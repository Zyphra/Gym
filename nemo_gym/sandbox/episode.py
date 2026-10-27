# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run one complete sandbox episode, sized from the sandbox it nests."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Mapping, Optional

from pydantic import BaseModel, Field, model_serializer

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxResources


# The episode cgroup also holds the placed Ray worker that starts the sandbox,
# stages files and collects results. Headroom above the nested sandbox cap keeps
# a runaway command's OOM inside the sandbox, so the worker survives to return a
# typed outcome instead of being group-killed with the episode.
EPISODE_MEMORY_HEADROOM_MIB = 2048


def sandbox_episode_resources(sandbox: SandboxResources) -> Any:
    """Episode CPU and PID limits are the sandbox's; memory adds the worker headroom."""
    from core.sandbox.executor import EpisodeResources

    if sandbox.gpu:
        raise ValueError("sandbox episodes do not support GPU resource requests")
    if sandbox.cpu is None or not float(sandbox.cpu).is_integer() or sandbox.cpu < 1:
        raise ValueError("sandbox resources.cpu must be a positive integer")
    if sandbox.memory_mib is None or sandbox.memory_mib < 1:
        raise ValueError("sandbox resources.memory_mib must be a positive integer")
    return EpisodeResources(
        num_cpus=int(sandbox.cpu),
        memory_bytes=(sandbox.memory_mib + EPISODE_MEMORY_HEADROOM_MIB) * 1024 * 1024,
        pids=sandbox.pids,
    )


_EPISODE_CONTROL: ContextVar[tuple[Optional[str], Optional[str], tuple[str, ...]]] = ContextVar(
    "sandbox_episode_control", default=(None, None, ())
)


@contextmanager
def sandbox_episode_control(name: Optional[str], episode_id: Optional[str] = None, excluded_node_ids=()):
    """Keep the caller-owned control identity inside one /run request."""
    token = _EPISODE_CONTROL.set((name, episode_id, tuple(excluded_node_ids)))
    try:
        yield
    finally:
        _EPISODE_CONTROL.reset(token)


async def run_sandbox_episode(
    fn: Callable, payload: Any, sandbox: SandboxResources, *, episode_id: Optional[str]
) -> object:
    from core.sandbox.executor import run_episode

    resources = sandbox_episode_resources(sandbox)
    control, enrolled_id, excluded_node_ids = _EPISODE_CONTROL.get()
    if control is not None:
        from core.sandbox import EpisodeOwnershipError
        from core.sandbox.lifecycle import RayEpisodeLifecycle

        if enrolled_id is None:
            raise EpisodeOwnershipError("controlled episode has no producer-enrolled identity")
        if episode_id is not None and episode_id != enrolled_id:
            raise EpisodeOwnershipError("sandbox episode identity differs from its enrolled identity")
        return await RayEpisodeLifecycle.attach(control).execute(enrolled_id, fn, payload, resources)
    return await run_episode(
        fn,
        payload,
        resources,
        episode_id=episode_id,
        **({"excluded_node_ids": excluded_node_ids} if excluded_node_ids else {}),
    )


class VerifierIntegrityMetrics(BaseModel):
    """Optional common metrics from a source-owned, quiescent verifier guard."""

    reward_hack_detected: Optional[int] = Field(default=None, ge=0, le=1)
    verifier_integrity_violations: Optional[int] = Field(default=None, ge=0)

    @model_serializer(mode="wrap")
    def preserve_default_wire(self, handler):
        result = handler(self)
        for field in ("reward_hack_detected", "verifier_integrity_violations"):
            if getattr(self, field) is None:
                result.pop(field, None)
        return result


class SandboxMemoryEvents(VerifierIntegrityMetrics):
    """Memory-limit events of the agent and grader sandboxes (instance and every command), read after close."""

    sandbox_oom: Optional[int] = None
    sandbox_oom_kill: Optional[int] = None
    sandbox_memory_max_events: Optional[int] = None
    grader_sandbox_oom: Optional[int] = None
    grader_sandbox_oom_kill: Optional[int] = None
    grader_sandbox_memory_max_events: Optional[int] = None


def sandbox_memory_events(sandbox: Any, prefix: str = "sandbox") -> dict[str, int]:
    """Metric fields from the stopped sandbox's retained memory.events; empty when none were kept."""
    events = sandbox.retained_memory_events() if isinstance(sandbox, AsyncSandbox) else {}
    if not events:
        return {}
    names = {"oom": "oom", "oom_kill": "oom_kill", "max": "memory_max_events"}
    return {f"{prefix}_{field}": events.get(name, 0) for name, field in names.items()}


def _memory_limit_kill(
    process: str, return_code: Optional[int], events: Mapping[str, int], prefix: str
) -> Optional[str]:
    oom_kill = events.get(f"{prefix}_oom_kill")
    if return_code not in (-9, 137) or not oom_kill:
        return None
    return f"{process} killed by the sandbox memory limit (exit {return_code}, oom_kill {oom_kill})"


def harness_memory_kill(return_code: Optional[int], events: Mapping[str, int]) -> Optional[str]:
    """Name an agent harness the sandbox memory limit killed: a SIGKILL exit beside a kernel OOM kill.

    A command the kernel killed while the harness carried on is the task's outcome, not this.
    """
    return _memory_limit_kill("agent harness", return_code, events, "sandbox")


def verifier_memory_kill(
    return_code: Optional[int], events: Mapping[str, int], prefix: str = "sandbox"
) -> Optional[str]:
    """Name a verifier the memory limit of its sandbox (``prefix``) killed: a SIGKILL exit beside a kernel OOM kill.

    Such a verifier did not complete, so a reward it wrote first is not a grade. A task
    subprocess the kernel killed while the verifier carried on is the task's outcome, not this.
    """
    return _memory_limit_kill("verifier", return_code, events, prefix)
