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
import asyncio
import hashlib
import importlib
import json
import os
import re
import shlex
import shutil
import sys
import tarfile
import tempfile
import time
import uuid
from asyncio import Semaphore
from contextlib import contextmanager
from pathlib import Path
from subprocess import Popen
from traceback import format_exc
from typing import Any, Dict, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle, harness_response_error
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.episode import (
    SandboxMemoryEvents,
    harness_memory_kill,
    run_sandbox_episode,
    sandbox_memory_events,
    verifier_memory_kill,
)
from nemo_gym.sandbox.providers.apptainer import ApptainerProvider
from nemo_gym.sandbox.providers.docker import DockerCreateConfig, DockerProvider
from nemo_gym.sandbox.providers.registry import create_provider
from nemo_gym.server_utils import apply_rollout_prefix
from nemo_gym.setup_timeout import setup_timeout_seconds
from responses_api_agents.anyterminal_agent.task_network import TaskNetwork


def _format_container(
    container_formatter: str | list[str], task_name: str, docker_image: str, runtime_image: str | None = None
) -> str:
    """Resolve the pullable/local image reference for a task from a formatter template."""

    fmt = container_formatter[0] if isinstance(container_formatter, list) else container_formatter
    fmt = fmt or "docker://{docker_image}"
    if fmt == "{runtime_image}":
        if not runtime_image or not Path(runtime_image).is_absolute():
            raise ValueError("runtime_image formatter requires an absolute admitted image path")
        return runtime_image
    docker_image = docker_image[len("docker://") :] if docker_image.startswith("docker://") else docker_image
    if fmt.endswith(".sif") or fmt.startswith(("/", ".")):
        return fmt.format(task_name=task_name, docker_image=docker_image)
    if fmt.startswith("docker://"):
        fmt = fmt[len("docker://") :]
    return f"docker://{fmt.format(task_name=task_name, docker_image=docker_image)}"


def _read_task_meta(task_dir: Path) -> dict:
    """Read workdir and timeouts from task.toml + Dockerfile at runtime (fallback when not in JSONL)."""
    result = {}
    toml_path = task_dir / "task.toml"
    if toml_path.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        with open(toml_path, "rb") as f:
            cfg = tomllib.load(f)
        result["agent_timeout_sec"] = (cfg.get("agent") or {}).get("timeout_sec")
        result["verifier_timeout_sec"] = (cfg.get("verifier") or {}).get("timeout_sec")
        if "storage_mb" in (cfg.get("environment") or {}):
            result["storage_mb"] = cfg["environment"]["storage_mb"]
    dockerfile = task_dir / "environment" / "Dockerfile"
    if dockerfile.exists():
        for line in dockerfile.read_text().splitlines():
            if line.strip().upper().startswith("WORKDIR"):
                parts = line.strip().split(None, 1)
                if len(parts) > 1:
                    result["workdir"] = parts[1]
    return result


_HARBOR_TEMPLATE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def task_verifier_environment(task_dir: Path, values: Mapping[str, str]) -> Dict[str, str]:
    """Resolve a Harbor task's ``[verifier.env]`` against caller-supplied values.

    The task declares which variables its verifier receives, as Harbor
    ``${NAME}`` / ``${NAME:-default}`` templates; the caller supplies only the
    values, never the worker's ambient environment. A referenced name the
    caller did not supply and that has no default is an infrastructure error.
    """
    toml_path = task_dir / "task.toml"
    if not toml_path.is_file():
        return {}
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]
    declared = (tomllib.loads(toml_path.read_text()).get("verifier") or {}).get("env") or {}
    resolved: Dict[str, str] = {}
    for name, template in declared.items():

        def substitute(match: "re.Match[str]") -> str:
            key, default = match.group(1), match.group(2)
            if values.get(key):
                return values[key]
            if default is not None:
                return default
            raise ValueError(f"task verifier variable {name} references {key}, which the caller did not supply")

        resolved[name] = _HARBOR_TEMPLATE.sub(substitute, str(template))
    return resolved


def _instruction_from_input(body: NeMoGymResponseCreateParamsNonStreaming) -> str:
    """Extract the task prompt from the Responses-API input messages.

    Joins the text of all messages (handling str or content-part list, dict or model form).
    """
    items = body.input
    if isinstance(items, str):
        return items
    parts: list[str] = []
    for item in items or []:
        content = getattr(item, "content", None) if not isinstance(item, dict) else item.get("content")
        if isinstance(content, list):
            content = "".join((p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content)
        if content:
            parts.append(content)
    return "\n".join(parts)


### Metrics


class TerminalBenchMetrics(SandboxMemoryEvents):
    resolved: Optional[bool] = None
    agent_timed_out: bool = False
    container_timed_out: bool = False
    sandbox_failed: bool = False
    mask_sample: bool = False
    # Labels only, never a mask: a harness or verifier failure that still
    # scores is named, so it is not indistinguishable from a policy miss.
    agent_exit_nonzero: bool = False
    agent_failed: bool = False
    agent_error: Optional[str] = None
    # The inner harness reports the agent itself ended the run (its own
    # completion), not a turn cap, timeout, failure or endpoint stop.
    finished_naturally: bool = False
    verifier_reward_unreadable: bool = False
    verifier_integrity_refused: bool = False
    verifier_integrity_problem: Optional[str] = None
    verifier_return_code: Optional[int] = None
    verifier_error_type: Optional[str] = None
    error_detail: Optional[str] = None

    ray_queue_time: Optional[float] = None
    agent_run_time: Optional[float] = None
    eval_run_time: Optional[float] = None
    total_run_time: Optional[float] = None
    submitted_at: Optional[float] = None
    task_started_at: Optional[float] = None
    sandbox_ready_at: Optional[float] = None
    agent_started_at: Optional[float] = None
    agent_completed_at: Optional[float] = None
    agent_phase_memory_events: Optional[dict[str, int]] = None
    verifier_phase_memory_events: Optional[dict[str, int]] = None
    memory_failure_phase: Optional[str] = None
    grading_started_at: Optional[float] = None
    grading_completed_at: Optional[float] = None

    def apply_timeout_policy(self, *, agent_timeout_is_failure: bool) -> None:
        """Keep policy timeouts in the scored population when explicitly enabled."""
        if self.agent_timed_out and agent_timeout_is_failure:
            self.resolved = False
        self.mask_sample = bool(
            self.mask_sample
            or self.container_timed_out
            or self.sandbox_failed
            or (self.agent_timed_out and not agent_timeout_is_failure)
        )


def update_metrics(metrics_fpath: Path, update_dict: Dict[str, Any], task_name: str = "") -> None:
    """Merge ``update_dict`` into the metrics file, starting from empty when it cannot be read.

    The file is seeded when the instance is set up, so an unreadable one means its directory went
    away under the episode (a sandbox or teardown failure). Raising here loses the metrics that
    name that failure and turns the episode into a 500 with no agent result and no reward, so the
    cause is printed once and the file is rewritten instead.
    """
    try:
        existing = {k: v for k, v in json.loads(metrics_fpath.read_text()).items() if v is not None}
    except (OSError, ValueError) as e:
        print(f"[{task_name}] metrics file unreadable, recording from empty: {metrics_fpath}: {e}", flush=True)
        existing = {}
    update = {k: v for k, v in update_dict.items() if v is not None}
    metrics_fpath.parent.mkdir(parents=True, exist_ok=True)
    metrics_fpath.write_text(json.dumps(existing | update))


# A dead process (killed -9, OOM, node loss) can leave a .lockdir behind forever, wedging every
# future waiter. 2h comfortably exceeds any real deps-setup run, so a lock that old is presumed dead.
_AGENT_DEPS_LOCK_STALE_AFTER_SECONDS = 7200  # 2 hours


@contextmanager
def _file_lock(file_path: Path, label: str, max_wait: float = 7200.0, poll_interval: float = 5.0):
    """Cross-node lock using mkdir, which is atomic on Lustre/NFS."""
    lock_dir = file_path.parent
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f".{file_path.name}.lockdir"

    print(f"Acquiring {label} lock at {lock_path}", flush=True)
    waited = 0.0
    while True:
        try:
            lock_path.mkdir(exist_ok=False)
            break
        except FileExistsError:
            try:
                lock_age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if lock_age >= _AGENT_DEPS_LOCK_STALE_AFTER_SECONDS:
                try:
                    if time.time() - lock_path.stat().st_mtime >= _AGENT_DEPS_LOCK_STALE_AFTER_SECONDS:
                        print(f"  Reclaiming stale {label} lock at {lock_path} (age {lock_age:.0f}s)", flush=True)
                        shutil.rmtree(lock_path, ignore_errors=True)
                except FileNotFoundError:
                    pass
                continue
            if waited >= max_wait:
                raise TimeoutError(
                    f"Timed out waiting for {label} lock at {lock_path} after {max_wait}s; lock age is {lock_age:.0f}s"
                )
            if waited % 30 == 0:
                print(f"  Waiting for {label} lock (held by another process, {waited:.0f}s elapsed)...", flush=True)
            time.sleep(poll_interval)
            waited += poll_interval

    try:
        yield
    finally:
        shutil.rmtree(lock_path, ignore_errors=True)


def _safe_config_json(params: "AnyTerminalInstanceConfig", indent: Optional[int] = None) -> str:
    """Serialize config without secrets."""

    def redact(value: Any, key: str = "") -> Any:
        normalized = key.lower()
        if (
            any(secret in normalized for secret in ("api_key", "apikey", "secret", "password"))
            or normalized == "token"
            or normalized.endswith("_token")
        ):
            return "***"
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                normalized_key = key.lower().replace("_", "").replace("-", "")
                is_secret = any(part in normalized_key for part in ("apikey", "password", "secret")) or (
                    normalized_key.endswith("token")
                )
                result[key] = "***" if is_secret else redact(item)
            return result
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    d = json.loads(params.model_dump_json())
    for key in ("agent_command_str", "agent_runtime_source", "agent_deps_url"):
        d.pop(key, None)
    return json.dumps(redact(d), indent=indent)


### Agent runner template
# Injected into the task container; imports any agent class and calls responses().

_RUNNER_TEMPLATE = """\
#!/usr/bin/env python3
import asyncio, json, os, sys
from pathlib import Path
from fastapi import Request

sys.path.insert(0, "/nemo_gym_mount")
# Append (not prepend) agent-deps bin so the task's own python/pip win — else the agent's
# builds/installs land in a Python the verifier can't see. Harness CLIs stay findable as a fallback.
os.environ["PATH"] = os.environ.get("PATH", os.defpath) + os.pathsep + "/agent_deps_mount/bin"

MODEL_URL    = os.environ.get("NGTB_MODEL_URL", "")
MODEL_NAME   = os.environ["NGTB_MODEL_NAME"]
INSTRUCTION  = Path("/trajectories_mount/instruction.txt").read_text()
AGENT_KWARGS = json.loads(os.environ.get("NGTB_AGENT_KWARGS", "{{}}"))
SAMPLING     = json.loads(os.environ.get("NGTB_SAMPLING", "{{}}"))

TASK_WORKDIR = os.environ.get("NGTB_TASK_WORKDIR") or str(Path.cwd())

openclaw_defaults = AGENT_KWARGS.get("openclaw_config", {{}}).get("agents", {{}}).get("defaults", {{}})
if openclaw_defaults.get("workspace") == ".":
    openclaw_defaults["workspace"] = str(Path.cwd())

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming, NeMoGymEasyInputMessage
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from {agent_module} import {agent_class}, {agent_cfg_class}

_mock_client = ServerClient.model_construct(global_config_dict={{}})
_mock_client._build_server_base_url = lambda cfg: MODEL_URL

_cfg_sampling = {{k: v for k, v in SAMPLING.items() if k in {agent_cfg_class}.model_fields}}
# Agents that declare a working directory run their tools in the task's, as under anyswe_agent.
if "working_directory" in {agent_cfg_class}.model_fields:
    AGENT_KWARGS.setdefault("working_directory", TASK_WORKDIR)

_model_server = ModelServerRef(name="policy_model", type="responses_api_models") if MODEL_URL else None
config = {agent_cfg_class}(
    host="0.0.0.0",
    port=0,
    name="{agent_class_lower}",
    entrypoint="app.py",
    model_server=_model_server,
    resources_server=ResourcesServerRef(name="anyterminal", type="resources_servers"),
    **{{**_cfg_sampling, **AGENT_KWARGS}},
)
agent = {agent_class}(config=config, server_client=_mock_client)

if MODEL_URL:
    _v1 = MODEL_URL if MODEL_URL.endswith("/v1") else MODEL_URL + "/v1"
    if hasattr(agent, "resolve_model_base_url"):
        object.__setattr__(agent, "resolve_model_base_url", lambda *args, **kwargs: _v1)
    if hasattr(agent, "_resolve_model_base_url"):
        agent._resolve_model_base_url = lambda *args, **kwargs: _v1
    if hasattr(agent, "_resolve_base_url"):
        agent._resolve_base_url = lambda *args, **kwargs: MODEL_URL

body = NeMoGymResponseCreateParamsNonStreaming(
    input=[NeMoGymEasyInputMessage(role="user", content=INSTRUCTION)],
    model=MODEL_NAME,
    **SAMPLING,
)
response = asyncio.run(agent.responses(request=Request({{"type": "http", "path_params": {{}}}}), body=body))
Path("/trajectories_mount/response.json").write_text(response.model_dump_json())
print(f"agent finished: {{len(response.output)}} output items", flush=True)
"""


### Agent harness installer
# Mirrors GymAgentHarnessProcessor in anyswe_agent: installs portable python + agent
# deps into a persistent prefix and writes agent_runner.py + instruction.txt.


class GymAgentHarnessProcessor(BaseModel):
    config: Any  # AnyTerminalAgentConfig at setup time; AnyTerminalInstanceConfig at run time

    @property
    def _parent(self) -> Path:
        return Path(__file__).parent

    @property
    def _agent_key(self) -> str:
        # responses_api_agents.hermes_agent.app -> hermes_agent
        return self.config.agent_server_module.split(".")[-2]

    def setup(self) -> Path:
        """Install agent deps into a portable prefix (idempotent, hash-keyed)."""
        agent_dir = PARENT_DIR / "responses_api_agents" / self._agent_key
        deps_dir = self._parent / "deps" / f"anyterminal_{self._agent_key}_deps"
        sentinel = deps_dir / ".installed"
        script = agent_dir / "scripts" / f"{self._agent_key}_deps.sh"
        shared = self._parent / "setup_scripts" / "_portable_python.sh"
        reqs = agent_dir / "requirements.txt"

        recipe_src = b"".join(p.read_bytes() for p in (script, shared, reqs) if p.exists()) or b"no-script"
        recipe = hashlib.sha256(recipe_src).hexdigest()
        if sentinel.exists() and sentinel.read_text().strip() == recipe:
            print(f"Agent deps already at {deps_dir}", flush=True)
            return deps_dir

        with _file_lock(deps_dir, f"{self._agent_key} deps setup"):
            if sentinel.exists() and sentinel.read_text().strip() == recipe:
                print(f"Agent deps already at {deps_dir}", flush=True)
                return deps_dir

            if not script.exists():
                print(f"No setup script for {self._agent_key}, skipping deps install", flush=True)
                deps_dir.mkdir(parents=True, exist_ok=True)
                sentinel.write_text(recipe)
                return deps_dir

            deps_dir.mkdir(parents=True, exist_ok=True)
            proc = Popen(
                f"PORTABLE_PYTHON_SH={shared} DEPS_DIR={deps_dir} NEMO_GYM_ROOT={PARENT_DIR} bash {script}",
                shell=True,
            )
            assert proc.wait() == 0, f"Agent deps setup failed ({script})"
            sentinel.write_text(recipe)
            return deps_dir

    def get_run_command(self) -> str:
        """Write instruction.txt and agent_runner.py; return the shell command to run the agent."""
        cfg: AnyTerminalInstanceConfig = self.config
        instruction = _instruction_from_input(cfg.body)
        (cfg.persistent_dir / "instruction.txt").write_text(instruction)
        runner = _RUNNER_TEMPLATE.format(
            agent_module=cfg.agent_server_module,
            agent_class=cfg.agent_server_class,
            agent_cfg_class=cfg.agent_config_class,
            agent_class_lower=cfg.agent_server_class.lower(),
        )
        (cfg.persistent_dir / "agent_runner.py").write_text(runner)
        return "/agent_deps_mount/bin/python /trajectories_mount/agent_runner.py"


### Configuration


class AnyTerminalAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: Optional[ModelServerRef] = None

    agent_server_module: str = Field(description="Import path to the agent module")
    agent_server_class: str = Field(description="Agent class name")
    agent_config_class: str = Field(description="Agent config class name")
    agent_kwargs: Dict[str, Any] = Field(default_factory=dict)

    container_formatter: str | list[str] = Field(
        default="docker://{docker_image}",
        description="Template for the task's image reference: use as a path if it ends with .sif or starts with / or ., else as a docker:// URI.",
    )
    sandbox_provider: Dict[str, Any] = Field(default_factory=lambda: {"docker": {}})
    sandbox_default_metadata: Dict[str, Any] = Field(default_factory=dict)
    apptainer_ignore_fakeroot_command_tasks: set[str] = Field(default_factory=set)
    apptainer_shared_pid_namespace: bool = False
    task_network: Optional[TaskNetwork] = None
    agent_environment: Dict[str, str] = Field(default_factory=dict)
    # Values for the names a task's [verifier.env] references (see task_verifier_environment).
    verifier_environment: Dict[str, str] = Field(default_factory=dict)

    @field_validator("task_network", mode="before")
    @classmethod
    def decode_task_network(cls, value):
        return TaskNetwork.model_validate_json(value) if isinstance(value, str) else value

    # Docker network for the agent container. "host" lets the in-container agent reach a
    # model server on host loopback; None uses the docker default (e.g. for a remote server).
    docker_network: Optional[str] = "host"
    sandbox_model_base_url: Optional[str] = None
    agent_runtime_source: str = Field(
        default="auto",
        description="auto, an absolute prepared dependency directory for local sandboxes, or a remote runtime archive/URL",
    )
    agent_timeout_is_failure: bool = Field(default=False, strict=True)
    tb_agent_timeout: int = 1800
    # When set, overrides the per-task agent_timeout_sec from the dataset for every task.
    global_agent_timeout: Optional[int] = Field(default=None, gt=0)
    tb_eval_timeout: int = 300
    tb_sandbox_ttl: int = 7200
    agent_overhead_mb: int = 2048  # extra container memory on top of the task's memory_mb for the
    # in-container agent harness
    concurrency: int = 256
    results_dir: Optional[Path] = None


class AnyTerminalRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class AnyTerminalServerConfig(BaseModel):
    run_session_id: str
    base_results_dir: Path
    model_server_url: str
    model_name: str = ""
    nemo_gym_root: Path
    agent_deps_dir: Path
    agent_deps_archive: Optional[Path] = None
    agent_deps_url: Optional[str] = None


class VerifierRuntime(BaseModel):
    """Immutable admitted Python verifier payloads; absent preserves stock execution."""

    model_config = ConfigDict(extra="forbid")
    helper_dir: Path
    cache_dir: Path
    helper_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cache_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    isolate_network: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def validate_payload_manifests(self):
        for directory, filename, digest in (
            (self.helper_dir, "helper-manifest.json", self.helper_manifest_sha256),
            (self.cache_dir, "verifier-cache.json", self.cache_manifest_sha256),
        ):
            if (
                not directory.is_absolute()
                or directory.resolve() != directory
                or directory.is_symlink()
                or not directory.is_dir()
                or any(c in str(directory) for c in (":", "\n", "\r", ","))
            ):
                raise ValueError("verifier payload must be a canonical bindable directory")
            manifest = directory / filename
            if manifest.is_symlink() or not manifest.is_file():
                raise ValueError("verifier payload manifest must be a regular file")
            if hashlib.sha256(manifest.read_bytes()).hexdigest() != digest:
                raise ValueError("verifier payload manifest identity differs from admission")
        return self

    def binds(self) -> list[str]:
        return [f"{self.helper_dir}:/staging/env_files:ro", f"{self.cache_dir}:/verifier_cache_mount:ro"]


class AnyTerminalInstanceConfig(AnyTerminalAgentConfig, AnyTerminalServerConfig):
    problem_info: Dict[str, Any]
    body: NeMoGymResponseCreateParamsNonStreaming
    persistent_dir: Path
    verifier_dir: Path
    agent_run_id: str
    metrics_fpath: Path
    container: str
    ray_queue_timestamp: float
    rollout_id: Optional[str] = None
    agent_command_str: Optional[str] = None
    verifier_runtime: Optional[VerifierRuntime] = None

    @property
    def task_name(self) -> str:
        return self.problem_info.get("task_name", self.problem_info.get("instance_id", "unknown"))

    @property
    def instance_id(self) -> str:
        return self.problem_info.get("instance_id", self.task_name)


class AnyTerminalVerifyResponse(TerminalBenchMetrics, BaseVerifyResponse):
    instance_config: Dict[str, Any]
    done_reason: Optional[str] = Field(default=None, exclude_if=lambda value: value is None)


def _episode_sandbox_resources(params: AnyTerminalInstanceConfig) -> SandboxResources:
    """Resolve the row-owned sandbox cap, separate from the episode aggregate."""

    def declared_int(key: str, *, optional: bool = False) -> Optional[int]:
        value = params.problem_info.get(key)
        if (
            value is None
            and optional
            and key == "pids"
            and (
                params.task_network is not None
                or next(iter(params.sandbox_provider), None) in {"enroot", "native_enroot"}
            )
        ):
            raw_pids = os.environ.get("NEMO_GYM_SANDBOX_PIDS", "")
            if not raw_pids.isdigit() or int(raw_pids) < 1:
                raise ValueError("private networking requires the profile PID limit")
            return int(raw_pids)
        if value is None and optional:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"AnyTerminal problem_info.{key} must be declared") from None
        if parsed < 1 or not parsed.is_integer():
            raise ValueError(f"AnyTerminal problem_info.{key} must be a positive integer")
        return int(parsed)

    if float(params.problem_info.get("gpus") or 0) > 0:
        raise ValueError("AnyTerminal episode executor does not support GPU resource requests")
    apptainer = params.sandbox_provider.get("apptainer") or {}
    create = apptainer.get("create") or {}
    disk_enabled = (
        next(iter(params.sandbox_provider), "docker") == "apptainer" and create.get("disk_backed_overlay") is True
    )
    storage_mib = declared_int("storage_mb", optional=True) if disk_enabled else None
    if storage_mib is not None and storage_mib % 1024:
        raise ValueError("AnyTerminal problem_info.storage_mb must be a whole number of GiB")
    return SandboxResources(
        cpu=declared_int("cpus"),
        memory_mib=declared_int("memory_mb") + params.agent_overhead_mb,
        disk_gib=None if storage_mib is None else storage_mib // 1024,
        pids=declared_int("pids", optional=True),
    )


### Sandbox provider selection


def _apt_root_sandbox(cfg: AnyTerminalInstanceConfig) -> str:
    # apt drops root to _apt before fetching; fakeroot's single-ID userns can't setgid to it.
    if next(iter(cfg.sandbox_provider), "docker") != "apptainer":
        return ""
    return (
        "mkdir -p /etc/apt/apt.conf.d && printf 'APT::Sandbox::User \"root\";\\n' "
        "> /etc/apt/apt.conf.d/99nemo-gym-apt-root; "
    )


def _build_provider(params: AnyTerminalInstanceConfig):
    """Build a sandbox provider with the per-instance mounts the run needs.

    Docker and Apptainer bind the local runtime directories at the paths expected by the
    agent runner. Other providers use the sandbox file API in RunTerminalAgent.
    """
    name = next(iter(params.sandbox_provider), "docker")
    if name in {"enroot", "native_enroot"}:
        from core.sandbox.terminal_provider import terminal_provider

        verifier_binds = [] if params.verifier_runtime is None else params.verifier_runtime.binds()
        return terminal_provider(
            params.sandbox_provider,
            [
                f"{params.persistent_dir}:/trajectories_mount",
                f"{params.nemo_gym_root}:/nemo_gym_mount:ro",
                f"{params.agent_deps_dir}:/agent_deps_mount:ro",
                f"{params.verifier_dir}:/logs/verifier",
                *verifier_binds,
            ],
            task_network=params.task_network,
        )
    if params.task_network is not None:
        if name != "apptainer":
            raise ValueError("private task networking requires Apptainer")
        runtime = getattr(params, "verifier_runtime", None)
        if runtime is not None and runtime.isolate_network:
            raise ValueError("private task verifier must inherit its instance network")
    verifier_binds = [] if params.verifier_runtime is None else params.verifier_runtime.binds()
    if verifier_binds and name not in {"apptainer", "docker", "enroot", "native_enroot"}:
        raise ValueError("admitted verifier runtime requires a local bind-mount provider")
    if name == "apptainer":
        appt = {
            k: v
            for k, v in (params.sandbox_provider.get("apptainer") or {}).items()
            if k in ("exec", "create", "probe")
        }
        exec_cfg = dict(appt.get("exec") or {})
        exec_cfg["default_binds"] = (
            list(exec_cfg.get("default_binds") or [])
            + [
                f"{params.persistent_dir}:/trajectories_mount",
                f"{params.nemo_gym_root}:/nemo_gym_mount:ro",
                f"{params.agent_deps_dir}:/agent_deps_mount:ro",
                f"{params.verifier_dir}:/logs/verifier",
            ]
            + verifier_binds
        )
        exec_args = list(exec_cfg.get("extra_exec_args") or [])
        if params.apptainer_shared_pid_namespace:
            exec_args = [arg for arg in exec_args if arg != "--pid"]
        exec_cfg["extra_exec_args"] = exec_args + [
            "--cleanenv",
            *([] if params.apptainer_shared_pid_namespace else ["--pid"]),
            "--no-mount",
            "tmp",
        ]
        appt["exec"] = exec_cfg

        create_cfg = dict(appt.get("create") or {})
        start_args = list(create_cfg.get("extra_start_args") or [])
        if params.problem_info.get("task_name") in params.apptainer_ignore_fakeroot_command_tasks:
            start_args.append("--ignore-fakeroot-command")
        if create_cfg.get("disk_backed_overlay") is True:
            start_args = [arg for arg in start_args if arg != "--writable-tmpfs"]
        elif "--writable-tmpfs" not in start_args:
            start_args.append("--writable-tmpfs")
        # Exec reuses the instance's mounts; exclude host /tmp when creating it.
        for index, arg in enumerate(start_args):
            if arg == "--no-mount":
                start_args[index + 1] += ",home,tmp"
                break
            if arg.startswith("--no-mount="):
                start_args[index] += ",home,tmp"
                break
        else:
            start_args += ["--no-mount", "home,tmp"]
        create_cfg["extra_start_args"] = start_args
        appt["create"] = create_cfg
        if params.task_network is not None:
            from core.sandbox.service_apptainer import register_private_network_provider

            register_private_network_provider()
            return create_provider(
                {
                    "apptainer_services": {
                        **appt,
                        "tools_root": str(params.task_network.tools_root),
                        "tools_sha256": params.task_network.tools_sha256,
                        "network_required": True,
                    }
                }
            )
        return ApptainerProvider(**appt)
    if name != "docker":
        return params.sandbox_provider
    return DockerProvider(
        create=DockerCreateConfig(
            network=params.docker_network,
            extra_run_args=[
                "-v",
                f"{params.persistent_dir}:/trajectories_mount",
                "-v",
                f"{params.nemo_gym_root}:/nemo_gym_mount:ro",
                "-v",
                f"{params.agent_deps_dir}:/agent_deps_mount:ro",
                "-v",
                f"{params.verifier_dir}:/logs/verifier",
            ]
            + [arg for bind in verifier_binds for arg in ("-v", bind)],
        ),
    )


### Container lifecycle


class RunTerminalAgent(BaseModel):
    """Single sandbox: agent runs, host stages tests, sandbox runs test.sh."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: AnyTerminalInstanceConfig

    @staticmethod
    def _uses_bind_mounts(cfg: AnyTerminalInstanceConfig) -> bool:
        return next(iter(cfg.sandbox_provider), "docker") in {"apptainer", "docker", "enroot", "native_enroot"}

    @staticmethod
    def _archive(source: Path) -> Path:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as temporary:
            archive = Path(temporary.name)
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(source, arcname=".")
        return archive

    async def _stage_remote_runtime(self, sandbox: AsyncSandbox, cfg: AnyTerminalInstanceConfig) -> None:
        result = await sandbox.exec(
            "mkdir -p /agent_deps_mount /trajectories_mount /logs/verifier",
            timeout_s=setup_timeout_seconds(30),
            user="root",
        )
        if result.return_code != 0:
            raise RuntimeError(result.stderr or "failed to create sandbox runtime directories")
        await sandbox.upload(cfg.persistent_dir / "instruction.txt", "/trajectories_mount/instruction.txt")
        await sandbox.upload(cfg.persistent_dir / "agent_runner.py", "/trajectories_mount/agent_runner.py")
        external_runtime = cfg.agent_deps_archive is not None or cfg.agent_deps_url is not None
        if cfg.agent_deps_archive is not None:
            await sandbox.upload(cfg.agent_deps_archive, "/tmp/anyterminal-agent-deps.tar.gz")
        elif cfg.agent_deps_url is not None:
            runtime_url = shlex.quote(cfg.agent_deps_url)
            python_fetch = shlex.quote(
                "import urllib.request; "
                f"urllib.request.urlretrieve({cfg.agent_deps_url!r}, '/tmp/anyterminal-agent-deps.tar.gz')"
            )
            result = await sandbox.exec(
                "if command -v curl >/dev/null 2>&1; then "
                f"curl -fsSL -o /tmp/anyterminal-agent-deps.tar.gz {runtime_url}; "
                "elif command -v wget >/dev/null 2>&1; then "
                f"wget -qO /tmp/anyterminal-agent-deps.tar.gz {runtime_url}; "
                "elif command -v python3 >/dev/null 2>&1; then "
                f"python3 -c {python_fetch}; "
                "else echo 'runtime download requires curl, wget, or python3' >&2; exit 127; fi",
                timeout_s=setup_timeout_seconds(1200),
                user="root",
            )
            if result.return_code != 0:
                raise RuntimeError(result.stderr or "failed to fetch agent runtime")
        else:
            result = await sandbox.exec(
                "test -x /agent_deps_mount/bin/python", timeout_s=setup_timeout_seconds(30), user="root"
            )
            if result.return_code != 0:
                raise RuntimeError("task image does not contain /agent_deps_mount/bin/python")
        if external_runtime:
            result = await sandbox.exec(
                "tar -xzf /tmp/anyterminal-agent-deps.tar.gz -C /agent_deps_mount && "
                "rm -f /tmp/anyterminal-agent-deps.tar.gz",
                timeout_s=setup_timeout_seconds(900),
                user="root",
            )
            if result.return_code != 0:
                raise RuntimeError(result.stderr or "failed to extract agent runtime")

    async def _stage_remote_tests(self, sandbox: AsyncSandbox, cfg: AnyTerminalInstanceConfig) -> None:
        archive = await asyncio.to_thread(self._archive, cfg.persistent_dir / "staging" / "tests")
        try:
            await sandbox.upload(archive, "/tmp/anyterminal-tests.tar.gz")
            result = await sandbox.exec(
                "mkdir -p /trajectories_mount/staging/tests && "
                "tar -xzf /tmp/anyterminal-tests.tar.gz -C /trajectories_mount/staging/tests",
                timeout_s=setup_timeout_seconds(300),
                user="root",
            )
            if result.return_code != 0:
                raise RuntimeError(result.stderr or "failed to stage verifier tests")
        finally:
            archive.unlink(missing_ok=True)

    async def _collect_remote_outputs(self, sandbox: AsyncSandbox, cfg: AnyTerminalInstanceConfig) -> None:
        for remote, local in (
            ("/trajectories_mount/response.json", cfg.persistent_dir / "response.json"),
            ("/logs/verifier/reward.txt", cfg.verifier_dir / "reward.txt"),
            ("/logs/verifier/test-stdout.txt", cfg.verifier_dir / "test-stdout.txt"),
        ):
            exists = await sandbox.exec(f"test -f {remote}", timeout_s=30, user="root")
            if exists.return_code == 0:
                await sandbox.download(remote, local)

    def _agent_env(self, cfg: AnyTerminalInstanceConfig) -> Dict[str, str]:
        sampling = {
            k: getattr(cfg.body, k)
            for k in ("temperature", "top_p", "max_output_tokens")
            if getattr(cfg.body, k, None) is not None
        }
        model_name = cfg.agent_kwargs.get("model") or cfg.body.model or "model"
        env = {
            "NGTB_MODEL_NAME": model_name,
            "NGTB_AGENT_KWARGS": json.dumps(cfg.agent_kwargs),
            "NGTB_SAMPLING": json.dumps(sampling),
        }
        if cfg.model_server_url:
            env["NGTB_MODEL_URL"] = cfg.model_server_url
        if cfg.problem_info.get("workdir"):
            env["NGTB_TASK_WORKDIR"] = str(cfg.problem_info["workdir"])
        env.update(cfg.agent_environment)
        return env

    async def _run_agent(self, sandbox: AsyncSandbox, cfg: AnyTerminalInstanceConfig) -> tuple[float, bool, bool, int]:
        from core.sandbox.memory_evidence import memory_event_delta, sandbox_memory_snapshot

        t0 = time.time()
        before = sandbox_memory_snapshot(sandbox)
        update_metrics(cfg.metrics_fpath, {"agent_started_at": time.time()}, cfg.task_name)
        try:
            result = await sandbox.exec(
                _apt_root_sandbox(cfg) + (cfg.agent_command_str or ""),
                timeout_s=cfg.tb_agent_timeout,
                user="root",
                env=self._agent_env(cfg),
            )
        finally:
            self._agent_phase_memory_events = memory_event_delta(before, sandbox_memory_snapshot(sandbox))
            update_metrics(
                cfg.metrics_fpath,
                {
                    "agent_completed_at": time.time(),
                    "agent_phase_memory_events": self._agent_phase_memory_events or None,
                },
                cfg.task_name,
            )
        (cfg.persistent_dir / "agent_result.json").write_text(
            json.dumps(
                {
                    "return_code": result.return_code,
                    "error_type": result.error_type,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
        )
        if result.return_code != 0:
            detail = result.stderr or result.stdout or ""
            print(f"[{cfg.task_name}] agent exit {result.return_code}: {detail[-2000:]}", flush=True)
        return time.time() - t0, result.error_type == "timeout", result.error_type == "sandbox", result.return_code

    async def _stage_tests(self, cfg: AnyTerminalInstanceConfig) -> None:
        """Copy the task's test files into the staging dir, visible to the sandbox at /tests."""
        src = Path(cfg.problem_info["task_dir"]) / "tests"
        staging_tests = cfg.persistent_dir / "staging" / "tests"
        if staging_tests.exists():
            shutil.rmtree(staging_tests)
        shutil.copytree(src, staging_tests)

    async def _exec_verifier(
        self, sandbox: AsyncSandbox, command: str, *, env: dict[str, str] | None, timeout_s: int | float
    ):
        # Apptainer's env-file shell reserves HOME. Set it only in the child
        # command, preserving the provider's validation and the agent's home.
        projected = None if env is None else dict(env)
        if projected is not None and "HOME" in projected:
            home = projected.pop("HOME")
            command = "env " + shlex.quote("HOME=" + home) + " sh -c " + shlex.quote(command)
        runtime = self.config.verifier_runtime
        if runtime is not None and runtime.isolate_network:
            # Only verifier children lose networking; task filesystem and the
            # agent/oracle network remain unchanged. No online fallback.
            # A fresh namespace starts with loopback down. Offline tests may
            # still start local servers; enable only lo, with no external route.
            setup = (
                "import fcntl,os,socket,struct,sys; "
                "s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); "
                "flags=struct.unpack('16sh',fcntl.ioctl(s,0x8913,"
                "struct.pack('16sh',b'lo',0))[:18])[1]; "
                "fcntl.ioctl(s,0x8914,struct.pack('16sh',b'lo',flags|1)); "
                "os.execvp('sh',['sh','-c',sys.argv[1]])"
            )
            command = (
                "unshare -n -- /staging/env_files/python/verifier-python/bin/python3 -I -c "
                + shlex.quote(setup)
                + " "
                + shlex.quote(command)
            )
        return await sandbox.exec(command, env=projected, timeout_s=timeout_s, user="root")

    async def _prepare_verifier_runtime(
        self, sandbox: AsyncSandbox, cfg: AnyTerminalInstanceConfig
    ) -> dict[str, str] | None:
        """Create only verifier-local state; never run task helper setup or alter /app."""
        if cfg.verifier_runtime is None:
            return None
        root = f"/tmp/gym-verifier-{uuid.uuid4().hex}"
        helper = "/staging/env_files/python"
        # Inspect before adding tools: task commands must keep their interpreter
        # and HOME. The explicit helper also works in Python-less task images.
        inspect = (
            "import json,os,platform,pwd,sys; "
            "assert sys.version_info[:2] == (3,13); "
            "libc=platform.libc_ver()[0]; "
            "assert sys.platform == 'linux' and libc in ('glibc','musl'); "
            "key='-'.join((sys.implementation.name,platform.python_version(),"
            "sys.platform,platform.machine(),'gnu' if libc == 'glibc' else 'musl')); "
            "print(json.dumps({'HOME':os.environ.get('HOME') or pwd.getpwuid(os.getuid()).pw_dir,"
            "'PATH':os.environ.get('PATH',os.defpath),'python_key':key}))"
        )
        task_environment = await sandbox.exec(
            f"{helper}/verifier-python/bin/python3 -I -c " + shlex.quote(inspect),
            timeout_s=cfg.tb_eval_timeout,
            user="root",
        )
        if task_environment.return_code != 0 or task_environment.error_type:
            (cfg.verifier_dir / "runtime-setup.json").write_text(
                json.dumps(
                    {
                        "phase": "task_environment",
                        "return_code": task_environment.return_code,
                        "error_type": task_environment.error_type,
                        "stdout": task_environment.stdout,
                        "stderr": task_environment.stderr,
                    }
                )
            )
            raise RuntimeError("private verifier runtime setup failed; see runtime-setup.json")
        task_env = json.loads(task_environment.stdout)
        env = {
            "HOME": task_env["HOME"],
            "PATH": f"{helper}/verifier-tools/bin:{task_env['PATH']}:{helper}/verifier-python/bin",
            "UV_CACHE_DIR": f"{root}/cache",
            "UV_TOOL_DIR": f"{root}/tools",
            "UV_PYTHON_INSTALL_DIR": f"{root}/python",
            "UV_PYTHON_PREFERENCE": "only-managed",
            "UV_PYTHON_DOWNLOADS": "never",
            "UV_OFFLINE": "true",
        }
        # uv discovers the already prepared helper independently of task PATH.
        managed_python = env["UV_PYTHON_INSTALL_DIR"] + "/" + task_env["python_key"]
        bootstrap_env = env["HOME"] + "/.local/bin/env"
        command = (
            f"umask 077; mkdir {shlex.quote(root)} && "
            f"mkdir -p {shlex.quote(env['UV_PYTHON_INSTALL_DIR'])} "
            f"{shlex.quote(env['HOME'] + '/.local/bin')} && "
            f"mkdir {shlex.quote(managed_python)} && ln -s {helper}/verifier-python {shlex.quote(managed_python + '/install')} && "
            f"cp -a /verifier_cache_mount/cache {shlex.quote(env['UV_CACHE_DIR'])} && "
            f"chmod -R u+w {shlex.quote(env['UV_CACHE_DIR'])} && "
            f"if [ ! -e {shlex.quote(bootstrap_env)} ] && [ ! -L {shlex.quote(bootstrap_env)} ]; "
            f"then printf '# Prepared verifier tools are supplied by the environment.\\n' "
            f"> {shlex.quote(bootstrap_env)}; fi && "
            f"{helper}/verifier-tools/bin/uv python find --offline 3.13"
        )
        result = await self._exec_verifier(sandbox, command, env=env, timeout_s=cfg.tb_eval_timeout)
        receipt = {
            "runtime": cfg.verifier_runtime.model_dump(mode="json"),
            "environment": env,
            "return_code": result.return_code,
            "error_type": result.error_type,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        (cfg.verifier_dir / "runtime-setup.json").write_text(json.dumps(receipt))
        if result.return_code != 0 or result.error_type:
            raise RuntimeError("private verifier runtime setup failed; see runtime-setup.json")
        return env

    async def _run_eval(
        self, sandbox: AsyncSandbox, cfg: AnyTerminalInstanceConfig
    ) -> tuple[float, bool, Optional[int], Optional[str]]:
        """Run test.sh: elapsed, timed out or sandbox error, and the verifier's return code and error type."""
        t0 = time.time()
        verifier_env = await self._prepare_verifier_runtime(sandbox, cfg)
        task_env = task_verifier_environment(Path(cfg.problem_info["task_dir"]), cfg.verifier_environment)
        if task_env:
            verifier_env = {**task_env, **(verifier_env or {})}
        test_cmd = (
            (
                "readlink /proc/self/ns/net > /logs/verifier/network-namespace.txt && "
                if cfg.task_network is not None
                else ""
            )
            + "rm -rf /tests && ln -s /trajectories_mount/staging/tests /tests && "
            # A minimal pytest.ini at / stops pytest from picking up Gym's pyproject.toml
            # (whose --pyargs breaks paths inside the sandbox).
            "printf '[pytest]\\naddopts =\\n' > /pytest.ini && "
            "mkdir -p /logs/verifier && bash /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1"
        )
        result = await self._exec_verifier(
            sandbox, _apt_root_sandbox(cfg) + test_cmd, timeout_s=cfg.tb_eval_timeout, env=verifier_env
        )
        if result.return_code != 0:
            print(f"[{cfg.task_name}] eval exit {result.return_code}: {(result.stderr or '')[-2000:]}", flush=True)
        return (
            time.time() - t0,
            result.error_type == "timeout",
            result.return_code,
            result.error_type,
        )

    async def process_single_datapoint(self) -> bool:
        from core.sandbox.memory_evidence import memory_event_delta, sandbox_memory_snapshot

        from responses_api_agents.anyterminal_agent.verifier_integrity import (
            VerifierIntegrity,
            VerifierIntegrityRefused,
        )

        cfg = self.config
        cfg.verifier_dir.mkdir(parents=True, exist_ok=True)
        (cfg.persistent_dir / "staging").mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        sandbox = AsyncSandbox(
            _build_provider(cfg),
            SandboxSpec(
                image=cfg.container.removeprefix("docker://") if not self._uses_bind_mounts(cfg) else cfg.container,
                ttl_s=cfg.tb_sandbox_ttl,
                workdir=cfg.problem_info.get("workdir"),
                metadata=cfg.sandbox_default_metadata,
                resources=_episode_sandbox_resources(cfg),
            ),
        )
        agent_timed_out = container_timed_out = False
        agent_return_code = eval_return_code = eval_error_type = None
        sandbox_failed = False
        agent_run_time = eval_run_time = None
        integrity = None
        verifier_memory_before = verifier_memory_after = {}
        integrity_problem = reward_problem = None
        resolved = False
        try:
            if self._uses_bind_mounts(cfg):
                integrity = VerifierIntegrity(cfg.verifier_dir, cfg.persistent_dir)
            await sandbox.start()
            update_metrics(cfg.metrics_fpath, {"sandbox_ready_at": time.time()}, cfg.task_name)
            if not self._uses_bind_mounts(cfg):
                await self._stage_remote_runtime(sandbox, cfg)
            agent_run_time, agent_timed_out, sandbox_failed, agent_return_code = await self._run_agent(sandbox, cfg)
            await self._stage_tests(cfg)
            if not self._uses_bind_mounts(cfg):
                await self._stage_remote_tests(sandbox, cfg)
            if integrity is not None:
                integrity.purge()
            grading_started_ns = time.time_ns()
            grading_started_at = grading_started_ns / 1_000_000_000
            if integrity is not None:
                integrity.mark_dispatch(grading_started_ns)
            update_metrics(cfg.metrics_fpath, {"grading_started_at": grading_started_at}, cfg.task_name)
            verifier_memory_before = sandbox_memory_snapshot(sandbox)
            try:
                eval_run_time, container_timed_out, eval_return_code, eval_error_type = await self._run_eval(
                    sandbox, cfg
                )
                sandbox_failed = sandbox_failed or eval_error_type == "sandbox"
            finally:
                verifier_memory_after = sandbox_memory_snapshot(sandbox)
                grading_completed_at = time.time()
                update_metrics(cfg.metrics_fpath, {"grading_completed_at": grading_completed_at}, cfg.task_name)
                (cfg.verifier_dir / "grading-timing.json").write_text(
                    json.dumps(
                        {
                            "started_at": grading_started_at,
                            "completed_at": grading_completed_at,
                        }
                    )
                )
            if not self._uses_bind_mounts(cfg):
                await self._collect_remote_outputs(sandbox, cfg)
        except VerifierIntegrityRefused as e:
            integrity_problem = str(e)[:512]
            print(f"[{cfg.task_name}] verifier integrity refused: {integrity_problem}", flush=True)
        except Exception as e:
            sandbox_failed = True
            print(f"[{cfg.task_name}] sandbox run failed: {e}", flush=True)
        finally:
            try:
                await sandbox.stop()
            except Exception as e:
                sandbox_failed = True
                print(f"[{cfg.task_name}] sandbox cleanup failed: {e}", flush=True)
            finally:
                if integrity is not None:
                    try:
                        if integrity_problem is None:
                            resolved = integrity.read_reward() > 0
                    except VerifierIntegrityRefused as e:
                        integrity_problem = str(e)[:512]
                    except (ValueError, OSError) as e:
                        reward_problem = f"{type(e).__name__}: {e}"
                    finally:
                        integrity.receipt["refused"] = integrity_problem
                        integrity.close()
                        try:
                            integrity.save_receipt(cfg.persistent_dir / "verifier-integrity.json")
                        except (VerifierIntegrityRefused, OSError) as e:
                            resolved = False
                            integrity_problem = f"{type(e).__name__}: {e}"[:512]
                            print(f"[{cfg.task_name}] verifier receipt refused: {integrity_problem}", flush=True)
            shutil.rmtree(cfg.persistent_dir / "staging", ignore_errors=True)

        total_run_time = time.time() - t0
        memory_events = sandbox_memory_events(sandbox)
        memory_kill = harness_memory_kill(agent_return_code, memory_events)
        # The agent and the verifier share this sandbox. A verifier the limit killed did not
        # complete, so a reward it wrote first is not a grade; any other exit keeps its reward.
        verifier_kill = verifier_memory_kill(eval_return_code, memory_events)
        verifier_phase_events = memory_event_delta(verifier_memory_before, verifier_memory_after)
        verifier_phase_kill = verifier_memory_kill(eval_return_code, verifier_phase_events)
        agent_phase_kill = harness_memory_kill(agent_return_code, getattr(self, "_agent_phase_memory_events", {}))
        if verifier_kill is not None and verifier_phase_kill is None:
            verifier_kill = "verifier SIGKILL with sandbox OOM history; execution phase UNKNOWN"
        if memory_kill is not None and agent_phase_kill is None:
            memory_kill = "agent harness SIGKILL with sandbox OOM history; execution phase UNKNOWN"
        sandbox_failed = sandbox_failed or memory_kill is not None or verifier_kill is not None

        reward_path = cfg.verifier_dir / "reward.txt"
        if integrity is None:
            try:
                resolved = float(reward_path.read_text().strip()) > 0
            except (ValueError, OSError) as e:
                reward_problem = f"{type(e).__name__}: {e}"
        # test.sh writes 0 or 1; a completed eval without a readable reward is a verifier failure.
        verifier_reward_unreadable = bool(
            reward_problem and eval_run_time is not None and not (container_timed_out or sandbox_failed)
        )
        if verifier_reward_unreadable:
            print(f"[{cfg.task_name}] eval finished without a readable reward.txt: {reward_problem}", flush=True)

        metrics = TerminalBenchMetrics(
            ray_queue_time=time.time() - cfg.ray_queue_timestamp,
            resolved=resolved,
            agent_timed_out=agent_timed_out,
            container_timed_out=container_timed_out,
            sandbox_failed=sandbox_failed,
            mask_sample=bool(container_timed_out or sandbox_failed),
            verifier_reward_unreadable=verifier_reward_unreadable,
            verifier_integrity_refused=integrity_problem is not None,
            verifier_integrity_problem=integrity_problem,
            verifier_return_code=eval_return_code,
            verifier_error_type=eval_error_type,
            agent_run_time=agent_run_time,
            eval_run_time=eval_run_time,
            total_run_time=total_run_time,
            error_detail=memory_kill or verifier_kill,
            verifier_phase_memory_events=verifier_phase_events or None,
            memory_failure_phase="verifier"
            if verifier_phase_kill
            else ("agent_harness" if agent_phase_kill else "unknown" if memory_kill or verifier_kill else None),
            **memory_events,
        )
        metrics.apply_timeout_policy(agent_timeout_is_failure=cfg.agent_timeout_is_failure)
        update_metrics(cfg.metrics_fpath, metrics.model_dump(), cfg.task_name)
        return bool(metrics.resolved)


def _terminal_response(params: AnyTerminalInstanceConfig) -> NeMoGymResponse:
    persisted = TerminalBenchMetrics.model_validate_json(params.metrics_fpath.read_text())
    persisted.apply_timeout_policy(agent_timeout_is_failure=params.agent_timeout_is_failure)
    mask_sample = persisted.mask_sample
    update_metrics(params.metrics_fpath, persisted.model_dump(), params.task_name)

    response_path = params.persistent_dir / "response.json"
    output_items, tools = [], []
    agent_stop_reason = None
    agent_error = None
    finished_naturally = False
    saved = None
    harness_error = None
    if response_path.exists():
        try:
            data = json.loads(response_path.read_text())
            data["model"] = params.model_name
            raw_observations = data.pop("_ng_agent_observations", None)
            observations = (
                AgentObservationBundle.model_validate(raw_observations) if raw_observations is not None else None
            )
            saved = NeMoGymResponse.model_validate(data)
            agent_stop_reason = (saved.metadata or {}).get("stop_reason")
            harness_error = harness_response_error(saved, observations)
            output_items = saved.output
            tools = saved.tools or []
            # The inner harness's own failure label (hermes status/error,
            # terminus metadata agent_error) is kept, not dropped.
            agent_error = (saved.metadata or {}).get("agent_error")
            finished_naturally = (saved.metadata or {}).get("finished_naturally") == "true"
            if saved.status == "failed" and not agent_error:
                agent_error = (saved.error.code if saved.error else None) or "failed"
        except (json.JSONDecodeError, ValueError) as e:
            agent_error = "response_unreadable"
            print(f"[{params.task_name}] response.json unreadable ({e}), treating as empty response", flush=True)
    elif not mask_sample:
        agent_error = "response_missing"
    try:
        agent_result = json.loads((params.persistent_dir / "agent_result.json").read_text())
    except (OSError, ValueError):
        agent_result = {}
    exit_code = agent_result.get("return_code")
    labels = {
        "agent_failed": agent_error is not None,
        "agent_error": agent_error,
        "mask_sample": mask_sample or harness_error is not None,
        # A timeout or sandbox error is already masked; a plain nonzero exit is a harness crash.
        "agent_exit_nonzero": agent_result.get("error_type") is None and exit_code not in (None, 0),
        "finished_naturally": finished_naturally,
    }
    update_metrics(params.metrics_fpath, labels, params.task_name)

    return NeMoGymResponse(
        id=f"anyterminal-{params.instance_id}",
        created_at=int(time.time()),
        model=params.model_name,
        object="response",
        output=output_items,
        parallel_tool_calls=params.body.parallel_tool_calls,
        tool_choice=params.body.tool_choice,
        tools=tools,
        status="failed" if harness_error is not None else None,
        error=harness_error,
        metadata={
            **({"stop_reason": agent_stop_reason} if agent_stop_reason == "context_exhausted" else {}),
            "input": json.dumps(params.body.model_dump(mode="json").get("input") or []),
            "metrics": params.metrics_fpath.read_text(),
            "instance_config": _safe_config_json(params),
        },
    )


async def run_terminal_episode(params_dict: dict) -> Dict[str, Any]:
    """Stable worker entry point for the complete Terminal harness and grading run."""

    task_started_at = time.time()
    AnyTerminalInstanceConfig.model_rebuild(force=True)
    RunTerminalAgent.model_rebuild(force=True)
    params = AnyTerminalInstanceConfig.model_validate(params_dict)
    update_metrics(params.metrics_fpath, {"task_started_at": task_started_at}, params.task_name)
    await RunTerminalAgent(config=params).process_single_datapoint()
    return _terminal_response(params).model_dump(mode="json")


### Agent server


class AnyTerminalAgent(SimpleResponsesAPIAgent):
    """Runs any Gym agent harness inside a Terminal Bench task sandbox."""

    config: AnyTerminalAgentConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _sem: Optional[Semaphore] = None
    _server: Optional[AnyTerminalServerConfig] = None

    def model_post_init(self, context: Any) -> None:
        self._sem = Semaphore(self.config.concurrency)
        self.config.sandbox_default_metadata = resolve_provider_metadata(
            self.config.sandbox_provider, self.server_client.global_config_dict
        )
        self.config.sandbox_provider = resolve_provider_config(
            self.config.sandbox_provider, self.server_client.global_config_dict
        )

        model_url = self.config.sandbox_model_base_url or ""
        if self.config.model_server is not None:
            model_cfg = get_first_server_config_dict(
                self.server_client.global_config_dict, self.config.model_server.name
            )
            if not model_url:
                model_url = self.server_client._build_server_base_url(model_cfg)

        # Real model identifier the policy server serves, set via +policy_model_name=... at run time.
        model_name = str(self.server_client.global_config_dict.get("policy_model_name") or "")

        workspace = Path(__file__).parent
        provider_name = next(iter(self.config.sandbox_provider), "docker")
        remote_provider = provider_name not in {"apptainer", "docker", "enroot", "native_enroot"}
        runtime_source = self.config.agent_runtime_source
        agent_deps_dir = workspace
        agent_deps_archive = None
        agent_deps_url = None
        if runtime_source == "auto":
            agent_deps_dir = GymAgentHarnessProcessor(config=self.config).setup()
        elif runtime_source == "baked":
            if not remote_provider:
                raise ValueError("agent_runtime_source=baked is only valid for remote sandbox providers")
        elif "://" in runtime_source:
            if not remote_provider:
                raise ValueError("URL agent_runtime_source is only valid for remote sandbox providers")
            agent_deps_url = runtime_source
        else:
            runtime_path = Path(runtime_source).expanduser()
            if not remote_provider:
                if not runtime_path.is_absolute() or not runtime_path.is_dir():
                    raise ValueError("local agent_runtime_source must be an absolute prepared dependency directory")
                agent_deps_dir = runtime_path.resolve(strict=True)
                python = agent_deps_dir / "bin" / "python"
                if not python.is_file() or not os.access(python, os.X_OK):
                    raise ValueError(f"prepared agent runtime has no executable bin/python: {agent_deps_dir}")
            else:
                agent_deps_archive = runtime_path
                if not agent_deps_archive.is_file():
                    raise ValueError(f"agent runtime archive not found: {agent_deps_archive}")
        if remote_provider and runtime_source == "auto":
            agent_deps_archive = workspace / f".{agent_deps_dir.name}.tar.gz"
            sentinel = agent_deps_dir / ".installed"
            if not agent_deps_archive.exists() or agent_deps_archive.stat().st_mtime < sentinel.stat().st_mtime:
                temporary = agent_deps_archive.with_suffix(".tmp")
                with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
                    archive.add(agent_deps_dir, arcname=".")
                temporary.replace(agent_deps_archive)
        base_results_dir = self.config.results_dir
        if base_results_dir is None:
            results_dir = workspace / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            session_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            base_results_dir = results_dir / f"anyterminal_results_{session_id}"
        else:
            session_id = base_results_dir.name
        base_results_dir.mkdir(parents=True, exist_ok=True)

        self._server = AnyTerminalServerConfig(
            run_session_id=session_id,
            base_results_dir=base_results_dir,
            model_server_url=model_url,
            model_name=model_name,
            nemo_gym_root=PARENT_DIR,
            agent_deps_dir=agent_deps_dir,
            agent_deps_archive=agent_deps_archive,
            agent_deps_url=agent_deps_url,
        )
        super().model_post_init(context)

    # Per-instance setup

    def _setup_params(
        self, body: NeMoGymResponseCreateParamsNonStreaming, rollout_id: Optional[str] = None
    ) -> AnyTerminalInstanceConfig:
        problem_info = dict(body.metadata or {})
        task_name = problem_info.get("task_name", problem_info.get("instance_id", "unknown"))

        # Fill in workdir and timeouts from task.toml/Dockerfile if not in JSONL metadata.
        task_dir = Path(problem_info["task_dir"])
        required_meta = ["workdir", "agent_timeout_sec", "verifier_timeout_sec"]
        if self.config.sandbox_provider.get("apptainer", {}).get("create", {}).get("disk_backed_overlay"):
            required_meta.append("storage_mb")
        if not all(k in problem_info for k in required_meta):
            problem_info.update({k: v for k, v in _read_task_meta(task_dir).items() if k not in problem_info})
        task_verifier_environment(task_dir, self.config.verifier_environment)

        instance_dir = f"{task_name}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        persistent_dir = self._server.base_results_dir / instance_dir
        persistent_dir.mkdir(parents=True, exist_ok=True)
        verifier_dir = persistent_dir / "verifier"
        verifier_dir.mkdir(parents=True, exist_ok=True)

        agent_run_id = f"{task_name}_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        # Per-task timeouts override config defaults when available, unless global_agent_timeout is set.
        config_overrides = {}
        if self.config.global_agent_timeout is not None:
            config_overrides["tb_agent_timeout"] = self.config.global_agent_timeout
        elif problem_info.get("agent_timeout_sec"):
            config_overrides["tb_agent_timeout"] = int(float(problem_info["agent_timeout_sec"]))
        if problem_info.get("verifier_timeout_sec"):
            config_overrides["tb_eval_timeout"] = int(float(problem_info["verifier_timeout_sec"]))

        # The container must outlive the task, or it gets torn down mid-run and the task scores as
        # a real failure instead of the infra issue it is. Mirrors anyswe_agent's derivation
        # (swebench_agent_timeout + swebench_tests_timeout + 600).
        effective_agent_timeout = config_overrides.get("tb_agent_timeout", self.config.tb_agent_timeout)
        effective_eval_timeout = config_overrides.get("tb_eval_timeout", self.config.tb_eval_timeout)
        required_ttl = effective_agent_timeout + effective_eval_timeout + 600
        if required_ttl > self.config.tb_sandbox_ttl:
            config_overrides["tb_sandbox_ttl"] = required_ttl

        server_config = self._server.model_dump()
        if not self.config.sandbox_model_base_url and rollout_id and server_config["model_server_url"]:
            server_config["model_server_url"] = apply_rollout_prefix(server_config["model_server_url"], rollout_id)

        params = AnyTerminalInstanceConfig(
            **{**self.config.model_dump(exclude={"task_network"}), **config_overrides},
            **server_config,
            problem_info=problem_info,
            task_network=(
                self.config.task_network
                if "task_network" not in problem_info
                else TaskNetwork.model_validate_json(problem_info["task_network"])
            ),
            verifier_runtime=(
                None
                if "verifier_runtime" not in problem_info
                else VerifierRuntime.model_validate_json(problem_info["verifier_runtime"])
            ),
            body=body,
            persistent_dir=persistent_dir,
            verifier_dir=verifier_dir,
            agent_run_id=agent_run_id,
            metrics_fpath=persistent_dir / "nemo_gym_metrics.json",
            container=_format_container(
                self.config.container_formatter,
                task_name,
                problem_info.get("docker_image", "ubuntu:22.04"),
                runtime_image=problem_info.get("runtime_image"),
            ),
            ray_queue_timestamp=time.time(),
            rollout_id=rollout_id,
        )
        params.metrics_fpath.write_text("{}")

        # Write instruction.txt + agent_runner.py, then resolve the in-sandbox run command.
        params.agent_command_str = GymAgentHarnessProcessor(config=params).get_run_command()

        return params

    # Request handlers

    async def _responses(
        self, body: NeMoGymResponseCreateParamsNonStreaming, rollout_id: Optional[str] = None
    ) -> NeMoGymResponse:
        params = self._setup_params(body, rollout_id)
        (params.persistent_dir / "params.json").write_text(_safe_config_json(params, indent=2))
        try:
            return await self._inner_responses(params)
        except Exception:
            tb_path = params.persistent_dir / "traceback.err"
            tb_path.write_text(format_exc())
            print(f"[{params.task_name}] exception: see {tb_path}", file=sys.stderr)
            raise

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        return await self._responses(body)

    async def _inner_responses(self, params: AnyTerminalInstanceConfig) -> NeMoGymResponse:
        payload = params.model_dump(mode="json")
        worker = importlib.import_module("responses_api_agents.anyterminal_agent.app").run_terminal_episode
        update_metrics(params.metrics_fpath, {"submitted_at": time.time()}, params.task_name)
        result = await run_sandbox_episode(
            worker,
            payload,
            _episode_sandbox_resources(params),
            episode_id=params.rollout_id,
            placement_key=params.instance_id,
        )
        return NeMoGymResponse.model_validate(result)

    async def run(self, body: AnyTerminalRunRequest) -> AnyTerminalVerifyResponse:
        async with self._sem:
            body.responses_create_params.parallel_tool_calls = True
            body.responses_create_params.tool_choice = "auto"
            response = await self._responses(body.responses_create_params, self.rollout_id_from_run(body))

            meta, response.metadata = response.metadata, None
            metrics = TerminalBenchMetrics.model_validate_json(meta["metrics"])

            return AnyTerminalVerifyResponse(
                responses_create_params=body.responses_create_params.model_dump()
                | {
                    "input": json.loads(meta["input"]),
                    "tools": [t.model_dump() for t in (response.tools or [])],
                    "model": response.model,
                },
                response=response,
                reward=1.0 if metrics.resolved else 0.0,
                done_reason="context_exhausted" if meta.get("stop_reason") == "context_exhausted" else None,
                **metrics.model_dump(),
                instance_config=AnyTerminalInstanceConfig.model_validate_json(meta["instance_config"]).model_dump(),
            )


if __name__ == "__main__":
    AnyTerminalAgent.run_webserver()
