# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run Harbor's Terminus-2 loop as a NeMo Gym agent server."""

import asyncio
import dataclasses
import errno
import importlib.util
import inspect
import json
import logging
import math
import os
import shlex
import shutil
import signal
import sys
import tempfile
from asyncio import Semaphore
from collections.abc import Mapping
from contextlib import ExitStack
from contextvars import ContextVar
from pathlib import Path
from time import time
from typing import Any, Callable, Literal, Optional
from uuid import uuid4

from fastapi import Request
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext
from openai.types.responses.response_error import ResponseError
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ObservationGap,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.terminus_2_agent.context_budget import ServedContextTerminus2
from responses_api_agents.terminus_2_agent.llm import EpisodeTerminatedError, NemoGymLLM
from responses_api_agents.terminus_2_agent.output import trajectory_to_responses


LOG = logging.getLogger(__name__)
_INTERNAL_OBSERVATIONS_KEY = "_ng_agent_observations"
_OBSERVATION_SOURCE = "terminus_2"
_ROOT_INVOCATION_ID = "root"
# Runs under the harness interpreter; see its docstring for the boundary it keeps.
COMMAND_NAMESPACE = Path(__file__).with_name("command_namespace.py")
NATIVE_COMMAND_PROCESS = Path("/native_command_process.py")
NATIVE_POLICY_BRIDGE = Path("/native_policy_bridge.py")
NATIVE_CGROUP_ROOT = Path("/sys/fs/cgroup")
TMUX_CONTROL_TIMEOUT_SECONDS = 5.0


def _message_text(item: Any) -> str:
    content = getattr(item, "content", None) if not isinstance(item, dict) else item.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "") for part in content)


# Harbor's compaction (Terminus2._summarize) asks three subagents for text that is handed to the
# next agent rather than executed: a summary, questions about it, and answers to them. The summary
# and answers calls reuse the policy history, whose first message demands the JSON action format,
# so without an explicit override the model replies with an unexecuted action.
HANDOFF_SUBAGENTS = frozenset(
    {
        "terminus-2-summarization-summary",
        "terminus-2-summarization-questions",
        "terminus-2-summarization-answers",
    }
)
HANDOFF_TEXT_CONTRACT = (
    "\n\nWrite this reply as plain text for the other agent. It is not a terminal turn and nothing "
    "in it will be executed, so do not use the JSON response format and do not include commands "
    "or keystrokes."
)


def _action_reply(content: str) -> Optional[dict[str, Any]]:
    """Return the Terminus JSON action a reply consists of, or None for a text reply."""
    start = content.find("{")
    if start < 0:
        return None
    try:
        action, _ = json.JSONDecoder().raw_decode(content[start:])
    except ValueError:
        return None
    if not isinstance(action, dict) or not {"analysis", "plan", "commands"} & action.keys():
        return None
    return action


def handoff_text(content: str, reasoning: Optional[str]) -> str:
    """The text a compaction subagent hands on: its reply, or its draft when it replied with an action."""
    action = _action_reply(content)
    if action is None:
        return content
    if reasoning and reasoning.strip():
        return reasoning.strip()
    prose = [action.get(key) for key in ("analysis", "plan")]
    return "\n\n".join(text.strip() for text in prose if isinstance(text, str) and text.strip()) or content


def _extract_instruction(body_input: Any) -> tuple[str, Optional[str]]:
    """Flatten Responses input into Terminus-2's single instruction."""
    if isinstance(body_input, str):
        return body_input, None

    messages: list[tuple[str, str]] = []
    for item in body_input or []:
        role = getattr(item, "role", None) if not isinstance(item, dict) else item.get("role")
        text = _message_text(item)
        if role and text:
            messages.append((role, text))

    conversation = [(role, text) for role, text in messages if role in {"user", "assistant"}]
    if len(conversation) <= 1 and (not conversation or conversation[0][0] == "user"):
        user_message = conversation[0][1] if conversation else ""
        system_text = "\n\n".join(text for role, text in messages if role in {"system", "developer"})
        return user_message, system_text or None
    return "\n\n".join(f"{role.title()}: {text}" for role, text in messages), None


def _require_directory(path: Path, message: str) -> None:
    """Validate directories through the kernel, avoiding fakeroot stat metadata."""
    try:
        descriptor = os.open(path, getattr(os, "O_PATH", os.O_RDONLY) | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError as error:
        if error.errno not in (errno.ENOENT, errno.ENOTDIR):
            raise
        raise FileNotFoundError(message) from error
    os.close(descriptor)


def _tmux_control_operation(command: str) -> str | None:
    # Reject executable shell syntax before shlex removes quoting/escaping
    # and treats unquoted newlines as whitespace. Quoted literals stay data.
    quote = None
    escaped = False
    for character in command:
        if escaped:
            escaped = False
        elif quote == "'":
            if character == quote:
                quote = None
        elif character == "\\":
            escaped = True
        elif character in {"$", "`"}:
            return None
        elif character == quote:
            quote = None
        elif quote is None:
            if character in {"'", '"'}:
                quote = character
            elif character in "\n;&|<>()#":
                return None
    try:
        words = shlex.split(command, comments=False, posix=True)
    except ValueError:
        return None
    return (
        words[1]
        if (
            len(words) >= 2
            and words[0] == "tmux"
            and words[1] in {"capture-pane", "send-keys", "has-session", "list-panes"}
        )
        else None
    )


class LocalEnvironment:
    """Execute Harbor environment commands in one local workspace."""

    def __init__(
        self, cwd: Path, command_timeout_sec: float, tmux_control_timeout_sec: float = TMUX_CONTROL_TIMEOUT_SECONDS
    ) -> None:
        self.cwd = cwd.resolve()
        self.command_timeout_sec = command_timeout_sec
        self.tmux_control_timeout_sec = tmux_control_timeout_sec
        self._memory_budget_events = []
        self._memory_control = None
        if not math.isfinite(tmux_control_timeout_sec) or tmux_control_timeout_sec <= 0:
            raise ValueError("tmux control timeout must be finite and positive")
        # Standalone Gym has no native pressure controller. An admitted native
        # policy binding requires its paired helper; missing/import errors
        # still propagate rather than silently disabling native recovery.
        if NATIVE_COMMAND_PROCESS.exists() or NATIVE_POLICY_BRIDGE.exists():
            spec = importlib.util.spec_from_file_location("native_command_process", NATIVE_COMMAND_PROCESS)
            if spec is None or spec.loader is None:
                raise ValueError("tmux reclaim recovery requires the admitted native command helper")
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            self._memory_control = helper.MemoryReclaimControl
        # Apptainer instances can share /tmp. Unique session names do not
        # isolate the tmux server: later sessions inherit its first namespace.
        self._tmux_directory = tempfile.TemporaryDirectory(prefix="gym-tmux-")

    async def close(self, *, terminate: bool = True) -> None:
        if self._tmux_directory is None:
            return
        try:
            # exec always selects this environment's private server, including
            # when the caller itself is running inside another tmux session.
            if terminate:
                await self.exec("tmux kill-server", cwd="/", timeout_sec=10)
        finally:
            self._tmux_directory.cleanup()
            self._tmux_directory = None

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | float | None = None,
        _control_retry: bool = True,
    ) -> ExecResult:
        if self._tmux_directory is None:
            raise RuntimeError("Terminus-2 environment is closed")
        original_command = command
        control_op = _tmux_control_operation(command)
        boundary = self._memory_control(NATIVE_CGROUP_ROOT) if control_op and self._memory_control else None
        run_cwd = Path(cwd).resolve() if cwd else self.cwd
        _require_directory(run_cwd, f"Terminus-2 working directory does not exist: {run_cwd}")

        process_env = os.environ.copy()
        if env:
            process_env.update({str(key): str(value) for key, value in env.items()})
        process_env.pop("TMUX", None)
        process_env.pop("TMUX_PANE", None)
        process_env["TMUX_TMPDIR"] = self._tmux_directory.name
        # A task image's login profile may replace PATH. Keep the admitted
        # agent runtime (including tmux) visible after bash reads that profile.
        command = f"export PATH={shlex.quote(process_env.get('PATH', ''))}; {command}"

        with ExitStack() as streams:
            # tmux passes output descriptors to its server through SCM_RIGHTS.
            # A stopped server can retain a pipe in that socket queue even after
            # the timed-out client is killed. Files let control cleanup reap
            # its client without waiting for the unresponsive server's EOF.
            control_stdout = streams.enter_context(tempfile.TemporaryFile()) if control_op is not None else None
            control_stderr = streams.enter_context(tempfile.TemporaryFile()) if control_op is not None else None
            proc = await asyncio.create_subprocess_exec(
                "bash",
                "-lc",
                command,
                cwd=str(run_cwd),
                env=process_env,
                stdout=control_stdout if control_stdout is not None else asyncio.subprocess.PIPE,
                stderr=control_stderr if control_stderr is not None else asyncio.subprocess.PIPE,
                start_new_session=True,
            )

            async def communicate():
                stdout, stderr = await proc.communicate()
                if control_stdout is not None:
                    control_stdout.seek(0)
                    control_stderr.seek(0)
                    return control_stdout.read(), control_stderr.read()
                return stdout, stderr

            timeout = self.command_timeout_sec if timeout_sec is None else float(timeout_sec)
            if control_op is not None:
                timeout = min(timeout, self.tmux_control_timeout_sec)
            try:
                stdout, stderr = await asyncio.wait_for(communicate(), timeout=timeout)
            except asyncio.CancelledError:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await communicate()
                raise
            except asyncio.TimeoutError:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = await communicate()
                if boundary is not None and _control_retry:
                    event = boundary.recover()
                    if event is not None:
                        self._memory_budget_events.append(event)
                        LOG.warning("Terminus tool resource feedback: %s", json.dumps(event, sort_keys=True))
                        return await self.exec(
                            command=original_command, cwd=cwd, env=env, timeout_sec=timeout_sec, _control_retry=False
                        )
                message = stderr.decode(errors="replace")
                message += f"\nCommand timed out after {timeout:g} seconds"
                return ExecResult(stdout=stdout.decode(errors="replace"), stderr=message.strip(), return_code=124)

            text = stdout.decode(errors="replace")
            if control_op == "capture-pane" and self._memory_budget_events:
                text = (
                    "\n".join(json.dumps(event, sort_keys=True) for event in self._memory_budget_events) + "\n" + text
                )
                self._memory_budget_events.clear()
            return ExecResult(
                stdout=text,
                stderr=stderr.decode(errors="replace"),
                return_code=int(proc.returncode or 0),
            )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path).resolve()
        target = Path(target_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if source != target:
            await asyncio.to_thread(shutil.copy2, source, target)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self.upload_file(source_path, str(target_path))


SETUP_FAILED_TMUX_UNAVAILABLE = "setup_failed:tmux_unavailable"
_SETUP_CAUSE_MAX_CHARS = 200
_SETUP_ERRORS: ContextVar[Optional[list[str]]] = ContextVar("terminus_2_setup_errors", default=None)


class _SetupErrorRecorder(logging.Handler):
    """Keep the first ERROR line Harbor logs while this task's Terminus setup runs.

    Harbor's tmux installer logs its failure and returns. The recorder keeps only
    the first nonempty cause (its first line, bounded) in the running task's own
    list, so concurrent setups never see each other's causes.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)

    def emit(self, record: logging.LogRecord) -> None:
        errors = _SETUP_ERRORS.get()
        if errors is None or errors:
            return
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        cause = next((line.strip() for line in message.splitlines() if line.strip()), "")
        if cause:
            errors.append(cause[:_SETUP_CAUSE_MAX_CHARS])


logging.getLogger("harbor.utils.logger").addHandler(_SetupErrorRecorder())


class TerminusSetupError(RuntimeError):
    """Terminus-2 could not start its tmux session: an infrastructure setup failure."""

    def __init__(self, label: str, first_cause: str, detail: str) -> None:
        self.label, self.first_cause = label, first_cause
        super().__init__(f"{label}: first cause: {first_cause}; {detail}")


class LocalTmuxSession(TmuxSession):
    """Use Harbor's Linux launcher and a BSD-compatible macOS launcher."""

    async def start(self) -> None:
        await super().start()
        # Linux script can return zero even when its child tmux failed. Prove
        # the session exists before allowing model calls against an empty pane.
        result = await self.environment.exec(
            f"tmux display-message -p -t {shlex.quote(self._session_name)} '#{{socket_path}}'"
        )
        if result.return_code != 0 or not result.stdout.strip():
            errors = _SETUP_ERRORS.get() or []
            detail = (result.stderr.strip().splitlines() or [""])[0][:_SETUP_CAUSE_MAX_CHARS]
            raise TerminusSetupError(
                SETUP_FAILED_TMUX_UNAVAILABLE,
                errors[0] if errors else detail,
                f"Terminus-2 tmux session is unavailable: {detail}",
            )
        socket = Path(result.stdout.strip())
        private_root = Path(self.environment._tmux_directory.name)
        if not socket.is_relative_to(private_root):
            raise RuntimeError(f"Terminus-2 tmux socket is outside its private directory: {socket}")
        self._logger.info("Terminus-2 tmux session=%s socket=%s", self._session_name, socket)

    @property
    def _tmux_start_session(self) -> str:
        if sys.platform != "darwin":
            # The tmux server, and every command the agent types into it, starts
            # in its own PID namespace as the preferred OOM victim: the agent's
            # kill commands and the task's memory use cannot end this harness.
            return (
                f"{shlex.quote(sys.executable)} -I {shlex.quote(str(COMMAND_NAMESPACE))} "
                f"bash -c {shlex.quote(super()._tmux_start_session)}"
            )

        session_name = shlex.quote(self._session_name)
        login_shell = shlex.quote("bash --login")
        pipe_command = shlex.quote(f"cat > {shlex.quote(str(self._logging_path))}")
        return (
            "export TERM=xterm-256color && "
            "export SHELL=/bin/bash && "
            f"tmux new-session -x {self._pane_width} -y {self._pane_height} -d "
            f"-s {session_name} {login_shell} \\; pipe-pane -t {session_name} {pipe_command}"
        )


class StandaloneTerminus2(ServedContextTerminus2):
    """Run Terminus-2 in a unique tmux session."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._compactions: list[ContextCompactionObservation] = []
        self._compaction_trigger = "context_length_exceeded"

    async def _check_proactive_summarization(self, *args: Any, **kwargs: Any) -> Any:
        self._compaction_trigger = "proactive_free_tokens"
        try:
            return await super()._check_proactive_summarization(*args, **kwargs)
        finally:
            self._compaction_trigger = "context_length_exceeded"

    async def _summarize(self, chat: Any, original_instruction: str, *args: Any, **kwargs: Any) -> Any:
        """Record each context summarization Harbor performs as a compaction event.

        Harbor owns the three-call hand-off, so this only observes it: ``tokens_before`` is the chat's
        token count when the hand-off starts (the served count, which includes hidden reasoning, once
        the endpoint has reported usage) and ``summary`` is the hand-off prompt the next context receives.
        A failed or cancelled hand-off is recorded as such; an empty chat is not compacted.
        """
        if not chat.messages:
            return await super()._summarize(chat, original_instruction, *args, **kwargs)
        tokens_before = None
        try:
            counted = self._count_total_tokens(chat)
            tokens_before = await counted if inspect.isawaitable(counted) else counted
        except Exception:
            LOG.debug("Terminus-2 could not count tokens before a compaction", exc_info=True)
        outcome: Literal["completed", "failed", "aborted"] = "failed"
        summary = None
        try:
            handoff, subagent_refs = await super()._summarize(chat, original_instruction, *args, **kwargs)
            outcome, summary = "completed", handoff
            return handoff, subagent_refs
        except asyncio.CancelledError:
            outcome = "aborted"
            raise
        finally:
            self._compactions.append(
                ContextCompactionObservation(
                    invocation_id=_ROOT_INVOCATION_ID,
                    observed_at=time(),
                    trigger=self._compaction_trigger,
                    tokens_before=tokens_before if type(tokens_before) is int and tokens_before >= 0 else None,
                    outcome=outcome,
                    summary=summary if isinstance(summary, str) else None,
                )
            )

    def observations(self) -> AgentObservationBundle:
        """The compactions this run performed, with what the harness cannot report as explicit gaps."""
        codes = ["model_call_ownership_unavailable"]
        if self._compactions:
            codes += ["compaction_model_call_boundary_unavailable", "compaction_tokens_after_unavailable"]
        gaps = [ObservationGap(code=code, invocation_id=_ROOT_INVOCATION_ID) for code in codes]
        return AgentObservationBundle(
            source=_OBSERVATION_SOURCE,
            records=[AgentInvocation(invocation_id=_ROOT_INVOCATION_ID), *self._compactions],
            gaps=gaps,
        )

    async def _handle_llm_interaction(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Track Terminus-2's consecutive two-response completion handshake."""
        result = await super()._handle_llm_interaction(*args, **kwargs)
        if result[1]:
            self._consecutive_completion_claims = getattr(self, "_consecutive_completion_claims", 0) + 1
        else:
            self._consecutive_completion_claims = 0
        return result

    @property
    def finished_naturally(self) -> bool:
        return getattr(self, "_consecutive_completion_claims", 0) >= 2

    async def _run_subagent(self, prompt: str, *args: Any, agent_name: str, **kwargs: Any) -> tuple[Any, ...]:
        """Ask compaction subagents for plain text and hand on text, never an unexecuted action.

        The request keeps harbor's history (and so the policy row's prefix); only its last user
        message gains the text contract. The trajectory records the reply as generated.
        """
        if agent_name not in HANDOFF_SUBAGENTS:
            return await super()._run_subagent(prompt, *args, agent_name=agent_name, **kwargs)
        response, trajectory_ref = await super()._run_subagent(
            prompt + HANDOFF_TEXT_CONTRACT, *args, agent_name=agent_name, **kwargs
        )
        text = handoff_text(response.content or "", response.reasoning_content)
        return dataclasses.replace(response, content=text), trajectory_ref

    async def run(self, instruction: str, environment: LocalEnvironment, context: AgentContext) -> None:
        """Keep completed turns when Terminus fails so Gym can score the partial trajectory."""
        self.context_length_exceeded = False
        self.agent_error = None
        self.model_transport_error = None
        try:
            session = getattr(self, "_session", None)
            if session is not None and not await session.is_session_alive():
                # Harbor ends its loop at once on a dead session, with no model call
                # and no error: name it as a setup failure with the setup's first cause.
                self.agent_error = SETUP_FAILED_TMUX_UNAVAILABLE
                causes = getattr(self, "_setup_errors", None) or ["tmux session is not alive after setup"]
                self.logger.warning("Agent setup failed: %s: %s", self.agent_error, causes[0])
            await super().run(instruction, environment, context)
        except EpisodeTerminatedError:
            pass  # A policy stop keeps the completed trajectory for grading.
        except Exception as exc:
            # The partial trajectory is still scored; name the cause so a model
            # or transport failure is not indistinguishable from a policy miss.
            self.agent_error = self.agent_error or type(exc).__name__
            self.logger.warning("Agent error: %s: %s. Returning completed turns.", type(exc).__name__, exc)
        finally:
            self._attach_routed_experts_to_trajectory()
            llm = getattr(self, "_llm", None)
            if isinstance(llm, NemoGymLLM):
                self.context_length_exceeded = llm.context_length_exceeded
                self.model_transport_error = getattr(llm, "model_transport_error", None)
                try:
                    await llm.aclose()
                except Exception as exc:
                    self.logger.warning("Model client close failed: %s: %s", type(exc).__name__, exc)

    def _attach_routed_experts_to_trajectory(self) -> None:
        llm = getattr(self, "_llm", None)
        if not isinstance(llm, NemoGymLLM):
            return

        modified = False
        for step in getattr(self, "_trajectory_steps", []):
            if getattr(step, "source", None) != "agent" or step.metrics is None:
                continue
            routed_experts = llm.pop_routed_experts_for_rollout_details(
                step.metrics.prompt_token_ids,
                step.metrics.completion_token_ids,
                step.metrics.logprobs,
            )
            if routed_experts is None:
                continue
            extra = step.metrics.extra or {}
            extra["routed_experts"] = routed_experts
            step.metrics.extra = extra
            modified = True

        if modified:
            self._dump_trajectory()

    async def setup(self, environment: LocalEnvironment) -> None:
        self._setup_errors = []
        token = _SETUP_ERRORS.set(self._setup_errors)
        try:
            await self._setup_session(environment)
        finally:
            _SETUP_ERRORS.reset(token)

    async def _setup_session(self, environment: LocalEnvironment) -> None:
        self._consecutive_completion_claims = 0
        self._standalone_session_name = f"terminus-2-{uuid4().hex[:12]}"
        recording_path = self.logs_dir / "recording.cast" if self._record_terminal_session else None
        self._session = LocalTmuxSession(
            session_name=self._standalone_session_name,
            environment=environment,
            logging_path=self.logs_dir / "terminus_2.pane",
            local_asciinema_recording_path=recording_path,
            remote_asciinema_recording_path=recording_path,
            pane_width=self._tmux_pane_width,
            pane_height=self._tmux_pane_height,
        )
        await self._session.start()

    async def close(self, environment: LocalEnvironment, *, terminate: bool = True) -> None:
        if self._session is not None and terminate:
            try:
                await self._session.stop()
            except Exception as exc:
                self.logger.warning("Could not stop Terminus-2 recording cleanly: %s", exc)
        session_name = getattr(self, "_standalone_session_name", None)
        if session_name and terminate:
            await environment.exec(
                f"tmux kill-session -t {shlex.quote(session_name)}",
                cwd="/",
                timeout_sec=30,
            )


class Terminus2AgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    concurrency: int = 1
    model: Optional[str] = None
    workspace_root: Optional[str] = None
    system_prompt: Optional[str] = None
    max_turns: Optional[int] = None
    parser_name: Literal["json", "xml"] = "json"
    temperature: float = 0.7
    reasoning_effort: Optional[Literal["none", "minimal", "low", "medium", "high", "default"]] = None
    collect_rollout_details: bool = False
    enable_summarize: bool = True
    proactive_summarization_threshold: int = 8000
    max_thinking_tokens: Optional[int] = None
    model_info: dict[str, Any] = Field(
        default_factory=lambda: {
            "max_input_tokens": 262144,
            "max_output_tokens": 81920,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        }
    )
    trajectory_config: dict[str, Any] = Field(default_factory=lambda: {"raw_content": False})
    tmux_pane_width: int = 160
    tmux_pane_height: int = 40
    store_all_messages: bool = False
    record_terminal_session: bool = False
    teardown_tmux: bool = True
    interleaved_thinking: bool = True
    model_timeout_sec: float = 2400
    command_timeout_sec: float = 1800
    timeout: float = 10800
    keep_logs: bool = False
    logs_root: str = "outputs/terminus_2_agent/runs"


class Terminus2AgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class Terminus2AgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    agent_timeout_error: int = 0
    context_length_exceeded_error: int = 0
    agent_failed: bool = False
    agent_error: Optional[str] = None
    ng_agent_observations: Optional[AgentObservationBundle] = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class Terminus2Agent(SimpleResponsesAPIAgent):
    """NeMo Gym Responses API wrapper for the Terminus-2 terminal loop."""

    config: Terminus2AgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    def _logs_dir(self) -> Path:
        if not self.config.keep_logs:
            return Path(tempfile.mkdtemp(prefix="nemo-gym-terminus-2-"))
        root = Path(self.config.logs_root).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        path = root / f"run-{int(time())}-{uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _model_name(self, body: NeMoGymResponseCreateParamsNonStreaming) -> str:
        global_model = getattr(self.server_client, "global_config_dict", {}).get("policy_model_name")
        return self.config.model or body.model or global_model or "model"

    def _rollout_id(self, request: Optional[Request]) -> Optional[str]:
        path_params = getattr(request, "path_params", None)
        return path_params.get("rollout_id") if isinstance(path_params, Mapping) else None

    def _build_agent(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        logs_dir: Path,
        api_base: str,
    ) -> StandaloneTerminus2:
        temperature = body.temperature if body.temperature is not None else self.config.temperature
        model_name = self._model_name(body)
        llm = NemoGymLLM(
            model_name=model_name,
            api_base=api_base,
            collect_rollout_details=self.config.collect_rollout_details,
            model_info=self.config.model_info,
            responses_create_params=body.model_dump(exclude_none=True),
            timeout_sec=self.config.model_timeout_sec,
        )
        return StandaloneTerminus2(
            logs_dir=logs_dir,
            model_name=model_name,
            max_turns=self.config.max_turns,
            parser_name=self.config.parser_name,
            api_base=api_base,
            temperature=temperature,
            reasoning_effort=self.config.reasoning_effort,
            collect_rollout_details=self.config.collect_rollout_details,
            enable_summarize=self.config.enable_summarize,
            proactive_summarization_threshold=self.config.proactive_summarization_threshold,
            max_thinking_tokens=self.config.max_thinking_tokens,
            model_info=self.config.model_info,
            trajectory_config=self.config.trajectory_config,
            tmux_pane_width=self.config.tmux_pane_width,
            tmux_pane_height=self.config.tmux_pane_height,
            store_all_messages=self.config.store_all_messages,
            record_terminal_session=self.config.record_terminal_session,
            interleaved_thinking=self.config.interleaved_thinking,
            llm=llm,
        )

    async def _run_terminus(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        instruction: str,
        api_base: str,
        observation_collector: Optional[Callable[[AgentObservationBundle], None]] = None,
    ) -> tuple[dict[str, Any], AgentContext, dict[str, Any], bool, bool]:
        logs_dir = self._logs_dir()
        temporary_workspace: Optional[Path] = None
        try:
            if self.config.workspace_root:
                workspace = Path(self.config.workspace_root).expanduser().resolve()
                _require_directory(workspace, f"Terminus-2 workspace does not exist: {workspace}")
            else:
                workspace = temporary_workspace = Path(tempfile.mkdtemp(prefix="nemo-gym-terminus-2-workspace-"))

            context = AgentContext()
            agent = self._build_agent(body, logs_dir, api_base)
            environment = LocalEnvironment(workspace, self.config.command_timeout_sec)
            timed_out = False
            try:
                await agent.setup(environment)
                try:
                    await asyncio.wait_for(agent.run(instruction, environment, context), timeout=self.config.timeout)
                except asyncio.TimeoutError:
                    timed_out = True
                    LOG.warning("Terminus-2 timed out after %gs", self.config.timeout)
            finally:
                if observation_collector is not None:
                    try:
                        observation_collector(agent.observations())
                    except Exception:
                        LOG.exception("failed to return Terminus-2 observations")
                teardown_kwargs = {} if self.config.teardown_tmux else {"terminate": False}
                try:
                    await agent.close(environment, **teardown_kwargs)
                except Exception as error:
                    LOG.warning("Terminus-2 agent teardown failed (ignored): %s", error)
                try:
                    await environment.close(**teardown_kwargs)
                except Exception as error:
                    LOG.warning("Terminus-2 environment teardown failed (ignored): %s", error)

            trajectory_path = logs_dir / "trajectory.json"
            trajectory = json.loads(trajectory_path.read_text()) if trajectory_path.exists() else {"steps": []}
            flags = {
                "context_length_exceeded": agent.context_length_exceeded,
                "agent_error": getattr(agent, "agent_error", None),
                "model_transport_error": getattr(agent, "model_transport_error", None),
                "stop_reason": agent._llm.terminal_stop_reason,
            }
            return trajectory, context, flags, timed_out, agent.finished_naturally
        finally:
            if not self.config.keep_logs:
                shutil.rmtree(logs_dir, ignore_errors=True)
            if temporary_workspace is not None:
                shutil.rmtree(temporary_workspace, ignore_errors=True)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if body.temperature is None:
            body.temperature = self.config.temperature
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_instruction, input_system = _extract_instruction(body.input)
        instruction_parts = [part for part in (self.config.system_prompt, input_system, user_instruction) if part]
        instruction = "\n\n".join(instruction_parts)

        api_base = self.resolve_model_base_url(
            self.config.model_server.name,
            self._rollout_id(request),
        )
        rollout_id = self._rollout_id(request)
        observations: Optional[AgentObservationBundle] = None

        def collect(bundle: AgentObservationBundle) -> None:
            nonlocal observations
            observations = bundle

        # Only a rollout-scoped request returns observations, as for every other harness.
        run_kwargs = {"observation_collector": collect} if isinstance(rollout_id, str) else {}
        async with self.sem:
            trajectory, context, flags, timed_out, finished_naturally = await self._run_terminus(
                body, instruction, api_base, **run_kwargs
            )
        output_items = trajectory_to_responses(trajectory)
        if not output_items:
            output_items = [
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                ).model_dump()
            ]

        # The agent itself ended the run with its confirmed task_complete, not a
        # timeout, a failure or an endpoint stop (turn cap, guard, context).
        finished_naturally = bool(
            finished_naturally
            and not timed_out
            and not flags.get("agent_error")
            and not flags.get("model_transport_error")
            and not flags.get("stop_reason")
            and not flags.get("context_length_exceeded")
        )
        final_metrics = trajectory.get("final_metrics", {})
        input_tokens = final_metrics.get("total_prompt_tokens", 0)
        output_tokens = final_metrics.get("total_completion_tokens", 0)
        cached_tokens = final_metrics.get("total_cached_tokens", 0)
        if input_tokens == 0 and output_tokens == 0:
            input_tokens = context.n_input_tokens or 0
            output_tokens = context.n_output_tokens or 0
            cached_tokens = context.n_cache_tokens or 0

        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self._model_name(body),
            object="response",
            output=output_items,
            parallel_tool_calls=False,
            temperature=body.temperature,
            top_p=body.top_p,
            tool_choice=body.tool_choice,
            tools=body.tools,
            background=False,
            reasoning={"effort": None, "generate_summary": None, "summary": None},
            service_tier="default",
            status="failed" if flags.get("model_transport_error") else "completed",
            error=ResponseError(code="server_error", message=flags["model_transport_error"])
            if flags.get("model_transport_error")
            else None,
            text={"format": {"type": "text"}, "verbosity": "medium"},
            top_logprobs=0,
            truncation="disabled",
            store=True,
            usage=NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=cached_tokens),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=input_tokens + output_tokens,
            ),
            metadata={
                "terminus_2": json.dumps(
                    {
                        "context": context.model_dump(mode="json"),
                        "agent_error_flags": flags,
                        "timed_out": timed_out,
                        "finished_naturally": finished_naturally,
                    }
                ),
                **({"model_transport_error": "true"} if flags.get("model_transport_error") else {}),
                **({"stop_reason": flags["stop_reason"]} if flags.get("stop_reason") else {}),
                # Harness-neutral labels a wrapping agent (anyterminal) carries.
                **({"agent_error": flags["agent_error"]} if flags.get("agent_error") else {}),
                "finished_naturally": "true" if finished_naturally else "false",
            },
        )
        if observations is None:
            return response
        return response.model_copy(update={_INTERNAL_OBSERVATIONS_KEY: observations.model_dump(mode="json")})

    async def run(self, request: Request, body: Terminus2AgentRunRequest) -> Terminus2AgentVerifyResponse:
        cookies = request.cookies
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_response)
        cookies = seed_response.cookies

        agent_response = await self.server_client.post(
            server_name=self.config.name,
            url_path=self.url_path_for_run("/v1/responses", body),
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(agent_response)
        cookies = agent_response.cookies
        response_json = await get_response_json(agent_response)
        raw_observations = response_json.pop(_INTERNAL_OBSERVATIONS_KEY, None)
        observations = (
            AgentObservationBundle.model_validate(raw_observations) if isinstance(raw_observations, dict) else None
        )

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=body.model_dump() | {"response": response_json},
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        verify_json = await get_response_json(verify_response)

        gym_response = NeMoGymResponse.model_validate(response_json)
        turns = sum(
            1
            for item in gym_response.output
            if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
        )
        terminus_metadata_json = (gym_response.metadata or {}).get("terminus_2", "{}")
        try:
            terminus_metadata = json.loads(terminus_metadata_json)
        except (json.JSONDecodeError, TypeError):
            terminus_metadata = {}
        flags = terminus_metadata.get("agent_error_flags", {})
        timed_out = bool(terminus_metadata.get("timed_out"))

        return Terminus2AgentVerifyResponse.model_validate(
            verify_json
            | {
                "turns_used": turns,
                "finished_naturally": (gym_response.metadata or {}).get("finished_naturally") == "true",
                "agent_timeout_error": int(timed_out),
                "context_length_exceeded_error": int(flags.get("context_length_exceeded", False)),
                "agent_failed": bool(flags.get("agent_error")),
                "agent_error": flags.get("agent_error"),
                **({"ng_agent_observations": observations} if observations is not None else {}),
            }
        )


if __name__ == "__main__":
    Terminus2Agent.run_webserver()
