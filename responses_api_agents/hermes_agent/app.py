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
import asyncio
import atexit
import json
import logging
import os
import shutil
import sys
import tempfile
from asyncio import Semaphore
from collections.abc import Mapping
from contextvars import ContextVar
from time import time
from typing import Any, Callable, Optional
from uuid import uuid4

import model_tools  # noqa: F401  # fail-fast if hermes-agent isn't installed  # pyright: ignore[reportMissingImports]
from fastapi import Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.rollout_observability import (
    AgentEpisode,
    AgentObservationBundle,
    ObservationGap,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.hermes_agent.observability import HermesAgentObserver


def _trajectory_to_output_items(messages, n_input):
    output_items = []
    for item in messages[n_input:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content", "") or ""
        if isinstance(content, list):
            content = "".join(c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "") for c in content)
        if role == "assistant":
            reasoning_text = item.get("reasoning") or ""
            if reasoning_text:
                content = ResponsesConverter._parse_think_tags(content)[1]
                output_items.append(
                    NeMoGymResponseReasoningItem(
                        id=f"rsn-{len(output_items)}",
                        summary=[NeMoGymSummary(type="summary_text", text=reasoning_text)],
                        type="reasoning",
                    )
                )
            output_items.append(
                NeMoGymResponseOutputMessageForTraining(
                    id=f"msg-{len(output_items)}",
                    content=[NeMoGymResponseOutputText(type="output_text", text=content, annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                    prompt_token_ids=item.get("prompt_token_ids") or [],
                    generation_token_ids=item.get("generation_token_ids") or [],
                    generation_log_probs=item.get("generation_log_probs") or [],
                    routed_experts=item.get("routed_experts"),
                )
            )
            for tc in item.get("tool_calls") or []:
                fn = tc.get("function") if isinstance(tc, dict) else None
                if not fn:
                    continue
                output_items.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=fn.get("arguments", ""),
                        call_id=tc.get("id", ""),
                        name=fn.get("name", ""),
                        type="function_call",
                        id=tc.get("id"),
                        status="completed",
                    )
                )
        elif role == "tool":
            output_items.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=item.get("tool_call_id", ""),
                    output=content,
                    status="completed",
                )
            )
    return output_items


LOG = logging.getLogger(__name__)
_INTERNAL_OBSERVATIONS_KEY = "_ng_agent_observations"


# if ray close sys.stderr mid-request, write to the original fd
class _SafeStderrHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = sys.__stderr__
            if stream is None:
                return
            stream.write(msg + "\n")
            stream.flush()
        except Exception:
            pass


if not LOG.handlers:
    LOG.addHandler(_SafeStderrHandler(level=logging.WARNING))


def _split_input_to_user_and_history(input_items) -> tuple[str, list[dict], Optional[str]]:
    items = list(input_items)
    system_message: Optional[str] = None
    if items:
        first = items[0]
        first_role = getattr(first, "role", None) or (first.get("role") if isinstance(first, dict) else None)
        first_content = getattr(first, "content", None) or (first.get("content") if isinstance(first, dict) else None)
        if first_role == "system":
            if isinstance(first_content, list):
                first_content = "".join(
                    (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in first_content
                )
            system_message = first_content or ""
            items = items[1:]

    user_message = ""
    history: list[dict] = []
    for idx, item in enumerate(items):
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
        if isinstance(content, list):
            content = "".join((p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content)
        content = content or ""
        if idx == len(items) - 1 and role == "user":
            user_message = content
        else:
            history.append({"role": role, "content": content})
    return user_message, history, system_message


def _execute_every_tool_call(agent: Any) -> None:
    """Run every tool call the policy emits in a turn.

    Hermes' post-call guardrails drop duplicate (name, arguments) pairs and
    delegate_task calls past MAX_CONCURRENT_CHILDREN with only a log line; the
    dropped calls get no tool result and leave the served turn. Every emitted
    call executes instead, in Hermes' own sequential or parallel-safe order.
    """
    agent._deduplicate_tool_calls = _keep_tool_calls
    agent._cap_delegate_task_calls = _keep_tool_calls


def _keep_tool_calls(tool_calls: list) -> list:
    return tool_calls


MAX_TOOL_RESULT_CHARS = 20_000


def _limit_tool_results(agent: Any) -> None:
    """Bound each new tool observation before Hermes sends its next request.

    The pinned Hermes executor has a larger, hardcoded result limit and no
    setting for it. Wrap its common dispatcher so both execution branches keep
    their ordering, errors and tool identities. Never rewrite earlier history.
    """
    execute_tool_calls = agent._execute_tool_calls

    def execute_then_limit(assistant_message, messages, *args, **kwargs):
        first_new_message = len(messages)
        try:
            return execute_tool_calls(assistant_message, messages, *args, **kwargs)
        finally:
            for message in messages[first_new_message:]:
                content = message.get("content")
                if message.get("role") != "tool" or not isinstance(content, str):
                    continue
                if len(content) > MAX_TOOL_RESULT_CHARS:
                    note = (
                        f"\n\n[Truncated: tool result was {len(content):,} characters; "
                        f"limit {MAX_TOOL_RESULT_CHARS:,} characters including this note. "
                        "Use a narrower search or read range.]"
                    )
                    message["content"] = content[: MAX_TOOL_RESULT_CHARS - len(note)] + note

    agent._execute_tool_calls = execute_then_limit


# OpenHands' built-in `finish` schema (openhands/sdk/tool/builtins/finish.py), so a
# harness mix offers one completion contract.
FINISH_TOOL_SCHEMA = {
    "name": "finish",
    "description": (
        "Signals the completion of the current task or conversation.\n\n"
        "Use this tool when:\n"
        "- You have successfully completed the user's requested task\n"
        "- You cannot proceed further due to technical limitations or missing information\n\n"
        "The message should include:\n"
        "- A clear summary of actions taken and their results\n"
        "- Any next steps for the user\n"
        "- Explanation if you're unable to complete the task\n"
        "- Any follow-up questions if more information is needed\n"
    ),
    "parameters": {
        "type": "object",
        "properties": {"message": {"type": "string", "description": "Final message to send to the user."}},
        "required": ["message"],
    },
}


def _finish_result(args: dict, **_: Any) -> str:
    return str(args.get("message", ""))


def _offer_finish_tool(agent: Any) -> Callable[[], bool]:
    """Offer `finish` and end the run after the turn that calls it.

    Hermes ends a run on a turn without tool calls. `finish` is the explicit
    form of that stop: Hermes runs it through its registry like any tool, every
    call in the turn still executes, and the loop then stops at its own
    between-turn interrupt check, before another model request. A finish on the
    last allowed turn skips Hermes' extra iteration-limit summary request.
    Returns whether the run was finished this way.
    """
    from tools.registry import registry  # pyright: ignore[reportMissingImports]

    registry.register(name="finish", toolset="finish", schema=FINISH_TOOL_SCHEMA, handler=_finish_result)
    agent.tools = [*(agent.tools or []), {"type": "function", "function": FINISH_TOOL_SCHEMA}]
    agent.valid_tool_names = {*(agent.valid_tool_names or ()), "finish"}
    execute_tool_calls = agent._execute_tool_calls
    handle_max_iterations = agent._handle_max_iterations
    message: Optional[str] = None

    def execute_then_finish(assistant_message, *args, **kwargs):
        nonlocal message
        result = execute_tool_calls(assistant_message, *args, **kwargs)
        for call in assistant_message.tool_calls or []:
            if call.function.name == "finish":
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                message = _finish_result(arguments if isinstance(arguments, dict) else {})
                agent._interrupt_requested = True
        return result

    def summary_unless_finished(*args, **kwargs):
        return message if message is not None else handle_max_iterations(*args, **kwargs)

    agent._execute_tool_calls = execute_then_finish
    agent._handle_max_iterations = summary_unless_finished
    return lambda: message is not None


def _endpoint_stop(response: Any) -> bool:
    """The serving endpoint's episode stop: a stop reason with no sampled tokens.

    The endpoint answers its turn cap, an episode guard or an exhausted context
    with an empty assistant message, which Hermes takes as a final answer.
    Terminus-2 reads the same contract (``terminus_2_agent/llm.py``).
    """
    choices = getattr(response, "choices", None) or []
    stop_reason = getattr(choices[0], "stop_reason", None) if choices else None
    return bool(stop_reason) and getattr(getattr(response, "usage", None), "completion_tokens", None) == 0


def _observe_endpoint_stop(agent: Any) -> Callable[[], bool]:
    """Record whether the latest main-loop model response was the endpoint's episode stop."""
    api_call = getattr(agent, "_interruptible_api_call", None)
    if api_call is None:
        # An unobservable stop counts as the endpoint's, so completion is never claimed.
        return lambda: True
    stopped = False

    def call_then_observe(*args, **kwargs):
        nonlocal stopped
        stopped = False
        response = api_call(*args, **kwargs)
        stopped = _endpoint_stop(response)
        return response

    agent._interruptible_api_call = call_then_observe
    return lambda: stopped


def _finished_naturally(
    result: dict,
    *,
    finished_by_tool: bool,
    ended_by_endpoint: bool,
    interrupted: bool,
    failed: bool,
) -> bool:
    """Whether the policy itself ended the run: a final answer, or the offered `finish` tool.

    Hermes' ``completed`` also covers the endpoint's empty episode stop and the
    synthetic error reply it appends near its turn limit. Only a final assistant
    message Hermes built from a policy response carries ``finish_reason``; that
    error reply and its iteration-limit summary do not.
    """
    if interrupted or failed or ended_by_endpoint or result.get("partial"):
        return False
    if finished_by_tool:
        return True
    messages = result.get("messages") or []
    last = messages[-1] if messages else None
    return bool(
        result.get("completed")
        and isinstance(last, dict)
        and last.get("role") == "assistant"
        and "finish_reason" in last
        and not last.get("tool_calls")
    )


class HermesAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    model: Optional[str] = None
    concurrency: int = 32
    max_turns: int = 90
    max_tokens: Optional[int] = None
    context_window: int | None = Field(default=None, gt=0)
    enabled_toolsets: Optional[list[str]] = None
    disabled_toolsets: Optional[list[str]] = None
    temperature: float | None = None
    terminal_backend: str = "local"
    terminal_timeout: int = 180
    api_timeout: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    system_prompt: Optional[str] = None
    compression_enabled: bool = True
    # Stock Hermes compresses at this fraction of the model context
    # (hermes_cli/config.py DEFAULT_CONFIG compression.threshold).
    compression_threshold: float = Field(default=0.50, gt=0, le=1)
    # Temperature of the compression summary request. None samples it like the
    # run's policy turns, never at Hermes' own 0.3 (context_compressor.py).
    compression_summary_temperature: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    # Hermes' compression.target_ratio: the kept recent tail as a fraction of the threshold.
    # None keeps Hermes' default (0.20); Hermes clamps it to [0.10, 0.80].
    compression_target_ratio: float | None = Field(default=None, ge=0.10, le=0.80, allow_inf_nan=False)
    # Total summary completion allowance, including reasoning; independent of tail retention.
    compression_summary_max_output_tokens: int = Field(default=32768, gt=0)
    chat_template_kwargs_enabled: bool = True
    api_key: Optional[str] = None
    delegation_max_iterations: int = 50
    checkpoints_enabled: bool = False
    # Hermes' own `mcp_servers` config (tools/mcp_tool.py): stdio or URL servers whose tools Hermes
    # registers beside its built-ins. A benchmark that ships its tools as an MCP server selects it
    # here; the mapping is written to config.yaml unchanged. Absent by default.
    mcp_servers: Optional[dict[str, dict[str, Any]]] = None
    # Offer OpenHands' `finish(message)` tool; the turn that calls it ends the run as completed.
    finish_tool: bool = False


class HermesAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class HermesAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    ng_agent_observations: AgentObservationBundle | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


_COMPRESSION_ROUTE: ContextVar[Optional[Callable[..., Any]]] = ContextVar("hermes_compression_route", default=None)


def _route_compression_calls() -> None:
    """Answer Hermes' own compression call from the calling run's policy client.

    Stock Hermes summarizes through ``call_llm(task="compression")``
    (``agent/context_compressor.py`` ``_generate_summary``); with no auxiliary
    provider configured, a self-hosted deployment resolves it to the main
    endpoint and model. A process-wide endpoint cannot carry concurrent rollout
    URLs, so compression calls made inside a run go to the route bound for that
    run. Everything else in the stock path is unchanged: the request, its
    max_tokens, an empty summary leaving only the compaction marker, and a
    failed call dropping the middle turns. Other calls keep Hermes' resolution.
    """
    import agent.context_compressor as context_compressor  # pyright: ignore[reportMissingImports]

    native_call_llm = context_compressor.call_llm
    if getattr(native_call_llm, "routes_compression", False):
        return

    def call_llm(task=None, **kwargs):
        route = _COMPRESSION_ROUTE.get()
        if task != "compression" or route is None:
            return native_call_llm(task, **kwargs)
        return route(**kwargs)

    call_llm.routes_compression = True
    context_compressor.call_llm = call_llm


def _policy_compression_route(agent, summary_temperature: float | None) -> Callable[..., Any]:
    """Send a compression request through the run's own policy request builder.

    ``agent._build_api_kwargs`` (the run's patched builder) supplies the model,
    timeout and ``extra_body`` (the thinking mode included) that the run's policy
    turns carry; the request then takes the compression messages only, with no
    tools and Hermes' summary ``max_tokens``. The summary is sampled at the policy
    turns' temperature, or ``summary_temperature`` when set; Hermes' own 0.3 is
    dropped because the summary is trained as a policy turn.
    """

    def complete(*, messages, temperature=None, max_tokens=None, tools=None, extra_body=None, **_):
        request: dict[str, Any] = agent._build_api_kwargs(messages)
        request["messages"] = messages
        for key in ("tools", "tool_choice", "max_tokens", "max_completion_tokens"):
            request.pop(key, None)
        if summary_temperature is not None:
            request["temperature"] = summary_temperature
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if tools:
            request["tools"] = tools
        if extra_body:
            request["extra_body"] = {**request.get("extra_body", {}), **extra_body}
        return agent.client.chat.completions.create(**request)

    return complete


class HermesAgent(SimpleResponsesAPIAgent):
    config: HermesAgentConfig
    sem: Semaphore = None
    # Set of agents currently running run_conversation, plus a flag tracking whether the single
    # shared SIGTERM dispatcher has been installed on the event loop. See _ensure_sigterm_handler.
    active_agents: set = None
    interrupted_agents: set = None
    sigterm_installed: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _ensure_sigterm_handler(self) -> None:
        """Install exactly one SIGTERM handler on the event loop that interrupts *every* in-flight
        agent. Registering a fresh per-call handler is unsafe under concurrency: add_signal_handler
        replaces the previous handler, so concurrent responses() calls clobber each other and the
        first to finish removes the only remaining handler — leaving later SIGTERMs unhandled and
        their trajectories lost. A single dispatcher over `active_agents` avoids that race."""
        if self.sigterm_installed:
            return
        import signal

        def _dispatch():
            for ag in list(self.active_agents):
                self.interrupted_agents.add(id(ag))
                if hasattr(ag, "interrupt"):
                    ag.interrupt("timeout")

        try:
            asyncio.get_event_loop().add_signal_handler(signal.SIGTERM, _dispatch)
            self.sigterm_installed = True
        except (NotImplementedError, OSError):
            pass  # not supported on this platform (e.g. Windows, non-main thread)

    def _build_config(self) -> str:
        import yaml

        config: dict[str, Any] = {
            "model": self._model_name(),
            "provider": "auto",
            "toolsets": ["hermes-cli"],
            "agent": {"max_turns": self.config.max_turns},
            "memory": {
                "memory_enabled": False,
                "user_profile_enabled": False,
            },
            "compression": {
                "enabled": self.config.compression_enabled,
                "threshold": self.config.compression_threshold,
                "summary_max_output_tokens": self.config.compression_summary_max_output_tokens,
                **(
                    {"target_ratio": self.config.compression_target_ratio}
                    if self.config.compression_target_ratio is not None
                    else {}
                ),
            },
            "terminal": {
                "backend": self.config.terminal_backend,
                "timeout": self.config.terminal_timeout,
            },
            "delegation": {
                "max_iterations": self.config.delegation_max_iterations,
            },
            "checkpoints": {
                "enabled": self.config.checkpoints_enabled,
            },
        }
        if self.config.mcp_servers:
            config["mcp_servers"] = self.config.mcp_servers
        if self.config.context_window is not None:
            config["model"] = {"default": self._model_name(), "context_length": self.config.context_window}
            reserve = self.config.max_tokens or 0
            if reserve >= self.config.context_window:
                raise ValueError("Hermes output reserve must be smaller than its context window")
        return yaml.dump(config, default_flow_style=False)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        self.active_agents = set()
        self.interrupted_agents = set()
        # hermes-agent reads these from env (cli.py / batch_runner.py); env vars are
        # process-global, so multiple HermesAgent instances in one process share them
        os.environ["TERMINAL_ENV"] = self.config.terminal_backend
        os.environ["TERMINAL_TIMEOUT"] = str(self.config.terminal_timeout)
        if self.config.api_timeout is not None:
            os.environ["HERMES_API_TIMEOUT"] = str(self.config.api_timeout)

        # Build config.yaml with config parameters
        hermes_home = tempfile.mkdtemp(prefix="hermes_agent_")
        atexit.register(shutil.rmtree, hermes_home, True)
        with open(os.path.join(hermes_home, "config.yaml"), "w") as _f:
            _f.write(self._build_config())
        os.environ["HERMES_HOME"] = hermes_home

    def _model_name(self) -> str:
        return self.config.model or str(self.config.model_server.name)

    async def _create_response(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        rollout_id: Optional[str] = None,
        observation_collector: Optional[Callable[[AgentObservationBundle], None]] = None,
    ) -> NeMoGymResponse:
        from run_agent import AIAgent  # from hermes-agent on path  # pyright: ignore[reportMissingImports]

        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, history, input_system = _split_input_to_user_and_history(body.input)
        system_message = self.config.system_prompt or input_system

        base_url = self.resolve_model_base_url(self.config.model_server.name, rollout_id)
        model_name = self._model_name()

        agent = AIAgent(
            base_url=base_url,
            api_key=self.config.api_key or os.environ.get("OPENAI_API_KEY", "gym"),  # pragma: allowlist secret
            model=model_name,
            use_streaming=False,
            temperature=self.config.temperature,
            insert_reasoning=True,
            max_iterations=self.config.max_turns,
            max_tokens=self.config.max_tokens,
            enabled_toolsets=self.config.enabled_toolsets,
            disabled_toolsets=self.config.disabled_toolsets,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            persist_session=False,
            save_trajectories=False,
        )
        _original_build_api_kwargs = agent._build_api_kwargs

        def _patched_build_api_kwargs(api_messages):
            kw = _original_build_api_kwargs(api_messages)
            if not self.config.chat_template_kwargs_enabled:
                return kw
            ctk = kw.setdefault("extra_body", {}).setdefault("chat_template_kwargs", {})
            ctk.setdefault("enable_thinking", True)
            ctk["truncate_history_thinking"] = False
            return kw

        agent._build_api_kwargs = _patched_build_api_kwargs
        compression_route = None
        if self.config.compression_enabled:
            _route_compression_calls()
            compression_route = _policy_compression_route(agent, self.config.compression_summary_temperature)
        _execute_every_tool_call(agent)
        _limit_tool_results(agent)
        ended_by_endpoint = _observe_endpoint_stop(agent)
        finished_by_tool = _offer_finish_tool(agent) if self.config.finish_tool else None
        observer = None
        if observation_collector is not None:
            try:
                observer = HermesAgentObserver(model_ref=self.config.model_server).instrument(agent)
            except Exception:
                LOG.exception("failed to initialize Hermes observability")

        # Interrupt the agent cleanly on SIGTERM so run_conversation returns with partial messages
        # instead of being killed mid-turn (which would leave response.json unwritten). A single
        # shared dispatcher interrupts every in-flight agent; we just register this one in the set.
        self._ensure_sigterm_handler()
        agent_id = id(agent)
        self.active_agents.add(agent)

        result = None
        agent_error: Optional[BaseException] = None
        interrupted_by_dispatch = False
        route_token = _COMPRESSION_ROUTE.set(compression_route)
        try:
            # to_thread copies this context, so the run's compression calls reach its own route.
            result = await asyncio.to_thread(
                agent.run_conversation,
                user_message,
                system_message,
                history,
            )
            if finished_by_tool is not None and finished_by_tool() and agent_id not in self.interrupted_agents:
                # The policy ended the run itself; the loop's interrupt check was only the stop point.
                result = {**result, "completed": True, "interrupted": False}
        except BaseException as exc:
            agent_error = exc
            raise
        finally:
            _COMPRESSION_ROUTE.reset(route_token)
            self.active_agents.discard(agent)
            interrupted_by_dispatch = agent_id in self.interrupted_agents
            self.interrupted_agents.discard(agent_id)
            if observation_collector is not None:
                try:
                    observations = (
                        observer.finish(result, error=agent_error)
                        if observer is not None
                        else AgentObservationBundle(
                            source="hermes",
                            gaps=[ObservationGap(code="observation_capture_failed")],
                        )
                    )
                except Exception:
                    LOG.exception("failed to finish Hermes observability")
                    observations = AgentObservationBundle(
                        source="hermes",
                        gaps=[ObservationGap(code="observation_capture_failed")],
                    )
                try:
                    observation_collector(observations)
                except Exception:
                    LOG.exception("failed to return Hermes observations")

        messages = result.get("messages") or []
        # aiagent omits system from returned messages
        n_input = len(history) + 1

        output_items = _trajectory_to_output_items(messages, n_input)

        has_assistant_message = any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        )
        if not has_assistant_message:
            LOG.warning(
                "Hermes agent ended without an assistant message. Padding empty assistant message. This should not happen often, investigate: error=%r",
                result.get("error"),
            )
            last_valid = next(
                (
                    m
                    for m in reversed(messages)
                    if isinstance(m, dict) and m.get("role") == "assistant" and m.get("generation_token_ids")
                ),
                None,
            )
            pti = last_valid["prompt_token_ids"] if last_valid else [0]
            gti = last_valid["generation_token_ids"] if last_valid else [0]
            glp = (last_valid.get("generation_log_probs") if last_valid else None) or [0.0]
            routed_experts = last_valid.get("routed_experts") if last_valid else None
            output_items.append(
                NeMoGymResponseOutputMessageForTraining(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text=result.get("error") or "", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                    prompt_token_ids=pti,
                    generation_token_ids=gti,
                    generation_log_probs=glp,
                    routed_experts=routed_experts,
                )
            )

        # The agent ran out of turns / blew its context window if run_conversation reports it did
        # not complete (api_call_count >= max_iterations). Mark the response incomplete so a
        # downstream consumer (e.g. anyswe) can mask an accidental pass instead of scoring it 1.0.
        agent_completed = bool(result.get("completed", True))

        # Expose harness run outcome fields for diagnosability
        was_interrupted = bool(result.get("interrupted")) or interrupted_by_dispatch

        harness_error = result.get("error")
        context_exhausted = (
            isinstance(harness_error, str)
            and harness_error.startswith("Context length exceeded")
            and harness_error.endswith("Cannot compress further.")
        )
        agent_failed = (bool(harness_error) or bool(result.get("failed"))) and not context_exhausted
        finished_naturally = _finished_naturally(
            result,
            finished_by_tool=finished_by_tool is not None and finished_by_tool(),
            ended_by_endpoint=ended_by_endpoint(),
            interrupted=was_interrupted,
            failed=agent_failed or context_exhausted,
        )
        metadata: dict[str, str] = {
            "interrupted": "true" if was_interrupted else "false",
            # Harness-neutral label a wrapping agent (anyterminal) carries.
            "finished_naturally": "true" if finished_naturally else "false",
            # `failed` = the underlying run flagged an API/provider failure (bad response shape,
            # rate limit, unrecoverable truncation) — the clean signal for an infra failure vs a
            # legitimate stop. `partial` = the response was truncated at the output-token limit.
            "failed": "true" if result.get("failed") else "false",
            "partial": "true" if result.get("partial") else "false",
        }
        # Turn count is `api_calls` in the result dict (present on every return).
        if isinstance(result.get("api_calls"), int):
            metadata["turns"] = str(result["api_calls"])
        if harness_error:
            metadata["hermes_error"] = str(harness_error)[:2000]
        if context_exhausted:
            metadata["stop_reason"] = "context_exhausted"

        # Populate the structured error field too. `code` must be one of OpenAI's Literals, so we
        # use the generic "server_error"; the real text lives in `message`.
        response_error = None
        if harness_error and not context_exhausted:
            from openai.types.responses import ResponseError  # pyright: ignore[reportMissingImports]

            response_error = ResponseError(code="server_error", message=str(harness_error)[:2000])

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=model_name,
            object="response",
            output=output_items,
            status="incomplete"
            if context_exhausted
            else ("failed" if agent_failed else ("completed" if agent_completed else "incomplete")),
            error=response_error,
            metadata=metadata,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=0,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=0,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=0,
            ),
        )

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        if not isinstance(rollout_id, str):
            return await self._create_response(body)
        episode = await self._create_episode(body, rollout_id=rollout_id)
        return episode.response.model_copy(
            update={_INTERNAL_OBSERVATIONS_KEY: episode.observations.model_dump(mode="json")}
        )

    async def _create_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        rollout_id: str,
    ) -> AgentEpisode:
        observations: Optional[AgentObservationBundle] = None

        def collect(bundle: AgentObservationBundle) -> None:
            nonlocal observations
            observations = bundle

        response = await self._create_response(
            body,
            rollout_id=rollout_id,
            observation_collector=collect,
        )
        if observations is None:
            observations = AgentObservationBundle(
                source="hermes",
                gaps=[ObservationGap(code="observation_capture_failed")],
            )
        observations.gaps.append(
            ObservationGap(
                code=(
                    "no_sandbox_runtime"
                    if self.config.terminal_backend == "local"
                    else "sandbox_observation_unavailable"
                ),
                detail=(
                    None
                    if self.config.terminal_backend == "local"
                    else f"terminal_backend={self.config.terminal_backend}"
                ),
            )
        )
        return AgentEpisode(response=response, observations=observations)

    async def run(self, request: Request, body: HermesAgentRunRequest) -> HermesAgentVerifyResponse:
        async with self.sem:
            cookies = request.cookies

            seed_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_resp)
            cookies = seed_resp.cookies

            rollout_id = self.rollout_id_from_run(body)
            agent_resp = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(agent_resp)
            cookies = agent_resp.cookies
            agent_resp_json = await get_response_json(agent_resp)
            raw_observations = (
                agent_resp_json.pop(_INTERNAL_OBSERVATIONS_KEY, None) if rollout_id is not None else None
            )
            observations = (
                AgentObservationBundle.model_validate(raw_observations) if isinstance(raw_observations, dict) else None
            )

            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": agent_resp_json},
                cookies=cookies,
            )
            await raise_for_status(verify_resp)
            verify_json = await get_response_json(verify_resp)

            gym_resp = NeMoGymResponse.model_validate(agent_resp_json)
            turns = sum(
                1
                for item in gym_resp.output
                if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            )
            naturally = (gym_resp.metadata or {}).get("finished_naturally") == "true"

            result = verify_json | {"turns_used": turns, "finished_naturally": naturally}
            if observations is not None:
                result["ng_agent_observations"] = observations.model_dump(mode="json")
            return HermesAgentVerifyResponse.model_validate(result)


if __name__ == "__main__":
    HermesAgent.run_webserver()
