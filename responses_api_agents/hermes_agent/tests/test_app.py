# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.rollout_observability import AgentEpisode, AgentObservationBundle
from nemo_gym.server_utils import ServerClient
from responses_api_agents.hermes_agent.app import (
    HermesAgent,
    HermesAgentConfig,
    HermesAgentRunRequest,
    ModelServerRef,
    ResourcesServerRef,
    _execute_every_tool_call,
    _split_input_to_user_and_history,
    _trajectory_to_output_items,
)
from responses_api_agents.hermes_agent.observability import HermesAgentObserver


class _FakeResponse:
    ok = True

    def __init__(self, payload: dict, cookies: dict | None = None) -> None:
        self.payload = payload
        self.cookies = cookies or {}

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _config(**kwargs) -> HermesAgentConfig:
    return HermesAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        model_server=ModelServerRef(type="responses_api_models", name=""),
        **kwargs,
    )


class TestSanity:
    @pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
    def test_api_timeout_rejects_invalid_budget(self, timeout) -> None:
        with pytest.raises(ValueError):
            _config(api_timeout=timeout)

    def test_api_timeout_projects_episode_budget(self, monkeypatch) -> None:
        monkeypatch.setenv("HERMES_API_TIMEOUT", "1800")
        HermesAgent(config=_config(api_timeout=21600), server_client=MagicMock(spec=ServerClient))
        assert float(os.environ["HERMES_API_TIMEOUT"]) == 21600

    def test_unset_api_timeout_preserves_runtime_setting(self, monkeypatch) -> None:
        monkeypatch.setenv("HERMES_API_TIMEOUT", "2400")
        HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        assert os.environ["HERMES_API_TIMEOUT"] == "2400"

    def test_construct(self) -> None:
        HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))

    def test_concurrency_semaphore_initialized(self) -> None:
        agent = HermesAgent(config=_config(concurrency=4), server_client=MagicMock(spec=ServerClient))
        assert agent.sem._value == 4

    def test_model_defaults_to_server_name(self) -> None:
        agent = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        assert agent._model_name() == ""

    def test_configured_model_overrides_server_name(self) -> None:
        agent = HermesAgent(config=_config(model="Qwen3.6-35B-A3B"), server_client=MagicMock(spec=ServerClient))
        assert agent._model_name() == "Qwen3.6-35B-A3B"


class _FakeAgent:
    """Stand-in for AIAgent — only needs .interrupt() for the SIGTERM dispatch path."""

    def __init__(self) -> None:
        self.interrupt_reason = None

    def interrupt(self, reason: str) -> None:
        self.interrupt_reason = reason


class TestSigtermHandler:
    """Regression tests for the concurrency-safe SIGTERM dispatcher.

    The old per-call add_signal_handler/remove_signal_handler approach raced: concurrent responses()
    calls clobbered each other's handler and the first to finish removed the only one left, so a
    later SIGTERM interrupted nobody. The fix registers a single dispatcher over a shared set of
    in-flight agents.
    """

    def test_active_agents_initialized_empty(self) -> None:
        agent = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        assert agent.active_agents == set()
        assert agent.sigterm_installed is False

    def test_handler_installed_once_and_interrupts_all_in_flight(self) -> None:
        agent = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))

        registered: list = []
        loop = asyncio.new_event_loop()
        loop.add_signal_handler = lambda sig, cb, *a: registered.append(cb)  # type: ignore[method-assign]
        asyncio.set_event_loop(loop)
        try:
            agent._ensure_sigterm_handler()
            assert agent.sigterm_installed is True
            assert len(registered) == 1  # exactly one dispatcher registered

            # Idempotent: a second concurrent call must NOT register another handler.
            agent._ensure_sigterm_handler()
            assert len(registered) == 1

            dispatch = registered[0]

            # Two concurrent in-flight agents: SIGTERM must interrupt BOTH (the old code lost one).
            a, b = _FakeAgent(), _FakeAgent()
            agent.active_agents.update({a, b})
            dispatch()
            assert a.interrupt_reason == "timeout"
            assert b.interrupt_reason == "timeout"

            # Once an agent finishes (discarded), a later SIGTERM no longer touches it.
            a.interrupt_reason = None
            b.interrupt_reason = None
            agent.active_agents.discard(a)
            dispatch()
            assert a.interrupt_reason is None
            assert b.interrupt_reason == "timeout"
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    def test_handler_install_survives_unsupported_platform(self) -> None:
        # On platforms where add_signal_handler raises (e.g. non-main thread), install is a no-op
        # rather than an error, and the agent stays usable.
        agent = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))

        loop = asyncio.new_event_loop()

        def _raise(*_a, **_k):
            raise NotImplementedError

        loop.add_signal_handler = _raise  # type: ignore[method-assign]
        asyncio.set_event_loop(loop)
        try:
            agent._ensure_sigterm_handler()
            assert agent.sigterm_installed is False
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    def test_interrupted_agent_id_cleared_when_run_conversation_raises(self, monkeypatch) -> None:
        import nemo_gym.base_responses_api_agent as base_agent
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: "http://h:1"
        hermes = HermesAgent(config=_config(), server_client=server_client)
        monkeypatch.setattr(hermes, "_ensure_sigterm_handler", lambda: None)

        class _FailingInterruptedAIAgent:
            def __init__(self, **kwargs) -> None:
                self._build_api_kwargs = lambda _messages: {}
                self._execute_tool_calls = lambda *_args, **_kwargs: None
                self.context_compressor = SimpleNamespace(managed_summary_callback=None)

            def run_conversation(self, *args, **kwargs) -> dict:
                hermes.interrupted_agents.add(id(self))
                raise RuntimeError("agent failed")

        monkeypatch.setattr("run_agent.AIAgent", _FailingInterruptedAIAgent)

        with pytest.raises(RuntimeError, match="agent failed"):
            asyncio.run(hermes.responses(request=None, body=NeMoGymResponseCreateParamsNonStreaming(input="hi")))

        assert hermes.active_agents == set()
        assert hermes.interrupted_agents == set()


class TestSplitInputToUserAndHistory:
    def test_user_only(self) -> None:
        items = [NeMoGymEasyInputMessage(role="user", content="hi")]
        user, history, system = _split_input_to_user_and_history(items)
        assert user == "hi"
        assert history == []
        assert system is None

    def test_system_plus_user(self) -> None:
        items = [
            NeMoGymEasyInputMessage(role="system", content="be helpful"),
            NeMoGymEasyInputMessage(role="user", content="hi"),
        ]
        user, history, system = _split_input_to_user_and_history(items)
        assert user == "hi"
        assert history == []
        assert system == "be helpful"

    def test_history_then_user(self) -> None:
        items = [
            NeMoGymEasyInputMessage(role="user", content="first"),
            NeMoGymEasyInputMessage(role="assistant", content="reply"),
            NeMoGymEasyInputMessage(role="user", content="follow-up"),
        ]
        user, history, system = _split_input_to_user_and_history(items)
        assert user == "follow-up"
        assert history == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ]
        assert system is None

    def test_resumed_ends_on_assistant(self) -> None:
        items = [
            NeMoGymEasyInputMessage(role="user", content="q"),
            NeMoGymEasyInputMessage(role="assistant", content="a"),
        ]
        user, history, system = _split_input_to_user_and_history(items)
        assert user == ""
        assert history == [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]

    def test_dict_inputs(self) -> None:
        items = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "ok"}]
        user, history, system = _split_input_to_user_and_history(items)
        assert user == "ok"
        assert history == []
        assert system == "be brief"


class TestTrajectoryToOutputItems:
    def test_empty(self) -> None:
        assert _trajectory_to_output_items([], 0) == []

    def test_drops_input_prefix(self) -> None:
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        out = _trajectory_to_output_items(msgs, 1)
        assert len(out) == 1
        assert isinstance(out[0], NeMoGymResponseOutputMessageForTraining)

    def test_assistant_with_tokens(self) -> None:
        routed_experts = [
            [[0, 1]],
            [[2, 3]],
            [[4, 5]],
            [[6, 7]],
        ]
        msgs = [
            {
                "role": "assistant",
                "content": "answer",
                "prompt_token_ids": [1, 2],
                "generation_token_ids": [3, 4],
                "generation_log_probs": [0.0, -0.1],
                "routed_experts": routed_experts,
            }
        ]
        out = _trajectory_to_output_items(msgs, 0)
        assert len(out) == 1
        assert isinstance(out[0], NeMoGymResponseOutputMessageForTraining)
        assert out[0].generation_token_ids == [3, 4]
        assert out[0].prompt_token_ids == [1, 2]
        assert out[0].routed_experts == routed_experts

    def test_assistant_reasoning_strips_inline_think_from_message(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "reasoning": "structured thoughts",
                "content": "<think>structured thoughts</think>final answer",
            }
        ]
        out = _trajectory_to_output_items(msgs, 0)
        assert len(out) == 2
        assert isinstance(out[0], NeMoGymResponseReasoningItem)
        assert out[0].summary[0].text == "structured thoughts"
        assert isinstance(out[1], NeMoGymResponseOutputMessageForTraining)
        assert out[1].content[0].text == "final answer"

    def test_assistant_with_tool_call_and_tool_result(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "terminal", "arguments": '{"cmd":"ls"}'}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "file.txt\n"},
        ]
        out = _trajectory_to_output_items(msgs, 0)
        assert len(out) == 3
        assert isinstance(out[0], NeMoGymResponseOutputMessageForTraining)
        assert isinstance(out[1], NeMoGymResponseFunctionToolCall)
        assert out[1].name == "terminal"
        assert out[1].arguments == '{"cmd":"ls"}'
        assert isinstance(out[2], NeMoGymFunctionCallOutput)
        assert out[2].call_id == "c1"
        assert out[2].output == "file.txt\n"

    def test_skips_non_dict_items(self) -> None:
        msgs = [None, "string", {"role": "assistant", "content": "ok"}]
        out = _trajectory_to_output_items(msgs, 0)
        assert len(out) == 1


class TestRolloutCorrelation:
    def test_responses_applies_rollout_prefix(self, monkeypatch) -> None:
        from fastapi.testclient import TestClient

        import nemo_gym.base_responses_api_agent as base_agent
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: "http://h:1"
        agent = HermesAgent(config=_config(), server_client=server_client)
        monkeypatch.setattr(agent, "_ensure_sigterm_handler", lambda: None)

        seen: dict = {}

        class _StubAIAgent:
            def __init__(self, **kwargs) -> None:
                seen["base_url"] = kwargs.get("base_url")
                self._build_api_kwargs = lambda _messages: {}
                self._execute_tool_calls = lambda *_args, **_kwargs: None
                self.context_compressor = SimpleNamespace(managed_summary_callback=None)
                self.compression_enabled = True

            def run_conversation(self, *args, **kwargs) -> dict:
                return {"messages": [{"role": "assistant", "content": "ok"}]}

        monkeypatch.setattr("run_agent.AIAgent", _StubAIAgent)
        client = TestClient(agent.setup_webserver())

        assert client.post("/ng-rollout/rid/v1/responses", json={"input": "hi"}).status_code == 200
        assert seen["base_url"] == "http://h:1/ng-rollout/rid/v1"

        direct = asyncio.run(agent.responses(request=None, body=NeMoGymResponseCreateParamsNonStreaming(input="hi")))
        assert seen["base_url"] == "http://h:1/v1"
        assert "_ng_agent_observations" not in direct.model_dump(mode="json")

        episode = asyncio.run(
            agent._create_episode(
                body=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
                rollout_id="rid",
            )
        )
        assert seen["base_url"] == "http://h:1/ng-rollout/rid/v1"
        assert episode.observations.source == "hermes"
        assert episode.observations.records[0].invocation_id == "root"


class TestMaxTokens:
    def _agent_and_seen(self, monkeypatch, *, retained_result=None, **config_kwargs) -> tuple[HermesAgent, dict]:
        import nemo_gym.base_responses_api_agent as base_agent

        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: "http://h:1"
        agent = HermesAgent(config=_config(**config_kwargs), server_client=server_client)
        monkeypatch.setattr(agent, "_ensure_sigterm_handler", lambda: None)

        seen: dict = {}

        class _StubAIAgent:
            def __init__(self, **kwargs) -> None:
                seen["max_tokens"] = kwargs.get("max_tokens")
                self._build_api_kwargs = lambda _messages: {}
                self._execute_tool_calls = lambda *_args, **_kwargs: None
                self.context_compressor = SimpleNamespace(managed_summary_callback=None)
                self.compression_enabled = True

            def run_conversation(self, *args, **kwargs) -> dict:
                return (
                    retained_result
                    if retained_result is not None
                    else {"messages": [{"role": "assistant", "content": "ok"}]}
                )

        monkeypatch.setattr("run_agent.AIAgent", _StubAIAgent)
        return agent, seen

    @pytest.mark.parametrize("context_stop", [True, False])
    def test_retained_context_stop_is_typed_and_other_errors_stay_failed(self, monkeypatch, context_stop):
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        raw = json.loads((Path(__file__).parent / "fixtures/context-stop.json").read_text())
        message = raw["error"]["message"] if context_stop else "API request failed"
        agent, _ = self._agent_and_seen(
            monkeypatch,
            retained_result={
                "error": message,
                "completed": False,
                "partial": True,
                "api_calls": int(raw["metadata"]["turns"]),
                "messages": [{"role": "assistant", "content": "retained partial output"}],
            },
        )
        response = asyncio.run(agent._create_response(NeMoGymResponseCreateParamsNonStreaming(input="continue")))
        assert response.metadata["hermes_error"] == message
        assert response.metadata["turns"] == raw["metadata"]["turns"]
        if context_stop:
            assert response.status == "incomplete" and response.error is None
            assert response.metadata["stop_reason"] == "context_exhausted"
        else:
            assert response.status == "failed" and response.error.code == "server_error"
            assert "stop_reason" not in response.metadata

    def test_max_tokens_passed_to_ai_agent(self, monkeypatch) -> None:
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        agent, seen = self._agent_and_seen(monkeypatch, max_tokens=4096)
        asyncio.run(agent.responses(request=None, body=NeMoGymResponseCreateParamsNonStreaming(input="hi")))
        assert seen["max_tokens"] == 4096

    def test_max_tokens_defaults_to_none(self, monkeypatch) -> None:
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        agent, seen = self._agent_and_seen(monkeypatch)
        asyncio.run(agent.responses(request=None, body=NeMoGymResponseCreateParamsNonStreaming(input="hi")))
        assert seen["max_tokens"] is None


class TestObservability:
    @pytest.mark.parametrize(
        ("terminal_backend", "runtime_gap"),
        [("local", "no_sandbox_runtime"), ("docker", "sandbox_observation_unavailable")],
    )
    def test_observation_failure_does_not_change_response(self, monkeypatch, terminal_backend, runtime_gap) -> None:
        import nemo_gym.base_responses_api_agent as base_agent
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: "http://h:1"
        agent = HermesAgent(config=_config(terminal_backend=terminal_backend), server_client=server_client)
        monkeypatch.setattr(agent, "_ensure_sigterm_handler", lambda: None)

        class _StubAIAgent:
            def __init__(self, **kwargs) -> None:
                self._build_api_kwargs = lambda _messages: {}
                self._execute_tool_calls = lambda *_args, **_kwargs: None
                self.context_compressor = SimpleNamespace(managed_summary_callback=None)

            def run_conversation(self, *args, **kwargs) -> dict:
                return {
                    "completed": True,
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "ok"},
                    ],
                }

        monkeypatch.setattr("run_agent.AIAgent", _StubAIAgent)
        body = NeMoGymResponseCreateParamsNonStreaming(input="hi")
        baseline = asyncio.run(agent.responses(request=None, body=body))

        def fail_finish(*args, **kwargs):
            raise RuntimeError("observer failed")

        monkeypatch.setattr(HermesAgentObserver, "finish", fail_finish)
        episode = asyncio.run(agent._create_episode(body=body, rollout_id="rid"))

        assert episode.response.output == baseline.output
        assert episode.response.usage == baseline.usage
        assert [gap.code for gap in episode.observations.gaps] == [
            "observation_capture_failed",
            runtime_gap,
        ]

    def test_run_returns_observations_without_leaking_internal_attachment(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {"observability_enabled": True}
        agent = HermesAgent(config=_config(), server_client=server_client)
        response = NeMoGymResponse.model_validate(
            {
                "id": "resp-1",
                "created_at": 1,
                "model": "model",
                "object": "response",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        observed_response = AsyncMock(
            return_value=AgentEpisode(
                response=response,
                observations=AgentObservationBundle(source="hermes"),
            )
        )

        async def post(server_name, url_path, json=None, cookies=None, **kwargs):
            if url_path == "/seed_session":
                return _FakeResponse({}, {"session": "1"})
            if url_path.endswith("/v1/responses"):
                response = await agent.responses(MagicMock(path_params={"rollout_id": "1-2"}), json)
                return _FakeResponse(response.model_dump(mode="json"), cookies)
            return _FakeResponse(json | {"reward": 1.0})

        server_client.post = AsyncMock(side_effect=post)
        request = MagicMock()
        request.cookies = {}
        body = HermesAgentRunRequest.model_validate(
            {
                "responses_create_params": {"input": "solve"},
                "_ng_task_index": 1,
                "_ng_rollout_index": 2,
            }
        )

        with patch.object(HermesAgent, "_create_episode", observed_response):
            result = asyncio.run(agent.run(request, body))

        assert result.ng_agent_observations is not None
        assert result.ng_agent_observations.source == "hermes"
        verify_json = server_client.post.await_args_list[-1].kwargs["json"]
        assert "_ng_agent_observations" not in verify_json["response"]
        assert "rollout_id" not in verify_json

    def test_observer_failure_does_not_mask_agent_exception(self, monkeypatch) -> None:
        import nemo_gym.base_responses_api_agent as base_agent
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: "http://h:1"
        agent = HermesAgent(config=_config(), server_client=server_client)
        monkeypatch.setattr(agent, "_ensure_sigterm_handler", lambda: None)

        class _FailingAIAgent:
            def __init__(self, **kwargs) -> None:
                self._build_api_kwargs = lambda _messages: {}
                self._execute_tool_calls = lambda *_args, **_kwargs: None
                self.context_compressor = SimpleNamespace(managed_summary_callback=None)

            def run_conversation(self, *args, **kwargs) -> dict:
                raise ValueError("agent failed")

        monkeypatch.setattr("run_agent.AIAgent", _FailingAIAgent)
        monkeypatch.setattr(
            HermesAgentObserver,
            "finish",
            MagicMock(side_effect=RuntimeError("observer failed")),
        )

        with pytest.raises(ValueError, match="agent failed"):
            asyncio.run(
                agent._create_episode(
                    body=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
                    rollout_id="rid",
                )
            )


class TestEveryToolCallExecutes:
    def test_hermes_guardrails_keep_duplicate_and_excess_delegate_calls(self) -> None:
        from types import SimpleNamespace

        from run_agent import AIAgent  # pyright: ignore[reportMissingImports]
        from tools.delegate_tool import MAX_CONCURRENT_CHILDREN  # pyright: ignore[reportMissingImports]

        def call(name: str, arguments: str) -> SimpleNamespace:
            return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))

        calls = [call("terminal", '{"command": "ls"}'), call("terminal", '{"command": "ls"}')] + [
            call("delegate_task", json.dumps({"goal": str(index)})) for index in range(MAX_CONCURRENT_CHILDREN + 1)
        ]
        agent = object.__new__(AIAgent)
        # Stock Hermes drops the duplicate and the delegate call past the cap.
        assert len(agent._cap_delegate_task_calls(agent._deduplicate_tool_calls(calls))) == len(calls) - 2

        _execute_every_tool_call(agent)
        assert agent._cap_delegate_task_calls(agent._deduplicate_tool_calls(calls)) == calls


class _ScriptedChatServer:
    """OpenAI chat-completions endpoint that replays scripted assistant turns."""

    def __init__(self, turns: list[dict]) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from threading import Thread

        self.turns = list(turns)
        self.requests: list[dict] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append(body)
                message = server.turns.pop(0) if server.turns else {"role": "assistant", "content": "unscripted"}
                choice = {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                }
                usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                if isinstance(message.get("stop_reason"), str):
                    # The serving endpoint's episode stop: an empty message, a stop
                    # reason and no sampled tokens.
                    choice = {
                        "index": 0,
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "stop",
                        "stop_reason": message["stop_reason"],
                    }
                    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                payload = json.dumps(
                    {
                        "id": f"chatcmpl-{len(server.requests)}",
                        "object": "chat.completion",
                        "created": 0,
                        "model": body.get("model", "model"),
                        "choices": [choice],
                        "usage": usage,
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _tool_turn(*calls: tuple[str, dict]) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": f"call_{index}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            for index, (name, args) in enumerate(calls)
        ],
    }


class TestFinishTool:
    """Real Hermes `AIAgent` loop against a scripted OpenAI-compatible endpoint."""

    def _run(self, monkeypatch, tmp_path, turns: list[dict], **config_kwargs):
        import nemo_gym.base_responses_api_agent as base_agent
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        server = _ScriptedChatServer(turns)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: server.url.removesuffix("/v1")
        agent = HermesAgent(
            config=_config(
                model="model",
                enabled_toolsets=["terminal", "file"],
                disabled_toolsets=["code_execution"],
                compression_enabled=False,
                chat_template_kwargs_enabled=False,
                **config_kwargs,
            ),
            server_client=server_client,
        )
        monkeypatch.setattr(agent, "_ensure_sigterm_handler", lambda: None)
        try:
            episode = asyncio.run(
                agent._create_episode(body=NeMoGymResponseCreateParamsNonStreaming(input="fix it"), rollout_id="rid")
            )
        finally:
            server.close()
        return episode, server.requests

    @staticmethod
    def _offered(request: dict) -> set[str]:
        return {tool["function"]["name"] for tool in request.get("tools") or []}

    def test_unoffered_finish_is_rejected_and_the_run_continues(self, monkeypatch, tmp_path) -> None:
        episode, requests = self._run(
            monkeypatch,
            tmp_path,
            [_tool_turn(("finish", {"message": "done"})), {"role": "assistant", "content": "done"}],
        )
        assert "finish" not in self._offered(requests[0])
        assert len(requests) == 2
        rejection = [item for item in episode.response.output if item.type == "function_call_output"]
        assert rejection[0].output.startswith("Tool 'finish' does not exist.")
        assert episode.response.status == "completed"

    def test_finish_is_offered_and_ends_the_run_without_another_request(self, monkeypatch, tmp_path) -> None:
        episode, requests = self._run(
            monkeypatch,
            tmp_path,
            [
                _tool_turn(
                    ("terminal", {"command": "echo checked > out.txt"}), ("finish", {"message": "Fix verified"})
                ),
                {"role": "assistant", "content": "unreachable"},
            ],
            finish_tool=True,
        )
        assert len(requests) == 1
        finish = next(tool["function"] for tool in requests[0]["tools"] if tool["function"]["name"] == "finish")
        assert finish["parameters"]["required"] == ["message"]
        # Every call in the finishing turn still executes.
        assert (tmp_path / "out.txt").read_text().strip() == "checked"
        output = episode.response.output
        assert [item.name for item in output if item.type == "function_call"] == ["terminal", "finish"]
        assert output[-1].type == "function_call_output" and output[-1].output == "Fix verified"
        assert episode.response.status == "completed"
        assert episode.response.metadata["interrupted"] == "false"
        root = next(
            record for record in episode.observations.records if getattr(record, "invocation_id", None) == "root"
        )
        assert root.status == "completed"

    def test_finish_on_the_last_turn_skips_the_iteration_limit_summary(self, monkeypatch, tmp_path) -> None:
        episode, requests = self._run(
            monkeypatch,
            tmp_path,
            [_tool_turn(("finish", {"message": "done"})), {"role": "assistant", "content": "summary"}],
            finish_tool=True,
            max_turns=1,
        )
        assert len(requests) == 1
        assert episode.response.status == "completed"

    def test_finish_is_the_agents_own_completion(self, monkeypatch, tmp_path) -> None:
        episode, _ = self._run(monkeypatch, tmp_path, [_tool_turn(("finish", {"message": "done"}))], finish_tool=True)
        assert episode.response.metadata["finished_naturally"] == "true"

    def test_runs_without_finish_keep_hermes_completion(self, monkeypatch, tmp_path) -> None:
        episode, requests = self._run(
            monkeypatch, tmp_path, [{"role": "assistant", "content": "all done"}], finish_tool=True
        )
        assert len(requests) == 1
        assert "finish" in self._offered(requests[0])
        assert episode.response.output[-1].content[0].text == "all done"
        assert episode.response.status == "completed"


class TestFinishedNaturally:
    """Which real Hermes endings are the agent's own completion."""

    _run = TestFinishTool._run

    def test_final_answer_is_the_agents_own_completion(self, monkeypatch, tmp_path) -> None:
        episode, requests = self._run(
            monkeypatch,
            tmp_path,
            [_tool_turn(("terminal", {"command": "echo ok > out.txt"})), {"role": "assistant", "content": "Done."}],
        )
        assert len(requests) == 2
        assert episode.response.status == "completed"
        assert episode.response.metadata["finished_naturally"] == "true"

    @pytest.mark.parametrize("reason", ["turn_limit", "context_exhausted", "identical_call", "rumination"])
    def test_endpoint_stop_is_not_the_agents_completion(self, monkeypatch, tmp_path, reason) -> None:
        # Hermes takes the endpoint's empty stop message as a final answer and
        # reports the run completed; the agent did not choose to end it.
        episode, requests = self._run(
            monkeypatch,
            tmp_path,
            [_tool_turn(("terminal", {"command": "echo ok > out.txt"})), {"stop_reason": reason}],
        )
        assert len(requests) == 2
        assert episode.response.status == "completed"
        assert episode.response.metadata["finished_naturally"] == "false"

    def test_turn_cap_summary_is_not_the_agents_completion(self, monkeypatch, tmp_path) -> None:
        episode, requests = self._run(
            monkeypatch,
            tmp_path,
            [_tool_turn(("terminal", {"command": "echo ok > out.txt"})), {"role": "assistant", "content": "summary"}],
            max_turns=1,
        )
        assert episode.response.output[-1].content[0].text == "summary"
        assert episode.response.metadata["finished_naturally"] == "false"

    @pytest.mark.parametrize(
        "result",
        [
            {"completed": True, "error": "API request failed"},  # harness failure
            {"completed": True, "failed": True},
            {"completed": True, "partial": True},
            {"completed": True, "interrupted": True},  # SIGTERM / timeout
            {"completed": False},
        ],
    )
    def test_failed_or_interrupted_run_is_not_the_agents_completion(self, result) -> None:
        from responses_api_agents.hermes_agent.app import _finished_naturally

        final = {"role": "assistant", "content": "Done.", "finish_reason": "stop"}
        result = {"messages": [final], **result}
        interrupted = bool(result.get("interrupted"))
        failed = bool(result.get("error") or result.get("failed"))
        assert not _finished_naturally(
            result, finished_by_tool=False, ended_by_endpoint=False, interrupted=interrupted, failed=failed
        )
        clean = {"messages": [final], "completed": True}
        assert _finished_naturally(
            clean, finished_by_tool=False, ended_by_endpoint=False, interrupted=False, failed=False
        )

    def test_synthetic_error_reply_is_not_the_agents_completion(self) -> None:
        from responses_api_agents.hermes_agent.app import _finished_naturally

        # Hermes appends this reply itself, without finish_reason, near its turn limit.
        result = {"completed": True, "messages": [{"role": "assistant", "content": "I apologize, but I encountered"}]}
        assert not _finished_naturally(
            result, finished_by_tool=False, ended_by_endpoint=False, interrupted=False, failed=False
        )
