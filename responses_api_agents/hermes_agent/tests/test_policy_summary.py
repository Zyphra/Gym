# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
from omegaconf import OmegaConf

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.hermes_agent.app import HermesAgent, HermesAgentConfig


SUMMARY_TEXT = "## Goal\nVerified fixture summary."
MAIN_TEXT = "Main fixture response."
CONTEXT_WINDOW = 131072


def _is_summary_request(body: dict) -> bool:
    first = body["messages"][0]["content"]
    return isinstance(first, str) and first.startswith("Create a structured handoff summary")


class _PolicyFixture:
    """A local OpenAI-compatible policy server recording every request it serves."""

    def __init__(self, summary: dict):
        self.requests: list[tuple[str, dict]] = []
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fixture.requests.append((self.path, body))
                if _is_summary_request(body):
                    status, choice = summary["status"], summary.get("choice")
                else:
                    status = 200
                    choice = {"message": {"role": "assistant", "content": MAIN_TEXT}, "finish_reason": "stop"}
                payload = (
                    {
                        "id": f"fixture-{len(fixture.requests)}",
                        "object": "chat.completion",
                        "created": 1,
                        "model": body["model"],
                        "choices": [{"index": 0, **choice}],
                        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
                    }
                    if status == 200
                    else {"error": {"message": "fixture refusal", "type": "invalid_request_error"}}
                )
                encoded = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        assert not self.thread.is_alive()

    def summaries(self) -> list[tuple[str, dict]]:
        return [(path, body) for path, body in self.requests if _is_summary_request(body)]

    def mains(self) -> list[tuple[str, dict]]:
        return [(path, body) for path, body in self.requests if not _is_summary_request(body)]


def _long_history_input() -> list[NeMoGymEasyInputMessage]:
    # 26 earlier messages of ~13,800 characters each: above Hermes' preflight
    # compression threshold (half of the 131,072-token window) and above the
    # protected head and tail, so the run compresses before its first request.
    items = [
        NeMoGymEasyInputMessage(
            role="user" if index % 2 == 0 else "assistant",
            content=f"history message {index}: " + "observed fixture state. " * 575,
        )
        for index in range(26)
    ]
    items.append(NeMoGymEasyInputMessage(role="user", content="Continue the fixture task."))
    return items


def _agent(tmp_path, monkeypatch, port: int, native_agents: list, **config) -> HermesAgent:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    import run_agent

    original = run_agent.AIAgent

    def record_native_agent(**kwargs):
        native = original(**kwargs)
        native_agents.append(native)
        return native

    monkeypatch.setattr(run_agent, "AIAgent", record_native_agent)
    client = ServerClient.model_construct(
        global_config_dict=OmegaConf.create(
            {"policy": {"responses_api_models": {"openai_model": {"host": "127.0.0.1", "port": port}}}}
        )
    )
    client._build_server_base_url = lambda _cfg: f"http://127.0.0.1:{port}"
    return HermesAgent(
        config=HermesAgentConfig(
            host="127.0.0.1",
            port=0,
            name="hermes",
            entrypoint="app.py",
            model_server=ModelServerRef(type="responses_api_models", name="policy"),
            resources_server=ResourcesServerRef(type="resources_servers", name="resource"),
            model="fixture-policy",
            max_turns=2,
            max_tokens=8192,
            context_window=CONTEXT_WINDOW,
            enabled_toolsets=[],
            compression_enabled=True,
            terminal_backend="local",
            **config,
        ),
        server_client=client,
    )


def _run(app: HermesAgent, *rollout_ids: str):
    async def exercise():
        return await asyncio.gather(
            *(
                app._create_response(
                    NeMoGymResponseCreateParamsNonStreaming(input=_long_history_input()), rollout_id=rollout_id
                )
                for rollout_id in rollout_ids
            )
        )

    return asyncio.run(exercise())


def _close(native_agents: list) -> None:
    for native in native_agents:
        native.client.close()


def test_stock_summary_request_goes_to_each_rollouts_own_policy(monkeypatch, record_property, tmp_path):
    summary = {
        "status": 200,
        "choice": {"message": {"role": "assistant", "content": SUMMARY_TEXT}, "finish_reason": "stop"},
    }
    native_agents: list = []
    with _PolicyFixture(summary) as policy:
        app = _agent(tmp_path, monkeypatch, policy.server.server_port, native_agents)
        try:
            responses = _run(app, "rollout-a", "rollout-b")
        finally:
            _close(native_agents)
        summaries, mains = policy.summaries(), policy.mains()

    assert [response.status for response in responses] == ["completed", "completed"]
    for native in native_agents:
        compressor = native.context_compressor
        assert compressor.context_length == CONTEXT_WINDOW
        # Stock threshold: half the window, with no output-reserve scaling.
        assert compressor.threshold_tokens == CONTEXT_WINDOW // 2

    # Each run's summary is served on its own rollout URL, before its main request.
    assert sorted(path for path, _ in summaries) == [
        "/ng-rollout/rollout-a/v1/chat/completions",
        "/ng-rollout/rollout-b/v1/chat/completions",
    ]
    assert sorted(path for path, _ in mains) == sorted(path for path, _ in summaries)
    for _, body in summaries:
        # Hermes' stock compression request: one user message, no tools, Hermes'
        # configured total-output budget and temperature.
        assert body["model"] == "fixture-policy"
        assert [message["role"] for message in body["messages"]] == ["user"]
        assert "TURNS TO SUMMARIZE:\n[ASSISTANT]: history message 3: " in body["messages"][0]["content"]
        assert "tools" not in body and "tool_choice" not in body
        assert body["max_tokens"] == 32768
        assert body["temperature"] == 0.3
    from agent.context_compressor import SUMMARY_PREFIX

    for _, body in mains:
        compacted = [m["content"] for m in body["messages"] if SUMMARY_PREFIX in (m.get("content") or "")]
        assert len(compacted) == 1 and compacted[0].startswith(f"{SUMMARY_PREFIX}\n{SUMMARY_TEXT}")
    import run_agent

    record_property("native_runtime", str(Path(run_agent.__file__).resolve()))


def test_summary_temperature_override_is_the_only_request_change(monkeypatch, tmp_path):
    summary = {
        "status": 200,
        "choice": {"message": {"role": "assistant", "content": SUMMARY_TEXT}, "finish_reason": "stop"},
    }
    native_agents: list = []
    with _PolicyFixture(summary) as policy:
        app = _agent(
            tmp_path, monkeypatch, policy.server.server_port, native_agents, compression_summary_temperature=1.0
        )
        try:
            _run(app, "rollout-t")
        finally:
            _close(native_agents)
        ((_, body),) = policy.summaries()

    assert body["temperature"] == 1.0
    assert body["max_tokens"] == 32768 and "tools" not in body


def test_empty_summary_leaves_only_the_compaction_marker_and_the_run_continues(monkeypatch, tmp_path):
    # Thinking spent the whole budget: no visible text, cut at max_tokens.
    summary = {"status": 200, "choice": {"message": {"role": "assistant", "content": ""}, "finish_reason": "length"}}
    native_agents: list = []
    with _PolicyFixture(summary) as policy:
        app = _agent(tmp_path, monkeypatch, policy.server.server_port, native_agents)
        try:
            (response,) = _run(app, "rollout-empty")
        finally:
            _close(native_agents)
        summaries, mains = policy.summaries(), policy.mains()

    from agent.context_compressor import SUMMARY_PREFIX

    assert len(summaries) == 1 and len(mains) == 1
    assert response.status == "completed"
    compacted = [m["content"] for m in mains[0][1]["messages"] if SUMMARY_PREFIX in (m.get("content") or "")]
    # Hermes merges the marker into the first kept turn when a standalone message would break role alternation.
    assert len(compacted) == 1
    assert compacted[0] == SUMMARY_PREFIX or compacted[0].startswith(f"{SUMMARY_PREFIX}\n\nhistory message")


def test_failed_summary_drops_the_middle_turns_and_the_run_continues(monkeypatch, tmp_path):
    native_agents: list = []
    with _PolicyFixture({"status": 400}) as policy:
        app = _agent(tmp_path, monkeypatch, policy.server.server_port, native_agents)
        try:
            (response,) = _run(app, "rollout-refused")
        finally:
            _close(native_agents)
        summaries, mains = policy.summaries(), policy.mains()

    assert len(summaries) == 1 and len(mains) == 1
    assert response.status == "completed"
    main_messages = mains[0][1]["messages"]
    assert not any("[CONTEXT COMPACTION]" in (m.get("content") or "") for m in main_messages)
    kept = [m["content"] for m in main_messages if (m.get("content") or "").startswith("history message")]
    assert 0 < len(kept) < 26


def test_compression_outside_a_run_keeps_hermes_resolution(monkeypatch, tmp_path):
    """Without a bound run, Hermes resolves its own compression endpoint as configured."""
    import yaml

    from responses_api_agents.hermes_agent.app import _COMPRESSION_ROUTE, _route_compression_calls

    summary = {
        "status": 200,
        "choice": {"message": {"role": "assistant", "content": SUMMARY_TEXT}, "finish_reason": "stop"},
    }
    with _PolicyFixture(summary) as configured:
        home = tmp_path / "hermes"
        home.mkdir()
        (home / "config.yaml").write_text(
            yaml.dump(
                {
                    "auxiliary": {
                        "compression": {
                            "base_url": f"http://127.0.0.1:{configured.server.server_port}/configured/v1",
                            "api_key": "fixture",  # pragma: allowlist secret
                            "model": "configured-summary-model",
                        }
                    }
                }
            )
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        _route_compression_calls()
        _route_compression_calls()
        import agent.context_compressor as context_compressor

        assert context_compressor.call_llm.routes_compression is True
        assert _COMPRESSION_ROUTE.get() is None
        response = context_compressor.call_llm(
            "compression",
            messages=[{"role": "user", "content": "Create a structured handoff summary of x"}],
            temperature=0.3,
            max_tokens=4000,
        )
        ((path, body),) = configured.requests

    assert response.choices[0].message.content == SUMMARY_TEXT
    assert path == "/configured/v1/chat/completions"
    assert body["model"] == "configured-summary-model"


def test_summary_request_carries_the_policy_turns_request_fields(monkeypatch, tmp_path):
    """The captured summary request carries every field a policy turn sends except tools and the turn's budget."""
    summary = {
        "status": 200,
        "choice": {"message": {"role": "assistant", "content": SUMMARY_TEXT}, "finish_reason": "stop"},
    }
    native_agents: list = []
    with _PolicyFixture(summary) as policy:
        app = _agent(
            tmp_path,
            monkeypatch,
            policy.server.server_port,
            native_agents,
            temperature=1.0,
            compression_summary_temperature=1.0,
        )
        try:
            _run(app, "rollout-fields")
        finally:
            _close(native_agents)
        ((summary_path, summary_body),) = policy.summaries()
        ((main_path, main_body),) = policy.mains()

    assert summary_path == main_path == "/ng-rollout/rollout-fields/v1/chat/completions"
    assert (
        summary_body["chat_template_kwargs"]
        == main_body["chat_template_kwargs"]
        == {
            "enable_thinking": True,
            "truncate_history_thinking": False,
        }
    )
    assert summary_body["model"] == main_body["model"] == "fixture-policy"
    assert summary_body["temperature"] == main_body["temperature"] == 1.0
    assert "tools" not in summary_body and "tool_choice" not in summary_body
    assert summary_body["max_tokens"] == 32768 and main_body["max_tokens"] == 8192
    assert [message["role"] for message in summary_body["messages"]] == ["user"]
    # Nothing else differs: token IDs and logprobs are sampled by the serving endpoint, not requested here.
    ignored = {"messages", "max_tokens", "tools", "tool_choice"}
    assert {k: v for k, v in summary_body.items() if k not in ignored} == {
        k: v for k, v in main_body.items() if k not in ignored
    }


def test_summary_request_without_template_kwargs_matches_policy_turns(monkeypatch, tmp_path):
    """With chat_template_kwargs disabled, neither the policy turn nor the summary sends them."""
    summary = {
        "status": 200,
        "choice": {"message": {"role": "assistant", "content": SUMMARY_TEXT}, "finish_reason": "stop"},
    }
    native_agents: list = []
    with _PolicyFixture(summary) as policy:
        app = _agent(
            tmp_path, monkeypatch, policy.server.server_port, native_agents, chat_template_kwargs_enabled=False
        )
        try:
            _run(app, "rollout-plain")
        finally:
            _close(native_agents)
        ((_, summary_body),) = policy.summaries()
        ((_, main_body),) = policy.mains()

    assert "chat_template_kwargs" not in summary_body and "chat_template_kwargs" not in main_body
    assert summary_body["temperature"] == 0.3


@pytest.mark.parametrize("target_ratio, expected", [(None, 0.20), (0.10, 0.10)])
def test_compression_target_ratio_reaches_hermes_compressor(monkeypatch, tmp_path, target_ratio, expected):
    import yaml

    native_agents: list = []
    summary = {
        "status": 200,
        "choice": {"message": {"role": "assistant", "content": SUMMARY_TEXT}, "finish_reason": "stop"},
    }
    with _PolicyFixture(summary) as policy:
        extra = {} if target_ratio is None else {"compression_target_ratio": target_ratio}
        app = _agent(tmp_path, monkeypatch, policy.server.server_port, native_agents, **extra)
        config = yaml.safe_load((Path(os.environ["HERMES_HOME"]) / "config.yaml").read_text())
        assert config["compression"].get("target_ratio") == target_ratio
        try:
            _run(app, "rollout-ratio")
        finally:
            _close(native_agents)
    compressor = native_agents[0].context_compressor
    assert compressor.summary_target_ratio == expected
    assert compressor.tail_token_budget == int(compressor.threshold_tokens * expected)


@pytest.mark.parametrize("budget", [None, 16384, 49152])
def test_summary_output_budget_reaches_compressor_request(monkeypatch, tmp_path, budget):
    import yaml

    native_agents: list = []
    summary = {
        "status": 200,
        "choice": {"message": {"role": "assistant", "content": SUMMARY_TEXT}, "finish_reason": "stop"},
    }
    expected = 32768 if budget is None else budget
    with _PolicyFixture(summary) as policy:
        extra = {} if budget is None else {"compression_summary_max_output_tokens": budget}
        app = _agent(tmp_path, monkeypatch, policy.server.server_port, native_agents, **extra)
        generated = yaml.safe_load(app._build_config())
        written = yaml.safe_load((Path(os.environ["HERMES_HOME"]) / "config.yaml").read_text())
        assert written == generated
        assert generated["compression"]["summary_max_output_tokens"] == expected
        try:
            (response,) = _run(app, "rollout-budget")
        finally:
            _close(native_agents)
        ((_, body),) = policy.summaries()
        ((_, main),) = policy.mains()
    assert response.status == "completed"
    compressor = native_agents[0].context_compressor
    assert compressor.summary_max_output_tokens == expected
    assert body["max_tokens"] == expected
    assert "Target ~" in body["messages"][0]["content"]
    assert f"Target ~{expected} tokens" not in body["messages"][0]["content"]
    assert body["chat_template_kwargs"]["enable_thinking"] is True
    assert body["chat_template_kwargs"] == main["chat_template_kwargs"]
    assert compressor.tail_token_budget == int(compressor.threshold_tokens * 0.20)
    assert any(SUMMARY_TEXT in (message.get("content") or "") for message in main["messages"])


@pytest.mark.parametrize("content, kept", [(SUMMARY_TEXT, SUMMARY_TEXT), ("", "")])
def test_compaction_observation_names_the_served_summary_call(monkeypatch, tmp_path, content, kept):
    from nemo_gym.rollout_observability import ContextCompactionObservation

    summary = {
        "status": 200,
        "choice": {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"},
    }
    native_agents: list = []
    with _PolicyFixture(summary) as policy:
        app = _agent(tmp_path, monkeypatch, policy.server.server_port, native_agents)
        try:
            episode = asyncio.run(
                app._create_episode(
                    NeMoGymResponseCreateParamsNonStreaming(input=_long_history_input()), rollout_id="rollout-o"
                )
            )
        finally:
            _close(native_agents)
        served = [(index + 1, body) for index, (_, body) in enumerate(policy.requests)]

    summaries = [index for index, body in served if _is_summary_request(body)]
    mains = [index for index, body in served if not _is_summary_request(body)]
    records = [r for r in episode.observations.records if isinstance(r, ContextCompactionObservation)]
    # Hermes' preflight repeats compression; the pass with nothing left to
    # summarize requests no summary, keeps the context, and is not a compaction.
    noops = [r for r in records if r.outcome == "aborted"]
    assert noops and all(r.model_calls == [] and r.tokens_before == r.tokens_after for r in noops)
    compactions = [r for r in records if r.outcome != "aborted"]
    assert summaries and len(compactions) == len(summaries)
    policy_ref = ModelServerRef(type="responses_api_models", name="policy")
    for compaction, served_index in zip(compactions, summaries, strict=True):
        # Each compaction names the summary call it was served, and the main
        # calls on either side of it.
        assert compaction.outcome == "completed"
        assert compaction.summary == kept
        assert [ref.response_id for ref in compaction.model_calls] == [f"fixture-{served_index}"]
        assert compaction.model_calls[0].model_ref == policy_ref
        earlier = [index for index in mains if index < served_index]
        later = [index for index in mains if index > served_index]
        assert (compaction.before_model_call is None) == (not earlier)
        if earlier:
            assert compaction.before_model_call.response_id == f"fixture-{earlier[-1]}"
        assert compaction.after_model_call.response_id == f"fixture-{later[0]}"
    codes = {gap.code for gap in episode.observations.gaps}
    assert not codes & {
        "compaction_summary_call_unavailable",
        "compaction_model_call_boundary_unavailable",
        "compaction_summary_unavailable",
        "compaction_after_model_call_unavailable",
    }
