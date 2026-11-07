# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unmodified upstream Hermes compaction against a recording OpenAI-compatible policy endpoint.

The agent runs Hermes' own ``AIAgent`` and ``ContextCompressor``; Gym only writes Hermes'
config. The fixture records every request so the wire contract the rollout endpoint relies on
is asserted directly: one per-rollout URL for every call, no client-chosen sampling parameters
(the endpoint supplies the policy's), echoed reasoning, and no auxiliary calls besides the
compression summary.
"""

import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
import yaml
from omegaconf import OmegaConf

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.hermes_agent.app import HermesAgent, HermesAgentConfig


SUMMARY_TEXT = "## Goal\nVerified fixture summary."
MAIN_TEXT = "Main fixture response."
CONTEXT_WINDOW = 131072
# The opening of upstream Hermes' summarizer prompt (agent/context_compressor.py _build_summary_prompt).
UPSTREAM_SUMMARY_OPENING = "You are a summarization agent creating a context checkpoint."


def _is_summary_request(body: dict) -> bool:
    first = body["messages"][0]["content"]
    return isinstance(first, str) and first.startswith(UPSTREAM_SUMMARY_OPENING)


class _PolicyFixture:
    """A local OpenAI-compatible policy server recording every request it serves."""

    def __init__(self, summary: dict, main_turns: list[dict] | None = None):
        self.requests: list[tuple[str, dict]] = []
        self.probes: list[str] = []
        self.refused_streams: list[str] = []
        self.main_turns = list(main_turns or [])
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if not self.path.endswith("/chat/completions"):
                    # Hermes' local-server probes (Ollama /api/show for num_ctx) are not policy requests.
                    fixture.probes.append(self.path)
                    self.send_error(404)
                    return
                if body.get("stream"):
                    # The rollout endpoint serves whole completions only and refuses a streamed request
                    # (recipe/nemo_async/episode_endpoint.py); Hermes then repeats it without streaming.
                    fixture.refused_streams.append(self.path)
                    encoded = json.dumps({"error": {"message": "non-streaming generation only"}}).encode()
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)
                    return
                fixture.requests.append((self.path, body))
                prompt_tokens = len(json.dumps(body["messages"])) // 4
                if _is_summary_request(body):
                    status, choice = summary["status"], summary.get("choice")
                else:
                    status = 200
                    message = (
                        fixture.main_turns.pop(0)
                        if fixture.main_turns
                        else {"role": "assistant", "content": MAIN_TEXT}
                    )
                    choice = {
                        "message": message,
                        "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                    }
                payload = (
                    {
                        "id": f"fixture-{len(fixture.requests)}",
                        "object": "chat.completion",
                        "created": 1,
                        "model": body["model"],
                        "choices": [{"index": 0, **choice}],
                        # Hermes triggers compaction from the provider's reported prompt size.
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": 5,
                            "total_tokens": prompt_tokens + 5,
                        },
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

    def summaries(self) -> list[tuple[str, dict]]:
        return [(path, body) for path, body in self.requests if _is_summary_request(body)]

    def mains(self) -> list[tuple[str, dict]]:
        return [(path, body) for path, body in self.requests if not _is_summary_request(body)]


def _long_history_input() -> list[NeMoGymEasyInputMessage]:
    # 34 earlier messages of ~13,800 characters each (~117K estimated tokens): above Hermes'
    # trigger for a 131,072-token window (at least 75% of the window less the 8,192-token reply
    # reserve) and inside the window, so the run compresses before its first request.
    items = [
        NeMoGymEasyInputMessage(
            role="user" if index % 2 == 0 else "assistant",
            content=f"history message {index}: " + "observed fixture state. " * 575,
        )
        for index in range(34)
    ]
    items.append(NeMoGymEasyInputMessage(role="user", content="Continue the fixture task."))
    return items


def _agent(tmp_path, monkeypatch, port: int, native_agents: list, **config) -> HermesAgent:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    import run_agent

    class RecordedAIAgent(run_agent.AIAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            native_agents.append(self)

    monkeypatch.setattr(run_agent, "AIAgent", RecordedAIAgent)
    client = ServerClient.model_construct(
        global_config_dict=OmegaConf.create(
            {"policy": {"responses_api_models": {"openai_model": {"host": "127.0.0.1", "port": port}}}}
        )
    )
    client._build_server_base_url = lambda _cfg: f"http://127.0.0.1:{port}"
    settings = dict(
        max_turns=3,
        max_tokens=8192,
        context_window=CONTEXT_WINDOW,
        enabled_toolsets=[],
        compression_enabled=True,
        terminal_backend="local",
    )
    settings.update(config)
    return HermesAgent(
        config=HermesAgentConfig(
            host="127.0.0.1",
            port=0,
            name="hermes",
            entrypoint="app.py",
            model_server=ModelServerRef(type="responses_api_models", name="policy"),
            resources_server=ResourcesServerRef(type="resources_servers", name="resource"),
            model="fixture-policy",
            **settings,
        ),
        server_client=client,
    )


def _run(app: HermesAgent, inputs, *rollout_ids: str):
    async def exercise():
        return await asyncio.gather(
            *(
                app._create_response(NeMoGymResponseCreateParamsNonStreaming(input=inputs), rollout_id=rollout_id)
                for rollout_id in rollout_ids
            )
        )

    return asyncio.run(exercise())


_SUMMARY_OK = {
    "status": 200,
    "choice": {"message": {"role": "assistant", "content": SUMMARY_TEXT}, "finish_reason": "stop"},
}


def _todo_turn(call_id: str = "call_0") -> dict:
    # One tool turn, so the run continues past the response whose usage crosses the trigger.
    arguments = json.dumps({"todos": [{"id": "1", "content": "fixture step", "status": "in_progress"}]})
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "todo", "arguments": arguments}}],
    }


def test_stock_summary_request_goes_to_each_rollouts_own_policy(monkeypatch, tmp_path, record_property):
    native_agents: list = []
    with _PolicyFixture(_SUMMARY_OK, main_turns=[_todo_turn("call_a"), _todo_turn("call_b")]) as policy:
        app = _agent(tmp_path, monkeypatch, policy.server.server_port, native_agents, enabled_toolsets=["todo"])
        responses = _run(app, _long_history_input(), "rollout-a", "rollout-b")
        summaries, mains = policy.summaries(), policy.mains()

    assert [response.status for response in responses] == ["completed", "completed"]
    for native in native_agents:
        assert native.context_compressor.context_length == CONTEXT_WINDOW
    # Every request of a run, the summary included, is served on that run's own rollout URL.
    assert (
        {path for path, _ in summaries}
        == {path for path, _ in mains}
        == {
            "/ng-rollout/rollout-a/v1/chat/completions",
            "/ng-rollout/rollout-b/v1/chat/completions",
        }
    )
    # Only main turns and compression summaries reach the policy: no title generation,
    # background review, memory or approval calls.
    assert len(summaries) + len(mains) == len(policy.requests)
    record_property("summary_requests", len(summaries))
    record_property("main_requests", len(mains))
    # Hermes streams its compression call; each refused stream is repeated once without streaming.
    record_property("refused_streams", len(policy.refused_streams))
    assert len(summaries) == 2 and sorted(policy.refused_streams) == sorted(path for path, _ in summaries)
    for _, body in summaries:
        # Upstream's own request: one user message, no tools, no output cap and no sampling
        # parameters, so the rollout endpoint samples it at the policy's temperature.
        assert [message["role"] for message in body["messages"]] == ["user"]
        assert "TASK BEING WORKED ON" not in body["messages"][0]["content"]
        assert "tools" not in body and "tool_choice" not in body
        assert "max_tokens" not in body and "max_completion_tokens" not in body
        assert not {"temperature", "top_p", "top_k"} & body.keys()
    for _, body in mains:
        assert not {"temperature", "top_p", "top_k"} & body.keys()
        assert body.get("stream") in (None, False)
        assert body["max_tokens"] == 8192
    # The turn after each compaction carries that run's summary.
    for rollout in ("rollout-a", "rollout-b"):
        last = [body for path, body in mains if f"/{rollout}/" in path][-1]
        assert "Verified fixture summary." in json.dumps(last["messages"])


def test_hermes_config_carries_the_compaction_trigger(monkeypatch, tmp_path):
    native_agents: list = []
    with _PolicyFixture(_SUMMARY_OK) as policy:
        app = _agent(
            tmp_path,
            monkeypatch,
            policy.server.server_port,
            native_agents,
            compression_threshold=0.75,
            compression_target_ratio=0.10,
            api_timeout=5400,
        )
        _run(app, "hi", "rollout-c")
    import os

    written = yaml.safe_load((Path(os.environ["HERMES_HOME"]) / "config.yaml").read_text())
    assert written["compression"] == {"enabled": True, "threshold": 0.75, "target_ratio": 0.10}
    assert written["model"]["context_length"] == CONTEXT_WINDOW and written["model"]["streaming"] is False
    assert written["auxiliary"] == {"title_generation": {"enabled": False}, "compression": {"timeout": 5400}}
    [native] = native_agents
    compressor = native.context_compressor
    assert compressor.context_length == CONTEXT_WINDOW
    assert compressor.threshold_percent == 0.75 and compressor.summary_target_ratio == 0.10
    # Hermes probes a local server for an Ollama context size; that is not a policy request.
    assert {path for path, _ in policy.requests} == {"/ng-rollout/rollout-c/v1/chat/completions"}


def test_reasoning_is_echoed_on_the_next_request(monkeypatch, tmp_path):
    turns = [
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "fixture reasoning before the call",
            "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "todo", "arguments": "{}"}}],
        },
        {"role": "assistant", "content": "done", "reasoning_content": "fixture reasoning before the answer"},
    ]
    native_agents: list = []
    with _PolicyFixture(_SUMMARY_OK, main_turns=turns) as policy:
        app = _agent(
            tmp_path,
            monkeypatch,
            policy.server.server_port,
            native_agents,
            enabled_toolsets=["todo"],
            compression_enabled=False,
        )
        [response] = _run(app, "use the todo tool once", "rollout-r")
        mains = policy.mains()
    assert response.status == "completed"
    assert len(mains) == 2
    [echoed] = [m for m in mains[1][1]["messages"] if m.get("role") == "assistant"]
    # The endpoint commits the reasoning-bearing message it served; Hermes must echo it.
    assert "fixture reasoning before the call" in (echoed.get("reasoning_content") or echoed.get("reasoning") or "")
    reasoning_items = [item for item in response.output if getattr(item, "type", None) == "reasoning"]
    assert [item.summary[0].text for item in reasoning_items][:1] == ["fixture reasoning before the call"]


@pytest.mark.parametrize(
    "summary, summary_requests",
    [
        ({"status": 200, "choice": {"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}}, 1),
        (
            {
                "status": 200,
                "choice": {"message": {"role": "assistant", "content": SUMMARY_TEXT}, "finish_reason": "length"},
            },
            1,
        ),
        ({"status": 400}, 1),
    ],
    ids=["empty", "length", "error"],
)
def test_rejected_summary_policy_calls_are_counted(monkeypatch, tmp_path, summary, summary_requests, record_property):
    """Upstream rejects an empty, length-stopped or failed summary without asking the policy again
    (its one retry is reserved for a distinct summary model), so no extra summary is sampled."""
    native_agents: list = []
    with _PolicyFixture(summary, main_turns=[_todo_turn()]) as policy:
        app = _agent(tmp_path, monkeypatch, policy.server.server_port, native_agents, enabled_toolsets=["todo"])
        [response] = _run(app, _long_history_input(), "rollout-f")
        summaries, mains = policy.summaries(), policy.mains()
    record_property("summary_requests", len(summaries))
    record_property("main_requests", len(mains))
    assert response.status in ("completed", "incomplete")
    assert len(summaries) == summary_requests, f"upstream sent {len(summaries)} summary requests"
    assert {path for path, _ in summaries + mains} == {"/ng-rollout/rollout-f/v1/chat/completions"}
