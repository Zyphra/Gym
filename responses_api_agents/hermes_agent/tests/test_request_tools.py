# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unmodified upstream Hermes with a request's tools, against recording policy and resources servers.

A rollout endpoint continues an episode only on the turns it served, so every request must replay
each sampled assistant turn exactly: names, calls and arguments as the policy wrote them.
"""

import asyncio
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.hermes_agent.tests.test_stock_compaction import (
    _SUMMARY_OK,
    _agent,
    _long_history_input,
    _PolicyFixture,
)


# An MCP tool name over the 64 characters some providers allow, as MiMo rows offer them.
LONG = "mcp__consumer_compliance_data_quality_control_register__run_response_timestamp_test"
REQUEST_TOOLS = [
    {
        "type": "function",
        "strict": False,
        "name": name,
        "description": f"The row's {name} tool.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    }
    for name in ("Bash", "Read", LONG)
]


class _Resources:
    """The resources server stand-in: records each tool call and its session cookie."""

    def __init__(self):
        self.calls: list[tuple[str, dict, str]] = []
        resources = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                resources.calls.append((self.path, body, self.headers.get("cookie")))
                text = f"ran {self.path[1:]}".encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(text)))
                self.end_headers()
                self.wfile.write(text)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"


def _call(call_id: str, name: str, path: str) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps({"path": path})}}


def _request_tools_agent(tmp_path, monkeypatch, policy, resources, **config):
    app = _agent(
        tmp_path,
        monkeypatch,
        policy.server.server_port,
        [],
        request_tools=True,
        enabled_toolsets=[],
        **config,
    )
    app.server_client._resolve_base_url = lambda _name: resources.url
    return app


def _run(app, inputs, cookies):
    body = NeMoGymResponseCreateParamsNonStreaming(input=inputs, tools=REQUEST_TOOLS)
    return asyncio.run(app._create_response(body, rollout_id="rollout-a", cookies=cookies))


def _offered(body: dict) -> list[str]:
    return sorted(tool["function"]["name"] for tool in body.get("tools") or [])


def test_sampled_turns_are_replayed_as_sampled_and_request_tools_run_in_the_session(monkeypatch, tmp_path):
    # One turn with an unknown-case name (`bash`) and a repeated call, then the answer.
    sampled = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "Look at the file twice.\n",
        "tool_calls": [
            _call("call_a", "bash", "x"),
            _call("call_b", "Read", "y"),
            _call("call_c", "Read", "y"),
            _call("call_d", LONG, "z"),
        ],
    }
    resources = _Resources()
    with _PolicyFixture(_SUMMARY_OK, main_turns=[sampled]) as policy:
        app = _request_tools_agent(tmp_path, monkeypatch, policy, resources, max_turns=4)
        response = _run(app, [NeMoGymEasyInputMessage(role="user", content="Do the task.")], {"session": "s-1"})
        mains = policy.mains()

    assert response.status == "completed"
    assert len(mains) == 2
    # The row's tools are the only tools offered.
    assert all(_offered(body) == sorted(["Bash", "Read", LONG]) for _, body in mains)
    replayed = [message for message in mains[1][1]["messages"] if message["role"] == "assistant"]
    assert len(replayed) == 1
    # Names, call count and arguments as sampled: no name repair, no duplicate dropped.
    assert [
        (c["id"], c["function"]["name"], json.loads(c["function"]["arguments"])) for c in replayed[0]["tool_calls"]
    ] == [(c["id"], c["function"]["name"], json.loads(c["function"]["arguments"])) for c in sampled["tool_calls"]]
    assert replayed[0]["reasoning_content"] == sampled["reasoning_content"]
    # Both Read calls ran on the resources server in the episode's session; `bash` got Hermes'
    # unknown-tool result and never reached it.
    assert sorted(resources.calls) == sorted(
        [("/Read", {"path": "y"}, "session=s-1")] * 2 + [(f"/{LONG}", {"path": "z"}, "session=s-1")]
    )
    results = {m["tool_call_id"]: m["content"] for m in mains[1][1]["messages"] if m["role"] == "tool"}
    assert "does not exist" in results["call_a"]
    assert "ran Read" in results["call_b"] and "ran Read" in results["call_c"]


def test_request_tools_survive_the_compaction_tool_refresh(monkeypatch, tmp_path):
    resources = _Resources()
    turns = [
        {"role": "assistant", "content": "", "tool_calls": [_call("call_a", "Read", "a")]},
        {"role": "assistant", "content": "", "tool_calls": [_call("call_b", "Read", "b")]},
    ]
    with _PolicyFixture(_SUMMARY_OK, main_turns=turns) as policy:
        app = _request_tools_agent(tmp_path, monkeypatch, policy, resources)
        response = _run(app, _long_history_input(), {"session": "s-2"})
        summaries, mains = policy.summaries(), policy.mains()

    assert response.status in ("completed", "incomplete")
    assert summaries, "the long history compacts"
    # Hermes rebuilds its tool surface from its toolsets after compacting; the row's tools stay.
    assert mains and all(_offered(body) == sorted(["Bash", "Read", LONG]) for _, body in mains)
    assert [path for path, _, _ in resources.calls] == ["/Read"] * len(resources.calls) and resources.calls


def test_runtime_process_runs_the_same_episode(monkeypatch, tmp_path):
    # This interpreter is a Hermes runtime: run the episode in one process of it.
    sampled = {"role": "assistant", "content": "", "tool_calls": [_call("call_a", "Read", "z")]}
    resources = _Resources()
    with _PolicyFixture(_SUMMARY_OK, main_turns=[sampled]) as policy:
        app = _request_tools_agent(tmp_path, monkeypatch, policy, resources, hermes_runtime=sys.prefix)
        response = _run(app, [NeMoGymEasyInputMessage(role="user", content="Do the task.")], {"session": "s-3"})
        mains = policy.mains()

    assert response.status == "completed"
    assert resources.calls == [("/Read", {"path": "z"}, "session=s-3")]
    assert [path for path, _ in mains] == ["/ng-rollout/rollout-a/v1/chat/completions"] * 2
