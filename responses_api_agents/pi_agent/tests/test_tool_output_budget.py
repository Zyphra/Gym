# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""tool_output_budget.ts through the installed Pi CLI against an endpoint that enforces its context window.

The endpoint counts one token per UTF-8 byte of the request's messages (what dense output costs a real
tokenizer) and answers a request whose prompt plus `max_tokens` passes the window with vLLM's 400 body."""

import asyncio
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import MagicMock

import pytest

import nemo_gym.base_responses_api_agent as base_agent
import responses_api_agents.pi_agent.app as pi_app
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.pi_agent.app import PiAgent, PiAgentConfig


pytestmark = pytest.mark.skipif(not os.environ.get("TASK_STOCK_PI_RUNTIME"), reason="native Pi runtime not supplied")

WINDOW = 40000
MAX_OUTPUT = 4096
TASK = "Inspect the numbers, then answer DONE."
# 2000 lines of digits, about 45 KB: under Pi's own 50 KB / 2000 line cap, above what the window can take.
DENSE = 'seq 1 1999 | awk \'{print "{\\"id\\":"$1",\\"v\\":["$1","$1","$1"]}"}\' | head -c 49000'
OVERFLOW = (
    '{"message":"This model\'s maximum context length is %d tokens. However, you requested %d output tokens '
    'and your prompt contains at least %d input tokens, for a total of at least %d tokens.","type":"BadRequestError",'
    '"param":"input_tokens","code":400}'
)


def _prompt_tokens(body: dict) -> int:
    total = 0
    for message in body["messages"]:
        content = message.get("content") or ""
        if not isinstance(content, str):
            content = "".join(block.get("text", "") for block in content)
        total += len(content.encode()) + 8
        for call in message.get("tool_calls") or []:
            total += len(call["function"]["arguments"].encode()) + 8
    return total + 500


class _Endpoint:
    def __init__(self, commands: list[str]) -> None:
        self.prompts: list[int] = []
        self.overflows = 0
        self.summaries = 0
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                prompt = _prompt_tokens(body)
                want = body.get("max_completion_tokens") or body.get("max_tokens") or 0
                if prompt + want > WINDOW:
                    endpoint.overflows += 1
                    raw = (OVERFLOW % (WINDOW, want, prompt, prompt + want)).encode()
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                if body.get("tools"):
                    endpoint.prompts.append(prompt)
                    n = len(endpoint.prompts)
                    if n <= len(commands):
                        call = {
                            "index": 0,
                            "id": f"call_{n}",
                            "type": "function",
                            "function": {"name": "bash", "arguments": json.dumps({"command": commands[n - 1]})},
                        }
                        delta, finish = {"role": "assistant", "content": "", "tool_calls": [call]}, "tool_calls"
                    else:
                        delta, finish = {"role": "assistant", "content": "DONE"}, "stop"
                else:
                    endpoint.summaries += 1
                    delta, finish = {"role": "assistant", "content": "## Goal\nInspect numbers."}, "stop"
                chunks = [
                    {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
                    {
                        "choices": [],
                        "usage": {"prompt_tokens": prompt, "completion_tokens": 20, "total_tokens": prompt + 20},
                    },
                ]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in chunks:
                    chunk = {
                        "id": "c",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": body["model"],
                        **chunk,
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        Thread(target=self.httpd.serve_forever, daemon=True).start()


def _episode(monkeypatch, tmp_path, commands):
    runtime = Path(os.environ["TASK_STOCK_PI_RUNTIME"])
    endpoint = _Endpoint(commands)
    monkeypatch.setenv("PATH", f"{runtime / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {})
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    client._build_server_base_url = lambda _cfg: endpoint.url
    agent = PiAgent(
        config=PiAgentConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="",
            name="pi",
            model="model",
            timeout=120,
            command=f"{runtime / 'bin/node'} {runtime / 'bin/pi'}",
            workspace_root=str(tmp_path / "pi"),
            resources_server=ResourcesServerRef(type="resources_servers", name=""),
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            max_turns=20,
            context_window=WINDOW,
            max_output_tokens=MAX_OUTPUT,
            compaction_enabled=True,
        ),
        server_client=client,
    )
    try:
        episode = asyncio.run(
            agent._create_episode(NeMoGymResponseCreateParamsNonStreaming(input=TASK), rollout_id="r")
        )
    finally:
        endpoint.httpd.shutdown()
    return episode.response, endpoint


def test_dense_output_is_cut_so_no_request_overflows(monkeypatch, tmp_path):
    response, endpoint = _episode(monkeypatch, tmp_path, [DENSE, DENSE])
    assert endpoint.overflows == 0
    assert all(prompt <= WINDOW for prompt in endpoint.prompts)
    assert response.metadata.get("stop_reason") is None
    assert int(response.metadata["pi_cut_tool_results"]) >= 1
    outputs = [item.output for item in response.output if item.type == "function_call_output"]
    assert any("Tool output cut to fit the context window" in output for output in outputs)
    assert [item for item in response.output if item.type == "message"][-1].content[0].text == "DONE"


def test_without_the_budget_the_same_episode_overflows(monkeypatch, tmp_path):
    # Control: the same episode with the extension replaced by one that does nothing.
    noop = tmp_path / "noop.ts"
    noop.write_text("export default function noop(_pi: any) {}\n")
    monkeypatch.setattr(pi_app, "TOOL_OUTPUT_BUDGET_EXTENSION", noop)
    response, endpoint = _episode(monkeypatch, tmp_path, [DENSE, DENSE])
    assert endpoint.overflows >= 1
    assert response.metadata["stop_reason"] == "context_exhausted"


def test_small_output_is_unchanged(monkeypatch, tmp_path):
    response, endpoint = _episode(monkeypatch, tmp_path, ["echo hello"])
    assert endpoint.overflows == 0
    assert response.metadata["pi_cut_tool_results"] == "0"
    outputs = [item.output for item in response.output if item.type == "function_call_output"]
    assert outputs == ["hello\n"] or outputs == ["hello"]
