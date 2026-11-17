# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real Pi with a request's tools (request_tools.ts), against loopback policy and resources servers.

Runs when ``TASK_STOCK_PI_RUNTIME`` names a prepared Pi runtime.
"""

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from starlette.requests import Request

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.pi_agent.app import PiAgent, PiAgentConfig


REQUEST_TOOLS = [
    {
        "type": "function",
        "name": "Bash",
        "description": "Executes a given bash command and returns its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        "strict": False,
    },
    {
        "type": "function",
        "name": "mcp__crm__get_account",
        "description": "Read one CRM account.",
        "parameters": {"type": "object", "properties": {"account_id": {"type": "string"}}, "required": ["account_id"]},
        "strict": False,
    },
]
SESSION_COOKIE = "session=episode-42"
FINAL = "The account is ACME and the workspace holds report.md."


def _serve(handler) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _policy(requests: list) -> ThreadingHTTPServer:
    """Chat completions: one turn calling both request tools, then the final answer (streamed or not)."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            requests.append(body)
            if len(requests) == 1:
                message = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "mcp__crm__get_account",
                                "arguments": json.dumps({"account_id": "A1"}),
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "Bash", "arguments": json.dumps({"command": "ls"})},
                        },
                    ],
                }
                finish = "tool_calls"
            else:
                message, finish = {"role": "assistant", "content": FINAL}, "stop"
            usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            if body.get("stream"):
                delta = dict(message)
                if "tool_calls" in delta:
                    delta["tool_calls"] = [dict(c, index=i) for i, c in enumerate(delta["tool_calls"])]
                chunks = [
                    {
                        "id": "c",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": body["model"],
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                    },
                    {
                        "id": "c",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": body["model"],
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                        "usage": usage,
                    },
                ]
                payload, kind = (
                    "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n",
                    "text/event-stream",
                )
            else:
                payload = json.dumps(
                    {
                        "id": "c",
                        "object": "chat.completion",
                        "created": 0,
                        "model": body["model"],
                        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                        "usage": usage,
                    }
                )
                kind = "application/json"
            data = payload.encode()
            self.send_response(200)
            self.send_header("content-type", kind)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return _serve(Handler)


def _resources(calls: list) -> ThreadingHTTPServer:
    """The resources server's tool route: POST /<tool>, plain-text reply."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            name = self.path.lstrip("/")
            args = json.loads(self.rfile.read(int(self.headers["content-length"])) or b"{}")
            calls.append((name, args, self.headers.get("cookie", "")))
            data = f"{name} output for {json.dumps(args, sort_keys=True)}".encode()
            self.send_response(200)
            self.send_header("content-type", "text/plain")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return _serve(Handler)


def test_pi_offers_and_runs_only_the_request_tools_in_the_session(tmp_path):
    runtime = os.environ.get("TASK_STOCK_PI_RUNTIME")
    if not runtime:
        pytest.skip("TASK_STOCK_PI_RUNTIME names no prepared Pi runtime")
    runtime = Path(runtime)
    requests, calls = [], []
    policy, resources = _policy(requests), _resources(calls)
    try:
        client = ServerClient.model_construct(
            global_config_dict=OmegaConf.create(
                {
                    "resource": {
                        "resources_servers": {"resource": {"host": "127.0.0.1", "port": resources.server_port}}
                    },
                    "policy": {
                        "responses_api_models": {"vllm_model": {"host": "127.0.0.1", "port": policy.server_port}}
                    },
                }
            )
        )
        agent = PiAgent(
            config=PiAgentConfig(
                host="127.0.0.1",
                port=0,
                name="pi",
                entrypoint="app.py",
                resources_server={"type": "resources_servers", "name": "resource"},
                model_server={"type": "responses_api_models", "name": "policy"},
                model="policy",
                command=f"{runtime}/bin/node {runtime}/bin/pi",
                request_tools=True,
                extra_args=[
                    "--no-builtin-tools",
                    "--no-extensions",
                    "--no-skills",
                    "--no-prompt-templates",
                    "--no-context-files",
                    "--offline",
                ],
                workspace_root=str(tmp_path),
                timeout=300,
            ),
            server_client=client,
        )
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/responses",
                "headers": [(b"cookie", SESSION_COOKIE.encode())],
                "path_params": {},
            }
        )
        body = NeMoGymResponseCreateParamsNonStreaming.model_validate(
            {
                "input": [{"role": "user", "content": "Look up account A1 and list the workspace."}],
                "tools": REQUEST_TOOLS,
            }
        )
        response = asyncio.run(agent.responses(request, body))
    finally:
        policy.shutdown()
        resources.shutdown()
    offered = {tuple(sorted(t["function"]["name"] for t in r.get("tools") or [])) for r in requests}
    assert offered == {("Bash", "mcp__crm__get_account")}
    assert sorted((name, args) for name, args, _ in calls) == [
        ("Bash", {"command": "ls"}),
        ("mcp__crm__get_account", {"account_id": "A1"}),
    ]
    assert {cookie for _, _, cookie in calls} == {SESSION_COOKIE}
    # A run that ended on its own completed.
    assert response.status == "completed"
    texts = [
        part.text
        for item in response.output
        if getattr(item, "type", None) == "message"
        for part in item.content
        if getattr(part, "type", None) == "output_text"
    ]
    assert texts and texts[-1] == FINAL
