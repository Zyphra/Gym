# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""max_turns through the installed Pi CLI against a loopback chat-completions endpoint."""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import MagicMock

import pytest

import nemo_gym.base_responses_api_agent as base_agent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.anyswe_agent.app import _should_mask_sample
from responses_api_agents.pi_agent.app import PiAgent, PiAgentConfig


pytestmark = pytest.mark.skipif(not os.environ.get("TASK_STOCK_PI_RUNTIME"), reason="native Pi runtime not supplied")


class _Endpoint:
    """Every request is answered with one bash call that appends to a marker file."""

    def __init__(self, marker: Path, extra: str = "") -> None:
        self.requests: list[dict] = []
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                endpoint.requests.append(body)
                n = len(endpoint.requests)
                call = {
                    "index": 0,
                    "id": f"call_{n}",
                    "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"command": f"echo {n} >> {marker}{extra}"})},
                }
                chunks = [
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": "", "tool_calls": [call]},
                                "finish_reason": None,
                            }
                        ]
                    },
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
                    {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
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


def _episode(monkeypatch, tmp_path, max_turns, extra="", **config):
    runtime = Path(os.environ["TASK_STOCK_PI_RUNTIME"])
    marker = tmp_path / "turns.txt"
    endpoint = _Endpoint(marker, extra)
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
            timeout=60,
            max_turns=max_turns,
            **config,
            command=f"{runtime / 'bin/node'} {runtime / 'bin/pi'}",
            workspace_root=str(tmp_path / "pi"),
            resources_server=ResourcesServerRef(type="resources_servers", name=""),
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        ),
        server_client=client,
    )
    try:
        episode = asyncio.run(
            agent._create_episode(NeMoGymResponseCreateParamsNonStreaming(input="go"), rollout_id="r")
        )
    finally:
        endpoint.httpd.shutdown()
    return episode.response, endpoint.requests, marker.read_text().split()


def test_max_turns_ends_after_the_last_turns_tools(monkeypatch, tmp_path):
    response, requests, ran = _episode(monkeypatch, tmp_path, max_turns=3)
    # Three requests reach the endpoint, the tools of all three turns run, and the fourth is never sent.
    assert len(requests) == 3 and ran == ["1", "2", "3"]
    assert response.error is None and response.status is None
    assert response.metadata == {"stop_reason": "max_iterations", "pi_stop_detail": "Pi reached max_turns=3"}
    assert [item.type for item in response.output].count("function_call") == 3
    assert not _should_mask_sample(True, "max_iteration", False, None, "max_iterations")


@pytest.mark.skipif(
    subprocess.run(["unshare", "--user", "--mount", "true"]).returncode != 0,
    reason="this host allows no unprivileged mount namespace",
)
def test_hidden_harness_mounts_are_empty_for_pi_bash(monkeypatch, tmp_path):
    """Pi's own bash tool runs through the hiding shell: a harness mount reads empty, the harness keeps it."""
    mount = Path(tempfile.mkdtemp(prefix="nemo-gym-", dir="/tmp"))
    try:
        (mount / "params.json").write_text("{}")
        listing = f"; ls -A {mount} | wc -l"
        response, _, ran = _episode(monkeypatch, tmp_path, 1, listing, hide_harness_mounts=True)
        output = [item.output for item in response.output if item.type == "function_call_output"]
        assert ran == ["1"] and output[0].strip() == "0"
        (tmp_path / "plain").mkdir()
        response, _, _ = _episode(monkeypatch, tmp_path / "plain", 1, listing)
        output = [item.output for item in response.output if item.type == "function_call_output"]
        assert output[0].strip() == "1" and (mount / "params.json").read_text() == "{}"
    finally:
        shutil.rmtree(mount)
