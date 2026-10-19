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
"""mini-swe-agent against a scripted chat-completions server: every request extends the last."""

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest


pytest.importorskip("minisweagent")

import uvicorn  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef  # noqa: E402
from nemo_gym.server_utils import ServerClient  # noqa: E402
from responses_api_agents.mini_swe_agent_local.app import MiniSweAgent, MiniSweAgentConfig, output_items  # noqa: E402


def _call(command: str, call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": command})},
            }
        ],
    }


SCRIPT = [
    _call("sed -i 's/a - b/a + b/' calc.py", "call-1"),
    # No tool call: upstream drops this turn from the history; the agent keeps it.
    {"role": "assistant", "content": "The fix is done."},
    _call("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "call-2"),
]


class _Server:
    """A chat-completions endpoint that serves SCRIPT and records each request."""

    def __init__(self, ended: dict | None = None):
        self.requests: list[dict] = []
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def complete(request: Request):
            body = await request.json()
            self.requests.append(body)
            if ended is not None and len(self.requests) > 1:
                return JSONResponse(status_code=400, content={"error": ended})
            message = SCRIPT[min(len(self.requests), len(SCRIPT)) - 1]
            return {
                "id": f"cmpl-{len(self.requests)}",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if "tool_calls" in message else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        while not self.server.started:
            time.sleep(0.01)
        return self

    def __exit__(self, *_):
        self.server.should_exit = True
        self.thread.join()


def _agent() -> MiniSweAgent:
    config = MiniSweAgentConfig(
        host="0.0.0.0",
        port=0,
        name="mini",
        entrypoint="app.py",
        model_server=ModelServerRef(name="policy_model", type="responses_api_models"),
        resources_server=ResourcesServerRef(name="anyswe", type="resources_servers"),
        model="policy",
        step_limit=10,
    )
    return MiniSweAgent(config=config, server_client=ServerClient.model_construct(global_config_dict={}))


def _repo(path: Path) -> Path:
    (path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    for command in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
    ):
        subprocess.run(command, cwd=path, check=True)
    return path


def test_every_request_extends_the_previous_one_including_after_a_format_error(tmp_path):
    with _Server() as server:
        agent = _agent()._agent(f"http://127.0.0.1:{server.port}/v1", None, str(_repo(tmp_path)))
        info = agent.run("Fix add() in calc.py.")
    assert info["exit_status"] == "Submitted"
    assert "return a + b" in (tmp_path / "calc.py").read_text()
    requests = server.requests
    assert len(requests) == len(SCRIPT)
    for previous, served, current in zip(requests, SCRIPT, requests[1:]):
        # The next request is the previous one, the turn exactly as served, then new context.
        assert current["messages"][: len(previous["messages"]) + 1] == previous["messages"] + [served]
    assert all(request["tools"][0]["function"]["name"] == "bash" for request in requests)
    # The submitting call ends the run instead of returning an observation.
    kinds = [item.type for item in output_items(agent.messages)]
    assert kinds == ["message", "function_call", "function_call_output", "message", "message", "function_call"]


@pytest.mark.parametrize(
    ("code", "reason"),
    [("context_length_exceeded", "context_exhausted"), ("episode_terminated", "episode_terminated")],
)
def test_an_ended_episode_stops_the_agent_with_its_reason(tmp_path, code, reason):
    ended = {"message": "ended", "type": "invalid_request_error", "param": "messages", "code": code}
    with _Server(ended=ended) as server:
        agent = _agent()._agent(f"http://127.0.0.1:{server.port}/v1", None, str(_repo(tmp_path)))
        info = agent.run("Fix add() in calc.py.")
    assert info == {"exit_status": "EpisodeEnded", "stop_reason": reason, "submission": ""}
    assert len(server.requests) == 2


def test_the_agent_model_loads_litellm_without_a_cost_map_fetch():
    """litellm's background cost-map fetch raced its own import in the sandbox (KeyError: 'litellm')."""
    script = (
        "import json, threading\n"
        "from responses_api_agents.mini_swe_agent_local.app import endpoint_model\n"
        "endpoint_model(base_url='http://127.0.0.1:9/v1', model_name='policy', api_key='gym',"  # pragma: allowlist secret
        " sampling={}, settings={}, timeout=1)\n"
        "from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map_source_info\n"
        "info = get_model_cost_map_source_info()\n"
        "threads = [thread.name for thread in threading.enumerate()]\n"
        "print(json.dumps({'source': info['source'], 'forced': info['is_env_forced'], 'threads': threads}))\n"
    )
    env = {key: value for key, value in os.environ.items() if key != "LITELLM_LOCAL_MODEL_COST_MAP"}
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert (loaded["source"], loaded["forced"]) == ("local", True)
    assert "litellm-model-cost-map-retry" not in loaded["threads"]
