# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""task_contract.ts through the installed Pi CLI against a scripted chat-completions endpoint."""

import asyncio
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import MagicMock

import pytest

import nemo_gym.base_responses_api_agent as base_agent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.pi_agent.app import PiAgent, PiAgentConfig, pi_episode_labels


pytestmark = pytest.mark.skipif(not os.environ.get("TASK_STOCK_PI_RUNTIME"), reason="native Pi runtime not supplied")

TASK = "Compute 6 times 7. Answer as <FINAL_ANSWER>value</FINAL_ANSWER>."
TAG = r"<FINAL_ANSWER>[\s\S]*?</FINAL_ANSWER>"
RESTATE = "Your earlier context was summarized."
SUMMARY = "## Goal\nArithmetic.\n## Progress\nRan one command."


def _bash(n: int) -> dict:
    call = {
        "index": 0,
        "id": f"call_{n}",
        "type": "function",
        "function": {"name": "bash", "arguments": '{"command": "echo hi"}'},
    }
    return {"delta": {"role": "assistant", "content": "", "tool_calls": [call]}, "finish": "tool_calls"}


def _text(text: str, finish: str = "stop") -> dict:
    return {"delta": {"role": "assistant", "content": text}, "finish": finish}


class _Endpoint:
    """Agent requests are answered from `script` in order (the last entry repeats); a request without tools
    (Pi's compaction summarizer) gets a fixed summary. `prompt_tokens` is reported on every reply."""

    def __init__(self, script: list[dict], prompt_tokens: list[int]) -> None:
        self.requests: list[dict] = []
        self.request_bytes: list[bytes] = []
        self.summaries = 0
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                pass

            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                body = json.loads(raw)
                if body.get("tools"):
                    endpoint.requests.append(body)
                    endpoint.request_bytes.append(raw)
                    n = len(endpoint.requests)
                    step = script[min(n, len(script)) - 1]
                    tokens = prompt_tokens[min(n, len(prompt_tokens)) - 1]
                else:
                    endpoint.summaries += 1
                    step, tokens = _text(SUMMARY), 50
                chunks = [
                    {"choices": [{"index": 0, "delta": step["delta"], "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": step["finish"]}]},
                    {
                        "choices": [],
                        "usage": {"prompt_tokens": tokens, "completion_tokens": 5, "total_tokens": tokens + 5},
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


def _episode(monkeypatch, tmp_path, script, prompt_tokens=(100,), **config):
    runtime = Path(os.environ["TASK_STOCK_PI_RUNTIME"])
    endpoint = _Endpoint(script, list(prompt_tokens))
    monkeypatch.setenv("PATH", f"{runtime / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {})
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    client._build_server_base_url = lambda _cfg: endpoint.url
    config = {
        "max_turns": 20,
        "context_window": 8192,
        "max_output_tokens": 1024,
        "compaction_enabled": True,
        "settings_config": {"compaction": {"reserveTokens": 1024, "keepRecentTokens": 1}},
        **config,
    }
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
            **config,
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


def _user_texts(request: dict) -> list[str]:
    texts = []
    for message in request["messages"]:
        if message["role"] != "user":
            continue
        content = message["content"]
        texts.append(content if isinstance(content, str) else "".join(b.get("text", "") for b in content))
    return texts


def _last_text(response) -> str:
    return [item for item in response.output if item.type == "message"][-1].content[0].text


def test_uncompacted_tagged_run_is_unchanged(monkeypatch, tmp_path):
    response, endpoint = _episode(
        monkeypatch, tmp_path, [_bash(1), _text("<FINAL_ANSWER>42</FINAL_ANSWER>")], answer_pattern=TAG
    )
    assert len(endpoint.requests) == 2 and endpoint.summaries == 0
    # Every request carries the instruction once, as the first user message, and nothing else is added.
    assert all(_user_texts(r) == [TASK] for r in endpoint.requests)
    assert response.metadata == {
        "pi_compactions": "0",
        "pi_restated_requests": "0",
        "pi_continuation": "none",
        "pi_command_timeouts": "0",
        "pi_cut_tool_results": "0",
    }


def test_uncompacted_request_bytes_match_the_baseline(monkeypatch, tmp_path):
    workdir = tmp_path / "same-task"
    workdir.mkdir()
    script = [_bash(1), _text("<FINAL_ANSWER>42</FINAL_ANSWER>")]
    _, candidate = _episode(monkeypatch, tmp_path / "candidate", script, working_directory=str(workdir))
    noop = tmp_path / "noop.ts"
    noop.write_text("export default function () {}\n")
    monkeypatch.setattr("responses_api_agents.pi_agent.app.TASK_CONTRACT_EXTENSION", noop)
    _, baseline = _episode(monkeypatch, tmp_path / "baseline", script, working_directory=str(workdir))
    assert len(candidate.request_bytes) == len(baseline.request_bytes) == 2
    assert candidate.request_bytes == baseline.request_bytes


def test_task_survives_compaction_and_empty_final_reply_gets_one_continuation(monkeypatch, tmp_path):
    # The first reply reports a context near the window, so Pi compacts before the second request.
    script = [_bash(1), _bash(2), _text(""), _text("<FINAL_ANSWER>42</FINAL_ANSWER>")]
    response, endpoint = _episode(
        monkeypatch, tmp_path, script, prompt_tokens=[7500, 200, 200, 200], answer_pattern=TAG
    )
    assert endpoint.summaries >= 1
    first, *after = endpoint.requests
    assert _user_texts(first) == [TASK]
    # Every request after the compaction carries the task again, right after the summary.
    assert after and all(any(t.startswith(RESTATE) and t.endswith(TASK) for t in _user_texts(r)) for r in after)
    # Each post-compaction request extends the previous one (append-only between compactions).
    for prev, nxt in zip(after, after[1:]):
        assert nxt["messages"][: len(prev["messages"])] == prev["messages"]
    # The empty reply got exactly one continuation, whose request asks for the final answer.
    assert len(endpoint.requests) == 4
    assert "Your last reply was empty." in _user_texts(endpoint.requests[-1])[-1]
    assert _last_text(response) == "<FINAL_ANSWER>42</FINAL_ANSWER>"
    assert response.metadata["pi_continuation"] == "empty_final_reply"
    assert int(response.metadata["pi_compactions"]) >= 1
    assert int(response.metadata["pi_restated_requests"]) == len(after)


@pytest.mark.parametrize("tokens", [100, 7500])
def test_training_empty_final_settles_with_no_answer_contract(monkeypatch, tmp_path, tokens):
    monkeypatch.setenv("PI_GYM_ANSWER_PATTERN", TAG)
    response, endpoint = _episode(monkeypatch, tmp_path, [_bash(1), _text("")], prompt_tokens=[tokens, 200])
    assert len(endpoint.requests) == 2
    assert endpoint.summaries == (1 if tokens == 7500 else 0)
    assert response.metadata["pi_continuation"] == "none"


def test_untagged_answer_continues_once_only(monkeypatch, tmp_path):
    response, endpoint = _episode(monkeypatch, tmp_path, [_text("42")], answer_pattern=TAG)
    # One continuation, never a second, even though the reply stays untagged.
    assert len(endpoint.requests) == 2
    assert response.metadata["pi_continuation"] == "answer_missing"


def test_answer_file_satisfies_the_contract(monkeypatch, tmp_path):
    answer = tmp_path / "answer.txt"
    answer.write_text("<FINAL_ANSWER>42</FINAL_ANSWER>")
    response, endpoint = _episode(
        monkeypatch, tmp_path, [_text("done")], answer_pattern=TAG, answer_files=[str(answer)]
    )
    assert len(endpoint.requests) == 1 and response.metadata["pi_continuation"] == "none"


def test_no_continuation_past_the_turn_limit_or_after_a_cut_tool_turn(monkeypatch, tmp_path):
    response, endpoint = _episode(monkeypatch, tmp_path, [_bash(1)], max_turns=2)
    # Two turns, then turn_limit.ts's cap request and its one re-ask (both replies are calls, blocked);
    # the run gets no continuation.
    assert len(endpoint.requests) == 4
    assert response.metadata["pi_continuation"] == "none"
    response, endpoint = _episode(monkeypatch, tmp_path / "b", [_text("")], max_turns=1)
    assert len(endpoint.requests) == 1 and response.metadata["pi_continuation"] == "none"


def test_a_cut_final_reply_is_retried_once_in_all(monkeypatch, tmp_path):
    """Pi's own length recovery retries a final reply cut at the output limit; the contract adds no second retry."""
    response, endpoint = _episode(monkeypatch, tmp_path, [_text("The answer is", "length"), _text("42")])
    assert len(endpoint.requests) == 2, [r["messages"][-1] for r in endpoint.requests]
    assert _last_text(response) == "42"


def test_labels_from_events():
    events = [
        {"type": "compaction_end"},
        {"type": "entry_appended", "entry": {"type": "custom", "customType": "pi_gym_command_timeout", "data": {}}},
        {
            "type": "entry_appended",
            "entry": {
                "type": "custom",
                "customType": "pi_gym_task_contract",
                "data": {"restated_requests": 3, "continuation": "answer_missing"},
            },
        },
    ]
    assert pi_episode_labels("\n".join(json.dumps(e) for e in events)) == {
        "pi_compactions": "1",
        "pi_restated_requests": "3",
        "pi_continuation": "answer_missing",
        "pi_command_timeouts": "1",
        "pi_cut_tool_results": "0",
    }


def test_loads_under_request_tool_flags(monkeypatch, tmp_path):
    """MiMo's Pi binding turns discovery off; PiAgent's own --extension still loads (the run has no tools,
    so the endpoint sees both of its requests as tool-less)."""
    flags = ["--no-builtin-tools", "--no-extensions", "--no-skills", "--no-prompt-templates", "--no-context-files"]
    response, endpoint = _episode(monkeypatch, tmp_path, [_text("42")], answer_pattern=TAG, extra_args=flags)
    assert endpoint.summaries == 2 and response.metadata["pi_continuation"] == "answer_missing"
