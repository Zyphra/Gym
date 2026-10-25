# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from responses_api_agents.hermes_agent.policy_summary import bind_policy_summary


@pytest.fixture
def native_policy(monkeypatch, tmp_path, record_property):
    """The admitted native compressor and OpenAI client over real local HTTP."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    run_agent = pytest.importorskip("run_agent")
    import agent.context_compressor as compressor_module

    monkeypatch.setattr(compressor_module, "get_model_context_length", lambda *_a, **_kw: 8192)
    requests = []
    response_kind = ["visible"]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            message = {"role": "assistant", "content": "Verified fixture summary."}
            finish = "stop"
            if response_kind[0] == "empty":
                message["content"] = ""
            elif response_kind[0] == "length":
                finish = "length"
            elif response_kind[0] == "tool_calls":
                finish = "tool_calls"
                message["tool_calls"] = [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ]
            payload = json.dumps(
                {
                    "id": f"fixture-{len(requests)}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": body["model"],
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    native = None
    try:
        native = run_agent.AIAgent(
            base_url=f"http://127.0.0.1:{server.server_port}/ng-rollout/summary-session/v1",
            api_key="local-control",
            model="fixture-policy",
            use_streaming=False,
            insert_reasoning=True,
            max_iterations=2,
            max_tokens=64,
            enabled_toolsets=[],
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            persist_session=False,
            save_trajectories=False,
        )
        if not hasattr(native.context_compressor, "managed_summary_callback"):
            pytest.skip("requires the admitted Hermes managed-summary runtime")
        record_property("native_runtime", str(Path(run_agent.__file__).resolve()))
        # This is the existing Gym task-turn template configuration. Keep the
        # same nested mapping across calls to catch summary mutation leaking out.
        original = native._build_api_kwargs
        task_template = {"enable_thinking": True, "truncate_history_thinking": False}

        def configured(messages):
            kwargs = original(messages)
            extra_body = dict(kwargs.get("extra_body") or {})
            extra_body["chat_template_kwargs"] = task_template
            kwargs["extra_body"] = extra_body
            return kwargs

        native._build_api_kwargs = configured
        bind_policy_summary(native)
        yield native, requests, response_kind, task_template
    finally:
        if native is not None:
            native.client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_native_first_and_iterative_summaries_preserve_task_mode_and_history(native_policy):
    native, requests, _, task_template = native_policy
    history = [
        {"role": "user", "content": "The local fixture was checked."},
        {"role": "assistant", "reasoning_content": "Retained native reasoning.", "content": "Checked."},
    ]
    original_history = deepcopy(history)
    before = deepcopy(native._build_api_kwargs(history))
    native.client.chat.completions.create(**native._build_api_kwargs(history))
    assert "Verified fixture summary." in native.context_compressor._generate_summary(history)
    assert "Verified fixture summary." in native.context_compressor._generate_summary(history)
    native.client.chat.completions.create(**native._build_api_kwargs(history))
    assert history == original_history
    assert native._build_api_kwargs(history) == before
    assert task_template == {"enable_thinking": True, "truncate_history_thinking": False}
    assert len(requests) == 4
    assert [path for path, _ in requests] == ["/ng-rollout/summary-session/v1/chat/completions"] * 4
    assert [body["model"] for _, body in requests] == ["fixture-policy"] * 4
    assert [body["chat_template_kwargs"]["enable_thinking"] for _, body in requests] == [True, False, False, True]
    assert [body["max_tokens"] for _, body in requests] == [64, 4000, 4000, 64]
    for _, summary in requests[1:3]:
        assert summary["tool_choice"] == "none"
        assert summary["chat_template_kwargs"]["truncate_history_thinking"] is False
        assert "Retained native reasoning." in summary["messages"][-1]["content"]
    assert requests[1][1]["messages"][-1]["content"].startswith("Create a structured handoff")
    assert requests[2][1]["messages"][-1]["content"].startswith("You are updating a context compaction summary")


@pytest.mark.parametrize("kind", ["empty", "length", "tool_calls"])
def test_native_summary_guard_still_rejects_unusable_handoffs(native_policy, kind):
    native, requests, response_kind, task_template = native_policy
    response_kind[0] = kind
    with pytest.raises(RuntimeError, match="empty, truncated, or emitted tool calls"):
        native.context_compressor._generate_summary([{"role": "user", "content": "Summarize the task."}])
    assert len(requests) == 1  # No auxiliary-provider fallback or extra policy call.
    assert requests[0][1]["max_tokens"] == 4000
    assert task_template["enable_thinking"] is True
