# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tool-result bounds using the real pinned Hermes dispatcher, without a model."""

import copy
import json
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from unittest.mock import patch

# Import the Gym adapter before Hermes, as the native agent entrypoint does.
from responses_api_agents.hermes_agent.app import MAX_TOOL_RESULT_CHARS, _limit_tool_results


class TestToolResultLimit(unittest.TestCase):
    def _run(self, contents, *, error=None):
        history = [
            {"role": "tool", "content": "old" * 10_000, "tool_call_id": "old"},
            {"role": "assistant", "content": "check", "reasoning_content": "thinking verbatim"},
        ]
        before = copy.deepcopy(history)

        def execute(_assistant, messages, _task_id):
            messages.extend(contents)
            if error:
                raise error
            return "native return"

        agent = SimpleNamespace(_execute_tool_calls=execute)
        _limit_tool_results(agent)
        if error:
            with self.assertRaisesRegex(type(error), str(error)):
                agent._execute_tool_calls(None, history, "task")
        else:
            self.assertEqual(agent._execute_tool_calls(None, history, "task"), "native return")
        self.assertEqual(history[:2], before)
        return history[2:]

    def test_small_and_exact_boundary_results_are_unchanged(self):
        for value in ("", "small\nresult", "x" * MAX_TOOL_RESULT_CHARS):
            with self.subTest(length=len(value)):
                result = {"role": "tool", "content": value, "tool_call_id": "call"}
                self.assertEqual(self._run([copy.deepcopy(result)]), [result])

    def test_oversize_search_result_is_bounded_with_note(self):
        content = json.dumps({"files": [f"path/{i:06d}.csv" for i in range(3000)]})
        result = self._run([{"role": "tool", "content": content, "tool_call_id": "search"}])[0]
        self.assertEqual(len(result["content"]), MAX_TOOL_RESULT_CHARS)
        self.assertTrue(result["content"].startswith(content[:100]))
        self.assertIn(f"was {len(content):,} characters", result["content"])
        self.assertTrue(result["content"].endswith("Use a narrower search or read range.]"))
        self.assertEqual(result["tool_call_id"], "search")

    def test_limit_counts_unicode_characters(self):
        content = "界" * (MAX_TOOL_RESULT_CHARS + 1)
        result = self._run([{"role": "tool", "content": content}])[0]["content"]
        self.assertEqual(len(result), MAX_TOOL_RESULT_CHARS)
        self.assertTrue(result.startswith("界" * 100))

    def test_new_assistant_thinking_and_nontext_payload_are_untouched(self):
        contents = [
            {"role": "assistant", "content": "a" * 30_000, "reasoning_content": "r" * 30_000},
            {"role": "tool", "content": [{"type": "text", "text": "x" * 30_000}]},
        ]
        self.assertEqual(self._run(copy.deepcopy(contents)), contents)

    def test_partial_results_are_bounded_without_replacing_native_error(self):
        result = self._run([{"role": "tool", "content": "x" * 30_000}], error=RuntimeError("native failure"))
        self.assertEqual(len(result[0]["content"]), MAX_TOOL_RESULT_CHARS)


class TestNativeSearchResultLimit(unittest.TestCase):
    def test_next_native_request_keeps_thinking_and_clipped_observation(self):
        from run_agent import AIAgent

        requests = []
        reasoning = "exact sampled thinking\n  with trailing space "
        result = json.dumps({"files": [f"path/{i:06d}.csv" for i in range(3000)]})

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(body)
                message = {"role": "assistant", "content": "finished", "reasoning_content": "second thinking"}
                if len(requests) == 1:
                    message = {
                        "role": "assistant",
                        "content": "inspect files",
                        "reasoning_content": reasoning,
                        "tool_calls": [
                            {
                                "id": "search",
                                "type": "function",
                                "function": {"name": "search_files", "arguments": '{"pattern":"*","target":"files"}'},
                            }
                        ],
                    }
                payload = json.dumps(
                    {
                        "id": f"reply-{len(requests)}",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "fixture",
                        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
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
        try:
            agent = AIAgent(
                model="fixture",
                base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
                api_key="gym",  # pragma: allowlist secret
                enabled_toolsets=["file"],
                max_iterations=4,
                use_streaming=False,
                insert_reasoning=True,
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                persist_session=False,
                save_trajectories=False,
            )
            _limit_tool_results(agent)
            with patch("run_agent.handle_function_call", return_value=result):
                agent.run_conversation("inspect the files")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(requests), 2)
        history = requests[1]["messages"]
        assistant = next(message for message in history if message.get("role") == "assistant")
        self.assertEqual(assistant["reasoning_content"], reasoning)
        self.assertEqual(assistant["content"], "inspect files")
        tool = next(message for message in history if message.get("role") == "tool")
        self.assertEqual(tool["tool_call_id"], "search")
        self.assertEqual(len(tool["content"]), MAX_TOOL_RESULT_CHARS)
        self.assertIn("[Truncated: tool result was", tool["content"])

    def test_native_sequential_and_concurrent_searches(self):
        from run_agent import AIAgent, _should_parallelize_tool_batch
        from tools.terminal_tool import cleanup_vm

        with tempfile.TemporaryDirectory(prefix="gym-hermes-tool-results-") as directory:
            root = Path(directory)
            for i in range(300):
                (root / (f"record-{i:04d}-" + "x" * 90 + ".csv")).touch()
            agent = AIAgent(
                model="model",
                base_url="http://127.0.0.1:1/v1",
                api_key="gym",  # pragma: allowlist secret
                enabled_toolsets=["file"],
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                persist_session=False,
                save_trajectories=False,
            )
            _limit_tool_results(agent)
            for count in (1, 2):
                with self.subTest(calls=count):
                    task_id = f"gym-tool-limit-{root.name}-{count}"
                    calls = [
                        SimpleNamespace(
                            id=f"search-{i}",
                            function=SimpleNamespace(
                                name="search_files",
                                arguments=json.dumps(
                                    {"pattern": "*.csv", "target": "files", "path": str(root), "limit": 500}
                                ),
                            ),
                        )
                        for i in range(count)
                    ]
                    self.assertEqual(_should_parallelize_tool_batch(calls), count == 2)
                    assistant = SimpleNamespace(tool_calls=calls)
                    previous = {"role": "assistant", "content": "check files", "reasoning": "keep this thinking"}
                    messages = [copy.deepcopy(previous)]
                    try:
                        agent._execute_tool_calls(assistant, messages, task_id)
                    finally:
                        cleanup_vm(task_id)
                    self.assertEqual(messages[0], previous)
                    self.assertEqual(len(messages), count + 1)
                    for i, result in enumerate(messages[1:]):
                        self.assertEqual(result["tool_call_id"], f"search-{i}")
                        self.assertEqual(len(result["content"]), MAX_TOOL_RESULT_CHARS)
                        self.assertIn("[Truncated: tool result was", result["content"])
                        self.assertIn("record-", result["content"])


if __name__ == "__main__":
    unittest.main()
