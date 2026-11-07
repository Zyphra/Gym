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
from datetime import datetime, timedelta, timezone

import orjson
import pytest

from nemo_gym.base_responses_api_model import (
    CaptureStore,
    ModelCallCaptureSelection,
    clear_model_call_captures_for_rollouts,
    read_available_model_call_records,
)
from nemo_gym.model_call_capture_files import REQUEST_PREFIX_KEY, seal_capture
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body


TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}] * 20


def chat_turn(index: int, *, system: str = "You operate a bank account backend. ") -> dict:
    """An episode-endpoint call: the whole chat history plus engine token ids and log-probs."""
    messages = [{"role": "system", "content": system * 30}, {"role": "user", "content": "Close account 7."}]
    for step in range(index):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": f"Step {step}: look up the account.",
                "tool_calls": [{"id": f"call-{step}", "function": {"name": "bash", "arguments": "{}"}}],
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"call-{step}", "content": "balance: 0\n" * 50})
    return {
        "request": {
            "model": "policy",
            "messages": messages,
            "tools": TOOLS,
            "temperature": 1.0,
            "top_p": 0.95,
            "metadata": {"traj_id": "t-1"},
        },
        "response": {"token_ids": [index, index + 1], "log_probs": [-0.5, -0.25], "stop_reason": None},
        "applied_defaults": ["temperature"],
        "request_started_at_unix_s": 1.0 + index,
    }


def responses_turn(index: int, role_text: str) -> dict:
    """A Gym middleware call in the Responses dialect, as a policy or a user simulator sends it."""
    items = [{"role": "system", "content": role_text * 20}]
    for step in range(index):
        items.append({"role": "user", "content": f"{role_text} message {step}"})
        items.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]})
    return {
        "model_call_id": f"{role_text}-{index}",
        "dialect": "responses",
        "status_code": 200,
        "request": {"input": items, "tools": TOOLS, "temperature": 0.7},
        "response": {"output": [{"type": "message", "content": [{"text": f"reply {index}"}]}]},
    }


def stored_lines(store: CaptureStore, rollout_id: str) -> list[dict]:
    return [orjson.loads(line) for line in store.path_for(rollout_id).read_bytes().splitlines() if line.strip()]


def test_appended_turns_store_only_new_messages_and_read_back_whole(tmp_path):
    store = CaptureStore(tmp_path)
    calls = [chat_turn(index) for index in range(6)]
    for call in calls:
        store.record("episode", call)

    assert store.read("episode") == calls
    lines = stored_lines(store, "episode")
    assert lines[0]["request"] == calls[0]["request"]
    for index, line in enumerate(lines[1:], start=1):
        # Two new messages per turn; tools inherited; scalar sampling stays in every line.
        assert len(line["request"]["messages"]) == 2
        assert "tools" not in line["request"]
        assert line["request"]["temperature"] == 1.0 and line["request"]["top_p"] == 0.95
        assert line[REQUEST_PREFIX_KEY]["base"]["count"] == len(calls[index - 1]["request"]["messages"])
        assert line[REQUEST_PREFIX_KEY]["base"]["inherit"] == ["metadata", "tools"]
        assert line["response"] == calls[index]["response"]


def test_interleaved_sessions_each_extend_their_own_requests(tmp_path):
    store = CaptureStore(tmp_path)
    calls = []
    for index in range(4):
        calls += [responses_turn(index, "policy"), responses_turn(index, "user simulator")]
    for call in calls:
        store.record("rollout", call)

    assert store.read("rollout") == calls
    lines = stored_lines(store, "rollout")
    assert "base" not in lines[0][REQUEST_PREFIX_KEY] and "base" not in lines[1][REQUEST_PREFIX_KEY]
    assert all(len(line["request"]["input"]) == 2 for line in lines[2:])


def test_a_rewritten_history_is_stored_whole(tmp_path):
    store = CaptureStore(tmp_path)
    compacted = chat_turn(1, system="Summary of the earlier conversation. ")
    calls = [chat_turn(0), chat_turn(1), chat_turn(2), compacted, chat_turn(1), chat_turn(1)]
    for call in calls:
        store.record("episode", call)

    assert store.read("episode") == calls
    lines = stored_lines(store, "episode")
    assert lines[3]["request"] == compacted["request"]
    assert "base" not in lines[3][REQUEST_PREFIX_KEY]
    # A retry of an earlier request stores no messages at all.
    assert lines[5]["request"]["messages"] == []


def test_requests_after_a_seal_extend_sealed_records(tmp_path):
    store = CaptureStore(tmp_path)
    calls = [chat_turn(index) for index in range(5)]
    for call in calls[:3]:
        store.record("episode", call)
    assert seal_capture(store.path_for("episode"))
    for call in calls[3:]:
        store.record("episode", call)

    assert "base" in stored_lines(store, "episode")[0][REQUEST_PREFIX_KEY]
    assert store.read("episode") == calls


def test_a_cleared_capture_restarts_with_a_whole_request(tmp_path):
    record = {"_ng_task_index": 0, "_ng_rollout_index": 0}
    rollout_id = maybe_rollout_id_from_run_body(record)
    store = CaptureStore(tmp_path)
    store.record(rollout_id, chat_turn(0))
    store.record(rollout_id, chat_turn(1))
    clear_model_call_captures_for_rollouts([record], [tmp_path])
    store.record(rollout_id, chat_turn(2))

    assert store.read(rollout_id) == [chat_turn(2)]
    assert stored_lines(store, rollout_id)[0]["request"] == chat_turn(2)["request"]


def test_a_request_built_on_a_damaged_line_is_reported_not_guessed(tmp_path):
    store = CaptureStore(tmp_path)
    for index in range(3):
        store.record("episode", chat_turn(index))
    path = store.path_for("episode")
    first, *rest = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"{damaged\n" + b"".join(rest))

    calls, invalid = read_available_model_call_records(store, "episode")
    # Line 1 extends the damaged line 0, and line 2 extends line 1.
    assert invalid == 3 and calls == []
    with pytest.raises(orjson.JSONDecodeError):
        store.read("episode")
    path.write_bytes(b"".join(rest))
    with pytest.raises(ValueError, match="does not hold"):
        store.read("episode")


def test_single_call_capture_line_keeps_its_whole_request(tmp_path):
    store = CaptureStore(tmp_path)
    call = responses_turn(0, "policy")
    store.record("single", call)

    [line] = stored_lines(store, "single")
    assert {key: value for key, value in line.items() if key != REQUEST_PREFIX_KEY} == call
    assert store.read("single") == [call]


def test_selected_capture_stores_prefixes_within_its_byte_bounds(tmp_path):
    selection = ModelCallCaptureSelection(
        model_server_names=["judge"],
        max_records_per_model=8,
        max_record_bytes=64 * 1024,
        max_total_bytes=1024 * 1024,
        pending_records_per_process=2,
        capture_until=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    store = CaptureStore(tmp_path / "private", selection)
    calls = [{**chat_turn(index), "model_ref": {"name": "judge"}} for index in range(4)]
    for call in calls:
        store.record("episode", call)

    read = store.read("episode")
    assert [{k: v for k, v in r.items() if k != "capture_retention"} for r in read] == calls
    assert all(len(line["request"]["messages"]) == 2 for line in stored_lines(store, "episode")[1:])
