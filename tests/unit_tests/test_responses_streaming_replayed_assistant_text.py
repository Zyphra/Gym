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
"""Replayed assistant text in streaming /v1/responses requests.

A streaming client replays its earlier turns as input. The Codex CLI replays the visible assistant
text of a tool-call turn as a ``message`` item whose ``output_text`` parts carry only ``text`` (no
message ``id``, no part ``annotations``), which the Responses API accepts. The sanitizer used to drop
that item as unsupported, so the backend saw the tool call without the text it had served and an
append-only backend rejected the continuation. ``fixtures/codex-0.144.4-replayed-assistant-text-requests.json``
holds two such second-turn bodies captured from codex-cli 0.144.4 (see its ``provenance``).
"""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.responses_streaming import sanitize_streaming_responses_body, validate_streaming_responses_params


CAPTURED = json.loads(
    (Path(__file__).parent / "fixtures/codex-0.144.4-replayed-assistant-text-requests.json").read_text()
)
CASES = [pytest.param(case, id=case["kind"]) for case in CAPTURED["cases"]]
CALL = {"type": "function_call", "name": "exec_command", "arguments": '{"cmd": "ls"}', "call_id": "call_1"}
OUTPUT = {"type": "function_call_output", "call_id": "call_1", "output": "README.md"}


def _codex_message(text: str) -> dict:
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _chat_messages(body: dict) -> list[dict]:
    cleaned, _ = sanitize_streaming_responses_body({**deepcopy(body), "stream": True})
    params = validate_streaming_responses_params(cleaned)
    chat = ResponsesConverter(return_token_id_information=False).responses_to_chat_completion_create_params(params)
    return chat.model_dump(exclude_none=True)["messages"]


@pytest.mark.parametrize("case", CASES)
def test_captured_codex_replay_keeps_the_served_assistant_text(case):
    messages = _chat_messages(case["turn2_request"])
    assistant = [message for message in messages if message["role"] == "assistant"]

    assert len(assistant) == 1
    assert assistant[0]["content"] == case["served_text"]
    assert [call["function"]["name"] for call in assistant[0]["tool_calls"]] == case["served_calls"]


@pytest.mark.parametrize("case", CASES)
def test_the_replayed_message_keeps_its_text_and_gains_only_id_status_and_annotations(case):
    original = next(
        item
        for item in case["turn2_request"]["input"]
        if item.get("type") == "message" and item.get("role") == "assistant"
    )
    cleaned, _ = sanitize_streaming_responses_body({**deepcopy(case["turn2_request"]), "stream": True})
    kept = next(item for item in cleaned["input"] if item.get("type") == "message" and item.get("role") == "assistant")

    assert kept["id"].startswith("msg_replayed_") and kept["status"] == "completed"
    assert [part["text"] for part in kept["content"]] == [part["text"] for part in original["content"]]
    assert all(part["annotations"] == [] for part in kept["content"])
    NeMoGymResponseCreateParamsNonStreaming.model_validate(cleaned)


@pytest.mark.parametrize("text", ["\n\n", "Let me look at the repository first.\n\n"])
def test_text_and_tool_call_merge_into_the_one_assistant_turn_the_backend_served(text):
    body = {"input": [{"role": "user", "content": "task"}, _codex_message(text), CALL, OUTPUT]}

    assert _chat_messages(body)[-2:] == [
        {
            "role": "assistant",
            "content": text,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "exec_command", "arguments": '{"cmd": "ls"}'},
                }
            ],
        },
        {"role": "tool", "content": "README.md", "tool_call_id": "call_1"},
    ]


def test_replayed_ids_are_positional_so_every_turn_converts_the_history_identically():
    turn2 = {"input": [{"role": "user", "content": "task"}, _codex_message("\n\n"), CALL, OUTPUT]}
    turn3 = {
        "input": [
            *turn2["input"],
            _codex_message("Next."),
            {**CALL, "call_id": "call_2"},
            {**OUTPUT, "call_id": "call_2"},
        ]
    }
    first, _ = sanitize_streaming_responses_body(deepcopy(turn2))
    second, _ = sanitize_streaming_responses_body(deepcopy(turn3))

    assert second["input"][: len(first["input"])] == first["input"]
    assert _chat_messages(turn3)[: len(_chat_messages(turn2))] == _chat_messages(turn2)


def test_a_complete_output_message_is_left_exactly_as_sent():
    complete = {
        "type": "message",
        "role": "assistant",
        "id": "msg_abc",
        "status": "completed",
        "content": [{"type": "output_text", "text": "hi", "annotations": []}],
    }
    cleaned, _ = sanitize_streaming_responses_body({"input": [deepcopy(complete)]})

    assert cleaned["input"] == [complete]


@pytest.mark.parametrize(
    "item",
    [
        pytest.param(
            {"type": "message", "role": "assistant", "content": [{"type": "mystery", "text": "x"}]}, id="unknown_part"
        ),
        pytest.param({"type": "mystery_item", "payload": 1}, id="unknown_item"),
    ],
)
def test_items_with_no_input_representation_are_still_dropped(item):
    cleaned, _ = sanitize_streaming_responses_body({"input": [{"role": "user", "content": "task"}, item]})

    assert cleaned["input"] == [{"role": "user", "content": "task"}]


def test_user_and_input_text_messages_are_untouched():
    user = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "task"}]}
    cleaned, _ = sanitize_streaming_responses_body({"input": [deepcopy(user)]})

    assert cleaned["input"] == [user]
