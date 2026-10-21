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
"""Responses -> Chat downconversion of real Codex CLI requests.

The fixture holds the three ``POST /v1/responses`` bodies codex-cli 0.144.4 (``codex exec``,
``danger-full-access``, fallback model metadata) sent to a local mock Responses server for one
two-tool-call episode. Only the probe's host paths are replaced (workdir -> ``/testbed``, Codex
home -> ``/root/.codex``). Every body carries ``include: []``.
"""

import json
from pathlib import Path

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.responses_streaming import sanitize_streaming_responses_body, validate_streaming_responses_params


CODEX_REQUESTS = json.loads((Path(__file__).parent / "fixtures/codex-0.144.4-responses-requests.json").read_text())


def _params(body: dict) -> NeMoGymResponseCreateParamsNonStreaming:
    """The params a Gym model server's streaming ``/v1/responses`` route hands to ``responses()``."""
    cleaned, _ = sanitize_streaming_responses_body(body)
    return validate_streaming_responses_params(cleaned)


def _chat(converter: ResponsesConverter, body: dict) -> dict:
    return converter.responses_to_chat_completion_create_params(_params(body)).model_dump(exclude_unset=True)


@pytest.fixture(params=[False, True], ids=["plain", "token_ids"])
def converter(request) -> ResponsesConverter:
    return ResponsesConverter(return_token_id_information=request.param)


def test_every_captured_codex_request_sends_an_empty_include():
    assert [body["include"] for body in CODEX_REQUESTS] == [[], [], []]
    assert all(body["stream"] is True and body["store"] is False for body in CODEX_REQUESTS)
    assert all("previous_response_id" not in body for body in CODEX_REQUESTS)


def test_captured_codex_history_downconverts_append_only(converter: ResponsesConverter):
    turns = [_chat(converter, body) for body in CODEX_REQUESTS]
    messages = [turn["messages"] for turn in turns]

    assert [[m["role"] for m in turn] for turn in messages] == [
        ["system", "user", "user"],
        ["system", "user", "user", "assistant", "tool"],
        ["system", "user", "user", "assistant", "tool", "assistant", "tool"],
    ]
    # Each request repeats the previous one's chat history exactly and only appends to it.
    assert messages[1][: len(messages[0])] == messages[0]
    assert messages[2][: len(messages[1])] == messages[1]

    # The Codex instructions and the leading developer message fold into one system message.
    system = messages[0][0]["content"]
    assert system.startswith(CODEX_REQUESTS[0]["instructions"])
    assert "<permissions instructions>" in system
    assert "<cwd>/testbed</cwd>" in messages[0][1]["content"][0]["text"]

    # Codex replays its reasoning items without ids (store=false); the streaming sanitizer drops
    # them, so each assistant turn is the replayed function call alone and its output follows.
    calls = [item for item in CODEX_REQUESTS[2]["input"] if item["type"] == "function_call"]
    outputs = [item for item in CODEX_REQUESTS[2]["input"] if item["type"] == "function_call_output"]
    assert messages[2][3:] == [
        message
        for call, output in zip(calls, outputs)
        for message in (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call["call_id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": call["arguments"]},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": output["call_id"], "content": output["output"]},
        )
    ]

    for turn in turns:
        assert "include" not in turn
        assert [tool["function"]["name"] for tool in turn["tools"]] == [
            "exec_command",
            "write_stdin",
            "update_plan",
            "request_user_input",
            "view_image",
        ]
        assert turn["tool_choice"] == "auto"
        assert turn["parallel_tool_calls"] is False


def test_empty_include_converts_exactly_like_an_absent_include(converter: ResponsesConverter):
    for body in CODEX_REQUESTS:
        without_include = {key: value for key, value in body.items() if key != "include"}
        assert _chat(converter, body) == _chat(converter, without_include)


@pytest.mark.parametrize(
    "include",
    [
        ["reasoning.encrypted_content"],
        ["message.output_text.logprobs"],
        ["reasoning.encrypted_content", "message.output_text.logprobs"],
    ],
)
def test_requested_include_values_are_still_rejected(converter: ResponsesConverter, include: list[str]):
    body = {**CODEX_REQUESTS[2], "include": include}

    with pytest.raises(NotImplementedError, match=r"\['include'\]"):
        converter.responses_to_chat_completion_create_params(_params(body))
