# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parsed reasoning survives the existing Chat and Responses JSON wire models."""

import json
from copy import deepcopy

import pytest

from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.responses_converter import ResponsesConverter


def through_wire(message, *, inline_parser=True, finish_reason="stop", extra_choices=None):
    envelope = {
        "id": "chatcmpl-parsed",
        "created": 0,
        "model": "wire-model",
        "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
    }
    if extra_choices:
        envelope["choices"].extend(extra_choices)
    before = deepcopy(envelope)
    chat = NeMoGymChatCompletion.model_validate_json(json.dumps(envelope))
    converter = ResponsesConverter(return_token_id_information=False, uses_reasoning_parser=inline_parser)
    response = converter.chat_completion_to_response(
        NeMoGymResponseCreateParamsNonStreaming(model="wire-model", input="question"),
        chat,
    )
    decoded = NeMoGymResponse.model_validate_json(response.model_dump_json())
    assert envelope == before
    assert chat.model_dump(exclude_none=True)["choices"][0]["message"].get("content") == message.get("content")
    return decoded


@pytest.mark.parametrize("inline_parser", [True, False])
@pytest.mark.parametrize("reasoning_key", ["reasoning_content", "reasoning"])
@pytest.mark.parametrize("reasoning_text", ["", "  Compute 6 times 7.\n"])
@pytest.mark.parametrize(
    ("finish_reason", "status", "incomplete_reason"),
    [
        ("stop", "completed", None),
        ("length", "incomplete", "max_output_tokens"),
        ("content_filter", "incomplete", "content_filter"),
        ("tool_calls", "completed", None),
    ],
)
def test_parsed_reasoning_preserves_final_and_status(
    inline_parser, reasoning_key, reasoning_text, finish_reason, status, incomplete_reason
):
    final = "  The final answer is \\boxed{42}.\n"
    response = through_wire(
        {
            "role": "assistant",
            "content": final,
            reasoning_key: reasoning_text,
        },
        inline_parser=inline_parser,
        finish_reason=finish_reason,
    )
    assert [item.type for item in response.output] == ["reasoning", "message"]
    reasoning, visible = response.output
    assert isinstance(reasoning, NeMoGymResponseReasoningItem)
    assert [(part.type, part.text) for part in reasoning.summary] == [("summary_text", reasoning_text)]
    assert isinstance(visible, NeMoGymResponseOutputMessage)
    assert visible.content[0].text == final
    assert response.status == status
    assert (response.incomplete_details.reason if response.incomplete_details else None) == incomplete_reason


@pytest.mark.parametrize("inline_parser", [True, False])
def test_parsed_reasoning_keeps_visible_think_text_and_does_not_duplicate(
    inline_parser,
):
    visible = "Quoted <think>same</think> text stays visible."
    response = through_wire(
        {
            "role": "assistant",
            "content": visible,
            "reasoning_content": "same",
            "reasoning": "same",
        },
        inline_parser=inline_parser,
    )
    assert [item.type for item in response.output] == ["reasoning", "message"]
    assert [part.text for part in response.output[0].summary] == ["same"]
    assert response.output[1].content[0].text == visible


def test_reasoning_content_has_existing_alias_precedence():
    response = through_wire(
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "legacy",
            "reasoning": "new",
        }
    )
    assert [item.type for item in response.output] == ["reasoning", "message"]
    assert response.output[0].summary[0].text == "legacy"


def test_explicit_empty_reasoning_content_precedes_alias_and_keeps_visible_text():
    visible = "Quoted <think>not parsed again</think> text."
    response = through_wire({"role": "assistant", "content": visible, "reasoning_content": "", "reasoning": "alias"})
    assert [item.type for item in response.output] == ["reasoning", "message"]
    assert response.output[0].summary[0].text == ""
    assert response.output[1].content[0].text == visible


@pytest.mark.parametrize("inline_parser", [True, False])
def test_literal_close_without_reasoning_field_preserves_legacy_visible_content(inline_parser):
    original = "work</think>answer"
    response = through_wire({"role": "assistant", "content": original}, inline_parser=inline_parser)
    assert [item.type for item in response.output] == ["message"]
    assert response.output[0].content[0].text == original


@pytest.mark.parametrize("reasoning_content", [None, [], {}, 7, False])
def test_empty_or_nontext_legacy_reasoning_can_use_textual_native_alias(
    reasoning_content,
):
    response = through_wire(
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": reasoning_content,
            "reasoning": "parsed",
        }
    )
    assert response.output[0].summary[0].text == "parsed"
    assert response.output[1].content[0].text == "answer"


@pytest.mark.parametrize("reasoning_content", [None, [], {}, 7, False])
@pytest.mark.parametrize("inline_parser", [True, False])
def test_nontext_reasoning_is_not_evidence(reasoning_content, inline_parser):
    response = through_wire(
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": reasoning_content,
        },
        inline_parser=inline_parser,
    )
    assert [item.type for item in response.output] == ["message"]
    assert response.output[0].content[0].text == "answer"


@pytest.mark.parametrize("inline_parser", [True, False])
def test_whitespace_reasoning_preserves_provider_bytes(inline_parser):
    response = through_wire(
        {"role": "assistant", "content": "answer", "reasoning_content": " \n"},
        inline_parser=inline_parser,
    )
    assert response.output[0].summary[0].text == " \n"
    assert response.output[1].content[0].text == "answer"


@pytest.mark.parametrize("inline_parser", [True, False])
def test_literal_closed_inline_reasoning_retains_legacy_toggle(inline_parser):
    original = "<think>work</think>answer"
    response = through_wire({"role": "assistant", "content": original}, inline_parser=inline_parser)
    if inline_parser:
        assert [item.type for item in response.output] == ["reasoning", "message"]
        assert response.output[0].summary[0].text == "work"
        assert response.output[1].content[0].text == "answer"
    else:
        assert [item.type for item in response.output] == ["message"]
        assert response.output[0].content[0].text == original


@pytest.mark.parametrize("reasoning_content", [None, []])
def test_empty_parsed_reasoning_keeps_inline_fallback(reasoning_content):
    response = through_wire(
        {
            "role": "assistant",
            "content": "<think>work</think>answer",
            "reasoning_content": reasoning_content,
        }
    )
    assert [item.type for item in response.output] == ["reasoning", "message"]
    assert response.output[0].summary[0].text == "work"
    assert response.output[1].content[0].text == "answer"


@pytest.mark.parametrize("reasoning_content", [None, [], "", "parsed"])
def test_refusal_remains_typed_output(reasoning_content):
    response = through_wire(
        {
            "role": "assistant",
            "content": None,
            "refusal": "Cannot answer.",
            "reasoning_content": reasoning_content,
        }
    )
    assert [item.type for item in response.output] == (
        ["reasoning", "message"] if isinstance(reasoning_content, str) else ["message"]
    )
    assert response.output[-1].content[0].type == "refusal"
    assert response.output[-1].content[0].refusal == "Cannot answer."


@pytest.mark.parametrize("inline_parser", [True, False])
def test_parsed_reasoning_and_tool_call_preserve_original_content(inline_parser):
    response = through_wire(
        {
            "role": "assistant",
            "content": "Checking now.",
            "reasoning_content": "Use the tool.",
            "tool_calls": [
                {
                    "id": "call-wire",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"x": 7}'},
                }
            ],
        },
        inline_parser=inline_parser,
        finish_reason="tool_calls",
    )
    assert [item.type for item in response.output] == [
        "reasoning",
        "message",
        "function_call",
    ]
    assert response.output[0].summary[0].text == "Use the tool."
    assert response.output[1].content[0].text == "Checking now."
    call = response.output[2]
    assert isinstance(call, NeMoGymResponseFunctionToolCall)
    assert (call.name, call.arguments, call.call_id, call.status) == (
        "lookup",
        '{"x": 7}',
        "call-wire",
        "completed",
    )


def test_multiple_choices_keep_existing_first_choice_projection():
    response = through_wire(
        {"role": "assistant", "content": "first", "reasoning_content": "first work"},
        extra_choices=[
            {
                "index": 1,
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": "second",
                    "reasoning_content": "second work",
                },
            }
        ],
    )
    assert [item.type for item in response.output] == ["reasoning", "message"]
    assert response.output[0].summary[0].text == "first work"
    assert response.output[1].content[0].text == "first"
    assert response.status == "completed"
