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
"""Usage token details on converted Responses objects.

The Responses API types ``input_tokens_details.cached_tokens`` and
``output_tokens_details.reasoning_tokens`` as integers. Chat backends often omit the
matching chat details, and the Codex CLI rejects a null inside either object, so the
converter reports an omitted detail as 0 and keeps every reported one.

``fixtures/codex-0.144.4-response-completed-usage.json`` records which ``response.completed``
usage shapes the real codex-cli 0.144.4 binary accepted (see its ``provenance``).
"""

import json
from pathlib import Path

import pytest
from openai.types.responses import ResponseUsage

from nemo_gym.anthropic_converter import AnthropicConverter
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionMessage,
    NeMoGymChoice,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.responses_streaming import synthesize_responses_sse


CODEX_USAGE = json.loads((Path(__file__).parent / "fixtures/codex-0.144.4-response-completed-usage.json").read_text())

# The usage block the wrapper's episode endpoint returns for a sampled turn, without and with
# vLLM's cached-token count, and the shapes other chat backends send.
ENDPOINT_USAGE = {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}
USAGE_CASES = [
    pytest.param(ENDPOINT_USAGE, (0, 0), id="endpoint_no_details"),
    pytest.param(
        {**ENDPOINT_USAGE, "prompt_tokens_details": {"cached_tokens": 9}}, (9, 0), id="endpoint_cached_tokens"
    ),
    pytest.param(
        {**ENDPOINT_USAGE, "prompt_tokens_details": None, "completion_tokens_details": None},
        (0, 0),
        id="vllm_null_details",
    ),
    pytest.param(
        {
            **ENDPOINT_USAGE,
            "prompt_tokens_details": {"cached_tokens": 9},
            "completion_tokens_details": {"reasoning_tokens": 4},
        },
        (9, 4),
        id="both_details",
    ),
    pytest.param(
        {**ENDPOINT_USAGE, "cached_input_tokens": 6, "reasoning_output_tokens": 2}, (6, 2), id="provider_aliases"
    ),
    pytest.param(
        {
            **ENDPOINT_USAGE,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
        (0, 0),
        id="reported_zeros",
    ),
]


def codex_accepts(usage) -> bool:
    """codex-cli 0.144.4's response.completed usage contract, as the fixture observed it: usage and
    each details object may be null or absent, but a member inside a details object must be an int."""
    if usage is None:
        return True
    for group, member in (("input_tokens_details", "cached_tokens"), ("output_tokens_details", "reasoning_tokens")):
        details = usage.get(group)
        if details is None:
            continue
        if type(details.get(member)) is not int:
            return False
    return all(type(usage.get(name)) is int for name in ("input_tokens", "output_tokens", "total_tokens"))


def _completion(usage: dict | None) -> NeMoGymChatCompletion:
    return NeMoGymChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "policy",
            "choices": [
                NeMoGymChoice(
                    index=0,
                    finish_reason="stop",
                    message=NeMoGymChatCompletionMessage(role="assistant", content="done"),
                ).model_dump()
            ],
            **({"usage": usage} if usage is not None else {}),
        }
    )


def _convert(usage: dict | None):
    return ResponsesConverter(return_token_id_information=False).chat_completion_to_response(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(model="policy", input="hello"),
        chat_completion=_completion(usage),
    )


def _completed_event(response) -> dict:
    events = list(synthesize_responses_sse(response.model_dump(mode="json")))
    payload = json.loads(events[-1].split("data: ", 1)[1])
    assert payload["type"] == "response.completed"
    return payload


@pytest.mark.parametrize("case", CODEX_USAGE["cases"], ids=lambda case: case["name"])
def test_contract_matches_every_shape_the_codex_binary_was_shown(case):
    assert codex_accepts(case["usage"]) is case["codex_accepts"]


def test_the_binary_rejected_a_null_detail_member_with_a_parse_error():
    rejected = [case for case in CODEX_USAGE["cases"] if not case["codex_accepts"]]
    assert {case["name"] for case in rejected} == {
        "both_null",
        "cached_int_reasoning_null",
        "cached_null_reasoning_int",
    }
    assert all("invalid type: null, expected i64" in case["codex_error"] for case in rejected)


@pytest.mark.parametrize("chat_usage, expected", USAGE_CASES)
def test_converted_usage_reports_omitted_details_as_zero_and_keeps_reported_ones(chat_usage, expected):
    usage = _convert(chat_usage).usage

    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (11, 3, 14)
    assert (usage.input_tokens_details.cached_tokens, usage.output_tokens_details.reasoning_tokens) == expected


@pytest.mark.parametrize("chat_usage, expected", USAGE_CASES)
def test_codex_parses_the_synthesized_response_completed(chat_usage, expected):
    usage = _completed_event(_convert(chat_usage))["response"]["usage"]

    assert codex_accepts(usage)
    assert (
        usage["input_tokens_details"]["cached_tokens"],
        usage["output_tokens_details"]["reasoning_tokens"],
    ) == expected


@pytest.mark.parametrize("chat_usage, expected", USAGE_CASES)
def test_converted_usage_validates_against_the_openai_responses_schema(chat_usage, expected):
    ResponseUsage.model_validate(_convert(chat_usage).usage.model_dump(mode="json"))


def test_the_old_null_details_fail_both_codex_and_the_openai_schema():
    null_details = {
        "input_tokens": 11,
        "input_tokens_details": {"cached_tokens": None},
        "output_tokens": 3,
        "output_tokens_details": {"reasoning_tokens": None},
        "total_tokens": 14,
    }
    assert not codex_accepts(null_details)
    with pytest.raises(ValueError):
        ResponseUsage.model_validate(null_details)


def test_a_completion_without_usage_still_has_no_usage():
    response = _convert(None)

    assert response.usage is None
    assert _completed_event(response)["response"]["usage"] is None
    assert codex_accepts(None)


@pytest.mark.parametrize("chat_usage, expected", USAGE_CASES)
def test_anthropic_messages_clients_see_the_same_usage(chat_usage, expected):
    message = AnthropicConverter().responses_to_anthropic_response(_convert(chat_usage), model="policy")

    assert message["usage"] == {"input_tokens": 11, "output_tokens": 3}
