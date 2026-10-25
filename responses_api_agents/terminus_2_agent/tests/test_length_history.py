# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.agents.terminus_2.terminus_json_plain_parser import TerminusJSONPlainParser
from harbor.llms.base import OutputLengthExceededError
from harbor.llms.chat import Chat

from responses_api_agents.terminus_2_agent.llm import NemoGymLLM


FIXTURES = Path(__file__).parent / "fixtures"


def response(message, finish_reason="stop", **kwargs):
    return {"choices": [{"message": message, "finish_reason": finish_reason}], **kwargs}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fixture",
    ["terminus_reasoning_length", "mimo_reasoning_length", "terminus_answer_length"],
)
async def test_native_harbor_length_retry_preserves_real_served_turn(fixture):
    pair = json.loads((FIXTURES / f"{fixture}.json").read_text())
    served = pair["served"]
    llm = NemoGymLLM(model_name="test", api_base="http://policy/v1", model_info={"max_output_tokens": 32768})
    chat = Chat(llm, interleaved_thinking=True)
    chat.messages.append({"role": "system", "content": "Complete the task."})
    agent = object.__new__(Terminus2)
    agent._llm = llm
    agent._parser = TerminusJSONPlainParser()
    agent._api_request_times = []
    agent.logger = logging.getLogger(__name__)
    requests = []
    replies = [
        response(served, "length"),
        response({"content": "ok"}),
        response({"content": "done"}),
    ]

    async def post(payload):
        requests.append(deepcopy(payload))
        return replies.pop(0)

    with patch.object(llm, "_post_chat_completions", side_effect=post):
        result = await agent._query_llm(chat, "Continue.", (None, None, None))
        assert result.content == "ok"
        await chat.chat("One more turn.")

    # The real Harbor handler appends the truncated answer and asks again.
    # Its entire next prefix must equal the original request plus the served turn.
    assert requests[1]["messages"][:-1] == requests[0]["messages"] + [served]
    assert requests[2]["messages"][2] == served
    assert chat.messages[2] == served
    assert len(requests) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["prefix", "answer", "reasoning"])
async def test_length_retry_does_not_repair_changed_history(changed):
    llm = NemoGymLLM(model_name="test", api_base="http://policy/v1")
    served = {"role": "assistant", "content": "partial", "reasoning_content": "reason"}
    with patch.object(
        llm,
        "_post_chat_completions",
        AsyncMock(return_value=response(served, "length")),
    ):
        with pytest.raises(OutputLengthExceededError):
            await llm.call("first")
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "partial"},
    ]
    if changed == "prefix":
        history[0]["content"] = "rewritten"
    elif changed == "answer":
        history[1]["content"] = "edited"
    else:
        history[1]["reasoning_content"] = "different"
    original = deepcopy(history)
    with patch.object(
        llm,
        "_post_chat_completions",
        AsyncMock(return_value=response({"content": "ok"})),
    ) as post:
        await llm.call("next", message_history=history)
    assert post.call_args.args[0]["messages"][:-1] == original
    assert history == original


@pytest.mark.asyncio
async def test_length_turn_uses_its_own_token_provenance():
    llm = NemoGymLLM(model_name="test", api_base="http://policy/v1", collect_rollout_details=True)
    served = {
        "content": "partial",
        "reasoning_content": "reason",
        "prompt_token_ids": [1, 2],
        "generation_token_ids": [3, 4],
        "generation_log_probs": [-0.3, -0.4],
    }
    with patch.object(
        llm,
        "_post_chat_completions",
        AsyncMock(return_value=response(served, "length")),
    ):
        with pytest.raises(OutputLengthExceededError):
            await llm.call("first")
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "partial"},
    ]
    with patch.object(
        llm,
        "_post_chat_completions",
        AsyncMock(return_value=response({"content": "ok"})),
    ) as post:
        await llm.call("next", message_history=history)
    echo = post.call_args.args[0]["messages"][1]
    assert echo["reasoning_content"] == "reason"
    assert echo["prompt_token_ids"] == [1, 2]
    assert echo["generation_token_ids"] == [3, 4]
    assert echo["generation_log_probs"] == [-0.3, -0.4]
