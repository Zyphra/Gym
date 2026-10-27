# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import inspect
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.base import OutputLengthExceededError
from harbor.llms.chat import Chat

from responses_api_agents.harbor_agent.custom_agents.llms.nemo_gym_llm import NemoGymLLM as HarborLLM
from responses_api_agents.terminus_2_agent.context_budget import ServedContextTerminus2, history_key
from responses_api_agents.terminus_2_agent.llm import NemoGymLLM


async def count_tokens(agent, chat):
    count = agent._count_total_tokens(chat)
    return await count if inspect.isawaitable(count) else count


@pytest.mark.asyncio
@pytest.mark.parametrize("llm_type", [NemoGymLLM, HarborLLM])
async def test_served_usage_drives_real_harbor_compaction_and_resets_on_rewrite(tmp_path, llm_type):
    totals = iter([(110000, 12000), (100, 10), (500, 10)])

    def complete(request):
        prompt, completion = next(totals)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "", "reasoning_content": "hidden reasoning"},
                         "finish_reason": "length" if prompt == 500 else "stop"}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        })

    llm = llm_type(model_name="gpt-4", api_base="http://model", model_info={
        "max_input_tokens": 131072, "max_output_tokens": 32768, "max_context_tokens": 131072,
    })
    llm._http_client = httpx.AsyncClient(transport=httpx.MockTransport(complete))
    agent = ServedContextTerminus2(logs_dir=tmp_path, model_name="gpt-4", llm=llm,
                                  record_terminal_session=False)
    chat = Chat(llm)
    chat.messages.append({"role": "system", "content": "task system prompt"})
    try:
        await chat.chat("continue")
        assert chat.messages[-1] == {"role": "assistant", "content": ""}
        assert await count_tokens(agent, chat) == 122000
        assert llm.get_model_context_limit() == 98304
        chat.messages.append({"role": "user", "content": "new terminal output"})
        assert await count_tokens(agent, chat) > 122000
        chat.messages.pop()
        with patch.object(agent, "_summarize", AsyncMock(return_value=("handoff", []))) as summarize:
            assert await agent._check_proactive_summarization(chat, "task", None) == ("handoff", [])
            summarize.assert_awaited_once()
        chat.messages.clear()
        assert await count_tokens(agent, chat) < 122000
        await chat.chat("new handoff")
        assert await count_tokens(agent, chat) == 110
        with pytest.raises(OutputLengthExceededError):
            await chat.chat("length stop")
        # Harbor appends truncated replies itself after the exception.
        chat.messages.extend([{"role": "user", "content": "length stop"},
                              {"role": "assistant", "content": ""}])
        assert await count_tokens(agent, chat) == 510
    finally:
        await llm.aclose()


@pytest.mark.parametrize("llm_type", [NemoGymLLM, HarborLLM])
def test_legacy_input_limit_without_explicit_joint_budget(llm_type):
    llm = llm_type(model_name="gpt-4", api_base="http://model", model_info={
        "max_input_tokens": 131072, "max_output_tokens": 32768,
    })
    assert llm.get_model_context_limit() == 131072


@pytest.mark.asyncio
@pytest.mark.parametrize("llm_type", [NemoGymLLM, HarborLLM])
async def test_cached_counter_matches_real_harbor_await_contract(tmp_path, llm_type):
    llm = llm_type(model_name="gpt-4", api_base="http://127.0.0.1:1", model_info={
        "max_input_tokens": 131072, "max_output_tokens": 32768, "max_context_tokens": 131072,
    })
    agent = ServedContextTerminus2(logs_dir=tmp_path, model_name="gpt-4", llm=llm,
                                  record_terminal_session=False)
    chat = Chat(llm)
    chat.messages.append({"role": "system", "content": "bounded counter diagnostic"})
    llm._served_context_tokens[history_key(chat.messages)] = 100
    try:
        count = agent._count_total_tokens(chat)
        assert inspect.isawaitable(count) == inspect.iscoroutinefunction(Terminus2._count_total_tokens)
        assert (await count if inspect.isawaitable(count) else count) == 100
        assert await agent._check_proactive_summarization(chat, "task", None) is None
        llm._served_context_tokens.clear()
        assert await count_tokens(agent, chat) > 0
    finally:
        await llm.aclose()
