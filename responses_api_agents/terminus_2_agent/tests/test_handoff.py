# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest
from harbor.models.trajectories import Step

from responses_api_agents.terminus_2_agent.app import HANDOFF_SYSTEM, StandaloneTerminus2
from responses_api_agents.terminus_2_agent.llm import NemoGymLLM


class FixtureLLM(NemoGymLLM):
    """A deterministic response fixture through the real Gym response decoder."""

    async def _post_chat_completions(self, payload):
        self.payload = copy.deepcopy(payload)
        return {
            "choices": [{"message": {"content": self.reply, "reasoning_content": "recorded reasoning"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 41, "completion_tokens": 17},
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["summary", "questions", "answers"])
async def test_handoff_contract_history_and_native_accounting(tmp_path, phase):
    llm = FixtureLLM(model_name="fixture", api_base="http://unused/v1")
    llm.reply = "Goal: fix TRAIN calculator. Completed: edited /app/calc.py. Remaining: test."
    agent = StandaloneTerminus2(logs_dir=tmp_path, model_name="fixture", llm=llm,
                               model_info={"max_input_tokens": 131072, "max_output_tokens": 32768})
    history = [] if phase == "questions" else [
        {"role": "system", "content": "Always return terminal action JSON."},
        {"role": "user", "content": "Repair the TRAIN calculator without changing the API."},
        {"role": "assistant", "content": '{"commands": []}', "reasoning_content": "Need regression test."},
        {"role": "user", "content": "pwd: /app; pytest: 1 failed, 4 passed"},
    ]
    steps = [Step(step_id=i, source=m["role"].replace("assistant", "agent"), message=m["content"])
             for i, m in enumerate(history, 1)]
    original_history, original_steps = copy.deepcopy(history), [s.model_dump() for s in steps]
    response, reference = await agent._run_subagent(
        prompt="Record task state.", message_history=history, steps=steps,
        session_id="train-handoff", agent_name=f"terminus-2-summarization-{phase}",
        filename_suffix=phase, summary_text="TRAIN fixture",
    )
    assert response.content == llm.reply
    assert response.reasoning_content == "recorded reasoning"
    assert llm.payload["messages"][0] == {"role": "system", "content": HANDOFF_SYSTEM}
    if history:
        assert llm.payload["messages"][1:-1] == history[1:]
    assert history == original_history
    assert [s.model_dump() for s in steps] == original_steps
    assert agent._subagent_metrics.total_prompt_tokens == 41
    assert agent._subagent_metrics.total_completion_tokens == 17
    saved = json.loads((tmp_path / reference.trajectory_path).read_text())
    assert saved["steps"][0]["message"] == HANDOFF_SYSTEM
    assert saved["steps"][-1]["message"] == llm.reply
    assert saved["final_metrics"]["total_completion_tokens"] == 17
    # The next ordinary terminal turn still has its original action contract.
    if history:
        await llm.call("Continue working.", message_history=history)
        assert llm.payload["messages"][0] == original_history[0]
