# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Each compaction call reports where its handed-on text came from, replayed from recorded episodes."""

import json
from pathlib import Path

import pytest
from aiohttp import web
from harbor.llms.chat import Chat

from nemo_gym.rollout_observability import AgentObservationBundle, ContextCompactionObservation
from responses_api_agents.terminus_2_agent.app import StandaloneTerminus2
from responses_api_agents.terminus_2_agent.llm import NemoGymLLM


FIXTURES = Path(__file__).parent / "fixtures"
CELL_A = json.loads((FIXTURES / "terminus_stock_compaction_cell_a.json").read_text())
TRAINING = json.loads((FIXTURES / "terminus_compaction_handoff.json").read_text())
CALLS = ("summary", "questions", "answers")


class _Screen:
    def __init__(self, screen: str) -> None:
        self.screen = screen

    async def capture_pane(self, capture_entire: bool = False) -> str:
        return self.screen


@pytest.fixture
async def endpoint():
    """A chat endpoint answering harbor's three compaction calls with a recorded episode's replies."""
    state: dict = {"replies": [], "requests": 0}

    async def complete(request):
        await request.json()
        reply = {"role": "assistant", **state["replies"][state["requests"] % 3]}
        state["requests"] += 1
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        return web.json_response({"choices": [{"message": reply, "finish_reason": "stop"}], "usage": usage})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/v1", state
    finally:
        await runner.cleanup()


async def _compaction(fixture, tmp_path, endpoint):
    api_base, state = endpoint
    state["replies"] = [fixture["replies"][name] for name in CALLS]
    llm = NemoGymLLM(model_name="policy", api_base=api_base)
    agent = StandaloneTerminus2(logs_dir=tmp_path, model_name="policy", llm=llm, enable_summarize=True)
    chat = Chat(llm)
    chat._messages = [dict(message) for message in fixture["history"]]
    try:
        handoff_prompt, _ = await agent._summarize(chat, fixture["instruction"], _Screen(fixture["screen"]))
    finally:
        await llm.aclose()
    bundle = AgentObservationBundle.model_validate(agent.observations().model_dump(mode="json"))
    [compaction] = [record for record in bundle.records if isinstance(record, ContextCompactionObservation)]
    return handoff_prompt, compaction


async def test_cell_a_action_replies_are_handed_on_as_generated(endpoint, tmp_path):
    # Hero check 243745: under Harbor's prompts every compaction reply was an action, and Harbor hands it on.
    handoff_prompt, compaction = await _compaction(CELL_A, tmp_path, endpoint)

    replies = CELL_A["replies"]
    assert [call.model_dump() for call in compaction.handoff_calls] == [
        {
            "call": name,
            "handoff_source": "content",
            "content_is_json_action": True,
            "content_bytes": len(replies[name]["content"].encode()),
            "reasoning_bytes": len(replies[name]["reasoning_content"].encode()),
        }
        for name in CALLS
    ]
    assert [call.content_bytes for call in compaction.handoff_calls] == [489, 1180, 722]
    assert [call.reasoning_bytes for call in compaction.handoff_calls] == [15607, 4412, 3185]
    assert compaction.summary == handoff_prompt and replies["answers"]["content"] in handoff_prompt
    assert replies["answers"]["reasoning_content"].strip() not in handoff_prompt


async def test_plain_text_and_action_replies_are_both_reported(endpoint, tmp_path):
    # Run 217294: the questions call answered in plain text; summary and answers were actions.
    _, compaction = await _compaction(TRAINING, tmp_path, endpoint)

    assert [(call.call, call.handoff_source, call.content_is_json_action) for call in compaction.handoff_calls] == [
        ("summary", "content", True),
        ("questions", "content", False),
        ("answers", "content", True),
    ]


async def test_empty_replies_report_empty(endpoint, tmp_path):
    empty = {**TRAINING, "replies": {name: {"content": "", "reasoning_content": "draft"} for name in CALLS}}
    _, compaction = await _compaction(empty, tmp_path, endpoint)

    assert [(call.handoff_source, call.content_is_json_action) for call in compaction.handoff_calls] == [
        ("empty", False)
    ] * 3
