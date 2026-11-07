# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Terminus compaction runs Harbor's stock prompts and handoff, replayed from a real training episode."""

import json
from pathlib import Path

import pytest
from aiohttp import web
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.chat import Chat

from responses_api_agents.terminus_2_agent.app import StandaloneTerminus2
from responses_api_agents.terminus_2_agent.llm import NemoGymLLM


FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "terminus_compaction_handoff.json").read_text())
REPLIES = [FIXTURE["replies"][name] for name in ("summary", "questions", "answers")]


class _Screen:
    async def capture_pane(self, capture_entire: bool = False) -> str:
        return FIXTURE["screen"]


@pytest.fixture
async def endpoint():
    """A chat endpoint answering harbor's three compaction calls with the replies the policy generated."""
    requests: list[dict] = []

    async def complete(request):
        requests.append(await request.json())
        reply = {"role": "assistant", **REPLIES[(len(requests) - 1) % 3]}
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
        yield f"http://127.0.0.1:{port}/v1", requests
    finally:
        await runner.cleanup()


async def _compact(agent_class, tmp_path, api_base):
    llm = NemoGymLLM(model_name="policy", api_base=api_base)
    agent = agent_class(logs_dir=tmp_path, model_name="policy", llm=llm, enable_summarize=True)
    chat = Chat(llm)
    chat._messages = [dict(message) for message in FIXTURE["history"]]
    try:
        handoff, refs = await agent._summarize(chat, FIXTURE["instruction"], _Screen())
    finally:
        await llm.aclose()
    return handoff, refs, chat.messages


async def test_compaction_sends_and_hands_on_exactly_what_stock_harbor_does(endpoint, tmp_path):
    api_base, requests = endpoint

    stock = await _compact(Terminus2, tmp_path / "stock", api_base)
    stock_requests = [body["messages"] for body in requests]
    requests.clear()
    ours = await _compact(StandaloneTerminus2, tmp_path / "ours", api_base)
    our_requests = [body["messages"] for body in requests]

    # Same three requests (summary, questions, answers), byte for byte, and the same handoff and history.
    assert len(our_requests) == 3
    assert our_requests == stock_requests
    assert ours[0] == stock[0]
    assert ours[2] == stock[2]
    assert len(ours[1]) == len(stock[1]) == 3
    # Harbor's own summary prompt ends the summary request; nothing is appended to it.
    history = FIXTURE["history"]
    assert our_requests[0][: len(history)] == history
    assert our_requests[0][-1]["content"].startswith("You are about to hand off your work")
    # The generated replies are handed on as generated, including a reply written as an action.
    assert f"**Summary from Previous Agent:**\n{REPLIES[0]['content']}" in our_requests[1][-1]["content"]
    assert REPLIES[2]["content"] in ours[0]
