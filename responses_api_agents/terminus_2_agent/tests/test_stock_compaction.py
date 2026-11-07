# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Terminus compaction runs Harbor's stock prompts and handoff plus the edited-files ask, replayed from a real episode."""

import json
from pathlib import Path

import pytest
from aiohttp import web
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.chat import Chat

from responses_api_agents.terminus_2_agent.app import SUMMARY_HANDOFF_ASK, StandaloneTerminus2
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
    tmp_path.mkdir()
    llm = NemoGymLLM(model_name="policy", api_base=api_base)
    agent = agent_class(logs_dir=tmp_path, model_name="policy", llm=llm, enable_summarize=True)
    chat = Chat(llm)
    chat._messages = [dict(message) for message in FIXTURE["history"]]
    try:
        handoff, refs = await agent._summarize(chat, FIXTURE["instruction"], _Screen())
    finally:
        await llm.aclose()
    return handoff, refs, chat.messages


async def _stock_and_ours(endpoint, tmp_path):
    api_base, requests = endpoint
    stock = await _compact(Terminus2, tmp_path / "stock", api_base)
    stock_requests = [body["messages"] for body in requests]
    requests.clear()
    ours = await _compact(StandaloneTerminus2, tmp_path / "ours", api_base)
    return stock, stock_requests, ours, [body["messages"] for body in requests]


async def test_compaction_sends_and_hands_on_what_stock_harbor_does_plus_the_ask(endpoint, tmp_path):
    stock, stock_requests, ours, our_requests = await _stock_and_ours(endpoint, tmp_path)

    # The same three requests (summary, questions, answers) and handoff and history, except that the
    # summary prompt, in the summary request and in the answers request that replays it, ends with the ask.
    assert len(our_requests) == 3
    history = FIXTURE["history"]
    stock_summary_prompt = stock_requests[0][-1]["content"]
    assert stock_summary_prompt.startswith("You are about to hand off your work")
    asked = {"role": "user", "content": stock_summary_prompt + SUMMARY_HANDOFF_ASK}
    assert our_requests[0] == history + [asked] == stock_requests[0][:-1] + [asked]
    assert our_requests[1] == stock_requests[1]
    stock_answers = stock_requests[2]
    assert stock_answers[len(history)] == {"role": "user", "content": stock_summary_prompt}
    assert our_requests[2] == stock_answers[: len(history)] + [asked] + stock_answers[len(history) + 1 :]
    assert ours[0] == stock[0]
    assert ours[2] == stock[2]
    assert len(ours[1]) == len(stock[1]) == 3
    # The answers trajectory records the summary prompt the answers request carried.
    answers_trajectory = json.loads((tmp_path / "ours" / ours[1][2].trajectory_path).read_text())
    assert [step["message"] for step in answers_trajectory["steps"] if step["message"] == asked["content"]] == [
        asked["content"]
    ]
    # The generated replies are handed on as generated, including a reply written as an action.
    assert f"**Summary from Previous Agent:**\n{REPLIES[0]['content']}" in our_requests[1][-1]["content"]
    assert REPLIES[2]["content"] in ours[0]


async def test_summary_request_asks_for_every_edited_file_and_the_last_check(endpoint, tmp_path):
    _, stock_requests, _, our_requests = await _stock_and_ours(endpoint, tmp_path)

    summary_request = our_requests[0][-1]["content"]
    stock_summary_request = stock_requests[0][-1]["content"]
    for ask in (
        "Every file you created or modified, each with its full path.",
        "The last test or verification command you ran",
        "and its result.",
    ):
        assert ask in summary_request and ask not in stock_summary_request
    # Harbor's own four sections stay first; the ask follows them.
    assert summary_request.index("4. **Current Status**") < summary_request.index("5. **Files Created or Modified**")
    assert summary_request.index("5. **Files Created or Modified**") < summary_request.index(
        "6. **Last Test or Verification**"
    )
