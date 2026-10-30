# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Terminus compaction hands the next agent text, replayed from a real training episode."""

import json
from pathlib import Path

import pytest
from aiohttp import web
from harbor.llms.chat import Chat

from responses_api_agents.terminus_2_agent.app import HANDOFF_TEXT_CONTRACT, StandaloneTerminus2, handoff_text
from responses_api_agents.terminus_2_agent.llm import NemoGymLLM


FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "terminus_compaction_handoff.json").read_text())
REPLIES = FIXTURE["replies"]


class _Screen:
    """The tmux pane at compaction time; harbor's questions prompt reads only this."""

    async def capture_pane(self, capture_entire: bool = False) -> str:
        return FIXTURE["screen"]


@pytest.fixture
async def served(tmp_path):
    """A chat endpoint answering harbor's three compaction calls with the replies the policy generated."""
    requests: list[dict] = []
    replies = [REPLIES[name] for name in ("summary", "questions", "answers")]

    async def complete(request):
        requests.append(await request.json())
        reply = replies[len(requests) - 1]
        message = {"role": "assistant", **reply}
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        return web.json_response({"choices": [{"message": message, "finish_reason": "stop"}], "usage": usage})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    llm = NemoGymLLM(model_name="policy", api_base=f"http://127.0.0.1:{port}/v1")
    agent = StandaloneTerminus2(logs_dir=tmp_path, model_name="policy", llm=llm, enable_summarize=True)
    try:
        yield agent, llm, requests
    finally:
        await llm.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_compaction_hands_on_the_summary_not_an_unexecuted_action(served):
    agent, llm, requests = served
    history = [dict(message) for message in FIXTURE["history"]]
    chat = Chat(llm)
    chat._messages = [dict(message) for message in history]

    handoff, refs = await agent._summarize(chat, FIXTURE["instruction"], _Screen())

    assert len(requests) == 3 and refs is not None and len(refs) == 3
    summary_request, questions_request, answers_request = (body["messages"] for body in requests)
    # The summary call extends the policy history unchanged (the served row's prefix); only its
    # last user message, harbor's summary prompt, carries the text contract.
    assert summary_request[: len(history)] == history
    assert summary_request[-1]["role"] == "user"
    assert summary_request[-1]["content"].startswith("You are about to hand off your work")
    assert summary_request[-1]["content"].endswith(HANDOFF_TEXT_CONTRACT)
    assert questions_request[-1]["content"].endswith(HANDOFF_TEXT_CONTRACT)
    assert answers_request[: len(history)] == history
    assert answers_request[-1]["content"].endswith(HANDOFF_TEXT_CONTRACT)

    # The policy replied with actions; the next agent receives its written summary instead.
    summary = REPLIES["summary"]["reasoning_content"].strip()
    assert '"commands"' in REPLIES["summary"]["content"] and '"commands"' in REPLIES["answers"]["content"]
    assert f"**Summary from Previous Agent:**\n{summary}\n" in questions_request[-1]["content"]
    assert answers_request[len(history) + 1] == {"role": "assistant", "content": summary}
    assert "data_indexer" in summary and '"keystrokes"' not in summary

    # The handoff carries the answers text; the questions were already plain text and pass through.
    assert handoff.startswith("Here are the answers the other agent provided.\n\n")
    assert REPLIES["answers"]["reasoning_content"].strip() in handoff
    assert '"keystrokes"' not in handoff and '"commands"' not in handoff
    assert chat.messages[1]["content"] == questions_request[-1]["content"].removesuffix(HANDOFF_TEXT_CONTRACT)
    assert chat.messages[2] == {"role": "assistant", "content": REPLIES["questions"]["content"]}


def test_handoff_text_keeps_text_and_recovers_actions():
    questions = REPLIES["questions"]
    assert handoff_text(questions["content"], questions["reasoning_content"]) == questions["content"]
    summary = REPLIES["summary"]
    assert handoff_text(summary["content"], summary["reasoning_content"]) == summary["reasoning_content"].strip()
    # With no draft, the action's own prose is handed on, never its commands.
    action = json.loads(summary["content"])
    assert handoff_text(summary["content"], None) == f"{action['analysis']}\n\n{action['plan']}"
    assert handoff_text("Run `ls` next {not json}", None) == "Run `ls` next {not json}"
