# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Terminus-2's context summarizations are reported as compaction observations, like every other harness's."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from fastapi import Request
from harbor.llms.chat import Chat

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    harness_response_error,
)
from responses_api_agents.terminus_2_agent.app import (
    _INTERNAL_OBSERVATIONS_KEY,
    StandaloneTerminus2,
    Terminus2Agent,
    Terminus2AgentRunRequest,
)
from responses_api_agents.terminus_2_agent.llm import NemoGymLLM
from responses_api_agents.terminus_2_agent.tests.test_app import _make_agent


FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "terminus_compaction_handoff.json").read_text())
REPLIES = [FIXTURE["replies"][name] for name in ("summary", "questions", "answers")]


class _Screen:
    async def capture_pane(self, capture_entire: bool = False) -> str:
        return FIXTURE["screen"]


def _compactions(bundle: AgentObservationBundle) -> list[ContextCompactionObservation]:
    return [record for record in bundle.records if isinstance(record, ContextCompactionObservation)]


def _zero_usage_response() -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response-1",
        created_at=0,
        model="policy",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=NeMoGymResponseUsage(
            input_tokens=0,
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
            output_tokens=0,
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
            total_tokens=0,
        ),
    )


def _chat(llm: NemoGymLLM) -> Chat:
    chat = Chat(llm)
    chat._messages = [dict(message) for message in FIXTURE["history"]]
    return chat


@pytest.fixture
async def policy(tmp_path):
    """A chat endpoint answering harbor's summary, questions and answers calls, round after round."""
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
    # A small window, so the history leaves fewer free tokens than the proactive threshold.
    model_info = {"max_input_tokens": 3000, "max_output_tokens": 1000}
    llm = NemoGymLLM(model_name="policy", api_base=f"http://127.0.0.1:{port}/v1", model_info=model_info)
    try:
        yield llm, requests
    finally:
        await llm.aclose()
        await runner.cleanup()


def _agent(tmp_path, llm) -> StandaloneTerminus2:
    return StandaloneTerminus2(logs_dir=tmp_path, model_name="policy", llm=llm, enable_summarize=True)


async def test_a_completed_summarization_is_one_compaction_event(policy, tmp_path):
    llm, requests = policy
    agent = _agent(tmp_path, llm)
    chat = _chat(llm)
    tokens_before = agent._count_total_tokens(chat)

    handoff, refs = await agent._summarize(chat, FIXTURE["instruction"], _Screen())

    assert len(requests) == 3 and len(refs) == 3
    bundle = AgentObservationBundle.model_validate(agent.observations().model_dump(mode="json"))
    [compaction] = _compactions(bundle)
    assert compaction.outcome == "completed"
    assert compaction.trigger == "context_length_exceeded"
    assert compaction.tokens_before == tokens_before > 0
    assert compaction.summary == handoff
    assert [record.invocation_id for record in bundle.records if isinstance(record, AgentInvocation)] == ["root"]
    # What the harness cannot report stays an explicit gap, and no invocation outcome is claimed.
    assert {gap.code for gap in bundle.gaps} >= {
        "compaction_model_call_boundary_unavailable",
        "compaction_tokens_after_unavailable",
    }
    # No invocation outcome is claimed, so a zero-token response is not reclassified as a harness failure.
    assert harness_response_error(_zero_usage_response(), bundle) is None


async def test_proactive_summarization_names_its_trigger(policy, tmp_path):
    llm, requests = policy
    agent = _agent(tmp_path, llm)

    proactive = await agent._check_proactive_summarization(_chat(llm), FIXTURE["instruction"], _Screen())
    await agent._summarize(_chat(llm), FIXTURE["instruction"], _Screen())

    assert proactive is not None and len(requests) == 6
    assert [c.trigger for c in _compactions(agent.observations())] == [
        "proactive_free_tokens",
        "context_length_exceeded",
    ]


async def test_a_failed_summarization_is_recorded_and_still_raises(policy, tmp_path):
    from responses_api_agents.terminus_2_agent.llm import EpisodeTerminatedError

    llm, requests = policy
    agent = _agent(tmp_path, llm)
    llm.terminal_stop_reason = "turn_limit"

    with pytest.raises(EpisodeTerminatedError):
        await agent._summarize(_chat(llm), FIXTURE["instruction"], _Screen())

    [compaction] = _compactions(agent.observations())
    assert (compaction.outcome, compaction.summary) == ("failed", None)
    assert compaction.tokens_before is not None and not requests


async def test_an_empty_chat_is_not_compacted(policy, tmp_path):
    llm, requests = policy
    agent = _agent(tmp_path, llm)
    empty = Chat(llm)

    handoff, refs = await agent._summarize(empty, FIXTURE["instruction"], _Screen())

    assert (handoff, refs, requests) == (FIXTURE["instruction"], None, [])
    bundle = agent.observations()
    assert _compactions(bundle) == []
    assert [gap.code for gap in bundle.gaps] == ["model_call_ownership_unavailable"]


class ScriptedTerminus2(StandaloneTerminus2):
    """The real Terminus-2 observer, with its terminal loop replaced by one real context summarization."""

    async def setup(self, environment) -> None:
        return None

    async def close(self, environment, **kwargs) -> None:
        return None

    async def run(self, instruction, environment, context) -> None:
        self.context_length_exceeded, self.agent_error, self.model_transport_error = False, None, None
        await self._summarize(_chat(self._llm), instruction, _Screen())


def _rollout_request(rollout_id: str | None) -> Request:
    return Request({"type": "http", "path_params": {"rollout_id": rollout_id} if rollout_id else {}})


async def test_a_rollout_scoped_request_returns_the_observations_through_run(policy, tmp_path):
    llm, _ = policy
    agent = _make_agent()
    # Each request runs its own Terminus-2, so a run reports only its own compactions.
    agent._build_agent = MagicMock(
        side_effect=lambda *_: ScriptedTerminus2(logs_dir=tmp_path, model_name="policy", llm=llm)
    )
    body = NeMoGymResponseCreateParamsNonStreaming(model="model", input="task")

    with patch.object(Terminus2Agent, "resolve_model_base_url", return_value="http://model/v1"):
        quiet = await agent.responses(request=_rollout_request(None), body=body)
        response = await agent.responses(request=_rollout_request("rollout-1"), body=body)

    # Only a rollout-scoped request carries observations, as for every other harness.
    assert _INTERNAL_OBSERVATIONS_KEY not in quiet.model_dump(mode="json")
    response_json = response.model_dump(mode="json")
    bundle = AgentObservationBundle.model_validate(response_json[_INTERNAL_OBSERVATIONS_KEY])
    assert [c.outcome for c in _compactions(bundle)] == ["completed"]

    verify_json = {"responses_create_params": body.model_dump(mode="json"), "response": response_json, "reward": 1.0}
    agent.server_client.post = AsyncMock(return_value=MagicMock(cookies={}))
    with (
        patch("responses_api_agents.terminus_2_agent.app.raise_for_status", new=AsyncMock()),
        patch(
            "responses_api_agents.terminus_2_agent.app.get_response_json",
            new=AsyncMock(side_effect=[response_json, verify_json]),
        ),
    ):
        verified = await agent.run(MagicMock(cookies={}), Terminus2AgentRunRequest(responses_create_params=body))

    assert verified.ng_agent_observations == bundle
    # The resources server verifies the response the harness produced, not its observation channel.
    verified_response = agent.server_client.post.await_args_list[-1].kwargs["json"]["response"]
    assert _INTERNAL_OBSERVATIONS_KEY not in verified_response
