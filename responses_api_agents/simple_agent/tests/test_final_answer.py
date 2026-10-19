# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
import inspect
import json
from contextlib import asynccontextmanager

import pytest
from aiohttp import ClientSession, web
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym import server_utils
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig


def _config(**kwargs):
    return SimpleAgentConfig(
        host="127.0.0.1",
        port=0,
        entrypoint="",
        name="simple",
        model_server={"type": "responses_api_models", "name": "model"},
        resources_server={"type": "resources_servers", "name": "resources"},
        max_steps=kwargs.pop("max_steps", 24),
        **kwargs,
    )


def _call(n, *, arguments='{"code":"print(42)"}', name="execute_python"):
    return {
        "type": "function_call",
        "id": f"fc-{n}",
        "call_id": f"call-{n}",
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def _answer(text):
    return {
        "type": "message",
        "id": "answer",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


@asynccontextmanager
async def episode(monkeypatch, outputs, **kwargs):
    requests, executed = [], []

    async def policy(request):
        body = await request.json()
        requests.append(body)
        output = [
            {
                "type": "reasoning",
                "id": f"reasoning-{len(requests)}",
                "summary": [{"type": "summary_text", "text": "deterministic calculation"}],
            }
        ]
        produced = outputs(body, len(requests))
        if inspect.isawaitable(produced):
            produced = await produced
        if isinstance(produced, web.StreamResponse):
            return produced
        output.extend(produced)
        return web.json_response(
            {
                "id": f"response-{len(requests)}",
                "created_at": 0.0,
                "model": "authored-deterministic-fixture",
                "object": "response",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "output": output,
            }
        )

    async def resource(request):
        assert request.cookies["session"] == "owned"
        executed.append((request.match_info["tool"], await request.json()))
        response = web.json_response({"stdout": "42\n", "stderr": "", "exit_code": 0, "timed_out": False})
        response.set_cookie("session", "owned")
        return response

    app = web.Application()
    app.router.add_post("/v1/responses", policy)
    app.router.add_post("/{tool}", resource)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    base = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    try:
        async with ClientSession() as session:
            monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", session)
            client = ServerClient(
                head_server_config=BaseServerConfig(host="127.0.0.1", port=1),
                global_config_dict=OmegaConf.create({"observability_enabled": True}),
            )
            client._server_base_urls.update({"model": base, "resources": base})
            agent = SimpleAgent(config=_config(**kwargs), server_client=client)
            body = NeMoGymResponseCreateParamsNonStreaming.model_validate(
                {
                    "input": [
                        {"role": "user", "content": "Compute 6*7; at most ten Python calls, then box the answer."}
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "name": "execute_python",
                            "parameters": {"type": "object"},
                            "strict": False,
                        }
                    ],
                    "parallel_tool_calls": False,
                }
            )
            yield agent, body, requests, executed
    finally:
        await runner.cleanup()


async def run(agent, body):
    return await agent._create_episode(
        body,
        model_url_path="/v1/responses",
        resources_server_cookies={"session": "owned"},
        collect_trajectory=True,
        task_id="deterministic-control",
        rollout_id="diagnostic-case-identity",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [r"\boxed{42}", "No answer.", r"\boxed{41}"])
async def test_exhausted_interactive_budget_gets_exactly_one_answer_request(monkeypatch, answer):
    async with episode(
        monkeypatch,
        lambda body, n: [_call(n)] if n <= 24 else [_answer(answer)],
        final_answer_after_max_steps=True,
        presented_tools_only=True,
    ) as (agent, body, requests, executed):
        response, trajectory, _, _ = await run(agent, body)
        assert len(executed) == 24 and len(requests) == 25
        assert all(request["tools"] == body.model_dump()["tools"] for request in requests[:24])
        assert requests[-1]["tools"] == [] and requests[-1]["tool_choice"] == "none"
        assert len([item for item in requests[-1]["input"] if item.get("type") == "function_call_output"]) == 24
        assert response.output[-1].content[0].text == answer
        assert trajectory.invocations[0].status == "completed"
        assert len(trajectory.tool_calls) == 24 and len(trajectory.turns) == 25
        assert len(body.input) == 1 and body.tools and body.tool_choice != "none"
        assert agent.config.max_steps == 24


@pytest.mark.asyncio
async def test_default_preserves_the_24th_call_and_original_incomplete_ending(monkeypatch):
    async with episode(monkeypatch, lambda body, n: [_call(n)]) as (agent, body, requests, executed):
        response, trajectory, _, _ = await run(agent, body)
        assert len(requests) == len(executed) == 24
        assert response.output[-1].type == "function_call_output"
        assert trajectory.invocations[0].status == "incomplete"
        assert all(request["tools"] == body.model_dump()["tools"] for request in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("calls", [0, 14])
async def test_existing_early_answer_and_more_than_ten_calls_are_unchanged(monkeypatch, calls):
    async with episode(
        monkeypatch,
        lambda body, n: [_call(n)] if n <= calls else [_answer(r"\boxed{42}")],
        final_answer_after_max_steps=True,
    ) as (agent, body, requests, executed):
        response, trajectory, _, _ = await run(agent, body)
        assert len(requests) == calls + 1 and len(executed) == calls
        assert all(request["tools"] == body.model_dump()["tools"] for request in requests)
        assert response.output[-1].content[0].text == r"\boxed{42}"
        assert trajectory.invocations[0].status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_calls", [1, 2])
async def test_final_request_calls_never_dispatch_or_invent_an_answer(monkeypatch, extra_calls):
    async with episode(
        monkeypatch,
        lambda body, n: [_call(n)] if n <= 3 else [_call(n + i) for i in range(extra_calls)],
        max_steps=3,
        final_answer_after_max_steps=True,
    ) as (agent, body, requests, executed):
        response, trajectory, _, _ = await run(agent, body)
        assert len(requests) == 4 and len(executed) == 3
        assert requests[-1]["tools"] == [] and requests[-1]["tool_choice"] == "none"
        assert all(t.error_type == "final_answer_tool_call" for t in trajectory.tool_calls[-extra_calls:])
        assert response.output[-1].type == "function_call_output"
        assert trajectory.invocations[0].status == "incomplete"


@pytest.mark.asyncio
async def test_context_exhaustion_keeps_the_native_incomplete_result(monkeypatch):
    from aiohttp import web

    error = {
        "error": {
            "type": "invalid_request_error",
            "message": "maximum context length is 131072 tokens; you requested 131073 tokens",
        }
    }
    async with episode(
        monkeypatch,
        lambda body, n: [_call(n)] if n <= 3 else web.json_response(error, status=400),
        max_steps=3,
        final_answer_after_max_steps=True,
    ) as (agent, body, requests, executed):
        response, trajectory, _, _ = await run(agent, body)
        assert len(requests) == 4 and len(executed) == 3
        assert response.status == "incomplete" and response.incomplete_details.reason == "max_output_tokens"
        assert trajectory.invocations[0].status == "incomplete"
        assert response.output[-1].type == "function_call_output"


@pytest.mark.asyncio
async def test_unrelated_model_error_is_not_converted_or_retried_away(monkeypatch):
    from aiohttp import ClientResponseError, web

    error = {"error": {"type": "original-invalid-control", "message": "original initiating error"}}
    async with episode(
        monkeypatch,
        lambda body, n: [_call(n)] if n <= 3 else web.json_response(error, status=422),
        max_steps=3,
        final_answer_after_max_steps=True,
    ) as (agent, body, requests, executed):
        with pytest.raises(ClientResponseError) as raised:
            await run(agent, body)
        assert raised.value.status == 422
        assert json.loads(raised.value.response_content) == error
        assert len(requests) == 4 and len(executed) == 3


@pytest.mark.asyncio
async def test_cancellation_on_final_request_propagates_and_releases(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()

    async def policy(body, n):
        if n <= 3:
            return [_call(n)]
        entered.set()
        await release.wait()
        return [_answer("not delivered after cancellation")]

    async with episode(monkeypatch, policy, max_steps=3, final_answer_after_max_steps=True) as (
        agent,
        body,
        requests,
        executed,
    ):
        task = asyncio.create_task(run(agent, body))
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled() and len(requests) == 4 and len(executed) == 3
        finally:
            release.set()


@pytest.mark.asyncio
async def test_native_output_length_ending_does_not_admit_an_extra_request(monkeypatch):
    from aiohttp import web

    def policy(body, n):
        if n == 1:
            return [_call(n)]
        return web.json_response(
            {
                "id": "cut",
                "created_at": 0.0,
                "model": "fixture",
                "object": "response",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "output": [],
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            }
        )

    async with episode(monkeypatch, policy, max_steps=2, final_answer_after_max_steps=True) as (
        agent,
        body,
        requests,
        executed,
    ):
        response, trajectory, _, _ = await run(agent, body)
        assert len(requests) == 2 and len(executed) == 1
        assert response.incomplete_details.reason == "max_output_tokens"
        assert trajectory.invocations[0].status == "incomplete"


@pytest.mark.parametrize("limit", [None, 0, -1])
def test_finalization_requires_a_finite_positive_interactive_limit(limit):
    with pytest.raises(ValidationError):
        _config(max_steps=limit, final_answer_after_max_steps=True)


@pytest.mark.parametrize("flag", ["true", 1, None])
def test_finalization_opt_in_is_strict(flag):
    with pytest.raises(ValidationError):
        _config(final_answer_after_max_steps=flag)
