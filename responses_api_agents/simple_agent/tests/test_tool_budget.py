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
        output.extend(outputs(body, len(requests)))
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
async def test_tenth_tool_result_is_followed_by_a_real_answer_request(monkeypatch, answer):
    async with episode(
        monkeypatch,
        lambda body, n: [_call(n)] if n <= 10 else [_answer(answer)],
        max_tool_calls=10,
        presented_tools_only=True,
    ) as (agent, body, requests, executed):
        response, trajectory, _, _ = await run(agent, body)
        assert len(executed) == 10 and len(requests) == 11
        assert all(request["tools"] == body.model_dump()["tools"] for request in requests[:10])
        assert requests[10]["tools"] == [] and requests[10]["tool_choice"] == "none"
        assert len([item for item in requests[10]["input"] if item.get("type") == "function_call_output"]) == 10
        assert response.output[-1].content[0].text == answer
        assert trajectory.invocations[0].status == "completed"
        assert len(trajectory.tool_calls) == 10 and all(t.status == "completed" for t in trajectory.tool_calls)
        assert len(body.input) == 1 and body.tools and body.tool_choice != "none"
        assert all(request.get("max_tool_calls") is None for request in requests)
        await run(agent, body)  # New episode has its own counter, never shared state.
        assert requests[11]["tools"] == body.model_dump()["tools"]


@pytest.mark.asyncio
async def test_disabled_option_preserves_the_24th_call_and_turn_cap(monkeypatch):
    async with episode(monkeypatch, lambda body, n: [_call(n)]) as (agent, body, requests, executed):
        response, trajectory, _, _ = await run(agent, body)
        assert len(requests) == len(executed) == 24
        assert response.output[-1].type == "function_call_output"
        assert trajectory.invocations[0].status == "incomplete"
        assert all(request["tools"] == body.model_dump()["tools"] for request in requests)


@pytest.mark.asyncio
async def test_parallel_calls_cannot_overrun_the_budget(monkeypatch):
    async with episode(
        monkeypatch, lambda body, n: [_call(i) for i in range(4)] if n == 1 else [_answer("done")], max_tool_calls=3
    ) as (agent, body, requests, executed):
        _, trajectory, _, _ = await run(agent, body)
        assert len(executed) == 3
        assert requests[1]["tools"] == [] and requests[1]["tool_choice"] == "none"
        assert trajectory.tool_calls[-1].error_type == "tool_budget_exhausted"
        assert "budget exhausted" in json.loads(trajectory.tool_calls[-1].output)["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["malformed", "unpresented"])
async def test_rejected_calls_consume_budget_without_resource_dispatch(monkeypatch, kind):
    call = _call(
        1, arguments="{" if kind == "malformed" else "{}", name="hidden" if kind == "unpresented" else "execute_python"
    )
    async with episode(
        monkeypatch,
        lambda body, n: [call] if n == 1 else [_answer("done")],
        max_tool_calls=1,
        presented_tools_only=True,
    ) as (agent, body, requests, executed):
        _, trajectory, _, _ = await run(agent, body)
        assert not executed and requests[1]["tools"] == []
        assert trajectory.tool_calls[0].status == "failed"


@pytest.mark.asyncio
async def test_last_model_turn_is_reserved_even_before_call_budget(monkeypatch):
    async with episode(
        monkeypatch, lambda body, n: [_call(n)] if n < 3 else [_answer("done")], max_tool_calls=10, max_steps=3
    ) as (agent, body, requests, executed):
        _, trajectory, _, _ = await run(agent, body)
        assert len(requests) == 3 and len(executed) == 2
        assert requests[2]["tools"] == [] and requests[2]["tool_choice"] == "none"
        assert trajectory.invocations[0].status == "completed"


@pytest.mark.asyncio
async def test_calls_despite_disabled_tools_do_not_execute_or_fabricate_an_answer(monkeypatch):
    async with episode(monkeypatch, lambda body, n: [_call(n)], max_tool_calls=1, max_steps=3) as (
        agent,
        body,
        requests,
        executed,
    ):
        response, trajectory, _, _ = await run(agent, body)
        assert len(requests) == 3 and len(executed) == 1
        assert [t.error_type for t in trajectory.tool_calls] == [
            None,
            "tool_budget_exhausted",
            "tool_budget_exhausted",
        ]
        assert response.output[-1].type == "function_call_output"
        assert trajectory.invocations[0].status == "incomplete"


@pytest.mark.parametrize("invalid", [-1, True, "10", 1.5])
def test_tool_budget_is_a_strict_nonnegative_integer(invalid):
    with pytest.raises(ValidationError):
        _config(max_tool_calls=invalid)


def test_zero_budget_is_supported_and_default_is_disabled():
    assert _config(max_tool_calls=0).max_tool_calls == 0
    assert _config().max_tool_calls is None
