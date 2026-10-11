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

import pytest
from aiohttp import ClientSession, web
from omegaconf import OmegaConf
from pytest import MonkeyPatch

from nemo_gym import server_utils
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.simple_agent.app import ModelServerRef, ResourcesServerRef, SimpleAgent, SimpleAgentConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("presented_tools_only", [False, True])
@pytest.mark.parametrize("present_tools", [False, True])
@pytest.mark.parametrize("collect_trajectory", [False, True])
async def test_presented_tools_only_checks_real_dispatch_and_returns_tool_feedback(
    monkeypatch: MonkeyPatch,
    presented_tools_only: bool,
    present_tools: bool,
    collect_trajectory: bool,
) -> None:
    requests = []
    executed = []

    async def model(request: web.Request) -> web.Response:
        payload = await request.json()
        requests.append(payload)
        output = (
            [
                {
                    "type": "function_call",
                    "id": f"fc-{name}",
                    "call_id": f"call-{name}",
                    "name": name,
                    "arguments": "{}",
                    "status": "completed",
                }
                for name in ("hidden", "advance")
            ]
            if len(requests) == 1
            else [
                {
                    "type": "message",
                    "id": "msg-done",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "done", "annotations": []}],
                }
            ]
        )
        return web.json_response(
            {
                "id": f"resp-{len(requests)}",
                "created_at": 1.0,
                "model": "policy",
                "object": "response",
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "output": output,
            }
        )

    async def resource(request: web.Request) -> web.Response:
        assert request.cookies["provider_session"] == "active"
        executed.append(request.match_info["tool"])
        response = web.json_response({"output": "executed"})
        response.set_cookie("provider_session", "active")
        return response

    upstream = web.Application()
    upstream.router.add_post("/v1/responses", model)
    upstream.router.add_post("/{tool}", resource)
    runner = web.AppRunner(upstream)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    upstream_url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    try:
        async with ClientSession() as gym_session:
            monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", gym_session)
            client = ServerClient(
                head_server_config=BaseServerConfig(host="127.0.0.1", port=1),
                global_config_dict=OmegaConf.create({"observability_enabled": collect_trajectory}),
            )
            client._server_base_urls.update({"model": upstream_url, "resources": upstream_url})
            config = SimpleAgentConfig(
                host="127.0.0.1",
                port=0,
                entrypoint="",
                name="simple",
                model_server=ModelServerRef(type="responses_api_models", name="model"),
                resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
                presented_tools_only=presented_tools_only,
            )
            agent = SimpleAgent(config=config, server_client=client)
            body = NeMoGymResponseCreateParamsNonStreaming.model_validate(
                {
                    "input": [{"role": "user", "content": "task"}],
                    "tools": [
                        {"type": "function", "name": "advance", "parameters": {"type": "object"}, "strict": False}
                    ]
                    if present_tools
                    else [],
                }
            )
            response, trajectory, _, cookies = await agent._create_episode(
                body,
                model_url_path="/v1/responses",
                resources_server_cookies={"provider_session": "active"},
                collect_trajectory=collect_trajectory,
            )
        expected = ["hidden", "advance"] if not presented_tools_only else (["advance"] if present_tools else [])
        assert executed == expected
        assert len(requests) == 2
        assert requests[0]["tools"] == requests[1]["tools"]
        feedback = [item for item in requests[1]["input"] if item.get("type") == "function_call_output"]
        assert [item["call_id"] for item in feedback] == ["call-hidden", "call-advance"]
        for item in feedback:
            name = item["call_id"].removeprefix("call-")
            payload = json.loads(item["output"])
            if name in expected:
                assert payload == {"output": "executed"}
            else:
                assert "was not presented" in payload["error"]
        assert response.output[-1].content[0].text == "done"
        assert body.input == [NeMoGymEasyInputMessage(role="user", content="task")]
        assert cookies["provider_session"].value == "active" if executed else cookies == {"provider_session": "active"}
        if collect_trajectory:
            assert [record.status for record in trajectory.tool_calls] == [
                "completed" if name in expected else "failed" for name in ("hidden", "advance")
            ]
            assert [record.error_type for record in trajectory.tool_calls] == [
                None if name in expected else "unpresented_tool" for name in ("hidden", "advance")
            ]
        else:
            assert trajectory is None
    finally:
        await runner.cleanup()
