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
from nemo_gym.base_resources_server import EPISODE_TERMINAL_HEADER
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.simple_agent.app import ModelServerRef, ResourcesServerRef, SimpleAgent, SimpleAgentConfig


@pytest.mark.asyncio
async def test_a_terminal_tool_response_ends_the_episode(monkeypatch: MonkeyPatch) -> None:
    requests = []
    executed = []

    async def model(request: web.Request) -> web.Response:
        requests.append(await request.json())
        call = {
            "type": "function_call",
            "id": f"fc-{len(requests)}",
            "call_id": f"call-{len(requests)}",
            "name": "act",
            "arguments": "{}",
            "status": "completed",
        }
        return web.json_response(
            {
                "id": f"resp-{len(requests)}",
                "created_at": 1.0,
                "model": "policy",
                "object": "response",
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "output": [call],
            }
        )

    async def resource(request: web.Request) -> web.Response:
        executed.append(request.match_info["tool"])
        # The second call is the one the resources server ends the episode on.
        headers = {EPISODE_TERMINAL_HEADER: "1"} if len(executed) == 2 else {}
        return web.json_response({"output": f"call {len(executed)}"}, headers=headers)

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
                global_config_dict=OmegaConf.create({"observability_enabled": False}),
            )
            client._server_base_urls.update({"model": upstream_url, "resources": upstream_url})
            agent = SimpleAgent(
                config=SimpleAgentConfig(
                    host="127.0.0.1",
                    port=0,
                    entrypoint="",
                    name="simple",
                    model_server=ModelServerRef(type="responses_api_models", name="model"),
                    resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
                    max_steps=10,
                ),
                server_client=client,
            )
            body = NeMoGymResponseCreateParamsNonStreaming.model_validate(
                {"input": [{"role": "user", "content": "task"}], "tools": []}
            )
            response, _, _, _ = await agent._create_episode(
                body, model_url_path="/v1/responses", collect_trajectory=False
            )
        # Without the header the loop would run to max_steps=10.
        assert executed == ["act", "act"]
        assert len(requests) == 2
        outputs = [item for item in response.output if item.type == "function_call_output"]
        assert json.loads(outputs[-1].output) == {"output": "call 2"}
    finally:
        await runner.cleanup()
