# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
from pathlib import Path

import pytest
import uvicorn
from aiohttp import ClientSession, ClientTimeout, web
from omegaconf import OmegaConf

from nemo_gym import server_utils
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig


@pytest.mark.skipif(not os.environ.get("SIMPLE_AGENT_LIVE_BASE_URL"), reason="explicit live model control only")
async def test_live_budget_instruction_reaches_policy_and_native_verify(monkeypatch):
    output = Path(os.environ["SIMPLE_AGENT_LIVE_OUTPUT"])
    output.mkdir(parents=True, exist_ok=False)
    state = {"policy_calls": 0, "verify_calls": 0}
    session = ClientSession(timeout=ClientTimeout(total=90))
    monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", session)

    async def model(request):
        if state["policy_calls"] >= 1:
            return web.json_response({"error": "one upstream POST guard"}, status=409)
        body = await request.json()
        body.update(model=os.environ["SIMPLE_AGENT_LIVE_MODEL"], max_output_tokens=512)
        state["policy_calls"] += 1
        (output / "request.json").write_text(json.dumps(body, sort_keys=True))
        async with session.post(
            os.environ["SIMPLE_AGENT_LIVE_BASE_URL"].rstrip("/") + "/responses",
            json=body,
            headers={"Authorization": "Bearer " + os.environ.get("JUDGE_API_KEY", "EMPTY")},
        ) as reply:
            raw = await reply.read()
            (output / "response.raw").write_bytes(raw)
            return web.Response(body=raw, status=reply.status, content_type="application/json")

    async def seed(request):
        return web.json_response({})

    async def verify(request):
        state["verify_calls"] += 1
        body = await request.json()
        answer = NeMoGymResponse.model_validate(body["response"]).output_text.strip()
        state["answer"] = answer
        return web.json_response(body | {"reward": float(answer == "50")})

    app = web.Application()
    app.router.add_post("/v1/responses", model)
    app.router.add_post("/seed_session", seed)
    app.router.add_post("/verify", verify)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    upstream = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    client = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=1), global_config_dict=OmegaConf.create({})
    )
    client._server_base_urls.update(model=upstream, resources=upstream)
    agent = SimpleAgent(
        config=SimpleAgentConfig(
            host="127.0.0.1",
            port=0,
            entrypoint="",
            name="simple",
            model_server={"type": "responses_api_models", "name": "model"},
            resources_server={"type": "resources_servers", "name": "resources"},
            max_steps=50,
            system_prompt_suffix="You have at most {max_steps} assistant turns to complete this task. Do not ask clarifying questions.",
        ),
        server_client=client,
    )
    server = uvicorn.Server(
        uvicorn.Config(agent.setup_webserver(), host="127.0.0.1", port=0, lifespan="off", log_level="error")
    )
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(500):
            if server.started and server.servers:
                break
            if task.done():
                await task
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("native agent HTTP server did not start")
        url = f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
        client._server_base_urls["simple"] = url
        async with session.post(
            url + "/run",
            json={
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": "Reply with only the integer for your maximum assistant-turn budget.",
                        }
                    ],
                    "tools": [],
                }
            },
        ) as reply:
            raw = await reply.read()
            (output / "native-run-response.json").write_bytes(raw)
            assert reply.status == 200, raw.decode()
            result = json.loads(raw)
        (output / "control.json").write_text(json.dumps(state, sort_keys=True))
        assert state == {"policy_calls": 1, "verify_calls": 1, "answer": "50"}
        assert result["reward"] == 1.0
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5)
        await session.close()
        await runner.cleanup()
