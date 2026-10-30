# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native SimpleAgent recovery with real aiohttp and resource HTTP responses."""

import asyncio
import json
import socket
import time
from contextlib import contextmanager
from threading import Thread

import pytest
import uvicorn
from aiohttp import web
from omegaconf import OmegaConf
from starlette.requests import Request

from nemo_gym import server_utils
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.api_workspace.app import APIRequest, WorkspaceSeed
from resources_servers.api_workspace.engine import VENDORS
from resources_servers.api_workspace.scenarios import scenario_from_id
from resources_servers.api_workspace.tests.test_app import gold_calls, server
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig


@contextmanager
def resource_http(app):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        port = listener.getsockname()[1]
        runner = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
        thread = Thread(target=runner.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while not runner.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert runner.started
        try:
            yield port
        finally:
            runner.should_exit = True
            thread.join(timeout=10)
            assert not thread.is_alive()


@pytest.mark.asyncio
@pytest.mark.parametrize("probe", ["unconnected_vendor", "absent_filter_field"])
async def test_native_agent_recovers_tool_queries_and_completes_world_without_mask(monkeypatch, probe):
    monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", None)
    transport = server_utils.set_global_aiohttp_client(server_utils.GlobalAIOHTTPAsyncClientConfig())
    native = server()
    identity = "api-workspace-v3:train:hr_location:0000"
    scenario = scenario_from_id(identity)
    unavailable = next(v for v in VENDORS.values() if v.function == "hr" and v.name not in scenario.connected)
    connected = next(VENDORS[name] for name in scenario.connected if VENDORS[name].function == "hr")
    first_call = {"vendor": unavailable.name, "method": "GET", "path": unavailable.path("employees")}
    if probe == "absent_filter_field":
        first_call = {
            "vendor": connected.name,
            "method": "GET",
            "path": connected.path("employees"),
            "params": {"filters": {connected.field("manager"): "middle"}},
        }
    stages = [
        [("workspace_request", first_call)],
        [("workspace_accounts", {"function": "hr", "connected_only": True})],
        [
            ("workspace_schema", {"vendor": name})
            for name in scenario.connected
            if VENDORS[name].function in {"hr", "accounting"}
        ],
        [("workspace_parallel", {"requests": gold_calls(scenario)})],
    ]
    received = []

    async def model_response(request):
        body = await request.json()
        received.append(body)
        turn = len(received) - 1
        output = (
            [
                {
                    "type": "function_call",
                    "id": f"call-{turn}-{i}",
                    "call_id": f"call-{turn}-{i}",
                    "name": name,
                    "arguments": json.dumps(arguments),
                }
                for i, (name, arguments) in enumerate(stages[turn])
            ]
            if turn < len(stages)
            else [
                {
                    "type": "message",
                    "id": "done",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "Completed the requested changes.", "annotations": []}
                    ],
                }
            ]
        )
        return web.json_response(
            {
                "id": f"response-{turn}",
                "created_at": 1.0,
                "model": "deterministic-fixture",
                "object": "response",
                "output": output,
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2,
                },
            }
        )

    model_app = web.Application()
    model_app.router.add_post("/v1/responses", model_response)
    model_runner = web.AppRunner(model_app)
    await model_runner.setup()
    model_site = web.TCPSite(model_runner, "127.0.0.1", 0)
    await model_site.start()
    model_port = model_site._server.sockets[0].getsockname()[1]
    try:
        app = native.setup_webserver()
        native.setup_exception_middleware(app)
        with resource_http(app) as resource_port:
            config = OmegaConf.create(
                {
                    "resources": {
                        "resources_servers": {"api_workspace": {"host": "127.0.0.1", "port": resource_port}}
                    },
                    "policy": {"responses_api_models": {"fixture": {"host": "127.0.0.1", "port": model_port}}},
                }
            )
            client = ServerClient(
                head_server_config=BaseServerConfig(host="127.0.0.1", port=9999), global_config_dict=config
            )
            seed = await client.post("resources", "/seed_session", json={"task_id": identity})
            assert seed.status == 200
            await seed.read()
            agent = SimpleAgent(
                config=SimpleAgentConfig(
                    host="127.0.0.1",
                    port=9997,
                    name="agent",
                    entrypoint="app.py",
                    max_steps=50,
                    resources_server={"type": "resources_servers", "name": "resources"},
                    model_server={"type": "responses_api_models", "name": "policy"},
                ),
                server_client=client,
            )
            response, trace, _, resource_cookies = await agent._create_episode(
                NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": scenario.prompt}]),
                model_url_path="/v1/responses",
                resources_server_cookies=seed.cookies,
                rollout_id="api-workspace-native-401",
                collect_trajectory=True,
            )
            assert len(received) == 5
            tool_outputs = [item for item in received[1]["input"] if item.get("type") == "function_call_output"]
            tool_result = json.loads(tool_outputs[0]["output"])
            if probe == "unconnected_vendor":
                assert tool_result["status"] == 401
                assert trace.tool_calls[0].error_type == "http_401"
            else:
                assert tool_result["status"] == 200
                assert tool_result["items"] == []
                assert trace.tool_calls[0].error_type is None
            assert trace.invocations[0].status == "completed"
            assert response.output[-1].type == "message"
            assert next(iter(native._sessions.values())).world.score()["reward"] == 1.0
            verified = await client.post(
                "resources",
                "/verify",
                json={
                    "task_id": identity,
                    "responses_create_params": {"input": []},
                    "response": response.model_dump(mode="json"),
                },
                cookies=resource_cookies,
            )
            assert verified.status == 200
            assert (await verified.json())["reward"] == 1.0
            assert not native._sessions
    finally:
        await model_runner.cleanup()
        await transport.close()


@pytest.mark.asyncio
async def test_reset_and_close_isolate_outstanding_calls():
    native = server()
    identity = "api-workspace-v3:train:hr_location:0000"
    request = Request({"type": "http", "session": {SESSION_ID_KEY: "concurrent-reset"}})
    await native.seed_session(request, WorkspaceSeed(task_id=identity))
    old = native._session(request)
    await old.lock.acquire()
    pending = asyncio.create_task(native._call(old, APIRequest(**gold_calls(old.world.scenario)[0])))
    await asyncio.sleep(0)
    assert not pending.done()
    await native.close_session(request)
    await native.seed_session(request, WorkspaceSeed(task_id=identity))
    new = native._session(request)
    old.lock.release()
    assert (await pending)[0] == 200
    assert new is not old
    assert new.world.records == new.world.scenario.records
    await native.close_session(request)
    assert not native._sessions
