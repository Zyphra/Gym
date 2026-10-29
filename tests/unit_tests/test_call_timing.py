# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-request call timing: scope, header contract and the agent -> resources -> judge roll-up."""

import asyncio
import json
import socket

import aiohttp
import pytest
import uvicorn
from fastapi import FastAPI
from omegaconf import DictConfig

from nemo_gym import call_timing, global_config, server_utils
from nemo_gym.call_timing import HEADER, CallTimingMiddleware, header_value, merge_header, record_call, timed_call
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import ServerClient


def _in_scope(function):
    async def run():
        token = call_timing._SCOPE.set({})
        try:
            return await function()
        finally:
            call_timing._SCOPE.reset(token)

    return asyncio.run(run())


def test_no_scope_is_a_no_op():
    record_call("judge", 5.0)
    with timed_call("judge"):
        pass
    merge_header('{"judge": {"n": 1, "sum_ms": 1, "max_ms": 1}}')
    assert header_value() is None


def test_record_accumulates_and_rejects_bad_input():
    async def body():
        record_call("judge", 10.0)
        record_call("judge", 30)
        record_call("judge", -1.0)
        record_call("judge", float("nan"))
        record_call("judge", True)
        record_call("not_a_role", 4.0)
        return json.loads(header_value())

    assert _in_scope(body) == {"judge": {"n": 2, "sum_ms": 40.0, "max_ms": 30.0}}


def test_timed_call_records_even_when_the_block_raises():
    async def body():
        with pytest.raises(RuntimeError):
            with timed_call("user_simulator"):
                await asyncio.sleep(0.02)
                raise RuntimeError("boom")
        return json.loads(header_value())

    entry = _in_scope(body)["user_simulator"]
    assert entry["n"] == 1 and entry["sum_ms"] >= 15.0 and entry["max_ms"] == entry["sum_ms"]


@pytest.mark.parametrize(
    "value",
    [
        "not json",
        "[1, 2]",
        '{"judge": 3}',
        '{"judge": {"n": -1, "sum_ms": 1, "max_ms": 1}}',
        '{"judge": {"n": 1, "sum_ms": "x", "max_ms": 1}}',
        '{"judge": {"n": 1.5, "sum_ms": 1, "max_ms": 1}}',
        '{"policy": {"n": 1, "sum_ms": 1, "max_ms": 1}}',
        '{"judge": {"n": 1, "sum_ms": NaN, "max_ms": 1}}',
    ],
)
def test_malformed_header_is_ignored(value):
    async def body():
        merge_header(value)
        return header_value()

    assert _in_scope(body) is None


def test_merge_adds_counts_and_keeps_the_longest_call():
    async def body():
        record_call("judge", 7.0)
        merge_header(
            json.dumps(
                {"judge": {"n": 2, "sum_ms": 30.0, "max_ms": 20.0}, "verify": {"n": 1, "sum_ms": 5, "max_ms": 5}}
            )
        )
        return json.loads(header_value())

    assert _in_scope(body) == {
        "judge": {"n": 3, "sum_ms": 37.0, "max_ms": 20.0},
        "verify": {"n": 1, "sum_ms": 5.0, "max_ms": 5.0},
    }


def test_middleware_adds_the_header_only_when_something_was_recorded():
    async def app(scope, receive, send):
        if scope["path"] == "/timed":
            record_call("judge", 12.0)
        await send({"type": "http.response.start", "status": 200, "headers": [(b"x-other", b"1")]})
        await send({"type": "http.response.body", "body": b"ok"})

    async def call(path):
        sent = []

        async def send(message):
            sent.append(message)

        await CallTimingMiddleware(app)({"type": "http", "path": path}, None, send)
        return dict(sent[0]["headers"])

    timed, untimed = asyncio.run(call("/timed")), asyncio.run(call("/untimed"))
    assert json.loads(timed[HEADER.encode()]) == {"judge": {"n": 1, "sum_ms": 12.0, "max_ms": 12.0}}
    assert timed[b"x-other"] == b"1"
    assert HEADER.encode() not in untimed


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Server:
    def __init__(self, app, port):
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))

    async def __aenter__(self):
        self.task = asyncio.create_task(self.server.serve())
        while not self.server.started:
            await asyncio.sleep(0.01)
        return self

    async def __aexit__(self, *exc):
        self.server.should_exit = True
        await self.task


def test_judge_and_verify_roll_up_through_two_hops(monkeypatch):
    resources_port, agent_port = _free_port(), _free_port()
    config = DictConfig({"resources": {"resources_servers": {"r": {"host": "127.0.0.1", "port": resources_port}}}})
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", config)
    client = ServerClient(head_server_config=BaseServerConfig(host="127.0.0.1", port=1), global_config_dict=config)

    resources = FastAPI()

    @resources.post("/verify")
    async def verify():
        # The judge call the resources server makes on behalf of this request.
        with timed_call("judge"):
            await asyncio.sleep(0.05)
        with timed_call("judge"):
            await asyncio.sleep(0.01)
        return {"reward": 1.0}

    agent = FastAPI()

    @agent.post("/run")
    async def run():
        response = await client.post(server_name="resources", url_path="/verify", json={})
        return await response.json()

    @agent.post("/plain")
    async def plain():
        return {"ok": True}

    resources.add_middleware(CallTimingMiddleware)
    agent.add_middleware(CallTimingMiddleware)

    async def main():
        async with _Server(resources, resources_port), _Server(agent, agent_port):
            async with aiohttp.ClientSession() as session:
                async with session.post(f"http://127.0.0.1:{agent_port}/run") as response:
                    ran = (await response.json(), dict(response.headers))
                async with session.post(f"http://127.0.0.1:{agent_port}/plain") as response:
                    plain_headers = dict(response.headers)
            # The shared aiohttp client is bound to this loop.
            await server_utils.close_global_aiohttp_client()
        return ran, plain_headers

    (body, headers), plain_headers = asyncio.run(main())
    assert body == {"reward": 1.0}
    timing = json.loads(headers[HEADER])
    assert timing["judge"]["n"] == 2 and timing["judge"]["sum_ms"] >= 55.0
    assert timing["judge"]["max_ms"] >= 45.0
    assert timing["verify"]["n"] == 1 and timing["verify"]["sum_ms"] >= timing["judge"]["sum_ms"]
    assert HEADER not in plain_headers
