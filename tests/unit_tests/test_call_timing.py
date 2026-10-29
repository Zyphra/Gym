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
from fastapi.responses import JSONResponse
from omegaconf import DictConfig
from pydantic import BaseModel

from nemo_gym import call_timing, global_config, server_utils
from nemo_gym.call_timing import HEADER, CallTimingMiddleware, header_value, merge_header, record_call, timed_call
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.judge import JudgeError, call_judge
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


# --- Real call_judge through real servers: multi-hop, error and cancellation ---------------------------------------


class _Verdict(BaseModel):
    verdict: str


def _graph_config(monkeypatch, judge_port, resources_port):
    config = DictConfig(
        {
            "judge": {"responses_api_models": {"j": {"host": "127.0.0.1", "port": judge_port}}},
            "resources": {"resources_servers": {"r": {"host": "127.0.0.1", "port": resources_port}}},
        }
    )
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", config)
    return ServerClient(head_server_config=BaseServerConfig(host="127.0.0.1", port=1), global_config_dict=config)


def _judge_app(delay_s=0.05, status=200):
    app = FastAPI()

    @app.post("/v1/responses")
    async def respond():
        await asyncio.sleep(delay_s)
        if status != 200:
            return JSONResponse({"error": "judge down"}, status_code=status)
        return {"verdict": "A"}

    return app


def _with_prod_order(app):
    """The exception middleware sits inside the call-timing middleware, as in ``SimpleServer.run_webserver``."""

    @app.middleware("http")
    async def exception_middleware(request, call_next):
        try:
            return await call_next(request)
        except Exception as error:
            return JSONResponse(repr(error), status_code=500)

    return app


async def _judge(client):
    return await call_judge(client, server_name="judge", url_path="/v1/responses", json={}, response_model=_Verdict)


def test_run_header_carries_the_judge_once_with_the_resource_servers_exact_values(monkeypatch):
    judge_port, resources_port, agent_port = _free_port(), _free_port(), _free_port()
    client = _graph_config(monkeypatch, judge_port, resources_port)
    resources, agent = FastAPI(), FastAPI()
    seen = {}

    @resources.post("/verify")
    async def verify():
        return {"reward": 1.0 if (await _judge(client)).verdict == "A" else 0.0}

    @agent.post("/run")
    async def run():
        response = await client.post(server_name="resources", url_path="/verify", json={})
        seen["verify_header"] = json.loads(response.headers[HEADER])
        return await response.json()

    resources.add_middleware(CallTimingMiddleware)
    agent.add_middleware(CallTimingMiddleware)

    async def main():
        async with _Server(_judge_app(), judge_port), _Server(resources, resources_port), _Server(agent, agent_port):
            async with aiohttp.ClientSession() as session:
                async with session.post(f"http://127.0.0.1:{agent_port}/run") as response:
                    result = (await response.json(), json.loads(response.headers[HEADER]))
            await server_utils.close_global_aiohttp_client()
            return result

    body, run_header = asyncio.run(main())
    assert body == {"reward": 1.0}
    # The resource server reported one judge call; the agent added only its own /verify call on top.
    assert seen["verify_header"] == {"judge": run_header["judge"]}
    assert run_header["judge"]["n"] == 1 and run_header["judge"]["sum_ms"] >= 45.0
    assert run_header["judge"]["max_ms"] == run_header["judge"]["sum_ms"]
    assert set(run_header) == {"judge", "verify"} and run_header["verify"]["n"] == 1
    assert run_header["verify"]["sum_ms"] >= run_header["judge"]["sum_ms"]


@pytest.mark.parametrize("handled", [False, True])
def test_a_failing_judge_call_is_timed_and_changes_neither_status_nor_body(monkeypatch, handled):
    judge_port = _free_port()
    client = _graph_config(monkeypatch, judge_port, 1)

    def build(with_timing):
        app = FastAPI()

        @app.post("/verify")
        async def verify():
            try:
                return {"reward": (await _judge(client)).verdict}
            except JudgeError:
                if not handled:
                    raise
                return {"reward": 0.0, "_ng_failure_class": "judge_failed"}

        _with_prod_order(app)
        if with_timing:
            app.add_middleware(CallTimingMiddleware)
        return app

    async def fetch(app, port):
        async with _Server(app, port):
            async with aiohttp.ClientSession() as session:
                async with session.post(f"http://127.0.0.1:{port}/verify") as response:
                    return response.status, await response.text(), dict(response.headers)

    async def main():
        async with _Server(_judge_app(status=500), judge_port):
            plain = await fetch(build(False), _free_port())
            timed = await fetch(build(True), _free_port())
            await server_utils.close_global_aiohttp_client()
            return plain, timed

    (plain_status, plain_body, plain_headers), (status, body, headers) = asyncio.run(main())
    assert (status, body) == (plain_status, plain_body)
    assert status == (200 if handled else 500)
    assert ("judge_failed" in body) if handled else ("JudgeError" in body)
    assert HEADER not in plain_headers
    timing = json.loads(headers[HEADER])
    assert set(timing) == {"judge"} and timing["judge"]["n"] == 1 and timing["judge"]["sum_ms"] >= 45.0


def test_a_raising_timed_block_records_and_reraises_the_original_exception():
    original = ValueError("kept as is")

    async def body():
        with pytest.raises(ValueError) as caught:
            with timed_call("judge"):
                raise original
        assert caught.value is original
        return json.loads(header_value())

    assert _in_scope(body)["judge"]["n"] == 1


def test_cancelling_a_request_mid_judge_call_resets_the_scope_and_leaves_no_task(monkeypatch):
    judge_port = _free_port()
    client = _graph_config(monkeypatch, judge_port, 1)
    scopes = []

    async def app(scope, receive, send):
        scopes.append(call_timing._SCOPE.get())
        await _judge(client)
        await send({"type": "http.response.start", "status": 200, "headers": []})

    async def main():
        async with _Server(_judge_app(delay_s=10), judge_port):
            await asyncio.sleep(0.05)
            baseline = asyncio.all_tasks()
            sent = []

            async def send(message):
                sent.append(message)

            middleware = CallTimingMiddleware(app)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(middleware({"type": "http", "path": "/x"}, None, send), 0.15)
            leaked_scope = call_timing._SCOPE.get()
            # A following request starts with an empty scope of its own.
            after = {}

            async def following(scope, receive, send):
                after["scope"] = dict(call_timing._SCOPE.get())

            await CallTimingMiddleware(following)({"type": "http", "path": "/y"}, None, send)
            await asyncio.sleep(0.05)
            pending = asyncio.all_tasks() - baseline
            await server_utils.close_global_aiohttp_client()
            return leaked_scope, after, sent, pending

    leaked_scope, after, sent, pending = asyncio.run(main())
    assert leaked_scope is None and after["scope"] == {} and sent == []
    # The cancelled call's partial time is recorded in the cancelled request's own scope, never raised.
    assert scopes[0]["judge"]["n"] == 1 and scopes[0]["judge"]["sum_ms"] >= 100.0
    assert not [task for task in pending if not task.done()]
