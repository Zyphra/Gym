# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import multiprocessing
import pickle
import socket
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager

import pytest
import uvicorn
from aiohttp import ClientResponseError, ClientSession
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from omegaconf import OmegaConf

from nemo_gym import server_utils
from nemo_gym.config_types import BaseRunServerInstanceConfig, BaseServerConfig
from nemo_gym.http_failure_provenance import (
    FAILURE_PROVENANCE_HEADER,
    MAX_FAILURE_PROVENANCE_HEADER,
    HTTPFailureProvenance,
    failure_provenance_headers,
)
from nemo_gym.rollout_correlation import RolloutContextMiddleware, rollout_context
from nemo_gym.server_utils import ServerClient, SimpleServer, raise_for_status


SENTINEL = "synthetic-body-query-message-marker"


class VerificationFailure(RuntimeError):
    failure_stage = "verification"


class ProbeServer(SimpleServer):
    def setup_webserver(self):
        return FastAPI()


def server_app(name):
    server = ProbeServer(
        config=BaseRunServerInstanceConfig(name=name, host="127.0.0.1", port=0, entrypoint=""),
        server_client=ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=1),
            global_config_dict=OmegaConf.create({}),
        ),
    )
    app = server.setup_webserver()
    # Match native resources-server registration: correlation is inside the
    # exception handler and has unwound by the time it constructs the response.
    app.add_middleware(RolloutContextMiddleware)
    server.setup_exception_middleware(app)
    return server, app


@asynccontextmanager
async def http_server(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    server = uvicorn.Server(uvicorn.Config(app, lifespan="off", access_log=False, log_config=None))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                await asyncio.sleep(0.001)
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 10)
        sock.close()


def return_exception(error):
    return error


@pytest.mark.asyncio
async def test_real_http_origin_survives_two_relays_and_process_pickle():
    _, origin = server_app("origin")
    relay_server, relay = server_app("relay")
    outer_server, outer = server_app("outer")
    counts = {"origin": 0, "relay": 0, "outer": 0}

    @origin.post("/fail")
    async def fail():
        counts["origin"] += 1
        raise VerificationFailure(SENTINEL)

    async with ClientSession(raise_for_status=raise_for_status) as client, http_server(origin) as origin_url:

        @relay.post("/fail")
        async def middle():
            counts["relay"] += 1
            await client.post(f"{origin_url}/ng-rollout/episode-1/fail?query={SENTINEL}")

        async with http_server(relay) as relay_url:

            @outer.post("/fail")
            async def last():
                counts["outer"] += 1
                await client.post(f"{relay_url}/fail")

            async with http_server(outer) as outer_url:
                with pytest.raises(ClientResponseError) as caught:
                    await client.post(f"{outer_url}/fail", json={"not_captured": SENTINEL})
    error = caught.value
    origin_body = JSONResponse(content=repr(VerificationFailure(SENTINEL))).body
    relay_body = JSONResponse(
        content=f"Hit an exception in {relay_server.get_session_middleware_key()} calling an inner server: {origin_body}"
    ).body
    expected = JSONResponse(
        content=f"Hit an exception in {outer_server.get_session_middleware_key()} calling an inner server: {relay_body}"
    ).body
    assert error.status == 500
    assert error.response_content == expected
    assert counts == {"origin": 1, "relay": 1, "outer": 1}
    record = error.nemo_gym_failure_provenance
    assert type(record) is HTTPFailureProvenance
    assert record.exception_class == "VerificationFailure"
    assert record.failure_stage == "verification"
    assert record.operation == "other"
    assert record.rollout_id == "episode-1"
    assert SENTINEL not in record.model_dump_json()
    assert set(record.model_dump()) == {
        "schema_version",
        "exception_class",
        "failure_stage",
        "operation",
        "rollout_id",
    }
    assert len(error.headers[FAILURE_PROVENANCE_HEADER]) <= MAX_FAILURE_PROVENANCE_HEADER
    assert record == pickle.loads(pickle.dumps(error)).nemo_gym_failure_provenance
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
        restored = executor.submit(return_exception, error).result(timeout=30)
    assert type(restored) is ClientResponseError
    assert restored.status == error.status
    assert restored.message == error.message
    assert restored.response_content == error.response_content
    assert restored.nemo_gym_failure_provenance == record


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 429, 500, 502])
async def test_real_http_preserves_direct_status_and_existing_relay_status(status):
    _, origin = server_app("origin")
    relay_server, relay = server_app("relay")
    record = HTTPFailureProvenance(exception_class="ValueError", operation="other")

    @origin.post("/fail")
    async def fail():
        return JSONResponse(
            content=SENTINEL, status_code=status, headers={FAILURE_PROVENANCE_HEADER: record.model_dump_json()}
        )

    async with ClientSession(raise_for_status=raise_for_status) as client, http_server(origin) as origin_url:
        with pytest.raises(ClientResponseError) as direct:
            await client.post(f"{origin_url}/fail")
        assert direct.value.status == status
        assert direct.value.nemo_gym_failure_provenance == record

        @relay.post("/fail")
        async def middle():
            await client.post(f"{origin_url}/fail")

        async with http_server(relay) as relay_url:
            with pytest.raises(ClientResponseError) as relayed:
                await client.post(f"{relay_url}/fail")
    # Managed deterministic client errors keep their status across relays.
    assert relayed.value.status == (status if 400 <= status < 500 else 500)
    assert (
        relayed.value.response_content
        == JSONResponse(
            content=f"Hit an exception in {relay_server.get_session_middleware_key()} calling an inner server: {direct.value.response_content}"
        ).body
    )
    assert relayed.value.nemo_gym_failure_provenance == record


@pytest.mark.asyncio
@pytest.mark.parametrize("method,expected_calls", [("POST", 1), ("GET", 3)])
async def test_real_native_request_preserves_managed_http_retry_policy(monkeypatch, method, expected_calls):
    _, app = server_app("origin")
    calls = 0

    @app.api_route("/fail", methods=["GET", "POST"])
    async def fail():
        nonlocal calls
        calls += 1
        raise VerificationFailure(SENTINEL)

    async with ClientSession(raise_for_status=raise_for_status) as client, http_server(app) as url:
        monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", client)
        with pytest.raises(ClientResponseError) as caught:
            await server_utils.request(method, f"{url}/fail")
    assert calls == expected_calls
    assert caught.value.nemo_gym_failure_provenance.exception_class == "VerificationFailure"


@pytest.mark.parametrize(
    "stage", ["setup", "transport", "verification", "generation", "cleanup", "storage", "scoring"]
)
def test_owned_known_stages_are_preserved(stage):
    error = RuntimeError(SENTINEL)
    error.failure_stage = stage
    headers = failure_provenance_headers(error, method="POST", path="/verify")
    assert HTTPFailureProvenance.from_header(headers[FAILURE_PROVENANCE_HEADER]).failure_stage == stage


@pytest.mark.parametrize("stage", [None, "verification?secret=oops", "post", 12, {"message": SENTINEL}])
def test_unknown_stage_is_not_inferred_from_verify_route(stage):
    error = RuntimeError(SENTINEL)
    error.failure_stage = stage
    record = HTTPFailureProvenance.from_header(
        failure_provenance_headers(error, method="POST", path="/verify")[FAILURE_PROVENANCE_HEADER]
    )
    assert record.failure_stage == "unknown"
    assert SENTINEL not in record.model_dump_json()


@pytest.mark.parametrize(
    "value",
    [
        None,
        "not-json",
        "x" * (MAX_FAILURE_PROVENANCE_HEADER + 1),
        '{"exception_class":"ValueError","operation":"other"}',
        '{"schema_version":"wrong","exception_class":"ValueError","operation":"other"}',
        '{"schema_version":"nemo_gym.http_failure.v1","exception_class":"http:secret","operation":"other"}',
        '{"schema_version":"nemo_gym.http_failure.v1","exception_class":"ValueError","operation":"other","message":"secret"}',
        '{"schema_version":"nemo_gym.http_failure.v1","exception_class":"ValueError","operation":"other","failure_stage":"made_up"}',
        '{"schema_version":"nemo_gym.http_failure.v1","exception_class":"ValueError","operation":"other","rollout_id":"url?token=secret"}',
    ],
)
def test_invalid_or_unowned_carriers_are_ignored(value):
    assert HTTPFailureProvenance.from_header(value) is None


def test_unowned_exception_attribute_cannot_be_relayed():
    error = RuntimeError(SENTINEL)
    error.nemo_gym_failure_provenance = {"message": SENTINEL}
    assert failure_provenance_headers(error, method="POST", path="/verify", relay=True) == {}
    error.nemo_gym_failure_provenance = HTTPFailureProvenance.model_construct(
        exception_class=SENTINEL + "?", operation="other"
    )
    assert failure_provenance_headers(error, method="POST", path="/verify", relay=True) == {}


def test_unsafe_class_name_cannot_become_payload():
    error = type("bad?query=secret", (RuntimeError,), {})(SENTINEL)
    record = HTTPFailureProvenance.from_header(
        failure_provenance_headers(error, method="POST", path="/verify")[FAILURE_PROVENANCE_HEADER]
    )
    assert record.exception_class == "unknown"
    assert "secret" not in record.model_dump_json()


@pytest.mark.asyncio
async def test_concurrent_origin_contexts_are_not_crossed():
    async def one(identity):
        with rollout_context(identity):
            await asyncio.sleep(0)
            return HTTPFailureProvenance.from_header(
                failure_provenance_headers(RuntimeError(), method="POST", path="/run")[FAILURE_PROVENANCE_HEADER]
            )

    records = await asyncio.gather(one("episode-a"), one("episode-b"))
    assert [r.rollout_id for r in records] == ["episode-a", "episode-b"]
    assert [r.operation for r in records] == ["run", "run"]


@pytest.mark.asyncio
async def test_absent_and_malformed_header_leave_transport_unknown():
    _, app = server_app("origin")

    @app.post("/fail")
    async def fail(request: Request):
        headers = {FAILURE_PROVENANCE_HEADER: "malformed"} if request.query_params.get("bad") else {}
        return JSONResponse(content=SENTINEL, status_code=500, headers=headers)

    async with ClientSession(raise_for_status=raise_for_status) as client, http_server(app) as url:
        for suffix in ("", "?bad=1"):
            with pytest.raises(ClientResponseError) as caught:
                await client.post(url + "/fail" + suffix)
            assert type(caught.value) is ClientResponseError
            assert caught.value.response_content == JSONResponse(content=SENTINEL).body
            assert not hasattr(caught.value, "nemo_gym_failure_provenance")
            assert failure_provenance_headers(caught.value, method="POST", path="/fail", relay=True) == {}
