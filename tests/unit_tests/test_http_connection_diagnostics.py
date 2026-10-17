# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import pickle
import socket
from contextlib import asynccontextmanager
from functools import partial

import pytest
import uvicorn
from aiohttp import ClientSession, ServerDisconnectedError
from fastapi import FastAPI, Request
from pydantic import ValidationError
from uvicorn.protocols.http.h11_impl import H11Protocol
from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol

from nemo_gym import server_utils
from nemo_gym.http_connection_diagnostics import (
    ConnectionDiagnosticMixin,
    client_trace_config,
    diagnostic_protocol,
    failure_context,
    request_metadata,
)
from nemo_gym.rollout_correlation import current_rollout_id, rollout_context
from nemo_gym.server_utils import GlobalAIOHTTPAsyncClientConfig
from nemo_gym.telemetry.connection_pool import build_connection_pool_connector


EPISODE = "381e945bdc1a401abe6ba6c62ab4dd26"


@asynccontextmanager
async def raw_peer(handler):
    tasks = set()

    async def run(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        try:
            await handler(reader, writer)
        finally:
            writer.close()
            await writer.wait_closed()
            tasks.remove(task)

    server = await asyncio.start_server(run, "127.0.0.1", 0)
    try:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    finally:
        server.close()
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(*list(tasks), return_exceptions=True)
        await server.wait_closed()


@asynccontextmanager
async def http_server(app, *, debug=True, idle=30, protocol=HttpToolsProtocol):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    config = uvicorn.Config(
        app,
        http=partial(diagnostic_protocol, base=protocol) if debug else protocol,
        timeout_keep_alive=idle,
        lifespan="off",
        access_log=False,
        log_config=None,
    )
    server = uvicorn.Server(config)
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


def events(capsys):
    return [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith('{"event":"http_server_connection_close"')
    ]


@pytest.mark.parametrize("value", ["true", "false", 1, 0, None])
def test_transport_policy_is_strict_and_separate(value):
    with pytest.raises(ValidationError):
        GlobalAIOHTTPAsyncClientConfig(global_http_transport_diagnostics=value)
    config = GlobalAIOHTTPAsyncClientConfig()
    assert config.global_http_transport_diagnostics is False
    assert config.global_aiohttp_client_request_debug is False


@pytest.mark.parametrize(
    "method,path,operation",
    [
        ("POST", f"/policy/episodes/{EPISODE}", "enroll"),
        ("DELETE", f"/policy/episodes/{EPISODE}", "release"),
        ("POST", f"/policy/episodes/{EPISODE}/finalize", "finalize"),
        ("GET", f"/policy/episodes/{EPISODE}/progress", "progress"),
        ("GET", f"/policy/episodes/{EPISODE}/snapshot", "snapshot"),
        ("POST", f"/ng-rollout/{EPISODE}/v1/responses", "model"),
        ("POST", "/run", "run"),
        ("POST", "/private-secret", "other"),
    ],
)
def test_route_metadata_contains_no_arbitrary_url_components(method, path, operation):
    result = request_metadata(method, "http://user:secret@127.0.0.1" + path + "?key=query-secret")
    assert result["operation"] == operation
    assert result["method"] == method
    encoded = json.dumps(result)
    for secret in ("user", "secret", "query", "private"):
        assert secret not in encoded
    assert result["rollout_id"] == (EPISODE if EPISODE in path else None)


@pytest.mark.asyncio
async def test_default_client_connector_and_trace_remain_unchanged(monkeypatch):
    monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", None)
    client = server_utils.set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
    try:
        assert not isinstance(client.connector, ConnectionDiagnosticMixin)
        assert client.trace_configs == []
        assert server_utils.is_global_aiohttp_client_request_debug_enabled() is False
    finally:
        await client.close()
        monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", None)


@pytest.mark.asyncio
async def test_enabled_client_does_not_enable_unsafe_legacy_body_debug(monkeypatch):
    monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", None)
    client = server_utils.set_global_aiohttp_client(
        GlobalAIOHTTPAsyncClientConfig(global_http_transport_diagnostics=True)
    )
    try:
        assert isinstance(client.connector, ConnectionDiagnosticMixin)
        assert len(client.trace_configs) == 1
        assert server_utils.is_global_aiohttp_client_request_debug_enabled() is False
    finally:
        await client.close()
        monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", None)


@pytest.mark.asyncio
async def test_post_disconnect_is_one_attempt_and_keeps_existing_finality(monkeypatch, capsys):
    requests = []

    async def drop(reader, writer):
        requests.append(await reader.readuntil(b"\r\n\r\n"))
        writer.transport.close()

    async with raw_peer(drop) as url:
        async with ClientSession(
            connector=build_connection_pool_connector(transport_debug=True),
            trace_configs=[client_trace_config()],
        ) as client:
            monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", client)
            monkeypatch.setattr(server_utils, "_NUM_SERVER_DISCONNECTED_ERROR", 99)
            with rollout_context(EPISODE):
                with pytest.raises(ServerDisconnectedError) as caught:
                    await server_utils.request("POST", url + f"/policy/episodes/{EPISODE}", _max_connection_retries=1)
            context = caught.value._gym_transport_context
            assert context["rollout_id"] == EPISODE
            assert context["connection"] == "new"
            assert context["phase"] in ("headers_sent", "body_sent")
            assert context["local"][0] == context["peer"][0] == "127.0.0.1"
            assert context["outcome"] == "exception"
            assert current_rollout_id() is None
            assert len(requests) == 1
            output = capsys.readouterr().out
            assert output.count("[request_retry") == 1
            assert '"final_attempt":true' in output
            assert '"operation":"enroll"' in output
        monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", None)


@pytest.mark.asyncio
async def test_reused_socket_keeps_safe_per_request_identity_and_idle_age():
    async def peer(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        await writer.drain()
        await reader.readuntil(b"\r\n\r\n")
        writer.transport.close()

    async with raw_peer(peer) as url:
        async with ClientSession(
            connector=build_connection_pool_connector(transport_debug=True),
            trace_configs=[client_trace_config()],
        ) as client:
            with rollout_context("first"):
                async with client.post(url) as response:
                    assert await response.text() == "ok"
            await asyncio.sleep(0.01)
            with rollout_context("second"):
                with pytest.raises(ServerDisconnectedError) as caught:
                    await client.post(url + "/private-secret?token=query-secret", data=b"body-secret")
            result = caught.value._gym_transport_context
            assert result["connection"] == "reused"
            assert result["rollout_id"] == "second"
            assert result["pool_idle_s"] >= 0.005
            assert result["operation"] == "other"
            assert "secret" not in json.dumps(result)
            assert current_rollout_id() is None


@pytest.mark.asyncio
async def test_concurrent_failures_and_cancellation_do_not_cross_contexts():
    received = asyncio.Event()

    async def wait(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        received.set()
        await asyncio.sleep(10)

    async with raw_peer(wait) as url:
        async with ClientSession(
            connector=build_connection_pool_connector(transport_debug=True),
            trace_configs=[client_trace_config()],
        ) as client:

            async def cancel_one(identifier):
                with rollout_context(identifier):
                    task = asyncio.create_task(client.post(url))
                    await received.wait()
                    await asyncio.sleep(0.01)
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError) as caught:
                        await task
                    context = caught.value._gym_transport_context
                    assert context["rollout_id"] == identifier
                    assert context["outcome"] == "client_cancelled"
                    assert current_rollout_id() == identifier
                    return context

            results = await asyncio.gather(cancel_one("one"), cancel_one("two"))
            assert {result["rollout_id"] for result in results} == {"one", "two"}
            assert current_rollout_id() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("protocol", [HttpToolsProtocol, H11Protocol])
async def test_real_server_idle_close_is_labelled_close_not_failure(debug, protocol, capsys):
    app = FastAPI()

    @app.post("/policy/episodes/{episode}")
    async def enroll(episode: str, request: Request):
        await request.body()
        return {"ok": True}

    async with http_server(app, debug=debug, idle=0.03, protocol=protocol) as url:
        async with ClientSession() as client:
            async with client.post(
                url + f"/policy/episodes/{EPISODE}?token=query-secret",
                data=b"body-secret",
                headers={"Authorization": "Bearer header-secret"},
            ) as response:
                assert response.status == 200
                await response.read()
            await asyncio.sleep(0.1)
        recorded = events(capsys)
        if not debug:
            assert recorded == []
        else:
            assert len(recorded) == 1
            event = recorded[0]
            assert event["event"] == "http_server_connection_close"
            assert event["local_trigger"] == "idle_expiry"
            assert event["last_parsed_request"]["rollout_id"] == EPISODE
            assert event["local"][0] == event["peer"][0] == "127.0.0.1"
            assert "secret" not in json.dumps(event)


@pytest.mark.asyncio
async def test_remote_close_is_unknown_and_factory_is_pickleable(capsys):
    factory = partial(diagnostic_protocol, base=HttpToolsProtocol)
    assert pickle.loads(pickle.dumps(factory)).func is diagnostic_protocol
    async with http_server(FastAPI(), idle=10) as url:
        async with ClientSession() as client:
            async with client.get(url + "/private-secret") as response:
                assert response.status == 404
                await response.read()
        await asyncio.sleep(0.02)
        recorded = events(capsys)
        assert len(recorded) == 1
        assert recorded[0]["local_trigger"] == "unknown"
        assert "secret" not in json.dumps(recorded)


def test_failure_metadata_does_not_claim_server_closer():
    error = ServerDisconnectedError("header-secret")
    with rollout_context("bad\\nsecret"):
        result = json.loads(failure_context(error, "POST", "/private-secret", replay_allowed=False, final=True))
    assert result == {
        "endpoint": {"scheme": None, "host": None, "port": None},
        "method": "POST",
        "operation": "other",
        "rollout_id": None,
        "replay_allowed": False,
        "final_attempt": True,
    }


@pytest.mark.asyncio
async def test_server_parser_reject_has_safe_local_trigger(capsys):
    async with http_server(FastAPI()) as url:
        port = int(url.rsplit(":", 1)[1])
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"INVALID secret-header\r\n\r\n")
        await writer.drain()
        response = await reader.read()
        assert b"400 Bad Request" in response
        writer.close()
        await writer.wait_closed()
        await asyncio.sleep(0)
        event = events(capsys)[0]
        assert event["local_trigger"] == "parser_reject"
        assert event["last_parsed_request"] is None
        assert "secret" not in json.dumps(event)


@pytest.mark.asyncio
async def test_server_idle_shutdown_has_safe_local_trigger(capsys):
    async with ClientSession() as client:
        async with http_server(FastAPI()) as url:
            async with client.get(url) as response:
                await response.read()
        event = events(capsys)[0]
        assert event["local_trigger"] == "shutdown_idle"


@pytest.mark.asyncio
async def test_default_failure_log_has_metadata_without_new_socket_tracing(monkeypatch, capsys):
    async def drop(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.transport.close()

    async with raw_peer(drop) as url:
        async with ClientSession() as client:
            monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", client)
            monkeypatch.setattr(server_utils, "_NUM_SERVER_DISCONNECTED_ERROR", 99)
            with pytest.raises(ServerDisconnectedError) as caught:
                await server_utils.request("POST", url + f"/policy/episodes/{EPISODE}", _max_connection_retries=1)
            assert not hasattr(caught.value, "_gym_transport_context")
            output = capsys.readouterr().out
            assert output.count("[request_retry") == 1
            assert '"operation":"enroll"' in output
            assert '"transport"' not in output
        monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("debug", [False, True])
async def test_actual_head_startup_selects_diagnostics_without_legacy_debug(debug, monkeypatch, capsys):
    from nemo_gym.config_types import BaseServerConfig

    monkeypatch.setattr(
        server_utils.ServerClient, "load_head_server_config", lambda: BaseServerConfig(host="127.0.0.1", port=0)
    )
    monkeypatch.setattr(server_utils, "get_global_config_dict", lambda: {"global_http_transport_diagnostics": debug})
    server, thread, head = server_utils.HeadServer.run_webserver()
    try:
        async with asyncio.timeout(10):
            while not server.started:
                await asyncio.sleep(0.005)
        port = server.servers[0].sockets[0].getsockname()[1]
        head.mark_ready()
        async with ClientSession() as client:
            async with client.get(f"http://127.0.0.1:{port}/health") as response:
                assert response.status == 200
                assert await response.json() == {"status": "ok"}
        await asyncio.sleep(0.03)
        recorded = events(capsys)
        assert bool(recorded) == debug
        if debug:
            assert recorded[0]["local_trigger"] == "unknown"
    finally:
        server.should_exit = True
        await asyncio.to_thread(thread.join, 10)
        assert not thread.is_alive()


@pytest.mark.asyncio
async def test_diagnostics_preserve_the_existing_metrics_connector(monkeypatch):
    from nemo_gym.telemetry import connection_pool

    monkeypatch.setattr(connection_pool, "is_metrics_exporter_active", lambda: True)
    connector = build_connection_pool_connector(transport_debug=True)
    try:
        assert isinstance(connector, ConnectionDiagnosticMixin)
        assert isinstance(connector, connection_pool.QueueTimedTCPConnector)
    finally:
        await connector.close()


@pytest.mark.asyncio
async def test_caller_trace_context_is_unchanged_and_not_copied_to_diagnostics():
    from aiohttp import TraceConfig

    caller_context = {"secret": "caller-secret"}
    seen = []
    caller_trace = TraceConfig()

    async def inspect_context(session, context, params):
        seen.append(context.trace_request_ctx)

    caller_trace.on_request_start.append(inspect_context)

    async def drop(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.transport.close()

    async with raw_peer(drop) as url:
        async with ClientSession(
            connector=build_connection_pool_connector(transport_debug=True),
            trace_configs=[caller_trace, client_trace_config()],
        ) as client:
            with pytest.raises(ServerDisconnectedError) as caught:
                await client.post(url, trace_request_ctx=caller_context)
            assert seen == [caller_context]
            assert seen[0] is caller_context
            assert caller_context == {"secret": "caller-secret"}
            assert "secret" not in json.dumps(caught.value._gym_transport_context)


@pytest.mark.asyncio
async def test_later_end_trace_failure_preserves_primary_exception_and_context():
    from aiohttp import TraceConfig

    from nemo_gym.http_connection_diagnostics import _ACTIVE_TRACE

    caller_trace = TraceConfig()
    primary = RuntimeError("original caller trace failure")

    async def fail_end(session, context, params):
        raise primary

    caller_trace.on_request_end.append(fail_end)

    async def respond(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        await writer.drain()
        await reader.read()

    async with raw_peer(respond) as url:
        async with ClientSession(
            connector=build_connection_pool_connector(transport_debug=True),
            trace_configs=[client_trace_config(), caller_trace],
        ) as client:
            with rollout_context("immutable-request-id"):
                with pytest.raises(RuntimeError) as caught:
                    await client.post(url)
                assert caught.value is primary
                assert _ACTIVE_TRACE.get() is None
                assert current_rollout_id() == "immutable-request-id"
                assert primary._gym_transport_context["rollout_id"] == "immutable-request-id"
            assert current_rollout_id() is None


@pytest.mark.asyncio
async def test_later_start_trace_failure_leaves_no_active_transport_context():
    from aiohttp import TraceConfig

    from nemo_gym.http_connection_diagnostics import _ACTIVE_TRACE

    caller_trace = TraceConfig()
    primary = RuntimeError("original caller start failure")

    async def fail_start(session, context, params):
        raise primary

    caller_trace.on_request_start.append(fail_start)
    async with ClientSession(trace_configs=[client_trace_config(), caller_trace]) as client:
        with rollout_context("immutable-start-id"):
            with pytest.raises(RuntimeError) as caught:
                await client.post("http://127.0.0.1:1")
            assert caught.value is primary
            assert _ACTIVE_TRACE.get() is None
            assert current_rollout_id() == "immutable-start-id"
        assert current_rollout_id() is None
