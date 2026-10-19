# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import select
import ssl
import subprocess
import threading
from contextlib import contextmanager

import aiohttp
import pytest
import uvloop
from aiohttp import TCPConnector, web

from nemo_gym.server_utils import PollSafeTCPConnector


@pytest.fixture(scope="module")
def tls_context(tmp_path_factory):
    root = tmp_path_factory.mktemp("pool-tls")
    key, cert = root / "key.pem", root / "cert.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
        ],
        check=True,
        capture_output=True,
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    context.verify_mode = ssl.CERT_OPTIONAL
    context.post_handshake_auth = True
    return context


class _ThreadedServer:
    """Real HTTP server whose transport loop remains independent of the client loop."""

    def __init__(self, tls_context=None):
        self._tls_context = tls_context
        self._ready = threading.Event()
        self._records_lock = threading.Lock()
        self._records = []
        self._warm_transport = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        assert self._ready.wait(5), "server did not start"

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._runner = web.AppRunner(web.Application())
        self._runner.app.router.add_post("/{name}", self._handle)
        self._loop.run_until_complete(self._start())
        self._ready.set()
        self._loop.run_forever()
        self._loop.run_until_complete(self._runner.cleanup())
        self._loop.close()

    async def _start(self):
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0, ssl_context=self._tls_context)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]

    async def _handle(self, request):
        await request.read()
        connection = id(request.transport)
        name = request.match_info["name"]
        with self._records_lock:
            self._records.append((name, connection))
            if name == "warm":
                self._warm_transport = request.transport
        return web.json_response({"name": name, "connection": connection})

    @property
    def url(self):
        return f"{'https' if self._tls_context else 'http'}://127.0.0.1:{self.port}"

    def records(self, name):
        with self._records_lock:
            return [connection for path, connection in self._records if path == name]

    def close_warm_while_client_loop_is_blocked(self):
        """Put a peer FIN in the kernel queue without advancing the client loop."""

        done = threading.Event()

        async def close_after_flush():
            assert self._warm_transport is not None
            self._warm_transport.close()
            # Ensure close() has reached the selector before releasing the blocked client loop.
            await asyncio.sleep(0.02)
            done.set()

        self._loop.call_soon_threadsafe(lambda: self._loop.create_task(close_after_flush()))
        assert done.wait(5), "server transport did not close"

    def stop(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(5)
        assert not self._thread.is_alive()


@contextmanager
def _server():
    server = _ThreadedServer()
    try:
        yield server
    finally:
        server.stop()


async def _post(session, server, name):
    async with session.post(f"{server.url}/{name}", json={"name": name}) as response:
        assert response.status == 200
        return await response.json()


@pytest.mark.parametrize("loop_factory", [asyncio.new_event_loop, uvloop.new_event_loop])
@pytest.mark.parametrize("mode", ["healthy", "queued_fin", "trace_fin", "trace_cancel", "trace_close"])
@pytest.mark.parametrize("tls", [False, True])
def test_native_lifecycle(loop_factory, mode, tls, tls_context):
    async def exercise():
        server = _ThreadedServer(tls_context if tls else None)
        connector = PollSafeTCPConnector(limit=2, keepalive_timeout=15, ssl=False)
        trace = aiohttp.TraceConfig()

        async def reused(session, context, params):
            if mode == "trace_fin":
                server.close_warm_while_client_loop_is_blocked()
            elif mode == "trace_cancel":
                raise asyncio.CancelledError
            elif mode == "trace_close":
                await connector.close()

        trace.on_connection_reuseconn.append(reused)
        try:
            async with aiohttp.ClientSession(connector=connector, trace_configs=[trace]) as session:
                warm = await _post(session, server, "warm")
                if mode == "queued_fin":
                    server.close_warm_while_client_loop_is_blocked()
                if mode in {"trace_cancel", "trace_close"}:
                    expected = asyncio.CancelledError if mode == "trace_cancel" else aiohttp.ClientConnectionError
                    with pytest.raises(expected):
                        await _post(session, server, "follow")
                    assert server.records("follow") == []
                    assert not connector._acquired
                    assert not connector._acquired_per_host
                else:
                    follow = await _post(session, server, "follow")
                    assert len(server.records("follow")) == 1
                    assert (warm["connection"] == follow["connection"]) == (mode == "healthy")
                    assert not connector._acquired
        finally:
            server.stop()

    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(exercise())


@pytest.mark.parametrize("loop_factory", [asyncio.new_event_loop, uvloop.new_event_loop])
def test_native_connector_reuse_trace_close_control(loop_factory):
    async def exercise():
        with _server() as server:
            connector = TCPConnector(limit=2, keepalive_timeout=15)
            trace = aiohttp.TraceConfig()

            async def reused(session, context, params):
                server.close_warm_while_client_loop_is_blocked()

            trace.on_connection_reuseconn.append(reused)
            async with aiohttp.ClientSession(connector=connector, trace_configs=[trace]) as session:
                await _post(session, server, "warm")
                with pytest.raises(aiohttp.ServerDisconnectedError):
                    await _post(session, server, "follow")
                assert server.records("follow") == []

    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(exercise())


@pytest.mark.parametrize("loop_factory", [asyncio.new_event_loop, uvloop.new_event_loop])
def test_high_descriptor_native_reuse(loop_factory):
    descriptors = []
    try:
        while not descriptors or descriptors[-1] <= 1100:
            descriptors.append(os.open(os.devnull, os.O_RDONLY))

        async def exercise():
            with _server() as server:
                connector = PollSafeTCPConnector(limit=1)
                async with aiohttp.ClientSession(connector=connector) as session:
                    warm = await _post(session, server, "warm")
                    follow = await _post(session, server, "follow")
                    assert warm["connection"] == follow["connection"]
                    assert len(server.records("follow")) == 1

        with asyncio.Runner(loop_factory=loop_factory) as runner:
            runner.run(exercise())
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


@pytest.mark.parametrize("loop_factory", [asyncio.new_event_loop, uvloop.new_event_loop])
def test_concurrent_pool_accounting(loop_factory):
    async def exercise():
        with _server() as server:
            connector = PollSafeTCPConnector(limit=2, limit_per_host=2)
            async with aiohttp.ClientSession(connector=connector) as session:
                await asyncio.gather(*(_post(session, server, "warm") for _ in range(2)))
                server.close_warm_while_client_loop_is_blocked()
                await asyncio.wait_for(
                    asyncio.gather(*(_post(session, server, f"follow-{i}") for i in range(8))),
                    timeout=5,
                )
                assert all(len(server.records(f"follow-{i}")) == 1 for i in range(8))
                assert not connector._acquired
                assert not connector._acquired_per_host

    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(exercise())


@pytest.mark.parametrize("loop_factory", [asyncio.new_event_loop, uvloop.new_event_loop])
def test_pending_healthy_tls_posthandshake_record(loop_factory, tls_context):
    async def exercise():
        server = _ThreadedServer(tls_context)
        client_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        client_context.check_hostname = False
        client_context.verify_mode = ssl.CERT_NONE
        client_context.post_handshake_auth = True
        connector = PollSafeTCPConnector(limit=1, ssl=client_context)
        try:
            async with aiohttp.ClientSession(connector=connector) as session:
                warm = await _post(session, server, "warm")

                async def send_posthandshake_record():
                    transport = server._warm_transport
                    assert not transport.is_closing()
                    ssl_object = transport.get_extra_info("ssl_object")
                    ssl_object.verify_client_post_handshake()
                    # Flush a real OpenSSL CertificateRequest through the real
                    # server TLS transport, without sending an HTTP body.
                    ssl_object.write(b"")
                    transport._ssl_protocol._process_outgoing()
                    await asyncio.sleep(0.02)
                    assert not transport.is_closing()

                asyncio.run_coroutine_threadsafe(send_posthandshake_record(), server._loop).result(5)
                protocol = next(iter(connector._conns.values()))[0][0]
                sock = protocol.transport.get_extra_info("socket")
                poller = select.poll()
                poller.register(sock, select.POLLIN)
                assert poller.poll(0), "fixture did not deliver the TLS control record"
                assert protocol.is_connected()
                follow = await _post(session, server, "follow")
                assert follow["connection"] != warm["connection"]
                assert len(server.records("follow")) == 1
                assert not connector._acquired
        finally:
            server.stop()

    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(exercise())
