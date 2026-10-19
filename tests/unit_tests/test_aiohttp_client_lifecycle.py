# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import queue
import threading

import pytest
from aiohttp import ClientSession, web

import nemo_gym.server_utils as owner
from nemo_gym.server_utils import PollSafeTCPConnector, close_global_aiohttp_client, global_aiohttp_client_exit


async def _live_client() -> ClientSession:
    app = web.Application()

    async def respond(request):
        return web.Response(text="owner-loop")

    app.router.add_get("/", respond)
    runner = web.AppRunner(app)
    await runner.setup()
    listener = web.TCPSite(runner, "127.0.0.1", 0)
    await listener.start()
    port = listener._server.sockets[0].getsockname()[1]
    client = ClientSession(connector=PollSafeTCPConnector())
    try:
        async with client.get(f"http://127.0.0.1:{port}/") as response:
            assert await response.text() == "owner-loop"
    except BaseException:
        await client.close()
        raise
    finally:
        await runner.cleanup()
    owner._GLOBAL_AIOHTTP_CLIENT = client
    return client


def test_exit_closes_real_connector_on_stopped_owner_loop_and_is_idempotent():
    loop = asyncio.new_event_loop()
    client = loop.run_until_complete(_live_client())
    connector = client.connector
    try:
        global_aiohttp_client_exit()
        global_aiohttp_client_exit()
        assert client.closed and connector.closed
        assert not owner.is_global_aiohttp_client_setup()
        assert not loop.is_closed()
    finally:
        loop.run_until_complete(client.close())
        owner._GLOBAL_AIOHTTP_CLIENT = None
        loop.close()


def test_exit_hands_real_client_to_running_owner_loop_in_another_thread():
    ready = queue.Queue()

    def serve():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        client = loop.run_until_complete(_live_client())
        ready.put((loop, client, client.connector))
        loop.run_forever()
        loop.run_until_complete(client.close())
        loop.close()

    thread = threading.Thread(target=serve)
    thread.start()
    loop, client, connector = ready.get(timeout=10)
    try:
        global_aiohttp_client_exit()
        assert client.closed and connector.closed
        assert not owner.is_global_aiohttp_client_setup()
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        owner._GLOBAL_AIOHTTP_CLIENT = None
    assert not thread.is_alive()


def test_explicit_async_close_uses_real_owner_loop_and_is_idempotent():
    async def scenario():
        client = await _live_client()
        connector = client.connector
        try:
            # Sync exit cannot block its own running loop. Explicit lifecycle
            # closure below is required and must not be reported as complete.
            global_aiohttp_client_exit()
            assert not client.closed and owner.is_global_aiohttp_client_setup()
            await close_global_aiohttp_client()
            await close_global_aiohttp_client()
            assert client.closed and connector.closed
            assert not owner.is_global_aiohttp_client_setup()
        finally:
            await client.close()
            owner._GLOBAL_AIOHTTP_CLIENT = None

    asyncio.run(scenario())


def test_explicit_async_close_rejects_real_foreign_loop_without_losing_client():
    loop = asyncio.new_event_loop()
    client = loop.run_until_complete(_live_client())
    try:
        with pytest.raises(RuntimeError, match="owning event loop"):
            asyncio.run(close_global_aiohttp_client())
        assert not client.closed and owner._GLOBAL_AIOHTTP_CLIENT is client
        loop.run_until_complete(close_global_aiohttp_client())
        assert client.closed and not owner.is_global_aiohttp_client_setup()
    finally:
        loop.run_until_complete(client.close())
        owner._GLOBAL_AIOHTTP_CLIENT = None
        loop.close()


def test_closed_owner_loop_reports_incomplete_cleanup_and_keeps_real_session():
    loop = asyncio.new_event_loop()
    client = loop.run_until_complete(_live_client())
    loop.close()
    try:
        with pytest.raises(RuntimeError, match="cleanup incomplete"):
            global_aiohttp_client_exit()
        assert not client.closed and owner._GLOBAL_AIOHTTP_CLIENT is client
    finally:
        # Test-only best effort disposal after the deliberate lifecycle failure.
        # There is no live owner loop on which asynchronous closure can settle.
        client.connector._close()
        client._connector = None
        owner._GLOBAL_AIOHTTP_CLIENT = None


def test_stalled_real_owner_loop_reports_deadline_without_claiming_close():
    import time

    ready = queue.Queue()
    stalled, release = threading.Event(), threading.Event()

    def serve():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        client = loop.run_until_complete(_live_client())
        ready.put((loop, client))

        def stall():
            stalled.set()
            assert release.wait(timeout=5)

        loop.call_soon(stall)
        loop.run_forever()
        loop.run_until_complete(client.close())
        loop.close()

    thread = threading.Thread(target=serve)
    thread.start()
    loop, client = ready.get(timeout=10)
    try:
        assert stalled.wait(timeout=2)
        before = time.monotonic()
        with pytest.raises(RuntimeError, match="cleanup incomplete.*deadline exceeded"):
            global_aiohttp_client_exit(timeout_s=0.05)
        assert time.monotonic() - before < 1
        assert not client.closed and owner._GLOBAL_AIOHTTP_CLIENT is client
    finally:
        release.set()
        # Explicitly settle on the now-responsive owner. Timeout above did not
        # certify closure; only this awaited operation releases the session.
        asyncio.run_coroutine_threadsafe(close_global_aiohttp_client(), loop).result(timeout=2)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        owner._GLOBAL_AIOHTTP_CLIENT = None
    assert not thread.is_alive() and client.closed


def test_stopped_owner_rejects_foreign_running_loop_without_losing_session():
    loop = asyncio.new_event_loop()
    client = loop.run_until_complete(_live_client())

    async def foreign():
        with pytest.raises(RuntimeError, match="cleanup incomplete.*another event loop"):
            global_aiohttp_client_exit()
        assert not client.closed and owner._GLOBAL_AIOHTTP_CLIENT is client

    try:
        asyncio.run(foreign())
        loop.run_until_complete(close_global_aiohttp_client())
        assert client.closed and not owner.is_global_aiohttp_client_setup()
    finally:
        loop.run_until_complete(client.close())
        owner._GLOBAL_AIOHTTP_CLIENT = None
        loop.close()


def test_async_close_closes_live_keepalive_transport_before_server_shutdown():
    async def scenario():
        app = web.Application()

        async def respond(request):
            return web.Response(text="live-keepalive")

        app.router.add_get("/", respond)
        runner = web.AppRunner(app)
        await runner.setup()
        listener = web.TCPSite(runner, "127.0.0.1", 0)
        await listener.start()
        port = listener._server.sockets[0].getsockname()[1]
        client = ClientSession(connector=PollSafeTCPConnector())
        connector = client.connector
        owner._GLOBAL_AIOHTTP_CLIENT = client
        try:
            async with client.get(f"http://127.0.0.1:{port}/") as response:
                assert await response.text() == "live-keepalive"
            transports = [proto.transport for values in connector._conns.values() for proto, _ in values]
            assert transports and all(not transport.is_closing() for transport in transports)
            await close_global_aiohttp_client()
            assert client.closed and connector.closed
            assert all(transport.is_closing() for transport in transports)
            assert owner._GLOBAL_AIOHTTP_CLIENT is None
        finally:
            await client.close()
            owner._GLOBAL_AIOHTTP_CLIENT = None
            await runner.cleanup()

    asyncio.run(scenario())
