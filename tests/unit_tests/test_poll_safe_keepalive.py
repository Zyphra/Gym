import asyncio
import http.client
import socket
import time
from contextlib import asynccontextmanager
from threading import Event, Thread

import pytest
import uvicorn
import uvloop
from fastapi import FastAPI
from uvicorn.protocols.http.auto import AutoHTTPProtocol
from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol

from nemo_gym.server_utils import HEAD_SERVER_HTTP_PROTOCOL, PollSafeHttpToolsProtocol


def _post(sock: socket.socket, path: str) -> tuple[int, bytes]:
    request = (
        f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        "Content-Length: 0\r\nConnection: keep-alive\r\n\r\n"
    ).encode()
    sock.sendall(request)
    response = http.client.HTTPResponse(sock)
    response.begin()
    return response.status, response.read()


@pytest.fixture
def event_loop_policy():
    return uvloop.EventLoopPolicy()


class _PollBlockingMixin:
    instances = {}
    entered = Event()
    hold_seconds = 0.25
    deferred_close_calls = 0

    def connection_made(self, transport) -> None:
        super().connection_made(transport)
        self.lost = asyncio.Event()
        self.scheduled_stage = None
        self.close_entered = False
        self.instances[transport.get_extra_info("peername")] = self

    def connection_lost(self, exc) -> None:
        super().connection_lost(exc)
        self.lost.set()

    def _defer_keep_alive_close(self) -> None:
        super()._defer_keep_alive_close()
        self.scheduled_stage = self.timeout_keep_alive_task

    def data_received(self, data: bytes) -> None:
        if b"POST /hold " in data:
            self.entered.set()
            time.sleep(self.hold_seconds)
        super().data_received(data)

    def _close_keep_alive_transport(self) -> None:
        self.close_entered = True
        type(self).deferred_close_calls += 1
        super()._close_keep_alive_transport()


class _PollBlockingProtocol(_PollBlockingMixin, PollSafeHttpToolsProtocol):
    pass


class _NativePollBlockingProtocol(_PollBlockingMixin, HttpToolsProtocol):
    pass


@asynccontextmanager
async def _server(protocol: type[HttpToolsProtocol], keepalive: float = 0.2):
    app = FastAPI()
    handled = {"followup": 0}

    @app.post("/warm")
    async def warm():
        return {"ok": True}

    @app.post("/hold")
    async def hold():
        return {"ok": True}

    @app.post("/followup")
    async def followup():
        handled["followup"] += 1
        return {"count": handled["followup"]}

    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            http=protocol,
            loop="none",
            lifespan="off",
            timeout_keep_alive=keepalive,
            log_level="warning",
            access_log=False,
        )
    )
    task = asyncio.create_task(server.serve(sockets=[listener]))
    while not server.started:
        await asyncio.sleep(0.001)
    try:
        yield port, handled
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 2)


def _connect(port: int) -> socket.socket:
    return socket.create_connection(("127.0.0.1", port), timeout=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", [_NativePollBlockingProtocol, _PollBlockingProtocol])
async def test_kernel_ready_post_is_handled_once_after_overdue_timer(protocol) -> None:
    protocol.entered.clear()
    protocol.deferred_close_calls = 0
    async with _server(protocol) as (port, handled):
        warm = await asyncio.to_thread(_connect, port)
        assert await asyncio.to_thread(_post, warm, "/warm") == (200, b'{"ok":true}')
        await asyncio.sleep(0.12)

        result: list[tuple[int, bytes]] = []
        errors: list[BaseException] = []

        def follow() -> None:
            _PollBlockingProtocol.entered.wait(1)
            try:
                result.append(_post(warm, "/followup"))
            except BaseException as error:
                errors.append(error)

        follower = Thread(target=follow)
        trigger = Thread(target=lambda: _post(_connect(port), "/hold"))
        follower.start()
        trigger.start()
        await asyncio.to_thread(trigger.join)
        await asyncio.to_thread(follower.join)

        if protocol is _NativePollBlockingProtocol:
            assert len(errors) == 1
            assert isinstance(errors[0], ConnectionResetError)
            assert result == []
            assert handled["followup"] == 0
        else:
            assert errors == []
            assert result == [(200, b'{"count":1}')]
            assert handled["followup"] == 1
        warm.close()


@pytest.mark.asyncio
async def test_nominal_idle_connection_still_closes() -> None:
    async with _server(PollSafeHttpToolsProtocol, keepalive=0.05) as (port, _):
        client = await asyncio.to_thread(_connect, port)
        assert await asyncio.to_thread(_post, client, "/warm") == (200, b'{"ok":true}')
        started = time.monotonic()
        assert await asyncio.to_thread(client.recv, 1) == b""
        assert 0.04 <= time.monotonic() - started < 0.5


@pytest.mark.asyncio
async def test_peer_disconnect_cancels_pending_close_stage() -> None:
    _PollBlockingProtocol.entered.clear()
    _PollBlockingProtocol.deferred_close_calls = 0
    async with _server(_PollBlockingProtocol) as (port, _):
        warm = await asyncio.to_thread(_connect, port)
        assert await asyncio.to_thread(_post, warm, "/warm") == (200, b'{"ok":true}')
        warm_protocol = _PollBlockingProtocol.instances[warm.getsockname()]
        await asyncio.sleep(0.12)

        def disconnect_during_hold() -> None:
            _PollBlockingProtocol.entered.wait(1)
            warm.close()

        disconnect = Thread(target=disconnect_during_hold)
        trigger = Thread(target=lambda: _post(_connect(port), "/hold"))
        disconnect.start()
        trigger.start()
        await asyncio.to_thread(trigger.join)
        await asyncio.to_thread(disconnect.join)
        await asyncio.wait_for(warm_protocol.lost.wait(), 2)
        assert warm_protocol.transport.is_closing()
        assert warm_protocol.timeout_keep_alive_task is None
        assert warm_protocol.scheduled_stage is not None
        assert warm_protocol.scheduled_stage.cancelled() or warm_protocol.close_entered


def test_head_server_preserves_uvicorn_auto_parser_choice() -> None:
    expected = PollSafeHttpToolsProtocol if AutoHTTPProtocol is HttpToolsProtocol else AutoHTTPProtocol
    assert HEAD_SERVER_HTTP_PROTOCOL is expected
