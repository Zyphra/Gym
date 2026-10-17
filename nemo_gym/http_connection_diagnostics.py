# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in socket metadata; never request contents, headers, queries or raw exceptions.

A client EOF identifies an observed transport outcome, not the process that closed
it. Server close events identify a local trigger only where the protocol directly
invokes close. Unknown closes remain unknown, including ASGI-driven closes.
"""

import asyncio
import ipaddress
import json
import os
import re
import time
from contextvars import ContextVar
from functools import lru_cache
from typing import Any
from urllib.parse import urlsplit

from aiohttp import TraceConfig

from nemo_gym.rollout_correlation import current_rollout_id


_ACTIVE_TRACE: ContextVar[dict | None] = ContextVar("gym_http_connection_trace", default=None)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_UUID = re.compile(
    r"(?:[a-fA-F0-9]{32}|[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12})\Z"
)
_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})


def request_metadata(method: str, url: str, rollout_id: str | None = None) -> dict:
    """Classify only known routes. Malformed diagnostic input never changes HTTP behavior."""
    try:
        return _request_metadata(method, url, rollout_id)
    except Exception:
        return {"method": "OTHER", "operation": "other", "rollout_id": None}


def _request_metadata(method: str, url: str, rollout_id: str | None) -> dict:
    parsed = urlsplit(str(url))
    path = parsed.path.split("/")
    operation = "other"
    path_id = None
    if len(path) > 3 and path[1] == "ng-rollout" and _ID.fullmatch(path[2]):
        path_id, path = path[2], [""] + path[3:]
    if len(path) in (4, 5) and path[1:3] == ["policy", "episodes"] and _UUID.fullmatch(path[3]):
        path_id = path[3]
        if len(path) == 4:
            operation = {"POST": "enroll", "DELETE": "release"}.get(method.upper(), "episode")
        elif path[4] in ("finalize", "progress", "snapshot"):
            operation = path[4]
    elif path == ["", "run"]:
        operation = "run"
    elif path[-2:] in (["v1", "responses"], ["v1", "completions"]):
        operation = "model"
    elif path[-3:] == ["v1", "chat", "completions"]:
        operation = "model"
    identifier = rollout_id if isinstance(rollout_id, str) and _ID.fullmatch(rollout_id) else path_id
    try:
        port = parsed.port
    except ValueError:
        port = None
    host = parsed.hostname
    if host is not None and not re.fullmatch(r"[A-Za-z0-9.:-]{1,253}", host):
        host = None
    return {
        "endpoint": {
            "scheme": parsed.scheme if parsed.scheme in ("http", "https") else None,
            "host": host,
            "port": port,
        },
        "method": method.upper() if method.upper() in _METHODS else "OTHER",
        "operation": operation,
        "rollout_id": identifier,
    }


def _socket_address(value: Any) -> list | None:
    if not isinstance(value, tuple) or len(value) < 2 or not isinstance(value[1], int):
        return None
    try:
        return [str(ipaddress.ip_address(value[0])), value[1]]
    except ValueError:
        return None


def _socket_metadata(transport: Any) -> dict:
    try:
        if transport is not None:
            return {
                "local": _socket_address(transport.get_extra_info("sockname")),
                "peer": _socket_address(transport.get_extra_info("peername")),
            }
    except Exception:
        pass
    return {}


def _exception_metadata(error: BaseException | None) -> list:
    chain = []
    seen = set()
    while error is not None and id(error) not in seen and len(chain) < 4:
        seen.add(id(error))
        # Class names are bounded identifiers, never str(error), repr, headers or body.
        name = type(error).__name__
        chain.append(
            {
                "class": name[:80] if name.isidentifier() else "Exception",
                "errno": error.errno if isinstance(error, OSError) and type(error.errno) is int else None,
            }
        )
        error = error.__cause__ or error.__context__
    return chain


def _capture_connection(protocol: Any, reused: bool) -> None:
    state = _ACTIVE_TRACE.get()
    if state is None:
        return
    try:
        state.update(_socket_metadata(protocol.transport))
        state["connection"] = "reused" if reused else "new"
        released = getattr(protocol, "_gym_diagnostic_released_at", None)
        if reused and released is not None:
            state["pool_idle_s"] = max(0.0, time.monotonic() - released)
    except Exception:
        # Diagnostic failures must not change connector acquisition or cancellation.
        pass


class ConnectionDiagnosticMixin:
    """Opt-in metadata around the connector selected by existing pool policy."""

    async def _get(self, *args: Any, **kwargs: Any) -> Any:
        connection = await super()._get(*args, **kwargs)
        if connection is not None:
            _capture_connection(connection.protocol, reused=True)
        return connection

    async def _create_connection(self, *args: Any, **kwargs: Any) -> Any:
        protocol = await super()._create_connection(*args, **kwargs)
        _capture_connection(protocol, reused=False)
        return protocol

    def _release(self, key: Any, protocol: Any, **kwargs: Any) -> None:
        try:
            protocol._gym_diagnostic_released_at = time.monotonic()
        except Exception:
            pass
        super()._release(key, protocol, **kwargs)


def client_trace_config() -> TraceConfig:
    """Keep one bounded state per attempt, copying identity before context unwind.

    Phases denote aiohttp callbacks, not peer acknowledgments or proven delivery.
    Only failure state is attached to the exception. Existing retry log sites decide
    whether to print it, retaining their current sampling/finality behavior.
    """
    config = TraceConfig()

    async def start(session: Any, context: Any, params: Any) -> None:
        state = request_metadata(params.method, str(params.url), current_rollout_id())
        state.update(pid=os.getpid(), started_unix_s=time.time(), phase="start", connection="unknown")
        context.state = state
        context.started = time.monotonic()
        # aiohttp runs start callbacks outside its try/exception cleanup. Delay
        # activation until a connection callback inside that protected region.
        context.token = None

    async def finish(session: Any, context: Any, params: Any) -> None:
        try:
            if hasattr(params, "exception"):
                context.state.update(
                    elapsed_s=max(0.0, time.monotonic() - context.started),
                    exception_chain=_exception_metadata(params.exception),
                    outcome="client_cancelled"
                    if isinstance(params.exception, asyncio.CancelledError)
                    else "exception",
                )
                params.exception._gym_transport_context = context.state.copy()
        except Exception:
            pass
        finally:
            # Another trace's end callback can fail after our end callback ran,
            # causing aiohttp to invoke exception callbacks for the same request.
            # Consume once and never replace the primary exception with cleanup.
            token = getattr(context, "token", None)
            context.token = None
            if token is not None:
                try:
                    _ACTIVE_TRACE.reset(token)
                except (RuntimeError, ValueError):
                    pass

    def phase_handler(phase: str):
        async def handler(session: Any, context: Any, params: Any) -> None:
            if context.token is None and phase in ("queued", "connecting", "reusing"):
                context.token = _ACTIVE_TRACE.set(context.state)
            context.state["phase"] = phase

        return handler

    config.on_request_start.append(start)
    config.on_request_end.append(finish)
    config.on_request_exception.append(finish)
    for signal, phase in (
        (config.on_connection_queued_start, "queued"),
        (config.on_connection_create_start, "connecting"),
        (config.on_connection_create_end, "connected"),
        (config.on_connection_reuseconn, "reusing"),
        (config.on_request_headers_sent, "headers_sent"),
        (config.on_request_chunk_sent, "body_sent"),
    ):
        signal.append(phase_handler(phase))
    return config


def failure_context(error: BaseException, method: str, url: str, *, replay_allowed: bool, final: bool) -> str:
    """Safe additions to an existing sampled failure log (no extra log sites)."""
    result = request_metadata(method, url, current_rollout_id())
    # This reports the caller's current retry policy, not a new claim that replay
    # is semantically safe. Private callers can pass False for nonreplayable POSTs.
    result.update(replay_allowed=replay_allowed, final_attempt=final)
    context = getattr(error, "_gym_transport_context", None)
    if context is not None:
        result["transport"] = context
    try:
        return json.dumps(result, separators=(",", ":"), ensure_ascii=True)
    except Exception:
        result.pop("transport", None)
        return json.dumps(result, separators=(",", ":"), ensure_ascii=True)


class ServerConnectionDiagnosticMixin:
    """Opt-in close events; last parsed request is not necessarily the failed request."""

    def connection_made(self, transport: Any) -> None:
        super().connection_made(transport)
        self._gym_close_trigger = "unknown"
        self._gym_opened = time.monotonic()
        self._gym_opened_unix = time.time()
        self._gym_socket = _socket_metadata(transport)
        self._gym_last_request = None

    def data_received(self, data: bytes) -> None:
        super().data_received(data)
        scope = getattr(self, "scope", None)
        if isinstance(scope, dict) and isinstance(scope.get("method"), str) and isinstance(scope.get("path"), str):
            self._gym_last_request = request_metadata(scope.get("method", "OTHER"), scope.get("path", ""))

    def timeout_keep_alive_handler(self) -> None:
        # Native httptools/h11 closes synchronously. A deferred PollSafe protocol
        # must mark only in its final _close_keep_alive_transport hook instead.
        if not hasattr(super(), "_close_keep_alive_transport") and not self.transport.is_closing():
            self._gym_close_trigger = "idle_expiry"
        super().timeout_keep_alive_handler()

    def _close_keep_alive_transport(self) -> None:
        if not self.transport.is_closing():
            self._gym_close_trigger = "idle_expiry"
        super()._close_keep_alive_transport()

    def send_400_response(self, *args: Any, **kwargs: Any) -> None:
        self._gym_close_trigger = "parser_reject"
        super().send_400_response(*args, **kwargs)

    def shutdown(self) -> None:
        if self.cycle is None or self.cycle.response_complete:
            self._gym_close_trigger = "shutdown_idle"
        super().shutdown()

    def connection_lost(self, exc: Exception | None) -> None:
        try:
            event = {
                "event": "http_server_connection_close",
                "pid": os.getpid(),
                "opened_unix_s": self._gym_opened_unix,
                "closed_unix_s": time.time(),
                "connection_age_s": max(0.0, time.monotonic() - self._gym_opened),
                "local_trigger": self._gym_close_trigger,
                "last_parsed_request": self._gym_last_request,
                "exception_chain": _exception_metadata(exc),
                **self._gym_socket,
            }
            print(json.dumps(event, separators=(",", ":"), ensure_ascii=True), flush=True)
        except Exception:
            pass
        super().connection_lost(exc)


@lru_cache(maxsize=4)
def _diagnostic_protocol_class(base: type) -> type:
    return type("ConnectionDiagnostic" + base.__name__, (ServerConnectionDiagnosticMixin, base), {})


def diagnostic_protocol(*args: Any, base: type, **kwargs: Any) -> Any:
    """Importable factory so uvicorn multi-worker configs remain pickleable."""
    return _diagnostic_protocol_class(base)(*args, **kwargs)
