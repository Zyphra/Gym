# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content-free per-request wall time of the calls a server makes on behalf of one request.

A server request opens a scope (``CallTimingMiddleware``). Code that calls a judge, a user simulator, a
tool simulator, a compaction model or a ``/verify`` endpoint records the elapsed time under a role. The
scope's totals leave the server in the ``x-nemo-gym-call-timing`` response header, and ``ServerClient``
merges that header into its caller's scope, so a resource server's judge time rolls up through the agent
into the agent's ``/run`` response header. Nothing here reads or stores a request or response body.

Header value: compact JSON ``{"<role>": {"n": <calls>, "sum_ms": <total>, "max_ms": <longest>}}``.
"""

import json
import math
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter


HEADER = "x-nemo-gym-call-timing"
ROLES = frozenset({"judge", "user_simulator", "tool_simulator", "compaction", "verify"})

_SCOPE: ContextVar = ContextVar("gym_call_timing", default=None)


def _add(scope, role, n, sum_ms, max_ms):
    entry = scope.setdefault(role, {"n": 0, "sum_ms": 0.0, "max_ms": 0.0})
    entry["n"] += n
    entry["sum_ms"] += sum_ms
    entry["max_ms"] = max(entry["max_ms"], max_ms)


def record_call(role, elapsed_ms):
    """Add one call to the active scope; a no-op without a scope, for an unknown role or a bad value."""
    scope = _SCOPE.get()
    if scope is None or role not in ROLES:
        return
    if type(elapsed_ms) not in (int, float) or not math.isfinite(elapsed_ms) or elapsed_ms < 0:
        return
    _add(scope, role, 1, float(elapsed_ms), float(elapsed_ms))


@contextmanager
def timed_call(role):
    """Time the enclosed block (including when it raises) under ``role``."""
    started = perf_counter()
    try:
        yield
    finally:
        record_call(role, 1000.0 * (perf_counter() - started))


def merge_header(value):
    """Fold a downstream server's header into the active scope; malformed input is ignored, never raised."""
    scope = _SCOPE.get()
    if scope is None or not value:
        return
    try:
        parsed = json.loads(value)
        items = list(parsed.items()) if isinstance(parsed, dict) else []
    except (TypeError, ValueError):
        return
    for role, entry in items:
        if role not in ROLES or not isinstance(entry, dict):
            continue
        n, sum_ms, max_ms = entry.get("n"), entry.get("sum_ms"), entry.get("max_ms")
        if (
            type(n) is not int
            or n < 0
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (sum_ms, max_ms))
        ):
            continue
        _add(scope, role, n, float(sum_ms), float(max_ms))


def header_value():
    """The active scope as a compact JSON header value, or ``None`` when nothing was recorded."""
    scope = _SCOPE.get()
    return json.dumps(scope, separators=(",", ":")) if scope else None


class CallTimingMiddleware:
    """Pure ASGI: one fresh scope per HTTP request; adds ``HEADER`` to the response when non-empty."""

    def __init__(self, app):
        self._app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return

        async def send_with_timing(message):
            if message["type"] == "http.response.start":
                value = header_value()
                if value is not None:
                    message = {**message, "headers": [*message.get("headers", []), (HEADER.encode(), value.encode())]}
            await send(message)

        token = _SCOPE.set({})
        try:
            await self._app(scope, receive, send_with_timing)
        finally:
            _SCOPE.reset(token)
