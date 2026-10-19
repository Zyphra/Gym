# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content-free projections for the existing model-call capture contract."""

import math
import re
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from time import perf_counter
from uuid import uuid4


_CALL = ContextVar("gym_model_call_metadata", default=None)
_OWNER = ContextVar("gym_model_call_owner", default=None)
_ROLES = {"policy", "user_simulator", "judge", "tool_simulator", "compaction", "unknown"}
_ID = re.compile(r"[a-f0-9]{32}\Z")


def _id(value):
    return value if isinstance(value, str) and _ID.fullmatch(value) else None


@contextmanager
def model_call_context(role, *, operation_id=None, depends_on=None, measured_spans_ms=None, preserve_operation=False):
    """Caller-owned semantic scope; never modifies native inference inputs."""
    previous = _OWNER.get() or {}
    reuse = preserve_operation and previous.get("role") == role
    spans = {**previous.get("measured_spans_ms", {}), **(measured_spans_ms or {})}
    owner = {
        "role": role if role in _ROLES else "unknown",
        "operation_id": _id(operation_id) or (previous.get("operation_id") if reuse else None) or uuid4().hex,
        "parent_operation_id": previous.get("parent_operation_id") if reuse else previous.get("operation_id"),
        "depends_on_operation_ids": [value for value in (previous.get("depends_on_operation_ids", []) if depends_on is None else depends_on) if _id(value)],
        "measured_spans_ms": {
            name: float(value) for name, value in spans.items()
            if name in {"session_lock_wait_ms", "decision_cache_wait_ms", "judge_semaphore_wait_ms", "owned_thread_start_delay_ms", "policy_inbox_wait_ms"}
            and type(value) in (int, float) and math.isfinite(value) and value >= 0
        },
    }
    token = _OWNER.set(owner)
    try:
        yield owner
    finally:
        _OWNER.reset(token)


def metadata_headers():
    owner = _OWNER.get()
    if owner is None:
        return {}
    headers = {"x-nemo-gym-call-role": owner["role"], "x-nemo-gym-operation-id": owner["operation_id"]}
    if owner["parent_operation_id"]:
        headers["x-nemo-gym-parent-operation-id"] = owner["parent_operation_id"]
    if owner["depends_on_operation_ids"]:
        headers["x-nemo-gym-depends-on"] = ",".join(owner["depends_on_operation_ids"])
    for name, value in owner["measured_spans_ms"].items():
        headers["x-nemo-gym-span-" + name] = str(value)
    return headers


@contextmanager
def metadata_exchange(headers):
    role = headers.get("x-nemo-gym-call-role")
    spans = {}
    for name in ("session_lock_wait_ms", "decision_cache_wait_ms", "judge_semaphore_wait_ms", "owned_thread_start_delay_ms", "policy_inbox_wait_ms"):
        try:
            value = float(headers.get("x-nemo-gym-span-" + name, "nan"))
            if math.isfinite(value) and value >= 0:
                spans[name] = value
        except ValueError:
            pass
    exchange = {
        "metadata_only": True, "role": role if role in _ROLES else "unknown",
        "operation_id": _id(headers.get("x-nemo-gym-operation-id")),
        "parent_operation_id": _id(headers.get("x-nemo-gym-parent-operation-id")),
        "depends_on_operation_ids": [value for value in headers.get("x-nemo-gym-depends-on", "").split(",") if _id(value)],
        "measured_spans_ms": spans,
        "server_queue_ms": None, "server_service_ms": None,
    }
    token = _CALL.set(exchange)
    try:
        yield exchange
    finally:
        _CALL.reset(token)


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def record_request_metadata(body, *, resolved=False):
    exchange = _CALL.get()
    if exchange is None:
        return
    profile = {}
    for name in ("temperature", "top_p", "top_k", "min_p", "max_tokens", "max_output_tokens", "max_completion_tokens"):
        value = _get(body, name)
        if type(value) in (int, float) and math.isfinite(value):
            profile[name] = value
    effort = _get(body, "reasoning_effort") or _get(_get(body, "reasoning"), "effort")
    if effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
        profile["reasoning_effort"] = effort
    thinking = _get(_get(body, "chat_template_kwargs"), "enable_thinking")
    if type(thinking) is bool:
        profile["enable_thinking"] = thinking
    exchange["resolved_profile" if resolved else "requested_profile"] = profile
    model = _get(body, "model")
    if isinstance(model, str):
        exchange["model"] = model


def record_response_metadata(response, *, raw_client=False):
    exchange = _CALL.get()
    if exchange is None:
        return
    from nemo_gym.base_responses_api_model import _cache_signal, extract_token_stats
    usage = _get(response, "usage")
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump(exclude_unset=True)
    if raw_client or not exchange.get("_raw_usage_observed"):
        exchange.update(extract_token_stats(usage))
        cached = _cache_signal(usage)[1]
        # Adapters can manufacture cache-zero while normalizing usage.
        # Only the decoded upstream response establishes an observed zero.
        exchange["cached_tokens"] = cached if raw_client or cached else None
    if raw_client:
        exchange["_raw_usage_observed"] = True
    # Only protocol identifiers/status; never choices, text, messages or raw bytes.
    exchange["response_id"] = _get(response, "id")
    exchange["response_status"] = _get(response, "status")
    choices = _get(response, "choices")
    if isinstance(choices, list) and choices:
        exchange["finish_reason"] = _get(choices[0], "finish_reason")


@contextmanager
def measured_model_client():
    started = perf_counter()
    try:
        yield
    finally:
        exchange = _CALL.get()
        if exchange is not None:
            exchange["client_elapsed_ms"] = 1000 * (perf_counter() - started)


def model_call_role(role):
    """Mark the actual role owner without changing its signature or arguments."""
    from functools import wraps
    def decorate(function):
        @wraps(function)
        async def invoke(*args, **kwargs):
            with model_call_context(role):
                return await function(*args, **kwargs)
        return invoke
    return decorate



@asynccontextmanager
async def observed_lock(lock):
    """The same acquire/release as asyncio.Lock; duration is caller lock wait only."""
    started = perf_counter()
    await lock.acquire()
    try:
        owner = _OWNER.get() or {}
        with model_call_context(owner.get("role", "unknown"), measured_spans_ms={
            "session_lock_wait_ms": 1000 * (perf_counter() - started),
        }):
            yield
    finally:
        lock.release()
