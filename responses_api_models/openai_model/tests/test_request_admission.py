# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
import json
import time
from contextlib import asynccontextmanager

import pytest
from aiohttp import ClientResponseError, ClientSession, ServerDisconnectedError, web
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.config_types import BaseServerConfig
from nemo_gym.openai_utils import (
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient
from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig
from responses_api_models.openai_model.request_admission import RequestAdmission, RequestRateLimit


def limits(**values):
    return RequestRateLimit(
        **(
            {
                "requests_per_period": 2,
                "tokens_per_period": 100_000,
                "period_seconds": 0.08,
                "output_token_reservation": 64,
                "input_token_overhead": 32,
                "admission_timeout_seconds": 2.0,
            }
            | values
        )
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("requests_per_period", 0),
        ("requests_per_period", True),
        ("tokens_per_period", -1),
        ("tokens_per_period", 1.5),
        ("period_seconds", float("inf")),
        ("output_token_reservation", 0),
        ("admission_timeout_seconds", float("nan")),
    ],
)
def test_invalid_rate_limits_fail_before_dispatch(field, value):
    with pytest.raises(ValidationError):
        limits(**{field: value})


def test_body_accounting_preserves_unicode_tools_and_output_limit():
    config = limits()
    body = {
        "input": [{"role": "user", "content": "こんにちは"}],
        "tools": [{"name": "lookup"}],
        "max_output_tokens": 512,
    }
    original = copy.deepcopy(body)
    cost = config.request_tokens(body)
    assert cost >= len("こんにちは".encode()) + 512
    assert body == original
    with pytest.raises(ValueError, match="positive integer"):
        config.request_tokens({"max_tokens": True})


@pytest.mark.asyncio
async def test_weighted_token_limit_delays_even_when_request_count_fits():
    body = {"input": "ordinary message"}
    config = limits(requests_per_period=100)
    budget = config.model_copy(update={"tokens_per_period": config.request_tokens(body)})
    gate = RequestAdmission(budget)
    await gate.acquire(body)
    started = time.monotonic()
    await gate.acquire(body)
    assert time.monotonic() - started >= 0.06


@pytest.mark.asyncio
async def test_waiting_cancellation_has_no_later_budget_cost():
    gate = RequestAdmission(limits(requests_per_period=1))
    await gate.acquire({"input": "first"})
    waiting = asyncio.create_task(gate.acquire({"input": "cancelled"}))
    await asyncio.sleep(0.01)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    await asyncio.sleep(0.08)
    started = time.monotonic()
    await gate.acquire({"input": "third"})
    assert time.monotonic() - started < 0.05


@pytest.mark.asyncio
async def test_finite_admission_timeout_does_not_dispatch_or_corrupt_budget():
    gate = RequestAdmission(limits(requests_per_period=1, period_seconds=1.0, admission_timeout_seconds=0.03))
    await gate.acquire({"input": "first"})
    with pytest.raises(TimeoutError, match="admission deadline"):
        await gate.acquire({"input": "second"})


@asynccontextmanager
async def upstream_model(
    rate_limit=None,
    status=200,
    max_http_attempts=None,
    error_code="invalid_request_error",
    disconnects=0,
    redirect=False,
):
    received = []

    async def response(request):
        body = await request.json()
        received.append((time.monotonic(), body, request.path))
        if redirect and len(received) == 1:
            return web.json_response(
                {"error": {"message": "redirect control"}}, status=307, headers={"Location": "/v1/redirected"}
            )
        if len(received) <= disconnects:
            request.transport.close()
            return web.Response()
        if status != 200:
            return web.json_response({"error": {"message": "local control", "code": error_code}}, status=status)
        if request.path.endswith("chat/completions"):
            result = {
                "id": "local-chat",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Acknowledged."}}
                ],
            }
        else:
            result = {
                "id": "local-response",
                "object": "response",
                "created_at": 0,
                "model": body["model"],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "status": "completed",
                "output": [
                    {
                        "id": "local-message",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Acknowledged.", "annotations": []}],
                    }
                ],
            }
        return web.json_response(result)

    app = web.Application()
    app.router.add_post("/v1/responses", response)
    app.router.add_post("/v1/chat/completions", response)
    app.router.add_post("/v1/redirected", response)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    port = runner.addresses[0][1]
    config = SimpleModelServerConfig(
        host="127.0.0.1",
        port=1,
        name="local_model",
        entrypoint="",
        openai_base_url=f"http://127.0.0.1:{port}/v1",
        openai_api_key="local-test-placeholder",
        openai_model="local-model",
        max_concurrent_requests=16,
        request_rate_limit=rate_limit,
        **({"max_http_attempts": max_http_attempts} if max_http_attempts is not None else {}),
    )
    client = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=1, name="head_server", entrypoint=""),
        global_config_dict=OmegaConf.create({"observability_enabled": False}),
    )
    model = SimpleModelServer(config=config, server_client=client)
    import nemo_gym.server_utils

    # Use an actual aiohttp session and close it on this fixture's event loop;
    # hosted Gym's global lifecycle remains owned by its existing implementation.
    async with ClientSession() as session:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(nemo_gym.server_utils, "get_global_aiohttp_client", lambda: session)
            try:
                yield model, received
            finally:
                await runner.cleanup()


@pytest.mark.asyncio
async def test_real_responses_and_chat_share_one_atomic_request_budget():
    async with upstream_model(limits()) as (model, received):
        response = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "hello"}], store=False)
        chat = NeMoGymChatCompletionCreateParamsNonStreaming(messages=[{"role": "user", "content": "hello"}])
        await asyncio.gather(
            model.responses(response),
            model.chat_completions(chat),
            model.responses(response),
            model.chat_completions(chat),
            model.responses(response),
        )
        times = sorted(t for t, _, _ in received)
        assert len(times) == 5
        assert all(times[index + 2] - times[index] >= 0.06 for index in range(3))
        assert all(body["model"] == "local-model" for _, body, _ in received)
        assert {path for _, _, path in received} == {"/v1/responses", "/v1/chat/completions"}
        assert response.store is False


@pytest.mark.asyncio
async def test_default_off_and_body_projection_are_preserved():
    async with upstream_model() as (model, received):
        assert model.config.request_rate_limit is None
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "hello"}], temperature=0.0, store=False
        )
        original = body.model_dump(exclude_unset=True)
        await asyncio.gather(*(model.responses(body) for _ in range(8)))
        assert len(received) == 8
        assert all(payload == original | {"model": "local-model"} for _, payload, _ in received)


@pytest.mark.asyncio
async def test_failed_upstream_call_still_consumes_rate_budget():
    async with upstream_model(limits(requests_per_period=1), status=400) as (model, received):
        body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "hello"}])
        for _ in range(2):
            with pytest.raises(ClientResponseError) as caught:
                await model.responses(body)
            assert caught.value.status == 400
        assert len(received) == 2
        assert received[1][0] - received[0][0] >= 0.06


@pytest.mark.asyncio
async def test_oversized_reservation_fails_before_upstream_without_truncation():
    async with upstream_model(limits(tokens_per_period=64)) as (model, received):
        body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "complete input"}])
        with pytest.raises(ValueError, match="admission budget"):
            await model.responses(body)
        assert received == []
        assert body.input[0].content == "complete input"


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", ["requests", "tokens"])
@pytest.mark.parametrize("traced", [False, True])
async def test_real_retry_attempts_reserve_each_wire_request_and_token_budget(monkeypatch, budget, traced):
    import nemo_gym.server_utils

    monkeypatch.setattr(nemo_gym.server_utils, "is_span_group_enabled", lambda group: traced)
    body = {"model": "local-model", "input": [{"role": "user", "content": "こんにちは"}], "max_output_tokens": 64}
    config = limits(requests_per_period=1 if budget == "requests" else 100, period_seconds=1.2)
    if budget == "tokens":
        config = config.model_copy(update={"tokens_per_period": config.request_tokens(body)})
    original = copy.deepcopy(body)
    async with upstream_model(config, status=429) as (model, received):
        with pytest.raises(ClientResponseError) as terminal:
            await model._client.create_response(**body)
        assert terminal.value.status == 429
        assert json.loads(terminal.value.response_content)["error"]["message"] == "local control"
        assert len(received) == 3
        assert all(received[index + 1][0] - received[index][0] >= 1.05 for index in range(2))
        assert all(payload == original for _, payload, _ in received)
        assert body == original


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_attempt", [1, 2])
async def test_admission_failure_is_not_retried_or_dispatched(failing_attempt):
    async with upstream_model(status=429) as (model, received):
        attempts = []

        async def admission(body):
            attempts.append(copy.deepcopy(body))
            if len(attempts) == failing_attempt:
                raise TimeoutError("bounded admission control")

        model._client.set_request_attempt_admission(admission)
        body = {"model": "local-model", "input": "unchanged"}
        with pytest.raises(TimeoutError, match="bounded admission control"):
            await model._client.create_response(**body)
        assert len(attempts) == failing_attempt
        assert len(received) == failing_attempt - 1
        assert all(payload == body for payload in attempts)


@pytest.mark.asyncio
async def test_cancellation_while_retry_admission_waits_prevents_second_wire_request():
    config = limits(requests_per_period=1, period_seconds=10, admission_timeout_seconds=5)
    async with upstream_model(config, status=429) as (model, received):
        body = {"model": "local-model", "input": "cancel while waiting"}
        waiting = asyncio.create_task(model._client.create_response(**body))
        try:
            async with asyncio.timeout(2):
                while not received:
                    await asyncio.sleep(0.01)
            await asyncio.sleep(0.6)
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
            assert len(received) == 1
            assert received[0][1] == body
        finally:
            if not waiting.done():
                waiting.cancel()
                try:
                    await waiting
                except asyncio.CancelledError:
                    pass


def test_token_reservation_covers_every_supplied_output_limit_without_mutating_body():
    body = {"input": "unchanged", "max_output_tokens": 64, "max_completion_tokens": 2048, "max_tokens": 4096}
    original = copy.deepcopy(body)
    assert limits().request_tokens(body) >= 4096
    assert body == original


def test_invalid_secondary_output_limit_fails_before_dispatch():
    with pytest.raises(ValueError, match="positive integer"):
        limits().request_tokens({"max_output_tokens": 64, "max_tokens": True})


@pytest.mark.asyncio
@pytest.mark.parametrize("attempts", [None, 1])
@pytest.mark.parametrize("endpoint", ["responses", "chat"])
async def test_real_hosted_default_three_and_explicit_one_preserve_terminal_error_body(attempts, endpoint):
    async with upstream_model(status=429, max_http_attempts=attempts) as (model, received):
        expected = 3 if attempts is None else attempts
        assert model.config.request_rate_limit is None
        assert model.config.max_http_attempts == expected
        assert model._client.max_http_attempts == expected
        assert model.ray_enabled is False
        if endpoint == "responses":
            body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "unchanged"}])
            call = model.responses
        else:
            body = NeMoGymChatCompletionCreateParamsNonStreaming(messages=[{"role": "user", "content": "unchanged"}])
            call = model.chat_completions
        original = body.model_dump(exclude_unset=True)
        with pytest.raises(ClientResponseError) as terminal:
            await call(body)
        assert terminal.value.status == 429
        assert json.loads(terminal.value.response_content) == {
            "error": {"message": "local control", "code": "invalid_request_error"}
        }
        assert len(received) == expected
        assert all(payload == original | {"model": "local-model"} for _, payload, _ in received)
        assert body.model_dump(exclude_unset=True) == original


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,code", [(429, "insufficient_quota"), (401, "invalid_api_key"), (403, "invalid_api_key")]
)
async def test_real_permanent_trip_preserves_error_and_prevents_future_wire_attempts(status, code):
    async with upstream_model(limits(), status=status, error_code=code) as (model, received):
        body = NeMoGymResponseCreateParamsNonStreaming(input="unchanged")
        for _ in range(2):
            with pytest.raises(ClientResponseError) as terminal:
                await model.responses(body)
            assert terminal.value.status == status
            assert json.loads(terminal.value.response_content)["error"]["code"] == code
        assert len(received) == 1
        assert len(model._request_admission._entries) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("traced", [False, True])
async def test_real_connection_retry_reserves_each_attempt_and_keeps_existing_bound(monkeypatch, traced):
    import nemo_gym.server_utils

    monkeypatch.setattr(nemo_gym.server_utils, "is_span_group_enabled", lambda group: traced)
    async with upstream_model(limits(requests_per_period=1, period_seconds=1.2), disconnects=3) as (model, received):
        model._client.max_connection_retries = 2
        with pytest.raises(ServerDisconnectedError):
            await model._client.create_response(model="local-model", input="unchanged")
        assert len(received) == 2
        assert received[1][0] - received[0][0] >= 1.05
        assert len(model._request_admission._entries) == 1


@pytest.mark.asyncio
async def test_real_retry_admission_deadline_stops_before_second_wire_attempt():
    config = limits(requests_per_period=1, period_seconds=10, admission_timeout_seconds=0.03)
    async with upstream_model(config, status=429) as (model, received):
        with pytest.raises(TimeoutError, match="admission deadline"):
            await model._client.create_response(model="local-model", input="unchanged")
        assert len(received) == 1
        assert len(model._request_admission._entries) == 1


def test_missing_finite_output_reservation_fails_closed_at_configuration():
    values = limits().model_dump()
    values.pop("output_token_reservation")
    with pytest.raises(ValidationError):
        RequestRateLimit(**values)
    body = {"input": "unchanged"}
    assert limits(output_token_reservation=4096).request_tokens(body) >= 4096


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_real_redirect_is_default_preserved_and_admitted_wire_fail_closed(enabled):
    async with upstream_model(limits() if enabled else None, redirect=True) as (model, received):
        response = await model._client._request(
            method="POST",
            url=f"{model.config.openai_base_url}/responses",
            json={"model": "local-model", "input": "unchanged"},
        )
        await response.read()
        assert response.status == (307 if enabled else 200)
        assert len(received) == (1 if enabled else 2)
        if enabled:
            assert len(model._request_admission._entries) == 1


@pytest.mark.asyncio
async def test_admitted_idempotent_method_rejected_before_any_wire_request():
    async with upstream_model(limits()) as (model, received):
        with pytest.raises(ValueError, match="model POST requests only"):
            await model._client.create_models()
        assert received == []
        assert not model._request_admission._entries
