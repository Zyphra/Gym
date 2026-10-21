# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real model-server and local HTTP controls; no inference service is called."""

import asyncio
from contextlib import asynccontextmanager

import httpx
import pytest
from aiohttp import ClientResponseError, web
from omegaconf import OmegaConf

from nemo_gym.base_responses_api_model import CaptureStore, read_model_call_records
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.model_call_metadata import metadata_exchange
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import (
    GlobalAIOHTTPAsyncClientConfig,
    ServerClient,
    close_global_aiohttp_client,
    set_global_aiohttp_client,
)
from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig


@asynccontextmanager
async def local_server(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    requests = []

    async def respond(request):
        body = await request.json()
        requests.append(body)
        marker = body.get("input") or body.get("messages", [{}])[0].get("content")
        if marker == "hold-private-input":
            entered.set()
            await release.wait()
        if marker == "error-private-input":
            return web.json_response({"error": "private upstream error"}, status=503)
        noop = marker == "noop-private-input"
        usage = {"prompt_tokens": 3, "completion_tokens": 0 if noop else 2, "total_tokens": 3 if noop else 5}
        if request.path.endswith("chat/completions"):
            return web.json_response(
                {
                    "id": "fixture-response",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "fixture-model",
                    "usage": usage,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "" if noop else "gold-private-output",
                            },
                        }
                    ],
                }
            )
        return web.json_response(
            {
                "id": "fixture-response",
                "object": "response",
                "created_at": 0,
                "model": "fixture-model",
                "status": "completed",
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "usage": {
                    "input_tokens": 3,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 0 if noop else 2,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 3 if noop else 5,
                },
                "output": []
                if noop
                else [
                    {
                        "id": "fixture-message",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "annotations": [], "text": "gold-private-output"}],
                    }
                ],
            }
        )

    app = web.Application()
    app.router.add_post("/v1/responses", respond)
    app.router.add_post("/v1/chat/completions", respond)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    config = OmegaConf.create(
        {
            "observability_enabled": True,
            "model_call_capture_enabled": False,
            "model_call_capture_metadata_only": True,
            "model_call_capture_dir": str(tmp_path),
        }
    )
    server = SimpleModelServer(
        config=SimpleModelServerConfig(
            host="127.0.0.1",
            port=1,
            name="fixture-helper",
            entrypoint="",
            num_workers=1,
            openai_base_url=f"http://127.0.0.1:{port}/v1",
            openai_api_key="local",
            openai_model="fixture-model",
            max_concurrent_requests=1,
        ),
        server_client=ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=1),
            global_config_dict=config,
        ),
    )
    set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
    try:
        yield server, entered, release, requests
    finally:
        release.set()
        await close_global_aiohttp_client()
        await runner.cleanup()


@pytest.mark.parametrize("dialect", ["responses", "chat/completions"])
async def test_metadata_only_separates_admission_and_upstream_without_content(tmp_path, dialect, record_property):
    async with local_server(tmp_path) as (server, entered, release, requests):
        app = server.setup_webserver()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:

            def body(marker):
                return {
                    "model": "fixture-model",
                    "temperature": 0.7,
                    **(
                        {"input": marker}
                        if dialect == "responses"
                        else {
                            "messages": [{"role": "user", "content": marker}],
                        }
                    ),
                }

            first = asyncio.create_task(client.post(f"/ng-rollout/gold/v1/{dialect}", json=body("hold-private-input")))
            await asyncio.wait_for(entered.wait(), 5)
            second = asyncio.create_task(
                client.post(f"/ng-rollout/noop/v1/{dialect}", json=body("noop-private-input"))
            )
            await asyncio.sleep(0.05)
            assert len(requests) == 1
            release.set()
            responses = await asyncio.wait_for(asyncio.gather(first, second), 5)
        assert [response.status_code for response in responses] == [200, 200]
        assert len(requests) == 2
        assert all(request["temperature"] == 0.7 for request in requests)
        gold = read_model_call_records(CaptureStore(tmp_path), "gold")[0]
        noop = read_model_call_records(CaptureStore(tmp_path), "noop")[0]
        assert gold.server_service_ms >= 40
        assert noop.server_queue_ms >= 20
        assert gold.server_queue_ms >= 0 and noop.server_service_ms >= 0
        record_property("gold_service_ms", gold.server_service_ms)
        record_property("noop_queue_ms", noop.server_queue_ms)
        record_property("noop_service_ms", noop.server_service_ms)
        assert gold.tokens_out == 2 and noop.tokens_out == 0
        expected_reasoning = 0 if dialect == "responses" else None
        assert gold.tokens_reasoning == expected_reasoning and noop.tokens_reasoning == expected_reasoning
        for record in (gold, noop):
            assert record.metadata_only is True
            assert record.request is None and record.response is None
            assert record.request_raw is None and record.response_raw is None
        text = "".join(path.read_text() for path in tmp_path.glob("*.capture.jsonl"))
        assert "private-input" not in text and "private-output" not in text
        assert "gold-private-output" in responses[0].text


async def test_upstream_error_releases_capacity_and_preserves_missing_usage(tmp_path, record_property):
    async with local_server(tmp_path) as (server, _entered, _release, requests):
        body = NeMoGymResponseCreateParamsNonStreaming(input="error-private-input")
        with metadata_exchange({}) as exchange:
            with pytest.raises(ClientResponseError) as raised:
                await server.responses(body)
        assert raised.value.status == 503
        assert exchange["server_queue_ms"] >= 0 and exchange["server_service_ms"] >= 0
        record_property("error_service_ms", exchange["server_service_ms"])
        assert exchange.get("tokens_out") is None
        with metadata_exchange({}) as following:
            await asyncio.wait_for(
                server.responses(NeMoGymResponseCreateParamsNonStreaming(input="noop-private-input")), 5
            )
        assert following["tokens_out"] == 0
        assert len(requests) == 2


@pytest.mark.parametrize("queued", [False, True])
async def test_cancellation_keeps_original_semaphore_and_records_only_reached_boundaries(
    tmp_path, queued, record_property
):
    async with local_server(tmp_path) as (server, entered, _release, requests):
        exchange = None
        if queued:
            await server._semaphore.acquire()

        async def call():
            nonlocal exchange
            with metadata_exchange({}) as exchange:
                await server.responses(NeMoGymResponseCreateParamsNonStreaming(input="hold-private-input"))

        task = asyncio.create_task(call())
        if queued:
            await asyncio.sleep(0.02)
        else:
            await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert exchange["server_queue_ms"] >= 0
        record_property("cancel_queue_ms", exchange["server_queue_ms"])
        record_property("cancel_service_ms", exchange["server_service_ms"])
        if queued:
            assert exchange["server_service_ms"] is None
            assert requests == []
            server._semaphore.release()
        else:
            assert exchange["server_service_ms"] >= 0
        with metadata_exchange({}):
            await asyncio.wait_for(
                server.responses(NeMoGymResponseCreateParamsNonStreaming(input="noop-private-input")), 5
            )
