# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content-free records from actual model wrappers and a local HTTP backend."""
import asyncio
import json
from uuid import uuid4

import pytest
from aiohttp import web
from omegaconf import OmegaConf

from nemo_gym.base_responses_api_model import CaptureStore, read_model_call_records
from nemo_gym.model_call_metadata import metadata_headers, model_call_context
from nemo_gym.server_utils import (
    BaseServerConfig,
    GlobalAIOHTTPAsyncClientConfig,
    ServerClient,
    close_global_aiohttp_client,
    set_global_aiohttp_client,
)
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


async def exercise(tmp_path, outcome):
    calls = []
    entered = asyncio.Event()
    release = asyncio.Event()

    async def backend(request):
        calls.append(await request.json())
        entered.set()
        if outcome == "cancel":
            await release.wait()
        else:
            await asyncio.sleep(0.02)
        if outcome == "error":
            return web.json_response({"error": "SENSITIVE_ERROR"}, status=500)
        return web.json_response({
            "id": "chatcmpl-fixture", "object": "chat.completion", "created": 0,
            "model": "fixture", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "SENSITIVE_RESPONSE"}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12,
                      "prompt_tokens_details": {"cached_tokens": 2},
                      "completion_tokens_details": {"reasoning_tokens": 3}},
        })

    app = web.Application()
    app.router.add_post('/v1/chat/completions', backend)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
    client = ServerClient(head_server_config=BaseServerConfig(host='127.0.0.1', port=0),
        global_config_dict=OmegaConf.create({'observability_enabled': True,
            'model_call_capture_dir': str(tmp_path), 'model_call_capture_metadata_only': True}))
    model = VLLMModel(config=VLLMModelConfig(host='127.0.0.1', port=0, entrypoint='', name='fixture',
        base_url=f'http://127.0.0.1:{port}/v1', api_key='NEVER_CAPTURE_APIKEY', model='fixture',
        return_token_id_information=False, uses_reasoning_parser=False,
        uses_interleaved_reasoning=False,
        sampling_overrides={'reasoning_effort': 'high', 'max_tokens': 17}), server_client=client)
    proxy = model.setup_webserver()
    rollout, operation = uuid4().hex, uuid4().hex
    body = json.dumps({'model': 'fixture', 'messages': [{'role':'user','content':'SENSITIVE_PROMPT'}],
                       'reasoning_effort': 'low', 'max_tokens': 3,
                       'metadata': {'secret': 'BODY_SECRET'}}).encode()
    delivered = False
    sent = []
    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {'type':'http.request','body':body,'more_body':False}
        await asyncio.Event().wait()
    async def send(message):
        sent.append(message)
    with model_call_context('user_simulator', operation_id=operation):
        headers = [(key.encode(), value.encode()) for key,value in metadata_headers().items()]
    scope = {'type':'http','asgi':{'version':'3.0'}, 'http_version':'1.1', 'method':'POST',
             'scheme':'http', 'path':f'/ng-rollout/{rollout}/v1/chat/completions',
             'raw_path':f'/ng-rollout/{rollout}/v1/chat/completions'.encode(),
             'query_string':b'', 'root_path':'', 'headers':[(b'content-type',b'application/json'),*headers],
             'client':('127.0.0.1',0), 'server':('127.0.0.1',0)}
    try:
        task = asyncio.create_task(proxy(scope, receive, send))
        await asyncio.wait_for(entered.wait(), 10)
        if outcome == 'cancel':
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif outcome == 'error':
            with pytest.raises(Exception):
                await task
        else:
            await task
        assert len(calls) == 1
        assert calls[0]['messages'][0]['content'] == 'SENSITIVE_PROMPT'
        assert calls[0]['reasoning_effort'] == 'high'
        assert calls[0]['max_tokens'] == 17
        store = CaptureStore(tmp_path)
        raw = store.path_for(rollout).read_text()
        for secret in ('SENSITIVE_PROMPT','SENSITIVE_RESPONSE','SENSITIVE_ERROR','BODY_SECRET','NEVER_CAPTURE_APIKEY'):
            assert secret not in raw
        records = read_model_call_records(store, rollout)
        assert len(records) == 1
        call = records[0]
        assert call.metadata_only and call.role == 'user_simulator'
        assert call.operation_id == operation
        assert call.model_call_id and call.latency_total_ms >= 0
        assert call.client_elapsed_ms >= 0
        assert call.server_queue_ms is None and call.server_service_ms is None
        assert call.request is None and call.response is None
        assert call.requested_profile['reasoning_effort'] == 'low'
        assert call.resolved_profile['reasoning_effort'] == 'high'
        if outcome == 'success':
            assert call.latency_total_ms >= call.client_elapsed_ms >= 20
            assert (call.tokens_in,call.tokens_out,call.cached_tokens,call.tokens_reasoning) == (8,4,2,3)
        else:
            assert call.tokens_in is None and call.tokens_out is None
            assert call.status_code is None
            assert call.error_category == ('CancelledError' if outcome == 'cancel' else 'ClientResponseError')
    finally:
        release.set()
        await close_global_aiohttp_client()
        await runner.cleanup()


@pytest.mark.parametrize('outcome', ['success','error','cancel'])
def test_metadata_only_real_native_model_and_http(tmp_path, outcome):
    asyncio.run(exercise(tmp_path, outcome))


def test_missing_role_profile_and_usage_stay_unknown(tmp_path):
    from nemo_gym.model_call_metadata import metadata_exchange, record_request_metadata, record_response_metadata
    with metadata_exchange({}) as exchange:
        record_request_metadata({'model':'fixture','messages':[{'content':'SENSITIVE'}]})
        record_response_metadata({'id':'response'})
    assert exchange['role'] == 'unknown'
    assert exchange['requested_profile'] == {}
    assert 'resolved_profile' not in exchange
    assert exchange['tokens_in'] is None and exchange['cached_tokens'] is None
    assert 'SENSITIVE' not in json.dumps(exchange)


def test_metadata_mode_preserves_existing_training_token_context(tmp_path):
    from nemo_gym.base_responses_api_model import _CaptureMiddleware
    from nemo_gym.token_id_capture import current_capture_context
    async def run():
        contexts = []
        async def app(scope, receive, send):
            contexts.append(current_capture_context())
            assert scope["path"] == "/v1/chat/completions"
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"SENSITIVE_RESPONSE"})
        async def receive():
            return {"type": "http.request", "body": b"SENSITIVE_PROMPT"}
        async def send(message):
            pass
        store = CaptureStore(tmp_path)
        path = "/ng-rollout/episode/training-token-capture/v1/chat/completions"
        await _CaptureMiddleware(app, store=store, model_server_name="fixture", metadata_only=True,
                                 token_capture_enabled=True)({"type": "http", "path": path, "headers": []}, receive, send)
        records = read_model_call_records(store, "episode")
        assert len(contexts) == len(records) == 1
        assert contexts[0] is not None
        assert contexts[0].model_call_id == records[0].model_call_id
        assert "SENSITIVE" not in store.path_for("episode").read_text()
    asyncio.run(run())


def test_actual_server_client_preserves_body_session_and_one_transport_call(monkeypatch):
    import nemo_gym.server_utils as module
    calls = []
    sentinel = object()
    body = {'messages': [{'content': 'SENSITIVE'}]}
    async def transport(**kwargs):
        calls.append(kwargs)
        return sentinel
    monkeypatch.setattr(module, 'request', transport)
    client = ServerClient(head_server_config=BaseServerConfig(host='127.0.0.1', port=1),
        global_config_dict=OmegaConf.create({'fixture': {'responses_api_models': {
            'openai_model': {'host': '127.0.0.1', 'port': 1234}}}}))
    async def run():
        with model_call_context('judge') as owner:
            result = await client.post('fixture', '/v1/responses', json=body,
                                       headers={'X-Session-ID': 'native-session'})
        assert result is sentinel and len(calls) == 1
        assert calls[0]['json'] is body
        assert calls[0]['url'] == 'http://127.0.0.1:1234/v1/responses'
        assert calls[0]['headers']['X-Session-ID'] == 'native-session'
        assert calls[0]['headers']['x-nemo-gym-operation-id'] == owner['operation_id']
        assert calls[0]['headers']['x-nemo-gym-call-role'] == 'judge'
        assert metadata_headers() == {}
    asyncio.run(run())


def test_raw_usage_missing_cache_survives_later_adapter_normalization():
    from nemo_gym.model_call_metadata import metadata_exchange, record_response_metadata
    with metadata_exchange({}) as exchange:
        record_response_metadata({'usage': {'prompt_tokens': 8, 'completion_tokens': 4}}, raw_client=True)
        record_response_metadata({'usage': {'input_tokens': 8, 'output_tokens': 4,
            'input_tokens_details': {'cached_tokens': 0}}})
    assert exchange['tokens_in'] == 8 and exchange['tokens_out'] == 4
    assert exchange['cached_tokens'] is None
    with metadata_exchange({}) as exchange:
        record_response_metadata({'usage': {'prompt_tokens_details': {'cached_tokens': 0}}}, raw_client=True)
    assert exchange['cached_tokens'] == 0
