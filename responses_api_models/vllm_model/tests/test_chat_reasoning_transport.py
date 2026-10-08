# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import socket
from contextlib import asynccontextmanager
from copy import deepcopy

import pytest
import uvicorn
from aiohttp import web
from omegaconf import DictConfig
from openai import OpenAI
from pydantic import ValidationError

from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import (
    GlobalAIOHTTPAsyncClientConfig,
    ServerClient,
    close_global_aiohttp_client,
    set_global_aiohttp_client,
)
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


_MESSAGE = {
    "role": "assistant",
    "content": "Visible answer.",
    "reasoning_content": "Parsed reasoning.",
    "reasoning": "Parsed reasoning.",
    "tool_calls": [
        {
            "id": "call_fixture",
            "type": "function",
            "function": {"name": "inspect", "arguments": '{"path":"memo.txt"}'},
        }
    ],
}
_USAGE = {"prompt_tokens": 17, "completion_tokens": 11, "total_tokens": 28}


def _config(**overrides):
    values = dict(
        name="fixture",
        host="127.0.0.1",
        port=0,
        entrypoint="app.py",
        base_url="http://127.0.0.1:1/v1",
        api_key="fixture",
        model="fixture",
        return_token_id_information=False,
        uses_reasoning_parser=True,
    )
    values.update(overrides)
    return VLLMModelConfig(**values)


@pytest.mark.parametrize("overrides", [{"uses_reasoning_parser": False}, {"use_completions_api": True}])
def test_chat_reasoning_preservation_requires_parsed_chat_backend(overrides):
    values = _config().model_dump()
    values.update(overrides, preserve_chat_completion_reasoning=True)
    with pytest.raises(ValidationError, match="preserve_chat_completion_reasoning requires"):
        VLLMModelConfig.model_validate(values)


def test_chat_reasoning_preservation_is_opt_in_for_existing_configs():
    values = _config().model_dump()
    values.pop("preserve_chat_completion_reasoning")
    assert VLLMModelConfig.model_validate(values).preserve_chat_completion_reasoning is False


@asynccontextmanager
async def _local_model(preserve):
    calls = []

    async def backend(request):
        calls.append(await request.json())
        return web.json_response(
            {
                "id": "completion_fixture",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "fixture",
                "choices": [{"index": 0, "message": deepcopy(_MESSAGE), "finish_reason": "tool_calls"}],
                "usage": dict(_USAGE),
            }
        )

    backend_app = web.Application()
    backend_app.router.add_post("/v1/chat/completions", backend)
    runner = web.AppRunner(backend_app)
    await runner.setup()
    backend_listener = socket.socket()
    backend_listener.bind(("127.0.0.1", 0))
    backend_listener.listen()
    backend_port = backend_listener.getsockname()[1]
    await web.SockSite(runner, backend_listener).start()
    set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
    model = VLLMModel(
        config=_config(
            preserve_chat_completion_reasoning=preserve,
            base_url=f"http://127.0.0.1:{backend_port}/v1",
        ),
        server_client=ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=0),
            global_config_dict=DictConfig({"observability_enabled": False}),
        ),
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(model.setup_webserver(), log_level="warning"))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{port}/v1", calls
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 10)
        listener.close()
        await close_global_aiohttp_client()
        await runner.cleanup()
        assert task.done() and listener.fileno() == -1 and backend_listener.fileno() == -1
        print({"frontend_port": port, "backend_port": backend_port, "owned_sockets_closed": True})


def _sdk_calls(base_url):
    with OpenAI(base_url=base_url, api_key="fixture", timeout=10, max_retries=0) as client:
        schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
        params = {
            "model": "fixture",
            "messages": [{"role": "user", "content": "Inspect memo.txt."}],
            "tools": [{"type": "function", "function": {"name": "inspect", "parameters": schema}}],
        }
        completion = client.chat.completions.create(**params)
        chunks = list(client.chat.completions.create(**params, stream=True, stream_options={"include_usage": True}))
        response = client.responses.create(
            model="fixture",
            input="Inspect memo.txt.",
            tools=[{"type": "function", "name": "inspect", "parameters": schema, "strict": False}],
        )
    return completion, chunks, response


@pytest.mark.asyncio
@pytest.mark.parametrize("preserve", [False, True])
async def test_chat_json_sse_preserve_reasoning_while_responses_remains_legacy(preserve):
    async with _local_model(preserve) as (base_url, calls):
        completion, chunks, response = await asyncio.to_thread(_sdk_calls, base_url)
        assert len(calls) == 3
        assert calls[0]["messages"] == calls[1]["messages"] == [{"role": "user", "content": "Inspect memo.txt."}]
        assert calls[2]["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "Inspect memo.txt."}]}
        ], calls[2]
        assert calls[0]["tools"] == calls[1]["tools"] == calls[2]["tools"]
        message = completion.choices[0].message.model_dump(exclude_unset=True)
        expected = deepcopy(_MESSAGE)
        if not preserve:
            expected.pop("reasoning_content")
            expected.pop("reasoning")
            expected["content"] = "<think>Parsed reasoning.</think>\n\nVisible answer."
        for key, value in expected.items():
            assert message[key] == value
        if not preserve:
            assert not message.get("reasoning_content") and not message.get("reasoning")
        assert {key: getattr(completion.usage, key) for key in _USAGE} == _USAGE
        text = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
        assert text == expected["content"]
        for key in ("reasoning_content", "reasoning"):
            reasoning = "".join(getattr(chunk.choices[0].delta, key, "") or "" for chunk in chunks if chunk.choices)
            assert reasoning == (expected.get(key) or "")
        tool_chunks = [chunk.choices[0].delta.tool_calls for chunk in chunks if chunk.choices]
        tool = next(call for group in tool_chunks if group for call in group)
        assert tool.id == "call_fixture"
        assert tool.function.name == "inspect"
        assert tool.function.arguments == '{"path":"memo.txt"}'
        assert {key: getattr(chunks[-1].usage, key) for key in _USAGE} == _USAGE
        # Responses still consumes the historical inline-think Chat conversion,
        # even when the same server opts into parsed fields on its Chat route.
        reasoning_items = [item for item in response.output if item.type == "reasoning"]
        assert [part.text for item in reasoning_items for part in item.summary] == ["Parsed reasoning."]
        messages = [item for item in response.output if item.type == "message"]
        assert [part.text for item in messages for part in item.content] == ["Visible answer."]
        functions = [item for item in response.output if item.type == "function_call"]
        assert [(item.call_id, item.name, item.arguments) for item in functions] == [
            ("call_fixture", "inspect", '{"path":"memo.txt"}')
        ]
        assert response.usage.input_tokens == 17 and response.usage.output_tokens == 11
