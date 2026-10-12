# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import aiohttp
import pytest
from aiohttp import web
from omegaconf import OmegaConf

from nemo_gym import server_utils
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_input,instructions",
    [
        (
            [{"role": "system", "content": "official rules"}, {"role": "user", "content": "question/gold/candidate"}],
            "complete",
        ),
        (
            [
                {"role": "system", "content": [{"type": "input_text", "text": "official rules"}]},
                {"role": "developer", "content": "additional rules"},
                {"role": "user", "content": "question/gold/candidate"},
            ],
            "complete",
        ),
        ("question/gold/candidate", "complete"),
        (
            [{"role": "system", "content": "official rules"}, {"role": "user", "content": "question/gold/candidate"}],
            None,
        ),
    ],
)
async def test_direct_responses_folds_like_chat_converter(response_input, instructions, monkeypatch):
    received = []

    async def upstream(request):
        payload = await request.json()
        received.append(payload)
        # A strict model template rejects a second system message, including one inserted by instructions.
        messages = payload["input"]
        if payload.get("instructions") and any(m.get("role") == "system" for m in messages):
            return web.json_response({"error": "System message must be at the beginning."}, status=400)
        assert [m["role"] for m in messages] == ["system", "user"]
        return web.json_response(
            {
                "id": "resp_test",
                "created_at": 0,
                "model": "test",
                "object": "response",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            }
        )

    app = web.Application()
    app.router.add_post("/v1/responses", upstream)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        host, port = runner.addresses[0]
        client = ServerClient(
            head_server_config=BaseServerConfig(host=host, port=port), global_config_dict=OmegaConf.create({})
        )
        server = SimpleModelServer(
            config=SimpleModelServerConfig(
                host=host,
                port=port,
                name="test",
                entrypoint="app.py",
                openai_base_url=f"http://{host}:{port}/v1",
                openai_api_key="test",  # pragma: allowlist secret
                openai_model="test",
                extra_body={"max_output_tokens": 128},
            ),
            server_client=client,
        )
        params = {"input": deepcopy(response_input), "temperature": 1.0, "top_p": 0.95}
        if instructions is not None:
            params["instructions"] = instructions
        body = NeMoGymResponseCreateParamsNonStreaming.model_validate(params)
        before = body.model_dump(mode="json", exclude_unset=True)
        expected = ResponsesConverter(return_token_id_information=False).responses_to_chat_completion_create_params(
            body.model_copy(deep=True)
        )
        async with aiohttp.ClientSession() as session:
            monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", session)
            result = await server.responses(body)
        assert result.id == "resp_test"
        assert body.model_dump(mode="json", exclude_unset=True) == before
        assert len(received) == 1
        forwarded = received[0]
        assert "instructions" not in forwarded
        assert forwarded["input"][0] == expected.messages[0]
        assert forwarded["input"][-1]["content"] == "question/gold/candidate"
        assert forwarded["temperature"] == 1.0 and forwarded["top_p"] == 0.95
        assert forwarded["max_output_tokens"] == 128
        if instructions is None:
            assert forwarded["input"] == response_input
    finally:
        await runner.cleanup()


def test_folding_preserves_responses_only_items():
    from nemo_gym.response_instructions import fold_instructions_into_messages

    items = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
        {"type": "reasoning", "id": "reasoning_1", "summary": []},
        {"type": "function_call", "call_id": "call_1", "name": "tool", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "result"},
    ]
    original = deepcopy(items)
    folded = fold_instructions_into_messages("instructions", items)
    assert folded[0] == {"role": "system", "content": "instructions\n\nrules"}
    assert folded[1:] == original[1:]
    assert items == original
    assert fold_instructions_into_messages(None, items) == original
