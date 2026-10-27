# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nemo_gym.global_config import GlobalConfigDictParserConfig, get_global_config_dict
from nemo_gym.server_utils import ServerClient
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


def test_explicit_assistant_boundary_option_reaches_converter():
    get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(dotenv_path=None, skip_load_from_cli=True)
    )
    base = dict(
        host="127.0.0.1",
        port=19000,
        name="policy_model",
        entrypoint="app.py",
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        model="policy",
        return_token_id_information=False,
        uses_reasoning_parser=False,
    )
    for boundaries in (False, True):
        for strict in (False, True):
            config = VLLMModelConfig(
                **base,
                preserve_explicit_assistant_boundaries=boundaries,
                preserve_function_tool_strict_for_template=strict,
            )
            model = VLLMModel(config=config, server_client=ServerClient.model_construct(global_config_dict={}))
            assert model.get_converter().preserve_explicit_assistant_boundaries is boundaries
            body = model._preprocess_chat_completion_create_params(
                None,
                {
                    "messages": [{"role": "user", "content": "Find a film"}],
                    "tools": [
                        {
                            "function": {"name": "find", "parameters": {"type": "object"}, "strict": True},
                            "type": "function",
                        }
                    ],
                },
            )
            assert list(body["tools"][0]) == ["function", "type"]
            assert ("strict" in body["tools"][0]["function"]) is strict


def test_chat_template_request_survives_native_http_forwarding():
    get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(dotenv_path=None, skip_load_from_cli=True)
    )
    captured = []

    class Backend(BaseHTTPRequestHandler):
        def do_POST(self):
            captured.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            payload = json.dumps(
                {
                    "id": "chat-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "policy",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
    thread = Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    body = {
        "model": "policy",
        "messages": [{"role": "user", "content": "hello"}],
        "chat_template_kwargs": {"enable_thinking": True, "truncate_history_thinking": False},
    }
    if fixture := os.environ.get("HERMES_CHAT_REPLAY_CAPTURE"):
        records = [json.loads(line) for line in Path(fixture).read_text().splitlines()]
        body = next(record["request"] for record in records if "chat_template_kwargs" in record.get("request", {}))
    server = VLLMModel(
        config=VLLMModelConfig(
            host="127.0.0.1",
            port=19000,
            name="policy_model",
            entrypoint="app.py",
            base_url=f"http://127.0.0.1:{backend.server_port}/v1",
            api_key="EMPTY",
            model=body["model"],
            return_token_id_information=False,
            uses_reasoning_parser=False,
            uses_interleaved_reasoning=False,
            chat_template_kwargs={"enable_thinking": False},
        ),
        server_client=ServerClient.model_construct(global_config_dict={}),
    )
    app = FastAPI()
    server.setup_session_middleware(app)
    app.post("/v1/chat/completions")(server.chat_completions)
    try:
        with TestClient(app) as client:
            response = client.post("/v1/chat/completions", json=body)
            assert response.status_code == 200, response.text
            assert len(captured) == 1
            assert captured[0]["chat_template_kwargs"] == body["chat_template_kwargs"]
            assert captured[0]["messages"] == body["messages"]
            rejected = client.post("/v1/chat/completions", json={**body, "unknown_option": True})
            assert rejected.status_code == 422
            assert len(captured) == 1
    finally:
        backend.shutdown()
        backend.server_close()
        thread.join()


def test_completion_render_preserves_explicit_template_options():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"unknown": 0}, unk_token="unknown")),
        chat_template="{{ enable_thinking }}|{{ truncate_history_thinking }}|{{ messages[0]['content'] }}",
    )
    server = VLLMModel.model_construct(
        config=VLLMModelConfig(
            host="127.0.0.1",
            port=19000,
            name="policy_model",
            entrypoint="app.py",
            base_url="http://127.0.0.1:19001/v1",
            api_key="EMPTY",
            model="policy",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            chat_template_kwargs={"enable_thinking": False, "truncate_history_thinking": True},
        ),
        server_client=ServerClient.model_construct(global_config_dict={}),
    )
    server._chat_template_tokenizer = tokenizer
    body = {
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": {"chat_template_kwargs": '{"enable_thinking": false}'},
        "chat_template_kwargs": {"enable_thinking": True, "truncate_history_thinking": False},
    }
    assert server._render_messages_via_chat_template(body) == "True|False|hello"
    del body["chat_template_kwargs"]
    assert server._render_messages_via_chat_template(body) == "False|True|hello"


def test_native_responses_honors_per_request_thinking_metadata(monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock

    from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming

    get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(dotenv_path=None, skip_load_from_cli=True)
    )
    payload = {
        "id": "resp-judge",
        "created_at": 0.0,
        "model": "judge",
        "object": "response",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "id": "msg-judge",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "EQUIVALENT", "annotations": []}],
            }
        ],
    }
    backend = AsyncMock()
    backend.create_response.return_value = payload
    monkeypatch.setattr(VLLMModel, "_resolve_client", lambda self, request: backend)
    server = VLLMModel(
        config=VLLMModelConfig(
            host="127.0.0.1",
            port=19000,
            name="judge_model",
            entrypoint="app.py",
            base_url="http://127.0.0.1:19001/v1",
            api_key="EMPTY",
            model="judge",
            is_responses_native=True,
            return_token_id_information=False,
            uses_reasoning_parser=True,
            uses_interleaved_reasoning=False,
            chat_template_kwargs={"enable_thinking": True, "preserve_other_option": True},
        ),
        server_client=ServerClient.model_construct(global_config_dict={}),
    )

    async def exercise():
        for thinking in (False, True, None):
            body = NeMoGymResponseCreateParamsNonStreaming(
                input="Compare answers",
                max_output_tokens=4096,
                metadata={}
                if thinking is None
                else {"chat_template_kwargs": json.dumps({"enable_thinking": thinking})},
            )
            response = await server._responses_native(None, body)
            assert isinstance(response, NeMoGymResponse)
            forwarded = backend.create_response.call_args.kwargs
            assert forwarded["chat_template_kwargs"] == {
                "enable_thinking": True if thinking is None else thinking,
                "preserve_other_option": True,
            }
            assert forwarded["max_output_tokens"] == 4096
        assert server.config.chat_template_kwargs == {
            "enable_thinking": True,
            "preserve_other_option": True,
        }

    asyncio.run(exercise())
