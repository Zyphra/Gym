# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from fastapi.testclient import TestClient

from nemo_gym.global_config import GlobalConfigDictParserConfig, get_global_config_dict
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient, close_global_aiohttp_client
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


@pytest.mark.parametrize("raw_history", [False, True])
def test_existing_explicit_raw_history_option_reaches_responses_converter(raw_history):
    model = VLLMModel(
        config=VLLMModelConfig(
            host="127.0.0.1",
            port=0,
            name="policy",
            entrypoint="app.py",
            base_url="http://127.0.0.1:1/v1",
            api_key="local-control",
            model="policy",
            return_token_id_information=False,
            uses_reasoning_parser=True,
            preserve_reasoning_in_assistant_content=raw_history,
        ),
        server_client=ServerClient.model_construct(global_config_dict={}),
    )
    params = model.get_converter().responses_to_chat_completion_create_params(
        NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "type": "reasoning",
                    "id": "reason",
                    "status": "completed",
                    "summary": [{"type": "summary_text", "text": "plan"}],
                },
                {"role": "assistant", "type": "message", "content": "answer"},
            ]
        )
    )
    if raw_history:
        assert params.messages == [{"role": "assistant", "content": "<think>plan</think>answer"}]
    else:
        assert params.messages == [{"role": "assistant", "content": "answer", "reasoning_content": "plan"}]


@pytest.mark.parametrize("api", ["chat/completions", "responses"])
@pytest.mark.parametrize(
    "reasoning,content",
    [("first plan", "literal <think>visible</think> answer"), ("", "answer"), ("unfinished plan", "")],
)
def test_parsed_reasoning_survives_native_http_roundtrip(api, reasoning, content, record_property):
    get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(dotenv_path=None, skip_load_from_cli=True)
    )
    captured = []

    class Backend(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            captured.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            payload = json.dumps(
                {
                    "id": "local-policy",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "policy",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content, "reasoning_content": reasoning},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
    thread = Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    model = VLLMModel(
        config=VLLMModelConfig(
            host="127.0.0.1",
            port=0,
            name="policy",
            entrypoint="app.py",
            base_url=f"http://127.0.0.1:{backend.server_port}/v1",
            api_key="local-control",
            model="policy",
            return_token_id_information=False,
            uses_reasoning_parser=True,
            uses_interleaved_reasoning=True,
        ),
        server_client=ServerClient.model_construct(global_config_dict={}),
    )
    opening = [{"role": "user", "content": "Continue."}]
    try:
        with TestClient(model.setup_webserver()) as client:
            try:
                key = "messages" if api == "chat/completions" else "input"
                first = {"model": "policy", key: opening}
                response = client.post("/v1/" + api, json=first)
                assert response.status_code == 200, response.text
                document = response.json()
                if api == "chat/completions":
                    assistant = document["choices"][0]["message"]
                    assert assistant["reasoning_content"] == reasoning
                    assert assistant["content"] == content
                    retained = [
                        {
                            key: value
                            for key, value in assistant.items()
                            if key in {"role", "content", "tool_calls", "reasoning_content", "reasoning"}
                            and value is not None
                        }
                    ]
                else:
                    retained = document["output"]
                    assert [
                        s["text"] for item in retained if item["type"] == "reasoning" for s in item["summary"]
                    ] == [reasoning]
                response = client.post(
                    "/v1/" + api, json={**first, key: [*opening, *retained, {"role": "user", "content": "Next."}]}
                )
                assert response.status_code == 200, response.text
                assert len(captured) == 2
                echoed = captured[1]["messages"][1]
                assert echoed["reasoning_content"] == reasoning
                assert (echoed.get("content") or "") == content
                record_property(
                    "native_reasoning_echo",
                    json.dumps({"api": api, "requests": 2, "reasoning_equal": True, "content_equal": True}),
                )
            finally:
                client.portal.call(close_global_aiohttp_client)
    finally:
        backend.shutdown()
        backend.server_close()
        thread.join()
