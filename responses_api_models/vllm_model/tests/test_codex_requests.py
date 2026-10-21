# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Real Codex CLI requests through the vLLM model server's streaming ``/v1/responses`` route.

Only the chat backend is replaced; the route, sanitizer, Responses -> Chat converter and SSE
synthesis are the server's own. See ``tests/unit_tests/test_responses_converter_codex_requests.py``
for the fixture's provenance.
"""

import json
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

import nemo_gym.server_utils
from nemo_gym import PARENT_DIR
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionMessage,
    NeMoGymChoice,
)
from nemo_gym.server_utils import ServerClient
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


CODEX_REQUESTS = json.loads(
    (PARENT_DIR / "tests/unit_tests/fixtures/codex-0.144.4-responses-requests.json").read_text()
)


def _server(monkeypatch: MonkeyPatch) -> tuple[VLLMModel, AsyncMock]:
    monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", MagicMock(return_value={}))
    server = VLLMModel(
        config=VLLMModelConfig(
            host="0.0.0.0",
            port=8081,
            base_url="http://backend.invalid/v1",
            api_key="dummy_key",  # pragma: allowlist secret
            model="policy",
            entrypoint="",
            name="",
            return_token_id_information=False,
            uses_reasoning_parser=True,
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    completion = NeMoGymChatCompletion(
        id="chatcmpl-1",
        object="chat.completion",
        created=0,
        model="policy",
        choices=[
            NeMoGymChoice(
                index=0,
                finish_reason="stop",
                message=NeMoGymChatCompletionMessage(role="assistant", content="done"),
            )
        ],
    )
    backend = MagicMock(spec=NeMoGymAsyncOpenAI)
    backend.create_chat_completion = AsyncMock(return_value=completion.model_dump())
    server._clients = [backend]
    return server, backend.create_chat_completion


def test_codex_requests_stream_to_completion_with_append_only_chat_history(monkeypatch: MonkeyPatch) -> None:
    server, create_chat_completion = _server(monkeypatch)
    client = TestClient(server.setup_webserver())

    for body in CODEX_REQUESTS:
        response = client.post("/v1/responses", json=body)

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "event: response.completed" in response.text
        assert "event: response.failed" not in response.text

    sent = [call.kwargs for call in create_chat_completion.await_args_list]
    assert [len(chat["messages"]) for chat in sent] == [3, 5, 7]
    assert sent[1]["messages"][:3] == sent[0]["messages"]
    assert sent[2]["messages"][:5] == sent[1]["messages"]
    assert all("include" not in chat for chat in sent)
    assert all(chat["model"] == "policy" for chat in sent)
