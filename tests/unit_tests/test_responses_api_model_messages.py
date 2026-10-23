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
"""Tests for the default ``/v1/messages`` route on ``SimpleResponsesAPIModel``.

Every Gym model server inherits an Anthropic Messages endpoint that maps Messages <-> Responses
around the server's own ``responses()``. These tests use minimal fake servers to exercise the
default mapping for both ``responses()`` signatures (with and without a leading ``request``).
"""

from time import time
from unittest.mock import MagicMock
from uuid import uuid4

from fastapi import Body, Request
from fastapi.testclient import TestClient

from nemo_gym.base_responses_api_model import BaseResponsesAPIModelConfig, SimpleResponsesAPIModel
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient


def _build_response(text: str, model: str = "downstream-model") -> NeMoGymResponse:
    return NeMoGymResponse(
        id=f"resp_{uuid4().hex}",
        created_at=int(time()),
        model=model,
        object="response",
        output=[
            {
                "type": "message",
                "id": f"msg_{uuid4().hex}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    )


class _BodyOnlyModel(SimpleResponsesAPIModel):
    """A server whose responses() takes only `body` (like openai_model)."""

    config: BaseResponsesAPIModelConfig
    last_params: object = None
    model_config = {"arbitrary_types_allowed": True}

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        object.__setattr__(self, "last_params", body)
        return _build_response("hi from body-only")

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        raise NotImplementedError


class _RequestAwareModel(SimpleResponsesAPIModel):
    """A server whose responses() also takes `request` (like vllm_model / azure)."""

    config: BaseResponsesAPIModelConfig
    saw_request: bool = False
    model_config = {"arbitrary_types_allowed": True}

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        object.__setattr__(self, "saw_request", isinstance(request, Request))
        return _build_response("hi from request-aware")

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        raise NotImplementedError


def _config(**overrides) -> BaseResponsesAPIModelConfig:
    return BaseResponsesAPIModelConfig(host="0.0.0.0", port=8099, entrypoint="", name="", **overrides)


def _client(model_cls, **overrides) -> TestClient:
    server = model_cls(config=_config(**overrides), server_client=MagicMock(spec=ServerClient, global_config_dict={}))
    return TestClient(server.setup_webserver()), server


class TestDefaultMessagesRoute:
    def test_messages_route_registered_alongside_openai_routes(self) -> None:
        server = _BodyOnlyModel(config=_config(), server_client=MagicMock(spec=ServerClient, global_config_dict={}))
        paths = {route.path for route in server.setup_webserver().routes}
        assert {"/v1/messages", "/v1/responses", "/v1/chat/completions"} <= paths

    def test_body_only_responses_signature(self) -> None:
        client, server = _client(_BodyOnlyModel)
        resp = client.post(
            "/v1/messages",
            json={"model": "claude-x", "max_tokens": 32, "messages": [{"role": "user", "content": "hello"}]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["role"] == "assistant"
        assert data["content"] == [{"type": "text", "text": "hi from body-only"}]
        assert data["model"] == "claude-x"  # request model echoed back
        # the inbound Anthropic request was translated to Responses params before delegating
        assert server.last_params.input[0].content == "hello"
        assert server.last_params.max_output_tokens == 32

    def test_request_aware_responses_signature(self) -> None:
        client, server = _client(_RequestAwareModel)
        resp = client.post(
            "/v1/messages",
            json={"model": "claude-x", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert resp.json()["content"] == [{"type": "text", "text": "hi from request-aware"}]
        assert server.saw_request is True  # request was forwarded to responses()

    def test_streaming_returns_anthropic_sse(self) -> None:
        client, _ = _client(_BodyOnlyModel)
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-x",
                "max_tokens": 8,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.text
        assert "event: message_start" in body
        assert "event: content_block_delta" in body
        assert "event: message_stop" in body


class TestClientIdentityDrop:
    IDENTITY = "You are a Claude agent, built on Anthropic's Claude Agent SDK."

    def _post(self, client, system) -> None:
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-x",
                "max_tokens": 8,
                "system": system,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200

    def test_off_by_default(self) -> None:
        client, server = _client(_BodyOnlyModel)
        self._post(client, [{"type": "text", "text": self.IDENTITY}, {"type": "text", "text": "You are ZAYA."}])
        assert server.last_params.instructions == self.IDENTITY + "\nYou are ZAYA."

    def test_drops_only_exact_identity_blocks(self) -> None:
        client, server = _client(_BodyOnlyModel, anthropic_drop_client_identity=True)
        system = [
            {"type": "text", "text": self.IDENTITY, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "You are ZAYA, a coding agent developed by Zyphra."},
            {"type": "text", "text": self.IDENTITY + " Extra."},
        ]
        self._post(client, system)
        assert (
            server.last_params.instructions
            == "You are ZAYA, a coding agent developed by Zyphra.\n" + self.IDENTITY + " Extra."
        )

    def test_every_cli_identity_variant(self) -> None:
        from nemo_gym.base_responses_api_model import CLAUDE_CODE_IDENTITY_BLOCKS, drop_client_identity

        assert len(CLAUDE_CODE_IDENTITY_BLOCKS) == 3
        for text in CLAUDE_CODE_IDENTITY_BLOCKS:
            body, dropped = drop_client_identity(
                {"system": [{"type": "text", "text": text}, {"type": "text", "text": "task"}]}
            )
            assert dropped == 1 and body["system"] == [{"type": "text", "text": "task"}]
            assert drop_client_identity({"system": text}) == ({"system": ""}, 1)
        unchanged = {"system": [{"type": "text", "text": "task"}]}
        assert drop_client_identity(unchanged) == (unchanged, 0)
        assert drop_client_identity({"messages": []}) == ({"messages": []}, 0)
