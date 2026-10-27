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

"""Real stock FastAPI serialization and native HTTP delegation with bounded fixture servers."""

import asyncio
import json
import socket
from contextlib import AsyncExitStack, asynccontextmanager
from copy import deepcopy

import aiohttp
import pytest
import uvicorn
from fastapi import FastAPI, Request
from omegaconf import OmegaConf

import nemo_gym.server_utils as server_utils
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.rollout_correlation import rollout_context
from nemo_gym.server_utils import ServerClient
from resources_servers.conversational_tool_use_simulation.app import (
    ConversationalToolUseSimulationConfig,
    ConversationalToolUseSimulationServer,
)
from resources_servers.single_step_tool_use_with_argument_comparison.app import (
    SingleStepToolUseArgumentComparisonResourcesServer,
    SingleStepToolUseArgumentComparisonResourcesServerConfig,
    SingleStepToolUseArgumentComparisonVerifyRequest,
)
from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
    StepRewardCategory,
)


@asynccontextmanager
async def serve_fastapi(app):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            lifespan="off",
            access_log=False,
            log_level="error",
            timeout_graceful_shutdown=1,
        )
    )
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("Fixture server exited before startup")
                await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=5)
        finally:
            if not task.done():
                task.cancel()
            listener.close()


def native_response(text, status="completed"):
    return NeMoGymResponse.model_validate(
        dict(
            id="native-response",
            created_at=0,
            model="fixture",
            object="response",
            status=status,
            output=[
                dict(
                    type="message",
                    id="native-message",
                    role="assistant",
                    status="completed",
                    content=[dict(type="output_text", text=text, annotations=[])],
                )
            ],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )
    )


def original_payload(*, explicit_id=None):
    value = dict(
        responses_create_params=dict(
            input=[
                dict(role="system", content="Preserve policy."),
                dict(role="user", content="Clarify what is missing."),
            ]
        ),
        response=native_response("Which detail should I clarify?").model_dump(mode="json"),
        expected_action=dict(type="function_call", name="different_gold", arguments="{}"),
        _ng_episode_control="fixture-control",
    )
    if explicit_id is not None:
        value["_ng_rollout_id"] = explicit_id
    return value


@asynccontextmanager
async def production_pipeline(monkeypatch, *, success=True, judge_status="completed"):
    inner_inputs, judge_inputs = [], []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
        monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", session)
        client = ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=1),
            global_config_dict=OmegaConf.create({}),
        )
        inner = ConversationalToolUseSimulationServer(
            config=ConversationalToolUseSimulationConfig(
                host="127.0.0.1",
                port=1,
                name="inner",
                entrypoint="app.py",
                enable_agent_step_verification=True,
                judge_model_server={"type": "responses_api_models", "name": "judge"},
                judge_responses_create_params={
                    "input": [],
                    "temperature": 0,
                    "max_output_tokens": 4000,
                },
                judge_provider_attempts=1,
                agent_step_verification_timeout_seconds=5,
            ),
            server_client=client,
        )
        outer = SingleStepToolUseArgumentComparisonResourcesServer(
            config=SingleStepToolUseArgumentComparisonResourcesServerConfig(
                host="127.0.0.1",
                port=1,
                name="outer",
                entrypoint="app.py",
                tool_call_comparator_config={"word_count_similarity_threshold": 0.1},
                agent_step_verifier={"type": "resources_servers", "name": "inner"},
                agent_step_verifier_timeout_seconds=8,
            ),
            server_client=client,
        )
        inner_app, outer_app = inner.setup_webserver(), outer.setup_webserver()

        @inner_app.middleware("http")
        async def observe_inner(request: Request, call_next):
            inner_inputs.append(dict(path=request.url.path, body=await request.json()))
            return await call_next(request)

        judge_app = FastAPI()
        judge_reply = native_response(
            '{"success":true,"explanation":"fixture decision"}'
            if success
            else '{"success":false,"explanation":"fixture decision"}',
            status=judge_status,
        )

        @judge_app.post("/v1/responses")
        @judge_app.post("/ng-rollout/{rollout_id}/v1/responses")
        async def fixture_judge(request: Request, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
            judge_inputs.append(dict(path=request.url.path, body=body.model_dump(mode="json")))
            return judge_reply

        async with AsyncExitStack() as stack:
            urls = {
                "judge": await stack.enter_async_context(serve_fastapi(judge_app)),
                "inner": await stack.enter_async_context(serve_fastapi(inner_app)),
                "outer": await stack.enter_async_context(serve_fastapi(outer_app)),
            }
            client._server_base_urls.update(urls)
            yield client, outer, inner, inner_inputs, judge_inputs, judge_reply


async def post_outer(client, payload, path="/verify"):
    response = await client.post(server_name="outer", url_path=path, json=payload)
    assert response.status == 200, await response.text()
    return await response.json()


def assert_episode_control_forwarded(original, observed):
    body = SingleStepToolUseArgumentComparisonVerifyRequest.model_validate(original)
    if "episode_control" in type(body).model_fields:
        assert body.episode_control == original["_ng_episode_control"] == observed["_ng_episode_control"]
    else:
        # Clean upstream has no optional episode-control contract. The wrapper's
        # bounded-lifecycle composition adds it and must preserve its exact value.
        assert "_ng_episode_control" not in observed


@pytest.mark.asyncio
@pytest.mark.parametrize("success", [True, False])
async def test_real_fastapi_union_preserves_receipt_reward_and_explicit_controls(monkeypatch, success):
    async with production_pipeline(monkeypatch, success=success) as (
        client,
        outer,
        inner,
        observed,
        calls,
        reply,
    ):
        payload = original_payload(explicit_id="explicit-id")
        untouched = deepcopy(payload)
        result = await post_outer(client, payload, path="/ng-rollout/different-path-id/verify")
        assert result["reward"] == int(success) and result["category"] == StepRewardCategory.AGENT_STEP_VERIFIED
        assert result["response"] == payload["response"]
        assert result["expected_action"] == payload["expected_action"]
        nested = result["agent_step_verification"]
        assert nested["verification_result"]["reward"] == int(success)
        assert nested["verification_result"]["responses"] == [reply.model_dump(mode="json")]
        assert nested["judge_settings"]["rollout_id"] == "explicit-id"
        assert len(observed) == len(calls) == 1
        assert observed[0]["path"] == "/ng-rollout/explicit-id/verify_agent_step"
        assert observed[0]["body"]["_ng_rollout_id"] == "explicit-id"
        assert_episode_control_forwarded(payload, observed[0]["body"])
        assert calls[0]["path"] == "/ng-rollout/explicit-id/v1/responses"
        assert nested["judge_request"] == calls[0]["body"]
        assert nested["judge_request"]["temperature"] == 0 and nested["judge_request"]["max_output_tokens"] == 4000
        assert nested["responses_create_params"] == observed[0]["body"]["responses_create_params"]
        assert payload == untouched and inner.session_id_to_state == {}


@pytest.mark.asyncio
async def test_real_fastapi_rollout_path_supplies_context_fallback_without_mutating_input(
    monkeypatch,
):
    async with production_pipeline(monkeypatch) as (
        client,
        outer,
        inner,
        observed,
        calls,
        reply,
    ):
        payload = original_payload()
        untouched = deepcopy(payload)
        result = await post_outer(client, payload, path="/ng-rollout/context-id/verify")
        assert result["reward"] == 1
        assert observed[0]["body"]["_ng_rollout_id"] == "context-id"
        assert_episode_control_forwarded(payload, observed[0]["body"])
        assert observed[0]["path"] == "/ng-rollout/context-id/verify_agent_step"
        assert calls[0]["path"] == "/ng-rollout/context-id/v1/responses"
        assert result["agent_step_verification"]["judge_settings"]["rollout_id"] == "context-id"
        assert payload == untouched and "_ng_rollout_id" not in payload
        assert inner.session_id_to_state == {} and len(calls) == 1


@pytest.mark.asyncio
async def test_original_native_typed_request_is_unchanged_by_delegation_with_context_fallback(
    monkeypatch,
):
    async with production_pipeline(monkeypatch) as (
        client,
        outer,
        inner,
        observed,
        calls,
        reply,
    ):
        body = SingleStepToolUseArgumentComparisonVerifyRequest.model_validate(original_payload())
        untouched = body.model_dump(mode="json")
        controls = (body.capture_rollout_id, getattr(body, "episode_control", None))
        with rollout_context("typed-context-id"):
            result = await outer.verify(body)
        assert result.reward == 1
        assert body.model_dump(mode="json") == untouched
        assert (body.capture_rollout_id, getattr(body, "episode_control", None)) == controls
        assert observed[0]["body"]["_ng_rollout_id"] == "typed-context-id"
        assert observed[0]["body"].get("_ng_episode_control") == controls[1]
        assert result.agent_step_verification["judge_settings"]["rollout_id"] == "typed-context-id"
        assert len(calls) == 1 and inner.session_id_to_state == {}


@pytest.mark.asyncio
async def test_real_fastapi_failed_envelope_survives_outer_failsafe_as_native_sentinel(
    monkeypatch,
):
    async with production_pipeline(monkeypatch, judge_status="incomplete") as (
        client,
        outer,
        inner,
        observed,
        calls,
        reply,
    ):
        payload = original_payload(explicit_id="failure-id")
        result = await post_outer(client, payload)
        assert result["reward"] == 0 and result["_ng_failure_class"] == "judge_failed"
        assert result["response"] == payload["response"] and result["expected_action"] == payload["expected_action"]
        failure = json.loads(result["_ng_failure_judge_error"])
        assert failure["judge_response"] == reply.model_dump(mode="json")
        assert failure["judge_request"] == calls[0]["body"]
        assert failure["judge_settings"]["rollout_id"] == "failure-id"
        assert_episode_control_forwarded(payload, observed[0]["body"])
        assert len(calls) == 1 and inner.session_id_to_state == {}
