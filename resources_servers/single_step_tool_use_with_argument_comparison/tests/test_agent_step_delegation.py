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
"""Real stock objects and native HTTP transport preserve step evidence and judge failure routing."""

import asyncio
import json

import aiohttp
import pytest
from aiohttp import web
from omegaconf import OmegaConf

import nemo_gym.server_utils as server_utils
from nemo_gym.config_types import BaseServerConfig, ModelServerRef, ResourcesServerRef
from nemo_gym.judge import judge_failsafe
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.server_utils import ServerClient
from resources_servers.single_step_tool_use_with_argument_comparison.app import (
    SingleStepToolUseArgumentComparisonResourcesServer,
    SingleStepToolUseArgumentComparisonResourcesServerConfig,
    SingleStepToolUseArgumentComparisonVerifyRequest,
)
from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
    StepRewardCategory,
    ToolCallComparatorConfig,
)
from responses_api_agents.tool_simulation_agent.app import (
    ToolSimulationAgent,
    ToolSimulationAgentConfig,
    ToolSimulationAgentRunRequest,
)


def policy_response():
    items = ResponsesConverter(
        return_token_id_information=False, uses_reasoning_parser=True
    ).postprocess_assistant_message_dict({"role": "assistant", "content": "A useful clarification."})
    return NeMoGymResponse(
        id="resp-policy",
        created_at=0,
        model="policy",
        object="response",
        output=items,
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        status="completed",
    )


async def run_case(monkeypatch, mode, *, enabled=True, through_agent=True):
    observed = []
    policy = policy_response()
    app = web.Application()
    async with aiohttp.ClientSession() as session:
        monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", session)
        runner = web.AppRunner(app)
        # Server objects use real native ServerClient instances; the fixture only owns HTTP replies.
        client = ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=1), global_config_dict=OmegaConf.create({})
        )
        cfg = SingleStepToolUseArgumentComparisonResourcesServerConfig(
            host="127.0.0.1",
            port=1,
            entrypoint="app.py",
            name="outer",
            tool_call_comparator_config=ToolCallComparatorConfig(word_count_similarity_threshold=0.1),
            agent_step_verifier=ResourcesServerRef(type="resources_servers", name="verifier") if enabled else None,
            agent_step_verifier_timeout_seconds=0.02 if mode == "timeout" else 2.0,
        )
        outer = SingleStepToolUseArgumentComparisonResourcesServer(config=cfg, server_client=client)

        async def model_endpoint(request):
            return web.json_response(policy.model_dump(mode="json"))

        async def verifier_endpoint(request):
            body = await request.json()
            observed.append(body)
            if mode == "http_error":
                return web.json_response({"error": "fixture unavailable"}, status=503)
            if mode == "timeout":
                await asyncio.sleep(0.1)
            if mode == "sentinel":
                return web.json_response(
                    body
                    | {
                        "reward": 0,
                        "_ng_failure_class": "judge_failed",
                        "_ng_failure_judge_error": "fixture judge error",
                    }
                )
            reward = 0 if mode == "valid_zero" else 1
            result = body | {
                "reward": reward,
                "verification_result": {"reward": reward, "explanation": "Fixture decision"},
                "judge_request": None,
            }
            if mode == "missing_evidence":
                result.pop("verification_result")
            elif mode == "different_response":
                result["response"] = policy.model_copy(update={"id": "different-policy-response"}).model_dump(
                    mode="json"
                )
            elif mode == "different_context":
                result["responses_create_params"] = {"input": [{"role": "user", "content": "substituted context"}]}
            elif mode == "inconsistent":
                result["verification_result"]["reward"] = 0
            elif mode == "generation_error":
                result["verification_result"]["generation_error"] = "unparseable fixture verdict"
            elif mode == "nonbinary":
                result["reward"] = 0.5
            return web.json_response(result)

        async def outer_endpoint(request):
            body = SingleStepToolUseArgumentComparisonVerifyRequest.model_validate(await request.json())
            result = await judge_failsafe(outer.verify)(body=body)
            if hasattr(result, "body"):
                return web.Response(body=result.body, status=result.status_code, content_type="application/json")
            return web.json_response(result.model_dump(mode="json"))

        app.router.add_post("/v1/responses", model_endpoint)
        app.router.add_post("/verify_agent_step", verifier_endpoint)
        app.router.add_post("/verify", outer_endpoint)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        url = f"http://127.0.0.1:{runner.addresses[0][1]}"
        client._server_base_urls.update({name: url for name in ("agent", "policy", "outer", "verifier")})
        row = ToolSimulationAgentRunRequest(
            responses_create_params={
                "input": [
                    {"role": "system", "content": "Only grounded statements."},
                    {"role": "user", "content": "Please clarify the missing credential."},
                ]
            },
            expected_action={"type": "function_call", "name": "transfer", "arguments": "{}"},
        )
        try:
            if through_agent:
                agent = ToolSimulationAgent(
                    config=ToolSimulationAgentConfig(
                        host="127.0.0.1",
                        port=1,
                        entrypoint="app.py",
                        name="agent",
                        resources_server=ResourcesServerRef(type="resources_servers", name="outer"),
                        model_server=ModelServerRef(type="responses_api_models", name="policy"),
                    ),
                    server_client=client,
                )
                result = (await agent.run(row)).model_dump(mode="json")
            else:
                body = SingleStepToolUseArgumentComparisonVerifyRequest.model_validate(
                    row.model_dump() | {"response": policy}
                )
                typed = await judge_failsafe(outer.verify)(body=body)
                result = json.loads(typed.body) if hasattr(typed, "body") else typed.model_dump(mode="json")
            return result, observed, policy, row
        finally:
            await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["valid_one", "valid_zero"])
async def test_stock_agent_preserves_complete_step_evidence(monkeypatch, mode):
    result, observed, policy, row = await run_case(monkeypatch, mode)
    assert len(observed) == 1
    assert result["reward"] == (0 if mode == "valid_zero" else 1)
    assert result["category"] == StepRewardCategory.AGENT_STEP_VERIFIED
    assert result["response"] == policy.model_dump(mode="json")
    assert observed[0]["responses_create_params"] == row.responses_create_params.model_dump(mode="json")
    assert result["agent_step_verification"]["verification_result"]["reward"] == result["reward"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "sentinel",
        "missing_evidence",
        "different_response",
        "different_context",
        "inconsistent",
        "generation_error",
        "nonbinary",
        "http_error",
        "timeout",
    ],
)
async def test_stock_agent_cannot_turn_verifier_failure_into_policy_zero(monkeypatch, mode):
    result, observed, policy, row = await run_case(monkeypatch, mode)
    assert len(observed) == 1
    assert result["reward"] == 0
    assert result["_ng_failure_class"] == "judge_failed"
    assert result["_ng_failure_judge_error"]
    assert result["response"] == policy.model_dump(mode="json")
    assert result["responses_create_params"] == row.responses_create_params.model_dump(mode="json")
    assert result["expected_action"] == row.expected_action


@pytest.mark.asyncio
async def test_disabled_delegation_keeps_stock_verdict_and_wire_shape(monkeypatch):
    result, observed, _, _ = await run_case(monkeypatch, "valid_one", enabled=False)
    assert observed == []
    assert result["reward"] == 0
    assert result["category"] == StepRewardCategory.NO_EXPECTED_TOOL_CALL
    assert "agent_step_verification" not in result
