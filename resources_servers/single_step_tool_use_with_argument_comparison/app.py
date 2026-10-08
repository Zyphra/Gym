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
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import Field, RootModel, ValidationError

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.rollout_correlation import current_rollout_id
from nemo_gym.server_utils import rollout_path_prefix
from resources_servers.single_step_tool_use_with_argument_comparison.common.response_utils import extract_action
from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
    ActionComparator,
    ExpectedAction,
    StepRewardCategory,
    ToolCallComparatorConfig,
)


class SingleStepToolUseArgumentComparisonResourcesServerConfig(BaseResourcesServerConfig):
    tool_call_comparator_config: ToolCallComparatorConfig
    agent_step_verifier: Optional[ResourcesServerRef] = None
    agent_step_verifier_timeout_seconds: float = Field(default=120.0, gt=0, allow_inf_nan=False)


class SingleStepToolUseArgumentComparisonRunRequest(BaseRunRequest):
    expected_action: ExpectedAction


class SingleStepToolUseArgumentComparisonVerifyRequest(
    SingleStepToolUseArgumentComparisonRunRequest, BaseVerifyRequest
):
    pass


class SingleStepToolUseArgumentComparisonVerifyResponse(BaseVerifyResponse):
    expected_action: ExpectedAction
    category: StepRewardCategory


class AgentStepComparisonVerifyResponse(SingleStepToolUseArgumentComparisonVerifyResponse):
    agent_step_verification: dict[str, Any]


class SingleStepToolUseArgumentComparisonResourcesServer(SimpleResourcesServer):
    config: SingleStepToolUseArgumentComparisonResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Additional server routes go here! e.g.:
        # app.post("/get_weather")(self.get_weather)

        return app

    async def verify(
        self, body: SingleStepToolUseArgumentComparisonVerifyRequest
    ) -> SingleStepToolUseArgumentComparisonVerifyResponse | AgentStepComparisonVerifyResponse:
        if self.config.agent_step_verifier is not None:
            return await self._verify_agent_step(body)
        actual_action = extract_action(body.response)
        if actual_action is None:
            return SingleStepToolUseArgumentComparisonVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                category=StepRewardCategory.NO_ACTION_FOUND,
            )

        action_comparator = ActionComparator(config=self.config.tool_call_comparator_config)
        result = action_comparator.compare_action(body.expected_action, actual_action)

        return SingleStepToolUseArgumentComparisonVerifyResponse(
            **body.model_dump(),
            reward=result.reward,
            category=result.category,
        )

    async def _verify_agent_step(
        self, body: SingleStepToolUseArgumentComparisonVerifyRequest
    ) -> AgentStepComparisonVerifyResponse:
        # The optional verifier's dependencies are needed only for the selected path.
        from resources_servers.conversational_tool_use_simulation.app import AgentStepVerifyResponse

        rollout_id = body.capture_rollout_id or current_rollout_id()
        payload = body.model_dump(mode="json")
        if rollout_id is not None:
            payload["_ng_rollout_id"] = rollout_id
        if body.episode_control is not None:
            payload["_ng_episode_control"] = body.episode_control
        try:
            async with asyncio.timeout(self.config.agent_step_verifier_timeout_seconds):
                reply = await call_judge(
                    self.server_client,
                    server_name=self.config.agent_step_verifier.name,
                    url_path=f"{rollout_path_prefix(rollout_id)}/verify_agent_step",
                    json=payload,
                    response_model=RootModel[dict],
                )
        except TimeoutError as error:
            raise JudgeError("Agent-step verifier exceeded its configured RPC deadline") from error
        data = reply.root
        # Inspect native failure routing before a typed model can ignore the sentinel.
        if data.get("_ng_failure_class") is not None:
            raise JudgeError(str(data.get("_ng_failure_judge_error", "Agent-step judge failed")))
        try:
            step = AgentStepVerifyResponse.model_validate(data)
        except ValidationError as error:
            raise JudgeError("Agent-step verifier returned an invalid response") from error
        if step.response != body.response or step.responses_create_params != body.responses_create_params:
            raise JudgeError("Agent-step verifier returned a different policy request or response")
        if step.reward not in (0.0, 1.0):
            raise JudgeError("Agent-step verifier returned a nonbinary reward")
        if step.verification_result.reward != step.reward or step.verification_result.generation_error is not None:
            raise JudgeError("Agent-step verifier returned inconsistent evaluation evidence")
        return AgentStepComparisonVerifyResponse(
            **body.model_dump(),
            reward=step.reward,
            category=StepRewardCategory.AGENT_STEP_VERIFIED,
            agent_step_verification=step.model_dump(mode="json"),
        )


if __name__ == "__main__":
    SingleStepToolUseArgumentComparisonResourcesServer.run_webserver()
