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
"""mini-swe-agent run natively inside the task container.

Upstream ``DefaultAgent`` with its own SWE-bench templates, its one ``bash`` tool and
``LocalEnvironment``. The model calls the rollout's chat-completions endpoint once
per turn: no client retries, and a turn that fails to parse stays in the history,
so every request extends the previous one.
"""

import asyncio
import logging
import os
from asyncio import Semaphore
from collections.abc import Mapping
from pathlib import Path
from time import time
from typing import Any, Optional
from uuid import uuid4

import yaml
from fastapi import Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


LOG = logging.getLogger(__name__)
# Importing litellm otherwise fetches its model cost map from the network and retries
# in a background thread whose logging can race the import itself (KeyError: 'litellm').
# The sandbox has no network: load the bundled map and never start that thread.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
# OpenAI error codes the episode endpoint answers once an episode has ended.
_ENDED_CODES = {"context_length_exceeded": "context_exhausted", "episode_terminated": "episode_terminated"}


class MiniSweAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    model: Optional[str] = None
    concurrency: int = 32
    # A config file shipped with mini-swe-agent (minisweagent/config/benchmarks/...).
    benchmark_config: str = "swebench.yaml"
    step_limit: int = Field(default=250, ge=1)
    command_timeout: int = Field(default=180, gt=0)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    # Replaces the benchmark config's system template when set.
    system_prompt: Optional[str] = None
    api_key: Optional[str] = None
    api_timeout: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)


class MiniSweAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class MiniSweAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    exit_status: str = ""


def benchmark_settings(name: str) -> dict:
    """The agent, environment and model sections of a mini-swe-agent benchmark config."""
    import minisweagent

    path = Path(minisweagent.__file__).parent / "config" / "benchmarks" / name
    return yaml.safe_load(path.read_text())


def endpoint_model(*, base_url: str, model_name: str, api_key: str, sampling: dict, settings: dict, timeout):
    """A mini-swe-agent model that makes exactly one chat-completions call per turn."""
    from minisweagent.exceptions import FormatError, LimitsExceeded
    from minisweagent.models.litellm_model import LitellmModel
    from minisweagent.models.utils.actions_toolcall import BASH_TOOL
    from openai import BadRequestError, OpenAI

    class EndpointModel(LitellmModel):
        def __init__(self):
            super().__init__(model_name=model_name, cost_tracking="ignore_errors", **settings)
            self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=timeout)

        @staticmethod
        def _ended(reason: str) -> LimitsExceeded:
            return LimitsExceeded(
                {
                    "role": "exit",
                    "content": reason,
                    "extra": {"exit_status": "EpisodeEnded", "stop_reason": reason, "submission": ""},
                }
            )

        def query(self, messages: list[dict], **kwargs) -> dict:
            history = [{key: value for key, value in message.items() if key != "extra"} for message in messages]
            try:
                response = self.client.chat.completions.create(
                    model=model_name, messages=history, tools=[BASH_TOOL], **sampling, **kwargs
                )
            except BadRequestError as error:
                code = (error.body or {}).get("code") if isinstance(error.body, Mapping) else None
                if code in _ENDED_CODES:
                    raise self._ended(_ENDED_CODES[code]) from error
                raise
            choice = response.choices[0]
            # The endpoint's terminal turn: the episode ended before this one was sampled.
            stop_reason = (choice.model_extra or {}).get("stop_reason")
            if isinstance(stop_reason, str):
                raise self._ended(stop_reason)
            # The turn exactly as served, so the next request echoes it unchanged.
            message = choice.message.model_dump(exclude_none=True)
            extra = {"response": response.model_dump(mode="json"), "cost": 0.0, "timestamp": time()}
            try:
                actions = self._parse_actions(response)
            except FormatError as error:
                # Upstream drops the unparsable turn; keeping it keeps the history append-only.
                error.messages = (message | {"extra": {**extra, "actions": []}}, *error.messages)
                raise
            return message | {"extra": {**extra, "actions": actions}}

    return EndpointModel()


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "") for part in content
        )
    return content or ""


def _task(input_items) -> tuple[str, Optional[str]]:
    """The last user message is the task; a leading system message is kept separately."""
    system, task = None, ""
    for item in input_items:
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
        if role == "system" and system is None:
            system = _text(content)
        elif role == "user":
            task = _text(content)
    return task, system


def output_items(messages: list[dict]) -> list:
    """Responses output items for the agent's turns after the system and task messages."""
    items = []
    for message in messages[2:]:
        role = message.get("role")
        if role == "assistant":
            items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg-{len(items)}",
                    content=[
                        NeMoGymResponseOutputText(
                            type="output_text", text=_text(message.get("content")), annotations=[]
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                items.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=function.get("arguments", ""),
                        call_id=call.get("id", ""),
                        name=function.get("name", ""),
                        type="function_call",
                        id=call.get("id"),
                        status="completed",
                    )
                )
        elif role == "tool":
            items.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=message.get("tool_call_id", ""),
                    output=_text(message.get("content")),
                    status="completed",
                )
            )
    return items


class MiniSweAgent(SimpleResponsesAPIAgent):
    config: MiniSweAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    def _model_name(self) -> str:
        return self.config.model or str(self.config.model_server.name)

    def _agent(self, base_url: str, system: Optional[str], cwd: str):
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.local import LocalEnvironment

        settings = benchmark_settings(self.config.benchmark_config)
        agent_settings = {
            **settings["agent"],
            "step_limit": self.config.step_limit,
            # The policy endpoint has no price: steps and the episode end it.
            "cost_limit": 0,
        }
        if self.config.system_prompt or system:
            agent_settings["system_template"] = self.config.system_prompt or system
        environment = dict(settings.get("environment") or {})
        env = dict(environment.get("env") or {})
        if env.get("BASH_ENV") and not os.path.isfile(env["BASH_ENV"]):
            del env["BASH_ENV"]
        sampling = {
            key: value
            for key, value in (("temperature", self.config.temperature), ("max_tokens", self.config.max_tokens))
            if value is not None
        }
        model_settings = {
            key: value
            for key, value in (settings.get("model") or {}).items()
            if key in {"observation_template", "format_error_template"}
        }
        model = endpoint_model(
            base_url=base_url,
            model_name=self._model_name(),
            api_key=self.config.api_key or os.environ.get("OPENAI_API_KEY", "gym"),  # pragma: allowlist secret
            sampling=sampling,
            settings=model_settings,
            timeout=self.config.api_timeout,
        )
        return DefaultAgent(
            model,
            LocalEnvironment(cwd=cwd, env=env, timeout=self.config.command_timeout),
            **{key: value for key, value in agent_settings.items() if key != "output_path"},
        )

    async def _create_response(
        self, body: NeMoGymResponseCreateParamsNonStreaming, *, rollout_id: Optional[str]
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]
        task, system = _task(body.input)
        base_url = self.resolve_model_base_url(self.config.model_server.name, rollout_id)
        agent = self._agent(base_url, system, os.getcwd())
        error: Optional[BaseException] = None
        try:
            info = await asyncio.to_thread(agent.run, task)
        except Exception as exc:  # DefaultAgent records the exit message before re-raising.
            LOG.warning("mini-swe-agent ended with %r", exc)
            error, info = exc, (agent.messages[-1].get("extra", {}) if agent.messages else {})
        exit_status = info.get("exit_status", "")
        stop_reason = info.get("stop_reason")
        metadata = {"exit_status": exit_status, "turns": str(agent.n_calls)}
        if stop_reason:
            metadata["stop_reason"] = stop_reason
        elif exit_status == "LimitsExceeded":
            metadata["stop_reason"] = "max_iterations"
        response_error = None
        if error is not None:
            from openai.types.responses import ResponseError

            metadata["harness_error"] = str(error)[:2000]
            response_error = ResponseError(code="server_error", message=str(error)[:2000])
        status = "failed" if error is not None else ("completed" if exit_status == "Submitted" else "incomplete")
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self._model_name(),
            object="response",
            output=output_items(agent.messages),
            status=status,
            error=response_error,
            metadata=metadata,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=0,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=0,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=0,
            ),
        )

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        return await self._create_response(body, rollout_id=rollout_id if isinstance(rollout_id, str) else None)

    async def run(self, request: Request, body: MiniSweAgentRunRequest) -> MiniSweAgentVerifyResponse:
        async with self.sem:
            cookies = request.cookies
            seed = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed)
            agent_response = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=body.responses_create_params,
                cookies=seed.cookies,
            )
            await raise_for_status(agent_response)
            response_json = await get_response_json(agent_response)
            verify = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": response_json},
                cookies=agent_response.cookies,
            )
            await raise_for_status(verify)
            response = NeMoGymResponse.model_validate(response_json)
            return MiniSweAgentVerifyResponse.model_validate(
                await get_response_json(verify)
                | {
                    "turns_used": int((response.metadata or {}).get("turns", 0)),
                    "exit_status": (response.metadata or {}).get("exit_status", ""),
                }
            )


if __name__ == "__main__":
    MiniSweAgent.run_webserver()
