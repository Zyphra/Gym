# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import shlex
import sys
from unittest.mock import patch

import pytest
from fastapi import Request
from omegaconf import OmegaConf
from openai.types.responses.response_error import ResponseError

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentInvocation, harness_response_error
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from responses_api_agents.anyswe_agent.app import _read_saved_response, _should_mask_sample
from responses_api_agents.pi_agent.app import PiAgent, PiAgentConfig, PiAgentRunRequest


def _events(*, end=True, usage=True):
    message = dict(role="assistant", content=[dict(type="text", text="partial answer")], responseId="response-fixture")
    if usage:
        message["usage"] = dict(input=2, output=3)
    events = [dict(type="message_end", message=message)]
    if end:
        events.append(dict(type="agent_end", messages=[dict(role="assistant", stopReason="stop")]))
    return "".join(json.dumps(event) + "\n" for event in events)


def _agent(root, stdout, exit_code=0, *, timeout=False):
    script = "import sys,time; sys.stdout.write(" + repr(stdout) + "); sys.stdout.flush(); "
    script += "time.sleep(5)" if timeout else "sys.exit(" + str(exit_code) + ")"
    client = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=1),
        global_config_dict=OmegaConf.create(
            {"policy": {"responses_api_models": {"fixture": {"host": "127.0.0.1", "port": 1}}}}
        ),
    )
    config = PiAgentConfig(
        host="127.0.0.1",
        port=1,
        name="pi",
        entrypoint="",
        resources_server=ResourcesServerRef(type="resources_servers", name="fixture"),
        model_server=ModelServerRef(type="responses_api_models", name="policy"),
        command=shlex.join([sys.executable, "-c", script]),
        model="fixture",
        workspace_root=str(root),
        timeout=1 if timeout else 5,
    )
    with patch("responses_api_agents.pi_agent.app.ensure_pi", return_value=None):
        return PiAgent(config=config, server_client=client)


async def _response(agent, observe):
    request = Request(
        {"type": "http", "headers": [], "path_params": {"rollout_id": "fixture-rollout"} if observe else {}}
    )
    response = await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="fixture"))
    return NeMoGymResponse.model_validate_json(response.model_dump_json())


def _stock_roundtrip(root, response):
    path = root / "response.json"
    path.write_text(response.model_dump_json())
    saved, observations = _read_saved_response(path)
    assert saved is not None
    return saved, observations, harness_response_error(saved, observations)


@pytest.mark.parametrize("observe", [False, True])
@pytest.mark.parametrize("stdout", ["", "not-json\n", '{"type":"agent_start"}\n'])
def test_nonzero_without_policy_is_classified_failure(tmp_path, observe, stdout):
    response = asyncio.run(_response(_agent(tmp_path / "work", stdout, 17), observe))
    saved, observations, error = _stock_roundtrip(tmp_path, response)
    assert saved.status == "failed" and saved.output == []
    assert saved.usage.total_tokens == 0
    assert saved.metadata["pi_return_code"] == "17"
    assert error is not None and error.message == "Pi process_exit_17"
    invocation = next(r for r in observations.records if isinstance(r, AgentInvocation))
    assert invocation.status == "failed" and invocation.error_type == "process_exit_17"
    assert _should_mask_sample(False, None, False, "harness")


@pytest.mark.parametrize("observe", [False, True])
@pytest.mark.parametrize("end", [False, True])
@pytest.mark.parametrize("usage", [False, True])
def test_nonzero_partial_events_survive_without_blanket_infra_mask(tmp_path, observe, end, usage):
    response = asyncio.run(_response(_agent(tmp_path / "work", _events(end=end, usage=usage), 17), observe))
    saved, observations, error = _stock_roundtrip(tmp_path, response)
    assert saved.status == "failed"
    assert saved.output[0].content[0].text == "partial answer"
    assert saved.usage.total_tokens == (5 if usage else 0)
    assert error is None
    assert not _should_mask_sample(True, None, False, None)
    if observe:
        invocation = next(r for r in observations.records if isinstance(r, AgentInvocation))
        assert invocation.status == "failed" and invocation.error_type == "process_exit_17"
        assert invocation.model_calls[0].response_id == "response-fixture"


@pytest.mark.parametrize("observe", [False, True])
def test_success_preserves_text_usage_and_completion(tmp_path, observe):
    response = asyncio.run(_response(_agent(tmp_path / "work", _events()), observe))
    saved, observations, error = _stock_roundtrip(tmp_path, response)
    assert saved.status not in ("failed", "incomplete") and saved.error is None
    assert saved.output[0].content[0].text == "partial answer" and saved.usage.total_tokens == 5
    assert error is None
    if observe:
        assert next(r for r in observations.records if isinstance(r, AgentInvocation)).status == "completed"


@pytest.mark.parametrize("observe", [False, True])
def test_timeout_preserves_partial_output_and_usage(tmp_path, observe):
    response = asyncio.run(_response(_agent(tmp_path / "work", _events(), timeout=True), observe))
    saved, observations, error = _stock_roundtrip(tmp_path, response)
    assert saved.status == "incomplete" and saved.error.message == "Pi process_timeout"
    assert saved.output[0].content[0].text == "partial answer" and saved.usage.total_tokens == 5
    assert error is None
    if observe:
        assert next(r for r in observations.records if isinstance(r, AgentInvocation)).status == "failed"


@pytest.mark.parametrize("observe", [False, True])
@pytest.mark.parametrize("exit_code,has_output", [(0, True), (17, True), (17, False)])
def test_run_gates_no_call_failure_and_natural_completion(tmp_path, observe, exit_code, has_output):
    # Only the HTTP boundary is in-process; Pi itself runs a real subprocess.
    class Reply:
        ok = True
        cookies = {}

        def __init__(self, payload):
            self.payload = payload

        async def read(self):
            return json.dumps(self.payload).encode()

    async def run():
        agent = _agent(tmp_path / "work", _events() if has_output else "", exit_code)
        calls = []

        async def post(client, server_name, url_path, json=None, **kwargs):
            calls.append(url_path)
            if url_path.endswith("/v1/responses"):
                response = await _response(agent, observe)
                return Reply(response.model_dump(mode="json"))
            if url_path == "/verify":
                return Reply(json | {"reward": 1.0})
            return Reply({})

        body = PiAgentRunRequest(responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="fixture"))
        request = Request({"type": "http", "headers": []})
        with patch.object(ServerClient, "post", post):
            if not has_output:
                with pytest.raises(RuntimeError, match="Pi process_exit_17"):
                    await agent.run(request, body)
                assert "/verify" not in calls
            else:
                result = await agent.run(request, body)
                assert result.reward == 1.0
                assert result.finished_naturally is (exit_code == 0)
                assert "/verify" in calls

    asyncio.run(run())


def test_existing_policy_stop_and_endpoint_failure_classification(tmp_path):
    response = asyncio.run(_response(_agent(tmp_path / "work", _events()), True))
    saved, observations, error = _stock_roundtrip(tmp_path, response)
    policy_stop = saved.model_copy(update={"status": "incomplete", "metadata": {"stop_reason": "context_exhausted"}})
    assert harness_response_error(policy_stop, observations) is None
    assert not _should_mask_sample(True, "context_window", False, None, "context_exhausted")
    endpoint_failure = saved.model_copy(
        update={
            "status": "failed",
            "error": ResponseError(code="server_error", message="Error code: 503 fixture unavailable"),
        }
    )
    assert harness_response_error(endpoint_failure, observations) is endpoint_failure.error
