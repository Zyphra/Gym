# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from omegaconf import OmegaConf

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from responses_api_agents.anyswe_agent import app
from responses_api_agents.anyswe_agent.tests.test_app import _config, _private_grader_config


@pytest.mark.parametrize("phase", ["start", "setup"])
async def test_private_sandbox_failure_retains_locator_without_private_error(tmp_path, monkeypatch, phase):
    head = BaseServerConfig(host="127.0.0.1", port=19000, name="head", entrypoint="app.py")
    config = _config(
        model_server=None,
        agent_runtime_source="baked",
        sandbox_model_base_url="http://127.0.0.1:19002/v1",
        private_grader=_private_grader_config(),
    )
    client = ServerClient(head_server_config=head, global_config_dict=OmegaConf.create({"results_dir": str(tmp_path)}))
    agent = app.AnySweAgent(config=config, server_client=client)
    params = agent._setup_params(
        NeMoGymResponseCreateParamsNonStreaming(
            input="Fix the fixture.",
            metadata={
                "instance_id": "test__test-1",
                "private_verifier": json.dumps({"protocol": "test-private-grader-v1"}),
            },
        ),
        rollout_id="fixture-episode",
    )
    closed = []

    class FailingSandbox:
        def __init__(self, provider, spec):
            pass

        async def start(self):
            if phase == "start":
                raise RuntimeError("PRIVATE_FIXTURE_DIAGNOSTIC")

        async def exec(self, command, **kwargs):
            assert phase == "setup" and command == "prepare-private-task"
            return SandboxExecResult("partial setup", "PRIVATE_FIXTURE_DIAGNOSTIC", 1, "timeout")

        async def stop(self):
            closed.append(True)

    monkeypatch.setattr(app, "AsyncSandbox", FailingSandbox)
    response = await app.AnySweSandboxRunner(params).run()
    path = params.persistent_dir / "sandbox_error.txt"
    assert path.is_file()
    if phase == "start":
        assert path.read_text() == "RuntimeError('PRIVATE_FIXTURE_DIAGNOSTIC')"
    else:
        assert "private grader setup failed" in path.read_text()

    async def responses(self, body, rollout_id=None):
        assert rollout_id == "fixture-episode"
        return response

    monkeypatch.setattr(app.AnySweAgent, "_responses", responses)
    monkeypatch.setattr(app.AnySweAgent, "rollout_id_from_run", lambda self, body: "fixture-episode")
    native = await agent.run(app.AnySweRunRequest.model_validate({"responses_create_params": {"input": []}}))
    assert native.artifact_path == str(path)
    assert native.error_kind == ("sandbox" if phase == "start" else "eval_timeout")
    assert native.error_detail == "grader exception type=RuntimeError"
    assert native.mask_sample is True and native.reward == 0
    assert native.instance_config["rollout_id"] == "fixture-episode"
    assert "PRIVATE_FIXTURE_DIAGNOSTIC" not in native.model_dump_json()
    assert closed == [True]
