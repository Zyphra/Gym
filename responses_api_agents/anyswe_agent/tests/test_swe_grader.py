# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A dataset-specific SWE grader replaces the pinned swebench spec and report."""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult, SandboxHandle
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from responses_api_agents.anyswe_agent import app as app_module
from responses_api_agents.anyswe_agent.app import AnySweAgent, AnySweAgentConfig


PATCH = "diff --git a/src/a.c b/src/a.c\n"
INSTANCE = {"instance_id": "jqlang__jq-2658", "repo": "jqlang/jq", "FAIL_TO_PASS": "[]", "PASS_TO_PASS": "[]"}
SCRIPT = "exec 2>&1\ngit apply /root/patch.diff\nmake check\n"
LOGS = []


def plan(instance, patch):
    assert (instance, patch) == (INSTANCE, PATCH)
    return {"script": SCRIPT, "grade": lambda log: LOGS.append(log) or "PASS" in log}


class RecordingProvider:
    def __init__(self, stdout):
        self.files = {}
        self.commands = []
        self.stdout = stdout

    async def create(self, spec):
        return SandboxHandle(sandbox_id="grader", provider_name="recording", raw=None)

    async def upload_file(self, handle, source, target):
        self.files[target] = Path(source).read_text()

    async def exec(self, handle, command, **kwargs):
        self.commands.append(command)
        return SandboxExecResult(self.stdout, "trace", 0, None)

    async def close(self, handle):
        pass

    async def aclose(self):
        pass


async def _grade(tmp_path, monkeypatch, stdout, **overrides):
    config = AnySweAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="anyswe_agent",
        agent_server_module="responses_api_agents.hermes_agent.app",
        agent_server_class="HermesAgent",
        agent_config_class="HermesAgentConfig",
        container_formatter="swebench/sweb.eval.x86_64.{instance_id}",
        sandbox_provider={"opensandbox": {}},
        agent_runtime_source="baked",
        sandbox_model_base_url="http://127.0.0.1:19002/v1",
        **overrides,
    )
    head = BaseServerConfig(host="127.0.0.1", port=19000, name="head", entrypoint="app.py")
    client = ServerClient(head_server_config=head, global_config_dict=OmegaConf.create({"results_dir": str(tmp_path)}))
    agent = AnySweAgent(config=config, server_client=client)
    params = agent._setup_params(
        NeMoGymResponseCreateParamsNonStreaming(
            input="Fix it.",
            metadata={
                "instance_id": INSTANCE["instance_id"],
                "problem_statement": "Fix it.",
                "instance_dict": json.dumps(INSTANCE),
            },
        )
    )
    provider = RecordingProvider(stdout)
    monkeypatch.setattr(app_module, "AsyncSandbox", lambda _provider, spec: AsyncSandbox(provider, spec))
    return provider, await agent._grade_sandbox_patch(params, PATCH)


@pytest.mark.parametrize("stdout,resolved", [("tests PASS", True), ("tests failed", False)])
async def test_swe_grader_script_runs_unchanged_and_its_grade_is_the_verdict(tmp_path, monkeypatch, stdout, resolved):
    LOGS.clear()
    provider, result = await _grade(tmp_path, monkeypatch, stdout, swe_grader=f"{__name__}:plan")

    assert result == (resolved, None, None)
    assert provider.files["/tmp/anyswe_eval.sh"] == SCRIPT
    assert provider.files["/root/patch.diff"] == PATCH
    assert provider.commands == ["bash /tmp/anyswe_eval.sh"]
    # The grader receives the sandbox output exactly as AnySWE captured it.
    assert LOGS == [f"{stdout}\ntrace"]
