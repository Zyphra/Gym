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
"""Opt-in test-authority restore step in AnySWE's default SWE-bench grader."""

import json
from pathlib import Path

from omegaconf import OmegaConf

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult, SandboxHandle
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from responses_api_agents.anyswe_agent import app as app_module
from responses_api_agents.anyswe_agent.app import AnySweAgent, AnySweAgentConfig, AnySweSandboxRunner


PATCH = "diff --git a/pkg/core.py b/pkg/core.py\n"
INSTANCE = {
    "instance_id": "astropy__astropy-13236",
    "repo": "astropy/astropy",
    "version": "5.0",
    "base_commit": "a" * 40,
    "environment_setup_commit": "b" * 40,
    "test_patch": "diff --git a/pkg/tests/test_a.py b/pkg/tests/test_a.py\n",
    "patch": "diff --git a/pkg/core.py b/pkg/core.py\n",
    "FAIL_TO_PASS": ["pkg/tests/test_a.py::test_a"],
    "PASS_TO_PASS": [],
    "problem_statement": "Fix it.",
}
STEP = {"files": {"/root/restore.py": "print('restored')\n"}, "command": "python3 /root/restore.py || exit 1\n"}


def restore_step(instance, patch):
    assert (instance, patch) == (INSTANCE, PATCH)
    return STEP


class RecordingProvider:
    def __init__(self):
        self.files = {}

    async def create(self, spec):
        return SandboxHandle(sandbox_id="grader", provider_name="recording", raw=None)

    async def upload_file(self, handle, source, target):
        self.files[target] = Path(source).read_text()

    async def exec(self, handle, command, **kwargs):
        return SandboxExecResult("", "", 0, None)

    async def close(self, handle):
        pass

    async def aclose(self):
        pass


async def _grader_files(tmp_path, monkeypatch, **overrides):
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
    provider = RecordingProvider()
    monkeypatch.setattr(app_module, "AsyncSandbox", lambda _provider, spec: AsyncSandbox(provider, spec))
    await agent._grade_sandbox_patch(params, PATCH)
    return provider.files


async def test_restore_step_runs_after_the_candidate_applies_and_before_hidden_tests(tmp_path, monkeypatch):
    default = await _grader_files(tmp_path / "default", monkeypatch)
    off = await _grader_files(tmp_path / "off", monkeypatch, test_authority_restore=None)
    on = await _grader_files(tmp_path / "on", monkeypatch, test_authority_restore=f"{__name__}:restore_step")
    assert off == default
    assert set(default) == {"/root/patch.diff", "/tmp/anyswe_eval.sh"}
    apply = AnySweSandboxRunner._apply_patch_script()
    script = default["/tmp/anyswe_eval.sh"]
    assert script.count(apply) == 1
    assert on == {
        **default,
        **STEP["files"],
        "/tmp/anyswe_eval.sh": script.replace(apply, apply + STEP["command"]),
    }
