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
"""Unit tests for anyswe_agent.

These exercise runner generation, image resolution, and configuration.
"""

import base64
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from responses_api_agents.anyswe_agent.agent_runner import (
    _agent_request,
    _extract_patch,
    _find_repo,
    _snapshot_repo,
)
from responses_api_agents.anyswe_agent.app import (
    AnySweAgent,
    AnySweAgentConfig,
    AnySweEpisodePayload,
    AnySweInstanceConfig,
    AnySweRunRequest,
    AnySweSandboxRunner,
    PrivateGraderConfig,
    PrivateGraderPlan,
    _classify_agent_error,
    _dataset_family,
    _overlay_files,
    _overlay_sha256,
    _r2e_resolved,
    _read_saved_response,
    _safe_config_json,
    _should_mask_sample,
    _write_grader_receipt,
)
from responses_api_agents.anyswe_agent.prepare import _to_gym_row

from nemo_gym.rollout_observability import (
    AgentObservationBundle,
    ToolCallObservation,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult, SandboxHandle
from nemo_gym.sandbox.episode import EPISODE_MEMORY_HEADROOM_MIB


def _config(**overrides) -> AnySweAgentConfig:
    base = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="anyswe_agent",
        model_server={"type": "responses_api_models", "name": "policy_model"},
        agent_server_module="responses_api_agents.hermes_agent.app",
        agent_server_class="HermesAgent",
        agent_config_class="HermesAgentConfig",
        container_formatter="swebench/sweb.eval.x86_64.{instance_id}",
        sandbox_provider={"opensandbox": {}},
    )
    base.update(overrides)
    return AnySweAgentConfig(**base)


@pytest.mark.parametrize("limit", [None, 4])
async def test_run_concurrency_null_is_unbounded_and_integer_is_bounded(tmp_path, monkeypatch, limit):
    import asyncio

    from omegaconf import OmegaConf

    from nemo_gym.server_utils import BaseServerConfig, ServerClient

    config = _config(
        concurrency=limit,
        model_server=None,
        agent_runtime_source="baked",
        sandbox_model_base_url="http://127.0.0.1:19002/v1",
    )
    head = BaseServerConfig(host="127.0.0.1", port=19000, name="head", entrypoint="app.py")
    client = ServerClient(head_server_config=head, global_config_dict=OmegaConf.create({"results_dir": str(tmp_path)}))
    agent = AnySweAgent(config=config, server_client=client)
    assert (agent._sem is None) == (limit is None)
    release = asyncio.Event()
    reached = asyncio.Event()
    active = peak = entered = 0
    expected = 24 if limit is None else limit

    class ControlledEnd(Exception):
        pass

    async def responses(self, body, rollout_id=None):
        nonlocal active, peak, entered
        active += 1
        entered += 1
        peak = max(peak, active)
        if active == expected:
            reached.set()
        try:
            await release.wait()
            raise ControlledEnd
        finally:
            active -= 1

    monkeypatch.setattr(AnySweAgent, "_responses", responses)
    body = AnySweRunRequest.model_validate({"responses_create_params": {"input": []}})
    requests = [asyncio.create_task(agent.run(body)) for _ in range(24)]
    try:
        await asyncio.wait_for(reached.wait(), timeout=2)
        assert active == expected
        assert entered == expected
    finally:
        release.set()
        results = await asyncio.gather(*requests, return_exceptions=True)
    assert all(isinstance(result, ControlledEnd) for result in results)
    assert peak == expected
    assert entered == 24
    assert active == 0


@pytest.mark.parametrize("limit", [0, -1])
def test_run_concurrency_refuses_nonpositive_limits(limit):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _config(concurrency=limit)


@pytest.mark.parametrize("harness", ["pi", "hermes"])
def test_pi_and_hermes_yaml_select_unbounded_concurrency(harness):
    import yaml

    path = Path(__file__).parents[1] / "configs" / f"anyswe_{harness}.yaml"
    raw = yaml.safe_load(path.read_text())
    assert raw[f"anyswe_{harness}"]["responses_api_agents"]["anyswe_agent"]["concurrency"] is None


@pytest.fixture
def episode_submissions(monkeypatch):
    """Exercise shared dispatch and its real payload without starting a Ray cluster."""
    from core.sandbox.executor import EpisodeResources, RayEpisodeExecutor

    monkeypatch.delenv("NEMO_GYM_SANDBOX_HEADROOM", raising=False)
    monkeypatch.delenv("NEMO_GYM_SANDBOX_MEMORY_MB", raising=False)
    monkeypatch.delenv("NEMO_GYM_SANDBOX_PIDS", raising=False)
    submissions = []

    async def submit(self, work, *, episode_id, resources):
        assert isinstance(resources, EpisodeResources)
        assert episode_id
        submissions.append(resources)
        return await work()

    monkeypatch.setattr(RayEpisodeExecutor, "submit", submit)
    return submissions


def _test_private_grader_plan(payload, patch):
    assert payload["protocol"] == "test-private-grader-v1"
    if patch is None:
        return {
            "command": "prepare-private-task",
            "workdir": "/workspace/repo",
            "env": {"SETUP": "1"},
            "files": {"/sandbox/public-setup.txt": "public setup"},
        }
    return {
        "command": "grade-private-task",
        "workdir": "/workspace/repo",
        "env": {"GRADE": "1"},
        "files": {"/sandbox/private/model.patch": patch},
    }


def _private_grader_config(**overrides) -> PrivateGraderConfig:
    values = {
        "target": f"{__name__}:_test_private_grader_plan",
        "metadata_key": "private_verifier",
        "unresolved_return_code": 86,
        "timeout_return_code": 80,
    }
    values.update(overrides)
    return PrivateGraderConfig(**values)


def test_grader_receipt_preserves_execution_streams(tmp_path) -> None:
    success = SandboxExecResult(
        stdout="test_example PASSED\n",
        stderr="collection warning\n",
        return_code=0,
        error_type=None,
    )
    _write_grader_receipt(tmp_path, success)
    assert (tmp_path / "grader.stdout").read_text() == success.stdout
    assert (tmp_path / "grader.stderr").read_text() == success.stderr
    assert json.loads((tmp_path / "grader_result.json").read_text()) == {
        "error_type": None,
        "return_code": 0,
        "stderr_path": "grader.stderr",
        "stdout_path": "grader.stdout",
    }
    sandbox_error = SandboxExecResult(
        stdout=None,
        stderr="sandbox execution failed\n",
        return_code=1,
        error_type="sandbox",
    )
    _write_grader_receipt(tmp_path, sandbox_error)
    assert (tmp_path / "grader.stdout").read_text() == ""
    assert (tmp_path / "grader.stderr").read_text() == sandbox_error.stderr
    assert json.loads((tmp_path / "grader_result.json").read_text()) == {
        "error_type": "sandbox",
        "return_code": 1,
        "stderr_path": "grader.stderr",
        "stdout_path": "grader.stdout",
    }


def test_episode_payload_keeps_private_grader_input_out_of_public_config() -> None:
    params = AnySweInstanceConfig.model_construct(private_grader_payload={"gold_patch": "secret"})
    payload = AnySweEpisodePayload.from_instance_config(params)
    worker_payload = AnySweEpisodePayload.worker_payload(params)

    assert payload.instance_config.private_grader_payload is None
    assert payload.unpack().private_grader_payload == {"gold_patch": "secret"}
    assert "private_grader_payload" not in payload.model_dump(exclude={"instance_config"})
    assert "private_grader_payload" not in worker_payload["instance_config"]
    assert worker_payload["private_grader_payload"] == {"gold_patch": "secret"}


def test_saved_response_preserves_standardized_agent_observations(tmp_path) -> None:
    from nemo_gym.openai_utils import NeMoGymResponse

    bundle = AgentObservationBundle(
        source="test-agent",
        records=[
            ToolCallObservation(
                invocation_id="rollout-1",
                tool_call_id="tool-1",
                tool_name="terminal",
                started_at=10.0,
                completed_at=10.25,
                duration_ms=250.0,
                timing_source="harness",
                status="completed",
            )
        ],
    )
    response = NeMoGymResponse(
        id="response-1",
        created_at=1,
        model="test-model",
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    ).model_copy(update={"_ng_agent_observations": bundle.model_dump(mode="json")})
    response_path = tmp_path / "response.json"
    response_path.write_text(response.model_dump_json())

    saved, observations = _read_saved_response(response_path)

    assert saved is not None and saved.id == "response-1"
    assert observations == bundle


@pytest.mark.parametrize("index", [0, 1], ids=["pi-no-call-failure", "hermes-route-denial"])
def test_saved_harness_failure_is_typed_and_masked(tmp_path, index: int) -> None:
    from nemo_gym import PARENT_DIR

    raw = json.loads((PARENT_DIR / "tests/unit_tests/fixtures/sandbox-harness-failures.json").read_text())[index]["response"]
    path = tmp_path / "response.json"
    path.write_text(json.dumps(raw))
    saved, observations = _read_saved_response(path)
    from nemo_gym.rollout_observability import harness_response_error

    assert harness_response_error(saved, observations) is not None
    assert observations is not None
    assert _should_mask_sample(False, "other", False, "harness") is True


async def test_grader_receipt_precedes_sandbox_teardown(tmp_path) -> None:
    from unittest.mock import AsyncMock, patch

    from omegaconf import OmegaConf

    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_utils import BaseServerConfig, ServerClient

    config = _config(
        model_server=None,
        agent_runtime_source="baked",
        sandbox_model_base_url="http://127.0.0.1:19002/v1",
    )
    head = BaseServerConfig(host="127.0.0.1", port=19000, name="head", entrypoint="app.py")
    client = ServerClient(head_server_config=head, global_config_dict=OmegaConf.create({"results_dir": str(tmp_path)}))
    agent = AnySweAgent(config=config, server_client=client)
    params = agent._setup_params(
        NeMoGymResponseCreateParamsNonStreaming(
            input="Fix the test.",
            metadata={
                "dataset_name": "r2e-gym",
                "instance_id": "test__test-1",
                "instance_dict": json.dumps(
                    {"eval_script": "true", "FAIL_TO_PASS": '["tests/test_example.py::test"]'}
                ),
                "problem_statement": "Fix the test.",
            },
        )
    )
    result = SandboxExecResult(
        stdout="PASSED tests/test_example.py::test\n",
        stderr="grader warning\n",
        return_code=0,
        error_type=None,
    )
    sandbox = AsyncMock()
    sandbox.exec.return_value = result
    sandbox.stop.side_effect = RuntimeError("sandbox teardown failed")

    with patch("responses_api_agents.anyswe_agent.app.AsyncSandbox", return_value=sandbox):
        with pytest.raises(RuntimeError, match="sandbox teardown failed"):
            await agent._grade_r2e_patch(params, "")

    sandbox.exec.assert_awaited_once()
    sandbox.stop.assert_awaited_once()
    assert (params.persistent_dir / "grader.stdout").read_text() == result.stdout
    assert (params.persistent_dir / "grader.stderr").read_text() == result.stderr


def test_private_grader_contract_is_typed() -> None:
    import pytest
    from pydantic import ValidationError

    assert _private_grader_config().timeout_return_code == 80
    with pytest.raises(ValidationError, match="module:function"):
        _private_grader_config(target="missing_separator")
    with pytest.raises(ValidationError, match="must differ"):
        _private_grader_config(timeout_return_code=86)
    with pytest.raises(ValidationError, match="must be absolute"):
        PrivateGraderPlan(command="true", workdir="relative")
    with pytest.raises(ValidationError, match="must be absolute"):
        PrivateGraderPlan(command="true", workdir="/workspace/repo", files={"relative": "data"})


async def test_private_setup_stages_only_planned_inputs(tmp_path, monkeypatch) -> None:
    from omegaconf import OmegaConf
    from responses_api_agents.anyswe_agent import app as app_module

    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_utils import BaseServerConfig, ServerClient

    head = BaseServerConfig(host="127.0.0.1", port=19000, name="head", entrypoint="app.py")
    config = _config(
        model_server=None,
        agent_runtime_source="baked",
        sandbox_model_base_url="http://127.0.0.1:19002/v1",
        private_grader=_private_grader_config(),
    )
    client = ServerClient(head_server_config=head, global_config_dict=OmegaConf.create({"results_dir": str(tmp_path)}))
    agent = AnySweAgent(config=config, server_client=client)
    gold = "PRIVATE-GOLD-MUST-NOT-REACH-THE-AGENT"
    params = agent._setup_params(
        NeMoGymResponseCreateParamsNonStreaming(
            input="Fix the test.",
            metadata={
                "instance_id": "test__test-1",
                "problem_statement": "Fix the test.",
                "private_verifier": json.dumps({"protocol": "test-private-grader-v1", "gold_patch": gold}),
            },
        )
    )
    assert params.private_grader_payload["gold_patch"] == gold
    assert "private_verifier" not in params.problem_info
    assert "private_verifier" not in (params.body.metadata or {})
    assert gold not in _safe_config_json(params)

    class FakeSandbox:
        instances = []

        def __init__(self, provider, spec):
            self.provider = provider
            self.spec = spec
            self.commands = []
            self.instances.append(self)

        async def start(self):
            return None

        async def stop(self):
            return None

        async def exec(self, command, **kwargs):
            self.commands.append((command, kwargs))
            return SandboxExecResult(
                stdout="setup complete\n" if command == "prepare-private-task" else "",
                stderr="",
                return_code=1 if command.startswith("test -f ") else 0,
                error_type=None,
            )

        async def upload(self, source, destination):
            raise AssertionError("baked runtime must not upload")

        async def download(self, source, destination):
            raise AssertionError("missing result files must not download")

    monkeypatch.setattr(app_module, "AsyncSandbox", FakeSandbox)
    await AnySweSandboxRunner(params).run()

    sandbox = FakeSandbox.instances[0]
    assert sandbox.spec.workdir == "/workspace/repo"
    assert sandbox.commands[0] == (
        "prepare-private-task",
        {
            "cwd": "/workspace/repo",
            "env": {"SETUP": "1"},
            "timeout_s": params.swebench_tests_timeout,
            "user": "root",
        },
    )
    agent_calls = [item for item in sandbox.commands if "agent_runner.py" in item[0]]
    assert agent_calls
    assert agent_calls[0][1]["cwd"] == "/workspace/repo"
    assert sandbox.spec.files["/sandbox/public-setup.txt"] == "public setup"
    assert gold not in json.dumps(sandbox.spec.files)
    assert (params.persistent_dir / "setup.stdout").read_text() == "setup complete\n"
    assert (params.persistent_dir / "setup.stderr").read_text() == ""
    assert json.loads((params.persistent_dir / "setup_result.json").read_text()) == {
        "error_type": None,
        "return_code": 0,
        "stderr_path": "setup.stderr",
        "stdout_path": "setup.stdout",
    }


async def test_private_grader_external_runtime_uses_plan_workdir(tmp_path, monkeypatch) -> None:
    import tarfile

    from omegaconf import OmegaConf
    from responses_api_agents.anyswe_agent import app as app_module

    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_utils import BaseServerConfig, ServerClient

    archive = tmp_path / "hermes-runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(name="bin/python")
        info.size = 0
        tar.addfile(info)

    head = BaseServerConfig(host="127.0.0.1", port=19000, name="head", entrypoint="app.py")
    config = _config(
        model_server=None,
        agent_runtime_source=str(archive),
        sandbox_model_base_url="http://127.0.0.1:19002/v1",
        private_grader=_private_grader_config(),
    )
    client = ServerClient(head_server_config=head, global_config_dict=OmegaConf.create({"results_dir": str(tmp_path)}))
    agent = AnySweAgent(config=config, server_client=client)
    params = agent._setup_params(
        NeMoGymResponseCreateParamsNonStreaming(
            input="Fix the test.",
            metadata={
                "instance_id": "test__test-1",
                "problem_statement": "Fix the test.",
                "private_verifier": json.dumps({"protocol": "test-private-grader-v1"}),
            },
        )
    )

    class RecordingSandbox:
        instances = []

        def __init__(self, provider, spec):
            self.provider = provider
            self.spec = spec
            self.commands = []
            self.uploads = []
            self.instances.append(self)

        async def start(self):
            return None

        async def stop(self):
            return None

        async def exec(self, command, **kwargs):
            self.commands.append((command, kwargs))
            return SandboxExecResult("", "", 0, None)

        async def upload(self, source, destination):
            self.uploads.append((source, destination))

        async def download(self, source, destination):
            return None

    monkeypatch.setattr(app_module, "AsyncSandbox", RecordingSandbox)
    await AnySweSandboxRunner(params).run()

    sandbox = RecordingSandbox.instances[0]
    assert sandbox.spec.workdir == "/workspace/repo"
    assert sandbox.uploads == [(archive, "/sandbox/anyswe_agent_deps.tar.gz")]
    extract_calls = [item for item in sandbox.commands if "tar -xzf" in item[0]]
    assert extract_calls
    assert extract_calls[0][1]["cwd"] == "/workspace/repo"
    agent_calls = [item for item in sandbox.commands if "agent_runner.py" in item[0]]
    assert agent_calls
    assert agent_calls[0][1]["cwd"] == "/workspace/repo"
    assert "/testbed" not in json.dumps([item[1] for item in extract_calls + agent_calls])


async def test_private_setup_timeout_is_retained_and_classified(tmp_path, monkeypatch) -> None:
    from omegaconf import OmegaConf
    from responses_api_agents.anyswe_agent import app as app_module

    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_utils import BaseServerConfig, ServerClient

    head = BaseServerConfig(host="127.0.0.1", port=19000, name="head", entrypoint="app.py")
    config = _config(
        model_server=None,
        agent_runtime_source="baked",
        sandbox_model_base_url="http://127.0.0.1:19002/v1",
        private_grader=_private_grader_config(),
    )
    client = ServerClient(head_server_config=head, global_config_dict=OmegaConf.create({"results_dir": str(tmp_path)}))
    agent = AnySweAgent(config=config, server_client=client)
    params = agent._setup_params(
        NeMoGymResponseCreateParamsNonStreaming(
            input="Fix the test.",
            metadata={
                "instance_id": "test__test-1",
                "problem_statement": "Fix the test.",
                "private_verifier": json.dumps({"protocol": "test-private-grader-v1"}),
            },
        )
    )

    class TimeoutSandbox:
        def __init__(self, provider, spec):
            self.provider = provider
            self.spec = spec

        async def start(self):
            return None

        async def stop(self):
            return None

        async def exec(self, command, **kwargs):
            assert command == "prepare-private-task"
            return SandboxExecResult("partial setup\n", "deadline\n", 1, "timeout")

    monkeypatch.setattr(app_module, "AsyncSandbox", TimeoutSandbox)
    response = await AnySweSandboxRunner(params).run()
    metrics = json.loads(response.metadata["metrics"])
    assert metrics["error_kind"] == "eval_timeout"
    assert metrics["mask_sample"] is True
    assert metrics["error_detail"] == "grader exception type=RuntimeError"
    assert "private grader setup failed" in (params.persistent_dir / "sandbox_error.txt").read_text()
    assert (params.persistent_dir / "setup.stdout").read_text() == "partial setup\n"
    assert (params.persistent_dir / "setup.stderr").read_text() == "deadline\n"


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (SandboxExecResult("passed", "", 0), (True, None, None)),
        (SandboxExecResult("failed", "", 86), (False, None, None)),
        (SandboxExecResult("timed out", "", 80), (False, "eval_timeout", "grader exec error_type=None rc=80")),
        (SandboxExecResult("broken", "", 17), (False, "grader", "grader exec error_type=None rc=17")),
        (
            SandboxExecResult("", "transport", 1, "sandbox"),
            (False, "sandbox", "grader exec error_type=sandbox rc=1"),
        ),
        (
            SandboxExecResult("", "deadline", 1, "timeout"),
            (False, "eval_timeout", "grader exec error_type=timeout rc=1"),
        ),
    ],
)
async def test_private_grader_return_code_contract(tmp_path, monkeypatch, result, expected) -> None:
    from omegaconf import OmegaConf
    from responses_api_agents.anyswe_agent import app as app_module

    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_utils import BaseServerConfig, ServerClient

    head = BaseServerConfig(host="127.0.0.1", port=19000, name="head", entrypoint="app.py")
    config = _config(
        model_server=None,
        agent_runtime_source="baked",
        sandbox_model_base_url="http://127.0.0.1:19002/v1",
        private_grader=_private_grader_config(),
    )
    client = ServerClient(head_server_config=head, global_config_dict=OmegaConf.create({"results_dir": str(tmp_path)}))
    agent = AnySweAgent(config=config, server_client=client)
    params = agent._setup_params(
        NeMoGymResponseCreateParamsNonStreaming(
            input="Fix the test.",
            metadata={
                "instance_id": "test__test-1",
                "problem_statement": "Fix the test.",
                "private_verifier": json.dumps({"protocol": "test-private-grader-v1", "gold_patch": "private"}),
            },
        )
    )

    class FakeSandbox:
        instance = None

        def __init__(self, provider, spec):
            self.provider = provider
            self.spec = spec
            self.command = None
            self.__class__.instance = self

        async def start(self):
            return None

        async def stop(self):
            return None

        async def exec(self, command, **kwargs):
            self.command = (command, kwargs)
            return result

    monkeypatch.setattr(app_module, "AsyncSandbox", FakeSandbox)
    assert await agent._grade_private_patch(params, "model patch") == expected
    assert FakeSandbox.instance.spec.files == {"/sandbox/private/model.patch": "model patch"}
    assert FakeSandbox.instance.command[0] == "grade-private-task"
    assert json.loads((params.persistent_dir / "grader_result.json").read_text())["return_code"] == result.return_code
    assert (params.persistent_dir / "grader.stdout").read_text() == result.stdout
    assert (params.persistent_dir / "grader.stderr").read_text() == result.stderr


async def test_private_grader_records_its_sandbox_memory_events(tmp_path, monkeypatch) -> None:
    """A grader the memory limit killed is told apart from a setup or scoring failure by its own events."""
    from omegaconf import OmegaConf
    from responses_api_agents.anyswe_agent import app as app_module

    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_utils import BaseServerConfig, ServerClient

    head = BaseServerConfig(host="127.0.0.1", port=19000, name="head", entrypoint="app.py")
    config = _config(
        model_server=None,
        agent_runtime_source="baked",
        sandbox_model_base_url="http://127.0.0.1:19002/v1",
        private_grader=_private_grader_config(),
    )
    client = ServerClient(head_server_config=head, global_config_dict=OmegaConf.create({"results_dir": str(tmp_path)}))
    agent = AnySweAgent(config=config, server_client=client)
    params = agent._setup_params(
        NeMoGymResponseCreateParamsNonStreaming(
            input="Fix the test.",
            metadata={
                "instance_id": "test__test-1",
                "problem_statement": "Fix the test.",
                "private_verifier": json.dumps({"protocol": "test-private-grader-v1", "gold_patch": "private"}),
            },
        )
    )

    class RetainedEventsProvider:
        closed = False

        async def create(self, spec):
            return SandboxHandle(sandbox_id="grader", provider_name="retained", raw=None)

        async def upload_file(self, handle, source, target):
            pass

        async def exec(self, handle, command, **kwargs):
            return SandboxExecResult("", "Killed", 137, None)

        async def close(self, handle):
            self.closed = True

        async def aclose(self):
            pass

        def retained_memory_events(self, handle):
            assert self.closed, "events are read after the sandbox is stopped"
            return {"low": 0, "high": 0, "max": 12, "oom": 1, "oom_kill": 1}

    monkeypatch.setattr(app_module, "AsyncSandbox", lambda provider, spec: AsyncSandbox(RetainedEventsProvider(), spec))
    assert await agent._grade_private_patch(params, "model patch") == (
        False,
        "grader",
        "grader exec error_type=None rc=137",
    )
    assert (params.persistent_dir / "grader.stdout").read_text() == ""
    assert (params.persistent_dir / "grader.stderr").read_text() == "Killed"
    assert json.loads((params.persistent_dir / "grader_result.json").read_text())["return_code"] == 137
    metrics = json.loads(params.metrics_fpath.read_text())
    assert (
        metrics["grader_sandbox_oom"],
        metrics["grader_sandbox_oom_kill"],
        metrics["grader_sandbox_memory_max_events"],
    ) == (1, 1, 12)
    assert "sandbox_oom_kill" not in metrics


@pytest.mark.skipif(not os.environ.get("ANYSWE_PATCH_PROBE_PARAMS"), reason="requires admitted SIF fixtures")
@pytest.mark.parametrize("case_index", [0, 1])
async def test_patch_baseline_in_admitted_sif(tmp_path, case_index):
    from responses_api_agents.anyswe_agent.app import AnySweInstanceConfig

    from nemo_gym.sandbox import AsyncSandbox

    paths = json.loads(Path(os.environ["ANYSWE_PATCH_PROBE_PARAMS"]).read_text())
    payload = json.loads(Path(paths[case_index]).read_text())
    params = AnySweInstanceConfig.model_validate(payload)
    gym = Path(__file__).resolve().parents[3]
    workdir = params.problem_info["workdir"]
    code = """import json, shutil, time
from pathlib import Path
from responses_api_agents.anyswe_agent.agent_runner import _snapshot_repo, _extract_patch
repo = Path.cwd()
index = Path('/trajectories_mount/baseline.index')
started = time.monotonic()
baseline = _snapshot_repo(repo, index)
elapsed = time.monotonic() - started
assert _extract_patch(repo, index, baseline) == ''
(repo / '.nemo-gym-patch-probe.txt').write_text('agent edit\\n')
shutil.rmtree(repo / '.git')
patch = _extract_patch(repo, index, baseline)
assert '.nemo-gym-patch-probe.txt' in patch and '+agent edit' in patch
assert not (repo / '.git').exists()
objects = index.with_name(index.name + '.git') / 'objects'
print(json.dumps({'snapshot_seconds': elapsed, 'object_bytes': sum(p.stat().st_size for p in objects.rglob('*') if p.is_file()), 'patch_chars': len(patch), 'metadata_removed': True}))
"""
    spec = AnySweAgent._sandbox_spec(
        params,
        workdir=workdir,
        files={"/trajectories_mount/patch_probe.py": code},
        binds=[f"{gym}:/nemo_gym_mount:ro", f"{params.agent_deps_dir}:/agent_deps_mount:ro"],
    )
    sandbox = AsyncSandbox(params.resolved_sandbox_provider, spec)
    try:
        await sandbox.start()
        result = await sandbox.exec(
            "/agent_deps_mount/bin/python /trajectories_mount/patch_probe.py",
            cwd=workdir,
            env={"PYTHONPATH": "/nemo_gym_mount"},
            timeout_s=180,
            user="root",
        )
        assert result.return_code == 0, (result.stdout, result.stderr)
        assert result.error_type is None
        measured = json.loads(result.stdout.strip().splitlines()[-1])
        assert measured["metadata_removed"] and measured["patch_chars"] > 0
        print(json.dumps({"task": params.instance_id, **measured}), flush=True)
    finally:
        await sandbox.stop()


class TestAgentRunner:
    @staticmethod
    def _source() -> str:
        return (Path(__file__).parent.parent / "agent_runner.py").read_text()

    def test_is_valid_python(self) -> None:
        source = self._source()
        compile(source, "<runner>", "exec")
        assert 'os.environ["NGSWE_AGENT_MODULE"]' in source
        assert '["git", "add", "-A"]' in source

    def test_rollout_identity_is_forwarded_to_the_inner_agent(self) -> None:
        assert _agent_request("rollout-1").path_params == {"rollout_id": "rollout-1"}
        assert _agent_request(None).path_params == {}

    def test_patch_extraction_excludes_image_dirt_and_includes_agent_files(self, tmp_path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
        (repo / "agent-edited.txt").write_text("committed\n")
        (repo / "image-dirty.txt").write_text("committed\n")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "initial"], cwd=repo, check=True)

        (repo / "agent-edited.txt").write_text("image baseline\n")
        (repo / "image-dirty.txt").write_text("pre-existing task image change\n")
        (repo / "image-untracked.txt").write_text("pre-existing untracked file\n")
        index_path = tmp_path / "baseline.index"
        baseline_tree = _snapshot_repo(repo, index_path)

        (repo / "agent-edited.txt").write_text("agent change\n")
        (repo / "agent-new.txt").write_text("new from agent\n")
        patch_text = _extract_patch(repo, index_path, baseline_tree)

        assert "agent-edited.txt" in patch_text
        assert "agent change" in patch_text
        assert "agent-new.txt" in patch_text
        assert "new from agent" in patch_text
        assert "image-dirty.txt" not in patch_text
        assert "image-untracked.txt" not in patch_text

    def test_patch_extraction_skips_only_empty_unborn_nested_repository(self, tmp_path) -> None:
        repo = tmp_path / "repo"
        self._init_repo(repo)
        (repo / "deleted.txt").write_text("delete me\n")
        subprocess.run(["git", "add", "deleted.txt"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "add deletion fixture"], cwd=repo, check=True)
        index_path = tmp_path / "baseline.index"
        baseline_tree = _snapshot_repo(repo, index_path)

        (repo / "tracked.txt").write_text("agent change\n")
        (repo / "deleted.txt").unlink()
        (repo / "new.txt").write_text("agent file\n")
        (repo / "agent-link").symlink_to("new.txt")
        nested = repo / "empty[repo]*?"
        subprocess.run(["git", "init", "-q", str(nested)], check=True)

        patch = _extract_patch(repo, index_path, baseline_tree)

        assert "agent change" in patch
        assert "agent file" in patch
        assert "deleted file mode" in patch
        assert "new file mode 120000" in patch
        assert "empty[repo]*?" not in patch
        assert (nested / ".git").is_dir()

    def test_nonempty_unborn_nested_repository_is_not_discarded(self, tmp_path) -> None:
        repo = tmp_path / "repo"
        self._init_repo(repo)
        index_path = tmp_path / "baseline.index"
        baseline_tree = _snapshot_repo(repo, index_path)
        nested = repo / "unborn"
        subprocess.run(["git", "init", "-q", str(nested)], check=True)
        (nested / "agent-file.txt").write_text("keep me\n")

        with pytest.raises(subprocess.CalledProcessError):
            _extract_patch(repo, index_path, baseline_tree)

        assert (nested / "agent-file.txt").read_text() == "keep me\n"

    def test_nested_repository_with_commit_is_exported_as_gitlink(self, tmp_path) -> None:
        repo = tmp_path / "repo"
        self._init_repo(repo)
        index_path = tmp_path / "baseline.index"
        baseline_tree = _snapshot_repo(repo, index_path)
        nested = repo / "committed"
        self._init_repo(nested)

        patch = _extract_patch(repo, index_path, baseline_tree)

        assert "new file mode 160000" in patch
        assert "committed" in patch
        assert "Subproject commit" in patch

    def test_symlinked_git_metadata_is_not_treated_as_empty_repository(self, tmp_path) -> None:
        repo = tmp_path / "repo"
        self._init_repo(repo)
        index_path = tmp_path / "baseline.index"
        baseline_tree = _snapshot_repo(repo, index_path)
        metadata = tmp_path / "metadata"
        subprocess.run(["git", "init", "--bare", "-q", str(metadata)], check=True)
        nested = repo / "linked-metadata"
        nested.mkdir()
        (nested / ".git").symlink_to(metadata)

        with pytest.raises(subprocess.CalledProcessError):
            _extract_patch(repo, index_path, baseline_tree)

        assert (nested / ".git").is_symlink()

    def _init_repo(self, repo: Path) -> None:
        repo.mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
        (repo / "tracked.txt").write_text("committed\n")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "initial"], cwd=repo, check=True)

    @pytest.mark.parametrize("remove_worktree", [False, True])
    def test_patch_survives_agent_removing_repository_metadata(self, tmp_path, remove_worktree) -> None:
        repo = tmp_path / "repo"
        self._init_repo(repo)
        index_path = tmp_path / "baseline.index"
        baseline_tree = _snapshot_repo(repo, index_path)
        (repo / "tracked.txt").write_text("agent change\n")
        (repo / "new.txt").write_text("agent file\n")
        shutil.rmtree(repo if remove_worktree else repo / ".git")
        patch = _extract_patch(repo, index_path, baseline_tree)
        target = tmp_path / "grader"
        self._init_repo(target)
        subprocess.run(["git", "apply", "-"], input=patch, text=True, check=True, cwd=target)
        if remove_worktree:
            assert not (target / "tracked.txt").exists()
            assert not repo.exists()
        else:
            assert (target / "tracked.txt").read_text() == "agent change\n"
            assert (target / "new.txt").read_text() == "agent file\n"
            assert not (repo / ".git").exists()

    def test_corrupt_baseline_is_not_an_empty_successful_patch(self, tmp_path) -> None:
        repo = tmp_path / "repo"
        self._init_repo(repo)
        index = tmp_path / "baseline.index"
        _snapshot_repo(repo, index)
        with pytest.raises(subprocess.CalledProcessError):
            _extract_patch(repo, index, "0" * 40)

    def test_private_baseline_preserves_excludes_and_filemode(self, tmp_path) -> None:
        repo = tmp_path / "repo"
        self._init_repo(repo)
        subprocess.run(["git", "config", "core.filemode", "false"], check=True, cwd=repo)
        (repo / ".git/info/exclude").write_text("ignored.txt\n")
        index_path = tmp_path / "baseline.index"
        baseline_tree = _snapshot_repo(repo, index_path)
        (repo / "tracked.txt").chmod(0o755)
        (repo / "ignored.txt").write_text("not part of the patch\n")
        shutil.rmtree(repo / ".git")
        assert _extract_patch(repo, index_path, baseline_tree) == ""

    def test_find_repo_uses_task_cwd_before_standard_candidates(self, tmp_path, monkeypatch) -> None:
        import responses_api_agents.anyswe_agent.agent_runner as runner

        scale = tmp_path / "workspace" / "auth0-python"
        standard = tmp_path / "testbed"
        self._init_repo(scale)
        self._init_repo(standard)
        monkeypatch.setattr(runner, "_REPO_CANDIDATES", (str(standard), str(tmp_path / "workspace" / "repo")))
        assert _find_repo(scale) == scale
        assert _find_repo(None) == standard

    def test_snapshot_extract_on_scale_workdir_and_standard_path(self, tmp_path, monkeypatch) -> None:
        import responses_api_agents.anyswe_agent.agent_runner as runner

        scale = tmp_path / "workspace" / "auth0-python"
        standard = tmp_path / "testbed"
        self._init_repo(scale)
        self._init_repo(standard)
        monkeypatch.setattr(runner, "_REPO_CANDIDATES", (str(standard),))

        for repo in (scale, standard):
            (repo / "tracked.txt").write_text("image baseline\n")
            index_path = tmp_path / f"{repo.name}.index"
            baseline_tree = _snapshot_repo(repo, index_path)
            (repo / "tracked.txt").write_text("agent change\n")
            (repo / "agent-new.txt").write_text("new from agent\n")
            patch_text = _extract_patch(repo, index_path, baseline_tree)
            assert "agent change" in patch_text
            assert "agent-new.txt" in patch_text
            assert _find_repo(repo) == repo

        assert _find_repo(scale) == scale
        assert _find_repo(None) == standard

    def test_sampling_is_forwarded(self) -> None:
        source = self._source()
        assert "NGSWE_SAMPLING" in source
        assert "**sampling," in source
        assert "**{**agent_kwargs, **config_sampling}" in source
        assert "config_class.model_fields" in source
        assert 'agent_kwargs["model"] = model_name' not in source


class TestSandboxAPI:
    def test_run_request_preserves_rollout_indices(self) -> None:
        request = AnySweRunRequest.model_validate(
            {
                "responses_create_params": {"input": [], "model": "model"},
                "_ng_task_index": 3,
                "_ng_rollout_index": 1,
            }
        )
        assert getattr(request, "_ng_task_index") == 3
        assert getattr(request, "_ng_rollout_index") == 1

    def test_default_provider_is_named_sandbox(self) -> None:
        config = AnySweAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="app.py",
            name="anyswe_agent",
            agent_server_module="responses_api_agents.hermes_agent.app",
            agent_server_class="HermesAgent",
            agent_config_class="HermesAgentConfig",
            container_formatter="registry.example.com/anyswe:{instance_id}",
        )
        assert config.sandbox_provider == "sandbox"
        assert config.agent_runtime_source == "baked"

    def test_image_uses_swebench_tag_format(self) -> None:
        image = AnySweAgent._sandbox_image(
            {
                "instance_id": "Astropy__Astropy-12907",
                "container_formatter": "docker://swebench/sweb.eval.x86_64.{instance_id}",
            }
        )
        assert image == "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"

    def test_spec_forwards_public_sandbox_fields(self) -> None:
        params = SimpleNamespace(
            sandbox_spec={
                "ttl_s": 600,
                "ready_timeout_s": 300,
                "resources": {"cpu": 2, "memory_mib": 4096},
                "provider_options": {"platform": {"arch": "amd64"}},
            },
            sandbox_default_metadata={"sandbox-api": "opensandbox-sdk"},
            swebench_agent_timeout=100,
            swebench_tests_timeout=200,
            instance_id="astropy__astropy-12907",
            container="image:tag",
        )
        spec = AnySweAgent._sandbox_spec(params, files={"/tmp/input": "data"})
        assert spec.image == "image:tag"
        assert spec.ttl_s == 600
        assert spec.resources.cpu == 2
        assert spec.metadata["sandbox-api"] == "opensandbox-sdk"
        assert spec.provider_options == {"platform": {"arch": "amd64"}}
        assert spec.files == {"/tmp/input": "data"}
        assert spec.workdir == "/testbed"
        overridden = AnySweAgent._sandbox_spec(params, files={"/tmp/input": "data"}, workdir="/workspace/auth0-python")
        assert overridden.workdir == "/workspace/auth0-python"

    def test_agent_config_is_forwarded_without_harness_specific_changes(self) -> None:
        params = SimpleNamespace(
            body=SimpleNamespace(model="model", temperature=1.0, top_p=0.95, max_output_tokens=None),
            agent_kwargs={"model": "configured-model", "custom": {"enabled": True}},
            model_server_url="http://model-host:8000/v1",
            agent_server_module="example.agent",
            agent_server_class="ExampleAgent",
            agent_config_class="ExampleAgentConfig",
            rollout_id="rollout-1",
            sandbox_model_base_url=None,
            agent_overlay_files={},
        )
        env = AnySweAgent._sandbox_agent_env(params)
        kwargs = json.loads(base64.b64decode(env["NGSWE_AGENT_KWARGS_B64"]))
        assert env["NGSWE_MODEL_NAME"] == "model"
        assert env["NGSWE_ROLLOUT_ID"] == "rollout-1"
        assert env["PYTHONPATH"] == ""
        assert kwargs == params.agent_kwargs

        params.agent_overlay_files = {"/trajectories_mount/agent_overlay/tools/example.py": "pass"}
        assert AnySweAgent._sandbox_agent_env(params)["PYTHONPATH"] == "/trajectories_mount/agent_overlay"

    def test_agent_error_classification_matches_swe_agents(self) -> None:
        assert _classify_agent_error("maximum iteration reached") == "max_iteration"
        assert _classify_agent_error("ContextWindowExceeded") == "context_window"
        assert _classify_agent_error("") is None

    def test_masking_matches_swe_agents(self) -> None:
        assert not _should_mask_sample(False, None, False, None)
        assert not _should_mask_sample(False, "max_iteration", False, None)
        assert _should_mask_sample(True, "max_iteration", False, None)
        assert _should_mask_sample(True, "context_window", False, None)
        assert not _should_mask_sample(True, "other", False, None)
        assert _should_mask_sample(False, None, True, None)
        assert _should_mask_sample(False, None, False, "eval_timeout")
        assert _should_mask_sample(False, None, False, "sandbox")
        assert _should_mask_sample(False, None, False, "grader")
        assert not _should_mask_sample(False, None, False, "eval_error")

    def test_dataset_routes(self) -> None:
        assert _dataset_family("princeton-nlp/SWE-bench_Verified") == "swebench"
        assert _dataset_family("SWE-bench_Multilingual") == "swebench_multilingual"
        assert _dataset_family("R2E-Gym/R2E-Gym-Subset") == "r2e"

    def test_r2e_required_tests(self) -> None:
        instance = {
            "FAIL_TO_PASS": ["tests/test_a.py::test_fix"],
            "PASS_TO_PASS": ["tests/test_b.py::test_stays_green"],
        }
        passing = "\n".join(
            [
                "PASSED tests/test_a.py::test_fix",
                "PASSED tests/test_b.py::test_stays_green",
            ]
        )
        failing = passing.replace("PASSED tests/test_a.py", "FAILED tests/test_a.py")
        skipped = passing.replace("PASSED tests/test_a.py", "SKIPPED tests/test_a.py")
        assert _r2e_resolved(instance, passing)
        assert not _r2e_resolved(instance, failing)
        assert not _r2e_resolved(instance, skipped)

    def test_safe_config_redacts_provider_key(self) -> None:
        class Params:
            def model_dump_json(self) -> str:
                return json.dumps(
                    {
                        "sandbox_provider": {"opensandbox": {"api_key": "secret"}},  # pragma: allowlist secret
                        "agent_runtime_source": "https://example.test/runtime?token=secret",
                        "agent_deps_url": "https://example.test/runtime?token=secret",
                    }
                )

        result = json.loads(_safe_config_json(Params()))
        assert result["sandbox_provider"]["opensandbox"]["api_key"] == "***"
        assert "agent_runtime_source" not in result
        assert "agent_deps_url" not in result


class TestAgentRuntimeOverlay:
    @staticmethod
    def _package(tmp_path: Path) -> Path:
        source = tmp_path / "openhands_env"
        (source / "sub").mkdir(parents=True)
        (source / "__init__.py").write_text('"""overlay package"""\n')
        (source / "app.py").write_text("VALUE = 1\n")
        (source / "sub" / "__init__.py").write_text("")
        (source / "notes.md").write_text("not a module\n")
        (source / "__pycache__").mkdir()
        (source / "__pycache__" / "app.cpython-313.pyc").write_bytes(b"stale")
        return source

    def test_only_packaged_modules_are_shipped(self, tmp_path) -> None:
        source = self._package(tmp_path)
        files = _overlay_files({"rl_tasks_openhands": str(source)})
        assert set(files) == {
            "/trajectories_mount/agent_overlay/rl_tasks_openhands/__init__.py",
            "/trajectories_mount/agent_overlay/rl_tasks_openhands/app.py",
            "/trajectories_mount/agent_overlay/rl_tasks_openhands/sub/__init__.py",
        }
        assert files["/trajectories_mount/agent_overlay/rl_tasks_openhands/app.py"] == "VALUE = 1\n"

    def test_package_typescript_extension(self, tmp_path) -> None:
        source = tmp_path / "extension"
        source.mkdir()
        (source / "tool.ts").write_text("export default function tool() {}\n")
        (source / "private.env").write_text("not a module")
        assert _overlay_files({"rl_tasks_extension": str(source)}) == {
            "/trajectories_mount/agent_overlay/rl_tasks_extension/tool.ts": "export default function tool() {}\n"
        }

    def test_identity_follows_content(self, tmp_path) -> None:
        source = self._package(tmp_path)
        before = _overlay_sha256(_overlay_files({"rl_tasks_openhands": str(source)}))
        assert before is not None and len(before) == 64
        assert _overlay_sha256({}) is None
        (source / "app.py").write_text("VALUE = 2\n")
        assert _overlay_sha256(_overlay_files({"rl_tasks_openhands": str(source)})) != before

    def test_selected_shared_core_module(self, tmp_path) -> None:
        module = tmp_path / "base_responses_api_agent.py"
        module.write_text("VALUE = 1\n")
        selected = _overlay_files({"nemo_gym/base_responses_api_agent.py": str(module)})
        assert selected == {"/trajectories_mount/agent_overlay/nemo_gym/base_responses_api_agent.py": "VALUE = 1\n"}

    def test_overlay_rejects_escaping_import_paths(self, tmp_path) -> None:
        import pytest

        source = self._package(tmp_path)
        for target in ("../nemo_gym", "/nemo_gym", "a/../../nemo_gym", "a\\b", "."):
            with pytest.raises(ValueError, match="Python import path"):
                _overlay_files({target: str(source)})

    def test_overlay_files_must_be_python_modules(self, tmp_path) -> None:
        import pytest

        source = tmp_path / "notes.md"
        source.write_text("not Python\n")
        with pytest.raises(ValueError, match="Python module"):
            _overlay_files({"nemo_gym/base.py": str(source)})

    def test_unset_and_missing_sources(self, tmp_path) -> None:
        import pytest

        assert _overlay_files({"rl_tasks_openhands": ""}) == {}
        with pytest.raises(ValueError, match="overlay directory not found"):
            _overlay_files({"rl_tasks_openhands": str(tmp_path / "absent")})
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="holds no modules"):
            _overlay_files({"rl_tasks_openhands": str(empty)})

    async def test_overlay_reaches_the_sandbox_import_path(self, tmp_path) -> None:
        from unittest.mock import AsyncMock, patch

        from omegaconf import OmegaConf

        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
        from nemo_gym.sandbox import SandboxExecResult
        from nemo_gym.server_utils import BaseServerConfig, ServerClient

        source = self._package(tmp_path)
        config = _config(
            model_server=None,
            agent_runtime_source="baked",
            sandbox_model_base_url="http://127.0.0.1:19002/v1",
            agent_runtime_overlay={"rl_tasks_openhands": str(source)},
        )
        head = BaseServerConfig(host="127.0.0.1", port=19000, name="head", entrypoint="app.py")
        client = ServerClient(
            head_server_config=head, global_config_dict=OmegaConf.create({"results_dir": str(tmp_path)})
        )
        agent = AnySweAgent(config=config, server_client=client)
        params = agent._setup_params(
            NeMoGymResponseCreateParamsNonStreaming(
                input="Fix the test.",
                metadata={
                    "dataset_name": "princeton-nlp/SWE-bench_Verified",
                    "instance_id": "test__test-1",
                    "instance_dict": json.dumps({"eval_script": "true"}),
                    "problem_statement": "Fix the test.",
                },
                model="zaya",
            )
        )
        sandbox = AsyncMock()
        sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0, error_type=None)

        with patch("responses_api_agents.anyswe_agent.app.AsyncSandbox", return_value=sandbox) as factory:
            await AnySweSandboxRunner(params).run()

        spec = factory.call_args.args[1]
        assert spec.files["/trajectories_mount/agent_overlay/rl_tasks_openhands/app.py"] == "VALUE = 1\n"
        runner_env = next(call.kwargs["env"] for call in sandbox.exec.await_args_list if call.kwargs.get("env"))
        assert runner_env["PYTHONPATH"] == "/trajectories_mount/agent_overlay"
        recorded = json.loads(_safe_config_json(params))
        assert "agent_overlay_files" not in recorded
        assert recorded["agent_overlay_sha256"] == _overlay_sha256(params.agent_overlay_files)
        assert recorded["agent_runtime_overlay"] == {"rl_tasks_openhands": str(source)}


class TestAgentRuntimeDirectory:
    @staticmethod
    def _runtime(tmp_path: Path) -> Path:
        runtime = tmp_path / "openhands-venv"
        (runtime / "bin").mkdir(parents=True)
        (runtime / "bin" / "python").write_text("#!/bin/sh\n")
        return runtime

    @staticmethod
    def _agent(tmp_path: Path, runtime: Path, provider: dict):
        from omegaconf import OmegaConf

        from nemo_gym.server_utils import BaseServerConfig, ServerClient

        config = _config(
            model_server=None,
            agent_runtime_source=str(runtime),
            sandbox_model_base_url="http://127.0.0.1:19002/v1",
            sandbox_provider=provider,
        )
        head = BaseServerConfig(host="127.0.0.1", port=19000, name="head", entrypoint="app.py")
        client = ServerClient(
            head_server_config=head, global_config_dict=OmegaConf.create({"results_dir": str(tmp_path)})
        )
        return AnySweAgent(config=config, server_client=client)

    @staticmethod
    def _params(agent):
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        return agent._setup_params(
            NeMoGymResponseCreateParamsNonStreaming(
                input="Fix the test.",
                metadata={
                    "dataset_name": "princeton-nlp/SWE-bench_Verified",
                    "instance_id": "test__test-1",
                    "instance_dict": json.dumps({"eval_script": "true"}),
                    "problem_statement": "Fix the test.",
                },
                model="zaya",
            )
        )

    def test_directory_needs_an_interpreter_and_a_binding_provider(self, tmp_path) -> None:
        import pytest

        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="lacks bin/python"):
            self._agent(tmp_path, empty, {"apptainer": {}})
        with pytest.raises(ValueError, match="apptainer provider"):
            self._agent(tmp_path, self._runtime(tmp_path), {"opensandbox": {}})

    async def test_directory_is_bind_mounted_not_copied(self, tmp_path) -> None:
        from unittest.mock import AsyncMock, patch

        runtime = self._runtime(tmp_path)
        agent = self._agent(tmp_path, runtime, {"apptainer": {}})
        params = self._params(agent)
        assert params.agent_deps_dir == runtime.resolve()
        assert params.agent_deps_archive is None and params.agent_deps_url is None
        sandbox = AsyncMock()
        sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0, error_type=None)
        with patch("responses_api_agents.anyswe_agent.app.AsyncSandbox", return_value=sandbox) as factory:
            await AnySweSandboxRunner(params).run()

        spec = factory.call_args.args[1]
        assert spec.provider_options["binds"] == [f"{runtime.resolve()}:/agent_deps_mount:ro"]
        sandbox.upload.assert_not_awaited()
        commands = [call.args[0] for call in sandbox.exec.await_args_list]
        assert "test -x /agent_deps_mount/bin/python" in commands
        assert any(
            command.endswith("/agent_deps_mount/bin/python /trajectories_mount/agent_runner.py")
            for command in commands
        )
        assert not any("tar -xzf" in command for command in commands)
        assert json.loads(_safe_config_json(params))["agent_deps_dir"] == str(runtime.resolve())

    def test_grading_spec_keeps_the_task_binds_only(self, tmp_path) -> None:
        agent = self._agent(tmp_path, self._runtime(tmp_path), {"apptainer": {}})
        spec = agent._sandbox_spec(self._params(agent), files={"/root/patch.diff": "x\n"})
        assert "binds" not in spec.provider_options


class TestSetupScriptsExist:
    def test_supported_agents_have_deps_scripts(self) -> None:
        scripts = Path(__file__).parent.parent / "setup_scripts"
        assert (scripts / "hermes_agent_deps.sh").exists()
        assert (scripts / "claude_code_agent_deps.sh").exists()
        assert (scripts / "cline_agent_deps.sh").exists()
        assert (scripts / "opencode_agent_deps.sh").exists()
        assert (scripts / "openclaw_agent_deps.sh").exists()
        assert (scripts / "pi_agent_deps.sh").exists()
        assert (scripts / "_portable_python.sh").exists()

    def test_portable_python_meets_project_minimum(self) -> None:
        script = (Path(__file__).parent.parent / "setup_scripts" / "_portable_python.sh").read_text()
        assert 'PYTHON_VERSION="${PYTHON_VERSION:-3.13.14}"' in script
        assert 'PBS_RELEASE="${PBS_RELEASE:-20260805}"' in script


class TestExampleData:
    def test_prepared_rows_do_not_set_sampling(self) -> None:
        row = _to_gym_row({"instance_id": "repo__repo-1", "problem_statement": "Fix it"}, "test")
        assert set(row["responses_create_params"]) == {"input", "metadata"}

    def test_example_jsonl_parses(self) -> None:
        example = Path(__file__).parent.parent / "data" / "example.jsonl"
        rows = [json.loads(line) for line in example.read_text().splitlines() if line.strip()]
        assert rows
        for row in rows:
            assert "metadata" in row["responses_create_params"]
            assert "instance_id" in row["responses_create_params"]["metadata"]


def test_configured_results_directory(tmp_path) -> None:
    from omegaconf import OmegaConf

    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_utils import BaseServerConfig, ServerClient

    head = BaseServerConfig(host="127.0.0.1", port=19000, name="head", entrypoint="app.py")
    config = _config(
        model_server=None, agent_runtime_source="baked", sandbox_model_base_url="http://127.0.0.1:19002/v1"
    )
    client = ServerClient(head_server_config=head, global_config_dict=OmegaConf.create({"results_dir": str(tmp_path)}))
    agent = AnySweAgent(config=config, server_client=client)
    params = agent._setup_params(
        NeMoGymResponseCreateParamsNonStreaming(
            input="Fix the test.", metadata={"instance_id": "test__test-1", "problem_statement": "Fix the test."}
        )
    )
    assert params.persistent_dir.is_relative_to(tmp_path)
    assert (params.persistent_dir / "instruction.txt").read_text() == "Fix the test."
    assert (params.persistent_dir / "agent_runner.py").is_file()
    for settings in ({}, {"results_dir": None}):
        fallback = AnySweAgent(
            config=config,
            server_client=ServerClient(head_server_config=head, global_config_dict=OmegaConf.create(settings)),
        )
        assert fallback._server.base_results_dir.parent == Path(__file__).parents[1]


async def test_sandbox_error_keeps_optional_model(tmp_path) -> None:
    import shutil

    import pytest
    from omegaconf import OmegaConf

    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_utils import BaseServerConfig, ServerClient

    binary = shutil.which("apptainer")
    if not binary:
        pytest.skip("requires the stock Apptainer executable")
    head = BaseServerConfig(host="127.0.0.1", port=19000, name="head", entrypoint="app.py")
    config = _config(
        model_server=None,
        agent_runtime_source="baked",
        sandbox_model_base_url="http://127.0.0.1:19002/v1",
        sandbox_provider={"apptainer": {"bin_path": str(Path(binary).parent)}},
    )
    client = ServerClient(head_server_config=head, global_config_dict=OmegaConf.create({"results_dir": str(tmp_path)}))
    agent = AnySweAgent(config=config, server_client=client)
    for model in (None, "requested-policy"):
        params = agent._setup_params(
            NeMoGymResponseCreateParamsNonStreaming(
                model=model,
                input="Fix the test.",
                metadata={
                    "instance_id": "test__test-1",
                    "problem_statement": "Fix the test.",
                    "image": str(tmp_path / "missing.sif"),
                },
            )
        )
        response = await AnySweSandboxRunner(params).run()
        assert response.model == (model or config.name)
        assert json.loads(response.metadata["metrics"])["error_kind"] == "sandbox"
        assert json.loads(response.metadata["metrics"])["mask_sample"]
        assert (params.persistent_dir / "sandbox_error.txt").is_file()


async def test_sandbox_error_detail_reaches_metrics(tmp_path) -> None:
    """The sandbox cause must survive in the metrics, not only in the discarded persistent_dir."""
    import shutil

    import pytest
    from omegaconf import OmegaConf
    from responses_api_agents.anyswe_agent.app import _MAX_ERROR_DETAIL_CHARS, _error_detail

    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_utils import BaseServerConfig, ServerClient

    binary = shutil.which("apptainer")
    if not binary:
        pytest.skip("requires the stock Apptainer executable")
    head = BaseServerConfig(host="127.0.0.1", port=19000, name="head", entrypoint="app.py")
    config = _config(
        model_server=None,
        agent_runtime_source="baked",
        sandbox_model_base_url="http://127.0.0.1:19002/v1",
        sandbox_provider={"apptainer": {"bin_path": str(Path(binary).parent)}},
    )
    client = ServerClient(head_server_config=head, global_config_dict=OmegaConf.create({"results_dir": str(tmp_path)}))
    agent = AnySweAgent(config=config, server_client=client)
    missing_image = tmp_path / "missing.sif"
    params = agent._setup_params(
        NeMoGymResponseCreateParamsNonStreaming(
            input="Fix the test.",
            metadata={
                "instance_id": "test__test-1",
                "problem_statement": "Fix the test.",
                "image": str(missing_image),
            },
        )
    )

    response = await AnySweSandboxRunner(params).run()

    metrics = json.loads(response.metadata["metrics"])
    assert metrics["error_kind"] == "sandbox"
    detail = metrics["error_detail"]
    # the cause, not just the bare token, and bounded so a traceback cannot bloat the metrics
    assert detail
    assert 0 < len(detail) <= _MAX_ERROR_DETAIL_CHARS
    assert "\n" not in detail
    # exactly the cause that previously reached only the discarded sandbox_error.txt
    assert detail == _error_detail((params.persistent_dir / "sandbox_error.txt").read_text())
    assert str(missing_image) in detail


@pytest.mark.parametrize(
    "stop_reason,grader_error,expected_mask",
    [
        ("max_iterations", None, False),
        (None, None, False),
        ("max_iterations", "eval_timeout", True),
        (None, "raised_runtime", True),
        (None, "raised_os_error", True),
    ],
)
async def test_expected_budget_stop_keeps_graded_patch(
    tmp_path, monkeypatch, episode_submissions, stop_reason, grader_error, expected_mask
):
    from omegaconf import OmegaConf
    from responses_api_agents.anyswe_agent import app as app_module

    from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_utils import BaseServerConfig, ServerClient

    head = BaseServerConfig(host="127.0.0.1", port=19000, name="head", entrypoint="app.py")
    config = _config(
        model_server=None,
        agent_runtime_source="baked",
        sandbox_model_base_url="http://127.0.0.1:19002/v1",
        sandbox_spec={"resources": {"cpu": 2, "memory_mib": 4096, "pids": 128}},
    )
    client = ServerClient(head_server_config=head, global_config_dict=OmegaConf.create({"results_dir": str(tmp_path)}))
    agent = AnySweAgent(config=config, server_client=client)
    body = NeMoGymResponseCreateParamsNonStreaming(
        model="test-model", input="Fix it", metadata={"instance_id": "test__test-1", "problem_statement": "Fix it"}
    )
    saved = NeMoGymResponse(
        id="budget-response",
        created_at=1,
        model="test-model",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        metadata={"stop_reason": stop_reason} if stop_reason else None,
    )
    patch = "diff --git a/fixed.txt b/fixed.txt\n+fixed\n"

    class ResultSandbox:
        def __init__(self, provider, spec):
            pass

        async def start(self):
            pass

        async def stop(self):
            pass

        async def exec(self, command, **kwargs):
            return SandboxExecResult("", "Agent reached maximum iterations limit (100).", 0, None)

        async def download(self, source, destination):
            destination.write_text(saved.model_dump_json() if source.endswith("response.json") else patch)

    graded = []
    boundaries = []

    async def policy_done(params):
        assert params is not None
        boundaries.append("policy_done")
        return True

    monkeypatch.setattr(app_module, "notify_policy_phase_done", policy_done)

    async def grade(self, params, received):
        assert boundaries == ["policy_done"]
        graded.append(received)
        if grader_error == "raised_runtime":
            raise RuntimeError("verifier cleanup failed")
        if grader_error == "raised_os_error":
            raise OSError("verifier output unavailable")
        return grader_error is None, grader_error, None

    monkeypatch.setattr(app_module, "AsyncSandbox", ResultSandbox)
    monkeypatch.setattr(AnySweSandboxRunner, "_grade_sandbox_patch", grade)
    result = await agent.run(AnySweRunRequest(responses_create_params=body))
    assert len(episode_submissions) == 1
    resources = episode_submissions[0]
    assert (resources.num_cpus, resources.memory_bytes, resources.pids) == (
        2,
        (4096 + EPISODE_MEMORY_HEADROOM_MIB) * 1024 * 1024,
        128,
    )
    assert graded == [patch]
    assert result.reward == (1.0 if grader_error is None else 0.0)
    assert result.mask_sample is expected_mask
    if grader_error in {"raised_runtime", "raised_os_error"}:
        assert result.error_kind == "grader"
        assert result.error_detail
    assert result.agent_stop_reason == stop_reason
    assert 0 < result.submitted_at <= result.task_started_at <= result.sandbox_ready_at <= result.grading_started_at <= result.grading_completed_at
    assert result.agent_error_kind == ("max_iteration" if stop_reason else None)


@pytest.mark.parametrize(
    "text,process_failed,declared_error,stop_reason,expected_kind,expected_mask",
    [
        ("maximum iterations reached", False, False, None, None, False),
        ("ContextWindowExceeded in the task fixture", False, False, None, None, False),
        ("stuck in a loop", False, False, None, None, False),
        ("maximum iterations reached", True, False, None, "max_iteration", True),
        ("ContextWindowExceeded", False, True, None, "context_window", True),
        ("ContextWindowExceeded", False, True, "max_iterations", "context_window", True),
        ("Context length exceeded (129,724 tokens). Cannot compress further.", False, True, "context_exhausted", "context_window", False),
    ],
)
async def test_only_declared_failures_affect_a_verified_patch(
    tmp_path, monkeypatch, episode_submissions, text, process_failed, declared_error, stop_reason, expected_kind, expected_mask
):
    from omegaconf import OmegaConf
    from responses_api_agents.anyswe_agent import app as app_module

    from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_utils import BaseServerConfig, ServerClient

    head = BaseServerConfig(host="127.0.0.1", port=19000, name="head", entrypoint="app.py")
    config = _config(
        model_server=None,
        agent_runtime_source="baked",
        sandbox_model_base_url="http://127.0.0.1:19002/v1",
        sandbox_spec={"resources": {"cpu": 2, "memory_mib": 4096, "pids": 128}},
    )
    client = ServerClient(head_server_config=head, global_config_dict=OmegaConf.create({"results_dir": str(tmp_path)}))
    agent = AnySweAgent(config=config, server_client=client)
    body = NeMoGymResponseCreateParamsNonStreaming(
        model="test-model", input="Fix it", metadata={"instance_id": "test__test-1", "problem_statement": "Fix it"}
    )
    saved = NeMoGymResponse(
        id="budget-response",
        created_at=1,
        model="test-model",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        metadata={"interrupted": "false", "failed": "false", "partial": "false", "turns": "100"}
        | ({"stop_reason": stop_reason} if stop_reason else {}),
        error={"code": "server_error", "message": text} if declared_error else None,
        instructions=text,
        status="failed" if declared_error else "completed",
    )
    patch = "diff --git a/fixed.txt b/fixed.txt\n+fixed\n"

    class ResultSandbox:
        def __init__(self, provider, spec):
            pass

        async def start(self):
            pass

        async def stop(self):
            pass

        async def exec(self, command, **kwargs):
            return SandboxExecResult(
                saved.model_dump_json(), text, 1 if process_failed and "agent_runner.py" in command else 0, None
            )

        async def download(self, source, destination):
            destination.write_text(saved.model_dump_json() if source.endswith("response.json") else patch)

    graded = []

    async def grade(self, params, received):
        graded.append(received)
        return True, None, None

    monkeypatch.setattr(app_module, "AsyncSandbox", ResultSandbox)
    monkeypatch.setattr(AnySweSandboxRunner, "_grade_sandbox_patch", grade)
    result = await agent.run(AnySweRunRequest(responses_create_params=body))
    assert len(episode_submissions) == 1
    resources = episode_submissions[0]
    assert (resources.num_cpus, resources.memory_bytes, resources.pids) == (
        2,
        (4096 + EPISODE_MEMORY_HEADROOM_MIB) * 1024 * 1024,
        128,
    )
    assert graded == [patch]
    assert result.reward == 1.0
    assert result.mask_sample is expected_mask
    assert result.agent_stop_reason == stop_reason
    assert result.agent_error_kind == expected_kind
    assert result.done_reason == ("context_exhausted" if stop_reason == "context_exhausted" else None)


@pytest.mark.parametrize(
    "agent_return_code,oom_kill,infrastructure,startup,max_events",
    [(0, 1, False, False, 9), (137, 2, False, False, 9), (137, 0, False, False, 9), (137, 2, True, True, 9)]
    + [pytest.param(raw["return_code"], raw["sandbox_oom_kill"], False, False,
                    raw["sandbox_memory_max_events"], id=raw["name"])
       for raw in json.loads((Path(__file__).parent / "fixtures/harness-memory-limit.json").read_text())],
)
async def test_memory_limit_death_after_harness_admission_is_zero_and_startup_is_infrastructure(
    tmp_path, monkeypatch, episode_submissions, agent_return_code, oom_kill, infrastructure, startup, max_events
):
    from omegaconf import OmegaConf

    from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_utils import BaseServerConfig, ServerClient
    from responses_api_agents.anyswe_agent import app as app_module

    head = BaseServerConfig(host="127.0.0.1", port=19000, name="head", entrypoint="app.py")
    config = _config(
        model_server=None,
        agent_runtime_source="baked",
        sandbox_model_base_url="http://127.0.0.1:19002/v1",
        sandbox_spec={"resources": {"cpu": 2, "memory_mib": 4096, "pids": 128}},
    )
    client = ServerClient(head_server_config=head, global_config_dict=OmegaConf.create({"results_dir": str(tmp_path)}))
    agent = AnySweAgent(config=config, server_client=client)
    body = NeMoGymResponseCreateParamsNonStreaming(
        model="test-model", input="Fix it", metadata={"instance_id": "test__test-1", "problem_statement": "Fix it"}
    )
    saved = NeMoGymResponse(
        id="r", created_at=1, model="test-model", object="response", output=[],
        parallel_tool_calls=False, tool_choice="auto", tools=[],
    )
    patch = "diff --git a/fixed.txt b/fixed.txt\n+fixed\n"

    class RetainedEventsProvider:
        """A provider whose sandbox cgroup kept memory.events at close, as the Apptainer provider does."""

        closed = False

        async def create(self, spec):
            return SandboxHandle(sandbox_id="box", provider_name="retained", raw=None)

        async def upload_file(self, handle, source, target):
            pass

        async def exec(self, handle, command, **kwargs):
            if startup and "mkdir -p" in command:
                raise RuntimeError("sandbox startup exhausted its memory limit before harness admission")
            code = agent_return_code if "agent_runner.py" in command else 0
            return SandboxExecResult(saved.model_dump_json(), "", code, None)

        async def download_file(self, handle, source, target):
            target.write_text(saved.model_dump_json() if source.endswith("response.json") else patch)

        async def close(self, handle):
            self.closed = True

        async def aclose(self):
            pass

        def retained_memory_events(self, handle):
            assert self.closed, "events are read after the sandbox is stopped"
            return {"low": 0, "high": 0, "max": max_events, "oom": oom_kill, "oom_kill": oom_kill}

    graded = []

    async def grade(self, params, received):
        graded.append(received)
        return True, None, None

    monkeypatch.setattr(app_module, "AsyncSandbox", lambda provider, spec: AsyncSandbox(RetainedEventsProvider(), spec))
    monkeypatch.setattr(AnySweSandboxRunner, "_grade_sandbox_patch", grade)
    result = await agent.run(AnySweRunRequest(responses_create_params=body))

    assert (result.sandbox_oom, result.sandbox_oom_kill, result.sandbox_memory_max_events) == (oom_kill, oom_kill, max_events)
    assert result.mask_sample is infrastructure
    if infrastructure:
        assert result.error_kind == "sandbox"
        assert "startup exhausted its memory limit" in result.error_detail
        assert graded == [] and result.reward == 0.0
    elif agent_return_code == 137 and oom_kill:
        assert result.error_kind == "agent_memory_limit"
        assert "in-harness causation unproven" in result.error_detail
        assert graded == [] and result.reward == 0.0
    else:
        assert result.error_kind is None
        assert graded == [patch] and result.reward == 1.0


@pytest.mark.parametrize("external", [False, True])
def test_anyswe_inner_agent_scopes_model_url_once(tmp_path, external):
    from omegaconf import OmegaConf

    from nemo_gym.config_types import ModelServerRef
    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_utils import BaseServerConfig, ServerClient

    head = BaseServerConfig(host="127.0.0.1", port=19000, name="head", entrypoint="app.py")
    global_config = OmegaConf.create(
        {
            "results_dir": str(tmp_path),
            "policy_model": {"responses_api_models": {"model": {"host": "127.0.0.1", "port": 19002}}},
        }
    )
    client = ServerClient(head_server_config=head, global_config_dict=global_config)
    config = AnySweAgentConfig(
        host="127.0.0.1",
        port=19001,
        name="anyswe",
        entrypoint="app.py",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        agent_server_module="responses_api_agents.hermes_agent.app",
        agent_server_class="HermesAgent",
        agent_config_class="HermesAgentConfig",
        container_formatter="test:{instance_id}",
        sandbox_provider={"opensandbox": {}},
        agent_runtime_source="baked",
        sandbox_model_base_url="http://external.example/proxy" if external else None,
    )
    outer = AnySweAgent(config=config, server_client=client)
    body = NeMoGymResponseCreateParamsNonStreaming(
        model="model",
        input="Fix the task",
        metadata={"instance_id": "test__task-1", "problem_statement": "Fix the task"},
    )
    params = outer._setup_params(body, rollout_id="episode-1")
    env = outer._sandbox_agent_env(params)
    # This is the existing agent_runner ServerClient binding, then the real
    # shared inner-agent resolver (inherited by Hermes and Pi). Exercise both ends, not the request object alone.
    inner_client = ServerClient(head_server_config=head, global_config_dict=global_config)
    inner_client._build_server_base_url = lambda config: env["NGSWE_MODEL_URL"]
    inner = AnySweAgent(config=config, server_client=inner_client)
    request = _agent_request(env.get("NGSWE_ROLLOUT_ID"))
    actual = inner.resolve_model_base_url("policy_model", request.path_params.get("rollout_id"))
    expected = "http://external.example/proxy/v1" if external else "http://127.0.0.1:19002/ng-rollout/episode-1/v1"
    assert actual == expected
