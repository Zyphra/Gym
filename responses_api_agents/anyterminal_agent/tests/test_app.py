# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Contracts for anyterminal_agent, including an opt-in allocated sandbox test.

Modeled on anyswe_agent's tests: these exercise pure logic (no real Apptainer/Docker) —
runner-script generation, provider selection, container discovery, deps-key derivation,
setup-script presence, and example-data shape. Heavy side effects in model_post_init (deps +
harness install) are bypassed by calling staticmethods/properties directly rather than
constructing the agent.
"""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from pydantic import ValidationError

from nemo_gym import PARENT_DIR
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult, SandboxHandle
from nemo_gym.sandbox.episode import (
    EPISODE_MEMORY_HEADROOM_MIB,
    harness_memory_kill,
    sandbox_episode_resources,
    verifier_memory_kill,
)
from nemo_gym.sandbox.providers.apptainer import ApptainerProvider
from nemo_gym.sandbox.providers.apptainer import provider as apptainer_provider
from nemo_gym.sandbox.providers.docker import DockerProvider
from nemo_gym.sandbox.providers.docker import provider as docker_provider
from nemo_gym.server_utils import ServerClient
from responses_api_agents.anyterminal_agent import app
from responses_api_agents.anyterminal_agent.app import (
    _RUNNER_TEMPLATE,
    AnyTerminalAgent,
    AnyTerminalAgentConfig,
    AnyTerminalInstanceConfig,
    AnyTerminalServerConfig,
    GymAgentHarnessProcessor,
    RunTerminalAgent,
    _build_provider,
    _episode_sandbox_resources,
    _file_lock,
    _format_container,
    _instruction_from_input,
    _read_task_meta,
    _safe_config_json,
    update_metrics,
)


def test_shared_episode_runtime_env_forwards_gym_source(monkeypatch: pytest.MonkeyPatch) -> None:
    pythonpath = os.pathsep.join((str(PARENT_DIR), "/existing/path"))
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", "/tasks")

    import sys

    from core.sandbox.executor import _episode_runtime_env

    monkeypatch.setenv("RAY_process_group_cleanup_enabled", "1")
    monkeypatch.setenv("NEMO_GYM_SERVICE_PYTHON", sys.executable)
    runtime_env = _episode_runtime_env()

    assert runtime_env["py_executable"]
    assert runtime_env["env_vars"]["PYTHONPATH"] == pythonpath
    assert runtime_env["env_vars"]["NEMO_GYM_EXTRA_ROOTS"] == "/tasks"
    assert runtime_env["env_vars"]["RAY_process_group_cleanup_enabled"] == "1"


def test_shared_episode_python_survives_submit_environment_removal(tmp_path, monkeypatch):
    import subprocess
    import sys
    import sysconfig
    import venv

    submit_env = tmp_path / "submit-node-only"
    venv.EnvBuilder(system_site_packages=True, with_pip=False).create(submit_env)
    pythonpath = os.pathsep.join(
        (str(PARENT_DIR), sysconfig.get_paths()["purelib"], os.environ.get("PYTHONPATH", ""))
    )
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.setenv("NEMO_GYM_SERVICE_PYTHON", sys.executable)
    monkeypatch.setenv("RAY_process_group_cleanup_enabled", "1")
    probe = """
import json, os, sys
from core.sandbox.executor import _episode_runtime_env
configured = _episode_runtime_env()
print(json.dumps({'configured': configured, 'python': sys.executable}))
"""
    result = json.loads(subprocess.check_output(
        [str(submit_env / "bin/python"), "-c", probe], text=True,
    ))
    assert result["configured"]["py_executable"] == sys.executable
    shutil.rmtree(submit_env)
    assert not Path(result["python"]).exists()
    env = {**os.environ, **result["configured"]["env_vars"]}
    worker_probe = """
from pathlib import Path
import ray
from nemo_gym.sandbox.providers.apptainer import provider
path = Path(provider.__file__).resolve()
print('worker imports PASS')
"""
    output = subprocess.check_output(
        [result["configured"]["py_executable"], "-c", worker_probe], env=env, text=True,
    )
    assert "worker imports PASS" in output

def _config(**overrides) -> AnyTerminalAgentConfig:
    base = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="anyterminal_agent",
        model_server={"type": "responses_api_models", "name": "policy_model"},
        agent_server_module="responses_api_agents.hermes_agent.app",
        agent_server_class="HermesAgent",
        agent_config_class="HermesAgentConfig",
    )
    base.update(overrides)
    return AnyTerminalAgentConfig(**base)


class TestRunnerTemplate:
    def _render(self) -> str:
        return _RUNNER_TEMPLATE.format(
            agent_module="responses_api_agents.hermes_agent.app",
            agent_class="HermesAgent",
            agent_cfg_class="HermesAgentConfig",
            agent_class_lower="hermesagent",
        )

    def test_renders_valid_python(self) -> None:
        rendered = self._render()
        # Must be syntactically valid Python and reference the agent class.
        compile(rendered, "<runner>", "exec")
        assert "HermesAgent(config=config" in rendered
        assert 'Request({"type": "http", "path_params": {}})' in rendered
        assert 'object.__setattr__(agent, "resolve_model_base_url"' in rendered

    def test_response_is_written_back(self) -> None:
        # The runner's agent-agnostic contract is to persist the response where the host reads it.
        assert "/trajectories_mount/response.json" in _RUNNER_TEMPLATE
        assert "response.model_dump_json()" in _RUNNER_TEMPLATE

    def test_sampling_is_forwarded(self) -> None:
        rendered = self._render()
        compile(rendered, "<runner>", "exec")
        # Read from env, forwarded onto the body, and filtered to the agent config's fields.
        assert "NGTB_SAMPLING" in rendered
        assert "**SAMPLING," in rendered
        assert "HermesAgentConfig.model_fields" in rendered


class TestAgentKey:
    def test_key_from_module(self) -> None:
        proc = GymAgentHarnessProcessor(config=_config())
        assert proc._agent_key == "hermes_agent"

    def test_key_for_claude(self) -> None:
        proc = GymAgentHarnessProcessor(
            config=_config(
                agent_server_module="responses_api_agents.claude_code_agent.app",
                agent_server_class="ClaudeCodeAgent",
                agent_config_class="ClaudeCodeAgentConfig",
            )
        )
        assert proc._agent_key == "claude_code_agent"


class TestFormatContainer:
    def test_default_template(self) -> None:
        assert _format_container(None, "t", "ubuntu:22.04") == "docker://ubuntu:22.04"

    def test_list_formatter_uses_first(self) -> None:
        assert _format_container(["docker://{docker_image}", "unused"], "t", "ubuntu:22.04") == "docker://ubuntu:22.04"

    def test_strips_existing_docker_scheme_from_image(self) -> None:
        assert _format_container("docker://{docker_image}", "t", "docker://ubuntu:22.04") == "docker://ubuntu:22.04"

    def test_task_name_placeholder(self) -> None:
        assert _format_container("docker://reg/{task_name}", "my-task", "ubuntu:22.04") == "docker://reg/my-task"

    def test_dotted_path_treated_as_local(self) -> None:
        assert _format_container("./images/{task_name}.sif", "t", "ubuntu:22.04") == "./images/t.sif"


@pytest.mark.parametrize("image", [None, "", "relative.sif", "docker://ubuntu:22.04"])
def test_admitted_image_formatter_rejects_missing_or_nonlocal_image(image):
    with pytest.raises(ValueError, match="absolute admitted image"):
        _format_container("{runtime_image}", "t", "ubuntu:22.04", runtime_image=image)


def test_admitted_image_does_not_change_default_formatter():
    assert _format_container(None, "t", "ubuntu:22.04", "/artifacts/task.sif") == "docker://ubuntu:22.04"


class TestSetupScriptsExist:
    def test_supported_agents_have_deps_scripts(self) -> None:
        assert (PARENT_DIR / "responses_api_agents" / "hermes_agent" / "scripts" / "hermes_agent_deps.sh").exists()
        assert (Path(__file__).parent.parent / "setup_scripts" / "_portable_python.sh").exists()


class TestExampleData:
    def _example(self) -> Path:
        # Data files are gitignored; test whichever materialized example is present, else skip.
        data_dir = Path(__file__).parent.parent / "data"
        for name in ("terminal_bench_smoke.jsonl", "terminal_bench_example.jsonl"):
            p = data_dir / name
            if p.exists():
                return p
        candidates = sorted(data_dir.glob("*.jsonl"))
        return candidates[0] if candidates else data_dir / "missing.jsonl"

    def test_example_jsonl_parses(self) -> None:
        example = self._example()
        if not example.exists():
            pytest.skip("no example .jsonl present (data/ is gitignored)")
        rows = [json.loads(line) for line in example.read_text().splitlines() if line.strip()]
        assert rows
        for row in rows:
            rcp = row["responses_create_params"]
            assert "input" in rcp
            assert "metadata" in rcp
            md = rcp["metadata"]
            assert "task_name" in md or "instance_id" in md


# ── helpers ───────────────────────────────────────────────────────────────────────


def _make_body(content: str = "solve this", **kwargs) -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "user", "content": content}], model="test-model", **kwargs
    )


def _make_instance_config(tmp_path: Path, **overrides) -> AnyTerminalInstanceConfig:
    persistent_dir = tmp_path / "persistent"
    persistent_dir.mkdir(parents=True, exist_ok=True)
    verifier_dir = tmp_path / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    metrics_fpath = tmp_path / "metrics.json"
    metrics_fpath.write_text("{}")
    defaults: dict = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="anyterminal_agent",
        model_server=None,
        agent_server_module="responses_api_agents.hermes_agent.app",
        agent_server_class="HermesAgent",
        agent_config_class="HermesAgentConfig",
        run_session_id="test_session",
        base_results_dir=tmp_path / "results",
        model_server_url="",
        model_name="test-model",
        nemo_gym_root=PARENT_DIR,
        agent_deps_dir=tmp_path / "deps",
        problem_info={
            "task_name": "fix-git",
            "task_dir": str(tmp_path),
            "docker_image": "ubuntu:22.04",
            "cpus": 2,
            "memory_mb": 4096,
            "pids": 512,
            "gpus": 0,
        },
        body=_make_body(),
        persistent_dir=persistent_dir,
        verifier_dir=verifier_dir,
        agent_run_id="fix-git_12345",
        metrics_fpath=metrics_fpath,
        container="docker://ubuntu:22.04",
        ray_queue_timestamp=0.0,
    )
    defaults.update(overrides)
    return AnyTerminalInstanceConfig(**defaults)


def _verifier_runtime(tmp_path):
    helper = tmp_path / "helper"
    cache = tmp_path / "cache"
    helper.mkdir()
    cache.mkdir()
    payload = b'{"schema":1}\n'
    (helper / "helper-manifest.json").write_bytes(payload)
    (cache / "verifier-cache.json").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    return app.VerifierRuntime(
        helper_dir=helper, cache_dir=cache,
        helper_manifest_sha256=digest, cache_manifest_sha256=digest,
    )


@pytest.mark.parametrize("provider", ["docker", "apptainer"])
def test_verifier_runtime_mounts_resolved_payloads_read_only(tmp_path, provider, monkeypatch):
    monkeypatch.setattr(apptainer_provider, "_require_apptainer", lambda _bin_path=None: "/usr/bin/apptainer")
    monkeypatch.setattr(docker_provider, "_require_docker", lambda: "/usr/bin/docker")
    runtime = _verifier_runtime(tmp_path)
    cfg = _make_instance_config(tmp_path, verifier_runtime=runtime, sandbox_provider={provider: {}})
    actual = _build_provider(cfg)
    if provider == "apptainer":
        binds = actual._exec_config.default_binds
    else:
        binds = actual._create_config.extra_run_args
    assert f"{runtime.helper_dir}:/staging/env_files:ro" in binds
    assert f"{runtime.cache_dir}:/verifier_cache_mount:ro" in binds


@pytest.mark.parametrize("change", ["symlink", "digest", "missing", "relative", "unknown"])
def test_verifier_runtime_rejects_unbound_payloads(tmp_path, change):
    runtime = _verifier_runtime(tmp_path)
    values = runtime.model_dump()
    if change == "symlink":
        linked = tmp_path / "linked"
        linked.symlink_to(runtime.helper_dir, target_is_directory=True)
        values["helper_dir"] = linked
    elif change == "digest":
        values["cache_manifest_sha256"] = "0" * 64
    elif change == "missing":
        (runtime.cache_dir / "verifier-cache.json").unlink()
    elif change == "relative":
        values["helper_dir"] = Path("helper")
    else:
        values["setup_command"] = "arbitrary setup is not an input"
    with pytest.raises(ValidationError):
        app.VerifierRuntime(**values)


@pytest.mark.asyncio
async def test_absent_verifier_runtime_never_provisions(tmp_path):
    cfg = _make_instance_config(tmp_path)
    sandbox = AsyncMock()
    assert await RunTerminalAgent(config=cfg)._prepare_verifier_runtime(sandbox, cfg) is None
    sandbox.exec.assert_not_called()


@pytest.mark.asyncio
async def test_verifier_environment_is_private_and_passed_to_grading(tmp_path):
    cfg = _make_instance_config(tmp_path, verifier_runtime=_verifier_runtime(tmp_path))
    sandbox = AsyncMock()
    result = SimpleNamespace(return_code=0, error_type=None, stdout="", stderr="")
    sandbox.exec.side_effect = [SimpleNamespace(
        return_code=0, error_type=None, stderr="", stdout=json.dumps({
            "HOME": "/root", "PATH": "/task/bin:/usr/bin", "python_key": "cpython-3.13.7-linux-x86_64-gnu",
        }),
    ), result, result]
    runner = RunTerminalAgent(config=cfg)
    await runner._run_eval(sandbox, cfg)
    inspection, setup, grading = sandbox.exec.await_args_list
    env = setup.kwargs["env"]
    assert grading.kwargs["env"] == env
    assert "HOME" not in env
    apptainer_provider._serialize_env_file(env)
    assert env["UV_CACHE_DIR"].startswith("/tmp/gym-verifier-")
    assert env["UV_OFFLINE"] == "true"
    assert env["UV_PYTHON_DOWNLOADS"] == "never"
    assert env["PATH"].index("/task/bin") < env["PATH"].index("verifier-python/bin")
    assert env["UV_PYTHON_PREFERENCE"] == "only-managed"
    receipt = json.loads((cfg.verifier_dir / "runtime-setup.json").read_text())
    assert receipt["runtime"]["cache_manifest_sha256"] == cfg.verifier_runtime.cache_manifest_sha256
    assert receipt["environment"]["HOME"] == "/root"
    assert {key: value for key, value in receipt["environment"].items() if key != "HOME"} == env


@pytest.mark.asyncio
@pytest.mark.parametrize("home", ["/tmp/private-verifier/home", "/tmp/private ' $(false); home"])
async def test_private_home_survives_apptainer_environment_boundary(tmp_path, home):
    import subprocess

    cfg = _make_instance_config(tmp_path)
    async def execute(command, *, env, timeout_s, user):
        apptainer_provider._serialize_env_file(env)
        assert "HOME" not in env
        return subprocess.run(["sh", "-c", command], env=os.environ | env,
                              capture_output=True, text=True, timeout=timeout_s, check=True)
    sandbox = SimpleNamespace(exec=execute)
    result = await RunTerminalAgent(config=cfg)._exec_verifier(
        sandbox, 'printf "%s\n%s" "$HOME" "$UV_OFFLINE"',
        env={"HOME": home, "UV_OFFLINE": "true"}, timeout_s=5,
    )
    assert result.stdout == home + "\ntrue"


@pytest.mark.asyncio
async def test_verifier_setup_failure_preserves_error_and_never_grades(tmp_path):
    cfg = _make_instance_config(tmp_path, verifier_runtime=_verifier_runtime(tmp_path))
    sandbox = AsyncMock()
    sandbox.exec.return_value = SimpleNamespace(return_code=1, error_type=None, stdout="", stderr="cache copy failed")
    with pytest.raises(RuntimeError, match="setup failed"):
        await RunTerminalAgent(config=cfg)._run_eval(sandbox, cfg)
    assert sandbox.exec.await_count == 1
    assert json.loads((cfg.verifier_dir / "runtime-setup.json").read_text())["stderr"] == "cache copy failed"


@pytest.mark.parametrize("value", ["true", "false", 1, None])
def test_verifier_network_isolation_requires_boolean(tmp_path, value):
    runtime = _verifier_runtime(tmp_path)
    assert runtime.isolate_network is False
    with pytest.raises(ValidationError):
        app.VerifierRuntime(**(runtime.model_dump() | {"isolate_network": value}))


@pytest.mark.asyncio
async def test_verifier_namespace_failure_never_runs_command(tmp_path):
    import subprocess

    runtime = _verifier_runtime(tmp_path).model_copy(update={"isolate_network": True})
    cfg = _make_instance_config(tmp_path, verifier_runtime=runtime)
    # Execute a failing namespace tool, not a mocked sandbox return value.
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "unshare").symlink_to(shutil.which("false"))
    marker = tmp_path / "must-not-exist"

    async def execute(command, *, env, timeout_s, user):
        result = subprocess.run(["/bin/sh", "-c", command], env=os.environ | (env or {}),
                                capture_output=True, text=True, timeout=timeout_s)
        return SimpleNamespace(return_code=result.returncode, error_type=None,
                               stdout=result.stdout, stderr=result.stderr)

    result = await RunTerminalAgent(config=cfg)._exec_verifier(
        SimpleNamespace(exec=execute), "touch " + str(marker),
        env={"PATH": str(tools)}, timeout_s=5,
    )
    assert result.return_code != 0
    assert not marker.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("isolated", [False, True])
async def test_verifier_network_namespace_preserves_task_files(tmp_path, isolated):
    from nemo_gym.sandbox import AsyncSandbox, SandboxSpec

    sif = os.environ.get("ANYTERMINAL_TEST_SIF")
    if not sif:
        pytest.skip("allocated Apptainer sandbox requires ANYTERMINAL_TEST_SIF")
    assert Path(sif).is_file()
    helper = Path(os.environ["ANYTERMINAL_TEST_VERIFIER_HELPER"])
    assert (helper / "python/verifier-python/bin/python3").is_file()
    runtime = _verifier_runtime(tmp_path).model_copy(update={
        "isolate_network": isolated, "helper_dir": helper,
        "helper_manifest_sha256": hashlib.sha256((helper / "helper-manifest.json").read_bytes()).hexdigest(),
    })
    (runtime.cache_dir / "cache").mkdir()
    cfg = _make_instance_config(
        tmp_path, verifier_runtime=runtime, container=sif,
        sandbox_provider={"apptainer": {"create": {"extra_start_args": ["--fakeroot"]}}},
    )
    cfg.agent_deps_dir.mkdir()
    sandbox = AsyncSandbox(_build_provider(cfg), SandboxSpec(image=sif, ttl_s=60))
    try:
        await sandbox.start()
        runner = RunTerminalAgent(config=cfg)
        task_identity = await sandbox.exec(
            "printf '%s\\n' \"$HOME\" \"$PATH\"; command -v python3 || true",
            timeout_s=10, user="root",
        )
        assert task_identity.return_code == 0, task_identity.stderr
        task_home, task_path, *task_python = task_identity.stdout.splitlines()
        marker = await sandbox.exec(
            "mkdir -p \"$HOME/.local/bin\"; "
            "printf '# retained task environment\\n' > \"$HOME/.local/bin/env\"; "
            "printf task-home-data > \"$HOME/verifier-home-marker\"",
            timeout_s=10, user="root",
        )
        assert marker.return_code == 0, marker.stderr
        verifier_env = await runner._prepare_verifier_runtime(sandbox, cfg)
        assert verifier_env["HOME"] == task_home
        assert task_path in verifier_env["PATH"]
        environment_probe = await runner._exec_verifier(
            sandbox, 'cat "$HOME/.local/bin/env" "$HOME/verifier-home-marker"; '
            "printf '\\n'; command -v python3; uv python find --offline 3.13",
            env=verifier_env, timeout_s=20,
        )
        assert environment_probe.return_code == 0, environment_probe.stderr
        probe_lines = environment_probe.stdout.splitlines()
        assert probe_lines[:2] == ["# retained task environment", "task-home-data"]
        if task_python:
            assert probe_lines[2] == task_python[0]
        else:
            assert probe_lines[2].startswith("/staging/env_files/python/verifier-python/bin/")
        assert "/python/cpython-3.13." in probe_lines[3] or probe_lines[3].startswith(
            "/staging/env_files/python/verifier-python/bin/"
        )
        system_tools = await runner._exec_verifier(
            sandbox, "ldconfig --version && start-stop-daemon --version",
            env=verifier_env, timeout_s=10,
        )
        assert system_tools.return_code == 0, system_tools.stderr
        parent = await sandbox.exec(
            "readlink /proc/self/ns/net; printf task-data > /tmp/verifier-network-test",
            timeout_s=10, user="root",
        )
        assert parent.return_code == 0, parent.stderr
        import shlex

        roundtrip = (
            "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); s.listen(1); "
            "c=socket.socket(); c.settimeout(2); c.connect(s.getsockname()); "
            "p,_=s.accept(); c.sendall(b'loopback'); print(p.recv(8).decode())"
        )
        result = await RunTerminalAgent(config=cfg)._exec_verifier(
            sandbox, "readlink /proc/self/ns/net; cat /tmp/verifier-network-test; "
            "printf '\n'; /staging/env_files/python/verifier-python/bin/python3 -I -c "
            + shlex.quote(roundtrip) + "; cat /proc/net/route",
            env={"HOME": "/tmp/private-verifier"}, timeout_s=10,
        )
        assert result.return_code == 0, result.stderr
        lines = result.stdout.splitlines()
        assert (lines[0] != parent.stdout.strip()) is isolated
        assert lines[1] == "task-data"
        assert lines[2] == "loopback"
        if isolated:
            assert not [line for line in lines[3:] if line.strip() and not line.startswith("Iface")]
    finally:
        await sandbox.stop()


# ── _read_task_meta ───────────────────────────────────────────────────────────────


class TestReadTaskMeta:
    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        assert _read_task_meta(tmp_path / "nonexistent") == {}

    def test_reads_timeouts_from_toml(self, tmp_path: Path) -> None:
        (tmp_path / "task.toml").write_bytes(b"[agent]\ntimeout_sec = 1800\n[verifier]\ntimeout_sec = 300\n")
        result = _read_task_meta(tmp_path)
        assert result["agent_timeout_sec"] == 1800
        assert result["verifier_timeout_sec"] == 300

    def test_reads_workdir_from_dockerfile(self, tmp_path: Path) -> None:
        (tmp_path / "environment").mkdir()
        (tmp_path / "environment" / "Dockerfile").write_text("FROM ubuntu:22.04\nWORKDIR /app\n")
        assert _read_task_meta(tmp_path).get("workdir") == "/app"

    def test_last_workdir_wins(self, tmp_path: Path) -> None:
        (tmp_path / "environment").mkdir()
        (tmp_path / "environment" / "Dockerfile").write_text("FROM ubuntu\nWORKDIR /first\nWORKDIR /workspace\n")
        assert _read_task_meta(tmp_path)["workdir"] == "/workspace"

    def test_toml_without_sections_returns_nones(self, tmp_path: Path) -> None:
        (tmp_path / "task.toml").write_bytes(b"[environment]\ndocker_image = 'ubuntu:22.04'\n")
        result = _read_task_meta(tmp_path)
        assert result.get("agent_timeout_sec") is None
        assert result.get("verifier_timeout_sec") is None


# ── AnyTerminalAgent._setup_params ──────────────────────────────────────────────────


def _make_setup_agent(tmp_path: Path, **config_overrides) -> AnyTerminalAgent:
    # model_post_init has heavy side effects (deps install, provider resolution) that
    # _setup_params doesn't touch, so bypass it and set only what _setup_params reads.
    with patch.object(AnyTerminalAgent, "model_post_init", lambda self, context: None):
        agent = AnyTerminalAgent(config=_config(**config_overrides), server_client=MagicMock(spec=ServerClient))
    agent._server = AnyTerminalServerConfig(
        run_session_id="test_session",
        base_results_dir=tmp_path / "results",
        model_server_url="",
        nemo_gym_root=PARENT_DIR,
        agent_deps_dir=tmp_path,
    )
    return agent


class TestSetupParams:
    def test_distinct_admitted_images_cross_real_request_metadata(self, tmp_path):
        agent = _make_setup_agent(tmp_path, container_formatter="{runtime_image}")
        images = [tmp_path / "first.sif", tmp_path / "second.sif"]
        containers = []
        for index, image in enumerate(images):
            body = _make_body(metadata={
                "task_name": f"task-{index}", "task_dir": str(tmp_path),
                "docker_image": "ubuntu:22.04", "runtime_image": str(image),
            })
            params = agent._setup_params(body)
            containers.append(params.container)
            assert params.problem_info["docker_image"] == "ubuntu:22.04"
        assert containers == [str(image) for image in images]

    def test_admitted_image_formatter_requires_row_metadata(self, tmp_path):
        agent = _make_setup_agent(tmp_path, container_formatter="{runtime_image}")
        body = _make_body(metadata={"task_name": "task", "task_dir": str(tmp_path)})
        with pytest.raises(ValueError, match="absolute admitted image"):
            agent._setup_params(body)

    def test_verifier_runtime_crosses_real_request_metadata(self, tmp_path):
        runtime = _verifier_runtime(tmp_path)
        agent = _make_setup_agent(tmp_path)
        body = _make_body(metadata={
            "task_name": "fix-git", "task_dir": str(tmp_path),
            "verifier_runtime": runtime.model_dump_json(),
        })
        params = agent._setup_params(body)
        assert params.verifier_runtime == runtime
        assert params.body.metadata["verifier_runtime"] == runtime.model_dump_json()

    @pytest.mark.parametrize("carrier", ["not-json", "{}", "null"])
    def test_verifier_runtime_rejects_invalid_request_metadata(self, tmp_path, carrier):
        agent = _make_setup_agent(tmp_path)
        body = _make_body(metadata={
            "task_name": "fix-git", "task_dir": str(tmp_path), "verifier_runtime": carrier,
        })
        with pytest.raises(ValidationError):
            agent._setup_params(body)

    def test_uses_per_task_timeout_by_default(self, tmp_path: Path) -> None:
        agent = _make_setup_agent(tmp_path)
        body = _make_body(metadata={"task_name": "fix-git", "task_dir": str(tmp_path), "agent_timeout_sec": "900"})

        params = agent._setup_params(body)

        assert params.tb_agent_timeout == 900

    def test_global_agent_timeout_overrides_per_task_timeout(self, tmp_path: Path) -> None:
        agent = _make_setup_agent(tmp_path, global_agent_timeout=7200)
        body = _make_body(metadata={"task_name": "fix-git", "task_dir": str(tmp_path), "agent_timeout_sec": "900"})

        params = agent._setup_params(body)

        assert params.tb_agent_timeout == 7200

    def test_global_agent_timeout_applies_without_per_task_timeout(self, tmp_path: Path) -> None:
        agent = _make_setup_agent(tmp_path, global_agent_timeout=7200)
        body = _make_body(metadata={"task_name": "fix-git", "task_dir": str(tmp_path)})

        params = agent._setup_params(body)

        assert params.tb_agent_timeout == 7200

    def test_sandbox_ttl_is_derived_when_timeouts_would_outlive_it(self, tmp_path: Path) -> None:
        agent = _make_setup_agent(tmp_path, global_agent_timeout=12500)
        body = _make_body(metadata={"task_name": "fix-git", "task_dir": str(tmp_path), "verifier_timeout_sec": "900"})

        params = agent._setup_params(body)

        assert params.tb_sandbox_ttl == 12500 + 900 + 600

    def test_sandbox_ttl_default_kept_when_already_sufficient(self, tmp_path: Path) -> None:
        agent = _make_setup_agent(tmp_path)
        body = _make_body(metadata={"task_name": "fix-git", "task_dir": str(tmp_path), "agent_timeout_sec": "900"})

        params = agent._setup_params(body)

        assert params.tb_sandbox_ttl == 7200

    def test_global_agent_timeout_rejects_zero(self) -> None:
        with pytest.raises(ValidationError, match="global_agent_timeout"):
            _config(global_agent_timeout=0)


# ── _instruction_from_input ───────────────────────────────────────────────────────


class TestInstructionFromInput:
    def _body(self, input_val):
        return SimpleNamespace(input=input_val)

    def test_str_input_passthrough(self) -> None:
        assert _instruction_from_input(self._body("hello world")) == "hello world"

    def test_dict_messages_joined(self) -> None:
        msgs = [{"role": "user", "content": "part1"}, {"role": "user", "content": "part2"}]
        assert _instruction_from_input(self._body(msgs)) == "part1\npart2"

    def test_object_messages(self) -> None:
        msgs = [SimpleNamespace(content="foo"), SimpleNamespace(content="bar")]
        assert _instruction_from_input(self._body(msgs)) == "foo\nbar"

    def test_content_part_list(self) -> None:
        msg = {"content": [{"type": "text", "text": "hello"}, {"type": "text", "text": " world"}]}
        assert _instruction_from_input(self._body([msg])) == "hello world"

    def test_none_content_skipped(self) -> None:
        msgs = [{"role": "user", "content": None}, {"role": "user", "content": "hi"}]
        assert _instruction_from_input(self._body(msgs)) == "hi"

    def test_empty_list(self) -> None:
        assert _instruction_from_input(self._body([])) == ""


# ── update_metrics ────────────────────────────────────────────────────────────────


class TestUpdateMetrics:
    def test_merges_with_existing(self, tmp_path: Path) -> None:
        fpath = tmp_path / "m.json"
        fpath.write_text('{"resolved": true}')
        update_metrics(fpath, {"total_run_time": 5.0})
        d = json.loads(fpath.read_text())
        assert d["resolved"] is True
        assert d["total_run_time"] == 5.0

    def test_none_values_excluded(self, tmp_path: Path) -> None:
        fpath = tmp_path / "m.json"
        fpath.write_text("{}")
        update_metrics(fpath, {"resolved": None, "total_run_time": 3.0})
        d = json.loads(fpath.read_text())
        assert "resolved" not in d
        assert d["total_run_time"] == 3.0

    def test_overwrites_existing_key(self, tmp_path: Path) -> None:
        fpath = tmp_path / "m.json"
        fpath.write_text('{"resolved": false}')
        update_metrics(fpath, {"resolved": True})
        assert json.loads(fpath.read_text())["resolved"] is True

    def test_missing_file_records_the_update(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """The instance directory can go away under the episode; its metrics still have to land."""
        fpath = tmp_path / "gone" / "nemo_gym_metrics.json"
        update_metrics(fpath, {"sandbox_failed": True, "mask_sample": True}, "fix-git")
        d = json.loads(fpath.read_text())
        assert d["sandbox_failed"] is True
        assert d["mask_sample"] is True
        out = capsys.readouterr().out
        assert "fix-git" in out
        assert str(fpath) in out

    def test_unparseable_file_records_the_update(self, tmp_path: Path) -> None:
        fpath = tmp_path / "m.json"
        fpath.write_text("{truncated")
        update_metrics(fpath, {"resolved": False, "sandbox_failed": True})
        assert json.loads(fpath.read_text()) == {"resolved": False, "sandbox_failed": True}


# ── _safe_config_json ─────────────────────────────────────────────────────────────


class TestSafeConfigJson:
    def test_api_key_redacted(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, agent_kwargs={"api_key": "sk-secret", "model": "gpt-4"})
        result = json.loads(_safe_config_json(cfg))
        assert result["agent_kwargs"]["api_key"] == "***"
        assert result["agent_kwargs"]["model"] == "gpt-4"

    def test_secret_key_variants_redacted(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(
            tmp_path,
            agent_kwargs={
                "my_secret": "x",
                "auth_token": "y",
                "hf_token": "hf",
                "password": "z",
                "anthropic_api_key": "a",
                "apiKey": "b",
                "aws_secret_access_key": "c",
                "password_hash": "d",
                "api_key_id": "e",
            },
        )
        result = json.loads(_safe_config_json(cfg))
        secret_keys = (
            "my_secret",
            "auth_token",
            "hf_token",
            "password",
            "anthropic_api_key",
            "apiKey",
            "aws_secret_access_key",
            "password_hash",
            "api_key_id",
        )
        for key in secret_keys:
            assert result["agent_kwargs"][key] == "***"

    def test_max_output_tokens_preserved_for_round_trip(self, tmp_path: Path) -> None:
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "solve this"}], model="test-model", max_output_tokens=12288
        )
        cfg = _make_instance_config(tmp_path, body=body)

        roundtripped = AnyTerminalInstanceConfig.model_validate_json(_safe_config_json(cfg))
        assert roundtripped.body.max_output_tokens == 12288

    def test_nested_provider_api_key_redacted(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(
            tmp_path,
            sandbox_provider={"opensandbox": {"connection": {"api_key": "secret"}}},  # pragma: allowlist secret
        )
        result = json.loads(_safe_config_json(cfg))
        assert result["sandbox_provider"]["opensandbox"]["connection"]["api_key"] == "***"

    def test_token_count_fields_are_not_redacted(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, body=_make_body().model_copy(update={"max_output_tokens": 123}))
        result = json.loads(_safe_config_json(cfg))
        assert result["body"]["max_output_tokens"] == 123

    def test_agent_command_str_excluded(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(
            tmp_path,
            agent_command_str="/agent_deps_mount/bin/python ...",
            agent_runtime_source="https://example.test/runtime?token=secret",
            agent_deps_url="https://example.test/runtime?token=secret",
        )
        result = json.loads(_safe_config_json(cfg))
        assert "agent_command_str" not in result
        assert "agent_runtime_source" not in result
        assert "agent_deps_url" not in result

    def test_indent_produces_multiline(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        assert "\n" in _safe_config_json(cfg, indent=2)


# ── AnyTerminalInstanceConfig properties ──────────────────────────────────────────


class TestInstanceConfigProperties:
    def test_task_name_from_problem_info(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, problem_info={"task_name": "my-task", "task_dir": str(tmp_path)})
        assert cfg.task_name == "my-task"

    def test_task_name_falls_back_to_instance_id(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, problem_info={"instance_id": "tb::my-task", "task_dir": str(tmp_path)})
        assert cfg.task_name == "tb::my-task"

    def test_task_name_unknown_when_neither_present(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, problem_info={"task_dir": str(tmp_path)})
        assert cfg.task_name == "unknown"

    def test_instance_id_from_problem_info(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(
            tmp_path,
            problem_info={"task_name": "t", "instance_id": "tb::t", "task_dir": str(tmp_path)},
        )
        assert cfg.instance_id == "tb::t"

    def test_instance_id_falls_back_to_task_name(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, problem_info={"task_name": "my-task", "task_dir": str(tmp_path)})
        assert cfg.instance_id == "my-task"


# ── _build_provider ────────────────────────────────────────────────────────────────


class TestBuildProvider:
    @pytest.fixture(autouse=True)
    def _provider_binaries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # These tests construct real provider configs without launching containers.
        monkeypatch.setattr(apptainer_provider, "_require_apptainer", lambda _bin_path=None: "/usr/bin/apptainer")
        monkeypatch.setattr(docker_provider, "_require_docker", lambda: "/usr/bin/docker")

    def test_default_is_docker(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        assert isinstance(_build_provider(cfg), DockerProvider)

    def test_apptainer_binds_are_wired(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, sandbox_provider={"apptainer": {}})
        provider = _build_provider(cfg)
        assert isinstance(provider, ApptainerProvider)
        binds = provider._exec_config.default_binds
        assert any(str(cfg.persistent_dir) in b and "/trajectories_mount" in b for b in binds)
        assert any(str(cfg.nemo_gym_root) in b and "/nemo_gym_mount" in b for b in binds)
        assert any(str(cfg.agent_deps_dir) in b and "/agent_deps_mount" in b for b in binds)
        assert any(str(cfg.verifier_dir) in b and "/logs/verifier" in b for b in binds)

    def test_apptainer_writable_and_no_home(self, tmp_path: Path) -> None:
        provider = _build_provider(_make_instance_config(tmp_path, sandbox_provider={"apptainer": {}}))
        start_args = provider._create_config.extra_start_args
        assert "--writable-tmpfs" in start_args
        assert "--no-mount" in start_args and "home,tmp" in start_args
        assert "tmp,bind-paths" not in provider._exec_config.extra_exec_args

    def test_docker_provider_selected(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, sandbox_provider={"docker": {}})
        provider = _build_provider(cfg)
        assert isinstance(provider, DockerProvider)

    def test_docker_run_args_include_mounts(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, sandbox_provider={"docker": {}})
        provider = _build_provider(cfg)
        assert any(str(cfg.persistent_dir) in a for a in provider._create_config.extra_run_args)
        assert any(str(cfg.verifier_dir) in a for a in provider._create_config.extra_run_args)

    def test_unknown_provider_returned_as_is(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, sandbox_provider={"opensandbox": {"foo": "bar"}})
        assert _build_provider(cfg) == {"opensandbox": {"foo": "bar"}}


class TestEpisodeResources:
    def _params(self, problem_info: dict, agent_overhead_mb: int = 2048) -> AnyTerminalInstanceConfig:
        return AnyTerminalInstanceConfig.model_construct(
            problem_info=problem_info,
            agent_overhead_mb=agent_overhead_mb,
        )

    def test_episode_adds_shared_headroom_to_declared_sandbox(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The global sandbox default neither caps nor rejects a row-owned sandbox.
        monkeypatch.setenv("NEMO_GYM_SANDBOX_MEMORY_MB", "4096")
        sandbox = _episode_sandbox_resources(self._params({"cpus": 2, "memory_mb": 4096, "pids": 512}))
        episode = sandbox_episode_resources(sandbox)

        assert (sandbox.cpu, sandbox.memory_mib, sandbox.pids) == (2, 4096 + 2048, 512)
        assert (episode.num_cpus, episode.pids) == (2, 512)
        assert episode.memory_bytes == (sandbox.memory_mib + EPISODE_MEMORY_HEADROOM_MIB) * 1024 * 1024

    def test_explicit_task_pids_win_global_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEMO_GYM_SANDBOX_PIDS", "2048")
        sandbox = _episode_sandbox_resources(self._params({"cpus": 2, "memory_mb": 4096, "pids": 512}))
        assert sandbox_episode_resources(sandbox).pids == 512

    @pytest.mark.parametrize("missing", ["cpus", "memory_mb"])
    def test_requires_every_task_resource_declaration(self, missing: str) -> None:
        problem_info = {"cpus": 2, "memory_mb": 4096, "pids": 512}
        problem_info.pop(missing)
        with pytest.raises(ValueError, match=missing):
            _episode_sandbox_resources(self._params(problem_info))

    def test_global_pids_owner_fills_missing_task_declaration(self, monkeypatch) -> None:
        monkeypatch.setenv("NEMO_GYM_SANDBOX_PIDS", "512")
        sandbox = _episode_sandbox_resources(self._params({"cpus": 2, "memory_mb": 4096}))
        assert sandbox_episode_resources(sandbox).pids == 512

    def test_rejects_fractional_cpu(self) -> None:
        with pytest.raises(ValueError, match="cpus.*positive integer"):
            _episode_sandbox_resources(self._params({"cpus": 2.5, "memory_mb": 4096, "pids": 512}))


# ── GymAgentHarnessProcessor.setup ───────────────────────────────────────────────


class TestHarnessProcessorSetup:
    def _proc_no_script(self) -> GymAgentHarnessProcessor:
        return GymAgentHarnessProcessor(
            config=_config(
                agent_server_module="responses_api_agents.no_such_agent.app",
                agent_server_class="NoSuchAgent",
                agent_config_class="NoSuchAgentConfig",
            )
        )

    def test_no_script_creates_empty_deps(self, tmp_path: Path) -> None:
        proc = self._proc_no_script()
        with patch.object(type(proc), "_parent", new_callable=PropertyMock, return_value=tmp_path):
            result = proc.setup()
        expected = tmp_path / "deps" / "anyterminal_no_such_agent_deps"
        assert result == expected
        assert expected.exists()
        assert (expected / ".installed").exists()

    def test_sentinel_match_skips_reinstall(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        proc = self._proc_no_script()
        with patch.object(type(proc), "_parent", new_callable=PropertyMock, return_value=tmp_path):
            proc.setup()
            proc.setup()
        assert "already at" in capsys.readouterr().out

    def test_rechecks_sentinel_after_acquiring_lock(self, tmp_path: Path) -> None:
        proc = self._proc_no_script()
        deps_dir = tmp_path / "deps" / "anyterminal_no_such_agent_deps"
        lock_path = deps_dir.parent / f".{deps_dir.name}.lockdir"
        lock_path.mkdir(parents=True)

        def finish_other_install(_seconds: float) -> None:
            deps_dir.mkdir()
            (deps_dir / ".installed").write_text(hashlib.sha256(b"no-script").hexdigest())
            shutil.rmtree(lock_path)

        with (
            patch.object(type(proc), "_parent", new_callable=PropertyMock, return_value=tmp_path),
            patch("responses_api_agents.anyterminal_agent.app.time.sleep", side_effect=finish_other_install),
        ):
            assert proc.setup() == deps_dir

        assert not lock_path.exists()

    def test_file_lock_retries_when_lock_disappears_before_stat(self, tmp_path: Path) -> None:
        setup_dir = tmp_path / "target"
        lock_path = setup_dir.parent / f".{setup_dir.name}.lockdir"
        lock_path.mkdir()
        original_stat = Path.stat
        removed = False

        def stat(path: Path, *args, **kwargs):
            nonlocal removed
            if path == lock_path and not removed:
                removed = True
                shutil.rmtree(lock_path)
                raise FileNotFoundError
            return original_stat(path, *args, **kwargs)

        with patch.object(Path, "stat", stat):
            with _file_lock(setup_dir, "test", max_wait=0, poll_interval=0):
                assert lock_path.exists()

    def test_file_lock_timeout_does_not_remove_existing_lock(self, tmp_path: Path) -> None:
        setup_dir = tmp_path / "target"
        lock_path = setup_dir.parent / f".{setup_dir.name}.lockdir"
        lock_path.mkdir()

        with pytest.raises(TimeoutError):
            with _file_lock(setup_dir, "test", max_wait=0, poll_interval=0):
                pass

        assert lock_path.exists()

    def test_stale_lock_is_reclaimed(self, tmp_path: Path) -> None:
        setup_dir = tmp_path / "target"
        lock_path = setup_dir.parent / f".{setup_dir.name}.lockdir"
        lock_path.mkdir(parents=True)
        old = time.time() - app._AGENT_DEPS_LOCK_STALE_AFTER_SECONDS - 1
        os.utime(lock_path, (old, old))

        with _file_lock(setup_dir, "test", max_wait=1, poll_interval=0):
            assert lock_path.exists()  # reacquired fresh by us, not the stale one

        assert not lock_path.exists()

    def test_fresh_lock_is_not_reclaimed(self, tmp_path: Path) -> None:
        setup_dir = tmp_path / "target"
        lock_path = setup_dir.parent / f".{setup_dir.name}.lockdir"
        lock_path.mkdir(parents=True)

        with pytest.raises(TimeoutError):
            with _file_lock(setup_dir, "test", max_wait=0, poll_interval=0):
                pass

        assert lock_path.exists()


# ── GymAgentHarnessProcessor.get_run_command ─────────────────────────────────────


class TestGetRunCommand:
    def test_writes_instruction_and_runner(self, tmp_path: Path) -> None:
        persistent_dir = tmp_path / "persistent"
        persistent_dir.mkdir()
        cfg = AnyTerminalInstanceConfig.model_construct(
            agent_server_module="responses_api_agents.hermes_agent.app",
            agent_server_class="HermesAgent",
            agent_config_class="HermesAgentConfig",
            body=SimpleNamespace(input=[SimpleNamespace(content="the task")]),
            persistent_dir=persistent_dir,
        )
        cmd = GymAgentHarnessProcessor(config=cfg).get_run_command()
        assert (persistent_dir / "instruction.txt").read_text() == "the task"
        runner = (persistent_dir / "agent_runner.py").read_text()
        assert "HermesAgent" in runner
        assert "HermesAgentConfig" in runner
        assert cmd == "/agent_deps_mount/bin/python /trajectories_mount/agent_runner.py"


# ── RunTerminalAgent ──────────────────────────────────────────────────────────────


def _sandbox_result(return_code: int = 0, error_type: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(stdout="", stderr="", return_code=return_code, error_type=error_type)


@pytest.mark.parametrize(
    "return_code,events,prefix,expected",
    [
        (137, {"sandbox_oom_kill": 2}, "sandbox", "verifier killed by the sandbox memory limit (exit 137, oom_kill 2)"),
        (-9, {"sandbox_oom_kill": 1}, "sandbox", "verifier killed by the sandbox memory limit (exit -9, oom_kill 1)"),
        (137, {"sandbox_oom_kill": 0}, "sandbox", None),  # a SIGKILL that is not the memory limit's
        (137, {}, "sandbox", None),  # no retained events
        (1, {"sandbox_oom_kill": 3}, "sandbox", None),  # a failing test beside a killed task subprocess
        (0, {"sandbox_oom_kill": 3}, "sandbox", None),  # the verifier survived the kill
        (None, {"sandbox_oom_kill": 3}, "sandbox", None),  # the verifier never ran
        (
            137,
            {"sandbox_oom_kill": 0, "grader_sandbox_oom_kill": 1},
            "grader_sandbox",
            "verifier killed by the sandbox memory limit (exit 137, oom_kill 1)",
        ),
        (137, {"sandbox_oom_kill": 1, "grader_sandbox_oom_kill": 0}, "grader_sandbox", None),
    ],
)
def test_verifier_memory_kill_needs_a_sigkill_exit_and_an_oom_kill_in_its_sandbox(
    return_code, events, prefix, expected
) -> None:
    assert verifier_memory_kill(return_code, events, prefix) == expected
    if prefix == "sandbox":
        assert verifier_memory_kill(return_code, events) == expected
        assert harness_memory_kill(return_code, events) == (expected and expected.replace("verifier", "agent harness"))


class TestRunAgentEnv:
    def test_model_url_included_when_set(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, model_server_url="http://model:8000")
        env = RunTerminalAgent(config=cfg)._agent_env(cfg)
        assert env["NGTB_MODEL_URL"] == "http://model:8000"

    def test_no_model_url_when_empty(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path, model_server_url="")
        env = RunTerminalAgent(config=cfg)._agent_env(cfg)
        assert "NGTB_MODEL_URL" not in env

    def test_sampling_and_kwargs_forwarded(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(
            tmp_path,
            agent_kwargs={"model": "my-model"},
            body=_make_body(),
        )
        env = RunTerminalAgent(config=cfg)._agent_env(cfg)
        assert env["NGTB_MODEL_NAME"] == "my-model"
        assert json.loads(env["NGTB_AGENT_KWARGS"]) == {"model": "my-model"}


class TestTerminalResponseLabels:
    def test_context_stop_keeps_partial_grade_and_typed_metadata(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        update_metrics(cfg.metrics_fpath, {"resolved": True})
        (cfg.persistent_dir / "response.json").write_text(json.dumps(self._response(
            status="incomplete", metadata={"stop_reason": "context_exhausted"},
        )))
        (cfg.persistent_dir / "agent_result.json").write_text(json.dumps({"return_code": 0, "error_type": None}))
        response = app._terminal_response(cfg)
        assert response.metadata["stop_reason"] == "context_exhausted"
        metrics = json.loads(response.metadata["metrics"])
        assert metrics["resolved"] is True
        assert metrics["mask_sample"] is False
        assert metrics["agent_failed"] is False

    def _response(self, **fields) -> dict:
        return {
            "id": "inner", "created_at": 0, "model": "inner-model", "object": "response", "output": [],
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], **fields,
        }

    def _labels(self, cfg: AnyTerminalInstanceConfig) -> dict:
        response = app._terminal_response(cfg)
        persisted = json.loads(cfg.metrics_fpath.read_text())
        assert json.loads(response.metadata["metrics"]) == persisted
        return {key: persisted.get(key) for key in ("agent_failed", "agent_error", "agent_exit_nonzero", "mask_sample")}

    def test_inner_agent_error_label_is_kept(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        (cfg.persistent_dir / "response.json").write_text(
            json.dumps(self._response(status="completed", metadata={"agent_error": "RuntimeError"}))
        )
        (cfg.persistent_dir / "agent_result.json").write_text(json.dumps({"return_code": 0, "error_type": None}))
        assert self._labels(cfg) == {
            "agent_failed": True, "agent_error": "RuntimeError", "agent_exit_nonzero": False, "mask_sample": False,
        }

    def test_inner_failed_status_and_nonzero_exit_are_labelled(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        error = {"code": "server_error", "message": "API call failed after 3 retries"}
        (cfg.persistent_dir / "response.json").write_text(json.dumps(self._response(status="failed", error=error)))
        (cfg.persistent_dir / "agent_result.json").write_text(json.dumps({"return_code": 1, "error_type": None}))
        assert self._labels(cfg) == {
            "agent_failed": True, "agent_error": "server_error", "agent_exit_nonzero": True, "mask_sample": False,
        }
        response = app._terminal_response(cfg)
        assert response.status != "failed"
        assert response.error is None

    @pytest.mark.parametrize("index", [0, 1], ids=["pi-no-call-failure", "hermes-route-denial"])
    def test_saved_harness_failures_are_masked_with_successful_process_exit(self, tmp_path: Path, index: int) -> None:
        raw = json.loads((PARENT_DIR / "tests/unit_tests/fixtures/sandbox-harness-failures.json").read_text())[index]["response"]
        cfg = _make_instance_config(tmp_path)
        (cfg.persistent_dir / "response.json").write_text(json.dumps(raw))
        (cfg.persistent_dir / "agent_result.json").write_text(json.dumps({"return_code": 0, "error_type": None}))
        response = app._terminal_response(cfg)
        assert response.status == "failed"
        assert response.error is not None
        assert json.loads(cfg.metrics_fpath.read_text())["mask_sample"] is True


    def test_missing_response_is_labelled_unless_masked(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        assert self._labels(cfg)["agent_error"] == "response_missing"
        masked = _make_instance_config(tmp_path / "masked")
        update_metrics(masked.metrics_fpath, {"agent_timed_out": True})
        assert self._labels(masked) == {
            "agent_failed": False, "agent_error": None, "agent_exit_nonzero": False, "mask_sample": True,
        }

    @pytest.mark.parametrize(("label", "expected"), [("true", True), ("false", False), (None, False)])
    def test_inner_completion_label_is_carried(self, tmp_path: Path, label, expected) -> None:
        cfg = _make_instance_config(tmp_path)
        metadata = {} if label is None else {"finished_naturally": label}
        (cfg.persistent_dir / "response.json").write_text(
            json.dumps(self._response(status="completed", metadata=metadata))
        )
        (cfg.persistent_dir / "agent_result.json").write_text(json.dumps({"return_code": 0, "error_type": None}))
        response = app._terminal_response(cfg)
        assert app.TerminalBenchMetrics.model_validate_json(response.metadata["metrics"]).finished_naturally is expected

    def test_missing_response_claims_no_completion(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        response = app._terminal_response(cfg)
        assert app.TerminalBenchMetrics.model_validate_json(response.metadata["metrics"]).finished_naturally is False

    def test_clean_run_carries_no_label(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        (cfg.persistent_dir / "response.json").write_text(json.dumps(self._response(status="completed")))
        (cfg.persistent_dir / "agent_result.json").write_text(json.dumps({"return_code": 0, "error_type": None}))
        assert self._labels(cfg) == {
            "agent_failed": False, "agent_error": None, "agent_exit_nonzero": False, "mask_sample": False,
        }


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fault", [None, "container_timed_out", "sandbox_failed", "mask_sample"])
def test_timeout_response_keeps_independent_infrastructure_masked(tmp_path, enabled, fault):
    cfg = _make_instance_config(tmp_path, agent_timeout_is_failure=enabled)
    metrics = {"resolved": True, "agent_timed_out": True}
    if fault:
        metrics[fault] = True
    update_metrics(cfg.metrics_fpath, metrics)
    response = app._terminal_response(cfg)
    observed = app.TerminalBenchMetrics.model_validate_json(response.metadata["metrics"])
    assert observed.agent_timed_out is True
    assert observed.resolved is (not enabled)
    assert observed.mask_sample is (not enabled or fault is not None)
    assert json.loads(response.metadata["instance_config"])["agent_timeout_is_failure"] is enabled


def test_timeout_policy_is_strict_and_default_off(tmp_path):
    assert _make_instance_config(tmp_path).agent_timeout_is_failure is False
    with pytest.raises(ValidationError, match="agent_timeout_is_failure"):
        _make_instance_config(tmp_path, agent_timeout_is_failure="true")


@pytest.mark.parametrize("harness_exit,oom_kills,infra", [(137, 1, True), (137, 0, False), (0, 1, False)])
def test_harness_oom_requires_both_process_exit_and_kernel_event(harness_exit, oom_kills, infra):
    problem = harness_memory_kill(harness_exit, {"sandbox_oom_kill": oom_kills})
    metrics = app.TerminalBenchMetrics(resolved=False, sandbox_failed=problem is not None)
    metrics.apply_timeout_policy(agent_timeout_is_failure=True)
    assert metrics.mask_sample is infra


class TestProcessSingleDatapoint:
    @pytest.fixture(autouse=True)
    def _no_real_provider(self):
        # AsyncSandbox is mocked below, so the provider it's built with is never used —
        # skip building a real ApptainerProvider (which hard-errors without the binary).
        with patch("responses_api_agents.anyterminal_agent.app._build_provider", return_value=None):
            yield

    async def test_resolved_when_reward_positive(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)

        async def verifier(command, **kwargs):
            if "bash /tests/test.sh" in command:
                (cfg.verifier_dir / "reward.txt").write_text("1.0")
            return _sandbox_result()

        sandbox = SimpleNamespace(
            start=AsyncMock(),
            exec=AsyncMock(side_effect=verifier),
            stop=AsyncMock(),
        )
        with patch(
            "responses_api_agents.anyterminal_agent.app.AsyncSandbox", return_value=sandbox
        ) as sandbox_type:
            with patch.object(RunTerminalAgent, "_stage_tests", new=AsyncMock(return_value=None)):
                result = await RunTerminalAgent(config=cfg).process_single_datapoint()

        assert result is True
        sandbox_spec = sandbox_type.call_args.args[1]
        assert sandbox_spec.resources.cpu == 2
        assert sandbox_spec.resources.memory_mib == 4096 + cfg.agent_overhead_mb
        assert sandbox_spec.resources.pids == 512
        metrics = json.loads(cfg.metrics_fpath.read_text())
        assert metrics["resolved"] is True
        assert 0 < metrics["sandbox_ready_at"] <= metrics["grading_started_at"] <= metrics["grading_completed_at"]
        timing = json.loads((cfg.verifier_dir / "grading-timing.json").read_text())
        assert timing == {
            "started_at": metrics["grading_started_at"],
            "completed_at": metrics["grading_completed_at"],
        }
        sandbox.stop.assert_awaited_once()

    async def test_receipt_permission_failure_closes_with_zero_and_flag(self, tmp_path: Path) -> None:
        if os.geteuid() == 0:
            pytest.skip("requires an unprivileged allocated worker")
        cfg = _make_instance_config(tmp_path)

        async def verifier(command, **kwargs):
            if "bash /tests/test.sh" in command:
                (cfg.verifier_dir / "reward.txt").write_text("1.0")
                cfg.persistent_dir.chmod(0o500)
            return _sandbox_result()

        sandbox = SimpleNamespace(start=AsyncMock(), exec=AsyncMock(side_effect=verifier), stop=AsyncMock())
        try:
            with patch("responses_api_agents.anyterminal_agent.app.AsyncSandbox", return_value=sandbox):
                with patch.object(RunTerminalAgent, "_stage_tests", new=AsyncMock(return_value=None)):
                    result = await RunTerminalAgent(config=cfg).process_single_datapoint()
        finally:
            cfg.persistent_dir.chmod(0o700)

        assert result is False
        metrics = json.loads(cfg.metrics_fpath.read_text())
        assert metrics["verifier_integrity_refused"] is True
        assert "PermissionError" in metrics["verifier_integrity_problem"]
        assert metrics["mask_sample"] is False
        sandbox.stop.assert_awaited_once()

    async def test_unresolved_when_no_reward_file(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        sandbox = SimpleNamespace(
            start=AsyncMock(),
            exec=AsyncMock(return_value=_sandbox_result()),
            stop=AsyncMock(),
        )
        with patch("responses_api_agents.anyterminal_agent.app.AsyncSandbox", return_value=sandbox):
            with patch.object(RunTerminalAgent, "_stage_tests", new=AsyncMock(return_value=None)):
                result = await RunTerminalAgent(config=cfg).process_single_datapoint()

        assert result is False
        metrics = json.loads(cfg.metrics_fpath.read_text())
        # test.sh writes 0 or 1: a completed eval without reward.txt is labelled, not masked.
        assert metrics["verifier_reward_unreadable"] is True
        assert metrics["mask_sample"] is False

    async def test_unparsable_reward_is_labelled_not_masked(self, tmp_path: Path, capsys) -> None:
        cfg = _make_instance_config(tmp_path)

        async def verifier(command, **kwargs):
            if "bash /tests/test.sh" in command:
                (cfg.verifier_dir / "reward.txt").write_text("PASSED")
            return _sandbox_result()
        sandbox = SimpleNamespace(
            start=AsyncMock(),
            exec=AsyncMock(side_effect=verifier),
            stop=AsyncMock(),
        )
        with patch("responses_api_agents.anyterminal_agent.app.AsyncSandbox", return_value=sandbox):
            with patch.object(RunTerminalAgent, "_stage_tests", new=AsyncMock(return_value=None)):
                result = await RunTerminalAgent(config=cfg).process_single_datapoint()

        assert result is False
        metrics = json.loads(cfg.metrics_fpath.read_text())
        assert metrics["verifier_reward_unreadable"] is True
        assert metrics["mask_sample"] is False
        assert "without a readable reward.txt: ValueError" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "agent_return_code,oom_kill,infrastructure",
        [(0, 1, False), (137, 2, True)],
        ids=["tool-killed-agent-finished", "agent-killed-by-limit"],
    )
    async def test_sandbox_memory_kills_are_counted_and_only_a_killed_harness_is_infrastructure(
        self, tmp_path: Path, agent_return_code: int, oom_kill: int, infrastructure: bool
    ) -> None:
        cfg = _make_instance_config(tmp_path, agent_command_str="run-the-agent")

        class RetainedEventsProvider:
            """A provider whose sandbox cgroup kept memory.events at close, as the Apptainer provider does."""

            closed = False
            seen_oom = 0

            async def create(self, spec):
                return SandboxHandle(sandbox_id="box", provider_name="retained", raw=None)

            async def exec(self, handle, command, **kwargs):
                if "run-the-agent" in command:
                    self.seen_oom = oom_kill
                if "bash /tests/test.sh" in command:
                    (cfg.verifier_dir / "reward.txt").write_text("1.0")
                return SandboxExecResult("", "", agent_return_code if "run-the-agent" in command else 0, None)

            async def close(self, handle):
                self.closed = True

            async def aclose(self):
                pass

            def memory_events(self, handle):
                return {"max": 9 if self.seen_oom else 0, "oom": self.seen_oom, "oom_kill": self.seen_oom}

            def retained_memory_events(self, handle):
                assert self.closed, "events are read after the sandbox is stopped"
                return {"low": 0, "high": 0, "max": 9, "oom": oom_kill, "oom_kill": oom_kill}

        with patch(
            "responses_api_agents.anyterminal_agent.app.AsyncSandbox",
            side_effect=lambda provider, spec: AsyncSandbox(RetainedEventsProvider(), spec),
        ):
            with patch.object(RunTerminalAgent, "_stage_tests", new=AsyncMock(return_value=None)):
                resolved = await RunTerminalAgent(config=cfg).process_single_datapoint()

        metrics = json.loads(cfg.metrics_fpath.read_text())
        assert (metrics["sandbox_oom"], metrics["sandbox_oom_kill"], metrics["sandbox_memory_max_events"]) == (
            oom_kill,
            oom_kill,
            9,
        )
        assert resolved is True
        assert metrics["sandbox_failed"] is infrastructure
        assert metrics["mask_sample"] is infrastructure
        assert ("sandbox memory limit (exit 137, oom_kill 2)" in metrics.get("error_detail", "")) is infrastructure

    @pytest.mark.parametrize(
        "verifier_return_code,reward,oom_kill,infrastructure",
        [(137, "1", 1, True), (-9, "1", 3, True), (1, "0", 0, False), (0, "0", 1, False)],
        ids=[
            "verifier-killed-after-writing-reward",
            "verifier-runtime-killed-after-writing-reward",
            "ordinary-failing-test",
            "task-subprocess-killed-verifier-finished",
        ],
    )
    @pytest.mark.parametrize("oom_phase", ["agent", "verifier", "unavailable"])
    async def test_only_a_verifier_the_memory_limit_killed_is_infrastructure(
        self, tmp_path: Path, verifier_return_code: int, reward: str, oom_kill: int, infrastructure: bool, oom_phase: str
    ) -> None:
        cfg = _make_instance_config(tmp_path, agent_command_str="run-the-agent")

        class VerifierProvider:
            """A provider whose verifier writes its reward and then exits with the given status."""

            closed = False
            seen_oom = 0

            async def create(self, spec):
                return SandboxHandle(sandbox_id="box", provider_name="retained", raw=None)

            async def exec(self, handle, command, **kwargs):
                if "run-the-agent" in command and oom_phase == "agent":
                    self.seen_oom = oom_kill
                if "bash /tests/test.sh" in command:
                    if oom_phase == "verifier":
                        self.seen_oom = oom_kill
                    (cfg.verifier_dir / "reward.txt").write_text(reward)
                    return SandboxExecResult("", "", verifier_return_code, None)
                return SandboxExecResult("", "", 0, None)

            async def close(self, handle):
                self.closed = True

            async def aclose(self):
                pass

            def memory_events(self, handle):
                if oom_phase == "unavailable":
                    return {}
                return {"max": 4 * self.seen_oom, "oom": self.seen_oom, "oom_kill": self.seen_oom}

            def retained_memory_events(self, handle):
                assert self.closed, "events are read after the sandbox is stopped"
                return {"low": 0, "high": 0, "max": 4 * oom_kill, "oom": oom_kill, "oom_kill": oom_kill}

        with patch(
            "responses_api_agents.anyterminal_agent.app.AsyncSandbox",
            side_effect=lambda provider, spec: AsyncSandbox(VerifierProvider(), spec),
        ):
            with patch.object(RunTerminalAgent, "_stage_tests", new=AsyncMock(return_value=None)):
                resolved = await RunTerminalAgent(config=cfg).process_single_datapoint()

        metrics = json.loads(cfg.metrics_fpath.read_text())
        # Reward reading is unchanged; the mask is what keeps an incomplete grade out of training.
        assert resolved is (reward == "1")
        assert metrics["resolved"] is (reward == "1")
        assert metrics["verifier_return_code"] == verifier_return_code
        assert "verifier_error_type" not in metrics  # None: an ordinary exit, no runtime marker
        assert metrics["sandbox_oom_kill"] == oom_kill
        assert metrics["verifier_reward_unreadable"] is False
        assert metrics["container_timed_out"] is False
        assert metrics["sandbox_failed"] is infrastructure
        assert metrics["mask_sample"] is infrastructure
        if infrastructure and oom_phase == "verifier":
            assert metrics["error_detail"] == (
                f"verifier killed by the sandbox memory limit (exit {verifier_return_code}, oom_kill {oom_kill})"
            )
            assert metrics["memory_failure_phase"] == "verifier"
        elif infrastructure:
            assert metrics["error_detail"] == "verifier SIGKILL with sandbox OOM history; execution phase UNKNOWN"
            assert metrics["memory_failure_phase"] == "unknown"
        else:
            assert "error_detail" not in metrics
        if oom_phase != "unavailable":
            assert metrics["verifier_phase_memory_events"]["sandbox_oom_kill"] == (
                oom_kill if oom_phase == "verifier" else 0
            )

    @pytest.mark.parametrize(
        "return_code,error_type,failed",
        [(0, None, False), (137, None, False), (-1, "sandbox", False), (-1, "timeout", True)],
    )
    async def test_run_eval_returns_the_verifier_status(
        self, tmp_path: Path, return_code: int, error_type: str | None, failed: bool
    ) -> None:
        cfg = _make_instance_config(tmp_path)
        sandbox = SimpleNamespace(exec=AsyncMock(return_value=_sandbox_result(return_code, error_type)))
        elapsed, timed_out_or_sandbox, got_return_code, got_error_type = await RunTerminalAgent(
            config=cfg
        )._run_eval(sandbox, cfg)
        assert elapsed >= 0
        assert (timed_out_or_sandbox, got_return_code, got_error_type) == (failed, return_code, error_type)

    async def test_agent_timeout_sets_flag_and_masks(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        sandbox = SimpleNamespace(
            start=AsyncMock(),
            exec=AsyncMock(return_value=_sandbox_result(return_code=124, error_type="timeout")),
            stop=AsyncMock(),
        )
        with patch("responses_api_agents.anyterminal_agent.app.AsyncSandbox", return_value=sandbox):
            with patch.object(RunTerminalAgent, "_stage_tests", new=AsyncMock(return_value=None)):
                await RunTerminalAgent(config=cfg).process_single_datapoint()

        metrics = json.loads(cfg.metrics_fpath.read_text())
        assert metrics["agent_timed_out"] is True
        assert metrics["mask_sample"] is True
        assert metrics["verifier_reward_unreadable"] is False

    async def test_container_killed_mid_run_sets_flag_and_masks(self, tmp_path: Path) -> None:
        # exec() on a container the TTL already removed returns error_type="sandbox" without
        # raising. Keep it masked as sandbox failure without inventing a timeout.
        cfg = _make_instance_config(tmp_path)
        sandbox = SimpleNamespace(
            start=AsyncMock(),
            exec=AsyncMock(return_value=_sandbox_result(return_code=125, error_type="sandbox")),
            stop=AsyncMock(),
        )
        with patch("responses_api_agents.anyterminal_agent.app.AsyncSandbox", return_value=sandbox):
            with patch.object(RunTerminalAgent, "_stage_tests", new=AsyncMock(return_value=None)):
                await RunTerminalAgent(config=cfg).process_single_datapoint()

        metrics = json.loads(cfg.metrics_fpath.read_text())
        assert metrics["agent_timed_out"] is False
        assert metrics["container_timed_out"] is False
        assert metrics["sandbox_failed"] is True
        assert metrics["mask_sample"] is True

    async def test_metrics_file_removed_mid_episode_is_still_recorded(self, tmp_path: Path) -> None:
        # Production shape: the instance's results directory went away under the episode, so
        # update_metrics raised FileNotFoundError out of the Ray task and the row came back as a
        # 500 with no agent result. The sample has to arrive masked and unresolved instead.
        cfg = _make_instance_config(tmp_path)

        async def _stop_and_lose_the_directory() -> None:
            cfg.metrics_fpath.unlink()

        sandbox = SimpleNamespace(
            start=AsyncMock(),
            exec=AsyncMock(return_value=_sandbox_result(return_code=125, error_type="sandbox")),
            stop=AsyncMock(side_effect=_stop_and_lose_the_directory),
        )
        with patch("responses_api_agents.anyterminal_agent.app.AsyncSandbox", return_value=sandbox):
            with patch.object(RunTerminalAgent, "_stage_tests", new=AsyncMock(return_value=None)):
                result = await RunTerminalAgent(config=cfg).process_single_datapoint()

        assert result is False
        metrics = json.loads(cfg.metrics_fpath.read_text())
        assert metrics["sandbox_failed"] is True
        assert metrics["mask_sample"] is True

    async def test_sandbox_start_failure_is_isolated(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        sandbox = SimpleNamespace(
            start=AsyncMock(side_effect=RuntimeError("boom")),
            exec=AsyncMock(),
            stop=AsyncMock(),
        )
        with patch("responses_api_agents.anyterminal_agent.app.AsyncSandbox", return_value=sandbox):
            result = await RunTerminalAgent(config=cfg).process_single_datapoint()

        assert result is False
        metrics = json.loads(cfg.metrics_fpath.read_text())
        assert metrics["sandbox_failed"] is True
        assert metrics["mask_sample"] is True
        sandbox.stop.assert_awaited_once()

    async def test_remote_provider_stages_and_collects(self, tmp_path: Path) -> None:
        archive = tmp_path / "deps.tar.gz"
        archive.write_bytes(b"deps")
        cfg = _make_instance_config(
            tmp_path,
            sandbox_provider={"opensandbox": {}},
            agent_deps_archive=archive,
        )
        sandbox = SimpleNamespace(
            start=AsyncMock(),
            exec=AsyncMock(return_value=_sandbox_result()),
            upload=AsyncMock(),
            download=AsyncMock(),
            stop=AsyncMock(),
        )
        with patch("responses_api_agents.anyterminal_agent.app.AsyncSandbox", return_value=sandbox):
            with (
                patch.object(RunTerminalAgent, "_stage_tests", new=AsyncMock()),
                patch.object(RunTerminalAgent, "_stage_remote_tests", new=AsyncMock()) as stage_tests,
                patch.object(RunTerminalAgent, "_collect_remote_outputs", new=AsyncMock()) as collect,
            ):
                await RunTerminalAgent(config=cfg).process_single_datapoint()

        uploaded = {call.args[1] for call in sandbox.upload.await_args_list}
        assert "/trajectories_mount/instruction.txt" in uploaded
        assert "/trajectories_mount/agent_runner.py" in uploaded
        assert "/tmp/anyterminal-agent-deps.tar.gz" in uploaded
        stage_tests.assert_awaited_once()
        collect.assert_awaited_once()

    async def test_remote_runtime_can_be_fetched_from_url(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(
            tmp_path,
            sandbox_provider={"opensandbox": {}},
            agent_runtime_source="https://relay.example/runtime.tar.gz",
            agent_deps_url="https://relay.example/runtime.tar.gz",
        )
        sandbox = SimpleNamespace(
            exec=AsyncMock(return_value=_sandbox_result()),
            upload=AsyncMock(),
        )

        await RunTerminalAgent(config=cfg)._stage_remote_runtime(sandbox, cfg)

        uploaded = {call.args[1] for call in sandbox.upload.await_args_list}
        assert "/tmp/anyterminal-agent-deps.tar.gz" not in uploaded
        commands = [call.args[0] for call in sandbox.exec.await_args_list]
        assert any("curl -fsSL" in command and "relay.example/runtime.tar.gz" in command for command in commands)
        assert any("tar -xzf" in command for command in commands)

    @pytest.mark.parametrize("verifier_error", ["timeout", "sandbox"])
    async def test_stops_sandbox_even_on_eval_timeout(self, tmp_path: Path, verifier_error: str) -> None:
        cfg = _make_instance_config(tmp_path)
        # First exec (agent) succeeds, second exec (eval) times out.
        sandbox = SimpleNamespace(
            start=AsyncMock(),
            exec=AsyncMock(side_effect=[_sandbox_result(), _sandbox_result(return_code=124, error_type=verifier_error)]),
            stop=AsyncMock(),
        )
        with patch("responses_api_agents.anyterminal_agent.app.AsyncSandbox", return_value=sandbox):
            with patch.object(RunTerminalAgent, "_stage_tests", new=AsyncMock(return_value=None)):
                await RunTerminalAgent(config=cfg).process_single_datapoint()

        metrics = json.loads(cfg.metrics_fpath.read_text())
        assert metrics["container_timed_out"] is (verifier_error == "timeout")
        assert metrics["sandbox_failed"] is (verifier_error == "sandbox")
        assert metrics["mask_sample"] is True
        sandbox.stop.assert_awaited_once()


# ── RunTerminalAgent._stage_tests ────────────────────────────────────────────────


class TestStageTests:
    async def test_copies_tests(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        (cfg.persistent_dir / "staging").mkdir(parents=True, exist_ok=True)
        tests_src = Path(cfg.problem_info["task_dir"]) / "tests"
        tests_src.mkdir()
        (tests_src / "test.sh").write_text("#!/bin/bash")

        await RunTerminalAgent(config=cfg)._stage_tests(cfg)

        assert (cfg.persistent_dir / "staging" / "tests" / "test.sh").exists()

    async def test_removes_stale_staging_before_copy(self, tmp_path: Path) -> None:
        cfg = _make_instance_config(tmp_path)
        staging_tests = cfg.persistent_dir / "staging" / "tests"
        staging_tests.mkdir(parents=True)
        (staging_tests / "stale.sh").write_text("old")
        tests_src = Path(cfg.problem_info["task_dir"]) / "tests"
        tests_src.mkdir()
        (tests_src / "fresh.sh").write_text("new")

        await RunTerminalAgent(config=cfg)._stage_tests(cfg)

        assert not (staging_tests / "stale.sh").exists()
        assert (staging_tests / "fresh.sh").exists()


@pytest.mark.parametrize("selected", [False, True])
def test_fakeroot_command_selection_is_task_scoped(tmp_path, monkeypatch, selected):
    monkeypatch.setattr(apptainer_provider, "_require_apptainer", lambda _bin_path=None: "/usr/bin/apptainer")
    cfg = _make_instance_config(tmp_path, sandbox_provider={"apptainer": {}},
        apptainer_ignore_fakeroot_command_tasks={"selected"})
    cfg.problem_info["task_name"] = "selected" if selected else "ordinary"
    provider = _build_provider(cfg)
    assert ("--ignore-fakeroot-command" in provider._create_config.extra_start_args) is selected


@pytest.mark.parametrize("provider", ["apptainer", "docker"])
def test_prepared_runtime_is_selected_without_install(tmp_path, provider):
    runtime = tmp_path / "prepared"
    (runtime / "bin").mkdir(parents=True)
    python = runtime / "bin/python"
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    config = _config(model_server=None, sandbox_provider={provider: {}},
                     agent_runtime_source=str(runtime), results_dir=tmp_path / "results")
    with patch.object(GymAgentHarnessProcessor, "setup", side_effect=AssertionError("unexpected install")):
        agent = AnyTerminalAgent(config=config, server_client=ServerClient.model_construct(global_config_dict={}))
    assert agent._server.agent_deps_dir == runtime.resolve()
    assert agent._server.agent_deps_archive is None
    assert agent._server.agent_deps_url is None


@pytest.mark.parametrize("kind", ["missing", "relative", "no_python", "not_executable"])
def test_invalid_prepared_runtime_fails_before_install(tmp_path, kind):
    runtime = tmp_path / "prepared"
    runtime.mkdir()
    if kind == "missing":
        runtime = tmp_path / "absent"
    elif kind == "relative":
        runtime = Path("relative-runtime")
    elif kind == "not_executable":
        (runtime / "bin").mkdir()
        (runtime / "bin/python").write_text("not executable")
    config = _config(model_server=None, sandbox_provider={"apptainer": {}},
                     agent_runtime_source=str(runtime), results_dir=tmp_path / "results")
    with patch.object(GymAgentHarnessProcessor, "setup", side_effect=AssertionError("unexpected install")):
        with pytest.raises(ValueError, match="prepared"):
            AnyTerminalAgent(config=config, server_client=ServerClient.model_construct(global_config_dict={}))


@pytest.mark.parametrize("args", [["--no-mount", "bind-paths"], ["--no-mount=bind-paths"]])
def test_apptainer_preserves_existing_mount_exclusions(tmp_path, monkeypatch, args):
    monkeypatch.setattr(apptainer_provider, "_require_apptainer", lambda _bin_path=None: "/usr/bin/apptainer")
    cfg = _make_instance_config(tmp_path, sandbox_provider={"apptainer": {"create": {"extra_start_args": args}}})
    provider = _build_provider(cfg)
    actual = provider._create_config.extra_start_args
    assert any("bind-paths,home,tmp" in value for value in actual)
    assert args in (["--no-mount", "bind-paths"], ["--no-mount=bind-paths"])
