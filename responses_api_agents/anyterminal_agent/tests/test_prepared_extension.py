# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Declared interpreter authority and opt-in real task/sandbox controls."""

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.anyterminal_agent import app, extension_runtime
from responses_api_agents.anyterminal_agent.tests.test_app import _make_setup_agent


IMAGE_PYTHON = {
    "executable": "/usr/local/bin/python3.13",
    "executable_sha256": "a" * 64,
    "version": "3.13.7",
    "soabi": "cpython-313-x86_64-linux-gnu",
    "extension_suffix": ".cpython-313-x86_64-linux-gnu.so",
    "uv_version": "0.9.5",
    "requirements": ["pytest==8.4.1", "setuptools==80.9.0", "pytest-json-ctrf==0.3.5"],
}


def declaration_toml():
    return (
        '''
[[verifier.isolation.prepared_artifacts]]
kind = "python_extension"
module = "example"
max_bytes = 1024
[verifier.isolation.prepared_artifacts.sources]
"tests/setup.py" = "'''
        + "b" * 64
        + '''"
"tests/example.c" = "'''
        + "c" * 64
        + """"
[verifier.isolation.prepared_artifacts.runtime]
mode = "image_python"
[verifier.isolation.prepared_artifacts.runtime.image_python]
"""
        + "\n".join(f"{k} = {json.dumps(v)}" for k, v in IMAGE_PYTHON.items())
    )


def test_task_declaration_selects_runtime_even_with_complete_row_metadata(tmp_path):
    (tmp_path / "task.toml").write_text(declaration_toml())
    agent = _make_setup_agent(tmp_path)
    body = NeMoGymResponseCreateParamsNonStreaming(
        input="instruction",
        metadata={
            "task_dir": str(tmp_path),
            "task_name": "example",
            "workdir": "/app",
            "agent_timeout_sec": "10",
            "verifier_timeout_sec": "20",
            "verifier_extension": "untrusted",
            "verifier_runtime": "untrusted",
        },
    )
    cfg = agent._setup_params(body)
    assert cfg.verifier_runtime.mode == "image_python"
    assert cfg.verifier_runtime.image_python.executable_sha256 == IMAGE_PYTHON["executable_sha256"]
    assert cfg.problem_info["verifier_extension"]["module"] == "example"
    assert cfg.verifier_runtime.binds() == [f"{Path(app.__file__).parent}:/gym_verifier_helper:ro"]


def test_row_cannot_add_extension_preparation(tmp_path):
    agent = _make_setup_agent(tmp_path)
    body = NeMoGymResponseCreateParamsNonStreaming(
        input="instruction",
        metadata={"task_dir": str(tmp_path), "verifier_extension": "untrusted"},
    )
    cfg = agent._setup_params(body)
    assert "verifier_extension" not in cfg.problem_info
    assert cfg.verifier_runtime is None


@pytest.mark.parametrize(
    "change",
    [{"version": "3.12.1"}, {"requirements": ["pytest"]}, {"executable": "/app/python"}, {"executable_sha256": "bad"}],
)
def test_image_interpreter_identity_fails_closed(change):
    with pytest.raises(ValidationError):
        app.ImageVerifierPython.model_validate({**IMAGE_PYTHON, **change})


def test_image_and_managed_runtime_cannot_be_mixed():
    with pytest.raises(ValidationError, match="only its exact image interpreter"):
        app.VerifierRuntime(mode="image_python", image_python=IMAGE_PYTHON, helper_dir=Path("/helper"))


def test_preparation_refuses_changed_sources_before_build(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "setup.py").write_text("raise RuntimeError('must never execute')")
    spec = {
        "workspace": "/app",
        "command": "exit 99",
        "sources": {"tests/setup.py": "0" * 64, "tests/example.c": "1" * 64},
        "module": "example",
        "output": "tests/example" + IMAGE_PYTHON["extension_suffix"],
        "max_bytes": 1024,
        "runtime": {"image_python": IMAGE_PYTHON},
        "declaration_sha256": "2" * 64,
        "task_manifest_sha256": "3" * 64,
        "sif_sha256": "4" * 64,
    }
    with pytest.raises(ValueError, match="source differs"):
        extension_runtime.prepare_extension(
            json.dumps(spec),
            source_tests=tests,
            output_dir=tmp_path / "output",
            receipt_dir=tmp_path / "receipt",
            run_phase=lambda *args: pytest.fail("build ran"),
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.skipif(not os.environ.get("GYM_ANYTERMINAL_NATIVE_ROWS"), reason="requires allocated native task images")
@pytest.mark.parametrize(
    "task,control,expected",
    [
        ("feal-differential-cryptanalysis", "oracle", True),
        ("feal-differential-cryptanalysis", "noop", False),
        ("regex-log", "oracle", True),
    ],
)
async def test_native_task_grades_with_original_pytest(tmp_path, task, control, expected):
    rows = [
        json.loads(line)
        for line in Path(os.environ["GYM_ANYTERMINAL_NATIVE_ROWS"]).read_text().splitlines()
        if line.strip()
    ]
    row = next(row for row in rows if row["responses_create_params"]["metadata"]["task_name"] == task)
    provider = json.loads(os.environ["GYM_ANYTERMINAL_NATIVE_PROVIDER"])
    root = Path(os.environ["GYM_ANYTERMINAL_NATIVE_OUTPUT"]) / f"{task}-{control}"
    root.mkdir(parents=True, exist_ok=False)
    agent = _make_setup_agent(
        root,
        container_formatter="{runtime_image}",
        sandbox_provider=provider,
        agent_timeout_is_failure=True,
        task_network=json.loads(os.environ["TASK_ANYTERMINAL_NETWORK"])
        if os.environ.get("TASK_ANYTERMINAL_NETWORK")
        else None,
    )
    cfg = agent._setup_params(NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"]))
    task_dir = Path(cfg.problem_info["task_dir"])
    source = task_dir / "solution/solve.sh"
    (cfg.persistent_dir / "control.sh").write_text(source.read_text() if control == "oracle" else "#!/bin/sh\ntrue\n")
    cfg.agent_command_str = "bash /trajectories_mount/control.sh"
    cfg.nemo_gym_root = Path(app.__file__).parents[2]
    cfg.agent_deps_dir = root / "empty-agent-deps"
    cfg.agent_deps_dir.mkdir()
    if task == "regex-log":
        assert cfg.verifier_runtime is None
    result = await app.RunTerminalAgent(config=cfg).process_single_datapoint()
    metrics = json.loads(cfg.metrics_fpath.read_text())
    stdout = (cfg.verifier_dir / "test-stdout.txt").read_text()
    (root / "control-result.json").write_text(
        json.dumps(
            {
                "resolved": result,
                "metrics": metrics,
                "task_dir": str(task_dir),
                "container": cfg.container,
                "test_stdout": stdout,
            },
            indent=2,
        )
        + "\n"
    )
    assert result is expected, metrics
    assert not metrics["mask_sample"], metrics
    assert "test session starts" in stdout
    if task == "feal-differential-cryptanalysis":
        assert "prepared_module" in stdout
        assert "trusted verifier Python is required" not in stdout
