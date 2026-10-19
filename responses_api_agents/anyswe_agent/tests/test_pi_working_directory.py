# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Pi subprocess and AnySWE runner controls without a CLI install or model IO."""

import asyncio
import json
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from nemo_gym.config_types import ResourcesServerRef
from responses_api_agents.anyswe_agent import agent_runner
from responses_api_agents.pi_agent.app import PiAgent, PiAgentConfig


SCRIPT = """
import json,os
from pathlib import Path
marker=Path('marker.py')
observed={'cwd':os.getcwd(),'marker_exists':marker.exists()}
if marker.exists(): marker.write_text('changed=1\\n')
print(json.dumps({'type':'message_end','message':{'role':'assistant','content':[{'type':'text','text':json.dumps(observed)}]}}))
"""


def kwargs(root):
    return {
        "model": "fixture",
        "workspace_root": str(root / "state"),
        "command": shlex.join([sys.executable, "-c", SCRIPT]),
    }


def make_agent(root, **extra):
    config = PiAgentConfig(
        host="127.0.0.1",
        port=1,
        name="fixture",
        entrypoint="",
        resources_server=ResourcesServerRef(type="resources_servers", name="fixture"),
        **(kwargs(root) | extra),
    )
    # Only startup installation is suppressed. The real config, _run_pi,
    # subprocess, event parser and cleanup execute; no model/HTTP server is used.
    with patch("responses_api_agents.pi_agent.app.ensure_pi", return_value=None):
        return PiAgent.model_construct(config=config, server_client=None)


@pytest.mark.parametrize("task_directory", [False, True])
def test_pi_separates_caller_directory_from_temporary_home(tmp_path, monkeypatch, task_directory):
    repo = tmp_path / "task repository"
    repo.mkdir()
    marker = repo / "marker.py"
    marker.write_text("original=1\n")
    monkeypatch.chdir(repo)
    extra = {"working_directory": str(repo)} if task_directory else {}
    result = asyncio.run(make_agent(tmp_path, **extra)._run_pi("fixture", None, collect_observations=False))
    observed = json.loads(result[0][0].content[0].text)
    assert observed["marker_exists"] is task_directory
    assert (Path(observed["cwd"]) == repo) is task_directory
    assert marker.read_text() == ("changed=1\n" if task_directory else "original=1\n")
    assert repo.is_dir()
    assert list((tmp_path / "state").iterdir()) == []


def test_invalid_pi_working_directory_preserves_caller_files(tmp_path):
    repo = tmp_path / "task repository"
    repo.mkdir()
    marker = repo / "marker.py"
    marker.write_text("original=1\n")
    with pytest.raises(FileNotFoundError):
        asyncio.run(
            make_agent(tmp_path, working_directory=str(repo / "missing"))._run_pi(
                "fixture",
                None,
                collect_observations=False,
            )
        )
    assert marker.read_text() == "original=1\n"
    assert list((tmp_path / "state").iterdir()) == []


@pytest.mark.parametrize("explicit_directory", [False, True])
def test_anyswe_actual_runner_projects_task_directory_and_extracts_patch(tmp_path, monkeypatch, explicit_directory):
    repo = tmp_path / "task repository"
    repo.mkdir()
    (repo / "marker.py").write_text("original=1\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "marker.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "instruction.txt").write_text("Read and change marker.py in the task repository.")
    other = tmp_path / "explicit directory"
    other.mkdir()
    (other / "marker.py").write_text("other=1\n")
    settings = kwargs(tmp_path)
    if explicit_directory:
        settings["working_directory"] = str(other)
    for key, value in {
        "NGSWE_MODEL_NAME": "fixture",
        "NGSWE_MODEL_URL": "",
        "NGSWE_AGENT_MODULE": "responses_api_agents.pi_agent.app",
        "NGSWE_AGENT_CLASS": "PiAgent",
        "NGSWE_AGENT_CONFIG_CLASS": "PiAgentConfig",
        "NGSWE_AGENT_KWARGS": json.dumps(settings),
        "NGSWE_SAMPLING": "{}",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("NGSWE_AGENT_KWARGS_B64", raising=False)
    monkeypatch.delenv("NGSWE_SAMPLING_B64", raising=False)
    monkeypatch.chdir(repo)

    class FixturePaths:
        def __new__(cls, path="."):
            path = Path(path)
            if path == Path("/trajectories_mount"):
                return artifacts
            return artifacts / path.name if path.parent == Path("/trajectories_mount") else path

        cwd = staticmethod(Path.cwd)

    # Relocate the runner's artifact/dependency mounts and repo search candidates.
    # Actual runner config construction, agent.responses, subprocess and Git patch
    # extraction run unchanged; there is no fake grader or model endpoint.
    monkeypatch.setattr(agent_runner, "Path", FixturePaths)
    monkeypatch.setattr(agent_runner, "_REPO_CANDIDATES", (str(repo),))
    deps = tmp_path / "agent dependencies"
    deps.mkdir()
    monkeypatch.setattr(agent_runner, "agent_deps_dir", str(deps), raising=False)
    with patch("responses_api_agents.pi_agent.app.ensure_pi", return_value=None):
        agent_runner.main()
    response = json.loads((artifacts / "response.json").read_text())
    observed = json.loads(response["output"][0]["content"][0]["text"])
    assert observed == {"cwd": str(other if explicit_directory else repo), "marker_exists": True}
    patch_text = (artifacts / "patch.diff").read_text()
    if explicit_directory:
        assert patch_text == ""
        assert (repo / "marker.py").read_text() == "original=1\n"
    else:
        assert "+changed=1" in patch_text and "-original=1" in patch_text
    assert repo.is_dir() and other.is_dir()
    assert list((tmp_path / "state").iterdir()) == []
