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
"""Codex inside a task sandbox: the task workdir, rollout correlation and the pinned offline CLI.

The agent is built the way AnySWE's ``agent_runner`` builds it inside the sandbox, and ``codex`` is a
real executable on PATH that answers ``--version`` and records what ``codex exec`` was given.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

import pytest
from fastapi import Request

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.codex_agent.app import CodexAgent, CodexAgentConfig


PINNED = "0.144.4"
MODEL_URL = "http://policy-model.invalid:9000"
ANYSWE = Path(__file__).resolve().parents[2] / "anyswe_agent"

_FAKE_CODEX = """#!{python}
import json, os, sys, tomllib
from pathlib import Path

if sys.argv[1:] == ["--version"]:
    print({version_line!r})
    sys.exit({version_exit})
config = tomllib.loads((Path(os.environ["CODEX_HOME"]) / "config.toml").read_text())
record = {{"argv": sys.argv, "base_url": config["model_providers"]["gym"]["base_url"], "config": config}}
Path(os.environ["FAKE_CODEX_RECORD"]).write_text(json.dumps(record))
print(json.dumps({{"type": "item.completed", "item": {{"id": "item_0", "type": "agent_message", "text": "done"}}}}))
print(json.dumps({{"type": "turn.completed", "usage": {{"input_tokens": 3, "output_tokens": 2}}}}))
"""

_FAKE_NPM = """#!{python}
import os, sys
from pathlib import Path

Path(os.environ["FAKE_NPM_RECORD"]).write_text(" ".join(sys.argv))
sys.exit(1)
"""


def _executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A PATH with a recording ``npm`` and no ``codex``, a private HOME, and record files."""
    bin_dir = tmp_path / "bin"
    _executable(bin_dir / "npm", _FAKE_NPM.format(python=sys.executable))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FAKE_CODEX_RECORD", str(tmp_path / "codex-record.json"))
    monkeypatch.setenv("FAKE_NPM_RECORD", str(tmp_path / "npm-record.txt"))
    return tmp_path


def _install_codex(directory: Path, version_line: str = f"codex-cli {PINNED}", version_exit: int = 0) -> Path:
    return _executable(
        directory / "codex",
        _FAKE_CODEX.format(python=sys.executable, version_line=version_line, version_exit=version_exit),
    )


def _agent(**config) -> CodexAgent:
    """The agent as AnySWE's agent_runner constructs it in the task sandbox."""
    client = ServerClient.model_construct(
        global_config_dict={"policy_model": {"responses_api_models": {"model": {"host": "0.0.0.0", "port": 0}}}}
    )
    client._build_server_base_url = lambda server_config: MODEL_URL
    return CodexAgent(
        config=CodexAgentConfig(
            host="0.0.0.0",
            port=0,
            name="codexagent",
            entrypoint="app.py",
            model_server=ModelServerRef(name="policy_model", type="responses_api_models"),
            resources_server=ResourcesServerRef(name="anyswe", type="resources_servers"),
            **{"codex_version": PINNED, **config},
        ),
        server_client=client,
    )


def _respond(agent: CodexAgent, rollout_id: Optional[str]) -> dict:
    path_params = {"rollout_id": rollout_id} if rollout_id else {}
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[NeMoGymEasyInputMessage(role="user", content="Fix the bug")], model="policy"
    )
    response = asyncio.run(agent.responses(request=Request({"type": "http", "path_params": path_params}), body=body))
    assert response.output[-1].content[0].text == "done"
    return json.loads(Path(os.environ["FAKE_CODEX_RECORD"]).read_text())


def _cd(record: dict) -> str:
    return record["argv"][record["argv"].index("--cd") + 1]


@pytest.mark.parametrize(
    ("config", "rollout_id", "base_url"),
    [
        ({}, "swe-gym-task-3", f"{MODEL_URL}/ng-rollout/swe-gym-task-3/v1"),
        ({}, None, f"{MODEL_URL}/v1"),
        # Under a policy sandbox the relay URL is already scoped and is used verbatim.
        ({"sandbox_model_base_url": "http://127.0.0.1:18080"}, "swe-gym-task-3", "http://127.0.0.1:18080/v1"),
    ],
)
def test_codex_runs_the_pinned_cli_in_the_task_workdir(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, config: dict, rollout_id: Optional[str], base_url: str
):
    codex = _install_codex(sandbox / "bin")
    testbed = sandbox / "testbed"
    (testbed / "pkg").mkdir(parents=True)
    monkeypatch.chdir(testbed)  # AnySWE starts the agent in the task workdir

    agent = _agent(codex_install="offline", cwd=".", **config)
    # A codex placed ahead on PATH after startup is never run: requests use the verified executable.
    _install_codex(sandbox / "later-bin")
    monkeypatch.setenv("PATH", f"{sandbox / 'later-bin'}{os.pathsep}{os.environ['PATH']}")
    record = _respond(agent, rollout_id)

    assert record["argv"][0] == str(codex)
    assert _cd(record) == str(testbed)
    assert record["base_url"] == base_url
    assert (testbed / "pkg").is_dir()  # the task repository is never removed


def test_default_cwd_is_still_a_fresh_scratch_dir(sandbox: Path, monkeypatch: pytest.MonkeyPatch):
    _install_codex(sandbox / "bin")
    testbed = sandbox / "testbed"
    testbed.mkdir()
    monkeypatch.chdir(testbed)

    agent = _agent()
    record = _respond(agent, None)

    assert agent._build_command("hi", "/work")[0] == "codex"
    assert Path(_cd(record)).name.startswith("nemo_gym_codex_ws_")
    assert not Path(_cd(record)).exists()


def test_cwd_that_is_not_a_directory_fails_the_request(sandbox: Path, monkeypatch: pytest.MonkeyPatch):
    _install_codex(sandbox / "bin")
    monkeypatch.chdir(sandbox)

    with pytest.raises(NotADirectoryError, match="'testbed' is not a directory"):
        _respond(_agent(codex_install="offline", cwd="testbed"), None)
    assert not Path(os.environ["FAKE_CODEX_RECORD"]).exists()


@pytest.mark.parametrize("version_line", ["codex-cli 0.145.0", "codex-cli 0.144.40", "codex 0.144.4-alpha.1", "?"])
def test_offline_startup_fails_fast_on_a_version_mismatch(sandbox: Path, version_line: str):
    _install_codex(sandbox / "bin", version_line)

    with pytest.raises(RuntimeError, match=f"but codex_version pins '{PINNED}'"):
        _agent(codex_install="offline")
    assert not Path(os.environ["FAKE_NPM_RECORD"]).exists()


@pytest.mark.parametrize("version_exit", [1, 23])
def test_offline_startup_fails_fast_when_the_version_probe_exits_non_zero(sandbox: Path, version_exit: int):
    _install_codex(sandbox / "bin", version_exit=version_exit)  # prints the pinned version, then fails

    with pytest.raises(RuntimeError, match=f"reports codex version None but codex_version pins '{PINNED}'"):
        _agent(codex_install="offline")
    assert not Path(os.environ["FAKE_NPM_RECORD"]).exists()


def test_offline_startup_fails_fast_when_codex_is_missing(sandbox: Path):
    # A pinned binary outside PATH (auto mode's ~/.local/bin fallback) is not a substitute.
    _install_codex(sandbox / "home" / ".local" / "bin")

    with pytest.raises(RuntimeError, match=f"codex {PINNED} is not on PATH"):
        _agent(codex_install="offline")
    assert not Path(os.environ["FAKE_NPM_RECORD"]).exists()


def test_auto_install_mode_keeps_using_any_codex_on_path(sandbox: Path):
    _install_codex(sandbox / "bin", "codex-cli 0.145.0")

    agent = _agent()

    assert agent.config.codex_install == "auto"
    assert agent._build_command("hi", "/work")[0] == "codex"
    assert not Path(os.environ["FAKE_NPM_RECORD"]).exists()


def test_anyswe_codex_runs_the_pinned_offline_cli_on_the_task_repository(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
):
    """AnySWE's Codex config, built as agent_runner builds it in a policy sandbox: the runtime's
    pinned CLI on the task repository, calling the relay's Responses API, with skills, the
    user-input tool and hosted features off."""
    from omegaconf import OmegaConf

    monkeypatch.setenv("ANYSWE_CONTAINER_FORMATTER", "docker://swebench/{instance_id}")
    monkeypatch.delenv("NEMO_GYM_SANDBOX_MODEL_BASE_URL", raising=False)
    loaded = OmegaConf.to_container(OmegaConf.load(ANYSWE / "configs" / "anyswe_codex.yaml"), resolve=True)
    anyswe = loaded["anyswe_codex"]["responses_api_agents"]["anyswe_agent"]
    assert (anyswe["agent_server_module"], anyswe["agent_server_class"], anyswe["agent_config_class"]) == (
        "responses_api_agents.codex_agent.app",
        "CodexAgent",
        "CodexAgentConfig",
    )
    kwargs = anyswe["agent_kwargs"]
    assert (kwargs["codex_version"], kwargs["codex_install"], kwargs["cwd"]) == (PINNED, "offline", ".")
    # The runtime installs exactly the version the agent accepts.
    assert f"\nCODEX_VERSION={PINNED}\n" in (ANYSWE / "setup_scripts" / "codex_agent_deps.sh").read_text()

    codex = _install_codex(sandbox / "bin")
    testbed = sandbox / "testbed"
    testbed.mkdir()
    monkeypatch.chdir(testbed)
    record = _respond(_agent(**kwargs, sandbox_model_base_url="http://127.0.0.1:18080"), "swe-gym-task-3")

    assert record["argv"][0] == str(codex)
    assert _cd(record) == str(testbed)
    config = record["config"]
    assert config["model_providers"]["gym"] | {"stream_idle_timeout_ms": None} == {
        "name": "gym",
        "base_url": "http://127.0.0.1:18080/v1",
        "env_key": "OPENAI_API_KEY",
        "wire_api": "responses",
        "stream_idle_timeout_ms": None,
    }
    assert config["model_providers"]["gym"]["stream_idle_timeout_ms"] == kwargs["timeout"] * 1000
    assert config["model"] == "gym-policy-model" and "model_context_window" not in config
    assert config["developer_instructions"] == kwargs["system_prompt"]
    assert (config["web_search"], config["check_for_update_on_startup"], config["analytics"]) == (
        "disabled",
        False,
        {"enabled": False},
    )
    assert config["project_doc_max_bytes"] == 0
    assert config["skills"] == {"bundled": {"enabled": False}, "include_instructions": False}
    assert config["tools"] == {"experimental_request_user_input": {"enabled": False}}
    assert {"multi_agent", "code_mode", "apps", "plugins", "memories", "enable_request_compression"} <= set(
        config["features"]
    )
    assert not any(config["features"].values())
