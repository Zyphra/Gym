# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The policy's terminal reads nothing from Terminus-2's own files while its episode runs.

Each command is one a Hero training policy actually typed (behavior audit, 7 Nov 2026), with
the recorded directory names and episode number replaced by this run's.
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest
from aiohttp import web
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.terminus_2_agent.app import LocalEnvironment, Terminus2Agent, Terminus2AgentConfig
from responses_api_agents.terminus_2_agent.output import trajectory_to_responses


RECORDED = {
    # The run directory: Harbor's trajectory and episodes, and the pane log.
    "run_dir": "grep -r 'IoU' /tmp/nemo-gym-terminus-2-wdirc4lr/ 2>/dev/null | head -5",
    # The tmux socket directory beside it.
    "gym_tmux": "ls -la /tmp/gym-tmux-l_n8n01i 2>/dev/null; ls -la /tmp/nemo-gym-terminus-2-i8hnkuto 2>/dev/null",
    # One episode's logged response.
    "episode": "cat /tmp/nemo-gym-terminus-2-drkqdiyi/episode-16/response.txt 2>/dev/null | head -100",
}


def _replayed(command: str, logs_dir: Path, tmux_dir: Path) -> str:
    command = re.sub(r"/tmp/nemo-gym-terminus-2-[\w]+", str(logs_dir), command)
    command = re.sub(r"/tmp/gym-tmux-[\w]+", str(tmux_dir), command)
    return re.sub(r"/episode-\d+/", "/episode-0/", command)


def _reply(commands: list[dict], task_complete: bool) -> str:
    return json.dumps({"analysis": "Look.", "plan": "Probe.", "commands": commands, "task_complete": task_complete})


async def _probe_run(tmp_path: Path, monkeypatch, keep_logs: bool, instruction: str, pane_marker: str):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runs: list[Path] = []
    tmux_dirs: list[Path] = []
    real_logs_dir, real_environment_init = Terminus2Agent._logs_dir, LocalEnvironment.__init__

    def recorded_logs_dir(self) -> Path:
        runs.append(real_logs_dir(self))
        return runs[-1]

    def recorded_environment(self, *args, **kwargs) -> None:
        real_environment_init(self, *args, **kwargs)
        tmux_dirs.append(Path(self._tmux_directory.name))

    monkeypatch.setattr(Terminus2Agent, "_logs_dir", recorded_logs_dir)
    monkeypatch.setattr(LocalEnvironment, "__init__", recorded_environment)
    harness: dict[str, str] = {}
    calls = 0

    async def complete(request):
        nonlocal calls
        await request.json()
        calls += 1
        commands, task_complete = [], calls > 2
        if calls == 1:
            # Terminal output the pane log keeps.
            commands = [{"keystrokes": f"echo {pane_marker}\n", "duration": 0.5}]
        elif calls == 2:
            # Harbor has written the history by now. The same commands from outside the
            # policy's namespace show what each one would read.
            for name, recorded in RECORDED.items():
                command = _replayed(recorded, runs[0], tmux_dirs[0])
                harness[name] = subprocess.run(["bash", "-c", command], capture_output=True, text=True).stdout
                commands.append({"keystrokes": f"{{ {command}; }} > {name}.txt 2>&1\n", "duration": 0.5})
            commands.append({"keystrokes": "touch done.txt\n", "duration": 0.5})
        return web.json_response(
            {
                "id": f"chatcmpl-{calls}",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": _reply(commands, task_complete)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16},
            }
        )

    model = web.Application()
    model.router.add_post("/v1/chat/completions", complete)
    runner = web.AppRunner(model)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    config = Terminus2AgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="terminus_2_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        workspace_root=str(workspace),
        logs_root=str(tmp_path / "logs"),
        keep_logs=keep_logs,
        model="gpt-4",
        timeout=60,
        model_timeout_sec=5,
        enable_summarize=False,
        max_turns=6,
    )
    client = ServerClient(head_server_config={"host": "127.0.0.1", "port": 0}, global_config_dict=OmegaConf.create({}))
    agent = Terminus2Agent(config=config, server_client=client)
    try:
        trajectory, _, flags, timed_out, finished = await agent._run_terminus(
            NeMoGymResponseCreateParamsNonStreaming(input=instruction, model="gpt-4"),
            instruction,
            f"http://127.0.0.1:{port}/v1",
        )
    finally:
        await runner.cleanup()
    assert flags["agent_error"] is None and not timed_out and finished
    return runs[0], workspace, harness, trajectory


@pytest.mark.parametrize("keep_logs", [False, True])
def test_recorded_log_reads_return_nothing_and_output_still_parses(tmp_path, monkeypatch, keep_logs):
    if shutil.which("tmux") is None:
        pytest.skip("tmux is unavailable")
    token = uuid4().hex
    instruction, pane_marker = f"Report the IoU {token}.", f"IoU-pane-{token}"
    run, workspace, harness, trajectory = asyncio.run(
        _probe_run(tmp_path, monkeypatch, keep_logs, instruction, pane_marker)
    )
    assert (workspace / "done.txt").exists()

    # Outside the policy's namespace each recorded command reads the harness's files ...
    assert token in harness["run_dir"] and "trajectory.json" in harness["run_dir"]
    assert f"tmux-{os.getuid()}" in harness["gym_tmux"] and "episode-0" in harness["gym_tmux"]
    assert '"task_complete"' in harness["episode"]
    # ... and from the policy's terminal the same commands read nothing.
    for name in RECORDED:
        seen = (workspace / f"{name}.txt").read_text()
        assert "IoU" not in seen and "tmux-" not in seen and "episode-" not in seen, (name, seen)
        assert "trajectory" not in seen and "task_complete" not in seen and "terminus_2.pane" not in seen, (name, seen)

    # Gym still reads the trajectory Harbor wrote and converts the policy's commands.
    calls = [item for item in trajectory_to_responses(trajectory) if item["type"] == "function_call"]
    assert any("run_dir.txt" in item["arguments"] for item in calls)
    assert token in trajectory["steps"][0]["message"]
    if keep_logs:
        assert json.loads((run / "trajectory.json").read_text()) == trajectory
        # The pane log is still written, though the policy cannot read it.
        assert pane_marker in (run / "terminus_2.pane").read_text(errors="replace")
    else:
        assert not run.exists()


def test_terminal_recording_is_rejected() -> None:
    with pytest.raises(ValidationError, match="readable by the policy"):
        Terminus2AgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="app.py",
            name="terminus_2_agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            record_terminal_session=True,
        )
