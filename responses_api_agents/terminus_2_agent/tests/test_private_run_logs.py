# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The policy's terminal cannot read Terminus-2's own run logs while its episode runs."""

import asyncio
import json
import os
import shlex
import shutil
from pathlib import Path
from uuid import uuid4

import pytest
from aiohttp import web
from omegaconf import OmegaConf

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.terminus_2_agent.app import HARNESS_LOGS, Terminus2Agent, Terminus2AgentConfig
from responses_api_agents.terminus_2_agent.output import trajectory_to_responses


def _reply(commands: list[dict], task_complete: bool) -> str:
    return json.dumps({"analysis": "Look.", "plan": "Probe.", "commands": commands, "task_complete": task_complete})


async def _probe_run(tmp_path: Path, monkeypatch, keep_logs: bool, instruction: str):
    """Run the real Terminus loop; on its second turn the policy tries to read the run's logs."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runs: list[Path] = []
    real_logs_dir = Terminus2Agent._logs_dir

    def recorded_logs_dir(self) -> Path:
        runs.append(real_logs_dir(self))
        return runs[-1]

    monkeypatch.setattr(Terminus2Agent, "_logs_dir", recorded_logs_dir)
    harness_view: list[str] = []
    calls = 0

    async def complete(request):
        nonlocal calls
        await request.json()
        calls += 1
        commands, task_complete = [], calls > 2
        if calls == 2:
            # Harbor has written these from the harness by the time it asks for this turn.
            harness = runs[0] / HARNESS_LOGS
            harness_view.extend(sorted(os.listdir(harness)))
            leaked = " ".join(shlex.quote(str(harness / name)) for name in ("trajectory.json", "episode-0/prompt.txt"))
            probe = (
                f"ls -A {shlex.quote(str(runs[0]))} > root.txt; ls -A {shlex.quote(str(harness))} > harness.txt; "
                f"cat {leaked} > leaked.txt 2>&1; echo $? > done.txt\n"
            )
            commands = [{"keystrokes": probe, "duration": 1.0}]
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
        record_terminal_session=False,
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
    return runs[0], workspace, harness_view, trajectory


@pytest.mark.parametrize("keep_logs", [False, True])
def test_policy_terminal_sees_no_run_logs_and_output_still_parses(tmp_path, monkeypatch, keep_logs):
    if shutil.which("tmux") is None:
        pytest.skip("tmux is unavailable")
    instruction = f"Do the task {uuid4().hex}."
    run, workspace, harness_view, trajectory = asyncio.run(_probe_run(tmp_path, monkeypatch, keep_logs, instruction))

    # The harness had the history on disk while the policy's commands ran ...
    assert {"trajectory.json", "episode-0", "episode-1"} <= set(harness_view)
    # ... and the policy's terminal saw an empty directory and could read none of it.
    assert (workspace / "done.txt").read_text().strip() != "0"
    assert (workspace / "harness.txt").read_text() == ""
    leaked = (workspace / "leaked.txt").read_text()
    assert "No such file or directory" in leaked and instruction not in leaked
    # Only the terminal's own pane log stays beside it.
    assert (workspace / "root.txt").read_text().split() == [HARNESS_LOGS, "terminus_2.pane"]

    # Gym still reads the trajectory Harbor wrote and converts the probe turn.
    calls = [item for item in trajectory_to_responses(trajectory) if item["type"] == "function_call"]
    assert any("leaked.txt" in item["arguments"] for item in calls)
    assert trajectory["steps"][0]["message"].find(instruction) >= 0

    if keep_logs:
        assert json.loads((run / HARNESS_LOGS / "trajectory.json").read_text()) == trajectory
        assert "leaked.txt" in (run / "terminus_2.pane").read_text(errors="replace")
    else:
        assert not run.exists()
