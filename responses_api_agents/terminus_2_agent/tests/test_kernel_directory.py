# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real filesystem and process checks for fakeroot-safe directory validation."""

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import uvloop
from aiohttp import web
from omegaconf import OmegaConf
from responses_api_agents.terminus_2_agent.app import (
    LocalEnvironment,
    Terminus2Agent,
    Terminus2AgentConfig,
    _require_directory,
)

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient


@pytest.mark.parametrize("loop_factory", [asyncio.new_event_loop, uvloop.new_event_loop])
def test_exec_uses_kernel_directory_when_stat_reports_a_file(loop_factory, tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "is_dir", lambda self: False)

    async def run():
        environment = LocalEnvironment(tmp_path, command_timeout_sec=5)
        try:
            result = await environment.exec("printf ran")
            assert (result.return_code, result.stdout) == (0, "ran"), result.stderr
        finally:
            await environment.close()

    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(run())


def test_completed_real_trajectory_survives_a_cleanup_error(tmp_path, monkeypatch, caplog):
    if shutil.which("tmux") is None:
        pytest.skip("tmux is unavailable")
    original_close = LocalEnvironment.close

    async def close_then_fail(environment, **kwargs):
        await original_close(environment, **kwargs)
        raise OSError("injected error after real cleanup")

    monkeypatch.setattr(LocalEnvironment, "close", close_then_fail)

    async def run():
        calls = 0

        async def complete(request):
            nonlocal calls
            await request.json()
            calls += 1
            content = json.dumps({"analysis": "Complete.", "plan": "Finish.", "commands": [], "task_complete": True})
            return web.json_response(
                {
                    "id": "chatcmpl-local",
                    "object": "chat.completion",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
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
        client = ServerClient(
            head_server_config={"host": "127.0.0.1", "port": 0}, global_config_dict=OmegaConf.create({})
        )
        config = Terminus2AgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="app.py",
            name="terminus_2_agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            workspace_root=str(tmp_path),
            logs_root=str(tmp_path / "logs"),
            keep_logs=True,
            model="gpt-4",
            timeout=30,
            model_timeout_sec=5,
            enable_summarize=False,
            max_turns=4,
            record_terminal_session=False,
        )
        agent = Terminus2Agent(config=config, server_client=client)
        try:
            trajectory, _, flags, timed_out, _ = await agent._run_terminus(
                NeMoGymResponseCreateParamsNonStreaming(input="Finish this empty task.", model="gpt-4"),
                "Finish this empty task.",
                f"http://127.0.0.1:{port}/v1",
            )
            assert calls >= 2
            assert trajectory["steps"]
            assert flags["agent_error"] is None and not timed_out
            assert "environment teardown failed (ignored)" in caplog.text
        finally:
            await runner.cleanup()

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_kernel_directory_rejects_missing_and_regular_files(tmp_path, kind):
    target = tmp_path / kind
    if kind == "file":
        target.write_text("regular")
    with pytest.raises(FileNotFoundError, match="not a directory"):
        _require_directory(target, "not a directory")


def test_kernel_directory_preserves_permission_errors(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root bypasses directory search permission")
    hidden = tmp_path / "hidden"
    child = hidden / "child"
    child.mkdir(parents=True)
    hidden.chmod(0)
    try:
        with pytest.raises(PermissionError):
            _require_directory(child, "not a directory")
    finally:
        hidden.chmod(0o700)


def test_kernel_directory_accepts_execute_only_directory(tmp_path):
    tmp_path.chmod(0o111)
    try:
        _require_directory(tmp_path, "not a directory")
    finally:
        tmp_path.chmod(0o700)


@pytest.mark.parametrize("loop_factory", [asyncio.new_event_loop, uvloop.new_event_loop])
def test_deleted_workspace_cleanup_reaps_its_real_tmux_server(loop_factory, tmp_path):
    if shutil.which("tmux") is None:
        pytest.skip("tmux is unavailable")

    async def run():
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        environment = LocalEnvironment(workspace, command_timeout_sec=5)
        socket_root = Path(environment._tmux_directory.name)
        env = {**os.environ, "TMUX_TMPDIR": str(socket_root)}
        env.pop("TMUX", None)
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", "directory-probe", "sleep 60"],
            cwd=workspace,
            env=env,
            check=True,
            timeout=5,
        )
        pid = int(subprocess.check_output(["tmux", "display-message", "-p", "#{pid}"], env=env, text=True, timeout=5))
        try:
            workspace.rmdir()
            await environment.close()
            assert not socket_root.exists()
            for _ in range(50):
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                await asyncio.sleep(0.02)
            else:
                assert Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z"
        finally:
            await environment.close()

    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(run())
