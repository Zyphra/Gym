# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real tmux process lifetime across the opted-in benchmark handoff."""
import asyncio
import os
import shutil
import signal

import pytest

from responses_api_agents.terminus_2_agent.app import LocalEnvironment, Terminus2AgentConfig


@pytest.mark.asyncio
async def test_preserved_tmux_shell_survives_agent_environment_close(tmp_path):
    if not shutil.which("tmux"):
        pytest.skip("requires the admitted tmux runtime")
    environment = LocalEnvironment(tmp_path, 10)
    started = await environment.exec("tmux new-session -d -s proof 'sleep 120'")
    assert started.return_code == 0, started
    identity = await environment.exec("tmux display-message -p -t proof '#{pid} #{pane_pid}'")
    assert identity.return_code == 0, identity
    server, shell = map(int, identity.stdout.split())
    try:
        await environment.close(terminate=False)
        os.kill(server, 0)
        os.kill(shell, 0)
    finally:
        for pid in (shell, server):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        await asyncio.sleep(0.05)


def test_standalone_terminus_keeps_teardown_enabled_by_default():
    assert Terminus2AgentConfig.model_fields["teardown_tmux"].default is True
