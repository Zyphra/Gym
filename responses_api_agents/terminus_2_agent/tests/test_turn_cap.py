# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Terminus-2's own max_turns stop is named, so wrapping agents count it like every other harness cap."""

import asyncio
import json
import shutil

import pytest
from aiohttp import web
from omegaconf import OmegaConf

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.terminus_2_agent.app import Terminus2Agent, Terminus2AgentConfig


def _reply(task_complete: bool) -> str:
    return json.dumps({"analysis": "Working.", "plan": "Continue.", "commands": [], "task_complete": task_complete})


async def _scripted_run(tmp_path, task_complete: bool, max_turns: int):
    calls = 0

    async def complete(request):
        nonlocal calls
        await request.json()
        calls += 1
        return web.json_response(
            {
                "id": f"chatcmpl-{calls}",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": _reply(task_complete)},
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
        workspace_root=str(tmp_path),
        logs_root=str(tmp_path / "logs"),
        keep_logs=True,
        model="gpt-4",
        timeout=60,
        model_timeout_sec=5,
        enable_summarize=False,
        max_turns=max_turns,
        record_terminal_session=False,
    )
    client = ServerClient(head_server_config={"host": "127.0.0.1", "port": 0}, global_config_dict=OmegaConf.create({}))
    agent = Terminus2Agent(config=config, server_client=client)
    try:
        _, _, flags, timed_out, finished = await agent._run_terminus(
            NeMoGymResponseCreateParamsNonStreaming(input="Do the task.", model="gpt-4"),
            "Do the task.",
            f"http://127.0.0.1:{port}/v1",
        )
    finally:
        await runner.cleanup()
    return calls, flags, timed_out, finished


@pytest.mark.parametrize(
    "task_complete, calls, turn_cap, finished",
    [
        (False, 3, True, False),  # never claims completion: Harbor's loop runs out at max_turns
        (True, 2, False, True),  # claim + confirmation inside the budget: a natural finish
    ],
)
def test_real_loop_names_its_own_turn_cap(tmp_path, task_complete, calls, turn_cap, finished):
    if shutil.which("tmux") is None:
        pytest.skip("tmux is unavailable")
    served, flags, timed_out, natural = asyncio.run(_scripted_run(tmp_path, task_complete, max_turns=3))
    assert served == calls
    assert flags["turn_cap"] is turn_cap and natural is finished
    assert flags["agent_error"] is None and flags["stop_reason"] is None and not timed_out


def test_confirmed_completion_on_the_last_turn_is_not_a_cap(tmp_path):
    if shutil.which("tmux") is None:
        pytest.skip("tmux is unavailable")
    served, flags, _, natural = asyncio.run(_scripted_run(tmp_path, True, max_turns=2))
    assert served == 2 and natural is True and flags["turn_cap"] is False
