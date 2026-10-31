# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A terminal harness's agent observations (context compactions, tool timings) reach the Gym /run result."""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from harbor.models.agent.context import AgentContext

from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.anyterminal_agent import app
from responses_api_agents.anyterminal_agent.app import (
    _INTERNAL_OBSERVATIONS_KEY,
    _RUNNER_TEMPLATE,
    AnyTerminalAgent,
    AnyTerminalRunRequest,
    RunTerminalAgent,
)
from responses_api_agents.anyterminal_agent.tests.test_app import _config, _make_body, _make_instance_config
from responses_api_agents.terminus_2_agent.app import Terminus2Agent, Terminus2AgentConfig


BUNDLE = AgentObservationBundle(
    source="terminus_2",
    records=[
        AgentInvocation(invocation_id="root"),
        ContextCompactionObservation(
            invocation_id="root", trigger="proactive_free_tokens", tokens_before=120_000, outcome="completed"
        ),
        ContextCompactionObservation(invocation_id="root", tokens_before=119_000, outcome="completed"),
    ],
)


class CompactingTerminus2Agent(Terminus2Agent):
    """The real Terminus-2 agent server, whose terminal loop is replaced by a run that compacted twice."""

    async def _run_terminus(self, body, instruction, api_base, observation_collector=None):
        if observation_collector is not None:
            observation_collector(BUNDLE)
        return {"steps": []}, AgentContext(), {}, False, True


def _run_runner(tmp_path: Path, monkeypatch, rollout_id: str | None) -> dict:
    """Execute the rendered in-sandbox runner for a real agent class and read the response it persists."""
    mount = tmp_path / "trajectories_mount"
    mount.mkdir(parents=True)
    (mount / "instruction.txt").write_text("solve the task")
    # The sandbox mounts the trajectory directory at an absolute path; point it at a temporary directory.
    rendered = _RUNNER_TEMPLATE.format(
        agent_module=CompactingTerminus2Agent.__module__,
        agent_class=CompactingTerminus2Agent.__name__,
        agent_cfg_class=Terminus2AgentConfig.__name__,
        agent_class_lower="compactingterminus2agent",
    ).replace("/trajectories_mount", str(mount))
    monkeypatch.setenv("PATH", os.environ.get("PATH", os.defpath))  # the runner appends to it
    monkeypatch.setenv("NGTB_MODEL_NAME", "policy")
    monkeypatch.setenv("NGTB_MODEL_URL", "http://policy.invalid/v1")
    monkeypatch.setenv("NGTB_AGENT_KWARGS", "{}")
    monkeypatch.setenv("NGTB_SAMPLING", "{}")
    if rollout_id is None:
        monkeypatch.delenv("NGTB_ROLLOUT_ID", raising=False)
    else:
        monkeypatch.setenv("NGTB_ROLLOUT_ID", rollout_id)
    exec(compile(rendered, "<runner>", "exec"), {})
    return json.loads((mount / "response.json").read_text())


def test_the_sandbox_env_carries_the_rollout_id_only_when_there_is_one(tmp_path):
    scoped = _make_instance_config(tmp_path / "scoped", rollout_id="rollout-7")
    unscoped = _make_instance_config(tmp_path / "unscoped")

    assert RunTerminalAgent(config=scoped)._agent_env(scoped)["NGTB_ROLLOUT_ID"] == "rollout-7"
    assert "NGTB_ROLLOUT_ID" not in RunTerminalAgent(config=unscoped)._agent_env(unscoped)


def test_the_runner_asks_the_harness_for_observations_only_for_a_rollout(tmp_path, monkeypatch):
    scoped = _run_runner(tmp_path / "scoped", monkeypatch, "rollout-7")
    unscoped = _run_runner(tmp_path / "unscoped", monkeypatch, None)

    assert AgentObservationBundle.model_validate(scoped[_INTERNAL_OBSERVATIONS_KEY]) == BUNDLE
    assert _INTERNAL_OBSERVATIONS_KEY not in unscoped


@pytest.fixture
def persisted_response(tmp_path, monkeypatch) -> dict:
    return _run_runner(tmp_path / "runner", monkeypatch, "rollout-7")


def _agent_server() -> AnyTerminalAgent:
    with patch.object(AnyTerminalAgent, "model_post_init", lambda self, context: None):
        agent = AnyTerminalAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
    agent._sem = asyncio.Semaphore(1)
    return agent


def _episode(tmp_path: Path, response: dict, rollout_id: str | None):
    cfg = _make_instance_config(tmp_path, rollout_id=rollout_id)
    (cfg.persistent_dir / "response.json").write_text(json.dumps(response))
    (cfg.persistent_dir / "agent_result.json").write_text(json.dumps({"return_code": 0, "error_type": None}))
    return cfg


async def test_run_returns_the_harness_observations_with_its_verdict(tmp_path, persisted_response):
    response = app._terminal_response(_episode(tmp_path, persisted_response, "rollout-7"))
    agent = _agent_server()
    agent._responses = AsyncMock(return_value=response)

    result = await agent.run(AnyTerminalRunRequest(responses_create_params=_make_body()))

    assert result.ng_agent_observations == BUNDLE
    assert result.model_dump(mode="json")["ng_agent_observations"] == BUNDLE.model_dump(mode="json")
    # The verdict is for the response the harness produced, not for its observation channel.
    assert _INTERNAL_OBSERVATIONS_KEY not in result.response.model_dump(mode="json")
    assert result.response.output[0].role == "assistant"


async def test_run_without_a_rollout_returns_no_observation_field(tmp_path, persisted_response):
    response = app._terminal_response(_episode(tmp_path, persisted_response, None))
    agent = _agent_server()
    agent._responses = AsyncMock(return_value=response)

    result = await agent.run(AnyTerminalRunRequest(responses_create_params=_make_body()))

    assert result.ng_agent_observations is None
    assert "ng_agent_observations" not in result.model_dump(mode="json")


def test_an_unreadable_response_returns_no_observations(tmp_path):
    unreadable = {_INTERNAL_OBSERVATIONS_KEY: BUNDLE.model_dump(mode="json"), "output": "not a list of items"}

    response = app._terminal_response(_episode(tmp_path, unreadable, "rollout-7"))

    assert _INTERNAL_OBSERVATIONS_KEY not in response.model_dump(mode="json")
    assert json.loads(response.metadata["metrics"])["agent_error"] == "response_unreadable"
