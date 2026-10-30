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
"""Contracts for the LHTB ``continue_until_timeout`` phases of anyterminal_agent.

The sandbox is a scripted stand-in for the allocated one; the staged tests, the
bound verifier directory, the integrity checks and the metrics file are real.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from responses_api_agents.anyterminal_agent import continue_until_timeout as phases
from responses_api_agents.anyterminal_agent.app import RunTerminalAgent, TerminalBenchMetrics, _reward
from responses_api_agents.anyterminal_agent.tests.test_app import _make_instance_config, _sandbox_result


INSTRUCTION = "Build the thing.\n"


def _task(root: Path, *, flag: bool | None = True) -> Path:
    task = root / "task"
    (task / "tests").mkdir(parents=True)
    (task / "tests" / "test.sh").write_text("echo hidden\n")
    (task / "tests" / "expected.txt").write_text("hidden expected values\n")
    agent = "" if flag is None else f"[agent]\ncontinue_until_timeout = {str(flag).lower()}\ntimeout_sec = 3600.0\n"
    (task / "task.toml").write_text(agent)
    return task


class ScriptedSandbox:
    """Runs the agent phases and the verifier from a reward schedule, recording what each sees."""

    def __init__(self, cfg, rewards, *, agent_seconds=0.0, agent_error=None):
        self.cfg, self.rewards = cfg, list(rewards)
        self.agent_seconds, self.agent_error = agent_seconds, agent_error
        self.events: list[str] = []
        self.agent_views: list[dict] = []
        self.start, self.stop = AsyncMock(), AsyncMock()

    async def exec(self, command, **kwargs):
        cfg = self.cfg
        if "run-the-agent" in command:
            self.events.append("agent")
            self.agent_views.append(
                {
                    "instruction": (cfg.persistent_dir / "instruction.txt").read_text(),
                    "verifier_entries": sorted(path.name for path in cfg.verifier_dir.iterdir()),
                    "staged_tests": (cfg.persistent_dir / "staging" / "tests").exists(),
                    "timeout_s": kwargs.get("timeout_s"),
                }
            )
            (cfg.persistent_dir / "response.json").write_text(json.dumps({"phase": len(self.agent_views) - 1}))
            await asyncio.sleep(self.agent_seconds)
            return _sandbox_result(124 if self.agent_error else 0, self.agent_error)
        if "kill -STOP" in command or "kill -CONT" in command:
            self.events.append("freeze" if "kill -STOP" in command else "thaw")
            return _sandbox_result()
        if "bash /tests/test.sh" in command:
            self.events.append("verify")
            (cfg.verifier_dir / "reward.txt").write_text(
                str(self.rewards.pop(0) if len(self.rewards) > 1 else self.rewards[0])
            )
            (cfg.verifier_dir / "pytest.log").write_text("FAILED expected 42\n")
            return _sandbox_result()
        return _sandbox_result()


async def _run(tmp_path, rewards, *, flag=True, scalar=True, agent_seconds=0.0, agent_error=None, **overrides):
    task = _task(tmp_path, flag=flag)
    cfg = _make_instance_config(
        tmp_path,
        problem_info={
            "task_name": "demo",
            "task_dir": str(task),
            "docker_image": "demo:1",
            "cpus": 2,
            "memory_mb": 4096,
        },
        agent_command_str="run-the-agent",
        continue_until_timeout=True,
        scalar_reward=scalar,
        **overrides,
    )
    (cfg.persistent_dir / "instruction.txt").write_text(INSTRUCTION)
    sandbox = ScriptedSandbox(cfg, rewards, agent_seconds=agent_seconds, agent_error=agent_error)
    return cfg, sandbox, task


@pytest.fixture(autouse=True)
def _no_real_provider():
    with patch("responses_api_agents.anyterminal_agent.app._build_provider", return_value=None):
        yield


async def _process(cfg, sandbox):
    with patch("responses_api_agents.anyterminal_agent.app.AsyncSandbox", return_value=sandbox):
        resolved = await RunTerminalAgent(config=cfg).process_single_datapoint()
    return resolved, json.loads(cfg.metrics_fpath.read_text())


def test_the_flag_is_read_from_the_task_declaration(tmp_path: Path) -> None:
    assert phases.task_continues_until_timeout(_task(tmp_path / "on", flag=True))
    assert not phases.task_continues_until_timeout(_task(tmp_path / "off", flag=False))
    assert not phases.task_continues_until_timeout(_task(tmp_path / "unset", flag=None))
    assert not phases.task_continues_until_timeout(tmp_path / "missing")


def test_the_rejection_names_no_verifier_detail() -> None:
    text = phases.continuation_instruction(INSTRUCTION, phase=2, remaining_s=1234.9)
    assert text.startswith("Build the thing.\n\n---\nYour submitted solution did not pass verification.")
    assert text.endswith("This is continuation 2; approximately 1234 seconds remain.")


async def test_a_failed_checkpoint_resumes_the_agent_and_a_pass_ends_the_trial(tmp_path: Path) -> None:
    cfg, sandbox, task = await _run(tmp_path, [0.4, 0.7, 1.0])
    resolved, metrics = await _process(cfg, sandbox)

    assert resolved is True
    assert (metrics["verifier_reward"], metrics["resolved"]) == (1.0, True)
    assert metrics["continue_phases"] == 3
    assert [point["reward"] for point in metrics["checkpoints"]] == [0.4, 0.7, 1.0]
    # Agent, freeze, verifier, thaw per checkpoint; the final verifier grades without a freeze.
    assert sandbox.events == ["agent", "freeze", "verify", "thaw"] * 3 + ["verify"]
    views = sandbox.agent_views
    assert views[0]["instruction"] == INSTRUCTION
    for phase, view in enumerate(views[1:], start=1):
        assert view["instruction"].startswith(INSTRUCTION.rstrip())
        assert f"This is continuation {phase};" in view["instruction"]
        assert "0.4" not in view["instruction"] and "42" not in view["instruction"]
    # Nothing of the grader survives into the next agent phase.
    assert [view["verifier_entries"] for view in views] == [[], [], []]
    assert [view["staged_tests"] for view in views] == [False, False, False]
    assert (cfg.persistent_dir / "instruction.txt").read_text() == INSTRUCTION
    assert not (cfg.persistent_dir / "staging").exists()
    # Each phase's agent output and each interim grade stay for audit.
    assert sorted(path.name for path in cfg.persistent_dir.glob("response-phase-*.json")) == [
        f"response-phase-{phase:04d}.json" for phase in range(3)
    ]
    interim = cfg.persistent_dir / "interim_verifier"
    assert [path.name for path in sorted(interim.iterdir())] == [f"phase-{phase:04d}" for phase in range(3)]
    assert (interim / "phase-0001" / "reward.txt").read_text() == "0.7"
    assert (task / "tests" / "expected.txt").read_text() == "hidden expected values\n"


async def test_each_phase_is_bounded_by_the_checkpoint_interval_and_the_budget_is_not_a_failure(
    tmp_path: Path,
) -> None:
    cfg, sandbox, _task_dir = await _run(
        tmp_path,
        [0.3],
        tb_agent_timeout=3,
        checkpoint_interval_s=1,
        agent_timeout_is_failure=True,
        agent_seconds=0.7,
        agent_error="timeout",
    )
    resolved, metrics = await _process(cfg, sandbox)

    assert metrics["continue_phases"] >= 3
    assert all(view["timeout_s"] <= 1 for view in sandbox.agent_views)
    # Spending the budget is the normal end: the final verifier's value is the reward.
    assert resolved is False
    assert (metrics["verifier_reward"], metrics["agent_timed_out"], metrics["mask_sample"]) == (0.3, False, False)
    assert _reward(TerminalBenchMetrics.model_validate(metrics), True) == 0.3


async def test_a_task_without_the_flag_runs_the_agent_once(tmp_path: Path) -> None:
    cfg, sandbox, _task_dir = await _run(tmp_path, [0.5], flag=False)
    _resolved, metrics = await _process(cfg, sandbox)

    assert sandbox.events == ["agent", "verify"]
    assert metrics["verifier_reward"] == 0.5 and "continue_phases" not in metrics


async def test_the_agent_is_thawed_even_when_the_interim_grade_fails(tmp_path: Path) -> None:
    cfg, sandbox, _task_dir = await _run(tmp_path, [1.0])

    async def broken_verifier(command, **kwargs):
        if "bash /tests/test.sh" in command:
            sandbox.events.append("verify")
            raise RuntimeError("sandbox lost")
        return await ScriptedSandbox.exec(sandbox, command, **kwargs)

    sandbox.exec = broken_verifier
    _resolved, metrics = await _process(cfg, sandbox)

    assert sandbox.events[:4] == ["agent", "freeze", "verify", "thaw"]
    assert metrics["sandbox_failed"] is True and metrics["mask_sample"] is True
    assert not (cfg.persistent_dir / "staging" / "tests").exists()


@pytest.mark.parametrize(("scalar", "expected"), [(True, 0.62), (False, 1.0)])
def test_the_reward_is_the_verifier_value_only_when_scalar(scalar: bool, expected: float) -> None:
    metrics = TerminalBenchMetrics(resolved=True, verifier_reward=0.62)
    assert _reward(metrics, scalar) == expected
    assert _reward(TerminalBenchMetrics(resolved=False), scalar) == 0.0


async def test_scalar_reward_calls_a_row_solved_at_the_lhtb_threshold(tmp_path: Path) -> None:
    for reward, solved in ((0.949, False), (0.95, True)):
        cfg, sandbox, _task_dir = await _run(tmp_path / str(reward), [reward], flag=False)
        resolved, metrics = await _process(cfg, sandbox)
        assert (resolved, metrics["resolved"], metrics["verifier_reward"]) == (solved, solved, reward)


def test_phases_need_a_bind_mount_provider(tmp_path: Path) -> None:
    from responses_api_agents.anyterminal_agent.tests.test_app import AnyTerminalAgent, ServerClient, _config

    config = _config(
        model_server=None,
        sandbox_provider={"e2b": {}},
        agent_runtime_source="baked",
        results_dir=tmp_path / "results",
        continue_until_timeout=True,
    )
    with pytest.raises(ValueError, match="bind-mount sandbox provider"):
        AnyTerminalAgent(config=config, server_client=ServerClient.model_construct(global_config_dict={}))
