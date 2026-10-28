# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import yaml
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.agents.terminus_2.tmux_session import TmuxSession as Terminus2SessionBase
from harbor.models.agent.context import AgentContext

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.terminus_2_agent.app import (
    COMMAND_NAMESPACE,
    SETUP_FAILED_TMUX_UNAVAILABLE,
    LocalEnvironment,
    LocalTmuxSession,
    StandaloneTerminus2,
    Terminus2Agent,
    Terminus2AgentConfig,
    Terminus2AgentRunRequest,
    TerminusSetupError,
    _extract_instruction,
)
from responses_api_agents.terminus_2_agent.llm import NemoGymLLM
from responses_api_agents.terminus_2_agent.output import trajectory_to_responses


def _config(**kwargs) -> Terminus2AgentConfig:
    return Terminus2AgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="terminus_2_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        **kwargs,
    )


def _make_agent(**kwargs) -> Terminus2Agent:
    with patch("responses_api_agents.terminus_2_agent.app.Terminus2Agent.model_post_init"):
        agent = Terminus2Agent(config=_config(**kwargs), server_client=MagicMock(spec=ServerClient))
    agent.sem = asyncio.Semaphore(agent.config.concurrency)
    return agent


class TestInstructionExtraction:
    def test_string_input(self) -> None:
        assert _extract_instruction("do the task") == ("do the task", None)

    def test_single_user_keeps_existing_format(self) -> None:
        items = [
            NeMoGymEasyInputMessage(role="system", content="be careful"),
            NeMoGymEasyInputMessage(role="user", content="solve it"),
        ]
        assert _extract_instruction(items) == ("solve it", "be careful")

    def test_multiturn_history_is_preserved_with_roles(self) -> None:
        items = [
            NeMoGymEasyInputMessage(role="system", content="be careful"),
            NeMoGymEasyInputMessage(role="user", content="first"),
            NeMoGymEasyInputMessage(role="assistant", content="previous answer"),
            NeMoGymEasyInputMessage(role="user", content="second"),
        ]
        assert _extract_instruction(items) == (
            "System: be careful\n\nUser: first\n\nAssistant: previous answer\n\nUser: second",
            None,
        )


class TestLocalEnvironment:
    @pytest.mark.asyncio
    async def test_exec_keeps_admitted_tools_after_login_path_reset(self, tmp_path: Path) -> None:
        helper_bin = tmp_path / "admitted tools" / "bin"
        helper_bin.mkdir(parents=True)
        helper = helper_bin / "admitted-tool"
        helper.write_text("#!/bin/sh\nprintf 'admitted-tool-ok'")
        helper.chmod(0o755)
        reset = tmp_path / "reset-path.sh"
        reset.write_text("export PATH=/usr/bin:/bin\n")
        environment = LocalEnvironment(tmp_path, command_timeout_sec=5)
        try:
            result = await environment.exec(
                "admitted-tool",
                env={"PATH": f"{helper_bin}:{os.environ['PATH']}", "BASH_ENV": str(reset)},
            )
            assert result.return_code == 0, result.stderr
            assert result.stdout == "admitted-tool-ok"
        finally:
            await environment.close()

    @pytest.mark.asyncio
    async def test_concurrent_environments_own_distinct_tmux_servers(self, tmp_path: Path, monkeypatch) -> None:
        # Contain the unfixed default socket too: never contact a user's server.
        inherited = tmp_path / "inherited"
        inherited.mkdir(mode=0o700)
        monkeypatch.setenv("TMUX_TMPDIR", str(inherited))
        monkeypatch.delenv("TMUX", raising=False)
        assert shutil.which("tmux"), "real tmux is required for this regression"
        environments = [LocalEnvironment(tmp_path, command_timeout_sec=10) for _ in range(2)]
        try:
            pids = []
            for index, environment in enumerate(environments):
                name = f"isolation-{os.getpid()}-{index}"
                started = await environment.exec(f"tmux new-session -d -s {shlex.quote(name)} 'sleep 60'")
                assert started.return_code == 0, started.stderr
                identity = await environment.exec(f"tmux display-message -p -t {shlex.quote(name)} '#{{pid}}'")
                assert identity.return_code == 0, identity.stderr
                pids.append(identity.stdout.strip())
                if index == 0:
                    nested = await environment.exec(
                        f"tmux display-message -p -t {shlex.quote(name)} '#{{socket_path}},#{{pid}},0'"
                    )
                    assert nested.return_code == 0, nested.stderr
                    monkeypatch.setenv("TMUX", nested.stdout.strip())
            assert pids[0] != pids[1], "unique sessions still share the first sandbox's tmux server"
            await environments[0].close()
            await environments[0].close()
            remaining = await environments[1].exec("tmux list-sessions")
            assert remaining.return_code == 0, "one episode's teardown stopped another episode"
            with pytest.raises(RuntimeError, match="closed"):
                await environments[0].exec("tmux list-sessions")
        finally:
            for environment in environments:
                close = getattr(environment, "close", None)
                if close is not None:
                    await close()
                else:
                    await environment.exec("tmux kill-server")

    @pytest.mark.asyncio
    async def test_exec_uses_workspace_and_env(self, tmp_path: Path) -> None:
        environment = LocalEnvironment(tmp_path, command_timeout_sec=5)
        result = await environment.exec('printf \'%s:%s\' "$PWD" "$VALUE"', env={"VALUE": "ok"})
        assert result.return_code == 0
        assert result.stdout == f"{tmp_path}:ok"

    @pytest.mark.asyncio
    async def test_exec_timeout_is_reported(self, tmp_path: Path) -> None:
        environment = LocalEnvironment(tmp_path, command_timeout_sec=0.01)
        result = await environment.exec("sleep 1")
        assert result.return_code == 124
        assert "timed out" in (result.stderr or "")


class TestLocalTmuxSession:
    @pytest.mark.asyncio
    async def test_start_requires_a_live_private_session(self, tmp_path: Path) -> None:
        assert shutil.which("tmux"), "real tmux is required for this regression"
        environment = LocalEnvironment(tmp_path, command_timeout_sec=10)
        session = LocalTmuxSession(
            session_name=f"ready-{os.getpid()}",
            environment=environment,
            logging_path=tmp_path / "pane.log",
            local_asciinema_recording_path=None,
            remote_asciinema_recording_path=None,
        )
        try:
            await session.start()
            result = await environment.exec(f"tmux has-session -t {session._session_name}")
            assert result.return_code == 0, result.stderr
        finally:
            await environment.close()

    @pytest.mark.asyncio
    async def test_zero_launcher_exit_without_session_fails(self, tmp_path: Path) -> None:
        class EmptyLauncher(LocalTmuxSession):
            @property
            def _tmux_start_session(self) -> str:
                return "true"

        assert shutil.which("tmux"), "real tmux is required for this regression"
        environment = LocalEnvironment(tmp_path, command_timeout_sec=10)
        session = EmptyLauncher(
            session_name=f"missing-{os.getpid()}",
            environment=environment,
            logging_path=tmp_path / "pane.log",
            local_asciinema_recording_path=None,
            remote_asciinema_recording_path=None,
        )
        try:
            with pytest.raises(RuntimeError, match="session is unavailable"):
                await session.start()
        finally:
            await environment.close()

    def test_macos_launcher_does_not_use_linux_script_flags(self, tmp_path: Path) -> None:
        session = LocalTmuxSession(
            session_name="terminus-test",
            environment=MagicMock(),
            logging_path=tmp_path / "pane.log",
            local_asciinema_recording_path=None,
            remote_asciinema_recording_path=None,
        )
        with patch("responses_api_agents.terminus_2_agent.app.sys.platform", "darwin"):
            command = session._tmux_start_session
        assert "script -qc" not in command
        assert "tmux new-session" in command

    def test_linux_launcher_runs_upstream_start_in_the_command_namespace(self, tmp_path: Path) -> None:
        session = LocalTmuxSession(
            session_name="terminus-test",
            environment=MagicMock(),
            logging_path=tmp_path / "pane.log",
            local_asciinema_recording_path=None,
            remote_asciinema_recording_path=None,
        )
        with patch("responses_api_agents.terminus_2_agent.app.sys.platform", "linux"):
            command = session._tmux_start_session
        upstream = Terminus2SessionBase._tmux_start_session.fget(session)
        assert shlex.split(command) == [sys.executable, "-I", str(COMMAND_NAMESPACE), "bash", "-c", upstream]
        assert "script -qc" in upstream


async def _wait_for(predicate, timeout_s: float = 20.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not await predicate():
        assert asyncio.get_running_loop().time() < deadline, "condition not reached"
        await asyncio.sleep(0.1)


class TestCommandNamespace:
    """The agent's commands cannot end the harness that types them."""

    async def _session(self, tmp_path: Path) -> tuple[LocalEnvironment, LocalTmuxSession]:
        assert shutil.which("tmux"), "real tmux is required for this regression"
        environment = LocalEnvironment(tmp_path, command_timeout_sec=30)
        session = LocalTmuxSession(
            session_name=f"contained-{os.getpid()}",
            environment=environment,
            logging_path=tmp_path / "pane.log",
            local_asciinema_recording_path=None,
            remote_asciinema_recording_path=None,
        )
        await session.start()
        return environment, session

    @pytest.mark.asyncio
    async def test_agent_kill_commands_reach_only_the_agent_namespace(self, tmp_path: Path) -> None:
        token = f"harness-sentinel-{uuid4().hex}"
        # A python process beside the harness, named so that an agent's pkill matches it.
        sentinel = subprocess.Popen([sys.executable, "-c", "import time, sys; time.sleep(300)", token])
        environment, session = await self._session(tmp_path)
        try:
            # "[h]arness" matches the sentinel's name but not these commands' own argv.
            pattern = "[h]" + token[1:]
            await session.send_keys(
                [
                    f"grep -l '{pattern}' /proc/[0-9]*/cmdline > seen.txt; "
                    f"pkill -9 -f '{pattern}'; pkill -9 -f python; echo $? > done.txt",
                    "Enter",
                ],
                min_timeout_sec=0.5,
            )

            async def finished() -> bool:
                return (tmp_path / "done.txt").exists()

            await _wait_for(finished)
            assert (tmp_path / "seen.txt").read_text() == ""
            assert sentinel.poll() is None
            assert await session.is_session_alive()

            # Everything the agent can signal goes, the harness side stays.
            await session.send_keys(["kill -9 -1", "Enter"], min_timeout_sec=0.5)

            async def session_gone() -> bool:
                return not await session.is_session_alive()

            await _wait_for(session_gone)
            assert sentinel.poll() is None
            result = await environment.exec("echo harness-alive")
            assert result.return_code == 0 and result.stdout.strip() == "harness-alive"
        finally:
            sentinel.kill()
            sentinel.wait()
            await environment.close()

    @pytest.mark.asyncio
    async def test_agent_commands_are_the_preferred_oom_victims(self, tmp_path: Path) -> None:
        harness_score = Path("/proc/self/oom_score_adj").read_text().strip()
        environment, session = await self._session(tmp_path)
        try:
            await session.send_keys(["cat /proc/self/oom_score_adj > adj.txt", "Enter"], min_timeout_sec=0.5)

            async def written() -> bool:
                return (tmp_path / "adj.txt").exists() and bool((tmp_path / "adj.txt").read_text().strip())

            await _wait_for(written)
            assert (tmp_path / "adj.txt").read_text().strip() == "1000"
            assert Path("/proc/self/oom_score_adj").read_text().strip() == harness_score
        finally:
            await environment.close()

    def test_start_failure_is_reported_with_its_cause(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing-command"
        result = subprocess.run(
            [sys.executable, "-I", str(COMMAND_NAMESPACE), str(missing)], capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 127
        assert str(missing) in result.stderr

    def test_command_status_returns_while_its_daemons_keep_running(self, tmp_path: Path) -> None:
        marker = tmp_path / "daemon.txt"
        script = f"(sleep 1; echo alive > {shlex.quote(str(marker))}) >/dev/null 2>&1 & exit 7"
        result = subprocess.run(
            [sys.executable, "-I", str(COMMAND_NAMESPACE), "bash", "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 7
        assert not marker.exists()
        deadline = time.monotonic() + 20
        while not marker.exists():
            assert time.monotonic() < deadline, "the namespace ended its command's daemon"
            time.sleep(0.05)


class TestStandaloneTerminus2:
    @pytest.mark.asyncio
    async def test_tracks_two_consecutive_completion_claims(self) -> None:
        agent = object.__new__(StandaloneTerminus2)
        agent._consecutive_completion_claims = 0
        non_completion = ([], False, "", "analysis", "plan", MagicMock())
        completion = ([], True, "", "analysis", "plan", MagicMock())

        with patch.object(Terminus2, "_handle_llm_interaction", new=AsyncMock(side_effect=[completion, completion])):
            await agent._handle_llm_interaction()
            assert agent.finished_naturally is False
            await agent._handle_llm_interaction()
            assert agent.finished_naturally is True

        with patch.object(Terminus2, "_handle_llm_interaction", new=AsyncMock(return_value=non_completion)):
            await agent._handle_llm_interaction()
        assert agent.finished_naturally is False

    @pytest.mark.asyncio
    async def test_swallowed_agent_error_is_named(self) -> None:
        agent = object.__new__(StandaloneTerminus2)
        agent.logger = MagicMock()

        with patch.object(Terminus2, "run", new=AsyncMock(side_effect=RuntimeError("model endpoint 500"))):
            await agent.run("task", MagicMock(), AgentContext())
        assert agent.agent_error == "RuntimeError"
        agent.logger.warning.assert_called_once()

        with patch.object(Terminus2, "run", new=AsyncMock(return_value=None)):
            await agent.run("task", MagicMock(), AgentContext())
        assert agent.agent_error is None


class TestTrajectoryOutput:
    def test_raw_content_preserves_training_fields_and_commands(self) -> None:
        trajectory = {
            "steps": [
                {
                    "source": "agent",
                    "message": 'prefix {"commands":[{"keystrokes":"ls\\n","duration":0.1}]} suffix',
                    "reasoning_content": "inspect",
                    "metrics": {
                        "prompt_token_ids": [1],
                        "completion_token_ids": [2],
                        "logprobs": [-0.1],
                        "extra": {"routed_experts": [[[3]]]},
                    },
                    "observation": {"results": [{"content": "file.txt"}]},
                }
            ]
        }

        output = trajectory_to_responses(trajectory)

        assert [item["type"] for item in output] == ["message", "function_call", "function_call_output"]
        assert output[0]["content"][0]["text"].startswith("<think>inspect</think>")
        assert output[0]["generation_token_ids"] == [2]
        assert output[0]["routed_experts"] == [[[3]]]
        assert output[1]["call_id"] == output[2]["call_id"] == "call_0_1"


class TestResponses:
    @pytest.mark.asyncio
    async def test_default_workspace_is_temporary_and_removed(self) -> None:
        agent = _make_agent()
        seen_workspace: Path | None = None
        terminus = MagicMock(finished_naturally=True)
        terminus.setup = AsyncMock()
        terminus.close = AsyncMock()

        async def capture_workspace(_instruction, environment, _context) -> None:
            nonlocal seen_workspace
            seen_workspace = environment.cwd
            assert seen_workspace.is_dir()

        terminus.run = AsyncMock(side_effect=capture_workspace)
        agent._build_agent = MagicMock(return_value=terminus)
        body = NeMoGymResponseCreateParamsNonStreaming(model="model", input="task")

        await agent._run_terminus(body, "task", "http://model/v1")

        assert seen_workspace is not None
        assert not seen_workspace.exists()

    @pytest.mark.asyncio
    async def test_converts_terminus_trajectory(self) -> None:
        agent = _make_agent(system_prompt="system config", model="test-model")
        trajectory = {
            "steps": [
                {"step_id": 1, "source": "user", "message": "task"},
                {
                    "step_id": 2,
                    "source": "agent",
                    "message": "Analysis: inspect\nPlan: list files",
                    "tool_calls": [
                        {
                            "tool_call_id": "call_0_1",
                            "function_name": "bash_command",
                            "arguments": {"keystrokes": "ls\n", "duration": 0.1},
                        }
                    ],
                    "observation": {"results": [{"content": "file.txt"}]},
                    "metrics": {"prompt_tokens": 10, "completion_tokens": 4},
                },
            ],
            "final_metrics": {"total_prompt_tokens": 10, "total_completion_tokens": 4},
        }
        context = AgentContext(n_input_tokens=10, n_output_tokens=4, n_cache_tokens=0)
        agent._run_terminus = AsyncMock(return_value=(trajectory, context, {}, False, True))

        body = NeMoGymResponseCreateParamsNonStreaming(
            model="ignored-request-model",
            input=[
                NeMoGymEasyInputMessage(role="system", content="input system"),
                NeMoGymEasyInputMessage(role="user", content="solve it"),
            ],
        )
        with patch.object(Terminus2Agent, "resolve_model_base_url", return_value="http://model/v1"):
            response = await agent.responses(request=None, body=body)

        assert response.model == "test-model"
        assert response.temperature == 0.7
        assert [item.type for item in response.output] == ["message", "function_call", "function_call_output"]
        assert response.usage.total_tokens == 14
        assert '"finished_naturally": true' in response.metadata["terminus_2"]
        called_instruction = agent._run_terminus.await_args.args[1]
        assert called_instruction == "system config\n\ninput system\n\nsolve it"

    @pytest.mark.asyncio
    async def test_empty_trajectory_gets_assistant_message(self) -> None:
        agent = _make_agent()
        agent._run_terminus = AsyncMock(return_value=({"steps": []}, AgentContext(), {}, False, False))
        body = NeMoGymResponseCreateParamsNonStreaming(model="model", input="task")

        with patch.object(Terminus2Agent, "resolve_model_base_url", return_value="http://model/v1"):
            response = await agent.responses(request=None, body=body)

        assert len(response.output) == 1
        assert response.output[0].type == "message"

    @pytest.mark.asyncio
    async def test_agent_error_is_labelled_through_run(self) -> None:
        agent = _make_agent()
        flags = {"context_length_exceeded": False, "agent_error": "RuntimeError"}
        agent._run_terminus = AsyncMock(return_value=({"steps": []}, AgentContext(), flags, False, False))
        body = NeMoGymResponseCreateParamsNonStreaming(model="model", input="task")
        with patch.object(Terminus2Agent, "resolve_model_base_url", return_value="http://model/v1"):
            response = await agent.responses(request=None, body=body)
        # A wrapping agent reads this harness-neutral key without parsing terminus_2.
        assert response.metadata["agent_error"] == "RuntimeError"

        response_json = response.model_dump(mode="json")
        verify_json = {
            "responses_create_params": body.model_dump(mode="json"),
            "response": response_json,
            "reward": 0.0,
        }
        agent.server_client.post = AsyncMock(return_value=MagicMock(cookies={}))
        with (
            patch("responses_api_agents.terminus_2_agent.app.raise_for_status", new=AsyncMock()),
            patch(
                "responses_api_agents.terminus_2_agent.app.get_response_json",
                new=AsyncMock(side_effect=[response_json, verify_json]),
            ),
        ):
            verified = await agent.run(MagicMock(cookies={}), Terminus2AgentRunRequest(responses_create_params=body))
        assert verified.agent_failed is True
        assert verified.agent_error == "RuntimeError"
        assert verified.reward == 0.0

        agent._run_terminus = AsyncMock(return_value=({"steps": []}, AgentContext(), {}, False, True))
        with patch.object(Terminus2Agent, "resolve_model_base_url", return_value="http://model/v1"):
            response = await agent.responses(request=None, body=body)
        assert "agent_error" not in response.metadata

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("flags", "timed_out", "confirmed", "expected"),
        [
            ({}, False, True, True),  # task_complete confirmed twice
            ({}, False, False, False),  # turn cap or an unconfirmed claim
            ({}, True, True, False),  # harness timeout
            ({"agent_error": "RuntimeError"}, False, True, False),  # harness exception
            ({"model_transport_error": "upstream 502"}, False, True, False),
            ({"stop_reason": "turn_limit"}, False, True, False),  # endpoint stop
            ({"stop_reason": "identical_call"}, False, True, False),  # endpoint guard
            ({"context_length_exceeded": True}, False, True, False),
        ],
    )
    async def test_finished_naturally_is_only_the_confirmed_completion(
        self, flags, timed_out, confirmed, expected
    ) -> None:
        agent = _make_agent()
        agent._run_terminus = AsyncMock(return_value=({"steps": []}, AgentContext(), flags, timed_out, confirmed))
        body = NeMoGymResponseCreateParamsNonStreaming(model="model", input="task")
        with patch.object(Terminus2Agent, "resolve_model_base_url", return_value="http://model/v1"):
            response = await agent.responses(request=None, body=body)
        # A wrapping agent reads this harness-neutral key without parsing terminus_2.
        assert response.metadata["finished_naturally"] == ("true" if expected else "false")
        assert json.loads(response.metadata["terminus_2"])["finished_naturally"] is expected

        response_json = response.model_dump(mode="json")
        verify_json = {
            "responses_create_params": body.model_dump(mode="json"),
            "response": response_json,
            "reward": 0.0,
        }
        agent.server_client.post = AsyncMock(return_value=MagicMock(cookies={}))
        with (
            patch("responses_api_agents.terminus_2_agent.app.raise_for_status", new=AsyncMock()),
            patch(
                "responses_api_agents.terminus_2_agent.app.get_response_json",
                new=AsyncMock(side_effect=[response_json, verify_json]),
            ),
        ):
            verified = await agent.run(MagicMock(cookies={}), Terminus2AgentRunRequest(responses_create_params=body))
        assert verified.finished_naturally is expected

    @pytest.mark.asyncio
    async def test_model_client_close_failure_is_logged(self) -> None:
        agent = object.__new__(StandaloneTerminus2)
        agent.logger = MagicMock()
        llm = object.__new__(NemoGymLLM)
        llm.context_length_exceeded = False
        llm.aclose = AsyncMock(side_effect=OSError("connection reset"))
        agent._llm = llm
        agent._attach_routed_experts_to_trajectory = MagicMock()

        with patch.object(Terminus2, "run", new=AsyncMock(return_value=None)):
            await agent.run("task", MagicMock(), AgentContext())
        agent.logger.warning.assert_called_once_with(
            "Model client close failed: %s: %s", "OSError", llm.aclose.side_effect
        )

    @pytest.mark.asyncio
    async def test_direct_response_waits_for_concurrency_slot(self) -> None:
        agent = _make_agent()
        agent.sem = asyncio.Semaphore(0)
        agent._run_terminus = AsyncMock(return_value=({"steps": []}, AgentContext(), {}, False, False))
        body = NeMoGymResponseCreateParamsNonStreaming(model="model", input="task")

        with patch.object(Terminus2Agent, "resolve_model_base_url", return_value="http://model/v1"):
            task = asyncio.create_task(agent.responses(request=None, body=body))
            await asyncio.sleep(0)
            agent._run_terminus.assert_not_awaited()
            agent.sem.release()
            await task

        agent._run_terminus.assert_awaited_once()


class TestConfig:
    def test_webserver_routes_build(self) -> None:
        _make_agent().setup_webserver()

    def test_defaults(self) -> None:
        config = _config()
        assert config.parser_name == "json"
        assert config.concurrency == 1
        assert config.record_terminal_session is False
        assert config.trajectory_config == {"raw_content": False}

    @pytest.mark.parametrize(
        "relative_path,config_name",
        [
            ("configs/terminus_2_agent.yaml", "terminus_2_agent"),
            ("../anyterminal_agent/configs/anyterminal_terminus_2.yaml", "anyterminal_terminus_2"),
        ],
    )
    def test_yaml_parses(self, relative_path: str, config_name: str) -> None:
        path = Path(__file__).resolve().parent.parent / relative_path
        data = yaml.safe_load(path.read_text())
        assert config_name in data

    def test_requirements_share_harbor_pin(self) -> None:
        agent_dir = Path(__file__).resolve().parent.parent
        terminus_requirement = next(
            line for line in (agent_dir / "requirements.txt").read_text().splitlines() if line.startswith("harbor @")
        )
        harbor_requirement = next(
            line
            for line in (agent_dir.parent / "harbor_agent" / "requirements.txt").read_text().splitlines()
            if line.startswith("harbor @")
        )
        assert terminus_requirement == harbor_requirement


def _setup_agent(logs_dir: Path) -> StandaloneTerminus2:
    agent = object.__new__(StandaloneTerminus2)
    agent.logs_dir = logs_dir
    agent._record_terminal_session = False
    agent._tmux_pane_width, agent._tmux_pane_height = 160, 40
    agent._session = None
    agent.logger = MagicMock()
    return agent


def _path_without_tmux() -> str:
    return os.pathsep.join(
        entry
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry and not (Path(entry) / "tmux").exists()
    )


class TestSetupFailureLabel:
    @pytest.mark.asyncio
    async def test_setup_without_tmux_fails_labelled_with_its_first_cause(self, tmp_path: Path, monkeypatch) -> None:
        # bash and tmux can share a directory. Keep a real shell available
        # while removing tmux, so this exercises the intended setup failure.
        shell_bin = tmp_path / "shell-bin"
        shell_bin.mkdir()
        for name in ("bash", "script"):
            executable = shutil.which(name)
            assert executable
            (shell_bin / name).symlink_to(executable)
        monkeypatch.setenv("PATH", str(shell_bin) + os.pathsep + _path_without_tmux())
        assert shutil.which("tmux") is None
        environment = LocalEnvironment(tmp_path, command_timeout_sec=120)
        agent = _setup_agent(tmp_path)
        try:
            with pytest.raises(TerminusSetupError) as raised:
                await agent.setup(environment)
        finally:
            await environment.close()
        assert raised.value.label == SETUP_FAILED_TMUX_UNAVAILABLE
        # Harbor's installer logged and returned; its first ERROR line is the cause.
        assert raised.value.first_cause.startswith("Failed to ") and "tmux" in raised.value.first_cause
        assert str(raised.value).startswith(f"{SETUP_FAILED_TMUX_UNAVAILABLE}: first cause: Failed to ")
        assert len(raised.value.first_cause) <= 200

    @pytest.mark.asyncio
    async def test_healthy_setup_and_run_carry_no_label(self, tmp_path: Path) -> None:
        assert shutil.which("tmux"), "real tmux is required for this control"
        environment = LocalEnvironment(tmp_path, command_timeout_sec=30)
        agent = _setup_agent(tmp_path)
        try:
            await agent.setup(environment)
            assert agent._setup_errors == []
            with patch.object(Terminus2, "run", new=AsyncMock(return_value=None)):
                await agent.run("task", environment, AgentContext())
            assert agent.agent_error is None
        finally:
            await agent.close(environment)
            await environment.close()

    @pytest.mark.asyncio
    async def test_run_on_a_dead_session_is_a_labelled_setup_failure(self, tmp_path: Path) -> None:
        assert shutil.which("tmux"), "real tmux is required for this control"
        environment = LocalEnvironment(tmp_path, command_timeout_sec=30)
        agent = _setup_agent(tmp_path)
        try:
            await agent.setup(environment)
            await environment.exec("tmux kill-server")
            with patch.object(Terminus2, "run", new=AsyncMock(return_value=None)) as native_run:
                await agent.run("task", environment, AgentContext())
            native_run.assert_awaited_once()
            assert agent.agent_error == SETUP_FAILED_TMUX_UNAVAILABLE
            agent.logger.warning.assert_any_call(
                "Agent setup failed: %s: %s", SETUP_FAILED_TMUX_UNAVAILABLE, "tmux session is not alive after setup"
            )
        finally:
            await environment.close()

    @pytest.mark.asyncio
    async def test_session_query_failure_keeps_the_run_cleanup(self, tmp_path: Path) -> None:
        assert shutil.which("tmux"), "real tmux is required for this control"
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        environment = LocalEnvironment(workspace, command_timeout_sec=30)
        agent = _setup_agent(tmp_path)
        llm = MagicMock(spec=NemoGymLLM)
        llm.context_length_exceeded = False
        llm.aclose = AsyncMock()
        agent._llm = llm
        agent._attach_routed_experts_to_trajectory = MagicMock()
        try:
            await agent.setup(environment)
            # The liveness query itself fails: its working directory is gone.
            shutil.rmtree(workspace)
            with patch.object(Terminus2, "run", new=AsyncMock(return_value=None)) as native_run:
                await agent.run("task", environment, AgentContext())
            native_run.assert_not_awaited()
            assert agent.agent_error is not None and agent.agent_error != SETUP_FAILED_TMUX_UNAVAILABLE
            agent._attach_routed_experts_to_trajectory.assert_called_once()
            llm.aclose.assert_awaited_once()
        finally:
            workspace.mkdir(exist_ok=True)
            await environment.exec("tmux kill-server")
            await environment.close()

    @pytest.mark.asyncio
    async def test_cancelled_session_query_keeps_the_run_cleanup(self, tmp_path: Path) -> None:
        assert shutil.which("tmux"), "real tmux is required for this control"
        environment = LocalEnvironment(tmp_path, command_timeout_sec=30)
        agent = _setup_agent(tmp_path)
        llm = MagicMock(spec=NemoGymLLM)
        llm.context_length_exceeded = False
        llm.aclose = AsyncMock()
        agent._llm = llm
        agent._attach_routed_experts_to_trajectory = MagicMock()
        querying = asyncio.Event()

        async def blocked_query() -> bool:
            querying.set()
            await asyncio.Event().wait()
            return True

        try:
            await agent.setup(environment)
            agent._session.is_session_alive = blocked_query
            run = asyncio.create_task(agent.run("task", environment, AgentContext()))
            await querying.wait()
            run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run
            agent._attach_routed_experts_to_trajectory.assert_called_once()
            llm.aclose.assert_awaited_once()
        finally:
            await agent.close(environment)
            await environment.close()


class TestSetupErrorRecorder:
    @pytest.mark.asyncio
    async def test_keeps_only_the_first_nonempty_cause_per_task(self) -> None:
        from responses_api_agents.terminus_2_agent.app import _SETUP_ERRORS

        harbor = logging.getLogger("harbor.utils.logger")

        async def setup(first: str) -> list[str]:
            errors: list[str] = []
            token = _SETUP_ERRORS.set(errors)
            try:
                harbor.error("   \n")
                await asyncio.sleep(0)
                harbor.error("%s\nsecond line", first)
                await asyncio.sleep(0)
                harbor.error("a later error " + "x" * 400)
            finally:
                _SETUP_ERRORS.reset(token)
            return errors

        one, two = await asyncio.gather(setup("Failed to install tmux from source"), setup("y" * 400))
        assert one == ["Failed to install tmux from source"]
        assert two == ["y" * 200]
        harbor.error("outside any setup")
        assert _SETUP_ERRORS.get() is None


@pytest.mark.asyncio
async def test_swallowed_model_transport_error_reaches_existing_harness_failure_rule():
    from nemo_gym.rollout_observability import harness_response_error

    agent = _make_agent()
    flags = {"agent_error": "HTTPStatusError", "model_transport_error": "Model endpoint HTTP 503"}
    trajectory = {"steps": [{"source": "agent", "message": '{"commands":[]}', "reasoning_content": "inspect"}]}
    agent._run_terminus = AsyncMock(return_value=(trajectory, AgentContext(), flags, False, False))
    body = NeMoGymResponseCreateParamsNonStreaming(model="model", input="task")
    with patch.object(Terminus2Agent, "resolve_model_base_url", return_value="http://model/v1"):
        response = await agent.responses(request=None, body=body)
    assert response.status == "failed"
    assert response.output
    assert response.metadata["model_transport_error"] == "true"
    assert harness_response_error(response) == response.error
    assert response.error.message == "Model endpoint HTTP 503"


@pytest.mark.asyncio
async def test_model_failure_custody_survives_standalone_exception_handler():
    agent = object.__new__(StandaloneTerminus2)
    agent.logger = MagicMock()
    agent._llm = NemoGymLLM(model_name="model", api_base="http://localhost/v1")
    agent._llm.model_transport_error = "ReadTimeout: model request expired"
    with patch.object(Terminus2, "run", new=AsyncMock(side_effect=RuntimeError("request failed"))):
        await agent.run("task", MagicMock(), AgentContext())
    assert agent.agent_error == "RuntimeError"
    assert agent.model_transport_error == "ReadTimeout: model request expired"
