# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-process regressions: the runner reaps what a harness leaves behind and enforces its deadline.

Each case runs the actual runner functions in a fresh interpreter so the child
subreaper setting never applies to the pytest process.
"""

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs Linux /proc and prctl")

DRIVER = textwrap.dedent(
    """
    import asyncio, json, os, resource, subprocess, sys, time
    from pathlib import Path
    from responses_api_agents.anyswe_agent import agent_runner as runner

    mode, out = sys.argv[1], Path(sys.argv[2])
    pids = out / "pids"

    class Body:
        model = "m"; tool_choice = "auto"; tools = []; parallel_tool_calls = True

    class Agent:
        async def responses(self, request=None, body=None):
            if mode == "transport":
                # A model call that timed out long before the agent's wall.
                raise TimeoutError("model read timed out")
            if mode == "thread":
                # A harness whose loop runs in a thread: cancellation never reaches it.
                (repo / "fixed.txt").write_text("fixed\\n")
                await asyncio.to_thread(time.sleep, 600)
                return runner._exception_response(RuntimeError("loop returned"), Body())
            if mode == "orphan":
                # A test runner left in the background, escaping into its own session.
                subprocess.run(
                    ["sh", "-c", f"setsid sh -c 'echo $$ >> {pids}; exec sleep 600' & "
                                 f"sh -c 'echo $$ >> {pids}; exec sleep 600' &"],
                    check=True,
                )
                for _ in range(100):
                    if pids.exists() and len(pids.read_text().split()) == 2:
                        break
                    time.sleep(0.05)
                return runner._exception_response(RuntimeError("done"), Body())
            # A tool that holds memory at its hard limit and never returns.
            def limit():
                resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
            code = (
                "import os, time\\n"
                f"open({str(pids)!r}, 'a').write(str(os.getpid()) + '\\\\n')\\n"
                "held = []\\n"
                "try:\\n"
                "    while True: held.append(bytearray(16 << 20))\\n"
                "except MemoryError: pass\\n"
                "time.sleep(600)\\n"
            )
            await asyncio.to_thread(subprocess.run, [sys.executable, "-c", code], preexec_fn=limit)
            return runner._exception_response(RuntimeError("tool returned"), Body())

    repo = tree = None
    if mode == "thread":
        repo = out / "repo"
        repo.mkdir()
        git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        (repo / "base.txt").write_text("base\\n")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run([*git, "commit", "-qm", "base"], cwd=repo, check=True)
        tree = runner._snapshot_repo(repo, out / "index")
    runner._become_subreaper()
    runner._run_and_capture(Agent(), None, Body(), repo, out / "index", tree, out)
    """
)


def _drive(tmp_path: Path, mode: str, **env) -> tuple[dict, dict, list[int]]:
    root = Path(__file__).resolve().parents[3]
    started = time.monotonic()
    done = subprocess.run(
        [sys.executable, "-c", DRIVER, mode, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": f"{root}:{os.environ.get('PYTHONPATH', '')}", **env},
    )
    assert done.returncode == 0, done.stderr[-2000:]
    # The runner may leave from its deadline thread, so the run is timed from here.
    summary = {"seconds": time.monotonic() - started, "stdout": done.stdout}
    response = json.loads((tmp_path / "response.json").read_text())
    pids = [int(pid) for pid in (tmp_path / "pids").read_text().split()] if (tmp_path / "pids").exists() else []
    return summary, response, pids


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as stream:
            return stream.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def test_orphaned_harness_processes_are_reaped_before_patch_extraction(tmp_path):
    summary, response, pids = _drive(tmp_path, "orphan")
    assert len(pids) == 2
    assert not any(_alive(pid) for pid in pids)
    assert "agent_wall_timeout" not in (response.get("metadata") or {})


def test_wall_deadline_kills_a_tool_held_at_its_memory_limit(tmp_path):
    summary, response, pids = _drive(tmp_path, "hog", NGSWE_AGENT_TIMEOUT_S="2")
    assert summary["seconds"] < 30
    # The wall is the agent's ending, not a failure of the harness or the model endpoint.
    assert response["metadata"] == {"agent_wall_timeout": "true"}
    assert response["error"] is None and response["status"] == "incomplete"
    assert len(pids) == 1 and not _alive(pids[0])


def test_wall_deadline_collects_the_patch_of_a_harness_that_does_not_stop(tmp_path):
    summary, response, _ = _drive(tmp_path, "thread", NGSWE_AGENT_TIMEOUT_S="2", NGSWE_AGENT_STOP_GRACE_S="2")
    # Without the runner's own exit the interpreter waits for the harness thread (asyncio joins
    # its executor for up to 300 s), which is where the sandbox exec timeout used to land first.
    assert summary["seconds"] < 60
    assert "agent did not stop at its wall" in summary["stdout"]
    assert response["metadata"] == {"agent_wall_timeout": "true"}
    assert response["error"] is None
    patch = (tmp_path / "patch.diff").read_text()
    assert "+++ b/fixed.txt" in patch and "+fixed" in patch
    assert not (tmp_path / "index").exists() and not (tmp_path / "index.git").exists()


def test_timeout_raised_before_the_wall_stays_a_transport_failure(tmp_path):
    _, response, _ = _drive(tmp_path, "transport", NGSWE_AGENT_TIMEOUT_S="600")
    assert response["metadata"] == {"runner_exception": "true", "model_transport_error": "true"}
    assert response["error"]["message"] == "Harness exception: TimeoutError"


def test_agent_phase_leaves_an_extraction_reserve():
    from responses_api_agents.anyswe_agent.app import _agent_phase_seconds

    assert _agent_phase_seconds(31800) == 31500
    assert _agent_phase_seconds(600) == 540
    assert _agent_phase_seconds(1) == 1


def test_stop_grace_is_a_fifth_of_the_extraction_reserve():
    from responses_api_agents.anyswe_agent.app import _agent_stop_grace_seconds

    assert _agent_stop_grace_seconds(3000) == 60
    assert _agent_stop_grace_seconds(600) == 12
    assert _agent_stop_grace_seconds(1) == 1
