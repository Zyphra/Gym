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

    runner._become_subreaper()
    started = time.monotonic()
    runner._run_and_capture(Agent(), None, Body(), None, out / "index", None, out)
    print(json.dumps({"seconds": time.monotonic() - started}))
    """
)


def _drive(tmp_path: Path, mode: str, **env) -> tuple[dict, dict, list[int]]:
    root = Path(__file__).resolve().parents[3]
    done = subprocess.run(
        [sys.executable, "-c", DRIVER, mode, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": f"{root}:{os.environ.get('PYTHONPATH', '')}", **env},
    )
    assert done.returncode == 0, done.stderr[-2000:]
    summary = json.loads(done.stdout.strip().splitlines()[-1])
    response = json.loads((tmp_path / "response.json").read_text())
    pids = [int(pid) for pid in (tmp_path / "pids").read_text().split()]
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
    assert response["metadata"]["agent_wall_timeout"] == "true"
    assert response["error"]["message"] == "Harness exception: TimeoutError"
    assert len(pids) == 1 and not _alive(pids[0])


def test_agent_phase_leaves_an_extraction_reserve():
    from responses_api_agents.anyswe_agent.app import _agent_phase_seconds

    assert _agent_phase_seconds(31800) == 31500
    assert _agent_phase_seconds(600) == 540
    assert _agent_phase_seconds(1) == 1
