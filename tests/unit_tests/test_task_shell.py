# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Task commands run through task_shell.py cannot reach the harness's mounts or processes.

Each sandbox test runs as root of its own user, mount and PID namespace with its
own ``/proc`` (``unshare -r -m -p -f --mount-proc``), as a remapped-root task
sandbox does, with a bind-mounted harness runtime and a plain run directory.
"""

import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from nemo_gym.sandbox import task_shell


TASK_SHELL = Path(task_shell.__file__)


def _sandbox(tmp_path: Path, harness_script: str) -> tuple[subprocess.CompletedProcess, Path, Path]:
    """Run ``harness_script`` as a sandbox's harness with the task shell on its PATH."""
    runtime, deps, run = tmp_path / "runtime", tmp_path / "agent_deps_mount", tmp_path / "trajectories_mount"
    (runtime / "bin").mkdir(parents=True)
    deps.mkdir()
    run.mkdir()
    (runtime / "harness.py").write_text("harness source\n")
    (runtime / "bin" / "python3.13").symlink_to(sys.executable)
    (run / "instruction.txt").write_text("run files\n")
    helper = run / "task_shell.py"
    shutil.copyfile(TASK_SHELL, helper)
    install = f"import runpy; print(runpy.run_path({str(helper)!r})['install']({str(helper)!r}, ({str(deps)!r}, {str(run)!r})))"
    script = (
        f"mount --bind {runtime} {deps} || exit 90\n"
        # A harness process whose root and cwd are in the harness's mount namespace.
        f"(cd {deps} && exec sleep 600) &\n"
        f"shim=$({sys.executable} -I -c {shlex.quote(install)}) || exit 91\n"
        f"export PATH=$shim:/usr/local/bin:/usr/bin:/bin:{deps}/bin\n"
        f"cd {tmp_path}\n" + harness_script.replace("@DEPS@", str(deps)).replace("@RUN@", str(run))
    )
    result = subprocess.run(
        # unshare --fork leaves SIGINT and SIGTERM ignored in its child; a sandbox does not.
        [
            "unshare",
            "-r",
            "-m",
            "-p",
            "-f",
            "--mount-proc",
            sys.executable,
            "-c",
            "import os, signal, sys; [signal.signal(n, signal.SIG_DFL) for n in (2, 15)]; "
            "os.execvp('bash', ['bash', '-c', sys.argv[1]])",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result, deps, run


def _lines(result: subprocess.CompletedProcess) -> dict[str, str]:
    return {line.split("=", 1)[0]: line.split("=", 1)[1] for line in result.stdout.splitlines() if "=" in line}


def test_task_commands_cannot_reach_the_harness_runtime_run_files_or_processes(tmp_path: Path) -> None:
    probe = (
        'echo "own_view=$(cat @DEPS@/harness.py 2>&1 | head -1)|$(ls -A @RUN@ 2>&1)"\n'
        'echo "path=$PATH"\n'
        'echo "harness_python=$(command -v python3.13 || echo none)"\n'
        'echo "cwd=$PWD"\n'
        'leaks=""\n'
        'for pid in $(ls /proc | grep -E "^[0-9]+$"); do\n'
        '  cat /proc/$pid/root/@DEPS@/harness.py >/dev/null 2>&1 && leaks="$leaks proc-root:$pid"\n'
        '  if command -v nsenter >/dev/null; then nsenter -t $pid -m cat @DEPS@/harness.py >/dev/null 2>&1 && leaks="$leaks nsenter:$pid"; fi\n'
        "done\n"
        'echo "background=$(pgrep -x sleep | wc -l)"\n'
        "umount -l /proc 2>/dev/null\n"
        'for pid in $(ls /proc 2>/dev/null | grep -E "^[0-9]+$"); do\n'
        '  cat /proc/$pid/root/@DEPS@/harness.py >/dev/null 2>&1 && leaks="$leaks under-proc:$pid"\n'
        "done\n"
        'echo "leaks=$leaks"\n'
    )
    harness = (
        "bash -c 'sleep 600 & exit 3'; echo \"first_status=$?\"\n"
        "cat > /tmp/probe.sh <<'PROBE'\n" + probe + "PROBE\n"
        "bash /tmp/probe.sh\n"
        'echo "harness_after=$(cat @DEPS@/harness.py)|$(cat @RUN@/instruction.txt)"\n'
    )
    result, _, _ = _sandbox(tmp_path, harness)
    assert result.returncode == 0, result.stdout + result.stderr
    seen = _lines(result)
    assert seen["first_status"] == "3"
    assert seen["own_view"].endswith("|") and "harness source" not in seen["own_view"]
    assert seen["path"] == "/usr/local/bin:/usr/bin:/bin"
    assert seen["harness_python"] == "none"
    assert seen["cwd"] == str(tmp_path)
    assert seen["leaks"] == ""
    # The first command's background process outlived it; the harness's did not enter the view.
    assert seen["background"] == "1"
    assert seen["harness_after"] == "harness source|run files"


def test_statuses_signals_and_pipes_behave_as_in_bash(tmp_path: Path) -> None:
    harness = (
        "bash -c 'exit 7'; echo \"exit=$?\"\n"
        "bash -c 'kill -TERM $$'; echo \"term=$?\"\n"
        "bash -c 'yes | head -1 >/dev/null; echo \"pipe=${PIPESTATUS[0]}\"'\n"
        "echo in | bash -c 'read value; echo \"stdin=$value\"'\n"
        "bash -c 'echo \"pid1=$(cat /proc/1/comm)\"'\n"
    )
    result, _, _ = _sandbox(tmp_path, harness)
    assert result.returncode == 0, result.stdout + result.stderr
    seen = _lines(result)
    assert seen["exit"] == "7"
    assert seen["term"] == "143"
    assert seen["pipe"] == "141"
    assert seen["stdin"] == "in"
    assert seen["pid1"] != "sleep"  # the task namespace's own init, not the harness's processes


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")
def test_a_tmux_pane_started_with_bash_runs_in_the_task_namespace(tmp_path: Path) -> None:
    harness = (
        "out=$(mktemp)\n"
        "tmux -S /tmp/task.sock new-session -d -x 80 -y 24 'bash --login'\n"
        'tmux -S /tmp/task.sock send-keys "cat @DEPS@/harness.py > $out 2>&1; echo done >> $out" Enter\n'
        "for _ in $(seq 100); do grep -q done $out && break; sleep 0.1; done\n"
        'echo "pane=$(tr "\\n" "|" < $out)"\n'
        "tmux -S /tmp/task.sock kill-server\n"
    )
    result, _, _ = _sandbox(tmp_path, harness)
    assert result.returncode == 0, result.stdout + result.stderr
    pane = _lines(result)["pane"]
    assert "done" in pane and "harness source" not in pane


def test_state_outside_a_hidden_path_is_refused(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(TASK_SHELL),
            "--state",
            str(tmp_path / "state"),
            "--hide",
            str(tmp_path / "x"),
            "--",
            "true",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2 and "hidden path" in result.stderr


def test_environment_keeps_only_entries_outside_hidden_paths() -> None:
    environment = {
        "PATH": "/agent_deps_mount/bin:/usr/bin:/trajectories_mount/.task_shell/bin",
        "HARNESS_HOME": "/agent_deps_mount/home",
        "HOME": "/root",
        "NOTE": "mentions /agent_deps_mountain",
    }
    assert task_shell.clean_environment(environment, ["/agent_deps_mount", "/trajectories_mount"]) == {
        "PATH": "/usr/bin",
        "HOME": "/root",
        "NOTE": "mentions /agent_deps_mountain",
    }
