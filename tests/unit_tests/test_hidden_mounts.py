# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real mount-namespace checks: the agent's shell sees the harness mounts empty, the harness keeps them."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nemo_gym import hidden_mounts


def _harness_tree(tmp_path: Path) -> Path:
    mount = tmp_path / "trajectories_mount"
    (mount / "staging").mkdir(parents=True)
    (mount / "params.json").write_text("{}")
    (tmp_path / "nemo-gym-abc").mkdir()
    (tmp_path / "nemo-gym-abc" / "grader.py").write_text("x")
    (tmp_path / "task").mkdir()
    (tmp_path / "task" / "keep.txt").write_text("task file")
    return mount


def _require_mount_namespace() -> None:
    probe = subprocess.run(
        [sys.executable, "-I", hidden_mounts.__file__, "--", "true"], capture_output=True, text=True, timeout=30
    )
    if probe.returncode != 0:
        pytest.skip(f"no mount namespace here: {probe.stderr.strip()}")


def test_shell_sees_hidden_paths_empty_and_the_rest_intact(tmp_path: Path) -> None:
    _require_mount_namespace()
    mount = _harness_tree(tmp_path)
    paths = [str(mount), str(tmp_path / "nemo-gym-*"), str(tmp_path / "absent")]
    script = f"ls -A {mount}; echo ---; ls -A {tmp_path}/nemo-gym-abc; echo ---; cat {tmp_path}/task/keep.txt; touch {mount}/x"
    result = subprocess.run(
        hidden_mounts.command(["bash", "-c", script], paths), capture_output=True, text=True, timeout=30
    )
    # Both hidden directories list empty; the task file outside them reads normally.
    assert [part.strip() for part in result.stdout.split("---")] == ["", "", "task file"]
    assert "Read-only file system" in result.stderr
    # The caller (the harness) still sees every file.
    assert sorted(p.name for p in mount.iterdir()) == ["params.json", "staging"]
    assert (tmp_path / "nemo-gym-abc" / "grader.py").read_text() == "x"


def test_exit_status_and_pid_are_the_commands(tmp_path: Path) -> None:
    _require_mount_namespace()
    proc = subprocess.Popen(
        hidden_mounts.command(["bash", "-c", "echo $$; exit 7"], [str(tmp_path)]), stdout=subprocess.PIPE, text=True
    )
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 7
    assert int(out.strip()) == proc.pid  # exec, not a child: the caller's kill/timeout reaches the command


def test_signal_to_the_process_group_reaches_the_command(tmp_path: Path) -> None:
    _require_mount_namespace()
    proc = subprocess.Popen(hidden_mounts.command(["sleep", "60"], [str(tmp_path)]), start_new_session=True)
    time.sleep(1)
    os.killpg(proc.pid, signal.SIGTERM)
    assert proc.wait(timeout=10) == -signal.SIGTERM


def test_shell_wrapper_runs_any_arguments_hidden(tmp_path: Path) -> None:
    _require_mount_namespace()
    mount = _harness_tree(tmp_path)
    wrapper = hidden_mounts.shell_wrapper(str(tmp_path), "bash", [str(mount)])
    result = subprocess.run([wrapper, "-lc", f"ls -A {mount} | wc -l"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and result.stdout.strip() == "0"


def test_shell_under_a_hidden_path_is_refused(tmp_path: Path) -> None:
    deps = tmp_path / "agent_deps_mount" / "bin"
    deps.mkdir(parents=True)
    shell = deps / "bash"
    shell.write_text("#!/bin/sh\n")
    shell.chmod(0o755)
    with pytest.raises(ValueError, match="hidden harness path"):
        hidden_mounts.shell_wrapper(str(tmp_path), str(shell), [str(tmp_path / "agent_deps_mount")])


def test_default_paths_cover_every_runner_mount() -> None:
    assert {
        "/trajectories_mount",
        "/nemo_gym_mount",
        "/agent_deps_mount",
        "/gym_verifier_helper",
        "/tmp/nemo-gym-*",
    } == set(hidden_mounts.HARNESS_MOUNTS)
