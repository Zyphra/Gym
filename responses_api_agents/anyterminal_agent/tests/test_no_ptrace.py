# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An agent process cannot open a verifier process's /proc entries; its own stay open."""

import shutil
import subprocess
import sys
import textwrap

import pytest

from responses_api_agents.anyterminal_agent import app


def _user_namespaces_available() -> bool:
    unshare = shutil.which("unshare")
    return unshare is not None and subprocess.run([unshare, "-Ur", "true"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _user_namespaces_available(), reason="needs unprivileged user namespaces")
def test_an_agent_command_cannot_reach_the_verifier_but_reaches_its_own_processes(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    agent = app._without_ptrace(
        textwrap.dedent(
            """
            v=$(cat "$DIR/verifier.pid")
            ls "/proc/$v/root/" >/dev/null 2>&1 && echo "reached root"
            cat "/proc/$v/environ" >/dev/null 2>&1 && echo "reached environ"
            for entry in fd/1 ns/mnt cwd exe; do
              readlink "/proc/$v/$entry" >/dev/null 2>&1 && echo "reached $entry"
            done
            (printf 1 > "/proc/$v/root$PRIVATE/reward.txt") 2>/dev/null && echo "wrote reward"
            sleep 5 & child=$!
            cat "/proc/$child/environ" >/dev/null && ls "/proc/$child/root/" >/dev/null && echo "own child ok"
            kill "$child"
            """
        )
    ).replace(app.PRIVATE_VERIFIER_PYTHON, sys.executable)
    # Namespace root, like the sandbox: the verifier keeps every capability, in its own mount
    # namespace with a private directory and a secret in its environment.
    script = textwrap.dedent(
        f"""
        TOKEN=secret unshare -m sh -c 'mount -t tmpfs t "$PRIVATE" && echo 0 > "$PRIVATE/reward.txt" &&
          echo $$ > "$DIR/verifier.pid" && exec sleep 3' &
        while [ ! -s "$DIR/verifier.pid" ]; do sleep 0.05; done; sleep 0.1
        {agent}
        echo "reward=$(cat "/proc/$(cat "$DIR/verifier.pid")/root$PRIVATE/reward.txt")"
        cat "/proc/$(cat "$DIR/verifier.pid")/environ" >/dev/null && echo "full caps reach it"
        kill %1
        """
    )
    result = subprocess.run(
        ["unshare", "-Ur", "sh", "-c", script],
        env={"DIR": str(tmp_path), "PRIVATE": str(private), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "reached" not in result.stdout and "wrote" not in result.stdout, result.stdout
    assert "own child ok" in result.stdout, result.stderr
    # The control: a process with every capability (the verifier's peers) still reaches it.
    assert "reward=0" in result.stdout and "full caps reach it" in result.stdout, result.stdout
