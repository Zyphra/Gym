# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider CLI commands never inherit the worker's stdin.

A command run without explicit input must read EOF, as a non-interactive
``docker exec`` (no ``-i``) does, instead of blocking forever on the worker's
own fd 0 (e.g. a dpkg conffile prompt). Explicit input bytes still arrive on a
pipe, and the detached start paths keep their existing ``/dev/null`` stdin.
"""

import asyncio
import contextlib
import os
import shutil
import time
from collections.abc import Iterator

import pytest

from nemo_gym.sandbox.providers.apptainer import provider as apptainer_provider
from nemo_gym.sandbox.providers.docker import provider as docker_provider
from nemo_gym.sandbox.providers.enroot import provider as enroot_provider


pytestmark = pytest.mark.skipif(shutil.which("sh") is None, reason="sh not available")

# Prints the read status: 1 at EOF, 0 when a line arrived.
READ_STDIN = ["sh", "-c", 'read line; echo "read=$? line=$line"']


@pytest.fixture
def held_open_stdin() -> Iterator[None]:
    """Make this process's fd 0 a pipe whose writer stays open (never EOF)."""
    read_end, write_end = os.pipe()
    saved = os.dup(0)
    os.dup2(read_end, 0)
    os.close(read_end)
    try:
        yield
    finally:
        os.dup2(saved, 0)
        os.close(saved)
        os.close(write_end)


def _providers(monkeypatch: pytest.MonkeyPatch, tmp_path) -> dict[str, object]:
    monkeypatch.setattr(apptainer_provider, "_require_apptainer", lambda _bin_path=None: "/bin/true")
    monkeypatch.setattr(enroot_provider, "_require_enroot", lambda: "/bin/true")
    monkeypatch.setattr(docker_provider, "_require_docker", lambda: "/bin/true")
    monkeypatch.delenv("NEMO_GYM_EPISODE_CGROUP_ROOT", raising=False)
    return {
        "apptainer": apptainer_provider.ApptainerProvider(),
        "enroot": enroot_provider.EnrootProvider(create={"base_dir": str(tmp_path / "enroot")}),
        "docker": docker_provider.DockerProvider(),
    }


@pytest.mark.parametrize("kind", ["apptainer", "enroot", "docker"])
async def test_command_without_input_reads_eof_immediately(kind, held_open_stdin, monkeypatch, tmp_path):
    provider = _providers(monkeypatch, tmp_path)[kind]
    started = time.monotonic()
    code, out, _err = await provider._run(READ_STDIN, timeout_s=10)
    assert time.monotonic() - started < 5, "the command waited on the worker's stdin"
    assert (code, out.strip()) == (0, "read=1 line=")


@pytest.mark.parametrize("kind", ["apptainer", "enroot", "docker"])
async def test_explicit_input_bytes_arrive_intact(kind, held_open_stdin, monkeypatch, tmp_path):
    provider = _providers(monkeypatch, tmp_path)[kind]
    payload = bytes(range(256)) * 64
    code, out, _err = await provider._run(["sh", "-c", "od -An -v -tx1 | tr -d ' \\n'"], timeout_s=10, stdin=payload)
    assert code == 0
    assert out == payload.hex()
    code, out, _err = await provider._run(READ_STDIN, timeout_s=10, stdin=b"answer\n")
    assert (code, out.strip()) == (0, "read=0 line=answer")


async def test_detached_paths_keep_dev_null_stdin(held_open_stdin, monkeypatch, tmp_path):
    providers = _providers(monkeypatch, tmp_path)
    code, out, _err = await providers["apptainer"]._run_daemonizing(READ_STDIN, timeout_s=10)
    assert (code, out.strip()) == (0, "read=1 line=")
    proc, out_f, err_f = await providers["enroot"]._start_detached(READ_STDIN)
    with contextlib.closing(out_f), contextlib.closing(err_f):
        assert await asyncio.wait_for(proc.wait(), 5) == 0
        out_f.seek(0)
        assert out_f.read().decode().strip() == "read=1 line="
