# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A task daemon remains in its instance through producer/verifier handoff."""

import json
import os
import select
import shlex
from pathlib import Path

import pytest

from nemo_gym.sandbox.providers.base import SandboxSpec
from responses_api_agents.anyterminal_agent.app import AnyTerminalInstanceConfig, _build_provider


pytestmark = [pytest.mark.integration, pytest.mark.sandbox, pytest.mark.skipduringci]


async def test_shared_instance_pid_namespace_keeps_daemon_until_close(tmp_path):
    image = os.environ.get("NEMO_GYM_APPTAINER_TEST_IMAGE")
    if not image:
        pytest.skip("set NEMO_GYM_APPTAINER_TEST_IMAGE for the real instance test")
    for name in ("episode", "verifier", "runtime"):
        (tmp_path / name).mkdir()
    cfg = AnyTerminalInstanceConfig.model_construct(
        problem_info={"task_name": "instance-lifecycle", "cpus": 1, "memory_mb": 512},
        persistent_dir=tmp_path / "episode",
        verifier_dir=tmp_path / "verifier",
        nemo_gym_root=tmp_path,
        agent_deps_dir=tmp_path / "runtime",
        sandbox_provider={
            "apptainer": {"create": {"extra_start_args": ["--fakeroot"]}, "exec": {"extra_exec_args": ["--pid"]}}
        },
        apptainer_shared_pid_namespace=True,
    )
    assert cfg.apptainer_shared_pid_namespace is True
    provider = _build_provider(cfg)
    assert "--pid" not in provider._exec_config.extra_exec_args
    handle = await provider.create(SandboxSpec(image=image))
    daemon_fd = None
    try:
        pidfile = shlex.quote(handle.raw.mount_point + "/daemon.pid")
        produced = await provider.exec(
            handle,
            f"set -eu; nohup sleep 120 </dev/null >/dev/null 2>&1 & echo $! > {pidfile}; readlink /proc/self/ns/pid",
        )
        assert produced.return_code == 0, produced.stderr
        checked = await provider.exec(
            handle, f'set -eu; pid=$(cat {pidfile}); kill -0 "$pid"; readlink /proc/self/ns/pid; echo "$pid"'
        )
        assert checked.return_code == 0, checked.stderr
        namespace, inner_pid = checked.stdout.strip().splitlines()
        assert namespace == produced.stdout.strip()
        assert namespace != os.readlink("/proc/self/ns/pid")
        code, listing, error = await provider._run(
            [provider._binary, "instance", "list", "--json", handle.raw.name], timeout_s=10
        )
        assert code == 0, error
        instance_pid = json.loads(listing)["instances"][0]["pid"]
        pending = [int(instance_pid)]
        while pending:
            pid = pending.pop()
            proc = Path(f"/proc/{pid}")
            try:
                status = proc.joinpath("status").read_text()
                nspid = next(line.split()[1:] for line in status.splitlines() if line.startswith("NSpid:"))
                if nspid[-1] == inner_pid and os.readlink(proc / "ns/pid") == namespace:
                    daemon_fd = os.pidfd_open(pid)
                    break
                pending.extend(map(int, proc.joinpath(f"task/{pid}/children").read_text().split()))
            except FileNotFoundError:
                continue
        assert daemon_fd is not None, "instance daemon must be visible in its owned process tree"
        assert not select.select([daemon_fd], [], [], 0)[0]
        verified = await provider.exec(handle, f'set -eu; kill -0 "$(cat {pidfile})"')
        assert verified.return_code == 0, verified.stderr
    finally:
        try:
            await provider.close(handle)
            if daemon_fd is not None:
                assert select.select([daemon_fd], [], [], 10)[0], "close must physically stop the daemon"
        finally:
            if daemon_fd is not None:
                os.close(daemon_fd)
