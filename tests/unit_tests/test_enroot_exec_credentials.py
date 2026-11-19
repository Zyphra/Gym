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
"""Sandbox credential changes are real: own mapped user namespace, or the provider's as namespace root."""

import asyncio
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from nemo_gym.sandbox.providers.base import SandboxSpec
from nemo_gym.sandbox.providers.enroot import provider as enroot_provider


pytestmark = pytest.mark.sandbox


def test_own_process_shares_every_namespace() -> None:
    assert enroot_provider._namespaces_not_shared(os.getpid()) == ()


def test_unreadable_init_keeps_enroot_exec(tmp_path: Path) -> None:
    exited = subprocess.Popen(["true"])
    exited.wait()
    assert enroot_provider._namespaces_not_shared(exited.pid) is None


def test_separate_user_namespace_keeps_enroot_exec() -> None:
    argv = enroot_provider._exec_argv("/opt/enroot/bin/enroot", 42, ("user", "mnt"), ["-e", "A"], ["sh", "-c", "true"])
    assert argv == ["/opt/enroot/bin/enroot", "exec", "-e", "A", "42", "sh", "-c", "true"]
    assert enroot_provider.is_exec_argv("/opt/enroot/bin/enroot", argv)


def test_shared_user_namespace_refuses_unknown_exec_options() -> None:
    with pytest.raises(ValueError, match="without Enroot's emulation"):
        enroot_provider._exec_argv("/opt/enroot/bin/enroot", 42, ("mnt",), ["--cleanenv"], ["sh"])


@pytest.mark.skipif(shutil.which("enroot-nsenter") is None, reason="requires Enroot's enroot-nsenter")
def test_shared_user_namespace_exec_passes_environment_and_workdir(tmp_path: Path) -> None:
    """The no-emulation exec gives the target's environment plus named caller variables and its cwd."""
    target = subprocess.Popen(["sleep", "60"], cwd=tmp_path, env={"PATH": os.environ["PATH"], "FROM_INIT": "init"})
    try:
        binary = str(Path(shutil.which("enroot-nsenter")).parent / "enroot")
        argv = enroot_provider._exec_argv(
            binary, target.pid, (), ["-e", "FROM_CALL"], ["sh", "-c", 'echo "$FROM_INIT $FROM_CALL $(pwd)"']
        )
        assert enroot_provider.is_exec_argv(binary, argv)
        result = subprocess.run(
            argv, env={**os.environ, "FROM_CALL": "call"}, capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == f"init call {tmp_path}"
    finally:
        target.kill()
        target.wait()


def test_mapped_user_namespace_exec_joins_it_first() -> None:
    """A sandbox in its own mapped user namespace is joined with nsenter, then entered without emulation."""
    argv = enroot_provider._exec_argv(
        "/opt/enroot/bin/enroot", 42, ("user", "pid", "mnt"), ["-e", "A"], ["sh", "-c", "true"], "/usr/bin/nsenter"
    )
    assert argv[:5] == ["/usr/bin/nsenter", "--target", "42", "--user", "--"]
    assert argv[5:9] == ["/bin/bash", "-c", enroot_provider._SHARED_USER_NAMESPACE_EXEC, "enroot-exec"]
    assert argv[9:] == [
        "/opt/enroot/bin/enroot-nsenter",
        "42",
        "A",
        "--",
        "--pid",
        "--mount",
        "--",
        "sh",
        "-c",
        "true",
    ]
    assert enroot_provider.is_exec_argv("/opt/enroot/bin/enroot", argv)


def test_rootfs_commands_are_found_behind_launcher_prefixes() -> None:
    binary = "/opt/enroot/bin/enroot"
    assert enroot_provider._manages_rootfs(binary, [binary, "create", "-n", "x", "img.sqsh"])
    assert enroot_provider._manages_rootfs(binary, ["/usr/bin/python3", "-c", "shim", "payload", binary, "start", "x"])
    assert enroot_provider._manages_rootfs(binary, [binary, "remove", "-f", "x"])
    assert not enroot_provider._manages_rootfs(binary, [binary, "exec", "42", "sh"])
    assert not enroot_provider._manages_rootfs(binary, [binary, "list", "-f"])


def test_subordinate_range_needs_a_full_range_for_the_caller(tmp_path: Path) -> None:
    subuid = tmp_path / "subuid"
    subuid.write_text("other:100000:65536\nrob:200000:1000\nrob:300000:65536\n2013:400000:65536\n")
    assert enroot_provider._subordinate_range(str(subuid), {"rob"}) == (300000, 65536)
    assert enroot_provider._subordinate_range(str(subuid), {"2013"}) == (400000, 65536)
    assert enroot_provider._subordinate_range(str(subuid), {"nobody"}) is None
    assert enroot_provider._subordinate_range(str(tmp_path / "absent"), {"rob"}) is None


def _full_map_root() -> bool:
    if os.geteuid() != 0:
        return False
    ranges = [line.split() for line in Path("/proc/self/uid_map").read_text().splitlines()]
    return sum(int(count) for _, _, count in ranges) >= 65536


@pytest.mark.skipif(
    not os.environ.get("GYM_ENROOT_CREDENTIALS_IMAGE")
    or shutil.which("enroot") is None
    or (enroot_provider._subordinate_ids() is None and not _full_map_root()),
    reason="requires Enroot, GYM_ENROOT_CREDENTIALS_IMAGE (.sqsh), and a subordinate ID range or namespace root",
)
def test_sandbox_changes_ownership_and_drops_privileges(tmp_path: Path) -> None:
    """A real sandbox really changes owners, uid, gid and groups, as under Docker."""
    config = {
        "create": {
            "remap_root": True,
            "rw": True,
            # Enroot resolves --rc /dev/null before /dev is mounted in a nested start.
            "bypass_entrypoint": False,
            # Enroot unpacks the image here, which needs a filesystem with extended attributes.
            "base_dir": str(Path(os.environ.get("GYM_ENROOT_CREDENTIALS_WORKDIR", tmp_path)) / tmp_path.name),
            "extra_start_args": ["--net", "--pid", "--ipc"],
        },
        "exec": {"default_timeout_s": 120},
    }

    async def run() -> list[str]:
        provider = enroot_provider.EnrootProvider(**config)
        handle = await provider.create(SandboxSpec(image=os.environ["GYM_ENROOT_CREDENTIALS_IMAGE"]))
        try:
            # Unprivileged, each sandbox has its own mapped user namespace; as namespace root, the provider's.
            assert ("user" in handle.raw.entered_namespaces) == (provider._subordinate_ids is not None)
            outputs = []
            for command in (
                "grep Seccomp: /proc/self/status | tr -d '\\t'",
                "cat /proc/self/setgroups",
                'f=$(mktemp) && chown 100:101 "$f" && stat -c %u:%g "$f"',
                "setpriv --reuid=100 --regid=101 --clear-groups sh -c 'echo $(id -u) $(id -g) $(id -G)'",
                # A file a task account owns in the shared folder is still removed at close.
                "mkdir -p /sandbox/owned/sub && touch /sandbox/owned/sub/f && chown -R 100:101 /sandbox/owned && echo ok",
            ):
                result = await provider.exec(handle, command)
                assert result.return_code == 0, result
                outputs.append(result.stdout.strip())
            return outputs
        finally:
            await provider.close(handle)
            assert not handle.raw.staging_dir.exists()

    assert asyncio.run(run()) == ["Seccomp:0", "allow", "100:101", "100 101 101", "ok"]
