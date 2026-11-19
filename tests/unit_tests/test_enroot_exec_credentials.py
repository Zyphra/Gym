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


BINARY = "/opt/enroot/bin/enroot"


def test_shared_user_namespace_keeps_enroot_exec() -> None:
    """An init in the provider's own user namespace is never entered without Enroot's namespace handling."""
    for nsenter in (None, "/usr/bin/nsenter"):
        argv = enroot_provider._exec_argv(BINARY, 42, ("pid", "mnt"), ["-e", "A"], ["sh"], nsenter)
        assert argv == [BINARY, "exec", "-e", "A", "42", "sh"]


def test_mapped_exec_refuses_unknown_exec_options() -> None:
    with pytest.raises(ValueError, match="without Enroot's emulation"):
        enroot_provider._exec_argv(BINARY, 42, ("user", "mnt"), ["--cleanenv"], ["sh"], "/usr/bin/nsenter")


@pytest.mark.parametrize("name", ["X[$(touch injected)]", "--", "A-B", "1A", "A B", ""])
def test_mapped_exec_rejects_environment_names_enroot_rejects(name: str) -> None:
    """A name is expanded by the worker-side exec script, so only identifiers pass (runtime.sh's rule)."""
    with pytest.raises(ValueError, match="invalid environment variable names"):
        enroot_provider._exec_argv(BINARY, 42, ("user", "mnt"), ["-e", name], ["sh"], "/usr/bin/nsenter")


def test_mapped_user_namespace_exec_joins_it_first() -> None:
    """A sandbox in its own mapped user namespace is joined with nsenter (keeping groups), then entered without emulation."""
    argv = enroot_provider._exec_argv(
        BINARY, 42, ("user", "pid", "mnt"), ["-e", "A"], ["sh", "-c", "true"], "/usr/bin/nsenter"
    )
    assert argv[:6] == ["/usr/bin/nsenter", "--target", "42", "--user", "--preserve-credentials", "--"]
    assert argv[6:10] == ["/bin/bash", "-c", enroot_provider._SHARED_USER_NAMESPACE_EXEC, "enroot-exec"]
    assert argv[10:] == [
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
    assert enroot_provider.is_exec_argv(BINARY, argv)


def test_rootfs_commands_are_found_behind_launcher_prefixes() -> None:
    assert enroot_provider._manages_rootfs(BINARY, [BINARY, "create", "-n", "x", "img.sqsh"])
    assert enroot_provider._manages_rootfs(BINARY, ["/usr/bin/python3", "-c", "shim", "payload", BINARY, "start", "x"])
    assert enroot_provider._manages_rootfs(BINARY, [BINARY, "remove", "-f", "x"])
    assert not enroot_provider._manages_rootfs(BINARY, [BINARY, "exec", "42", "sh"])
    assert not enroot_provider._manages_rootfs(BINARY, [BINARY, "list", "-f"])


def test_subordinate_range_needs_a_full_range_for_the_caller(tmp_path: Path) -> None:
    subuid = tmp_path / "subuid"
    subuid.write_text("other:100000:65536\nrob:200000:1000\nrob:300000:65536\n2013:400000:65536\n")
    assert enroot_provider._subordinate_range(str(subuid), {"rob"}) == (300000, 65536)
    assert enroot_provider._subordinate_range(str(subuid), {"2013"}) == (400000, 65536)
    assert enroot_provider._subordinate_range(str(subuid), {"nobody"}) is None
    assert enroot_provider._subordinate_range(str(tmp_path / "absent"), {"rob"}) is None


IDS = enroot_provider._SubordinateIds(
    uid=(100000, 65536), gid=(100000, 65536), unshare="u", newuidmap="nu", newgidmap="ng", nsenter="ns"
)


def test_credentials_mode_is_explicit_and_real_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="auto, real or emulated"):
        enroot_provider._select_credentials("sometimes")
    assert enroot_provider._select_credentials("emulated")[:2] == (None, "emulated")

    monkeypatch.setattr(enroot_provider, "_subordinate_ids", lambda: None)
    with pytest.raises(enroot_provider.EnrootCreateError, match="not possible here"):
        enroot_provider._select_credentials("real")
    ids, mode, reason = enroot_provider._select_credentials("auto")
    assert (ids, mode) == (None, "emulated") and reason

    monkeypatch.setattr(enroot_provider, "_subordinate_ids", lambda: IDS)
    monkeypatch.setattr(enroot_provider, "_own_seccomp", lambda: 2)
    with pytest.raises(enroot_provider.EnrootCreateError, match="seccomp filter"):
        enroot_provider._select_credentials("real")
    assert enroot_provider._select_credentials("auto")[1] == "emulated"

    monkeypatch.setattr(enroot_provider, "_own_seccomp", lambda: 0)
    assert enroot_provider._select_credentials("real")[:2] == (IDS, "real")


@pytest.mark.skipif(
    not os.environ.get("GYM_ENROOT_CREDENTIALS_IMAGE")
    or shutil.which("enroot") is None
    or enroot_provider._subordinate_ids() is None,
    reason="requires Enroot, GYM_ENROOT_CREDENTIALS_IMAGE (.sqsh) with setpriv, and a subordinate ID range",
)
def test_real_credentials_and_containment(tmp_path: Path) -> None:
    """Real credentials as under Docker, and nothing beyond the caller's own ids and binds."""
    outside = tmp_path / "outside-secret"
    outside.write_text("caller only\n")
    outside.chmod(0o600)
    config = {
        "create": {
            "credentials": "real",
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
    ids = enroot_provider._subordinate_ids()

    async def run() -> dict[str, str]:
        provider = enroot_provider.EnrootProvider(**config)
        assert provider.credentials_mode == "real"
        handle = await provider.create(SandboxSpec(image=os.environ["GYM_ENROOT_CREDENTIALS_IMAGE"]))
        out: dict[str, str] = {}
        try:
            assert "user" in handle.raw.entered_namespaces and handle.raw.credentials == "real"
            for key, command in {
                "seccomp": "grep Seccomp: /proc/self/status | tr -d '\\t'",
                "setgroups": "cat /proc/self/setgroups",
                "uid_map": "tr -s ' ' < /proc/self/uid_map | sed 's/^ //' | tr '\\n' ';'",
                "chown": 'f=$(mktemp) && chown 100:101 "$f" && stat -c %u:%g "$f"',
                "drop": "setpriv --reuid=100 --regid=101 --clear-groups sh -c 'echo $(id -u) $(id -g) $(id -G)'",
                "high": 'f=$(mktemp) && chown 70000 "$f" 2>/dev/null; echo $?',
                "outside": f"cat {outside} 2>/dev/null; echo $?",
                "exec_groups": "grep ^Groups: /proc/self/status | tr -d '\\t'",
                "init_groups": "grep ^Groups: /proc/1/status | tr -d '\\t'",
                "private": "mkdir -p /sandbox/owned/sub && echo data > /sandbox/owned/sub/f && chmod 700 /sandbox/owned "
                "&& chown -R 100:101 /sandbox/owned && echo ok",
            }.items():
                result = await provider.exec(handle, command)
                assert result.return_code == 0, (key, result)
                out[key] = result.stdout.strip()
            # A task account's private file still moves through the staging folder.
            target = tmp_path / "downloaded"
            await provider.download_file(handle, "/sandbox/owned/sub/f", target)
            out["download"] = target.read_text().strip()
            src = tmp_path / "up"
            src.write_text("up\n")
            await provider.upload_file(handle, src, "/sandbox/owned/sub/up")
            out["upload"] = (await provider.exec(handle, "cat /sandbox/owned/sub/up")).stdout.strip()
        finally:
            await provider.close(handle)
            assert not handle.raw.staging_dir.exists()
        return out

    out = asyncio.run(run())
    assert out["seccomp"] == "Seccomp:0" and out["setgroups"] == "allow"
    # Exactly the caller (as 0) and its own range: host root and every other id are unmapped.
    assert out["uid_map"] == f"0 {os.getuid()} 1;1 {ids.uid[0]} {ids.uid[1]};"
    assert out["chown"] == "100:101" and out["drop"] == "100 101 101"
    assert out["high"] != "0"  # an id outside the range is refused, never faked
    assert out["outside"] == "1"  # caller files outside the binds are not reachable
    assert out["exec_groups"] == out["init_groups"]
    assert out["download"] == "data" and out["upload"] == "up"
