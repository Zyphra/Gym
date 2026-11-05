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

"""Run a shell whose view of the sandbox lacks the harness mounts.

Usage: ``python -I hidden_mounts.py [--hide PATH]... -- ARGV...``

A harness running inside the task sandbox (Hermes, OpenHands) needs the runner
directory, Gym and its own runtime; the agent's shell commands do not, and a
benchmark's task container has none of them. This process enters a new mount
namespace, covers each ``--hide`` PATH that is a directory with an empty
read-only tmpfs, then replaces itself with ARGV. The process id, session,
stdio, signals and exit status stay ARGV's, so a caller's own timeout and
process-group handling are unchanged. A trailing ``*`` in PATH is expanded.

This file is standard-library only and is executed by path, so it can run from
an overlay without importing Gym.
"""

import ctypes
import glob
import os
import shlex
import shutil
import sys


# Bind mounts the AnySWE/AnyTerminal runners place in the agent's sandbox, and
# their staging directories. Paths absent from a sandbox are left alone.
HARNESS_MOUNTS = (
    "/trajectories_mount",
    "/nemo_gym_mount",
    "/agent_deps_mount",
    "/gym_verifier_helper",
    "/tmp/nemo-gym-*",
)
SETUP_FAILURE_EXIT = 125

_MS_RDONLY = 0x1
_MS_NOSUID = 0x2
_MS_NODEV = 0x4
_MS_NOEXEC = 0x8
_MS_REC = 0x4000
_MS_PRIVATE = 1 << 18


def _mount(libc, source: bytes, target: bytes, fstype, flags: int, data=None) -> None:
    if libc.mount(source, target, fstype, flags, data) != 0:
        code = ctypes.get_errno()
        raise OSError(code, f"mount {target.decode()}: {os.strerror(code)}")


def _unshare_mounts() -> None:
    """A fakeroot sandbox holds CAP_SYS_ADMIN in its user namespace; otherwise nest one mapping only this user."""
    try:
        os.unshare(os.CLONE_NEWNS)
        return
    except PermissionError:
        pass
    uid, gid = os.geteuid(), os.getegid()
    os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWNS)
    with open("/proc/self/setgroups", "w") as handle:
        handle.write("deny")
    with open("/proc/self/uid_map", "w") as handle:
        handle.write(f"{uid} {uid} 1")
    with open("/proc/self/gid_map", "w") as handle:
        handle.write(f"{gid} {gid} 1")


def expand(paths) -> list[str]:
    """Existing directories named by PATHS; a trailing ``*`` matches by prefix."""
    found = []
    for path in paths:
        for match in sorted(glob.glob(path)) if path.endswith("*") else [path]:
            if os.path.isdir(match) and not os.path.islink(match) and match not in found:
                found.append(match)
    return found


def hide(paths) -> list[str]:
    """Enter a new mount namespace and cover each existing directory in PATHS; returns those covered."""
    covered = expand(paths)
    _unshare_mounts()
    libc = ctypes.CDLL(None, use_errno=True)
    # Keep these mounts out of the sandbox's other mount namespaces.
    _mount(libc, b"none", b"/", None, _MS_REC | _MS_PRIVATE)
    for path in covered:
        _mount(
            libc,
            b"none",
            os.fsencode(path),
            b"tmpfs",
            _MS_RDONLY | _MS_NOSUID | _MS_NODEV | _MS_NOEXEC,
            b"size=4k,mode=0555",
        )
    return covered


def command(argv, paths=None, python: str | None = None) -> list[str]:
    """ARGV run through this file with PATHS (default: HARNESS_MOUNTS) hidden, by the given (default: this) interpreter."""
    paths = HARNESS_MOUNTS if paths is None else paths
    hides = [arg for path in paths for arg in ("--hide", path)]
    return [python or sys.executable, "-I", os.path.abspath(__file__), *hides, "--", *argv]


def shell_wrapper(directory: str, shell: str, paths=None, python: str | None = None) -> str:
    """Write an executable ``bash`` stand-in in DIRECTORY that runs SHELL with PATHS hidden; returns its path.

    A harness that names its shell by path (Hermes' ``_find_bash``) takes this file instead, so every
    command it spawns, with whatever arguments, runs without the harness mounts.
    """
    paths = HARNESS_MOUNTS if paths is None else paths
    resolved = shutil.which(shell) or shell
    if any(resolved == p or resolved.startswith(p.rstrip("*").rstrip("/") + "/") for p in paths):
        # The shell itself would disappear under its own mount; keep the harness working.
        raise ValueError(f"shell {resolved} lives under a hidden harness path")
    path = os.path.join(directory, "bash")
    with open(path, "w") as handle:
        handle.write("#!/bin/sh\nexec " + shlex.join(command([resolved], paths, python)) + ' "$@"\n')
    os.chmod(path, 0o755)
    return path


def main(argv: list[str]) -> int:
    paths = []
    while len(argv) >= 2 and argv[0] == "--hide":
        paths.append(argv[1])
        argv = argv[2:]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        print("usage: hidden_mounts.py [--hide PATH]... -- ARGV...", file=sys.stderr)
        return 2
    try:
        hide(paths)
    except OSError as error:
        print(f"hidden_mounts: {error}", file=sys.stderr)
        return SETUP_FAILURE_EXIT
    try:
        os.execvp(argv[0], argv)
    except OSError as error:
        print(f"hidden_mounts: {argv[0]}: {error}", file=sys.stderr)
        return 127
    return 0  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
