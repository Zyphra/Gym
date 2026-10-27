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

"""Run the agent's task shell without the harness's own mounts.

Usage: ``python -I task_shell.py [--hide PATH]... -- ARGV...``

The harness runs inside the task sandbox, so the commands the agent types would
otherwise see the harness runtime (``/agent_deps_mount``), the Gym tree
(``/nemo_gym_mount``) and the run's files (``/trajectories_mount``). ARGV runs in
its own mount namespace in which each hidden path is detached (or, when it is a
plain directory, covered by an empty read-only tmpfs), with every environment
entry under a hidden path removed. The harness's own namespace keeps them.

Only the standard library is used: this runs under the harness interpreter,
which is loaded before its mount is detached.
"""

import ctypes
import errno
import os
import sys


HARNESS_MOUNTS = ("/agent_deps_mount", "/nemo_gym_mount", "/trajectories_mount")
_MS_RDONLY = 0x1
_MS_NOSUID = 0x2
_MS_NODEV = 0x4
_MS_NOEXEC = 0x8
_MS_REC = 0x4000
_MS_PRIVATE = 0x40000
_MNT_DETACH = 0x2
SETUP_FAILURE_EXIT = 125


def _unshare_mounts() -> None:
    """Enter a new mount namespace, in a nested user namespace when not privileged."""
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


def _hide(libc: ctypes.CDLL, path: str) -> None:
    target = path.encode()
    if libc.umount2(target, _MNT_DETACH) == 0:
        return
    code = ctypes.get_errno()
    if code not in (errno.EINVAL, errno.ENOENT):
        raise OSError(code, f"detach {path}: {os.strerror(code)}")
    if not os.path.isdir(path):
        return  # absent: nothing to hide
    # A plain directory: cover it, so no task command reads its contents.
    flags = _MS_RDONLY | _MS_NOSUID | _MS_NODEV | _MS_NOEXEC
    if libc.mount(b"none", target, b"tmpfs", flags, b"size=4k,mode=000") != 0:
        code = ctypes.get_errno()
        raise OSError(code, f"cover {path}: {os.strerror(code)}")


def _under(value: str, roots: list[str]) -> bool:
    return any(value == root or value.startswith(root.rstrip("/") + "/") for root in roots)


def clean_environment(environment: dict[str, str], roots: list[str]) -> dict[str, str]:
    """``environment`` without entries under a hidden root: path lists keep their other entries."""
    cleaned = {}
    for name, value in environment.items():
        if not any(root in value for root in roots):
            cleaned[name] = value
        elif ":" in value:
            kept = [entry for entry in value.split(":") if not _under(entry, roots)]
            if kept:
                cleaned[name] = ":".join(kept)
        elif not _under(value, roots):
            cleaned[name] = value
    return cleaned


def main(argv: list[str]) -> int:
    roots: list[str] = []
    while argv and argv[0] == "--hide" and len(argv) > 1:
        roots.append(os.path.normpath(argv[1]))
        argv = argv[2:]
    if not argv or argv[0] != "--" or len(argv) < 2:
        print("usage: task_shell.py [--hide PATH]... -- ARGV...", file=sys.stderr)
        return 2
    argv = argv[1:]
    roots = roots or list(HARNESS_MOUNTS)
    environment = clean_environment(dict(os.environ), roots)
    inside_hidden = _under(os.getcwd(), roots)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        _unshare_mounts()
        if libc.mount(b"none", b"/", None, _MS_REC | _MS_PRIVATE, None) != 0:
            code = ctypes.get_errno()
            raise OSError(code, f"private /: {os.strerror(code)}")
        for root in roots:
            _hide(libc, root)
    except OSError as error:
        print(f"task shell: {error}", file=sys.stderr)
        return SETUP_FAILURE_EXIT
    if inside_hidden:
        os.chdir("/")
    try:
        os.execvpe(argv[0], argv, environment)
    except OSError as error:
        print(f"task shell: {argv[0]}: {error}", file=sys.stderr)
        return 127 if error.errno == errno.ENOENT else 126


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
