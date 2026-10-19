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

"""Start the agent's terminal in its own PID and mount namespace.

Usage: ``python -I command_namespace.py ARGV...``

The harness runs inside the task sandbox, beside the commands it types. Harbor's
own environments keep the harness outside the task container; this keeps that
boundary inside one sandbox:

- ARGV (the tmux session start) runs as the first child of a new PID namespace
  whose init is this process, with a private ``/proc``. A ``pkill`` or
  ``kill -9 -1`` typed by the agent reaches only processes in that namespace,
  never the harness, and the namespace init ignores signals sent from inside it.
- Everything ARGV starts carries ``oom_score_adj`` 1000, so a sandbox memory
  limit kills the task's command that exhausted it, not the harness.

The caller's exit status is ARGV's. The namespace init then stays as the reaper
of the agent's processes until the last one exits; tmux daemonizes, so the
session and anything the agent left running keep living after ARGV returns.

Only the standard library is used: this runs under the admitted harness
interpreter inside arbitrary task images.
"""

import ctypes
import errno
import os
import signal
import sys


_MS_NOSUID = 0x2
_MS_NODEV = 0x4
_MS_NOEXEC = 0x8
_MS_REC = 0x4000
_MS_PRIVATE = 0x40000
_COMMAND_OOM_SCORE_ADJ = "1000"
SETUP_FAILURE_EXIT = 125


def _mount(libc: ctypes.CDLL, source: bytes, target: bytes, fstype: bytes | None, flags: int) -> None:
    if libc.mount(source, target, fstype, flags, None) != 0:
        code = ctypes.get_errno()
        raise OSError(code, f"mount {target.decode()}: {os.strerror(code)}")


def _unshare() -> None:
    """Enter new PID (for children) and mount namespaces.

    A fakeroot sandbox already holds CAP_SYS_ADMIN in its own user namespace and
    takes the first path. An unprivileged caller gets the same isolation inside a
    nested user namespace that maps only its own user and group; the kernel then
    requires dropping supplementary groups for the agent's commands.
    """
    try:
        os.unshare(os.CLONE_NEWPID | os.CLONE_NEWNS)
        return
    except PermissionError:
        pass
    uid, gid = os.geteuid(), os.getegid()
    os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWPID | os.CLONE_NEWNS)
    with open("/proc/self/setgroups", "w") as handle:
        handle.write("deny")
    with open("/proc/self/uid_map", "w") as handle:
        handle.write(f"{uid} {uid} 1")
    with open("/proc/self/gid_map", "w") as handle:
        handle.write(f"{gid} {gid} 1")


def _report(fd: int, message: str) -> None:
    try:
        os.write(fd, message.encode("utf-8", errors="replace"))
    finally:
        os.close(fd)


def _run_init(argv: list[str], status: int) -> None:
    """PID 1 of the new namespace: start ARGV, report its status, reap the rest."""
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        # Keep the private /proc out of the sandbox's other mount namespaces.
        _mount(libc, b"none", b"/", None, _MS_REC | _MS_PRIVATE)
        _mount(libc, b"proc", b"/proc", b"proc", _MS_NOSUID | _MS_NODEV | _MS_NOEXEC)
    except OSError as error:
        _report(status, f"E {error}\n")
        os._exit(SETUP_FAILURE_EXIT)
    # A namespace init receives a signal from inside only when it installed a
    # handler. Python installs one for SIGINT; restore the default for all.
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT, signal.SIGUSR1, signal.SIGUSR2):
        signal.signal(number, signal.SIG_DFL)
    command = os.fork()
    if command == 0:
        os.close(status)
        try:
            with open("/proc/self/oom_score_adj", "w") as handle:
                handle.write(_COMMAND_OOM_SCORE_ADJ)
            os.execvp(argv[0], argv)
        except OSError as error:
            os.write(2, f"terminus command namespace: {argv[0]}: {error}\n".encode(errors="replace"))
            os._exit(127 if error.errno == errno.ENOENT else 126)
    # Release the caller's output pipes: only ARGV writes to them.
    null = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(null, fd)
    os.close(null)
    _, wait_status = os.waitpid(command, 0)
    code = os.waitstatus_to_exitcode(wait_status)
    _report(status, f"{code if code >= 0 else 128 - code}\n")
    while True:
        try:
            os.wait()
        except ChildProcessError:
            os._exit(0)
        except InterruptedError:
            continue


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: command_namespace.py ARGV...", file=sys.stderr)
        return 2
    read_end, write_end = os.pipe()
    sys.stdout.flush()
    sys.stderr.flush()
    intermediate = os.fork()
    if intermediate == 0:
        os.close(read_end)
        try:
            _unshare()
            init = os.fork()
        except OSError as error:
            _report(write_end, f"E namespace setup: {error}\n")
            os._exit(SETUP_FAILURE_EXIT)
        if init == 0:
            _run_init(argv, write_end)
        os._exit(0)
    os.close(write_end)
    os.waitpid(intermediate, 0)
    chunks = []
    while chunk := os.read(read_end, 4096):
        chunks.append(chunk)
    os.close(read_end)
    reply = b"".join(chunks).decode("utf-8", errors="replace").strip()
    if reply.startswith("E "):
        print(f"terminus command namespace: {reply[2:]}", file=sys.stderr)
        return SETUP_FAILURE_EXIT
    try:
        return int(reply)
    except ValueError:
        print("terminus command namespace: init exited before reporting a status", file=sys.stderr)
        return SETUP_FAILURE_EXIT


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
