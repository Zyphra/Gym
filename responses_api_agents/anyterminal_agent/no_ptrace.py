# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run an agent command without ``CAP_SYS_PTRACE``: ``python -I -S no_ptrace.py PROGRAM [ARG...]``.

The verifier runs in the agent's sandbox, as the same (namespace) root user. Linux lets a
process open another's ``/proc/<pid>/root``, ``environ``, ``fd`` and ``ns`` entries only if
it holds every capability the other holds, or ``CAP_SYS_PTRACE``. Dropping that one
capability from an agent process's bounding, inheritable and ambient sets, before it
executes anything, therefore keeps it and every process it starts out of the verifier's
private mount namespace and environment, which keep the full set. Agent processes still
reach each other (they hold the same set), so ``gdb -p``/``strace -p`` on its own processes,
PID files, ``apt`` and services work as before. Docker's default capability set omits
``CAP_SYS_PTRACE`` as well. Standard library only; runs from the read-only agent runtime.
"""

import ctypes
import os
import sys


CAP_SYS_PTRACE = 19
PR_CAPBSET_DROP, PR_CAP_AMBIENT, PR_CAP_AMBIENT_LOWER = 24, 47, 3
LINUX_CAPABILITY_VERSION_3 = 0x20080522
FAILED = 125


class _Header(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class _Data(ctypes.Structure):
    _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32), ("inheritable", ctypes.c_uint32)]


def drop_ptrace() -> None:
    """Remove ``CAP_SYS_PTRACE`` from this process and anything it will execute."""
    libc = ctypes.CDLL(None, use_errno=True)

    def check(result: int, call: str) -> None:
        if result != 0:
            error = ctypes.get_errno()
            raise OSError(error, f"{call}: {os.strerror(error)}")

    check(libc.prctl(PR_CAPBSET_DROP, CAP_SYS_PTRACE, 0, 0, 0), "PR_CAPBSET_DROP")
    libc.prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_LOWER, CAP_SYS_PTRACE, 0, 0)  # EINVAL without ambient support
    header, data = _Header(LINUX_CAPABILITY_VERSION_3, 0), (_Data * 2)()
    check(libc.capget(ctypes.byref(header), data), "capget")
    keep = ~(1 << CAP_SYS_PTRACE) & 0xFFFFFFFF
    data[0].effective &= keep
    data[0].permitted &= keep
    data[0].inheritable &= keep
    check(libc.capset(ctypes.byref(header), data), "capset")


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: no_ptrace.py PROGRAM [ARG...]", file=sys.stderr)
        return FAILED
    try:
        drop_ptrace()
    except OSError as error:
        # Never run an agent command that could reach the verifier.
        print(f"no_ptrace: {error}", file=sys.stderr)
        return FAILED
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
