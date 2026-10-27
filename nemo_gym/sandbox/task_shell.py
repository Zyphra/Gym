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

"""Run the agent's task commands without the harness's own mounts and processes.

Usage: ``python -I task_shell.py --state DIR [--hide PATH]... -- ARGV...``

A harness runs inside the task sandbox, so the commands the agent types would
otherwise see the harness runtime (``/agent_deps_mount``), the Gym tree
(``/nemo_gym_mount``), the run's files (``/trajectories_mount``) and the
harness's processes. Every task command of an episode runs in one shared task
namespace, created by the first command and kept by its init process:

- a mount namespace in which each hidden path is detached (or, when it is a
  plain directory, covered by an empty read-only tmpfs);
- a PID namespace with its own ``/proc``; the sandbox's ``/proc`` is detached,
  so no harness process, and no ``/proc/PID/root`` or ``nsenter`` into the
  harness's mount namespace, is reachable. Background processes outlive the
  command that started them, as in a plain shell;
- ARGV gets the caller's environment without entries under a hidden path,
  its working directory, and no other inherited file descriptor.

``DIR`` keeps the namespace's init identity and must lie under a hidden path,
so no task command can redirect a later one. :func:`activate` writes the
``bash`` that harnesses find on their ``PATH``. A namespace created where a
cover or ``/proc`` layer could be unmounted (a plain hidden directory, or
mounts this sandbox cannot detach) also enters a nested user namespace with
the same IDs, which locks those mounts.

Only the standard library is used: this runs under the harness interpreter
inside arbitrary task images.
"""

import ctypes
import errno
import fcntl
import os
import signal
import sys
import tempfile


HARNESS_MOUNTS = ("/agent_deps_mount", "/nemo_gym_mount", "/trajectories_mount")
_MS_RDONLY = 0x1
_MS_NOSUID = 0x2
_MS_NODEV = 0x4
_MS_NOEXEC = 0x8
_MS_MOVE = 0x2000
_MS_REC = 0x4000
_MS_PRIVATE = 0x40000
_MNT_DETACH = 0x2
_NS_GET_USERNS = 0xB701
SETUP_FAILURE_EXIT = 125
BIN_VARIABLE = "NEMO_GYM_TASK_SHELL_BIN"
_FORWARDED = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT, signal.SIGUSR1, signal.SIGUSR2)


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def _check(result: int, action: str) -> None:
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, f"{action}: {os.strerror(code)}")


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


def _detach(libc: ctypes.CDLL, path: str) -> bool:
    """Detach every mount layer at ``path``; return whether one remains (a locked mount)."""
    while libc.umount2(path.encode(), _MNT_DETACH) == 0:
        pass
    code = ctypes.get_errno()
    if code not in (errno.EINVAL, errno.ENOENT):
        raise OSError(code, f"detach {path}: {os.strerror(code)}")
    return os.path.ismount(path)


def _hide(libc: ctypes.CDLL, path: str) -> bool:
    """Hide ``path``; return whether that took a cover, which a task root could unmount."""
    if not _detach(libc, path) and not os.path.isdir(path):
        return False  # absent: nothing to hide
    if not os.path.ismount(path) and not any(os.scandir(path)):
        return False  # an empty mount point: everything mounted there is gone
    # A plain directory, or a mount this namespace cannot detach: cover it.
    flags = _MS_RDONLY | _MS_NOSUID | _MS_NODEV | _MS_NOEXEC
    _check(libc.mount(b"none", path.encode(), b"tmpfs", flags, b"size=4k,mode=000"), f"cover {path}")
    return True


def _own_maps() -> dict[str, str]:
    """The caller's ID ranges, as identity maps for a nested user namespace."""
    maps = {}
    for name in ("uid_map", "gid_map"):
        with open(f"/proc/self/{name}") as handle:
            ranges = [line.split() for line in handle if line.strip()]
        maps[name] = "".join(f"{inside} {inside} {count}\n" for inside, _, count in ranges)
    return maps


def _write_maps(proc: int, pid: int, maps: dict[str, str]) -> None:
    """Write ``pid``'s maps through the caller's ``/proc``, which the init has replaced in this mount namespace."""
    for name, text in maps.items():
        fd = os.open(f"{pid}/{name}", os.O_WRONLY | os.O_CLOEXEC, dir_fd=proc)
        try:
            os.write(fd, text.encode())
        finally:
            os.close(fd)


def _private_proc(libc: ctypes.CDLL) -> bool:
    """Replace every ``/proc`` layer with one for this PID namespace; return whether one is locked below."""
    staging = tempfile.mkdtemp(prefix=".task-shell-proc-", dir="/tmp")
    try:
        # The kernel admits a new proc only beside a fully visible one: mount
        # first, then detach the sandbox's layers and move this one into place.
        _check(libc.mount(b"proc", staging.encode(), b"proc", _MS_NOSUID | _MS_NODEV | _MS_NOEXEC, None), "mount proc")
        locked = _detach(libc, "/proc")
        _check(libc.mount(staging.encode(), b"/proc", None, _MS_MOVE, None), "move proc")
    finally:
        os.rmdir(staging)
    return locked


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


def _start_ticks(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat") as handle:
            return int(handle.read().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _init(report: int, covered: bool, request: int, granted: int) -> None:
    """PID 1 of the task namespace: finish hiding the harness, then reap until the sandbox ends."""
    try:
        libc = _libc()
        if _private_proc(libc) or covered:
            # A task root could unmount a cover or this /proc and see what is
            # below. Mounts carried into a nested user namespace are locked
            # together, so enter one that keeps every ID.
            os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWNS)
            os.write(request, b"M")
            if os.read(granted, 1) != b"G":
                raise OSError(errno.EPERM, "task namespace ID maps were not written")
        os.close(request)
        os.close(granted)
        os.setsid()
        os.chdir("/")
        null = os.open(os.devnull, os.O_RDWR)
        for fd in (0, 1, 2):
            os.dup2(null, fd)
        os.closerange(3, report)
        os.closerange(report + 1, 65536)
    except OSError as error:
        os.write(report, f"E {error}\n".encode(errors="replace"))
        os._exit(SETUP_FAILURE_EXIT)
    # A namespace init receives only signals it installed a handler for: with
    # the defaults restored, no task command can stop it.
    for number in _FORWARDED:
        signal.signal(number, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGCHLD})
    os.write(report, b"R\n")
    os.close(report)
    while True:
        signal.sigwait({signal.SIGCHLD})
        while True:
            try:
                if os.waitpid(-1, os.WNOHANG)[0] == 0:
                    break
            except ChildProcessError:
                break


def _create(roots: list[str]) -> int:
    """Start the namespace init; return its PID in the caller's namespace."""
    read_end, write_end = os.pipe()
    sys.stdout.flush()
    sys.stderr.flush()
    intermediate = os.fork()
    if intermediate == 0:
        os.close(read_end)
        try:
            _unshare_mounts()
            libc = _libc()
            _check(libc.mount(b"none", b"/", None, _MS_REC | _MS_PRIVATE, None), "private /")
            covered = any([_hide(libc, root) for root in roots])
            maps = _own_maps()
            proc = os.open("/proc", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            os.unshare(os.CLONE_NEWPID)
            request_read, request_write = os.pipe()
            granted_read, granted_write = os.pipe()
            init = os.fork()
            if init == 0:
                os.close(request_read)
                os.close(granted_write)
                _init(write_end, covered, request_write, granted_read)
            os.close(request_write)
            os.close(granted_read)
            if os.read(request_read, 1) == b"M":
                _write_maps(proc, init, maps)
                os.write(granted_write, b"G")
        except OSError as error:
            os.write(write_end, f"E {error}\n".encode(errors="replace"))
            os._exit(SETUP_FAILURE_EXIT)
        os.write(write_end, f"P {init}\n".encode())
        os._exit(0)
    os.close(write_end)
    os.waitpid(intermediate, 0)
    chunks = []
    while chunk := os.read(read_end, 4096):
        chunks.append(chunk)
    os.close(read_end)
    lines = b"".join(chunks).decode(errors="replace").split("\n")
    errors = [line[2:] for line in lines if line.startswith("E ")]
    pids = [int(line[2:]) for line in lines if line.startswith("P ")]
    if errors or len(pids) != 1 or "R" not in lines:
        raise OSError(errno.EIO, f"task namespace setup: {'; '.join(errors) or 'init did not report ready'}")
    return pids[0]


def _holder(state: str, roots: list[str]) -> tuple[int, int]:
    """The episode's namespace init, started by the first command that needs it."""
    os.makedirs(state, mode=0o700, exist_ok=True)
    record = os.path.join(state, "init")
    lock = os.open(os.path.join(state, "lock"), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            with open(record) as handle:
                pid, ticks = (int(field) for field in handle.read().split())
            if _start_ticks(pid) == ticks:
                return pid, ticks
        except (OSError, ValueError):
            pass
        pid = _create(roots)
        ticks = _start_ticks(pid)
        if ticks is None:
            raise OSError(errno.ESRCH, "task namespace init exited")
        temporary = record + ".tmp"
        with open(temporary, "w") as handle:
            handle.write(f"{pid} {ticks}\n")
        os.replace(temporary, record)
        return pid, ticks
    finally:
        os.close(lock)


def _enter(pid: int, ticks: int) -> None:
    """Join the init's namespaces in the order the kernel admits.

    Joining a PID namespace takes CAP_SYS_ADMIN in its owning user namespace
    and in the caller's own, so that owner is joined first (when it is not the
    caller's), then the PID namespace, then the init's user namespace (when the
    namespace was locked) and its mounts.
    """
    handles = {name: os.open(f"/proc/{pid}/ns/{name}", os.O_RDONLY | os.O_CLOEXEC) for name in ("pid", "user", "mnt")}
    handles["owner"] = fcntl.ioctl(handles["pid"], _NS_GET_USERNS)
    try:
        if _start_ticks(pid) != ticks:
            raise OSError(errno.ESRCH, "task namespace init changed")
        if os.fstat(handles["mnt"]).st_ino == os.stat("/proc/self/ns/mnt").st_ino:
            raise OSError(errno.EPERM, "task namespace shares the harness mounts")
        for name, kind in (("owner", os.CLONE_NEWUSER), ("pid", os.CLONE_NEWPID), ("user", os.CLONE_NEWUSER)):
            if kind != os.CLONE_NEWUSER or os.fstat(handles[name]).st_ino != os.stat("/proc/self/ns/user").st_ino:
                os.setns(handles[name], kind)
        os.setns(handles["mnt"], os.CLONE_NEWNS)
    finally:
        for handle in handles.values():
            os.close(handle)


def _run(argv: list[str], environment: dict[str, str], cwd: str) -> int:
    """Fork ARGV into the joined PID namespace and return its status."""
    child = os.fork()
    if child == 0:
        os.closerange(3, 65536)
        # As subprocess does: the interpreter ignores these; ARGV must not.
        for number in (signal.SIGPIPE, signal.SIGXFSZ):
            signal.signal(number, signal.SIG_DFL)
        try:
            os.chdir(cwd)
        except OSError:
            os.chdir("/")
        try:
            os.execvpe(argv[0], argv, environment)
        except OSError as error:
            os.write(2, f"task shell: {argv[0]}: {error}\n".encode(errors="replace"))
            os._exit(127 if error.errno == errno.ENOENT else 126)

    def forward(received: int, _frame) -> None:
        try:
            os.kill(child, received)
        except ProcessLookupError:
            pass

    for number in _FORWARDED:
        signal.signal(number, forward)
    while True:
        try:
            _, status = os.waitpid(child, 0)
            break
        except InterruptedError:
            continue
    code = os.waitstatus_to_exitcode(status)
    return code if code >= 0 else 128 - code


def main(argv: list[str]) -> int:
    roots: list[str] = []
    state = None
    while len(argv) > 1 and argv[0] in ("--hide", "--state"):
        if argv[0] == "--hide":
            roots.append(os.path.normpath(argv[1]))
        else:
            state = os.path.normpath(argv[1])
        argv = argv[2:]
    if not argv or argv[0] != "--" or len(argv) < 2 or state is None:
        print("usage: task_shell.py --state DIR [--hide PATH]... -- ARGV...", file=sys.stderr)
        return 2
    argv = argv[1:]
    roots = roots or list(HARNESS_MOUNTS)
    if not _under(state, roots):
        print("task shell: --state must lie under a hidden path", file=sys.stderr)
        return 2
    environment = clean_environment(dict(os.environ), roots)
    try:
        cwd = os.getcwd()
    except OSError:
        cwd = "/"
    if _under(cwd, roots):
        cwd = "/"
    try:
        _enter(*_holder(state, roots))
    except OSError as error:
        print(f"task shell: {error}", file=sys.stderr)
        return SETUP_FAILURE_EXIT
    return _run(argv, environment, cwd)


def install(helper: str, hidden: tuple[str, ...] = HARNESS_MOUNTS) -> str:
    """Write the ``bash`` that runs task shells; return its directory for the harness ``PATH``.

    ``helper`` is this file's path inside the sandbox, under a hidden path; the
    namespace state lives beside it. A harness that looks ``bash`` up on
    ``PATH`` then starts every task shell through it, and inside, ``PATH`` loses
    every hidden entry, so ``bash`` there is the image's own.
    """
    import shlex

    root = os.path.join(os.path.dirname(helper), ".task_shell")
    directory = os.path.join(root, "bin")
    os.makedirs(directory, exist_ok=True)
    if os.statvfs(directory).f_flag & os.ST_NOEXEC:
        # A noexec run directory: the bash lives on its own small tmpfs, still under the hidden path.
        flags = _MS_NOSUID | _MS_NODEV
        _check(
            _libc().mount(b"tmpfs", directory.encode(), b"tmpfs", flags, b"size=64k,mode=0755"), f"mount {directory}"
        )
    words = [sys.executable, "-I", helper, "--state", os.path.join(root, "state")]
    for path in hidden:
        words += ["--hide", path]
    shim = os.path.join(directory, "bash")
    with open(shim, "w") as handle:
        handle.write(f'#!/bin/sh\nexec {shlex.join(words)} -- bash "$@"\n')
    os.chmod(shim, 0o755)
    return directory


def activate(helper: str, hidden: tuple[str, ...] = HARNESS_MOUNTS) -> str:
    """Lead this harness's ``PATH`` with the task shell's ``bash``; return its directory.

    The directory is also exported as ``BIN_VARIABLE``, so a harness whose own
    control commands run through ``bash`` can leave it out of their ``PATH``.
    """
    directory = install(helper, hidden)
    os.environ[BIN_VARIABLE] = directory
    os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", os.defpath)
    return directory


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
