# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a same-sandbox verifier command against a ``/logs/verifier`` that no other process sees.

Run as the sandbox root with ``python -I -S private_verifier.py COMMAND``, from the read-only
Gym bind and the read-only agent runtime interpreter. Standard library only.

Agent processes stay alive during grading in a live sandbox, because some tasks grade a
service the agent started. Those processes share the sandbox's mount namespace, so a
process left behind could rewrite ``/logs/verifier/reward.txt`` after ``test.sh`` wrote it.
This program gives the command its own mount namespace in which ``/logs/verifier`` is an
empty tmpfs. Everything else is the sandbox's own view: the same files, processes, network
and services. With ``--tests SOURCE SHA256`` the command also reads its tests from a private ``/tests``:
the program copies them from ``SOURCE`` (the staged tree, which sandbox processes can
rewrite) and admits a file only if its bytes match the host's manifest ``SOURCE-manifest.json``
whose sha256 the host passes. Files staged there but absent from the manifest are ignored.
Both private directories must still be mounted when the command ends; a sandbox process
that detached one (by renaming its mount point in the sandbox's namespace) leaves an
``integrity`` problem the host refuses to grade.

With ``--startup BASELINE SHA256 WORKDIR`` the command also runs without the startup files
added or changed since the sandbox started (``BASELINE``, taken by ``--scan-startup`` before the
agent ran): Python ``sitecustomize``/``usercustomize`` and ``.pth`` files no installer recorded,
modules that shadow the test runner, ``conftest.py`` in ``/`` and the working directory and the agent's git configuration.
They are hidden in this namespace only, so services the agent left running keep them, and git runs
with hooks and fsmonitor off. After the command exits, the program
reports every output it wrote on stdout, as one ``MARKER`` line:

* small files inline (``data``, base64), so the host never reads them from the sandbox;
* larger files by size and sha256, copied into the shared ``/logs/verifier`` for the host
  to verify before use.

The command's own stdout and stderr go to this program's stderr. The report carries the
command's exit status; this program exits 0 once it wrote the report, or ``SETUP_FAILED``
with a ``setup`` problem in the report when the private directory could not be made, in
which case the command never runs. Any other exit (this program killed) means the report on
stdout, if any, is not its own, so the host never grades it.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time


MARKER = "NEMO_GYM_PRIVATE_VERIFIER "
VERIFIER_DIR = "/logs/verifier"
TESTS_DIR = "/tests"
TESTS_MANIFEST_SUFFIX = "-manifest.json"
COPY_CHUNK = 1 << 20
SETUP_FAILED = 125
# Inline outputs keep the report well under a provider's 1 MiB exec output capture.
INLINE_FILE_BYTES = 64 * 1024
INLINE_TOTAL_BYTES = 384 * 1024
# What a command stopped by its timeout had written is copied out within these bounds, inside the
# provider's grace period between SIGTERM and SIGKILL.
UNADMITTED_BYTES = 64 << 20
UNADMITTED_SECONDS = 10.0

CLONE_NEWNS = 0x00020000
MS_NOSUID, MS_NODEV, MS_BIND, MS_REC, MS_PRIVATE = 0x2, 0x4, 0x1000, 0x4000, 0x40000

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_char_p]


def _mount(source: str | None, target: str, fstype: str | None, flags: int, data: str | None = None) -> None:
    encode = lambda value: None if value is None else value.encode()  # noqa: E731
    if _libc.mount(encode(source), encode(target), encode(fstype), flags, encode(data)) != 0:
        error = ctypes.get_errno()
        raise OSError(error, f"mount {target}: {os.strerror(error)}")


def _mount_ids() -> dict[str, str]:
    """Mount point -> mount id of this namespace's mounts."""
    with open("/proc/self/mountinfo") as stream:
        fields = [line.split() for line in stream]
    return {field[4]: field[0] for field in fields}


def _private_tmpfs(path: str) -> str:
    """Mount an empty tmpfs at ``path`` in this namespace; return its mount id."""
    _mount("tmpfs", path, "tmpfs", MS_NOSUID | MS_NODEV, "mode=0755")
    # A link raced in at ``path`` would have moved the mount elsewhere.
    mount_id = _mount_ids().get(path)
    if mount_id is None or os.path.realpath(path) != path:
        raise RuntimeError(f"private {path} is not mounted at {path}")
    if os.listdir(path):
        raise RuntimeError(f"private {path} is not empty")
    return mount_id


def private_directory() -> tuple[int, dict[str, str]]:
    """Cover ``/logs/verifier`` in a new mount namespace; return a descriptor of the shared one."""
    os.makedirs(VERIFIER_DIR, exist_ok=True)
    shared = os.open(VERIFIER_DIR, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    os.unshare(CLONE_NEWNS)
    # Nothing mounted here may propagate back into the sandbox's namespace.
    _mount(None, "/", None, MS_REC | MS_PRIVATE)
    return shared, {VERIFIER_DIR: _private_tmpfs(VERIFIER_DIR)}


def private_tests(source: str, manifest_sha256: str) -> str:
    """Fill a private ``/tests`` with exactly the manifest's files, each verified as written."""
    with open(source + TESTS_MANIFEST_SUFFIX, "rb") as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise RuntimeError("tests manifest differs from the host's")
    manifest = json.loads(raw)
    if os.path.islink(TESTS_DIR) or (os.path.lexists(TESTS_DIR) and not os.path.isdir(TESTS_DIR)):
        os.unlink(TESTS_DIR)
    os.makedirs(TESTS_DIR, exist_ok=True)
    mount_id = _private_tmpfs(TESTS_DIR)
    for directory in manifest["directories"]:
        os.mkdir(os.path.join(TESTS_DIR, directory), 0o755)
    for link in manifest["links"]:
        os.symlink(link["target"], os.path.join(TESTS_DIR, link["path"]))
    for entry in manifest["files"]:
        digest, size = hashlib.sha256(), 0
        target = os.path.join(TESTS_DIR, entry["path"])
        with (
            open(os.path.join(source, entry["path"]), "rb") as src,
            open(
                os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600), "wb"
            ) as dst,
        ):
            while chunk := src.read(COPY_CHUNK):
                digest.update(chunk)
                size += len(chunk)
                dst.write(chunk)
        if size != entry["size"] or digest.hexdigest() != entry["sha256"]:
            raise RuntimeError(f"staged test {entry['path']} differs from the task source")
        os.chmod(target, entry["mode"])
    return mount_id


STARTUP_FILES = ("sitecustomize.py", "usercustomize.py")
SHADOW_MODULES = ("pytest.py", "py.py", "_pytest.py", "pluggy.py", "unittest.py")
# Directories Python adds at startup, by layout rather than by asking an interpreter (which the agent can replace).
PYTHON_DIR_GLOBS = (
    "/usr/lib/python3*",
    "/usr/lib/python3*/site-packages",
    "/usr/lib/python3*/dist-packages",
    "/usr/lib/python3/dist-packages",
    "/usr/local/lib/python3*",
    "/usr/local/lib/python3*/site-packages",
    "/usr/local/lib/python3*/dist-packages",
    "/opt/*/lib/python3*",
    "/opt/*/lib/python3*/site-packages",
    "/opt/*/*/lib/python3*",
    "/opt/*/*/lib/python3*/site-packages",
    "/root/.local/lib/python3*/site-packages",
    "/home/*/.local/lib/python3*/site-packages",
    "/root/.local/share/uv/python/*/lib/python3*",
    "/root/.local/share/uv/python/*/lib/python3*/site-packages",
    "/*/.venv/lib/python3*/site-packages",
    "/*/*/.venv/lib/python3*/site-packages",
)
# pytest's rootdir (``/pytest.ini``) and root's home.
ROOT_DIRS = ("/", "/root")
GIT_CONFIGS = ("/etc/gitconfig", "/root/.gitconfig", "/root/.config/git/config")
# Command-line git settings outrank every config file: no hook or fsmonitor runs inside grading.
GIT_SETTINGS = (
    ("core.hooksPath", "/dev/null"),
    ("core.fsmonitor", "false"),
    ("commit.gpgSign", "false"),
    ("tag.gpgSign", "false"),
)


def _digest(path: str) -> str | None:
    try:
        with open(path, "rb") as stream:
            return hashlib.sha256(stream.read()).hexdigest()
    except OSError:
        return None


def startup_files(workdir: str) -> dict[str, str]:
    """Files that run code when Python, pytest or git start, with their sha256."""
    import glob

    found = {}
    for pattern in PYTHON_DIR_GLOBS:
        for directory in glob.glob(pattern):
            if not os.path.isdir(directory) or os.path.islink(directory):
                continue
            for name in os.listdir(directory):
                if name in STARTUP_FILES or name.endswith(".pth"):
                    found[os.path.join(directory, name)] = None
    for directory in {*ROOT_DIRS, workdir}:
        for name in SHADOW_MODULES + (("conftest.py",) if directory in (ROOT_DIRS[0], workdir) else ()):
            found[os.path.join(directory, name)] = None
    for path in GIT_CONFIGS:
        found[path] = None
    return {path: digest for path in sorted(found) if os.path.isfile(path) and (digest := _digest(path)) is not None}


def _installed(path: str) -> bool:
    """A ``.pth`` file an installer recorded (``*.dist-info/RECORD``) beside it."""
    import glob

    directory, name = os.path.split(path)
    for record in glob.glob(os.path.join(glob.escape(directory), "*.dist-info", "RECORD")):
        try:
            with open(record, errors="replace") as stream:
                if any(line.split(",", 1)[0] == name for line in stream):
                    return True
        except OSError:
            continue
    return False


def neutralize_startup_files(baseline: dict[str, str], workdir: str) -> tuple[list[str], dict[str, str]]:
    """Hide, in this namespace only, every startup file added or changed since the sandbox started.

    Returns the hidden paths and the environment that keeps the agent's git settings out.
    """
    hidden, env = [], {}
    for path, digest in startup_files(workdir).items():
        if baseline.get(path) == digest:
            continue
        if path in GIT_CONFIGS:
            system = path == GIT_CONFIGS[0]
            env["GIT_CONFIG_NOSYSTEM" if system else "GIT_CONFIG_GLOBAL"] = "1" if system else "/dev/null"
            hidden.append(path)
            continue
        if path.endswith(".pth") and _installed(path):
            continue
        _mount("/dev/null", path, None, MS_BIND)
        hidden.append(path)
    count = int(os.environ.get("GIT_CONFIG_COUNT", "0") or 0)
    for index, (key, value) in enumerate(GIT_SETTINGS, start=count):
        env[f"GIT_CONFIG_KEY_{index}"], env[f"GIT_CONFIG_VALUE_{index}"] = key, value
    env["GIT_CONFIG_COUNT"] = str(count + len(GIT_SETTINGS))
    return hidden, env


def _baseline(path: str, sha256: str) -> dict[str, str]:
    with open(path, "rb") as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise RuntimeError("startup baseline differs from the host's")
    return json.loads(raw)


def detached(mounts: dict[str, str]) -> list[str]:
    """Private mounts that are no longer where the command found them."""
    current = _mount_ids()
    return [path for path, mount_id in sorted(mounts.items()) if current.get(path) != mount_id]


def _copy_out(shared: int, relative: str, data: bytes, mode: int) -> None:
    """Write one output under the shared directory without following any link there."""
    *parents, name = relative.split("/")
    directory = os.dup(shared)
    try:
        for part in parents:
            try:
                os.mkdir(part, 0o755, dir_fd=directory)
            except FileExistsError:
                pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
            os.close(directory)
            directory = child
        try:
            os.unlink(name, dir_fd=directory)
        except FileNotFoundError:
            pass
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, mode, dir_fd=directory)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
    finally:
        os.close(directory)


def collect(shared: int | None) -> dict:
    """Describe every output in the private directory; copy the large ones out."""
    files, links, problems = [], [], []
    inline_left = INLINE_TOTAL_BYTES
    for root, directories, names in os.walk(VERIFIER_DIR):
        directories.sort()
        for name in sorted(names) + [d for d in directories if os.path.islink(os.path.join(root, d))]:
            path = os.path.join(root, name)
            relative = os.path.relpath(path, VERIFIER_DIR)
            info = os.lstat(path)
            if stat.S_ISLNK(info.st_mode):
                links.append({"path": relative, "target": os.readlink(path)})
                continue
            if not stat.S_ISREG(info.st_mode):
                problems.append({"path": relative, "error": "not a regular file"})
                continue
            with open(path, "rb") as stream:
                data = stream.read()
            entry = {
                "path": relative,
                "size": len(data),
                "mode": stat.S_IMODE(info.st_mode),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            if len(data) <= INLINE_FILE_BYTES and len(data) <= inline_left:
                inline_left -= len(data)
                entry["data"] = base64.b64encode(data).decode()
            elif shared is None:
                problems.append({"path": relative, "error": "no shared directory to copy into"})
            else:
                try:
                    _copy_out(shared, relative, data, stat.S_IMODE(info.st_mode))
                except OSError as error:
                    problems.append({"path": relative, "error": f"copy out: {error}"})
            files.append(entry)
    return {"files": files, "links": links, "problems": problems}


class _Terminated(Exception):
    """SIGTERM arrived while the command ran."""


def _terminated(signum, frame) -> None:
    raise _Terminated


def keep_unfinished(shared: int) -> None:
    """Copy the regular files the stopped command wrote into the shared directory, within bounds.

    Links and other special files are neither followed nor copied. The host never grades these.
    """
    deadline = time.monotonic() + UNADMITTED_SECONDS
    left = UNADMITTED_BYTES
    for root, directories, names in os.walk(VERIFIER_DIR):
        directories.sort()
        for name in sorted(names):
            if time.monotonic() > deadline:
                return
            path = os.path.join(root, name)
            try:
                info = os.lstat(path)
                if not stat.S_ISREG(info.st_mode) or info.st_size > left:
                    continue
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
                with os.fdopen(fd, "rb") as stream:
                    data = stream.read(left + 1)
                if len(data) > left:
                    continue
                _copy_out(shared, os.path.relpath(path, VERIFIER_DIR), data, stat.S_IMODE(info.st_mode))
                left -= len(data)
            except OSError:
                continue


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "--scan-startup":
        sys.stdout.write(json.dumps(startup_files(argv[1]), sort_keys=True))
        return 0
    tests = startup = snapshot_after = None
    while len(argv) > 1 and argv[0] in ("--tests", "--startup", "--snapshot-after"):
        if argv[0] == "--tests":
            tests, argv = (argv[1], argv[2]), argv[3:]
        elif argv[0] == "--snapshot-after":
            snapshot_after, argv = float(argv[1]), argv[2:]
        else:
            startup, argv = (argv[1], argv[2], argv[3]), argv[4:]
    if len(argv) != 1:
        print(
            "usage: private_verifier.py [--tests SOURCE SHA256] [--startup BASELINE SHA256 WORKDIR]"
            " [--snapshot-after SECONDS] COMMAND | --scan-startup WORKDIR",
            file=sys.stderr,
        )
        return SETUP_FAILED
    started = time.monotonic()
    report: dict = {"schema": "private-verifier/v1", "setup": None, "exit_code": None, "integrity": []}
    try:
        shared, mounts = private_directory()
        if tests is not None:
            mounts[TESTS_DIR] = private_tests(*tests)
        env = None
        if startup is not None:
            report["hidden_startup_files"], extra = neutralize_startup_files(_baseline(*startup[:2]), startup[2])
            env = {**os.environ, **extra}
    except BaseException as error:  # noqa: BLE001 - reported; the command never runs
        report["setup"] = f"{type(error).__name__}: {error}"
        sys.stdout.write(MARKER + json.dumps(report) + "\n")
        return SETUP_FAILED
    report["setup_s"] = round(time.monotonic() - started, 4)
    command_started = time.monotonic()
    # Until the report is out, a SIGTERM (the provider's timeout) leaves the outputs, not a report.
    signal.signal(signal.SIGTERM, _terminated)
    process = None
    try:
        # The command's output must not mix with the report on stdout.
        process = subprocess.Popen(["/bin/sh", "-c", argv[0]], stdin=subprocess.DEVNULL, stdout=2, stderr=2, env=env)
        try:
            status = process.wait(timeout=snapshot_after)
        except subprocess.TimeoutExpired:
            # Near the host's timeout, a provider that kills without a grace period leaves no
            # chance later: copy what the command wrote so far and let it run on.
            keep_unfinished(shared)
            status = process.wait()
        report["command_s"] = round(time.monotonic() - command_started, 4)
        report["exit_code"] = status if status >= 0 else 128 - status
        report["integrity"] = [f"private {path} was detached during the command" for path in detached(mounts)]
        collect_started = time.monotonic()
        report.update(collect(shared))
        report["collect_s"] = round(time.monotonic() - collect_started, 4)
        sys.stdout.write(MARKER + json.dumps(report) + "\n")
        sys.stdout.flush()
    except _Terminated:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if process is not None and process.poll() is None:
            process.kill()
        # This namespace and its private directory remain until this program exits.
        keep_unfinished(shared)
        return 128 + signal.SIGTERM
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
