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
``integrity`` problem the host refuses to grade. After the command exits, the program
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

CLONE_NEWNS = 0x00020000
MS_NOSUID, MS_NODEV, MS_REC, MS_PRIVATE = 0x2, 0x4, 0x4000, 0x40000

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


def main(argv: list[str]) -> int:
    tests = None
    if len(argv) == 4 and argv[0] == "--tests":
        tests, argv = (argv[1], argv[2]), argv[3:]
    if len(argv) != 1:
        print("usage: private_verifier.py [--tests SOURCE MANIFEST_SHA256] COMMAND", file=sys.stderr)
        return SETUP_FAILED
    started = time.monotonic()
    report: dict = {"schema": "private-verifier/v1", "setup": None, "exit_code": None, "integrity": []}
    try:
        shared, mounts = private_directory()
        if tests is not None:
            mounts[TESTS_DIR] = private_tests(*tests)
    except BaseException as error:  # noqa: BLE001 - reported; the command never runs
        report["setup"] = f"{type(error).__name__}: {error}"
        sys.stdout.write(MARKER + json.dumps(report) + "\n")
        return SETUP_FAILED
    report["setup_s"] = round(time.monotonic() - started, 4)
    command_started = time.monotonic()
    # The command's output must not mix with the report on stdout.
    status = subprocess.call(["/bin/sh", "-c", argv[0]], stdin=subprocess.DEVNULL, stdout=2, stderr=2)
    report["command_s"] = round(time.monotonic() - command_started, 4)
    report["exit_code"] = status if status >= 0 else 128 - status
    report["integrity"] = [f"private {path} was detached during the command" for path in detached(mounts)]
    collect_started = time.monotonic()
    report.update(collect(shared))
    report["collect_s"] = round(time.monotonic() - collect_started, 4)
    sys.stdout.write(MARKER + json.dumps(report) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
