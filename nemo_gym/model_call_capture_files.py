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
"""Model-call capture files: ``<id>.capture.jsonl`` records, sealed into ``<id>.capture.jsonl.zst``.

Writers append JSONL records under an exclusive ``flock``. ``seal_capture`` moves a
capture's records into its zstd file (one frame per seal) and removes the plain file;
later appends start a new plain file. Readers use ``open_capture``/``read_capture_bytes``,
which return the sealed records followed by the plain ones. Standard library and
``zstandard`` only, so offline readers need no server dependencies.
"""

import fcntl
import io
import itertools
import os
import stat
import struct
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


CAPTURE_SUFFIX = ".capture.jsonl"
SEALED_CAPTURE_SUFFIX = CAPTURE_SUFFIX + ".zst"


def sealed_capture_path(path: str | Path) -> Path:
    """The zstd file holding the sealed records of the capture ``<id>.capture.jsonl``."""
    path = Path(path)
    return path.with_name(path.name + ".zst")


def _is_current(fd: int, path: Path) -> bool:
    """Whether ``fd`` is still the file at ``path`` (a seal unlinks the file it moved)."""
    try:
        current = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return False
    opened = os.fstat(fd)
    return (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino)


def _append_locked(path: Path, line: bytes, open_fd) -> None:
    """Append one record under the capture's exclusive lock, to the file no seal has moved."""
    while True:
        with os.fdopen(open_fd(path), "ab") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                if not _is_current(handle.fileno(), path):
                    continue
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
                return
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# Each sealed segment ends with a zstd skippable frame (zstd readers skip it) naming the
# plain file it came from. A seal publishes the sealed file before removing that plain
# file, so a plain file the mark names, unchanged since before the sealed file was
# written, is one whose removal was interrupted: its records are already sealed. (Appends
# to such a file before the next seal finishes it would read twice; the window is the
# instant between the two calls, and the idle sweep seals only files nobody is writing.)
_SEAL_MARK_MAGIC = 0x184D2A5C
_SEAL_MARK = struct.Struct("<IIQQQ")


def _seal_mark(info: os.stat_result) -> bytes:
    return _SEAL_MARK.pack(_SEAL_MARK_MAGIC, _SEAL_MARK.size - 8, info.st_dev, info.st_ino, info.st_size)


def _already_sealed(path: Path, plain) -> bool:
    info = os.fstat(plain.fileno())
    try:
        with sealed_capture_path(path).open("rb") as sealed:
            sealed_mtime_ns = os.fstat(sealed.fileno()).st_mtime_ns
            sealed.seek(-_SEAL_MARK.size, os.SEEK_END)
            mark = sealed.read(_SEAL_MARK.size)
    except OSError:
        return False
    # A later plain file can reuse the inode, but it is written after the sealed file.
    return mark == _seal_mark(info) and info.st_mtime_ns < sealed_mtime_ns


def _open_locked(path: Path, operation: int):
    """Open and lock ``path``, retrying while a seal moves it; ``None`` when absent."""
    while True:
        try:
            handle = open(path, "rb", opener=lambda name, flags: os.open(name, flags | os.O_NOFOLLOW | os.O_CLOEXEC))
        except FileNotFoundError:
            return None
        fcntl.flock(handle.fileno(), operation)
        if _is_current(handle.fileno(), path):
            return handle
        handle.close()


@contextmanager
def open_capture(path: str | Path) -> Iterator[Iterator[bytes]]:
    """Stream the JSONL lines of the capture ``<id>.capture.jsonl``, sealed or not.

    A capture is its sealed ``.zst`` records followed by any records appended to the
    plain file since. Raises ``FileNotFoundError`` when neither file exists. Plain
    captures read without ``zstandard``.
    """
    path = Path(path)
    # Shared lock: no append is half-written and no seal moves records mid-read.
    plain = _open_locked(path, fcntl.LOCK_SH)
    sealed = None
    try:
        try:
            sealed = sealed_capture_path(path).open("rb")
        except FileNotFoundError:
            if plain is None:
                raise
        parts = []
        if sealed is not None:
            import zstandard

            reader = zstandard.ZstdDecompressor().stream_reader(sealed, read_across_frames=True, closefd=False)
            parts.append(io.BufferedReader(reader, 1 << 20))
        if plain is not None and not (sealed is not None and _already_sealed(path, plain)):
            parts.append(plain)
        yield itertools.chain.from_iterable(parts)
    finally:
        for handle in (sealed, plain):
            if handle is not None:
                handle.close()


def read_capture_bytes(path: str | Path) -> bytes:
    """The JSONL bytes of a capture, sealed or not (see ``open_capture``)."""
    with open_capture(path) as lines:
        return b"".join(lines)


def seal_capture(path: str | Path) -> bool:
    """Move a capture's plain records into its zstd file and remove the plain file.

    Appends that arrive later start a new plain file, read after the sealed records
    and sealed again by the next call. Returns whether any records were moved.
    """
    import zstandard

    path = Path(path)
    plain = _open_locked(path, fcntl.LOCK_EX)
    if plain is None:
        return False
    with plain:
        info = os.fstat(plain.fileno())
        moved = bool(info.st_size) and not _already_sealed(path, plain)
        if moved:
            sealed = sealed_capture_path(path)
            temporary = sealed.with_name(f".{sealed.name}.{os.getpid()}.tmp")
            mode = stat.S_IMODE(info.st_mode)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
            try:
                with os.fdopen(os.open(temporary, flags, mode), "wb") as out:
                    os.fchmod(out.fileno(), mode)
                    try:
                        with sealed.open("rb") as previous:
                            while chunk := previous.read(1 << 20):
                                out.write(chunk)
                    except FileNotFoundError:
                        pass
                    # One frame per seal keeps an episode's repeated context inside zstd's window.
                    zstandard.ZstdCompressor(level=3).copy_stream(plain, out)
                    out.write(_seal_mark(info))
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(temporary, sealed)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        # Writers waiting on this lock find the path gone and start a new plain file.
        os.unlink(path)
        return moved


def capture_paths(directory: str | Path) -> list[Path]:
    """The ``<id>.capture.jsonl`` path of every capture in ``directory``, sealed or not."""
    names = set()
    for entry in os.scandir(directory):
        for suffix in ("", ".zst"):
            if entry.name.endswith(CAPTURE_SUFFIX + suffix) and not entry.name.startswith("."):
                names.add(entry.name.removesuffix(suffix) if suffix else entry.name)
    return [Path(directory) / name for name in sorted(names)]


def seal_idle_captures(directory: str | Path, idle_s: float) -> int:
    """Seal every capture in ``directory`` with no append for ``idle_s`` seconds; returns the count."""
    sealed = 0
    now = time.time()
    try:
        entries = list(os.scandir(directory))
    except FileNotFoundError:
        return 0
    for entry in entries:
        if not entry.name.endswith(CAPTURE_SUFFIX) or not entry.is_file(follow_symlinks=False):
            continue
        try:
            idle = now - entry.stat(follow_symlinks=False).st_mtime
        except FileNotFoundError:
            continue
        if idle >= idle_s and seal_capture(entry.path):
            sealed += 1
    return sealed
