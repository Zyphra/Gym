# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned output directories and reward freshness; no writer attribution."""

import json
import os
import shutil
import stat
import uuid
from pathlib import Path


class VerifierIntegrityRefused(ValueError):
    """Policy-visible output cannot be safely purged or admitted."""


class VerifierIntegrity:
    def __init__(self, verifier: Path, trajectories: Path):
        self.roots = []
        self.receipt = {
            "schema": "stock-verifier-freshness/v1",
            "pre_grade": [],
            "trusted_dispatch_start_ns": None,
            "filesystem_dispatch_mtime_ns": None,
            "reward": {},
            "refused": None,
        }
        try:
            for label, path in (
                ("/logs/verifier", verifier),
                ("/trajectories_mount/verifier", trajectories / "verifier"),
            ):
                path.mkdir(exist_ok=True)
                fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
                self.roots.append((label, path, fd))
        except BaseException:
            self.close()
            raise

    def close(self):
        while self.roots:
            os.close(self.roots.pop()[2])

    def save_receipt(self, path: Path):
        """Replace the receipt leaf atomically; never follow a policy-created link."""
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        temporary = ".verifier-integrity-" + uuid.uuid4().hex
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600, dir_fd=directory)
            with os.fdopen(fd, "w") as stream:
                json.dump(self.receipt, stream)
            try:
                os.replace(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory)
            except IsADirectoryError as error:
                raise VerifierIntegrityRefused("receipt leaf is a directory") from error
        finally:
            try:
                try:
                    os.unlink(temporary, dir_fd=directory)
                except FileNotFoundError:
                    pass
            finally:
                os.close(directory)

    def _check_roots(self):
        for _, path, fd in self.roots:
            try:
                current, pinned = path.lstat(), os.fstat(fd)
            except OSError as error:
                raise VerifierIntegrityRefused("verifier output root is unavailable") from error
            if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != (pinned.st_dev, pinned.st_ino):
                raise VerifierIntegrityRefused("verifier output root was replaced")

    def purge(self):
        """Empty both pinned bind sources in place, never following child links."""
        try:
            self._check_roots()
            if not shutil.rmtree.avoids_symlink_attacks:
                raise VerifierIntegrityRefused("descriptor-relative purge is unavailable")
            seen = set()
            for label, _, fd in self.roots:
                info = os.fstat(fd)
                record = {"root": label, "device": info.st_dev, "inode": info.st_ino, "entries": [], "count": 0}
                self.receipt["pre_grade"].append(record)
                identity = info.st_dev, info.st_ino
                if identity in seen:
                    record["alias_of_previous"] = True
                    continue
                seen.add(identity)
                with os.scandir(fd) as entries:
                    for entry in entries:
                        metadata = entry.stat(follow_symlinks=False)
                        record["count"] += 1
                        if len(record["entries"]) < 128:
                            record["entries"].append(
                                {
                                    "name": entry.name[:256],
                                    "mode": metadata.st_mode,
                                    "size": metadata.st_size,
                                    "mtime_ns": metadata.st_mtime_ns,
                                }
                            )
                        if stat.S_ISDIR(metadata.st_mode):
                            shutil.rmtree(entry.name, dir_fd=fd)
                        else:
                            os.unlink(entry.name, dir_fd=fd)
                with os.scandir(fd) as remaining:
                    if next(remaining, None) is not None:
                        raise VerifierIntegrityRefused("verifier output was repopulated during purge")
            self._check_roots()
        except (OSError, ValueError) as error:
            self.receipt["refused"] = str(error)[:512]
            raise VerifierIntegrityRefused(self.receipt["refused"]) from error

    def mark_dispatch(self, trusted_dispatch_start_ns: int):
        """Capture the output filesystem clock before dispatch; leave no marker behind."""
        marker = ".verifier-dispatch-" + uuid.uuid4().hex
        directory = self.roots[0][2]
        try:
            self._check_roots()
            fd = os.open(
                marker,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=directory,
            )
            try:
                self.receipt["trusted_dispatch_start_ns"] = trusted_dispatch_start_ns
                self.receipt["filesystem_dispatch_mtime_ns"] = os.fstat(fd).st_mtime_ns
            finally:
                try:
                    os.unlink(marker, dir_fd=directory)
                finally:
                    os.close(fd)
            self._check_roots()
        except OSError as error:
            raise VerifierIntegrityRefused(f"filesystem dispatch anchor unavailable: errno={error.errno}") from error

    def read_reward(self):
        """Read one regular bounded leaf after sandbox stop; preserve float semantics."""
        self._check_roots()
        try:
            fd = os.open("reward.txt", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.roots[0][2])
        except FileNotFoundError:
            raise
        except OSError as error:
            raise VerifierIntegrityRefused(f"reward leaf cannot be safely opened: errno={error.errno}") from error
        try:
            info = os.fstat(fd)
            self.receipt["reward"] = {"mtime_ns": info.st_mtime_ns, "size": info.st_size, "mode": info.st_mode}
            start = self.receipt["filesystem_dispatch_mtime_ns"]
            if not stat.S_ISREG(info.st_mode) or info.st_size > 4096:
                raise VerifierIntegrityRefused("reward is not a bounded regular file")
            if start is None or info.st_mtime_ns < start:
                raise VerifierIntegrityRefused("reward predates trusted verifier dispatch")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                text = stream.read(4097)
            if len(text) > 4096:
                raise VerifierIntegrityRefused("reward exceeds the bounded read")
            return float(text.decode("utf-8").strip())
        finally:
            os.close(fd)
