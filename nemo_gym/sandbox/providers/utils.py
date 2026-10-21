# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers shared by the built-in sandbox providers.

These were previously duplicated verbatim across the enroot, apptainer, and
opensandbox providers. They live here so there is a single implementation.
"""

import asyncio
import posixpath
from collections.abc import Mapping
from typing import Any


def coerce_config(value: Any, config_cls: type[Any]) -> Any:
    """Accept either a config dataclass instance or a plain mapping (Hydra YAML)."""
    if value is None:
        return config_cls()
    if isinstance(value, config_cls):
        return value
    if isinstance(value, Mapping):
        return config_cls(**value)
    raise TypeError(f"{config_cls.__name__} must be a mapping or {config_cls.__name__} instance")


def path_under_mount(mount_point: str, path: str) -> str | None:
    """If `path` is inside the mount, return its path relative to the mount; else None."""
    if not path.startswith("/"):
        return None
    mp = posixpath.normpath(mount_point.rstrip("/") or "/")
    normalized = posixpath.normpath(path)
    if normalized == mp:
        return ""
    try:
        if posixpath.commonpath([mp, normalized]) != mp:
            return None
    except ValueError:
        return None
    if mp == "/":
        return normalized.lstrip("/")
    return normalized[len(mp) + 1 :]


class CapturedTimeout(TimeoutError):
    def __init__(self, message: str, stdout: str, stderr: str, facts: dict[str, int | bool]) -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr
        self.facts = facts


class PipeCapture:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.retained = bytearray()
        self.seen = 0
        self.complete = False

    async def drain(self, reader: asyncio.StreamReader) -> None:
        while chunk := await reader.read(64 * 1024):
            self.seen += len(chunk)
            if len(self.retained) < self.limit:
                self.retained.extend(chunk[: self.limit - len(self.retained)])
        self.complete = True


def capture_facts(stdout: PipeCapture, stderr: PipeCapture) -> dict[str, int | bool]:
    return {
        "stdout_bytes": stdout.seen,
        "stderr_bytes": stderr.seen,
        "stdout_truncated": stdout.seen > len(stdout.retained) or not stdout.complete,
        "stderr_truncated": stderr.seen > len(stderr.retained) or not stderr.complete,
        "capture_complete": stdout.complete and stderr.complete,
    }
