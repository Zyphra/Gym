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

import os
import posixpath
from collections.abc import Iterable, Mapping
from typing import Any


# What a sandbox runtime client (``apptainer``, ``enroot``) and its helpers read from the environment
# they are started in. Everything a container needs travels explicitly (``--env-file`` or ``-e``);
# nothing reaches it by inheriting the worker's environment. Some Apptainer paths hang when the
# client environment holds a variable over 64 KiB (the trainer's run record is one), and a worker
# environment carries credentials no sandbox should see.
CLIENT_ENV_NAMES = frozenset(
    {
        "PATH",
        "LD_LIBRARY_PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "TZ",
        "TERM",
        "TMPDIR",
        "DBUS_SESSION_BUS_ADDRESS",
        # Enroot's cgroup hooks mount the delegated Slurm step cgroup only when this is set.
        "SLURM_STEP_ID",
        # Kept by ``apptainer --cleanenv`` itself, so the client must still receive them.
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "FTP_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "ftp_proxy",
        "no_proxy",
    }
)
CLIENT_ENV_PREFIXES = ("LC_", "XDG_")


def sandbox_client_environment(
    extra_prefixes: Iterable[str] = (),
    *,
    extra_names: Iterable[str] = (),
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The allow-listed subset of ``environ`` (default ``os.environ``) a sandbox client process receives."""
    names = CLIENT_ENV_NAMES.union(extra_names)
    prefixes = (*CLIENT_ENV_PREFIXES, *extra_prefixes)
    source = os.environ if environ is None else environ
    return {name: value for name, value in source.items() if name in names or name.startswith(prefixes)}


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
