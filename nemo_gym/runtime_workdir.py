# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private process directories that preserve component-relative read paths."""

from __future__ import annotations

import os
import shlex
import stat
import tempfile
from pathlib import Path


def component_runtime_directory(source: Path, root: Path, server_name: str) -> Path:
    """Project immediate source entries into a new private runtime directory.

    Inputs remain aliases to the original source authority. New relative outputs
    live in the private directory. Source mounts must retain their read-only
    policy; this function never copies input bodies or makes source writable.
    The owner retains the directory until its normal job-cache cleanup, after
    service descendants have stopped.
    """
    source = source.absolute()
    root = root.absolute()
    if not source.is_dir():
        raise ValueError(f"Component source is not a directory: {source}")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError(f"Runtime directory root must be owned mode0700: {root}")
    # Unique per service start, including two aliases for one source component.
    runtime = Path(tempfile.mkdtemp(prefix="component-", dir=root))
    for child in source.iterdir():
        (runtime / child.name).symlink_to(child)
    print(
        f"[gym-component-cwd] name={server_name!r} source={str(source)!r} runtime={str(runtime)!r}",
        flush=True,
    )
    return runtime


def component_launch_command(
    setup_command: str,
    source: Path,
    entrypoint: Path,
    *,
    runtime_root: Path | None = None,
    server_name: str = "",
    environment: dict[str, str] | None = None,
    python_executable: str = "python",
) -> str:
    """Run native setup at its source, then launch from a private local cwd."""
    if entrypoint.is_absolute():
        raise ValueError("Component entrypoint must be relative to its source")
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in (environment or {}).items())
    interpreter = shlex.quote(python_executable)
    python_command = f"{prefix} {interpreter}" if prefix else interpreter
    if runtime_root is None:
        return f"{setup_command} && {python_command} {shlex.quote(str(entrypoint))}"
    source = source.absolute()
    script = source / entrypoint
    if not script.is_file():
        raise ValueError(f"Component entrypoint is unavailable: {script}")
    runtime = component_runtime_directory(source, runtime_root, server_name)
    return f"{setup_command} && cd {shlex.quote(str(runtime))} && {python_command} {shlex.quote(str(script))}"
