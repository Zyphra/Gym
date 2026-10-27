# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable private-network capability admitted by a task artifact."""

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskNetwork(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tools_root: Path
    tools_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: Literal["private"] = "private"

    @model_validator(mode="after")
    def validate_manifest(self):
        root = self.tools_root
        if (
            not root.is_absolute()
            or root.resolve() != root
            or root.is_symlink()
            or not root.is_dir()
            or any(c in str(root) for c in (":", "\n", "\r", ","))
        ):
            raise ValueError("network tools require a canonical bindable directory")
        manifest = root / "manifest.json"
        if manifest.is_symlink() or not manifest.is_file():
            raise ValueError("network tools manifest must be a regular file")
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != self.tools_sha256:
            raise ValueError("network tools manifest differs from task admission")
        return self
