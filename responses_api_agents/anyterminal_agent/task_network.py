# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable private-network capability admitted by a task artifact."""

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskService(BaseModel):
    """A task-declared dependency service (a Harbor compose sidecar) on the agent's private network.

    The provider starts ``image`` (a locked SIF, verified against ``image_sha256``) in its own
    mount and PID namespace, reachable from the agent only as ``http://<hostname>:<port>``.
    ``command`` (the image's Docker CMD, which a SIF build does not keep) runs in ``workdir``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    hostname: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    port: int = Field(ge=1, le=65535)
    image: Path
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    health_path: str = Field(default="/", pattern=r"^/[A-Za-z0-9/_.-]*$")
    command: str = Field(min_length=1)
    workdir: str = Field(default="/", pattern=r"^/")

    @model_validator(mode="after")
    def validate_image(self):
        if not self.image.is_absolute() or any(c in str(self.image) for c in (":", "\n", "\r", ",")):
            raise ValueError("service image must be a canonical bindable absolute path")
        return self


class TaskNetwork(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tools_root: Path
    tools_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: Literal["private"] = "private"
    # False: loopback only, plus the one route to the policy's model endpoint.
    outbound: bool = True
    services: tuple[TaskService, ...] = ()

    def declared(self, declaration: str) -> "TaskNetwork":
        """This network with a row's ``task_network_declaration`` (outbound and services) applied."""
        values = TaskNetworkDeclaration.model_validate_json(declaration)
        return self.model_validate({**self.model_dump(), **values.model_dump()})

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


class TaskNetworkDeclaration(BaseModel):
    """The task-owned part of a private network a row declares; the run owns the tools."""

    model_config = ConfigDict(extra="forbid")
    outbound: bool = True
    services: tuple[TaskService, ...] = ()

    @model_validator(mode="after")
    def unique_hostnames(self):
        names = [service.hostname for service in self.services]
        if len(set(names)) != len(names):
            raise ValueError("task services repeat a hostname")
        return self
