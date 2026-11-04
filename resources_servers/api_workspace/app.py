# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native, session-isolated HTTP resource server for fictional SaaS workspaces."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.api_workspace import realapi
from resources_servers.api_workspace.engine import APIError, Workspace
from resources_servers.api_workspace.scenarios import REVISION, REVISION_V5, scenario_from_id


class APIWorkspaceConfig(BaseResourcesServerConfig):
    num_workers: Literal[1] = 1
    max_active_sessions: int = Field(default=65536, ge=1)
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED


class WorkspaceSeed(BaseSeedSessionRequest):
    task_id: str


class WorkspaceVerify(BaseVerifyRequest):
    task_id: str


class AccountsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    function: str | None = None
    connected_only: bool = False
    query: str = ""


class CatalogRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = ""
    function: str | None = None
    connected_only: bool = False
    method: str | None = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class SchemaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    vendor: str


class APIRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    vendor: str
    method: str
    path: str
    params: dict[str, Any] | str | None = None
    body: dict[str, Any] | str | None = None
    idempotency_key: str | None = None

    @field_validator("params", "body", mode="before")
    @classmethod
    def request_object(cls, value: Any) -> dict[str, Any]:
        """Accept a JSON object or its serialized representation; null means absent."""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Expected a JSON object, a serialized JSON object, or null; Python None is not JSON."
                ) from error
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("Request containers must decode to an object, not an array or scalar.")
        return value


class ParallelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    requests: list[APIRequest] = Field(min_length=1, max_length=20)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    connected_only: bool = False


class FetchRequest(BaseModel):
    """Containers stay untyped here so the API itself explains JSON-string mistakes."""

    model_config = ConfigDict(extra="forbid")
    method: Any
    url: Any
    params: Any = None
    body: Any = None


class CodecRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    operation: Literal["base64url_encode", "base64url_decode"]
    text: str


class WorkspaceVerifyResponse(BaseVerifyResponse):
    # Named native grading record; the reward stays the binary final-state score.
    task_evidence: dict[str, dict[str, Any]]


@dataclass
class _Session:
    world: Workspace
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _revision_error(session: _Session, revision: str) -> JSONResponse | None:
    if session.world.scenario.revision == revision:
        return None
    return JSONResponse(
        {"error": {"code": "tool_not_in_revision", "message": "This tool is not part of this task's API surface."}},
        status_code=400,
    )


class APIWorkspaceResourcesServer(SimpleResourcesServer):
    ray_enabled: ClassVar[bool] = False
    config: APIWorkspaceConfig
    initialization_preflight_read_only: ClassVar[bool] = True
    _sessions: dict[str, _Session] = PrivateAttr(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/workspace_catalog")(self.workspace_catalog)
        app.post("/workspace_accounts")(self.workspace_accounts)
        app.post("/workspace_schema")(self.workspace_schema)
        app.post("/workspace_request")(self.workspace_request)
        app.post("/workspace_parallel")(self.workspace_parallel)
        app.post("/workspace_search")(self.workspace_search)
        app.post("/workspace_fetch")(self.workspace_fetch)
        app.post("/workspace_codec")(self.workspace_codec)
        app.post("/close")(self.close_session)
        return app

    def _session(self, request: Request) -> _Session:
        session = self._sessions.get(request.session[SESSION_ID_KEY])
        if session is None:
            raise HTTPException(404, "workspace_session_missing; seed a canonical task first")
        return session

    async def seed_session(self, request: Request, body: WorkspaceSeed) -> dict[str, Any]:
        try:
            scenario = scenario_from_id(body.task_id)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        identity = request.session[SESSION_ID_KEY]
        if identity not in self._sessions and len(self._sessions) >= self.config.max_active_sessions:
            raise HTTPException(503, "workspace_capacity_exhausted; close completed sessions")
        previous = self._sessions.get(identity)
        if previous is not None:
            async with previous.lock:
                self._sessions[identity] = _Session(Workspace(scenario))
        else:
            self._sessions[identity] = _Session(Workspace(scenario))
        return {"task_id": scenario.task_id, "namespace": scenario.namespace, "reset": True}

    def _v4_session(self, request: Request) -> tuple[_Session, JSONResponse | None]:
        session = self._session(request)
        return session, _revision_error(session, REVISION)

    async def workspace_catalog(self, request: Request, body: CatalogRequest) -> JSONResponse:
        session, refused = self._v4_session(request)
        if refused is not None:
            return refused
        try:
            return JSONResponse(session.world.catalog(**body.model_dump()))
        except APIError as error:
            return JSONResponse(error.payload(), status_code=error.status)

    async def workspace_accounts(self, request: Request, body: AccountsRequest) -> JSONResponse:
        session, refused = self._v4_session(request)
        if refused is not None:
            return refused
        try:
            return JSONResponse(session.world.discover(**body.model_dump()))
        except APIError as error:
            return JSONResponse(error.payload(), status_code=error.status)

    async def workspace_schema(self, request: Request, body: SchemaRequest) -> JSONResponse:
        session, refused = self._v4_session(request)
        if refused is not None:
            return refused
        try:
            return JSONResponse(session.world.schema(body.vendor))
        except APIError as error:
            return JSONResponse(error.payload(), status_code=error.status)

    async def _call(self, session: _Session, body: APIRequest) -> tuple[int, dict[str, Any]]:
        async with session.lock:
            try:
                result = session.world.request(**body.model_dump())
                return result["status"], result
            except APIError as error:
                return error.status, error.payload()

    async def workspace_request(self, request: Request, body: APIRequest) -> JSONResponse:
        session, refused = self._v4_session(request)
        if refused is not None:
            return refused
        status, result = await self._call(session, body)
        return JSONResponse(
            result,
            status_code=status,
            headers={"WWW-Authenticate": 'Bearer realm="fictional-workspace"'} if status == 401 else None,
        )

    async def workspace_parallel(self, request: Request, body: ParallelRequest) -> Any:
        session, refused = self._v4_session(request)
        if refused is not None:
            return refused
        results = await asyncio.gather(*(self._call(session, item) for item in body.requests))
        return {"results": [{"status": status, "body": payload} for status, payload in results]}

    async def workspace_search(self, request: Request, body: SearchRequest) -> JSONResponse:
        session = self._session(request)
        refused = _revision_error(session, REVISION_V5)
        return refused or JSONResponse(realapi.search(session.world, body.query, body.top_k, body.connected_only))

    async def workspace_fetch(self, request: Request, body: FetchRequest) -> JSONResponse:
        session = self._session(request)
        refused = _revision_error(session, REVISION_V5)
        if refused is not None:
            return refused
        async with session.lock:
            status, payload = realapi.fetch(session.world, body.method, body.url, body.params, body.body)
        return JSONResponse(
            {"status": status, "body": payload},
            status_code=200 if status == 204 else status,
            headers={"WWW-Authenticate": 'Bearer realm="fictional-workspace"'} if status == 401 else None,
        )

    async def workspace_codec(self, request: Request, body: CodecRequest) -> JSONResponse:
        session = self._session(request)
        refused = _revision_error(session, REVISION_V5)
        if refused is not None:
            return refused
        status, payload = realapi.codec(body.operation, body.text)
        return JSONResponse(payload, status_code=status)

    async def score_session(self, request: Request, task_id: str) -> dict[str, Any]:
        session = self._session(request)
        async with session.lock:
            if session.world.scenario.task_id != task_id:
                raise HTTPException(409, "workspace_task_identity_mismatch")
            return session.world.score()

    async def verify(self, request: Request, body: WorkspaceVerify) -> WorkspaceVerifyResponse:
        details = await self.score_session(request, body.task_id)
        await self.close_session(request)
        return WorkspaceVerifyResponse(
            **body.model_dump(),
            reward=details["reward"],
            task_evidence={"api_workspace_grading": details["assertions"]},
        )

    async def close_session(self, request: Request) -> dict[str, bool]:
        self._sessions.pop(request.session[SESSION_ID_KEY], None)
        return {"success": True}


if __name__ == "__main__":
    APIWorkspaceResourcesServer.run_webserver()
