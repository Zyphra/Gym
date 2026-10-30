# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.workplace_assistant.utils import (
    execute_tracked_action,
    get_tools,
    initialize_effect_tracking,
    state_matches_reference,
)


class WorkbenchResourcesServerConfig(BaseResourcesServerConfig):
    pass
    csv_seed_root: Path | None = None
    reject_noop_reference: bool = False


class WorkbenchRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class WorkbenchResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class WorkbenchVerifyRequest(BaseVerifyRequest):
    ground_truth: list[Dict[str, str]] | str
    id: int
    category: str
    environment_name: str


class WorkbenchVerifyResponse(BaseVerifyResponse):
    pass


class WorkbenchResourcesServer(SimpleResourcesServer):
    config: WorkbenchResourcesServerConfig
    session_id_to_tool_env: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        from resources_servers.workplace_assistant.workplace_assistant_tools._csv import configure_seed_root

        configure_seed_root(self.config.csv_seed_root)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/{path}")(self.route_to_python_function)
        return app

    # Register all 27 workplace tools as MCP tools via the catch-all route when expose_tools_over_mcp is enabled.
    def mcp_tools(self, harvested, catchall):
        specs = get_tools(["email", "calendar", "analytics", "project_management", "customer_relationship_manager"])[
            "schemas"
        ]
        return harvested + [catchall.tool(s["name"], s["parameters"], s.get("description")) for s in specs]

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        # init session once for each sample.
        session_id = request.session[SESSION_ID_KEY]
        toolkits = [
            "email",
            "calendar",
            "analytics",
            "project_management",
            "customer_relationship_manager",
        ]
        self.session_id_to_tool_env[session_id] = get_tools(toolkits)
        initialize_effect_tracking(self.session_id_to_tool_env[session_id])
        return BaseSeedSessionResponse()

    async def route_to_python_function(self, path: str, body: WorkbenchRequest, request: Request) -> WorkbenchResponse:
        session_id = request.session[SESSION_ID_KEY]

        # Check if session exists
        if session_id not in self.session_id_to_tool_env:
            raise HTTPException(
                status_code=400,
                detail="Session not initialized. Please call seed_session first.",
            )

        tool_env = self.session_id_to_tool_env[session_id]
        args = {key: value for key, value in body.model_dump(exclude_unset=True).items() if value is not None}

        try:
            result = execute_tracked_action(tool_env, path, args)
            return WorkbenchResponse(output=result)
        except Exception as e:
            return WorkbenchResponse(
                output=f"Error executing tool '{path}': {str(e)}"
            )  # return error to model so that it can correct itself

    async def verify(self, request: Request, body: WorkbenchVerifyRequest) -> WorkbenchVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        try:
            tool_env = self.session_id_to_tool_env.get(session_id)
            if tool_env is None:
                raise HTTPException(status_code=409, detail="Session state unavailable for effect verification")
            reward = state_matches_reference(
                tool_env, body.ground_truth, reject_noop_reference=self.config.reject_noop_reference
            )
            return WorkbenchVerifyResponse(**body.model_dump(), reward=float(reward))
        finally:
            self.session_id_to_tool_env.pop(session_id, None)


if __name__ == "__main__":
    WorkbenchResourcesServer.run_webserver()
