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
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from resources_servers.workplace_assistant.utils import get_tools, is_correct


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


_PAGINATION_PARAMETERS = {"page", "page_size", "limit", "offset", "max_results"}


def unknown_argument_error(tool_name: str, args: Dict[str, Any], schemas: List[Dict[str, Any]]) -> Optional[str]:
    """Name the arguments a tool's schema does not accept, with the accepted list.

    Without this the tool function raises a TypeError whose text never says
    which parameters exist, so the model keeps guessing.
    """
    schema = next((s for s in schemas if s["name"] == tool_name), None)
    if schema is None:
        return None
    properties = schema["parameters"].get("properties", {})
    unknown = [name for name in args if name not in properties]
    if not unknown:
        return None
    noun = "parameter" if len(unknown) == 1 else "parameters"
    message = (
        f"Error executing tool '{tool_name}': unknown {noun} {', '.join(repr(name) for name in unknown)}. "
        f"Accepted parameters: {', '.join(properties) or '(none)'}."
    )
    hints = []
    for name in unknown:
        word = re.compile(rf"\b{re.escape(name)}\b")
        covering = [p for p, spec in properties.items() if word.search(spec.get("description", ""))]
        if covering:
            description = properties[covering[0]]["description"].rstrip(".")
            hints.append(f"Hint: pass the {name} value in '{covering[0]}' ({description}).")
        elif name in _PAGINATION_PARAMETERS and not _PAGINATION_PARAMETERS & properties.keys():
            hints.append(f"Hint: '{tool_name}' has no pagination; it returns every matching record in one response.")
    return " ".join([message, *hints])


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

        argument_error = unknown_argument_error(path, args, tool_env["schemas"])
        if argument_error is not None:
            return WorkbenchResponse(output=argument_error)

        try:
            function = tool_env["functions"][path]
            result = function(**args)
            return WorkbenchResponse(output=result)
        except Exception as e:
            return WorkbenchResponse(
                output=f"Error executing tool '{path}': {str(e)}"
            )  # return error to model so that it can correct itself

    async def verify(self, request: Request, body: WorkbenchVerifyRequest) -> WorkbenchVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        try:
            ground_truth = body.ground_truth
            response = body.response.output

            total_score = 0.0

            # Convert list of ResponseFunctionToolCall objects into list of dictionaries
            predicted_function_calls = []

            for message in response:
                if message.type == "function_call":
                    predicted_function_calls.append(message.model_dump())

            predicted_chat_content = []

            for message in response:
                if message.type == "output_text":
                    predicted_chat_content.append(message.model_dump())

            total_score += (
                is_correct(
                    predicted_function_calls,
                    ground_truth,
                    None,
                    reject_noop_reference=self.config.reject_noop_reference,
                )
                * 1.0
            )
            return WorkbenchVerifyResponse(**body.model_dump(), reward=total_score)
        finally:
            self.session_id_to_tool_env.pop(session_id, None)


if __name__ == "__main__":
    WorkbenchResourcesServer.run_webserver()
