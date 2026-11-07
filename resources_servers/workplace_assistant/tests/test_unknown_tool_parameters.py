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
"""A call with a parameter the tool schema does not declare is refused with the accepted list.

The error names the unknown parameter, lists the schema's parameters and, when one applies, says which accepted
parameter already covers it. Nothing executes, so state is unchanged. Real server, real seeded tools.
"""

import inspect
from unittest.mock import MagicMock

import pytest
from fastapi import Request

from nemo_gym.base_resources_server import BaseSeedSessionRequest
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.workplace_assistant.app import (
    WorkbenchRequest,
    WorkbenchResourcesServer,
    WorkbenchResourcesServerConfig,
)
from resources_servers.workplace_assistant.utils import get_tools


TOOLKITS = ["email", "calendar", "analytics", "project_management", "customer_relationship_manager"]


@pytest.fixture
async def session():
    config = WorkbenchResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    server = WorkbenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    request = Request(scope={"type": "http", "session": {SESSION_ID_KEY: "unknown-params"}})
    await server.seed_session(request, BaseSeedSessionRequest())

    async def call(tool, **arguments):
        response = await server.route_to_python_function(
            path=tool, body=WorkbenchRequest(**arguments), request=request
        )
        return response.output

    return server.session_id_to_tool_env["unknown-params"], call


def snapshot(env):
    containers = env["containers"]
    return {
        "calendar": containers["calendar"]._calendar_events.copy(),
        "tasks": containers["project_management"]._project_tasks.copy(),
    }


def assert_unchanged(before, env):
    after = snapshot(env)
    for name, frame in before.items():
        assert frame.equals(after[name]), name


async def test_participant_email_on_calendar_search_names_query(session):
    env, call = session
    before = snapshot(env)
    output = await call("calendar_search_events", participant_email="aisha.chen@atlas.com")
    assert output == (
        "Error executing tool 'calendar_search_events': unknown parameter 'participant_email'. "
        "Accepted parameters: query, time_min, time_max, page, page_size. "
        "Hint: pass the participant_email value in 'query' (Query to search for. "
        "Terms will be matched in the event_name and participant_email fields)."
    )
    assert "unexpected keyword argument" not in output
    assert_unchanged(before, env)


async def test_page_on_task_search_says_there_is_no_pagination(session):
    env, call = session
    before = snapshot(env)
    output = await call("project_management_search_tasks", board="Back end", page=2)
    assert output == (
        "Error executing tool 'project_management_search_tasks': unknown parameter 'page'. "
        "Accepted parameters: task_name, assigned_to_email, list_name, due_date, board. "
        "Hint: 'project_management_search_tasks' has no pagination; it returns every matching record in one response."
    )
    assert_unchanged(before, env)


async def test_unknown_write_parameter_writes_nothing(session):
    env, call = session
    before = snapshot(env)
    output = await call(
        "calendar_create_event",
        event_name="Sync",
        participant_email="aisha.chen@atlas.com",
        event_start="2023-12-04 10:00:00",
        duration="30",
        location="Room 1",
        notes="x",
    )
    assert output.startswith("Error executing tool 'calendar_create_event': unknown parameters 'location', 'notes'. ")
    assert_unchanged(before, env)


async def test_declared_parameters_still_run(session):
    _, call = session
    events = await call("calendar_search_events", query="aisha.chen@atlas.com", page=1, page_size=2)
    assert isinstance(events, dict) and len(events["events"]) <= 2
    tasks = await call("project_management_search_tasks", board="Back end")
    assert isinstance(tasks, list) and tasks


def test_every_schema_matches_its_python_signature():
    """The check trusts the schema, so a schema missing a real parameter would refuse valid calls."""
    env = get_tools(TOOLKITS)
    schemas = {schema["name"]: schema for schema in env["schemas"]}
    assert set(schemas) == set(env["functions"])
    for name, function in env["functions"].items():
        assert set(schemas[name]["parameters"]["properties"]) == set(inspect.signature(function).parameters), name
