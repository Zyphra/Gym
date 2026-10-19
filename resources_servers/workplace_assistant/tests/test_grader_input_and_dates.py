# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json

import pandas as pd
import pytest
from fastapi import Request
from omegaconf import OmegaConf

from nemo_gym.base_resources_server import BaseSeedSessionRequest
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.workplace_assistant.app import (
    WorkbenchRequest,
    WorkbenchResourcesServer,
    WorkbenchResourcesServerConfig,
    WorkbenchVerifyRequest,
)
from resources_servers.workplace_assistant.utils import get_tools, is_correct
from resources_servers.workplace_assistant.workplace_assistant_tools.customer_relationship_manager import (
    CustomerRelationshipManagerTool,
)


def action(name, **kwargs):
    return {"name": name, "arguments": json.dumps(kwargs)}


def creation(name="Correct meeting"):
    return action(
        "calendar_create_event",
        event_name=name,
        participant_email="person@example.com",
        event_start="2023-12-01 09:00:00",
        duration="30",
    )


@pytest.mark.parametrize("value", [["one"], ["one", "two"], {"name": "bad"}])
def test_structured_tool_string_argument_is_wrong_not_infra(value):
    assert not is_correct([creation(value)], [creation()], None)


@pytest.mark.parametrize(
    "table,id_column",
    [("customer_relationship_manager", "customer_id"), ("project_management", "task_id")],
)
@pytest.mark.parametrize("value", [[{"bad": 1}], [["bad"]]])
def test_structured_id_mutation_is_wrong_not_infra(table, id_column, value):
    env = get_tools([table])
    tool = env["containers"][table]
    frame = tool._crm_data if table == "customer_relationship_manager" else tool._project_tasks
    record_id = frame.iloc[0][id_column]
    mutation = action(
        table + ("_update_customer" if table == "customer_relationship_manager" else "_update_task"),
        **{id_column: record_id, "field": id_column, "new_value": value},
    )
    # Depending on pandas' string dtype, the tool may reject the value or
    # persist it. Neither case may satisfy a required state change or crash.
    assert not is_correct([mutation], [creation()], None)


def server_and_request(**config_overrides):
    server = WorkbenchResourcesServer(
        config=WorkbenchResourcesServerConfig(
            host="127.0.0.1", port=8080, entrypoint="app.py", name="workplace_assistant", **config_overrides
        ),
        server_client=ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=8081), global_config_dict=OmegaConf.create({})
        ),
    )
    request = Request({"type": "http", "session": {SESSION_ID_KEY: "grader-regression"}})
    return server, request


def verify_body(actions, ground_truth):
    return WorkbenchVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="Complete the task", model="policy"),
        response=NeMoGymResponse(
            id="regression",
            object="response",
            created_at=0,
            model="policy",
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            output=[dict(item, type="function_call", call_id=str(index)) for index, item in enumerate(actions)],
        ),
        ground_truth=ground_truth,
        id=0,
        category="regression",
        environment_name="workplace_assistant",
    )


@pytest.mark.parametrize("value", [[{"bad": 1}], [["bad"]]])
def test_verify_scores_structured_id_mutation_zero_and_cleans_session(value):
    async def run():
        server, request = server_and_request()
        await server.seed_session(request, BaseSeedSessionRequest())
        env = server.session_id_to_tool_env["grader-regression"]
        record_id = env["containers"]["customer_relationship_manager"]._crm_data.iloc[0]["customer_id"]
        mutation = action(
            "customer_relationship_manager_update_customer",
            customer_id=record_id,
            field="customer_id",
            new_value=value,
        )
        result = await server.route_to_python_function(
            mutation["name"], WorkbenchRequest(**json.loads(mutation["arguments"])), request
        )
        assert result.output == "Customer updated successfully." or result.output.startswith(
            "Error executing tool 'customer_relationship_manager_update_customer': Invalid value for dtype"
        )
        result = await server.verify(request, verify_body([mutation], [creation()]))
        assert result.reward == 0.0
        assert "grader-regression" not in server.session_id_to_tool_env

    asyncio.run(run())


@pytest.mark.parametrize("enabled", [False, True])
def test_verify_noop_reference_gate_is_opt_in(enabled):
    async def run():
        server, request = server_and_request(reject_noop_reference=enabled)
        await server.seed_session(request, BaseSeedSessionRequest())
        result = await server.verify(request, verify_body([], []))
        assert result.reward == (0.0 if enabled else 1.0)

    asyncio.run(run())


def test_runtime_noop_reference_gate_is_opt_in():
    assert is_correct([], [], None)
    assert not is_correct([], [], None, reject_noop_reference=True)
    read_only = [action("calendar_search_events", query="meeting")]
    assert not is_correct(read_only, read_only, None, reject_noop_reference=True)
    assert is_correct([creation()], [creation()], None, reject_noop_reference=True)
    assert not is_correct([], [creation()], None, reject_noop_reference=True)


@pytest.mark.parametrize("column", ["last_contact_date", "follow_up_by"])
def test_crm_inclusive_date_bound_keeps_entire_day(column):
    tool = CustomerRelationshipManagerTool()
    dates = pd.to_datetime(tool._crm_data[column], errors="coerce")
    selected = dates.dropna().max().normalize()
    text = selected.strftime("%Y-%m-%d")
    expected = tool._crm_data.loc[dates < selected + pd.Timedelta(days=1), "customer_id"].tolist()
    result = tool.search_customers(**{column + "_max": text}, page_size=len(tool._crm_data) + 1)
    assert sorted(c["customer_id"] for c in result["customers"]) == sorted(expected)
    # Explicit midnight is narrower than an inclusive date, not silently widened.
    precise = tool.search_customers(**{column + "_max": text + " 00:00:00"}, page_size=len(tool._crm_data) + 1)
    expected_precise = tool._crm_data.loc[dates <= selected, "customer_id"].tolist()
    assert sorted(c["customer_id"] for c in precise["customers"]) == sorted(expected_precise)
    assert len(expected_precise) < len(expected)


def test_crm_min_max_intersection_preserves_index_alignment():
    tool = CustomerRelationshipManagerTool()
    dates = pd.to_datetime(tool._crm_data["last_contact_date"])
    start, end = dates.min().normalize(), dates.max().normalize()
    result = tool.search_customers(
        last_contact_date_min=start.strftime("%Y-%m-%d"),
        last_contact_date_max=end.strftime("%Y-%m-%d"),
        page_size=1000,
    )
    assert len(result["customers"]) == len(tool._crm_data)


@pytest.mark.parametrize("column", ["last_contact_date", "follow_up_by"])
@pytest.mark.parametrize("row_index", [0, 1])
def test_crm_search_preserves_seed_timestamps_after_date_only_update(column, row_index):
    tool = CustomerRelationshipManagerTool()
    original_dates = tool._crm_data[column].map(pd.Timestamp)
    selected = original_dates.max().normalize()
    date_text = selected.strftime("%Y-%m-%d")
    record_id = tool._crm_data.iloc[row_index]["customer_id"]
    assert tool.update_customer(record_id, column, date_text) == "Customer updated successfully."
    expected_dates = original_dates.copy()
    expected_dates.iloc[row_index] = selected
    expected = tool._crm_data.loc[
        (expected_dates >= selected) & (expected_dates < selected + pd.Timedelta(days=1)), "customer_id"
    ].tolist()
    result = tool.search_customers(**{column + "_min": date_text, column + "_max": date_text}, page_size=1000)
    assert sorted(customer["customer_id"] for customer in result["customers"]) == sorted(expected)
    assert record_id in expected


def test_calendar_start_boundary_matches_published_schema():
    env = get_tools(["calendar"])
    schema = next(s for s in env["schemas"] if s["name"] == "calendar_search_events")
    assert "event's start time" in schema["parameters"]["properties"]["time_min"]["description"]
