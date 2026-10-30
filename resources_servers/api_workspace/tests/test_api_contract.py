# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Boundary and preservation controls for real HTTP error/status contracts."""

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser
from nemo_gym.train_data_utils import TrainDataProcessor, TrainDataProcessorConfig
from resources_servers.api_workspace.engine import COLLECTIONS, VENDORS, canonical_json
from resources_servers.api_workspace.scenarios import row_from_id, scenario_from_id, task_ids
from resources_servers.api_workspace.tests.test_app import gold_calls, server


@pytest.fixture
def http_world():
    native = server()
    identity = "api-workspace-v4:train:acct_reconcile:0000"
    with TestClient(native.setup_webserver()) as client:
        assert client.post("/seed_session", json={"task_id": identity}).status_code == 200
        world = next(iter(native._sessions.values())).world
        yield native, client, world


def test_invalid_methods_items_and_bodies_do_not_mutate(http_world):
    _, client, world = http_world
    goal = world.scenario.updates[0]
    vendor = VENDORS[goal["vendor"]]
    path = vendor.path(goal["collection"])
    item = vendor.path(goal["collection"], goal["id"])
    missing = vendor.path(goal["collection"], "absent_record")
    valid = {"vendor": vendor.name, "method": "GET", "path": path}
    original = canonical_json(world.records)
    for changes, status in (
        ({"method": "PUT"}, 405),
        ({"path": missing}, 404),
        ({"body": {"amount": 1}}, 400),
        ({"path": item, "params": {"q": "anything"}}, 400),
        ({"method": "DELETE"}, 400),
        ({"method": "DELETE", "path": item, "body": {"amount": 1}}, 400),
        ({"method": "DELETE", "path": missing}, 404),
        ({"method": "PATCH", "path": missing, "body": vendor.wire({"paid": 1})}, 404),
        ({"method": "PATCH", "path": item}, 422),
        ({"method": "POST", "path": item, "body": vendor.wire({"amount": 1})}, 405),
        ({"method": "POST", "body": vendor.wire({"amount": 1})}, 422),
        ({"method": "PATCH", "path": item, "params": {"q": "anything"}, "body": vendor.wire({"paid": 1})}, 400),
        ({"path": path + "/../invoices"}, 400),
        ({"path": path + "%3Fq=x"}, 400),
        ({"path": vendor.prefix + "/unknown_records"}, 404),
    ):
        result = client.post("/workspace_request", json=valid | changes)
        assert result.status_code == status, (changes, result.text)
        assert canonical_json(world.records) == original
    result = client.post("/workspace_request", json=valid | {"path": item})
    assert result.status_code == 200
    assert result.json()["item"] == vendor.wire(world.records[vendor.name][goal["collection"]][goal["id"]])


def test_creation_requires_complete_valid_schema_and_bounded_key(http_world):
    _, client, world = http_world
    call = next(x for x in gold_calls(world.scenario) if x["method"] == "POST")
    original = canonical_json(world.records)
    for changes, code in (
        ({"idempotency_key": None}, "idempotency_key_required"),
        ({"idempotency_key": "x" * 201}, "idempotency_key_required"),
        ({"body": call["body"] | {"shared_with": [1]}}, "invalid_field_type"),
    ):
        response = client.post("/workspace_request", json=call | changes)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == code
        assert canonical_json(world.records) == original
    response = client.post("/workspace_request", json=call)
    assert response.status_code == 201


@pytest.mark.parametrize("operator", ["eq", "in", "gt", "gte", "lt", "lte"])
def test_numeric_search_comparisons_have_correct_boundary_semantics(http_world, operator):
    _, client, world = http_world
    vendor = next(VENDORS[name] for name in world.scenario.connected if VENDORS[name].function == "accounting")
    records = list(world.records[vendor.name]["invoices"].values())
    boundary = records[0]["amount"]
    value = [boundary] if operator == "in" else boundary
    expected = {
        "eq": lambda x: x == boundary,
        "in": lambda x: x in [boundary],
        "gt": lambda x: x > boundary,
        "gte": lambda x: x >= boundary,
        "lt": lambda x: x < boundary,
        "lte": lambda x: x <= boundary,
    }
    response = client.post(
        "/workspace_request",
        json={
            "vendor": vendor.name,
            "method": "GET",
            "path": vendor.path("invoices"),
            "params": {"limit": 100, "filters": {"amount": {operator: value}}},
        },
    )
    assert response.status_code == 200
    assert {x[vendor.field("id")] for x in response.json()["items"]} == {
        x["id"] for x in records if expected[operator](x["amount"])
    }


@pytest.mark.parametrize(
    "condition",
    ["middle", {"eq": "middle"}, {"in": ["middle"]}, {"gt": "a"}, {"gte": "middle"}, {"lt": "z"}, {"lte": "middle"}],
)
def test_missing_schema_field_does_not_match_or_mutate(condition):
    native = server()
    app = native.setup_webserver()
    # Production catches unexpected resource exceptions outside session handling.
    native.setup_exception_middleware(app)
    identity = "api-workspace-v4:train:hr_location:0000"
    with TestClient(app) as client:
        assert client.post("/seed_session", json={"task_id": identity}).status_code == 200
        world = next(iter(native._sessions.values())).world
        vendor = next(VENDORS[name] for name in world.scenario.connected if VENDORS[name].function == "hr")
        employees = world.records[vendor.name]["employees"]
        # Native records carry every schema field; remove one to exercise records persisted without it.
        for row in employees.values():
            del row["manager"]
        request = {
            "vendor": vendor.name,
            "method": "GET",
            "path": vendor.path("employees"),
            "params": {"limit": 100, "filters": {vendor.field("manager"): condition}},
        }
        original = canonical_json(world.records)
        result = client.post("/workspace_request", json=request)
        assert result.status_code == 200, result.text
        assert result.json()["items"] == []
        assert canonical_json(world.records) == original
        identity = next(iter(employees))
        patched = client.post(
            "/workspace_request",
            json={
                "vendor": vendor.name,
                "method": "PATCH",
                "path": vendor.path("employees", identity),
                "body": {vendor.field("manager"): "middle"},
            },
        )
        assert patched.status_code == 200
        original = canonical_json(world.records)
        result = client.post("/workspace_request", json=request)
        assert result.status_code == 200
        assert [row[vendor.field("id")] for row in result.json()["items"]] == [identity]
        assert canonical_json(world.records) == original


def test_search_validation_and_unknown_accounts(http_world):
    _, client, world = http_world
    vendor = next(VENDORS[name] for name in world.scenario.connected if VENDORS[name].function == "accounting")
    request = {"vendor": vendor.name, "method": "GET", "path": vendor.path("invoices")}
    for params in (
        {"q": True},
        {"filters": []},
        {"limit": 0},
        {"filters": {"amount": {"bad": 1}}},
        {"filters": {"amount": {"gt": 1, "lt": 10}}},
        {"filters": {"amount": {"in": "bad"}}},
        {"filters": {"amount": {"in": [True]}}},
        {"filters": {"amount": {"gt": "1"}}},
        {"filters": {"amount": "1"}},
        {"cursor": 10},
    ):
        assert client.post("/workspace_request", json=request | {"params": params}).status_code == 400
    search = client.post("/workspace_request", json=request | {"params": {"q": "definitely_missing_text"}})
    assert search.json()["items"] == []
    assert client.post("/workspace_accounts", json={"function": "invented"}).status_code == 400
    assert client.post("/workspace_accounts", json={"query": "definitely_missing_vendor"}).json()["accounts"] == []
    assert client.post("/workspace_schema", json={"vendor": "invented"}).status_code == 404


def test_capacity_and_seed_authority_preserve_active_world():
    native = server()
    native.config.max_active_sessions = 1
    app = native.setup_webserver()
    identity = task_ids("train")[0]
    with TestClient(app) as first, TestClient(app) as second:
        assert first.post("/seed_session", json={"task_id": identity}).status_code == 200
        assert second.post("/seed_session", json={"task_id": identity}).status_code == 503
        assert first.post("/seed_session", json={"task_id": identity}).status_code == 200
        assert second.post("/seed_session", json={"task_id": "not_a_task"}).status_code == 422
        assert second.post("/workspace_accounts", json={}).status_code == 404
        assert first.post("/close", json={}).status_code == 200
        assert second.post("/seed_session", json={"task_id": identity}).status_code == 200
        assert second.post("/close", json={}).status_code == 200
    assert not native._sessions


def test_native_verify_scores_real_world_then_releases_it(http_world):
    native, client, world = http_world
    for call in gold_calls(world.scenario):
        client.post("/workspace_request", json=call).raise_for_status()
    payload = deepcopy(row_from_id(world.scenario.task_id))
    payload["response"] = {
        "id": "done",
        "created_at": 1.0,
        "object": "response",
        "model": "fixture",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    result = client.post("/verify", json=payload)
    assert result.status_code == 200
    assert result.json()["reward"] == 1
    assert not native._sessions


def test_grader_rejects_wrong_valid_values_extra_collections_and_missing_rows():
    from resources_servers.api_workspace.engine import Workspace

    scenario = scenario_from_id("api-workspace-v4:train:hr_location:0000")
    world = Workspace(scenario)
    for call in gold_calls(scenario):
        world.request(**call)
    assert world.score()["reward"] == 1
    goal = scenario.updates[0]
    world.records[goal["vendor"]][goal["collection"]][goal["id"]]["location"] = "Wrong office"
    assert world.score()["reward"] == 0
    world.records[goal["vendor"]][goal["collection"]][goal["id"]].update(goal["fields"])
    world.records[goal["vendor"]]["hidden_extra_collection"] = {}
    assert world.score()["reward"] == 0
    del world.records[goal["vendor"]]["hidden_extra_collection"]
    del world.records[goal["vendor"]][goal["collection"]][goal["id"]]
    assert world.score()["reward"] == 0
    assert len(COLLECTIONS) == 6


def test_native_local_dataset_binding_needs_no_registry_credentials(tmp_path):
    rows = tmp_path / "train.jsonl"
    rows.write_text("{}\n")
    # This is the supported literal path override accepted by the native CLI.
    config = OmegaConf.create(
        {
            "agent": {
                "responses_api_agents": {
                    "simple_agent": {
                        "entrypoint": "app.py",
                        "datasets": [
                            {
                                "name": "authored_train",
                                "type": "train",
                                "jsonl_fpath": str(rows),
                                "license": "Apache 2.0",
                            }
                        ],
                    }
                }
            }
        }
    )
    instances = GlobalConfigDictParser().filter_for_server_instance_configs(config)
    assert len(instances) == 1 and instances[0].datasets[0].jsonl_fpath == str(rows)
    TrainDataProcessor().load_datasets(
        TrainDataProcessorConfig(
            output_dirpath=str(tmp_path / "prepared"),
            mode="train_preparation",
            should_download=False,
        ),
        instances,
    )
