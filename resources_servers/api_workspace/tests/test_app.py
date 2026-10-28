# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real native HTTP controls plus full-corpus gold/no-op and mutation checks."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import ServerClient
from resources_servers.api_workspace.app import APIWorkspaceConfig, APIWorkspaceResourcesServer
from resources_servers.api_workspace.engine import VENDORS, APIError, Workspace, canonical_json
from resources_servers.api_workspace.scenarios import FAMILIES, row_from_id, scenario_from_id, task_ids


def server():
    client = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=9999), global_config_dict=OmegaConf.create({})
    )
    return APIWorkspaceResourcesServer(
        config=APIWorkspaceConfig(host="127.0.0.1", port=9998, entrypoint="app.py", name="api_workspace_resources"),
        server_client=client,
    )


def gold_calls(scenario):
    calls = []
    for method, goals in (("PATCH", scenario.updates), ("DELETE", scenario.deletes), ("POST", scenario.creates)):
        for i, goal in enumerate(goals):
            vendor = VENDORS[goal["vendor"]]
            calls.append(
                {
                    "vendor": vendor.name,
                    "method": method,
                    "path": vendor.path(goal["collection"], goal.get("id")),
                    "body": vendor.wire(goal.get("fields", {})),
                    "idempotency_key": f"gold-{i}",
                }
            )
    return calls


@pytest.fixture
def seeded():
    native = server()
    with TestClient(native.setup_webserver()) as client:
        identity = "api-workspace-v3:train:acct_reconcile:0000"
        assert client.post("/seed_session", json={"task_id": identity}).status_code == 200
        yield native, client, scenario_from_id(identity)


@pytest.mark.parametrize("family", FAMILIES)
def test_all_immutable_tasks_gold_noop_wrong_and_preservation(family):
    fingerprints = set()
    for identity in task_ids("train"):
        if f":{family}:" not in identity:
            continue
        scenario = scenario_from_id(identity)
        world = Workspace(scenario)
        assert world.score()["reward"] == 0.0, identity
        before = canonical_json(world.records)
        fingerprints.add(hashlib.sha256(before.encode()).hexdigest())
        for call in reversed(gold_calls(scenario)):
            result = world.request(**call)
            assert result["status"] in (200, 201), identity
        assert world.score()["reward"] == 1.0, (identity, world.score())
        # Unrelated-state tampering must invalidate a completed task.
        vendor = scenario.connected[0]
        collection = next(iter(world.records[vendor]))
        if not world.records[vendor][collection]:
            collection = next(k for k, v in world.records[vendor].items() if v)
        record = next(iter(world.records[vendor][collection].values()))
        record["injected_authority"] = "success"
        assert world.score()["reward"] == 0.0, identity
        assert canonical_json(scenario.records) == before
    assert len(fingerprints) == 80


@pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
def test_created_text_files_accept_line_endings_without_mutating_state(line_ending):
    checked = 0
    for family in FAMILIES:
        scenario = scenario_from_id(f"api-workspace-v3:train:{family}:0000")
        goals = [goal for goal in scenario.creates if goal["collection"] == "files"]
        if not goals:
            continue
        world = Workspace(scenario)
        for call in gold_calls(scenario):
            world.request(**call)
        for goal in goals:
            originals = scenario.records[goal["vendor"]]["files"]
            created = next(row for key, row in world.records[goal["vendor"]]["files"].items() if key not in originals)
            created["content"] = created["content"].replace("\n", line_ending) + line_ending
        before_score = canonical_json(world.records)
        assert world.score()["reward"] == 1.0, family
        assert canonical_json(world.records) == before_score
        checked += 1
    assert checked > 0


@pytest.mark.parametrize("suffix", ["\n\n", " ", "\nextra"])
def test_created_text_files_reject_changed_content_and_extra_blank_lines(suffix):
    scenario = scenario_from_id("api-workspace-v3:train:hr_onboarding:0000")
    world = Workspace(scenario)
    for call in gold_calls(scenario):
        world.request(**call)
    goal = next(goal for goal in scenario.creates if goal["collection"] == "files")
    originals = scenario.records[goal["vendor"]]["files"]
    created = next(row for key, row in world.records[goal["vendor"]]["files"].items() if key not in originals)
    created["content"] += suffix
    assert world.score()["reward"] == 0.0


def test_created_text_file_equivalence_does_not_relax_preserved_records():
    scenario = scenario_from_id("api-workspace-v3:train:hr_onboarding:0000")
    world = Workspace(scenario)
    for call in gold_calls(scenario):
        world.request(**call)
    assert world.score()["reward"] == 1.0
    goal = next(goal for goal in scenario.creates if goal["collection"] == "files")
    preserved_id = next(iter(scenario.records[goal["vendor"]]["files"]))
    world.records[goal["vendor"]]["files"][preserved_id]["content"] += "\n"
    assert world.score()["reward"] == 0.0


def test_discovery_real_401_and_recovery(seeded):
    _, client, scenario = seeded
    accounts = client.post("/workspace_accounts", json={"function": "accounting"}).json()["accounts"]
    assert len(accounts) == 3 and sum(x["connected"] for x in accounts) == 2
    disconnected = next(x["vendor"] for x in accounts if not x["connected"])
    schema = client.post("/workspace_schema", json={"vendor": disconnected}).json()
    denied = client.post(
        "/workspace_request", json={"vendor": disconnected, "method": "GET", "path": schema["collections"][0]["path"]}
    )
    assert denied.status_code == 401
    assert denied.headers["www-authenticate"].startswith("Bearer")
    assert denied.json()["error"]["code"] == "account_not_connected"
    connected = client.post("/workspace_accounts", json={"function": "accounting", "connected_only": True}).json()
    found = []
    for account in connected["accounts"]:
        name = account["vendor"]
        v = VENDORS[name]
        response = client.post(
            "/workspace_request",
            json={"vendor": name, "method": "GET", "path": v.path("invoices"), "params": {"limit": 100}},
        )
        assert response.status_code == 200
        found.extend(response.json()["items"])
    assert len(found) >= 4
    assert all(
        x["namespace"] == scenario.namespace
        for x in [client.post("/seed_session", json={"task_id": scenario.task_id}).json()]
    )


def test_query_url_wrong_path_and_parameter_errors_are_atomic(seeded):
    native, client, scenario = seeded
    vendor = next(VENDORS[name] for name in scenario.connected if VENDORS[name].function == "crm")
    before = canonical_json(next(iter(native._sessions.values())).world.records)
    for path, params, status in (
        (vendor.path("accounts") + "?q=Atelier", {}, 400),
        ("/incorrect/accounts", {}, 404),
        (vendor.path("accounts"), {"limit": True}, 400),
        (vendor.path("accounts"), {"unknown": "x"}, 400),
        (vendor.path("accounts"), {"filters": {"amount": "x"}}, 400),
    ):
        response = client.post(
            "/workspace_request", json={"vendor": vendor.name, "method": "GET", "path": path, "params": params}
        )
        assert response.status_code == status, response.text
    assert canonical_json(next(iter(native._sessions.values())).world.records) == before


def test_pagination_filter_cursor_scope_and_schema_dialects(seeded):
    _, client, scenario = seeded
    vendor = next(VENDORS[name] for name in scenario.connected if VENDORS[name].function == "hr")
    params = {"limit": 1}
    collected = []
    while True:
        response = client.post(
            "/workspace_request",
            json={"vendor": vendor.name, "method": "GET", "path": vendor.path("employees"), "params": params},
        )
        assert response.status_code == 200
        result = response.json()
        collected.extend(result["items"])
        cursor = result["next_cursor"]
        if not cursor:
            break
        params = {"limit": 1, "cursor": cursor}
    expected = scenario.records[vendor.name]["employees"]
    assert {x[vendor.field("id")] for x in collected} == set(expected)
    bad = client.post(
        "/workspace_request",
        json={
            "vendor": vendor.name,
            "method": "GET",
            "path": vendor.path("employees"),
            "params": {"q": "different", "cursor": params["cursor"]},
        },
    )
    assert bad.status_code == 400
    identity = next(iter(expected))
    result = client.post(
        "/workspace_request",
        json={
            "vendor": vendor.name,
            "method": "GET",
            "path": vendor.path("employees"),
            "params": {"filters": {vendor.field("id"): identity}},
        },
    ).json()
    assert result["items"] == [vendor.wire(expected[identity])]


def test_parallel_idempotency_order_and_session_isolation(seeded):
    native, client, scenario = seeded
    creation = next(x for x in gold_calls(scenario) if x["method"] == "POST")
    result = client.post("/workspace_parallel", json={"requests": [creation, creation]}).json()
    assert [x["status"] for x in result["results"]] == [201, 201]
    assert result["results"][0] == result["results"][1]
    changed = deepcopy(creation)
    changed["body"]["content"] = "conflicting body"
    assert client.post("/workspace_request", json=changed).status_code == 409
    app = native.setup_webserver()
    with TestClient(app) as other:
        assert other.post("/seed_session", json={"task_id": scenario.task_id}).status_code == 200
        assert len(native._sessions) == 2
        second_world = list(native._sessions.values())[-1].world
        assert second_world.records == scenario.records
        for call in gold_calls(scenario):
            assert client.post("/workspace_request", json=call).status_code in (200, 201)
        worlds = list(native._sessions.values())
        assert worlds[0].world.score()["reward"] == 1
        assert worlds[1].world.score()["reward"] == 0


def test_concurrent_independent_clients_reset_and_close():
    native = server()
    app = native.setup_webserver()
    identity = "api-workspace-v3:train:storage_index:0000"
    scenario = scenario_from_id(identity)

    def run(index):
        with TestClient(app) as client:
            assert client.post("/seed_session", json={"task_id": identity}).status_code == 200
            for call in gold_calls(scenario):
                assert client.post("/workspace_request", json=call).status_code in (200, 201)
            reset = client.post("/seed_session", json={"task_id": identity})
            assert reset.status_code == 200
            assert client.post("/close", json={}).json() == {"success": True}
        return index

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sorted(pool.map(run, range(16))) == list(range(16))
    assert not native._sessions


def test_corpus_split_prompt_authority_and_noncanonical_ids():
    assert len(task_ids("train")) == 2400
    assert len(task_ids("val")) == 240
    assert not set(task_ids("train")) & set(task_ids("val"))
    row = row_from_id(task_ids("train")[0])
    assert "50 model turns" in row["responses_create_params"]["input"][0]["content"]
    assert not {"records", "updates", "creates", "deletes", "goals"} & row.keys()
    for identity in (
        "api-workspace-v3:train:hr_onboarding:0",
        "api-workspace-v3:train:hr_onboarding:0080",
        "api-workspace-v3:eval:hr_onboarding:0000",
        "api-workspace-v3:train:invented:0000",
    ):
        with pytest.raises(ValueError):
            scenario_from_id(identity)


def test_api_write_validation_and_grader_ignores_claims(seeded):
    native, client, scenario = seeded
    world = next(iter(native._sessions.values())).world
    goal = scenario.updates[0]
    v = VENDORS[goal["vendor"]]
    before = canonical_json(world.records)
    for body in ({v.field("id"): "replacement"}, {"amount": -1}, {"amount": True}, {"goals": []}):
        result = client.post(
            "/workspace_request",
            json={"vendor": v.name, "method": "PATCH", "path": v.path(goal["collection"], goal["id"]), "body": body},
        )
        assert result.status_code == 422
    assert canonical_json(world.records) == before
    assert world.score()["reward"] == 0
    for call in gold_calls(scenario):
        world.request(**call)
    assert world.score()["reward"] == 1
    with pytest.raises(APIError):
        world.request(vendor="invented_vendor", method="GET", path="/v1/invoices")


def test_catalog_metadata_pagination_and_encoded_request_recovery(seeded):
    import json

    native, client, scenario = seeded
    page = client.post(
        "/workspace_catalog", json={"query": "employees location", "method": "PATCH", "limit": 1}
    ).json()
    assert page["total"] == 3 and page["next_offset"] == 1
    assert page["catalog_only"] is True
    second = client.post(
        "/workspace_catalog", json={"query": "employees location", "method": "PATCH", "limit": 1, "offset": 1}
    ).json()
    assert second["endpoints"][0]["vendor"] != page["endpoints"][0]["vendor"]
    entries = client.post("/workspace_catalog", json={"query": "employees", "method": "GET"}).json()["endpoints"]
    assert len(entries) == 3
    assert scenario.namespace not in json.dumps(entries)
    assert client.post("/workspace_catalog", json={"query": scenario.namespace}).json()["endpoints"] == []
    unavailable = next(entry for entry in entries if not entry["connected"])
    assert (
        client.post(
            "/workspace_request", json={"vendor": unavailable["vendor"], "method": "GET", "path": unavailable["path"]}
        ).status_code
        == 401
    )
    available = next(entry for entry in entries if entry["connected"])
    base = {"vendor": available["vendor"], "method": "GET", "path": available["path"]}
    before = canonical_json(next(iter(native._sessions.values())).world.records)
    for invalid in ("None", "[]", '"text"', "true", "{not-json}"):
        rejected = client.post("/workspace_request", json=base | {"params": invalid})
        assert rejected.status_code == 422, rejected.text
    assert canonical_json(next(iter(native._sessions.values())).world.records) == before
    read = client.post("/workspace_request", json=base | {"params": json.dumps({"limit": 100}), "body": "null"})
    assert read.status_code == 200 and read.json()["items"]
    vendor = VENDORS[available["vendor"]]
    observed_id = read.json()["items"][0][vendor.field("id")]
    body = vendor.wire({"location": "Temporary Office"})
    changed = client.post(
        "/workspace_request",
        json=base
        | {"method": "PATCH", "path": vendor.path("employees", observed_id), "params": None, "body": json.dumps(body)},
    )
    assert changed.status_code == 200
    reread = client.post("/workspace_request", json=base | {"path": vendor.path("employees", observed_id)})
    assert reread.json()["item"][vendor.field("location")] == "Temporary Office"


def test_catalog_validates_selectors_and_can_filter_connections(seeded):
    _, client, _ = seeded
    data = client.post("/workspace_catalog", json={"function": "hr", "connected_only": True, "method": "GET"}).json()
    assert data["total"] == 4 and all(item["connected"] for item in data["endpoints"])
    for payload, expected in (
        ({"function": "unknown"}, 400),
        ({"method": "HEAD"}, 400),
        ({"limit": 0}, 422),
        ({"offset": -1}, 422),
    ):
        assert client.post("/workspace_catalog", json=payload).status_code == expected


@pytest.mark.parametrize("family", FAMILIES)
def test_independent_sql_reference_rederives_every_gold(family):
    from resources_servers.api_workspace.tests.independent_reference import derive_calls

    for identity in task_ids("train"):
        if f":{family}:" not in identity:
            continue
        scenario = scenario_from_id(identity)
        calls = derive_calls(scenario.prompt, scenario.records, family)
        assert calls, identity
        world = Workspace(scenario)
        for call in calls:
            world.request(**call)
        assert world.score()["reward"] == 1, (identity, world.score())
        # A correct-looking report without these persisted operations earns zero.
        assert Workspace(scenario).score()["reward"] == 0
        vendor = next(
            VENDORS[name]
            for name in scenario.connected
            if VENDORS[name].function == "hr" and scenario.records[name]["employees"]
        )
        identity = next(iter(scenario.records[vendor.name]["employees"]))
        world.request(
            vendor=vendor.name,
            method="PATCH",
            path=vendor.path("employees", identity),
            body=vendor.wire({"location": "Wrong reference result"}),
        )
        assert world.score()["reward"] == 0
