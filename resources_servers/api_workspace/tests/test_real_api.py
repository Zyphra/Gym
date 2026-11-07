# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V5 real-API conventions: full-corpus gold/no-op/wrong controls, dialect contracts and V4 isolation."""

from __future__ import annotations

import base64
import hashlib
import json

import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import ServerClient
from resources_servers.api_workspace import realapi
from resources_servers.api_workspace.app import APIWorkspaceConfig, APIWorkspaceResourcesServer
from resources_servers.api_workspace.engine import Workspace
from resources_servers.api_workspace.realapi import APPS, REAL_VENDORS, parse_a1
from resources_servers.api_workspace.scenarios import (
    FAMILIES_V5,
    REVISION_V5,
    SYSTEM_PROMPT_V5,
    row_from_id,
    scenario_from_id,
    task_ids,
)
from resources_servers.api_workspace.tests.real_api_gold import (
    gold_plan,
    guard_breach,
    raw_message,
    response_ok,
    wrong_value,
)


def _server() -> APIWorkspaceResourcesServer:
    client = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=9999), global_config_dict=OmegaConf.create({})
    )
    return APIWorkspaceResourcesServer(
        config=APIWorkspaceConfig(host="127.0.0.1", port=9998, entrypoint="app.py", name="api_workspace_resources"),
        server_client=client,
    )


def _apply(world: Workspace, plan: list) -> list[bool]:
    """Execute a plan in process with the same handlers the HTTP tools call."""
    results = []
    for endpoint, arguments in plan:
        if endpoint == "/workspace_codec":
            status, payload = realapi.codec(arguments["operation"], arguments["text"])
            results.append(status == 200)
            continue
        status, body = realapi.fetch(world, **arguments)
        results.append(response_ok(endpoint, status, {"body": body}))
    return results


def _connected(scenario, app: str) -> realapi.RealVendor:
    return next(REAL_VENDORS[name] for name in scenario.connected if REAL_VENDORS[name].app == app)


@pytest.mark.parametrize("family", FAMILIES_V5)
def test_every_v5_task_gold_noop_missing_guard_and_wrong_value(family):
    fingerprints = set()
    for split in ("train", "val"):
        for identity in task_ids(split, REVISION_V5):
            if f":{family}:" not in identity:
                continue
            scenario = scenario_from_id(identity)
            fingerprints.add(hashlib.sha256(json.dumps(scenario.records, sort_keys=True).encode()).hexdigest())
            plan = gold_plan(scenario)
            noop = Workspace(scenario).score()
            assert noop["reward"] == 0 and noop["assertions"]["partial"] == 0, identity
            gold = Workspace(scenario)
            assert all(_apply(gold, plan)), identity
            score = gold.score()
            assert score["reward"] == 1 and score["assertions"]["partial"] == 1, (identity, score)
            for broken, guarded in (
                (plan[:-1], False),
                (plan + [guard_breach(scenario)], True),
                (wrong_value(plan), False),
            ):
                world = Workspace(scenario)
                _apply(world, broken)
                result = world.score()
                assert result["reward"] == 0 and result["assertions"]["partial"] < 1, (identity, result)
                # A missing or wrong write fails only its assertion; the extra action breaks a guard.
                assert bool(result["assertions"]["guards_broken"]) is guarded, (identity, result)
            assert json.dumps(scenario.records, sort_keys=True) == json.dumps(
                scenario_from_id(identity).records, sort_keys=True
            )
    assert len(fingerprints) == 88


def test_v5_corpus_rows_tools_and_split():
    train, val = task_ids("train", REVISION_V5), task_ids("val", REVISION_V5)
    assert len(train) == 960 and len(val) == 96 and not set(train) & set(val)
    row = row_from_id(train[0])
    assert row["responses_create_params"]["input"][0]["content"] == SYSTEM_PROMPT_V5
    assert [tool["name"] for tool in row["responses_create_params"]["tools"]] == [
        "workspace_search",
        "workspace_fetch",
        "workspace_codec",
    ]
    assert row["metadata"]["revision"] == REVISION_V5
    assert not {"records", "updates", "creates", "deletes", "guards", "context"} & row.keys()
    for bad in (f"{REVISION_V5}:train:rx_invoice_intake:0080", f"{REVISION_V5}:train:hr_onboarding:0000"):
        with pytest.raises(ValueError):
            scenario_from_id(bad)


def test_v5_records_carry_exactly_their_schema_and_goals_change_visible_state():
    for identity in task_ids("train", REVISION_V5)[:120] + task_ids("val", REVISION_V5):
        scenario = scenario_from_id(identity)
        for vendor, collections in scenario.records.items():
            schema = APPS[REAL_VENDORS[vendor].app]
            for collection, rows in collections.items():
                for key, row in rows.items():
                    assert row["id"] == key and row.keys() == {"id", *schema[collection]}, (identity, collection)
                    assert all(type(row[name]) is kind for name, kind in schema[collection].items()), identity
        for goal in scenario.updates:
            row = scenario.records[goal["vendor"]][goal["collection"]][goal["id"]]
            assert any(row[key] != value for key, value in goal["fields"].items()), (identity, goal)
        assert scenario.updates or scenario.creates


def test_http_tools_gold_verify_and_revision_isolation():
    native = _server()
    with TestClient(native.setup_webserver()) as client:
        identity = f"{REVISION_V5}:train:rx_refund_review:0003"
        scenario = scenario_from_id(identity)
        assert client.post("/seed_session", json={"task_id": identity}).status_code == 200
        refused = client.post("/workspace_request", json={"vendor": "gw_postquill", "method": "GET", "path": "/v1/x"})
        assert refused.status_code == 400 and refused.json()["error"]["code"] == "tool_not_in_revision"
        for endpoint, arguments in gold_plan(scenario):
            response = client.post(endpoint, json=arguments)
            assert response_ok(endpoint, response.status_code, response.json()), response.text
        world = next(iter(native._sessions.values())).world
        assert world.score()["reward"] == 1
        row = row_from_id(identity)
        empty = {"id": "r", "created_at": 0, "model": "m", "object": "response", "output": []}
        empty |= {"parallel_tool_calls": True, "tool_choice": "auto", "tools": []}
        verified = client.post(
            "/verify",
            json={"task_id": identity, "responses_create_params": row["responses_create_params"], "response": empty},
        ).json()
        assert verified["reward"] == 1
        assert verified["task_evidence"]["api_workspace_grading"] == {
            "partial": 1.0,
            "assertions_passed": len(scenario.updates) + len(scenario.creates),
            "assertions_total": len(scenario.updates) + len(scenario.creates),
            "guards_broken": 0,
        }
        v4 = "api-workspace-v4:train:hr_location:0000"
        assert client.post("/seed_session", json={"task_id": v4}).status_code == 200
        refused = client.post("/workspace_search", json={"query": "mail"})
        assert refused.status_code == 400


def test_disconnected_vendor_is_401_and_containers_must_be_json_strings():
    scenario = scenario_from_id(f"{REVISION_V5}:train:rx_signup_import:0000")
    world = Workspace(scenario)
    mail = _connected(scenario, "mail")
    other = next(REAL_VENDORS[name] for name in realapi.APP_VENDORS["mail"] if name != mail.name)
    status, body = realapi.fetch(world, "GET", f"{other.base}/mail/v1/users/me/messages")
    assert status == 401 and body["error"]["status"] == "UNAUTHENTICATED"
    url = f"{mail.base}/mail/v1/users/me/messages"
    assert realapi.fetch(world, "GET", url, params={"q": "x"})[0] == 400
    assert realapi.fetch(world, "GET", url, params="{not json")[0] == 400
    assert realapi.fetch(world, "GET", url, params="[1]")[0] == 400
    assert realapi.fetch(world, "GET", url + "?q=signup", params='{"q": "x"}')[0] == 400
    status, body = realapi.fetch(world, "GET", url + "?maxResults=2", params='{"q": "is:unread"}')
    assert status == 200 and len(body["messages"]) == 2 and "nextPageToken" in body
    second = realapi.fetch(
        world, "GET", url, params=json.dumps({"q": "is:unread", "maxResults": 2, "pageToken": body["nextPageToken"]})
    )
    assert second[0] == 200 and not {m["id"] for m in second[1]["messages"]} & {m["id"] for m in body["messages"]}
    assert realapi.fetch(world, "GET", url, params=json.dumps({"pageToken": body["nextPageToken"]}))[0] == 400
    assert realapi.fetch(world, "GET", url, body="{}")[0] == 400


def test_mail_send_requires_encoded_rfc2822_and_parses_headers():
    scenario = scenario_from_id(f"{REVISION_V5}:train:rx_budget_alerts:0000")
    world = Workspace(scenario)
    url = f"{_connected(scenario, 'mail').base}/mail/v1/users/me/messages/send"
    before = json.dumps(world.records, sort_keys=True)
    for body in (
        '{"to": "a@b.invalid", "subject": "s", "body": "t"}',
        json.dumps({"raw": "%%%not-base64%%%"}),
        json.dumps({"raw": realapi.b64url_encode("just text, no headers")}),
        json.dumps(realapi.b64url_encode(raw_message(["a@b.invalid"], "s", "t"))),
        json.dumps({"raw": realapi.b64url_encode(raw_message(["a@b.invalid"], "s", "t")), "threadId": "missing"}),
    ):
        assert realapi.fetch(world, "POST", url, body=body)[0] in (400, 404), body
    assert json.dumps(world.records, sort_keys=True) == before
    multipart = (
        'To: "Ops" <Ops@Example.invalid>, b@c.invalid\r\nCc: d@e.invalid\r\nSubject: Hi there\r\nMIME-Version: 1.0\r\n'
        'Content-Type: multipart/alternative; boundary="x"\r\n\r\n--x\r\nContent-Type: text/plain; charset=utf-8\r\n'
        "Content-Transfer-Encoding: base64\r\n\r\n"
        + base64.b64encode("Line one\nLine two\n".encode()).decode()
        + "\r\n--x\r\nContent-Type: text/html\r\n\r\n<p>Line one</p>\r\n--x--\r\n"
    )
    status, body = realapi.fetch(
        world, "POST", url, body=json.dumps({"raw": base64.b64encode(multipart.encode()).decode()})
    )
    assert status == 200
    sent = world.records[_connected(scenario, "mail").name]["messages"][body["id"]]
    assert sent["to"] == ["b@c.invalid", "ops@example.invalid"] and sent["cc"] == ["d@e.invalid"]
    assert sent["subject"] == "Hi there" and sent["body"] == "Line one\nLine two\n" and sent["labels"] == ["SENT"]


MALFORMED_MIME = {
    "no empty line before the body": (
        "To: a@b.invalid\r\nSubject: Re: X\r\nRefund approved for WO-1.",
        "no empty line between its headers and its body",
    ),
    "body text in the header block": (
        "To: a@b.invalid\nSubject: Re: X\nBody: Refund approved for WO-1.",
        "'Body' is not a mail header",
    ),
    "typed escapes for every line break": (
        "To: a@b.invalid\\r\\nSubject: Re: X\\r\\n\\r\\nRefund approved for WO-1.",
        "To header holds typed backslash-r text",
    ),
    "typed escapes after the subject": (
        "To: a@b.invalid\r\nSubject: Re: X\\n\\nRefund approved for WO-1.",
        "Subject header holds typed backslash-n text",
    ),
    "typed escapes in the body": (
        "To: a@b.invalid\r\nSubject: Re: X\r\n\r\nLine one\\nLine two",
        "body holds typed backslash-n",
    ),
}


@pytest.mark.parametrize("case", sorted(MALFORMED_MIME))
def test_mail_send_refuses_a_malformed_mime_layout_and_changes_nothing(case):
    raw, reason = MALFORMED_MIME[case]
    scenario = scenario_from_id(f"{REVISION_V5}:train:rx_refund_review:0000")
    world = Workspace(scenario)
    url = f"{_connected(scenario, 'mail').base}/mail/v1/users/me/messages/send"
    before = json.dumps(world.records, sort_keys=True)
    status, body = realapi.fetch(world, "POST", url, body=json.dumps({"raw": realapi.b64url_encode(raw)}))
    assert status == 400 and body["error"]["status"] == "INVALID_ARGUMENT"
    assert reason in body["error"]["message"] and "then one empty line, then the body" in body["error"]["message"]
    assert json.dumps(world.records, sort_keys=True) == before


@pytest.mark.parametrize(
    "raw, stored",
    [
        ("To: a@b.invalid\nSubject: Re: X\n\nRefund approved for WO-1.", "Refund approved for WO-1."),
        ("To: a@b.invalid\r\nSubject: Re: X\r\nX-Mailer: m\r\n\r\nSee C:\\new\r\nok", "See C:\\new\r\nok"),
        ("To: a@b.invalid\r\nSubject: Re: X\r\n\r\nWO-1: approved", "WO-1: approved"),
    ],
)
def test_mail_send_accepts_well_formed_layouts(raw, stored):
    scenario = scenario_from_id(f"{REVISION_V5}:train:rx_refund_review:0000")
    world = Workspace(scenario)
    vendor = _connected(scenario, "mail")
    url = f"{vendor.base}/mail/v1/users/me/messages/send"
    status, body = realapi.fetch(world, "POST", url, body=json.dumps({"raw": realapi.b64url_encode(raw)}))
    assert status == 200, body
    assert world.records[vendor.name]["messages"][body["id"]]["body"] == stored


@pytest.mark.parametrize(
    "app, method, path, params, body",
    [
        ("mail", "GET", "/mail/v1/users/me/messages", {"pageToken": True}, None),
        ("mail", "GET", "/mail/v1/users/me/messages/{message}", {"format": []}, None),
        ("sheets", "GET", "/v4/spreadsheets/{sheet}/values/Teams!A1:B2", {"majorDimension": ["ROWS"]}, None),
        (
            "sheets",
            "POST",
            "/v4/spreadsheets/{sheet}/values:batchUpdate",
            None,
            {"valueInputOption": ["RAW"], "data": []},
        ),
        (
            "sheets",
            "POST",
            "/v4/spreadsheets/{sheet}/values/Teams!A:E:append",
            {"valueInputOption": "RAW", "insertDataOption": {}},
            {"values": [["x"]]},
        ),
        ("chat", "GET", "/api/users.info", {"user": ["U1"]}, None),
    ],
)
def test_json_values_of_the_wrong_type_are_api_errors(app, method, path, params, body):
    """A policy's wrongly typed JSON value gets the vendor's own error reply, never a server exception."""
    scenario = scenario_from_id(f"{REVISION_V5}:train:rx_budget_alerts:0001")
    world = Workspace(scenario)
    vendor = _connected(scenario, app)
    store = world.records[vendor.name]
    ids = {
        name: next(iter(store.get(key) or [""])) for name, key in (("message", "messages"), ("sheet", "spreadsheets"))
    }
    url = vendor.base + path.format(**ids)
    status, reply = realapi.fetch(
        world, method, url, json.dumps(params) if params else None, json.dumps(body) if body else None
    )
    assert (status, reply.get("ok")) == (200, False) if app == "chat" else status == 400


def test_crm_search_rejects_unhashable_filters_and_compares_infinite_values():
    for identity in task_ids("train", REVISION_V5)[:24]:
        scenario = scenario_from_id(identity)
        crm = _connected(scenario, "crm")
        if crm.dialect == "rest":
            break
    world = Workspace(scenario)
    url = f"{crm.base}/crm/v3/objects/deals/search"
    for item in ({"propertyName": ["amount"], "operator": "EQ"}, {"propertyName": "amount", "operator": ["EQ"]}):
        assert realapi.fetch(world, "POST", url, body=json.dumps({"filterGroups": [{"filters": [item]}]}))[0] == 400
    infinite = {"filterGroups": [{"filters": [{"propertyName": "amount", "operator": "GT", "value": float("inf")}]}]}
    status, reply = realapi.fetch(world, "POST", url, body=json.dumps(infinite))
    assert status == 200 and reply["results"] == []


def test_a1_ranges_values_envelopes_and_value_input_option():
    titles = ["Intake Log", "Rules"]
    assert parse_a1("'Intake Log'!B2:D9", titles) == realapi.A1Range("Intake Log", 2, 2, 9, 4, False)
    assert parse_a1("Rules!A:C", titles) == realapi.A1Range("Rules", None, 1, None, 3, False)
    assert parse_a1("intake log!3:5", titles) == realapi.A1Range("Intake Log", 3, None, 5, None, False)
    assert parse_a1("A2:C", titles) == realapi.A1Range("Intake Log", 2, 1, None, 3, False)
    assert parse_a1("Rules", titles) == realapi.A1Range("Rules", None, None, None, None, False)
    assert parse_a1("Rules!C7", titles).single
    for bad in ("Missing!A1", "Rules!A", "Rules!C1:A1", "Rules!1A", "Rules!A1:B2:C3", "Rules!AAA1"):
        with pytest.raises(realapi._Fail):
            parse_a1(bad, titles)
    scenario = scenario_from_id(f"{REVISION_V5}:train:rx_budget_alerts:0001")
    world = Workspace(scenario)
    sheets = _connected(scenario, "sheets")
    tab = next(row for row in world.records[sheets.name]["worksheets"].values() if row["title"] == "Teams")
    base = f"{sheets.base}/v4/spreadsheets/{tab['spreadsheet']}/values"
    assert realapi.fetch(world, "PUT", f"{base}/Teams!F1", body='{"values": [["x"]]}')[0] == 400
    assert realapi.fetch(world, "PUT", f"{base}/Teams!F1", '{"valueInputOption": "RAW"}', '[["x"]]')[0] == 400
    assert (
        realapi.fetch(world, "PUT", f"{base}/Teams!F1:F1", '{"valueInputOption": "RAW"}', '{"values": [["x", "y"]]}')[
            0
        ]
        == 400
    )
    status, body = realapi.fetch(
        world, "PUT", f"{base}/Teams!F2", '{"valueInputOption": "RAW"}', '{"values": [[7, true]]}'
    )
    assert status == 200 and body["updatedRange"] == "Teams!F2:G2"
    status, body = realapi.fetch(world, "GET", f"{base}/Teams!F2:G2")
    assert body["values"] == [["7", "TRUE"]]
    status, body = realapi.fetch(
        world, "POST", f"{base}/Teams!A:E:append", '{"valueInputOption": "RAW"}', '{"values": [["Z"]]}'
    )
    assert status == 200 and body["updates"]["updatedRange"] == f"Teams!A{len(tab['values'])}:A{len(tab['values'])}"
    assert realapi.fetch(world, "POST", f"{base}:batchUpdate", '{"valueInputOption": "RAW"}', '{"data": []}')[0] == 400


def test_chat_methods_take_ids_in_json_bodies():
    scenario = scenario_from_id(f"{REVISION_V5}:train:rx_offboarding:0002")
    world = Workspace(scenario)
    chat = _connected(scenario, "chat")
    channel_id, channel = next(iter(world.records[chat.name]["channels"].items()))
    post = f"{chat.base}/api/chat.postMessage"
    assert (
        realapi.fetch(world, "POST", post, params=json.dumps({"channel": channel_id, "text": "x"}))[1]["ok"] is False
    )
    named = realapi.fetch(world, "POST", post, body=json.dumps({"channel": "#" + channel["name"], "text": "x"}))[1]
    assert named == {"ok": False, "error": "channel_not_found", "detail": named["detail"]} and "ID" in named["detail"]
    user = next(iter(world.records[chat.name]["users"].values()))
    invite = f"{chat.base}/api/conversations.invite"
    assert (
        realapi.fetch(world, "POST", invite, body=json.dumps({"channel": channel_id, "users": user["email"]}))[1][
            "error"
        ]
        == "user_not_found"
    )
    status, body = realapi.fetch(
        world, "GET", f"{chat.base}/api/users.lookupByEmail", params=json.dumps({"email": user["email"]})
    )
    assert body["user"]["id"] == user["id"]
    assert realapi.fetch(world, "GET", f"{chat.base}/api/chat.postMessage")[1]["error"] == "invalid_request"
    archived = next(key for key, row in world.records[chat.name]["channels"].items() if row["archived"])
    member = world.records[chat.name]["channels"][archived]["members"][0]
    kick = json.dumps({"channel": archived, "user": member})
    assert realapi.fetch(world, "POST", f"{chat.base}/api/conversations.kick", body=kick)[1]["error"] == "is_archived"


def test_crm_rest_properties_envelope_and_graphql_dialect():
    for identity in task_ids("train", REVISION_V5)[:24]:
        scenario = scenario_from_id(identity)
        crm = _connected(scenario, "crm")
        world = Workspace(scenario)
        deals = world.records[crm.name]["deals"]
        if not deals:
            continue
        deal_id = next(iter(deals))
        if crm.dialect == "rest":
            url = f"{crm.base}/crm/v3/objects/deals/{deal_id}"
            assert realapi.fetch(world, "PATCH", url, body='{"stage": "x"}')[0] == 400
            assert realapi.fetch(world, "PATCH", url, body='{"properties": {"amount": "250"}}')[0] == 200
            assert deals[deal_id]["amount"] == 250
            search = {"filterGroups": [{"filters": [{"propertyName": "amount", "operator": "EQ", "value": 250}]}]}
            found = realapi.fetch(world, "POST", f"{crm.base}/crm/v3/objects/deals/search", body=json.dumps(search))
            assert [x["id"] for x in found[1]["results"]] == [deal_id]
        else:
            url = f"{crm.base}/graphql"
            query = "query($f: DealFilter) { all: deals(first: 1, filter: $f) { nodes { id closeDate } pageInfo { hasNextPage endCursor } totalCount } }"
            status, body = realapi.fetch(world, "POST", url, body=json.dumps({"query": query, "variables": {"f": {}}}))
            assert status == 200 and body["data"]["all"]["totalCount"] == len(deals)
            assert body["data"]["all"]["pageInfo"]["hasNextPage"] == (len(deals) > 1)
            following = body["data"]["all"]["pageInfo"]["endCursor"]
            status, body = realapi.fetch(
                world,
                "POST",
                url,
                body=json.dumps(
                    {"query": query.replace("first: 1", f'first: 1, after: "{following}"'), "variables": {"f": {}}}
                ),
            )
            assert body["data"]["all"]["nodes"][0]["id"] != deal_id
            assert "errors" in realapi.fetch(world, "POST", url, body='{"query": "{ deals { nodes { nope } } }"}')[1]
            assert realapi.fetch(world, "POST", url, body='{"query": "{ deals { "}')[0] == 400
            mutation = 'mutation { updateDeal(id: "%s", input: {amount: 250}) { id amount } }' % deal_id
            assert realapi.fetch(world, "POST", url, body=json.dumps({"query": mutation}))[1]["data"][
                "updateDeal"
            ] == {
                "id": deal_id,
                "amount": 250,
            }
            assert deals[deal_id]["amount"] == 250


def test_catalog_search_ranks_endpoints_and_reports_connections():
    scenario = scenario_from_id(f"{REVISION_V5}:train:rx_label_holds:0000")
    world = Workspace(scenario)
    results = realapi.search(world, "append rows spreadsheet values", top_k=4)["results"]
    assert results and all(entry["app"] == "sheets" for entry in results[:2])
    assert {entry["id"].split(".", 1)[1] for entry in results[:2]} == {"values.append"}
    connected = realapi.search(world, "send email", top_k=20, connected_only=True)["results"]
    assert connected and all(entry["connected"] for entry in connected)
    assert all("records" not in json.dumps(entry).lower() or entry["app"] for entry in connected)
