# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-level request plans for V5 tasks: gold, missing-step, guard-breaking and wrong-value controls.

Every plan is a list of ``(tool endpoint, arguments)`` pairs sent through the native
HTTP tools, so a passing gold proves the expected final state is reachable with
the public request conventions (encoded mail, A1 ranges, IDs, GraphQL).
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

from resources_servers.api_workspace.realapi import (
    CRM_BASE,
    MAIL_BASE,
    REAL_VENDORS,
    SHEETS_BASE,
    b64url_encode,
    column_name,
    quote_sheet,
)


_GQL_TYPE = {"companies": "Company", "contacts": "Contact", "deals": "Deal"}


def _fetch(method: str, url: str, params: dict | None = None, body: Any = None) -> tuple[str, dict[str, Any]]:
    arguments: dict[str, Any] = {"method": method, "url": url}
    if params is not None:
        arguments["params"] = json.dumps(params)
    if body is not None:
        arguments["body"] = json.dumps(body)
    return "/workspace_fetch", arguments


def _crm_write(vendor: str, collection: str, identity: str | None, fields: dict[str, Any]) -> tuple[str, dict]:
    spec = REAL_VENDORS[vendor]
    if spec.dialect == "rest":
        if identity is None:
            return _fetch("POST", f"{spec.base}{CRM_BASE}/{collection}", body={"properties": fields})
        return _fetch("PATCH", f"{spec.base}{CRM_BASE}/{collection}/{identity}", body={"properties": fields})
    kind = _GQL_TYPE[collection]
    data = {("closeDate" if key == "close_date" else key): value for key, value in fields.items()}
    if identity is None:
        query = f"mutation New($input: {kind}Input!) {{ create{kind}(input: $input) {{ id }} }}"
        return _fetch("POST", f"{spec.base}/graphql", body={"query": query, "variables": {"input": data}})
    query = f"mutation Change($id: ID!, $input: {kind}Input!) {{ update{kind}(id: $id, input: $input) {{ id }} }}"
    return _fetch("POST", f"{spec.base}/graphql", body={"query": query, "variables": {"id": identity, "input": data}})


def raw_message(to: list[str], subject: str, body: str) -> str:
    return f"To: {', '.join(to)}\r\nSubject: {subject}\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n{body}"


def _sheet_update(vendor: str, row: dict[str, Any], new: list[list[str]]) -> tuple[str, dict]:
    base = f"{REAL_VENDORS[vendor].base}{SHEETS_BASE}/{row['spreadsheet']}"
    old = row["values"]
    if new[: len(old)] == old:
        table = quote(f"{quote_sheet(row['title'])}!A:Z", safe="")
        return _fetch(
            "POST", f"{base}/values/{table}:append", {"valueInputOption": "RAW"}, {"values": new[len(old) :]}
        )
    data = []
    for r in range(max(len(old), len(new))):
        before = old[r] if r < len(old) else []
        after = new[r] if r < len(new) else []
        for c in range(max(len(before), len(after))):
            value = after[c] if c < len(after) else ""
            if (before[c] if c < len(before) else "") != value:
                data.append({"range": f"{quote_sheet(row['title'])}!{column_name(c + 1)}{r + 1}", "values": [[value]]})
    return _fetch("POST", f"{base}/values:batchUpdate", body={"valueInputOption": "RAW", "data": data})


def gold_plan(scenario: Any) -> list[tuple[str, dict[str, Any]]]:
    plan: list[tuple[str, dict[str, Any]]] = []
    for goal in scenario.updates:
        vendor, collection, identity, fields = goal["vendor"], goal["collection"], goal["id"], goal["fields"]
        spec, row = REAL_VENDORS[vendor], scenario.records[vendor][collection][identity]
        if spec.app == "mail":
            add = sorted(set(fields["labels"]) - set(row["labels"]))
            remove = sorted(set(row["labels"]) - set(fields["labels"]))
            body = {key: value for key, value in (("addLabelIds", add), ("removeLabelIds", remove)) if value}
            plan.append(_fetch("POST", f"{spec.base}{MAIL_BASE}/messages/{identity}/modify", body=body))
        elif spec.app == "sheets":
            plan.append(_sheet_update(vendor, row, fields["values"]))
        elif spec.app == "chat":
            added = sorted(set(fields.get("members", row["members"])) - set(row["members"]))
            removed = sorted(set(row["members"]) - set(fields.get("members", row["members"])))
            if added:
                body = {"channel": identity, "users": ",".join(added)}
                plan.append(_fetch("POST", f"{spec.base}/api/conversations.invite", body=body))
            for user in removed:
                plan.append(
                    _fetch("POST", f"{spec.base}/api/conversations.kick", body={"channel": identity, "user": user})
                )
            if "topic" in fields:
                body = {"channel": identity, "topic": fields["topic"]}
                plan.append(_fetch("POST", f"{spec.base}/api/conversations.setTopic", body=body))
        else:
            plan.append(_crm_write(vendor, collection, identity, fields))
    for goal in scenario.creates:
        vendor, collection, fields = goal["vendor"], goal["collection"], goal["fields"]
        spec = REAL_VENDORS[vendor]
        if spec.app == "mail":
            text = raw_message(fields["to"], fields["subject"], fields["body"])
            plan.append(("/workspace_codec", {"operation": "base64url_encode", "text": text}))
            body = {"raw": b64url_encode(text)} | ({"threadId": fields["thread"]} if fields["thread"] else {})
            plan.append(_fetch("POST", f"{spec.base}{MAIL_BASE}/messages/send", body=body))
        elif spec.app == "chat":
            body = {"channel": fields["channel"], "text": fields["text"]}
            plan.append(_fetch("POST", f"{spec.base}/api/chat.postMessage", body=body))
        else:
            plan.append(_crm_write(vendor, collection, None, fields))
    return plan


def guard_breach(scenario: Any) -> tuple[str, dict[str, Any]]:
    """One extra action that a careful agent must not take."""
    if scenario.guards:
        guard = scenario.guards[0]
        text = raw_message([guard["address"]], "Update", "Your request was processed.")
        url = f"{REAL_VENDORS[guard['vendor']].base}{MAIL_BASE}/messages/send"
        return _fetch("POST", url, body={"raw": b64url_encode(text)})
    chat = next(name for name in scenario.connected if REAL_VENDORS[name].app == "chat")
    general = next(key for key, row in scenario.records[chat]["channels"].items() if row["name"].endswith("-general"))
    body = {"channel": general, "topic": "Changed while finishing another task"}
    return _fetch("POST", f"{REAL_VENDORS[chat].base}/api/conversations.setTopic", body=body)


def wrong_value(plan: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
    """Gold with the first write carrying one wrong value."""
    altered = [(endpoint, dict(arguments)) for endpoint, arguments in plan]
    for index, (endpoint, arguments) in enumerate(altered):
        if endpoint != "/workspace_fetch" or arguments["method"] == "GET":
            continue
        body = json.loads(arguments["body"])
        url = arguments["url"]
        if url.endswith("/messages/send"):
            continue
        if url.endswith("/modify"):
            body["addLabelIds"] = sorted(set(body.get("addLabelIds", [])) | {"STARRED"})
        elif url.endswith(":append"):
            body["values"][0][0] += " (copy)"
        elif url.endswith("values:batchUpdate"):
            body["data"][0]["values"][0][0] += " (copy)"
        elif url.endswith("chat.postMessage") or url.endswith("setTopic"):
            key = "text" if "text" in body else "topic"
            body[key] += " (copy)"
        elif url.endswith(("conversations.invite", "conversations.kick")):
            continue
        elif url.endswith("/graphql"):
            data = body["variables"]["input"]
            key = next(iter(data))
            data[key] = data[key] + 1 if isinstance(data[key], int) else data[key] + " (copy)"
        else:
            data = body["properties"]
            key = next(iter(data))
            data[key] = data[key] + 1 if isinstance(data[key], int) else data[key] + " (copy)"
        altered[index] = (endpoint, arguments | {"body": json.dumps(body)})
        return altered
    for index, (endpoint, arguments) in enumerate(altered):
        if endpoint == "/workspace_fetch" and arguments["url"].endswith("/messages/send"):
            body = json.loads(arguments["body"])
            text = raw_message(["someone.else@unrelated.invalid"], "Wrong recipient", "x")
            altered[index] = (endpoint, arguments | {"body": json.dumps(body | {"raw": b64url_encode(text)})})
            return altered
    for index, (endpoint, arguments) in enumerate(altered):
        if endpoint == "/workspace_fetch" and arguments["url"].endswith("conversations.kick"):
            return altered[:index] + altered[index + 1 :]
    raise AssertionError("plan has no write to alter")


def response_ok(endpoint: str, status: int, payload: dict[str, Any]) -> bool:
    if endpoint == "/workspace_codec":
        return status == 200
    body = payload.get("body", {})
    if status not in (200, 201, 204):
        return False
    return body.get("ok", True) is not False and not body.get("errors")
