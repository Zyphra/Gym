# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SQL reference derived from visible instructions and initial records, never goal metadata."""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

from resources_servers.api_workspace.engine import VENDORS


def derive_calls(prompt: str, records: dict[str, Any], family: str) -> list[dict[str, Any]]:
    """Independently join physical records and apply each published workflow rule."""
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE records(vendor TEXT, function TEXT, collection TEXT, data TEXT)")
    for vendor, collections in records.items():
        for collection, rows in collections.items():
            db.executemany(
                "INSERT INTO records VALUES(?,?,?,?)",
                [(vendor, VENDORS[vendor].function, collection, json.dumps(row)) for row in rows.values()],
            )
    calls: list[dict[str, Any]] = []

    def rows(
        collection: str, function: str = "", condition: str = "1", parameters: tuple = ()
    ) -> list[tuple[str, dict]]:
        query = "SELECT vendor,data FROM records WHERE collection=? AND (?='' OR function=?) AND " + condition
        return [(v, json.loads(d)) for v, d in db.execute(query, (collection, function, function, *parameters))]

    def select(collection: str, field: str, value: str, function: str = "") -> tuple[str, dict]:
        found = rows(collection, function, f"json_extract(data,'$.{field}')=?", (value,))
        assert len(found) == 1, (collection, field, value, len(found))
        return found[0]

    def update(item: tuple[str, dict], collection: str, **fields: Any) -> None:
        vendor, row = item
        calls.append(
            {
                "vendor": vendor,
                "method": "PATCH",
                "path": VENDORS[vendor].path(collection, row["id"]),
                "body": VENDORS[vendor].wire(fields),
            }
        )

    def create(function: str, collection: str, fields: dict) -> None:
        placement = min if policy["new_records_vendor_order"] == "alphabetical_first" else max
        vendor = placement(v for v in records if VENDORS[v].function == function)
        calls.append(
            {
                "vendor": vendor,
                "method": "POST",
                "path": VENDORS[vendor].path(collection),
                "body": VENDORS[vendor].wire(fields),
                "idempotency_key": f"independent-sql-{len(calls)}",
            }
        )

    approved = rows("files", "storage", "json_extract(data,'$.status')='approved'")
    policies = [
        item
        for item in approved
        if {"expense_limit", "renewal_discount_percent", "cost_center"} <= json.loads(item[1]["content"]).keys()
    ]
    assert len(policies) == 1, "The current operating policy must be uniquely discoverable"
    policy = json.loads(policies[0][1]["content"])
    coordinator = policy["coordinator"]
    folder = policy["operations_folder"]
    channel = policy["dispatch_channel"]
    namespace = re.match(r"gw_[a-f0-9]+", policies[0][1]["name"])[0]
    subject = f"{namespace} {family}"
    company_match = re.search(r"GW Atelier [a-f0-9]+-\d+", prompt)
    email_match = re.search(r"staff-[a-f0-9]+-\d+@gw-example\.invalid", prompt)
    company = company_match[0] if company_match else ""
    email = email_match[0] if email_match else ""

    def document(content: str, extension: str = ".txt", sharing: bool = False) -> None:
        create(
            "storage",
            "files",
            {
                "name": subject + extension,
                "content": content,
                "folder": folder,
                "owner": coordinator,
                "status": "draft",
                "access": "private",
                "shared_with": [coordinator] if sharing else [],
                "version": 1,
            },
        )

    def draft(body: str, to: list[str], thread: str = "", title: str | None = None) -> None:
        create(
            "mail",
            "drafts",
            {"subject": title or subject, "body": body, "to": to, "thread": thread, "status": "draft"},
        )

    def post(body: str) -> None:
        create(
            "chat",
            "messages",
            {"channel": channel, "body": body, "sender": coordinator, "thread": "", "status": "posted"},
        )

    def company_file(name: str) -> tuple[str, dict]:
        matches = [x for x in rows("files", "storage") if json.loads(x[1]["content"]).get("account") == name]
        assert len(matches) == 1
        return matches[0]

    def company_mail(name: str) -> tuple[str, dict]:
        return next(x for x in rows("messages", "mail") if json.loads(x[1]["body"])["account"] == name)

    if family == "hr_onboarding":
        selected = rows(
            "employees",
            condition="json_extract(data,'$.department')=? AND json_extract(data,'$.status')='pending'",
            parameters=(policy["department"],),
        )
        for item in selected:
            update(item, "employees", status="active", location=policy["location"], cost_center=policy["cost_center"])
        document("\n".join(sorted(x[1]["email"] for x in selected)))
    elif family == "hr_location":
        update(select("employees", "email", email), "employees", location=policy["location"])
        update(select("expenses", "employee", email), "expenses", cost_center=policy["cost_center"])
    elif family == "hr_leave_cover":
        delegate = re.findall(r"staff-[a-f0-9]+-\d+@gw-example\.invalid", prompt)[1]
        update(select("leave", "employee", email), "leave", status="approved", delegate=delegate)
        update(select("accounts", "owner", email), "accounts", owner=delegate)
    elif family == "hr_conversion":
        update(
            select("employees", "email", email),
            "employees",
            employment_type="employee",
            cost_center=policy["cost_center"],
        )
        update(select("expenses", "employee", email), "expenses", category="onboarding")
    elif family == "hr_departure":
        update(select("employees", "email", email), "employees", status="inactive")
        for item in rows("accounts", condition="json_extract(data,'$.owner')=?", parameters=(email,)):
            update(item, "accounts", owner=coordinator)
        for item in rows("files", "storage", "json_extract(data,'$.owner')=?", (email,)):
            update(item, "files", owner=coordinator)
    elif family == "crm_renewal":
        item = select("opportunities", "account", company)
        amount = db.execute(
            "SELECT CAST(json_extract(data,'$.amount')*(100-?)/100 AS INTEGER) FROM records WHERE collection='opportunities' AND json_extract(data,'$.account')=?",
            (policy["renewal_discount_percent"], company),
        ).fetchone()[0]
        update(item, "opportunities", amount=amount, stage="proposal", close_date=policy["effective_date"])
        draft(str(amount), [select("contacts", "account", company)[1]["email"]])
    elif family == "crm_territory":
        selected = rows("accounts", condition="json_extract(data,'$.region')=?", parameters=(policy["region"],))
        for item in selected:
            update(item, "accounts", owner=coordinator, tier="priority")
        joined = db.execute(
            "SELECT o.vendor,o.data FROM records o JOIN records a ON json_extract(o.data,'$.account')=json_extract(a.data,'$.name') WHERE o.collection='opportunities' AND a.collection='accounts' AND json_extract(a.data,'$.region')=?",
            (policy["region"],),
        )
        for vendor, data in joined:
            update((vendor, json.loads(data)), "opportunities", owner=coordinator)
    elif family == "crm_contacts":
        item = company_mail(company)
        update(
            select("contacts", "account", company),
            "contacts",
            email=json.loads(item[1]["body"])["requested_contact"],
            status="verified",
        )
        update(item, "messages", status="read", labels=["contact-processed"])
    elif family == "crm_pipeline":
        selected = rows("opportunities", condition="json_extract(data,'$.probability')>=60")
        for item in selected:
            update(item, "opportunities", stage="review")
        post("\n".join(sorted(f"{x[1]['account']}={x[1]['amount']}" for x in selected)))
    elif family == "crm_dormant":
        update(select("accounts", "name", company), "accounts", status="paused")
        update(select("opportunities", "account", company), "opportunities", stage="on_hold", probability=0)
        update(company_mail(company), "messages", labels=["account-paused"])
    elif family == "acct_reconcile":
        matched = db.execute(
            "SELECT i.vendor,i.data,json_extract(p.data,'$.amount') FROM records i JOIN records p ON json_extract(i.data,'$.number')=json_extract(p.data,'$.invoice') WHERE i.collection='invoices' AND p.collection='payments' AND json_extract(p.data,'$.status')='settled'"
        ).fetchall()
        for vendor, raw, amount in matched:
            item = (vendor, json.loads(raw))
            update(item, "invoices", paid=amount, status="paid" if item[1]["amount"] == amount else "partial")
        document(str(sum(x[2] for x in matched)))
    elif family in {"acct_dispute", "acct_credit"}:
        invoice = select("invoices", "account", company)
        approved = json.loads(company_file(company)[1]["content"])["approved_amount"]
        if family == "acct_dispute":
            update(invoice, "invoices", status="disputed", owner=coordinator)
            mail = company_mail(company)[1]
            draft(f"{invoice[1]['number']}={approved}", [mail["sender"]], mail["thread"])
        else:
            update(invoice, "invoices", amount=approved, status="adjusted")
            document(f"{invoice[1]['number']}={invoice[1]['amount'] - approved}")
    elif family == "acct_expenses":
        for item in rows("expenses", condition="json_extract(data,'$.status')='submitted'"):
            update(
                item,
                "expenses",
                status="approved" if item[1]["amount"] <= policy["expense_limit"] else "review",
                cost_center=policy["cost_center"],
            )
    elif family == "acct_overdue":
        for item in rows(
            "invoices",
            condition="json_extract(data,'$.status')='open' AND json_extract(data,'$.amount')>?",
            parameters=(policy["expense_limit"],),
        ):
            update(item, "invoices", status="followup")
            draft(
                str(item[1]["amount"]),
                [select("contacts", "account", item[1]["account"])[1]["email"]],
                title=f"Followup {item[1]['number']}",
            )
    elif family in {"storage_publish", "storage_archive"}:
        selected = []
        for item in rows("files", "storage"):
            data = json.loads(item[1]["content"])
            if "account" not in data:
                continue
            if (family == "storage_publish" and item[1]["status"] == "approved") or (
                family == "storage_archive" and data["updated"] < policy["archive_before"]
            ):
                selected.append(item)
                update(
                    item,
                    "files",
                    **(
                        {"access": "team", "shared_with": [coordinator], "status": "published"}
                        if family == "storage_publish"
                        else {"status": "archived", "folder": "archive"}
                    ),
                )
        if family == "storage_publish":
            post("\n".join(sorted(x[1]["name"] for x in selected)))
        else:
            document(str(len(selected)))
    elif family == "storage_handoff":
        item = company_file(company)
        update(item, "files", owner=coordinator, shared_with=[item[1]["owner"]])
        update(select("accounts", "name", company), "accounts", owner=coordinator)
    elif family == "storage_index":
        joined = db.execute(
            "SELECT json_extract(a.data,'$.name'),json_extract(c.data,'$.email') FROM records a JOIN records c ON json_extract(a.data,'$.name')=json_extract(c.data,'$.account') WHERE a.collection='accounts' AND c.collection='contacts' ORDER BY json_extract(a.data,'$.name')"
        ).fetchall()
        document("\n".join(f"{a},{b}" for a, b in joined), ".csv", True)
        update(select("folders", "name", folder), "folders", shared_with=[coordinator])
    elif family == "storage_redact":
        item = company_file(company)
        value = json.loads(item[1]["content"])
        del value["private_phone"]
        update(
            item,
            "files",
            content=json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
            version=item[1]["version"] + 1,
        )
        draft(item[1]["name"], [select("contacts", "account", company)[1]["email"]])
    elif family == "mail_reply":
        mail = company_mail(company)
        invoice = select("invoices", "account", company)[1]
        draft(f"{invoice['number']}:{invoice['amount'] - invoice['paid']}", [mail[1]["sender"]], mail[1]["thread"])
        update(mail, "messages", status="read", labels=["replied"])
    elif family == "mail_campaign":
        joined = db.execute(
            "SELECT json_extract(c.data,'$.email') FROM records c JOIN records a ON json_extract(c.data,'$.account')=json_extract(a.data,'$.name') WHERE c.collection='contacts' AND a.collection='accounts' AND json_extract(a.data,'$.region')=?",
            (policy["region"],),
        ).fetchall()
        for (contact,) in joined:
            draft(policy["effective_date"], [contact])
        post(str(len(joined)))
    elif family == "mail_triage":
        high = []
        for item in rows("messages", "mail"):
            data = json.loads(item[1]["body"])
            urgent = data["priority"] == "high"
            update(item, "messages", status="read", labels=["urgent" if urgent else "routine"])
            if urgent:
                high.append(data["account"])
        document("\n".join(sorted(high)))
    elif family == "mail_followup":
        mail = company_mail(company)[1]
        employee = json.loads(mail["body"])["employee"]
        update(select("leave", "employee", employee), "leave", delegate=coordinator)
        draft(policy["effective_date"], [employee], mail["thread"])
    elif family == "mail_deduplicate":
        thread = re.search(r"thread (gw_[a-f0-9]+-thread-\d+)", prompt)[1]
        vendor = re.search(r"copy in vendor (gw_\w+)", prompt)[1]
        for item in rows("messages", "mail", "json_extract(data,'$.thread')=?", (thread,)):
            if item[0] == vendor:
                update(item, "messages", labels=["canonical"])
            else:
                calls.append(
                    {"vendor": item[0], "method": "DELETE", "path": VENDORS[item[0]].path("messages", item[1]["id"])}
                )
        post(thread)
    elif family in {"chat_incident", "chat_closeout"}:
        for item in rows("messages", "chat"):
            data = json.loads(item[1]["body"])
            selected = (
                (data["severity"] == "urgent" and not data["resolved"])
                if family == "chat_incident"
                else data["resolved"]
            )
            if selected:
                update(item, "messages", status="escalated" if family == "chat_incident" else "closed")
                if family == "chat_incident":
                    update(
                        select("invoices", "account", data["account"]), "invoices", owner=coordinator, status="review"
                    )
                else:
                    update(select("accounts", "name", data["account"]), "accounts", tier="reviewed")
    elif family == "chat_roster":
        selected = rows(
            "employees", condition="json_extract(data,'$.department')=?", parameters=(policy["department"],)
        )
        members = sorted(x[1]["email"] for x in selected)
        update(select("channels", "name", channel), "channels", members=sorted([coordinator, *members]))
        document("\n".join(members))
    elif family == "chat_digest":
        joined = db.execute(
            "SELECT json_extract(m.data,'$.thread'),json_extract(i.data,'$.amount') "
            "FROM records m JOIN records i ON "
            "json_extract(json_extract(m.data,'$.body'),'$.invoice')=json_extract(i.data,'$.number') "
            "WHERE m.collection='messages' AND m.function='chat' AND i.collection='invoices'"
        ).fetchall()
        post("\n".join(sorted(f"{thread}={amount}" for thread, amount in joined)))
        draft(str(sum(x[1] for x in joined)), [coordinator])
    elif family == "chat_topic":
        update(
            select("channels", "name", channel),
            "channels",
            topic=f"{policy['department']} | {policy['effective_date']} | {policy['cost_center']}",
        )
        update(select("folders", "name", folder), "folders", owner=coordinator, access="team")
    else:
        raise ValueError(f"No independent reference for {family}")
    db.close()
    return calls
