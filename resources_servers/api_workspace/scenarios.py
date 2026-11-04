# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independently authored, deterministic fictional business workflows.

No external task corpus, company, document, benchmark or model output is an input.
The generator varies record count, topology, values and decision boundaries as
well as names. Goal records stay server-side; rows carry only canonical IDs.
"""

from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from resources_servers.api_workspace.engine import COLLECTIONS, VENDOR_NAMES, VENDORS, canonical_json
from resources_servers.api_workspace.realapi import APP_VENDORS, APPS


REVISION = "api-workspace-v4"
ROOT_SEED = 20261028
TRAIN_VARIANTS = 80
VAL_VARIANTS = 8
FAMILIES = (
    "hr_onboarding",
    "hr_location",
    "hr_leave_cover",
    "hr_conversion",
    "hr_departure",
    "crm_renewal",
    "crm_territory",
    "crm_contacts",
    "crm_pipeline",
    "crm_dormant",
    "acct_reconcile",
    "acct_dispute",
    "acct_expenses",
    "acct_overdue",
    "acct_credit",
    "storage_publish",
    "storage_archive",
    "storage_handoff",
    "storage_index",
    "storage_redact",
    "mail_reply",
    "mail_campaign",
    "mail_triage",
    "mail_followup",
    "mail_deduplicate",
    "chat_incident",
    "chat_roster",
    "chat_digest",
    "chat_topic",
    "chat_closeout",
)
SYSTEM_PROMPT = """You are working in a fictional workspace containing simulated business records.
The available tools provide an API catalog, vendor schemas and HTTP requests.
Follow the current workspace operating policy, including its rules for placing new records.
Completion is determined by the resulting workspace state within the available turn and context budget.
"""


def tools() -> list[dict[str, Any]]:
    request_fields = {
        "vendor": {"type": "string"},
        "method": {"type": "string", "enum": ["GET", "POST", "PATCH", "DELETE"]},
        "path": {"type": "string"},
        "params": {"anyOf": [{"type": "object", "additionalProperties": True}, {"type": "string"}, {"type": "null"}]},
        "body": {"anyOf": [{"type": "object", "additionalProperties": True}, {"type": "string"}, {"type": "null"}]},
        "idempotency_key": {"type": ["string", "null"]},
    }
    entries = [
        (
            "workspace_catalog",
            "Search fictional API endpoint metadata by operation, collection, function or field. Returns routes and connection status, not records.",
            {
                "query": {"type": "string"},
                "function": {"type": ["string", "null"], "enum": [*COLLECTIONS, None]},
                "connected_only": {"type": "boolean"},
                "method": {"type": ["string", "null"], "enum": ["GET", "POST", "PATCH", "DELETE", None]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                "offset": {"type": "integer", "minimum": 0},
            },
            [],
        ),
        (
            "workspace_schema",
            "Discover exact API paths, field aliases and parameter contract for a vendor.",
            {"vendor": {"type": "string"}},
            ["vendor"],
        ),
        (
            "workspace_request",
            "Issue a request to a fictional vendor API using its method, path and optional containers.",
            request_fields,
            ["vendor", "method", "path"],
        ),
        (
            "workspace_parallel",
            "Execute independent API requests concurrently in input order; returns each HTTP status.",
            {
                "requests": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "properties": request_fields,
                        "required": ["vendor", "method", "path"],
                        "additionalProperties": False,
                    },
                }
            },
            ["requests"],
        ),
    ]
    return [
        {
            "type": "function",
            "name": name,
            "description": description,
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": fields,
                "required": required,
                "additionalProperties": False,
            },
        }
        for name, description, fields, required in entries
    ]


@dataclass(frozen=True)
class Scenario:
    task_id: str
    namespace: str
    family: str
    split: str
    variant: int
    connected: tuple[str, ...]
    records: dict[str, Any]
    prompt: str
    updates: tuple[dict[str, Any], ...]
    creates: tuple[dict[str, Any], ...]
    deletes: tuple[dict[str, Any], ...]
    revision: str = REVISION
    # Named guards, such as a recipient that must not receive mail; graded by the final state.
    guards: tuple[dict[str, Any], ...] = ()
    # Session facts the V5 APIs need: the connected mailbox, chat user and clock.
    context: dict[str, Any] = field(default_factory=dict)


class _Builder:
    def __init__(self, split: str, family: str, variant: int):
        self.task_id = f"{REVISION}:{split}:{family}:{variant:04d}"
        digest = hashlib.sha256(f"{ROOT_SEED}:{self.task_id}".encode()).hexdigest()
        self.rng = random.Random(int(digest, 16))
        self.namespace = "gw_" + digest[:10]
        self.family, self.split, self.variant = family, split, variant
        self.connected: list[str] = []
        self.vendors: dict[str, list[str]] = {}
        self.records: dict[str, Any] = {}
        for function, names in VENDOR_NAMES.items():
            chosen = self.rng.sample(list(names), 2)
            self.vendors[function] = chosen
            self.connected.extend(chosen)
            for name in chosen:
                self.records[name] = {collection: {} for collection in COLLECTIONS[function]}
        self.updates: list[dict[str, Any]] = []
        self.creates: list[dict[str, Any]] = []
        self.deletes: list[dict[str, Any]] = []
        self.rows: dict[str, list[dict[str, Any]]] = {
            collection: [] for collections in COLLECTIONS.values() for collection in collections
        }
        self.count = self.rng.randint(4, 8)
        self.department = self.rng.choice(("Field Research", "Quality Engineering", "Design Operations"))
        self.region = self.rng.choice(("North", "West", "Coastal"))
        self.threshold = self.rng.choice((12500, 25000, 45000))
        self.manager = f"coordinator-{digest[10:16]}@gw-example.invalid"
        self.folder_name = f"{self.namespace}-operations"
        self.channel_name = f"{self.namespace}-dispatch"
        self.policy_name = f"{self.namespace}-operating-policy.json"
        self.policy = {
            "department": self.department,
            "coordinator": self.manager,
            "operations_folder": self.folder_name,
            "dispatch_channel": self.channel_name,
            "new_records_vendor_order": self.rng.choice(("alphabetical_first", "alphabetical_last")),
            "region": self.region,
            "location": self.rng.choice(("Tern Office", "Willow Office", "Reed Office")),
            "cost_center": f"CC-{self.rng.randint(210, 890)}",
            "expense_limit": self.threshold,
            "renewal_discount_percent": self.rng.choice((5, 10, 15)),
            "archive_before": "2026-06-01",
            "effective_date": "2026-11-01",
        }
        self.fixture()

    def add(self, function: str, collection: str, fields: dict[str, Any], index: int = 0) -> dict[str, Any]:
        name = self.vendors[function][index % 2]
        token = hashlib.sha256(f"{self.task_id}:{collection}:{len(self.rows[collection])}".encode()).hexdigest()[:14]
        row = {"id": "r_" + token, **fields}
        self.records[name][collection][row["id"]] = row
        handle = {"vendor": name, "collection": collection, "id": row["id"], "row": row}
        self.rows[collection].append(handle)
        return handle

    def update(self, handle: dict[str, Any], **fields: Any) -> None:
        self.updates.append({key: handle[key] for key in ("vendor", "collection", "id")} | {"fields": fields})

    def create(self, function: str, collection: str, **fields: Any) -> None:
        self.creates.append({"vendor": self.vendors[function][0], "collection": collection, "fields": fields})

    def delete(self, handle: dict[str, Any]) -> None:
        self.deletes.append({key: handle[key] for key in ("vendor", "collection", "id")})

    def document(
        self, name: str, content: str, *, access: str = "private", shared_with: list[str] | None = None
    ) -> None:
        self.create(
            "storage",
            "files",
            name=name,
            content=content,
            folder=self.folder_name,
            owner=self.manager,
            status="draft",
            access=access,
            shared_with=shared_with or [],
            version=1,
        )

    def draft(self, subject: str, body: str, to: list[str], thread: str = "") -> None:
        self.create("mail", "drafts", subject=subject, body=body, to=to, thread=thread, status="draft")

    def post(self, body: str, thread: str = "") -> None:
        self.create(
            "chat",
            "messages",
            channel=self.channel_name,
            body=body,
            sender=self.manager,
            thread=thread,
            status="posted",
        )

    def fixture(self) -> None:
        self.add(
            "storage",
            "files",
            {
                "name": self.policy_name,
                "content": canonical_json(self.policy),
                "folder": self.folder_name,
                "owner": self.manager,
                "status": "approved",
                "access": "private",
                "shared_with": [],
                "version": 2,
            },
            1,
        )
        self.add(
            "storage",
            "folders",
            {
                "name": self.folder_name,
                "owner": self.manager,
                "access": "private",
                "shared_with": [],
                "status": "active",
            },
        )
        self.add(
            "chat",
            "channels",
            {"name": self.channel_name, "topic": "Work intake", "members": [self.manager], "status": "active"},
        )
        for i in range(self.count):
            email = f"staff-{self.namespace[3:]}-{i}@gw-example.invalid"
            account = f"GW Atelier {self.namespace[3:]}-{i}"
            amount = self.threshold + self.rng.choice((-4000, -1, 0, 1, 5000, 12000))
            if i < 2:
                amount = self.threshold + (-1 if i == 0 else 1)
            self.add(
                "hr",
                "employees",
                {
                    "name": f"Colleague {self.namespace[3:]} {i}",
                    "email": email,
                    "department": self.department if i % 3 else "Supply Design",
                    "manager": self.manager,
                    "status": "pending" if i % 2 == 0 else "active",
                    "location": "Remote",
                    "start_date": f"2026-11-{i + 1:02d}",
                    "employment_type": "contractor",
                    "cost_center": "CC-UNASSIGNED",
                },
                i,
            )
            self.add(
                "hr",
                "leave",
                {
                    "employee": email,
                    "start_date": "2026-11-10",
                    "end_date": "2026-11-14",
                    "status": "requested",
                    "delegate": "",
                },
                i + 1,
            )
            self.add(
                "crm",
                "accounts",
                {
                    "name": account,
                    "owner": email,
                    "region": self.region if i % 2 else "Central",
                    "status": "active",
                    "tier": "standard",
                    "renewal_date": f"2026-11-{i + 1:02d}",
                },
                i,
            )
            self.add(
                "crm",
                "contacts",
                {
                    "name": f"Contact {self.namespace[3:]} {i}",
                    "email": f"contact-{self.namespace[3:]}-{i}@gw-example.invalid",
                    "account": account,
                    "role": "billing" if i % 2 == 0 else "operations",
                    "status": "unverified",
                },
                i + 1,
            )
            self.add(
                "crm",
                "opportunities",
                {
                    "name": f"Renewal {account}",
                    "account": account,
                    "owner": email,
                    "stage": "qualified",
                    "amount": amount * (i + 2),
                    "close_date": f"2026-11-{i + 1:02d}",
                    "probability": self.rng.choice((35, 60, 85)),
                },
                i + 1,
            )
            self.add(
                "accounting",
                "invoices",
                {
                    "number": f"{self.namespace}-INV-{i}",
                    "account": account,
                    "amount": amount,
                    "paid": 0,
                    "currency": "USD",
                    "due_date": f"2026-10-{15 + i:02d}",
                    "status": "open",
                    "owner": email,
                    "reference": f"{self.namespace}-PO-{i}",
                },
                i,
            )
            self.add(
                "accounting",
                "payments",
                {
                    "reference": f"{self.namespace}-PAY-{i}",
                    "invoice": f"{self.namespace}-INV-{i}",
                    "amount": max(1, amount - i * 1000),
                    "currency": "USD",
                    "status": "settled" if i % 2 == 0 else "pending",
                },
                i + 1,
            )
            self.add(
                "accounting",
                "expenses",
                {
                    "employee": email,
                    "amount": amount,
                    "category": "equipment" if i % 2 == 0 else "travel",
                    "status": "submitted",
                    "cost_center": "CC-UNASSIGNED",
                    "reference": f"{self.namespace}-EXP-{i}",
                },
                i,
            )
            contact = self.rows["contacts"][-1]["row"]["email"]
            content = canonical_json(
                {
                    "account": account,
                    "email": contact,
                    "owner": email,
                    "approved_amount": max(0, amount - 2000),
                    "classification": "internal",
                    "version": i + 1,
                    "updated": "2026-05-01" if i % 2 == 0 else "2026-09-01",
                    "private_phone": f"fictional-extension-{i}",
                }
            )
            self.add(
                "storage",
                "files",
                {
                    "name": f"{self.namespace}-account-{i}.json",
                    "content": content,
                    "folder": self.folder_name,
                    "owner": email,
                    "status": "approved" if i % 2 == 0 else "draft",
                    "access": "private",
                    "shared_with": [],
                    "version": i + 1,
                },
                i,
            )
            self.add(
                "mail",
                "messages",
                {
                    "subject": f"{self.namespace} request {i}",
                    "body": canonical_json(
                        {
                            "account": account,
                            "invoice": f"{self.namespace}-INV-{i}",
                            "employee": email,
                            "requested_contact": f"verified-{contact}",
                            "priority": "high" if i % 2 == 0 else "normal",
                        }
                    ),
                    "sender": contact,
                    "to": [self.manager],
                    "thread": f"{self.namespace}-thread-{i}",
                    "status": "unread",
                    "labels": [],
                },
                i + 1,
            )
            self.add(
                "chat",
                "messages",
                {
                    "channel": self.channel_name,
                    "body": canonical_json(
                        {
                            "account": account,
                            "employee": email,
                            "invoice": f"{self.namespace}-INV-{i}",
                            "severity": "urgent" if i % 2 == 0 else "routine",
                            "resolved": i % 3 == 0,
                        }
                    ),
                    "sender": email,
                    "thread": f"{self.namespace}-case-{i}",
                    "status": "posted",
                },
                i,
            )
        # Plausible near-name distractors are kept immutable by the grader.
        self.add(
            "storage",
            "files",
            {
                "name": self.policy_name.replace(".json", "-superseded.json"),
                "content": canonical_json(self.policy | {"expense_limit": self.threshold * 3}),
                "folder": "archive",
                "owner": self.manager,
                "status": "superseded",
                "access": "private",
                "shared_with": [],
                "version": 1,
            },
        )

    def finish(self, request: str) -> Scenario:
        # Surface variation is deterministic and independent of any benchmark.
        style = (FAMILIES.index(self.family) + self.variant) % 6
        guidance = (
            f"The operating-policy document is '{self.policy_name}'. The operations folder is "
            f"'{self.folder_name}', and {self.manager} coordinates this work.",
            f"Use the operating rules in '{self.policy_name}'; its current version is marked approved.",
            "Use the approved operating policy in the document store for the company details and operating rules.",
            "Apply the current workspace operating policy. The superseded policy is retained for history.",
            f"The coordinator for this work is {self.manager}. Other workspace details are in the approved operating policy.",
            f"The dispatch destination is '{self.channel_name}'. The governing policy document is '{self.policy_name}'.",
        )[style]
        renderings = (
            f"{guidance} {request}",
            f"{request}\n\nContext: {guidance}",
            f"Please complete the following work.\n{request}\n\n{guidance}",
            f"Work item\n\n{request}\n\nOperating context\n{guidance}",
            f"{guidance}\n\nRequested changes: {request}",
            f"{request}\n{guidance}",
        )
        placement = min if self.policy["new_records_vendor_order"] == "alphabetical_first" else max
        # New-record placement is a business rule in the discoverable policy.
        for goal in self.creates:
            function = VENDORS[goal["vendor"]].function
            goal["vendor"] = placement(self.vendors[function])
        return Scenario(
            self.task_id,
            self.namespace,
            self.family,
            self.split,
            self.variant,
            tuple(sorted(self.connected)),
            self.records,
            renderings[style],
            tuple(self.updates),
            tuple(self.creates),
            tuple(self.deletes),
        )


DOCUMENT = (
    "in the operations folder with coordinator owner, status 'draft', access 'private', shared_with [] and version 1"
)


def _workflow(b: _Builder) -> str:
    f, p, rows = b.family, b.policy, b.rows
    employees, accounts = rows["employees"], rows["accounts"]
    target = b.rng.randrange(b.count)
    employee, account = employees[target], accounts[target]
    email, company = employee["row"]["email"], account["row"]["name"]
    invoice, file = rows["invoices"][target], rows["files"][target + 1]
    # mail and chat share a collection name, so select their typed physical records.
    mails = [x for x in rows["messages"] if VENDORS[x["vendor"]].function == "mail"]
    chats = [x for x in rows["messages"] if VENDORS[x["vendor"]].function == "chat"]
    mail = mails[target]
    thread = mail["row"]["thread"]
    subject = f"{b.namespace} {f}"
    if f == "hr_onboarding":
        selected = [x for x in employees if x["row"]["department"] == b.department and x["row"]["status"] == "pending"]
        for x in selected:
            b.update(x, status="active", location=p["location"], cost_center=p["cost_center"])
        content = "\n".join(sorted(x["row"]["email"] for x in selected))
        b.document(subject + ".txt", content)
        return (
            f"Set status 'active' on pending employees in {b.department}, and set their location and cost_center to the "
            f"policy values. Save '{subject}.txt' containing their emails sorted one per line, {DOCUMENT}. "
            "Leave other employees untouched."
        )
    if f == "hr_location":
        b.update(employee, location=p["location"])
        b.update(rows["expenses"][target], cost_center=p["cost_center"])
        return (
            f"Relocate {email} to the policy location and update only that person's submitted expense's cost_center "
            "to the policy value. Keep employment status and all expense amounts unchanged."
        )
    if f == "hr_leave_cover":
        delegate = employees[(target + 1) % b.count]["row"]["email"]
        b.update(rows["leave"][target], status="approved", delegate=delegate)
        b.update(account, owner=delegate)
        return (
            f"Approve {email}'s requested November leave: set its status 'approved' and delegate {delegate}. Transfer "
            "only their CRM account's owner to that delegate during leave; do not alter the employee record."
        )
    if f == "hr_conversion":
        b.update(employee, employment_type="employee", cost_center=p["cost_center"])
        b.update(rows["expenses"][target], category="onboarding")
        return (
            f"Convert contractor {email}: set employment_type 'employee' and set their cost_center to the policy "
            "cost_center. Set their expense's category to 'onboarding'; preserve its approval status and amount."
        )
    if f == "hr_departure":
        b.update(employee, status="inactive")
        owned = [x for x in accounts if x["row"]["owner"] == email]
        for x in owned:
            b.update(x, owner=b.manager)
        b.update(file, owner=b.manager)
        return (
            f"Offboard {email}: set HR status 'inactive', transfer all their CRM accounts and their account JSON file "
            "to the coordinator, preserving everything else. Do not delete historical records."
        )
    if f == "crm_renewal":
        opp = rows["opportunities"][target]
        amount = opp["row"]["amount"] * (100 - p["renewal_discount_percent"]) // 100
        b.update(opp, amount=amount, stage="proposal", close_date=p["effective_date"])
        b.draft(subject, str(amount), [rows["contacts"][target]["row"]["email"]])
        return (
            f"For {company}'s renewal opportunity apply the policy renewal_discount_percent to amount, round down "
            "to integer cents, set stage 'proposal' and close_date to the policy effective_date. Create a mail draft to the "
            f"account's CRM contact email with subject '{subject}', body only the discounted integer, empty thread and status 'draft'."
        )
    if f == "crm_territory":
        selected = [x for x in accounts if x["row"]["region"] == b.region]
        for x in selected:
            b.update(x, owner=b.manager, tier="priority")
        selected_names = {x["row"]["name"] for x in selected}
        for x in rows["opportunities"]:
            if x["row"]["account"] in selected_names:
                b.update(x, owner=b.manager)
        return (
            f"Reassign accounts in the policy region ({b.region}) and their opportunities to the coordinator; "
            "set those accounts' tier to 'priority'. Keep opportunities' amounts and stages unchanged."
        )
    if f == "crm_contacts":
        new_email = json.loads(mail["row"]["body"])["requested_contact"]
        b.update(rows["contacts"][target], email=new_email, status="verified")
        b.update(mail, status="read", labels=["contact-processed"])
        return (
            f"Read the mail request for {company}. Set the matching CRM contact's email to the request's "
            "requested_contact and its status to 'verified'. Mark only that request status 'read' with labels exactly "
            '["contact-processed"].'
        )
    if f == "crm_pipeline":
        selected = [x for x in rows["opportunities"] if x["row"]["probability"] >= 60]
        for x in selected:
            b.update(x, stage="review")
        body = "\n".join(sorted(f"{x['row']['account']}={x['row']['amount']}" for x in selected))
        b.post(body)
        return (
            "Set stage 'review' on qualified opportunities with probability at least 60. Post one line per such "
            "opportunity, '<account>=<amount>', sorted by account, as a chat message in the dispatch channel with coordinator "
            "sender, empty thread and status 'posted'. Do not include lower-probability opportunities."
        )
    if f == "crm_dormant":
        b.update(account, status="paused")
        b.update(rows["opportunities"][target], stage="on_hold", probability=0)
        b.update(mail, labels=["account-paused"])
        return (
            f"Set {company}'s CRM account status to 'paused', set its renewal opportunity stage 'on_hold' with "
            'probability 0, and set its related mail labels exactly ["account-paused"]. Leave the mail unread status and '
            "all financial records intact."
        )
    if f == "acct_reconcile":
        settled = {
            x["row"]["invoice"]: x["row"]["amount"] for x in rows["payments"] if x["row"]["status"] == "settled"
        }
        for x in rows["invoices"]:
            if x["row"]["number"] in settled:
                amount = settled[x["row"]["number"]]
                b.update(x, paid=amount, status="paid" if amount == x["row"]["amount"] else "partial")
        b.document(subject + ".txt", str(sum(settled.values())))
        return (
            "Reconcile all settled payments to their invoice number: set paid to the settled amount, and status 'paid' "
            "if equal to the invoice amount, otherwise 'partial'. Ignore pending payments. Save "
            f"'{subject}.txt' containing only the total settled cents, {DOCUMENT}."
        )
    if f == "acct_dispute":
        amount = json.loads(file["row"]["content"])["approved_amount"]
        b.update(invoice, status="disputed", owner=b.manager)
        b.draft(subject, f"{invoice['row']['number']}={amount}", [mail["row"]["sender"]], thread)
        return (
            f"Set {company}'s invoice status 'disputed' and owner coordinator. Read its account JSON file's approved_amount. "
            f"Create a mail draft to the sender of {company}'s mail request, on that request's thread, subject '{subject}', "
            "body exactly '<invoice number>=<approved_amount>', status 'draft'. Do not change invoice amount or mark it paid."
        )
    if f == "acct_expenses":
        for x in rows["expenses"]:
            b.update(
                x, status="approved" if x["row"]["amount"] <= b.threshold else "review", cost_center=p["cost_center"]
            )
        return (
            "Process all submitted expenses: amount at or below the policy expense_limit gets status 'approved'; above "
            "gets 'review'. Set their cost_center to the policy value. Never edit amount, category or employee."
        )
    if f == "acct_overdue":
        selected = [x for x in rows["invoices"] if x["row"]["amount"] > b.threshold]
        lookup = {x["row"]["account"]: x["row"]["email"] for x in rows["contacts"]}
        for x in selected:
            b.update(x, status="followup")
            b.draft(f"Followup {x['row']['number']}", str(x["row"]["amount"]), [lookup[x["row"]["account"]]])
        return (
            "For open invoices strictly above the policy expense_limit, set status 'followup' and create one mail draft "
            "to each invoice account's CRM contact email: subject 'Followup <invoice number>', body only the invoice amount in "
            "cents, empty thread, status 'draft'. Leave invoices at or below the limit untouched."
        )
    if f == "acct_credit":
        approved = json.loads(file["row"]["content"])["approved_amount"]
        credit = invoice["row"]["amount"] - approved
        b.update(invoice, amount=approved, status="adjusted")
        b.document(subject + ".txt", f"{invoice['row']['number']}={credit}")
        return (
            f"Set {company}'s invoice amount to the approved_amount in its account JSON file, and its status to 'adjusted'. "
            f"Save '{subject}.txt' containing only '<invoice number>=<credit>', where credit is the old invoice amount minus "
            f"approved_amount in integer cents, {DOCUMENT}. Preserve paid and payment records."
        )
    if f == "storage_publish":
        selected = [x for x in rows["files"] if "-account-" in x["row"]["name"] and x["row"]["status"] == "approved"]
        for x in selected:
            b.update(x, access="team", shared_with=[b.manager], status="published")
        b.post("\n".join(sorted(x["row"]["name"] for x in selected)))
        return (
            "Publish approved account JSON files only: set status 'published', access 'team' and shared_with exactly "
            "[coordinator]. Post their filenames sorted one per line as a chat message in the dispatch channel with coordinator "
            "sender, empty thread and status 'posted'. Do not publish drafts or either policy file."
        )
    if f == "storage_archive":
        selected = [
            x
            for x in rows["files"]
            if "-account-" in x["row"]["name"] and json.loads(x["row"]["content"])["updated"] < p["archive_before"]
        ]
        for x in selected:
            b.update(x, status="archived", folder="archive")
        b.document(subject + ".txt", str(len(selected)))
        return (
            "Archive account JSON files whose content.updated is before the policy archive_before: set status 'archived' "
            f"and folder 'archive'. Save '{subject}.txt' containing only the archived count, {DOCUMENT}. "
            "Keep newer files and policy files intact."
        )
    if f == "storage_handoff":
        b.update(file, owner=b.manager, shared_with=[email])
        b.update(account, owner=b.manager)
        return (
            f"Hand off {company}: transfer its account JSON file and CRM account to the coordinator, and share "
            "the file with exactly its prior employee owner. Preserve access level and content."
        )
    if f == "storage_index":
        lookup = {x["row"]["account"]: x["row"]["email"] for x in rows["contacts"]}
        content = "\n".join(
            f"{x['row']['name']},{lookup[x['row']['name']]}" for x in sorted(accounts, key=lambda a: a["row"]["name"])
        )
        b.document(subject + ".csv", content, shared_with=[b.manager])
        b.update(rows["folders"][0], shared_with=[b.manager])
        return (
            f"Join all CRM accounts to their contact emails. Save '{subject}.csv' with '<account>,<email>' rows sorted "
            "by account, one per line, no header, in the operations folder with coordinator owner, status 'draft', access "
            "'private', shared_with exactly [coordinator] and version 1. Also set the operations folder record's shared_with "
            "to exactly [coordinator]; change no other folder field."
        )
    if f == "storage_redact":
        original = json.loads(file["row"]["content"])
        redacted = {k: v for k, v in original.items() if k != "private_phone"}
        b.update(file, content=canonical_json(redacted), version=file["row"]["version"] + 1)
        b.draft(subject, file["row"]["name"], [rows["contacts"][target]["row"]["email"]])
        return (
            f"Remove only private_phone from {company}'s account JSON content and serialize compact JSON with sorted "
            "keys. Increment the file record's version field by 1; keep every other JSON value, including the version inside "
            f"the content, unchanged. Create a mail draft to the account's CRM contact email with subject '{subject}', body "
            "only the filename, empty thread and status 'draft'. Keep the file private."
        )
    if f == "mail_reply":
        b.draft(
            subject,
            f"{invoice['row']['number']}:{invoice['row']['amount'] - invoice['row']['paid']}",
            [mail["row"]["sender"]],
            thread,
        )
        b.update(mail, status="read", labels=["replied"])
        return (
            f"For {company}'s mail request, create a mail draft replying to its sender on its thread with subject "
            f"'{subject}', body '<invoice number>:<outstanding cents>' (invoice amount minus paid) and status 'draft'. "
            "Set the original request status 'read' with labels exactly [\"replied\"]."
        )
    if f == "mail_campaign":
        selected_names = {x["row"]["name"] for x in accounts if x["row"]["region"] == b.region}
        for x in rows["contacts"]:
            if x["row"]["account"] in selected_names:
                b.draft(subject, p["effective_date"], [x["row"]["email"]])
        b.post(str(len(selected_names)))
        return (
            f"For accounts in the policy region create one mail draft per CRM contact email: subject '{subject}', body "
            "only the policy effective_date value, empty thread, status 'draft'. Post only the recipient count as a chat "
            "message in the dispatch channel with coordinator sender, empty thread and status 'posted'. Do not send the drafts."
        )
    if f == "mail_triage":
        for x in mails:
            priority = json.loads(x["row"]["body"])["priority"]
            b.update(x, labels=["urgent" if priority == "high" else "routine"], status="read")
        b.document(
            subject + ".txt",
            "\n".join(
                sorted(
                    json.loads(x["row"]["body"])["account"]
                    for x in mails
                    if json.loads(x["row"]["body"])["priority"] == "high"
                )
            ),
        )
        return (
            'Triage request messages by body.priority: high gets labels exactly ["urgent"], normal gets ["routine"]; '
            f"set each status 'read'. Save '{subject}.txt' listing high-priority account names sorted one per line, "
            f"{DOCUMENT}."
        )
    if f == "mail_followup":
        b.draft(subject, p["effective_date"], [email], thread)
        b.update(rows["leave"][target], delegate=b.manager)
        return (
            f"For the employee named in {company}'s mail request, set their leave delegate to the coordinator and create "
            f"a mail draft to that employee on the request thread: subject '{subject}', body only the policy effective_date "
            "value, status 'draft'. Preserve leave approval status."
        )
    if f == "mail_deduplicate":
        fields = {k: v for k, v in mail["row"].items() if k != "id"}
        duplicate = b.add("mail", "messages", fields, 1 - b.vendors["mail"].index(mail["vendor"]))
        b.delete(duplicate)
        b.update(mail, labels=["canonical"])
        # The canonical account is explicitly named, not hidden in a timestamp assumption.
        b.post(thread)
        return (
            f"The thread {thread} has duplicate messages across mail accounts. Keep its copy in vendor "
            f'{mail["vendor"]} and set its labels exactly ["canonical"]; delete the other copy. Post only the thread '
            "ID as a chat message in the dispatch channel with coordinator sender, empty thread and status 'posted'. "
            "Do not delete other messages."
        )
    if f == "chat_incident":
        selected = [
            x
            for x in chats
            if json.loads(x["row"]["body"])["severity"] == "urgent" and not json.loads(x["row"]["body"])["resolved"]
        ]
        for x in selected:
            company_name = json.loads(x["row"]["body"])["account"]
            inv = next(y for y in rows["invoices"] if y["row"]["account"] == company_name)
            b.update(inv, owner=b.manager, status="review")
            b.update(x, status="escalated")
        return (
            "Find dispatch chat cases that are urgent and unresolved in their JSON body. Set those messages' status "
            "'escalated', and set each linked invoice's owner to the coordinator and status 'review'. Leave routine and "
            "resolved cases unchanged."
        )
    if f == "chat_roster":
        members = sorted(x["row"]["email"] for x in employees if x["row"]["department"] == b.department)
        b.update(rows["channels"][0], members=sorted([b.manager, *members]))
        b.document(subject + ".txt", "\n".join(members))
        return (
            f"Set dispatch channel members to the coordinator plus all HR employees in {b.department}, sorted. "
            f"Save '{subject}.txt' containing only those employees' emails (not the coordinator) sorted one per line, "
            f"{DOCUMENT}."
        )
    if f == "chat_digest":
        amounts = {x["row"]["number"]: x["row"]["amount"] for x in rows["invoices"]}
        content = "\n".join(
            sorted(f"{x['row']['thread']}={amounts[json.loads(x['row']['body'])['invoice']]}" for x in chats)
        )
        b.post(content)
        b.draft(subject, str(sum(amounts.values())), [b.manager])
        return (
            "Join every dispatch chat case to its invoice. Post '<case thread>=<invoice amount>' lines sorted by thread, "
            "one per line, as a chat message in the dispatch channel with coordinator sender, empty thread and status "
            f"'posted'. Create a mail draft to the coordinator with subject '{subject}', body only the sum of those invoice "
            "cents, empty thread and status 'draft'."
        )
    if f == "chat_topic":
        # The folder starts with a previous owner, so the requested ownership transfer is a real state change.
        rows["folders"][0]["row"]["owner"] = email
        b.update(rows["channels"][0], topic=f"{b.department} | {p['effective_date']} | {p['cost_center']}")
        b.update(rows["folders"][0], owner=b.manager, access="team")
        return (
            "Read the current policy and set the dispatch channel topic to '<department> | <effective_date> | "
            "<cost_center>' with one space around each pipe. Set the operations folder record's owner to the coordinator "
            "and access to 'team'. Preserve its sharing list and the policy files."
        )
    if f == "chat_closeout":
        selected = [x for x in chats if json.loads(x["row"]["body"])["resolved"]]
        for x in selected:
            b.update(x, status="closed")
            name = json.loads(x["row"]["body"])["account"]
            b.update(next(y for y in accounts if y["row"]["name"] == name), tier="reviewed")
        return (
            "Set status 'closed' on dispatch chat messages whose JSON resolved is true, and set their CRM account tier "
            "'reviewed'. Preserve message bodies, account status and all unresolved cases."
        )
    raise ValueError("unknown workflow family")


# ---------------------------------------------------------------------------
# V5: multi-app workflows over real-API conventions (mail, spreadsheets, chat, CRM).
#
# Each family writes to at least two apps, reads its rules from a current policy
# that lives in mail or in a spreadsheet next to a superseded version, and guards
# recipients and unrelated records. Names, domains, IDs and values are generated
# here from fictional word lists; no benchmark task, record or assertion is used.

REVISION_V5 = "api-workspace-v5"
ROOT_SEED_V5 = 20261104
TRAIN_VARIANTS_V5 = 80
VAL_VARIANTS_V5 = 8
FAMILIES_V5 = (
    "rx_invoice_intake",
    "rx_escalation_routing",
    "rx_renewal_quotes",
    "rx_roster_invites",
    "rx_label_holds",
    "rx_payment_reminders",
    "rx_deal_handoff",
    "rx_signup_import",
    "rx_budget_alerts",
    "rx_offboarding",
    "rx_refund_review",
    "rx_vendor_onboarding",
)
SYSTEM_PROMPT_V5 = """You are working in a fictional company workspace whose business apps are simulated HTTP APIs.
Find endpoints with the search tool, then call them with exact URLs, query parameters and request bodies.
Follow the current policies stated in the workspace's mail and spreadsheets.
Completion is determined by the resulting workspace state within the available turn and context budget.
"""
NOW_V5 = "2026-11-12T16:00:00Z"
_ORG_HEAD = ("Tarn", "Corrie", "Brack", "Holm", "Skerry", "Fell", "Gill", "Wold", "Mire", "Lyth", "Scaur", "Rigg")
_ORG_TAIL = ("haven", "stead", "wick", "combe", "mere", "holt", "fold", "ford")
_ORG_KIND = ("Provisioning", "Outfitters", "Instruments", "Textiles", "Hardware", "Ceramics", "Optics", "Joinery")
_FIRM_HEAD = (
    "Bram",
    "Cinder",
    "Dusk",
    "Ember",
    "Fallow",
    "Gorse",
    "Heron",
    "Inkle",
    "Juniper",
    "Kestle",
    "Larch",
    "Marten",
    "Nettle",
    "Osier",
    "Pike",
    "Quarry",
    "Rook",
    "Sedge",
    "Teasel",
    "Umber",
    "Vetch",
    "Yarrow",
)
_FIRM_TAIL = ("brook", "gate", "hurst", "ley", "ridge", "shaw", "worth", "field")
_FIRM_KIND = (
    "Kilnworks",
    "Ropery",
    "Glassworks",
    "Cartage",
    "Millwrights",
    "Chandlery",
    "Cooperage",
    "Foundry",
    "Apiary",
    "Bindery",
    "Saddlery",
    "Tilery",
)
_FIRST = (
    "Ardith",
    "Belen",
    "Corvin",
    "Delphine",
    "Emrys",
    "Fennick",
    "Galen",
    "Hesper",
    "Isolde",
    "Jarrah",
    "Kestrel",
    "Linnea",
    "Marlow",
    "Nerys",
    "Oriel",
    "Perrin",
    "Quilla",
    "Sabine",
    "Tamsin",
    "Ulric",
    "Veda",
    "Ysolde",
    "Anselm",
    "Briony",
    "Caspian",
    "Elowen",
    "Florian",
    "Greer",
    "Ione",
    "Lowri",
)
_LAST = (
    "Ashdown",
    "Brackwell",
    "Corrigan",
    "Dunmore",
    "Elsworth",
    "Farrow",
    "Garrick",
    "Hollis",
    "Ingram",
    "Jessop",
    "Kendrick",
    "Lockhart",
    "Marchbank",
    "Norwood",
    "Oakes",
    "Pembury",
    "Quarrie",
    "Radley",
    "Stroud",
    "Thackeray",
    "Upton",
    "Whitlow",
    "Yelland",
    "Avenell",
)
_LABEL_NAMES = ("Receipts", "Trips", "Archive/2025", "Chase later", "Board papers", "Newsletters")
_REGIONS = ("north", "south", "east", "west")
_TEAMS = ("Assembly", "Dispatch", "Fieldwork", "Glazing", "Kiln", "Packing", "Repairs", "Sampling", "Workshop")
_PROJECTS = ("Lantern", "Bellfounder", "Saltmarsh", "Weathervane", "Quayside", "Larkspur", "Millrace")
_CATEGORIES = ("packaging", "freight", "catering", "printing", "cleaning", "maintenance")


def tools_v5() -> list[dict[str, Any]]:
    entries = [
        (
            "workspace_search",
            "Search the API catalog of the workspace's mail, spreadsheet, chat and CRM apps by keywords. Returns "
            "endpoint URLs, parameters, request formats and whether each vendor account is connected; no records.",
            {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                "connected_only": {"type": "boolean"},
            },
            ["query"],
        ),
        (
            "workspace_fetch",
            "Send an HTTP request to a workspace API. params is a JSON-encoded object of query parameters and body "
            "is the JSON-encoded request body. Returns the HTTP status and the response body.",
            {
                "method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"]},
                "url": {"type": "string"},
                "params": {"type": ["string", "null"]},
                "body": {"type": ["string", "null"]},
            },
            ["method", "url"],
        ),
        (
            "workspace_codec",
            "Encode UTF-8 text as base64url, or decode base64url back to text.",
            {
                "operation": {"type": "string", "enum": ["base64url_encode", "base64url_decode"]},
                "text": {"type": "string"},
            },
            ["operation", "text"],
        ),
    ]
    return [
        {
            "type": "function",
            "name": name,
            "description": description,
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": fields,
                "required": required,
                "additionalProperties": False,
            },
        }
        for name, description, fields, required in entries
    ]


def _trim_grid(grid: list[list[str]]) -> list[list[str]]:
    rows = []
    for row in grid:
        row = [str(cell) for cell in row]
        while row and row[-1] == "":
            row.pop()
        rows.append(row)
    while rows and not rows[-1]:
        rows.pop()
    return rows


class _RealBuilder(_Builder):
    """V5 world construction. Scenario, task identities and final-state grading are shared with V4."""

    def __init__(self, split: str, family: str, variant: int):
        self.task_id = f"{REVISION_V5}:{split}:{family}:{variant:04d}"
        digest = hashlib.sha256(f"{ROOT_SEED_V5}:{self.task_id}".encode()).hexdigest()
        self.rng = random.Random(int(digest, 16))
        self.namespace = "gw_" + digest[:10]
        self.family, self.split, self.variant = family, split, variant
        self.vendor = {app: self.rng.choice(APP_VENDORS[app]) for app in APPS}
        self.connected = sorted(self.vendor.values())
        self.records = {name: {collection: {} for collection in APPS[app]} for app, name in self.vendor.items()}
        self.updates: list[dict[str, Any]] = []
        self.creates: list[dict[str, Any]] = []
        self.deletes: list[dict[str, Any]] = []
        self.guards: list[dict[str, Any]] = []
        self.staged: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.serial = 0
        self.clock = 0
        self.labels: dict[str, str] = {}
        self.books: dict[str, dict[str, Any]] = {}
        head, tail, kind = (self.rng.choice(options) for options in (_ORG_HEAD, _ORG_TAIL, _ORG_KIND))
        self.organization = f"{head}{tail} {kind}"
        self.slug = f"{head}{tail}".lower()
        self.domain = f"{self.slug}-{digest[10:14]}.invalid"
        self.token = digest[14:18]
        pairs = self.rng.sample([(first, last) for first in _FIRST for last in _LAST], 40)
        self.staff = [self.person(first, last, self.domain) for first, last in pairs[:10]]
        self.names = pairs[10:]
        self.me = self.staff[0]
        firms = self.rng.sample([(h, t, k) for h in _FIRM_HEAD for t in _FIRM_TAIL for k in _FIRM_KIND], 9)
        self.firms = [
            {
                "name": f"{h}{t} {k}",
                "domain": f"{h.lower()}{t}-{self.token}.invalid",
                "code": (h[:2] + t[:1]).upper(),
            }
            for h, t, k in firms
        ]
        self.context = {
            "mailbox": self.me["email"],
            "chat_user": self.me["user"],
            "organization": self.organization,
            "now": NOW_V5,
        }
        self.fixture()

    # -- records ------------------------------------------------------------
    def ident(self, kind: str, length: int) -> str:
        self.serial += 1
        return hashlib.sha256(f"{self.task_id}:{kind}:{self.serial}".encode()).hexdigest()[:length]

    def put(self, app: str, collection: str, identity: str, fields: dict[str, Any]) -> dict[str, Any]:
        name = self.vendor[app]
        row = {"id": identity, **fields}
        self.records[name][collection][identity] = row
        return {"vendor": name, "collection": collection, "id": identity, "row": row}

    def person(self, first: str, last: str, domain: str) -> dict[str, Any]:
        user = "U" + self.ident("user", 8).upper()
        email = f"{first}.{last}@{domain}".lower()
        self.put(
            "chat",
            "users",
            user,
            {"name": f"{first}.{last}".lower(), "real_name": f"{first} {last}", "email": email, "deleted": False},
        )
        return {"name": f"{first} {last}", "email": email, "user": user}

    def outsider(self, firm: dict[str, Any], chat: bool = False) -> dict[str, Any]:
        first, last = self.names.pop()
        if chat:
            return self.person(first, last, firm["domain"])
        return {"name": f"{first} {last}", "email": f"{first}.{last}@{firm['domain']}".lower(), "user": ""}

    def label(self, name: str) -> str:
        if name not in self.labels:
            identity = f"Label_{len(self.labels) + 1}"
            self.put("mail", "labels", identity, {"name": name, "type": "user"})
            self.labels[name] = identity
        return self.labels[name]

    def mail_in(self, sender: str, subject: str, body: str, *, unread: bool = True) -> dict[str, Any]:
        self.clock += 1
        date = f"2026-11-{2 + self.clock // 9:02d}T{8 + self.clock % 9:02d}:{(self.clock * 7) % 60:02d}:00Z"
        return self.put(
            "mail",
            "messages",
            "18" + self.ident("message", 14),
            {
                "thread": "18" + self.ident("thread", 14),
                "sender": sender,
                "to": [self.me["email"]],
                "cc": [],
                "bcc": [],
                "subject": subject,
                "body": body,
                "labels": sorted({"INBOX", *(["UNREAD"] if unread else [])}),
                "date": date,
            },
        )

    def workbook(self, title: str) -> dict[str, Any]:
        if title not in self.books:
            identity = "1" + self.ident("spreadsheet", 21)
            self.books[title] = self.put(
                "sheets", "spreadsheets", identity, {"title": title, "owner": self.me["email"]}
            )
        return self.books[title]

    def tab(self, book: dict[str, Any], title: str, values: list[list[Any]]) -> dict[str, Any]:
        sheets = self.records[self.vendor["sheets"]]["worksheets"].values()
        index = sum(row["spreadsheet"] == book["id"] for row in sheets)
        identity = str(int(self.ident("worksheet", 7), 16))
        fields = {"spreadsheet": book["id"], "title": title, "index": index, "values": _trim_grid(values)}
        return self.put("sheets", "worksheets", identity, fields)

    def channel(self, name: str, members: list[str], archived: bool = False) -> dict[str, Any]:
        topic = self.rng.choice(("", "Daily coordination", "Questions welcome", "Read the pinned notes"))
        fields = {"name": name, "topic": topic, "members": sorted(set(members)), "archived": archived}
        return self.put("chat", "channels", "C" + self.ident("channel", 8).upper(), fields)

    def crm(self, collection: str, fields: dict[str, Any]) -> dict[str, Any]:
        self.serial += 1
        return self.put("crm", collection, str(100000 + self.serial * 7), fields)

    # -- expected final state ----------------------------------------------
    def current(self, handle: dict[str, Any], name: str) -> Any:
        key = (handle["vendor"], handle["collection"], handle["id"])
        return deepcopy(self.staged.get(key, {}).get(name, handle["row"][name]))

    def change(self, handle: dict[str, Any], **fields: Any) -> None:
        key = (handle["vendor"], handle["collection"], handle["id"])
        self.staged.setdefault(key, {}).update(deepcopy(fields))

    def append_rows(self, tab: dict[str, Any], rows: list[list[Any]]) -> None:
        self.change(tab, values=_trim_grid(self.current(tab, "values") + rows))

    def set_cell(self, tab: dict[str, Any], row: int, column: int, value: str) -> None:
        grid = self.current(tab, "values")
        while len(grid) < row:
            grid.append([])
        grid[row - 1] += [""] * (column - len(grid[row - 1]))
        grid[row - 1][column - 1] = value
        self.change(tab, values=_trim_grid(grid))

    def relabel(self, message: dict[str, Any], add: tuple[str, ...] = (), remove: tuple[str, ...] = ()) -> None:
        labels = set(self.current(message, "labels"))
        self.change(message, labels=sorted((labels - set(remove)) | set(add)))

    def expect_mail(self, to: list[str], subject: str, body: str, thread: str = "") -> None:
        fields = {
            "thread": thread,
            "sender": self.me["email"],
            "to": sorted(to),
            "cc": [],
            "bcc": [],
            "subject": subject,
            "body": body,
            "labels": ["SENT"],
            "date": NOW_V5,
        }
        self.creates.append({"vendor": self.vendor["mail"], "collection": "messages", "fields": fields})

    def expect_post(self, channel: dict[str, Any], text: str) -> None:
        fields = {"channel": channel["id"], "user": self.me["user"], "text": text, "thread_ts": ""}
        self.creates.append({"vendor": self.vendor["chat"], "collection": "messages", "fields": fields})

    def forbid_mail(self, address: str) -> None:
        guard = {"type": "not_sent_to", "vendor": self.vendor["mail"], "address": address}
        if guard not in self.guards:
            self.guards.append(guard)

    def policy(self, title: str, current: list[tuple[str, str]], previous: list[tuple[str, str]]) -> str:
        """Place current and superseded rules in mail or a spreadsheet and say where the current ones are."""
        owner = self.staff[1 + self.rng.randrange(3)]
        if self.rng.random() < 0.5:
            for effective, pairs in (("2026-09-01", previous), ("2026-11-01", current)):
                lines = "\n".join(f"{key}: {value}" for key, value in pairs)
                body = f"Effective: {effective}\n{lines}\n\n{owner['name']}"
                self.mail_in(owner["email"], f"{title} - effective {effective}", body, unread=False)
            return f"the latest '{title}' email from {owner['email']}"
        book = self.workbook(f"{self.organization} Operating Rules")
        self.tab(book, f"{title} (superseded)", [["Setting", "Value"], *map(list, previous)])
        self.tab(book, title, [["Setting", "Value"], *map(list, current)])
        return f"the '{title}' tab of the '{self.organization} Operating Rules' spreadsheet"

    def fixture(self) -> None:
        for name in self.rng.sample(_LABEL_NAMES, 3):
            self.label(name)
        self.channel(f"{self.slug}-general", [person["user"] for person in self.staff])
        self.channel(f"{self.slug}-social", [person["user"] for person in self.staff[::2]])
        for firm in self.firms[-2:]:
            self.mail_in(
                f"news@{firm['domain']}",
                f"{firm['name']} monthly update",
                f"Highlights from {firm['name']} this month.\nNo action is needed.",
                unread=self.rng.random() < 0.5,
            )

    def finish(self, request: str) -> Scenario:
        for (vendor, collection, identity), fields in self.staged.items():
            row = self.records[vendor][collection][identity]
            changed = {key: value for key, value in fields.items() if row[key] != value}
            if changed:
                self.updates.append({"vendor": vendor, "collection": collection, "id": identity, "fields": changed})
        if not self.updates and not self.creates:
            raise ValueError("V5 variant has no required change")
        style = (FAMILIES_V5.index(self.family) + self.variant) % 6
        identity = f"You are {self.me['name']} <{self.me['email']}> in the {self.organization} workspace."
        prompt = (
            f"{identity} {request}",
            f"{request}\n\n{identity}",
            f"Please handle this today.\n{request}\n\n{identity}",
            f"Task\n\n{request}\n\nContext\n{identity}",
            f"{identity}\n\n{request}",
            f"{request}\n{identity}",
        )[style]
        return Scenario(
            self.task_id,
            self.namespace,
            self.family,
            self.split,
            self.variant,
            tuple(self.connected),
            self.records,
            prompt,
            tuple(self.updates),
            tuple(self.creates),
            tuple(self.deletes),
            revision=REVISION_V5,
            guards=tuple(self.guards),
            context=dict(self.context),
        )


def _companies(b: _RealBuilder, firms: list[dict[str, Any]], tiers: tuple[str, ...]) -> list[dict[str, Any]]:
    """CRM company, contact and deal records for firms; returns one bundle per firm."""
    bundles = []
    for i, firm in enumerate(firms):
        tier = tiers[i] if i < len(tiers) else b.rng.choice(tiers)
        owner = b.staff[1 + i % 5]["email"]
        company = b.crm(
            "companies",
            {
                "name": firm["name"],
                "domain": firm["domain"],
                "tier": tier,
                "region": b.rng.choice(_REGIONS),
                "owner": owner,
            },
        )
        contacts = {}
        for role in ("billing", "operations"):
            person = b.outsider(firm)
            contacts[role] = b.crm(
                "contacts",
                {
                    "name": person["name"],
                    "email": person["email"],
                    "company": firm["name"],
                    "role": role,
                    "lifecycle": "customer",
                    "owner": owner,
                },
            )
        deal = b.crm(
            "deals",
            {
                "name": f"{firm['name']} supply agreement",
                "company": firm["name"],
                "amount": b.rng.randrange(120, 480) * 100,
                "stage": b.rng.choice(("qualified", "proposal", "negotiation")),
                "owner": owner,
                "close_date": f"2026-12-{b.rng.randint(1, 28):02d}",
            },
        )
        b.crm(
            "deals",
            {
                "name": f"{firm['name']} pilot order",
                "company": firm["name"],
                "amount": b.rng.randrange(20, 90) * 100,
                "stage": b.rng.choice(("closed_won", "closed_lost")),
                "owner": owner,
                "close_date": f"2026-0{b.rng.randint(3, 9)}-{b.rng.randint(1, 28):02d}",
            },
        )
        bundles.append({"firm": firm, "company": company, "contacts": contacts, "deal": deal, "tier": tier})
    return bundles


def _rx_invoice_intake(b: _RealBuilder) -> str:
    suppliers = b.firms[:5]
    blocked = suppliers[: b.rng.choice((1, 2))]
    allowed = suppliers[len(blocked) :]
    limit = b.rng.choice((2500, 4000, 6000))
    where = b.policy(
        "Payables rules",
        [("approval_limit", str(limit)), ("blocked_domains", ", ".join(x["domain"] for x in blocked))],
        [("approval_limit", str(limit + b.rng.choice((-1500, 2000)))), ("blocked_domains", allowed[-1]["domain"])],
    )
    book = b.workbook(f"{b.organization} Payables")
    earlier = allowed[0]
    logged = [f"{earlier['code']}-{b.rng.randint(10000, 49999)}", earlier["domain"], str(limit // 2), "approve"]
    log = b.tab(book, "Intake Log", [["Invoice", "Supplier domain", "Amount", "Decision"], logged])
    clerk = b.staff[2]
    channel = b.channel(f"{b.slug}-payables", [b.me["user"], clerk["user"]])
    plan = [(earlier, int(logged[2]), False, logged[0]), (blocked[0], limit - 100, True, None)]
    amounts = [limit, limit + 1, *b.rng.sample((limit // 2, limit - 750, limit + 900, limit * 2), b.rng.randint(1, 3))]
    for i, amount in enumerate(amounts):
        plan.append((allowed[i % len(allowed)], amount, True, None))
    rows = []
    numbers = {logged[0]}
    for supplier, amount, unread, number in plan:
        while number is None or number in numbers and not (supplier is earlier and not unread):
            number = f"{supplier['code']}-{b.rng.randint(50000, 99999)}"
        numbers.add(number)
        body = (
            f"Invoice: {number}\nAmount due: {amount}\nDue date: 2026-12-{b.rng.randint(1, 28):02d}\n\n"
            f"Thank you,\n{supplier['name']} accounts"
        )
        message = b.mail_in(f"billing@{supplier['domain']}", f"Invoice {number}", body, unread=unread)
        b.forbid_mail(f"billing@{supplier['domain']}")
        if unread and supplier not in blocked:
            rows.append([number, supplier["domain"], str(amount), "approve" if amount <= limit else "review"])
            b.relabel(message, remove=("UNREAD",))
    b.append_rows(log, sorted(rows))
    b.expect_post(channel, f"{len(rows)} invoices logged")
    return (
        f"Process the unread supplier invoice emails in the mailbox under the current payables rules ({where}). "
        f"Append one row per invoice to the 'Intake Log' tab of the '{b.organization} Payables' spreadsheet: "
        "invoice number, the sender's email domain, the amount due as a plain integer, and 'approve' if the amount "
        "is at or below the approval limit or 'review' otherwise. Add the rows after the existing ones, sorted by "
        "invoice number. Skip invoices from blocked supplier domains entirely and leave those emails unread; mark "
        "each logged email read by removing its UNREAD label. Then post exactly '<n> invoices logged', where n is "
        f"the number of rows you added, to the #{channel['row']['name']} chat channel. Do not send any email."
    )


def _rx_escalation_routing(b: _RealBuilder) -> str:
    tiers = ("gold", "silver", "bronze")
    bundles = _companies(b, b.firms[:6], ("gold", "silver", "bronze"))
    style = b.rng.choice(("{tier}-escalations@{domain}", "escalations-{tier}@{domain}", "{tier}.desk@{domain}"))
    boxes = {tier: style.format(tier=tier, domain=b.domain) for tier in tiers}
    rotated = dict(zip(tiers, [boxes[t] for t in tiers[1:] + tiers[:1]]))
    where = b.policy("Escalation desk routing", list(boxes.items()), list(rotated.items()))
    chosen = b.rng.sample(bundles, b.rng.randint(2, 3))
    handled = next(x for x in bundles if x not in chosen)
    for bundle, unread in [(handled, False)] + [(x, True) for x in chosen]:
        contact = bundle["contacts"]["operations"]["row"]
        order = f"SO-{b.rng.randint(100000, 999999)}"
        body = (
            f"Hello,\n\nOrder {order} arrived damaged and we need it replaced quickly.\n\n"
            f"{contact['name']}\n{bundle['firm']['name']}"
        )
        b.mail_in(contact["email"], f"Problem with order {order}", body, unread=unread)
        b.forbid_mail(contact["email"])
        if unread:
            body = f"{bundle['firm']['name']} | {contact['email']} | {order}"
            b.expect_mail([boxes[bundle["tier"]]], f"Escalation {order}", body)
            b.change(bundle["deal"], stage="at_risk")
    return (
        "Route each unread 'Problem with order ...' email using the current escalation desk routing "
        f"({where}). Find the sender in the CRM, take their company's tier, and send one new email to that tier's "
        "escalation mailbox with subject 'Escalation <order number>' and body exactly "
        "'<company name> | <sender email> | <order number>'. Then set the stage of that company's open deal (the "
        "one not closed_won or closed_lost) to 'at_risk'. Do not reply to the customers, and leave the complaint "
        "emails unread."
    )


def _rx_renewal_quotes(b: _RealBuilder) -> str:
    tiers = ("gold", "silver", "bronze")
    bundles = _companies(b, b.firms[:6], ("gold", "silver", "bronze", "gold"))
    current = dict(zip(tiers, b.rng.sample((4, 6, 8, 10, 12, 15), 3)))
    previous = {tier: value + 5 for tier, value in current.items()}
    where = b.policy(
        "Renewal discount schedule",
        [(f"{tier}_discount_percent", str(value)) for tier, value in current.items()],
        [(f"{tier}_discount_percent", str(value)) for tier, value in previous.items()],
    )
    chosen = b.rng.sample(bundles, 4)
    chosen[0]["contacts"]["billing"]["row"]["lifecycle"] = "opted_out"
    for bundle in chosen:
        contact = bundle["contacts"]["billing"]["row"]
        deal = bundle["deal"]["row"]
        subject = f"Renewal quote for {bundle['firm']['name']}"
        body = f"Hi,\n\nPlease send the renewal quote for {deal['name']}.\n\nBest,\n{contact['name']}"
        message = b.mail_in(contact["email"], subject, body)
        if contact["lifecycle"] == "opted_out":
            b.forbid_mail(contact["email"])
            continue
        price = deal["amount"] * (100 - current[bundle["tier"]]) // 100
        b.expect_mail(
            [contact["email"]], f"Re: {subject}", f"Quote for {deal['name']}: {price}", message["row"]["thread"]
        )
        b.change(bundle["deal"], amount=price, stage="quoted")
    return (
        "Answer each unread renewal-quote request in the mailbox. Apply the discount for the requesting company's "
        f"CRM tier from the current renewal discount schedule ({where}) to the amount of the company's open deal "
        "(the one not closed_won or closed_lost), rounding down to a whole number. Reply in the request's thread "
        "to its sender with subject 'Re: <original subject>' and body exactly 'Quote for <deal name>: <discounted "
        "amount>', then set that deal's amount to the discounted amount and its stage to 'quoted'. Skip requests "
        "from contacts whose CRM lifecycle is 'opted_out': do not email them or change their deals."
    )


def _rx_roster_invites(b: _RealBuilder) -> str:
    project = b.rng.choice(_PROJECTS)
    affiliations = ("staff", "partner", "contractor")
    allowed = b.rng.choice((("staff", "partner"), ("staff", "contractor"), ("partner", "staff", "contractor")))
    previous = ("staff",) if len(allowed) > 2 else tuple(x for x in affiliations if x not in allowed) + ("staff",)
    where = b.policy(
        "Channel access rules",
        [("allowed_affiliations", ", ".join(allowed))],
        [("allowed_affiliations", ", ".join(previous))],
    )
    people = [(person, "staff") for person in b.staff[3:7]]
    people += [(b.outsider(b.firms[i], chat=True), affiliations[1 + i % 2]) for i in range(4)]
    b.rng.shuffle(people)
    statuses = ["active"] * 6 + ["inactive"] * 2
    b.rng.shuffle(statuses)
    members = [b.me["user"], people[0][0]["user"]]
    channel = b.channel(f"{b.slug}-{project.lower()}", members)
    owner = b.staff[1]
    if not any(
        status == "active" and affiliation in allowed and person["user"] not in members
        for (person, affiliation), status in zip(people, statuses)
    ):
        # Every variant requires at least one invitation.
        people[-1], statuses[-1] = (people[-1][0], allowed[0]), "active"
    roster = [["Name", "Email", "Status", "Affiliation"]]
    invited = []
    for (person, affiliation), status in zip(people, statuses):
        roster.append([person["name"], person["email"], status, affiliation])
        if status == "active" and affiliation in allowed and person["user"] not in members:
            invited.append(person)
    book = b.workbook(f"{b.organization} {project} Roster")
    b.tab(book, "Roster", roster)
    b.change(channel, members=sorted(set(members) | {x["user"] for x in invited}))
    b.expect_mail([owner["email"]], f"{project} roster synced", "\n".join(sorted(x["email"] for x in invited)))
    return (
        f"Sync the #{channel['row']['name']} chat channel with the 'Roster' tab of the '{book['row']['title']}' "
        "spreadsheet: invite every roster person whose Status is 'active', whose Affiliation is allowed by the "
        f"current channel access rules ({where}), and who is not already a member. Chat invitations take user IDs; "
        f"match people by email. Do not remove anyone. Then email {owner['email']} with subject "
        f"'{project} roster synced' and a body listing the emails of the people you invited, sorted alphabetically, "
        "one per line."
    )


def _rx_label_holds(b: _RealBuilder) -> str:
    hold = b.label("Supplier quarantine")
    held, others = b.firms[:2], b.firms[2:5]
    where = b.policy(
        "Mail hold rules",
        [("held_domains", ", ".join(x["domain"] for x in held))],
        [("held_domains", ", ".join(x["domain"] for x in others[:2]))],
    )
    book = b.workbook(f"{b.organization} Compliance Log")
    earlier = f"accounts@{others[2]['domain']}"
    log = b.tab(book, "Holds", [["Date", "Sender", "Subject"], ["2026-10-14", earlier, "Statement for September"]])
    channel = b.channel(f"{b.slug}-compliance", [b.me["user"], b.staff[4]["user"]])
    topics = ("Updated bank details", "Overdue balance notice", "Revised price list", "Contract renewal terms")
    picks = [held[0], held[1], held[b.rng.randrange(2)], *b.rng.sample(others, 2)]
    b.rng.shuffle(picks)
    rows = []
    for firm in picks:
        sender = f"{b.rng.choice(('accounts', 'sales', 'notices'))}@{firm['domain']}"
        subject = f"{b.rng.choice(topics)} ({firm['name']})"
        message = b.mail_in(
            sender, subject, f"Please review the attached terms.\n\n{firm['name']}", unread=b.rng.random() < 0.6
        )
        if firm in held:
            b.relabel(message, add=(hold,), remove=("INBOX",))
            rows.append([message["row"]["date"][:10], sender, subject, message["row"]["date"]])
    rows.sort(key=lambda row: row[3])
    b.append_rows(log, [row[:3] for row in rows])
    b.expect_post(channel, f"{len(rows)} messages held")
    return (
        f"Apply the current mail hold rules ({where}). Give every inbox message sent from a held domain the "
        "'Supplier quarantine' label and take it out of the inbox by removing the INBOX label; leave its read/unread state "
        "alone. Labels are changed by label ID. Append one row per held message to the 'Holds' tab of the "
        f"'{b.organization} Compliance Log' spreadsheet: date received (YYYY-MM-DD), sender email and subject, "
        "oldest message first, after the existing rows. Finally post exactly '<n> messages held' to "
        f"#{channel['row']['name']}. Do not send any email."
    )


def _rx_payment_reminders(b: _RealBuilder) -> str:
    bundles = _companies(b, b.firms[:7], ("strategic", "standard", "standard", "standard"))
    days = b.rng.choice((30, 45, 60))
    where = b.policy(
        "Collections policy",
        [("reminder_after_days", str(days)), ("exempt_tiers", "strategic")],
        [("reminder_after_days", str(days + 15)), ("exempt_tiers", "none")],
    )
    book = b.workbook(f"{b.organization} Receivables")
    grid = [["Invoice", "Company", "Amount", "Days Overdue", "Status"]]
    overdue = [days + 20, days, days - 1, days + 5, days + 40, days - 10, days + 2]
    statuses = ["open", "open", "open", "paid", "open", "open", "reminded"]
    targets = []
    for i, bundle in enumerate(bundles):
        invoice = f"AR-{b.rng.randint(20000, 99999)}"
        amount = b.rng.randrange(300, 9000)
        grid.append([invoice, bundle["firm"]["name"], str(amount), str(overdue[i]), statuses[i]])
        b.forbid_mail(bundle["contacts"]["operations"]["row"]["email"])
        if statuses[i] == "open" and overdue[i] >= days and bundle["tier"] != "strategic":
            targets.append((i + 2, invoice, amount, overdue[i], bundle))
        elif bundle["tier"] == "strategic":
            b.forbid_mail(bundle["contacts"]["billing"]["row"]["email"])
    tab = b.tab(book, "Open Items", grid)
    for row, invoice, amount, late, bundle in targets:
        billing = bundle["contacts"]["billing"]["row"]["email"]
        b.expect_mail(
            [billing], f"Payment reminder {invoice}", f"Invoice {invoice} for {amount} is {late} days overdue."
        )
        b.set_cell(tab, row, 5, "reminded")
    return (
        f"Send payment reminders under the current collections policy ({where}). For each row of the 'Open Items' "
        f"tab in the '{b.organization} Receivables' spreadsheet whose Status is 'open' and whose Days Overdue is at "
        "or above the policy threshold, email the company's billing contact (the CRM contact with role 'billing') "
        "with subject 'Payment reminder <invoice>' and body exactly 'Invoice <invoice> for <amount> is <days> days "
        "overdue.', then change that row's Status cell to 'reminded'. Companies in an exempt CRM tier get nothing. "
        "Change no other cells."
    )


def _rx_deal_handoff(b: _RealBuilder) -> str:
    leaving, successor, other = b.staff[2], b.staff[3], b.staff[4]
    where = b.policy(
        "Account handoff",
        [("from_owner", leaving["email"]), ("to_owner", successor["email"]), ("scope", "open deals only")],
        [("from_owner", leaving["email"]), ("to_owner", other["email"]), ("scope", "open deals only")],
    )
    stages = [
        "qualified",
        "proposal",
        "negotiation",
        "closed_won",
        "closed_lost",
        "proposal",
        "qualified",
        "negotiation",
    ]
    owners = [leaving, leaving, leaving, leaving, leaving, other, successor, b.staff[5]]
    if b.rng.random() < 0.5:
        stages[5], owners[5] = "negotiation", leaving
    moved = []
    for firm, stage, owner in zip(b.firms[:8], stages, owners):
        deal = b.crm(
            "deals",
            {
                "name": f"{firm['name']} {b.rng.choice(('framework', 'restock', 'service plan'))}",
                "company": firm["name"],
                "amount": b.rng.randrange(40, 400) * 100,
                "stage": stage,
                "owner": owner["email"],
                "close_date": f"2026-12-{b.rng.randint(1, 28):02d}",
            },
        )
        if owner is leaving and not stage.startswith("closed"):
            b.change(deal, owner=successor["email"])
            moved.append(deal["row"]["name"])
    book = b.workbook(f"{b.organization} Sales Ops")
    tab = b.tab(book, "Reassignments", [["Deal", "Previous owner", "New owner"]])
    channel = b.channel(f"{b.slug}-sales-desk", [b.me["user"], leaving["user"], successor["user"]])
    b.append_rows(tab, [[name, leaving["email"], successor["email"]] for name in sorted(moved)])
    b.expect_post(channel, f"{len(moved)} deals moved to {successor['email']}")
    return (
        f"Carry out the current account handoff ({where}). In the CRM, reassign the deals it covers from the "
        "departing owner to the new owner; closed deals (closed_won or closed_lost) stay where they are. Append one "
        "row per reassigned deal (deal name, previous owner email, new owner email) to the 'Reassignments' tab of "
        f"the '{b.organization} Sales Ops' spreadsheet, sorted by deal name, and post exactly '<n> deals moved to "
        f"<new owner email>' in #{channel['row']['name']}."
    )


def _rx_signup_import(b: _RealBuilder) -> str:
    owner, other = b.staff[3], b.staff[4]
    excluded = b.firms[5]
    where = b.policy(
        "Lead intake rules",
        [("lead_owner", owner["email"]), ("excluded_domains", excluded["domain"])],
        [("lead_owner", other["email"]), ("excluded_domains", b.firms[6]["domain"])],
    )
    existing = b.outsider(b.firms[0])
    b.crm(
        "contacts",
        {
            "name": existing["name"],
            "email": existing["email"],
            "company": b.firms[0]["name"],
            "role": "operations",
            "lifecycle": "customer",
            "owner": other["email"],
        },
    )
    entries = [(existing, b.firms[0], "duplicate"), (b.outsider(excluded), excluded, "excluded")]
    entries += [(b.outsider(firm), firm, "created") for firm in b.rng.sample(b.firms[1:5], b.rng.randint(2, 3))]
    b.rng.shuffle(entries)
    book = b.workbook(f"{b.organization} Marketing")
    tab = b.tab(book, "Signup ledger", [["Email", "Result"]])
    for person, firm, result in entries:
        body = f"Name: {person['name']}\nEmail: {person['email']}\nCompany: {firm['name']}\n"
        message = b.mail_in(f"forms@{b.domain}", f"New signup: {person['name']}", body)
        b.relabel(message, remove=("UNREAD",))
        if result == "created":
            fields = {
                "name": person["name"],
                "email": person["email"],
                "company": firm["name"],
                "role": "unknown",
                "lifecycle": "lead",
                "owner": owner["email"],
            }
            b.creates.append({"vendor": b.vendor["crm"], "collection": "contacts", "fields": fields})
    b.append_rows(tab, sorted([person["email"], result] for person, _, result in entries))
    return (
        f"Import the unread 'New signup' emails into the CRM following the current lead intake rules ({where}). "
        "For each signup whose email is not already a CRM contact and whose email domain is not excluded, create a "
        "contact with the name, email and company from the email, role 'unknown', lifecycle 'lead' and owner set to "
        "the rules' lead owner. Record every signup in the 'Signup ledger' tab of the "
        f"'{b.organization} Marketing' spreadsheet as an '<email>', '<result>' row, sorted by email, where result "
        "is 'created', 'duplicate' or 'excluded'. Mark each signup email read. Do not send any email."
    )


def _rx_budget_alerts(b: _RealBuilder) -> str:
    percent = b.rng.choice((75, 80, 90))
    where = b.policy("Budget alert policy", [("alert_percent", str(percent))], [("alert_percent", str(percent - 15))])
    teams = sorted(b.rng.sample(_TEAMS, 6))
    owners = b.staff[2:8]
    ratios = [percent, percent - 1, percent + 12, percent - 20, 100, percent + 5]
    flags = ["", "", "", "", "", "alerted"]
    order = list(range(6))
    b.rng.shuffle(order)
    grid = [["Team", "Owner", "Budget", "Spent", "Flag"]]
    alerted = []
    for slot in order:
        team, owner = teams[slot], owners[slot]
        budget = b.rng.randrange(20, 90) * 1000
        spent = budget * ratios[slot] // 100
        grid.append([team, owner["email"], str(budget), str(spent), flags[slot]])
        if spent * 100 >= budget * percent and not flags[slot]:
            alerted.append((len(grid), team, owner, spent, budget))
        else:
            b.forbid_mail(owner["email"])
    book = b.workbook(f"{b.organization} Budgets")
    tab = b.tab(book, "Teams", grid)
    channel = b.channel(f"{b.slug}-finance-desk", [b.me["user"], b.staff[1]["user"]])
    for row, team, owner, spent, budget in alerted:
        b.expect_mail([owner["email"]], f"Budget alert: {team}", f"{team} has spent {spent} of {budget}.")
        b.set_cell(tab, row, 5, "alerted")
    b.expect_post(channel, "Budget alerts: " + ", ".join(sorted(x[1] for x in alerted)))
    return (
        f"Run the budget check from the current budget alert policy ({where}) on the 'Teams' tab of the "
        f"'{b.organization} Budgets' spreadsheet. For each team whose Spent is at least the alert percentage of its "
        "Budget and whose Flag cell is empty, email the team's Owner with subject 'Budget alert: <team>' and body "
        "exactly '<team> has spent <spent> of <budget>.', and set its Flag cell to 'alerted'. Then post exactly "
        f"'Budget alerts: <teams>' to #{channel['row']['name']}, listing those teams alphabetically separated by "
        "', '. Leave every other cell unchanged."
    )


def _rx_offboarding(b: _RealBuilder) -> str:
    leaving = b.staff[5 : 5 + b.rng.randint(1, 2)]
    cancelled, successor = b.staff[8], b.staff[2]
    where = b.policy(
        "Offboarding notice",
        [("departing", ", ".join(x["email"] for x in leaving)), ("crm_successor", successor["email"])],
        [("departing", cancelled["email"]), ("crm_successor", b.staff[3]["email"])],
    )
    removed = {person["email"]: 0 for person in leaving}
    for i, name in enumerate(("design-review", "stockroom", "night-shift", "suppliers", "launch-2025")):
        archived = name == "launch-2025"
        members = [b.me["user"], cancelled["user"], b.staff[1 + i]["user"]]
        members += [person["user"] for j, person in enumerate(leaving) if (i + j) % 2 == 0 or archived]
        channel = b.channel(f"{b.slug}-{name}", members, archived=archived)
        gone = {person["user"] for person in leaving if person["user"] in channel["row"]["members"]}
        if gone and not archived:
            for person in leaving:
                removed[person["email"]] += person["user"] in gone
            b.change(channel, members=sorted(set(channel["row"]["members"]) - gone))
    for i, firm in enumerate(b.firms[:6]):
        owner = (leaving + [cancelled, b.staff[1], b.staff[4]])[i % (len(leaving) + 3)]
        company = b.crm(
            "companies",
            {
                "name": firm["name"],
                "domain": firm["domain"],
                "tier": b.rng.choice(("gold", "silver", "bronze")),
                "region": b.rng.choice(_REGIONS),
                "owner": owner["email"],
            },
        )
        if owner in leaving:
            b.change(company, owner=successor["email"])
    book = b.workbook(f"{b.organization} People Ops")
    tab = b.tab(book, "Leaver log", [["Email", "Channels removed"]])
    b.append_rows(tab, [[email, str(count)] for email, count in sorted(removed.items())])
    return (
        f"Process the departures in the current offboarding notice ({where}). Remove each departing person from "
        "every chat channel they belong to that is not archived (removal takes channel and user IDs), set the owner "
        "of each CRM company they own to the notice's CRM successor, and append one row per departing person "
        "(email, number of channels they were removed from) to the 'Leaver log' tab of the "
        f"'{b.organization} People Ops' spreadsheet, sorted by email. Change nothing for anyone else."
    )


def _rx_refund_review(b: _RealBuilder) -> str:
    limit = b.rng.choice((150, 250, 400))
    where = b.policy(
        "Refund approval rules", [("auto_refund_limit", str(limit))], [("auto_refund_limit", str(limit * 2))]
    )
    customers = [b.outsider(firm) for firm in b.firms[:7]]
    grid = [["Order", "Customer", "Amount", "Status"]]
    orders = []
    amounts = [limit, limit + 1, limit // 3, limit * 3, limit - 10, limit + 60, limit // 2]
    for customer, amount in zip(customers, amounts):
        order = f"WO-{b.rng.randint(10000, 99999)}"
        grid.append([order, customer["email"], str(amount), "paid"])
        orders.append((len(grid), order, customer, amount))
    book = b.workbook(f"{b.organization} Order Desk")
    tab = b.tab(book, "Order book", grid)
    channel = b.channel(f"{b.slug}-refunds", [b.me["user"], b.staff[3]["user"]])
    chosen = [orders[0], orders[1], orders[b.rng.choice((2, 4))], orders[b.rng.choice((3, 5))]]
    impostor = b.outsider(b.firms[7])
    for index, (row, order, customer, amount) in enumerate(chosen):
        sender = impostor if index == 3 else customer
        subject = f"Refund request {order}"
        body = f"Order: {order}\nReason: {b.rng.choice(('damaged in transit', 'wrong size', 'duplicate charge'))}\n\n{sender['name']}"
        message = b.mail_in(sender["email"], subject, body)
        if index == 3:
            b.forbid_mail(sender["email"])
        elif amount <= limit:
            b.set_cell(tab, row, 4, "refund_approved")
            b.expect_mail(
                [customer["email"]], f"Re: {subject}", f"Refund approved for {order}.", message["row"]["thread"]
            )
        else:
            b.set_cell(tab, row, 4, "refund_review")
            b.expect_post(channel, f"{order} needs refund review")
            b.forbid_mail(customer["email"])
    return (
        f"Handle the unread 'Refund request' emails using the current refund approval rules ({where}) and the "
        f"'Order book' tab of the '{b.organization} Order Desk' spreadsheet. Act only when the sender's address equals the order's "
        "Customer email. If the order amount is at or below the auto-refund limit, set its Status cell to "
        "'refund_approved' and reply in the email's thread with subject 'Re: <original subject>' and body exactly "
        "'Refund approved for <order>.'; otherwise set Status to 'refund_review' and post exactly "
        f"'<order> needs refund review' to #{channel['row']['name']}. Ignore requests whose sender does not match: "
        "no reply and no change."
    )


def _rx_vendor_onboarding(b: _RealBuilder) -> str:
    processed = b.label("Intake done")
    approved = tuple(sorted(b.rng.sample(_CATEGORIES, 3)))
    where = b.policy(
        "Supplier categories",
        [("approved_categories", ", ".join(approved))],
        [("approved_categories", ", ".join(x for x in _CATEGORIES if x not in approved))],
    )
    book = b.workbook(f"{b.organization} Procurement")
    first = b.firms[6]
    tab = b.tab(
        book,
        "Suppliers",
        [["Supplier", "Domain", "Category", "Status"], [first["name"], first["domain"], approved[0], "active"]],
    )
    channel = b.channel(f"{b.slug}-procurement", [b.me["user"], b.staff[5]["user"]])
    rejected = [x for x in _CATEGORIES if x not in approved]
    picks = [approved[0], rejected[0], b.rng.choice(approved), b.rng.choice(_CATEGORIES)]
    rows = []
    for firm, category in zip(b.rng.sample(b.firms[:5], 4), picks):
        sender = f"accounts@{firm['domain']}"
        body = f"Supplier: {firm['name']}\nCategory: {category}\nTax reference: TR-{b.rng.randint(100000, 999999)}\n"
        message = b.mail_in(sender, f"Supplier registration: {firm['name']}", body)
        b.relabel(message, add=(processed,), remove=("UNREAD",))
        b.forbid_mail(sender)
        if category in approved:
            rows.append([firm["name"], firm["domain"], category, "pending_review"])
    b.append_rows(tab, sorted(rows))
    b.expect_post(channel, f"{len(rows)} suppliers registered")
    return (
        "Review the unread 'Supplier registration' emails against the current supplier categories "
        f"({where}). Append each supplier whose category is approved to the 'Suppliers' tab of the "
        f"'{b.organization} Procurement' spreadsheet as a row of supplier name, the sender's email domain, "
        "category and 'pending_review', sorted by supplier name, after the existing rows. Give every registration "
        "email the 'Intake done' label and mark it read (labels are changed by label ID), post exactly "
        f"'<n> suppliers registered' to #{channel['row']['name']}, and do not reply to the suppliers."
    )


_REAL_WORKFLOWS = {family: globals()[f"_{family}"] for family in FAMILIES_V5}
_REVISIONS = {
    REVISION: (ROOT_SEED, FAMILIES, TRAIN_VARIANTS, VAL_VARIANTS),
    REVISION_V5: (ROOT_SEED_V5, FAMILIES_V5, TRAIN_VARIANTS_V5, VAL_VARIANTS_V5),
}


def scenario_from_id(task_id: str) -> Scenario:
    parts = task_id.split(":")
    if (
        len(parts) != 4
        or parts[0] not in _REVISIONS
        or parts[1] not in {"train", "val"}
        or parts[2] not in _REVISIONS[parts[0]][1]
    ):
        raise ValueError("unknown API workspace task identity")
    try:
        variant = int(parts[3])
    except ValueError as error:
        raise ValueError("invalid task variant") from error
    _, _, train, val = _REVISIONS[parts[0]]
    bound = train if parts[1] == "train" else val
    if not 0 <= variant < bound or parts[3] != f"{variant:04d}":
        raise ValueError("task variant outside immutable V3 corpus")
    if parts[0] == REVISION:
        builder = _Builder(parts[1], parts[2], variant)
        return builder.finish(_workflow(builder))
    real = _RealBuilder(parts[1], parts[2], variant)
    return real.finish(_REAL_WORKFLOWS[parts[2]](real))


def task_ids(split: str, revision: str = REVISION) -> tuple[str, ...]:
    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    if revision not in _REVISIONS:
        raise ValueError("unknown API workspace revision")
    _, families, train, val = _REVISIONS[revision]
    count = train if split == "train" else val
    if revision == REVISION:
        families = tuple(FAMILIES[offset + domain * 5] for offset in range(5) for domain in range(6))
    return tuple(f"{revision}:{split}:{family}:{variant:04d}" for variant in range(count) for family in families)


def row_from_id(task_id: str) -> dict[str, Any]:
    scenario = scenario_from_id(task_id)
    real = scenario.revision == REVISION_V5
    return {
        "id": task_id,
        "task_id": task_id,
        "data_source": "api_workspace",
        "agent_ref": {"type": "responses_api_agents", "name": "api_workspace_agent"},
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT_V5 if real else SYSTEM_PROMPT},
                {"role": "user", "content": scenario.prompt},
            ],
            "tools": tools_v5() if real else tools(),
        },
        "metadata": {
            "split": scenario.split,
            "family": scenario.family,
            "revision": scenario.revision,
            "seed": ROOT_SEED_V5 if real else ROOT_SEED,
            "authoring": "independent_procedural_fiction",
        },
    }
