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
from dataclasses import dataclass
from typing import Any

from resources_servers.api_workspace.engine import COLLECTIONS, VENDOR_NAMES, VENDORS, canonical_json


REVISION = "api-workspace-v3"
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
You have at most50 model turns. Completion is determined by the resulting workspace state.
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
                    "coordinator": self.manager,
                    "operations_folder": self.folder_name,
                    "dispatch_channel": self.channel_name,
                    "new_records_vendor_order": self.rng.choice(("alphabetical_first", "alphabetical_last")),
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
                            "requested_contact": contact,
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
            f"Activate pending employees in {b.department}, setting location and cost_center from policy. "
            f"Save '{subject}.txt' with their emails sorted one per line in the operations folder; use coordinator owner, "
            "draft status, private access, no sharing, version1. Leave other employees untouched."
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
            f"Approve {email}'s requested November leave, with {delegate} as delegate. Transfer only their CRM "
            "account's owner to that delegate during leave; do not alter the employee record."
        )
    if f == "hr_conversion":
        b.update(employee, employment_type="employee", cost_center=p["cost_center"])
        b.update(rows["expenses"][target], category="onboarding")
        return (
            f"Convert contractor {email} to employment_type 'employee' using policy cost_center, and recategorize "
            "their expense as onboarding. Preserve its approval status and amount."
        )
    if f == "hr_departure":
        b.update(employee, status="inactive")
        owned = [x for x in accounts if x["row"]["owner"] == email]
        for x in owned:
            b.update(x, owner=b.manager)
        b.update(file, owner=b.manager)
        return (
            f"Offboard {email}: set HR status inactive, transfer all their CRM accounts and their account JSON file "
            "to the coordinator, preserving everything else. Do not delete historical records."
        )
    if f == "crm_renewal":
        opp = rows["opportunities"][target]
        amount = opp["row"]["amount"] * (100 - p["renewal_discount_percent"]) // 100
        b.update(opp, amount=amount, stage="proposal", close_date=p["effective_date"])
        b.draft(subject, str(amount), [rows["contacts"][target]["row"]["email"]])
        return (
            f"For {company}'s renewal opportunity apply policy renewal_discount_percent to amount, round down "
            "to integer cents, set stage proposal and close_date to policy effective_date. Create a draft to the CRM "
            f"contact with subject '{subject}', body containing only the discounted integer, empty thread and draft status."
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
            "set those accounts' tier priority. Keep opportunities' amounts and stages unchanged."
        )
    if f == "crm_contacts":
        new_email = json.loads(mail["row"]["body"])["requested_contact"]
        b.update(rows["contacts"][target], email=new_email, status="verified")
        b.update(mail, status="read", labels=["contact-processed"])
        return (
            f"Read the mail request for {company}; verify the matching CRM contact using its requested_contact "
            "email. Mark only that request read with labels exactly [contact-processed]."
        )
    if f == "crm_pipeline":
        selected = [x for x in rows["opportunities"] if x["row"]["probability"] >= 60]
        for x in selected:
            b.update(x, stage="review")
        body = "\n".join(sorted(f"{x['row']['account']}={x['row']['amount']}" for x in selected))
        b.post(body)
        return (
            "Move qualified opportunities with probability at least60 to stage review. Post their "
            "account=amount pairs sorted by account, one per line, to dispatch; coordinator sender, empty thread, posted status. "
            "Do not include lower-probability opportunities."
        )
    if f == "crm_dormant":
        b.update(account, status="paused")
        b.update(rows["opportunities"][target], stage="on_hold", probability=0)
        b.update(mail, labels=["account-paused"])
        return (
            f"Pause {company}'s account, put its renewal on_hold with probability0, and label its related mail "
            "exactly [account-paused]. Leave the mail unread status and all financial records intact."
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
            "Reconcile all settled payments to their invoice number: set paid to settled amount, status paid if "
            "equal to invoice amount, otherwise partial. Ignore pending payments. Save "
            f"'{subject}.txt' containing only the total settled cents; operations folder,coordinator owner,draft,private,no sharing,version1."
        )
    if f == "acct_dispute":
        amount = json.loads(file["row"]["content"])["approved_amount"]
        b.update(invoice, status="disputed", owner=b.manager)
        b.draft(subject, f"{invoice['row']['number']}={amount}", [mail["row"]["sender"]], thread)
        return (
            f"Flag {company}'s invoice disputed and assign coordinator. Read its account JSON file's approved_amount. "
            f"Draft to the matching mail sender on that thread, subject '{subject}', body exactly invoice-number=approved_amount, "
            "draft status. Do not change invoice amount or mark it paid."
        )
    if f == "acct_expenses":
        for x in rows["expenses"]:
            b.update(
                x, status="approved" if x["row"]["amount"] <= b.threshold else "review", cost_center=p["cost_center"]
            )
        return (
            "Process all submitted expenses: amount at or below policy expense_limit becomes approved; above becomes "
            "review. Set their cost_center to the policy value. Never edit amount, category or employee."
        )
    if f == "acct_overdue":
        selected = [x for x in rows["invoices"] if x["row"]["amount"] > b.threshold]
        lookup = {x["row"]["account"]: x["row"]["email"] for x in rows["contacts"]}
        for x in selected:
            b.update(x, status="followup")
            b.draft(f"Followup {x['row']['number']}", str(x["row"]["amount"]), [lookup[x["row"]["account"]]])
        return (
            "For open invoices strictly above policy expense_limit, mark status followup and create one draft to each "
            "account's CRM contact: subject 'Followup ' plus invoice number, body only amount cents, empty thread,draft status. "
            "Leave invoices at or below the limit untouched."
        )
    if f == "acct_credit":
        approved = json.loads(file["row"]["content"])["approved_amount"]
        credit = invoice["row"]["amount"] - approved
        b.update(invoice, amount=approved, status="adjusted")
        b.document(subject + ".txt", f"{invoice['row']['number']}={credit}")
        return (
            f"Use {company}'s account JSON approved_amount to adjust its invoice amount and set status adjusted. "
            f"Save '{subject}.txt' with invoice-number=old-minus-approved-credit as integer cents; "
            "operations folder,coordinator owner,draft,private,no sharing,version1. Preserve paid and payment records."
        )
    if f == "storage_publish":
        selected = [x for x in rows["files"] if "-account-" in x["row"]["name"] and x["row"]["status"] == "approved"]
        for x in selected:
            b.update(x, access="team", shared_with=[b.manager], status="published")
        b.post("\n".join(sorted(x["row"]["name"] for x in selected)))
        return (
            "Publish approved account JSON files only: status published,access team,shared_with exactly coordinator. "
            "Post filenames sorted one per line to dispatch using coordinator sender,empty thread,posted status. "
            "Do not publish drafts or either policy file."
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
            "Archive account JSON files whose content.updated is before policy archive_before; set status archived "
            f"and folder archive. Save '{subject}.txt' with only the archived count; operations folder,coordinator owner, "
            "draft,private,no sharing,version1. Keep newer files and policy files intact."
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
            f"Join all CRM accounts to their contact emails. Save '{subject}.csv' with account,email rows sorted "
            "by account,one per line,no header. Use operations folder,coordinator owner,draft,private,shared_with exactly "
            "coordinator,version1. Share the operations folder with exactly coordinator too."
        )
    if f == "storage_redact":
        original = json.loads(file["row"]["content"])
        redacted = {k: v for k, v in original.items() if k != "private_phone"}
        b.update(file, content=canonical_json(redacted), version=file["row"]["version"] + 1)
        b.draft(subject, file["row"]["name"], [rows["contacts"][target]["row"]["email"]])
        return (
            f"Remove only private_phone from {company}'s account JSON content, serialize compact JSON with sorted "
            f"keys, and increment file version by1. Draft a notice subject '{subject}' to the CRM contact, body only "
            "the filename,empty thread,draft status. Keep the file private and all other JSON values."
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
            f"For {company}'s mail request, draft a reply to sender on its thread using subject '{subject}' "
            "and body invoice-number:outstanding-cents (amount minus paid). Mark original read with labels exactly [replied]."
        )
    if f == "mail_campaign":
        selected_names = {x["row"]["name"] for x in accounts if x["row"]["region"] == b.region}
        for x in rows["contacts"]:
            if x["row"]["account"] in selected_names:
                b.draft(subject, p["effective_date"], [x["row"]["email"]])
        b.post(str(len(selected_names)))
        return (
            f"For accounts in policy region create one draft per CRM contact: subject '{subject}',body only "
            "policy effective_date,empty thread,draft status. Post only recipient count to dispatch,coordinator sender, "
            "empty thread,posted status. Do not send the drafts."
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
            "Triage request messages by body.priority: high gets labels exactly [urgent],normal gets [routine]; "
            f"mark each read. Save '{subject}.txt' listing high-priority account names sorted one per line; "
            "operations folder,coordinator owner,draft,private,no sharing,version1."
        )
    if f == "mail_followup":
        b.draft(subject, p["effective_date"], [email], thread)
        b.update(rows["leave"][target], delegate=b.manager)
        return (
            f"For the employee named in {company}'s request, set their leave delegate coordinator and draft "
            f"a follow-up to that employee on the request thread: subject '{subject}',body only policy effective_date,draft status. "
            "Preserve leave approval status."
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
            f"{mail['vendor']} and set labels exactly [canonical]; delete the other copy. Post only the thread ID "
            "to dispatch with coordinator sender,empty thread,posted status. Do not delete other messages."
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
            "Find dispatch chat cases that are urgent and unresolved in their JSON body. Set those message "
            "statuses escalated and the linked invoice owner coordinator,status review. Leave routine and resolved cases unchanged."
        )
    if f == "chat_roster":
        members = sorted(x["row"]["email"] for x in employees if x["row"]["department"] == b.department)
        b.update(rows["channels"][0], members=sorted([b.manager, *members]))
        b.document(subject + ".txt", "\n".join(members))
        return (
            f"Set dispatch channel members to coordinator plus all HR employees in {b.department}, sorted. "
            f"Save '{subject}.txt' containing only those employees' emails sorted one per line; "
            "operations folder,coordinator owner,draft,private,no sharing,version1."
        )
    if f == "chat_digest":
        amounts = {x["row"]["number"]: x["row"]["amount"] for x in rows["invoices"]}
        content = "\n".join(
            sorted(f"{x['row']['thread']}={amounts[json.loads(x['row']['body'])['invoice']]}" for x in chats)
        )
        b.post(content)
        b.draft(subject, str(sum(amounts.values())), [b.manager])
        return (
            "Join every dispatch case to its invoice. Post thread=invoice-amount pairs sorted by thread,one per "
            f"line,coordinator sender,empty thread,posted status. Draft subject '{subject}' to coordinator,body only "
            "sum of those invoice cents,empty thread,draft status."
        )
    if f == "chat_topic":
        b.update(rows["channels"][0], topic=f"{b.department} | {p['effective_date']} | {p['cost_center']}")
        b.update(rows["folders"][0], owner=b.manager, access="team")
        return (
            "Read current policy and set dispatch topic to department | effective_date | cost_center with one "
            "space around each pipe. Transfer operations folder owner to coordinator and access team. Preserve its sharing list."
        )
    if f == "chat_closeout":
        selected = [x for x in chats if json.loads(x["row"]["body"])["resolved"]]
        for x in selected:
            b.update(x, status="closed")
            name = json.loads(x["row"]["body"])["account"]
            b.update(next(y for y in accounts if y["row"]["name"] == name), tier="reviewed")
        return (
            "Close dispatch chat messages whose JSON resolved is true and set their CRM account tier reviewed. "
            "Preserve message body,account status,and all unresolved cases."
        )
    raise ValueError("unknown workflow family")


def scenario_from_id(task_id: str) -> Scenario:
    parts = task_id.split(":")
    if len(parts) != 4 or parts[0] != REVISION or parts[1] not in {"train", "val"} or parts[2] not in FAMILIES:
        raise ValueError("unknown API workspace task identity")
    try:
        variant = int(parts[3])
    except ValueError as error:
        raise ValueError("invalid task variant") from error
    bound = TRAIN_VARIANTS if parts[1] == "train" else VAL_VARIANTS
    if not 0 <= variant < bound or parts[3] != f"{variant:04d}":
        raise ValueError("task variant outside immutable V3 corpus")
    builder = _Builder(parts[1], parts[2], variant)
    return builder.finish(_workflow(builder))


def task_ids(split: str) -> tuple[str, ...]:
    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    count = TRAIN_VARIANTS if split == "train" else VAL_VARIANTS
    ordered = tuple(FAMILIES[offset + domain * 5] for offset in range(5) for domain in range(6))
    return tuple(f"{REVISION}:{split}:{family}:{variant:04d}" for variant in range(count) for family in ordered)


def row_from_id(task_id: str) -> dict[str, Any]:
    scenario = scenario_from_id(task_id)
    return {
        "id": task_id,
        "task_id": task_id,
        "data_source": "api_workspace",
        "agent_ref": {"type": "responses_api_agents", "name": "api_workspace_agent"},
        "responses_create_params": {
            "input": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": scenario.prompt}],
            "tools": tools(),
        },
        "metadata": {
            "split": scenario.split,
            "family": scenario.family,
            "revision": REVISION,
            "seed": ROOT_SEED,
            "authoring": "independent_procedural_fiction",
        },
    }
