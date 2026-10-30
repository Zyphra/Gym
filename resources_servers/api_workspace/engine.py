# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline SaaS API semantics and final-state verification; no network or filesystem access."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


COLLECTIONS = {
    "hr": {
        "employees": {
            "name": str,
            "email": str,
            "department": str,
            "manager": str,
            "status": str,
            "location": str,
            "start_date": str,
            "employment_type": str,
            "cost_center": str,
        },
        "leave": {"employee": str, "start_date": str, "end_date": str, "status": str, "delegate": str},
    },
    "crm": {
        "accounts": {"name": str, "owner": str, "region": str, "status": str, "tier": str, "renewal_date": str},
        "contacts": {"name": str, "email": str, "account": str, "role": str, "status": str},
        "opportunities": {
            "name": str,
            "account": str,
            "owner": str,
            "stage": str,
            "amount": int,
            "close_date": str,
            "probability": int,
        },
    },
    "accounting": {
        "invoices": {
            "number": str,
            "account": str,
            "amount": int,
            "paid": int,
            "currency": str,
            "due_date": str,
            "status": str,
            "owner": str,
            "reference": str,
        },
        "payments": {"reference": str, "invoice": str, "amount": int, "currency": str, "status": str},
        "expenses": {
            "employee": str,
            "amount": int,
            "category": str,
            "status": str,
            "cost_center": str,
            "reference": str,
        },
    },
    "storage": {
        "files": {
            "name": str,
            "content": str,
            "folder": str,
            "owner": str,
            "status": str,
            "access": str,
            "shared_with": list,
            "version": int,
        },
        "folders": {"name": str, "owner": str, "access": str, "shared_with": list, "status": str},
    },
    "mail": {
        "messages": {
            "subject": str,
            "body": str,
            "sender": str,
            "to": list,
            "thread": str,
            "status": str,
            "labels": list,
        },
        "drafts": {"subject": str, "body": str, "to": list, "thread": str, "status": str},
    },
    "chat": {
        "channels": {"name": str, "topic": str, "members": list, "status": str},
        "messages": {"channel": str, "body": str, "sender": str, "thread": str, "status": str},
    },
}

# These names and API dialects were authored here. They are not benchmark imports.
VENDOR_NAMES = {
    "hr": ("gw_stafffern", "gw_peoplequill", "gw_teamcairn"),
    "crm": ("gw_dealwillow", "gw_clientreed", "gw_pipelinefin"),
    "accounting": ("gw_ledgerwren", "gw_tallymoss", "gw_bookcairn"),
    "storage": ("gw_foliofern", "gw_cloudreed", "gw_filewren"),
    "mail": ("gw_postquill", "gw_lettercairn", "gw_mailmoss"),
    "chat": ("gw_chimefern", "gw_talkquill", "gw_chatwren"),
}


@dataclass(frozen=True)
class Vendor:
    name: str
    function: str
    dialect: int

    @property
    def prefix(self) -> str:
        return ("/v1", "/api/v2", "/rest/2026")[self.dialect]

    def collection(self, name: str) -> str:
        return (name, f"{name}_records", f"workspace_{name}")[self.dialect]

    def field(self, name: str) -> str:
        aliases = (
            {},
            {"id": "key", "name": "display_name", "status": "state"},
            {"id": "uuid", "name": "title", "owner": "assignee"},
        )
        return aliases[self.dialect].get(name, name)

    def wire(self, data: dict[str, Any]) -> dict[str, Any]:
        return {self.field(key): deepcopy(value) for key, value in data.items()}

    def canonical(self, data: dict[str, Any]) -> dict[str, Any]:
        inverse = {self.field(key): key for schema in COLLECTIONS[self.function].values() for key in schema}
        inverse[self.field("id")] = "id"
        return {inverse.get(key, key): deepcopy(value) for key, value in data.items()}

    def path(self, collection: str, identity: str | None = None) -> str:
        return f"{self.prefix}/{self.collection(collection)}" + (f"/{identity}" if identity else "")


VENDORS = {
    name: Vendor(name, function, index) for function, names in VENDOR_NAMES.items() for index, name in enumerate(names)
}


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message
        super().__init__(message)

    def payload(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message}, "status": self.status}


def _error(status: int, code: str, message: str) -> None:
    raise APIError(status, code, message)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _creation_for_score(collection: str, row: dict[str, Any]) -> str:
    # Text-file tasks specify lines, not an operating-system line ending or
    # whether the last line has a terminator. Preserve every other character.
    if collection == "files" and isinstance(row.get("content"), str):
        row = {**row, "content": row["content"].replace("\r\n", "\n").removesuffix("\n")}
    return canonical_json(row)


class Workspace:
    """One episode's state. Its immutable scenario is reconstructed by the server."""

    def __init__(self, scenario: Any):
        self.scenario = scenario
        self.records = deepcopy(scenario.records)
        self.initial = deepcopy(scenario.records)
        self.idempotency: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
        self.sequence = 0

    def discover(self, function: str | None = None, connected_only: bool = False, query: str = "") -> dict[str, Any]:
        if function is not None and function not in COLLECTIONS:
            _error(400, "invalid_function", "Unknown job function; omit function to list all.")
        return {
            "accounts": [
                {
                    "vendor": v.name,
                    "function": v.function,
                    "connected": v.name in self.scenario.connected,
                    "account_id": f"{self.scenario.namespace}:{v.name}",
                    "api_prefix": v.prefix,
                }
                for v in VENDORS.values()
                if (function is None or v.function == function)
                and (not connected_only or v.name in self.scenario.connected)
                and query.casefold() in v.name.casefold()
            ]
        }

    def catalog(
        self,
        *,
        query: str = "",
        function: str | None = None,
        connected_only: bool = False,
        method: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Search route metadata without inspecting any task records or goals."""
        if function is not None and function not in COLLECTIONS:
            _error(400, "invalid_function", "Unknown function; omit it to search all functions.")
        if method is not None and method not in {"GET", "POST", "PATCH", "DELETE"}:
            _error(400, "invalid_method", "Use GET, POST, PATCH or DELETE.")
        if type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or offset < 0:
            _error(400, "invalid_pagination", "limit must be1..100 and offset nonnegative.")
        terms = re.findall(r"[a-z0-9_]+", query.casefold())
        results = []
        actions = {
            "GET": "read list search records",
            "POST": "create record",
            "PATCH": "update fields",
            "DELETE": "delete record",
        }
        for vendor in VENDORS.values():
            connected = vendor.name in self.scenario.connected
            if function is not None and vendor.function != function or connected_only and not connected:
                continue
            for collection, fields in COLLECTIONS[vendor.function].items():
                for verb, description in actions.items():
                    if method is not None and verb != method:
                        continue
                    path = vendor.path(collection) + ("/{id}" if verb in {"PATCH", "DELETE"} else "")
                    entry = {
                        "vendor": vendor.name,
                        "function": vendor.function,
                        "connected": connected,
                        "collection": collection,
                        "method": verb,
                        "path": path,
                        "description": f"{description} in {collection}; schema includes {', '.join(fields)}",
                        "schema_tool": "workspace_schema",
                    }
                    text = canonical_json(entry).casefold()
                    if terms and not all(term in text for term in terms):
                        continue
                    results.append(entry)
        return {
            "endpoints": results[offset : offset + limit],
            "total": len(results),
            "next_offset": offset + limit if offset + limit < len(results) else None,
            "catalog_only": True,
            "usage": "Catalog entries describe routes, not record contents. Inspect vendor schema, then request records.",
        }

    def schema(self, vendor: str) -> dict[str, Any]:
        v = self._vendor(vendor)
        return {
            "vendor": vendor,
            "connected": vendor in self.scenario.connected,
            "methods": {
                "GET": "list/search or get",
                "POST": "create; requires idempotency_key",
                "PATCH": "partial update",
                "DELETE": "delete record",
            },
            "params": {
                "q": "case-insensitive full-text search",
                "filters": "field:value or field:{eq|in|gt|gte|lt|lte:value}; all predicates must match",
                "limit": "integer 1..100; default 3",
                "cursor": "opaque next_cursor from same query",
            },
            "collections": [
                {
                    "name": name,
                    "path": v.path(name),
                    "id_field": v.field("id"),
                    "fields": {v.field(key): value.__name__ for key, value in fields.items()},
                }
                for name, fields in COLLECTIONS[v.function].items()
            ],
            "contract": "Use exact path; put query fields in params, never ?query in path. "
            "Amounts are integer cents. IDs are immutable. Lists have explicit pagination. "
            "Only fictional task accounts exist. Schema discovery does not connect an account.",
        }

    @staticmethod
    def _vendor(name: str) -> Vendor:
        if name not in VENDORS:
            _error(404, "vendor_not_found", "Vendor does not exist; discover accounts first.")
        return VENDORS[name]

    def request(
        self,
        *,
        vendor: str,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        v = self._vendor(vendor)
        if vendor not in self.scenario.connected:
            _error(401, "account_not_connected", "This vendor is not connected. Discover connected alternatives.")
        if "?" in path or "#" in path or "%" in path or ".." in path:
            _error(400, "invalid_path", "Path must be exact. Supply query parameters separately in params.")
        method = method.upper()
        match = re.fullmatch(re.escape(v.prefix) + r"/([a-z_]+)(?:/([a-zA-Z0-9_-]+))?", path)
        if not match:
            _error(404, "route_not_found", "Use vendor schema to discover the API path.")
        collection = next((key for key in COLLECTIONS[v.function] if v.collection(key) == match[1]), None)
        if collection is None:
            _error(404, "collection_not_found", "Collection is not available on this vendor.")
        identity = match[2]
        params, body = params or {}, body or {}
        bucket = self.records[vendor][collection]
        if method == "GET":
            if body:
                _error(400, "unexpected_body", "GET accepts params, not body.")
            if identity:
                if params:
                    _error(400, "unexpected_params", "Item GET has no query parameters.")
                if identity not in bucket:
                    _error(404, "record_not_found", "No such record in this connected account.")
                return {"status": 200, "item": v.wire(bucket[identity])}
            return self._list(v, collection, params)
        if params:
            _error(400, "unexpected_params", "Write endpoints do not accept query parameters.")
        if method not in {"POST", "PATCH", "DELETE"}:
            _error(405, "method_not_allowed", "Supported methods: GET, POST, PATCH, DELETE.")
        wire_fields = {v.field(key) for key in COLLECTIONS[v.function][collection]}
        if set(body) - wire_fields - {v.field("id")}:
            _error(422, "unknown_fields", "Use only field names declared in this vendor's schema.")
        data = v.canonical(body)
        if method == "DELETE":
            if body or not identity:
                _error(400, "invalid_delete", "DELETE requires item path and no body.")
            if identity not in bucket:
                _error(404, "record_not_found", "No such record.")
            del bucket[identity]
            return {"status": 200, "deleted": identity}
        if "id" in data:
            _error(422, "immutable_id", "Record IDs are server assigned and immutable.")
        schema = COLLECTIONS[v.function][collection]
        if set(data) - schema.keys():
            _error(422, "unknown_fields", "Use only fields declared in this vendor's schema.")
        for key, value in data.items():
            if type(value) is not schema[key] or (isinstance(value, list) and any(type(x) is not str for x in value)):
                _error(422, "invalid_field_type", f"Field {v.field(key)} requires {schema[key].__name__}.")
            if schema[key] is int and value < 0:
                _error(422, "invalid_amount", "Integer business values must be nonnegative.")
        if method == "PATCH":
            if not identity or identity not in bucket:
                _error(404, "record_not_found", "PATCH requires an existing item path.")
            if not data:
                _error(422, "empty_patch", "PATCH must contain at least one field.")
            bucket[identity].update(data)
            return {"status": 200, "item": v.wire(bucket[identity])}
        if identity:
            _error(405, "invalid_create_path", "POST uses a collection path.")
        if set(data) != schema.keys():
            _error(422, "missing_fields", "POST requires every field declared in the collection schema.")
        if not idempotency_key or len(idempotency_key) > 200:
            _error(422, "idempotency_key_required", "POST requires a nonempty idempotency_key up to200 characters.")
        fingerprint = canonical_json({"path": path, "body": data})
        slot = (vendor, idempotency_key)
        if slot in self.idempotency:
            previous, result = self.idempotency[slot]
            if previous != fingerprint:
                _error(409, "idempotency_conflict", "This key was already used for another request.")
            return deepcopy(result)
        self.sequence += 1
        identity = f"new_{self.sequence:06d}"
        bucket[identity] = {"id": identity, **data}
        result = {"status": 201, "item": v.wire(bucket[identity])}
        self.idempotency[slot] = (fingerprint, deepcopy(result))
        return result

    def _list(self, v: Vendor, collection: str, params: dict[str, Any]) -> dict[str, Any]:
        if set(params) - {"q", "filters", "limit", "cursor"}:
            _error(400, "unknown_query_parameter", "Allowed params: q, filters, limit, cursor.")
        q, filters, limit = params.get("q", ""), params.get("filters", {}), params.get("limit", 3)
        if type(q) is not str or type(filters) is not dict or type(limit) is not int or not 1 <= limit <= 100:
            _error(400, "invalid_query", "q must be text, filters an object, limit integer1..100.")
        fields = {"id": str, **COLLECTIONS[v.function][collection]}
        if set(filters) - {v.field(key) for key in fields}:
            _error(400, "unknown_filter_field", "Filter fields must use the vendor schema.")
        filters = v.canonical(filters)
        if set(filters) - fields.keys():
            _error(400, "unknown_filter_field", "Filter fields must use the vendor schema.")
        for key, condition in filters.items():
            if isinstance(condition, dict):
                if len(condition) != 1 or next(iter(condition)) not in {"eq", "in", "gt", "gte", "lt", "lte"}:
                    _error(400, "invalid_filter", "Use exactly one supported comparison per field.")
                operator, value = next(iter(condition.items()))
                if operator == "in":
                    if type(value) is not list or any(type(x) is not fields[key] for x in value):
                        _error(400, "invalid_filter", "in requires a list of values matching the field type.")
                elif type(value) is not fields[key] or (operator != "eq" and fields[key] not in {int, str}):
                    _error(400, "invalid_filter", "Comparison type must match the schema field.")
            elif type(condition) is not fields[key]:
                _error(400, "invalid_filter", "Equality filter type must match the schema field.")

        def matches(row: dict[str, Any]) -> bool:
            if q.casefold() not in canonical_json(row).casefold():
                return False
            for key, condition in filters.items():
                if key not in row:
                    return False
                operator, value = next(iter(condition.items())) if isinstance(condition, dict) else ("eq", condition)
                actual = row[key]
                if not {
                    "eq": lambda: actual == value,
                    "in": lambda: actual in value,
                    "gt": lambda: actual > value,
                    "gte": lambda: actual >= value,
                    "lt": lambda: actual < value,
                    "lte": lambda: actual <= value,
                }[operator]():
                    return False
            return True

        rows = [row for _, row in sorted(self.records[v.name][collection].items()) if matches(row)]
        fingerprint = hashlib.sha256(
            canonical_json({"vendor": v.name, "collection": collection, "q": q, "filters": filters}).encode()
        ).hexdigest()[:16]
        cursor = params.get("cursor")
        offset = 0
        if cursor is not None:
            if type(cursor) is not str or not re.fullmatch(re.escape(fingerprint) + r":\d+", cursor):
                _error(400, "invalid_cursor", "Cursor must come from this account and the same query.")
            offset = int(cursor.split(":")[1])
        next_offset = offset + limit
        return {
            "status": 200,
            "items": [v.wire(row) for row in rows[offset:next_offset]],
            "next_cursor": f"{fingerprint}:{next_offset}" if next_offset < len(rows) else None,
        }

    def score(self) -> dict[str, Any]:
        """Compare semantic world contents, ignoring generated create IDs and action order."""
        expected = deepcopy(self.initial)
        for goal in self.scenario.updates:
            row = expected[goal["vendor"]][goal["collection"]][goal["id"]]
            row.update(deepcopy(goal["fields"]))
        for goal in self.scenario.deletes:
            del expected[goal["vendor"]][goal["collection"]][goal["id"]]
        missing = 0
        unexpected = 0
        for vendor, collections in expected.items():
            if set(self.records.get(vendor, {})) != set(collections):
                unexpected += 1
            for collection, records in collections.items():
                actual = self.records.get(vendor, {}).get(collection, {})
                for identity, record in records.items():
                    if actual.get(identity) != record:
                        missing += 1
                creations = [
                    goal["fields"]
                    for goal in self.scenario.creates
                    if goal["vendor"] == vendor and goal["collection"] == collection
                ]
                actual_new = [
                    {key: value for key, value in row.items() if key != "id"}
                    for identity, row in actual.items()
                    if identity not in records
                ]
                # Deleted original records must not be reintroduced under another ID.
                if sorted(_creation_for_score(collection, row) for row in creations) != sorted(
                    _creation_for_score(collection, row) for row in actual_new
                ):
                    unexpected += 1
        success = missing == 0 and unexpected == 0 and set(self.records) == set(expected)
        return {
            "reward": float(success),
            "strict_success": success,
            "mismatched_records": missing,
            "unexpected_record_groups": unexpected,
            "task_id": self.scenario.task_id,
            "grading_basis": "final_world_state",
            "state_sha256": hashlib.sha256(canonical_json(self.records).encode()).hexdigest(),
        }
