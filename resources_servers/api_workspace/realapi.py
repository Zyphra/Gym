# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-API conventions for API Workspace V5 vendors; offline, deterministic, no network.

V5 worlds are served through one HTTP-style tool. Requests carry an absolute URL,
query parameters as a JSON-encoded string and a JSON-encoded body. The fictional
vendors follow widely used public API conventions: base64url RFC 2822 mail,
label IDs, spreadsheet A1 ranges with valueInputOption, chat methods that take
channel and user IDs, a CRM REST dialect with ``properties`` envelopes and a CRM
GraphQL dialect. Disconnected vendors answer 401. Vendor names, hosts and paths
were authored here; no benchmark contract or record is an input.
"""

from __future__ import annotations

import base64
import binascii
import email
import email.policy
import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from email.utils import getaddresses
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit


APPS: dict[str, dict[str, dict[str, type]]] = {
    "mail": {
        "messages": {
            "thread": str,
            "sender": str,
            "to": list,
            "cc": list,
            "bcc": list,
            "subject": str,
            "body": str,
            "labels": list,
            "date": str,
        },
        "labels": {"name": str, "type": str},
    },
    "sheets": {
        "spreadsheets": {"title": str, "owner": str},
        "worksheets": {"spreadsheet": str, "title": str, "index": int, "values": list},
    },
    "chat": {
        "channels": {"name": str, "topic": str, "members": list, "archived": bool},
        "messages": {"channel": str, "user": str, "text": str, "thread_ts": str},
        "users": {"name": str, "real_name": str, "email": str, "deleted": bool},
    },
    "crm": {
        "companies": {"name": str, "domain": str, "tier": str, "region": str, "owner": str},
        "contacts": {"name": str, "email": str, "company": str, "role": str, "lifecycle": str, "owner": str},
        "deals": {"name": str, "company": str, "amount": int, "stage": str, "owner": str, "close_date": str},
    },
}


@dataclass(frozen=True)
class RealVendor:
    name: str
    app: str
    host: str
    dialect: str

    @property
    def base(self) -> str:
        return f"https://{self.host}"

    @property
    def short(self) -> str:
        return self.name.removeprefix("gw_")


REAL_VENDORS = {
    vendor.name: vendor
    for vendor in (
        RealVendor("gw_parcelwick", "mail", "api.parcelwick.example", "rest"),
        RealVendor("gw_inkharbor", "mail", "api.inkharbor.example", "rest"),
        RealVendor("gw_gridmoor", "sheets", "api.gridmoor.example", "rest"),
        RealVendor("gw_cellbrook", "sheets", "api.cellbrook.example", "rest"),
        RealVendor("gw_buzzfold", "chat", "api.buzzfold.example", "rest"),
        RealVendor("gw_chirpline", "chat", "api.chirpline.example", "rest"),
        RealVendor("gw_ledgerlark", "crm", "api.ledgerlark.example", "rest"),
        RealVendor("gw_orbitmint", "crm", "api.orbitmint.example", "graphql"),
    )
}
APP_VENDORS = {app: tuple(sorted(v.name for v in REAL_VENDORS.values() if v.app == app)) for app in APPS}
_HOSTS = {vendor.host: vendor for vendor in REAL_VENDORS.values()}
SYSTEM_LABELS = ("IMPORTANT", "INBOX", "SENT", "STARRED", "TRASH", "UNREAD")
MAIL_BASE = "/mail/v1/users/me"
SHEETS_BASE = "/v4/spreadsheets"
CRM_BASE = "/crm/v3/objects"
GRID_ROWS, GRID_COLUMNS = 1000, 26
_ABSENT = object()


class _Fail(Exception):
    def __init__(self, status: int, body: dict[str, Any]):
        super().__init__(status)
        self.status, self.body = status, body


def _google(status: int, kind: str, message: str) -> _Fail:
    return _Fail(status, {"error": {"code": status, "status": kind, "message": message}})


def _bad(message: str) -> _Fail:
    return _google(400, "INVALID_ARGUMENT", message)


def _missing(message: str = "Requested entity was not found.") -> _Fail:
    return _google(404, "NOT_FOUND", message)


def _no_route(method: str, path: str) -> _Fail:
    return _missing(f"No handler for {method} {path}. Search the catalog with workspace_search for the endpoint URL.")


def _wrong_method(method: str, path: str, allowed: str) -> _Fail:
    return _google(405, "METHOD_NOT_ALLOWED", f"{method} is not supported for {path}; use {allowed}.")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def b64url_encode(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def b64url_decode(text: str) -> bytes:
    """Decode base64url with or without padding; the standard alphabet is tolerated."""
    compact = re.sub(r"\s+", "", text).replace("+", "-").replace("/", "_").rstrip("=")
    if not compact or not re.fullmatch(r"[A-Za-z0-9_-]+", compact) or len(compact) % 4 == 1:
        raise ValueError("not base64url")
    try:
        return base64.urlsafe_b64decode(compact + "=" * (-len(compact) % 4))
    except (binascii.Error, ValueError) as error:
        raise ValueError("not base64url") from error


def codec(operation: str, text: str) -> tuple[int, dict[str, Any]]:
    if operation == "base64url_encode":
        return 200, {"result": b64url_encode(text)}
    try:
        return 200, {"result": b64url_decode(text).decode()}
    except (ValueError, UnicodeDecodeError):
        return 400, {"error": {"code": 400, "message": "Input is not base64url-encoded UTF-8 text."}}


# ---------------------------------------------------------------------------
# Request decoding


def _decode_container(value: Any, label: str) -> Any:
    if value is None or isinstance(value, str) and not value.strip():
        return _ABSENT
    if not isinstance(value, str):
        example = "'{\"limit\": 10}'" if label == "params" else '\'{"name": "value"}\''
        raise _bad(f"{label} must be a JSON-encoded string (or null), for example {example}.")
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise _bad(f"Invalid JSON in {label}: {error.msg} at character {error.pos}.") from error


def _query(url_query: str, params: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key, value in parse_qsl(url_query, keep_blank_values=True):
        if key not in merged:
            merged[key] = value
        elif isinstance(merged[key], list):
            merged[key].append(value)
        else:
            merged[key] = [merged[key], value]
    decoded = _decode_container(params, "params")
    if decoded is _ABSENT:
        return merged
    if not isinstance(decoded, dict):
        raise _bad("params must encode a JSON object of query parameters, for example '{\"limit\": 10}'.")
    for key, value in decoded.items():
        if key in merged:
            raise _bad(f"Query parameter {key!r} appears both in the URL and in params; supply it once.")
        merged[key] = value
    return merged


def fetch(world: Any, method: Any, url: Any, params: Any = None, body: Any = None) -> tuple[int, dict[str, Any]]:
    """Execute one HTTP-style request against the session's V5 world."""
    try:
        if type(method) is not str or method.upper() not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            raise _bad("method must be one of GET, POST, PUT, PATCH or DELETE.")
        method = method.upper()
        if type(url) is not str or not url.strip():
            raise _bad("url must be an absolute https URL taken from workspace_search results.")
        parts = urlsplit(url.strip())
        if parts.scheme != "https" or not parts.hostname:
            raise _bad("url must be an absolute https URL taken from workspace_search results.")
        vendor = _HOSTS.get(parts.hostname.lower())
        if vendor is None:
            raise _missing(f"No API is served at host {parts.hostname}. Use a URL from workspace_search.")
        if vendor.name not in world.scenario.connected:
            raise _google(
                401,
                "UNAUTHENTICATED",
                f"The {vendor.short} account is not connected to this workspace, so the request has no valid "
                f"credentials. Use the connected {vendor.app} vendor shown by workspace_search.",
            )
        query = _query(parts.query, params)
        payload = _decode_container(body, "body")
        path = unquote(parts.path).rstrip("/") or "/"
        if method == "GET" and payload is not _ABSENT:
            raise _bad("GET requests take query parameters in params, not a body.")
        handler = {
            ("mail", "rest"): _mail,
            ("sheets", "rest"): _sheets,
            ("chat", "rest"): _chat,
            ("crm", "rest"): _crm_rest,
            ("crm", "graphql"): _crm_graphql,
        }[(vendor.app, vendor.dialect)]
        return handler(world, vendor, method, path, query, payload)
    except _Fail as failure:
        return failure.status, failure.body


def _store(world: Any, vendor: RealVendor) -> dict[str, dict[str, dict[str, Any]]]:
    return world.records[vendor.name]


def _new_id(world: Any) -> int:
    world.sequence += 1
    return world.sequence


def _only(query: dict[str, Any], allowed: set[str], hint: str = "") -> None:
    unknown = sorted(set(query) - allowed)
    if unknown:
        raise _bad(f"Unknown query parameter {unknown[0]!r}." + (f" {hint}" if hint else ""))


def _integer(value: Any, name: str, default: int, low: int, high: int) -> int:
    if value is None:
        return default
    if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        value = int(value)
    if type(value) is not int or not low <= value <= high:
        raise _bad(f"{name} must be an integer from {low} to {high}.")
    return value


def _flag(value: Any, name: str) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if type(value) is not bool:
        raise _bad(f"{name} must be true or false.")
    return value


def _token(scope: dict[str, Any], offset: int) -> str:
    digest = hashlib.sha256(_canonical(scope).encode()).hexdigest()[:12]
    return base64.urlsafe_b64encode(f"{digest}:{offset}".encode()).decode().rstrip("=")


def _offset(scope: dict[str, Any], token: Any, name: str) -> int:
    if token is None or token == "":
        return 0
    try:
        digest, offset = b64url_decode(token).decode().split(":")
    except (ValueError, UnicodeDecodeError, AttributeError) as error:
        raise _bad(f"Invalid {name}; reuse the token returned for the same request.") from error
    if digest != hashlib.sha256(_canonical(scope).encode()).hexdigest()[:12] or not offset.isdigit():
        raise _bad(f"Invalid {name}; it must come from the same request with the same filters.")
    return int(offset)


# ---------------------------------------------------------------------------
# Mail: messages resource with base64url bodies, label IDs and RFC 2822 sends.


def _labels(store: dict[str, Any]) -> list[dict[str, str]]:
    system = [{"id": label, "name": label, "type": "system"} for label in SYSTEM_LABELS]
    user = [{"id": key, "name": row["name"], "type": row["type"]} for key, row in sorted(store["labels"].items())]
    return system + user


def _thread(row: dict[str, Any], identity: str) -> str:
    return row["thread"] or identity


def _mail_resource(identity: str, row: dict[str, Any], form: str) -> dict[str, Any]:
    headers = [{"name": "From", "value": row["sender"]}, {"name": "To", "value": ", ".join(row["to"])}]
    if row["cc"]:
        headers.append({"name": "Cc", "value": ", ".join(row["cc"])})
    headers += [{"name": "Subject", "value": row["subject"]}, {"name": "Date", "value": row["date"]}]
    resource: dict[str, Any] = {
        "id": identity,
        "threadId": _thread(row, identity),
        "labelIds": list(row["labels"]),
        "snippet": " ".join(row["body"].split())[:40],
    }
    if form == "minimal":
        return resource
    if form == "raw":
        lines = [f"{h['name']}: {h['value']}" for h in headers]
        return resource | {"raw": b64url_encode("\r\n".join(lines) + "\r\n\r\n" + row["body"])}
    payload: dict[str, Any] = {"mimeType": "text/plain", "headers": headers}
    if form == "full":
        data = row["body"].encode()
        payload["body"] = {"size": len(data), "data": b64url_encode(row["body"])}
    return resource | {"payload": payload}


def _mail_query_terms(text: str) -> list[tuple[bool, str, str]]:
    terms = []
    for match in re.finditer(r'(-?)(?:([a-zA-Z]+):)?("([^"]*)"|\S+)', text):
        negated, operator, raw, quoted = match.groups()
        value = quoted if quoted is not None else raw.strip('"')
        if operator and operator.lower() not in {
            "from",
            "to",
            "cc",
            "subject",
            "label",
            "is",
            "in",
            "after",
            "before",
        }:
            value, operator = f"{operator}:{value}", None
        terms.append((bool(negated), (operator or "").lower(), value.lower()))
    return terms


def _mail_matches(row: dict[str, Any], terms: list[tuple[bool, str, str]], names: dict[str, str]) -> bool:
    labels = set(row["labels"])
    for negated, operator, value in terms:
        if operator == "from":
            hit = value in row["sender"].lower()
        elif operator == "to":
            hit = any(value in address for address in row["to"] + row["cc"] + row["bcc"])
        elif operator == "cc":
            hit = any(value in address for address in row["cc"])
        elif operator == "subject":
            hit = value in row["subject"].lower()
        elif operator == "label":
            hit = names.get(value, names.get(value.replace("-", " "), value.upper())) in labels
        elif operator == "is":
            states = {
                "unread": "UNREAD" in labels,
                "read": "UNREAD" not in labels,
                "starred": "STARRED" in labels,
                "important": "IMPORTANT" in labels,
            }
            hit = states.get(value, False)
        elif operator == "in":
            hit = value == "anywhere" or value.upper() in labels
        elif operator in {"after", "before"}:
            day = value.replace("/", "-")
            hit = row["date"][:10] > day if operator == "after" else row["date"][:10] < day
        else:
            text = " ".join([row["subject"], row["body"], row["sender"], *row["to"]]).lower()
            hit = value in text
        if hit == negated:
            return False
    return True


def _addresses(message: Any, header: str) -> list[str]:
    values = [str(value) for value in message.get_all(header, [])]
    found = []
    for _, address in getaddresses(values):
        address = address.strip().lower()
        if not re.fullmatch(r"[^@\s<>,;]+@[^@\s<>,;]+\.[^@\s<>,;]+", address):
            raise _bad(f"Invalid {header} header in the decoded message: {', '.join(values)!r}.")
        found.append(address)
    if values and not found:
        raise _bad(f"Invalid {header} header in the decoded message: {', '.join(values)!r}.")
    return found


def _mail_send(world: Any, store: dict[str, Any], payload: Any) -> tuple[int, dict[str, Any]]:
    if payload is _ABSENT:
        raise _bad('Request body is required: {"raw": "<base64url-encoded RFC 2822 message>"}.')
    if not isinstance(payload, dict):
        raise _bad(
            'Invalid JSON payload: the body must be a JSON object such as {"raw": "<base64url RFC 2822 message>"}.'
        )
    unknown = sorted(set(payload) - {"raw", "threadId"})
    if unknown:
        raise _bad(
            f"Invalid JSON payload received. Unknown name {unknown[0]!r}: a message is sent as "
            '{"raw": "<base64url RFC 2822 message>"} with an optional "threadId"; recipients, subject and text '
            "belong in the encoded message's headers and body."
        )
    raw = payload.get("raw")
    if not isinstance(raw, str) or not raw.strip():
        raise _bad("'raw' is required and must be a base64url-encoded RFC 2822 message.")
    try:
        data = b64url_decode(raw)
    except ValueError as error:
        raise _bad("Invalid value at 'message.raw': it is not valid base64url.") from error
    message = email.message_from_bytes(data, policy=email.policy.default)
    to, cc, bcc = (_addresses(message, header) for header in ("To", "Cc", "Bcc"))
    if not to + cc + bcc:
        raise _bad(
            "Recipient address required. No To, Cc or Bcc header was found in the decoded RFC 2822 message; "
            "headers come first, then a blank line, then the body."
        )
    part = message
    if message.is_multipart():
        parts = [p for p in message.walk() if not p.is_multipart() and p.get_content_maintype() == "text"]
        part = next((p for p in parts if p.get_content_type() == "text/plain"), parts[0] if parts else None)
    try:
        text = part.get_content() if part is not None and part.get_content_maintype() == "text" else ""
    except (LookupError, ValueError) as error:
        raise _bad("The message body could not be decoded; use UTF-8 text.") from error
    thread = payload.get("threadId")
    if thread is not None:
        known = {_thread(row, key) for key, row in store["messages"].items()}
        if not isinstance(thread, str) or thread not in known:
            raise _missing(f"Requested entity was not found: thread {thread!r}. Use a threadId from a message.")
    identity = f"19{_new_id(world):014x}"
    row = {
        "id": identity,
        "thread": thread or "",
        "sender": world.scenario.context["mailbox"],
        "to": sorted(set(to)),
        "cc": sorted(set(cc)),
        "bcc": sorted(set(bcc)),
        "subject": str(message.get("Subject", "")).strip(),
        "body": text if isinstance(text, str) else "",
        "labels": ["SENT"],
        "date": world.scenario.context["now"],
    }
    store["messages"][identity] = row
    return 200, {"id": identity, "threadId": _thread(row, identity), "labelIds": ["SENT"]}


def _mail_modify(store: dict[str, Any], identity: str, row: dict[str, Any], payload: Any) -> tuple[int, dict]:
    if not isinstance(payload, dict) or set(payload) - {"addLabelIds", "removeLabelIds"}:
        raise _bad('Body must be {"addLabelIds": [...], "removeLabelIds": [...]} with label IDs.')
    add, remove = payload.get("addLabelIds", []), payload.get("removeLabelIds", [])
    if not all(isinstance(x, list) and all(isinstance(y, str) for y in x) for x in (add, remove)) or not add + remove:
        raise _bad("addLabelIds and removeLabelIds must be lists of label IDs, and at least one must be nonempty.")
    valid = {label["id"] for label in _labels(store)}
    names = {label["name"].lower(): label["id"] for label in _labels(store)}
    for label in add + remove:
        if label not in valid:
            hint = f" The ID for that name is {names[label.lower()]}." if label.lower() in names else ""
            raise _bad(f"Invalid label: {label!r}. Use label IDs from GET {MAIL_BASE}/labels.{hint}")
        if label == "SENT":
            raise _bad("The SENT label is managed by the mail service.")
    row["labels"] = sorted((set(row["labels"]) - set(remove)) | set(add))
    return 200, {"id": identity, "threadId": _thread(row, identity), "labelIds": list(row["labels"])}


def _mail(world: Any, vendor: RealVendor, method: str, path: str, query: dict, payload: Any) -> tuple[int, dict]:
    store = _store(world, vendor)
    if not path.startswith(MAIL_BASE + "/"):
        raise _no_route(method, path)
    rest = path[len(MAIL_BASE) :]
    names = {label["name"].lower(): label["id"] for label in _labels(store)}
    if rest == "/profile":
        if method != "GET":
            raise _wrong_method(method, path, "GET")
        _only(query, set())
        return 200, {"emailAddress": world.scenario.context["mailbox"], "messagesTotal": len(store["messages"])}
    if rest == "/labels":
        if method != "GET":
            raise _wrong_method(method, path, "GET")
        _only(query, set())
        return 200, {"labels": _labels(store)}
    if rest == "/messages/send":
        if method != "POST":
            raise _wrong_method(method, path, "POST")
        _only(query, set(), "Message fields go in the JSON body as an encoded raw message.")
        return _mail_send(world, store, payload)
    if rest == "/messages":
        if method != "GET":
            raise _wrong_method(method, path, "GET to list, or POST .../messages/send to send")
        _only(query, {"q", "maxResults", "pageToken", "labelIds", "includeSpamTrash"})
        text = query.get("q", "")
        labels = query.get("labelIds", [])
        labels = [labels] if isinstance(labels, str) else labels
        if not isinstance(text, str) or not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
            raise _bad("q must be text and labelIds a label ID or list of label IDs.")
        limit = _integer(query.get("maxResults"), "maxResults", 10, 1, 50)
        trash = _flag(query.get("includeSpamTrash"), "includeSpamTrash")
        terms = _mail_query_terms(text)
        rows = sorted(store["messages"].items(), key=lambda item: (item[1]["date"], item[0]), reverse=True)
        found = [
            (key, row)
            for key, row in rows
            if all(label in row["labels"] for label in labels)
            and (trash or "TRASH" not in row["labels"])
            and _mail_matches(row, terms, names)
        ]
        scope = {"vendor": vendor.name, "q": text, "labelIds": labels, "trash": trash}
        start = _offset(scope, query.get("pageToken"), "pageToken")
        page = found[start : start + limit]
        result: dict[str, Any] = {
            "messages": [{"id": key, "threadId": _thread(row, key)} for key, row in page],
            "resultSizeEstimate": len(found),
        }
        if start + limit < len(found):
            result["nextPageToken"] = _token(scope, start + limit)
        return 200, result
    match = re.fullmatch(r"/messages/([A-Za-z0-9_-]+)(/modify)?", rest)
    if not match:
        raise _no_route(method, path)
    identity, modify = match.groups()
    row = store["messages"].get(identity)
    if row is None:
        raise _missing(f"Requested entity was not found: message {identity!r}.")
    if modify:
        if method != "POST":
            raise _wrong_method(method, path, "POST")
        _only(query, set(), "Label changes go in the JSON body.")
        return _mail_modify(store, identity, row, payload)
    if method != "GET":
        raise _wrong_method(method, path, "GET")
    _only(query, {"format"})
    form = query.get("format", "full")
    if form not in {"full", "metadata", "minimal", "raw"}:
        raise _bad("format must be full, metadata, minimal or raw.")
    return 200, _mail_resource(identity, row, form)


# ---------------------------------------------------------------------------
# Sheets: A1 ranges, valueInputOption query parameter and values envelopes.


def column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _column_index(letters: str) -> int:
    index = 0
    for char in letters.upper():
        index = index * 26 + ord(char) - 64
    return index


@dataclass(frozen=True)
class A1Range:
    sheet: str
    row1: int | None
    col1: int | None
    row2: int | None
    col2: int | None
    single: bool

    def label(self, rows: int | None = None, cols: int | None = None) -> str:
        r1, c1 = self.row1 or 1, self.col1 or 1
        r2 = self.row2 or max(r1, rows or r1)
        c2 = self.col2 or max(c1, cols or c1)
        return f"{quote_sheet(self.sheet)}!{column_name(c1)}{r1}:{column_name(c2)}{r2}"


def quote_sheet(title: str) -> str:
    return title if re.fullmatch(r"[A-Za-z0-9_]+", title) else "'" + title.replace("'", "''") + "'"


def parse_a1(text: str, titles: list[str]) -> A1Range:
    """Parse ``Sheet!A1:C5``, ``'Sheet name'!A:C``, ``Sheet!2:4``, ``Sheet`` or a range on the first sheet."""
    lookup = {title.lower(): title for title in titles}
    text = text.strip()
    if "!" in text:
        sheet, cells = text.rsplit("!", 1)
        if len(sheet) >= 2 and sheet[0] == sheet[-1] == "'":
            sheet = sheet[1:-1].replace("''", "'")
    elif text.lower() in lookup or text.startswith("'") and text.endswith("'"):
        sheet, cells = text.strip("'").replace("''", "'"), ""
    else:
        sheet, cells = titles[0] if titles else "", text
    if sheet.lower() not in lookup:
        raise _bad(f"Unable to parse range: {text}. No worksheet is titled {sheet!r}.")
    sheet = lookup[sheet.lower()]
    if cells == "":
        return A1Range(sheet, None, None, None, None, False)
    bounds = cells.split(":")
    parsed = []
    for part in bounds:
        match = re.fullmatch(r"([A-Za-z]{1,2})?([1-9][0-9]*)?", part)
        if not part or not match:
            raise _bad(f"Unable to parse range: {text}.")
        letters, digits = match.groups()
        parsed.append((int(digits) if digits else None, _column_index(letters) if letters else None))
    if len(parsed) == 1:
        (row, col) = parsed[0]
        if row is None or col is None:
            raise _bad(f"Unable to parse range: {text}.")
        result = A1Range(sheet, row, col, row, col, True)
    elif len(parsed) == 2:
        (r1, c1), (r2, c2) = parsed
        shape = (r1 is not None, c1 is not None, r2 is not None, c2 is not None)
        # A1:C5, A2:C (open rows), A:C (columns) and 2:5 (rows) are the accepted forms.
        if shape not in {
            (True, True, True, True),
            (True, True, False, True),
            (False, True, False, True),
            (True, False, True, False),
        }:
            raise _bad(f"Unable to parse range: {text}.")
        if r1 is not None and r2 is not None and r1 > r2 or c1 is not None and c2 is not None and c1 > c2:
            raise _bad(f"Unable to parse range: {text}. The start must precede the end.")
        result = A1Range(sheet, r1, c1, r2, c2, False)
    else:
        raise _bad(f"Unable to parse range: {text}.")
    if (
        max(x or 0 for x in (result.row1, result.row2)) > GRID_ROWS
        or max(x or 0 for x in (result.col1, result.col2)) > GRID_COLUMNS
    ):
        raise _bad(f"Range {text} exceeds the grid limits ({GRID_ROWS} rows, columns A-Z).")
    return result


def _cell(value: Any, where: str) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if type(value) is bool:
        return "TRUE" if value else "FALSE"
    if type(value) is int:
        return str(value)
    if type(value) is float and math.isfinite(value):
        return str(int(value)) if value.is_integer() else repr(value)
    raise _bad(f"Invalid value at '{where}': cells must be strings, numbers, booleans or null.")


def _trim(grid: list[list[str]]) -> list[list[str]]:
    rows = []
    for row in grid:
        row = list(row)
        while row and row[-1] == "":
            row.pop()
        rows.append(row)
    while rows and not rows[-1]:
        rows.pop()
    return rows


def _matrix(payload: Any, where: str, dimension: str = "ROWS") -> list[list[str]]:
    values = payload.get("values") if isinstance(payload, dict) else None
    if not isinstance(values, list) or not values or not all(isinstance(row, list) for row in values):
        raise _bad(f"Invalid value at '{where}.values': expected a 2D array of cell values ([[...], ...]).")
    matrix = [
        [_cell(value, f"{where}.values[{i}][{j}]") for j, value in enumerate(row)] for i, row in enumerate(values)
    ]
    if dimension == "COLUMNS":
        width = max(len(row) for row in matrix)
        matrix = [[row[i] if i < len(row) else "" for row in matrix] for i in range(width)]
    return matrix


def _write(sheet: dict[str, Any], start_row: int, start_col: int, matrix: list[list[str]]) -> tuple[int, int]:
    grid = [list(row) for row in sheet["values"]]
    height = len(matrix)
    width = max((len(row) for row in matrix), default=0)
    if start_row + height - 1 > GRID_ROWS or start_col + width - 1 > GRID_COLUMNS:
        raise _bad("Writing would exceed the grid limits (1000 rows, columns A-Z).")
    for i, values in enumerate(matrix):
        r = start_row - 1 + i
        while len(grid) <= r:
            grid.append([])
        for j, value in enumerate(values):
            c = start_col - 1 + j
            while len(grid[r]) <= c:
                grid[r].append("")
            grid[r][c] = value
    sheet["values"] = _trim(grid)
    return height, width


def _check_fit(target: A1Range, matrix: list[list[str]], text: str) -> None:
    if target.single:
        return
    height, width = len(matrix), max(len(row) for row in matrix)
    if target.row2 is not None and (target.row1 or 1) + height - 1 > target.row2:
        raise _bad(
            f"Requested writing within range [{text}], but tried writing to row [{(target.row1 or 1) + height - 1}]."
        )
    if target.col2 is not None and (target.col1 or 1) + width - 1 > target.col2:
        last = column_name((target.col1 or 1) + width - 1)
        raise _bad(f"Requested writing within range [{text}], but tried writing to column [{last}].")


def _read(sheet: dict[str, Any], target: A1Range, dimension: str) -> dict[str, Any]:
    grid = sheet["values"]
    r1, c1 = target.row1 or 1, target.col1 or 1
    r2 = target.row2 or max(len(grid), r1)
    c2 = target.col2 or max([len(row) for row in grid] + [c1])
    block = [[(row[c - 1] if c - 1 < len(row) else "") for c in range(c1, c2 + 1)] for row in grid[r1 - 1 : r2]]
    block = _trim(block)
    if dimension == "COLUMNS" and block:
        width = max(len(row) for row in block)
        block = _trim([[row[i] if i < len(row) else "" for row in block] for i in range(width)])
    result: dict[str, Any] = {
        "range": target.label(r1 + len(block) - 1 if block else r1, c2),
        "majorDimension": dimension,
    }
    if block:
        result["values"] = block
    return result


def _value_input(query: dict[str, Any], payload: Any, operation: str, extra: set[str]) -> None:
    if isinstance(payload, dict) and "valueInputOption" in payload:
        raise _bad(f"valueInputOption is a query parameter for {operation}; move it from the body to params.")
    _only(query, {"valueInputOption", "includeValuesInResponse", *extra})
    if query.get("valueInputOption") not in {"RAW", "USER_ENTERED"}:
        raise _bad(
            f"'valueInputOption' is required for {operation}. Pass it as a query parameter: RAW or USER_ENTERED."
        )


def _sheets(world: Any, vendor: RealVendor, method: str, path: str, query: dict, payload: Any) -> tuple[int, dict]:
    store = _store(world, vendor)
    if path == SHEETS_BASE:
        if method != "GET":
            raise _wrong_method(method, path, "GET")
        _only(query, {"q", "pageSize", "pageToken"})
        text = query.get("q", "")
        if not isinstance(text, str):
            raise _bad("q must be text matched against spreadsheet titles.")
        size = _integer(query.get("pageSize"), "pageSize", 10, 1, 100)
        found = [
            {"spreadsheetId": key, "title": row["title"]}
            for key, row in sorted(store["spreadsheets"].items(), key=lambda item: (item[1]["title"], item[0]))
            if text.lower() in row["title"].lower()
        ]
        scope = {"vendor": vendor.name, "q": text}
        start = _offset(scope, query.get("pageToken"), "pageToken")
        result: dict[str, Any] = {"spreadsheets": found[start : start + size]}
        if start + size < len(found):
            result["nextPageToken"] = _token(scope, start + size)
        return 200, result
    match = re.fullmatch(re.escape(SHEETS_BASE) + r"/([^/:]+)(.*)", path)
    if not match:
        raise _no_route(method, path)
    sid, rest = match.groups()
    if sid not in store["spreadsheets"]:
        raise _missing(f"Requested entity was not found: spreadsheet {sid!r}. List spreadsheets to find its ID.")
    tabs = sorted(
        ((key, row) for key, row in store["worksheets"].items() if row["spreadsheet"] == sid),
        key=lambda item: item[1]["index"],
    )
    titles = [row["title"] for _, row in tabs]
    by_title = {row["title"]: row for _, row in tabs}
    if rest == "":
        if method != "GET":
            raise _wrong_method(method, path, "GET")
        _only(query, {"fields", "includeGridData"})
        return 200, {
            "spreadsheetId": sid,
            "properties": {"title": store["spreadsheets"][sid]["title"]},
            "sheets": [
                {
                    "properties": {
                        "sheetId": int(key),
                        "title": row["title"],
                        "index": row["index"],
                        "gridProperties": {"rowCount": GRID_ROWS, "columnCount": GRID_COLUMNS},
                    }
                }
                for key, row in tabs
            ],
        }
    if rest == "/values:batchGet":
        if method != "GET":
            raise _wrong_method(method, path, "GET")
        _only(query, {"ranges", "majorDimension"})
        ranges = query.get("ranges")
        ranges = [ranges] if isinstance(ranges, str) else ranges
        if not isinstance(ranges, list) or not ranges or not all(isinstance(x, str) for x in ranges):
            raise _bad("ranges is required: one A1 range or a list of A1 ranges.")
        dimension = query.get("majorDimension", "ROWS")
        if dimension not in {"ROWS", "COLUMNS"}:
            raise _bad("majorDimension must be ROWS or COLUMNS.")
        values = []
        for text in ranges:
            target = parse_a1(text, titles)
            values.append(_read(by_title[target.sheet], target, dimension))
        return 200, {"spreadsheetId": sid, "valueRanges": values}
    if rest == "/values:batchUpdate":
        if method != "POST":
            raise _wrong_method(method, path, "POST")
        if query:
            raise _bad("values.batchUpdate takes valueInputOption and data in the JSON body, not in params.")
        if not isinstance(payload, dict) or set(payload) - {"valueInputOption", "data", "includeValuesInResponse"}:
            raise _bad('Body must be {"valueInputOption": "RAW", "data": [{"range": "...", "values": [[...]]}]}.')
        if payload.get("valueInputOption") not in {"RAW", "USER_ENTERED"}:
            raise _bad("'valueInputOption' is required in the batchUpdate body: RAW or USER_ENTERED.")
        data = payload.get("data")
        if not isinstance(data, list) or not data or not all(isinstance(x, dict) for x in data):
            raise _bad("data must be a nonempty list of {range, values} objects.")
        staged = deepcopy(by_title)
        cells = 0
        for i, item in enumerate(data):
            if not isinstance(item.get("range"), str) or set(item) - {"range", "values", "majorDimension"}:
                raise _bad(f"data[{i}] must contain range and values.")
            target = parse_a1(item["range"], titles)
            matrix = _matrix(item, f"data[{i}]", item.get("majorDimension", "ROWS"))
            _check_fit(target, matrix, item["range"])
            _write(staged[target.sheet], target.row1 or 1, target.col1 or 1, matrix)
            cells += sum(len(row) for row in matrix)
        for key, row in tabs:
            row["values"] = staged[row["title"]]["values"]
        return 200, {"spreadsheetId": sid, "totalUpdatedCells": cells, "totalUpdatedRanges": len(data)}
    if not rest.startswith("/values/"):
        raise _no_route(method, path)
    text = rest[len("/values/") :]
    operation = "update"
    for suffix in (":append", ":clear"):
        if text.endswith(suffix):
            text, operation = text[: -len(suffix)], suffix[1:]
    target = parse_a1(text, titles)
    sheet = by_title[target.sheet]
    if operation == "update" and method == "GET":
        _only(query, {"majorDimension", "valueRenderOption"})
        dimension = query.get("majorDimension", "ROWS")
        if dimension not in {"ROWS", "COLUMNS"}:
            raise _bad("majorDimension must be ROWS or COLUMNS.")
        return 200, _read(sheet, target, dimension)
    if operation == "clear":
        if method != "POST":
            raise _wrong_method(method, path, "POST")
        _only(query, set())
        r1, c1 = target.row1 or 1, target.col1 or 1
        r2, c2 = target.row2 or GRID_ROWS, target.col2 or GRID_COLUMNS
        grid = [list(row) for row in sheet["values"]]
        for r in range(r1 - 1, min(r2, len(grid))):
            for c in range(c1 - 1, min(c2, len(grid[r]))):
                grid[r][c] = ""
        sheet["values"] = _trim(grid)
        return 200, {"spreadsheetId": sid, "clearedRange": target.label()}
    if operation == "update" and method != "PUT":
        raise _wrong_method(
            method, path, "GET to read or PUT to write values (POST .../values/{range}:append adds rows)"
        )
    if operation == "append" and method != "POST":
        raise _wrong_method(method, path, "POST")
    _value_input(query, payload, f"values.{operation}", {"insertDataOption"} if operation == "append" else set())
    if operation == "append" and query.get("insertDataOption", "INSERT_ROWS") not in {"INSERT_ROWS", "OVERWRITE"}:
        raise _bad("insertDataOption must be INSERT_ROWS or OVERWRITE.")
    if not isinstance(payload, dict):
        raise _bad('Invalid JSON payload: the body must be an object such as {"values": [["a", "b"]]}.')
    if set(payload) - {"values", "range", "majorDimension"}:
        raise _bad(f"Unknown field {sorted(set(payload) - {'values', 'range', 'majorDimension'})[0]!r} in the body.")
    if "range" in payload and parse_a1(str(payload["range"]), titles) != target:
        raise _bad("The body range must match the range in the URL.")
    dimension = payload.get("majorDimension", "ROWS")
    if dimension not in {"ROWS", "COLUMNS"}:
        raise _bad("majorDimension must be ROWS or COLUMNS.")
    matrix = _matrix(payload, "data", dimension)
    if operation == "update":
        _check_fit(target, matrix, text)
        row, col = target.row1 or 1, target.col1 or 1
        height, width = _write(sheet, row, col, matrix)
        updated = (
            f"{quote_sheet(sheet['title'])}!{column_name(col)}{row}:{column_name(col + width - 1)}{row + height - 1}"
        )
        return 200, {
            "spreadsheetId": sid,
            "updatedRange": updated,
            "updatedRows": height,
            "updatedColumns": width,
            "updatedCells": sum(len(x) for x in matrix),
        }
    first, col = target.row1 or 1, target.col1 or 1
    last_col = target.col2 or GRID_COLUMNS
    last = first - 1
    for index, values in enumerate(sheet["values"], start=1):
        if index >= first and (target.row2 is None or index <= target.row2):
            if any(cell != "" for cell in values[col - 1 : last_col]):
                last = index
    height, width = _write(sheet, last + 1, col, matrix)
    updated = (
        f"{quote_sheet(sheet['title'])}!{column_name(col)}{last + 1}:{column_name(col + width - 1)}{last + height}"
    )
    return 200, {
        "spreadsheetId": sid,
        "tableRange": f"{quote_sheet(sheet['title'])}!{column_name(col)}{first}:{column_name(last_col)}{max(last, first)}",
        "updates": {
            "spreadsheetId": sid,
            "updatedRange": updated,
            "updatedRows": height,
            "updatedColumns": width,
            "updatedCells": sum(len(x) for x in matrix),
        },
    }


# ---------------------------------------------------------------------------
# Chat: method-style endpoints, ok/error envelopes, IDs instead of names.


CHAT_READS = {
    "auth.test",
    "conversations.history",
    "conversations.info",
    "conversations.list",
    "conversations.members",
    "users.info",
    "users.list",
    "users.lookupByEmail",
}
CHAT_WRITES = {"chat.postMessage", "conversations.invite", "conversations.kick", "conversations.setTopic"}


def _slack(error: str, detail: str = "") -> _Fail:
    return _Fail(200, {"ok": False, "error": error} | ({"detail": detail} if detail else {}))


def _channel(store: dict[str, Any], value: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, str) or not value:
        raise _slack("channel_not_found", "Supply the channel ID, for example C04AB12CD.")
    if value in store["channels"]:
        return value, store["channels"][value]
    if any(row["name"] == value.lstrip("#") for row in store["channels"].values()):
        raise _slack(
            "channel_not_found",
            f"{value!r} is a channel name. Pass the channel ID from conversations.list instead.",
        )
    raise _slack("channel_not_found")


def _user_resource(identity: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": identity,
        "name": row["name"],
        "real_name": row["real_name"],
        "deleted": row["deleted"],
        "profile": {"email": row["email"], "real_name": row["real_name"]},
    }


def _channel_resource(identity: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": identity,
        "name": row["name"],
        "is_archived": row["archived"],
        "topic": {"value": row["topic"]},
        "num_members": len(row["members"]),
    }


def _page(rows: list[Any], query: dict[str, Any], scope: dict[str, Any], default: int) -> tuple[list[Any], str]:
    limit = _integer(query.get("limit"), "limit", default, 1, 200)
    try:
        start = _offset(scope, query.get("cursor"), "cursor")
    except _Fail as failure:
        raise _slack("invalid_cursor", "Reuse next_cursor from the same request.") from failure
    following = _token(scope, start + limit) if start + limit < len(rows) else ""
    return rows[start : start + limit], following


def _chat(world: Any, vendor: RealVendor, method: str, path: str, query: dict, payload: Any) -> tuple[int, dict]:
    store = _store(world, vendor)
    match = re.fullmatch(r"/api/([A-Za-z]+\.[A-Za-z]+)", path)
    if not match or match[1] not in CHAT_READS | CHAT_WRITES:
        raise _Fail(404, {"ok": False, "error": "unknown_method", "detail": f"No method at {path}."})
    name = match[1]
    if name in CHAT_READS:
        if method != "GET":
            raise _slack("invalid_request", f"{name} is a read method: use GET with query parameters in params.")
        return 200, _chat_read(world, store, vendor, name, query)
    if method != "POST":
        raise _slack("invalid_request", f"{name} is a write method: use POST with a JSON body.")
    if query:
        raise _slack(
            "invalid_arguments",
            f"{name} reads its arguments from the JSON body; query parameters are not accepted for writes.",
        )
    if not isinstance(payload, dict):
        raise _slack("invalid_arguments", f"{name} requires a JSON object body.")
    return 200, _chat_write(world, store, name, payload)


def _chat_read(world: Any, store: dict, vendor: RealVendor, name: str, query: dict) -> dict[str, Any]:
    context = world.scenario.context
    if name == "auth.test":
        return {"ok": True, "user_id": context["chat_user"], "team": context["organization"]}
    if name == "conversations.list":
        allowed = {"limit", "cursor", "exclude_archived", "types"}
        if set(query) - allowed:
            raise _slack("invalid_arguments", f"Unknown argument {sorted(set(query) - allowed)[0]!r}.")
        hide = _flag(query.get("exclude_archived"), "exclude_archived")
        rows = [
            _channel_resource(key, row)
            for key, row in sorted(store["channels"].items(), key=lambda item: (item[1]["name"], item[0]))
            if not (hide and row["archived"])
        ]
        page, following = _page(rows, query, {"vendor": vendor.name, "list": "channels", "hide": hide}, 5)
        return {"ok": True, "channels": page, "response_metadata": {"next_cursor": following}}
    if name == "users.list":
        if set(query) - {"limit", "cursor"}:
            raise _slack("invalid_arguments", "users.list accepts limit and cursor.")
        rows = [_user_resource(key, row) for key, row in sorted(store["users"].items())]
        page, following = _page(rows, query, {"vendor": vendor.name, "list": "users"}, 8)
        return {"ok": True, "members": page, "response_metadata": {"next_cursor": following}}
    if name == "users.lookupByEmail":
        address = query.get("email")
        if set(query) != {"email"} or not isinstance(address, str):
            raise _slack("invalid_arguments", "users.lookupByEmail requires email.")
        for key, row in sorted(store["users"].items()):
            if row["email"].lower() == address.strip().lower():
                return {"ok": True, "user": _user_resource(key, row)}
        raise _slack("users_not_found")
    if name == "users.info":
        if set(query) != {"user"} or query["user"] not in store["users"]:
            raise _slack("user_not_found", "users.info requires a user ID such as U04AB12CD.")
        return {"ok": True, "user": _user_resource(query["user"], store["users"][query["user"]])}
    if set(query) - {"channel", "limit", "cursor"}:
        raise _slack("invalid_arguments", f"{name} accepts channel, limit and cursor.")
    identity, channel = _channel(store, query.get("channel"))
    if name == "conversations.info":
        return {"ok": True, "channel": _channel_resource(identity, channel)}
    if name == "conversations.members":
        page, following = _page(channel["members"], query, {"vendor": vendor.name, "members": identity}, 10)
        return {"ok": True, "members": page, "response_metadata": {"next_cursor": following}}
    rows = [
        {"type": "message", "ts": key, "user": row["user"], "text": row["text"]}
        | ({"thread_ts": row["thread_ts"]} if row["thread_ts"] else {})
        for key, row in sorted(store["messages"].items(), reverse=True)
        if row["channel"] == identity
    ]
    page, following = _page(rows, query, {"vendor": vendor.name, "history": identity}, 10)
    return {"ok": True, "messages": page, "has_more": bool(following), "response_metadata": {"next_cursor": following}}


def _chat_write(world: Any, store: dict, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    if name == "chat.postMessage":
        ignored = {"mrkdwn", "unfurl_links", "unfurl_media", "link_names", "parse"}
        if set(payload) - {"channel", "text", "thread_ts"} - ignored:
            raise _slack("invalid_arguments", "chat.postMessage accepts channel, text and optional thread_ts.")
        identity, channel = _channel(store, payload.get("channel"))
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise _slack("no_text")
        if channel["archived"]:
            raise _slack("is_archived")
        thread = payload.get("thread_ts") or ""
        if thread and (
            not isinstance(thread, str)
            or thread not in store["messages"]
            or store["messages"][thread]["channel"] != identity
        ):
            raise _slack("thread_not_found")
        sequence = _new_id(world)
        ts = f"{1762900000 + sequence}.{sequence:06d}"
        row = {
            "id": ts,
            "channel": identity,
            "user": world.scenario.context["chat_user"],
            "text": text,
            "thread_ts": thread,
        }
        store["messages"][ts] = row
        message = {"type": "message", "user": row["user"], "text": text, "ts": ts}
        return {
            "ok": True,
            "channel": identity,
            "ts": ts,
            "message": message | ({"thread_ts": thread} if thread else {}),
        }
    identity, channel = _channel(store, payload.get("channel"))
    if channel["archived"]:
        raise _slack("is_archived", "Archived channels cannot be changed.")
    if name == "conversations.setTopic":
        if set(payload) != {"channel", "topic"} or not isinstance(payload["topic"], str):
            raise _slack("invalid_arguments", "conversations.setTopic requires channel and topic.")
        channel["topic"] = payload["topic"]
        return {"ok": True, "channel": _channel_resource(identity, channel)}
    if name == "conversations.invite":
        users = payload.get("users")
        if set(payload) != {"channel", "users"}:
            raise _slack("invalid_arguments", "conversations.invite requires channel and users.")
        users = users.split(",") if isinstance(users, str) else users
        if not isinstance(users, list) or not users or not all(isinstance(x, str) for x in users):
            raise _slack("invalid_arguments", "users is a comma-separated string of user IDs.")
        users = [x.strip() for x in users]
        for user in users:
            if user not in store["users"]:
                raise _slack(
                    "user_not_found",
                    f"{user!r} is not a user ID. Find IDs with users.lookupByEmail or users.list.",
                )
        if set(users) <= set(channel["members"]):
            raise _slack("already_in_channel")
        channel["members"] = sorted(set(channel["members"]) | set(users))
        return {"ok": True, "channel": _channel_resource(identity, channel)}
    user = payload.get("user")
    if set(payload) != {"channel", "user"} or not isinstance(user, str):
        raise _slack("invalid_arguments", "conversations.kick requires channel and user.")
    if user not in store["users"]:
        raise _slack("user_not_found", f"{user!r} is not a user ID.")
    if user not in channel["members"]:
        raise _slack("not_in_channel")
    channel["members"] = sorted(set(channel["members"]) - {user})
    return {"ok": True}


# ---------------------------------------------------------------------------
# CRM REST dialect: properties envelopes, after-cursor paging and filter search.


def _crm_error(status: int, category: str, message: str) -> _Fail:
    return _Fail(status, {"status": "error", "category": category, "message": message})


def _crm_value(schema: dict[str, type], key: str, value: Any) -> Any:
    kind = schema[key]
    if kind is int and isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        value = int(value)
    if type(value) is not kind or kind is int and value < 0:
        raise _crm_error(400, "VALIDATION_ERROR", f"Property {key!r} must be a {kind.__name__}.")
    return value


def _crm_object(identity: str, row: dict[str, Any]) -> dict[str, Any]:
    properties = {key: deepcopy(value) for key, value in row.items() if key != "id"}
    return {"id": identity, "properties": properties, "archived": False}


def _crm_compare(actual: Any, operator: str, value: Any) -> bool:
    if operator == "HAS_PROPERTY":
        return actual not in ("", None)
    if operator in {"IN", "NOT_IN"}:
        values = value if isinstance(value, list) else []
        hit = any(str(actual).lower() == str(x).lower() for x in values)
        return hit if operator == "IN" else not hit
    if operator == "CONTAINS_TOKEN":
        return str(value).lower().strip("*") in str(actual).lower()
    if isinstance(actual, int):
        try:
            value = int(value)
        except (TypeError, ValueError):
            return False
    else:
        actual, value = str(actual).lower(), str(value).lower()
    return {
        "EQ": actual == value,
        "NEQ": actual != value,
        "LT": actual < value,
        "LTE": actual <= value,
        "GT": actual > value,
        "GTE": actual >= value,
    }[operator]


def _crm_search(store: dict, kind: str, payload: Any, scope: dict) -> dict[str, Any]:
    allowed = {"filterGroups", "limit", "after", "properties", "sorts", "query"}
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise _crm_error(
            400, "VALIDATION_ERROR", "Search body accepts filterGroups, query, limit, after and properties."
        )
    schema = APPS["crm"][kind]
    groups = payload.get("filterGroups", [])
    operators = {"EQ", "NEQ", "LT", "LTE", "GT", "GTE", "CONTAINS_TOKEN", "IN", "NOT_IN", "HAS_PROPERTY"}
    if not isinstance(groups, list):
        raise _crm_error(400, "VALIDATION_ERROR", "filterGroups must be a list of {filters: [...]} objects.")
    for group in groups:
        filters = group.get("filters") if isinstance(group, dict) else None
        if not isinstance(filters, list) or not filters:
            raise _crm_error(400, "VALIDATION_ERROR", "Each filter group needs a nonempty filters list.")
        for item in filters:
            if (
                not isinstance(item, dict)
                or item.get("propertyName") not in {*schema, "id"}
                or item.get("operator") not in operators
            ):
                raise _crm_error(
                    400,
                    "VALIDATION_ERROR",
                    "Each filter needs propertyName (a property of this object), operator (EQ, NEQ, LT, LTE, GT, "
                    "GTE, CONTAINS_TOKEN, IN, NOT_IN, HAS_PROPERTY) and value (or values for IN).",
                )
    text = payload.get("query", "")
    if not isinstance(text, str):
        raise _crm_error(400, "VALIDATION_ERROR", "query must be text.")

    def matches(identity: str, record: dict[str, Any]) -> bool:
        if text and text.lower() not in _canonical(record).lower():
            return False
        return not groups or any(
            all(
                _crm_compare(record[f["propertyName"]], f["operator"], f.get("values", f.get("value")))
                for f in group["filters"]
            )
            for group in groups
        )

    rows = [(key, row) for key, row in sorted(store[kind].items()) if matches(key, row)]
    limit = payload.get("limit", 10)
    if type(limit) is not int or not 1 <= limit <= 100:
        raise _crm_error(400, "VALIDATION_ERROR", "limit must be an integer from 1 to 100.")
    scope = scope | {"groups": groups, "query": text}
    try:
        start = _offset(scope, payload.get("after"), "after")
    except _Fail as failure:
        raise _crm_error(400, "VALIDATION_ERROR", "Invalid after cursor; reuse paging.next.after.") from failure
    result: dict[str, Any] = {
        "total": len(rows),
        "results": [_crm_object(k, r) for k, r in rows[start : start + limit]],
    }
    if start + limit < len(rows):
        result["paging"] = {"next": {"after": _token(scope, start + limit)}}
    return result


def _crm_properties(payload: Any, schema: dict[str, type], create: bool) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {"properties"}:
        stray = sorted(set(payload) - {"properties"}) if isinstance(payload, dict) else []
        hint = f" Unexpected top-level field {stray[0]!r}." if stray else ""
        raise _crm_error(
            400,
            "VALIDATION_ERROR",
            'Invalid input JSON: fields must be nested under "properties", e.g. {"properties": {"stage": "..."}}.'
            + hint,
        )
    properties = payload["properties"]
    if not isinstance(properties, dict) or not properties:
        raise _crm_error(400, "VALIDATION_ERROR", "properties must be a nonempty object.")
    unknown = sorted(set(properties) - set(schema))
    if unknown:
        raise _crm_error(400, "VALIDATION_ERROR", f"Property {unknown[0]!r} does not exist on this object type.")
    if create and set(properties) != set(schema):
        missing = sorted(set(schema) - set(properties))
        raise _crm_error(400, "VALIDATION_ERROR", f"Creating this object requires every property; missing {missing}.")
    return {key: _crm_value(schema, key, value) for key, value in properties.items()}


def _crm_rest(world: Any, vendor: RealVendor, method: str, path: str, query: dict, payload: Any) -> tuple[int, dict]:
    store = _store(world, vendor)
    match = re.fullmatch(re.escape(CRM_BASE) + r"/([A-Za-z_]+)(?:/([A-Za-z0-9_-]+))?", path)
    if not match:
        raise _no_route(method, path)
    kind, identity = match.groups()
    if kind not in APPS["crm"]:
        raise _crm_error(404, "OBJECT_NOT_FOUND", f"Unknown object type {kind!r}; use companies, contacts or deals.")
    schema = APPS["crm"][kind]
    scope = {"vendor": vendor.name, "kind": kind}
    if identity == "search":
        if method != "POST":
            raise _wrong_method(method, path, "POST with a JSON search body")
        if query:
            raise _crm_error(400, "VALIDATION_ERROR", "Search criteria go in the JSON body, not query parameters.")
        return 200, _crm_search(store, kind, payload, scope)
    if identity is None:
        if method == "GET":
            if set(query) - {"limit", "after", "properties", "archived"}:
                raise _crm_error(400, "VALIDATION_ERROR", "List accepts limit, after and properties parameters.")
            limit = _integer(query.get("limit"), "limit", 5, 1, 100)
            rows = sorted(store[kind].items())
            try:
                start = _offset(scope, query.get("after"), "after")
            except _Fail as failure:
                raise _crm_error(
                    400, "VALIDATION_ERROR", "Invalid after cursor; reuse paging.next.after."
                ) from failure
            result: dict[str, Any] = {"results": [_crm_object(k, r) for k, r in rows[start : start + limit]]}
            if start + limit < len(rows):
                result["paging"] = {"next": {"after": _token(scope, start + limit)}}
            return 200, result
        if method != "POST":
            raise _wrong_method(method, path, "GET or POST")
        if query:
            raise _crm_error(400, "VALIDATION_ERROR", "Object properties go in the JSON body, not query parameters.")
        key = str(800000 + _new_id(world))
        row = {"id": key, **_crm_properties(payload, schema, True)}
        store[kind][key] = row
        return 201, _crm_object(key, row)
    if identity not in store[kind]:
        raise _crm_error(404, "OBJECT_NOT_FOUND", f"No {kind[:-1] if kind != 'companies' else 'company'} {identity}.")
    if method == "GET":
        if set(query) - {"properties", "archived"}:
            raise _crm_error(400, "VALIDATION_ERROR", "Item reads accept only properties.")
        return 200, _crm_object(identity, store[kind][identity])
    if query:
        raise _crm_error(400, "VALIDATION_ERROR", "Object properties go in the JSON body, not query parameters.")
    if method == "PATCH":
        store[kind][identity].update(_crm_properties(payload, schema, False))
        return 200, _crm_object(identity, store[kind][identity])
    if method == "DELETE":
        if payload is not _ABSENT:
            raise _crm_error(400, "VALIDATION_ERROR", "DELETE takes no body.")
        del store[kind][identity]
        return 204, {}
    raise _wrong_method(method, path, "GET, PATCH or DELETE")


# ---------------------------------------------------------------------------
# CRM GraphQL dialect: a small, strict executor for one fictional schema.


GQL_TYPES: dict[str, dict[str, str]] = {
    "Company": {
        "id": "ID",
        "name": "String",
        "domain": "String",
        "tier": "String",
        "region": "String",
        "owner": "String",
    },
    "Contact": {
        "id": "ID",
        "name": "String",
        "email": "String",
        "company": "String",
        "role": "String",
        "lifecycle": "String",
        "owner": "String",
    },
    "Deal": {
        "id": "ID",
        "name": "String",
        "company": "String",
        "amount": "Int",
        "stage": "String",
        "owner": "String",
        "closeDate": "String",
    },
    "PageInfo": {"hasNextPage": "Boolean", "endCursor": "String"},
}
_GQL_KINDS = {"Company": "companies", "Contact": "contacts", "Deal": "deals"}
for _name in _GQL_KINDS:
    GQL_TYPES[f"{_name}Connection"] = {"nodes": f"[{_name}]", "pageInfo": "PageInfo", "totalCount": "Int"}
GQL_QUERY = {
    **{kind: (f"{name}Connection", {"first", "after", "filter"}) for name, kind in _GQL_KINDS.items()},
    **{name.lower(): (name, {"id"}) for name in _GQL_KINDS},
}
GQL_MUTATION = {
    **{f"update{name}": (name, {"id", "input"}) for name in _GQL_KINDS},
    **{f"create{name}": (name, {"input"}) for name in _GQL_KINDS},
}
GQL_SCHEMA_SDL = (
    "type Query { companies(first: Int = 5, after: String, filter: CompanyFilter): CompanyConnection! "
    "contacts(first: Int = 5, after: String, filter: ContactFilter): ContactConnection! "
    "deals(first: Int = 5, after: String, filter: DealFilter): DealConnection! company(id: ID!): Company "
    "contact(id: ID!): Contact deal(id: ID!): Deal } "
    "type Mutation { updateCompany(id: ID!, input: CompanyInput!): Company updateContact(id: ID!, "
    "input: ContactInput!): Contact updateDeal(id: ID!, input: DealInput!): Deal createCompany(input: "
    "CompanyInput!): Company createContact(input: ContactInput!): Contact createDeal(input: DealInput!): Deal } "
    "type Company { id name domain tier region owner } type Contact { id name email company role lifecycle "
    "owner } type Deal { id name company amount: Int stage owner closeDate } type XConnection { nodes: [X!]! "
    "pageInfo { hasNextPage endCursor } totalCount: Int } Filters and inputs use the same field names as the "
    "object (filters match exact values; inputs for create need every field except id)."
)
_GQL_TOKENS = re.compile(
    r"(?P<skip>[\s,﻿]+|#[^\n]*)|(?P<spread>\.\.\.)|(?P<punct>[!$():=@\[\]{}|&])"
    r'|(?P<block>"""(?:[^"\\]|\\.|"(?!""))*""")|(?P<string>"(?:[^"\\\n]|\\.)*")'
    r"|(?P<number>-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)|(?P<name>[_A-Za-z][_0-9A-Za-z]*)"
)


class _GraphQLError(Exception):
    pass


@dataclass
class _Field:
    alias: str
    name: str
    arguments: dict[str, Any]
    selections: list[_Field] | None


class _Parser:
    def __init__(self, text: str):
        self.tokens: list[tuple[str, str]] = []
        position = 0
        while position < len(text):
            match = _GQL_TOKENS.match(text, position)
            if not match:
                raise _GraphQLError(f"Syntax Error: unexpected character {text[position]!r} at {position}.")
            position = match.end()
            if match.lastgroup != "skip":
                self.tokens.append((match.lastgroup, match.group()))
        self.index = 0

    def peek(self, value: str | None = None) -> bool:
        if self.index >= len(self.tokens):
            return False
        return value is None or self.tokens[self.index][1] == value

    def take(self, value: str | None = None, kind: str | None = None) -> str:
        if self.index >= len(self.tokens):
            raise _GraphQLError(f"Syntax Error: expected {value or kind}, found end of document.")
        token_kind, token = self.tokens[self.index]
        if value is not None and token != value or kind is not None and token_kind != kind:
            raise _GraphQLError(f"Syntax Error: expected {value or kind}, found {token!r}.")
        self.index += 1
        return token

    def document(self) -> list[tuple[str, str | None, dict[str, Any], list[_Field]]]:
        operations = []
        while self.peek():
            if self.peek("{"):
                operations.append(("query", None, {}, self.selections()))
                continue
            kind = self.take(kind="name")
            if kind not in {"query", "mutation"}:
                raise _GraphQLError(f"Syntax Error: unexpected {kind!r}; only query and mutation are supported.")
            name = self.take(kind="name") if self.peek() and self.tokens[self.index][0] == "name" else None
            defaults: dict[str, Any] = {}
            if self.peek("("):
                self.take("(")
                while not self.peek(")"):
                    self.take("$")
                    variable = self.take(kind="name")
                    self.take(":")
                    self.type_reference()
                    defaults[variable] = self.value() if self.peek("=") and self.take("=") else _ABSENT
                self.take(")")
            operations.append((kind, name, defaults, self.selections()))
        if not operations:
            raise _GraphQLError("Syntax Error: the document contains no operation.")
        return operations

    def type_reference(self) -> None:
        if self.peek("["):
            self.take("[")
            self.type_reference()
            self.take("]")
        else:
            self.take(kind="name")
        if self.peek("!"):
            self.take("!")

    def selections(self) -> list[_Field]:
        self.take("{")
        fields = []
        while not self.peek("}"):
            if self.peek("..."):
                raise _GraphQLError("Fragments are not supported by this API; select fields directly.")
            alias = name = self.take(kind="name")
            if self.peek(":"):
                self.take(":")
                name = self.take(kind="name")
            arguments: dict[str, Any] = {}
            if self.peek("("):
                self.take("(")
                while not self.peek(")"):
                    key = self.take(kind="name")
                    self.take(":")
                    arguments[key] = self.value()
                self.take(")")
            fields.append(_Field(alias, name, arguments, self.selections() if self.peek("{") else None))
        self.take("}")
        if not fields:
            raise _GraphQLError("Syntax Error: a selection set cannot be empty.")
        return fields

    def value(self) -> Any:
        if self.peek("$"):
            self.take("$")
            return ("variable", self.take(kind="name"))
        if self.peek("["):
            self.take("[")
            items = []
            while not self.peek("]"):
                items.append(self.value())
            self.take("]")
            return ("list", items)
        if self.peek("{"):
            self.take("{")
            fields = {}
            while not self.peek("}"):
                key = self.take(kind="name")
                self.take(":")
                fields[key] = self.value()
            self.take("}")
            return ("object", fields)
        if self.index >= len(self.tokens):
            raise _GraphQLError("Syntax Error: expected a value, found end of document.")
        kind, token = self.tokens[self.index]
        self.index += 1
        if kind == "string":
            return ("literal", json.loads(token))
        if kind == "block":
            return ("literal", token[3:-3].replace('\\"""', '"""'))
        if kind == "number":
            return ("literal", float(token) if re.search(r"[.eE]", token) else int(token))
        if kind == "name":
            return ("literal", {"true": True, "false": False, "null": None}.get(token, token))
        raise _GraphQLError(f"Syntax Error: unexpected {token!r} where a value was expected.")


def _gql_resolve(node: Any, variables: dict[str, Any], defaults: dict[str, Any]) -> Any:
    kind, value = node
    if kind == "variable":
        if value in variables:
            return variables[value]
        if defaults.get(value, _ABSENT) is not _ABSENT:
            return _gql_resolve(defaults[value], variables, defaults)
        if value not in defaults:
            raise _GraphQLError(f"Variable ${value} is not defined by the operation.")
        raise _GraphQLError(f"Variable ${value} was not provided.")
    if kind == "list":
        return [_gql_resolve(item, variables, defaults) for item in value]
    if kind == "object":
        return {key: _gql_resolve(item, variables, defaults) for key, item in value.items()}
    return value


def _gql_validate(type_name: str, selections: list[_Field] | None, path: str) -> None:
    base = type_name.strip("[]!")
    if base not in GQL_TYPES:
        if selections:
            raise _GraphQLError(f'Field "{path}" must not have a selection since type "{base}" has no subfields.')
        return
    if not selections:
        raise _GraphQLError(f'Field "{path}" of type "{base}" must have a selection of subfields.')
    for field in selections:
        if field.name == "__typename":
            continue
        if field.name not in GQL_TYPES[base]:
            raise _GraphQLError(f'Cannot query field "{field.name}" on type "{base}".')
        if field.arguments:
            raise _GraphQLError(f'Field "{base}.{field.name}" takes no arguments.')
        _gql_validate(GQL_TYPES[base][field.name], field.selections, f"{path}.{field.name}")


def _gql_project(value: Any, type_name: str, selections: list[_Field] | None) -> Any:
    if value is None or selections is None:
        return value
    if isinstance(value, list):
        return [_gql_project(item, type_name.strip("[]!"), selections) for item in value]
    base = type_name.strip("[]!")
    out = {}
    for field in selections:
        if field.name == "__typename":
            out[field.alias] = base
        else:
            out[field.alias] = _gql_project(value[field.name], GQL_TYPES[base][field.name], field.selections)
    return out


def _gql_node(identity: str, row: dict[str, Any]) -> dict[str, Any]:
    fields = {("closeDate" if key == "close_date" else key): value for key, value in row.items() if key != "id"}
    return {"id": identity, **fields}


def _gql_input(value: Any, kind: str, create: bool) -> dict[str, Any]:
    schema = APPS["crm"][kind]
    if not isinstance(value, dict) or not value:
        raise _GraphQLError("input must be a nonempty input object.")
    data = {("close_date" if key == "closeDate" else key): item for key, item in value.items()}
    unknown = sorted(set(data) - set(schema))
    if unknown:
        raise _GraphQLError(f'Field "{unknown[0]}" is not defined by the input type.')
    if create and set(data) != set(schema):
        missing = sorted(("closeDate" if key == "close_date" else key) for key in set(schema) - set(data))
        raise _GraphQLError(f"Create input requires every field; missing {missing}.")
    for key, item in data.items():
        if type(item) is not schema[key] or schema[key] is int and item < 0:
            expected = "Int" if schema[key] is int else "String"
            raise _GraphQLError(f'Field "{key}" expects type {expected}.')
    return data


def _gql_field(world: Any, store: dict, field: _Field, operation: str, arguments: dict[str, Any]) -> Any:
    if operation == "query" and field.name in _GQL_KINDS.values():
        kind = field.name
        first = arguments.get("first", 5)
        if type(first) is not int or not 1 <= first <= 100:
            raise _GraphQLError("first must be an integer from 1 to 100.")
        criteria = arguments.get("filter") or {}
        if not isinstance(criteria, dict):
            raise _GraphQLError("filter must be an input object of field values.")
        criteria = {("close_date" if key == "closeDate" else key): value for key, value in criteria.items()}
        if set(criteria) - {"id", *APPS["crm"][kind]}:
            raise _GraphQLError(f'filter field "{sorted(set(criteria) - set(APPS["crm"][kind]))[0]}" is not defined.')
        rows = [
            _gql_node(key, row)
            for key, row in sorted(store[kind].items())
            if all(row[name] == value for name, value in criteria.items())
        ]
        scope = {"kind": kind, "filter": criteria}
        start = _offset(scope, arguments.get("after"), "after") if arguments.get("after") else 0
        following = start + first < len(rows)
        return {
            "nodes": rows[start : start + first],
            "pageInfo": {"hasNextPage": following, "endCursor": _token(scope, start + first) if following else None},
            "totalCount": len(rows),
        }
    if operation == "query":
        kind = _GQL_KINDS[field.name.capitalize()]
        identity = arguments.get("id")
        row = store[kind].get(identity) if isinstance(identity, str) else None
        return None if row is None else _gql_node(identity, row)
    kind = _GQL_KINDS[field.name.removeprefix("update").removeprefix("create")]
    if field.name.startswith("update"):
        identity = arguments.get("id")
        if not isinstance(identity, str) or identity not in store[kind]:
            raise _GraphQLError(f"No {kind} record has id {identity!r}.")
        store[kind][identity].update(_gql_input(arguments.get("input"), kind, False))
        return _gql_node(identity, store[kind][identity])
    data = _gql_input(arguments.get("input"), kind, True)
    identity = str(800000 + _new_id(world))
    store[kind][identity] = {"id": identity, **data}
    return _gql_node(identity, data)


def _crm_graphql(world: Any, vendor: RealVendor, method: str, path: str, query: dict, payload: Any) -> tuple:
    if path != "/graphql":
        raise _no_route(method, path)
    if method != "POST":
        raise _wrong_method(method, path, "POST with a JSON body {query, variables}")
    if query:
        raise _bad("GraphQL requests carry query and variables in the JSON body, not in params.")
    if not isinstance(payload, dict) or not isinstance(payload.get("query"), str):
        raise _bad('Body must be {"query": "<GraphQL document>", "variables": {...}}.')
    if set(payload) - {"query", "variables", "operationName"}:
        raise _bad("GraphQL body accepts query, variables and operationName.")
    variables = payload.get("variables") or {}
    if not isinstance(variables, dict):
        raise _bad("variables must be a JSON object.")
    try:
        operations = _Parser(payload["query"]).document()
    except _GraphQLError as error:
        return 400, {"errors": [{"message": str(error)}]}
    wanted = payload.get("operationName")
    chosen = [x for x in operations if wanted is None or x[1] == wanted]
    if len(chosen) != 1:
        message = (
            "Must provide operationName when the document has several operations."
            if wanted is None
            else (f'Unknown operation named "{wanted}".')
        )
        return 200, {"errors": [{"message": message}], "data": None}
    operation, _, defaults, selections = chosen[0]
    roots = GQL_QUERY if operation == "query" else GQL_MUTATION
    root_type = "Query" if operation == "query" else "Mutation"
    try:
        for field in selections:
            if field.name == "__typename":
                continue
            if field.name not in roots:
                raise _GraphQLError(f'Cannot query field "{field.name}" on type "{root_type}".')
            result_type, accepted = roots[field.name]
            unknown = sorted(set(field.arguments) - accepted)
            if unknown:
                raise _GraphQLError(f'Unknown argument "{unknown[0]}" on field "{root_type}.{field.name}".')
            _gql_validate(result_type, field.selections, field.name)
        resolved = [
            {key: _gql_resolve(node, variables, defaults) for key, node in field.arguments.items()}
            for field in selections
        ]
    except _GraphQLError as error:
        return 200, {"errors": [{"message": str(error)}], "data": None}
    data: dict[str, Any] = {}
    errors = []
    store = _store(world, vendor)
    for field, arguments in zip(selections, resolved):
        if field.name == "__typename":
            data[field.alias] = root_type
            continue
        try:
            value = _gql_field(world, store, field, operation, arguments)
            data[field.alias] = _gql_project(value, roots[field.name][0], field.selections)
        except (_GraphQLError, _Fail) as error:
            message = str(error) if isinstance(error, _GraphQLError) else "Invalid cursor; reuse pageInfo.endCursor."
            errors.append({"message": message, "path": [field.alias]})
            data[field.alias] = None
    return 200, {"data": data} | ({"errors": errors} if errors else {})


# ---------------------------------------------------------------------------
# Catalog search over endpoint descriptions (BM25), never over task records.


def _p(location: str, kind: str, required: bool, description: str) -> dict[str, Any]:
    return {"in": location, "type": kind, "required": required, "description": description}


_ENDPOINTS: dict[str, list[dict[str, Any]]] = {
    "mail": [
        {
            "op": "messages.list",
            "method": "GET",
            "path": MAIL_BASE + "/messages",
            "description": "List or search mailbox messages; returns message and thread IDs only, newest first.",
            "parameters": {
                "q": _p("query", "string", False, "Search: from:, to:, subject:, label:, is:unread, in:inbox, words."),
                "maxResults": _p("query", "integer", False, "Page size, default 10, max 50."),
                "pageToken": _p("query", "string", False, "nextPageToken from the previous page."),
                "labelIds": _p("query", "string|array", False, "Only messages carrying all of these label IDs."),
            },
            "response": "{messages: [{id, threadId}], nextPageToken?, resultSizeEstimate}",
        },
        {
            "op": "messages.get",
            "method": "GET",
            "path": MAIL_BASE + "/messages/{id}",
            "description": "Get one message: headers, label IDs and the body as base64url in payload.body.data.",
            "parameters": {
                "id": _p("path", "string", True, "Message ID."),
                "format": _p("query", "string", False, "full (default), metadata, minimal or raw."),
            },
            "response": "{id, threadId, labelIds, snippet, payload: {headers: [{name, value}], body: {size, data}}}",
        },
        {
            "op": "messages.send",
            "method": "POST",
            "path": MAIL_BASE + "/messages/send",
            "description": "Send an email. The whole RFC 2822 message (To, Cc, Subject headers, blank line, text) "
            "is base64url-encoded into raw. Reply in a thread by adding threadId.",
            "parameters": {
                "raw": _p("body", "string", True, "base64url-encoded RFC 2822 message."),
                "threadId": _p("body", "string", False, "Thread to reply in."),
            },
            "request": 'Body: {"raw": "<base64url of \'To: a@b.example\\r\\nSubject: Hi\\r\\n\\r\\nText\'>", '
            '"threadId": "<optional>"}',
            "response": "{id, threadId, labelIds}",
        },
        {
            "op": "messages.modify",
            "method": "POST",
            "path": MAIL_BASE + "/messages/{id}/modify",
            "description": "Add or remove labels on a message, e.g. remove UNREAD to mark read or INBOX to archive.",
            "parameters": {
                "id": _p("path", "string", True, "Message ID."),
                "addLabelIds": _p("body", "array", False, "Label IDs to add."),
                "removeLabelIds": _p("body", "array", False, "Label IDs to remove."),
            },
            "request": 'Body: {"addLabelIds": ["Label_2"], "removeLabelIds": ["UNREAD"]}',
            "response": "{id, threadId, labelIds}",
        },
        {
            "op": "labels.list",
            "method": "GET",
            "path": MAIL_BASE + "/labels",
            "description": "List system and user labels with their IDs and names.",
            "parameters": {},
            "response": "{labels: [{id, name, type}]}",
        },
        {
            "op": "profile.get",
            "method": "GET",
            "path": MAIL_BASE + "/profile",
            "description": "Get the connected mailbox address.",
            "parameters": {},
            "response": "{emailAddress, messagesTotal}",
        },
    ],
    "sheets": [
        {
            "op": "spreadsheets.list",
            "method": "GET",
            "path": SHEETS_BASE,
            "description": "List spreadsheets (workbooks) and their IDs; q filters by title.",
            "parameters": {
                "q": _p("query", "string", False, "Title substring."),
                "pageSize": _p("query", "integer", False, "Default 10."),
                "pageToken": _p("query", "string", False, "nextPageToken."),
            },
            "response": "{spreadsheets: [{spreadsheetId, title}], nextPageToken?}",
        },
        {
            "op": "spreadsheets.get",
            "method": "GET",
            "path": SHEETS_BASE + "/{spreadsheetId}",
            "description": "Get spreadsheet metadata including worksheet (tab) titles.",
            "parameters": {"spreadsheetId": _p("path", "string", True, "Spreadsheet ID.")},
            "response": "{spreadsheetId, properties: {title}, sheets: [{properties: {sheetId, title, index}}]}",
        },
        {
            "op": "values.get",
            "method": "GET",
            "path": SHEETS_BASE + "/{spreadsheetId}/values/{range}",
            "description": "Read cell values in an A1 range such as 'Tab name'!A1:D20, Tab!A:D or Tab.",
            "parameters": {
                "spreadsheetId": _p("path", "string", True, "Spreadsheet ID."),
                "range": _p("path", "string", True, "A1 range; quote tab names with spaces."),
                "majorDimension": _p("query", "string", False, "ROWS (default) or COLUMNS."),
            },
            "response": "{range, majorDimension, values: [[...]]} (trailing empty cells omitted)",
        },
        {
            "op": "values.update",
            "method": "PUT",
            "path": SHEETS_BASE + "/{spreadsheetId}/values/{range}",
            "description": "Write cell values into an A1 range (a single start cell expands to fit).",
            "parameters": {
                "spreadsheetId": _p("path", "string", True, "Spreadsheet ID."),
                "range": _p("path", "string", True, "A1 range."),
                "valueInputOption": _p("query", "string", True, "RAW or USER_ENTERED."),
                "values": _p("body", "array", True, "2D array of rows."),
            },
            "request": 'Params: {"valueInputOption": "RAW"}; Body: {"values": [["a", "b"]]}',
            "response": "{spreadsheetId, updatedRange, updatedRows, updatedColumns, updatedCells}",
        },
        {
            "op": "values.append",
            "method": "POST",
            "path": SHEETS_BASE + "/{spreadsheetId}/values/{range}:append",
            "description": "Append rows after the last row of the table found in the range.",
            "parameters": {
                "spreadsheetId": _p("path", "string", True, "Spreadsheet ID."),
                "range": _p("path", "string", True, "A1 range of the table, e.g. Log!A:D."),
                "valueInputOption": _p("query", "string", True, "RAW or USER_ENTERED."),
                "values": _p("body", "array", True, "2D array of new rows."),
            },
            "request": 'Params: {"valueInputOption": "RAW"}; Body: {"values": [["a", "b"], ["c", "d"]]}',
            "response": "{spreadsheetId, tableRange, updates: {updatedRange, updatedRows}}",
        },
        {
            "op": "values.batchGet",
            "method": "GET",
            "path": SHEETS_BASE + "/{spreadsheetId}/values:batchGet",
            "description": "Read several A1 ranges at once.",
            "parameters": {
                "spreadsheetId": _p("path", "string", True, "Spreadsheet ID."),
                "ranges": _p("query", "array", True, "A1 ranges."),
            },
            "response": "{spreadsheetId, valueRanges: [{range, values}]}",
        },
        {
            "op": "values.batchUpdate",
            "method": "POST",
            "path": SHEETS_BASE + "/{spreadsheetId}/values:batchUpdate",
            "description": "Write several A1 ranges at once; valueInputOption goes in the body here.",
            "parameters": {
                "spreadsheetId": _p("path", "string", True, "Spreadsheet ID."),
                "valueInputOption": _p("body", "string", True, "RAW or USER_ENTERED."),
                "data": _p("body", "array", True, "[{range, values}]"),
            },
            "request": 'Body: {"valueInputOption": "RAW", "data": [{"range": "Tab!E2", "values": [["x"]]}]}',
            "response": "{spreadsheetId, totalUpdatedCells, totalUpdatedRanges}",
        },
    ],
    "chat": [
        {
            "op": "conversations.list",
            "method": "GET",
            "path": "/api/conversations.list",
            "description": "List channels with their IDs, names and topics (paginated by cursor).",
            "parameters": {
                "limit": _p("query", "integer", False, "Default 5."),
                "cursor": _p("query", "string", False, "response_metadata.next_cursor."),
                "exclude_archived": _p("query", "boolean", False, "Hide archived channels."),
            },
            "response": "{ok, channels: [{id, name, is_archived, topic, num_members}], response_metadata}",
        },
        {
            "op": "conversations.history",
            "method": "GET",
            "path": "/api/conversations.history",
            "description": "Read a channel's messages, newest first.",
            "parameters": {
                "channel": _p("query", "string", True, "Channel ID."),
                "limit": _p("query", "integer", False, "Default 10."),
                "cursor": _p("query", "string", False, "next_cursor."),
            },
            "response": "{ok, messages: [{ts, user, text, thread_ts?}], has_more, response_metadata}",
        },
        {
            "op": "conversations.members",
            "method": "GET",
            "path": "/api/conversations.members",
            "description": "List the user IDs that are members of a channel.",
            "parameters": {"channel": _p("query", "string", True, "Channel ID.")},
            "response": "{ok, members: [user IDs], response_metadata}",
        },
        {
            "op": "conversations.info",
            "method": "GET",
            "path": "/api/conversations.info",
            "description": "Get one channel by ID.",
            "parameters": {"channel": _p("query", "string", True, "Channel ID.")},
            "response": "{ok, channel}",
        },
        {
            "op": "users.list",
            "method": "GET",
            "path": "/api/users.list",
            "description": "List workspace users with IDs and emails.",
            "parameters": {
                "limit": _p("query", "integer", False, "Default 8."),
                "cursor": _p("query", "string", False, ""),
            },
            "response": "{ok, members: [{id, name, real_name, profile: {email}}], response_metadata}",
        },
        {
            "op": "users.lookupByEmail",
            "method": "GET",
            "path": "/api/users.lookupByEmail",
            "description": "Find a user's ID from their email address.",
            "parameters": {"email": _p("query", "string", True, "Email address.")},
            "response": "{ok, user: {id, name, profile}}",
        },
        {
            "op": "chat.postMessage",
            "method": "POST",
            "path": "/api/chat.postMessage",
            "description": "Post a message to a channel (by channel ID); thread_ts replies in a thread.",
            "parameters": {
                "channel": _p("body", "string", True, "Channel ID, not its name."),
                "text": _p("body", "string", True, "Message text."),
                "thread_ts": _p("body", "string", False, "Parent message ts."),
            },
            "request": 'Body: {"channel": "C04AB12CD", "text": "..."}',
            "response": "{ok, channel, ts, message}; failures return ok: false with an error code",
        },
        {
            "op": "conversations.invite",
            "method": "POST",
            "path": "/api/conversations.invite",
            "description": "Invite users (comma-separated user IDs) to a channel.",
            "parameters": {
                "channel": _p("body", "string", True, "Channel ID."),
                "users": _p("body", "string", True, "Comma-separated user IDs."),
            },
            "request": 'Body: {"channel": "C04AB12CD", "users": "U01,U02"}',
            "response": "{ok, channel}",
        },
        {
            "op": "conversations.kick",
            "method": "POST",
            "path": "/api/conversations.kick",
            "description": "Remove one user (by ID) from a channel.",
            "parameters": {
                "channel": _p("body", "string", True, "Channel ID."),
                "user": _p("body", "string", True, "User ID."),
            },
            "response": "{ok}",
        },
        {
            "op": "conversations.setTopic",
            "method": "POST",
            "path": "/api/conversations.setTopic",
            "description": "Set a channel topic.",
            "parameters": {
                "channel": _p("body", "string", True, "Channel ID."),
                "topic": _p("body", "string", True, "New topic."),
            },
            "response": "{ok, channel}",
        },
    ],
    "crm_rest": [
        {
            "op": "objects.list",
            "method": "GET",
            "path": CRM_BASE + "/{objectType}",
            "description": "List CRM companies, contacts or deals; page with paging.next.after.",
            "parameters": {
                "objectType": _p("path", "string", True, "companies, contacts or deals."),
                "limit": _p("query", "integer", False, "Default 5, max 100."),
                "after": _p("query", "string", False, "paging.next.after."),
            },
            "response": "{results: [{id, properties: {...}}], paging?: {next: {after}}}",
        },
        {
            "op": "objects.search",
            "method": "POST",
            "path": CRM_BASE + "/{objectType}/search",
            "description": "Search CRM records by property filters (filters in one group are ANDed; groups are ORed).",
            "parameters": {
                "objectType": _p("path", "string", True, "companies, contacts or deals."),
                "filterGroups": _p("body", "array", False, "[{filters: [{propertyName, operator, value}]}]"),
                "query": _p("body", "string", False, "Free text."),
                "limit": _p("body", "integer", False, "Default 10."),
                "after": _p("body", "string", False, "paging.next.after."),
            },
            "request": 'Body: {"filterGroups": [{"filters": [{"propertyName": "email", "operator": "EQ", '
            '"value": "x@y.example"}]}]}',
            "response": "{total, results: [{id, properties}], paging?}",
        },
        {
            "op": "objects.get",
            "method": "GET",
            "path": CRM_BASE + "/{objectType}/{id}",
            "description": "Get one CRM record.",
            "parameters": {"objectType": _p("path", "string", True, ""), "id": _p("path", "string", True, "")},
            "response": "{id, properties}",
        },
        {
            "op": "objects.update",
            "method": "PATCH",
            "path": CRM_BASE + "/{objectType}/{id}",
            "description": "Update CRM record properties.",
            "parameters": {
                "objectType": _p("path", "string", True, ""),
                "id": _p("path", "string", True, ""),
                "properties": _p("body", "object", True, "Only the properties to change."),
            },
            "request": 'Body: {"properties": {"stage": "..."}}',
            "response": "{id, properties}",
        },
        {
            "op": "objects.create",
            "method": "POST",
            "path": CRM_BASE + "/{objectType}",
            "description": "Create a CRM record; every property is required.",
            "parameters": {
                "objectType": _p("path", "string", True, ""),
                "properties": _p("body", "object", True, "All properties of the object type."),
            },
            "request": 'Body: {"properties": {...}}',
            "response": "{id, properties}",
        },
    ],
    "crm_graphql": [
        {
            "op": "graphql",
            "method": "POST",
            "path": "/graphql",
            "description": "GraphQL API for CRM companies, contacts and deals: queries with first/after/filter "
            "paging and update/create mutations.",
            "parameters": {
                "query": _p("body", "string", True, "GraphQL document."),
                "variables": _p("body", "object", False, "Variables for the document."),
            },
            "request": 'Body: {"query": "mutation($id: ID!, $input: DealInput!) { updateDeal(id: $id, input: $input) '
            '{ id stage } }", "variables": {"id": "...", "input": {"stage": "..."}}}. Schema: ' + GQL_SCHEMA_SDL,
            "response": "{data, errors?}",
        },
    ],
}


def _endpoint_family(vendor: RealVendor) -> str:
    return f"crm_{vendor.dialect}" if vendor.app == "crm" else vendor.app


CATALOG = [
    {
        "id": f"{vendor.short}.{spec['op']}",
        "vendor": vendor.short,
        "app": vendor.app,
        "method": spec["method"],
        "url": vendor.base + spec["path"],
        "description": spec["description"],
        "parameters": spec["parameters"],
        **({"request": spec["request"]} if "request" in spec else {}),
        "response": spec["response"],
    }
    for vendor in REAL_VENDORS.values()
    for spec in _ENDPOINTS[_endpoint_family(vendor)]
]


def _terms(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower())
    return [word[:-1] if len(word) > 3 and word.endswith("s") else word for word in words]


_DOCUMENTS = [
    _terms(
        " ".join(
            [entry["id"], entry["app"], entry["method"], entry["url"], entry["description"], *entry["parameters"]]
        )
    )
    for entry in CATALOG
]
_AVERAGE = sum(map(len, _DOCUMENTS)) / len(_DOCUMENTS)
_FREQUENCY = {term: sum(term in doc for doc in _DOCUMENTS) for doc in _DOCUMENTS for term in doc}


def search(world: Any, query: str, top_k: int = 5, connected_only: bool = False) -> dict[str, Any]:
    terms = set(_terms(query))
    scored = []
    for index, (entry, document) in enumerate(zip(CATALOG, _DOCUMENTS)):
        connected = f"gw_{entry['vendor']}" in world.scenario.connected
        if connected_only and not connected:
            continue
        score = 0.0
        for term in terms:
            count = document.count(term)
            if count:
                idf = math.log(1 + (len(_DOCUMENTS) - _FREQUENCY[term] + 0.5) / (_FREQUENCY[term] + 0.5))
                score += idf * count * 2.2 / (count + 1.2 * (0.25 + 0.75 * len(document) / _AVERAGE))
        if score > 0:
            scored.append((-score, index, entry | {"connected": connected}))
    results = [entry for _, _, entry in sorted(scored, key=lambda item: item[:2])[:top_k]]
    return {
        "results": results,
        "usage": "Results describe endpoints, not records. Call workspace_fetch with the url (fill {placeholders}), "
        "JSON-encoded params for query parameters and a JSON-encoded body for request bodies.",
    }
