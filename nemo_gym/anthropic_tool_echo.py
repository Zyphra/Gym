# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Send the backend the assistant turns it served, whatever the Messages client echoes.

A Messages client stores the assistant turns it receives and sends them back on
every later request. Claude Code changes them on the way: it drops a leading
``cd <cwd> && `` from Bash commands, drops whitespace-only text blocks, fills the
input-schema defaults the model left out (Edit ``replace_all``) and strips
trailing whitespace from Edit's ``new_string`` lines. An append-only episode
endpoint compares the history with what it served and rejects the continuation.

The model server knows what it served. ``messages()`` records every served turn
that has tool calls, keyed by rollout and by the turn's ``tool_use`` ids. Before it
converts the next request, each echoed assistant turn that matches a served turn
gets the exact served content back. A turn matches when it is an assistant
message with the same ``tool_use`` ids and tool names in the same order, and its
blocks are the served blocks in served order, some possibly left out. Every
difference is written to the rollout's restore log, classified as
``cd_strip``, ``schema_default_fill``, ``whitespace`` or ``other``.

Anything else passes through: turns that match no served turn (a compaction
summary, a dropped or added turn, other ids, reordered or added blocks), and all
user and ``tool_result`` content. The endpoint's append-only check still sees
those.

Served turns are per-rollout JSONL files (``<rollout>.jsonl``, removed a day after
their last write) under one directory per server identity, shared by all of the
server's uvicorn workers; ``NEMO_GYM_SERVED_TOOL_INPUT_DIR`` sets the parent
directory. The restore log sits beside them as ``<rollout>.restores.jsonl`` and is
kept: one entry per changed turn and change set, with the turn's ``tool_use`` ids,
counts per kind and the served and echoed value of each difference. Requests
without a rollout id use a bounded per-process map and log through the module
logger.
"""

from __future__ import annotations

import contextlib
import copy
import fcntl
import hashlib
import itertools
import json
import logging
import os
import re
import stat
import tempfile
import threading
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Mapping, Optional


logger = logging.getLogger(__name__)

STORE_DIR_ENV = "NEMO_GYM_SERVED_TOOL_INPUT_DIR"
RETENTION_SECONDS = 24 * 3600
PRUNE_INTERVAL_SECONDS = 3600
MEMORY_LIMIT = 4096

# Fields a client adds to blocks it echoes; they do not reach the converted history.
_CLIENT_BLOCK_FIELDS = frozenset({"signature", "cache_control"})
# Claude Code's own test for a directory it strips from ``cd <dir> && <command>``
# (2.1.284: HOn): an absolute POSIX or drive path without whitespace, quotes or
# shell metacharacters. Used only to name the difference in the log.
_CLI_CWD = re.compile(r"""^(?:/|[A-Za-z]:[\\/])[^\s"'`$;&|<>*?(){}\[\]!#]{0,1023}$""")
_ROLLOUT_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_SEGMENT_SEARCH_LIMIT = 12


def turn_key(tool_use_ids: list[str]) -> str:
    return "turn:" + json.dumps(list(tool_use_ids))


def _tool_uses(content: list) -> list[Mapping[str, Any]]:
    return [block for block in content if isinstance(block, Mapping) and block.get("type") == "tool_use"]


def _client_view(block: Any) -> Any:
    if not isinstance(block, Mapping):
        return block
    return {key: value for key, value in block.items() if key not in _CLIENT_BLOCK_FIELDS}


def _blank_text(block: Any) -> bool:
    return isinstance(block, Mapping) and block.get("type") == "text" and not str(block.get("text") or "").strip()


def _align_segment(served: list, echoed: list, offset: int) -> Optional[list[Optional[int]]]:
    """Map a run of non-tool blocks onto its echo: same types in served order, fewest changes."""
    best: Optional[tuple[int, tuple[int, ...]]] = None
    candidates = (
        itertools.combinations(range(len(served)), len(echoed))
        if len(served) <= _SEGMENT_SEARCH_LIMIT
        else [tuple(range(len(echoed)))]
    )
    for chosen in candidates:
        pairs = list(zip(chosen, echoed))
        if any(
            not isinstance(echo, Mapping) or echo.get("type") != served[index].get("type") for index, echo in pairs
        ):
            continue
        changed = sum(_client_view(served[index]) != _client_view(echo) for index, echo in pairs)
        left_out = sum(not _blank_text(served[index]) for index in set(range(len(served))) - set(chosen))
        if best is None or changed + left_out < best[0]:
            best = (changed + left_out, chosen)
    if best is None:
        return None
    where = dict(zip(best[1], range(len(echoed))))
    return [offset + where[index] if index in where else None for index in range(len(served))]


def _align(served: list, echoed: list) -> Optional[list[Optional[int]]]:
    """For each served block, the index of its echo, or None when the echo left it out.

    The echo matches when it keeps every served ``tool_use`` (same id and name, same
    order) and, between them, the served blocks in served order with some left out.
    Otherwise None.
    """
    served_tools = [index for index, block in enumerate(served) if block.get("type") == "tool_use"]
    echoed_tools = [
        index for index, block in enumerate(echoed) if isinstance(block, Mapping) and block.get("type") == "tool_use"
    ]
    if len(served_tools) != len(echoed_tools):
        return None
    for served_index, echoed_index in zip(served_tools, echoed_tools):
        for field in ("id", "name"):
            if served[served_index].get(field) != echoed[echoed_index].get(field):
                return None
    mapping: list[Optional[int]] = []
    served_start = echoed_start = 0
    for served_end, echoed_end in [*zip(served_tools, echoed_tools), (len(served), len(echoed))]:
        segment = _align_segment(served[served_start:served_end], echoed[echoed_start:echoed_end], echoed_start)
        if segment is None:
            return None
        mapping.extend(segment)
        if served_end < len(served):
            mapping.append(echoed_end)
        served_start, echoed_start = served_end + 1, echoed_end + 1
    return mapping


def _strip_lines(text: str) -> str:
    return "\n".join(line.rstrip(" \t") for line in text.split("\n"))


def _text_kind(served: Any, echoed: Any) -> str:
    if isinstance(served, str) and isinstance(echoed, str):
        if served.strip() == echoed.strip() or _strip_lines(served) == _strip_lines(echoed):
            return "whitespace"
    return "other"


def _cd_strip(tool: str, key: str, served: Any, echoed: Any) -> bool:
    if tool != "Bash" or key != "command" or not isinstance(served, str) or not served.startswith("cd "):
        return False
    directory, separator, rest = served[3:].partition(" && ")
    return bool(separator) and bool(_CLI_CWD.match(directory)) and rest == echoed


def _schema_defaults(tools: Any) -> dict[str, dict[str, Any]]:
    defaults: dict[str, dict[str, Any]] = {}
    for tool in tools or []:
        if isinstance(tool, Mapping) and isinstance(tool.get("name"), str):
            properties = (tool.get("input_schema") or {}).get("properties") or {}
            defaults[tool["name"]] = {
                key: schema["default"]
                for key, schema in properties.items()
                if isinstance(schema, Mapping) and "default" in schema
            }
    return defaults


def _differences(served: list, echoed: list, mapping: list, defaults: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    """Field-level differences between a served turn and its echo, each with a kind."""
    found = []
    for index, (block, echo_index) in enumerate(zip(served, mapping)):
        kind_of = block.get("type")
        if echo_index is None:
            blank = _blank_text(block)
            found.append(
                {
                    "block": index,
                    "type": kind_of,
                    "field": None,
                    "change": "left_out",
                    "kind": "whitespace" if blank else "other",
                    "served": block,
                    "echoed": None,
                }
            )
            continue
        echo = _client_view(echoed[echo_index])
        base = _client_view(block)
        if base == echo:
            continue
        if kind_of == "tool_use":
            tool = block.get("name")
            served_input = base.get("input") if isinstance(base.get("input"), Mapping) else {}
            echo_input = echo.get("input") if isinstance(echo.get("input"), Mapping) else {}
            for key in sorted(set(served_input) | set(echo_input)):
                if key in served_input and key in echo_input and served_input[key] == echo_input[key]:
                    continue
                if key not in served_input:
                    filled = key in defaults.get(tool, {}) and defaults[tool][key] == echo_input[key]
                    kind, change = ("schema_default_fill" if filled else "other"), "added"
                elif key not in echo_input:
                    kind, change = "other", "removed"
                elif _cd_strip(tool, key, served_input[key], echo_input[key]):
                    kind, change = "cd_strip", "changed"
                else:
                    kind, change = _text_kind(served_input[key], echo_input[key]), "changed"
                found.append(
                    {
                        "block": index,
                        "type": "tool_use",
                        "tool": tool,
                        "id": block.get("id"),
                        "field": f"input.{key}",
                        "change": change,
                        "kind": kind,
                        "served": served_input.get(key),
                        "echoed": echo_input.get(key),
                    }
                )
            for key in sorted((set(base) | set(echo)) - {"input"}):
                if base.get(key) != echo.get(key):
                    found.append(
                        {
                            "block": index,
                            "type": "tool_use",
                            "field": key,
                            "change": "changed",
                            "kind": "other",
                            "served": base.get(key),
                            "echoed": echo.get(key),
                        }
                    )
        else:
            for key in sorted(set(base) | set(echo)):
                if base.get(key) != echo.get(key):
                    found.append(
                        {
                            "block": index,
                            "type": kind_of,
                            "field": key,
                            "change": "changed",
                            "kind": _text_kind(base.get(key), echo.get(key)),
                            "served": base.get(key),
                            "echoed": echo.get(key),
                        }
                    )
    return found


class ServedToolInputStore:
    """Served turns with tool calls, and the restore log, of one model server, per rollout."""

    def __init__(
        self, root: Optional[Path], *, retention_s: float = RETENTION_SECONDS, memory_limit: int = MEMORY_LIMIT
    ):
        self.root = root
        self.retention_s = retention_s
        self.memory_limit = memory_limit
        self._memory: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._logged: "OrderedDict[str, None]" = OrderedDict()
        self._lock = threading.Lock()
        self._last_prune = 0.0
        if root is not None:
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            info = root.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise ValueError(f"served tool input store {root} must be a private directory owned by this user")

    def _path(self, rollout_id: str, suffix: str = ".jsonl") -> Path:
        if not _ROLLOUT_ID.match(rollout_id) or self.root is None:
            raise ValueError(f"invalid rollout id for the served tool input store: {rollout_id!r}")
        return self.root / f"{rollout_id}{suffix}"

    def _remember(self, mapping: "OrderedDict", key: str, value: Any) -> None:
        mapping[key] = value
        mapping.move_to_end(key)
        while len(mapping) > self.memory_limit:
            mapping.popitem(last=False)

    def record(self, rollout_id: Optional[str], records: list[dict[str, Any]]) -> None:
        if not records:
            return
        if rollout_id is None or self.root is None:
            with self._lock:
                for record in records:
                    self._remember(self._memory, record["id"], record)
            return
        self._append(self._path(rollout_id), records)
        self._prune()

    def lookup(self, rollout_id: Optional[str], ids: set[str]) -> dict[str, dict[str, Any]]:
        if not ids:
            return {}
        if rollout_id is None or self.root is None:
            with self._lock:
                return {key: self._memory[key] for key in ids if key in self._memory}
        found: dict[str, dict[str, Any]] = {}
        for record in self._read(self._path(rollout_id)):
            if record.get("id") in ids:
                found[record["id"]] = record
        return found

    def log_restores(self, rollout_id: Optional[str], entries: list[dict[str, Any]]) -> None:
        """Append each restore entry once per rollout (a turn is echoed again on every later request)."""
        if not entries:
            return
        if rollout_id is None or self.root is None:
            with self._lock:
                new = [entry for entry in entries if entry["key"] not in self._logged]
                for entry in new:
                    self._remember(self._logged, entry["key"], None)
            for entry in new:
                logger.info("restored a served turn the client changed: %s", json.dumps(entry, default=str))
            return
        path = self._path(rollout_id, ".restores.jsonl")
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            with open(descriptor, "rb", closefd=False) as stream:
                stream.seek(0)
                seen = set()
                for line in stream:
                    with contextlib.suppress(ValueError):
                        seen.add(json.loads(line).get("key"))
            new = [entry for entry in entries if entry["key"] not in seen]
            if new:
                os.write(descriptor, "".join(json.dumps(entry, default=str) + "\n" for entry in new).encode())
        finally:
            os.close(descriptor)

    def _append(self, path: Path, records: list[dict[str, Any]]) -> None:
        data = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode()
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            os.write(descriptor, data)
        finally:
            os.close(descriptor)

    def _read(self, path: Path) -> list[dict[str, Any]]:
        try:
            stream = path.open("rb")
        except FileNotFoundError:
            return []
        records = []
        with stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
            for line in stream:
                with contextlib.suppress(ValueError):
                    records.append(json.loads(line))
        return records

    def _prune(self) -> None:
        now = time.time()
        if self.root is None or now - self._last_prune < PRUNE_INTERVAL_SECONDS:
            return
        self._last_prune = now
        for path in self.root.glob("*.jsonl"):
            if path.name.endswith(".restores.jsonl"):
                continue
            with contextlib.suppress(OSError):
                if now - path.stat().st_mtime > self.retention_s:
                    path.unlink()


_STORES: dict[str, ServedToolInputStore] = {}
_STORES_LOCK = threading.Lock()


def store_for(server_name: str, host: Any, port: Any) -> ServedToolInputStore:
    """One store per server identity: every uvicorn worker of a server opens the same directory.

    ``NEMO_GYM_SERVED_TOOL_INPUT_DIR`` overrides the parent directory (default: the
    temporary directory). A store that cannot be made private falls back to memory.
    """
    identity = re.sub(r"[^A-Za-z0-9._-]", "_", f"{server_name}-{host}-{port}")
    parent = os.environ.get(STORE_DIR_ENV) or str(
        Path(tempfile.gettempdir()) / f"nemo-gym-served-tool-inputs-{os.geteuid()}"
    )
    key = str(Path(parent) / identity)
    with _STORES_LOCK:
        store = _STORES.get(key)
        if store is None:
            try:
                Path(parent).mkdir(mode=0o700, parents=True, exist_ok=True)
                store = ServedToolInputStore(Path(key))
            except (OSError, ValueError) as error:
                logger.warning("served tool inputs stay in process memory: %s", error)
                store = ServedToolInputStore(None)
            _STORES[key] = store
        return store


def served_turn(anthropic_response: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """The served turn keyed by its ``tool_use`` ids; turns without tool calls are not recorded."""
    content = list(anthropic_response.get("content") or [])
    ids = [block.get("id") for block in _tool_uses(content)]
    if not ids or not all(isinstance(value, str) for value in ids):
        return None
    return {"id": turn_key(ids), "content": copy.deepcopy(content)}


def record_served_turn(
    anthropic_response: Mapping[str, Any], store: ServedToolInputStore, rollout_id: Optional[str]
) -> None:
    """Remember what this response served; a failure only costs the next request its restoration."""
    turn = served_turn(anthropic_response)
    if turn is None:
        return
    try:
        store.record(rollout_id, [turn])
    except (OSError, ValueError) as error:
        logger.warning("could not record a served turn for rollout %r: %s", rollout_id, error)


def restore_served_turns(
    body: Mapping[str, Any], store: ServedToolInputStore, rollout_id: Optional[str]
) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    """Return ``body`` with each echoed assistant turn that matches a served turn replaced by it.

    The caller's body is not changed: when any turn differed, a new body is returned that
    shares every unchanged message, with one entry per changed turn (also appended to
    the rollout's restore log).
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body, []
    turns = []
    for index, message in enumerate(messages):
        if (
            isinstance(message, Mapping)
            and message.get("role") == "assistant"
            and isinstance(message.get("content"), list)
        ):
            ids = [block.get("id") for block in _tool_uses(message["content"])]
            if ids and all(isinstance(value, str) for value in ids):
                turns.append((index, ids))
    if not turns:
        return body, []
    try:
        served = store.lookup(rollout_id, {turn_key(ids) for _, ids in turns})
    except (OSError, ValueError) as error:
        logger.warning("could not read served turns for rollout %r: %s", rollout_id, error)
        return body, []
    defaults = _schema_defaults(body.get("tools"))
    restored_body = None
    entries = []
    for index, ids in turns:
        record = served.get(turn_key(ids))
        if record is None:
            continue
        echoed = messages[index]["content"]
        mapping = _align(record["content"], echoed)
        if mapping is None:
            continue
        differences = _differences(record["content"], echoed, mapping, defaults)
        if not differences:
            continue
        if restored_body is None:
            restored_body = {**body, "messages": list(messages)}
        restored_body["messages"][index] = {**messages[index], "content": copy.deepcopy(record["content"])}
        counts = Counter(difference["kind"] for difference in differences)
        digest = hashlib.sha256(json.dumps(differences, sort_keys=True, default=str).encode()).hexdigest()[:16]
        entries.append(
            {
                "key": f"{turn_key(ids)}#{digest}",
                "at": time.time(),
                "tool_use_ids": ids,
                "counts": dict(counts),
                "differences": differences,
            }
        )
    if restored_body is None:
        return body, []
    try:
        store.log_restores(rollout_id, entries)
    except (OSError, ValueError) as error:
        logger.warning("could not log restored turns for rollout %r: %s", rollout_id, error)
    return restored_body, entries
