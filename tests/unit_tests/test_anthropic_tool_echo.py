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
"""The default /v1/messages route sends the backend the assistant turns it served.

Real model servers (FastAPI app, Anthropic converter, rollout-prefix middleware and
the served-turn store) serve a turn and then receive the next request with that
turn echoed the way Claude Code 2.1.284 echoes it: ``cd /testbed && `` dropped from
Bash commands, whitespace-only text blocks dropped, Edit's ``replace_all`` default
filled in, trailing whitespace stripped from ``new_string`` lines. The backend must
receive the served turn again and the restore log must name each change. Turns that
do not match a served turn, and user and tool_result content, must reach the
backend unchanged so append-only checks still see them.
"""

import copy
import json
import logging
import os
from time import time
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import Body
from fastapi.testclient import TestClient

import nemo_gym.anthropic_tool_echo as tool_echo
from nemo_gym.base_responses_api_model import BaseResponsesAPIModelConfig, SimpleResponsesAPIModel
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient


SERVED = {"command": "cd /testbed && git status --short", "description": "Show the repository state"}
STRIPPED = {"command": "git status --short", "description": "Show the repository state"}
EDIT = {"file_path": "/testbed/app.py", "old_string": "alpha", "new_string": "beta  \n\tgamma \nend"}
EDIT_ECHOED = {
    "file_path": "/testbed/app.py",
    "old_string": "alpha",
    "new_string": "beta\n\tgamma\nend",
    "replace_all": False,
}
TOOLS = [
    {
        "name": "Bash",
        "description": "Run a shell command",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}, "description": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "Read",
        "description": "Read a file",
        "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
    },
    {
        "name": "Edit",
        "description": "Replace text in a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean", "default": False},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
]
TASK = {"role": "user", "content": "Fix the bug in /testbed."}


def _response(output: list) -> NeMoGymResponse:
    return NeMoGymResponse(
        id=f"resp_{uuid4().hex}",
        created_at=int(time()),
        model="policy",
        object="response",
        output=output,
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    )


def _turn(*parts) -> NeMoGymResponse:
    """A served turn: ("reasoning", text), ("text", text) and ("tool", call_id, name, arguments) parts, in order."""
    output = []
    for part in parts:
        if part[0] == "reasoning":
            output.append(
                {
                    "type": "reasoning",
                    "id": f"rs_{uuid4().hex}",
                    "summary": [{"type": "summary_text", "text": part[1]}],
                }
            )
        elif part[0] == "text":
            output.append(
                {
                    "type": "message",
                    "id": f"msg_{uuid4().hex}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": part[1], "annotations": []}],
                }
            )
        else:
            output.append(
                {
                    "type": "function_call",
                    "id": part[1],
                    "call_id": part[1],
                    "name": part[2],
                    "arguments": json.dumps(part[3]),
                    "status": "completed",
                }
            )
    return _response(output)


def _text(text: str) -> NeMoGymResponse:
    return _turn(("text", text))


class _ScriptedModel(SimpleResponsesAPIModel):
    """Serves its scripted responses in order and keeps every Responses request it received."""

    config: BaseResponsesAPIModelConfig
    script: list = []
    received: list = []
    model_config = {"arbitrary_types_allowed": True}

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        self.received.append(body)
        return self.script.pop(0) if self.script else _text("done")

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        raise NotImplementedError


@pytest.fixture(autouse=True)
def private_store(tmp_path, monkeypatch):
    monkeypatch.setenv(tool_echo.STORE_DIR_ENV, str(tmp_path / "served"))
    monkeypatch.setattr(tool_echo, "_STORES", {})
    return tmp_path / "served"


def _server(
    script: list, name: str = "policy_model", port: int = 8099, case_repair: bool = False
) -> tuple[TestClient, _ScriptedModel]:
    config = BaseResponsesAPIModelConfig(
        host="127.0.0.1", port=port, entrypoint="", name=name, anthropic_tool_name_case_repair=case_repair
    )
    server = _ScriptedModel(
        config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}), script=list(script)
    )
    return TestClient(server.setup_webserver()), server


def _post(client: TestClient, messages: list, prefix: str = "/ng-rollout/roll-1", stream: bool = True):
    response = client.post(
        f"{prefix}/v1/messages",
        json={"model": "policy", "max_tokens": 64, "stream": stream, "tools": TOOLS, "messages": messages},
    )
    assert response.status_code == 200
    return response


def _thinking(text: str) -> dict:
    return {"type": "thinking", "thinking": text, "signature": ""}


def _blank(text: str = "\n\n") -> dict:
    return {"type": "text", "text": text}


def _tool(call_id: str, name: str, arguments: dict) -> dict:
    return {"type": "tool_use", "id": call_id, "name": name, "input": arguments}


def _results(blocks: list, content: str = "ok") -> dict:
    ids = [block["id"] for block in blocks if block["type"] == "tool_use"]
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": i, "content": content} for i in ids]}


def _history(*turns: list) -> list:
    """The task, then each assistant turn followed by its tool results."""
    messages = [TASK]
    for blocks in turns:
        messages += [{"role": "assistant", "content": blocks}, _results(blocks)]
    return messages


def _arguments(server: _ScriptedModel, call_id: str) -> dict:
    calls = [item for item in server.received[-1].input if getattr(item, "type", None) == "function_call"]
    (call,) = [item for item in calls if item.call_id == call_id]
    return json.loads(call.arguments)


def _backend_input(server: _ScriptedModel) -> list:
    return [item.model_dump() if hasattr(item, "model_dump") else item for item in server.received[-1].input]


def _texts(server: _ScriptedModel) -> list:
    """Assistant text the backend received, in order (the converter emits it as string content)."""
    texts = []
    for item in _backend_input(server):
        if item.get("type") == "message" and item.get("role") == "assistant":
            content = item.get("content")
            texts.extend([content] if isinstance(content, str) else [part.get("text") for part in content or []])
    return texts


def _comparable(server: _ScriptedModel) -> dict:
    """The backend request, with the id the converter mints for each reasoning item masked."""
    dumped = server.received[-1].model_dump()
    dumped["input"] = [
        {**item, "id": "<minted>"} if isinstance(item, dict) and item.get("type") == "reasoning" else item
        for item in dumped["input"]
    ]
    return dumped


def _restore_log(private_store, rollout: str = "roll-1") -> list:
    paths = list(private_store.glob(f"*/{rollout}.restores.jsonl"))
    return [json.loads(line) for path in paths for line in path.read_text().splitlines()]


def _kinds(entries: list) -> dict:
    total: dict = {}
    for entry in entries:
        for kind, count in entry["counts"].items():
            total[kind] = total.get(kind, 0) + count
    return total


def _restored_and_exact(served: list, normalized: list, unnormalized: list) -> tuple:
    """Serve ``served`` turns to two servers; echo the normalized history to one and the served one to the other."""
    restored_client, restored = _server(served, name="restored")
    exact_client, exact = _server(copy.deepcopy(served), name="exact")
    for index in range(len(served)):
        _post(restored_client, _history(*normalized[:index]))
        _post(exact_client, _history(*unnormalized[:index]))
    _post(restored_client, _history(*normalized))
    _post(exact_client, _history(*unnormalized))
    return restored, exact


class TestServedTurnRestoration:
    def test_cd_strip_is_restored_and_logged(self, private_store) -> None:
        client, server = _server([_turn(("tool", "call_1", "Bash", SERVED))])
        streamed = _post(client, [TASK])
        assert "cd /testbed && git status --short" in streamed.text
        _post(client, _history([_tool("call_1", "Bash", STRIPPED)]))
        assert _arguments(server, "call_1") == SERVED
        (entry,) = _restore_log(private_store)
        assert entry["counts"] == {"cd_strip": 1}
        (difference,) = entry["differences"]
        assert difference["field"] == "input.command"
        assert (difference["served"], difference["echoed"]) == (SERVED["command"], STRIPPED["command"])

    def test_dropped_whitespace_block_is_restored_and_logged(self, private_store) -> None:
        call = _tool("call_1", "Bash", {"command": "ls"})
        restored, exact = _restored_and_exact(
            [_turn(("reasoning", "Look first."), ("text", "\n\n"), ("tool", "call_1", "Bash", {"command": "ls"}))],
            normalized=[[_thinking("Look first."), call]],
            unnormalized=[[_thinking("Look first."), _blank(), call]],
        )
        assert "\n\n" in _texts(restored)
        assert _comparable(restored) == _comparable(exact)
        (entry,) = _restore_log(private_store)
        assert entry["counts"] == {"whitespace": 1}
        assert entry["differences"][0]["change"] == "left_out"

    def test_edit_replace_all_default_fill_is_restored_and_logged(self, private_store) -> None:
        served = {"file_path": "/testbed/app.py", "old_string": "alpha", "new_string": "beta"}
        client, server = _server([_turn(("tool", "call_e", "Edit", served))])
        _post(client, [TASK])
        _post(client, _history([_tool("call_e", "Edit", {**served, "replace_all": False})]))
        assert _arguments(server, "call_e") == served
        (entry,) = _restore_log(private_store)
        assert entry["counts"] == {"schema_default_fill": 1}
        assert entry["differences"][0]["field"] == "input.replace_all"

    def test_trailing_spaces_stripped_from_new_string_are_restored_and_logged(self, private_store) -> None:
        stripped = {**EDIT, "new_string": "beta\n\tgamma\nend"}
        client, server = _server([_turn(("tool", "call_e", "Edit", EDIT))])
        _post(client, [TASK])
        _post(client, _history([_tool("call_e", "Edit", stripped)]))
        assert _arguments(server, "call_e") == EDIT
        (entry,) = _restore_log(private_store)
        assert entry["counts"] == {"whitespace": 1}
        assert entry["differences"][0]["field"] == "input.new_string"

    def test_an_unknown_future_normalization_is_restored_and_logged_as_other(self, private_store) -> None:
        reasoning = "Check the tree.  "
        client, server = _server(
            [_turn(("reasoning", reasoning), ("text", "Looking."), ("tool", "call_1", "Bash", SERVED))]
        )
        _post(client, [TASK])
        normalized = {"command": "git status --short", "description": "show the repository state", "timeout": 120000}
        _post(
            client,
            _history(
                [_thinking("check the tree"), {"type": "text", "text": "Looking"}, _tool("call_1", "Bash", normalized)]
            ),
        )
        assert _arguments(server, "call_1") == SERVED
        assert _texts(server) == ["Looking."]
        (entry,) = _restore_log(private_store)
        assert entry["counts"] == {"cd_strip": 1, "other": 4}
        fields = {(difference["type"], difference["field"]) for difference in entry["differences"]}
        assert fields == {
            ("thinking", "thinking"),
            ("text", "text"),
            ("tool_use", "input.command"),
            ("tool_use", "input.description"),
            ("tool_use", "input.timeout"),
        }

    def test_every_claude_code_normalization_at_once_matches_an_unnormalized_echo(self, private_store) -> None:
        bash, edit = _tool("call_1", "Bash", SERVED), _tool("call_e", "Edit", EDIT)
        restored, exact = _restored_and_exact(
            [
                _turn(("reasoning", "Inspect."), ("text", "\n\n"), ("tool", "call_1", "Bash", SERVED)),
                _turn(("text", "\n"), ("tool", "call_e", "Edit", EDIT)),
            ],
            normalized=[
                [_thinking("Inspect."), _tool("call_1", "Bash", STRIPPED)],
                [_tool("call_e", "Edit", EDIT_ECHOED)],
            ],
            unnormalized=[[_thinking("Inspect."), _blank(), bash], [_blank("\n"), edit]],
        )
        assert _comparable(restored) == _comparable(exact)
        assert _arguments(restored, "call_1") == SERVED and _arguments(restored, "call_e") == EDIT
        entries = _restore_log(private_store)
        assert [entry["tool_use_ids"] for entry in entries] == [["call_1"], ["call_e"]]
        assert _kinds(entries) == {"cd_strip": 1, "whitespace": 3, "schema_default_fill": 1}

    def test_a_turn_echoed_on_every_later_request_is_logged_once(self, private_store) -> None:
        client, server = _server([_turn(("tool", "call_1", "Bash", SERVED)), _text("a"), _text("b")])
        _post(client, [TASK])
        history = _history([_tool("call_1", "Bash", STRIPPED)])
        _post(client, history)
        _post(client, history + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "go on"}])
        assert _arguments(server, "call_1") == SERVED
        assert len(_restore_log(private_store)) == 1

    def test_an_unchanged_echo_is_not_logged(self, private_store) -> None:
        client, server = _server([_turn(("text", "Run it."), ("tool", "call_1", "Bash", SERVED))])
        _post(client, [TASK])
        _post(
            client,
            _history(
                [
                    {"type": "text", "text": "Run it.", "cache_control": {"type": "ephemeral"}},
                    _tool("call_1", "Bash", SERVED),
                ]
            ),
        )
        assert _arguments(server, "call_1") == SERVED
        assert _restore_log(private_store) == []

    def test_non_streaming_turns_are_recorded_too(self) -> None:
        client, server = _server([_turn(("tool", "call_1", "Bash", SERVED))])
        _post(client, [TASK], stream=False)
        _post(client, _history([_tool("call_1", "Bash", STRIPPED)]))
        assert _arguments(server, "call_1") == SERVED

    def test_uvicorn_workers_of_one_server_share_the_records(self, private_store) -> None:
        # Stock runs scale vllm_model over several uvicorn worker processes; the turn
        # after a served tool call may land on another worker.
        first_client, _ = _server([_turn(("text", "\n\n"), ("tool", "call_1", "Bash", SERVED))])
        _post(first_client, [TASK])
        tool_echo._STORES.clear()  # a fresh process: nothing cached in memory
        second_client, second = _server([])
        _post(second_client, _history([_tool("call_1", "Bash", STRIPPED)]))
        assert _arguments(second, "call_1") == SERVED
        assert "\n\n" in _texts(second)
        (record,) = [json.loads(line) for line in next(private_store.glob("*/roll-1.jsonl")).read_text().splitlines()]
        assert record["id"] == tool_echo.turn_key(["call_1"])
        assert _kinds(_restore_log(private_store)) == {"cd_strip": 1, "whitespace": 1}

    def test_a_request_without_a_rollout_uses_process_memory_and_the_logger(self, private_store, caplog) -> None:
        client, server = _server([_turn(("tool", "call_1", "Bash", SERVED))])
        _post(client, [TASK], prefix="")
        with caplog.at_level(logging.INFO, logger=tool_echo.__name__):
            _post(client, _history([_tool("call_1", "Bash", STRIPPED)]), prefix="")
            _post(client, _history([_tool("call_1", "Bash", STRIPPED)]), prefix="")
        assert _arguments(server, "call_1") == SERVED
        assert len([record for record in caplog.records if "cd_strip" in record.getMessage()]) == 1
        assert not list(private_store.glob("*/*.jsonl"))

    def test_the_caller_body_is_not_changed(self) -> None:
        store = tool_echo.ServedToolInputStore(None)
        store.record(None, [tool_echo.served_turn({"content": [_tool("call_1", "Bash", SERVED)]})])
        body = {"messages": _history([_tool("call_1", "Bash", STRIPPED)])}
        original = copy.deepcopy(body)
        restored, entries = tool_echo.restore_served_turns(body, store, None)
        assert body == original
        assert restored["messages"][1]["content"] == [_tool("call_1", "Bash", SERVED)]
        assert restored["messages"][2] == body["messages"][2]
        assert [entry["counts"] for entry in entries] == [{"cd_strip": 1}]


class TestNothingElseIsMasked:
    """Echoes that do not match a served turn, and user content, reach the backend as sent."""

    def _unchanged(self, served: list, messages: list) -> list:
        client, server = _server(served + [_text("x"), _text("y")])
        for index in range(len(served)):
            _post(client, [TASK] if index == 0 else messages[: 2 * index + 1])
        _post(client, messages)
        with_records = _comparable(server)
        _post(client, messages, prefix="/ng-rollout/no-records")  # same echo, nothing served under this rollout
        assert with_records == _comparable(server)
        return _backend_input(server)

    @pytest.mark.parametrize(
        "echoed",
        [
            [_tool("call_other", "Bash", STRIPPED)],
            [_tool("call_1", "Read", STRIPPED)],
            [_tool("call_2", "Bash", STRIPPED), _tool("call_1", "Bash", STRIPPED)],
            [_tool("call_1", "Bash", STRIPPED)],
        ],
        ids=["other-id", "other-name", "reordered-ids", "a-tool-call-left-out"],
    )
    def test_mismatched_tool_call_ids_pass_through(self, echoed: list, private_store) -> None:
        served = [_turn(("tool", "call_1", "Bash", SERVED), ("tool", "call_2", "Bash", SERVED))]
        if echoed[0]["name"] == "Read" or echoed[0]["id"] == "call_other":
            served = [_turn(("tool", "call_1", "Bash", SERVED))]
        self._unchanged(served, _history(echoed))
        assert _restore_log(private_store) == []

    @pytest.mark.parametrize(
        "echoed",
        [
            [_tool("call_1", "Bash", STRIPPED), {"type": "text", "text": "Let me look."}],
            [
                {"type": "text", "text": "Done."},
                {"type": "text", "text": "Let me look."},
                _tool("call_1", "Bash", STRIPPED),
            ],
            [_thinking("Planning."), {"type": "text", "text": "Let me look."}, _tool("call_1", "Bash", STRIPPED)],
        ],
        ids=["block-moved-after-the-call", "block-added", "thinking-added"],
    )
    def test_reordered_or_added_blocks_pass_through(self, echoed: list, private_store) -> None:
        self._unchanged([_turn(("text", "Let me look."), ("tool", "call_1", "Bash", SERVED))], _history(echoed))
        assert _restore_log(private_store) == []

    def test_a_dropped_turn_stays_dropped(self, private_store) -> None:
        first, second = _tool("call_1", "Bash", SERVED), _tool("call_2", "Bash", SERVED)
        client, server = _server(
            [_turn(("tool", "call_1", "Bash", SERVED)), _turn(("tool", "call_2", "Bash", SERVED))]
        )
        _post(client, [TASK])
        _post(client, _history([first]))
        _post(client, _history([second]))  # the first turn and its result left out
        calls = [item["call_id"] for item in _backend_input(server) if item.get("type") == "function_call"]
        assert calls == ["call_2"]
        assert _restore_log(private_store) == []

    def test_a_compacted_history_passes_through(self, private_store) -> None:
        summary = [TASK, {"role": "user", "content": "Summary: ran git status (cd /testbed omitted)."}]
        client, server = _server([_turn(("tool", "call_1", "Bash", SERVED))])
        _post(client, [TASK])
        _post(client, summary)
        assert [item.get("content") for item in _backend_input(server)][-1] == summary[-1]["content"]
        assert _restore_log(private_store) == []

    def test_an_edited_tool_result_reaches_the_backend_while_the_turn_is_restored(self, private_store) -> None:
        client, server = _server([_turn(("tool", "call_1", "Bash", SERVED))])
        _post(client, [TASK])
        blocks = [_tool("call_1", "Bash", STRIPPED)]
        _post(
            client, [TASK, {"role": "assistant", "content": blocks}, _results(blocks, content="edited by the client")]
        )
        assert _arguments(server, "call_1") == SERVED
        (output,) = [item for item in _backend_input(server) if item.get("type") == "function_call_output"]
        assert output["output"] == "edited by the client"
        (entry,) = _restore_log(private_store)
        assert {difference["type"] for difference in entry["differences"]} == {"tool_use"}

    def test_user_content_that_looks_like_a_served_turn_is_not_touched(self, private_store) -> None:
        client, server = _server([_turn(("tool", "call_1", "Bash", SERVED))])
        _post(client, [TASK])
        quoted = {"role": "user", "content": [{"type": "text", "text": json.dumps(_tool("call_1", "Bash", STRIPPED))}]}
        _post(client, [TASK, {"role": "assistant", "content": "ok"}, quoted])
        assert _backend_input(server)[-1]["content"] == quoted["content"][0]["text"]
        assert _restore_log(private_store) == []

    def test_records_belong_to_their_rollout(self, private_store) -> None:
        client, server = _server([_turn(("tool", "call_1", "Bash", SERVED))])
        _post(client, [TASK], prefix="/ng-rollout/roll-1")
        _post(client, _history([_tool("call_1", "Bash", STRIPPED)]), prefix="/ng-rollout/roll-2")
        assert _arguments(server, "call_1") == STRIPPED
        assert _restore_log(private_store, "roll-2") == []

    def test_unsafe_rollout_ids_never_reach_the_filesystem(self, private_store) -> None:
        store = tool_echo.store_for("policy_model", "127.0.0.1", 1)
        body = {"messages": _history([_tool("call_1", "Bash", STRIPPED)])}
        assert tool_echo.restore_served_turns(body, store, "../escape") == (body, [])
        with pytest.raises(ValueError):
            store.record("../escape", [tool_echo.served_turn({"content": [_tool("call_1", "Bash", SERVED)]})])
        assert not list(private_store.parent.glob("escape*"))


class TestToolNameTruncation:
    """Claude Code echoes a tool name longer than 200 characters as its first 200."""

    LONG = "python3 -c \"" + "x = 1; " * 60 + "\""

    def test_a_truncated_model_made_name_is_restored_and_logged(self, private_store) -> None:
        served = _tool("call_1", self.LONG, {"command": "ls"})
        echoed = _tool("call_1", self.LONG[:200], {"command": "ls"})
        client, server = _server([_turn(("tool", "call_1", self.LONG, {"command": "ls"}))])
        _post(client, [TASK])
        _post(client, _history([echoed]))
        (call,) = [item for item in _backend_input(server) if item.get("type") == "function_call"]
        assert call["name"] == self.LONG
        assert _kinds(_restore_log(private_store)) == {"name_truncation": 1}

    @pytest.mark.parametrize("echoed_name", ["Bash", LONG[:199], LONG[1:201]])
    def test_any_other_name_change_passes_through(self, echoed_name, private_store) -> None:
        client, server = _server([_turn(("tool", "call_1", self.LONG, {"command": "ls"}))], name=f"n{len(echoed_name)}")
        _post(client, [TASK])
        _post(client, _history([_tool("call_1", echoed_name, {"command": "ls"})]))
        (call,) = [item for item in _backend_input(server) if item.get("type") == "function_call"]
        assert call["name"] == echoed_name
        assert _restore_log(private_store) == []

    def test_a_short_name_is_never_treated_as_truncated(self) -> None:
        assert tool_echo._same_or_truncated_name("Bash", "Bash")
        assert not tool_echo._same_or_truncated_name("Bash", "Bas")
        assert not tool_echo._same_or_truncated_name("b" * 200, "b" * 199)


class TestToolNameCase:
    """A served tool name that differs from one offered tool only in case reaches the client as that tool."""

    @staticmethod
    def _client_names(response) -> list:
        return [block["name"] for block in response.json()["content"] if block["type"] == "tool_use"]

    def test_the_client_gets_the_offered_name_and_the_backend_gets_the_served_one(self, private_store) -> None:
        client, server = _server([_turn(("tool", "call_1", "bash", {"command": "ls"}))], case_repair=True)
        assert self._client_names(_post(client, [TASK], stream=False)) == ["Bash"]
        _post(client, _history([_tool("call_1", "Bash", {"command": "ls"})]))
        (call,) = [item for item in _backend_input(server) if item.get("type") == "function_call"]
        assert call["name"] == "bash"
        (entry,) = _restore_log(private_store)
        assert entry["counts"] == {"name_case": 1}
        assert entry["differences"][0]["served"] == "bash" and entry["differences"][0]["echoed"] == "Bash"

    def test_the_streamed_turn_carries_the_offered_name(self) -> None:
        client, _ = _server([_turn(("tool", "call_1", "bash", {"command": "ls"}))], case_repair=True)
        streamed = _post(client, [TASK]).text
        assert '"Bash"' in streamed and '"bash"' not in streamed

    @pytest.mark.parametrize("name", ["Bash", "Done", "bash_tool"])
    def test_an_offered_or_unmatched_name_is_sent_as_served(self, name) -> None:
        client, _ = _server([_turn(("tool", "call_1", name, {"command": "ls"}))], case_repair=True)
        assert self._client_names(_post(client, [TASK], stream=False)) == [name]

    def test_off_by_default(self) -> None:
        client, _ = _server([_turn(("tool", "call_1", "bash", {"command": "ls"}))])
        assert self._client_names(_post(client, [TASK], stream=False)) == ["bash"]

    def test_a_name_matching_several_offered_tools_is_not_renamed(self) -> None:
        served = {"content": [_tool("call_1", "read", {})], "stop_reason": "tool_use"}
        tools = [{"name": "Read"}, {"name": "READ"}, {"name": "Bash"}]
        assert tool_echo.repair_tool_name_case(served, tools) == (served, [])

    def test_only_the_mismatched_block_is_renamed_and_the_input_is_unchanged(self) -> None:
        served = {"content": [_tool("call_1", "Read", {}), _tool("call_2", "bash", {"command": "ls"})]}
        original = copy.deepcopy(served)
        repaired, renamed = tool_echo.repair_tool_name_case(served, TOOLS)
        assert [block["name"] for block in repaired["content"]] == ["Read", "Bash"]
        assert renamed == [{"id": "call_2", "served": "bash", "sent": "Bash"}]
        assert served == original

    def test_an_echo_with_any_other_name_change_still_passes_through(self, private_store) -> None:
        client, server = _server([_turn(("tool", "call_1", "bash", {"command": "ls"}))], case_repair=True)
        _post(client, [TASK])
        _post(client, _history([_tool("call_1", "Shell", {"command": "ls"})]))
        (call,) = [item for item in _backend_input(server) if item.get("type") == "function_call"]
        assert call["name"] == "Shell"
        assert _restore_log(private_store) == []


class TestEmptyTurnNudge:
    """Claude Code drops a served turn that presents nothing and appends a nudge to the last user message."""

    @staticmethod
    def _nudged(results: dict) -> dict:
        return {**results, "content": [*results["content"], {"type": "text", "text": tool_echo.EMPTY_TURN_NUDGE}]}

    @staticmethod
    def _shape(server: _ScriptedModel) -> list:
        shape = []
        for item in _backend_input(server):
            kind, role = item.get("type"), item.get("role")
            if kind == "message":
                content = item.get("content")
                text = content if isinstance(content, str) else "".join(part.get("text", "") for part in content or [])
                shape.append((role, text))
            else:
                shape.append((kind, item.get("call_id")))
        return shape

    @pytest.mark.parametrize("empty", [_text(""), _turn(("reasoning", "thinking only")), _turn(("text", "\n\n"))])
    def test_the_dropped_empty_turn_is_restored_before_its_nudge(self, empty, private_store) -> None:
        first = _tool("call_1", "Bash", SERVED)
        client, server = _server([_turn(("tool", "call_1", "Bash", SERVED)), copy.deepcopy(empty), _text("ok")])
        _post(client, [TASK])
        served = _post(client, _history([first]), stream=False).json()["content"]  # the empty turn
        # The CLI never echoes that turn; it appends its nudge to the last user message instead.
        nudged = _history([first])
        nudged[-1] = self._nudged(nudged[-1])
        _post(client, nudged)
        # The backend gets what an exact echo would give: the served turn, then the nudge as a new user turn.
        exact_client, exact = _server([_turn(("tool", "call_1", "Bash", SERVED)), copy.deepcopy(empty)], name="exact")
        _post(exact_client, [TASK])
        _post(exact_client, _history([first]))
        _post(exact_client, _history([first]) + [
            {"role": "assistant", "content": served},
            {"role": "user", "content": [{"type": "text", "text": tool_echo.EMPTY_TURN_NUDGE}]},
        ])
        assert _comparable(server)["input"] == _comparable(exact)["input"]
        assert self._shape(server)[-1] == ("user", tool_echo.EMPTY_TURN_NUDGE)
        (entry,) = _restore_log(private_store)
        assert entry["counts"] == {"empty_turn_nudge": 1} and entry["key"] == "empty:3#empty_turn_nudge"

    def test_later_requests_keep_the_restored_turn_and_restore_tool_turns_after_it(self, private_store) -> None:
        first, second = _tool("call_1", "Bash", SERVED), _tool("call_2", "Bash", SERVED)
        client, server = _server(
            [_turn(("tool", "call_1", "Bash", SERVED)), _text(""), _turn(("tool", "call_2", "Bash", SERVED))]
        )
        _post(client, [TASK])
        _post(client, _history([first]))
        nudged = _history([first])
        nudged[-1] = self._nudged(nudged[-1])
        _post(client, nudged)
        # The next request: the CLI's history keeps the merged message and echoes call_2 with the cd strip.
        later = nudged + [{"role": "assistant", "content": [_tool("call_2", "Bash", STRIPPED)]}, _results([second])]
        _post(client, later)
        shape = self._shape(server)
        assert shape[-5:] == [
            ("function_call_output", "call_1"),
            ("assistant", ""),
            ("user", tool_echo.EMPTY_TURN_NUDGE),
            ("function_call", "call_2"),
            ("function_call_output", "call_2"),
        ]
        assert _arguments(server, "call_2") == SERVED
        kinds = _kinds(_restore_log(private_store))
        assert kinds["empty_turn_nudge"] == 1 and kinds["cd_strip"] == 1

    def test_a_nudge_without_a_served_empty_turn_passes_through(self, private_store) -> None:
        first = _tool("call_1", "Bash", SERVED)
        client, server = _server([_turn(("tool", "call_1", "Bash", SERVED)), _text("visible answer")])
        _post(client, [TASK])
        _post(client, _history([first]))  # served visible text: nothing recorded
        nudged = _history([first])
        nudged[-1] = self._nudged(nudged[-1])
        _post(client, nudged)
        assert self._shape(server)[-1] == ("user", tool_echo.EMPTY_TURN_NUDGE)
        assert ("assistant", "visible answer") not in self._shape(server)
        assert _restore_log(private_store) == []

    def test_only_the_exact_nudge_after_other_content_counts(self, private_store) -> None:
        first = _tool("call_1", "Bash", SERVED)
        for last in ([{"type": "text", "text": tool_echo.EMPTY_TURN_NUDGE + " "}], None):
            client, server = _server([_turn(("tool", "call_1", "Bash", SERVED)), _text("")], name=f"s{last is None}")
            _post(client, [TASK])
            _post(client, _history([first]))
            nudged = _history([first])
            if last is None:  # the nudge alone, with no tool result before it
                nudged[-1] = {"role": "user", "content": [{"type": "text", "text": tool_echo.EMPTY_TURN_NUDGE}]}
            else:
                nudged[-1] = {**nudged[-1], "content": [*nudged[-1]["content"], *last]}
            _post(client, nudged)
            assert ("assistant", "") not in self._shape(server)
        assert _restore_log(private_store) == []

    def test_a_length_capped_turn_dropped_for_the_output_limit_nudge_is_restored(self, private_store) -> None:
        first = _tool("call_1", "Bash", SERVED)
        client, server = _server([_turn(("tool", "call_1", "Bash", SERVED))])
        capped = {"content": [{"type": "text", "text": "partial answer cut at the"}], "stop_reason": "max_tokens"}
        store = tool_echo.store_for("policy_model", "127.0.0.1", 8099)
        _post(client, [TASK])
        tool_echo.record_served_turn(capped, store, "roll-1", request_length=3)
        nudged = _history([first])
        nudged[-1] = {**nudged[-1], "content": [*nudged[-1]["content"], {"type": "text", "text": tool_echo.OUTPUT_LIMIT_NUDGE}]}
        _post(client, nudged)
        assert self._shape(server)[-2:] == [("assistant", "partial answer cut at the"), ("user", tool_echo.OUTPUT_LIMIT_NUDGE)]
        (entry,) = _restore_log(private_store)
        assert entry["counts"] == {"output_limit_nudge": 1}
        # A turn that was neither empty nor length-capped is never keyed by request length.
        assert tool_echo.served_turns({"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn"}, 3) == []

    def test_served_turn_keys(self) -> None:
        empty = {"content": [{"type": "text", "text": ""}]}
        assert tool_echo.served_turn(empty) is None
        assert tool_echo.served_turn(empty, 7)["id"] == "empty:7"
        assert tool_echo.served_turn({"content": [{"type": "thinking", "thinking": "x"}]}, 7)["id"] == "empty:7"
        assert tool_echo.served_turn({"content": [{"type": "text", "text": "hi"}]}, 7) is None
        called = {"content": [{"type": "tool_use", "id": "call_1", "name": "Bash", "input": {}}]}
        assert tool_echo.served_turn(called, 7)["id"] == tool_echo.turn_key(["call_1"])


class TestDifferenceKinds:
    """Classification of what the client changed; every one of these turns is restored."""

    def _kinds_of(self, served: list, echoed: list, tools: list = TOOLS) -> list:
        store = tool_echo.ServedToolInputStore(None)
        store.record(None, [tool_echo.served_turn({"content": served})])
        body = {"tools": tools, "messages": [TASK, {"role": "assistant", "content": echoed}]}
        restored, entries = tool_echo.restore_served_turns(body, store, None)
        assert restored["messages"][1]["content"] == served
        return [(difference["field"], difference["kind"]) for entry in entries for difference in entry["differences"]]

    @pytest.mark.parametrize(
        "served,echoed,kind",
        [
            ("cd /testbed && ls", "ls", "cd_strip"),
            ("cd /testbed && cd src && ls", "cd src && ls", "cd_strip"),
            ("cd C:\\repo && dir", "dir", "cd_strip"),
            ("cd '/testbed' && ls", "ls", "other"),
            ("cd /my repo && ls", "ls", "other"),
            ("cd src && ls", "ls", "other"),
            ("cd /testbed && ls", "ls -la", "other"),
            ("ls  ", "ls", "whitespace"),
        ],
    )
    def test_bash_commands(self, served: str, echoed: str, kind: str) -> None:
        assert self._kinds_of(
            [_tool("a", "Bash", {"command": served})], [_tool("a", "Bash", {"command": echoed})]
        ) == [("input.command", kind)]

    def test_a_cd_strip_is_only_named_for_bash(self) -> None:
        served, echoed = {"command": "cd /testbed && ls"}, {"command": "ls"}
        assert self._kinds_of([_tool("a", "Read", served)], [_tool("a", "Read", echoed)]) == [
            ("input.command", "other")
        ]

    @pytest.mark.parametrize(
        "added,kind",
        [({"replace_all": False}, "schema_default_fill"), ({"replace_all": True}, "other"), ({"extra": 1}, "other")],
    )
    def test_added_inputs(self, added: dict, kind: str) -> None:
        served = {"file_path": "/f", "old_string": "a", "new_string": "b"}
        (field,) = added
        assert self._kinds_of([_tool("a", "Edit", served)], [_tool("a", "Edit", {**served, **added})]) == [
            (f"input.{field}", kind)
        ]

    def test_a_removed_input_is_other(self) -> None:
        assert self._kinds_of(
            [_tool("a", "Edit", {"file_path": "/f", "replace_all": False})], [_tool("a", "Edit", {"file_path": "/f"})]
        ) == [("input.replace_all", "other")]

    def test_left_out_blocks(self) -> None:
        tool = _tool("a", "Bash", {"command": "ls"})
        assert self._kinds_of([_blank(" \n\t"), {"type": "text", "text": "Plan."}, tool], [tool]) == [
            (None, "whitespace"),
            (None, "other"),
        ]

    def test_the_whitespace_block_is_the_one_left_out(self) -> None:
        tool = _tool("a", "Bash", {"command": "ls"})
        plan = {"type": "text", "text": "Plan."}
        assert self._kinds_of([_blank(), plan, tool], [plan, tool]) == [(None, "whitespace")]

    @pytest.mark.parametrize(
        "served,echoed,kind",
        [("Plan A.\n", "Plan A.", "whitespace"), ("Plan A.", "Plan B.", "other")],
    )
    def test_thinking(self, served: str, echoed: str, kind: str) -> None:
        tool = _tool("a", "Bash", {"command": "ls"})
        assert self._kinds_of([_thinking(served), tool], [_thinking(echoed), tool]) == [("thinking", kind)]

    def test_a_signature_or_cache_control_alone_is_no_difference(self) -> None:
        store = tool_echo.ServedToolInputStore(None)
        served = [{"type": "thinking", "thinking": "p", "signature": ""}, _tool("a", "Bash", {"command": "ls"})]
        store.record(None, [tool_echo.served_turn({"content": served})])
        echoed = [
            {"type": "thinking", "thinking": "p", "signature": "x"},
            {**served[1], "cache_control": {"type": "ephemeral"}},
        ]
        body = {"messages": [{"role": "assistant", "content": echoed}]}
        assert tool_echo.restore_served_turns(body, store, None) == (body, [])


class TestStoreFiles:
    def test_stale_served_turns_are_pruned_and_restore_logs_kept(self, tmp_path) -> None:
        store = tool_echo.ServedToolInputStore(tmp_path / "store", retention_s=60)
        store.record("old", [tool_echo.served_turn({"content": [_tool("call_1", "Bash", SERVED)]})])
        body = {"messages": _history([_tool("call_1", "Bash", STRIPPED)])}
        assert tool_echo.restore_served_turns(body, store, "old")[1]
        stale = time() - 3600
        for path in (tmp_path / "store").iterdir():
            os.utime(path, (stale, stale))
        store._last_prune = 0.0
        store.record("new", [tool_echo.served_turn({"content": [_tool("call_2", "Bash", SERVED)]})])
        assert sorted(path.name for path in (tmp_path / "store").iterdir()) == ["new.jsonl", "old.restores.jsonl"]

    def test_a_store_directory_others_can_read_is_refused(self, tmp_path) -> None:
        root = tmp_path / "shared"
        root.mkdir(mode=0o755)
        root.chmod(0o755)
        with pytest.raises(ValueError):
            tool_echo.ServedToolInputStore(root)
