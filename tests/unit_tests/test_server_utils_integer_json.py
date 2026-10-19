# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
import datetime
import json
import uuid

import orjson
import pytest
from aiohttp import ClientSession, web

from nemo_gym import server_utils


@pytest.mark.parametrize(
    "value",
    [
        {"value": -(1 << 63)},
        {"value": (1 << 64) - 1},
        {"value": 42, "flag": True, "none": None, "text": "é"},
        {"value": [1, 2, {"nested": 3}]},
        {"value": (1, 2)},
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)},
        {"value": uuid.UUID(int=0)},
    ],
)
def test_normal_json_bytes_unchanged(value):
    assert server_utils._encode_json_body(value) == orjson.dumps(value)


@pytest.mark.parametrize("integer", [1 << 64, -(1 << 63) - 1, 10**100, -(10**80)])
def test_oversized_integer_value_is_exact(integer):
    value = {"nested": [False, None, {"integer": integer}]}
    assert json.loads(server_utils._encode_json_body(value)) == value


@pytest.mark.parametrize(
    "value",
    [
        {"value": object()},
        {"value": {1, 2}},
        {"value": b"bytes"},
        {"value": 1 << 80, "unsupported": object()},
        {"value": 1 << 80, "unsupported": {1, 2}},
    ],
)
def test_unsupported_values_still_raise(value):
    with pytest.raises(TypeError):
        server_utils._encode_json_body(value)


@pytest.mark.parametrize("kind", ["dict", "list", "mixed"])
def test_cycles_remain_json_type_errors(kind):
    if kind == "dict":
        value = {"integer": 1 << 80}
        value["cycle"] = value
    elif kind == "list":
        value = [1 << 80]
        value.append(value)
    else:
        cycle = []
        cycle.append(cycle)
        value = {"integer": 1 << 80, "cycle": cycle}
    with pytest.raises(TypeError, match="Recursion limit reached"):
        server_utils._encode_json_body(value)


class OverrideInt(int):
    def __str__(self):
        return "0"

    def __int__(self):
        return 0

    def __le__(self, other):
        return True

    def __ge__(self, other):
        return True


@pytest.mark.parametrize("integer", [1 << 80, -(1 << 80)])
def test_integer_subclass_overrides_cannot_change_wire_value(integer):
    encoded = server_utils._encode_json_body({"value": OverrideInt(integer)})
    assert json.loads(encoded) == {"value": integer}


def test_shared_child_is_serializable_and_not_mutated():
    child = {"integer": 1 << 80}
    value = {"first": child, "second": child}
    assert json.loads(server_utils._encode_json_body(value)) == value
    assert value["first"] is child and value["second"] is child
    assert child == {"integer": 1 << 80}


async def test_native_request_preserves_integer_headers_and_cookie(monkeypatch):
    assert server_utils._GLOBAL_AIOHTTP_CLIENT is None
    received = []

    async def receive(request):
        body = json.loads(await request.read())
        received.append((body, request.cookies.get("session"), request.headers.get("X-Test")))
        response = web.json_response(body)
        response.set_cookie("session", "retained")
        return response

    app = web.Application()
    app.router.add_post("/tool", receive)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    client = ClientSession(connector=server_utils.PollSafeTCPConnector())
    monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", client)
    value = {"integer": 1 << 80, "nested": [True, None]}
    try:
        response = await server_utils.request(
            "POST",
            f"http://127.0.0.1:{port}/tool",
            json=value,
            cookies={"session": "retained"},
            headers={"X-Test": "integer"},
        )
        async with response:
            assert response.status == 200
            assert response.cookies["session"].value == "retained"
            assert await response.json() == value
        assert received == [(value, "retained", "integer")]
    finally:
        await server_utils.close_global_aiohttp_client()
        await runner.cleanup()
    assert client.closed and server_utils._GLOBAL_AIOHTTP_CLIENT is None
    assert site._server is None or not site._server.is_serving()
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), 1)
    except ConnectionRefusedError:
        pass
    else:
        writer.close()
        await writer.wait_closed()
        pytest.fail("Listener still accepts connections after cleanup")
