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
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.base_responses_api_model import (
    CaptureStore,
    ModelCallCaptureConfig,
    ModelCallCaptureSelection,
    _CaptureMiddleware,
    install_model_call_capture,
)
from nemo_gym.global_config import NEMO_GYM_RESERVED_TOP_LEVEL_KEYS


def selection(**overrides):
    return ModelCallCaptureSelection(
        **{
            "model_server_names": ["judge"],
            "max_records_per_model": 4,
            "max_record_bytes": 4096,
            "max_total_bytes": 32768,
            "pending_records_per_process": 2,
            "capture_until": datetime.now(timezone.utc) + timedelta(hours=1),
            **overrides,
        }
    )


def exchange():
    return {"model_ref": {"name": "judge"}, "request": {"input": "question"}, "response": {"output": "answer"}}


@pytest.mark.parametrize(
    "changes",
    [
        {"model_server_names": []},
        {"model_server_names": ["judge", "judge"]},
        {"model_server_names": ["*"]},
        {"max_records_per_model": 0},
        {"max_record_bytes": 0},
        {"max_total_bytes": 1},
        {"pending_records_per_process": 0},
        {"capture_until": datetime(2026, 1, 1)},
    ],
)
def test_selection_rejects_unbounded_or_ambiguous_input(changes):
    with pytest.raises(ValidationError):
        selection(**changes)


def test_selected_capture_requires_full_mode():
    with pytest.raises(ValidationError):
        ModelCallCaptureConfig(model_call_capture_selection=selection())
    assert "model_call_capture_selection" in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS


def test_private_store_shared_quota_and_redaction(tmp_path):
    policy = selection(max_records_per_model=3)
    stores = [CaptureStore(tmp_path / "private", policy) for _ in range(2)]
    row = exchange()
    row["request"]["api_key"] = "not-for-retention"
    row["response_raw"] = "opaque"
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda i: stores[i % 2].record("episode", row), range(12)))
    rows = stores[0].read("episode")
    assert len(rows) == 3
    assert all(r["request"]["api_key"] == "[REDACTED]" for r in rows)
    assert all(not r["capture_retention"]["replay_eligible"] for r in rows)
    assert all("response_raw" not in r for r in rows)
    assert (tmp_path / "private").stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in (tmp_path / "private").iterdir())
    quota = json.loads((tmp_path / "private/.capture-quota.json").read_text())
    assert quota["models"] == {"judge": 3}
    assert quota["bytes"] == stores[0].path_for("episode").stat().st_size


def test_byte_cap_window_and_model_exclusion(tmp_path):
    store = CaptureStore(tmp_path / "bytes", selection(max_total_bytes=4096))
    row = exchange()
    row["request"]["input"] = "q" * 5000
    store.record("oversize", row)
    assert not store.path_for("oversize").exists()
    row = exchange()
    row["model_ref"]["name"] = "policy"
    store.record("excluded", row)
    assert not store.path_for("excluded").exists()
    expired = CaptureStore(
        tmp_path / "expired", selection(capture_until=datetime.now(timezone.utc) - timedelta(seconds=1))
    )
    assert not expired.admit()
    assert store.dropped == {"record_byte_limit": 1, "unselected_model": 1}


def test_private_paths_and_quota_policy_refuse(tmp_path):
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    with pytest.raises(ValueError):
        CaptureStore(public, selection())
    store = CaptureStore(tmp_path / "private", selection())
    store.record("one", exchange())
    other = CaptureStore(store.root, selection(max_records_per_model=5))
    with pytest.raises(ValueError, match="policy differs"):
        other.record("two", exchange())
    target = tmp_path / "foreign"
    target.write_text("preserve")
    store.path_for("link").symlink_to(target)
    with pytest.raises(OSError):
        store.record("link", exchange())
    assert target.read_text() == "preserve"


def test_nonblocking_terminal_and_bounded_pending_records(tmp_path):
    store = CaptureStore(tmp_path / "private", selection(pending_records_per_process=1))
    entered, release = threading.Event(), threading.Event()
    original = store.record

    def slow_record(*args):
        entered.set()
        assert release.wait(5)
        original(*args)

    store.record = slow_record
    delivered = []

    async def app(scope, receive, send):
        await receive()
        await send(
            {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]}
        )
        await send(
            {
                "type": "http.response.body",
                "body": b'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed","output":[]}}\n\n',
                "more_body": False,
            }
        )

    async def receive():
        return {"type": "http.request", "body": b'{"input":"question"}'}

    async def send(message):
        delivered.append(dict(message))

    scope = {"type": "http", "path": "/ng-rollout/episode/v1/responses", "headers": []}
    middleware = _CaptureMiddleware(app, store=store, model_server_name="judge")
    try:
        asyncio.run(middleware(scope, receive, send))
        assert delivered[-1]["more_body"] is False
        assert entered.wait(2)
        # The fsync path is still blocked, but terminal delivery already finished.
        assert not store.admit()
        assert store.dropped["pending_limit"] == 1
    finally:
        release.set()
        store._writer.shutdown(wait=True)
    [row] = store.read("episode")
    assert row["error_category"] is None
    assert row["response"]["status"] == "completed"


def test_install_excludes_policy_and_keeps_response_unchanged(tmp_path):
    app = FastAPI()

    @app.post("/v1/responses")
    async def responses():
        return {"output": "same"}

    config = ModelCallCaptureConfig(
        observability_enabled=True,
        model_call_capture_dir=tmp_path / "private",
        model_call_capture_selection=selection(),
    )
    install_model_call_capture(app, config, model_server_name="policy")
    with TestClient(app) as client:
        assert client.post("/ng-rollout/episode/v1/responses", json={"input": "q"}).json() == {"output": "same"}
    assert not (tmp_path / "private").exists()
