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
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import orjson
import pytest

from nemo_gym.base_responses_api_model import (
    CaptureStore,
    ModelCallCaptureSelection,
    clear_model_call_captures_for_rollouts,
)
from nemo_gym.model_call_capture_files import (
    _append_locked,
    capture_paths,
    open_capture,
    read_capture_bytes,
    seal_capture,
    seal_idle_captures,
    sealed_capture_path,
)
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body


def selection():
    return ModelCallCaptureSelection(
        model_server_names=["judge"],
        max_records_per_model=4,
        max_record_bytes=4096,
        max_total_bytes=32768,
        pending_records_per_process=2,
        capture_until=datetime.now(timezone.utc) + timedelta(hours=1),
    )


def exchange():
    return {"model_ref": {"name": "judge"}, "request": {"input": "question"}, "response": {"output": "answer"}}


def turn(index: int) -> dict:
    """One agentic policy call: each request resends the whole conversation so far."""
    messages = [{"role": "system", "content": "You are a careful software engineer. " * 40}]
    for step in range(index):
        messages.append(
            {"role": "assistant", "content": f"Inspecting module {step}: " + "def handler(x): return x\n" * 30}
        )
        messages.append({"role": "tool", "content": f"$ pytest -q tests/test_{step}.py\n" + "PASSED test case\n" * 40})
    return {
        "model_call_id": f"call-{index}",
        "dialect": "chat_completions",
        "model_ref": {"type": "responses_api_models", "name": "policy_model"},
        "status_code": 200,
        "request": {"model": "policy", "messages": messages, "temperature": 1.0},
        "response": {"choices": [{"message": {"role": "assistant", "content": f"Next I run step {index}."}}]},
    }


def test_sealed_capture_round_trip_and_late_appends(tmp_path):
    store = CaptureStore(tmp_path)
    for index in range(3):
        store.record("episode", turn(index))
    raw = store.path_for("episode").read_bytes()

    assert seal_capture(store.path_for("episode"))
    assert not store.path_for("episode").exists()
    assert sealed_capture_path(store.path_for("episode")).name == "episode.capture.jsonl.zst"
    assert read_capture_bytes(store.path_for("episode")) == raw
    assert store.read("episode") == [turn(index) for index in range(3)]

    # A judge call after the seal starts a plain tail that reads after the sealed records.
    store.record("episode", exchange())
    assert store.path_for("episode").exists()
    assert store.read("episode") == [turn(0), turn(1), turn(2), exchange()]
    assert seal_capture(store.path_for("episode"))
    assert not seal_capture(store.path_for("episode"))
    assert store.read("episode") == [turn(0), turn(1), turn(2), exchange()]
    assert store.exists("episode")


def test_readers_accept_plain_sealed_and_mixed_captures(tmp_path):
    store = CaptureStore(tmp_path)
    for name in ("plain", "sealed", "mixed"):
        store.record(name, turn(1))
    seal_capture(store.path_for("sealed"))
    seal_capture(store.path_for("mixed"))
    store.record("mixed", turn(2))
    (tmp_path / "unrelated.json").write_text("{}")

    assert [path.name for path in capture_paths(tmp_path)] == [
        "mixed.capture.jsonl",
        "plain.capture.jsonl",
        "sealed.capture.jsonl",
    ]
    assert store.read("plain") == store.read("sealed") == [turn(1)]
    assert store.read("mixed") == [turn(1), turn(2)]
    assert store.read("missing") == []


@pytest.mark.parametrize("killed_at", ["replace sealed", "unlink plain"])
@pytest.mark.parametrize("earlier_seal", [False, True])
def test_interrupted_seal_neither_loses_nor_duplicates_records(tmp_path, monkeypatch, killed_at, earlier_seal):
    store = CaptureStore(tmp_path)
    path = store.path_for("episode")
    expected = [turn(1), turn(2)] if earlier_seal else [turn(2)]
    if earlier_seal:
        store.record("episode", turn(1))
        seal_capture(path)
    store.record("episode", turn(2))
    os.utime(path, (0, 0))  # sealing selects idle captures

    def killed(*args):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace" if killed_at == "replace sealed" else "unlink", killed)
        with pytest.raises(KeyboardInterrupt):
            seal_capture(path)
    assert path.exists() and not any(p.name.endswith(".tmp") for p in tmp_path.iterdir())

    assert store.read("episode") == expected
    assert [p.name for p in capture_paths(tmp_path)] == ["episode.capture.jsonl"]
    # The next seal finishes the interrupted one: it moves records only if none were sealed.
    assert seal_capture(path) is (killed_at == "replace sealed")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["episode.capture.jsonl.zst"]
    assert store.read("episode") == expected
    store.record("episode", turn(3))
    assert store.read("episode") == expected + [turn(3)]


def test_an_identical_record_after_a_seal_is_kept(tmp_path):
    store = CaptureStore(tmp_path)
    store.record("episode", exchange())
    seal_capture(store.path_for("episode"))
    store.record("episode", exchange())

    assert store.read("episode") == [exchange(), exchange()]


def test_append_racing_a_seal_lands_in_the_new_plain_file(tmp_path):
    store = CaptureStore(tmp_path)
    store.record("episode", turn(1))
    path = store.path_for("episode")
    opened = []

    def open_then_lose_race(target):
        fd = os.open(target, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o666)
        if not opened:
            # The seal takes the lock first and unlinks the file this writer opened.
            assert seal_capture(target)
        opened.append(fd)
        return fd

    _append_locked(path, orjson.dumps(turn(2)) + b"\n", open_then_lose_race)

    assert len(opened) == 2
    assert store.read("episode") == [turn(1), turn(2)]


def test_idle_sweep_and_clear(tmp_path):
    store = CaptureStore(tmp_path)
    store.record("old", turn(1))
    store.record("active", turn(1))
    os.utime(store.path_for("old"), (0, 0))

    assert seal_idle_captures(tmp_path, idle_s=600) == 1
    assert sealed_capture_path(store.path_for("old")).exists()
    assert store.path_for("active").exists()
    assert seal_idle_captures(tmp_path / "absent", idle_s=0) == 0

    record = {"_ng_task_index": 0, "_ng_rollout_index": 0}
    rollout_id = maybe_rollout_id_from_run_body(record)
    store.record(rollout_id, turn(1))
    seal_capture(store.path_for(rollout_id))
    clear_model_call_captures_for_rollouts([record], [tmp_path])
    assert not store.exists(rollout_id)


def test_selected_capture_seal_keeps_owner_private_mode(tmp_path):
    store = CaptureStore(tmp_path / "private", selection())
    store.record("episode", exchange())
    seal_capture(store.path_for("episode"))
    store.record("episode", exchange())

    assert sealed_capture_path(store.path_for("episode")).stat().st_mode & 0o777 == 0o600
    assert len(store.read("episode")) == 2


def test_open_capture_streams_sealed_then_plain_lines(tmp_path):
    store = CaptureStore(tmp_path)
    store.record("episode", turn(1))
    seal_capture(store.path_for("episode"))
    store.record("episode", turn(2))

    with open_capture(store.path_for("episode")) as lines:
        assert [orjson.loads(line) for line in lines] == [turn(1), turn(2)]
    with pytest.raises(FileNotFoundError):
        with open_capture(tmp_path / "missing.capture.jsonl"):
            pass


def test_plain_captures_read_without_zstandard(tmp_path):
    store = CaptureStore(tmp_path)
    store.record("episode", turn(1))
    # -S leaves out site-packages: only the standard library and this checkout import.
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "from nemo_gym.model_call_capture_files import read_capture_bytes;"
        "sys.stdout.buffer.write(read_capture_bytes(sys.argv[2]))"
    )
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-S", "-c", code, str(root), str(store.path_for("episode"))], capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == store.path_for("episode").read_bytes()


def test_agentic_capture_compresses_by_more_than_twenty_times(tmp_path):
    store = CaptureStore(tmp_path)
    for index in range(40):
        store.record("episode", turn(index))
    raw = store.path_for("episode").stat().st_size
    seal_capture(store.path_for("episode"))
    sealed = sealed_capture_path(store.path_for("episode")).stat().st_size

    assert raw > 1_000_000
    assert raw / sealed > 20
