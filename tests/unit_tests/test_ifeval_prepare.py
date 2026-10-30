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
import hashlib
import json
import re
from pathlib import Path

import pytest

from benchmarks.ifeval import prepare as prepare_module


FIXTURE = b"".join(
    json.dumps(row).encode() + b"\n"
    for row in (
        {
            "key": 1,
            "prompt": "Write in lowercase.",
            "instruction_id_list": ["change_case:english_lowercase"],
            "kwargs": [{}],
        },
        {
            "key": 2,
            "prompt": "Two paragraphs, no commas.",
            "instruction_id_list": ["length_constraints:number_paragraphs", "punctuation:no_comma"],
            "kwargs": [{"num_paragraphs": 2}, {}],
        },
    )
)


def _redirect(monkeypatch, tmp_path: Path, payload: bytes, *, digest: str, counts: tuple[int, int]) -> None:
    monkeypatch.setattr(prepare_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(prepare_module, "RAW_FPATH", tmp_path / "input_data.jsonl")
    monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", tmp_path / "ifeval_benchmark.jsonl")
    monkeypatch.setattr(prepare_module, "MANIFEST_FPATH", tmp_path / "ifeval_benchmark.manifest.json")
    monkeypatch.setattr(prepare_module, "_fetch", lambda url: payload)
    monkeypatch.setattr(prepare_module, "SOURCE_SHA256", digest)
    monkeypatch.setattr(prepare_module, "EXPECTED_PROMPTS", counts[0])
    monkeypatch.setattr(prepare_module, "EXPECTED_INSTRUCTIONS", counts[1])


def test_source_is_an_immutable_commit() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", prepare_module.UPSTREAM_COMMIT)
    assert prepare_module.UPSTREAM_COMMIT in prepare_module.URL
    assert "master" not in prepare_module.URL and "main" not in prepare_module.URL
    assert re.fullmatch(r"[0-9a-f]{64}", prepare_module.SOURCE_SHA256)


def test_prepare_converts_and_records_digests(monkeypatch, tmp_path: Path) -> None:
    _redirect(monkeypatch, tmp_path, FIXTURE, digest=hashlib.sha256(FIXTURE).hexdigest(), counts=(2, 3))

    output = prepare_module.prepare()

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == [0, 1]
    assert rows[1]["verifier_metadata"]["instruction_id_list"] == [
        "length_constraints:number_paragraphs",
        "punctuation:no_comma",
    ]
    assert rows[1]["verifier_metadata"]["grading_mode"] == "binary"
    assert rows[0]["responses_create_params"]["input"] == [{"role": "user", "content": "Write in lowercase."}]
    manifest = json.loads((tmp_path / "ifeval_benchmark.manifest.json").read_text(encoding="utf-8"))
    assert manifest["upstream_commit"] == prepare_module.UPSTREAM_COMMIT
    assert manifest["source_sha256"] == hashlib.sha256(FIXTURE).hexdigest()
    assert manifest["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert (manifest["prompts"], manifest["instructions"]) == (2, 3)


def test_prepare_refuses_a_changed_source_before_writing(monkeypatch, tmp_path: Path) -> None:
    _redirect(monkeypatch, tmp_path, FIXTURE + b"\n", digest=hashlib.sha256(FIXTURE).hexdigest(), counts=(2, 3))

    with pytest.raises(ValueError, match="differs from the pinned"):
        prepare_module.prepare()

    assert list(tmp_path.iterdir()) == []


def test_prepare_refuses_a_wrong_inventory(monkeypatch, tmp_path: Path) -> None:
    _redirect(monkeypatch, tmp_path, FIXTURE, digest=hashlib.sha256(FIXTURE).hexdigest(), counts=(541, 834))

    with pytest.raises(ValueError, match="expected 541 and 834"):
        prepare_module.prepare()

    assert not (tmp_path / "ifeval_benchmark.jsonl").exists()
