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

"""Prepare IFEval evaluation data for NeMo Gym.

Downloads the original IFEval test data from the google-research GitHub repo
and converts it to Gym JSONL format compatible with the
`instruction_following` resources server.

Each output row has two keys:
  responses_create_params  — the agent input (user message with the prompt)
  verifier_metadata        — scoring fields read by the resources server

grading_mode is set to "binary" to align with the original paper's
prompt-level strict accuracy metric (1 if all instructions pass, 0 otherwise).
"""

import hashlib
import json
import urllib.request
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
RAW_FPATH = DATA_DIR / "input_data.jsonl"
OUTPUT_FPATH = DATA_DIR / "ifeval_benchmark.jsonl"
MANIFEST_FPATH = DATA_DIR / "ifeval_benchmark.manifest.json"

# Pinned to the last upstream commit that changed the data file (2024-06-11), not to
# `master`: these prompts and kwargs are the scoring input, so a moving branch would let
# an upstream edit change scores between runs of identical code. Bumping the commit
# requires re-reading the new file and updating the digest and counts below.
UPSTREAM_COMMIT = "26d8ccdab6fec61b5c83ad6327ea8bda9e580288"  # pragma: allowlist secret
URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    f"{UPSTREAM_COMMIT}/instruction_following_eval/data/input_data.jsonl"
)
SOURCE_SHA256 = "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49"  # pragma: allowlist secret
EXPECTED_PROMPTS = 541
EXPECTED_INSTRUCTIONS = 834


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()


def prepare() -> Path:
    """Download IFEval input data and convert to Gym JSONL format."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading IFEval input data from {URL} ...")
    raw = _fetch(URL)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"IFEval source digest {digest} differs from the pinned {SOURCE_SHA256}; refusing to prepare")
    RAW_FPATH.write_bytes(raw)

    rows = []
    instructions = 0
    with open(RAW_FPATH, "rt", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            entry = json.loads(line)
            instructions += len(entry["instruction_id_list"])
            prompt = entry["prompt"]
            row = {
                "id": idx,
                "responses_create_params": {
                    "input": [{"role": "user", "content": prompt}],
                    "tools": [],
                    "parallel_tool_calls": False,
                },
                "verifier_metadata": {
                    "prompt": prompt,
                    "instruction_id_list": entry["instruction_id_list"],
                    "kwargs": entry["kwargs"],
                    "grading_mode": "binary",
                },
            }
            rows.append(row)

    if (len(rows), instructions) != (EXPECTED_PROMPTS, EXPECTED_INSTRUCTIONS):
        raise ValueError(
            f"IFEval has {len(rows)} prompts and {instructions} instructions; "
            f"expected {EXPECTED_PROMPTS} and {EXPECTED_INSTRUCTIONS}"
        )

    with open(OUTPUT_FPATH, "wt", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")

    MANIFEST_FPATH.write_text(
        json.dumps(
            {
                "upstream_commit": UPSTREAM_COMMIT,
                "source_url": URL,
                "source_sha256": digest,
                "prompts": len(rows),
                "instructions": instructions,
                "output_sha256": hashlib.sha256(OUTPUT_FPATH.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} problems to {OUTPUT_FPATH} (manifest {MANIFEST_FPATH})")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
