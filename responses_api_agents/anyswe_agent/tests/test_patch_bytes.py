# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Patch transport preserves real Git file bytes, including binary deletions."""

import shutil
import subprocess

import pytest

from responses_api_agents.anyswe_agent.agent_runner import _extract_patch, _snapshot_repo


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (b"inventory\n", b"# compressed inventory\nx\xda\x85\xff\rABC\n"),
        (b"compiled\x00\xff\r\n", None),
        (b"old\r\n", b"new\r\nembedded\rcarriage\r\n"),
        (b"old\n", b"plain text\n"),
    ],
    ids=["non-utf8-non-nul", "binary-deletion", "carriage-returns", "plain-text"],
)
def test_extracted_patch_recreates_exact_candidate(tmp_path, monkeypatch, before, after):
    monkeypatch.delenv("GIT_DIR", raising=False)
    monkeypatch.delenv("GIT_WORK_TREE", raising=False)
    monkeypatch.delenv("GIT_INDEX_FILE", raising=False)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")

    def git(root, *args, data=None):
        return subprocess.run(
            ["git", *args],
            cwd=root,
            input=data,
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout

    repo = tmp_path / "agent"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "core.autocrlf", "false")
    (repo / "fixture").write_bytes(before)
    (repo / "plain.py").write_bytes(b"baseline = 1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "baseline")
    grader = tmp_path / "grader"
    shutil.copytree(repo, grader)
    index = tmp_path / "baseline.index"
    baseline = _snapshot_repo(repo, index)
    attributes = index.with_name(index.name + ".git") / "info" / "attributes"
    original_attributes = attributes.read_bytes() if attributes.exists() else None
    if after is None:
        (repo / "fixture").unlink()
    else:
        (repo / "fixture").write_bytes(after)
    (repo / "plain.py").write_bytes(b"baseline = 2\n")
    (repo / "plain.py").chmod(0o755)
    shutil.rmtree(repo / ".git")
    patch = _extract_patch(repo, index, baseline)
    patch_file = tmp_path / "patch.diff"
    patch_file.write_text(patch, encoding="utf-8")
    git(grader, "apply", "--check", "-", data=patch_file.read_bytes())
    git(grader, "apply", "-", data=patch_file.read_bytes())
    assert (grader / "fixture").read_bytes() == after if after is not None else not (grader / "fixture").exists()
    assert (grader / "plain.py").read_bytes() == b"baseline = 2\n"
    assert (grader / "plain.py").stat().st_mode & 0o111 == 0o111
    assert (attributes.read_bytes() if attributes.exists() else None) == original_attributes
