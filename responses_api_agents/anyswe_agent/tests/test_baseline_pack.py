# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-Git regression, ready to fold into AnySWE's existing test_app.py.

Run with the hydrated Gym source on PYTHONPATH. The source filesystem must
offer writable /dev/shm distinct from pytest's tmp_path; otherwise skip.
This exercises the actual runner functions, without subprocess doubles.
"""

import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from responses_api_agents.anyswe_agent.agent_runner import (
    _extract_patch,
    _snapshot_repo,
)


def test_baseline_export_crosses_filesystems_and_survives_metadata_deletion(tmp_path, monkeypatch):
    source_mount = Path("/dev/shm")
    if not source_mount.is_dir() or not os.access(source_mount, os.W_OK):
        pytest.skip("a writable /dev/shm is required for the cross-filesystem regression")
    if source_mount.stat().st_dev == tmp_path.stat().st_dev:
        pytest.skip("source repository and baseline archive must use different filesystems")

    # Keep Git discovery/config independent of the test caller's checkout.
    for key in tuple(os.environ):
        if key.startswith("GIT_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("LC_ALL", "C")

    def git(repo, *args, **kwargs):
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
            **kwargs,
        )

    with TemporaryDirectory(prefix="anyswe-pack-regression-", dir=source_mount) as source:
        repo = Path(source) / "repo"
        repo.mkdir()
        git(repo, "init", "-q")
        git(repo, "config", "user.email", "test@example.com")
        git(repo, "config", "user.name", "Test")
        git(repo, "config", "core.filemode", "true")
        (repo / "tracked.txt").write_text("committed\n")
        (repo / "run.sh").write_text("#!/bin/sh\necho baseline\n")
        (repo / "removed.txt").write_text("delete after the snapshot\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "initial")
        # The hardened image exposes an exact shallow base, not its ancestors.
        (repo / ".git/shallow").write_text(git(repo, "rev-parse", "HEAD").stdout)
        (repo / "tracked.txt").write_text("image baseline\n")
        (repo / "image-dirty.txt").write_text("publisher fixture\n")

        # Reproduce the previous command's real EXDEV rename failure first.
        old_pack_dir = tmp_path / "old-export"
        old_pack_dir.mkdir()
        tree = git(repo, "rev-parse", "HEAD^{tree}").stdout
        original = subprocess.run(
            ["git", "pack-objects", "--revs", str(old_pack_dir / "pack")],
            cwd=repo,
            input=tree,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert original.returncode == 128, original.stderr
        assert "unable to rename" in original.stderr, original.stderr
        assert "Invalid cross-device link" in original.stderr, original.stderr

        index = tmp_path / "baseline.index"
        baseline = _snapshot_repo(repo, index)
        archive = index.with_name(index.name + ".git")
        assert archive.stat().st_dev != repo.stat().st_dev
        assert not (archive / "objects/info/alternates").exists()

        # This grader starts from the exact image tree that was snapshotted.
        grader = tmp_path / "grader"
        shutil.copytree(repo, grader, ignore=shutil.ignore_patterns(".git"))
        git(grader, "init", "-q")
        git(grader, "config", "user.email", "test@example.com")
        git(grader, "config", "user.name", "Test")
        git(grader, "config", "core.filemode", "true")
        git(grader, "add", "-A")
        git(grader, "commit", "-qm", "image baseline")

        (repo / "tracked.txt").write_text("agent edit\n")
        (repo / "run.sh").chmod(0o755)
        (repo / "removed.txt").unlink()
        (repo / "new.sh").write_text("#!/bin/sh\necho agent\n")
        (repo / "new.sh").chmod(0o755)
        (repo / "agent-link").symlink_to("new.sh")
        shutil.rmtree(repo / ".git")

        # The private archive must carry every object without the source store.
        archive_env = {**os.environ, "GIT_DIR": str(archive)}
        kinds = git(
            grader,
            "cat-file",
            "--batch-all-objects",
            "--batch-check=%(objecttype)",
            env=archive_env,
        ).stdout.splitlines()
        assert set(kinds) == {"tree", "blob"}
        patch = _extract_patch(repo, index, baseline)
        assert "image-dirty.txt" not in patch
        git(grader, "apply", "--check", "-", input=patch)
        git(grader, "apply", "-", input=patch)
        assert (grader / "tracked.txt").read_text() == "agent edit\n"
        assert (grader / "image-dirty.txt").read_text() == "publisher fixture\n"
        assert (grader / "run.sh").stat().st_mode & 0o111 == 0o111
        assert (grader / "new.sh").read_text() == "#!/bin/sh\necho agent\n"
        assert (grader / "new.sh").stat().st_mode & 0o111 == 0o111
        assert (grader / "agent-link").is_symlink()
        assert os.readlink(grader / "agent-link") == "new.sh"
        assert not (grader / "removed.txt").exists()
        assert not (repo / ".git").exists()
