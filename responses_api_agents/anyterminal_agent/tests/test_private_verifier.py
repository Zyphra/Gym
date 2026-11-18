# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A process left in the sandbox cannot reach the directory the reward is read from."""

import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from nemo_gym.sandbox.providers.docker import provider as docker_provider
from responses_api_agents.anyterminal_agent import app, private_verifier
from responses_api_agents.anyterminal_agent.tests.test_app import _make_instance_config
from responses_api_agents.anyterminal_agent.verifier_integrity import VerifierIntegrityRefused


HELPER = Path(private_verifier.__file__)


def _namespaces_available() -> bool:
    unshare = shutil.which("unshare")
    return unshare is not None and subprocess.run([unshare, "-rm", "true"], capture_output=True).returncode == 0


def _run_helper(private: Path, command: str) -> subprocess.CompletedProcess:
    """The real helper as namespace root, with its verifier path at ``private``."""
    driver = textwrap.dedent(
        """
        import runpy, sys
        module = runpy.run_path(sys.argv[1], run_name="private_verifier")
        module["main"].__globals__["VERIFIER_DIR"] = sys.argv[2]
        raise SystemExit(module["main"](sys.argv[3:]))
        """
    )
    return subprocess.run(
        ["unshare", "-rm", sys.executable, "-I", "-S", "-c", driver, str(HELPER), str(private), command],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_sandbox_paths_of_the_verifier_output_are_a_scratch_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(docker_provider, "_require_docker", lambda: "/usr/bin/docker")
    cfg = _make_instance_config(tmp_path, sandbox_provider={"docker": {}})
    args = app._build_provider(cfg)._create_config.extra_run_args
    binds = [args[index + 1] for index, value in enumerate(args) if value == "-v"]
    assert f"{cfg.sandbox_verifier_dir}:/logs/verifier" in binds
    # The persistent directory holds verifier_dir; its sandbox path is covered after it is bound.
    covered = binds.index(f"{cfg.sandbox_verifier_dir}:/trajectories_mount/verifier")
    assert binds.index(f"{cfg.persistent_dir}:/trajectories_mount") < covered
    assert not any(bind.startswith(f"{cfg.verifier_dir}:") for bind in binds)


@pytest.mark.skipif(not _namespaces_available(), reason="needs unprivileged user and mount namespaces")
def test_a_left_behind_writer_cannot_change_the_reported_reward(tmp_path):
    shared = tmp_path / "logs-verifier"
    shared.mkdir()
    # The writer every grading-isolation probe used: it waits for the reward, then overwrites it.
    writer = subprocess.Popen(
        [
            "sh",
            "-c",
            f'while :; do [ -e "{shared}/reward.txt" ] && printf 1 > "{shared}/reward.txt"; sleep 0.01; done',
        ],
    )
    try:
        result = _run_helper(
            shared,
            f'printf 0 > "{shared}/reward.txt"; head -c 200000 /dev/zero > "{shared}/pytest.log"; '
            f'sleep 0.3; [ "$(cat "{shared}/reward.txt")" = 0 ]',
        )
    finally:
        writer.kill()
        writer.wait()
    assert result.returncode == 0, result.stderr
    (line,) = [line for line in result.stdout.splitlines() if line.startswith(private_verifier.MARKER)]
    report = json.loads(line[len(private_verifier.MARKER) :])
    files = {entry["path"]: entry for entry in report["files"]}
    assert files["reward.txt"]["data"] == "MA=="  # b"0", inline: never read back from the sandbox
    assert "data" not in files["pytest.log"]  # too large to inline: copied out with its digest
    assert (shared / "pytest.log").stat().st_size == 200000
    assert not (shared / "reward.txt").exists()  # the writer never saw the reward file


@pytest.mark.skipif(not _namespaces_available(), reason="needs unprivileged user and mount namespaces")
def test_setup_failure_is_reported_and_the_command_never_runs(tmp_path):
    marker = tmp_path / "ran"
    (tmp_path / "file").write_text("")
    result = _run_helper(tmp_path / "file" / "verifier", f"touch {marker}")
    report = json.loads(result.stdout.split(private_verifier.MARKER, 1)[1])
    assert result.returncode == private_verifier.SETUP_FAILED and report["setup"]
    assert not marker.exists()


def _report(cfg, monkeypatch, outputs: dict[str, bytes]) -> str:
    """A report from the real collector over a private directory holding ``outputs``."""
    private = cfg.persistent_dir.parent / "private"
    for name, data in outputs.items():
        (private / name).parent.mkdir(parents=True, exist_ok=True)
        (private / name).write_bytes(data)
    cfg.sandbox_verifier_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(private_verifier, "VERIFIER_DIR", str(private))
    shared = os.open(cfg.sandbox_verifier_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        report = private_verifier.collect(shared)
    finally:
        os.close(shared)
    report.update(schema="private-verifier/v1", setup=None, exit_code=0)
    return "other output\n" + private_verifier.MARKER + json.dumps(report) + "\n"


def test_host_writes_exactly_what_test_sh_wrote(tmp_path, monkeypatch):
    cfg = _make_instance_config(tmp_path)
    large = os.urandom(private_verifier.INLINE_FILE_BYTES + 1)
    stdout = _report(cfg, monkeypatch, {"reward.txt": b"1\n", "artifacts/score.json": b"{}", "pytest.log": large})
    (cfg.sandbox_verifier_dir / "reward.txt").write_text("1")  # what a sandbox process wrote is never read
    started = time.monotonic()
    receipt = app._write_private_outputs(stdout, cfg)
    assert (cfg.verifier_dir / "reward.txt").read_bytes() == b"1\n"
    assert (cfg.verifier_dir / "artifacts/score.json").read_bytes() == b"{}"
    assert (cfg.verifier_dir / "pytest.log").read_bytes() == large
    assert 0 <= receipt["host_s"] <= time.monotonic() - started
    recorded = json.loads((cfg.verifier_dir / app.PRIVATE_VERIFIER_RECEIPT).read_text())
    assert {entry["path"] for entry in recorded["files"]} == {"reward.txt", "artifacts/score.json", "pytest.log"}
    assert all("data" not in entry for entry in recorded["files"])


def test_a_large_output_changed_in_the_sandbox_is_refused(tmp_path, monkeypatch):
    cfg = _make_instance_config(tmp_path)
    stdout = _report(cfg, monkeypatch, {"pytest.log": b"x" * (private_verifier.INLINE_FILE_BYTES + 1)})
    (cfg.sandbox_verifier_dir / "pytest.log").write_text("forged")
    with pytest.raises(VerifierIntegrityRefused, match="differs"):
        app._write_private_outputs(stdout, cfg)


@pytest.mark.parametrize("path", ["../reward.txt", "/logs/verifier/reward.txt", "a//b", ""])
def test_a_report_cannot_name_a_path_outside_the_verifier_directory(tmp_path, path):
    cfg = _make_instance_config(tmp_path)
    entry = {"path": path, "size": 1, "mode": 0o644, "sha256": "0" * 64, "data": "MQ=="}
    stdout = app.PRIVATE_VERIFIER_MARKER + json.dumps({"setup": None, "files": [entry], "links": []})
    with pytest.raises(VerifierIntegrityRefused, match="not relative"):
        app._write_private_outputs(stdout, cfg)


@pytest.mark.parametrize(
    "stdout,message",
    [
        ("", "wrote 0 reports"),
        (app.PRIVATE_VERIFIER_MARKER + json.dumps({"setup": "OSError: mount"}), "setup failed"),
    ],
)
def test_a_missing_or_failed_private_verifier_is_never_graded(tmp_path, stdout, message):
    cfg = _make_instance_config(tmp_path)
    with pytest.raises(RuntimeError, match=message):
        app._write_private_outputs(stdout, cfg)
    assert not (cfg.verifier_dir / "reward.txt").exists()
