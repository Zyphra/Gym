# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A process left in the sandbox cannot reach the directory the reward is read from."""

import base64
import hashlib
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


def _run_helper(private: Path, command: str, tests: Path | None = None, *options: str) -> subprocess.CompletedProcess:
    """The real helper as namespace root, with its verifier path at ``private`` and tests at ``tests``."""
    driver = textwrap.dedent(
        """
        import runpy, sys
        module = runpy.run_path(sys.argv[1], run_name="private_verifier")
        module["main"].__globals__["VERIFIER_DIR"] = sys.argv[2]
        module["main"].__globals__["TESTS_DIR"] = sys.argv[3]
        raise SystemExit(module["main"](sys.argv[4:]))
        """
    )
    tests_dir = str(tests or private.parent / "tests")
    return subprocess.run(
        [
            "unshare",
            "-rm",
            sys.executable,
            "-I",
            "-S",
            "-c",
            driver,
            str(HELPER),
            str(private),
            tests_dir,
            *options,
            command,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _helper_report(result: subprocess.CompletedProcess) -> dict:
    (line,) = [line for line in result.stdout.splitlines() if line.startswith(private_verifier.MARKER)]
    return json.loads(line[len(private_verifier.MARKER) :])


def _staged_task(tmp_path: Path) -> tuple[Path, Path, str]:
    """A task's pristine tests, their staged copy and the host's manifest digest for it."""
    source = tmp_path / "task" / "tests"
    (source / "data").mkdir(parents=True)
    (source / "test.sh").write_text('echo "$(cat "$(dirname "$0")/data/expected")" > "$1/reward.txt"\n')
    (source / "test.sh").chmod(0o755)
    (source / "data" / "expected").write_text("0")
    (source / "link").symlink_to("data/expected")
    staged = tmp_path / "staging" / "tests"
    shutil.copytree(source, staged, symlinks=True)
    manifest = app.tests_manifest(source)
    (tmp_path / "staging" / "tests-manifest.json").write_bytes(manifest)
    return source, staged, hashlib.sha256(manifest).hexdigest()


@pytest.mark.skipif(not _namespaces_available(), reason="needs unprivileged user and mount namespaces")
def test_test_sh_runs_the_host_tests_whatever_was_staged(tmp_path):
    _source, staged, digest = _staged_task(tmp_path)
    (staged / "planted.py").write_text("raise SystemExit(0)")  # staged, but not the task's
    shared, tests = tmp_path / "logs-verifier", tmp_path / "tests"
    shared.mkdir()
    tests.symlink_to(staged)  # what the staged tree used to be reached through
    # A left-behind writer rewrites the staged test.sh for as long as the verifier runs.
    writer = subprocess.Popen(
        ["sh", "-c", f'while :; do printf "echo 1 > \\"\\$1/reward.txt\\"\\n" > "{staged}/test.sh"; sleep 0.01; done']
    )
    try:
        time.sleep(0.1)
        result = _run_helper(
            shared,
            f'sleep 0.3; ls -A "{tests}" > "{shared}/listing"; bash "{tests}/test.sh" "{shared}"',
            tests,
            "--tests",
            str(staged),
            digest,
        )
        report = _helper_report(result)
        assert report["setup"] and "differs from the task source" in report["setup"]  # refused, never run
    finally:
        writer.kill()
        writer.wait()
    shutil.copy2(tmp_path / "task" / "tests" / "test.sh", staged / "test.sh")
    result = _run_helper(
        shared,
        f'ls -A "{tests}" > "{shared}/listing"; bash "{tests}/test.sh" "{shared}"',
        tests,
        "--tests",
        str(staged),
        digest,
    )
    report = _helper_report(result)
    assert result.returncode == 0 and report["exit_code"] == 0 and report["integrity"] == []
    files = {entry["path"]: base64.b64decode(entry["data"]) for entry in report["files"]}
    assert files["reward.txt"] == b"0\n"
    assert files["listing"].split() == [b"data", b"link", b"test.sh"]  # planted.py never reached the grader
    (staged / "test.sh").write_text("echo 1")
    (tmp_path / "staging" / "tests-manifest.json").write_bytes(app.tests_manifest(staged))
    result = _run_helper(shared, "true", tests, "--tests", str(staged), digest)
    assert "manifest differs" in _helper_report(result)["setup"]  # a restaged manifest is not the host's


@pytest.mark.skipif(not _namespaces_available(), reason="needs unprivileged user and mount namespaces")
def test_a_private_directory_detached_during_grading_is_reported(tmp_path):
    _source, staged, digest = _staged_task(tmp_path)
    shared, tests = tmp_path / "logs" / "verifier", tmp_path / "tests"
    shared.mkdir(parents=True)
    # Renaming a mount point in the sandbox's namespace detaches the verifier's private mount
    # at it; the swapped-in directory then serves the tests and receives the reward.
    swap = (
        f'(sleep 0.2; mv "{tests}" "{tests}.old"; mkdir "{tests}"; echo "echo 1 > \\"\\$1/reward.txt\\"" > "{tests}/test.sh"; '
        f'mv "{tmp_path}/logs" "{tmp_path}/logs.old"; mkdir -p "{shared}") &'
    )
    subprocess.run(["sh", "-c", swap])
    result = _run_helper(
        shared, f'sleep 0.6; bash "{tests}/test.sh" "{shared}"', tests, "--tests", str(staged), digest
    )
    report = _helper_report(result)
    assert result.returncode == 0 and report["exit_code"] == 0
    assert sorted(report["integrity"]) == [
        f"private {shared} was detached during the command",
        f"private {tests} was detached during the command",
    ]
    stdout = private_verifier.MARKER + json.dumps(report) + "\n"
    with pytest.raises(VerifierIntegrityRefused, match="detached"):
        app._write_private_outputs(stdout, 0, _make_instance_config(tmp_path / "episode"))


def test_the_tests_manifest_is_the_task_source_plus_host_held_files(tmp_path):
    source, _staged, _digest = _staged_task(tmp_path)
    manifest = json.loads(app.tests_manifest(source, {"ext.so": (b"\x7fELF", 0o444)}))
    assert manifest["directories"] == ["data"]
    assert manifest["links"] == [{"path": "link", "target": "data/expected"}]
    files = {entry["path"]: entry for entry in manifest["files"]}
    assert set(files) == {"data/expected", "test.sh", "ext.so"}
    assert files["test.sh"]["mode"] == 0o755 and files["ext.so"]["mode"] == 0o444
    assert files["ext.so"]["sha256"] == hashlib.sha256(b"\x7fELF").hexdigest()


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
    assert report["exit_code"] == 0  # the command's own status; the helper exits 0 after its report
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
    receipt = app._write_private_outputs(stdout, 0, cfg)
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
        app._write_private_outputs(stdout, 0, cfg)


@pytest.mark.parametrize("path", ["../reward.txt", "/logs/verifier/reward.txt", "a//b", ""])
def test_a_report_cannot_name_a_path_outside_the_verifier_directory(tmp_path, path):
    cfg = _make_instance_config(tmp_path)
    entry = {"path": path, "size": 1, "mode": 0o644, "sha256": "0" * 64, "data": "MQ=="}
    stdout = app.PRIVATE_VERIFIER_MARKER + json.dumps({"setup": None, "files": [entry], "links": []})
    with pytest.raises(VerifierIntegrityRefused, match="not relative"):
        app._write_private_outputs(stdout, 0, cfg)


@pytest.mark.parametrize(
    "stdout,message",
    [
        ("", "wrote no report"),
        (app.PRIVATE_VERIFIER_MARKER + json.dumps({"setup": "OSError: mount"}), "setup failed"),
    ],
)
def test_a_missing_or_failed_private_verifier_is_never_graded(tmp_path, stdout, message):
    cfg = _make_instance_config(tmp_path)
    with pytest.raises(RuntimeError, match=message):
        app._write_private_outputs(stdout, 0, cfg)
    assert not (cfg.verifier_dir / "reward.txt").exists()


def test_a_report_from_a_killed_or_shadowed_helper_is_never_graded(tmp_path, monkeypatch):
    cfg = _make_instance_config(tmp_path)
    forged = _report(cfg, monkeypatch, {"reward.txt": b"1"})
    # A sandbox process printed a report on the helper's stdout and killed it.
    with pytest.raises(RuntimeError, match="exited 137"):
        app._write_private_outputs(forged, 137, cfg)
    # Or printed one beside the helper's own.
    with pytest.raises(VerifierIntegrityRefused, match="2 reports"):
        app._write_private_outputs(forged + forged.splitlines()[-1] + "\n", 0, cfg)
    assert not (cfg.verifier_dir / "reward.txt").exists()


def test_grader_outputs_live_beside_the_episode_and_are_published_after_it(tmp_path):
    persistent = tmp_path / "episode"
    cfg = app.AnyTerminalInstanceConfig.model_construct(
        persistent_dir=persistent, verifier_dir=persistent / "verifier"
    )
    trusted = cfg.host_only_dir("verifier")
    # /trajectories_mount is the episode directory: nothing the host grades from is below it.
    assert trusted.parent == tmp_path and app.interim_snapshot(cfg, 3).parent.parent == tmp_path
    cfg.verifier_dir.mkdir(parents=True)
    (cfg.verifier_dir / "reward.txt").write_text("1")  # a sandbox process uncovered and wrote this
    trusted.mkdir()
    (trusted / "reward.txt").write_text("0")
    (persistent / "interim_verifier").mkdir()  # empty: nothing written there
    interim = app.interim_snapshot(cfg, 0)
    interim.mkdir(parents=True)
    (interim / "reward.txt").write_text("0.5")

    app._publish_host_only_outputs(cfg)

    assert (persistent / "verifier" / "reward.txt").read_text() == "0"
    assert (persistent / "verifier.sandbox-written" / "reward.txt").read_text() == "1"
    assert (persistent / "interim_verifier" / "phase-0000" / "reward.txt").read_text() == "0.5"
    assert not (persistent / "interim_verifier.sandbox-written").exists()
    assert not trusted.exists() and not interim.parent.exists()
