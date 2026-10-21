# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real CLI subprocess capture/custody; sandbox image admission is separate."""

import asyncio
import os
import sys
from pathlib import Path

import pytest

from nemo_gym.sandbox.providers.apptainer import provider as apptainer
from nemo_gym.sandbox.providers.enroot import provider as enroot
from nemo_gym.sandbox.providers.utils import CapturedTimeout


def _provider(tmp_path, monkeypatch):
    monkeypatch.setattr(enroot, "_require_enroot", lambda: sys.executable)
    return enroot.EnrootProvider(create={"base_dir": str(tmp_path / "home")}, exec={"max_output_bytes": 128})


def test_timeout_retains_bounded_original_streams_and_reaps_process(tmp_path, monkeypatch):
    provider = _provider(tmp_path, monkeypatch)
    pidfile = tmp_path / "pid"
    script = (
        "import os,sys,time;from pathlib import Path;"
        "Path(sys.argv[1]).write_text(str(os.getpid()));"
        "os.write(1,b'O'*16384+b'\\xff');os.write(2,b'E'*8192);time.sleep(30)"
    )
    facts = {}
    with pytest.raises(CapturedTimeout) as caught:
        asyncio.run(provider._run([sys.executable, "-c", script, str(pidfile)], timeout_s=0.5, capture_facts=facts))
    error = caught.value
    assert error.stdout == "O" * 128
    assert error.stderr == "E" * 128
    assert error.facts == facts
    assert facts["stdout_bytes"] == 16385
    assert facts["stderr_bytes"] == 8192
    assert facts["stdout_truncated"] and facts["stderr_truncated"]
    assert not facts["capture_complete"]
    pid = int(pidfile.read_text())
    assert not Path(f"/proc/{pid}").exists()
    with pytest.raises(ProcessLookupError):
        os.killpg(pid, 0)


def test_success_keeps_stdin_byte_counts_and_replacement_decoding(tmp_path, monkeypatch):
    provider = _provider(tmp_path, monkeypatch)
    script = "import os,sys;data=sys.stdin.buffer.read();os.write(1,b'\\xff'+data);os.write(2,b'error')"
    facts = {}
    code, out, err = asyncio.run(
        provider._run([sys.executable, "-c", script], stdin=b"x" * 200, timeout_s=3, capture_facts=facts)
    )
    assert code == 0 and out == "\ufffd" + "x" * 127 and err == "error"
    assert facts == {
        "stdout_bytes": 201,
        "stderr_bytes": 5,
        "stdout_truncated": True,
        "stderr_truncated": False,
        "capture_complete": True,
    }


def test_cancellation_reaps_owned_process_without_leaving_reader_tasks(tmp_path, monkeypatch):
    provider = _provider(tmp_path, monkeypatch)
    pidfile = tmp_path / "pid"
    script = (
        "import os,sys,time;from pathlib import Path;Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(30)"
    )

    async def exercise():
        task = asyncio.create_task(
            provider._run([sys.executable, "-c", script, str(pidfile)], timeout_s=30, capture_facts={})
        )
        for _ in range(100):
            if pidfile.exists():
                break
            await asyncio.sleep(0.01)
        assert pidfile.exists()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not Path(f"/proc/{int(pidfile.read_text())}").exists()
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]

    asyncio.run(exercise())


def test_apptainer_uses_same_capture_owner_after_move(tmp_path, monkeypatch):
    monkeypatch.setattr(apptainer, "_require_apptainer", lambda _: sys.executable)
    monkeypatch.setenv("APPTAINER_CONFIGDIR", str(tmp_path))
    provider = apptainer.ApptainerProvider(
        create={"apply_resource_limits": False}, exec={"max_output_bytes": 128, "timeout_grace_s": 0}
    )
    facts = {}
    with pytest.raises(CapturedTimeout) as caught:
        asyncio.run(
            provider._run(
                [sys.executable, "-c", "import os,time;os.write(1,b'partial');time.sleep(30)"],
                timeout_s=0.5,
                capture_facts=facts,
            )
        )
    assert caught.value.stdout == "partial"
    assert caught.value.facts == facts
