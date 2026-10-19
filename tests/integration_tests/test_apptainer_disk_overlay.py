# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real disk/RAM containment proof; run only on an allocated worker."""

import asyncio
import json
import os
import shlex
from pathlib import Path

import pytest

from nemo_gym.sandbox.providers.apptainer.provider import ApptainerProvider
from nemo_gym.sandbox.providers.base import SandboxResources, SandboxSpec


pytestmark = [pytest.mark.integration, pytest.mark.sandbox, pytest.mark.skipduringci]


def counters(path):
    return {key: int(value) for key, value in (line.split() for line in path.read_text().splitlines())}


async def test_real_disk_budget_is_independent_of_ram_and_cleanup_is_owned():
    names = ("NEMO_GYM_APPTAINER_TEST_IMAGE", "NEMO_GYM_DELEGATED_CGROUP_TEST_ROOT", "NEMO_GYM_DISK_OVERLAY_TEST_ROOT")
    if not os.environ.get("SLURM_JOB_ID") or not all(os.environ.get(name) for name in names):
        pytest.skip("requires an allocated worker, native test image, delegated cgroup and local disk root")
    image, cgroup_root, storage_root = (os.environ[name] for name in names)
    provider = ApptainerProvider(
        create={
            "disk_backed_overlay": True,
            "staging_root": storage_root,
            "cgroup_root": cgroup_root,
            "extra_start_args": ["--fakeroot", "--no-home", "--no-mount", "tmp,home"],
        },
        exec={"timeout_grace_s": 1},
    )
    handle = await provider.create(
        SandboxSpec(
            image=image,
            resources=SandboxResources(cpu=1, memory_mib=512, disk_gib=1, pids=128),
        )
    )
    group = handle.raw.cgroup.path
    owned = handle.raw.cleanup_dir
    receipt = {"image": image, "owned_storage": str(owned), "sandbox_cgroup": str(group)}
    try:
        assert (group / "memory.max").read_text().strip() == str(512 * 1024**2)
        assert (group / "cpu.max").read_text().strip() == "100000 100000"
        hidden = await provider.exec(handle, "test ! -e " + shlex.quote(str(owned / "writable-overlay.img")))
        assert hidden.return_code == 0, hidden
        written = await provider.exec(
            handle, "dd if=/dev/zero of=/disk-probe bs=1M count=768 conv=fsync", timeout_s=180
        )
        assert written.return_code == 0, written
        stats = counters(group / "memory.stat")
        events = counters(group / "memory.events")
        assert stats["shmem"] < 64 * 1024**2, stats
        assert events["oom_kill"] == 0, events
        full = await provider.exec(
            handle,
            "dd if=/dev/zero of=/disk-probe bs=1M count=512 oflag=append conv=notrunc,fsync",
            timeout_s=180,
        )
        assert full.return_code != 0 and "No space left on device" in (full.stderr or ""), full
        alive = await provider.exec(handle, "printf harness-alive", timeout_s=30)
        assert alive.return_code == 0 and alive.stdout == "harness-alive", alive
        assert counters(group / "memory.events")["oom_kill"] == 0
        receipt.update(
            {
                "disk_write_bytes": 768 * 1024**2,
                "after_write_memory_stat": stats,
                "after_write_memory_events": events,
                "disk_full_exit": full.return_code,
                "disk_full_stderr": full.stderr,
                "harness_alive": True,
            }
        )
    finally:
        await provider.close(handle)
    assert not owned.exists() and not group.exists()

    # Cancellation after create must drain the instance before releasing backing.
    handle = await provider.create(
        SandboxSpec(image=image, resources=SandboxResources(cpu=1, memory_mib=512, disk_gib=1))
    )
    owned, group = handle.raw.cleanup_dir, handle.raw.cgroup.path
    closing = asyncio.create_task(provider.close(handle))
    await asyncio.sleep(0)
    closing.cancel()
    try:
        await closing
    except asyncio.CancelledError:
        pass
    assert not owned.exists() and not group.exists()
    receipt.update({"cleanup_passed": True, "cancelled_close_cleanup_passed": True})
    if destination := os.environ.get("NEMO_GYM_DISK_OVERLAY_TEST_RECEIPT"):
        Path(destination).write_text(json.dumps(receipt, indent=2) + "\n")
