# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path

import pytest

from nemo_gym.sandbox.providers.apptainer import provider as apptainer
from nemo_gym.sandbox.providers.base import SandboxResources, SandboxSpec


pytestmark = pytest.mark.sandbox


@pytest.fixture
def overlay_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(apptainer, "_require_apptainer", lambda _path=None: "/usr/bin/apptainer")
    monkeypatch.setattr(apptainer, "_require_disk_backing", lambda _path: None)
    provider = apptainer.ApptainerProvider(
        create={"disk_backed_overlay": True, "staging_root": str(tmp_path), "extra_start_args": ["--writable-tmpfs"]},
        probe={"command": None},
    )
    calls = []

    async def run(argv, **kwargs):
        calls.append(list(argv))
        if argv[1:3] == ["overlay", "create"]:
            size_mib = int(argv[argv.index("--size") + 1])
            with Path(argv[-1]).open("wb") as image:
                image.truncate(size_mib * 1024 * 1024)
        return 0, "", ""

    monkeypatch.setattr(provider, "_run", run)
    return provider, calls, run


def spec(disk_gib=2):
    return SandboxSpec(image="/task.sif", resources=SandboxResources(cpu=4, memory_mib=10240, disk_gib=disk_gib))


@pytest.mark.asyncio
async def test_overlay_is_sized_unique_private_and_removed_after_stop(overlay_provider):
    provider, calls, _run = overlay_provider
    first, second = await asyncio.gather(provider.create(spec()), provider.create(spec()))
    create_calls = [argv for argv in calls if argv[1:3] == ["overlay", "create"]]
    overlays = [Path(argv[-1]) for argv in create_calls]
    assert len(set(overlays)) == 2
    assert all(image.stat().st_size == 2 * 1024**3 for image in overlays)
    assert all("--sparse" in argv for argv in create_calls)
    for handle in (first, second):
        start = next(argv for argv in calls if argv[1:3] == ["instance", "start"] and argv[-1] == handle.sandbox_id)
        image = Path(start[start.index("--overlay") + 1])
        assert not image.is_relative_to(handle.raw.staging_dir)
        assert f"{handle.raw.staging_dir}:/sandbox" in start
        assert "--writable-tmpfs" not in start
        assert start[start.index("--cpus") + 1] == "4"
        assert start[start.index("--memory") + 1] == "10240m"
        assert handle.raw.cleanup_dir == image.parent
        await provider.close(handle)
        assert not image.parent.exists()


@pytest.mark.parametrize("budget", [None, 0, -1, 1.5, True])
@pytest.mark.asyncio
async def test_overlay_requires_exact_positive_disk_budget(overlay_provider, budget):
    provider, calls, _run = overlay_provider
    with pytest.raises(apptainer.ApptainerCreateError, match="positive integer resources.disk_gib"):
        await provider.create(spec(budget))
    assert calls == []


@pytest.mark.parametrize("failure", ["overlay", "size", "start", "probe", "cancel"])
@pytest.mark.asyncio
async def test_failed_create_drains_instance_before_removing_owned_overlay(overlay_provider, monkeypatch, failure):
    provider, calls, run = overlay_provider
    if failure == "probe":
        provider._probe = apptainer.ApptainerProbeConfig()

    async def failing_run(argv, **kwargs):
        outcome = await run(argv, **kwargs)
        if argv[1:3] == ["overlay", "create"] and failure == "overlay":
            return 1, "", "mkfs failed"
        if argv[1:3] == ["overlay", "create"] and failure == "size":
            with Path(argv[-1]).open("wb") as image:
                image.truncate(3 * 1024**3)
        if argv[1:3] == ["instance", "start"]:
            if failure == "cancel":
                raise asyncio.CancelledError()
            if failure == "start":
                return 1, "", "instance failed"
        if argv[1] == "exec" and failure == "probe":
            return 1, "", "probe failed"
        return outcome

    monkeypatch.setattr(provider, "_run", failing_run)
    error = (
        asyncio.CancelledError
        if failure == "cancel"
        else (apptainer.ApptainerCreateError, apptainer.ApptainerCreateVerificationError)
    )
    with pytest.raises(error):
        await provider.create(spec())
    image = Path(next(argv[-1] for argv in calls if argv[1:3] == ["overlay", "create"]))
    assert not image.parent.exists()
    assert any(argv[1:3] == ["instance", "stop"] for argv in calls)


@pytest.mark.asyncio
async def test_failed_stop_retains_disk_until_instance_is_drained(overlay_provider, monkeypatch):
    provider, _calls, run = overlay_provider
    handle = await provider.create(spec())

    async def failed_stop(argv, **kwargs):
        if argv[1:3] == ["instance", "stop"]:
            return 1, "", "still running"
        return await run(argv, **kwargs)

    monkeypatch.setattr(provider, "_run", failed_stop)
    with pytest.raises(RuntimeError, match="instance stop failed"):
        await provider.close(handle)
    assert handle.raw.cleanup_dir.exists()
    monkeypatch.setattr(provider, "_run", run)
    await provider.close(handle)
    assert not handle.raw.cleanup_dir.exists()


def test_backing_mount_uses_deepest_covering_filesystem_and_decodes_spaces(tmp_path):
    root = tmp_path / "disk space"
    root.mkdir()
    escaped = str(root).replace(" ", r"\040")
    mounts = f"1 0 0:1 / / rw - tmpfs none rw\n2 1 8:1 / {escaped} rw - ext4 /dev/nvme0n1 rw\n"
    assert apptainer._disk_backing_filesystem(root / "child", mounts) == "ext4"


@pytest.mark.parametrize("filesystem", ["tmpfs", "ramfs", "nfs", "ceph", "overlay"])
def test_disk_mode_refuses_memory_shared_or_unproven_overlay_backing(tmp_path, monkeypatch, filesystem):
    monkeypatch.setattr(apptainer, "_disk_backing_filesystem", lambda _path, _mounts: filesystem)
    with pytest.raises(apptainer.ApptainerCreateError, match="requires a local disk filesystem"):
        apptainer._require_disk_backing(tmp_path)


def test_disk_mode_is_default_off():
    assert apptainer.ApptainerCreateConfig().disk_backed_overlay is False


@pytest.mark.parametrize("invalid", ["false", "true", 0, 1, None])
def test_disk_mode_requires_an_actual_boolean(invalid):
    with pytest.raises(ValueError, match="disk_backed_overlay must be a boolean"):
        apptainer.ApptainerCreateConfig(disk_backed_overlay=invalid)


@pytest.mark.asyncio
async def test_mode_creates_missing_private_parent(overlay_provider, tmp_path):
    provider, _calls, _run = overlay_provider
    root = tmp_path / "new" / "owned"
    provider._create_config = apptainer.ApptainerCreateConfig(disk_backed_overlay=True, staging_root=str(root))
    handle = await provider.create(spec())
    assert handle.raw.cleanup_dir.parent == root
    assert root.stat().st_mode & 0o777 == 0o700
    await provider.close(handle)


@pytest.mark.asyncio
async def test_overlay_storage_cannot_be_exposed_by_default_or_per_instance_bind(overlay_provider, tmp_path):
    provider, calls, _run = overlay_provider
    request = SandboxSpec(
        image="/task.sif",
        resources=SandboxResources(disk_gib=2),
        provider_options={"binds": [f"{tmp_path}:/host-data"]},
    )
    with pytest.raises(apptainer.ApptainerCreateError, match="must not be exposed"):
        await provider.create(request)
    assert calls == []


@pytest.mark.asyncio
async def test_overlay_fakeroot_creation_matches_instance_mode(overlay_provider):
    provider, calls, _run = overlay_provider
    provider._create_config = apptainer.ApptainerCreateConfig(
        disk_backed_overlay=True,
        staging_root=provider._create_config.staging_root,
        extra_start_args=["--fakeroot"],
    )
    handle = await provider.create(spec())
    overlay = next(argv for argv in calls if argv[1:3] == ["overlay", "create"])
    assert "--fakeroot" in overlay
    await provider.close(handle)
