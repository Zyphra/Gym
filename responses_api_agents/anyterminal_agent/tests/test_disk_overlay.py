# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_gym.sandbox.providers.apptainer import provider as apptainer
from responses_api_agents.anyterminal_agent.app import _build_provider, _episode_sandbox_resources
from responses_api_agents.anyterminal_agent.tests.test_app import _make_instance_config


def task(tmp_path, **overrides):
    return {
        "task_name": "task",
        "task_dir": str(tmp_path),
        "docker_image": "image",
        "cpus": "4",
        "memory_mb": "8192",
        "storage_mb": "10240",
        "gpus": "0",
        **overrides,
    }


def test_declared_disk_projection_preserves_existing_cpu_and_memory(tmp_path):
    params = _make_instance_config(
        tmp_path,
        problem_info=task(tmp_path),
        agent_overhead_mb=2048,
        sandbox_provider={"apptainer": {"create": {"disk_backed_overlay": True}}},
    )
    actual = _episode_sandbox_resources(params)
    assert actual.cpu == 4
    assert actual.memory_mib == 10240
    assert actual.disk_gib == 10


@pytest.mark.parametrize("storage", [0, -1, 512, 1025, 1366, 1024.5])
def test_disk_projection_never_silently_rounds_declared_storage_up(tmp_path, storage):
    params = _make_instance_config(
        tmp_path,
        problem_info=task(tmp_path, storage_mb=storage),
        sandbox_provider={"apptainer": {"create": {"disk_backed_overlay": True}}},
    )
    with pytest.raises(ValueError, match="storage_mb"):
        _episode_sandbox_resources(params)


def test_legacy_row_without_disk_budget_retains_resource_projection(tmp_path):
    metadata = task(tmp_path)
    metadata.pop("storage_mb")
    params = _make_instance_config(tmp_path, problem_info=metadata)
    assert _episode_sandbox_resources(params).disk_gib is None


@pytest.mark.parametrize("storage", [1366, "invalid", 0, -1, 1024.5, {}, None])
@pytest.mark.parametrize("create", [{}, {"disk_backed_overlay": False}])
def test_default_off_ignores_storage_metadata_and_preserves_original_resources(tmp_path, storage, create):
    params = _make_instance_config(
        tmp_path,
        problem_info=task(tmp_path, storage_mb=storage),
        sandbox_provider={"apptainer": {"create": create}},
    )
    resources = _episode_sandbox_resources(params)
    assert resources.cpu == 4
    assert resources.memory_mib == 10240
    assert resources.disk_gib is None


def test_string_false_cannot_enable_overlay_or_storage_parsing(tmp_path, monkeypatch):
    monkeypatch.setattr(apptainer, "_require_apptainer", lambda _path=None: "/usr/bin/apptainer")
    params = _make_instance_config(
        tmp_path,
        problem_info=task(tmp_path, storage_mb=1366),
        sandbox_provider={"apptainer": {"create": {"disk_backed_overlay": "false"}}},
    )
    assert _episode_sandbox_resources(params).disk_gib is None
    with pytest.raises(ValueError, match="disk_backed_overlay must be a boolean"):
        _build_provider(params)


@pytest.mark.parametrize("enabled", [False, True])
def test_anyterminal_selects_one_writable_backing_without_changing_other_flags(tmp_path, monkeypatch, enabled):
    monkeypatch.setattr(apptainer, "_require_apptainer", lambda _path=None: "/usr/bin/apptainer")
    params = _make_instance_config(
        tmp_path,
        problem_info=task(tmp_path),
        sandbox_provider={
            "apptainer": {
                "create": {
                    "disk_backed_overlay": enabled,
                    "staging_root": str(tmp_path),
                    "extra_start_args": ["--fakeroot", "--writable-tmpfs", "--no-home"],
                }
            }
        },
    )
    provider = _build_provider(params)
    config = provider._create_config
    assert config.disk_backed_overlay is enabled
    assert config.staging_root == str(tmp_path)
    assert ("--writable-tmpfs" in config.extra_start_args) is (not enabled)
    assert "--fakeroot" in config.extra_start_args
    assert "--no-home" in config.extra_start_args
    assert _episode_sandbox_resources(params).memory_mib == 10240
