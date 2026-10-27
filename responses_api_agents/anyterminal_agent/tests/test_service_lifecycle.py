# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in benchmark lifecycle projections on real provider/config objects."""
import hashlib

import pytest

from nemo_gym.sandbox.providers.apptainer import provider as apptainer
from responses_api_agents.anyterminal_agent.app import (
    AnyTerminalInstanceConfig,
    _build_provider,
    _episode_sandbox_resources,
    _read_task_meta,
)


@pytest.mark.parametrize("shared", [False, True])
def test_task_pid_namespace_is_owned_by_selected_instance(tmp_path, monkeypatch, shared):
    monkeypatch.setattr(apptainer, "_require_apptainer", lambda _path=None: "/usr/bin/apptainer")
    cfg = AnyTerminalInstanceConfig.model_construct(
        problem_info={"task_name": "example", "cpus": 1, "memory_mb": 512},
        persistent_dir=tmp_path / "episode", verifier_dir=tmp_path / "verifier",
        nemo_gym_root=tmp_path / "gym", agent_deps_dir=tmp_path / "runtime",
        sandbox_provider={"apptainer": {"exec": {"extra_exec_args": ["--pid"]}}},
        apptainer_shared_pid_namespace=shared,
    )
    actual = _build_provider(cfg)
    assert "--pid" not in actual._create_config.extra_start_args
    assert ("--pid" in actual._exec_config.extra_exec_args) is not shared


def test_disk_budget_falls_back_to_real_task_declaration(tmp_path):
    (tmp_path / "task.toml").write_text("[environment]\nstorage_mb = 10240\n")
    metadata = {"cpus": 1, "memory_mb": 2048, **_read_task_meta(tmp_path)}
    cfg = AnyTerminalInstanceConfig.model_construct(
        problem_info=metadata,
        sandbox_provider={"apptainer": {"create": {"disk_backed_overlay": True}}},
    )
    resources = _episode_sandbox_resources(cfg)
    assert resources.disk_gib == 10
    assert resources.memory_mib == 4096


def test_config_network_descriptor_keeps_original_profile_resources(tmp_path, monkeypatch):
    from responses_api_agents.anyterminal_agent.task_network import TaskNetwork

    root = tmp_path / "tools"
    root.mkdir()
    manifest = b'{"schema":1}\n'
    (root / "manifest.json").write_bytes(manifest)
    selected = TaskNetwork(tools_root=root, tools_sha256=hashlib.sha256(manifest).hexdigest())
    monkeypatch.setenv("NEMO_GYM_SANDBOX_PIDS", "128")
    cfg = AnyTerminalInstanceConfig.model_construct(
        problem_info={"cpus": 1, "memory_mb": 512}, task_network=selected,
    )
    resources = _episode_sandbox_resources(cfg)
    assert (resources.cpu, resources.memory_mib, resources.pids) == (1, 2560, 128)
