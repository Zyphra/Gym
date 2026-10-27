# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemo_gym.sandbox.providers import registry
from nemo_gym.sandbox.providers.apptainer import ApptainerProvider
from nemo_gym.sandbox.providers.apptainer import provider as apptainer_provider
from responses_api_agents.anyterminal_agent.app import (
    AnyTerminalInstanceConfig,
    _build_provider,
    _episode_sandbox_resources,
)
from responses_api_agents.anyterminal_agent.task_network import TaskNetwork


def network(tmp_path):
    root = tmp_path / "network"
    root.mkdir()
    data = b'{"schema":1}\n'
    (root / "manifest.json").write_bytes(data)
    return TaskNetwork(tools_root=root, tools_sha256=hashlib.sha256(data).hexdigest())


def params(tmp_path, **overrides):
    return AnyTerminalInstanceConfig.model_construct(**({
        "problem_info": {"cpus": 1, "memory_mb": 4096},
        "persistent_dir": tmp_path / "episode",
        "verifier_dir": tmp_path / "verifier",
        "agent_deps_dir": tmp_path / "agent",
        "nemo_gym_root": tmp_path / "gym",
        "sandbox_provider": {"apptainer": {}},
    } | overrides))


@pytest.mark.parametrize("change", ["relative", "symlink", "missing", "digest", "mode", "unknown"])
def test_network_descriptor_rejects_unadmitted_inputs(tmp_path, change):
    selected = network(tmp_path)
    values = selected.model_dump()
    if change == "relative":
        values["tools_root"] = Path("network")
    elif change == "symlink":
        linked = tmp_path / "linked"
        linked.symlink_to(selected.tools_root, target_is_directory=True)
        values["tools_root"] = linked
    elif change == "missing":
        (selected.tools_root / "manifest.json").unlink()
    elif change == "digest":
        values["tools_sha256"] = "0" * 64
    elif change == "mode":
        values["mode"] = "host"
    else:
        values["command"] = "arbitrary"
    with pytest.raises(ValidationError):
        TaskNetwork(**values)


def test_private_network_requires_original_profile_pid_limit(tmp_path, monkeypatch):
    selected = network(tmp_path)
    monkeypatch.setenv("NEMO_GYM_SANDBOX_PIDS", "2048")
    resources = _episode_sandbox_resources(params(tmp_path, task_network=selected))
    assert (resources.cpu, resources.memory_mib, resources.pids) == (1, 6144, 2048)
    monkeypatch.delenv("NEMO_GYM_SANDBOX_PIDS")
    with pytest.raises(ValueError, match="profile PID limit"):
        _episode_sandbox_resources(params(tmp_path, task_network=selected))
    assert _episode_sandbox_resources(params(tmp_path)).pids is None


def test_factory_preserves_instance_mounts_and_default_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(apptainer_provider, "_require_apptainer", lambda _bin_path=None: "/usr/bin/apptainer")

    class ObservedApptainer(ApptainerProvider):
        def __init__(self, *, tools_root, tools_sha256, network_required, **kwargs):
            super().__init__(**kwargs)
            self.selection = (tools_root, tools_sha256, network_required)

    monkeypatch.setattr(registry, "_PROVIDER_REGISTRY", {})
    registry.register_provider("apptainer_services", ObservedApptainer)
    selected = network(tmp_path)
    cfg = params(tmp_path, task_network=selected)
    actual = _build_provider(cfg)
    assert isinstance(actual, ObservedApptainer)
    assert actual.selection == (str(selected.tools_root), selected.tools_sha256, True)
    for directory, target in ((cfg.persistent_dir, "/trajectories_mount"),
                              (cfg.verifier_dir, "/logs/verifier"),
                              (cfg.agent_deps_dir, "/agent_deps_mount:ro")):
        assert f"{directory}:{target}" in actual._exec_config.default_binds
    assert type(_build_provider(params(tmp_path))) is ApptainerProvider


def test_private_network_rejects_remote_provider(tmp_path):
    with pytest.raises(ValueError, match="requires Apptainer"):
        _build_provider(params(tmp_path, task_network=network(tmp_path), sandbox_provider={"docker": {}}))
