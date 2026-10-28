# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from nemo_gym.sandbox.episode import run_sandbox_episode, sandbox_episode_resources
from nemo_gym.sandbox.providers.base import SandboxResources


def test_instance_affinity_preserves_real_resource_limits(monkeypatch):
    from core.sandbox import executor

    sandbox = SandboxResources(cpu=8, memory_mib=8192, pids=512)
    original = sandbox_episode_resources(sandbox)
    observed = []

    async def execute(fn, payload, resources, **kwargs):
        observed.append((resources, kwargs))
        return payload

    monkeypatch.setattr(executor, "run_episode", execute)
    for index in range(2):
        assert (
            asyncio.run(
                run_sandbox_episode(
                    lambda value: value,
                    index,
                    sandbox,
                    episode_id=f"sample-{index}",
                    placement_key="shared-instance",
                )
            )
            == index
        )
    for resources, _ in observed:
        assert resources.placement_key == "shared-instance"
        assert resources.memory_bytes == original.memory_bytes
        assert resources.num_cpus == original.num_cpus
        assert resources.pids == original.pids
        assert resources.scheduler_memory_bytes == original.scheduler_memory_bytes
    assert observed[0][1]["episode_id"] != observed[1][1]["episode_id"]
