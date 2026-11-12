# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Apptainer client environment is explicit: a worker-sized environment cannot hang a fakeroot exec."""

import os

import pytest

from nemo_gym.sandbox.providers.apptainer.provider import ApptainerProvider
from nemo_gym.sandbox.providers.base import SandboxSpec


pytestmark = [pytest.mark.integration, pytest.mark.sandbox, pytest.mark.skipduringci]
_IMAGE_ENV = "NEMO_GYM_APPTAINER_TEST_IMAGE"


def _live_shaped_environment() -> dict[str, str]:
    """A Ray episode worker's environment: 592 variables, 250 KB, one variable of 89.5 KB."""
    environment = dict(os.environ)
    environment["VERL_RUN_RECORD_JSON"] = "x" * 89_500
    environment["PYXIS_MOUNTS"] = "m" * 43_394
    for index in range(588):
        environment[f"TASK_VALUE_{index}"] = "v" * 150
    return environment


async def test_fakeroot_exec_does_not_depend_on_the_worker_environment() -> None:
    """Under a root-mapped user namespace (the Hero runtime image inside pyxis) Apptainer reads the
    client's environment from ``fakeroot-sysv env`` and stops at a line over 64 KiB, so a client
    started with the worker's whole environment hangs; the allow-listed one does not."""
    image = os.environ.get(_IMAGE_ENV)
    if not image:
        pytest.skip(f"set {_IMAGE_ENV} for the real Apptainer client-environment test")
    live = _live_shaped_environment()
    provider = ApptainerProvider()
    handle = await provider.create(SandboxSpec(image=image))
    try:
        assert max(len(value) for value in provider._subprocess_env.values()) < 65_536
        assert len(provider._subprocess_env) < 50
        ok = await provider.exec(handle, "id -u", user="root", timeout_s=60)
        assert ok.error_type != "timeout", ok.stderr
        # A host that cannot enter the user namespace fails the exec quickly; it never hangs.

        # What the unpatched provider did: hand the client the worker's environment.
        patched = provider._subprocess_env
        provider._subprocess_env = live
        try:
            hung = await provider.exec(handle, "id -u", user="root", timeout_s=20)
        finally:
            provider._subprocess_env = patched
        if hung.error_type != "timeout":
            pytest.skip("this host does not run the fakeroot environment capture (no root-mapped namespace)")
    finally:
        await provider.close(handle)
