# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A command's status and output stay the command's through real Apptainer exec."""

import os
import shlex
import signal

import pytest

from nemo_gym.sandbox.providers.apptainer.provider import SANDBOX_RUNTIME_RETURN_CODE, ApptainerProvider
from nemo_gym.sandbox.providers.base import SandboxSpec


pytestmark = [pytest.mark.integration, pytest.mark.sandbox, pytest.mark.skipduringci]
_IMAGE_ENV = "NEMO_GYM_APPTAINER_TEST_IMAGE"


async def test_command_text_and_signals_are_not_apptainer_failures() -> None:
    image = os.environ.get(_IMAGE_ENV)
    if not image:
        pytest.skip(f"set {_IMAGE_ENV} for the real Apptainer exec-status test")
    provider = ApptainerProvider(create={"extra_start_args": ["--fakeroot"]})
    handle = await provider.create(SandboxSpec(image=image))
    try:
        quoted = shlex.quote("echo 'Directory does not exist'")
        # The words Apptainer also uses, printed by the command, then its own nonzero exit.
        failed = await provider.exec(handle, f"echo {quoted} >&2; echo 'instance not found' >&2; exit 3")
        assert (failed.return_code, failed.error_type) == (3, None)
        assert "does not exist" in failed.stderr

        # A command that kills its own process (an agent's pkill reaching its harness):
        # Apptainer re-raises the signal, so the status is the signal's.
        killed = await provider.exec(handle, "echo 'Directory does not exist' >&2; kill -TERM $$")
        assert (killed.return_code, killed.error_type) == (-signal.SIGTERM, None)

        # A command printing a FATAL line keeps its status unless Apptainer exits 255 itself.
        printed = await provider.exec(handle, "echo 'FATAL: from the command' >&2; exit 1")
        assert (printed.return_code, printed.error_type) == (1, None)

        # Apptainer's own failure: the working directory cannot be entered.
        missing = await provider.exec(handle, "true", cwd="/nonexistent-working-directory")
        assert (missing.return_code, missing.error_type) == (SANDBOX_RUNTIME_RETURN_CODE, "sandbox")
        assert missing.stderr.rstrip().splitlines()[-1].startswith("FATAL:")
    finally:
        await provider.close(handle)
