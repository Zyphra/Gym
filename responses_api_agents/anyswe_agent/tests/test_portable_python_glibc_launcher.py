# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The portable CPython runs in glibc and musl (Alpine) task images.

Builds a runtime with the real setup script, then runs its interpreter, a child
interpreter started from sys.executable, and ssl inside an Alpine and a Debian image.
Needs Apptainer and network access (python-build-standalone and the two images).
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "setup_scripts" / "_portable_python.sh"
APPTAINER = os.environ.get("APPTAINER") or shutil.which("apptainer") or shutil.which("singularity")
IMAGES = {"alpine": "docker://alpine:3.19", "debian": "docker://debian:12-slim"}
PROBE = (
    "import ssl, subprocess, sys; "
    "child = subprocess.run([sys.executable, '-c', 'import ssl, sys; print(sys.version_info[:2])'], "
    "capture_output=True, text=True, check=True); "
    "print(sys.executable, child.stdout.strip())"
)

pytestmark = pytest.mark.skipif(
    APPTAINER is None or os.environ.get("NEMO_GYM_RUN_CONTAINER_TESTS") != "1",
    reason="needs Apptainer and NEMO_GYM_RUN_CONTAINER_TESTS=1 (downloads a CPython and two images)",
)


@pytest.fixture(scope="module")
def runtime(tmp_path_factory) -> Path:
    deps = tmp_path_factory.mktemp("deps")
    subprocess.run(
        ["bash", "-c", f'source "{SCRIPT}" && install_portable_python'],
        env={**os.environ, "DEPS_DIR": str(deps)},
        check=True,
    )
    return deps


@pytest.mark.parametrize("image", sorted(IMAGES))
def test_interpreter_and_child_run_in_image(runtime: Path, image: str, tmp_path: Path) -> None:
    sif = tmp_path / f"{image}.sif"
    subprocess.run([APPTAINER, "pull", "--force", str(sif), IMAGES[image]], check=True, capture_output=True)
    result = subprocess.run(
        [APPTAINER, "exec", "--cleanenv", "--no-home", "--bind", f"{runtime}:/agent_deps_mount:ro",
         str(sif), "/agent_deps_mount/bin/python", "-c", PROBE],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    executable, child = result.stdout.split()[0], result.stdout.strip()
    assert executable.startswith("/agent_deps_mount/bin/python3.")
    assert "(3," in child
    if image == "alpine":
        # Without glibc the launcher must stay sys.executable so children use the loader.
        assert not executable.endswith(".real")


def test_launcher_bundles_the_loader(runtime: Path) -> None:
    assert (runtime / "glibc" / "ld-linux-x86-64.so.2").is_file()
    launcher = os.path.realpath(runtime / "bin" / "python3")
    assert Path(launcher).read_text().startswith("#!/bin/sh")
    assert Path(launcher + ".real").is_file()
