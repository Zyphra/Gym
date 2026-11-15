# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The portable CPython runs in musl (Alpine), older-glibc and current-glibc task images.

Builds a runtime with the real setup script and a cryptography wheel resolved on the
build host, then runs its interpreter, a child interpreter started from sys.executable,
ssl and cryptography's Rust bindings inside Alpine, Debian 11 (glibc 2.31) and Debian 12
images. Needs Apptainer and network access (python-build-standalone, PyPI and the images).
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "setup_scripts" / "_portable_python.sh"
APPTAINER = os.environ.get("APPTAINER") or shutil.which("apptainer") or shutil.which("singularity")
IMAGES = {
    "alpine": "docker://alpine:3.19",
    "debian11": "docker://debian:11-slim",
    "debian": "docker://debian:12-slim",
}
PROBE = (
    "import ssl, subprocess, sys; "
    "from cryptography.hazmat.bindings._rust import openssl; "
    "child = subprocess.run([sys.executable, '-c', 'import ssl, sys; print(sys.version_info[:2])'], "
    "capture_output=True, text=True, check=True); "
    "print(sys.executable, child.stdout.strip())"
)

needs_containers = pytest.mark.skipif(
    APPTAINER is None or os.environ.get("NEMO_GYM_RUN_CONTAINER_TESTS") != "1",
    reason="needs Apptainer and NEMO_GYM_RUN_CONTAINER_TESTS=1 (downloads a CPython and two images)",
)


@pytest.fixture(scope="module")
def runtime(tmp_path_factory) -> Path:
    deps = tmp_path_factory.mktemp("deps")
    subprocess.run(
        ["bash", "-c", f'source "{SCRIPT}" && install_portable_python && install_python_packages cryptography'],
        env={**os.environ, "DEPS_DIR": str(deps)},
        check=True,
    )
    return deps


@needs_containers
@pytest.mark.parametrize("image", sorted(IMAGES))
def test_interpreter_and_child_run_in_image(runtime: Path, image: str, tmp_path: Path) -> None:
    sif = tmp_path / f"{image}.sif"
    subprocess.run([APPTAINER, "pull", "--force", str(sif), IMAGES[image]], check=True, capture_output=True)
    result = subprocess.run(
        [
            APPTAINER,
            "exec",
            "--cleanenv",
            "--no-home",
            "--bind",
            f"{runtime}:/agent_deps_mount:ro",
            str(sif),
            "/agent_deps_mount/bin/python",
            "-c",
            PROBE,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    executable, child = result.stdout.split()[0], result.stdout.strip()
    assert executable.startswith("/agent_deps_mount/bin/python3.")
    assert "(3," in child
    if image != "debian":
        # Without a glibc as new as the bundled one the launcher must stay sys.executable,
        # so children use the bundled loader too.
        assert not executable.endswith(".real")


@needs_containers
def test_launcher_bundles_the_loader(runtime: Path) -> None:
    assert (runtime / "glibc" / "ld-linux-x86-64.so.2").is_file()
    launcher = os.path.realpath(runtime / "bin" / "python3")
    assert Path(launcher).read_text().startswith("#!/bin/sh")
    assert Path(launcher + ".real").is_file()


def _bundled_minor() -> int:
    version = subprocess.run(["getconf", "GNU_LIBC_VERSION"], capture_output=True, text=True, check=True).stdout
    return int(version.split()[1].split(".")[1])


def _stub(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)


def _install(deps: Path, *entries: str) -> None:
    subprocess.run(
        ["bash", "-c", f'source "{SCRIPT}" && install_glibc_launcher "$@"', "install", *entries],
        env={**os.environ, "DEPS_DIR": str(deps)},
        check=True,
    )


def _launcher(tmp_path: Path) -> Path:
    """Build the real launcher around stub interpreter and loader scripts."""
    deps = tmp_path / "deps"
    (deps / "bin").mkdir(parents=True)
    (deps / "lib").mkdir()
    _stub(deps / "bin" / "python3.13", "echo placeholder")
    (deps / "bin" / "python3").symlink_to("python3.13")
    # install_glibc_launcher copies the host's loader and libraries; only then stub them.
    _install(deps)
    _stub(deps / "bin" / "python3.13.real", "echo system-loader")
    _stub(deps / "glibc" / "ld-linux-x86-64.so.2", "echo bundled-loader")
    return deps / "bin" / "python3.13"


def _loader_version(minor: int) -> str:
    return f"echo 'ld.so (GNU libc) stable release version 2.{minor}.'"


# A loader older than glibc 2.33 has no --version: it tries to load a program of that name.
_PRE_233_LOADER = "echo '--version: cannot open shared object file' >&2; exit 127"


@pytest.mark.parametrize(
    ("loader", "musl", "expected"),
    [
        (_loader_version(99), False, "system-loader"),
        ("same", False, "system-loader"),
        ("older", False, "bundled-loader"),
        (_PRE_233_LOADER, False, "bundled-loader"),
        ("", False, "bundled-loader"),
        (None, False, "bundled-loader"),
        (_loader_version(99), True, "bundled-loader"),
    ],
    ids=[
        "newer-glibc",
        "same-glibc",
        "older-glibc",
        "pre-2.33-glibc",
        "unreadable-loader",
        "no-glibc",
        "gcompat-over-musl",
    ],
)
def test_launcher_uses_image_glibc_only_when_as_new_as_bundled(
    tmp_path: Path, loader: str | None, musl: bool, expected: str
) -> None:
    launcher = _launcher(tmp_path)
    root = tmp_path / "root"
    (root / "lib64").mkdir(parents=True)
    (root / "lib").mkdir()
    if loader is not None:
        body = {"same": _loader_version(_bundled_minor()), "older": _loader_version(_bundled_minor() - 1)}.get(
            loader, loader
        )
        if body:
            _stub(root / "lib64" / "ld-linux-x86-64.so.2", body)
        else:
            (root / "lib64" / "ld-linux-x86-64.so.2").write_text("")
    if musl:
        (root / "lib" / "ld-musl-x86_64.so.1").write_text("")
    result = subprocess.run(
        [str(launcher)], env={**os.environ, "NGSWE_SYSROOT": str(root)}, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == expected


def test_existing_runtime_gains_launchers_for_each_entry(tmp_path: Path) -> None:
    """A prebuilt runtime (Python and Node, no launcher) is converted in place; a rerun keeps the binaries."""
    deps = tmp_path / "runtime"
    (deps / "bin").mkdir(parents=True)
    _stub(deps / "bin" / "python3.14", "echo python")
    (deps / "bin" / "python3").symlink_to("python3.14")
    (deps / "bin" / "python").symlink_to("python3.14")
    _stub(deps / "bin" / "node", "echo node")
    _install(deps, "python3", "node")
    _install(deps, "python3", "node")
    for name, output in (("python3.14", "python"), ("node", "node")):
        assert (deps / "bin" / name).read_text().startswith("#!/bin/sh")
        assert f'--argv0 "$R/bin/{name}"' in (deps / "bin" / name).read_text()
        real = subprocess.run([str(deps / "bin" / f"{name}.real")], capture_output=True, text=True, check=True)
        assert real.stdout.strip() == output
    assert os.readlink(deps / "bin" / "python") == "python3.14"
    assert os.readlink(deps / "bin" / "python3") == "python3.14"
    # The bundled glibc is the host's, so on the host the launcher runs the real entry point directly.
    host = subprocess.run([str(deps / "bin" / "node")], capture_output=True, text=True, check=True)
    assert host.stdout.strip() == "node"


def test_entry_point_outside_runtime_is_refused(tmp_path: Path) -> None:
    deps = tmp_path / "runtime"
    (deps / "bin").mkdir(parents=True)
    outside = tmp_path / "python3.13"
    _stub(outside, "echo outside")
    (deps / "bin" / "python3").symlink_to(outside)
    result = subprocess.run(
        ["bash", "-c", f'source "{SCRIPT}" && install_glibc_launcher'],
        env={**os.environ, "DEPS_DIR": str(deps)},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "resolves outside" in result.stderr
    assert outside.read_text() == "#!/bin/sh\necho outside\n"
