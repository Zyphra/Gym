# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted, episode-local Python extension preparation for Terminal verifiers.

Standard library only: the same file runs in stock and separate verifier images.
The caller owns the lifecycle boundary; this module owns source/output validation,
runtime attestation, and the initial module import before pytest.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Callable, TypedDict


SETUP_FAILED = 125
RECEIPT = "extension-preparation.json"


class ExtensionPreparationPlan(TypedDict):
    workspace: str
    command: str
    sources: dict[str, str]
    module: str
    output: str
    max_bytes: int
    runtime: dict
    declaration_sha256: str
    task_manifest_sha256: str
    sif_sha256: str


class ExtensionPreparationResult(TypedDict):
    ok: bool
    declaration_sha256: str
    task_manifest_sha256: str
    sif_sha256: str
    sources: dict[str, str]
    module: str
    output: str
    output_sha256: str
    output_size: int
    runtime: dict
    runtime_manifest_sha256: str
    runtime_root: str


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def site_tree_sha256(directory: str | Path) -> str:
    root = Path(directory)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts:
            continue
        if path.is_symlink():
            raise ValueError("prepared package environment contains a symlink")
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def _regular(path: Path, *, size: int | None = None) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or (size is not None and info.st_size != size):
        raise ValueError(f"prepared extension path is not the declared regular file: {path}")
    return info


def _plan(spec_text: str) -> ExtensionPreparationPlan:
    spec = json.loads(spec_text)
    if set(spec) != {
        "workspace",
        "command",
        "sources",
        "output",
        "module",
        "max_bytes",
        "runtime",
        "declaration_sha256",
        "task_manifest_sha256",
        "sif_sha256",
    }:
        raise ValueError("extension preparation specification is incomplete")
    for key in ("declaration_sha256", "task_manifest_sha256", "sif_sha256"):
        if not isinstance(spec[key], str) or not re.fullmatch(r"[0-9a-f]{64}", spec[key]):
            raise ValueError(f"extension {key} is invalid")
    if not isinstance(spec["module"], str) or not spec["module"].isidentifier() or spec["module"].startswith("_"):
        raise ValueError("extension module is invalid")
    if not isinstance(spec["max_bytes"], int) or not 0 < spec["max_bytes"] <= 64 * 1024 * 1024:
        raise ValueError("extension output bound is invalid")
    if (
        not isinstance(spec["output"], str)
        or spec["output"] != "tests/" + spec["module"] + spec["runtime"]["image_python"]["extension_suffix"]
    ):
        raise ValueError("extension output path is unsafe")
    if not isinstance(spec["command"], str) or len(spec["command"]) > 16384:
        raise ValueError("extension build command is invalid")
    sources = spec["sources"]
    if not isinstance(sources, dict) or "tests/setup.py" not in sources or not 2 <= len(sources) <= 16:
        raise ValueError("extension source inventory is incomplete")
    for relative, expected in sources.items():
        if (
            not isinstance(relative, str)
            or not relative.startswith("tests/")
            or "/" in relative.removeprefix("tests/")
            or not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
        ):
            raise ValueError("extension source declaration is unsafe")
    return spec


def prepare_extension(
    spec_text: str,
    *,
    source_tests: Path,
    output_dir: Path | None,
    receipt_dir: Path,
    run_phase: Callable[[str, Path], None],
) -> ExtensionPreparationResult:
    """Build only hash-bound sources and stage one validated result."""
    spec = _plan(spec_text)
    root = Path(tempfile.mkdtemp(prefix="verdict-preparation-", dir="/tmp"))
    build = root / "build"
    build.mkdir(mode=0o700)
    for relative, expected in spec["sources"].items():
        source = source_tests / relative.removeprefix("tests/")
        _regular(source)
        if sha256(source) != expected:
            raise ValueError(f"extension source differs: {relative}")
        target = build / source.name
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        target.chmod(0o444)
        if sha256(target) != expected:
            raise ValueError(f"extension copied source differs: {relative}")
    run_phase(spec["command"], root)
    manifest_path = root / "runtime.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = spec["runtime"]["image_python"]
    for key in ("executable_sha256", "version", "soabi", "extension_suffix", "uv_version"):
        if manifest.get(key) != runtime[key]:
            raise ValueError(f"extension runtime {key} differs")
    python = Path(manifest["python"])
    if not python.is_relative_to(root) or not python.is_file():
        raise ValueError("extension package environment differs")
    pins = {item.split("==")[0].replace("_", "-").lower(): item.split("==")[1] for item in runtime["requirements"]}
    if manifest.get("packages") != pins:
        raise ValueError("extension package environment differs")
    site = Path(manifest["site_packages"])
    if not site.is_relative_to(root) or site_tree_sha256(site) != manifest.get("site_tree_sha256"):
        raise ValueError("prepared package tree differs")
    built = build / spec["output"].removeprefix("tests/")
    info = _regular(built)
    if not 0 < info.st_size <= spec["max_bytes"] or sha256(built) != manifest.get("output_sha256"):
        raise ValueError("built extension output differs")
    if output_dir is None:
        output_dir = root / "prepared"
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    receipt_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    staged = output_dir / built.name
    with built.open("rb") as src, staged.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    staged.chmod(0o444)
    if sha256(staged) != manifest["output_sha256"]:
        raise ValueError("staged extension differs")
    result: ExtensionPreparationResult = {
        "ok": True,
        "declaration_sha256": spec["declaration_sha256"],
        "task_manifest_sha256": spec["task_manifest_sha256"],
        "sif_sha256": spec["sif_sha256"],
        "sources": spec["sources"],
        "module": spec["module"],
        "output": spec["output"],
        "output_sha256": manifest["output_sha256"],
        "output_size": info.st_size,
        "runtime": manifest,
        "runtime_manifest_sha256": sha256(manifest_path),
        "runtime_root": str(root),
    }
    with (receipt_dir / RECEIPT).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True)
    return result


def attest_extension(
    declaration_sha256: str,
    *,
    receipt_dir: Path,
    extension_dir: Path | None,
) -> dict:
    """Recheck derived bytes and interpreter immediately before tests."""
    path = receipt_dir / RECEIPT
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "r", encoding="utf-8") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("preparation receipt is not regular")
        receipt = json.load(stream)
    if not receipt.get("ok") or receipt.get("declaration_sha256") != declaration_sha256:
        raise ValueError("preparation declaration differs")
    output = receipt["output"]
    if output != "tests/" + receipt["module"] + receipt["runtime"]["extension_suffix"]:
        raise ValueError("preparation output path is unsafe")
    root = Path(receipt["runtime_root"])
    if not root.is_absolute() or root.resolve() != root or not root.is_dir():
        raise ValueError("prepared runtime root is not a canonical directory")
    target = (root / "prepared" if extension_dir is None else extension_dir) / output.removeprefix("tests/")
    if extension_dir is None and not target.resolve().is_relative_to(root):
        raise ValueError("prepared extension is outside runtime")
    _regular(target, size=receipt["output_size"])
    if sha256(target) != receipt["output_sha256"]:
        raise ValueError("staged extension differs before tests")
    python = Path(receipt["runtime"]["python"])
    if not python.is_relative_to(root) or not python.is_file():
        raise ValueError("prepared interpreter is outside runtime")
    if sha256(receipt["runtime"]["base_executable"]) != receipt["runtime"]["executable_sha256"]:
        raise ValueError("prepared base interpreter differs")
    if sha256(root / "runtime.json") != receipt["runtime_manifest_sha256"]:
        raise ValueError("prepared runtime manifest differs")
    site = Path(receipt["runtime"]["site_packages"])
    if not site.is_relative_to(root) or site_tree_sha256(site) != receipt["runtime"]["site_tree_sha256"]:
        raise ValueError("prepared package tree differs before tests")
    return {
        "python": str(python),
        "base_executable": receipt["runtime"]["base_executable"],
        "executable_sha256": receipt["runtime"]["executable_sha256"],
        "version": receipt["runtime"]["version"],
        "soabi": receipt["runtime"]["soabi"],
        "module": receipt["module"],
        "extension": str(target),
        "extension_sha256": receipt["output_sha256"],
        "site_packages": str(site),
        "site_tree_sha256": receipt["runtime"]["site_tree_sha256"],
    }


def pytest_entry(argv: list[str]) -> int:
    """Load the exact extension before pytest or candidate application code."""
    try:
        expected = json.loads(os.environ["NEMO_VERIFIER_ATTESTATION"])
        base = os.path.realpath(getattr(sys, "_base_executable", sys.executable))
        target = Path(expected["extension"])
        if (
            sys.executable != expected["python"]
            or base != os.path.realpath(expected["base_executable"])
            or sha256(base) != expected["executable_sha256"]
            or platform.python_version() != expected["version"]
            or sysconfig.get_config_var("SOABI") != expected["soabi"]
            or site_tree_sha256(expected["site_packages"]) != expected["site_tree_sha256"]
            or sha256(target) != expected["extension_sha256"]
        ):
            raise ValueError("test Python or extension identity differs")
        spec = importlib.util.spec_from_file_location(expected["module"], target)
        if spec is None or spec.loader is None:
            raise ValueError("prepared extension has no loader")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if (
            os.path.realpath(module.__file__) != os.path.realpath(target)
            or sha256(target) != expected["extension_sha256"]
        ):
            raise ValueError("initial imported extension differs")
        sys.modules[expected["module"]] = module
        print(
            json.dumps(
                {
                    "prepared_module": expected["module"],
                    "origin": os.path.realpath(module.__file__),
                    "sha256": expected["extension_sha256"],
                }
            ),
            flush=True,
        )
    except BaseException as error:  # setup failure, not a graded test result
        print(f"test interpreter attestation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return SETUP_FAILED
    import pytest

    return int(pytest.main(argv))


def _stock_phase(command: str, root: Path) -> None:
    environment = dict(os.environ, NEMO_PREP_ROOT=str(root), PYTHONPATH="", PYTEST_ADDOPTS="")
    result = subprocess.run(["/bin/bash", "-c", command], cwd="/gym_verifier_helper", env=environment, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"extension setup failed: {result.returncode}")


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--attest-python":
        return pytest_entry(argv[1:])
    if len(argv) == 2 and argv[0] == "--prepare-stock":
        try:
            spec = json.loads(argv[1])
            output_dir = Path("/trajectories_mount/prepared-extension")
            prepare_extension(
                json.dumps(spec),
                source_tests=Path("/tests"),
                output_dir=output_dir,
                receipt_dir=output_dir,
                run_phase=_stock_phase,
            )
            return 0
        except BaseException as error:
            print(f"extension preparation failed: {type(error).__name__}: {error}", file=sys.stderr)
            return SETUP_FAILED
    if len(argv) == 2 and argv[0] == "--attest-stock":
        try:
            attestation = attest_extension(
                argv[1],
                receipt_dir=Path("/trajectories_mount/prepared-extension"),
                extension_dir=Path("/tests"),
            )
            print(json.dumps(attestation, sort_keys=True))
            return 0
        except BaseException as error:
            print(f"extension attestation failed: {type(error).__name__}: {error}", file=sys.stderr)
            return SETUP_FAILED
    print("usage: extension_runtime.py --attest-python [pytest args] | --prepare-stock JSON", file=sys.stderr)
    return SETUP_FAILED


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
