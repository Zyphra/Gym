# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The glibc floor check reads real ELF version needs of libraries this process has mapped."""

import importlib.util
import os
import platform
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or platform.libc_ver()[0] != "glibc", reason="ELF/glibc check needs a glibc Linux host"
)

_SCRIPT = Path(__file__).parents[1] / "scripts" / "check_glibc_floor.py"
_spec = importlib.util.spec_from_file_location("check_glibc_floor", _SCRIPT)
floor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(floor)


def _loaded_extension() -> str:
    """pydantic_core's compiled extension, mapped by the agent's own imports."""
    from pydantic_core import _pydantic_core

    return os.path.realpath(_pydantic_core.__file__)


def test_required_glibc_reads_symbol_versions_of_a_mapped_extension():
    needed = floor.required_glibc(_loaded_extension())

    assert needed, "a dynamically linked extension requires at least one GLIBC_ symbol version"
    assert all(isinstance(v, tuple) and all(isinstance(p, int) for p in v) for v in needed)
    assert needed == sorted(needed)
    assert needed[0] >= (2, 2, 5)


def test_offenders_names_libraries_above_the_floor_and_skips_optional_packages():
    path = _loaded_extension()
    package_dir = os.path.dirname(path)
    prefix = os.path.dirname(package_dir)
    newest = floor.required_glibc(path)[-1]

    assert path in floor.mapped_libraries(prefix)
    assert any(line.startswith(path) for line in floor.offenders(prefix, (2, 0)))
    assert not any(line.startswith(path) for line in floor.offenders(prefix, newest))
    optional = (os.path.basename(package_dir),)
    assert not any(line.startswith(path) for line in floor.offenders(prefix, (2, 0), optional))


def test_required_glibc_rejects_non_elf(tmp_path):
    text = tmp_path / "not-elf.so"
    text.write_text("plain text")

    with pytest.raises(ValueError):
        floor.required_glibc(str(text))
