# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path


LOG = logging.getLogger(__name__)

_CODEX_PKG = "@openai/codex"
_NODE_VERSION = "22.15.0"
_NODE_DIST_URL = f"https://nodejs.org/dist/v{_NODE_VERSION}/node-v{_NODE_VERSION}-linux-x64.tar.xz"
_LOCAL_PREFIX = Path(__file__).parent / ".codex_node"
# `codex --version` prints e.g. "codex-cli 0.144.4".
_VERSION_LINE = re.compile(r"^codex(?:-cli)?\s+v?(\S+)$")


def _npm_install(npm_bin: str, version: str | None) -> None:
    pkg = f"{_CODEX_PKG}@{version}" if version else f"{_CODEX_PKG}@latest"
    subprocess.run([npm_bin, "install", "-g", pkg], check=True)


def _install_node_locally() -> Path:
    node_bin = _LOCAL_PREFIX / "bin" / "node"
    if node_bin.is_file():
        return _LOCAL_PREFIX / "bin"

    _LOCAL_PREFIX.mkdir(parents=True, exist_ok=True)
    tarball = _LOCAL_PREFIX / "node.tar.xz"

    LOG.info("downloading Node.js %s", _NODE_VERSION)
    urllib.request.urlretrieve(_NODE_DIST_URL, tarball)  # noqa: S310

    with tarfile.open(tarball, "r:xz") as tf:
        tf.extractall(_LOCAL_PREFIX, filter="data")

    nested = next(p for p in _LOCAL_PREFIX.iterdir() if p.is_dir() and p.name.startswith("node-"))
    for item in nested.iterdir():
        item.rename(_LOCAL_PREFIX / item.name)
    nested.rmdir()
    tarball.unlink(missing_ok=True)
    return _LOCAL_PREFIX / "bin"


def _codex_cli_version(executable: str) -> str | None:
    """The version ``<executable> --version`` reports (``codex-cli 0.144.4`` -> ``0.144.4``), or None.

    A run that exits non-zero reports nothing, whatever it printed. The probe runs with a
    throwaway CODEX_HOME, so it never writes into a real Codex home.
    """
    with tempfile.TemporaryDirectory(prefix="codex_version_") as codex_home:
        try:
            result = subprocess.run(
                [executable, "--version"],
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, "CODEX_HOME": codex_home},
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        match = _VERSION_LINE.match(line.strip())
        if match:
            return match.group(1)
    return None


def require_pinned_codex(version: str) -> str:
    """Return the ``codex`` on PATH if it reports exactly ``version``; never install or download.

    For runtimes that ship the pinned binary and run without network: a missing or mismatched
    binary raises at startup instead of being replaced or used silently.
    """
    executable = shutil.which("codex")
    if executable is None:
        raise RuntimeError(
            f"codex {version} is not on PATH and offline provisioning never installs it; "
            "ship the pinned binary with the runtime"
        )
    reported = _codex_cli_version(executable)
    if reported != version:
        raise RuntimeError(
            f"{executable} reports codex version {reported!r} but codex_version pins {version!r}; "
            "offline provisioning never installs a replacement"
        )
    return executable


def ensure_codex(version: str | None = None) -> None:
    """Ensure ``codex`` is on PATH, installing it if necessary."""
    if shutil.which("codex"):
        return

    # Check ~/.local/bin
    local_bin = Path.home() / ".local" / "bin"
    if (local_bin / "codex").is_file():
        os.environ["PATH"] = str(local_bin) + os.pathsep + os.environ.get("PATH", "")
        return

    npm = shutil.which("npm")
    if npm:
        LOG.info("installing codex via system npm (%s)", npm)
        _npm_install(npm, version)
    else:
        LOG.info("npm not found; installing local Node.js")
        bin_dir = _install_node_locally()
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
        npm = shutil.which("npm")
        if not npm:
            raise RuntimeError(f"npm not found after local Node.js install in {bin_dir}")
        _npm_install(npm, version)

    # npm install -g may put the binary in a prefix not yet on PATH. `npm bin -g` was removed in
    # npm >= 9 (Node 22.15 ships npm 10), so resolve the global prefix and append its bin dir.
    if not shutil.which("codex"):
        npm_prefix = subprocess.run(
            [shutil.which("npm") or "npm", "prefix", "-g"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        npm_bin_dir = str(Path(npm_prefix) / "bin") if npm_prefix else ""
        if npm_bin_dir and Path(npm_bin_dir).is_dir():
            os.environ["PATH"] = npm_bin_dir + os.pathsep + os.environ.get("PATH", "")

    # Also check ~/.local/bin after install
    if not shutil.which("codex") and (local_bin / "codex").is_file():
        os.environ["PATH"] = str(local_bin) + os.pathsep + os.environ.get("PATH", "")

    if not shutil.which("codex"):
        raise RuntimeError("codex install appeared to succeed but 'codex' is still not on PATH")

    LOG.info("codex is ready at %s", shutil.which("codex"))
