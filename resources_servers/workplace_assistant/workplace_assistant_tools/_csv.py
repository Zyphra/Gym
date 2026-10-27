# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Parse each packaged CSV once per process and hand every tool a private copy.

Every session seed and every ``is_correct`` call builds fresh tool state from the
same read-only CSVs; re-parsing them dominated that cost (7 ms of a 7.4 ms
seed). A copy of the parsed frame is equal in content and dtypes.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


_SEED_ROOT: Path | None = None
_WORLD_STARTED = False


def configure_seed_root(root: Path | None) -> None:
    """Select one immutable CSV world before any tool or reference initializes."""
    global _SEED_ROOT
    selected = Path(root).resolve() if root is not None else None
    if (_WORLD_STARTED or _SEED_ROOT is not None) and selected != _SEED_ROOT:
        raise RuntimeError("Workplace CSV seed root cannot change within a process")
    if selected is not None and not selected.is_dir():
        raise FileNotFoundError(selected)
    _SEED_ROOT = selected


@lru_cache(maxsize=None)
def _parsed(
    reader, path: str, dtype: Optional[type], no_header: bool, names: Optional[Tuple[str, ...]]
) -> pd.DataFrame:
    kwargs = {"dtype": dtype}
    if no_header:
        kwargs["header"] = None
    if names is not None:
        kwargs["names"] = list(names)
    return reader(path, **kwargs)


def read_csv(path: str, *, dtype: Optional[type] = None, no_header: bool = False, names=None) -> pd.DataFrame:
    """``pd.read_csv(path, ...)`` for the packaged data, parsed once per process.

    The cache is keyed on the current ``pd.read_csv`` as well, so a replaced
    reader (a test double) is always called rather than served from the cache.
    """
    global _WORLD_STARTED
    if _SEED_ROOT is not None:
        packaged = Path(__file__).resolve().parent.parent / "csv_data"
        relative = Path(path).resolve().relative_to(packaged)
        selected = _SEED_ROOT / relative
        if not selected.is_file():
            raise FileNotFoundError(selected)
        path = str(selected)
    _WORLD_STARTED = True
    return _parsed(pd.read_csv, path, dtype, no_header, tuple(names) if names else None).copy()
