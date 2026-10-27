# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A process selects one native CSV world for policy and reference state."""

import os
import subprocess
import sys


def run(code, root):
    subprocess.run([sys.executable, "-c", code, str(root)], check=True, env=os.environ.copy())


PREFIX = """
import sys
from pathlib import Path
from resources_servers.workplace_assistant.workplace_assistant_tools import _csv
root = Path(sys.argv[1])
packaged = Path(_csv.__file__).resolve().parent.parent/'csv_data/processed/project_tasks.csv'
"""


def test_default_csv_bytes_and_started_root_are_preserved(tmp_path):
    run(
        PREFIX
        + """
import pandas as pd
_csv.configure_seed_root(None)
assert _csv.read_csv(str(packaged), dtype=str).equals(pd.read_csv(packaged, dtype=str))
try: _csv.configure_seed_root(root)
except RuntimeError: pass
else: raise AssertionError('root switched after default read')
""",
        tmp_path,
    )


def test_alternate_csv_cache_is_private_and_root_is_immutable(tmp_path):
    target = tmp_path / "processed"
    target.mkdir()
    (target / "project_tasks.csv").write_text("task_id,task_name\n09999999,Alternate fact\n")
    run(
        PREFIX
        + """
_csv.configure_seed_root(root)
frame = _csv.read_csv(str(packaged), dtype=str)
assert frame.iloc[0].to_dict() == {'task_id':'09999999','task_name':'Alternate fact'}
frame.iloc[0,0] = 'mutated'
assert _csv.read_csv(str(packaged), dtype=str).iloc[0,0] == '09999999'
try: _csv.configure_seed_root(None)
except RuntimeError: pass
else: raise AssertionError('configured world switched')
""",
        tmp_path,
    )


def test_missing_alternate_csv_never_falls_back_to_packaged_data(tmp_path):
    run(
        PREFIX
        + """
_csv.configure_seed_root(root)
try: _csv.read_csv(str(packaged))
except FileNotFoundError as error: assert Path(error.args[0]) == root/'processed/project_tasks.csv'
else: raise AssertionError('missing world accepted')
""",
        tmp_path,
    )
