# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import stat

import pytest

from nemo_gym.sandbox.utils import remove_private_directory


@pytest.mark.parametrize("mode", [0o500, 0o000])
def test_read_only_tree_cleanup_preserves_link_targets(tmp_path, mode):
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("keep")
    private = tmp_path / "private"
    nested = private / "runtime" / "bin"
    nested.mkdir(parents=True)
    (nested / "tool").write_text("tool")
    (nested / "outside").symlink_to(outside, target_is_directory=True)
    (nested / "dangling").symlink_to(tmp_path / "missing")
    before = outside.stat().st_mode
    for path in [nested, nested.parent, private]:
        path.chmod(mode)
    remove_private_directory(private)
    assert not private.exists()
    assert sentinel.read_text() == "keep"
    assert outside.stat().st_mode == before


def test_cleanup_refuses_root_symlink(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    target.chmod(0o750)
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="expected a directory"):
        remove_private_directory(link)
    assert link.is_symlink()
    assert stat.S_IMODE(target.stat().st_mode) == 0o750
