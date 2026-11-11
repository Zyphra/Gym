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

from nemo_gym.sandbox.providers.utils import sandbox_client_environment


def test_only_allow_listed_names_and_prefixes_pass() -> None:
    environ = {
        "PATH": "/usr/bin",
        "HOME": "/home/user",
        "SLURM_STEP_ID": "3",
        "SLURM_JOB_NODELIST": "cnode-[1-9]",
        "LC_CTYPE": "C.UTF-8",
        "XDG_RUNTIME_DIR": "/run/user/1",
        "ENROOT_DATA_PATH": "/data",
        "APPTAINER_CACHEDIR": "/cache",
        "VERL_RUN_RECORD_JSON": "x" * 89_500,
        "OPENAI_API_KEY": "secret",
    }
    assert sandbox_client_environment(environ=environ) == {
        "PATH": "/usr/bin",
        "HOME": "/home/user",
        "SLURM_STEP_ID": "3",
        "LC_CTYPE": "C.UTF-8",
        "XDG_RUNTIME_DIR": "/run/user/1",
    }
    assert set(sandbox_client_environment(("ENROOT",), extra_names=("OPENAI_API_KEY",), environ=environ)) == {
        "PATH",
        "HOME",
        "SLURM_STEP_ID",
        "LC_CTYPE",
        "XDG_RUNTIME_DIR",
        "ENROOT_DATA_PATH",
        "OPENAI_API_KEY",
    }


def test_defaults_to_the_process_environment(monkeypatch) -> None:
    monkeypatch.setenv("VERL_RUN_RECORD_JSON", "x")
    monkeypatch.setenv("LANG", "C.UTF-8")
    env = sandbox_client_environment()
    assert env["LANG"] == "C.UTF-8" and "VERL_RUN_RECORD_JSON" not in env
