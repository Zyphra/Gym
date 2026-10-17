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
"""Checker infrastructure failures must remain distinct from policy failures."""

import asyncio
import errno

import pytest
from fastapi import HTTPException
from omegaconf import OmegaConf
from verifiable_instructions import instructions_util

from nemo_gym.base_resources_server import NeMoGymResponse
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.server_utils import ServerClient
from resources_servers.instruction_following.app import (
    InstructionFollowingResourcesServer,
    InstructionFollowingResourcesServerConfig,
    InstructionFollowingVerifyRequest,
)


@pytest.fixture
def server():
    client = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=0),
        global_config_dict=OmegaConf.create({}),
    )
    return InstructionFollowingResourcesServer(
        config=InstructionFollowingResourcesServerConfig(
            name="checker_control",
            host="127.0.0.1",
            port=0,
            entrypoint="app.py",
        ),
        server_client=client,
    )


def request(text):
    converter = ResponsesConverter(return_token_id_information=False, uses_reasoning_parser=False)
    response = NeMoGymResponse(
        id="response_control",
        created_at=0.0,
        model="policy",
        object="response",
        output=converter.postprocess_assistant_message_dict({"role": "assistant", "content": text}),
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )
    return InstructionFollowingVerifyRequest(
        id=1,
        response=response,
        responses_create_params={"input": [{"role": "user", "content": "Write at least two sentences."}]},
        verifier_metadata={
            "instruction_id_list": ["length_constraints:number_sentences"],
            "prompt": "Write at least two sentences.",
            "kwargs": [{"num_sentences": 2, "relation": "at least"}],
            "grading_mode": "binary",
        },
    )


@pytest.mark.parametrize("text,expected", [("One sentence. Another sentence.", True), ("Only one sentence.", False)])
def test_native_checker_keeps_valid_and_invalid_answers(server, text, expected):
    result = asyncio.run(server.verify(request(text)))
    assert result.follow_instruction_list == [expected]
    assert result.follow_all_instructions is expected
    assert result.reward == float(expected)


@pytest.mark.parametrize(
    "error", [FileNotFoundError(errno.ENOENT, "tokenizer missing"), ModuleNotFoundError("tokenizer module missing")]
)
def test_native_checker_dependency_failure_is_unscoreable(server, monkeypatch, error):
    def fail_dependency(text):
        raise error

    # Keep the real registry, checker, verifier, DTOs and client. Only the
    # native sentence helper's dependency boundary fails in this test process.
    monkeypatch.setattr(instructions_util, "count_sentences", fail_dependency)
    with pytest.raises(HTTPException) as caught:
        asyncio.run(server.verify(request("One sentence. Another sentence.")))
    assert caught.value.status_code == 502
    assert caught.value.__cause__ is error
    assert caught.value.detail == {
        "stage": "verification",
        "cause": "instruction_checker_dependency_error",
        "exception_class": type(error).__name__,
        "instruction_id": "length_constraints:number_sentences",
    }


def test_other_checker_exceptions_keep_existing_failed_check(server, monkeypatch):
    def fail_check(text):
        raise ValueError("invalid checker input")

    monkeypatch.setattr(instructions_util, "count_sentences", fail_check)
    result = asyncio.run(server.verify(request("One sentence. Another sentence.")))
    assert result.follow_instruction_list == [False]
    assert result.follow_all_instructions is False
    assert result.reward == 0.0
