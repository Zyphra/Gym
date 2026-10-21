# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quoted English words reach the actual validator and verification endpoint."""

import asyncio
from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.verifif.app import (
    TuringVIFResourcesServer,
    TuringVIFResourcesServerConfig,
    TuringVIFVerifyRequest,
)
from resources_servers.verifif.vif_validators.en.validator import keyword_frequency


@pytest.mark.parametrize(
    "response,keyword,count",
    [
        ("clearly", "clearly", 1),
        ("'clearly", "clearly", 1),
        ("clearly'", "clearly", 1),
        ("'clearly'", "clearly", 1),
        ("'Clearly stated' and 'clearly shown'", "clearly", 2),
        ('"clearly"', "clearly", 1),
        ("unclearly", "clearly", 0),
        ("clearly's", "clearly", 0),
        ("someone'clearly", "clearly", 0),
        ("clearly-written", "clearly", 0),
        ("not-clearly", "clearly", 0),
        ("'don't'", "don't", 1),
        ("'well-being'", "well-being", 1),
        ("unwell-being", "well-being", 0),
        ("Nothing relevant", "clearly", 0),
    ],
)
def test_keyword_boundaries_preserve_whole_words(response, keyword, count):
    assert keyword_frequency(response, keyword) == count


def _verify(text, instruction, *, source="system", misaligned=False):
    server = TuringVIFResourcesServer(
        config=TuringVIFResourcesServerConfig(host="127.0.0.1", port=0, name="verifif", entrypoint="app.py"),
        server_client=MagicMock(spec=ServerClient),
    )
    response = NeMoGymResponse.model_validate(
        {
            "id": "planted",
            "created_at": 0,
            "model": "none",
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "id": "answer",
                    "role": "assistant",
                    "type": "message",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
        }
    )
    request = TuringVIFVerifyRequest(
        id=1,
        language="en",
        llm_judge=[],
        response=response,
        responses_create_params={"input": [{"role": "user", "content": "Independent planted keyword control"}]},
        instructions=[{"uid": 1, "source": source, "is_misalignment_check": misaligned, **instruction}],
    )
    return asyncio.run(server.verify(request))


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A precise answer avoids the forbidden term.", 1),
        ("The note says 'clearly demonstrated'.", 0),
        ("The note says clearly demonstrated.", 0),
    ],
)
def test_forbidden_words_full_verify_gold_and_wrong(text, expected):
    result = _verify(text, {"instruction_id": "keywords:forbidden_words", "forbidden_words": ["clearly"]})
    assert result.reward == expected
    assert result.follow_instruction_list == [bool(expected)]


@pytest.mark.parametrize(
    "identifier,kwargs",
    [
        ("keywords:existence", {"keywords": ["clearly"]}),
        ("keywords:frequency", {"keyword": "clearly", "relation": "equal to", "frequency": 1}),
    ],
)
def test_positive_keyword_contract_uses_same_quote_boundary(identifier, kwargs):
    assert _verify("The note says 'clearly demonstrated'.", {"instruction_id": identifier, **kwargs}).reward == 1
    assert _verify("The note says merely demonstrated.", {"instruction_id": identifier, **kwargs}).reward == 0


@pytest.mark.parametrize(
    "source,misaligned,expected",
    [
        ("system", False, 0),
        ("system", True, 0),
        ("user", False, 0),
        ("user", True, 1),
    ],
)
def test_existing_user_misalignment_inversion_is_preserved(source, misaligned, expected):
    result = _verify(
        "The note says 'clearly demonstrated'.",
        {"instruction_id": "keywords:forbidden_words", "forbidden_words": ["clearly"]},
        source=source,
        misaligned=misaligned,
    )
    assert result.reward == expected
    assert result.follow_instruction_list == [bool(expected)]
