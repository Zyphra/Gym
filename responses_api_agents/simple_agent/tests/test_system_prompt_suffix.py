# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig


def config(**kwargs):
    return SimpleAgentConfig(
        host="127.0.0.1",
        port=1,
        entrypoint="",
        name="agent",
        model_server={"type": "responses_api_models", "name": "model"},
        resources_server={"type": "resources_servers", "name": "resources"},
        **kwargs,
    )


@pytest.mark.parametrize("limit", [2, 50])
def test_budget_instruction_matches_real_policy_call_limit(limit):
    agent = SimpleAgent(
        config=config(
            max_steps=limit,
            system_prompt_suffix="You have at most {max_steps} assistant turns. Do not ask clarifying questions.",
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    payload = {
        "id": "response",
        "created_at": 1.0,
        "model": "fixture",
        "object": "response",
        "output": [
            {
                "type": "function_call",
                "id": "fc",
                "call_id": "call",
                "name": "read_state",
                "arguments": "{}",
                "status": "completed",
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }
    reply = MagicMock(status=200, ok=True, cookies={})
    reply.read = AsyncMock(return_value=json.dumps(payload).encode())
    reply.content.read = AsyncMock(return_value=b'{"state": "open"}')
    agent.server_client.post.return_value = reply
    client = TestClient(agent.setup_webserver())
    response = client.post(
        "/v1/responses",
        json={
            "input": [
                {"role": "system", "content": "Original task instructions."},
                {"role": "user", "content": "Use the state tool."},
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "read_state",
                    "description": "Read.",
                    "parameters": {"type": "object", "properties": {}},
                    "strict": False,
                }
            ],
        },
    )
    assert response.status_code == 200, response.text
    policy = [c.kwargs["json"] for c in agent.server_client.post.call_args_list if c.kwargs["server_name"] == "model"]
    assert len(policy) == limit
    for request in policy:
        messages = [m.model_dump() if hasattr(m, "model_dump") else m for m in request.input]
        assert messages[0]["content"] == "Original task instructions."
        assert messages[1]["role"] == "system"
        assert messages[1]["content"] == f"You have at most {limit} assistant turns. Do not ask clarifying questions."
        assert messages[2]["role"] == "user"
        assert sum(m.get("role") == "system" for m in messages) == 2


@pytest.mark.parametrize(
    "suffix,limit", [("", 50), ("   ", 50), ("Budget {max_steps}", None), ("Budget {max_steps}", 0)]
)
def test_invalid_budget_instruction_fails_before_any_model_call(suffix, limit):
    with pytest.raises(ValueError):
        config(max_steps=limit, system_prompt_suffix=suffix)


def test_unselected_instruction_default():
    assert config().system_prompt_suffix is None
