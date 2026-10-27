# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json

import pytest
from fastapi import HTTPException

from nemo_gym.base_resources_server import BaseSeedSessionRequest
from resources_servers.workplace_assistant.app import WorkbenchRequest
from resources_servers.workplace_assistant.tests.test_grader_input_and_dates import (
    creation,
    server_and_request,
    verify_body,
)
from resources_servers.workplace_assistant.utils import execute_actions_and_reset_state, state_matches_reference


@pytest.mark.parametrize("kind", ["gold", "noop", "wrong", "report_only", "fabricated_calls", "omitted_trace"])
def test_verify_reads_actual_session(kind):
    async def run():
        server, request = server_and_request()
        await server.seed_session(request, BaseSeedSessionRequest())
        gold = creation()
        executed = (
            [creation("Wrong meeting")] if kind == "wrong" else ([gold] if kind in {"gold", "omitted_trace"} else [])
        )
        for action in executed:
            result = await server.route_to_python_function(
                action["name"], WorkbenchRequest(**json.loads(action["arguments"])), request
            )
            assert not str(result.output).startswith("Error"), result
        claimed = [gold] if kind in {"gold", "fabricated_calls", "wrong"} else []
        body = verify_body(claimed, [gold])
        if kind == "report_only":
            from nemo_gym.openai_utils import NeMoGymResponseOutputMessage

            body.response.output.append(
                NeMoGymResponseOutputMessage(
                    id="report",
                    role="assistant",
                    type="message",
                    status="completed",
                    content=[{"type": "output_text", "text": "I created the correct meeting.", "annotations": []}],
                )
            )
        result = await server.verify(request, body)
        assert result.reward == float(kind in {"gold", "omitted_trace"})
        assert not server.session_id_to_tool_env

    asyncio.run(run())


def test_missing_session_is_unavailable_not_replayed():
    async def run():
        server, request = server_and_request()
        with pytest.raises(HTTPException) as caught:
            await server.verify(request, verify_body([creation()], [creation()]))
        assert caught.value.status_code == 409

    asyncio.run(run())


def test_live_state_retains_unintended_effects_absent_from_transcript():
    gold = creation()
    env = execute_actions_and_reset_state([gold, creation("Unrequested meeting")])
    assert not state_matches_reference(env, [gold])


def test_missing_tracking_is_unavailable():
    env = execute_actions_and_reset_state([creation()])
    del env["sent_emails"]
    with pytest.raises(ValueError, match="tracking is unavailable"):
        state_matches_reference(env, [creation()])
