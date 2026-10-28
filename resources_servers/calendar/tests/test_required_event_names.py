"""Mandatory name presence and the legacy stricter name-type guard."""

import asyncio
import json

import pytest
from omegaconf import OmegaConf

from nemo_gym.config_types import BaseServerConfig
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.calendar.app import (
    CalendarResourcesServer,
    CalendarResourcesServerConfig,
    CalendarVerifyRequest,
)


STATE = {"0": {"event_id": 0, "duration": 30, "constraint": None, "min_time": "10:00", "max_time": "16:00"}}


def grade(event, enabled):
    client = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=0), global_config_dict=OmegaConf.create({})
    )
    srv = CalendarResourcesServer(
        config=CalendarResourcesServerConfig(
            host="127.0.0.1", port=0, entrypoint="app.py", name="calendar", require_event_names=enabled
        ),
        server_client=client,
    )
    response = NeMoGymResponse.model_validate(
        {
            "id": "control",
            "created_at": 0,
            "model": "fixture",
            "object": "response",
            "status": "completed",
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
            "output": [
                {
                    "id": "answer",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(event if isinstance(event, list) else [event]),
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                }
            ],
        }
    )
    body = CalendarVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "Schedule event 0, Coding workshop."}]},
        exp_cal_state=STATE,
        response=response,
    )
    return asyncio.run(srv.verify(body)).reward


@pytest.mark.parametrize("name,stock_reward", [(None, 0), ("", 1), ("   ", 1), (7, 1), ([], 0), ({}, 1)])
def test_name_presence_is_mandatory_and_legacy_type_guard_is_preserved(name, stock_reward):
    event = {"event_id": 0, "duration": 30, "start_time": "10:00"}
    if name is not None:
        event["event_name"] = name
    assert grade(event, False) == stock_reward
    assert grade(event, True) == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_valid_named_event_parity_and_wrong_duration(enabled):
    event = {"event_id": 0, "event_name": "Coding workshop", "duration": 30, "start_time": "10:00"}
    assert grade(event, enabled) == 1
    assert grade({**event, "duration": 31}, enabled) == 0


@pytest.mark.parametrize("bad", [1, None, "not an event"])
def test_non_object_list_entries_are_explicitly_rejected_without_an_exception(bad):
    named = {"event_id": 0, "event_name": "Coding workshop", "duration": 30, "start_time": "10:00"}
    assert grade([bad, named], True) == 0
