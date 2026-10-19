# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import pandas as pd

from resources_servers.workplace_assistant.utils import get_tools, is_correct


TOOLKITS = ["email", "calendar", "analytics", "project_management", "customer_relationship_manager"]
DATA = os.path.join(os.path.dirname(__file__), "..", "csv_data")


def _frames(env):
    containers = env["containers"]
    return {
        "emails": containers["email"]._emails,
        "calendar_events": containers["calendar"]._calendar_events,
        "analytics_data": containers["analytics"]._analytics_data,
        "project_tasks": containers["project_management"]._project_tasks,
        "customer_relationship_manager_data": containers["customer_relationship_manager"]._crm_data,
        "email_addresses": containers["company_directory"]._emails,
    }


def test_tool_state_equals_a_fresh_parse_of_each_csv():
    frames = _frames(get_tools(TOOLKITS))
    for name, frame in frames.items():
        if name == "email_addresses":
            expected = pd.read_csv(os.path.join(DATA, "raw", f"{name}.csv"), header=None, names=["email_address"])
        else:
            expected = pd.read_csv(os.path.join(DATA, "processed", f"{name}.csv"), dtype=str)
            if name == "analytics_data":
                expected["user_engaged"] = expected["user_engaged"] == "True"
        pd.testing.assert_frame_equal(frame, expected)


def test_sessions_do_not_share_tool_state():
    first, second = get_tools(TOOLKITS), get_tools(TOOLKITS)
    first["functions"]["email_delete_email"](email_id="00000057")
    first["functions"]["calendar_delete_event"](event_id="00000013")
    assert len(first["containers"]["email"]._emails) == len(second["containers"]["email"]._emails) - 1
    assert len(get_tools(TOOLKITS)["containers"]["email"]._emails) == len(second["containers"]["email"]._emails)
    assert (
        len(first["containers"]["calendar"]._calendar_events)
        == len(second["containers"]["calendar"]._calendar_events) - 1
    )


def test_is_correct_still_compares_state_changes():
    reply = [{"name": "email_delete_email", "arguments": '{"email_id": "00000057"}'}]
    other = [{"name": "email_delete_email", "arguments": '{"email_id": "00000058"}'}]
    assert is_correct(reply, reply, None)
    assert not is_correct(reply, other, None)
