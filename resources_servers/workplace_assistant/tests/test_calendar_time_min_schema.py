# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calendar discovery describes the same inclusive start bound the tool applies."""

import pandas as pd
from resources_servers.workplace_assistant.utils import get_tools


def test_calendar_time_min_includes_start_boundary_and_excludes_prior_overlap():
    env = get_tools(["calendar"])
    create = env["functions"]["calendar_create_event"]
    query = "calendar-time-min-boundary"
    before = create(query, "boundary@example.com", "2023-11-20 09:30:00", "90")
    boundary = create(query, "boundary@example.com", "2023-11-20 10:00:00", "30")
    later = create(query, "boundary@example.com", "2023-11-20 10:01:00", "30")
    information = env["functions"]["calendar_get_event_information_by_id"]
    prior_start = information(event_id=before, field="event_start")["event_start"]
    prior_duration = information(event_id=before, field="duration")["duration"]
    minimum = "2023-11-20 10:00:00"
    assert pd.Timestamp(prior_start) < pd.Timestamp(minimum)
    assert pd.Timestamp(prior_start) + pd.Timedelta(minutes=int(prior_duration)) > pd.Timestamp(minimum)
    result = env["functions"]["calendar_search_events"](query=query, time_min=minimum)
    assert [event["event_id"] for event in result["events"]] == [later, boundary]
    assert result["pagination"]["total_events"] == 2
    assert before not in {event["event_id"] for event in result["events"]}


def test_calendar_get_tools_schema_describes_inclusive_lower_start_bound():
    schema = next(s for s in get_tools(["calendar"])["schemas"] if s["name"] == "calendar_search_events")
    parameter = schema["parameters"]["properties"]["time_min"]
    assert parameter["type"] == "string"
    assert parameter["description"] == (
        "Lower bound (inclusive) for an event's start time to filter by. Format: YYYY-MM-DD HH:MM:SS"
    )
