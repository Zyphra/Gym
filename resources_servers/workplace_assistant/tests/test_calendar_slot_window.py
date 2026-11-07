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
"""Meetings are booked Monday-Friday, start at or after 09:00 and end by 18:00.

The dataset system prompt says "Meetings must not start before 9am or end after 6pm". A slot outside that window
is refused with an error and changes no state, live and in the grader's replay.
"""

import json

import pytest

from resources_servers.workplace_assistant.utils import execute_actions_and_reset_state, get_tools, is_correct


def action(name, **arguments):
    return {"name": name, "arguments": json.dumps(arguments)}


def booking(start, duration="60", name="Handoff"):
    return action(
        "calendar_create_event",
        event_name=name,
        participant_email="aisha.chen@atlas.com",
        event_start=start,
        duration=duration,
    )


def rows(env):
    return len(env["containers"]["calendar"]._calendar_events)


WITHIN = [
    ("2023-12-04 09:00:00", "60"),  # Monday, opening
    ("2023-12-04 17:00:00", "60"),  # ends exactly at closing
    ("2023-12-08 16:30:00", "90"),  # Friday
    ("2023-12-05T10:00:00", "030"),  # ISO form, zero-padded duration
]
OUTSIDE = [
    ("2023-12-04 08:59:00", "30"),  # before 09:00
    ("2023-12-04 17:30:00", "120"),  # ends 19:30
    ("2023-12-04 17:01:00", "60"),  # ends one minute late
    ("2023-12-04 23:30:00", "60"),  # crosses midnight
    ("2023-12-02 11:00:00", "60"),  # Saturday
    ("2023-12-03 11:00:00", "60"),  # Sunday
]


@pytest.mark.parametrize("start,duration", WITHIN)
def test_in_window_slot_is_booked(start, duration):
    env = get_tools(["calendar"])
    before = len(env["containers"]["calendar"]._calendar_events)
    result = env["functions"]["calendar_create_event"]("Handoff", "aisha.chen@atlas.com", start, duration)
    assert result.isdigit()
    assert len(env["containers"]["calendar"]._calendar_events) == before + 1


@pytest.mark.parametrize("start,duration", OUTSIDE)
def test_out_of_window_slot_is_refused_with_the_rule_and_changes_nothing(start, duration):
    env = get_tools(["calendar"])
    before = env["containers"]["calendar"]._calendar_events.copy()
    result = env["functions"]["calendar_create_event"]("Handoff", "aisha.chen@atlas.com", start, duration)
    assert "outside the booking window" in result
    assert "Monday to Friday" in result and "09:00" in result and "18:00" in result
    assert env["containers"]["calendar"]._calendar_events.equals(before)


def test_update_cannot_move_an_event_out_of_the_window_but_other_fields_are_free():
    env = get_tools(["calendar"])
    create, update, calendar = (
        env["functions"]["calendar_create_event"],
        env["functions"]["calendar_update_event"],
        env["containers"]["calendar"],
    )
    event_id = create("Handoff", "aisha.chen@atlas.com", "2023-12-04 16:00:00", "60")
    assert "outside the booking window" in update(event_id, "event_start", "2023-12-04 17:30:00")
    assert "outside the booking window" in update(event_id, "duration", "180")
    assert "outside the booking window" in update(event_id, "event_start", "2023-12-09 10:00:00")
    stored = calendar._calendar_events.set_index("event_id").loc[event_id]
    assert (stored["event_start"], stored["duration"]) == ("2023-12-04 16:00:00", "60")
    assert update(event_id, "event_start", "2023-12-05 10:00:00") == "Event updated successfully."
    assert update(event_id, "duration", "90") == "Event updated successfully."
    assert update(event_id, "event_name", "Renamed") == "Event updated successfully."


def test_renaming_a_seeded_event_is_not_blocked_by_its_own_slot():
    env = get_tools(["calendar"])
    calendar = env["containers"]["calendar"]
    seeded = calendar._calendar_events.iloc[0]
    assert env["functions"]["calendar_update_event"](seeded["event_id"], "event_name", "Renamed") == (
        "Event updated successfully."
    )


# Full path through the grader: the reference books a legal slot.
GOLD = [booking("2023-12-04 16:00:00", "60")]


def test_gold_passes():
    assert is_correct(GOLD, GOLD, None)


def test_no_op_fails():
    assert not is_correct([], GOLD, None)


@pytest.mark.parametrize("start,duration", OUTSIDE)
def test_wrong_answer_in_an_out_of_window_slot_fails_and_leaves_no_event(start, duration):
    wrong = [booking(start, duration)]
    assert rows(execute_actions_and_reset_state(wrong)) == rows(execute_actions_and_reset_state([]))
    assert not is_correct(wrong, GOLD, None)


def test_refused_slot_then_the_legal_slot_scores_like_the_reference():
    """Feedback works: the refusal is a no-op, so retrying inside the window earns the reference state."""
    retry = [booking("2023-12-04 17:30:00", "120"), *GOLD]
    assert is_correct(retry, GOLD, None)


def test_legal_slot_at_a_different_time_still_fails():
    assert not is_correct([booking("2023-12-04 15:00:00", "60")], GOLD, None)
