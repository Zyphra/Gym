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
import json

from resources_servers.workplace_assistant.utils import execute_actions_and_reset_state, is_correct


def action(name, **arguments):
    return {"name": name, "arguments": json.dumps(arguments)}


def calendar_creations():
    return [
        action(
            "calendar_create_event",
            event_name="Handoff A",
            participant_email="aisha.chen@atlas.com",
            event_start="2023-12-01 09:00:00",
            duration="30",
        ),
        action(
            "calendar_create_event",
            event_name="Handoff B",
            participant_email="amir.ali@atlas.com",
            event_start="2023-12-02 11:00:00",
            duration="60",
        ),
    ]


def test_reversed_forward_recipients_are_equivalent():
    gold = [
        action("email_forward_email", email_id="00000031", recipient=recipient)
        for recipient in ["amir.ali@atlas.com", "akira.sato@atlas.com"]
    ]
    assert is_correct(list(reversed(gold)), gold, None)
    wrong = [gold[0], action("email_forward_email", email_id="00000031", recipient="aisha.chen@atlas.com")]
    assert not is_correct(wrong, gold, None)
    assert not is_correct(gold[:1], gold, None)
    assert not is_correct(gold + gold[:1], gold, None)
    assert not is_correct([], gold, None)


def test_plot_order_is_irrelevant_but_content_and_multiplicity_matter():
    gold = [
        action(
            "analytics_create_plot",
            time_min="2023-10-19",
            time_max="2023-11-30",
            value_to_plot=metric,
            plot_type="bar",
        )
        for metric in ["visits_social_media", "visits_search_engine"]
    ]
    assert is_correct(list(reversed(gold)), gold, None)
    assert not is_correct([gold[0], gold[0]], gold, None)
    assert not is_correct(gold[:1], gold, None)
    assert not is_correct(gold + gold[:1], gold, None)


def test_generated_calendar_ids_follow_content_after_reordered_creation():
    gold = calendar_creations()
    seed = execute_actions_and_reset_state([])["containers"]["calendar"]._calendar_events
    first_id = str(int(seed["event_id"].max()) + 1).zfill(8)
    second_id = str(int(first_id) + 1).zfill(8)
    update_first = action("calendar_update_event", event_id=first_id, field="event_name", new_value="Updated A")
    update_second = action("calendar_update_event", event_id=second_id, field="event_name", new_value="Updated A")
    assert is_correct(list(reversed(gold)), gold, None)
    assert is_correct([*reversed(gold), update_second], [*gold, update_first], None)
    assert not is_correct([*reversed(gold), update_first], [*gold, update_first], None)


def test_duplicate_generated_identity_is_invalid():
    gold = [
        action(
            "project_management_create_task",
            task_name=name,
            assigned_to_email="aisha.chen@atlas.com",
            board="Back end",
            list_name="Backlog",
            due_date="2023-12-06",
        )
        for name in ["Handoff A", "Handoff B"]
    ]
    seed = execute_actions_and_reset_state([])["containers"]["project_management"]._project_tasks
    first_id = str(int(seed["task_id"].max()) + 1).zfill(8)
    second_id = str(int(first_id) + 1).zfill(8)
    duplicate = action("project_management_update_task", task_id=second_id, field="task_id", new_value=first_id)
    state = execute_actions_and_reset_state([*gold, duplicate])["containers"]["project_management"]._project_tasks
    assert state["task_id"].duplicated().any()
    assert not is_correct([*gold, duplicate], gold, None)


def test_seeded_identity_and_unrequested_records_are_preserved():
    gold = [action("email_delete_email", email_id="00000031")]
    wrong = [action("email_delete_email", email_id="00000032")]
    assert not is_correct(wrong, gold, None)
    assert not is_correct([*gold, *wrong], gold, None)


def test_deleted_seed_identity_is_not_restored_by_generated_id_reuse():
    seed = execute_actions_and_reset_state([])["containers"]["calendar"]._calendar_events
    highest = seed["event_id"].max()
    delete = action("calendar_delete_event", event_id=highest)
    create = calendar_creations()[0]
    gold = [delete, create]
    legal = [create, delete]
    assert is_correct(legal, gold, None)
    assert not is_correct([create], gold, None)


def test_existing_case_rules_and_error_rejection_are_preserved():
    upper = [action("email_reply_email", email_id="00000155", body="THANK YOU")]
    lower = [action("email_reply_email", email_id="00000155", body="thank you")]
    assert is_correct(lower, upper, None)
    assert not is_correct(lower, upper, "execution failed")
    gold = [
        action(
            "customer_relationship_manager_update_customer",
            customer_id="00000015",
            field="status",
            new_value="Proposal",
        )
    ]
    wrong = [
        action(
            "customer_relationship_manager_update_customer",
            customer_id="00000015",
            field="status",
            new_value="proposal",
        )
    ]
    assert not is_correct(wrong, gold, None)
