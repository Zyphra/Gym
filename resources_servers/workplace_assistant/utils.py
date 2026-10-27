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
import copy
import json
from collections import Counter
from typing import Dict, List

import pandas as pd

from resources_servers.workplace_assistant.workplace_assistant_tools.analytics import (
    AnalyticsTool,
    analytics_tool_schemas,
)
from resources_servers.workplace_assistant.workplace_assistant_tools.calendar import (
    CalendarTool,
    calendar_tool_schemas,
)
from resources_servers.workplace_assistant.workplace_assistant_tools.company_directory import (
    CompanyDirectoryTool,
    company_directory_tool_schemas,
)
from resources_servers.workplace_assistant.workplace_assistant_tools.customer_relationship_manager import (
    CustomerRelationshipManagerTool,
    customer_relationship_manager_tool_schemas,
)
from resources_servers.workplace_assistant.workplace_assistant_tools.email import (
    EmailTool,
    email_tool_schemas,
)
from resources_servers.workplace_assistant.workplace_assistant_tools.project_management import (
    ProjectManagementTool,
    project_management_tool_schemas,
)


def get_tools(toolkits):
    tool_env = {
        "containers": {},
        "functions": {},
        "schemas": [],
    }
    company_directory = CompanyDirectoryTool()
    tool_env["containers"]["company_directory"] = company_directory
    tool_env["functions"]["company_directory_find_email_address"] = company_directory.find_email_address
    tool_env["schemas"].extend(company_directory_tool_schemas)
    if "email" in toolkits:
        email = EmailTool()
        tool_env["containers"]["email"] = email
        tool_env["functions"]["email_get_email_information_by_id"] = email.get_email_information_by_id
        tool_env["functions"]["email_search_emails"] = email.search_emails
        tool_env["functions"]["email_send_email"] = email.send_email
        tool_env["functions"]["email_delete_email"] = email.delete_email
        tool_env["functions"]["email_forward_email"] = email.forward_email
        tool_env["functions"]["email_reply_email"] = email.reply_email
        tool_env["schemas"].extend(email_tool_schemas)
    if "calendar" in toolkits:
        calendar = CalendarTool()
        tool_env["containers"]["calendar"] = calendar
        tool_env["functions"]["calendar_get_event_information_by_id"] = calendar.get_event_information_by_id
        tool_env["functions"]["calendar_search_events"] = calendar.search_events
        tool_env["functions"]["calendar_create_event"] = calendar.create_event
        tool_env["functions"]["calendar_delete_event"] = calendar.delete_event
        tool_env["functions"]["calendar_update_event"] = calendar.update_event
        tool_env["schemas"].extend(calendar_tool_schemas)
    if "analytics" in toolkits:
        analytics = AnalyticsTool()
        tool_env["containers"]["analytics"] = analytics
        tool_env["functions"]["analytics_engaged_users_count"] = analytics.engaged_users_count
        tool_env["functions"]["analytics_get_visitor_information_by_id"] = analytics.get_visitor_information_by_id
        tool_env["functions"]["analytics_create_plot"] = analytics.create_plot
        tool_env["functions"]["analytics_traffic_source_count"] = analytics.traffic_source_count
        tool_env["functions"]["analytics_total_visits_count"] = analytics.total_visits_count
        tool_env["functions"]["analytics_get_average_session_duration"] = analytics.get_average_session_duration
        tool_env["schemas"].extend(analytics_tool_schemas)
    if "project_management" in toolkits:
        project_management = ProjectManagementTool()
        tool_env["containers"]["project_management"] = project_management
        tool_env["functions"]["project_management_get_task_information_by_id"] = (
            project_management.get_task_information_by_id
        )
        tool_env["functions"]["project_management_search_tasks"] = project_management.search_tasks
        tool_env["functions"]["project_management_create_task"] = project_management.create_task
        tool_env["functions"]["project_management_delete_task"] = project_management.delete_task
        tool_env["functions"]["project_management_update_task"] = project_management.update_task
        tool_env["schemas"].extend(project_management_tool_schemas)
    if "customer_relationship_manager" in toolkits:
        customer_relationship_manager = CustomerRelationshipManagerTool()
        tool_env["containers"]["customer_relationship_manager"] = customer_relationship_manager
        tool_env["functions"]["customer_relationship_manager_search_customers"] = (
            customer_relationship_manager.search_customers
        )
        tool_env["functions"]["customer_relationship_manager_update_customer"] = (
            customer_relationship_manager.update_customer
        )
        tool_env["functions"]["customer_relationship_manager_add_customer"] = (
            customer_relationship_manager.add_customer
        )
        tool_env["functions"]["customer_relationship_manager_delete_customer"] = (
            customer_relationship_manager.delete_customer
        )
        tool_env["schemas"].extend(customer_relationship_manager_tool_schemas)
    return tool_env


def execute_actions_and_reset_state(actions: List[Dict[str, str]]):
    toolkits = [
        "email",
        "calendar",
        "analytics",
        "project_management",
        "customer_relationship_manager",
    ]
    tool_env = get_tools(toolkits)
    initialize_effect_tracking(tool_env)
    for action in actions:
        execute_tracked_action(tool_env, action["name"], json.loads(action["arguments"]))
    return tool_env


def initialize_effect_tracking(tool_env):
    """Capture seeded identities before the first live or replayed operation."""
    tool_env["stable_ids"] = {
        name: set(frame[id_column]) if id_column is not None else set()
        for name, (frame, id_column) in _state_tables(tool_env).items()
    }

    # Every email an action sends, as sent. A later delete removes the row from
    # the mailbox, but the recipient already has the message, so grading keeps it.
    tool_env["sent_emails"] = []
    tool_env["observations"] = []


def execute_tracked_action(tool_env, name, arguments):
    """Execute once and retain real effects, including writes before an error."""
    emails_before = set(tool_env["containers"]["email"]._emails["email_id"])
    try:
        result = tool_env["functions"][name](**arguments)
    except Exception as error:
        result = f"Error executing tool '{name}': {error}"
    finally:
        emails = tool_env["containers"]["email"]._emails
        tool_env["sent_emails"].extend(emails[~emails["email_id"].isin(emails_before)].to_dict(orient="records"))
        # Once a seeded record is removed, reusing its ID does not restore its
        # identity. Track this per replay so delete/create order can also vary.
        for table, (frame, id_column) in _state_tables(tool_env).items():
            if id_column is not None:
                # Canonicalization rejects non-scalar IDs. Keep them out of
                # hash-based identity tracking so they reach that validation.
                tool_env["stable_ids"][table].intersection_update(
                    value for value in frame[id_column] if pd.api.types.is_scalar(value)
                )
    tool_env["observations"].append(
        {"tool": name, "arguments": copy.deepcopy(arguments), "result": copy.deepcopy(result)}
    )
    return result


def _state_tables(tool_env):
    containers = tool_env["containers"]
    return {
        "calendar": (containers["calendar"]._calendar_events, "event_id"),
        "email": (containers["email"]._emails, "email_id"),
        "analytics": (containers["analytics"]._plots_data, None),
        "project_management": (containers["project_management"]._project_tasks, "task_id"),
        "customer_relationship_manager": (containers["customer_relationship_manager"]._crm_data, "customer_id"),
    }


def _sent_email_rows(tool_env):
    """Canonical multiset of every email sent during a replay, deleted or not.

    Generated email IDs are positional (max + 1), so a deleted send and its
    replacement can share one; identity is the sent content.
    """
    columns = tool_env["containers"]["email"]._emails.columns
    frame = pd.DataFrame(tool_env["sent_emails"], columns=columns).drop(columns=["email_id"])
    return _canonical_rows(frame, None, set(), "email")


def _canonical_rows(frame, id_column, stable_ids, table_name=None):
    """Match seeded records by ID and newly created records by complete content.

    Workplace tables have no foreign keys between generated records. Generated
    IDs identify rows only within a replay and can differ with creation order.
    Keep row multiplicity, every other field, and the original case rules.
    """
    columns = tuple(sorted(frame.columns))
    if id_column is not None:
        if not all(pd.api.types.is_scalar(value) for value in frame[id_column]):
            return None
        if frame[id_column].duplicated().any():
            return None
    rows = []
    for values in frame.loc[:, columns].itertuples(index=False, name=None):
        row = []
        for column, value in zip(columns, values):
            # Tool schemas declare scalar table cells. Malformed persisted tool
            # arguments are incorrect actions, not verifier infrastructure errors.
            if not pd.api.types.is_scalar(value):
                return None
            if pd.isna(value):
                value = None
            elif table_name == "calendar" and column in {"event_start", "duration"}:
                # Calendar APIs validate these values with Timestamp/int but
                # retain the caller's spelling. Compare their validated meaning.
                try:
                    value = pd.Timestamp(value) if column == "event_start" else int(value)
                except (ValueError, TypeError, OverflowError):
                    # Invalid persisted values retain their exact representation.
                    pass
            elif isinstance(value, str) and column not in {"status", "list_name", "board"}:
                value = value.lower()
            if column == id_column and value not in stable_ids:
                value = ("generated_id",)
            row.append(value)
        rows.append(tuple(row))
    return columns, Counter(rows)


def is_correct(
    predicted_actions: Dict[str, str],
    ground_truth_actions: Dict[str, str],
    error: str,
    *,
    reject_noop_reference: bool = False,
) -> bool:
    """
    Checks if the prediction is correct by comparing the state change after executing the actions.

    Parameters
    ----------
    predicted_actions : list
        List of predicted actions as strings.
    ground_truth_actions : list
        List of ground truth actions as strings.
    error : str
        Error message from the prediction.

    Returns
    -------
    bool
        True if the predicted actions result in the same state change as the ground truth actions
        and send the same emails, counting sent emails that were later deleted.

    """
    if error:
        return False
    predicted_env = execute_actions_and_reset_state(predicted_actions)
    return state_matches_reference(predicted_env, ground_truth_actions, reject_noop_reference=reject_noop_reference)


def state_matches_reference(predicted_env, ground_truth_actions, *, reject_noop_reference=False):
    """Compare server-owned final tables to a fresh reference replay.

    Submitted action lists and assistant reports cannot create predicted effects.
    Missing session tracking is an unavailable verifier, never inferred success.
    """
    if "stable_ids" not in predicted_env or "sent_emails" not in predicted_env:
        raise ValueError("session effect tracking is unavailable")
    reference_env = execute_actions_and_reset_state(ground_truth_actions)
    predicted = _state_tables(predicted_env)
    reference = _state_tables(reference_env)
    if reject_noop_reference:
        seed_env = execute_actions_and_reset_state([])
        seed = _state_tables(seed_env)
        unchanged = True
        for name, (reference_frame, id_column) in reference.items():
            expected = _canonical_rows(reference_frame, id_column, reference_env["stable_ids"][name], name)
            initial = _canonical_rows(seed[name][0], id_column, seed_env["stable_ids"][name], name)
            if expected is None or initial is None:
                return False
            unchanged = unchanged and expected == initial
        if unchanged:
            return False
    for name, (reference_frame, id_column) in reference.items():
        predicted_rows = _canonical_rows(predicted[name][0], id_column, predicted_env["stable_ids"][name], name)
        reference_rows = _canonical_rows(reference_frame, id_column, reference_env["stable_ids"][name], name)
        if predicted_rows is None or reference_rows is None or predicted_rows != reference_rows:
            return False
    # The final mailbox hides a wrong email that was sent and then deleted.
    predicted_sent = _sent_email_rows(predicted_env)
    reference_sent = _sent_email_rows(reference_env)
    return predicted_sent is not None and reference_sent is not None and predicted_sent == reference_sent
