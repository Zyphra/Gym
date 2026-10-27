# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemo_gym.base_responses_api_model import ModelCallRecord
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ModelCallRef,
    ObservationGap,
    SandboxObservation,
    ToolCallObservation,
    TrajectoryModelCall,
    TrajectoryRecord,
    bounded_error_text,
    endpoint_stop_reason,
    harness_response_error,
    join_model_call_observations,
)


@pytest.mark.parametrize("index", [0, 1], ids=["pi-no-call-failure", "hermes-route-denial"])
def test_saved_harness_failure_survives_response_projection(index: int) -> None:
    raw = json.loads((Path(__file__).parent / "fixtures/sandbox-harness-failures.json").read_text())[index]["response"]
    observations = AgentObservationBundle.model_validate(raw.pop("_ng_agent_observations"))
    response = NeMoGymResponse.model_validate(raw)
    error = harness_response_error(response, observations)
    assert error is not None
    if response.error is not None:
        assert error == response.error


@pytest.mark.parametrize("status,calls", [("completed", []), ("failed", [ModelCallRef(model_call_id="empty-stop")])])
def test_empty_model_stop_is_not_a_zero_call_harness_failure(status: str, calls: list[ModelCallRef]) -> None:
    raw = json.loads((Path(__file__).parent / "fixtures/sandbox-harness-failures.json").read_text())[0]["response"]
    raw.pop("_ng_agent_observations")
    response = NeMoGymResponse.model_validate(raw)
    observations = AgentObservationBundle(source="test", records=[AgentInvocation(
        invocation_id="root", status=status, model_calls=calls,
    )])
    assert harness_response_error(response, observations) is None
    assert harness_response_error(response) is None  # Missing evidence is not a zero-call receipt.


@pytest.mark.parametrize("message", [
    "Model used all output tokens on reasoning with none left for the response. Try lowering reasoning effort or increasing max_tokens.",
    "Model generated invalid tool call: finish",
    "Context length exceeded: max compression attempts (3) reached.",
])
def test_policy_errors_remain_scored_outcomes(message: str) -> None:
    raw = json.loads((Path(__file__).parent / "fixtures/sandbox-harness-failures.json").read_text())[1]["response"]
    observations = AgentObservationBundle.model_validate(raw.pop("_ng_agent_observations"))
    raw["error"]["message"] = message
    raw["usage"]["total_tokens"] = 100
    raw["usage"]["input_tokens"] = 100
    response = NeMoGymResponse.model_validate(raw)
    assert harness_response_error(response, observations) is None


def _failed_response(message: str, **metadata: str) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate({
        "id": "r", "created_at": 0, "model": "policy", "object": "response", "output": [],
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], "status": "failed",
        "error": {"code": "server_error", "message": message}, "metadata": metadata,
    })


@pytest.mark.parametrize("response", [
    _failed_response("Connection error."),
    _failed_response("Connection error"),
    _failed_response("anything", model_transport_error="true"),
    _failed_response("Error code: 503 - upstream"),
], ids=["sdk-connection-error", "sdk-connection-error-no-period", "declared-transport", "http-503"])
def test_an_unreachable_or_failing_endpoint_is_a_harness_failure(response: NeMoGymResponse) -> None:
    assert harness_response_error(response) == response.error


def test_a_failed_setup_label_is_a_harness_failure_without_an_error() -> None:
    response = NeMoGymResponse.model_validate({
        "id": "r", "created_at": 0, "model": "policy", "object": "response", "output": [],
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], "status": "completed",
        "metadata": {"agent_error": "setup_failed:tmux_unavailable"},
    })
    error = harness_response_error(response)
    assert error is not None and error.message == "setup_failed:tmux_unavailable"


@pytest.mark.parametrize("response", [
    _failed_response("Connection error while parsing the policy's tool call"),
    _failed_response("Request timed out."),
    _failed_response("HTTP 400: This episode has ended (empty_turns); no further turns are served."),
    _failed_response("Model generated invalid tool call: finish"),
    _failed_response("RuntimeError", agent_error="RuntimeError"),
], ids=["lookalike", "client-timeout", "episode-ended", "invalid-tool-call", "exception-class"])
def test_a_cause_that_may_be_the_policys_is_not_a_harness_failure(response: NeMoGymResponse) -> None:
    assert harness_response_error(response) is None


def test_bounded_error_text_redacts_secrets_and_bounds_length() -> None:
    assert bounded_error_text("Error code: 400\n  - bad   request") == "Error code: 400 - bad request"
    redacted = bounded_error_text("failed with Authorization: Bearer sk-abc123 and api_key=sk-xyz")
    assert "sk-abc123" not in redacted and "sk-xyz" not in redacted and "[redacted]" in redacted
    assert len(bounded_error_text("x" * 1000)) == 300
    assert bounded_error_text("a\x00b\x1bc") == "abc"
    assert bounded_error_text("   ") is None and bounded_error_text(None) is None and bounded_error_text(7) is None


def test_endpoint_stop_reason_reads_only_the_refusal_code() -> None:
    refusal = 'b\'{"error":{"code":"episode_terminated"}}\''
    assert endpoint_stop_reason(None, "other", refusal) == "episode_terminated"
    assert endpoint_stop_reason("HTTP 400: invalid tool schema", None, 7) is None


@pytest.mark.parametrize(
    "value",
    ({}, {"response_id": "resp-1"}, {"model_ref": {"name": "policy", "type": "responses_api_models"}}),
)
def test_model_call_ref_rejects_incomplete_join_keys(value: dict) -> None:
    with pytest.raises(ValidationError, match="model_call_id or both model_ref and response_id"):
        ModelCallRef.model_validate(value)


def test_observation_bundle_rejects_duplicate_invocation_ids() -> None:
    with pytest.raises(ValidationError, match="invocation_id must be unique"):
        AgentObservationBundle(
            source="test",
            records=[AgentInvocation(invocation_id="root"), AgentInvocation(invocation_id="root")],
        )


@pytest.mark.parametrize(
    "records",
    (
        [AgentInvocation(invocation_id="root", parent_invocation_id="root")],
        [
            AgentInvocation(invocation_id="a", parent_invocation_id="b"),
            AgentInvocation(invocation_id="b", parent_invocation_id="a"),
        ],
    ),
)
def test_observation_bundle_rejects_parent_cycles(records: list[AgentInvocation]) -> None:
    with pytest.raises(ValidationError, match="parent_invocation_id must not form a cycle"):
        AgentObservationBundle(source="test", records=records)


def test_observation_bundle_allows_missing_parent() -> None:
    bundle = AgentObservationBundle(
        source="test",
        records=[AgentInvocation(invocation_id="child", parent_invocation_id="unobserved")],
    )
    assert bundle.records[0].parent_invocation_id == "unobserved"


def test_observation_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="producer_extension"):
        ModelCallRef.model_validate({"model_call_id": "call-1", "producer_extension": "unexpected"})
    with pytest.raises(ValidationError, match="output"):
        ToolCallObservation.model_validate(
            {"invocation_id": "root", "tool_call_id": "tool-1", "output": "trajectory-only"}
        )


def test_turns_are_versioned_trajectory_records_not_legacy_observations() -> None:
    with pytest.raises(ValidationError, match="turn"):
        AgentObservationBundle.model_validate({"source": "test", "records": [{"kind": "turn"}]})


def test_trajectory_rejects_duplicate_model_call_ids() -> None:
    with pytest.raises(ValidationError, match="model_call_id must be unique"):
        TrajectoryRecord(
            task_id="task",
            rollout_id="rollout",
            model_calls=[TrajectoryModelCall(model_call_id="call-1"), TrajectoryModelCall(model_call_id="call-1")],
        )


@pytest.mark.parametrize(
    "timing",
    (
        {"duration_ms": -1},
        {"started_at": 2.0, "completed_at": 1.0},
    ),
)
def test_tool_call_observation_rejects_invalid_timing(timing: dict) -> None:
    with pytest.raises(ValidationError):
        ToolCallObservation(invocation_id="root", tool_call_id="call-1", **timing)


def test_agent_invocation_rejects_negative_duration() -> None:
    with pytest.raises(ValidationError):
        AgentInvocation(invocation_id="root", duration_ms=-1)


@pytest.mark.parametrize(
    "values",
    (
        {"tokens_before": -1},
        {"tokens_after": -1},
    ),
)
def test_context_compaction_rejects_invalid_token_counts(values: dict) -> None:
    with pytest.raises(ValidationError):
        ContextCompactionObservation(invocation_id="root", **values)


def test_context_compaction_preserves_producer_reported_token_counts() -> None:
    observation = ContextCompactionObservation(
        invocation_id="root",
        tokens_before=10,
        tokens_after=11,
        outcome="completed",
    )
    assert observation.tokens_after == 11


def test_join_model_calls_resolves_exact_references_and_reports_unowned_calls() -> None:
    model_ref = ModelServerRef(name="policy", type="responses_api_models")
    bundle = AgentObservationBundle(
        source="test",
        records=[
            AgentInvocation(
                invocation_id="root",
                model_calls=[ModelCallRef(model_ref=model_ref, response_id="resp-1")],
            )
        ],
        gaps=[
            ObservationGap(code="model_call_ownership_unavailable", detail="direct_provider"),
            ObservationGap(code="model_call_ownership_unavailable", detail="capture:stale"),
        ],
    )
    calls = [
        ModelCallRecord(
            model_call_id="call-1",
            response_id="resp-1",
            model_ref=model_ref,
            call_index=0,
        ),
        ModelCallRecord(model_call_id="call-2", model_ref=model_ref, call_index=1),
    ]

    joined = join_model_call_observations(bundle, calls)

    [invocation] = [record for record in joined.records if isinstance(record, AgentInvocation)]
    [joined_call] = invocation.model_calls
    assert joined_call.model_call_id == "call-1"
    assert joined_call.model_ref == model_ref
    assert joined_call.response_id == "resp-1"
    ownership_gaps = [gap for gap in joined.gaps if gap.code == "model_call_ownership_unavailable"]
    assert [gap.detail for gap in ownership_gaps] == ["direct_provider", "capture:call-2:call_index=1"]


def test_join_model_calls_uses_exact_client_session_ownership() -> None:
    bundle = AgentObservationBundle(
        source="test",
        records=[
            AgentInvocation(invocation_id="root"),
            AgentInvocation(invocation_id="child", parent_invocation_id="root"),
        ],
    )
    calls = [
        ModelCallRecord(model_call_id="call-root", client_session_id="root", call_index=0),
        ModelCallRecord(model_call_id="call-child", client_session_id="child", call_index=1),
        ModelCallRecord(model_call_id="call-unknown", client_session_id="unknown", call_index=2),
    ]

    joined = join_model_call_observations(bundle, calls)
    joined_again = join_model_call_observations(joined, calls)

    assert all(not invocation.model_calls for invocation in bundle.records)
    assert joined_again.model_dump() == joined.model_dump()
    invocations = {record.invocation_id: record for record in joined.records if isinstance(record, AgentInvocation)}
    assert [reference.model_call_id for reference in invocations["root"].model_calls] == ["call-root"]
    assert [reference.model_call_id for reference in invocations["child"].model_calls] == ["call-child"]
    assert [gap.detail for gap in joined.gaps if gap.code == "model_call_ownership_unavailable"] == [
        "capture:call-unknown:call_index=2"
    ]


def test_client_session_association_does_not_mutate_or_retain_phantom_calls() -> None:
    bundle = AgentObservationBundle(source="test", records=[AgentInvocation(invocation_id="root")])
    first = [ModelCallRecord(model_call_id="first", client_session_id="root", call_index=0)]
    second = [ModelCallRecord(model_call_id="second", client_session_id="root", call_index=0)]

    first_result = join_model_call_observations(bundle, first)
    second_result = join_model_call_observations(bundle, second)

    assert not bundle.records[0].model_calls
    assert [reference.model_call_id for reference in first_result.records[0].model_calls] == ["first"]
    assert [reference.model_call_id for reference in second_result.records[0].model_calls] == ["second"]


def test_client_session_association_skips_calls_without_a_session_id() -> None:
    bundle = AgentObservationBundle(source="test", records=[AgentInvocation(invocation_id="")])
    calls = [ModelCallRecord(model_call_id="unowned", call_index=0)]

    joined = join_model_call_observations(bundle, calls)

    assert not joined.records[0].model_calls
    assert [gap.detail for gap in joined.gaps if gap.code == "model_call_ownership_unavailable"] == [
        "capture:unowned:call_index=0"
    ]


def test_client_session_association_deduplicates_existing_exact_references() -> None:
    model_ref = ModelServerRef(name="policy", type="responses_api_models")
    bundle = AgentObservationBundle(
        source="test",
        records=[
            AgentInvocation(
                invocation_id="root",
                model_calls=[ModelCallRef(model_ref=model_ref, response_id="response")],
            )
        ],
    )
    calls = [
        ModelCallRecord(
            model_call_id="call",
            client_session_id="root",
            model_ref=model_ref,
            response_id="response",
            call_index=0,
        )
    ]

    associated = join_model_call_observations(bundle, calls)

    assert len(associated.records[0].model_calls) == 1


def test_client_session_association_scales_to_long_rollouts() -> None:
    bundle = AgentObservationBundle(source="test", records=[AgentInvocation(invocation_id="root")])
    calls = [
        ModelCallRecord(model_call_id=f"call-{index}", client_session_id="root", call_index=index)
        for index in range(2_000)
    ]

    associated = join_model_call_observations(bundle, calls)

    assert len(associated.records[0].model_calls) == len(calls)


def test_join_model_calls_does_not_guess_ambiguous_response_ids() -> None:
    model_ref = ModelServerRef(name="policy", type="responses_api_models")
    bundle = AgentObservationBundle(
        source="test",
        records=[
            AgentInvocation(
                invocation_id="root",
                model_calls=[ModelCallRef(model_ref=model_ref, response_id="resp-1")],
            )
        ],
    )
    calls = [
        ModelCallRecord(model_call_id="call-1", response_id="resp-1", model_ref=model_ref, call_index=0),
        ModelCallRecord(model_call_id="call-2", response_id="resp-1", model_ref=model_ref, call_index=1),
    ]

    joined = join_model_call_observations(bundle, calls)

    [invocation] = [record for record in joined.records if isinstance(record, AgentInvocation)]
    assert invocation.model_calls[0].model_call_id is None
    assert "model_call_reference_ambiguous" in {gap.code for gap in joined.gaps}
    assert [gap.detail for gap in joined.gaps if gap.code == "model_call_ownership_unavailable"] == [
        "capture:call-1:call_index=0",
        "capture:call-2:call_index=1",
    ]


def test_join_model_calls_reports_conflicting_and_unmatched_references() -> None:
    model_ref = ModelServerRef(name="policy", type="responses_api_models")
    bundle = AgentObservationBundle(
        source="test",
        records=[
            AgentInvocation(
                invocation_id="root",
                model_calls=[
                    ModelCallRef(model_call_id="call-1"),
                    ModelCallRef(model_ref=model_ref, response_id="resp-1"),
                    ModelCallRef(model_call_id="missing"),
                ],
            )
        ],
        gaps=[
            ObservationGap(
                code="model_call_ownership_unavailable",
                invocation_id="root",
                detail="producer gap",
            )
        ],
    )
    calls = [
        ModelCallRecord(
            model_call_id="call-1",
            response_id="resp-1",
            model_ref=model_ref,
            call_index=0,
        )
    ]

    joined = join_model_call_observations(bundle, calls)
    joined_again = join_model_call_observations(joined, calls)

    assert joined_again.model_dump() == joined.model_dump()
    assert {gap.code for gap in joined.gaps} == {
        "model_call_ownership_unavailable",
        "model_call_reference_conflict",
        "model_call_reference_unmatched",
    }
    assert any(gap.detail == "producer gap" for gap in joined.gaps)


def test_join_model_calls_resolves_compaction_boundaries() -> None:
    model_ref = ModelServerRef(name="policy", type="responses_api_models")
    references = {
        response_id: ModelCallRef(model_ref=model_ref, response_id=response_id)
        for response_id in ("before", "compaction", "after")
    }
    bundle = AgentObservationBundle(
        source="test",
        records=[
            AgentInvocation(
                invocation_id="root",
                model_calls=list(references.values()),
            ),
            ContextCompactionObservation(
                invocation_id="root",
                before_model_call=references["before"],
                model_calls=[references["compaction"]],
                after_model_call=references["after"],
            ),
        ],
    )
    calls = [
        ModelCallRecord(
            model_call_id="call-before",
            response_id="before",
            model_ref=model_ref,
            call_index=0,
        ),
        ModelCallRecord(
            model_call_id="call-compaction",
            response_id="compaction",
            model_ref=model_ref,
            call_index=1,
        ),
        ModelCallRecord(
            model_call_id="call-after",
            response_id="after",
            model_ref=model_ref,
            call_index=2,
        ),
    ]

    joined = join_model_call_observations(bundle, calls)
    [compaction] = [record for record in joined.records if isinstance(record, ContextCompactionObservation)]

    assert compaction.before_model_call.model_call_id == "call-before"
    assert compaction.model_calls[0].model_call_id == "call-compaction"
    assert compaction.after_model_call.model_call_id == "call-after"
    assert joined.gaps == []


def test_join_model_calls_rejects_cross_invocation_compaction_ownership() -> None:
    model_ref = ModelServerRef(name="policy", type="responses_api_models")
    owned = ModelCallRef(model_ref=model_ref, response_id="owned")
    unowned = ModelCallRef(model_ref=model_ref, response_id="unowned")
    bundle = AgentObservationBundle(
        source="test",
        records=[
            AgentInvocation(invocation_id="root"),
            AgentInvocation(invocation_id="other", model_calls=[owned]),
            ContextCompactionObservation(
                invocation_id="root",
                before_model_call=owned,
                model_calls=[owned, unowned],
            ),
        ],
    )
    calls = [
        ModelCallRecord(model_call_id="call-owned", response_id="owned", model_ref=model_ref, call_index=0),
        ModelCallRecord(model_call_id="call-unowned", response_id="unowned", model_ref=model_ref, call_index=1),
    ]

    joined = join_model_call_observations(bundle, calls)
    compaction = next(record for record in joined.records if isinstance(record, ContextCompactionObservation))

    assert compaction.before_model_call.model_call_id is None
    assert all(reference.model_call_id is None for reference in compaction.model_calls)
    assert {(gap.code, gap.detail) for gap in joined.gaps if gap.invocation_id == "root"} == {
        ("model_call_reference_conflict", "before_model_call:call-owned"),
        ("model_call_reference_conflict", "model_calls:call-owned"),
        ("model_call_ownership_unavailable", "model_calls:call-unowned"),
    }


def test_sandbox_observation_rejects_negative_usage() -> None:
    with pytest.raises(ValidationError):
        SandboxObservation(role="agent", cpu_time_s=-1)
