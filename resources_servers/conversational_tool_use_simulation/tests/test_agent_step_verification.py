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

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.base_resources_server import BaseVerifyRequest
from nemo_gym.judge import JudgeError, judge_failsafe
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.conversational_tool_use_simulation.app import (
    ConversationalToolUseSimulationConfig,
    ConversationalToolUseSimulationServer,
    ConversationSessionState,
    CustomerScenario,
    Source,
)


def make_server(**overrides):
    values = dict(
        host="127.0.0.1",
        port=0,
        name="stock_step",
        entrypoint="app.py",
        enable_agent_step_verification=True,
        judge_model_server={"type": "responses_api_models", "name": "judge"},
        judge_responses_create_params={"input": [], "temperature": 0, "max_output_tokens": 4000},
        judge_provider_attempts=1,
    )
    values.update(overrides)
    return ConversationalToolUseSimulationServer(
        config=ConversationalToolUseSimulationConfig(**values), server_client=MagicMock(spec=ServerClient)
    )


def message(text="visible", parts=None):
    return dict(
        type="message",
        id="message",
        role="assistant",
        status="completed",
        content=parts or [dict(type="output_text", text=text, annotations=[])],
    )


def response(output, **overrides):
    values = dict(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=output,
        status="completed",
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )
    values.update(overrides)
    return NeMoGymResponse.model_validate(values)


def tool_call(arguments='{ "value": 1, "label": "USD 7" }', name="write", call_id="call1"):
    return dict(type="function_call", id=call_id, name=name, call_id=call_id, arguments=arguments)


def request(output=None):
    return BaseVerifyRequest.model_validate(
        dict(
            _ng_rollout_id="runtime-id",
            responses_create_params=dict(
                input=[
                    dict(role="developer", content="Follow policy exactly."),
                    dict(role="user", content="User request."),
                    dict(type="function_call", name="lookup", call_id="prior_call", arguments='{"query":"x"}'),
                    dict(type="function_call_output", call_id="prior_call", output='{"amount":"USD 7"}'),
                ],
                instructions="Instruction field.",
                tool_choice="auto",
                parallel_tool_calls=False,
                tools=[
                    dict(
                        type="function",
                        name="write",
                        description="Complete offered definition.",
                        strict=False,
                        parameters=dict(
                            type="object",
                            required=["value", "label"],
                            additionalProperties=False,
                            properties=dict(value=dict(type="number"), label=dict(type="string")),
                        ),
                    )
                ],
            ),
            response=response(output if output is not None else [tool_call()]).model_dump(mode="json"),
        )
    )


def judge(text='{"success":true,"explanation":"grounded"}', **overrides):
    return response([message(text)], **overrides)


def test_named_instance_format_preserves_stock_rubric_and_native_evidence():
    stock = make_server()
    instance = make_server(agent_step_verdict_format="evaluation_instance_v1")
    body = request([message("narration"), tool_call()])
    stock_params = stock.prepare_agent_step_judge_request(body).judge_params
    instance_params = instance.prepare_agent_step_judge_request(body).judge_params
    stock_system = stock_params.input[0].content
    assert instance_params.input[0].content.startswith(stock_system + "\n")
    assert instance_params.input[1:] == stock_params.input[1:]
    assert instance_params.model_dump(exclude={"input"}) == stock_params.model_dump(exclude={"input"})
    assert '"explanation": "string", "success": "boolean"' in instance_params.input[0].content[len(stock_system) :]
    assert instance._agent_step_judge_settings(None)["agent_step_verdict_format"] == "evaluation_instance_v1"
    assert stock._agent_step_judge_settings(None)["agent_step_verdict_format"] == "stock_schema"
    assert instance.parse_agent_step_judge_response(judge()).reward == 1
    with pytest.raises(JudgeError):
        instance.parse_agent_step_judge_response(judge('{"properties":{"success":true,"explanation":"x"}}'))


def test_named_instance_format_requires_enabled_agent_step_mode():
    with pytest.raises(ValidationError, match="requires agent-step verification"):
        make_server(enable_agent_step_verification=False, agent_step_verdict_format="evaluation_instance_v1")


def test_named_evidence_policy_preserves_evidence_and_format_is_independent():
    stock = make_server()
    grounded = make_server(agent_step_evidence_policy="grounded_complete_turn_v1")
    body = request([message("narration"), tool_call()])
    original = body.model_dump(mode="json")
    stock_params = stock.prepare_agent_step_judge_request(body).judge_params
    grounded_params = grounded.prepare_agent_step_judge_request(body).judge_params
    assert grounded_params.input[0].content.startswith(stock_params.input[0].content + "\n")
    assert grounded_params.input[1:] == stock_params.input[1:]
    assert grounded_params.model_dump(exclude={"input"}) == stock_params.model_dump(exclude={"input"})
    assert body.model_dump(mode="json") == original
    settings = grounded._agent_step_judge_settings(None)
    assert settings["agent_step_evidence_policy"] == "grounded_complete_turn_v1"
    assert settings["agent_step_verdict_format"] == "stock_schema"
    assert stock._agent_step_judge_settings(None)["agent_step_evidence_policy"] == "stock"


def test_named_evidence_policy_requires_enabled_agent_step_mode():
    with pytest.raises(ValidationError, match="evidence policy requires agent-step verification"):
        make_server(enable_agent_step_verification=False, agent_step_evidence_policy="grounded_complete_turn_v1")


@pytest.mark.parametrize("status", [None, "queued", "cancelled", "in_progress", "incomplete", "failed"])
def test_noncompleted_candidate_root_returns_zero_without_judge(status):
    body = request()
    body.response = response([tool_call()], status=status)
    prepared = make_server().prepare_agent_step_judge_request(body)
    assert prepared.judge_params is None and prepared.verification_result.reward == 0


def test_candidate_incomplete_details_returns_zero_without_judge():
    body = request()
    body.response = response([tool_call()], incomplete_details=dict(reason="max_output_tokens"))
    prepared = make_server().prepare_agent_step_judge_request(body)
    assert prepared.judge_params is None and prepared.verification_result.reward == 0


@pytest.mark.parametrize("kind", ["message", "function_call"])
@pytest.mark.parametrize("status", ["in_progress", "incomplete"])
def test_incomplete_visible_candidate_item_returns_zero_without_judge(kind, status):
    item = message() if kind == "message" else tool_call()
    item["status"] = status
    prepared = make_server().prepare_agent_step_judge_request(request([item]))
    assert prepared.judge_params is None and prepared.verification_result.reward == 0


def test_completed_root_with_omitted_native_call_status_remains_supported():
    prepared = make_server().prepare_agent_step_judge_request(request([tool_call()]))
    assert prepared.judge_params is not None and prepared.verification_result is None


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_timeout_is_positive_finite(timeout):
    with pytest.raises(ValidationError):
        make_server(agent_step_verification_timeout_seconds=timeout)


@pytest.mark.parametrize(
    "settings",
    [
        dict(input=[], instructions="injected"),
        dict(input=[], previous_response_id="old"),
        dict(input=[], conversation="old"),
        dict(input="injected"),
    ],
)
def test_enabled_judge_settings_cannot_carry_context(settings):
    with pytest.raises(ValidationError):
        make_server(judge_responses_create_params=settings)


def test_enabled_requires_explicit_settings_and_judge_reference():
    values = dict(host="127.0.0.1", port=0, name="step", entrypoint="app.py", enable_agent_step_verification=True)
    with pytest.raises(ValidationError):
        ConversationalToolUseSimulationConfig(**values)
    with pytest.raises(ValidationError):
        ConversationalToolUseSimulationConfig(
            **values, judge_model_server={"type": "responses_api_models", "name": "judge"}
        )


def test_default_off_and_mcp_filter():
    server = make_server(enable_agent_step_verification=False)
    with pytest.raises(HTTPException) as exc:
        server.prepare_agent_step_judge_request(request())
    assert exc.value.status_code == 404
    tool = MagicMock(name="advertised")
    tool.name = "advertised"
    verifier = MagicMock(name="verifier")
    verifier.name = "verify_agent_step"
    assert server.mcp_tools([tool, verifier], None) == [tool]
    paths = [route.path for route in server.setup_webserver().routes]
    assert paths.index("/verify_agent_step") < paths.index("/{tool_name}")


def test_complete_projection_and_no_session_mutation_or_hidden_reasoning():
    server = make_server()
    hidden = dict(type="reasoning", id="private", summary=[dict(type="summary_text", text="HIDDEN_REASONING")])
    body = request(
        [
            hidden,
            message(
                parts=[
                    dict(type="output_text", text="narration1", annotations=[]),
                    dict(type="refusal", refusal="refusal2"),
                ]
            ),
            tool_call(call_id="first"),
            tool_call(call_id="second"),
        ]
    )
    initial = body.model_dump(mode="json")
    prepared = server.prepare_agent_step_judge_request(body)
    system, user = [item.content for item in prepared.judge_params.input]
    assert "Follow policy exactly." in system and "Instruction field." in system
    assert "Complete offered definition." in system and '"additionalProperties": false' in system
    for token in ("User request.", "prior_call", "USD 7", "narration1", "refusal2", "first", "second"):
        assert token in user
    assert user.index("narration1") < user.index("first") < user.index("second")
    assert "HIDDEN_REASONING" not in system + user
    assert body.model_dump(mode="json") == initial
    assert server.session_id_to_state == {}
    assert prepared.judge_params.temperature == 0 and prepared.judge_params.max_output_tokens == 4000
    assert prepared.judge_params.tools == [] and prepared.judge_params.tool_choice == "none"


@pytest.mark.parametrize(
    "arguments",
    ['{"value":1,"label":"USD 7"}', '{ "label" : "USD 7", "value" : 1 }', '{\n"value":1,\n"label":"USD 7"\n}'],
)
def test_argument_layout_is_canonical_without_changing_raw_receipt(arguments):
    server = make_server()
    body = request([tool_call(arguments)])
    expected = server.prepare_agent_step_judge_request(request()).judge_params.model_dump(mode="json")
    assert server.prepare_agent_step_judge_request(body).judge_params.model_dump(mode="json") == expected
    assert body.response.output[0].arguments == arguments


@pytest.mark.parametrize(
    "parts",
    [
        [dict(type="output_text", text="", annotations=[])],
        [dict(type="output_text", text=" \n\t", annotations=[])],
        [dict(type="output_text", text=" ", annotations=[]), dict(type="output_text", text="\n", annotations=[])],
    ],
)
@pytest.mark.parametrize("blank_first", [True, False])
@pytest.mark.asyncio
async def test_blank_transport_messages_preserve_call_only_judgment_and_raw_receipt(parts, blank_first):
    server = make_server()
    output = [message(parts=parts), tool_call()]
    if not blank_first:
        output.reverse()
    body = request(output)
    original = body.model_dump(mode="json")
    expected = server.prepare_agent_step_judge_request(request([tool_call()])).judge_params
    prepared = server.prepare_agent_step_judge_request(body)
    assert prepared.judge_params == expected and prepared.verification_result is None
    server._call_judge_model = AsyncMock(return_value=judge())
    result = await server.verify_agent_step(body)
    assert result.reward == 1 and result.judge_request == expected
    assert result.response == body.response and len(result.response.output) == 2
    assert body.model_dump(mode="json") == original
    server._call_judge_model.assert_awaited_once()


@pytest.mark.parametrize("refusal", ["", " ", "refused"])
def test_refusal_is_never_filtered_as_blank_transport(refusal):
    server = make_server()
    body = request([message(parts=[dict(type="refusal", refusal=refusal)]), tool_call()])
    prepared = server.prepare_agent_step_judge_request(body)
    call_only = server.prepare_agent_step_judge_request(request([tool_call()])).judge_params
    assert prepared.judge_params != call_only
    assert '"type": "refusal"' in prepared.judge_params.input[1].content


@pytest.mark.parametrize(
    "output",
    [
        [],
        [dict(type="reasoning", id="private", summary=[])],
        [message("   ")],
        [message("")],
        [message(""), message(" \n\t")],
        [tool_call(name="unknown")],
        [tool_call(arguments="{bad")],
        [tool_call(arguments='{"value":"1","label":"USD 7"}')],
        [tool_call(arguments='{"value":1,"value":2,"label":"USD 7"}')],
        [tool_call(arguments='{"value":NaN,"label":"USD 7"}')],
    ],
)
@pytest.mark.asyncio
async def test_invalid_policy_is_zero_without_judge_call(output):
    server = make_server()
    server._call_judge_model = AsyncMock()
    body = request(output)
    result = await server.verify_agent_step(body)
    assert result.reward == 0 and result.verification_result.reward == 0
    assert result.response == body.response and result.judge_request is None
    server._call_judge_model.assert_not_awaited()


@pytest.mark.parametrize(
    "text",
    [
        "bad",
        "null",
        "[]",
        "1",
        '{"success":"true","explanation":"x"}',
        '{"success":true}',
        '{"success":true,"explanation":"x","extra":1}',
        '{"success":false,"success":true,"explanation":"x"}',
    ],
)
def test_malformed_or_unsupported_verdict_is_judge_failure(text):
    server = make_server()
    reply = judge(text)
    with pytest.raises(JudgeError) as exc:
        server.parse_agent_step_judge_response(reply)
    assert json.loads(str(exc.value))["judge_response"] == reply.model_dump(mode="json")


@pytest.mark.parametrize(
    "overrides",
    [
        dict(status="failed"),
        dict(status="incomplete"),
        dict(error=dict(code="server_error", message="failed")),
        dict(incomplete_details=dict(reason="max_output_tokens")),
    ],
)
def test_parseable_bad_judge_envelope_never_pays(overrides):
    with pytest.raises(JudgeError):
        make_server().parse_agent_step_judge_response(judge(**overrides))


def test_multiple_judge_parts_and_refusal_are_unsupported():
    server = make_server()
    for reply in [
        response([message(), message()]),
        response([message(parts=[dict(type="refusal", refusal="no")])]),
        response(
            [
                message(
                    parts=[
                        dict(type="output_text", text="{}", annotations=[]),
                        dict(type="output_text", text="{}", annotations=[]),
                    ]
                )
            ]
        ),
    ]:
        with pytest.raises(JudgeError):
            server.parse_agent_step_judge_response(reply)


@pytest.mark.parametrize("success", [True, False])
@pytest.mark.asyncio
async def test_production_and_offline_helpers_are_identical_with_one_call(success):
    server = make_server()
    body = request([message("narration"), tool_call()])
    reply = judge(json.dumps(dict(success=success, explanation="evaluated")))
    server._call_judge_model = AsyncMock(return_value=reply)
    prepared = server.prepare_agent_step_judge_request(body)
    expected = server.parse_agent_step_judge_response(reply)
    actual = await server.verify_agent_step(body)
    assert actual.verification_result == expected
    assert actual.reward == int(success)
    assert actual.judge_request == prepared.judge_params
    assert actual.response == body.response and actual.responses_create_params == body.responses_create_params
    server._call_judge_model.assert_awaited_once()
    call = server._call_judge_model.call_args.kwargs
    assert call["params"] == prepared.judge_params and call["messages"] == prepared.judge_params.input
    assert call["rollout_id"] == "runtime-id" and actual.judge_settings["strict_agent_step"] is True
    assert server.session_id_to_state == {}


@pytest.mark.asyncio
async def test_failure_failsafe_retains_body_effective_request_and_actual_reply():
    server = make_server()
    body = request()
    reply = judge(status="incomplete")
    server._call_judge_model = AsyncMock(return_value=reply)
    result = await judge_failsafe(server.verify_agent_step)(body)
    receipt = json.loads(result.body)
    assert receipt["_ng_failure_class"] == "judge_failed"
    assert receipt["response"] == body.response.model_dump(mode="json")
    failure = json.loads(receipt["_ng_failure_judge_error"])
    assert failure["judge_response"] == reply.model_dump(mode="json")
    assert failure["judge_request"] == server.prepare_agent_step_judge_request(body).judge_params.model_dump(
        mode="json"
    )
    server._call_judge_model.assert_awaited_once()


@pytest.mark.asyncio
async def test_total_deadline_is_judge_failure_with_request_retained():
    server = make_server(agent_step_verification_timeout_seconds=0.01)

    async def blocked(**kwargs):
        await asyncio.Event().wait()

    server._call_judge_model = AsyncMock(side_effect=blocked)
    with pytest.raises(JudgeError) as exc:
        await server.verify_agent_step(request())
    assert json.loads(str(exc.value))["reason"] == "TimeoutError"
    assert json.loads(str(exc.value))["judge_request"]["max_output_tokens"] == 4000
    server._call_judge_model.assert_awaited_once()


@pytest.mark.asyncio
async def test_legacy_full_conversation_retains_parseable_incomplete_and_generation_error_behavior():
    server = make_server(enable_agent_step_verification=False, generation_attempts=1)
    state = ConversationSessionState(
        domain_name="domain", policy="policy", tool_signatures=[], customer_scenario=CustomerScenario()
    )
    server._call_judge_model = AsyncMock(return_value=judge(status="incomplete"))
    legacy = await server._generate_judge_evaluation(Source.AGENT, "step", state)
    assert legacy.reward == 1
    server._call_judge_model = AsyncMock(return_value=judge("bad"))
    legacy = await server._generate_judge_evaluation(Source.AGENT, "step", state)
    assert legacy.reward is None and legacy.generation_error


def test_real_asgi_endpoint_uses_failsafe_before_catchall():
    server = make_server()
    server._call_judge_model = AsyncMock(return_value=judge(status="failed"))
    with TestClient(server.setup_webserver()) as client:
        result = client.post("/verify_agent_step", json=request().model_dump(mode="json"))
    assert result.status_code == 200 and result.json()["_ng_failure_class"] == "judge_failed"
    assert result.json()["response"] == request().response.model_dump(mode="json")


def test_v2_evidence_audit_preserves_actual_evidence_and_default_settings():
    body = request([message("all narration"), tool_call()])
    initial = body.model_dump(mode="json")
    old = make_server(agent_step_evidence_policy="grounded_complete_turn_v1")
    new = make_server(agent_step_evidence_policy="grounded_complete_turn_v2")
    a = old.prepare_agent_step_judge_request(body).judge_params
    b = new.prepare_agent_step_judge_request(body).judge_params
    assert b.input[1:] == a.input[1:]
    assert b.model_dump(exclude={"input"}) == a.model_dump(exclude={"input"})
    assert body.model_dump(mode="json") == initial
    assert new._agent_step_judge_settings(None)["agent_step_evidence_policy"] == "grounded_complete_turn_v2"
    assert make_server().config.agent_step_evidence_policy == "stock"
    assert make_server().config.agent_step_verdict_format == "stock_schema"


def test_instance_v3_preserves_evidence_and_all_transport_fields():
    from resources_servers.conversational_tool_use_simulation.app import Evaluation

    body = request([message("narration"), tool_call()])
    original = body.model_dump(mode="json")
    stock = make_server().prepare_agent_step_judge_request(body).judge_params
    server = make_server(agent_step_verdict_format="evaluation_instance_v3")
    prepared = server.prepare_agent_step_judge_request(body).judge_params
    assert prepared.input[1:] == stock.input[1:]
    assert prepared.model_dump(exclude={"input"}) == stock.model_dump(exclude={"input"})
    assert body.model_dump(mode="json") == original
    assert json.dumps(Evaluation.model_json_schema()) not in prepared.input[0].content
    assert server.parse_agent_step_judge_response(judge()).reward == 1
    with pytest.raises(JudgeError):
        server.parse_agent_step_judge_response(judge('{"properties":{"success":true,"explanation":"x"}}'))
    assert server._agent_step_judge_settings(None)["agent_step_verdict_format"] == "evaluation_instance_v3"


@pytest.mark.parametrize("policy", ["stock", "grounded_complete_turn_v2", "grounded_complete_turn_v3"])
def test_instance_v4_preserves_rubric_evidence_transport_and_strict_parser(policy):
    body = request([message("narration"), tool_call()])
    original = body.model_dump(mode="json")
    stock = make_server(agent_step_evidence_policy=policy).prepare_agent_step_judge_request(body).judge_params
    server = make_server(agent_step_verdict_format="evaluation_instance_v4", agent_step_evidence_policy=policy)
    prepared = server.prepare_agent_step_judge_request(body).judge_params
    assert prepared.input[1:] == stock.input[1:]
    assert prepared.model_dump(exclude={"input"}) == stock.model_dump(exclude={"input"})
    assert body.model_dump(mode="json") == original
    expected_rubric = stock.input[0].content.split("\n\nPlease output the evaluation", 1)[0]
    assert prepared.input[0].content.startswith(expected_rubric + "\n")
    assert "using the following JSON schema" not in prepared.input[0].content
    assert server.parse_agent_step_judge_response(judge()).reward == 1
    assert server.parse_agent_step_judge_response(judge('{"explanation":"violation","success":false}')).reward == 0
    for text in ['{"evaluation":"x","success":true}', '<evaluation>x</evaluation>{"explanation":"x","success":true}']:
        with pytest.raises(JudgeError):
            server.parse_agent_step_judge_response(judge(text))


def test_v3_authority_policy_preserves_optional_tool_parameter_and_source_evidence():
    body = request()
    tool = body.responses_create_params.tools[0]
    tool["parameters"]["properties"]["api_key"] = {"type": "string"}
    body.responses_create_params.input.insert(2, message("I believe all these tools require an API key."))
    old = make_server(agent_step_evidence_policy="grounded_complete_turn_v2").prepare_agent_step_judge_request(body)
    server = make_server(agent_step_evidence_policy="grounded_complete_turn_v3")
    new = server.prepare_agent_step_judge_request(body)
    assert new.judge_params.input[1:] == old.judge_params.input[1:]
    assert new.judge_params.model_dump(exclude={"input"}) == old.judge_params.model_dump(exclude={"input"})
    assert tool["parameters"]["required"] == ["value", "label"]
    assert body.response.output[0].arguments == '{ "value": 1, "label": "USD 7" }'
    assert server._agent_step_judge_settings(None)["agent_step_evidence_policy"] == "grounded_complete_turn_v3"
