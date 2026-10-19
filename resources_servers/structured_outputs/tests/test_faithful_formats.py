# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""faithful_format_grading: each format is graded by its own faithful representations; off grades as before."""
import asyncio
import json
from nemo_gym.config_types import BaseServerConfig
from omegaconf import OmegaConf
from jsonschema import Draft202012Validator

import pytest

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.structured_outputs.app import (
    StructuredOutputsResourcesServer,
    StructuredOutputsResourcesServerConfig,
    StructuredOutputsVerifyRequest,
)

PERSON = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "born": {"type": "string"},
        "age": {"type": ["integer", "null"]},
        "medal": {"type": ["string", "null"], "enum": ["Gold", "Silver", None]},
        "location": {"type": "object", "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
                     "required": ["city"]},
        "towns": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"},
                  "km": {"type": "number"}}, "required": ["name"]}},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "location"],
    "additionalProperties": False,
}


def server(faithful: bool) -> StructuredOutputsResourcesServer:
    cfg = StructuredOutputsResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="",
                                                 faithful_format_grading=faithful)
    return StructuredOutputsResourcesServer(config=cfg, server_client=ServerClient(head_server_config=BaseServerConfig(host="127.0.0.1", port=0), global_config_dict=OmegaConf.create({})))


def grade(srv, text, schema_type="json", schema=PERSON, fences_allowed=None):
    return srv.evaluate_structured_output_response(schema_type, json.dumps(schema), text, fences_allowed=fences_allowed)[0]


def test_default_off_keeps_every_stock_verdict():
    srv = server(False)
    assert grade(srv, json.dumps({"name": "A", "location": {"city": "B"}})) == 0.0  # stock strictify requires all
    assert grade(srv, "\n\n<?xml version='1.0'?><r><name>A</name></r>", "xml") == 0.0
    assert grade(srv, "name,location.city\nA,B", "csv") == 0.0


def test_required_list_is_the_schemas_own_and_extras_stay_forbidden():
    srv = server(True)
    assert grade(srv, json.dumps({"name": "A", "location": {"city": "B"}})) == 1.0
    assert grade(srv, json.dumps({"name": "A"})) == 0.0  # location is required by the schema
    assert grade(srv, json.dumps({"name": "A", "location": {"city": "B"}, "x": 1})) == 0.0
    allows = {**PERSON, "additionalProperties": True}
    assert grade(srv, json.dumps({"name": "A", "location": {"city": "B"}, "x": 1}), schema=allows) == 1.0


def test_optional_fields_and_impossible_required_names_preserve_the_schema():
    srv = server(True)
    loose = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": ["string", "null"]}}}
    assert grade(srv, "{}", schema=loose) == 1.0
    assert grade(srv, json.dumps({"a": "x"}), schema=loose) == 1.0  # nullable b may be omitted
    empty_list = {**loose, "required": []}
    assert grade(srv, "{}", schema=empty_list) == 1.0
    undeclared = {**loose, "required": ["a", "ghost"], "additionalProperties": False}
    assert grade(srv, json.dumps({"a": "x"}), schema=undeclared) == 0.0
    nested = {"type": "object", "properties": {"loc": {"type": "object", "properties": {"c": {"type": ["string", "null"]}}}},
              "required": ["loc"]}
    assert grade(srv, "<r><loc/></r>", "xml", nested) == 1.0  # an empty element is an empty object
    assert grade(srv, "<r><a>x</a></r>", "xml", {**loose, "required": ["a", "b"]}) == 1.0  # omitted nullable = null
    assert grade(srv, "a,b\n,\n", "csv", {**loose, "required": ["b"]}) == 1.0  # a row of empty cells is a row
    named = {"type": "object", "properties": {"required": {"type": "boolean"}, "properties": {"type": "string"}},
             "required": ["required"]}
    assert grade(srv, json.dumps({"required": True}), schema=named) == 1.0  # field names are not keywords


def test_fences_unwrap_only_where_the_row_allows_them():
    srv = server(True)
    fenced = "```yaml\nname: A\nlocation:\n  city: B\n```"
    assert grade(srv, fenced, "yaml", fences_allowed=True) == 1.0
    assert grade(srv, fenced, "yaml", fences_allowed=False) == 0.0
    assert grade(srv, fenced, "yaml") == 0.0
    assert grade(srv, "\n\nname: A\nlocation:\n  city: B\n\n", "yaml") == 1.0


def test_yaml_dates_stay_text():
    srv = server(True)
    assert grade(srv, "name: A\nborn: 1974-05-30\nlocation:\n  city: B", "yaml") == 1.0
    assert grade(server(False), "name: A\nborn: 1974-05-30\nlocation:\n  city: B\nage: 1\nmedal: Gold\n"
                 "towns: []\ntags: []", "yaml") == 0.0


def test_xml_root_unions_arrays_and_digit_names():
    srv = server(True)
    xml = ("\n\n<?xml version=\"1.0\"?>\n<response><name>A</name><age>49</age><medal/>"
           "<location><city>B</city></location><towns><town><name>T</name><km>2.5</km></town>"
           "<town><name>U</name><km>3</km></town></towns><tags><tag>x</tag></tags></response>")
    assert grade(srv, xml, "xml") == 1.0
    assert grade(srv, "<response><name>A</name><age>old</age><location><city>B</city></location></response>", "xml") == 0.0
    arr = {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}}
    assert grade(srv, "<items><item><id>1</id></item><item><id>2</id></item></items>", "xml", arr) == 1.0
    digit = {"type": "object", "properties": {"401k": {"type": "boolean"}}, "required": ["401k"]}
    assert grade(srv, "<plan><_401k>true</_401k></plan>", "xml", digit) == 1.0
    only = {"type": "object", "properties": {"output": {"type": "object", "properties": {"a": {"type": "string"}}}}}
    assert grade(srv, "<output><a>x</a></output>", "xml", only) == 1.0  # the root is the only property
    assert grade(srv, "<output><output><a>x</a></output></output>", "xml", only) == 1.0  # wrapper named like it


def test_toml_nulls_by_omission_and_null_literal_is_invalid():
    srv = server(True)
    req_null = {**PERSON, "required": ["name", "location", "medal"]}
    toml = 'name = "A"\n[location]\ncity = "B"\n'
    assert grade(srv, toml, "toml", req_null) == 1.0  # medal omitted = null
    assert grade(srv, 'name = "A"\nmedal = null\n[location]\ncity = "B"\n', "toml", req_null) == 0.0
    assert grade(srv, 'name = "A"\nmedal = ""\n[location]\ncity = "B"\n', "toml", req_null) == 0.0  # placeholder
    assert grade(server(False), toml, "toml", req_null) == 0.0
    arr = {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}}
    assert grade(srv, "[[items]]\nid = 1\n[[items]]\nid = 2\n", "toml", arr) == 1.0


@pytest.mark.parametrize("csv_text", [
    "name,location.city,location.country,towns[0].name,towns[0].km,towns[1].name,tags\n"
    'A,B,C,T,2.5,U,"[""x"",""y""]"',
    'name,location,towns,tags\nA,"{""city"":""B""}","[{""name"":""T"",""km"":2.5}]","[""x""]"',
    "path,value\nname,A\nlocation.city,B\ntowns.0.name,T\ntowns.0.km,2.5\ntowns.1.name,U",
    "name,location.city,towns.name,towns.km\nA,B,T,2.5\nA,B,U,3",
    "name,location/city,age\nA,B,",
])
def test_csv_nested_objects_from_the_layouts_models_write(csv_text):
    assert grade(server(True), csv_text, "csv") == 1.0
    assert grade(server(False), csv_text, "csv") == 0.0


def test_csv_rejects_non_csv_pipe_lists_and_conflicts():
    srv = server(True)
    assert grade(srv, json.dumps({"name": "A", "location": {"city": "B"}}), "csv") == 0.0
    assert grade(srv, "name,location.city,tags\nA,B,x | y", "csv") == 0.0  # lists in a cell must be JSON arrays
    assert grade(srv, "name,location.city\nA,B\nZ,B", "csv") == 0.0  # rows disagree on a scalar
    assert grade(srv, "name,location.city,extra\nA,B,1", "csv") == 0.0
    flat = {"type": "array", "items": {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}}
    assert grade(srv, "\n\na\n1\n2\n", "csv", flat) == 1.0


def test_tool_call_payloads_use_the_schemas_own_required_list():
    payload = {"name": "A", "location": {"city": "B"}}
    for faithful, want in ((True, 1.0), (False, 0.0)):
        assert server(faithful).evaluate_structured_object_response(json.dumps(PERSON), payload)[0] == want


def test_verify_passes_the_rows_fence_permission():
    srv = server(True)
    text = "```json\n" + json.dumps({"name": "A", "location": {"city": "B"}}) + "\n```"
    resp = NeMoGymResponse(id="r", created_at=0.0, model="m", object="response", parallel_tool_calls=False,
                           tool_choice="none", tools=[], output=[NeMoGymResponseOutputMessage(
                               id="m", role="assistant", status="completed", type="message",
                               content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")])])
    params = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "q"}])
    for allowed, want in ((True, 1.0), (False, 0.0)):
        body = StructuredOutputsVerifyRequest(responses_create_params=params, response=resp, schema_str=json.dumps(PERSON),
                                              schema_type="json", fences_allowed=allowed)
        assert asyncio.run(srv.verify(body)).reward == want


@pytest.mark.parametrize("schema", [
    {"type": "object", "properties": {"a": {"type": "string"}}},
    {"type": "object", "properties": {"a": {"type": "string"}}, "required": []},
    {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": False},
    {"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": {"type": "integer"}},
    {"type": "object", "properties": {}, "patternProperties": {"^x_": {"type": "integer"}}, "required": ["x_id"], "additionalProperties": False},
    {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a", "ghost"], "additionalProperties": False},
    {"type": "object", "properties": {"loc": {"type": "object", "properties": {"city": {"type": "string"}}}}, "required": ["loc"]},
    {"type": "object", "properties": {"required": {"type": "boolean"}}, "required": ["required"]},
])
@pytest.mark.parametrize("payload", [{}, {"a": "x"}, {"a": 1}, {"a": "x", "extra": 2}, {"x_id": 1}, {"loc": {}}, {"required": True}])
def test_json_format_matches_actual_schema_validator(schema, payload):
    expected = float(Draft202012Validator(schema).is_valid(payload))
    assert grade(server(True), json.dumps(payload), schema=schema) == expected
