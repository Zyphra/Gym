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
import csv
import io
import json
import re
import tomllib
from collections import defaultdict
from enum import StrEnum
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import xmltodict
import yaml
from fastapi import FastAPI
from openapi_schema_validator import validate as validate_against_schema_openapi

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class StructuredOutputsResourcesServerConfig(BaseResourcesServerConfig):
    xml_coerce_types: bool = True
    # Opt-in. Grade every format by its own faithful representations of the schema: the schema's own required
    # lists, whitespace strip, fences only where the row allows them, XML root unwrapping and union-type coercion,
    # TOML nulls by omission, and schema-guided CSV decoding of nested objects. Off grades exactly as before.
    faithful_format_grading: bool = False


class _YamlStringDatesLoader(yaml.SafeLoader):
    """SafeLoader that keeps YAML timestamps as the text the model wrote (JSON Schema has no date type)."""


_YamlStringDatesLoader.yaml_implicit_resolvers = {
    k: [(tag, regexp) for tag, regexp in v if tag != "tag:yaml.org,2002:timestamp"]
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_FENCE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\n(.*?)\n?```", re.S)


class SchemaType(StrEnum):
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    TOML = "toml"
    CSV = "csv"


class StructuredOutputsVerifyRequest(BaseVerifyRequest):
    schema_str: str
    schema_type: SchemaType = SchemaType.JSON
    problem_type: Optional[str] = None
    schema_repr: Optional[str] = None
    source_format: Optional[str] = None
    num_turns: Optional[int] = None
    source_record_id: Optional[str] = None
    response_mode: Optional[str] = "text"
    tool_name: Optional[str] = None
    tool_schema_mode: Optional[str] = None
    tool_payload_key: Optional[str] = None
    tool_choice: Optional[str] = None
    parallel_tool_calls: Optional[bool] = None
    source_schema_type: Optional[str] = None
    num_tools: Optional[int] = None
    num_distractors: Optional[int] = None
    has_distractors: Optional[bool] = None
    instruction_layout: Optional[str] = None
    instruction_detail_level: Optional[str] = None
    system_instruction_style: Optional[str] = None
    tool_name_style: Optional[str] = None
    distractor_style: Optional[str] = None
    tool_union_mode: Optional[str] = None
    fences_allowed: Optional[bool] = None


class StructuredOutputsVerifyResponse(BaseVerifyResponse):
    schema_str: str
    schema_type: SchemaType
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    problem_type: Optional[str] = None
    schema_repr: Optional[str] = None
    source_format: Optional[str] = None
    num_turns: Optional[int] = None
    source_record_id: Optional[str] = None
    response_mode: Optional[str] = "text"
    tool_name: Optional[str] = None
    tool_schema_mode: Optional[str] = None
    tool_payload_key: Optional[str] = None
    tool_choice: Optional[str] = None
    parallel_tool_calls: Optional[bool] = None
    source_schema_type: Optional[str] = None
    num_tools: Optional[int] = None
    num_distractors: Optional[int] = None
    has_distractors: Optional[bool] = None
    instruction_layout: Optional[str] = None
    instruction_detail_level: Optional[str] = None
    system_instruction_style: Optional[str] = None
    tool_name_style: Optional[str] = None
    distractor_style: Optional[str] = None
    tool_union_mode: Optional[str] = None
    fences_allowed: Optional[bool] = None


class StructuredOutputsResourcesServer(SimpleResourcesServer):
    config: StructuredOutputsResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        by_fmt: Dict[str, List[float]] = defaultdict(list)
        by_problem: Dict[str, List[float]] = defaultdict(list)
        by_repr: Dict[str, List[float]] = defaultdict(list)
        by_response_mode: Dict[str, List[float]] = defaultdict(list)
        by_tool_schema_mode: Dict[str, List[float]] = defaultdict(list)
        by_tool_choice: Dict[str, List[float]] = defaultdict(list)
        by_parallel_tool_calls: Dict[str, List[float]] = defaultdict(list)
        by_num_tools: Dict[str, List[float]] = defaultdict(list)
        by_has_distractors: Dict[str, List[float]] = defaultdict(list)
        by_instruction_layout: Dict[str, List[float]] = defaultdict(list)
        by_instruction_detail_level: Dict[str, List[float]] = defaultdict(list)
        by_system_instruction_style: Dict[str, List[float]] = defaultdict(list)
        by_tool_name_style: Dict[str, List[float]] = defaultdict(list)
        by_distractor_style: Dict[str, List[float]] = defaultdict(list)
        by_tool_union_mode: Dict[str, List[float]] = defaultdict(list)

        for rollouts in tasks:
            for r in rollouts:
                reward = r.get("reward", 0.0)
                by_fmt[r.get("schema_type", "unknown")].append(reward)
                pt = r.get("problem_type")
                if pt:
                    by_problem[pt].append(reward)
                sr = r.get("schema_repr")
                if sr:
                    by_repr[sr].append(reward)
                response_mode = r.get("response_mode")
                if response_mode:
                    by_response_mode[response_mode].append(reward)
                tool_schema_mode = r.get("tool_schema_mode")
                if tool_schema_mode:
                    by_tool_schema_mode[tool_schema_mode].append(reward)
                tool_choice = r.get("tool_choice")
                if tool_choice:
                    by_tool_choice[tool_choice].append(reward)
                parallel_tool_calls = r.get("parallel_tool_calls")
                if parallel_tool_calls is not None:
                    by_parallel_tool_calls[str(bool(parallel_tool_calls)).lower()].append(reward)
                num_tools = r.get("num_tools")
                if num_tools is not None:
                    by_num_tools[str(num_tools)].append(reward)
                has_distractors = r.get("has_distractors")
                if has_distractors is not None:
                    by_has_distractors[str(bool(has_distractors)).lower()].append(reward)
                instruction_layout = r.get("instruction_layout")
                if instruction_layout:
                    by_instruction_layout[instruction_layout].append(reward)
                instruction_detail_level = r.get("instruction_detail_level")
                if instruction_detail_level:
                    by_instruction_detail_level[instruction_detail_level].append(reward)
                system_instruction_style = r.get("system_instruction_style")
                if system_instruction_style:
                    by_system_instruction_style[system_instruction_style].append(reward)
                tool_name_style = r.get("tool_name_style")
                if tool_name_style:
                    by_tool_name_style[tool_name_style].append(reward)
                distractor_style = r.get("distractor_style")
                if distractor_style:
                    by_distractor_style[distractor_style].append(reward)
                tool_union_mode = r.get("tool_union_mode")
                if tool_union_mode:
                    by_tool_union_mode[tool_union_mode].append(reward)

        metrics = {f"mean/reward_{k}": mean(v) for k, v in by_fmt.items() if v}
        metrics.update({f"mean/reward_{k}": mean(v) for k, v in by_problem.items() if v})
        metrics.update({f"mean/reward_repr_{k}": mean(v) for k, v in by_repr.items() if v})
        metrics.update({f"mean/reward_response_mode_{k}": mean(v) for k, v in by_response_mode.items() if v})
        metrics.update({f"mean/reward_tool_schema_mode_{k}": mean(v) for k, v in by_tool_schema_mode.items() if v})
        metrics.update({f"mean/reward_tool_choice_{k}": mean(v) for k, v in by_tool_choice.items() if v})
        metrics.update(
            {f"mean/reward_parallel_tool_calls_{k}": mean(v) for k, v in by_parallel_tool_calls.items() if v}
        )
        metrics.update({f"mean/reward_num_tools_{k}": mean(v) for k, v in by_num_tools.items() if v})
        metrics.update({f"mean/reward_has_distractors_{k}": mean(v) for k, v in by_has_distractors.items() if v})
        metrics.update({f"mean/reward_instruction_layout_{k}": mean(v) for k, v in by_instruction_layout.items() if v})
        metrics.update(
            {f"mean/reward_instruction_detail_level_{k}": mean(v) for k, v in by_instruction_detail_level.items() if v}
        )
        metrics.update(
            {f"mean/reward_system_instruction_style_{k}": mean(v) for k, v in by_system_instruction_style.items() if v}
        )
        metrics.update({f"mean/reward_tool_name_style_{k}": mean(v) for k, v in by_tool_name_style.items() if v})
        metrics.update({f"mean/reward_distractor_style_{k}": mean(v) for k, v in by_distractor_style.items() if v})
        metrics.update({f"mean/reward_tool_union_mode_{k}": mean(v) for k, v in by_tool_union_mode.items() if v})
        return metrics

    async def verify(self, body: StructuredOutputsVerifyRequest) -> StructuredOutputsVerifyResponse:
        schema_type = body.schema_type
        schema_str = body.schema_str

        if schema_type not in list(SchemaType):
            raise NotImplementedError(f"SchemaType must be one of {list(SchemaType)}, got {schema_type} !")

        if body.response_mode == "tool_call":
            response_obj, error_type, error_message = self.extract_tool_call_payload(body)
            if error_type is not None:
                return StructuredOutputsVerifyResponse(
                    **body.model_dump(), reward=0.0, error_type=error_type, error_message=error_message
                )

            reward, error_type, error_message = self.evaluate_structured_object_response(schema_str, response_obj)
            return StructuredOutputsVerifyResponse(
                **body.model_dump(), reward=reward, error_type=error_type, error_message=error_message
            )

        # get model generation.
        assistant_responses = []
        for output_item in body.response.output:
            if output_item.type != "message":
                continue

            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue

                assistant_responses.append(content_item.text)
        response_text = "".join(assistant_responses)

        reward, error_type, error_message = self.evaluate_structured_output_response(
            schema_type, schema_str, response_text, fences_allowed=body.fences_allowed
        )
        return StructuredOutputsVerifyResponse(
            **body.model_dump(), reward=reward, error_type=error_type, error_message=error_message
        )

    # ----- Helpers ----- #
    def extract_tool_call_payload(
        self, body: StructuredOutputsVerifyRequest
    ) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
        if body.response.error is not None:
            return None, "api_error", str(body.response.error)[:200]

        function_calls = [output_item for output_item in body.response.output if output_item.type == "function_call"]
        if not function_calls:
            return None, "missing_tool_call", "No function_call item found in assistant response"
        if len(function_calls) > 1:
            called_names = ", ".join(function_call.name for function_call in function_calls)
            return None, "multiple_tool_calls", f"Expected exactly one function_call, got: {called_names}"

        selected_call = None
        if body.tool_name:
            for function_call in function_calls:
                if function_call.name == body.tool_name:
                    selected_call = function_call
                    break
            if selected_call is None:
                called_names = ", ".join(function_call.name for function_call in function_calls)
                return None, "wrong_tool_name", f"Expected tool '{body.tool_name}', got: {called_names}"
        else:
            selected_call = function_calls[0]

        arguments = selected_call.arguments
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                return None, "tool_arguments_parse_error", f"{type(e).__name__}: {str(e)[:200]}"
        elif isinstance(arguments, dict):
            parsed_arguments = arguments
        else:
            return None, "tool_arguments_parse_error", f"Unsupported arguments type: {type(arguments).__name__}"

        if body.tool_payload_key:
            if not isinstance(parsed_arguments, dict) or body.tool_payload_key not in parsed_arguments:
                return None, "missing_tool_payload_key", f"Missing tool payload key: {body.tool_payload_key}"
            return parsed_arguments[body.tool_payload_key], None, None

        return parsed_arguments, None, None

    def parse_content(self, schema_type: SchemaType, content: str):
        match schema_type.lower():
            case SchemaType.JSON:
                parsed = json.loads(content)
            case SchemaType.YAML:
                parsed = yaml.safe_load(content)
            case SchemaType.XML:
                parsed = xmltodict.parse(content)
            case SchemaType.TOML:
                parsed = tomllib.loads(content)
            case SchemaType.CSV:
                parsed = list(csv.DictReader(io.StringIO(content)))
            case _:
                parsed = None
        return parsed

    def strictify_schema(self, schema: Dict[str, Any]):
        """Make a schema strict as per OpenAPI guidelines"""
        if isinstance(schema, Dict):
            if "properties" in schema:
                schema["required"] = list(schema["properties"])
                schema["additionalProperties"] = False
            for k, v in schema.items():
                self.strictify_schema(v)

    def coerce_xml_types(self, data: Any, schema: Dict[str, Any]) -> Any:
        """Recursively coerce xmltodict string values to match the JSON schema types.

        xmltodict.parse() returns all leaf values as strings. This method walks the
        parsed data alongside the schema and converts values where possible.
        On conversion failure the original value is returned so that schema
        validation can report the error.
        """
        if not isinstance(schema, dict) or "type" not in schema:
            return data

        schema_type = schema["type"]

        if schema_type == "object" and isinstance(data, dict):
            properties = schema.get("properties", {})
            coerced = {}
            for key, value in data.items():
                if key in properties:
                    coerced[key] = self.coerce_xml_types(value, properties[key])
                else:
                    coerced[key] = value
            return coerced

        if schema_type == "array":
            items_schema = schema.get("items", {})
            # xmltodict represents repeated child elements as {"tagName": [values]},
            # e.g. <skills><string>a</string><string>b</string></skills> becomes
            # {"string": ["a", "b"]}. For single elements, xmltodict gives
            # {"string": "python"} instead of a list. In both cases, unwrap the
            # single-key dict since we're at an array schema position -- a dict here
            # is always the xmltodict wrapping artifact, not a meaningful structure.
            if isinstance(data, dict) and len(data) == 1:
                data = next(iter(data.values()))
            if not isinstance(data, list):
                data = [data] if data is not None else []
            return [self.coerce_xml_types(item, items_schema) for item in data]

        # xmltodict returns None for empty tags like <field/> or <field></field>.
        # Coerce to "" only for string types (parity with JSON/YAML where "" is valid).
        # Non-string types (integer, boolean, etc.) intentionally left as None so
        # they fail validation -- 0 and False are meaningful values, not "empty".
        if data is None and schema_type == "string":
            return ""

        if isinstance(data, str):
            try:
                if schema_type == "integer":
                    return int(data)
                if schema_type == "number":
                    return float(data)
                if schema_type == "boolean":
                    lower = data.lower()
                    if lower in ("true", "1"):
                        return True
                    if lower in ("false", "0"):
                        return False
            except (ValueError, AttributeError):
                pass

        return data

    def coerce_csv_types(self, rows: List[Dict[str, str]], schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Coerce CSV string values to match JSON Schema types.

        csv.DictReader returns all values as strings.  Walk rows alongside the
        items schema and convert values where possible.  On conversion failure
        the original string is kept so that schema validation reports the error.
        """
        items_schema = schema.get("items", schema)
        properties = items_schema.get("properties", {})

        coerced_rows = []
        for row in rows:
            coerced: Dict[str, Any] = {}
            for key, value in row.items():
                prop_schema = properties.get(key, {})
                prop_type = prop_schema.get("type", "string")
                coerced[key] = self._coerce_csv_scalar(value, prop_type)
            coerced_rows.append(coerced)
        return coerced_rows

    def _coerce_csv_scalar(self, value: str, target_type) -> Any:
        """Coerce a single CSV string to the target JSON Schema type."""
        if isinstance(target_type, list):
            if (value is None or value == "") and "null" in target_type:
                return None
            for t in target_type:
                if t == "null":
                    continue
                result = self._coerce_csv_scalar(value, t)
                if not isinstance(result, str) or t == "string":
                    return result
            return value

        if value is None or value == "":
            return value

        try:
            if target_type == "integer":
                return int(value)
            if target_type == "number":
                return float(value)
            if target_type == "boolean":
                lower = value.lower()
                if lower in ("true", "1"):
                    return True
                if lower in ("false", "0"):
                    return False
        except (ValueError, AttributeError):
            pass
        return value

    def evaluate_structured_object_response(
        self, schema_str: str, response_obj: Any
    ) -> Tuple[float, Optional[str], Optional[str]]:
        """Returns (reward, error_type, error_message) for already-parsed tool-call arguments."""
        try:
            schema = json.loads(schema_str)
        except Exception as e:
            return 0.0, "schema_error", str(e)[:200]

        self.strict_schema(schema)

        try:
            validate_against_schema_openapi(response_obj, schema)
            return 1.0, None, None
        except Exception as e:
            return 0.0, "validation_error", f"{type(e).__name__}: {str(e)[:200]}"

    def evaluate_structured_output_response(
        self,
        schema_type: SchemaType,
        schema_str: str,
        response_text: str,
        fences_allowed: Optional[bool] = None,
    ) -> Tuple[float, Optional[str], Optional[str]]:
        """Returns (reward, error_type, error_message)."""
        if not response_text or not response_text.strip():
            return 0.0, "empty_response", "No assistant response text"

        try:
            schema = json.loads(schema_str)
        except Exception as e:
            return 0.0, "schema_error", str(e)[:200]

        self.strict_schema(schema)
        faithful = self.config.faithful_format_grading
        if faithful:
            response_text = self.faithful_text(response_text, fences_allowed)

        try:
            if faithful:
                response_obj = self.faithful_parse(schema_type, response_text, schema)
            else:
                response_obj = self.parse_content(schema_type, response_text)
        except Exception as e:
            return 0.0, "parse_error", f"{type(e).__name__}: {str(e)[:200]}"

        try:
            if faithful:
                response_obj = self.faithful_coerce(schema_type, response_obj, schema)
            else:
                if schema_type == SchemaType.XML and self.config.xml_coerce_types:
                    response_obj = self.coerce_xml_types(response_obj, schema)
                if schema_type == SchemaType.CSV:
                    response_obj = self.coerce_csv_types(response_obj, schema)
            validate_against_schema_openapi(response_obj, schema)
            return 1.0, None, None
        except Exception as e:
            return 0.0, "validation_error", f"{type(e).__name__}: {str(e)[:200]}"

    # ----- Faithful format grading (opt-in) ----- #
    def strict_schema(self, schema: Dict[str, Any]) -> None:
        # Faithful grading uses the supplied schema unchanged. Requiring optional
        # properties, forbidding unspecified extras, or deleting required names
        # would change the task. Source contradictions belong to source repair.
        if not self.config.faithful_format_grading:
            self.strictify_schema(schema)

    @staticmethod
    def schema_types(schema: Any) -> List[str]:
        if not isinstance(schema, dict):
            return []
        t = schema.get("type")
        if isinstance(t, list):
            return [str(x) for x in t]
        if t is None:
            if "properties" in schema:
                return ["object"]
            if "items" in schema:
                return ["array"]
            return []
        return [str(t)]

    def faithful_text(self, text: str, fences_allowed: Optional[bool]) -> str:
        """Strip surrounding whitespace; unwrap one whole-answer code fence only where the row allows fences."""
        text = text.strip()
        if fences_allowed:
            m = _FENCE.fullmatch(text)
            if m:
                text = m.group(1).strip()
        return text

    def faithful_parse(self, schema_type: SchemaType, text: str, schema: Dict[str, Any]) -> Any:
        match schema_type.lower():
            case SchemaType.JSON:
                return json.loads(text)
            case SchemaType.YAML:
                return yaml.load(text, Loader=_YamlStringDatesLoader)
            case SchemaType.XML:
                parsed = xmltodict.parse(text)
                if isinstance(parsed, dict) and len(parsed) == 1:
                    root, inner = next(iter(parsed.items()))
                    types = self.schema_types(schema)
                    # XML needs exactly one root element; unwrap it unless it is itself the schema's only property.
                    # <p><p>...</p></p> for a schema whose property is p reads as wrapper plus property.
                    nested_same = isinstance(inner, dict) and list(inner) == [root]
                    if "array" in types or root not in (schema.get("properties") or {}) or nested_same:
                        return {} if inner is None and "object" in types else inner
                return parsed
            case SchemaType.TOML:
                parsed = tomllib.loads(text)
                # A TOML document is a table; a top-level array can only be written under one key.
                if "array" in self.schema_types(schema) and isinstance(parsed, dict) and len(parsed) == 1:
                    only = next(iter(parsed.values()))
                    if isinstance(only, list):
                        return only
                return parsed
            case SchemaType.CSV:
                return self.decode_csv(text, schema)
            case _:
                return None

    def faithful_coerce(self, schema_type: SchemaType, data: Any, schema: Dict[str, Any]) -> Any:
        if schema_type == SchemaType.XML:
            if self.config.xml_coerce_types:
                data = self.coerce_text_values(data, schema, xml=True)
            return self.fill_omitted_nulls(data, schema)  # XML has no standard null: an omitted element is null
        if schema_type == SchemaType.TOML:
            return self.fill_omitted_nulls(data, schema)
        return data

    def coerce_text_values(self, data: Any, schema: Any, xml: bool = False) -> Any:
        """Schema-guided coercion of text leaves (XML, CSV), aware of union types such as ["integer", "null"]."""
        types = self.schema_types(schema)
        if not types:
            return data
        if "object" in types and isinstance(data, dict):
            props = schema.get("properties", {}) or {}
            out = {}
            for key, value in data.items():
                name = key
                # XML element names cannot start with a digit; "_401k" is the usual spelling of "401k".
                if xml and name not in props and name.startswith("_") and name[1:] in props and name[1:] not in data:
                    name = name[1:]
                out[name] = self.coerce_text_values(value, props[name], xml) if name in props else value
            return out
        if "array" in types:
            items = schema.get("items", {})
            if xml and isinstance(data, dict) and len(data) == 1:
                data = next(iter(data.values()))
            if isinstance(data, dict) and not xml and "object" in self.schema_types(items):
                data = [data]
            if data is None:
                return None if "null" in types else []
            if not isinstance(data, list):
                if not xml:  # CSV: a list in a cell must be a JSON array; plain text stays text and fails
                    return data
                data = [data]  # XML: a single child element is a list of one
            return [self.coerce_text_values(item, items, xml) for item in data]
        if xml and data is None and "object" in types and "null" not in types:
            return {}  # an empty element is an empty object
        if data is None or (isinstance(data, str) and data.strip() == ""):
            if "null" in types:
                return None
            if "string" in types:
                return ""
            return data
        if isinstance(data, str):
            value = data.strip()
            if "string" in types:
                return data
            if "null" in types and value.lower() in ("null", "none"):
                return None
            for t in types:
                try:
                    if t == "integer" and re.fullmatch(r"[-+]?\d+", value):
                        return int(value)
                    if t == "number" and re.fullmatch(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", value):
                        return float(value) if re.search(r"[.eE]", value) else int(value)
                    if t == "boolean" and value.lower() in ("true", "false", "1", "0"):
                        return value.lower() in ("true", "1")
                except ValueError:
                    continue
        return data

    def fill_omitted_nulls(self, data: Any, schema: Any) -> Any:
        """TOML and XML have no null value: an omitted property reads as null wherever the schema allows null."""
        types = self.schema_types(schema)
        if "object" in types and isinstance(data, dict):
            props = schema.get("properties", {}) or {}
            out = {k: self.fill_omitted_nulls(v, props.get(k, {})) for k, v in data.items()}
            for k, sub in props.items():
                if k not in out and "null" in self.schema_types(sub):
                    out[k] = None
            return out
        if "array" in types and isinstance(data, list):
            return [self.fill_omitted_nulls(v, schema.get("items", {})) for v in data]
        return data

    # CSV: one table per answer. Nested objects are written with path headers ("a.b", "a[0].b", "a.0.b", "a/b"),
    # JSON values in cells, a long "path, ..., value" layout, or one row per item of an array of objects. Lists in a
    # cell must be JSON arrays.
    def decode_csv(self, text: str, schema: Dict[str, Any]) -> Any:
        rows = [r for r in csv.reader(io.StringIO(text)) if len(r) > 1 or (r and r[0].strip())]  # skip blank lines
        if not rows:
            raise ValueError("empty CSV")
        header = [h.strip() for h in rows[0]]
        if len(set(header)) != len(header):
            raise ValueError("duplicate CSV header")
        body = rows[1:]
        for r in body:
            if len(r) != len(header):
                raise ValueError(f"CSV row has {len(r)} cells, header has {len(header)}")
        types = self.schema_types(schema)
        if "array" in types:
            items = schema.get("items", {})
            return [self.csv_record(dict(zip(header, r)), items) for r in body]
        if (
            len(header) >= 2
            and header[-1].lower() in ("value", "values")
            and len(body) >= 1
            and not (set(header) & set((schema.get("properties") or {}).keys()))
        ):
            obj: Dict[str, Any] = {}
            for r in body:
                parts = [p for cell in r[:-1] for p in self.csv_path(cell.strip(), schema)]
                if parts:
                    self.set_path(obj, parts, r[-1])
            return self.coerce_text_values(self.compact_lists(obj), schema)
        if not body:
            raise ValueError("CSV has a header but no data row")
        record = self.csv_record(dict(zip(header, body[0])), schema)
        for r in body[1:]:  # further rows add items to array fields
            extra = self.csv_record(dict(zip(header, r)), schema)
            for k, v in extra.items():
                if isinstance(record.get(k), list) and isinstance(v, list):
                    record[k].extend(v)
                elif k not in record:
                    record[k] = v
                elif record[k] != v:
                    raise ValueError(f"CSV rows disagree on non-list field {k!r}")
        return record

    def csv_record(self, flat: Dict[str, str], schema: Dict[str, Any]) -> Any:
        obj: Dict[str, Any] = {}
        for header, value in flat.items():
            if value is None or value.strip() == "":
                parts = self.csv_path(header, schema)
                if parts and "null" in self.schema_types(self.schema_at(schema, parts)):
                    self.set_path(obj, parts, None)
                continue
            parts = self.csv_path(header, schema)
            if parts:
                self.set_path(obj, parts, value)
        obj = self.compact_lists(obj)
        return self.coerce_csv_cells(obj, schema)

    def coerce_csv_cells(self, data: Any, schema: Any) -> Any:
        """Decode JSON cells where the schema expects an object or array, then coerce text leaves."""
        types = self.schema_types(schema)
        if isinstance(data, str) and ("object" in types or "array" in types) and data.strip()[:1] in "[{":
            try:
                data = json.loads(data)
            except ValueError:
                return data
        if isinstance(data, dict) and "object" in types:
            props = schema.get("properties", {}) or {}
            data = {k: self.coerce_csv_cells(v, props[k]) if k in props else v for k, v in data.items()}
        elif isinstance(data, list) and "array" in types:
            data = [self.coerce_csv_cells(v, schema.get("items", {})) for v in data]
        return self.coerce_text_values(data, schema)

    def csv_path(self, header: str, schema: Dict[str, Any]) -> List[Any]:
        if header in (schema.get("properties") or {}):
            return [header]
        parts: List[Any] = []
        for token in re.split(r"\.|/|\[|\]", header):
            if token == "":
                continue
            parts.append(int(token) if token.isdigit() else token)
        return parts

    def schema_at(self, schema: Any, parts: List[Any]) -> Any:
        for p in parts:
            if isinstance(p, int):
                schema = schema.get("items", {}) if isinstance(schema, dict) else {}
            else:
                schema = ((schema.get("properties") or {}).get(p, {})) if isinstance(schema, dict) else {}
        return schema

    @staticmethod
    def set_path(obj: Any, parts: List[Any], value: Any) -> None:
        cur = obj
        for i, p in enumerate(parts):
            last = i == len(parts) - 1
            nxt = None if last else parts[i + 1]
            if isinstance(cur, list):
                if not isinstance(p, int):
                    raise ValueError(f"CSV path step {p!r} indexes a list")
                while len(cur) <= p:
                    cur.append(None)
                if last:
                    cur[p] = value
                else:
                    if cur[p] is None:
                        cur[p] = [] if isinstance(nxt, int) else {}
                    cur = cur[p]
            else:
                if isinstance(p, int):
                    raise ValueError(f"CSV path index {p} on an object")
                if last:
                    if p in cur and cur[p] != value:
                        raise ValueError(f"CSV path {p!r} written twice")
                    cur[p] = value
                else:
                    if p not in cur or cur[p] is None:
                        cur[p] = [] if isinstance(nxt, int) else {}
                    cur = cur[p]

    def compact_lists(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: self.compact_lists(v) for k, v in data.items()}
        if isinstance(data, list):
            if any(v is None for v in data):
                raise ValueError("CSV list indices have gaps")
            return [self.compact_lists(v) for v in data]
        return data


if __name__ == "__main__":
    StructuredOutputsResourcesServer.run_webserver()
