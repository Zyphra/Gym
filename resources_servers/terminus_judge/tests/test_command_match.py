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
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.terminus_judge.app import (
    ENTER,
    FailureCode,
    TerminusJudgeResourcesServer,
    TerminusJudgeResourcesServerConfig,
    TerminusJudgeVerifyRequest,
    command_list,
    command_match,
    extract_json_object,
    normalize_keystrokes,
)


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def answer(*keystrokes: str, task_complete: bool = False) -> dict:
    return {
        "analysis": "a",
        "plan": "p",
        "commands": [{"keystrokes": k, "duration": 1.0} for k in keystrokes],
        "task_complete": task_complete,
    }


def server(**overrides) -> TerminusJudgeResourcesServer:
    config = TerminusJudgeResourcesServerConfig(
        host="127.0.0.1",
        port=20002,
        entrypoint="",
        name="terminus_judge_command_match_test",
        enable_string_similarity=True,
        string_similarity_threshold=0.9,
        enable_llm_judge=False,
        **overrides,
    )
    return TerminusJudgeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def request(model_output: str, expected: dict) -> TerminusJudgeVerifyRequest:
    response = NeMoGymResponse(
        id="r",
        created_at=0,
        model="m",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="m", content=[NeMoGymResponseOutputText(annotations=[], text=model_output)]
            )
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )
    return TerminusJudgeVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[NeMoGymEasyInputMessage(role="user", content="x")]
        ),
        response=response,
        expected_answer=json.dumps(expected),
        metadata={"harness": "terminus_2"},
    )


COMMAND_MATCH = dict(json_extraction="terminus_2", command_scoring="command_match")


class TestExtractJsonObject:
    def test_whole_text(self):
        assert extract_json_object(json.dumps(answer("ls\n"))) == answer("ls\n")

    def test_code_fence_and_prose(self):
        text = "Here is my answer.\n```json\n" + json.dumps(answer("ls\n"), indent=2) + "\n```\n"
        assert extract_json_object(text) == answer("ls\n")

    def test_first_object_of_several_is_the_one_executed(self):
        text = json.dumps(answer("ls\n")) + "\n" + json.dumps(answer("pwd\n"))
        assert extract_json_object(text) == answer("ls\n")

    def test_braces_inside_strings_do_not_end_the_object(self):
        first = answer("awk '{print $1}' f\n")
        assert extract_json_object("x " + json.dumps(first) + " }") == first

    def test_unbalanced_quote_in_prose_falls_back_to_decoding(self):
        text = 'The file is 5" wide.\n' + json.dumps(answer("ls\n"))
        assert extract_json_object(text) == answer("ls\n")

    def test_truncated_or_absent_json_is_rejected(self):
        assert extract_json_object(json.dumps(answer("ls\n"))[:-5]) is None
        assert extract_json_object("no json here") is None


class TestNormalizeKeystrokes:
    def test_equivalent_quoting_and_spacing(self):
        assert normalize_keystrokes("cat 'a b'  >x\n") == normalize_keystrokes('cat "a b" > x')
        assert normalize_keystrokes("ls -la /app;cat f\n") == normalize_keystrokes("ls  -la /app ; cat f")

    def test_word_boundaries_are_kept(self):
        assert normalize_keystrokes("echo 'a b'") != normalize_keystrokes("echo a b")

    def test_unbalanced_quote_falls_back_to_whitespace(self):
        assert normalize_keystrokes('"\n') == '"'


class TestCommandList:
    def test_waits_drop_and_bare_enter_joins_previous_command(self):
        assert command_list(answer("", '"', "\n", "echo hi\n")) == ['"', "echo\x1fhi"]

    def test_leading_bare_enter_is_a_command(self):
        assert command_list(answer("\n", "cat f\n")) == [ENTER, "cat\x1ff"]


class TestCommandMatch:
    def test_superset_in_order_within_extra_budget_passes(self):
        gt = answer("cat a\n", "cat b\n", "cat c\n", "cat d\n")
        pred = answer("cat a\n", "cat b\n", "cat c\n", "cat d\n", "cat e\n", "cat f\n")
        assert command_match(gt, pred, 0.9, 2) is True
        assert command_match(gt, pred, 0.9, 1) is False

    def test_reordered_batch_does_not_pass(self):
        gt = answer("cat a\n", "cat b\n")
        assert command_match(gt, answer("cat b\n", "cat a\n"), 0.9, 2) is False

    def test_prefix_of_multi_command_batch_does_not_pass(self):
        gt = answer("cat a\n", "cat b\n", "cat c\n")
        assert command_match(gt, answer("cat a\n", "cat b\n"), 0.9, 2) is False

    def test_single_intent_first_command_passes_with_follow_ups(self):
        gt = answer("python3 run.py --check\n")
        pred = answer("python3 run.py --check\n", "ls\n", "cat out\n", "echo done\n")
        assert command_match(gt, pred, 0.9, 2) is True

    def test_closing_an_open_quote_matches_split_teacher_keystrokes(self):
        assert command_match(answer('"', "\n"), answer('"\n'), 0.9, 2) is True

    def test_wait_steps(self):
        assert command_match(answer(), answer(""), 0.9, 2) is True
        assert command_match(answer(""), answer("ls\n"), 0.9, 2) is False
        assert command_match(answer("ls\n"), answer(), 0.9, 2) is False

    def test_enter_is_not_a_wait(self):
        assert command_match(answer("\n"), answer(), 0.9, 2) is False
        assert command_match(answer("\n", "cat db/init.sql\n"), answer("C-c", "cat db/init.sql\n"), 0.9, 2) is False

    def test_premature_task_complete_never_passes(self):
        assert command_match(answer("ls\n"), answer("ls\n", task_complete=True), 0.9, 2) is False
        assert command_match(answer(task_complete=True), answer(task_complete=True), 0.9, 2) is True


class TestVerifyModes:
    @pytest.mark.asyncio
    async def test_default_stays_strict(self):
        gt = answer("cat a\n", "cat b\n", "cat c\n", "cat d\n")
        fenced = "```json\n" + json.dumps(gt) + "\n```"
        strict = server()
        assert (await strict.verify(request(fenced, gt))).failure_reason == FailureCode.MODEL_OUTPUT_INVALID
        superset = json.dumps(answer("cat a\n", "cat b\n", "cat c\n", "cat d\n", "cat e\n"))
        response = await strict.verify(request(superset, gt))
        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.STRING_SIMILARITY_BELOW_THRESHOLD

    @pytest.mark.asyncio
    async def test_command_match_mode(self):
        gt = answer("cat a\n", "cat b\n", "cat c\n", "cat d\n")
        lenient = server(**COMMAND_MATCH)
        text = "<think>x</think>Plan first.\n```json\n" + json.dumps(answer(*gt_keys(gt), "cat e\n")) + "\n```"
        response = await lenient.verify(request(text, gt))
        assert response.reward == 1.0
        assert response.similarity_score < 0.9
        assert response.parsed_output["commands"][-1]["keystrokes"] == "cat e\n"

        response = await lenient.verify(request(json.dumps(answer("rm -rf /app\n")), gt))
        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.STRING_SIMILARITY_BELOW_THRESHOLD

    @pytest.mark.asyncio
    async def test_concatenated_pass_is_kept(self):
        gt = answer("cd /app/build/output/directory\n", "ls\n")
        merged = json.dumps(answer("cd /app/build/output/directory && ls\n"))
        assert not command_match(gt, json.loads(merged), 0.9, 2)
        response = await server(**COMMAND_MATCH).verify(request(merged, gt))
        assert response.reward == 1.0
        assert response.similarity_score >= 0.9

    @pytest.mark.asyncio
    async def test_task_complete_still_required(self):
        gt = answer(task_complete=True)
        lenient = server(**COMMAND_MATCH)
        response = await lenient.verify(request(json.dumps(answer()), gt))
        assert response.failure_reason == FailureCode.TASK_COMPLETE_CHECK_FAILED


def gt_keys(gt: dict) -> list[str]:
    return [c["keystrokes"] for c in gt["commands"]]


def test_command_match_config_overrides_string_only_server():
    base = OmegaConf.load(CONFIGS / "terminus_judge_string_only.yaml")
    overlay = OmegaConf.load(CONFIGS / "terminus_judge_string_only_command_match.yaml")
    assert overlay.config_paths == ["resources_servers/terminus_judge/configs/terminus_judge_string_only.yaml"]
    merged = OmegaConf.merge(base, overlay)
    fields = merged.terminus_judge_string_only_resources_server.resources_servers.terminus_judge
    config = TerminusJudgeResourcesServerConfig(host="h", port=1, name="n", **OmegaConf.to_container(fields))
    assert (config.json_extraction, config.command_scoring) == ("terminus_2", "command_match")
    assert config.string_similarity_threshold == 0.9 and not config.enable_llm_judge
