# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional verifier safety metrics survive native typed response projection."""

import pytest
from pydantic import ValidationError

from responses_api_agents.anyswe_agent.app import SWEBenchMetrics
from responses_api_agents.anyterminal_agent.app import TerminalBenchMetrics


@pytest.mark.parametrize("model", [SWEBenchMetrics, TerminalBenchMetrics])
def test_default_off_wire_omits_guard_metrics(model):
    payload = model().model_dump(mode="json")
    assert "reward_hack_detected" not in payload
    assert "verifier_integrity_violations" not in payload


@pytest.mark.parametrize("model", [SWEBenchMetrics, TerminalBenchMetrics])
@pytest.mark.parametrize("hack,violations", [(0, 0), (1, 3)])
def test_active_metrics_survive_json_and_preserve_infra_mask(model, hack, violations):
    metrics = model(
        resolved=False, mask_sample=True, reward_hack_detected=hack, verifier_integrity_violations=violations
    )
    restored = model.model_validate_json(metrics.model_dump_json())
    assert restored.reward_hack_detected == hack
    assert restored.verifier_integrity_violations == violations
    assert restored.mask_sample is True
    assert restored.resolved is False


def test_invalid_common_metrics_fail_typed_validation():
    with pytest.raises(ValidationError):
        SWEBenchMetrics(reward_hack_detected=2)
    with pytest.raises(ValidationError):
        TerminalBenchMetrics(verifier_integrity_violations=-1)
