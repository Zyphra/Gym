# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import multiprocessing
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from nemo_gym.base_responses_api_model import CaptureStore, ModelCallCaptureSelection


def policy(**changes):
    return ModelCallCaptureSelection(
        **{
            "model_server_names": ["customer"],
            "model_server_role": "user_simulator",
            "rollout_ids": ["selected"],
            "max_records_per_rollout": 20,
            "max_records_per_model": 100,
            "max_record_bytes": 4096,
            "max_total_bytes": 131072,
            "pending_records_per_process": 1,
            "capture_until": datetime(2099, 1, 1, tzinfo=timezone.utc),
            **changes,
        }
    )


def row(model="customer"):
    return {"model_ref": {"name": model}, "request": {"input": "customer turn"}, "response": {"output": "reply"}}


@pytest.mark.parametrize(
    "changes",
    [
        {"model_server_role": "policy"},
        {"rollout_ids": None},
        {"rollout_ids": []},
        {"rollout_ids": ["selected", "selected"]},
        {"rollout_ids": ["../foreign"]},
        {"max_records_per_rollout": None},
        {"max_records_per_rollout": 0},
    ],
)
def test_user_capture_requires_exact_bounded_identity(changes):
    with pytest.raises(ValidationError):
        policy(**changes)


def write_many(root, serialized):
    store = CaptureStore(root, ModelCallCaptureSelection.model_validate_json(serialized))
    for _ in range(30):
        store.record("selected", row())


def test_quota_is_shared_between_processes_on_one_capture_root(tmp_path):
    root = tmp_path / "capture"
    selection = policy()
    context = multiprocessing.get_context("spawn")
    children = [context.Process(target=write_many, args=(root, selection.model_dump_json())) for _ in range(2)]
    for child in children:
        child.start()
    for child in children:
        child.join(20)
        assert child.exitcode == 0
    store = CaptureStore(root, selection)
    assert len(store.read("selected")) == 20
    quota = json.loads((root / ".capture-quota.json").read_text())
    assert quota["rollouts"] == {"selected": 20}
    assert quota["models"] == {"customer": 20}
    assert all(item["capture_retention"]["quota_scope"] == "capture_root" for item in store.read("selected"))


def test_rollout_cap_combines_models_but_independent_roots_have_independent_quota(tmp_path):
    selection = policy(model_server_names=["customer", "second"], max_records_per_rollout=2)
    store = CaptureStore(tmp_path / "one", selection)
    store.record("selected", row())
    store.record("selected", row("second"))
    store.record("selected", row())
    assert len(store.read("selected")) == 2
    assert store.dropped == {"rollout_record_limit": 1}
    other = CaptureStore(tmp_path / "two", selection)
    other.record("selected", row())
    assert len(other.read("selected")) == 1


def test_unselected_identity_and_expired_window_write_no_body(tmp_path):
    store = CaptureStore(tmp_path / "one", policy())
    store.record("unselected", row())
    store.record("selected", row("policy"))
    assert not store.path_for("unselected").exists()
    assert not store.path_for("selected").exists()
    assert store.dropped == {"unselected_rollout": 1, "unselected_model": 1}
    expired = CaptureStore(
        tmp_path / "expired", policy(capture_until=datetime.now(timezone.utc) - timedelta(seconds=1))
    )
    assert not expired.admit()


def test_existing_redaction_and_no_response_mutation(tmp_path):
    store = CaptureStore(tmp_path / "capture", policy())
    original = row()
    original["request"]["api_key"] = "private-secret"
    store.record("selected", original)
    retained = store.read("selected")[0]
    assert retained["request"]["api_key"] == "[REDACTED]"
    assert retained["capture_retention"]["replay_eligible"] is False
    assert original["request"]["api_key"] == "private-secret"


def write_first_eligible(root, serialized, identity, model="customer"):
    store = CaptureStore(root, ModelCallCaptureSelection.model_validate_json(serialized))
    for _ in range(25):
        store.record(identity, row(model))


def test_first_eligible_cohort_is_locked_across_processes_and_never_rolls_over(tmp_path):
    selection = policy(
        rollout_selection="first_eligible", rollout_ids=None, max_rollouts=1, model_server_names=["customer", "second"]
    )
    root = tmp_path / "first"
    context = multiprocessing.get_context("spawn")
    children = [
        context.Process(target=write_first_eligible, args=(root, selection.model_dump_json(), identity, model))
        for identity, model in [("one", "customer"), ("two", "second")]
    ]
    for child in children:
        child.start()
    for child in children:
        child.join(20)
        assert child.exitcode == 0
    store = CaptureStore(root, selection)
    quota = json.loads((root / ".capture-quota.json").read_text())
    assert len(quota["selected_rollouts"]) == 1
    selected = quota["selected_rollouts"][0]
    assert len(store.read(selected)) == 20
    assert quota["rollouts"] == {selected: 20}
    assert sum(quota["models"].values()) == 20
    other = ({"one", "two"} - {selected}).pop()
    assert not store.path_for(other).exists()
    store.record("three", row())
    assert json.loads((root / ".capture-quota.json").read_text()) == quota


def test_five_episode_cohort_has_no_padding_or_sixth_replacement(tmp_path):
    selection = policy(
        rollout_selection="first_eligible", rollout_ids=None, max_rollouts=5, model_server_names=["customer", "second"]
    )
    store = CaptureStore(tmp_path / "cohort", selection)
    for index in range(6):
        for attempt in range(21):
            store.record(f"episode-{index}", row("customer" if attempt % 2 == 0 else "second"))
    quota = json.loads((store.root / ".capture-quota.json").read_text())
    assert quota["selected_rollouts"] == [f"episode-{index}" for index in range(5)]
    assert quota["models"] == {"customer": 50, "second": 50}
    assert not store.path_for("episode-5").exists()


def test_first_eligible_includes_failed_exchange_and_preserves_byte_bound(tmp_path):
    store = CaptureStore(
        tmp_path / "errors",
        policy(rollout_selection="first_eligible", rollout_ids=None, max_rollouts=1, max_total_bytes=4096),
    )
    failed = row()
    failed["status_code"] = 500
    failed["request"]["input"] = "q" * 3000
    store.record("failed-first", failed)
    store.record("failed-first", failed)
    store.record("successful-later", row())
    quota = json.loads((store.root / ".capture-quota.json").read_text())
    assert quota["selected_rollouts"] == ["failed-first"]
    assert len(store.read("failed-first")) == 1
    assert not store.path_for("successful-later").exists()
    assert quota["bytes"] <= 4096


def test_legacy_judge_quota_does_not_grow_with_unbounded_episode_names(tmp_path):
    selection = policy(
        model_server_role="judge", rollout_ids=None, max_records_per_rollout=None, max_records_per_model=1000
    )
    store = CaptureStore(tmp_path / "legacy", selection)
    for index in range(64):
        store.record(f"prompt.{index:064x}.{index:032x}", row())
    raw = (store.root / ".capture-quota.json").read_bytes()
    assert len(raw) < 1024
    assert json.loads(raw)["models"] == {"customer": 64}


@pytest.mark.parametrize(
    "changes",
    [
        {"model_server_role": "judge", "rollout_ids": None, "max_rollouts": 1},
        {"rollout_ids": ["selected"], "max_rollouts": 1},
        {"rollout_ids": None, "max_rollouts": None},
        {"rollout_ids": None, "max_rollouts": 1, "max_records_per_rollout": None},
    ],
)
def test_first_eligible_is_explicit_bounded_and_user_only(changes):
    with pytest.raises(ValidationError):
        policy(rollout_selection="first_eligible", **changes)


def test_known_unselected_rollouts_skip_capture_without_storage_io(tmp_path):
    selection = policy(rollout_selection="first_eligible", rollout_ids=None, max_rollouts=1)
    root = tmp_path / "capture"
    writer = CaptureStore(root, selection)
    other_process_view = CaptureStore(root, selection)
    writer.record("one", row())
    other_process_view.record("two", row())
    root.rename(tmp_path / "moved-away")
    assert not writer.admit("two")
    assert not other_process_view.admit("three")
    assert not root.exists()
    assert writer.admit("one")
    writer._pending.release()
