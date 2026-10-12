# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copied FD seeds and native reference worlds share one payout contract."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import ServerClient
from resources_servers.indian_banking.app import IndianBankingResourcesServer
from resources_servers.indian_banking.core import banking_tools, engine
from resources_servers.indian_banking.core import reward as reward_mod
from resources_servers.indian_banking.tests.test_app import (
    CUSTOMER,
    _config,
    _tiny_db,
)
from resources_servers.indian_banking.tests.test_app import (
    configured_engine as _configured_engine,
)


configured_engine = _configured_engine


@pytest.fixture
def seeded(configured_engine):
    raw = _tiny_db()
    customer = raw["customers"][CUSTOMER]
    for payout in ["monthly", "quarterly", "cumulative"]:
        deposit_id = "FD_SEED_" + payout
        customer["deposits"][deposit_id] = {
            "deposit_id": deposit_id, "kind": "FD", "principal": 20000,
            "rate": 6.8, "tenure_months": 12, "payout_type": payout,
            "maturity_amount": json.loads(banking_tools.calculate_fd_maturity(20000, 6.8, 12))["maturity_amount"],
            "status": "ACTIVE", "start_date": "2025-07-16", "maturity_date": "2026-07-16",
            "maturity_instruction": "no_renewal", "payout_account": "SB0000000001",
        }
        customer["login_context"]["linked_products"].append(deposit_id)
    customer["deposits"]["RD_UNCHANGED"] = {"kind": "RD", "maturity_amount": 23456}
    path = Path(engine._DB_PATH)
    path.write_text(json.dumps(raw))
    engine.configure(db_path=str(path), kb_path=engine._KB_PATH)
    return path, path.read_bytes(), raw


@pytest.mark.parametrize("enabled", [False, True])
def test_only_periodic_contract_maturity_changes_in_independent_seed(seeded, enabled):
    path, source_bytes, original = seeded
    engine.configure(db_path=str(path), kb_path=engine._KB_PATH, fd_payout_consistency=not enabled)
    episode = engine.seed_world(CUSTOMER, fd_payout_consistency=enabled)
    expected = deepcopy(original)
    if enabled:
        for payout in ["monthly", "quarterly"]:
            record = expected["customers"][CUSTOMER]["deposits"]["FD_SEED_" + payout]
            record["maturity_amount"] = json.loads(banking_tools.calculate_fd_maturity(
                record["principal"], record["rate"], record["tenure_months"], payout
            ))["maturity_amount"]
    assert episode["db"] == expected
    assert episode["fd_payout_consistency"] is enabled
    assert engine.get_base_db() == original
    assert path.read_bytes() == source_bytes
    episode["db"]["customers"][CUSTOMER]["accounts"]["SB0000000001"]["available_balance"] = 1
    assert engine.get_base_db() == original


@pytest.mark.parametrize("payout", ["monthly", "quarterly"])
def test_native_matured_payout_and_gold_keep_seed_flag_after_reconfigure(seeded, payout):
    path, source_bytes, original = seeded
    deposit_id = "FD_SEED_" + payout
    episode = engine.seed_world(CUSTOMER, fd_payout_consistency=True)
    engine.configure(db_path=str(path), kb_path=engine._KB_PATH, fd_payout_consistency=False)
    assert episode["db"]["customers"][CUSTOMER]["deposits"][deposit_id]["maturity_amount"] == 20000
    arguments = {"deposit_id": deposit_id, "payout_account": "SB0000000001"}
    result = json.loads(engine.apply_tool(episode, "close_deposit", arguments))
    assert not episode["calls"][-1]["error"], result
    account = episode["db"]["customers"][CUSTOMER]["accounts"]["SB0000000001"]
    assert account["available_balance"] == 70000
    assert account["transactions"][:-1] == original["customers"][CUSTOMER]["accounts"]["SB0000000001"]["transactions"]
    assert account["transactions"][-1]["amount"] == 20000
    task = {"customer": CUSTOMER, "evaluation_criteria": {
        "reward_basis": ["ACTION", "DB"], "actions": [{"name": "close_deposit", "arguments": arguments}]}}
    store = {reward_mod.WORLD_KEY: episode, reward_mod.TASK_KEY: task, "nl_history": []}
    assert reward_mod.score_trajectory(store)["strict"] == 1
    noop = engine.seed_world(CUSTOMER, fd_payout_consistency=True)
    assert reward_mod.score_trajectory(store | {reward_mod.WORLD_KEY: noop})["strict"] == 0
    wrong = engine.seed_world(CUSTOMER, fd_payout_consistency=False)
    engine.apply_tool(wrong, "close_deposit", arguments)
    assert not wrong["calls"][-1]["error"]
    wrong["fd_payout_consistency"] = True
    assert reward_mod.score_trajectory(store | {reward_mod.WORLD_KEY: wrong})["strict"] == 0
    assert path.read_bytes() == source_bytes
    assert engine.get_base_db() == original


async def test_server_reset_selects_seed_before_another_servers_global_setting(seeded):
    path, source_bytes, _ = seeded
    client = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=8080, name="head", entrypoint=""),
        global_config_dict=OmegaConf.create({}),
    )
    legacy = IndianBankingResourcesServer(
        config=_config(db_fpath=str(path), kb_fpath=engine._KB_PATH), server_client=client
    )
    corrected = IndianBankingResourcesServer(
        config=_config(db_fpath=str(path), kb_fpath=engine._KB_PATH, fd_payout_consistency=True),
        server_client=client,
    )
    metadata = {"customer": CUSTOMER, "opening_message": "Please check my quarterly deposit."}
    await legacy.reset(metadata, session_id="legacy-seed")
    await corrected.reset(metadata, session_id="corrected-seed")
    old = legacy.session_state["legacy-seed"][reward_mod.WORLD_KEY]
    new = corrected.session_state["corrected-seed"][reward_mod.WORLD_KEY]
    assert old["fd_payout_consistency"] is False
    assert new["fd_payout_consistency"] is True
    assert old["db"]["customers"][CUSTOMER]["deposits"]["FD_SEED_quarterly"]["maturity_amount"] > 20000
    assert new["db"]["customers"][CUSTOMER]["deposits"]["FD_SEED_quarterly"]["maturity_amount"] == 20000
    assert old["db"]["customers"][CUSTOMER]["accounts"] == new["db"]["customers"][CUSTOMER]["accounts"]
    assert path.read_bytes() == source_bytes
