# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf

from nemo_gym.cli.env import _model_endpoint_timeout_seconds
from nemo_gym.sandbox.providers.enroot import provider
from nemo_gym.setup_timeout import setup_timeout_seconds


@pytest.mark.parametrize("existing,expected", [(None, None), (0, 0), (-1, -1), (30, 1800), (600, 1800), (3600, 3600)])
def test_setup_floor_preserves_longer_and_disabled(existing, expected):
    assert setup_timeout_seconds(existing, config={"setup_timeout": 1800}) == expected
    assert setup_timeout_seconds(existing, config={}) == existing


def test_real_provider_resolves_defaults_before_floor(monkeypatch, tmp_path):
    import nemo_gym.global_config as global_config

    monkeypatch.setattr(global_config, "maybe_get_global_config_dict", lambda: {"setup_timeout": 1800})
    monkeypatch.setattr(provider, "_require_enroot", lambda: "/bin/true")
    instance = provider.EnrootProvider(
        create=provider.EnrootCreateConfig(
            base_dir=str(tmp_path), import_timeout_s=3600, create_timeout_s=None, start_timeout_s=30
        )
    )
    assert instance._create_config.import_timeout_s == 3600
    assert instance._create_config.create_timeout_s is None
    assert instance._create_config.start_timeout_s == 1800
    assert instance._probe.timeout_s == 1800
    assert instance._probe.deadline_s is None
    assert instance._exec_config.default_timeout_s == 180


@pytest.mark.parametrize("existing,expected", [(None, 1800), (0, 0), (-1, -1), (30, 1800), (3600, 3600)])
def test_model_readiness_floor_preserves_disabled(existing, expected):
    config = OmegaConf.create({"setup_timeout": 1800, "model_endpoint_readiness_timeout_seconds": existing})
    assert _model_endpoint_timeout_seconds(config) == expected


@pytest.mark.parametrize("name", ["apptainer", "docker"])
def test_other_real_providers_floor_setup_only(monkeypatch, name):
    import importlib

    import nemo_gym.global_config as global_config

    module = importlib.import_module(f"nemo_gym.sandbox.providers.{name}.provider")
    monkeypatch.setattr(global_config, "maybe_get_global_config_dict", lambda: {"setup_timeout": 1800})
    monkeypatch.setattr(module, f"_require_{name}", lambda *args: "/bin/true")
    cls = getattr(module, name.title() + "Provider")
    instance = cls()
    assert instance._create_config.start_timeout_s == 1800
    assert instance._probe.timeout_s == 1800
    assert instance._probe.deadline_s is None
    assert instance._exec_config.default_timeout_s == 180


def test_setup_floor_survives_real_global_config_parser():
    from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig

    initial = OmegaConf.merge(GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT, {"setup_timeout": 1800})
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial, skip_load_from_cli=True, skip_load_from_dotenv=True, offline=True
        )
    )
    assert resolved["setup_timeout"] == 1800
    assert setup_timeout_seconds(600, config=resolved) == 1800
