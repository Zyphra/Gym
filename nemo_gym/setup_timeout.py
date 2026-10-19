# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply the caller's setup floor without changing execution budgets."""

from collections.abc import Mapping


def setup_timeout_seconds(existing: float | None, *, config: Mapping | None = None) -> float | None:
    """Keep disabled/unbounded values and longer existing startup allowances."""
    if existing is None or existing <= 0:
        return existing
    if config is None:
        from nemo_gym.global_config import maybe_get_global_config_dict

        config = maybe_get_global_config_dict()
    floor = None if config is None else config.get("setup_timeout")
    return existing if floor is None else max(existing, floor)
