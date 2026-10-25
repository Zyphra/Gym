# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional notification at the real policy-to-grading boundary."""

import asyncio
import logging

from aiohttp import ClientTimeout

from nemo_gym.server_utils import request


async def notify_policy_phase_done(params) -> bool:
    """Release training policy capacity; an unavailable observer never skips grading.

    The training integration supplies this URL in request metadata. Ordinary Gym
    requests have none. The receiver owns idempotence and rejects later policy turns.
    """
    url = (params.metadata or {}).get("_ng_policy_completion_url")
    if not url:
        return False
    try:
        async with asyncio.timeout(2):
            response = await request(
                "POST",
                url,
                _internal=True,
                _max_connection_retries=1,
                timeout=ClientTimeout(total=2),
            )
            async with response:
                response.raise_for_status()
                await response.read()
        return True
    except Exception as error:
        logging.getLogger(__name__).warning("Policy completion notification failed: %s", type(error).__name__)
        return False
