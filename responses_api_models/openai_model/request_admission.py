# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conservative, per-process admission before an upstream model request."""

import asyncio
import json
import time
from collections import deque
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RequestRateLimit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requests_per_period: int = Field(gt=0, strict=True)
    tokens_per_period: int = Field(gt=0, strict=True)
    period_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    output_token_reservation: int = Field(gt=0, strict=True)
    input_token_overhead: int = Field(default=1024, ge=0, strict=True)
    admission_timeout_seconds: float = Field(default=300.0, gt=0, allow_inf_nan=False)

    def request_tokens(self, body: dict[str, Any]) -> int:
        """Reserve UTF-8 bytes plus output capacity; never alter the API body.

        The byte count deliberately overestimates text tokens and includes
        tool schemas and message framing. The caller declares output capacity
        when an upstream request omits an explicit output limit. This is quota
        accounting, not a generation limit or a tokenizer replacement.
        """
        encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        explicit = [
            body[name]
            for name in ("max_output_tokens", "max_completion_tokens", "max_tokens")
            if body.get(name) is not None
        ]
        if any(type(value) is not int or value < 1 for value in explicit):
            raise ValueError("output token limit must be a positive integer")
        return len(encoded) + self.input_token_overhead + max([self.output_token_reservation, *explicit])


class RequestAdmission:
    """Reserve both budgets atomically; every dispatched attempt costs a slot.

    Reservations expire after the full period, including failed and cancelled
    upstream requests. Waiting cancellation reserves nothing. No generation
    or stateful environment call is retried here.
    """

    def __init__(self, config: RequestRateLimit):
        self.config = config
        self._lock = asyncio.Lock()
        self._entries: deque[tuple[float, int]] = deque()
        self._tokens = 0

    async def acquire(self, body: dict[str, Any]) -> None:
        cost = self.config.request_tokens(body)
        if cost > self.config.tokens_per_period:
            raise ValueError("request exceeds configured model token admission budget")
        deadline = time.monotonic() + self.config.admission_timeout_seconds
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._entries and self._entries[0][0] + self.config.period_seconds <= now:
                    _, expired = self._entries.popleft()
                    self._tokens -= expired
                if (
                    len(self._entries) < self.config.requests_per_period
                    and self._tokens + cost <= self.config.tokens_per_period
                ):
                    self._entries.append((now, cost))
                    self._tokens += cost
                    return
                remaining = deadline - now
                if remaining <= 0:
                    raise TimeoutError("model request rate admission deadline exceeded")
                delay = min(self._entries[0][0] + self.config.period_seconds - now, remaining)
            await asyncio.sleep(delay)
