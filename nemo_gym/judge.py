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
"""Shared LLM-as-judge failure abstraction.

A failed judge call is a distinct outcome, not a wrong answer. Resources servers
issue judge calls through ``call_judge``, or wrap a bespoke call in
``reraise_judge_errors``; ``judge_failsafe`` wraps every verify endpoint so a
JudgeError becomes a row tagged ``_ng_failure_class="judge_failed"``, which
rollout_collection routes to ``<output>_failures.jsonl`` — excluded from the
aggregate metrics and from the file-based re-aggregation (``gym eval
aggregate``), and retryable on resume.

Boundary: a failed *call* (transport/timeout/auth/HTTP) → JudgeError → sidecar; a
*received-but-unparseable* response is a legitimate wrong answer (let the parser
score it, don't raise). Empty output is a per-benchmark call, so servers differ.

Wrap the call, not the scoring around it: ``reraise_judge_errors`` relabels every
exception it sees, so covering a whole ``verify()`` would misfile ordinary bugs
(a KeyError in prompt assembly, say) as judge failures.
"""

import functools
import re
from typing import Any, Awaitable, Callable, Mapping, Type, TypeVar

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from nemo_gym.server_utils import ServerClient, get_response_json, raise_for_status


ResponseT = TypeVar("ResponseT", bound=BaseModel)


class JudgeError(Exception):
    """A judge call failed; judge_failsafe routes the row to the failures sidecar."""


def parse_judge_grade(judge_text: str, labels: Mapping[str, str], default: str) -> str:
    """Map a judge reply to its grade letter, anchored to where a verdict is written.

    ``labels`` maps each label word to its letter (``{"CORRECT": "A", ...}``); the
    letters are the valid grades. Tried in order, the first hit wins:

    1. a leading letter followed by a non-letter, on the whole reply or its last
       line (covers the bare letter, ``A: CORRECT`` and ``B - INCORRECT``);
    2. an explicit ``Grade: X`` / ``grade is X`` (the last one);
    3. one distinct label word, upper case, as a whole word.

    Anything else returns ``default``. A letter or label is never matched inside
    other text, so ``NOT_ATTEMPTED`` or ``INCORRECT`` cannot read as ``A``.
    """
    letters = "".join(sorted(set(labels.values())))
    text = judge_text.strip().strip("*_`\"'()[] \t")
    last_line = text.rsplit("\n", 1)[-1].strip().strip("*_`\"'()[] \t")
    for candidate in (text, last_line):
        if m := re.match(rf"([{letters}])(?![A-Za-z])", candidate):
            return m.group(1)
    if graded := re.findall(rf"(?i:\bgrade)\s*(?:is|:|=)?\s*[*_`\"'(\[]*([{letters}])(?![A-Za-z])", text):
        return graded[-1]
    words = "|".join(sorted(map(re.escape, labels), key=len, reverse=True))
    found = {labels[w] for w in re.findall(rf"(?<![A-Za-z_])({words})(?![A-Za-z_])", text)}
    return found.pop() if len(found) == 1 else default


async def reraise_judge_errors(coro: Awaitable[Any]) -> Any:
    """Await a judge call; re-raise any exception as JudgeError (recorded verbatim)."""
    try:
        return await coro
    except JudgeError:
        raise
    except Exception as e:
        raise JudgeError(f"{type(e).__name__}: {e}") from e


async def call_judge(
    server_client: ServerClient,
    *,
    server_name: str,
    url_path: str,
    json: Any,
    response_model: Type[ResponseT],
) -> ResponseT:
    """POST to a judge model server and parse the reply; raise JudgeError on failure.

    ``url_path`` is the judge's API surface (``/v1/responses`` or
    ``/v1/chat/completions``), paired with the matching ``response_model``. The
    status check comes before parsing so an auth or server error reports the HTTP
    failure itself, not a validation error against the error body.
    """

    async def _call() -> ResponseT:
        http_response = await server_client.post(server_name=server_name, url_path=url_path, json=json)
        await raise_for_status(http_response)
        return response_model.model_validate(await get_response_json(http_response))

    from nemo_gym.call_timing import timed_call
    from nemo_gym.model_call_metadata import model_call_context

    with model_call_context("judge", preserve_operation=True), timed_call("judge"):
        return await reraise_judge_errors(_call())


def judge_failsafe(verify_fn: Callable) -> Callable:
    """Wrap verify() so a JudgeError returns a sidecar-routed row (reward 0.0, the
    routing keys, the request's ``response`` carried) instead of propagating.
    functools.wraps keeps verify's signature so FastAPI injects the same params
    (``body``, and ``request`` for servers that take it); ``*args, **kwargs`` pass
    them straight through."""

    @functools.wraps(verify_fn)
    async def wrapper(*args, **kwargs):
        try:
            return await verify_fn(*args, **kwargs)
        except JudgeError as e:
            body = kwargs.get("body") or next(
                (a for a in (*kwargs.values(), *args) if hasattr(a, "model_dump") and hasattr(a, "response")),
                None,
            )
            if body is None:  # verify always has a request body; guard against an opaque 500
                raise RuntimeError("judge_failsafe: could not locate the verify request body") from e
            data = body.model_dump() | {
                "reward": 0.0,
                "_ng_failure_class": "judge_failed",
                "_ng_failure_judge_error": str(e),
            }
            return JSONResponse(content=jsonable_encoder(data))

    return wrapper
