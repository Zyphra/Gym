# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Use served token usage for Terminus histories, including hidden reasoning."""

import hashlib
import inspect
import json

from harbor.agents.terminus_2.terminus_2 import Terminus2


def history_key(messages):
    # Token provenance and optional reasoning echoes do not change visible history.
    visible = [{"role": m.get("role"), "content": m.get("content") or ""} for m in messages]
    return (len(messages), hashlib.sha256(json.dumps(visible, sort_keys=True).encode()).digest())


def record_served_context(llm, messages, content, usage):
    if usage is not None and usage.prompt_tokens > 0:
        history = [*messages, {"role": "assistant", "content": content}]
        llm._served_context_tokens[history_key(history)] = usage.prompt_tokens + usage.completion_tokens


def _served_count(count):
    if inspect.iscoroutinefunction(Terminus2._count_total_tokens):

        async def completed():
            return count

        return completed()
    return count


class ServedContextTerminus2(Terminus2):
    def _count_total_tokens(self, chat):
        counts = getattr(self._llm, "_served_context_tokens", {})
        for end in range(len(chat.messages), 0, -1):
            served = counts.get(history_key(chat.messages[:end]))
            if served is not None:
                if end == len(chat.messages):
                    return _served_count(served)
                from litellm.utils import token_counter

                # Only unserved appended messages need a local estimate.
                return _served_count(served + token_counter(model=self._model_name, messages=chat.messages[end:]))
        return super()._count_total_tokens(chat)
