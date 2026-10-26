# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def bind_policy_summary(agent) -> None:
    """Use Hermes' existing summary callback with this run's policy client.

    A process-global auxiliary provider cannot carry concurrent rollout URLs.
    The admitted native compressor exposes this per-instance callback; older
    runtimes must be replaced rather than silently making untracked summaries.
    """
    compressor = getattr(agent, "context_compressor", None)
    if compressor is None or not hasattr(compressor, "managed_summary_callback"):
        raise RuntimeError("Hermes compression requires the native managed summary callback")

    def summary(messages, max_tokens):
        kwargs = agent._build_api_kwargs(messages)
        kwargs.pop("max_completion_tokens", None)
        kwargs.update(model=agent.model, max_tokens=max_tokens, tool_choice="none")
        # The handoff is visible output. Close only this request's reasoning
        # prefill; prior reasoning and ordinary task-turn settings stay exact.
        extra_body = dict(kwargs.get("extra_body") or {})
        extra_body["chat_template_kwargs"] = {
            **(extra_body.get("chat_template_kwargs") or {}),
            "enable_thinking": False,
        }
        kwargs["extra_body"] = extra_body
        response = agent.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        content = choice.message.content
        if (
            choice.finish_reason != "stop"
            or choice.message.tool_calls
            or not isinstance(content, str)
            or not content.strip()
        ):
            raise RuntimeError("Hermes policy summary was empty, truncated, or emitted tool calls")
        return content

    compressor.managed_summary_callback = summary
