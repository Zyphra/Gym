// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * PiAgentConfig.max_turns for `pi --print`, which has no turn limit of its own. Every policy request
 * passes Pi's `context` event first. The request after the last allowed one is the cap request: it
 * carries a request-local message saying the turns are used up and is sent with tool calls forbidden
 * (`tool_choice: "none"`, tools still offered so the request extends the cached context), so the
 * policy's answer reaches the run as text, as Hermes' turn-cap summary and simple_agent's
 * `final_answer_after_max_steps` do. A tool call in that reply is blocked and ends the run, and any
 * later request is aborted before it is sent. The marker on stderr tells the Gym agent that this
 * extension, not a failure, stopped the run.
 */
export const MARKER = "pi_gym_turn_limit";

export function capMessage(limit: number): string {
  return (
    `You have used all ${limit} turns; tools no longer run. ` +
    "Reply now with your final answer as text, in the format the task requires."
  );
}

export default function turnLimit(pi: any) {
  const limit = Number(process.env.PI_GYM_MAX_TURNS);
  if (!Number.isInteger(limit) || limit < 1) throw new Error(`PI_GYM_MAX_TURNS must be a positive integer`);
  let requests = 0;
  pi.on("context", (event: any, ctx: any) => {
    requests += 1;
    if (requests <= limit) return undefined;
    if (requests === limit + 1) {
      process.stderr.write(`${MARKER}: ${limit} turns\n`);
      const message = { role: "user", content: [{ type: "text", text: capMessage(limit) }], timestamp: Date.now() };
      return { messages: [...event.messages, message] };
    }
    ctx.abort();
    return undefined;
  });
  pi.on("before_provider_request", (event: any) => {
    if (requests !== limit + 1 || !event.payload || typeof event.payload !== "object") return undefined;
    return { ...event.payload, tool_choice: "none" };
  });
  pi.on("tool_call", () =>
    requests > limit ? { block: true, reason: `the ${limit}-turn limit is reached`, terminate: true } : undefined,
  );
}
