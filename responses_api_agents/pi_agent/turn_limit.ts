// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * PiAgentConfig.max_turns for `pi --print`, which has no turn limit of its own. Every policy request
 * passes Pi's `context` event first; the request after the last allowed one is aborted before it is
 * sent, so the run ends after the tools of its last allowed turn have run. The marker on stderr tells
 * the Gym agent that this extension, not a failure, stopped the run.
 */
export const MARKER = "pi_gym_turn_limit";

export default function turnLimit(pi: any) {
  const limit = Number(process.env.PI_GYM_MAX_TURNS);
  if (!Number.isInteger(limit) || limit < 1) throw new Error(`PI_GYM_MAX_TURNS must be a positive integer`);
  let requests = 0;
  pi.on("context", (_event: any, ctx: any) => {
    requests += 1;
    if (requests <= limit) return undefined;
    process.stderr.write(`${MARKER}: ${limit} turns\n`);
    ctx.abort();
    return undefined;
  });
}
