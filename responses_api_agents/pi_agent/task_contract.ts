// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * Keeps the task in front of `pi --print` across context compaction. An evaluation binding can
 * declare an answer contract for one continuation. PiAgent loads it on every run.
 *
 * - Restate: Pi's compaction replaces the messages that carried the task (the first user message) with
 *   a summary. Once a compaction summary is in context and the instruction is not, every request carries
 *   the original instruction again as a user message right after the summary. It is request-local (the
 *   `context` event), so a run that never compacts sends exactly the requests it sent before, and each
 *   request after a compaction extends the previous one.
 * - Continue: when an answer contract is declared and the run settles on an assistant reply (not on a terminating tool, an abort, an error or
 *   a reply cut at the output limit, which Pi's own length recovery retries once) that is empty or carries neither the answer the task declares
 *   (PI_GYM_ANSWER_PATTERN on the reply) nor its answer file (PI_GYM_ANSWER_FILES), the extension asks
 *   for the final answer once. Never past PI_GYM_MAX_TURNS, which turn_limit.ts enforces.
 * - Label: the last settle appends a `pi_gym_task_contract` entry (restated requests, continuation reason)
 *   that PiAgent copies into the response metadata.
 */
import { existsSync, readFileSync } from "node:fs";

export const LABEL = "pi_gym_task_contract";
export const RESTATE_PREFIX =
  "Your earlier context was summarized. The task, restated exactly as it was given:\n\n";

const textOf = (content: any): string =>
  typeof content === "string"
    ? content
    : Array.isArray(content)
      ? content.filter((b: any) => b?.type === "text").map((b: any) => b.text ?? "").join("\n")
      : "";

const answerFiles = (): string[] => {
  const raw = process.env.PI_GYM_ANSWER_FILES;
  if (!raw) return [];
  const files = JSON.parse(raw);
  if (!Array.isArray(files) || files.some((f) => typeof f !== "string")) {
    throw new Error("PI_GYM_ANSWER_FILES must be a JSON list of paths");
  }
  return files;
};

export function missingReason(last: any, pattern: RegExp | null, files: string[]): string | null {
  if (!pattern && files.length === 0) return null;
  if (last?.role !== "assistant") return null; // ended on a terminating tool's result
  // A reply cut at the output limit ("length") already had Pi's one compact-and-retry; aborted (turn limit)
  // and error ends are not ours to retry.
  if (last.stopReason !== "stop") return null;
  const content = Array.isArray(last.content) ? last.content : [];
  if (content.some((b: any) => b?.type === "toolCall")) return null;
  const reply = textOf(content).trim();
  if (!reply) return "empty_final_reply";
  if (pattern && pattern.test(reply)) return null;
  for (const file of files) {
    if (!existsSync(file)) continue;
    const text = readFileSync(file, "utf8");
    if (text.trim() && (!pattern || pattern.test(text))) return null;
  }
  return "answer_missing";
}

export function continuationMessage(reason: string): string {
  const what = {
    empty_final_reply: "Your last reply was empty.",
    answer_missing: "Your last reply does not contain the final answer in the form the task requires.",
  }[reason];
  return `${what} The task is not finished until its deliverable exists. Give your final answer now, ` +
    "exactly in the format and location the task specifies, and keep it short.";
}

export default function taskContract(pi: any) {
  const patternSource = process.env.PI_GYM_ANSWER_PATTERN;
  const pattern = patternSource ? new RegExp(patternSource) : null;
  const files = answerFiles();
  const maxTurns = Number(process.env.PI_GYM_MAX_TURNS) || Infinity;
  let instruction: string | null = null;
  let requests = 0;
  let restated = 0;
  let continuation: string | null = null;

  pi.on("before_agent_start", (event: any) => {
    if (instruction === null && typeof event.prompt === "string" && event.prompt.trim()) instruction = event.prompt;
    return undefined;
  });

  pi.on("context", (event: any) => {
    requests += 1;
    const messages: any[] = event.messages;
    const summaryAt = messages.map((m) => m?.role).lastIndexOf("compactionSummary");
    if (instruction === null || summaryAt < 0) return undefined;
    const task = instruction;
    if (messages.some((m) => m?.role === "user" && textOf(m.content).includes(task))) return undefined;
    restated += 1;
    const restatement = {
      role: "user",
      content: [{ type: "text", text: RESTATE_PREFIX + task }],
      timestamp: messages[summaryAt].timestamp,
    };
    return { messages: [...messages.slice(0, summaryAt + 1), restatement, ...messages.slice(summaryAt + 1)] };
  });

  pi.on("agent_before_settle", (event: any) => {
    // An aborted or failed run (turn_limit.ts aborts the request after the last turn) settles untouched.
    if (event.outcome !== "completed") return undefined;
    const label = { type: "custom", customType: LABEL, data: { restated_requests: restated, continuation } };
    if (continuation !== null || requests >= maxTurns) return { entries: [label] };
    const history = event.context?.contextMessages ?? [];
    const reason = missingReason(history[history.length - 1], pattern, files);
    if (reason === null) return { entries: [label] };
    continuation = reason;
    return {
      entries: [
        { type: "custom", customType: LABEL, data: { restated_requests: restated, continuation } },
        { type: "custom_message", customType: LABEL, content: continuationMessage(reason), display: true },
      ],
      continue: true,
    };
  });
}
