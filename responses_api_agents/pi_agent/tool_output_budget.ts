// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * Keeps every request Pi sends inside the model's context window when tool output is dense. PiAgent loads
 * it on every run.
 *
 * Pi sizes a request from the last reply's real token usage plus 4 characters per token for everything
 * appended since (`estimateContextTokens`), asks for `contextWindow - estimate - 4096` output tokens, and
 * compacts first only when that estimate passes `contextWindow - reserveTokens`. Its tool output cap is
 * 50 KB, and dense JSON or numbers tokenize at about one token per byte, so one such result can sit 30k
 * tokens above the estimate: the request overruns the window, the server answers 400, and the overflow
 * recovery's summary request overruns it too.
 *
 * Before a tool result enters the history, this extension counts it at one token per UTF-8 byte (a bound
 * for a byte-level tokenizer) and cuts it so the next request fits, whichever of Pi's two sizes it gets:
 * - with the model's full output budget: real context <= contextWindow - maxTokens, or
 * - with an output budget clamped by the estimate: real context - estimate <= Pi's 4096-token safety margin.
 * A cut result keeps its head and its last lines (where Pi's own truncation notice is) around a notice.
 * Each cut appends a `pi_gym_tool_output_budget` entry that PiAgent counts into the response metadata.
 */

export const LABEL = "pi_gym_tool_output_budget";
// Pi's own margins: CONTEXT_SAFETY_TOKENS in pi-ai's clampMaxTokensToContext, and chars per token.
const PI_SAFETY_TOKENS = 4096;
const PI_CHARS_PER_TOKEN = 4;
// Our slack inside those, for chat-template framing of the result and the request.
const MARGIN_TOKENS = 1024;
const MESSAGE_OVERHEAD = 16;
const TAIL_CHARS = 400;
const NO_USAGE_SCHEMA_TOKENS = 4096;

const bytes = (text: string): number => Buffer.byteLength(text, "utf8");

const textOf = (content: any): string =>
  typeof content === "string"
    ? content
    : Array.isArray(content)
      ? content.filter((b: any) => b?.type === "text").map((b: any) => b.text ?? "").join("\n")
      : "";

/** Upper bound (bytes) and Pi's estimate (chars / 4) of one message's tokens. */
export function messageCost(message: any): { real: number; est: number } {
  let text = "";
  if (message?.role === "assistant" && Array.isArray(message.content)) {
    for (const block of message.content) {
      if (block?.type === "text") text += block.text ?? "";
      else if (block?.type === "thinking") text += block.thinking ?? "";
      else if (block?.type === "toolCall") text += (block.name ?? "") + JSON.stringify(block.arguments ?? {});
    }
  } else if (message?.role === "compactionSummary") {
    text = message.summary ?? "";
  } else {
    text = textOf(message?.content);
  }
  return { real: bytes(text) + MESSAGE_OVERHEAD, est: Math.ceil(text.length / PI_CHARS_PER_TOKEN) };
}

const usageTokens = (usage: any): number =>
  !usage ? 0 : usage.totalTokens || (usage.input ?? 0) + (usage.output ?? 0) + (usage.cacheRead ?? 0) + (usage.cacheWrite ?? 0);

/**
 * The context the next request would carry, as Pi estimates it and as an upper bound, from the last reply
 * with real usage (as Pi's estimateContextTokens anchors it) plus everything after it.
 */
export function contextNow(messages: any[], systemPrompt: string, pending: Map<string, string>) {
  let anchor = -1;
  let latestPrefix = -Infinity;
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    if (m?.role === "assistant" && (m.timestamp ?? 0) >= latestPrefix && m.stopReason !== "error" &&
        m.stopReason !== "aborted" && usageTokens(m.usage) > 0) anchor = i;
    latestPrefix = Math.max(latestPrefix, m?.timestamp ?? 0);
  }
  let real = 0;
  let est = 0;
  if (anchor >= 0) {
    real = est = usageTokens(messages[anchor].usage);
  } else {
    // No reply since the last compaction: the system prompt, plus room for the tool schemas Pi also sends.
    const system = messageCost({ content: systemPrompt });
    real = system.real + NO_USAGE_SCHEMA_TOKENS;
    est = system.est;
  }
  const seen = new Set<string>();
  for (const m of messages.slice(anchor + 1)) {
    if (m?.role === "toolResult" && m.toolCallId) seen.add(m.toolCallId);
    const cost = messageCost(m);
    real += cost.real;
    est += cost.est;
  }
  // Results of this turn's other calls that are decided but not yet appended (parallel tool calls).
  for (const [id, text] of pending) {
    if (seen.has(id)) continue;
    const cost = messageCost({ content: text });
    real += cost.real;
    est += cost.est;
  }
  return { real, est };
}

/** Whether a result of `text` keeps the next request inside the window under either of Pi's output sizes. */
export function fits(text: string, now: { real: number; est: number }, contextWindow: number, maxTokens: number) {
  const cost = messageCost({ content: text });
  const real = now.real + cost.real;
  const est = now.est + cost.est;
  return real <= contextWindow - maxTokens - MARGIN_TOKENS || real - est <= PI_SAFETY_TOKENS - MARGIN_TOKENS;
}

export function notice(keptBytes: number, totalBytes: number): string {
  return `\n\n[Tool output cut to fit the context window: kept ${keptBytes} of ${totalBytes} bytes (start and end). ` +
    "Narrow the command (filter, head, grep, jq) to see the rest.]\n\n";
}

/** The longest head + notice + tail of `text` that fits; the notice alone when nothing more does. */
export function cut(text: string, ok: (candidate: string) => boolean): string {
  const total = bytes(text);
  const tail = text.length > TAIL_CHARS * 2 ? text.slice(-TAIL_CHARS) : "";
  const build = (n: number) => {
    const head = text.slice(0, n);
    return head + notice(bytes(head) + bytes(tail), total) + tail;
  };
  let lo = 0;
  let hi = Math.max(0, text.length - tail.length);
  if (!ok(build(0))) return notice(0, total).trim();
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (ok(build(mid))) lo = mid;
    else hi = mid - 1;
  }
  return build(lo);
}

export default function toolOutputBudget(pi: any) {
  // Results this extension returned since the last reply, by tool call id.
  const pending = new Map<string, string>();

  pi.on("message_end", (event: any) => {
    if (event.message?.role === "assistant") pending.clear();
    return undefined;
  });

  pi.on("tool_result", (event: any, ctx: any) => {
    const contextWindow = ctx.model?.contextWindow ?? 0;
    const maxTokens = ctx.model?.maxTokens ?? 0;
    const content: any[] = Array.isArray(event.content) ? event.content : [];
    if (contextWindow <= 0 || !content.some((b) => b?.type === "text")) return undefined;
    const text = textOf(content);
    const messages = ctx.sessionManager.buildSessionProjection().messages;
    const now = contextNow(messages, ctx.getSystemPrompt?.() ?? "", pending);
    const ok = (candidate: string) => fits(candidate, now, contextWindow, maxTokens);
    if (ok(text)) {
      pending.set(event.toolCallId, text);
      return undefined;
    }
    const kept = cut(text, ok);
    pending.set(event.toolCallId, kept);
    try {
      pi.appendEntry(LABEL, { tool: event.toolName, bytes: bytes(text), kept_bytes: bytes(kept) });
    } catch {
      /* no session to record in */
    }
    const images = content.filter((b) => b?.type !== "text");
    return { content: [{ type: "text", text: kept }, ...images] };
  });
}
