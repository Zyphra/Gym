// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import http from "node:http";
import { pathToFileURL } from "node:url";
import { join } from "node:path";
import { readFile } from "node:fs/promises";

const [runtime, compatJSON] = process.argv.slice(2);
const pkg = join(runtime, "lib/node_modules/@earendil-works/pi-coding-agent/node_modules/@earendil-works/pi-ai");
const version = JSON.parse(await readFile(join(pkg, "package.json"), "utf8")).version;
const { streamSimple } = await import(pathToFileURL(join(pkg, "dist/api/openai-completions.js")));
const requests = [];
const server = http.createServer(async (req, res) => {
  let body = "";
  for await (const chunk of req) body += chunk;
  requests.push({ path: req.url, body: JSON.parse(body) });
  res.writeHead(200, { "content-type": "text/event-stream" });
  const chunk = (delta, finish_reason = null) => ({ id: "fixture", object: "chat.completion.chunk", created: 1,
    model: "fixture-policy", choices: [{ index: 0, delta, finish_reason }] });
  for (const x of [chunk({ role: "assistant", reasoning_content: "exact sampled thinking" }),
    chunk({ content: "answer" }), chunk({}, "stop")]) res.write(`data: ${JSON.stringify(x)}\n\n`);
  res.end("data: [DONE]\n\n");
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
try {
  const base = { id: "fixture-policy", name: "fixture-policy", api: "openai-completions", provider: "nemo",
    baseUrl: `http://127.0.0.1:${server.address().port}/ng-rollout/pi-session/v1`, reasoning: true, input: ["text"],
    contextWindow: 131072, maxTokens: 64, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
  const counts = [];
  for (const compat of [{ supportsDeveloperRole: false, supportsReasoningEffort: false }, JSON.parse(compatJSON)]) {
    const model = { ...base, compat };
    const options = { apiKey: "local-fixture", maxTokens: 64, maxRetries: 0 };
    const first = await streamSimple(model, { messages: [{ role: "user", content: "fixture", timestamp: 1 }] }, options).result();
    assert.equal(first.stopReason, "stop");
    let echoes = 0;
    for (const history of [
      [{ role: "user", content: "fixture", timestamp: 1 }, first, { role: "user", content: "continue", timestamp: 2 }],
      [first, { role: "user", content: "continue after summary", timestamp: 3 }],
    ]) {
      const response = await streamSimple(model, { messages: history }, options).result();
      assert.equal(response.stopReason, "stop");
      const assistant = requests.at(-1).body.messages.find(m => m.role === "assistant");
      assert.equal(assistant.reasoning_content, "exact sampled thinking");
      assert.equal(assistant.content, "answer");
      assert.equal(requests.at(-1).body.model, "fixture-policy");
      echoes++;
    }
    counts.push(echoes);
  }
  console.log(JSON.stringify({ version, baseline_echoes: counts[0], explicit_echoes: counts[1], paths: requests.map(r => r.path) }));
} finally {
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
}
