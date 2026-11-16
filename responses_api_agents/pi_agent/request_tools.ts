// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * PiAgentConfig.request_tools: the request's function tools as Pi tools. The agent writes the
 * tools, the resources server URL and the episode's session cookie to the file named by
 * PI_GYM_REQUEST_TOOLS. Each tool keeps the request's name, description and JSON schema, and a
 * call is `POST {url}/{name}` with the arguments as the JSON body: the request Gym's
 * simple_agent makes. The plain-text reply is the tool output.
 */
import { readFileSync } from "node:fs";
import { request } from "node:http";

type Forwarded = { name: string; description: string; parameters: Record<string, unknown> };
type Forwarding = { url: string; cookie: string; tools: Forwarded[] };

// node:http rather than fetch: undici's 300 s header timeout would cut a tool call that runs to a
// 300 s ceiling; the resources server owns every tool deadline.
function post(url: string, cookie: string, body: string, signal: AbortSignal): Promise<[number, string]> {
  return new Promise((resolve, reject) => {
    const req = request(url, {
      method: "POST",
      headers: { "content-type": "application/json", "content-length": Buffer.byteLength(body), cookie },
      signal,
    }, (res) => {
      const chunks: Buffer[] = [];
      res.on("data", (chunk: Buffer) => chunks.push(chunk));
      res.on("end", () => resolve([res.statusCode ?? 0, Buffer.concat(chunks).toString("utf8")]));
      res.on("error", reject);
    });
    req.on("error", reject);
    req.end(body);
  });
}

// Tool input schemas must be objects; some providers reject object schemas without properties.
const toParameters = (schema: Record<string, unknown>) => ({
  ...schema,
  type: schema.type ?? "object",
  ...(schema.properties === undefined ? { properties: {} } : {}),
});

export default function requestTools(pi: any) {
  const path = process.env.PI_GYM_REQUEST_TOOLS;
  if (!path) throw new Error("PI_GYM_REQUEST_TOOLS is not set");
  const forwarding: Forwarding = JSON.parse(readFileSync(path, "utf8"));
  for (const tool of forwarding.tools) {
    pi.registerTool({
      name: tool.name,
      label: tool.name,
      description: tool.description,
      parameters: toParameters(tool.parameters),
      async execute(_id: string, params: unknown, signal: AbortSignal) {
        const url = `${forwarding.url}/${encodeURIComponent(tool.name)}`;
        const [status, text] = await post(url, forwarding.cookie, JSON.stringify(params ?? {}), signal);
        // As in simple_agent, a resources-server error reply is the model-visible tool output.
        const failed = status < 200 || status >= 400;
        return { content: [{ type: "text", text }], details: undefined, ...(failed ? { isError: true } : {}) };
      },
    });
  }
}
