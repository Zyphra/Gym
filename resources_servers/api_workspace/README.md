# API Workspace

API Workspace is an offline simulation of fictional SaaS vendors for HR, CRM, accounting, storage, email and chat. An agent receives a business task, discovers the available APIs, reads records and policy documents, and changes the workspace through API calls. A deterministic grader checks the final state.

Example: approve leave and transfer a CRM account to the delegate while preserving the employee record. Other workflows cover invoices, expenses, files, email and chat.

## Tools and discovery

| Tool | Purpose |
| --- | --- |
| `workspace_catalog` | Search endpoint metadata by operation, domain, collection or field; inspect account connection status. |
| `workspace_schema` | Get a vendor’s exact paths, field aliases, methods and query parameters. |
| `workspace_request` | Read or modify records with `GET`, `POST`, `PATCH` or `DELETE`. |
| `workspace_parallel` | Batch up to 20 independent requests; results preserve input order and individual status codes. |

Catalog results contain API metadata. Task records and document contents must be fetched with requests. The server also retains `/workspace_accounts` for direct clients; it is not a fifth tool exposed in task rows.

There are 18 fictional vendors: three in each domain. Each task connects two per domain and distributes records across them. Vendors share business concepts but use different URL prefixes, collection names and field aliases. Calling a disconnected account returns 401; schema discovery does not connect it.

An approved JSON policy document in the file store supplies business values and record-placement rules. A marked superseded policy is a distractor. Records are spread across a domain’s two connected vendors and paginate, so the agent combines results across vendors and pages.

## How tasks are built

[scenarios.py](scenarios.py) defines 30 workflow templates authored in code and deterministic world construction. A task ID, root seed and V4 source determine the prompt, initial records, policy document and expected changes. Variants change values, names, record counts, vendor connections, placement and instruction layout. Construction uses no model calls or external task corpus.

TRAIN contains 80 variants per family (**2,400 tasks**); VAL contains eight (**240 tasks**). VAL has different seeds but shares TRAIN templates. It is an instance-validation split, not evidence of transfer to unseen workflow families. The pilot selects one TRAIN variant per family.

[RLTasks’ API Workspace codec](https://github.com/Zyphra/rl_tasks/tree/wrapper/envs/api_workspace) prepares rows containing the prompt, four tools and canonical task ID. Expected changes remain server-side. The specs lock source and prompt versions. This source rejects task IDs from other revisions, so prepared data and source must come from the same revision.

## Scoring

[Workspace.score](engine.py) returns **1 for a correct final state and 0 otherwise**. It checks required field updates, deletions and creations, and preservation of unrelated records and fields. Any missing or extra change to the final state fails. Action order and generated IDs for created records do not affect the score.

New file contents and message/draft bodies allow CRLF/LF and one trailing newline; other content and preserved records are exact. Scoring uses stored state, with no model judge. Training cost penalties belong to the consuming run.

## Request and session behavior

- Supply the exact API `path`, query `params` and write `body` separately. Embedded query strings in paths are rejected. Amounts are integer cents.
- `params` and `body` accept JSON objects, serialized JSON objects or null; malformed containers return validation errors.
- List/search requests default to **three records per page** and return `next_cursor`. Reuse cursors only with the same vendor, collection and query. Catalog pagination uses `offset`/`next_offset` instead.
- `POST` requires an idempotency key. Identical retries return the same object; conflicting reuse returns 409. `PATCH` and `DELETE` use item paths.
- A batch is not atomic: each request succeeds or fails on its own, results return in input order, and all requests share the world’s session lock. Put dependent operations in separate calls.
- Each session cookie owns an isolated world reconstructed from its task ID. Reseeding resets it. Native verification scores and closes it; the RLTasks provider exposes scoring and close separately.

The stock `simple_agent` limit is `AGENT_MAX_ASSISTANT_TURNS` (**200** when unset), which the wrapper sets from `runtime.agent_max_assistant_turns`. It counts model turns. Context and response-token limits can end an episode earlier.

## Run and inspect

Prepare a dataset using the [RLTasks instructions](https://github.com/Zyphra/rl_tasks/blob/wrapper/envs/api_workspace/README.md#prepare-and-run), then use a writable output directory:

```bash
gym env test --resources-server api_workspace
gym eval run --config resources_servers/api_workspace/configs/api_workspace.yaml \
  --model-type vllm_model --split train --limit 5 --num-repeats 1 --concurrency 2 \
  --model-url "$POLICY_BASE_URL" --model "$POLICY_MODEL" \
  --output "$OUTPUT/rollouts.jsonl" --max-output-tokens 16384 --temperature 1.0 \
  "++api_workspace_agent.responses_api_agents.simple_agent.datasets=[{name:authored_train,type:train,jsonl_fpath:$API_WORKSPACE_DATA_JSONL,license:Apache\ 2.0}]"
```

The config’s default dataset is five example rows. The override above selects prepared rows. Sampling and the output-token value are settings for this example invocation.

Tests in [tests/](tests/) cover goal replay, wrong-state controls, preservation, schemas, API errors and session behavior. Inspect real rollouts for discoverability and answerability; mechanical checks do not establish learnability or transfer.

Reviewed V4 TRAIN tasks are selected without a policy pass-rate prerequisite. Current model results, mixture choices and production adoption are owned by the consuming run; keep VAL out of TRAIN.
