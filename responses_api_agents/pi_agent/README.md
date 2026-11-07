# Pi Agent

Runs the [pi](https://github.com/earendil-works/pi) CLI (`pi --print --mode json --no-session`). 
pi runs its own tools internally. Resources server for verifier.

The optional Gym model-server binding routes main and compaction calls through
the admitted episode endpoint. Token capture belongs to that model server.

## Quick start

pi must be on PATH (auto-installed on first start, or `npm install -g @earendil-works/pi-coding-agent`).
Put `policy_base_url`, `policy_api_key`, and `policy_model_name` in `env.yaml`.

```bash
gym env start \
  --config environments/pi_math/config.yaml \
  --model-type openai_model

gym eval run --no-serve --agent pi_math_agent \
  --input environments/pi_math/data/example.jsonl \
  --output pi_rollout.jsonl --limit 5
```

Per request the agent writes `models.json` into an isolated `HOME`, runs one `pi` invocation with
stdin from `/dev/null`, then parses the jsonl `message_end` events. Example rollouts are in
`environments/pi_math/data/example_rollouts.jsonl`.

## Model id

`model` is `<provider>/<model-id>`. Define the provider in `models_config` (written to
`~/.pi/agent/models.json`) and reference it here:

```yaml
model: nvinf/nvidia/qwen/qwen3-next-80b-a3b-instruct
models_config:
  providers:
    nvinf:
      baseUrl: ${policy_base_url}
      api: openai-completions
      apiKey: ${policy_api_key}
      models:
      - id: nvidia/qwen/qwen3-next-80b-a3b-instruct
        reasoning: false
```

Alternatively, set `model_server` to a Gym model server and set `model` to its served model id. The
agent creates the Pi provider entry automatically. Without `model_server`, the existing provider
configuration is unchanged.

The generated provider explicitly keeps thinking in `reasoning_content`, rather
than folding it into answer text or relying on URL detection. Pi-ai 0.99.2
already replays a sampled thinking block under its captured wire signature;
the explicit compatibility settings preserve this behavior for episode URLs
and supply an empty reasoning field for assistant turns without one. They do
not change the model's thinking format or generation effort.

The native provider regression uses an installed runtime and loopback HTTP:
`TASK_STOCK_PI_RUNTIME=<runtime> pytest responses_api_agents/pi_agent/tests/test_app.py -k native_thinking`.
This checks two-turn reasoning echo, including a shortened history, without a
model call. Model rollout and multi-segment training qualification are separate.

## Config fields

- `concurrency`: max simultaneous `run()` calls
- `command`: the pi command, split on spaces so a multi-word launcher works
- `model`: `<provider>/<model-id>` (see Model id)
- `model_server`: optional Gym model server used to generate the provider entry
- `context_window`: context limit for a generated model entry
- `max_output_tokens`: output limit for a generated model entry
- `env`: extra env vars for the subprocess (e.g. provider API keys)
- `workspace_root`: where per-request HOMEs are created and deleted
- `working_directory`: optional CLI working directory. When absent, the CLI uses
  its temporary per-request workspace. An explicitly supplied directory is never
  removed; only the temporary HOME workspace is removed after the invocation. AnySWE
  supplies its task working directory unless explicitly configured otherwise.
- `thinking`: passed to `--thinking` (off, minimal, low, medium, high, xhigh)
- `system_prompt`: appended via `--append-system-prompt`
- `timeout`: seconds for the `pi` run
- `hide_harness_mounts`: run Pi's shell commands without the runner, Gym and harness-runtime mounts
  (`nemo_gym/hidden_mounts.py`, as Hermes does). Pi's bash tool takes the hiding shell as its `shellPath`;
  an extension that spawns its own shell reads `PI_HIDE_HARNESS_MOUNTS=1`.
- `max_turns`: assistant requests per run (Pi has none of its own). When set, the agent loads
  `turn_limit.ts`, which aborts the request after the last allowed turn, once that turn's tools have run.
- `extra_args`: extra flags appended to the `pi` command
- `models_config`: written to `~/.pi/agent/models.json`
- `pi_version`: npm version to pin on install (null means latest)

See `configs/pi_agent.yaml`.

## Budget stops

A run that ends on a budget reports it the way the OpenHands adapter does, with no response error and
`metadata.stop_reason`, so AnySWE grades and keeps the partial patch:

- `max_iterations`: `max_turns` was reached;
- `context_exhausted`: the last request failed with the endpoint's context-overflow error after Pi's
  own overflow recovery, or the last reply was cut where the context window ends.

Any other non-zero exit stays a `failed` response with error `Pi process_exit_<code>`.
