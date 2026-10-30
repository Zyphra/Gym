# Hermes Agent

# Quick start

## Create env.yaml in Gym/

```
policy_base_url: https://api.openai.com/v1
policy_api_key: sk...
policy_model_name: gpt-4o
```

## Launch nemo gym servers

```bash
gym env start \
    --config environments/hermes_math/config.yaml \
    --model-type openai_model
```

## Collect rollouts

```bash
gym eval run --no-serve \
    --agent hermes_math_agent \
    --input environments/hermes_math/data/example.jsonl \
    --output hermes_agent_rollout.jsonl \
    --limit 1
```

Example math rollouts are in `environments/hermes_math/data/example_rollouts.jsonl`.

Example training reward for small multi environment test is shown [here](https://github.com/NVIDIA-NeMo/Gym/pull/1033#issuecomment-4399509664).

## Description

Runs [hermes-agent](https://github.com/NousResearch/hermes-agent) in a nemo gym agent server via the `run_agent.AIAgent` entrypoint, which matches the hermes-agent CLI and user experience. Can be used for benchmarks with hermes agent, or training in the harness.

Every newly produced tool result is limited to **20,000 characters, including a
truncation note**. Smaller results remain unchanged. This applies to every tool
and both sequential and concurrent execution, for all Hermes lanes. It bounds
each result, not the combined observations from a multi-tool turn or the total
context. Earlier history, assistant reasoning, tool IDs, execution and rewards
are unchanged. The pinned Hermes runtime's `file_read_max_chars` setting covers
file reads only; its larger general result ceiling is hardcoded, so Gym applies
this limit at the common tool-dispatch boundary.

Focused CPU controls exercise real Hermes search execution and preserve small
results, previous history and thinking. They do not establish a model pass-rate
gain or qualify a changed source under an earlier end-to-end run. Run the focused
controls with the declared Hermes runtime and selected Gym source:

```bash
python -m unittest responses_api_agents.hermes_agent.tests.test_tool_results -v
```

## Setup

`hermes-agent` is pinned in `requirements.txt` to a fork branch with patches for token id tracking, chat template, and sampling parameters needed for training.

For agent integrations like this, the agent must point at Gym's model server, it must include prompt and generation token id in requests for Nemo RL and other trainer integration on policy token id correction, it must not override sampling parameters like temperature and top p, and it must preserve sampled reasoning in retained history. When compression is enabled, Hermes runs its stock compressor (`agent/context_compressor.py`) unchanged, and its `call_llm(task="compression")` request is answered by the calling run's policy client, model and rollout URL, as a self-hosted deployment without an auxiliary provider resolves it. The request is Hermes' own: one user message, no tools, Hermes' summary `max_tokens` and temperature. An empty summary leaves only the compaction marker and a failed call drops the middle turns; the run continues in both cases.

## Resources server compatibility

Works with any resources server based verifier, but does not work for resources server tools or other endpoints out of the box. Hermes Agent ships its own toolset (terminal, file, code_execution, web, etc.), so it does not rely on tools defined in the dataset. It may work with Gymnasium style resources servers, though. In testing, only the resources server's task data and `verify` are used. This means existing benchmarks (math, code, reasoning_gym, mcqa, instruction_following, ...) can be used as-is by adding a `<server>_hermes_agent` config.

## Configuration example

```yaml
hermes_agent:
  responses_api_agents:
    hermes_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_verifier
      model_server:
        type: responses_api_models
        name: policy_model
      model: served-model-name
      enabled_toolsets: [terminal, file, code_execution]
      max_turns: 30
      concurrency: 32
      temperature: 1.0
      system_prompt: |
        your system prompt here.
```

| field | default | description |
|-------|---------|-------------|
| `enabled_toolsets` | `null` (all) | forwarded to `AIAgent(enabled_toolsets=...)` |
| `disabled_toolsets` | `null` | forwarded to `AIAgent(disabled_toolsets=...)` |
| `model` | `null` | served model id; defaults to `model_server.name` for backward compatibility |
| `max_turns` | `30` | maps to `AIAgent.max_iterations` |
| `concurrency` | `32` | max simultaneous `run()` calls |
| `temperature` | `1.0` | sampling temperature passed to `AIAgent` |
| `terminal_backend` | `local` | sets `TERMINAL_ENV` (process-global); `local`, `docker`, `daytona`, `modal`, `ssh` |
| `terminal_timeout` | `60` | sets `TERMINAL_TIMEOUT` (process-global); per-command wall-clock seconds |
| `system_prompt` | `null` | passed as `system_message` to `run_conversation`; falls back to any system item in `body.input` |
| `finish_tool` | `false` | offers OpenHands' `finish(message)` tool; every call in the turn that calls it executes, then the run ends as completed without another model request |

The model-server url is resolved at request time and passed to `AIAgent(base_url=..., api_key="gym")`. <!-- pragma: allowlist secret -->

`context_window` binds the native compressor to the served model window; model-name fallback metadata must not override a configured training window. `compression_threshold` (default `0.50`, Hermes' `compression.threshold` default) is the fraction of that window at which Hermes compresses. `compression_summary_temperature` (default `null`) replaces the temperature of the compression request; `null` keeps Hermes' own `0.3`.

### Summary output budget

`compression_summary_max_output_tokens` defaults to `32768` and projects to
Hermes' `compression.summary_max_output_tokens`. It includes reasoning and visible
summary tokens. Thinking and the visible-summary target remain unchanged;
`compression_target_ratio` only controls recent-tail retention. Choose an output
budget that fits the policy model's context window together with the summary input.
Requires the Hermes runtime's summary-output-budget configuration support.

## Local command output budget

The pinned Hermes source bounds command output while it drains the process.
Above 64 MiB it kills the owned command group and returns exit125 with a bounded
head/tail observation, so runaway output cannot keep growing the agent's buffer.
Finite file-operation reads remain complete within that budget; terminal display
keeps its existing presentation limit. Existing sealed runtime archives need to
be rebuilt and selected explicitly to consume this source change.
