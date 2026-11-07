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

## Setup

Hermes runs unmodified: `scripts/hermes_agent_deps.sh` pins a NousResearch/hermes-agent
commit, places that source tree in the agent runtime and installs Hermes' own `uv.lock`
(Python 3.14), then Gym's dependencies. Hermes refuses wheel builds, so it is not listed in
`requirements.txt`. Where the two lock files pin different versions, Gym's pin wins (the OpenAI
SDK: Gym mirrors its audited version's Responses types).

Gym adds no Hermes behaviour. Training data comes from the policy endpoint, which records the
served prompt and generated token IDs, log probabilities and routes; Hermes' messages are not
read for tokens. Sampling parameters are the endpoint's: Hermes sends no temperature, top-p or
top-k on its turns or on its compression summary, so both are sampled at the run's policy
settings. Hermes is configured only through its own `config.yaml` and constructor:

- `provider: custom` with the run's rollout URL, so Hermes' auxiliary routing sends the
  compression summary to the same per-run endpoint and model;
- `model.context_length` from `context_window`, `model.streaming: false`;
- `compression.threshold` / `target_ratio` from the fields below;
- `auxiliary.title_generation.enabled: false` and no background review, so the compression
  summary is the only auxiliary request that reaches the policy;
- `auxiliary.compression.timeout` from `api_timeout`.

Hermes streams its compression request; the rollout endpoint serves whole completions only and
refuses it, and Hermes repeats the request without streaming. When the endpoint ends an episode
(an empty reply with a stop reason and no sampled tokens), Gym stops the run at Hermes' own
interrupt check, as the `finish` tool does, instead of letting Hermes nudge or retry.

The Gym-side controls run real Hermes against a recording OpenAI-compatible endpoint:

```bash
python -m pytest responses_api_agents/hermes_agent/tests -q
```

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
| `terminal_backend` | `local` | sets `TERMINAL_ENV` (process-global); `local`, `docker`, `daytona`, `modal`, `ssh` |
| `terminal_timeout` | `60` | sets `TERMINAL_TIMEOUT` (process-global); per-command wall-clock seconds |
| `system_prompt` | `null` | passed as `system_message` to `run_conversation`; falls back to any system item in `body.input` |
| `finish_tool` | `false` | offers OpenHands' `finish(message)` tool; every call in the turn that calls it executes, then the run ends as completed without another model request |

The model-server url is resolved at request time and passed to `AIAgent(base_url=..., api_key="gym")`. <!-- pragma: allowlist secret -->

`context_window` binds Hermes' compressor to the served model window. `compression_threshold`
(default `0.50`) is Hermes' `compression.threshold`: Hermes compresses when the provider-reported
prompt reaches that fraction of the window less the turn's output reserve (`max_tokens`), with a
floor of 0.75 for windows under 512K tokens. A threshold of `1.0` therefore compresses when the
next turn could no longer receive its full `max_tokens`. `compression_target_ratio` is Hermes'
`compression.target_ratio`; Hermes' default `lean` tail mode keeps a 10K-25K token recent tail.
