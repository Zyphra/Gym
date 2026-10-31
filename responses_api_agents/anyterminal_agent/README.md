# AnyTerminal

AnyTerminal runs a Gym agent inside a task-specific Linux container and grades the container with the task's private `tests/test.sh`. The agent receives the task instruction and works through shell and filesystem tools; any compatible Gym agent can supply the harness (`hermes_agent`, `claude_code_agent`, `terminus_2_agent`, `pi_agent`, `openclaw_agent`, `nemo_fabric_agent`, each with a config under `configs/`). The task, not the harness, defines what counts as success.

## Grading lifecycles

"Terminal task graded by AnyTerminal" does not name one verifier boundary. Three are in use; name the one a run selected.

| Route | Where `test.sh` runs | Boundary |
| --- | --- | --- |
| Stock `anyterminal_agent` (this directory) | Sequentially, in the same sandbox the agent just used. The host copies the task's `tests/` in after the agent finishes. | State the agent left behind (files outside the workspace, installed packages, running processes) is visible to the verifier. Bind-mount providers (Docker, Apptainer, Enroot) purge the verifier output directories before grading and accept only a `reward.txt` newer than the grading dispatch; this does not isolate the verdict from an agent running as root in the same container. |
| RLTasks fresh verification (`anyterminal_separate_grading`) | In a fresh instance of the same image. The agent instance is stopped, its declared workspace is hardened and handed over, and the tests and a private output directory are created after it stopped. | Only the declared workspace carries over; installed system packages and services do not. The test runner and its children run isolated from the verdict. This is that component's default for tasks that declare no mode. |
| RLTasks live verification (the same component, `verification_mode = "live"`) | In the agent's own instance after a lockdown step, with pristine tests copied in afterward. | Keeps system installs and running services for tasks that need them. Isolation is weaker than a fresh verifier, and a lockdown violation scores 0 as `reward_hack_detected`. A task declares the mode, or a training binding sets a default. |

The RLTasks component is in [`core/providers/anyterminal_grading/app.py`](https://github.com/Zyphra/rl_tasks/blob/wrapper/core/providers/anyterminal_grading/app.py); the shared contract for fresh and live verification is [`core/sandbox/HARDENING.md`](https://github.com/Zyphra/rl_tasks/blob/wrapper/core/sandbox/HARDENING.md). Do not describe every terminal route as "same-container" or "fresh": the RLTasks TMax, TerminalHorizon and Terminal-Lego bindings use the separate-grading component, while its Terminal-Bench 2.1 evaluation binding and the Recursive Task Synthesis binding compose this stock same-sandbox component.

## Task sources

The stock `prepare.py` below builds rows from a Harbor dataset (`terminal-bench@2.0` by default). RLTasks supplies the training sources and their bindings; each entry page states its source, task construction, grader and selected verifier route:

| Source | Entry page |
| --- | --- |
| TMax (locked Prime Tasks tree) | [`envs/tmax/README.md`](https://github.com/Zyphra/rl_tasks/blob/wrapper/envs/tmax/README.md) |
| TMax-15K publisher reconstruction | [`envs/tmax_15k/README.md`](https://github.com/Zyphra/rl_tasks/blob/wrapper/envs/tmax_15k/README.md) |
| TerminalHorizon | [`envs/terminalhorizon/README.md`](https://github.com/Zyphra/rl_tasks/blob/wrapper/envs/terminalhorizon/README.md) |
| Recursive Task Synthesis | [`envs/recursive_task_synthesis/README.md`](https://github.com/Zyphra/rl_tasks/blob/wrapper/envs/recursive_task_synthesis/README.md) |
| Terminal-Bench 2.1 (evaluation) | [`envs/terminal_bench_21/README.md`](https://github.com/Zyphra/rl_tasks/blob/wrapper/envs/terminal_bench_21/README.md) |

Terminal-Bench 2.1 is held-out evaluation, not a training source. Keep its protocol separate from training caps and from the optional training penalties described below.

## Scoring

`tests/test.sh` writes a reward to `/logs/verifier/reward.txt`; the host reads it from the bound verifier directory after the sandbox stops.

- `reward` is `1.0` when the row is `resolved` and `0.0` otherwise. `resolved` means a positive verifier reward, or a reward at or above `solved_reward_threshold` when that is set. With `fractional_reward` the reward is the verifier's own value in `[0, 1]`.
- `mask_sample` is set when the verifier timed out, the sandbox failed (including a memory kill), or the agent timed out (unless `agent_timeout_is_failure` keeps that in the scored population). A completed grading run without a readable reward sets `verifier_reward_unreadable`; a refused integrity check sets `verifier_integrity_refused`.
- Infrastructure and masking fields, the native verdict (`resolved`, `verifier_reward`) and any training penalty are different fields. The RLTasks component can subtract an additive penalty from a scored episode (a wrong-harness final action, or a run the agent declared complete that the verifier failed) and records the native verdict beside it; see the [TMax-15K entry page](https://github.com/Zyphra/rl_tasks/blob/wrapper/envs/tmax_15k/README.md).

Sandbox startup, execution and cleanup exceptions persist their type and message in `nemo_gym_metrics.json`'s `error_detail` (at most 2,000 characters), alongside the infrastructure and masking fields. When execution and cleanup both fail, the initiating error comes first. Memory-kill attribution takes precedence; this diagnostic field does not change reward or masking policy.

## Prerequisites

Tasks run in the sandbox selected by `sandbox_provider`: Docker by default, or Apptainer, native Enroot or another configured provider. Apptainer is required for private task networking and where a config selects it; it is not bundled with Gym.

```bash
apt-get update && apt-get install -y wget
cd /tmp
wget https://github.com/apptainer/apptainer/releases/download/v1.4.2/apptainer_1.4.2_amd64.deb
apt-get install -y ./apptainer_1.4.2_amd64.deb
apptainer --version
```

Tasks can opt into private outbound networking through an artifact-admitted
`task_network` metadata JSON descriptor with `tools_root`, `tools_sha256` and
`mode: private`. The registered `apptainer_services` provider uses the original
sandbox resource limits and the profile PID limit, with a separate network
namespace per instance and host loopback disabled. Its immutable native tools
manifest must match the admitted task artifact. The verifier inherits that same
instance network; tasks without this descriptor keep their existing provider.

## Quickstart

**1. Prepare the dataset.** `prepare.py` downloads tasks through the Harbor CLI and writes the input JSONL:

```bash
# Download the tasks and write data/terminal_bench.jsonl (no image build)
python responses_api_agents/anyterminal_agent/prepare.py

# Smoke test: first 5 tasks only
python responses_api_agents/anyterminal_agent/prepare.py --limit 5

# A different Harbor dataset reference
python responses_api_agents/anyterminal_agent/prepare.py \
  --dataset-name terminal-bench/terminal-bench-2-1

# Also build one Apptainer SIF per task into a chosen directory
python responses_api_agents/anyterminal_agent/prepare.py --build-image --image-dir /path/to/sifs
```

Requires the `harbor` CLI on `PATH` (and `apptainer` for `--build-image`). Tasks are cached under `--tasks-cache` (default `~/.cache/harbor/tasks`) and an existing download is skipped.

**2. Start the environment** with Hermes and a model server:

```bash
gym env start \
  --config responses_api_agents/anyterminal_agent/configs/anyterminal_hermes.yaml \
  --model-type vllm_model
```

To use prebuilt SIFs, override `container_formatter` (see Container images).

**3. Collect rollouts:**

```bash
gym eval run --no-serve \
  --agent anyterminal_hermes \
  --input responses_api_agents/anyterminal_agent/data/terminal_bench.jsonl \
  --output results/anyterminal_rollouts.jsonl
```

Each rollout row contains `reward` (0.0 or 1.0 unless `fractional_reward` is set), the full agent trajectory, and `mask_sample`.

## Agent wiring

Swap the agent by changing three fields in the YAML (or overriding on the CLI):


```yaml
agent_server_module: responses_api_agents.hermes_agent.app
agent_server_class: HermesAgent
agent_config_class: HermesAgentConfig
agent_kwargs:
  max_turns: 30
  terminal_backend: local
```

Agents whose configuration declares `working_directory` (Pi) run their tools in
the task's working directory unless `agent_kwargs` supplies one, as under
`anyswe_agent`.

For a rollout-scoped request the harness runs with the rollout id (`NGTB_ROLLOUT_ID`), returns its agent
observations (context compactions, tool timings) in `response.json`, and `/run` returns them as
`ng_agent_observations`, as `anyswe_agent` does. A harness that reports none (such as an agent without
observability) leaves the field out.

Agent dependencies are installed once at startup into a portable Python prefix
mounted read-only inside the task container at `/agent_deps_mount`. To support a
new agent, add `responses_api_agents/<agent_dir>/scripts/<agent_dir>_deps.sh` (see
`responses_api_agents/hermes_agent/scripts/hermes_agent_deps.sh` for the pattern).

## Container images

Each task row carries the task's Docker image (from `task.toml`); `container_formatter` turns it into the reference the sandbox starts. The template may use `{docker_image}`, `{task_name}` and `{runtime_image}`:

- **Pull at runtime** (default, `docker://{docker_image}`): the provider pulls the image on first use. Requires registry access from the worker.
- **Prebuilt SIFs** (`prepare.py --build-image --image-dir PATH`): set `container_formatter` to `PATH/{task_name}.sif`. A template that ends in `.sif` or starts with `/` or `.` is used as a path. Works on air-gapped clusters.

## Key config options

| Field | Default | Description |
|---|---|---|
| `sandbox_provider` | `{docker: {}}` | Sandbox backend: `docker`, `apptainer`, `enroot` or `native_enroot`, or another configured provider. The first four use bind mounts for the runtime, trajectory and verifier directories. |
| `container_formatter` | `docker://{docker_image}` | Image reference template; see Container images. |
| `tb_agent_timeout` | `1800` | Seconds before the agent is killed, used when a task declares no `agent.timeout_sec`. A task's own value overrides it; `global_agent_timeout` overrides both. |
| `tb_eval_timeout` | `300` | Seconds for `test.sh` to complete, used when a task declares no `verifier.timeout_sec`. |
| `agent_timeout_is_failure` | `false` | Opt in to counting agent timeouts as unmasked reward-zero failures; independent infrastructure remains masked. |
| `agent_overhead_mb` | `2048` | Memory added to the task's `memory_mb` for the in-container agent harness. |
| `verifier_environment` | `{}` | Values for the names a task's `[verifier.env]` references. `test.sh` receives each declared variable with Harbor's `${NAME}` / `${NAME:-default}` templates resolved against these values, never the worker environment; an unsupplied reference without a default fails the row before the agent runs. |
| `continue_until_timeout` | `false` | Honour each task's `[agent].continue_until_timeout = true`, as Long-Horizon Terminal-Bench's Harbor does (stock Harbor ignores it). The agent runs in fresh phases of at most `checkpoint_interval_s` on the same instance. After each phase the hidden `test.sh` runs with the agent's tmux descendants frozen; the staged tests and `/logs/verifier` are removed before the agent resumes. A reward below 1.0 re-invokes the agent with the original instruction plus a binary rejection that names no verifier detail; a pass or the end of the agent budget ends the run, and the final verifier grades it. The budget ending is not an agent timeout. Phase outputs and interim grades are kept as `response-phase-NNNN.json` and `interim_verifier/phase-NNNN/`. Needs a bind-mount sandbox provider; background processes outside tmux are not frozen. |
| `checkpoint_interval_s` | `1800` | Longest agent phase under `continue_until_timeout`. |
| `fractional_reward` | `false` | Report `reward.txt`'s value (0 to 1; anything else is unreadable) as the reward instead of 1.0 for any positive grade, for verifiers that award partial credit. |
| `solved_reward_threshold` | unset | Call a row `resolved` only at or above this verifier reward (LHTB's "solved" is `0.95`) instead of at any positive reward. The reward itself is unchanged. |
| `concurrency` | `256` | Max concurrent tasks dispatched to Ray; the shipped configs set `6` (`1` for `nemo_fabric`). |
