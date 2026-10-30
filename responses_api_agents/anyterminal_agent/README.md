# anyterminal_agent

Sandbox startup, execution and cleanup exceptions persist their type and message
in `nemo_gym_metrics.json`'s `error_detail` (at most 2,000 characters), alongside
the existing infrastructure and masking fields. When execution and cleanup both
fail, the initiating error comes first. Existing memory-kill attribution takes
precedence; this diagnostic field does not change reward or masking policy.

Runs any Gym agent inside a Terminal Bench task container and evaluates the result
by running the task's `tests/test.sh` in the same container. Works with
`hermes_agent`, `claude_code_agent`, `terminus_2_agent`, `pi_agent` (`configs/anyterminal_pi.yaml`),
or any other compatible Gym agent.

Unlike `anyswe_agent` (which runs agent and eval in two concurrent containers),
anyterminal runs everything sequentially in one container: agent finishes, then
`test.sh` runs and writes a reward to `/logs/verifier/reward.txt`. The test
directory is mounted read-only so the agent cannot tamper with the tests before
they run.

## Prerequisites

Every task runs inside an [Apptainer](https://apptainer.org/) (formerly Singularity) container,
so Apptainer must be installed on each machine that runs rollouts. It is not bundled with Gym.

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

**1. Prepare the dataset** — downloads tasks via Harbor and writes the input JSONL:

```bash
# Download tasks + build dataset + build SIFs (default)
python responses_api_agents/anyterminal_agent/prepare.py

# Skip SIF builds — Apptainer will pull docker:// images at runtime
python responses_api_agents/anyterminal_agent/prepare.py --no-build-sif

# Build SIFs into a custom directory
python responses_api_agents/anyterminal_agent/prepare.py --sif-dir /shared/sifs

# Smoke test — first 5 tasks only
python responses_api_agents/anyterminal_agent/prepare.py --limit 5 --no-build-sif
```

Requires the `harbor` CLI on PATH. Tasks are downloaded automatically and cached at
`~/.cache/harbor/tasks/terminal-bench/`; subsequent runs skip the download.

**2. Start the environment** with Hermes and a model server:

```bash
gym env start \
  --config responses_api_agents/anyterminal_agent/configs/anyterminal_hermes.yaml \
  --model-type vllm_model
```

If you pre-built SIFs into a custom directory, override `tb_sif_dir`:

```bash
gym env start --config ... \
  ++anyterminal_hermes.responses_api_agents.anyterminal_agent.tb_sif_dir=/shared/sifs
```

**3. Collect rollouts:**

```bash
gym eval run --no-serve \
  --agent anyterminal_hermes \
  --input responses_api_agents/anyterminal_agent/data/terminal_bench.jsonl \
  --output results/anyterminal_rollouts.jsonl
```

Each rollout row contains `reward` (0.0 or 1.0), the full agent trajectory, and
`mask_sample` (set when a timeout made the reward unreliable).

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

Agent dependencies are installed once at startup into a portable Python prefix
mounted read-only inside the task container at `/agent_deps_mount`. To support a
new agent, add `responses_api_agents/<agent_dir>/scripts/<agent_dir>_deps.sh` (see
`responses_api_agents/hermes_agent/scripts/hermes_agent_deps.sh` for the pattern).

## Container images

Each Terminal Bench task specifies a Docker image in its `task.toml`. You can
either:

- **Pull at runtime** (default, `tb_sif_dir: null`): Apptainer pulls
  `docker://<image>` on first use. Requires internet access on compute nodes.
- **Pre-build SIFs** (`prepare.py --sif-dir PATH`): Converts each image to a
  `.sif` file. Faster and works on air-gapped clusters.

## Key config options

| Field | Default | Description |
|---|---|---|
| `tb_tasks_cache_dir` | `~/.cache/harbor/tasks` | Where Harbor stores downloaded task definitions |
| `tb_sif_dir` | `null` | Pre-built SIF directory; `null` = pull docker:// at runtime |
| `tb_agent_timeout` | `1800` | Seconds before the agent is killed |
| `agent_timeout_is_failure` | `false` | Opt in to counting agent timeouts as unmasked reward-zero failures; independent infrastructure remains masked. |
| `tb_eval_timeout` | `300` | Seconds for `test.sh` to complete |
| `verifier_environment` | `{}` | Values for the names a task's `[verifier.env]` references. `test.sh` receives each declared variable with Harbor's `${NAME}` / `${NAME:-default}` templates resolved against these values, never the worker environment; an unsupplied reference without a default fails the row before the agent runs. |
| `continue_until_timeout` | `false` | Honour each task's `[agent].continue_until_timeout = true`, as Long-Horizon Terminal-Bench's Harbor does (stock Harbor ignores it). The agent runs in fresh phases of at most `checkpoint_interval_s` on the same instance. After each phase the hidden `test.sh` runs with the agent's tmux descendants frozen; the staged tests and `/logs/verifier` are removed before the agent resumes. A reward below 1.0 re-invokes the agent with the original instruction plus a binary rejection that names no verifier detail; a pass or the end of the agent budget ends the run, and the final verifier grades it. The budget ending is not an agent timeout. Phase outputs and interim grades are kept as `response-phase-NNNN.json` and `interim_verifier/phase-NNNN/`. Needs a bind-mount sandbox provider; background processes outside tmux are not frozen. |
| `checkpoint_interval_s` | `1800` | Longest agent phase under `continue_until_timeout`. |
| `fractional_reward` | `false` | Report `reward.txt`'s value (0 to 1; anything else is unreadable) as the reward instead of 1.0 for any positive grade, for verifiers that award partial credit. |
| `solved_reward_threshold` | unset | Call a row `resolved` only at or above this verifier reward (LHTB's "solved" is `0.95`) instead of at any positive reward. The reward itself is unchanged. |
| `apptainer_memory_limit_mb` | `32768` | Per-container memory cap via `ulimit -v` |
| `concurrency` | `256` | Max concurrent tasks dispatched to Ray |
