# anyterminal_agent

Runs any Gym agent inside a Terminal Bench task container and evaluates the result
by running the task's `tests/test.sh` in the same container. Works with
`hermes_agent`, `claude_code_agent`, `terminus_2_agent`, or any other compatible Gym agent.

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

A harness failure is masked only when its cause is outside the policy: an unreachable model
endpoint, a failed harness setup, a sandbox error, or an OOM kill while the sandbox's own memory limit
never fired (an ancestor cgroup or the host).
Such a row carries `error_kind: "harness"` (or `sandbox_failed`) and `mask_sample=true`. A harness
that exits nonzero with no such evidence keeps scoring, because the policy's own output (an empty
compaction summary) or its own commands can end the process; the row names it with
`agent_exit_nonzero`, `agent_return_code` and `agent_exception` (the traceback's class name).

Every failed row also carries `agent_error`, `agent_error_text` (the harness's message, redacted and
cut to 300 characters) and `agent_stop_reason`. A harness the episode endpoint refused after it had
ended the episode (a guard, degeneration or turn stop) reports that HTTP 400 as an API failure of its
own; its `agent_stop_reason` is `episode_terminated`, and it stays a scored outcome of the policy.

A harness the sandbox's own memory limit killed (exit 137 or -9 beside `sandbox_oom_kill` while the
sandbox `oom` or `max` counter moved) scores 0 and is typed `error_kind: "agent_memory_limit"`, not masked:
limits are meant to be generous enough for normal solutions and to stop inefficient or broken ones. When the
counters do not say whose limit fired, it also scores and the detail says so; only a kill with both counters
present and zero is masked as a sandbox failure.

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
| `apptainer_memory_limit_mb` | `32768` | Per-container memory cap via `ulimit -v` |
| `concurrency` | `256` | Max concurrent tasks dispatched to Ray |
