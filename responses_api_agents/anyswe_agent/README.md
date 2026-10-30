# AnySWE

AnySWE is the shared Gym runner for repository-repair tasks. The agent receives
an issue and a repository at the task's base commit, edits the code inside the
task image with its own harness, and submits a patch. The runner extracts the
patch and grades it in a fresh sandbox with private tests; the verdict is the
reward. The harness and the dataset are separate choices: running another
harness over the same instances does not create a new task, and the dataset,
not the harness, selects the grader.

## Datasets and graders

| Dataset (population) | Grader | Role and task binding |
| --- | --- | --- |
| SWE-bench Verified | Default path: pinned `swebench` 4.1.0 `make_test_spec` and `get_eval_report`. | Held-out evaluation. [Verified binding](https://github.com/Zyphra/rl_tasks/blob/wrapper/envs/swe/splits/swe_openhands/README.md); `prepare.py` below builds rows from `princeton-nlp/SWE-bench_Verified`. |
| SWE-bench Multilingual (300-instance locked test split) | The RLTasks binding sets `swe_grader`: the dataset's own `eval_script` behind the patch-apply chain, scored with `swebench` 5.0.2 `get_eval_report`. | Evaluation only. [Binding](https://github.com/Zyphra/rl_tasks/blob/wrapper/envs/swe/splits/swebench_multilingual/README.md). |
| R2E-Gym (dataset name contains `R2E-Gym`) | Built-in: the image's `run_tests.sh` (or the row's `eval_script`) runs under `pytest -rA`; every `FAIL_TO_PASS` and `PASS_TO_PASS` test must report `PASSED`, and the required list must be nonempty. | R2E test directories are removed from the agent sandbox before the agent starts. |
| Scale-SWE (17,202 TRAIN tasks in the pinned source inventory) | `private_grader`: a verifier running the Prime-derived scorer; every expected test (`FAIL_TO_PASS` plus `PASS_TO_PASS`) must pass. | Training. Emitted rows are only the tasks named by an admitted-image authority map. [Binding](https://github.com/Zyphra/rl_tasks/blob/wrapper/envs/swe/splits/scale_swe/README.md). |
| SWE-Gym (2,396 admitted issues; 7,188 rows = three harness copies) | `private_grader`: the pinned `swegym` builder, commands and parser, not `swebench`. Every `FAIL_TO_PASS` test must be `PASSED`; `XFAIL` does not count. | Training. [Binding](https://github.com/Zyphra/rl_tasks/blob/wrapper/envs/swe/splits/swe_gym/README.md). |

Counts describe each source's locked inventory or recipe, not the rows a given
run samples. For each rollout AnySWE picks the first matching grader: a
configured `private_grader`, then an R2E-Gym dataset name, then a configured
`swe_grader`, then the default SWE-bench path.

## How a rollout is scored

1. **Agent.** The harness runs inside the task image. Before it starts, the
   runner snapshots the repository, including changes already present in the
   image. Afterwards it emits `git diff --binary` against that snapshot, with
   untracked files included, so the submitted patch holds only the agent's edits.
2. **Grading.** A nonempty patch goes to a fresh grading sandbox, where the
   dataset's private tests run; they are kept from the agent. The default
   SWE-bench and R2E paths apply the patch with `git apply`, then `--reject`,
   then `patch --fuzz=5`; a private grader applies it in its own command. An
   empty patch is not graded and earns `0`.
3. **Reward.** Resolved patches receive reward `1`; other completed attempts
   receive `0`. An integrity violation reported by the sandbox policy forces
   `resolved=false`.
4. **Masks.** `mask_sample=true` marks outcomes that say nothing about the
   policy: an agent timeout; an evaluation timeout; a sandbox, grader or
   harness error; and a resolved patch whose episode ended on a max-iteration or
   context-window error unless the harness reported that stop as its expected
   budget stop. An unresolved patch after an ordinary max-iteration or
   context-window stop keeps its `0`. Keep masked samples out of the success
   denominator, and do not treat every timeout as the same event.

Each rollout includes the trajectory, patch, reward, grading fields and
`mask_sample`.

Sandbox failures retain their private exception in `sandbox_error.txt` under the
request's results directory. The native response publishes only that file's
`artifact_path` alongside the existing class-level error and mask, so callers
can retain the locator without copying private diagnostics. A locator does not
guarantee later file retention; responses that never reach this writer have none.

A `private_grader` configuration names `target` (`module:function(payload,
patch)` returning the grading command, working directory, environment and
files), the row `metadata_key` holding the private payload, an
`unresolved_return_code` and an optional `timeout_return_code`. Exit `0`
resolves, the unresolved code is an ordinary `0` reward, the timeout code is an
evaluation timeout, and any other result is a grader error and masked. Called
with no patch, the same target supplies the setup command that runs before the
agent starts. A configuration may also set `swe_grader`, a trusted
`module:function(instance, patch)` returning `{"script": ..., "grade":
callable(log) -> bool}`: AnySWE runs the script (which must apply the patch and
run the tests) in the grading sandbox and takes `grade(log)` as the verdict,
instead of the pinned `swebench` path. `test_authority_restore`, a trusted
`module:function(instance, patch)` returning `{files, command}`, runs in the
default SWE-bench grader after the candidate patch applies and before the hidden
tests. All three are unset by default.

## Evaluation boundary

SWE-bench Verified and Multilingual are evaluation sets: do not mix their rows
into TRAIN data. A shared harness name is not a matched protocol. Scores compare
only under the same dataset binding, grader, grading-network policy, prompt and
turn and time budgets. The RLTasks
[official-parity cell](https://github.com/Zyphra/rl_tasks/blob/wrapper/envs/swe/splits/swe_official/README.md)
changes the grading network, output capture, turn cap and visible prompt while
keeping the agent offline; it is not full upstream parity.

## Run

Create `env.yaml` for the policy model:

```yaml
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY
policy_model_name: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

Set the OpenSandbox credentials, the model URL reachable from its sandboxes, and
the task-image format:

```bash
export OPENSANDBOX_API_KEY=...
export OPENSANDBOX_DOMAIN=...
export NEMO_GYM_SANDBOX_MODEL_BASE_URL=...
export ANYSWE_CONTAINER_FORMATTER='registry.example.com/anyswe/swebench:{instance_id}'
```

Prepare the dataset, start the environment, and collect rollouts:

```bash
python3 responses_api_agents/anyswe_agent/prepare.py --limit 5

gym env start \
  --config responses_api_agents/anyswe_agent/configs/anyswe_hermes.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --model-type vllm_model

gym eval run --no-serve \
  --agent anyswe_hermes \
  --input responses_api_agents/anyswe_agent/data/swebench_verified.jsonl \
  --output results/anyswe_rollouts.jsonl \
  --limit 5
```

`prepare.py` writes `data/swebench_verified.jsonl`. Each row must resolve to a
task image through its `image` field or `container_formatter`.

## Agents

The included configurations run Hermes Agent, Claude Code, Cline, Pi, OpenClaw, or OpenCode
with the same sandbox and grading path.

Agents whose configuration declares `working_directory` receive the sandbox's
task working directory unless `agent_kwargs` explicitly supplies one. Pi uses
this directory for its CLI tools while keeping its temporary HOME separate;
standalone Pi retains its temporary-workspace default. The submitted patch still
comes from the task repository through the normal AnySWE extraction path.

Configure another Gym agent with:

```yaml
agent_server_module: responses_api_agents.hermes_agent.app
agent_server_class: HermesAgent
agent_config_class: HermesAgentConfig
agent_kwargs:
  max_turns: 100
  terminal_backend: local
```

For large runs, bake `/agent_deps_mount/bin/python`, NeMo Gym, and the selected
agent into each task image. `agent_runtime_source` controls other delivery modes:

- `baked` uses the runtime in the task image and is the default.
- `auto` builds the portable runtime with `setup_scripts/<agent>_deps.sh` once
  and uploads the resulting archive to each sandbox.
- A local tarball path uploads a prebuilt runtime.
- An HTTP(S) URL downloads a prebuilt runtime inside the sandbox.

Each archive must contain the runtime's `bin/python` at its root.

A directory `agent_runtime_source` (an unpacked runtime on a filesystem every
node mounts) is bind-mounted read-only at `/agent_deps_mount` instead of being
copied; it needs the Apptainer provider. Archives and URLs remain the path for
providers without bind mounts.

`agent_runtime_overlay` maps package paths to directories or selected `.py` paths
to Python files. Modules take precedence on `PYTHONPATH`; shared Gym modules fall
back to the pinned archive. Use it
for first-party agent packages so editing them does not repack the archive;
third-party dependencies stay in the pinned runtime.

Package directories also ship `.ts` source files for agent extensions. Other
assets remain excluded from the overlay.

Private task images can pass registry credentials through
`sandbox_spec.provider_options.image_auth`.

## Concurrency

`concurrency: null` runs requests without an agent-level semaphore; the Pi and
Hermes configs select it. Positive integer values keep a bounded semaphore.
Zero and negative values are rejected rather than creating a permanently
blocked queue. Other configurations retain the default 256 request limit.
Sandbox resources and the model server retain their own admission controls.

## Runtime notes

SWE and terminal episodes pass their stable instance identity as the native
executor's placement preference. Samples of the same instance can share the
node-local image cache. The executor may spill to another eligible node as soon
as the preferred node lacks resources; readiness and node exclusions remain
mandatory. Episode identities, hard resource limits and grading remain unchanged.
