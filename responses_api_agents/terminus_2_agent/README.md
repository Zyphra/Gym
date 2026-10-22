# Terminus-2 Agent

Runs [Harbor's Terminus-2](https://github.com/harbor-framework/harbor/tree/main/src/harbor/agents/terminus_2)
as a NeMo Gym Responses API agent.

Terminus-2 executes terminal commands in a fresh temporary workspace by default. Set
`workspace_root` to run in an existing directory instead.

## AnyTerminal

Prepare the Terminal Bench dataset first:

```bash
python responses_api_agents/anyterminal_agent/prepare.py --no-build-sif
```

```bash
gym eval run \
  --config responses_api_agents/anyterminal_agent/configs/anyterminal_terminus_2.yaml \
  --config nemo_gym/sandbox/providers/docker/configs/docker.yaml \
  --config responses_api_models/vllm_model/configs/vllm_model.yaml \
  --agent anyterminal_terminus_2 \
  --input responses_api_agents/anyterminal_agent/data/terminal_bench.jsonl \
  --output terminus_2_rollouts.jsonl
```

Use `configs/terminus_2_agent.yaml` to connect the standalone agent to another
resources server. The host must provide `bash`, `tmux`, and `script`. AnyTerminal
sets `workspace_root: "."` because each task already runs in an isolated container.

The agent converts the Terminus-2 trajectory to Responses API messages and tool
calls. Sampling parameters are forwarded to the configured model server.


`interleaved_thinking` defaults to `true` in both the typed agent and standalone
configuration. Harbor keeps the exact returned `reasoning_content` in retained
assistant messages and echoes it on the next request, for training and evaluation.
Compaction can discard old messages; retained messages keep their reasoning.
The native summarization subagent's content-only prompt remains unchanged.
