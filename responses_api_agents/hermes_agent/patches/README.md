# Hermes source fixes

`scripts/hermes_agent_deps.sh` applies these to the pinned hermes-agent commit, in name order,
with `patch -p1 --fuzz=0`. Each fixes a defect in Hermes' own tool handling that the policy
cannot work around; none changes a tool schema or description.

| Patch | File | Fix |
|---|---|---|
| `0001-anyof-coercion.patch` | `tools/arg_coercion.py` | Arguments for a union schema (`anyOf`/`oneOf`, e.g. `terminal.notify`: boolean or array of strings) are coerced against the union's branch types instead of being left uncoerced. |
| `0002-guard-notify-text.patch` | `tools/terminal_tool_guards.py` | The background-command guard names the terminal tool's real argument `notify`, not the nonexistent `notify_on_complete`. |
| `0003-deferred-tool-name.patch` | `agent/conversation_loop.py` | A call to a deferred tool by its own name is told to load it with `tool_describe` and call it through `tool_call`, instead of "does not exist". |

When the pin moves, drop a patch that upstream has absorbed; one that no longer applies fails
the build.
