# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Harbor's LHTB ``continue_until_timeout`` agent phases, for the same-instance verifier.

Long-Horizon Terminal-Bench (https://github.com/zli12321/LHTB, Apache-2.0) tasks
declare ``[agent].continue_until_timeout = true``. Stock Harbor ignores the flag;
LHTB's Harbor (``harbor/patches/single_step.py.harbor-0.20.0``) runs the agent in
phases instead: each phase is a fresh agent run of at most one checkpoint interval,
followed by the hidden verifier with the agent's process tree frozen. A full pass
ends the trial; anything less re-invokes the agent with the original instruction
and a binary rejection that names no verifier detail, until the budget is spent.
The text and the freeze command below follow that module.
"""

from __future__ import annotations

from pathlib import Path


try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

# Reward at which the verifier's pass ends a trial (``_verifier_passed``) and the
# LHTB report's "solved" threshold.
PASS_REWARD = 1.0
SOLVED_REWARD = 0.95


def task_continues_until_timeout(task_dir: Path) -> bool:
    """Whether the task's ``task.toml`` declares ``[agent].continue_until_timeout = true``."""
    path = task_dir / "task.toml"
    if not path.is_file():
        return False
    return (tomllib.loads(path.read_text(encoding="utf-8")).get("agent") or {}).get("continue_until_timeout") is True


def continuation_instruction(base_instruction: str, *, phase: int, remaining_s: int) -> str:
    """The original instruction followed by the binary rejection (verifier detail is never shown)."""
    rejection = (
        "Your submitted solution did not pass verification. No verifier details, "
        "test output, scalar reward, or hidden requirements are available. "
        "Continue working from the current project state. Re-read the original "
        "requirements, inspect your implementation, and run your own focused "
        f"checks before submitting again. This is continuation {phase}; "
        f"approximately {max(0, int(remaining_s))} seconds remain."
    )
    return f"{base_instruction.rstrip()}\n\n---\n{rejection}"


# Freeze or thaw the descendants of the agent's tmux servers around a shared-instance
# verifier pass. It walks /proc rather than using ps, pgrep or awk, none of which the
# slim task images guarantee; sed is assumed. With no tmux server it signals nothing.
AGENT_TREE_SIGNAL = r"""
roots=""
for d in /proc/[0-9]*; do
  [ -r "$d/status" ] || continue
  name=$(sed -n 's/^Name:[[:space:]]*//p' "$d/status" 2>/dev/null)
  case "$name" in tmux*) roots="$roots ${d#/proc/}";; esac
done
[ -n "$roots" ] || exit 0

marked="$roots"
changed=1
while [ "$changed" -eq 1 ]; do
  changed=0
  for d in /proc/[0-9]*; do
    pid=${d#/proc/}
    case " $marked " in *" $pid "*) continue;; esac
    ppid=$(sed -n 's/^PPid:[[:space:]]*//p' "$d/status" 2>/dev/null)
    [ -n "$ppid" ] || continue
    case " $marked " in
      *" $ppid "*) marked="$marked $pid"; changed=1;;
    esac
  done
done

for pid in $marked; do
  case " $roots " in *" $pid "*) continue;; esac
  kill -__SIGNAL__ "$pid" 2>/dev/null || true
done
exit 0
"""


def signal_command(signal: str) -> str:
    if signal not in {"STOP", "CONT"}:
        raise ValueError(f"unsupported signal: {signal!r}")
    return AGENT_TREE_SIGNAL.replace("__SIGNAL__", signal)
