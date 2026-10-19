# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in containment checks through real Apptainer instances and commands."""

import asyncio
import contextlib
import json
import os
import shlex
import time
from pathlib import Path

import pytest

from nemo_gym import PARENT_DIR
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox.providers.apptainer.provider import ApptainerProvider
from nemo_gym.sandbox.providers.base import SandboxHandle, SandboxSpec
from responses_api_agents.anyterminal_agent.app import AnyTerminalInstanceConfig, RunTerminalAgent


pytestmark = [pytest.mark.integration, pytest.mark.sandbox, pytest.mark.skipduringci]
_ROOT_ENV = "NEMO_GYM_DELEGATED_CGROUP_TEST_ROOT"
_IMAGE_ENV = "NEMO_GYM_APPTAINER_TEST_IMAGE"


def _counter(raw: str, name: str) -> int:
    return int(dict(line.split() for line in raw.splitlines())[name])


def _counters(raw: str) -> dict[str, int]:
    return {name: int(value) for name, value in (line.split() for line in raw.splitlines())}


async def _wait_for_exec_leaf(handle: SandboxHandle, timeout_s: float = 5) -> Path:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        leaves = list(handle.raw.cgroup.path.glob("exec-*"))
        if len(leaves) == 1 and (leaves[0] / "cgroup.procs").read_text().split():
            return leaves[0]
        await asyncio.sleep(0.01)
    raise AssertionError("real Apptainer exec did not become populated in its dedicated cgroup leaf")


async def test_real_apptainer_aggregate_containment_and_cleanup() -> None:
    root = os.environ.get(_ROOT_ENV)
    image = os.environ.get(_IMAGE_ENV)
    if not root or not image:
        pytest.skip(f"set both {_ROOT_ENV} and {_IMAGE_ENV} for the real delegated-provider test")

    provider = ApptainerProvider(
        create={"cgroup_root": root, "extra_start_args": ["--fakeroot"]},
        exec={"timeout_grace_s": 0.1, "concurrency": 2},
    )
    primary: SandboxHandle | None = None
    sibling: SandboxHandle | None = None
    primary_path: Path | None = None
    sibling_path: Path | None = None
    delegated_root = Path(root)
    receipt: dict[str, object] = {"image": image, "root": root}
    primary_closed = False
    sibling_closed = False
    try:
        primary = await provider.create(SandboxSpec(image=image, resources={"cpu": 1, "memory_mib": 128, "pids": 48}))
        sibling = await provider.create(SandboxSpec(image=image, resources={"cpu": 1, "memory_mib": 128, "pids": 48}))
        primary_path = primary.raw.cgroup.path
        sibling_path = sibling.raw.cgroup.path

        assert (primary_path / "cpu.max").read_text().strip() == "100000 100000"
        assert (primary_path / "memory.max").read_text().strip() == str(128 * 1024 * 1024)
        assert (primary_path / "memory.swap.max").read_text().strip() == "0"
        assert (primary_path / "memory.oom.group").read_text().strip() == "0"
        assert (primary_path / "pids.max").read_text().strip() == "48"
        assert (primary_path / "instance" / "cgroup.procs").read_text().split()
        receipt["configured"] = {
            name: (primary_path / name).read_text().strip()
            for name in ("cpu.max", "memory.max", "memory.swap.max", "memory.oom.group", "pids.max")
        }

        provider_pid = os.getpid()
        provider_root = f"/proc/{provider_pid}/root"
        sibling_via_provider = f"{provider_root}{sibling_path}"
        manager_via_provider = f"{provider_root}{root}/init"
        provider_fds = f"/proc/{provider_pid}/fd"
        escape_command = (
            primary,
            "set -eu; test ! -w /sys/fs/cgroup/cgroup.procs; "
            "! printf %s $$ > /sys/fs/cgroup/cgroup.procs 2>/dev/null; "
            'test -z "$(find /sys/fs/cgroup -mindepth 1 -maxdepth 1 -type d -print -quit)"; '
            'for fd in /proc/self/fd/*; do target=$(readlink "$fd" 2>/dev/null || true); '
            "case $target in *cgroup.procs*) exit 1;; esac; done; "
            f"test ! -w {shlex.quote(manager_via_provider + '/cgroup.procs')}; "
            f"test ! -w {shlex.quote(sibling_via_provider + '/cpu.max')}; "
            f"! printf %s '100000 100000' > {shlex.quote(sibling_via_provider + '/cpu.max')} 2>/dev/null; "
            f"for fd in {shlex.quote(provider_fds)}/*; do "
            'target=$(readlink "$fd" 2>/dev/null || true); '
            "case $target in *cgroup.procs*) exit 1;; esac; done",
        )
        escape = await provider.exec(*escape_command)
        assert escape.return_code == 0, escape.stderr
        receipt["escape"] = {"provider_pid": provider_pid, "return_code": escape.return_code}

        detached = await provider.exec(
            sibling,
            "sh -c 'nohup sleep 60 </dev/null >/dev/null 2>&1 &'",
            timeout_s=10,
        )
        assert detached.return_code == 0, detached.stderr
        service_leaves = [
            path for path in sibling.raw.cgroup.path.glob("exec-*") if (path / "cgroup.procs").read_text().split()
        ]
        assert len(service_leaves) == 1
        service_pids = (service_leaves[0] / "cgroup.procs").read_text().split()
        service_probe = await provider.exec(sibling, "true")
        assert service_probe.return_code == 0, service_probe.stderr
        assert (service_leaves[0] / "cgroup.procs").read_text().split() == service_pids
        receipt["detached_service"] = {"leaf": service_leaves[0].name, "pid_count": len(service_pids)}

        placement_task = asyncio.create_task(provider.exec(primary, "sleep 30", timeout_s=60))
        placement_leaf = await _wait_for_exec_leaf(primary)
        placement_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await placement_task
        assert not placement_leaf.exists()

        timed = asyncio.create_task(
            provider.exec(primary, "sh -c 'sleep 300 & wait'", timeout_s=2)
        )
        timeout_leaf = await _wait_for_exec_leaf(primary)
        timeout_result = await timed
        assert timeout_result.error_type == "timeout"
        assert not timeout_leaf.exists()
        assert timeout_leaf not in primary.raw.cgroup.exec_leaves
        assert primary.raw.cgroup.instance.populated()
        assert (await provider.exec(primary, "true")).return_code == 0
        receipt["timeout_subtree"] = {
            "leaf": timeout_leaf.name,
            "removed": True,
            "instance_survived": True,
            "subsequent_exec_passed": True,
        }

        cpu_before = (primary_path / "cpu.stat").read_text()
        cpu = await provider.exec(
            primary,
            "pids=''; for i in 1 2 3 4; do yes >/dev/null & pids=\"$pids $!\"; done; "
            "sleep 2; kill $pids; wait $pids 2>/dev/null || true",
            timeout_s=10,
        )
        assert cpu.return_code == 0, cpu.stderr
        cpu_after = (primary_path / "cpu.stat").read_text()
        assert _counter(cpu_after, "nr_throttled") > _counter(cpu_before, "nr_throttled")
        assert _counter(cpu_after, "usage_usec") - _counter(cpu_before, "usage_usec") < 3_500_000
        receipt["cpu_before"] = _counters(cpu_before)
        receipt["cpu_after"] = _counters(cpu_after)

        pids = await provider.exec(
            primary,
            "python3 - <<'PY'\n"
            "import threading, time\n"
            "threads = []\n"
            "try:\n"
            "    while True:\n"
            "        thread = threading.Thread(target=time.sleep, args=(2,))\n"
            "        thread.start()\n"
            "        threads.append(thread)\n"
            "except RuntimeError:\n"
            "    print(f'limited={len(threads)}')\n"
            "for thread in threads:\n"
            "    thread.join()\n"
            "PY",
            timeout_s=10,
        )
        assert pids.return_code == 0 and "limited=" in (pids.stdout or ""), pids.stderr
        created_threads = int((pids.stdout or "").strip().split("limited=", 1)[1])
        assert 0 < created_threads < 48
        pid_events = [
            _counters(counters["pids.events"])
            for counters in primary.raw.cgroup.exec_counters
            if "pids.events" in counters
        ]
        assert any(events["max"] > 0 for events in pid_events)
        receipt["pids"] = {"created_threads": created_threads, "events": pid_events}

        root_local_before = (delegated_root / "memory.events.local").read_text()
        sandbox_local_before = (primary_path / "memory.events.local").read_text()
        heartbeat = asyncio.create_task(provider.exec(sibling, "sleep 3; printf sibling-alive", timeout_s=10))
        oom = await provider.exec(
            primary,
            "python3 - <<'PY'\n"
            "import multiprocessing, time\n"
            "def allocate():\n"
            "    value = bytearray(64 * 1024 * 1024)\n"
            "    time.sleep(10)\n"
            "children = [multiprocessing.Process(target=allocate) for _ in range(3)]\n"
            "for child in children: child.start()\n"
            "for child in children: child.join()\n"
            "PY",
            timeout_s=15,
        )
        # Only the child that exhausted the sandbox is killed; the command reaps it and exits.
        assert oom.return_code == 0, oom.stderr
        sandbox_events = (primary_path / "memory.events").read_text()
        assert _counter(sandbox_events, "oom_kill") > 0
        sandbox_local_after = (primary_path / "memory.events.local").read_text()
        root_local_after = (delegated_root / "memory.events.local").read_text()
        assert _counter(sandbox_local_after, "oom") > _counter(sandbox_local_before, "oom")
        assert _counter(root_local_after, "oom_kill") == _counter(root_local_before, "oom_kill")
        receipt["oom"] = {
            "sandbox_hierarchical": _counters(sandbox_events),
            "sandbox_local_before": _counters(sandbox_local_before),
            "sandbox_local_after": _counters(sandbox_local_after),
            "root_local_before": _counters(root_local_before),
            "root_local_after": _counters(root_local_after),
        }
        sibling_result = await heartbeat
        assert sibling_result.return_code == 0 and sibling_result.stdout == "sibling-alive"
        sibling_probe = await provider.exec(sibling, "true")
        assert sibling_probe.return_code == 0, sibling_probe.stderr

        cleanup_entered = asyncio.Event()
        original_cleanup = primary.raw.cgroup.cleanup

        async def delayed_cleanup(grace_s: float) -> None:
            cleanup_entered.set()
            await asyncio.sleep(0.2)
            await original_cleanup(grace_s)

        primary.raw.cgroup.cleanup = delayed_cleanup
        primary_close = asyncio.create_task(provider.close(primary))
        await cleanup_entered.wait()
        primary_close.cancel()
        primary_close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await primary_close
        primary_closed = True
        assert not primary_path.exists()

        blockers = [asyncio.create_task(provider.exec(sibling, "sleep 1", timeout_s=5)) for _ in range(2)]
        await asyncio.sleep(0.1)
        sibling_close = asyncio.create_task(provider.close(sibling))
        await asyncio.sleep(0.1)
        sibling_close.cancel()
        sibling_close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sibling_close
        sibling_closed = True
        blocker_results = await asyncio.gather(*blockers)
        assert all(result.return_code == 0 for result in blocker_results)
        assert not sibling_path.exists()
        receipt["cancelled_close"] = {"running_cleanup": True, "queued_stop": True}
    finally:
        try:
            if primary is not None and not primary_closed:
                await provider.close(primary)
        finally:
            try:
                if sibling is not None and not sibling_closed:
                    await provider.close(sibling)
            finally:
                with contextlib.suppress(Exception):
                    await provider.aclose()

    assert primary_path is not None and not primary_path.exists()
    assert sibling_path is not None and not sibling_path.exists()
    assert primary is not None and primary.raw.cgroup.counters
    assert provider.retained_memory_events(primary)["oom_kill"] > 0
    assert sibling is not None and sibling.raw.cgroup.counters
    receipt["cleanup"] = {
        "primary_removed": not primary_path.exists(),
        "sibling_removed": not sibling_path.exists(),
        "primary_counters": primary.raw.cgroup.counters,
        "sibling_counters": sibling.raw.cgroup.counters,
    }
    print(json.dumps(receipt, sort_keys=True))


# test.sh bodies for the stock runner's verifier. The allocation exceeds the 128 MiB sandbox.
_ALLOCATE = "python3 -c 'import time; held = [bytearray(32 << 20) for _ in range(16)]; time.sleep(30)'"
_VERIFIERS = {
    # The reward is on disk before the kernel kills the verifier itself.
    "verifier-killed-after-writing-reward": f"printf 1 > /logs/verifier/reward.txt\nexec {_ALLOCATE}\n",
    "ordinary-failing-test": "printf 0 > /logs/verifier/reward.txt\nexit 1\n",
    # A task process the verifier started is killed; the verifier grades and exits normally.
    "task-subprocess-killed-verifier-finished": (
        f"{_ALLOCATE}\necho task process exit $?\nprintf 0 > /logs/verifier/reward.txt\nexit 0\n"
    ),
}


@pytest.mark.parametrize("case", list(_VERIFIERS))
async def test_real_stock_runner_types_only_a_verifier_the_memory_limit_killed(tmp_path: Path, case: str) -> None:
    root = os.environ.get(_ROOT_ENV)
    image = os.environ.get(_IMAGE_ENV)
    if not root or not image:
        pytest.skip(f"set both {_ROOT_ENV} and {_IMAGE_ENV} for the real delegated-provider test")

    task_dir = tmp_path / "task"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "tests" / "test.sh").write_text(_VERIFIERS[case])
    for name in ("persistent", "verifier", "deps"):
        (tmp_path / name).mkdir()
    cfg = AnyTerminalInstanceConfig(
        host="127.0.0.1",
        port=1,
        entrypoint="app.py",
        name="verifier-status",
        agent_server_module="responses_api_agents.terminus_2_agent.app",
        agent_server_class="Terminus2Agent",
        agent_config_class="Terminus2AgentConfig",
        run_session_id=case,
        base_results_dir=tmp_path,
        model_server_url="",
        nemo_gym_root=Path(PARENT_DIR),
        agent_deps_dir=tmp_path / "deps",
        problem_info={"task_name": case, "task_dir": str(task_dir), "cpus": 1, "memory_mb": 128, "pids": 64},
        body=NeMoGymResponseCreateParamsNonStreaming(input=[], model="verifier-status"),
        persistent_dir=tmp_path / "persistent",
        verifier_dir=tmp_path / "verifier",
        agent_run_id=case,
        metrics_fpath=tmp_path / "metrics.json",
        container=image,
        ray_queue_timestamp=time.time(),
        agent_command_str="true",
        agent_overhead_mb=0,
        tb_agent_timeout=60,
        tb_eval_timeout=60,
        sandbox_provider={
            "apptainer": {
                "create": {"cgroup_root": root, "extra_start_args": ["--fakeroot"]},
                "exec": {"timeout_grace_s": 0.1},
            }
        },
    )
    cfg.metrics_fpath.write_text("{}")

    resolved = await RunTerminalAgent(config=cfg).process_single_datapoint()

    metrics = json.loads(cfg.metrics_fpath.read_text())
    reward = (cfg.verifier_dir / "reward.txt").read_text()
    print(json.dumps({"case": case, "reward": reward, "metrics": metrics}, sort_keys=True))
    assert metrics["agent_timed_out"] is False and metrics["container_timed_out"] is False
    assert metrics["verifier_reward_unreadable"] is False
    assert "verifier_error_type" not in metrics  # the kernel's kill is an ordinary exit status to the runtime
    if case == "verifier-killed-after-writing-reward":
        assert reward == "1" and resolved is True
        assert metrics["verifier_return_code"] in (-9, 137)
        assert metrics["sandbox_oom_kill"] > 0
        assert metrics["sandbox_failed"] is True and metrics["mask_sample"] is True
        assert metrics["error_detail"].startswith("verifier killed by the sandbox memory limit (exit ")
    elif case == "ordinary-failing-test":
        assert reward == "0" and resolved is False
        assert metrics["verifier_return_code"] == 1
        assert metrics["sandbox_oom_kill"] == 0
        assert metrics["sandbox_failed"] is False and metrics["mask_sample"] is False
        assert "error_detail" not in metrics
    else:
        assert reward == "0" and resolved is False
        assert metrics["verifier_return_code"] == 0
        assert metrics["sandbox_oom_kill"] > 0
        assert metrics["sandbox_failed"] is False and metrics["mask_sample"] is False
        assert "error_detail" not in metrics
