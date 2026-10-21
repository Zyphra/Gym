# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Apptainer provider implementation."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import posixpath
import re
import shlex
import shutil
import signal
import tempfile
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxCreateVerificationError,
    SandboxExecResult,
    SandboxHandle,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.sandbox.providers.utils import CapturedTimeout as _CapturedTimeout
from nemo_gym.sandbox.providers.utils import PipeCapture as _PipeCapture
from nemo_gym.sandbox.providers.utils import capture_facts as _capture_facts
from nemo_gym.sandbox.providers.utils import coerce_config as _coerce_config
from nemo_gym.sandbox.providers.utils import path_under_mount as _path_under_mount
from nemo_gym.sandbox.utils import remove_private_directory
from nemo_gym.setup_timeout import setup_timeout_seconds


if TYPE_CHECKING:
    from core.sandbox.cgroups import CgroupLeaf as _CgroupLeaf
    from core.sandbox.cgroups import SandboxCgroup as _SandboxCgroup


LOGGER = logging.getLogger(__name__)

DEFAULT_MOUNT_POINT = "/sandbox"
INSTANCE_NAME_PREFIX = "nemo-gym-"
READY_PROBE_COMMAND = "printf apptainer-sandbox-ready"
READY_PROBE_EXPECTED = "apptainer-sandbox-ready"
SANDBOX_RUNTIME_RETURN_CODE = 125
# Apptainer reports its own failure to run a command as a final "FATAL:" line and
# exit status 255. A command's output may contain any text, including these words.
APPTAINER_FATAL_EXIT_CODE = 255
APPTAINER_FATAL_PREFIX = "FATAL:"
APPTAINER_MISSING_INSTANCE_MARKERS = ("no instance found", "instance not found", "does not exist")
APPTAINER_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
APPTAINER_ENV_FILE_READONLY = frozenset({"EUID", "GID", "HOME", "IFS", "OPTIND", "PWD", "UID"})
ACTIVE_EPISODE_CGROUP_ENV = "NEMO_GYM_ACTIVE_EPISODE_CGROUP"


class ApptainerCreateError(SandboxCreateError):
    """Raised when Apptainer cannot create a sandbox."""


class ApptainerCreateVerificationError(SandboxCreateVerificationError):
    """Raised when a newly-created sandbox cannot execute a probe command."""


def _require_apptainer(bin_path: str | None = None) -> str:
    """Return the apptainer binary path or hard-error if it is not installed."""
    if bin_path:
        path = Path(bin_path) / "apptainer"
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    else:
        path = shutil.which("apptainer")
        if path is not None:
            return path
    raise RuntimeError(
        "The 'apptainer' binary is required for the apptainer sandbox provider. "
        "Install Apptainer before using env.sandbox.provider.name=apptainer."
    )


def _apptainer_subprocess_env(bin_path: str | None) -> dict[str, str]:
    env = os.environ.copy()
    # Instances are children of the enclosing Slurm/Pyxis job. Asking
    # Apptainer to create a user-systemd scope moves lifecycle ownership away
    # from the job and races scope teardown during instance start. This host
    # setting is controlled by apptainer.conf, not a container env override.
    config_dir = Path(env.get("APPTAINER_CONFIGDIR", tempfile.gettempdir()))
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"nemo-gym-apptainer-{os.getpid()}.conf"
    config_path.write_text("systemd cgroups = no\n", encoding="utf-8")
    env["APPTAINER_CONFIG_FILE"] = str(config_path)
    if bin_path:
        env["PATH"] = f"{bin_path}:{env.get('PATH', '')}"
    return env


@dataclass(frozen=True)
class ApptainerCreateConfig:
    """Settings for creating an Apptainer sandbox instance."""

    mount_point: str = DEFAULT_MOUNT_POINT
    start_timeout_s: float | None = 600
    extra_start_args: list[str] = field(default_factory=list)
    apply_resource_limits: bool = True
    cgroup_root: str | None = None
    disk_backed_overlay: bool = False
    staging_root: str | None = None

    def __post_init__(self) -> None:
        if type(self.disk_backed_overlay) is not bool:
            raise ValueError("create.disk_backed_overlay must be a boolean")
        if self.start_timeout_s is not None and self.start_timeout_s <= 0:
            raise ValueError("create.start_timeout_s must be > 0")
        if not self.mount_point.startswith("/"):
            raise ValueError("create.mount_point must be an absolute path")
        if self.cgroup_root is not None and not Path(self.cgroup_root).is_absolute():
            raise ValueError("create.cgroup_root must be an absolute path")
        if self.cgroup_root is not None and not self.apply_resource_limits:
            raise ValueError("create.apply_resource_limits cannot be false when create.cgroup_root is set")
        if self.staging_root is not None and not Path(self.staging_root).is_absolute():
            raise ValueError("create.staging_root must be an absolute path")


@dataclass(frozen=True)
class ApptainerExecConfig:
    """Settings for running commands inside an Apptainer sandbox."""

    default_timeout_s: float | None = 180
    timeout_grace_s: float = 15
    fakeroot_for_root: bool = True
    default_binds: list[str] = field(default_factory=list)
    extra_exec_args: list[str] = field(default_factory=list)
    concurrency: int = 32
    max_output_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.default_timeout_s is not None and self.default_timeout_s <= 0:
            raise ValueError("exec.default_timeout_s must be > 0")
        if self.timeout_grace_s < 0:
            raise ValueError("exec.timeout_grace_s must be >= 0")
        if self.concurrency < 1:
            raise ValueError("exec.concurrency must be >= 1")
        if self.max_output_bytes is not None and self.max_output_bytes < 1:
            raise ValueError("exec.max_output_bytes must be > 0")


@dataclass(frozen=True)
class ApptainerProbeConfig:
    """Post-create probe settings: a test command confirming the sandbox is usable."""

    command: str | None = READY_PROBE_COMMAND
    expected_stdout: str | None = READY_PROBE_EXPECTED
    timeout_s: int = 30
    deadline_s: float | None = None
    stable_count: int = 1
    stable_delay_s: float = 0.0

    def __post_init__(self) -> None:
        if self.command is not None and self.timeout_s <= 0:
            raise ValueError("probe.timeout_s must be > 0")
        if self.deadline_s is not None and self.deadline_s <= 0:
            raise ValueError("probe.deadline_s must be > 0")
        if self.stable_count < 1:
            raise ValueError("probe.stable_count must be >= 1")
        if self.stable_delay_s < 0:
            raise ValueError("probe.stable_delay_s must be >= 0")


@dataclass
class _ApptainerInstance:
    """Provider-private state stashed on SandboxHandle.raw."""

    name: str  # what the instance is called
    staging_dir: Path  # the shared folder on the host
    mount_point: str  # where the folder shows up inside
    image: str  # what it was built from
    env: dict[str, str] = field(default_factory=dict)
    cgroup: "_SandboxCgroup | None" = None
    cleanup_dir: Path | None = None


def _load_cgroup_module() -> Any:
    """Load task-owned containment only when an explicit root enables it."""
    try:
        from core.sandbox import cgroups
    except ModuleNotFoundError as error:
        raise RuntimeError("create.cgroup_root requires the rl_tasks core.sandbox.cgroups module") from error
    return cgroups


def _project_active_cgroup_root(configured: str | None) -> str | None:
    """Keep provider cgroups inside the executor-owned aggregate episode."""
    active = os.environ.get(ACTIVE_EPISODE_CGROUP_ENV)
    if active is None:
        # Pooled workers inherit their node's delegated root without an active episode.
        return configured if configured is not None else os.environ.get("NEMO_GYM_EPISODE_CGROUP_ROOT")
    active_path = Path(active)
    if not active_path.is_absolute():
        raise ValueError(f"{ACTIVE_EPISODE_CGROUP_ENV} must be an absolute path")
    active_path = active_path.resolve(strict=True)
    if configured is None:
        return str(active_path)

    configured_path = Path(configured).resolve(strict=True)
    try:
        configured_path.relative_to(active_path)
    except ValueError:
        raise ValueError("create.cgroup_root must be the active episode cgroup or one of its descendants") from None
    return str(configured_path)


def _resource_flags(resources: SandboxResources) -> list[str]:
    """Translate neutral resources into apptainer CLI flags."""
    return _resource_limit_flags(resources) + _resource_passthrough_flags(resources)


def _resource_limit_flags(resources: SandboxResources) -> list[str]:
    flags: list[str] = []
    if resources.cpu is not None:
        flags += ["--cpus", str(resources.cpu)]
    if resources.memory_mib is not None:
        flags += ["--memory", f"{resources.memory_mib}m"]
    return flags


def _resource_passthrough_flags(resources: SandboxResources) -> list[str]:
    flags: list[str] = []
    if resources.gpu:
        flags.append("--nv")
    return flags


def _resolve_image(image: str) -> str:
    if "://" in image or image.startswith(("/", ".")) or image.endswith(".sif"):
        return image
    return f"docker://{image}"


def _disk_backing_filesystem(path: Path, mountinfo: str) -> str:
    """Find the actual mount covering the private overlay's parent directory."""
    covering: list[tuple[int, int, str]] = []
    for index, line in enumerate(mountinfo.splitlines()):
        before, separator, after = line.partition(" - ")
        fields, filesystem = before.split(), after.split()
        if not separator or len(fields) < 5 or not filesystem:
            continue
        mount = Path(re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), fields[4]))
        if path.is_relative_to(mount):
            covering.append((len(mount.parts), index, filesystem[0]))
    if not covering:
        raise ApptainerCreateError(f"cannot identify disk backing filesystem for {path}")
    return max(covering)[2]


def _require_disk_backing(path: Path) -> None:
    """Fail closed instead of putting a task's declared disk budget in RAM or shared storage."""
    filesystem = _disk_backing_filesystem(path, Path("/proc/self/mountinfo").read_text())
    # Overlay mounts can themselves have a tmpfs upper directory; they do not
    # prove disk backing. The caller must choose an explicitly mounted local disk.
    if filesystem not in {"ext2", "ext3", "ext4", "xfs", "btrfs", "zfs", "f2fs"}:
        raise ApptainerCreateError(f"disk_backed_overlay requires a local disk filesystem; {path} uses {filesystem}")


def _to_sandbox_status(state: str | None) -> SandboxStatus:
    """Map an apptainer-reported state string to the neutral status enum."""
    normalized = str(state or "").lower()
    if normalized in {"running", "active", "ready"}:
        return SandboxStatus.RUNNING
    if normalized in {"starting", "creating", "pending"}:
        return SandboxStatus.STARTING
    if normalized in {"stopped", "exited", "terminated"}:
        return SandboxStatus.STOPPED
    if normalized in {"error", "failed", "unhealthy"}:
        return SandboxStatus.ERROR
    return SandboxStatus.UNKNOWN


def _is_runtime_failure(code: int, stderr: str) -> bool:
    """Did apptainer itself fail to run the command, rather than the command fail?

    Only Apptainer's own exit (status 255 whose last stderr line is its FATAL
    line) qualifies. Text anywhere else in stderr belongs to the command, and a
    command killed by a signal or exiting nonzero keeps its own status. Apptainer
    fails before the command writes anything, so its few lines are never cut by a
    bounded capture. A command that itself exits 255 after printing a final
    "FATAL:" line is indistinguishable and counts as Apptainer's.
    """
    if code != APPTAINER_FATAL_EXIT_CODE:
        return False
    lines = [line for line in (stderr or "").splitlines() if line.strip()]
    return bool(lines) and lines[-1].startswith(APPTAINER_FATAL_PREFIX)


def _is_missing_instance(stderr: str) -> bool:
    low = stderr.lower()
    return any(marker in low for marker in APPTAINER_MISSING_INSTANCE_MARKERS)


def _open_starter_pidfd(listing: str, name: str) -> int | None:
    """A pidfd for the starter that owns the named instance, or None.

    ``instance list --json`` names the container process; its parent is the
    ``Apptainer instance: ...`` starter that ``instance stop`` waits for.
    """
    try:
        entries = json.loads(listing).get("instances") or []
        pid = next(int(entry["pid"]) for entry in entries if entry.get("instance") == name)
        stat = Path(f"/proc/{pid}/stat").read_text()
        starter = int(stat[stat.rindex(")") + 2 :].split()[1])
        return os.pidfd_open(starter)
    except (AttributeError, IndexError, KeyError, OSError, StopIteration, TypeError, ValueError):
        return None


def _open_container_pidfd(listing: str, name: str) -> int | None:
    """Pin the listed container identity before stopping its PID namespace."""
    try:
        entries = json.loads(listing).get("instances") or []
        pid = next(int(entry["pid"]) for entry in entries if entry.get("instance") == name)
        return os.pidfd_open(pid)
    except (AttributeError, KeyError, OSError, StopIteration, TypeError, ValueError):
        return None


async def _wait_container_exit(pidfd: int, timeout_s: float) -> None:
    """Wait for this container's kernel exit, including PID-namespace reaping."""
    loop = asyncio.get_running_loop()
    exited = loop.create_future()

    def ready() -> None:
        if not exited.done():
            exited.set_result(None)

    loop.add_reader(pidfd, ready)
    try:
        await asyncio.wait_for(exited, timeout_s)
    finally:
        loop.remove_reader(pidfd)


async def _reap_adopted_instance(pidfd: int) -> None:
    """Reap a stopped instance starter that this process adopted as a child subreaper.

    ``apptainer instance start`` daemonizes, so the starter is re-parented to
    the nearest subreaper, e.g. a contained episode worker. ``instance stop``
    waits until the starter PID disappears, and an unreaped zombie keeps it, so
    every stop waited out its full timeout (10 s). A pidfd names exactly this
    process; a starter that is not our child is left to its own parent.
    """
    while True:
        try:
            if os.waitid(os.P_PIDFD, pidfd, os.WEXITED | os.WNOHANG) is not None:
                return
        except ChildProcessError:
            return
        await asyncio.sleep(0.01)


def _coerce_binds(value: Any) -> list[str]:
    """Normalize ``spec.provider_options['binds']`` into a list of bind strings.

    Accepts a single ``"src:dst[:opts]"`` string or a list of them. These are
    extra per-sandbox bind mounts, added on top of the staging mount and the
    provider-level ``exec.default_binds``.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    raise ApptainerCreateError(f"provider_options['binds'] must be a string or list, got {type(value).__name__}")


def _serialize_env_file(env: Mapping[str, str]) -> bytes:
    """Serialize environment variables for Apptainer ``--env-file``.

    Apptainer shell-evaluates env files once while reading them. POSIX
    shell-quoting preserves each value through that evaluation, while the
    accompanying ``--no-eval`` flag prevents a second evaluation during
    container environment injection.
    """
    assignments: list[str] = []
    for key, value in env.items():
        if not isinstance(key, str) or APPTAINER_ENV_NAME.fullmatch(key) is None:
            raise ValueError("Apptainer environment variable names must be POSIX shell identifiers")
        if key in APPTAINER_ENV_FILE_READONLY:
            raise ValueError("Apptainer --env-file cannot set a shell readonly environment variable")
        if not isinstance(value, str):
            raise TypeError("Apptainer environment variable values must be strings")
        if "\x00" in value:
            raise ValueError("Apptainer environment variable values cannot contain NUL")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("Apptainer environment variable values must be UTF-8 encodable") from None
        assignments.append(f"{key}={shlex.quote(value)}")
    return ("\n".join(assignments) + "\n").encode("utf-8") if assignments else b""


@contextlib.contextmanager
def _private_env_file(staging_dir: Path, content: bytes) -> Iterator[Path | None]:
    """Yield a unique mode-0600 env file and remove it on every exit path."""
    if not content:
        yield None
        return

    fd = -1
    path: Path | None = None
    try:
        fd, raw_path = tempfile.mkstemp(prefix=".apptainer-env-", dir=staging_dir)
        path = Path(raw_path)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            fd = -1
            stream.write(content)
        yield path
    finally:
        if fd >= 0:
            os.close(fd)
        if path is not None:
            path.unlink(missing_ok=True)


class ApptainerProvider:
    """Sandbox provider backed by the local Apptainer CLI."""

    name = "apptainer"

    def __init__(
        self,
        *,
        exec: ApptainerExecConfig | Mapping[str, Any] | None = None,
        create: ApptainerCreateConfig | Mapping[str, Any] | None = None,
        probe: ApptainerProbeConfig | Mapping[str, Any] | None = None,
        bin_path: str | None = None,
    ) -> None:
        if not hasattr(os, "pidfd_open") or not hasattr(os, "P_PIDFD"):
            raise RuntimeError("Apptainer sandbox lifecycle requires Python pidfd_open and P_PIDFD")
        self._exec_config = _coerce_config(exec, ApptainerExecConfig)
        create_config = _coerce_config(create, ApptainerCreateConfig)
        projected_cgroup_root = _project_active_cgroup_root(create_config.cgroup_root)
        self._create_config = (
            create_config
            if projected_cgroup_root == create_config.cgroup_root
            else replace(create_config, cgroup_root=projected_cgroup_root)
        )
        self._probe = _coerce_config(probe, ApptainerProbeConfig)
        self._create_config = replace(
            self._create_config,
            start_timeout_s=setup_timeout_seconds(self._create_config.start_timeout_s),
        )
        self._probe = replace(
            self._probe,
            timeout_s=setup_timeout_seconds(self._probe.timeout_s),
            deadline_s=setup_timeout_seconds(self._probe.deadline_s),
        )
        self._binary = _require_apptainer(bin_path)
        self._subprocess_env = _apptainer_subprocess_env(bin_path)
        self._semaphore = asyncio.Semaphore(self._exec_config.concurrency)
        self._cgroup_root = (
            _load_cgroup_module().DelegatedCgroupRoot(self._create_config.cgroup_root)
            if self._create_config.cgroup_root is not None
            else None
        )

    async def _run(
        self,
        argv: list[str],
        *,
        timeout_s: float | None,
        stdin: bytes | None = None,
        daemonize: bool = False,
        cgroup: _CgroupLeaf | None = None,
        capture_facts: dict[str, int | bool] | None = None,
    ) -> tuple[int, str, str]:
        """Run an apptainer CLI command. Returns (return_code, stdout, stderr).

        Enforces timeout via asyncio.wait_for and kills the whole process group
        on timeout so child processes do not linger. Bounds concurrency with a
        shared semaphore. Decodes output with errors="replace".

        Set ``daemonize=True`` for commands that fork a long-lived background
        process (``apptainer instance start``). Such commands hand the started
        instance a copy of the child's stdout/stderr, so reading those pipes to
        EOF (``communicate()``) blocks until the *instance* exits — i.e. the call
        appears to hang until ``timeout_s`` even though the foreground process
        finished in under a second. In that mode we capture output to temp files
        (which the instance may inherit harmlessly) and only wait for the
        foreground process to exit.
        """
        async with self._semaphore:
            if daemonize:
                return await self._run_daemonizing(argv, timeout_s=timeout_s, cgroup=cgroup)

            command = cgroup.attach_command(argv) if cgroup is not None else argv
            proc = await asyncio.create_subprocess_exec(
                *command,
                # Without explicit input the command reads EOF (like a non-interactive `docker exec`),
                # never the worker's own stdin, which an interactive prompt would block on forever.
                stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                env=self._subprocess_env,
            )
            limit = self._exec_config.max_output_bytes if capture_facts is not None else None
            stdout_capture = _PipeCapture(limit) if limit is not None else None
            stderr_capture = _PipeCapture(limit) if limit is not None else None
            stdout_task = asyncio.create_task(stdout_capture.drain(proc.stdout)) if limit is not None else None
            stderr_task = asyncio.create_task(stderr_capture.drain(proc.stderr)) if limit is not None else None
            try:
                if limit is None:
                    stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(input=stdin), timeout=timeout_s)
                else:

                    async def finish_bounded() -> None:
                        if stdin is not None:
                            proc.stdin.write(stdin)
                            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                                await proc.stdin.drain()
                            proc.stdin.close()
                        await proc.wait()
                        await asyncio.shield(asyncio.gather(stdout_task, stderr_task))

                    await asyncio.wait_for(finish_bounded(), timeout=timeout_s)
                    stdout_b, stderr_b = bytes(stdout_capture.retained), bytes(stderr_capture.retained)
                    capture_facts.update(_capture_facts(stdout_capture, stderr_capture))
            except (asyncio.TimeoutError, asyncio.CancelledError) as e:
                # Graceful timeout: SIGTERM the process group first and give it a short grace period
                # to react (e.g. an in-container agent flushing a partial trace on its SIGTERM
                # handler), THEN SIGKILL if it hasn't exited.
                if cgroup is not None:
                    with contextlib.suppress(Exception):
                        await cgroup.terminate(
                            self._exec_config.timeout_grace_s, reap_owned_children=True, waiter_pid=proc.pid
                        )
                else:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    with contextlib.suppress(asyncio.TimeoutError, Exception):
                        await asyncio.wait_for(proc.wait(), timeout=self._exec_config.timeout_grace_s)
                    if proc.returncode is None:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                with contextlib.suppress(Exception):
                    if limit is None:
                        await proc.wait()
                    else:
                        await asyncio.wait_for(proc.wait(), timeout=max(1.0, self._exec_config.timeout_grace_s + 1))
                if limit is not None:
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(asyncio.gather(stdout_task, stderr_task)),
                            timeout=max(1.0, self._exec_config.timeout_grace_s + 1),
                        )
                    except asyncio.TimeoutError:
                        for fd in (1, 2):
                            pipe = proc._transport.get_pipe_transport(fd)
                            if pipe is not None:
                                pipe.close()
                        stdout_task.cancel()
                        stderr_task.cancel()
                        with contextlib.suppress(asyncio.TimeoutError):
                            await asyncio.wait_for(
                                asyncio.gather(stdout_task, stderr_task, return_exceptions=True), timeout=1.0
                            )
                    stdout_b, stderr_b = bytes(stdout_capture.retained), bytes(stderr_capture.retained)
                    facts = _capture_facts(stdout_capture, stderr_capture)
                    capture_facts.update(facts)
                if isinstance(e, asyncio.CancelledError):
                    raise
                if limit is not None:
                    raise _CapturedTimeout(
                        f"apptainer command timed out after {timeout_s:g}s: {argv}",
                        stdout_b.decode(errors="replace"),
                        stderr_b.decode(errors="replace"),
                        facts,
                    ) from e
                raise TimeoutError(f"apptainer command timed out after {timeout_s:g}s: {argv}") from e
            return_code = proc.returncode if proc.returncode is not None else SANDBOX_RUNTIME_RETURN_CODE
            return return_code, stdout_b.decode(errors="replace"), stderr_b.decode(errors="replace")

    async def _run_daemonizing(
        self, argv: list[str], *, timeout_s: float | None, cgroup: _CgroupLeaf | None = None
    ) -> tuple[int, str, str]:
        """Run a command that daemonizes a child (e.g. ``apptainer instance start``).

        Captures stdout/stderr to temp files instead of pipes so the long-lived
        instance inheriting those descriptors cannot wedge the read, then waits
        only for the foreground process to exit.
        """
        with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
            command = cgroup.attach_command(argv) if cgroup is not None else argv
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=out_f,
                stderr=err_f,
                start_new_session=True,
                env=self._subprocess_env,
            )
            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout_s)
            except (asyncio.TimeoutError, asyncio.CancelledError) as e:
                # Graceful timeout: SIGTERM the process group first and give it a short grace period
                # to react (e.g. an in-container agent flushing a partial trace on its SIGTERM
                # handler), THEN SIGKILL if it hasn't exited.
                if cgroup is not None:
                    with contextlib.suppress(Exception):
                        await cgroup.terminate(self._exec_config.timeout_grace_s)
                else:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    with contextlib.suppress(asyncio.TimeoutError, Exception):
                        await asyncio.wait_for(proc.wait(), timeout=self._exec_config.timeout_grace_s)
                    if proc.returncode is None:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                with contextlib.suppress(Exception):
                    await proc.wait()
                if isinstance(e, asyncio.CancelledError):
                    raise
                raise TimeoutError(f"apptainer command timed out after {timeout_s:g}s: {argv}") from e

            out_f.seek(0)
            err_f.seek(0)
            stdout_b = out_f.read()
            stderr_b = err_f.read()

        return_code = proc.returncode if proc.returncode is not None else SANDBOX_RUNTIME_RETURN_CODE
        return return_code, stdout_b.decode(errors="replace"), stderr_b.decode(errors="replace")

    async def _stop_instance(self, name: str, *, sandbox_cgroup: _SandboxCgroup | None = None) -> tuple[int, str, str]:
        """Run ``apptainer instance stop`` and reap the instance if it is our child."""
        pidfd = None
        container_pidfd = None
        container_reaper = None
        if hasattr(os, "pidfd_open"):
            with contextlib.suppress(Exception):
                code, out, _err = await self._run(
                    [self._binary, "instance", "list", "--json", name],
                    timeout_s=self._exec_config.default_timeout_s,
                )
                if code == 0:
                    container_pidfd = _open_container_pidfd(out, name)
                    pidfd = _open_starter_pidfd(out, name)
        reaper = asyncio.create_task(_reap_adopted_instance(pidfd)) if pidfd is not None else None
        try:
            result = await self._run(
                [self._binary, "instance", "stop", name],
                timeout_s=self._exec_config.default_timeout_s,
            )
            if container_pidfd is not None and (result[0] == 0 or _is_missing_instance(result[2])):
                # The stopped starter hands its container to our admitted subreaper.
                # Reap that exact child while cleanup waits for physical PID drain.
                container_reaper = asyncio.create_task(_reap_adopted_instance(container_pidfd))
                if sandbox_cgroup is not None:
                    await sandbox_cgroup.cleanup(self._exec_config.timeout_grace_s, reap_owned_children=True)
                timeout_s = self._exec_config.default_timeout_s or max(1.0, self._exec_config.timeout_grace_s)
                await _wait_container_exit(container_pidfd, timeout_s)
            return result
        finally:
            if container_reaper is not None:
                container_reaper.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await container_reaper
            if container_pidfd is not None:
                os.close(container_pidfd)
            if reaper is not None:
                reaper.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await reaper
                os.close(pidfd)

    async def _cleanup_failed_start(self, name: str, staging_dir: Path, sandbox_cgroup: _SandboxCgroup | None) -> None:
        stopped = False
        try:
            code, _out, err = await self._stop_instance(name, sandbox_cgroup=sandbox_cgroup)
            stopped = code == 0 or _is_missing_instance(err)
        except Exception:
            pass
        if sandbox_cgroup is not None:
            await sandbox_cgroup.cleanup(self._exec_config.timeout_grace_s, reap_owned_children=True)
        if self._create_config.disk_backed_overlay and not stopped:
            LOGGER.error("retaining overlay staging dir %s because instance stop failed", staging_dir)
            return
        shutil.rmtree(staging_dir, ignore_errors=True)

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Start an apptainer instance and return a ready handle.

        Steps:
        1. Warn once if spec.ttl_s is set (unsupported by apptainer).
        2. Resolve the image source (local .sif path or remote docker://, oras://,
           library:// URI) from spec.image. -- just use as is
        3. Make a host staging dir (tempfile.mkdtemp), pick
           mount_point = self._create_config.mount_point, generate a unique
           name = INSTANCE_NAME_PREFIX + uuid4().hex.
        4. Build argv: [binary, "instance", "start", <--bind staging:mount_point>,
           <config default_binds>, <spec.provider_options["binds"]>, <--env-file ...>,
           _resource_flags(spec.resources), <extra_start_args>, image, name].
        5. await self._run(argv, timeout_s=self._create_config.start_timeout_s);
           on non-zero return, clean up the staging dir and raise
           ApptainerCreateError(stderr).
        6. Build the handle:
           SandboxHandle(sandbox_id=name, provider_name=self.name,
               raw=_ApptainerInstance(name, staging_dir, mount_point, image)).
        7. Verify readiness via self._verify_created_handle(handle); on failure
           clean up and raise ApptainerCreateVerificationError.
        8. Return the handle.
        """
        # ttl_s has no apptainer equivalent; warn once, then ignore it.
        if spec.ttl_s is not None:
            LOGGER.warning("ttl_s is not supported by the apptainer provider; it will be ignored.")

        if spec.image is None:
            raise ApptainerCreateError("spec.image is required for the apptainer provider")
        if self._create_config.disk_backed_overlay:
            if type(spec.resources.disk_gib) is not int or spec.resources.disk_gib < 1:
                raise ApptainerCreateError("disk_backed_overlay requires a positive integer resources.disk_gib")
            if any(
                arg in {"--overlay", "--writable"} or arg.startswith("--overlay=")
                for arg in self._create_config.extra_start_args
            ):
                raise ApptainerCreateError("disk_backed_overlay owns --overlay and cannot use --writable")
        if spec.resources.pids is not None and self._cgroup_root is None:
            raise ApptainerCreateError("resources.pids requires create.cgroup_root for enforced containment")
        image = _resolve_image(spec.image)
        try:
            env_file_content = _serialize_env_file(spec.env)
        except (TypeError, ValueError) as e:
            raise ApptainerCreateError(f"invalid Apptainer environment: {e}") from e

        # Extra per-sandbox bind mounts (validated before we allocate anything).
        extra_binds = _coerce_binds(spec.provider_options.get("binds"))

        # host staging dir (bind-mounted in), mount point, unique name.
        mount_point = self._create_config.mount_point
        staging_root = Path(self._create_config.staging_root or tempfile.gettempdir())
        staging_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        staging_root = staging_root.resolve(strict=True)
        if self._create_config.disk_backed_overlay:
            _require_disk_backing(staging_root)
            for bind in [*self._exec_config.default_binds, *extra_binds]:
                source = Path(bind.split(":", 1)[0]).resolve()
                if staging_root.is_relative_to(source):
                    raise ApptainerCreateError("disk-backed overlay storage must not be exposed by a sandbox bind")
        staging_dir = Path(
            tempfile.mkdtemp(prefix="nemo-gym-apptainer-", dir=staging_root)
            if self._create_config.staging_root is not None
            else tempfile.mkdtemp(prefix="nemo-gym-apptainer-")
        )  # create a new empty temp directory on the host and returns that path
        payload_dir = staging_dir
        if self._create_config.disk_backed_overlay:
            # The writable image must never be exposed through the sandbox's
            # staging bind: task commands could otherwise modify its raw blocks.
            payload_dir = staging_dir / "payload"
            payload_dir.mkdir(mode=0o700)
        name = INSTANCE_NAME_PREFIX + uuid.uuid4().hex
        sandbox_cgroup: _SandboxCgroup | None = None
        if self._cgroup_root is not None:
            try:
                sandbox_cgroup = self._cgroup_root.create_sandbox(name, spec.resources)
            except (OSError, RuntimeError, ValueError) as e:
                shutil.rmtree(staging_dir, ignore_errors=True)
                raise ApptainerCreateError(f"could not create delegated sandbox cgroup: {e}") from e

        # build the `apptainer instance start` command line.
        argv: list[str] = [self._binary, "instance", "start"]
        # Apptainer implicitly binds the caller's working directory read-write.
        # That cwd belongs to this host process (e.g. a Ray worker's node
        # scratch root), not to the task, so it is not bound. This closes the
        # implicit cwd bind only; other mounts and binds are configured apart.
        argv += ["--no-mount", "cwd"]
        argv += ["--bind", f"{payload_dir}:{mount_point}"]
        for bind in self._exec_config.default_binds:
            argv += ["--bind", bind]
        for bind in extra_binds:
            argv += ["--bind", bind]
        start_args = list(self._create_config.extra_start_args)
        if self._create_config.disk_backed_overlay:
            start_args = [arg for arg in start_args if arg != "--writable-tmpfs"]
            # Apptainer's implicit host home/tmp binds would expose backing
            # stored there. The payload bind remains the only owned writable
            # directory the task can see.
            start_args += ["--no-mount", "home,tmp"]
            overlay_path = staging_dir / "writable-overlay.img"
            overlay_argv = [
                self._binary,
                "overlay",
                "create",
                "--sparse",
                "--size",
                str(spec.resources.disk_gib * 1024),
            ]
            if "--fakeroot" in start_args:
                overlay_argv.append("--fakeroot")
            overlay_argv.append(str(overlay_path))
            try:
                code, _out, err = await self._run(overlay_argv, timeout_s=self._create_config.start_timeout_s)
                if code != 0:
                    raise ApptainerCreateError(f"could not create disk-backed overlay (code={code}): {err.strip()}")
                if not overlay_path.is_file():
                    raise ApptainerCreateError("overlay create completed without the owned overlay image")
                if overlay_path.stat().st_size != spec.resources.disk_gib * 1024**3:
                    raise ApptainerCreateError("overlay image size differs from the declared disk budget")
            except BaseException:
                await self._cleanup_failed_start(name, staging_dir, sandbox_cgroup)
                raise
            start_args += ["--overlay", str(overlay_path)]
        resource_limit_flags = _resource_limit_flags(spec.resources)
        if resource_limit_flags and self._create_config.apply_resource_limits and sandbox_cgroup is None:
            if "--fakeroot" in start_args:
                LOGGER.warning(
                    "Skipping apptainer CPU/memory resource flags because create.extra_start_args contains --fakeroot."
                )
            else:
                argv += resource_limit_flags
        argv += _resource_passthrough_flags(spec.resources)
        argv += start_args
        if sandbox_cgroup is not None:
            argv += sandbox_cgroup.instance.readonly_bind_args()

        # start the instance; clean up the staging dir on any failure.
        try:
            with _private_env_file(staging_dir, env_file_content) as env_file:
                if env_file is not None:
                    argv += ["--no-eval", "--env-file", str(env_file)]
                argv += [image, name]
                code, _out, err = await self._run(
                    argv,
                    timeout_s=self._create_config.start_timeout_s,
                    daemonize=True,
                    cgroup=sandbox_cgroup.instance if sandbox_cgroup is not None else None,
                )
        except asyncio.CancelledError:
            await self._cleanup_failed_start(name, staging_dir, sandbox_cgroup)
            raise
        except TimeoutError as e:
            await self._cleanup_failed_start(name, staging_dir, sandbox_cgroup)
            raise ApptainerCreateError(f"apptainer instance start timed out for image={image!r}: {e}") from e
        except BaseException:
            await self._cleanup_failed_start(name, staging_dir, sandbox_cgroup)
            raise
        if code != 0:
            await self._cleanup_failed_start(name, staging_dir, sandbox_cgroup)
            raise ApptainerCreateError(
                f"apptainer instance start failed (code={code}) for image={image!r}: {err.strip()}"
            )

        # wrap provider-private state on the handle.
        handle = SandboxHandle(
            sandbox_id=name,
            provider_name=self.name,
            raw=_ApptainerInstance(
                name=name,
                staging_dir=payload_dir,
                mount_point=mount_point,
                image=image,
                env=dict(spec.env),
                cgroup=sandbox_cgroup,
                cleanup_dir=staging_dir if self._create_config.disk_backed_overlay else None,
            ),
        )

        # Verify the sandbox can actually run a command before handing it back.
        # On any failure, tear down the half-created sandbox so we don't leak a
        # running instance / staging dir.
        try:
            await self._verify_created_handle(handle)
        except BaseException:
            await self._cleanup_failed_create_handle(handle)
            raise

        return handle

    async def _verify_created_handle(self, handle: SandboxHandle) -> None:
        """Run the readiness probe until the sandbox responds, or raise.

        - probe.command is None      -> skip (no verification).
        - probe.deadline_s is None   -> single attempt; a failure raises immediately.
        - probe.deadline_s is set    -> poll until the sandbox passes the probe
          `stable_count` consecutive times, or the deadline elapses.
        """
        probe = self._probe
        if probe.command is None:
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + probe.deadline_s if probe.deadline_s is not None else None
        consecutive = 0
        last_detail = "no probe attempt completed"

        while True:
            result = await self.exec(handle, probe.command, timeout_s=probe.timeout_s)
            passed = result.return_code == 0 and (
                probe.expected_stdout is None or probe.expected_stdout in (result.stdout or "")
            )
            if passed:
                consecutive += 1
                if consecutive >= probe.stable_count:
                    return
            else:
                consecutive = 0
                last_detail = f"return_code={result.return_code}, stderr={(result.stderr or '').strip()!r}"
                if deadline is None:
                    raise ApptainerCreateVerificationError(
                        f"sandbox {handle.sandbox_id!r} failed readiness probe: {last_detail}"
                    )

            if deadline is not None and loop.time() >= deadline:
                raise ApptainerCreateVerificationError(
                    f"sandbox {handle.sandbox_id!r} did not pass readiness probe within "
                    f"{probe.deadline_s:g}s: {last_detail}"
                )
            if probe.stable_delay_s > 0:
                await asyncio.sleep(probe.stable_delay_s)

    async def _cleanup_failed_create_handle(self, handle: SandboxHandle) -> None:
        """Best-effort teardown of a sandbox that failed verification."""
        inst = handle.raw
        await self._cleanup_failed_start(inst.name, inst.cleanup_dir or inst.staging_dir, inst.cgroup)

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
        stdin: bytes | None = None,
    ) -> SandboxExecResult:
        """Run a command inside the instance.

        Maps the neutral ``user`` parameter onto apptainer:
        - None            -> run as the default (launching) user.
        - "root" / 0      -> add --fakeroot (root inside the container).
        - other user/uid  -> --fakeroot + wrap in ``su`` to switch to that user.

        ``stdin``, when given, is piped to the command's standard input. This is an
        apptainer-provider extension to the base protocol, useful for feeding large
        inputs (e.g. prompts) that would exceed the kernel's argv length limit.
        """
        inst = handle.raw

        flags: list[str] = []
        if cwd is not None:
            flags += ["--pwd", cwd]
        merged_env = dict(getattr(inst, "env", {}))
        if env:
            merged_env.update(env)
        env_file_content = _serialize_env_file(merged_env)

        effective_command = command
        is_root = user == "root" or user == 0
        if is_root:
            if self._exec_config.fakeroot_for_root:
                flags.append("--fakeroot")
        elif user is not None:
            # Need root inside the container to switch users, then su to the target.
            flags.append("--fakeroot")
            effective_command = f"su -s /bin/sh -c {shlex.quote(command)} {shlex.quote(str(user))}"

        flags += list(self._exec_config.extra_exec_args)

        effective_timeout = timeout_s if timeout_s is not None else self._exec_config.default_timeout_s

        try:
            with _private_env_file(inst.staging_dir, env_file_content) as env_file:
                if env_file is not None:
                    flags += ["--no-eval", "--env-file", str(env_file)]
                exec_cgroup = inst.cgroup.new_exec_leaf() if inst.cgroup is not None else None
                if exec_cgroup is not None:
                    flags += exec_cgroup.readonly_bind_args()
                argv = [self._binary, "exec", *flags, f"instance://{inst.name}", "sh", "-c", effective_command]
                completed = False
                capture_facts: dict[str, int | bool] = {}
                try:
                    code, out, err = await self._run(
                        argv,
                        timeout_s=effective_timeout,
                        stdin=stdin,
                        cgroup=exec_cgroup,
                        **({"capture_facts": capture_facts} if self._exec_config.max_output_bytes is not None else {}),
                    )
                    completed = code == 0
                finally:
                    if exec_cgroup is not None:
                        if completed and exec_cgroup.populated():
                            exec_cgroup.retain_counters()
                        else:
                            await asyncio.shield(
                                inst.cgroup.remove_exec_leaf(exec_cgroup, 0, reap_owned_children=True)
                            )
        except TimeoutError as e:
            facts = e.facts if isinstance(e, _CapturedTimeout) else {}
            return SandboxExecResult(
                stdout=e.stdout if isinstance(e, _CapturedTimeout) else None,
                stderr=e.stderr if isinstance(e, _CapturedTimeout) else str(e),
                return_code=SANDBOX_RUNTIME_RETURN_CODE,
                error_type="timeout",
                **facts,
            )

        if _is_runtime_failure(code, err):
            return SandboxExecResult(
                stdout=out,
                stderr=err,
                return_code=SANDBOX_RUNTIME_RETURN_CODE,
                error_type="sandbox",
                **capture_facts,
            )
        return SandboxExecResult(stdout=out, stderr=err, return_code=code, error_type=None, **capture_facts)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        """Upload one host file into the sandbox.

        Fast path (target under the bind mount): write directly to the host side
        of the shared folder. Fallback (arbitrary path): stage into the shared
        folder, then cp inside the container.
        """
        inst = handle.raw

        rel = _path_under_mount(inst.mount_point, target_path)
        if rel is not None:
            dest = inst.staging_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source_path.read_bytes())
            return

        tmp_name = uuid.uuid4().hex
        host_tmp = inst.staging_dir / tmp_name
        host_tmp.write_bytes(source_path.read_bytes())
        try:
            container_tmp = f"{inst.mount_point.rstrip('/')}/{tmp_name}"
            parent = posixpath.dirname(target_path)
            script = f"mkdir -p {shlex.quote(parent)} && cp {shlex.quote(container_tmp)} {shlex.quote(target_path)}"
            result = await self.exec(handle, script, user="root")
            if result.return_code != 0:
                raise RuntimeError(f"apptainer upload to {target_path!r} failed: {result.stderr}")
        finally:
            host_tmp.unlink(missing_ok=True)

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        """Download one sandbox file to the host.

        Fast path (source under the bind mount): read directly from the host side
        of the shared folder. Fallback (arbitrary path): cp inside the container
        into the shared folder, then read the host side.
        """
        inst = handle.raw
        target_path.parent.mkdir(parents=True, exist_ok=True)

        rel = _path_under_mount(inst.mount_point, source_path)
        if rel is not None:
            target_path.write_bytes((inst.staging_dir / rel).read_bytes())
            return

        tmp_name = uuid.uuid4().hex
        host_tmp = inst.staging_dir / tmp_name
        try:
            container_tmp = f"{inst.mount_point.rstrip('/')}/{tmp_name}"
            script = f"cp {shlex.quote(source_path)} {shlex.quote(container_tmp)}"
            result = await self.exec(handle, script, user="root")
            if result.return_code != 0:
                raise RuntimeError(f"apptainer download from {source_path!r} failed: {result.stderr}")
            target_path.write_bytes(host_tmp.read_bytes())
        finally:
            host_tmp.unlink(missing_ok=True)

    def retained_memory_events(self, handle: SandboxHandle) -> dict[str, int]:
        """The sandbox cgroup's hierarchical memory.events (instance and every command), kept at close."""
        cgroup = handle.raw.cgroup
        raw = cgroup.counters.get("memory.events") if cgroup is not None else None
        return {name: int(value) for name, value in (line.split() for line in raw.splitlines())} if raw else {}

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        """Return the instance's lifecycle status by querying ``apptainer instance list``.
        Runs apptainer instance list --json
        On timeout, non-zero, unparseable JSON ---> UNKNOWN
        Look for the instance name of this sandbox. If it is found --> RUNNING. If it's gone --> STOPPED
        """
        inst = handle.raw

        try:
            # Scope the query to this sandbox's own instance. An unscoped list
            # parses every instance file of the user, so an unrelated, partially
            # written file from another job makes the whole query fail.
            code, out, _err = await self._run(
                [self._binary, "instance", "list", "--json", inst.name],
                timeout_s=self._exec_config.default_timeout_s,
            )
        except TimeoutError:
            return SandboxStatus.UNKNOWN

        if code != 0:
            return SandboxStatus.UNKNOWN

        try:
            instances = json.loads(out).get("instances", [])
        except (json.JSONDecodeError, AttributeError):
            return SandboxStatus.UNKNOWN

        for entry in instances:
            if entry.get("instance") == inst.name:
                return _to_sandbox_status(entry.get("state") or "running")

        # Not listed -> it has been stopped (or never existed anymore).
        return SandboxStatus.STOPPED

    async def _close(self, handle: SandboxHandle) -> None:
        """Stop the instance and clean up the host staging dir.
        Runs apptainer instance stop <name>
        If there is no instance --> SUCCESS
        Removes the host staging dir afterward
        """
        inst = handle.raw

        stop_error: Exception | None = None
        try:
            code, _out, err = await self._stop_instance(inst.name, sandbox_cgroup=inst.cgroup)
            if code != 0 and not _is_missing_instance(err):
                stop_error = RuntimeError(
                    f"apptainer instance stop failed (code={code}) for {inst.name!r}: {err.strip()}"
                )
        except TimeoutError as e:
            stop_error = e

        containment_clean = not self._create_config.disk_backed_overlay or stop_error is None
        if inst.cgroup is not None:
            try:
                await inst.cgroup.cleanup(self._exec_config.timeout_grace_s, reap_owned_children=True)
            except Exception as e:
                containment_clean = False
                if stop_error is None:
                    stop_error = e

        if containment_clean:
            try:
                remove_private_directory(inst.cleanup_dir or inst.staging_dir)
            except OSError as e:
                LOGGER.warning("failed to remove staging dir %s: %s", inst.staging_dir, e)
        else:
            LOGGER.error("retaining staging dir %s because instance or cgroup cleanup failed", inst.staging_dir)

        if stop_error is not None:
            raise stop_error

    async def close(self, handle: SandboxHandle) -> None:
        """Drain teardown even when the caller is cancelled, then propagate cancellation."""
        cleanup_task = asyncio.create_task(self._close(handle))
        cancelled: asyncio.CancelledError | None = None
        cleanup_error: BaseException | None = None
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError as e:
                cancelled = e
                current = asyncio.current_task()
                if current is not None:
                    current.uncancel()
            except BaseException as e:
                if cancelled is None:
                    raise
                cleanup_error = e
                break
        if cancelled is not None:
            if cleanup_error is None and not cleanup_task.cancelled():
                cleanup_error = cleanup_task.exception()
            if cleanup_error is not None:
                LOGGER.error(
                    "sandbox cleanup failed while close caller was cancelled: %s",
                    cleanup_error,
                    exc_info=(type(cleanup_error), cleanup_error, cleanup_error.__traceback__),
                )
            raise cancelled
        await cleanup_task

    async def aclose(self) -> None:
        """No provider-wide resources to close."""
        return None
