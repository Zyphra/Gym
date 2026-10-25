# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import asyncio
import base64
import hashlib
import importlib
import json
import shlex
import shutil
import tarfile
import tempfile
import time
import uuid
from asyncio import Semaphore
from collections.abc import Callable
from contextlib import AsyncExitStack, nullcontext
from pathlib import Path
from subprocess import Popen
from traceback import format_exc
from typing import Any, Dict, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.policy_completion import notify_policy_phase_done
from nemo_gym.rollout_observability import AgentObservationBundle, harness_response_error
from nemo_gym.sandbox import AsyncSandbox, SandboxCreateError, SandboxExecResult, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.episode import (
    SandboxMemoryEvents,
    harness_memory_kill,
    run_sandbox_episode,
    sandbox_memory_events,
)
from nemo_gym.setup_timeout import setup_timeout_seconds


AGENT_OVERLAY_DIR = "/trajectories_mount/agent_overlay"
_INTERNAL_OBSERVATIONS_KEY = "_ng_agent_observations"


def _overlay_files(overlay: Dict[str, str]) -> Dict[str, str]:
    """Read selected Python modules or packages over the pinned runtime's copy."""
    files: Dict[str, str] = {}
    for package, source in overlay.items():
        if not source:
            continue
        target = Path(package)
        parts = target.parts[:-1] if target.suffix == ".py" else target.parts
        if target.is_absolute() or not target.parts or any(not part.isidentifier() for part in parts):
            raise ValueError(f"agent runtime overlay requires a Python import path: {package}")
        if target.suffix == ".py" and not target.stem.isidentifier():
            raise ValueError(f"agent runtime overlay requires a Python import path: {package}")
        root = Path(source).expanduser()
        if root.is_file():
            if root.suffix != ".py" or target.suffix != ".py":
                raise ValueError("agent runtime overlay files must select a Python module")
            files[f"{AGENT_OVERLAY_DIR}/{target}"] = root.read_text()
            continue
        if not root.is_dir():
            raise ValueError(f"agent runtime overlay directory not found: {root}")
        if target.suffix == ".py":
            raise ValueError("agent runtime overlay module must select a Python file")
        modules = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix in (".py", ".ts") and "__pycache__" not in path.parts
        )
        if not modules:
            raise ValueError(f"agent runtime overlay holds no modules: {root}")
        for module in modules:
            files[f"{AGENT_OVERLAY_DIR}/{package}/{module.relative_to(root)}"] = module.read_text()
    return files


def _overlay_sha256(files: Dict[str, str]) -> Optional[str]:
    """Content identity of the shipped overlay, recorded with every instance."""
    if not files:
        return None
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(files[path].encode())
        digest.update(b"\0")
    return digest.hexdigest()


class SWEBenchMetrics(SandboxMemoryEvents):
    resolved: Optional[bool] = None
    patch_exists: Optional[bool] = None
    model_patch: Optional[str] = None

    agent_timed_out: Optional[bool] = None
    agent_error_kind: Optional[str] = None
    agent_stop_reason: Optional[str] = None
    eval_timed_out: Optional[bool] = None
    error_kind: Optional[str] = None
    error_detail: Optional[str] = None
    artifact_path: Optional[str] = None
    mask_sample: Optional[bool] = None

    agent_run_time: Optional[float] = None
    submitted_at: Optional[float] = None
    task_started_at: Optional[float] = None
    sandbox_ready_at: Optional[float] = None
    grading_started_at: Optional[float] = None
    grading_completed_at: Optional[float] = None


class AnySweRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class PrivateGraderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str = Field(min_length=3)
    metadata_key: str = Field(min_length=1)
    unresolved_return_code: int = Field(ge=1, le=255)
    timeout_return_code: Optional[int] = Field(default=None, ge=1, le=255)

    @field_validator("target")
    @classmethod
    def validate_target(cls, value: str) -> str:
        module_name, separator, function_name = value.partition(":")
        if not separator or not module_name or not function_name or ":" in function_name:
            raise ValueError("private grader target must be 'module:function'")
        return value

    @model_validator(mode="after")
    def validate_return_codes(self) -> "PrivateGraderConfig":
        if self.timeout_return_code == self.unresolved_return_code:
            raise ValueError("private grader timeout and unresolved return codes must differ")
        return self

    def load(self) -> Callable[[Dict[str, Any], Optional[str]], Dict[str, Any]]:
        module_name, function_name = self.target.split(":", 1)
        planner = getattr(importlib.import_module(module_name), function_name)
        if not callable(planner):
            raise TypeError(f"private grader target is not callable: {self.target}")
        return planner


class PrivateGraderPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = Field(min_length=1)
    workdir: str = Field(min_length=1)
    env: Dict[str, str] = Field(default_factory=dict)
    files: Dict[str, str] = Field(default_factory=dict)

    @field_validator("workdir")
    @classmethod
    def validate_workdir(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("private grader workdir must be absolute")
        return value

    @field_validator("files")
    @classmethod
    def validate_file_paths(cls, value: Dict[str, str]) -> Dict[str, str]:
        if any(not Path(path).is_absolute() for path in value):
            raise ValueError("private grader file paths must be absolute")
        return value


_MAX_ERROR_DETAIL_CHARS = 500


def _error_detail(*parts: Any) -> Optional[str]:
    """Bounded, single-line cause carried in the metrics so it survives the run."""
    text = " ".join(str(part) for part in parts if part not in (None, "")).replace("\n", " ").strip()
    return text[:_MAX_ERROR_DETAIL_CHARS] or None


def _grading_exception_detail(params: "AnySweInstanceConfig", exc: BaseException) -> Optional[str]:
    """Keep private grading diagnostics in receipts and project only the exception class."""
    if params.private_grader is not None:
        return _error_detail(f"grader exception type={type(exc).__name__}")
    return _error_detail(repr(exc))


def update_metrics(metrics_fpath: Path, update_dict: Dict[str, Any]) -> None:
    existing = {k: v for k, v in json.loads(metrics_fpath.read_text()).items() if v is not None}
    update = {k: v for k, v in update_dict.items() if v is not None}
    metrics_fpath.write_text(json.dumps(existing | update))


def _write_grader_receipt(persistent_dir: Path, result: SandboxExecResult, *, label: str = "grader") -> None:
    stdout_path = f"{label}.stdout"
    stderr_path = f"{label}.stderr"
    (persistent_dir / stdout_path).write_text(result.stdout or "")
    (persistent_dir / stderr_path).write_text(result.stderr or "")
    (persistent_dir / f"{label}_result.json").write_text(
        json.dumps(
            {
                "error_type": result.error_type,
                "return_code": result.return_code,
                "stderr_path": stderr_path,
                "stdout_path": stdout_path,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _safe_config_json(params: "AnySweInstanceConfig", indent: Optional[int] = None) -> str:
    def redact(value: Any, key: str = "") -> Any:
        normalized = key.lower()
        if (
            any(s in normalized for s in ("api_key", "apikey", "secret", "password"))
            or normalized == "token"
            or normalized.endswith("_token")
        ):
            return "***"
        if isinstance(value, dict):
            return {k: redact(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [redact(v) for v in value]
        return value

    d = json.loads(params.model_dump_json())
    d.pop("agent_runtime_source", None)
    d.pop("agent_deps_url", None)
    d.pop("agent_overlay_files", None)
    return json.dumps(redact(d), indent=indent)


def _classify_agent_error(error: str) -> Optional[str]:
    text = error.lower()
    if "maximum iteration" in text or "max iterations" in text or "max_turns" in text:
        return "max_iteration"
    if "contextwindow" in text or "context window" in text:
        return "context_window"
    if "stuck in a loop" in text:
        return "stuck_in_loop"
    return "other" if text.strip() else None


def _policy_agent_execution(policy: Any):
    factory = getattr(policy, "agent_execution_context", None)
    return factory() if factory is not None else nullcontext()


def _notify_agent_finished(policy: Any, result: Any) -> None:
    """Report the actual host exec result before owned policy teardown."""
    callback = getattr(policy, "agent_finished", None)
    if callback is not None:
        callback(result)


def _should_mask_sample(
    resolved: bool,
    agent_error_kind: Optional[str],
    agent_timed_out: bool,
    error_kind: Optional[str],
    agent_stop_reason: Optional[str] = None,
) -> bool:
    expected_budget_stop = (agent_error_kind, agent_stop_reason) in {
        ("max_iteration", "max_iterations"),
        ("context_window", "context_exhausted"),
    }
    return bool(
        (resolved and agent_error_kind in ("max_iteration", "context_window") and not expected_budget_stop)
        or agent_timed_out
        or error_kind in ("eval_timeout", "sandbox", "grader", "harness")
    )


def _dataset_family(dataset_name: str) -> str:
    if "R2E-Gym" in dataset_name:
        return "r2e"
    if "SWE-bench_Multilingual" in dataset_name:
        return "swebench_multilingual"
    return "swebench"


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            return list(json.loads(value))
        except json.JSONDecodeError:
            return [value] if value else []
    return list(value or [])


def _r2e_resolved(instance: Dict[str, Any], log: str) -> bool:
    statuses: Dict[str, str] = {}
    for line in log.splitlines():
        fields = line.strip().replace(" - ", " ").split()
        if len(fields) > 1 and fields[0] in ("PASSED", "FAILED", "ERROR", "XFAIL", "SKIPPED"):
            statuses[fields[1]] = fields[0]
    required = _as_list(instance.get("FAIL_TO_PASS") or instance.get("fail_to_pass"))
    required += _as_list(instance.get("PASS_TO_PASS") or instance.get("pass_to_pass"))
    return bool(required) and all(statuses.get(test) == "PASSED" for test in required)


class AnySweAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: Optional[ModelServerRef] = None

    agent_server_module: str = Field(
        description="Import path to the agent module, e.g. responses_api_agents.hermes_agent.app"
    )
    agent_server_class: str = Field(description="Agent class name, e.g. HermesAgent")
    agent_config_class: str = Field(description="Agent config class name, e.g. HermesAgentConfig")
    agent_kwargs: Dict[str, Any] = Field(default_factory=dict)

    container_formatter: str = Field(description="Baked task image containing the AnySWE agent runtime")
    sandbox_provider: str | Dict[str, Any] = "sandbox"
    sandbox_spec: Dict[str, Any] = Field(default_factory=dict)
    sandbox_grading_policy: Optional[str] = Field(
        default=None,
        description="Trusted module:function projecting a fresh grading provider and spec",
    )
    sandbox_policy: Optional[str] = Field(
        default=None,
        description="Trusted module:function providing an async policy sandbox context",
    )
    agent_runtime_source: str = "baked"
    agent_runtime_overlay: Dict[str, str] = Field(
        default_factory=dict,
        description="Import path to host Python module or package directory shadowing the runtime's copy",
    )
    swebench_tests_timeout: int = 1800
    swebench_agent_timeout: int = 2700
    concurrency: Optional[int] = Field(default=256, gt=0)
    private_grader: Optional[PrivateGraderConfig] = None
    test_authority_restore: Optional[str] = Field(
        default=None,
        description="Trusted module:function(instance, patch) returning {files, command}; the default SWE-bench "
        "grader runs the command after the candidate patch applies and before the hidden tests",
    )


class AnySweServerConfig(BaseModel):
    run_session_id: str
    base_results_dir: Path
    model_server_url: str
    agent_deps_archive: Optional[Path] = None
    agent_deps_url: Optional[str] = None
    agent_deps_dir: Optional[Path] = None
    agent_overlay_files: Dict[str, str] = Field(default_factory=dict)
    agent_overlay_sha256: Optional[str] = None
    resolved_sandbox_provider: Dict[str, Any]
    sandbox_default_metadata: Dict[str, Any]


class AnySweInstanceConfig(AnySweAgentConfig, AnySweServerConfig):
    problem_info: Dict[str, Any]
    body: NeMoGymResponseCreateParamsNonStreaming
    persistent_dir: Path
    metrics_fpath: Path
    container: str
    rollout_id: Optional[str] = None
    mask_sample: bool = False
    private_grader_payload: Optional[Dict[str, Any]] = Field(default=None, exclude=True, repr=False)

    @property
    def instance_id(self) -> str:
        return self.problem_info["instance_id"]


class AnySweEpisodePayload(BaseModel):
    """Worker transport with private grader input kept out of serialized config."""

    model_config = ConfigDict(extra="forbid")

    instance_config: AnySweInstanceConfig
    private_grader_payload: Optional[Dict[str, Any]] = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def private_payload_has_one_owner(self) -> "AnySweEpisodePayload":
        if self.instance_config.private_grader_payload is not None:
            raise ValueError("private grader payload must be carried by the episode payload")
        return self

    @classmethod
    def from_instance_config(cls, params: AnySweInstanceConfig) -> "AnySweEpisodePayload":
        return cls(
            instance_config=params.model_copy(update={"private_grader_payload": None}),
            private_grader_payload=params.private_grader_payload,
        )

    def unpack(self) -> AnySweInstanceConfig:
        return self.instance_config.model_copy(update={"private_grader_payload": self.private_grader_payload})

    @classmethod
    def worker_payload(cls, params: AnySweInstanceConfig) -> Dict[str, Any]:
        """Build a plain Ray payload without attaching service-module model classes."""

        return {
            "instance_config": params.model_dump(mode="json"),
            "private_grader_payload": params.private_grader_payload,
        }


class AnySweVerifyResponse(SWEBenchMetrics, BaseVerifyResponse):
    instance_config: Dict[str, Any]
    done_reason: Optional[str] = Field(default=None, exclude_if=lambda value: value is None)
    ng_agent_observations: Optional[AgentObservationBundle] = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


def _read_saved_response(
    response_path: Path,
) -> tuple[Optional[NeMoGymResponse], Optional[AgentObservationBundle]]:
    if not response_path.exists():
        return None, None
    payload = json.loads(response_path.read_text())
    raw_observations = payload.pop(_INTERNAL_OBSERVATIONS_KEY, None)
    observations = (
        AgentObservationBundle.model_validate(raw_observations) if isinstance(raw_observations, dict) else None
    )
    return NeMoGymResponse.model_validate(payload), observations


class GymAgentHarnessProcessor(BaseModel):
    config: Any

    @property
    def _agent_key(self) -> str:
        return self.config.agent_server_module.split(".")[-2]

    def setup(self) -> Path:
        deps_dir = Path(__file__).parent / f"anyswe_{self._agent_key}_deps"
        sentinel = deps_dir / ".installed"
        scripts = Path(__file__).parent / "setup_scripts"
        script = scripts / f"{self._agent_key}_deps.sh"
        sources = (
            script,
            scripts / "_portable_python.sh",
            PARENT_DIR / "responses_api_agents" / self._agent_key / "requirements.txt",
            *sorted((PARENT_DIR / "responses_api_agents" / self._agent_key).glob("*.py")),
        )
        recipe = hashlib.sha256(b"".join(path.read_bytes() for path in sources if path.exists())).hexdigest()
        if sentinel.exists() and sentinel.read_text().strip() == recipe:
            return deps_dir

        lock = deps_dir.parent / f".{deps_dir.name}.lockdir"
        while True:
            try:
                lock.mkdir(exist_ok=False)
                break
            except FileExistsError:
                if time.time() - lock.stat().st_mtime > 3600:
                    shutil.rmtree(lock, ignore_errors=True)
                else:
                    time.sleep(5)

        try:
            if sentinel.exists() and sentinel.read_text().strip() == recipe:
                return deps_dir
            if not script.exists():
                raise ValueError(f"missing agent runtime setup script: {script}")
            deps_dir.mkdir(parents=True, exist_ok=True)
            proc = Popen(f"DEPS_DIR={deps_dir} NEMO_GYM_ROOT={PARENT_DIR} bash {script}", shell=True)
            if proc.wait() != 0:
                raise RuntimeError(f"agent runtime setup failed: {script}")
            sentinel.write_text(recipe)
            return deps_dir
        finally:
            shutil.rmtree(lock, ignore_errors=True)

    def write_runner(self) -> None:
        cfg: AnySweInstanceConfig = self.config

        (cfg.persistent_dir / "instruction.txt").write_text(cfg.problem_info.get("problem_statement", ""))
        runner = Path(__file__).with_name("agent_runner.py").read_text()
        (cfg.persistent_dir / "agent_runner.py").write_text(runner)


class _AnySweAgentServer(SimpleResponsesAPIAgent):
    """Runs a Gym agent in each task container."""

    config: AnySweAgentConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _sem: Optional[Semaphore] = None
    _server: Optional[AnySweServerConfig] = None

    def model_post_init(self, context: Any) -> None:
        self._sem = Semaphore(self.config.concurrency) if self.config.concurrency is not None else None

        model_url = self.config.sandbox_model_base_url or ""
        if self.config.model_server is not None:
            model_cfg = get_first_server_config_dict(
                self.server_client.global_config_dict, self.config.model_server.name
            )
            if not model_url:
                model_url = self.server_client._build_server_base_url(model_cfg)

        workspace = Path(__file__).parent
        agent_deps_archive = None
        agent_deps_url = None
        agent_deps_dir = None
        if self.config.agent_runtime_source == "auto":
            agent_deps_dir = GymAgentHarnessProcessor(config=self.config).setup()
            agent_deps_archive = workspace / f".{agent_deps_dir.name}.tar.gz"
            sentinel = agent_deps_dir / ".installed"
            if not agent_deps_archive.exists() or agent_deps_archive.stat().st_mtime < sentinel.stat().st_mtime:
                temporary = agent_deps_archive.with_suffix(".tmp")
                with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
                    archive.add(agent_deps_dir, arcname=".")
                temporary.replace(agent_deps_archive)
        elif self.config.agent_runtime_source != "baked":
            if "://" in self.config.agent_runtime_source:
                agent_deps_url = self.config.agent_runtime_source
            else:
                source = Path(self.config.agent_runtime_source).expanduser()
                if source.is_dir():
                    if not (source / "bin" / "python").is_file():
                        raise ValueError(f"agent runtime directory lacks bin/python: {source}")
                    agent_deps_dir = source.resolve()
                elif source.is_file():
                    agent_deps_archive = source
                else:
                    raise ValueError(f"agent runtime archive not found: {source}")
        results_dir = Path(self.server_client.global_config_dict.get("results_dir") or workspace)
        session_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        agent_overlay_files = _overlay_files(self.config.agent_runtime_overlay)
        agent_overlay_sha256 = _overlay_sha256(agent_overlay_files)
        if agent_overlay_sha256 is not None:
            print(
                f"[anyswe_agent] agent_runtime_overlay packages={sorted(self.config.agent_runtime_overlay)} "
                f"files={len(agent_overlay_files)} sha256={agent_overlay_sha256}",
                flush=True,
            )

        resolved_sandbox_provider = resolve_provider_config(
            self.config.sandbox_provider, self.server_client.global_config_dict
        )
        if agent_deps_dir is not None and not any(
            name in resolved_sandbox_provider for name in ("apptainer", "enroot", "native_enroot", "swe_gym")
        ):
            raise ValueError(
                "a directory agent_runtime_source is bind-mounted and needs an Apptainer or Enroot provider; "
                "other providers take an archive path or URL"
            )
        self._server = AnySweServerConfig(
            run_session_id=session_id,
            base_results_dir=results_dir / f"anyswe_results_{session_id}",
            model_server_url=model_url,
            agent_deps_archive=agent_deps_archive,
            agent_deps_url=agent_deps_url,
            agent_deps_dir=agent_deps_dir,
            agent_overlay_files=agent_overlay_files,
            agent_overlay_sha256=agent_overlay_sha256,
            resolved_sandbox_provider=resolved_sandbox_provider,
            sandbox_default_metadata=resolve_provider_metadata(
                self.config.sandbox_provider, self.server_client.global_config_dict
            ),
        )
        super().model_post_init(context)

    @staticmethod
    def _sandbox_image(problem_info: Dict[str, Any]) -> str:
        instance = problem_info.get("instance_dict", {})
        instance = json.loads(instance) if isinstance(instance, str) else instance
        explicit_image = problem_info.get("image") or instance.get("image") or instance.get("docker_image")
        if explicit_image:
            return str(explicit_image).removeprefix("docker://")
        formatter = problem_info["container_formatter"]
        if formatter.endswith(".sif"):
            raise ValueError("sandbox_provider requires a container image, not a .sif file")
        formatter = formatter.removeprefix("docker://")
        instance_id = problem_info["instance_id"].replace("__", "_1776_").lower()
        image = formatter.format(instance_id=instance_id)
        if ":" not in image.rsplit("/", 1)[-1]:
            image += ":latest"
        return image


class AnySweSandboxRunner:
    """Runs one complete AnySWE harness and grading episode."""

    def __init__(self, params: AnySweInstanceConfig):
        self.params = params

    @staticmethod
    def _sandbox_spec(
        params: AnySweInstanceConfig,
        *,
        files: Optional[Dict[str, str]] = None,
        binds: Sequence[str] = (),
        workdir: Optional[str] = None,
    ) -> SandboxSpec:
        config = dict(params.sandbox_spec)
        provider_options = dict(config.pop("provider_options", {}))
        if binds:
            existing = provider_options.get("binds", [])
            existing = [existing] if isinstance(existing, str) else list(existing)
            provider_options["binds"] = existing + list(binds)
        metadata = {
            **params.sandbox_default_metadata,
            **config.pop("metadata", {}),
            "nemo_gym_agent": "anyswe_agent",
            "instance_id": params.instance_id[:63],
        }
        resolved_workdir = workdir if workdir is not None else config.pop("workdir", "/testbed")
        if workdir is not None:
            config.pop("workdir", None)
        return SandboxSpec(
            image=params.container,
            ttl_s=config.pop("ttl_s", params.swebench_agent_timeout + params.swebench_tests_timeout + 600),
            ready_timeout_s=config.pop("ready_timeout_s", 1200),
            workdir=resolved_workdir,
            env=config.pop("env", {}),
            files=files or {},
            metadata=metadata,
            resources=SandboxResources.from_mapping(config.pop("resources", {})),
            entrypoint=config.pop("entrypoint", None),
            provider_options=provider_options,
        )

    @staticmethod
    def _sandbox_agent_env(params: AnySweInstanceConfig) -> Dict[str, str]:
        sampling = {
            key: getattr(params.body, key)
            for key in ("temperature", "top_p", "max_output_tokens")
            if getattr(params.body, key, None) is not None
        }
        model_name = params.body.model or "model"
        agent_kwargs = dict(params.agent_kwargs)
        if params.sandbox_model_base_url is not None:
            agent_kwargs["sandbox_model_base_url"] = params.sandbox_model_base_url
        env = {
            "NGSWE_MODEL_NAME": model_name,
            "NGSWE_MODEL_URL": params.model_server_url,
            "NGSWE_AGENT_MODULE": params.agent_server_module,
            "NGSWE_AGENT_CLASS": params.agent_server_class,
            "NGSWE_AGENT_CONFIG_CLASS": params.agent_config_class,
            "NGSWE_AGENT_KWARGS_B64": base64.b64encode(json.dumps(agent_kwargs).encode()).decode(),
            "NGSWE_SAMPLING_B64": base64.b64encode(json.dumps(sampling).encode()).decode(),
            # Never leak the worker's wrapper/Task imports into the task image.
            "PYTHONPATH": AGENT_OVERLAY_DIR if getattr(params, "agent_overlay_files", None) else "",
        }
        # Invocation identity remains available even when a sandbox relay owns
        # upstream URL scoping. Its explicit endpoint prevents double scoping.
        if params.rollout_id:
            env["NGSWE_ROLLOUT_ID"] = params.rollout_id
        return env

    @staticmethod
    def _instance_dict(params: AnySweInstanceConfig) -> Dict[str, Any]:
        value = params.problem_info["instance_dict"]
        return json.loads(value) if isinstance(value, str) else dict(value)

    @staticmethod
    def _private_grader_plan(params: AnySweInstanceConfig, patch: Optional[str]) -> PrivateGraderPlan:
        if params.private_grader is None or params.private_grader_payload is None:
            raise RuntimeError("private grader is not configured for this instance")
        raw_plan = params.private_grader.load()(dict(params.private_grader_payload), patch)
        return PrivateGraderPlan.model_validate(raw_plan)

    @staticmethod
    def _apply_patch_script() -> str:
        return (
            "cd /testbed && "
            "if git apply --verbose /root/patch.diff || "
            "git apply --verbose --reject /root/patch.diff || "
            "patch --batch --fuzz=5 -p1 -i /root/patch.diff; then "
            "echo '>>>>> Applied Patch'; else "
            "echo '>>>>> Patch Apply Failed'; exit 1; fi\n"
        )

    @staticmethod
    def _grading_sandbox(params: AnySweInstanceConfig, spec: SandboxSpec) -> AsyncSandbox:
        provider = params.resolved_sandbox_provider
        if params.sandbox_grading_policy is not None:
            module, separator, name = params.sandbox_grading_policy.partition(":")
            if not separator or not module or not name or ":" in name:
                raise ValueError("sandbox_grading_policy must be module:function")
            callback = getattr(importlib.import_module(module), name)
            provider, spec = callback(provider, spec)
        return AsyncSandbox(provider, spec)

    async def _grade_sandbox_patch(
        self, params: AnySweInstanceConfig, patch: str
    ) -> tuple[bool, Optional[str], Optional[str]]:
        if params.private_grader is not None:
            return await self._grade_private_patch(params, patch)
        if _dataset_family(params.problem_info.get("dataset_name", "")) == "r2e":
            return await self._grade_r2e_patch(params, patch)

        from swebench.harness.grading import get_eval_report
        from swebench.harness.test_spec.test_spec import make_test_spec

        instance = self._instance_dict(params)
        test_spec = make_test_spec(instance, namespace="swebench")
        restore = {"files": {}, "command": ""}
        if params.test_authority_restore is not None:
            module, _, name = params.test_authority_restore.partition(":")
            restore = getattr(importlib.import_module(module), name)(instance, patch)
        eval_script = (
            'export PYTEST_ADDOPTS="-rA ${PYTEST_ADDOPTS:-}"\n'
            + self._apply_patch_script()
            + restore["command"]
            + test_spec.eval_script
        )
        spec = self._sandbox_spec(
            params,
            files={
                **restore["files"],
                "/root/patch.diff": patch if patch.endswith("\n") else patch + "\n",
                "/tmp/anyswe_eval.sh": eval_script,
            },
        )
        sandbox = self._grading_sandbox(params, spec)
        try:
            await sandbox.start()
            result = await sandbox.exec(
                "bash /tmp/anyswe_eval.sh",
                cwd="/testbed",
                timeout_s=params.swebench_tests_timeout,
                user="root",
            )
            _write_grader_receipt(params.persistent_dir, result)
        finally:
            await sandbox.stop()
            update_metrics(params.metrics_fpath, sandbox_memory_events(sandbox, "grader_sandbox"))

        if result.error_type in ("timeout", "sandbox"):
            return (
                False,
                "eval_timeout" if result.error_type == "timeout" else "sandbox",
                _error_detail(
                    f"grader exec error_type={result.error_type}", f"rc={result.return_code}", result.stderr
                ),
            )

        log = "\n".join(part for part in (result.stdout, result.stderr) if part)
        with tempfile.NamedTemporaryFile("w", suffix=".log") as log_file:
            log_file.write(log)
            log_file.flush()
            prediction = {
                "instance_id": params.instance_id,
                "model_name_or_path": params.body.model or "anyswe",
                "model_patch": patch,
            }
            report = get_eval_report(test_spec, prediction, log_file.name, include_tests_status=True)[
                params.instance_id
            ]
        return bool(report["resolved"]), None, None

    async def _grade_private_patch(
        self, params: AnySweInstanceConfig, patch: str
    ) -> tuple[bool, Optional[str], Optional[str]]:
        plan = self._private_grader_plan(params, patch)
        sandbox = self._grading_sandbox(
            params,
            self._sandbox_spec(params, files=plan.files, workdir=plan.workdir),
        )
        try:
            await sandbox.start()
            result = await sandbox.exec(
                plan.command,
                cwd=plan.workdir,
                env=plan.env,
                timeout_s=params.swebench_tests_timeout,
                user="root",
            )
            _write_grader_receipt(params.persistent_dir, result)
        finally:
            await sandbox.stop()
            update_metrics(params.metrics_fpath, sandbox_memory_events(sandbox, "grader_sandbox"))

        if result.error_type == "timeout":
            return (
                False,
                "eval_timeout",
                _error_detail(f"grader exec error_type={result.error_type}", f"rc={result.return_code}"),
            )
        if result.error_type is not None:
            return (
                False,
                "sandbox" if result.error_type == "sandbox" else "grader",
                _error_detail(f"grader exec error_type={result.error_type}", f"rc={result.return_code}"),
            )
        if result.return_code == 0:
            return True, None, None
        if result.return_code == params.private_grader.unresolved_return_code:
            return False, None, None
        if result.return_code == params.private_grader.timeout_return_code:
            return (
                False,
                "eval_timeout",
                _error_detail(f"grader exec error_type={result.error_type}", f"rc={result.return_code}"),
            )
        return (
            False,
            "grader",
            _error_detail(f"grader exec error_type={result.error_type}", f"rc={result.return_code}"),
        )

    async def _grade_r2e_patch(
        self, params: AnySweInstanceConfig, patch: str
    ) -> tuple[bool, Optional[str], Optional[str]]:
        instance = self._instance_dict(params)
        eval_script = instance.get("eval_script") or params.problem_info.get("eval_script")
        if not eval_script:
            eval_script = (
                "if [ -f /run_tests.sh ]; then bash /run_tests.sh; "
                "elif [ -f /testbed/run_tests.sh ]; then bash /testbed/run_tests.sh; "
                "elif [ -f /root/run_tests.sh ]; then bash /root/run_tests.sh; "
                "else echo 'R2E eval script not found'; exit 127; fi"
            )
        script = 'export PYTEST_ADDOPTS="-rA ${PYTEST_ADDOPTS:-}"\n' + self._apply_patch_script() + str(eval_script)
        spec = self._sandbox_spec(
            params,
            files={
                "/root/patch.diff": patch if patch.endswith("\n") else patch + "\n",
                "/tmp/anyswe_eval.sh": script,
            },
        )
        sandbox = self._grading_sandbox(params, spec)
        try:
            await sandbox.start()
            result = await sandbox.exec(
                "bash /tmp/anyswe_eval.sh",
                cwd="/testbed",
                timeout_s=params.swebench_tests_timeout,
                user="root",
            )
            _write_grader_receipt(params.persistent_dir, result)
        finally:
            await sandbox.stop()
            update_metrics(params.metrics_fpath, sandbox_memory_events(sandbox, "grader_sandbox"))

        if result.error_type in ("timeout", "sandbox"):
            return (
                False,
                "eval_timeout" if result.error_type == "timeout" else "sandbox",
                _error_detail(
                    f"grader exec error_type={result.error_type}", f"rc={result.return_code}", result.stderr
                ),
            )

        log = "\n".join(part for part in (result.stdout, result.stderr) if part)
        return _r2e_resolved(instance, log), None, None

    async def run(self) -> NeMoGymResponse:
        params = self.params
        files = {
            "/trajectories_mount/instruction.txt": (params.persistent_dir / "instruction.txt").read_text(),
            "/trajectories_mount/agent_runner.py": (params.persistent_dir / "agent_runner.py").read_text(),
            **params.agent_overlay_files,
        }
        setup_plan = None
        if params.private_grader is not None:
            setup_error = None
            try:
                setup_plan = self._private_grader_plan(params, None)
                collisions = files.keys() & setup_plan.files.keys()
                if collisions:
                    raise ValueError(f"private grader setup files conflict with agent files: {sorted(collisions)}")
                files.update(setup_plan.files)
            except Exception as exc:
                # Setup precedes the runner's sandbox guard. Keep the original private
                # diagnostic here; Gym's HTTP exception middleware receives only its class.
                (params.persistent_dir / "eval_error.txt").write_text(repr(exc))
                setup_error = RuntimeError(f"private grader setup exception type={type(exc).__name__}")
            if setup_error is not None:
                # Detach the private cause before crossing the worker/HTTP boundary.
                raise setup_error from None
        workdir = setup_plan.workdir if setup_plan is not None else None
        spec = self._sandbox_spec(
            params,
            files=files,
            binds=(f"{params.agent_deps_dir}:/agent_deps_mount:ro",) if params.agent_deps_dir else (),
            workdir=workdir,
        )
        agent_workdir = spec.workdir
        result = None
        sandbox_failed = False
        sandbox_error_kind = None
        error_detail: Optional[str] = None
        sandbox = None
        policy_stack = AsyncExitStack()
        policy = None
        started = time.time()
        try:
            if params.sandbox_policy is not None:
                module, separator, name = params.sandbox_policy.partition(":")
                if not separator or not module or not name or ":" in name:
                    raise ValueError("sandbox_policy must be module:function")
                callback = getattr(importlib.import_module(module), name)
                policy = await policy_stack.enter_async_context(callback(params, spec))
                sandbox = AsyncSandbox(policy.provider, policy.spec)
            else:
                sandbox = AsyncSandbox(params.resolved_sandbox_provider, spec)
            await sandbox.start()
            if policy is not None:
                await policy.started(sandbox)
            update_metrics(params.metrics_fpath, {"sandbox_ready_at": time.time()})
            if setup_plan is not None:
                setup = await sandbox.exec(
                    setup_plan.command,
                    cwd=setup_plan.workdir,
                    env=setup_plan.env,
                    timeout_s=params.swebench_tests_timeout,
                    user="root",
                )
                _write_grader_receipt(params.persistent_dir, setup, label="setup")
                if setup.error_type is not None or setup.return_code != 0:
                    sandbox_error_kind = {
                        "timeout": "eval_timeout",
                        "sandbox": "sandbox",
                    }.get(setup.error_type, "grader")
                    raise RuntimeError(
                        f"private grader setup failed: error_type={setup.error_type!r} return_code={setup.return_code}"
                    )
            await sandbox.exec(
                "mkdir -p /agent_deps_mount /sandbox/home /sandbox/tmp /trajectories_mount",
                timeout_s=setup_timeout_seconds(30),
                user="root",
            )
            external_runtime = params.agent_deps_archive is not None or params.agent_deps_url is not None
            if params.agent_deps_archive is not None:
                await sandbox.upload(params.agent_deps_archive, "/sandbox/anyswe_agent_deps.tar.gz")
            elif params.agent_deps_url is not None:
                runtime_url = shlex.quote(params.agent_deps_url)
                python_fetch = shlex.quote(
                    "import urllib.request; "
                    f"urllib.request.urlretrieve({params.agent_deps_url!r}, '/sandbox/anyswe_agent_deps.tar.gz')"
                )
                fetched = await sandbox.exec(
                    "if command -v curl >/dev/null 2>&1; then "
                    f"curl -fsSL -o /sandbox/anyswe_agent_deps.tar.gz {runtime_url}; "
                    "elif command -v wget >/dev/null 2>&1; then "
                    f"wget -qO /sandbox/anyswe_agent_deps.tar.gz {runtime_url}; "
                    "elif command -v python3 >/dev/null 2>&1; then "
                    f"python3 -c {python_fetch}; "
                    "else echo 'runtime download requires curl, wget, or python3' >&2; exit 127; fi",
                    timeout_s=setup_timeout_seconds(600),
                    user="root",
                )
                if fetched.return_code != 0:
                    raise RuntimeError(f"agent runtime fetch failed: {(fetched.stderr or '')[:300]}")
            else:
                runtime = await sandbox.exec(
                    "test -x /agent_deps_mount/bin/python", timeout_s=setup_timeout_seconds(30), user="root"
                )
                if runtime.return_code != 0:
                    raise RuntimeError("task image does not contain /agent_deps_mount/bin/python")
            if external_runtime:
                unpacked = await sandbox.exec(
                    "mkdir -p /sandbox/agent_deps_runtime && "
                    "tar -xzf /sandbox/anyswe_agent_deps.tar.gz -C /sandbox/agent_deps_runtime && "
                    "rm -f /sandbox/anyswe_agent_deps.tar.gz",
                    cwd=agent_workdir,
                    timeout_s=setup_timeout_seconds(600),
                    user="root",
                )
                if unpacked.return_code != 0:
                    raise RuntimeError(f"agent runtime extraction failed: {(unpacked.stderr or '')[:300]}")
            if _dataset_family(params.problem_info.get("dataset_name", "")) == "r2e":
                await sandbox.exec(
                    "rm -rf /r2e_tests /root/r2e_tests /testbed/r2e_tests; "
                    "for f in /run_tests.sh /root/run_tests.sh /testbed/run_tests.sh; do "
                    'if grep -qs r2e_tests "$f"; then rm -f "$f"; fi; done',
                    timeout_s=setup_timeout_seconds(30),
                    user="root",
                )
            runtime_dir = "/sandbox/agent_deps_runtime" if external_runtime else "/agent_deps_mount"
            agent_env = self._sandbox_agent_env(policy.agent_params if policy is not None else params)
            agent_env["NGSWE_AGENT_DEPS_DIR"] = runtime_dir
            command = (
                f"HOME=/sandbox/home TMPDIR=/sandbox/tmp {runtime_dir}/bin/python /trajectories_mount/agent_runner.py"
            )
            if policy is not None:
                await policy.prepare(sandbox)
                command = policy.wrap_command(command, runtime_dir)
            with _policy_agent_execution(policy):
                result = await sandbox.exec(
                    command,
                    cwd=agent_workdir,
                    env=agent_env,
                    timeout_s=params.swebench_agent_timeout,
                    user="root",
                )
            _notify_agent_finished(policy, result)
            await notify_policy_phase_done(params.body)
            for remote, local in (
                ("/trajectories_mount/response.json", params.persistent_dir / "response.json"),
                ("/trajectories_mount/patch.diff", params.persistent_dir / "patch.diff"),
            ):
                exists = await sandbox.exec(f"test -f {shlex.quote(remote)}", timeout_s=30, user="root")
                if exists.return_code == 0:
                    await sandbox.download(remote, local)
        except Exception as exc:
            sandbox_failed = True
            sandbox_error_kind = sandbox_error_kind or "sandbox"
            error_detail = _grading_exception_detail(params, exc)
            error_path = params.persistent_dir / "sandbox_error.txt"
            update_metrics(
                params.metrics_fpath,
                {
                    "resolved": False,
                    "patch_exists": False,
                    "error_kind": sandbox_error_kind,
                    "error_detail": error_detail,
                    "artifact_path": str(error_path),
                    "mask_sample": True,
                    "agent_run_time": time.time() - started,
                },
            )
            error_path.write_text(repr(exc))
        finally:
            try:
                if sandbox is not None:
                    await sandbox.stop()
            finally:
                # A policy callback may attest process quiescence here. Downloads
                # above remain untrusted until the context exits successfully.
                await policy_stack.aclose()
        memory_events = sandbox_memory_events(sandbox)
        integrity = getattr(policy, "verifier_integrity", None)
        integrity_metrics = integrity.metrics() if integrity is not None else {}

        patch_path = params.persistent_dir / "patch.diff"
        patch = patch_path.read_bytes().decode("utf-8") if patch_path.exists() else ""
        response_path = params.persistent_dir / "response.json"
        saved, observations = _read_saved_response(response_path)
        harness_error = harness_response_error(saved, observations) if saved is not None else None

        agent_error = ""
        agent_timed_out = False
        error_kind = sandbox_error_kind if sandbox_failed else None
        if result is not None:
            (params.persistent_dir / "agent_result.json").write_text(
                json.dumps(
                    {
                        "return_code": result.return_code,
                        "error_type": result.error_type,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                )
            )
            # Successful stdout contains the serialized conversation; neither that
            # text nor a normal budget warning is a structured harness failure.
            if result.return_code != 0 or result.error_type:
                agent_error = "\n".join(part for part in (result.stdout, result.stderr) if part)
            agent_timed_out = result.error_type == "timeout"
            if result.error_type == "sandbox":
                error_kind = "sandbox"
                error_detail = error_detail or _error_detail(
                    f"agent exec error_type=sandbox rc={result.return_code}", result.stderr
                )
            memory_kill = harness_memory_kill(result.return_code, memory_events)
            if memory_kill is not None:
                # This result comes from the admitted harness. Kernel counters
                # prove its memory-limit death, not independent infrastructure
                # causation. Keep that unproven outcome as a task zero; failures
                # during host setup still take the typed sandbox exception path.
                error_kind = "agent_memory_limit"
                error_detail = f"{memory_kill}; in-harness causation unproven"
        agent_stop_reason = (saved.metadata or {}).get("stop_reason") if saved is not None else None
        if not isinstance(agent_stop_reason, str):
            agent_stop_reason = None
        if saved is not None:
            declared_errors = [
                saved.error.message if saved.error is not None else None,
                (saved.metadata or {}).get("hermes_error"),
            ]
            agent_error += "\n" + "\n".join(str(part) for part in declared_errors if part)
        if agent_stop_reason == "max_iterations" and not agent_error.strip():
            agent_error += "\nmax iterations"
        agent_error_kind = (
            "context_window" if agent_stop_reason == "context_exhausted" else _classify_agent_error(agent_error)
        )
        resolved = False
        # A runner exception outside the typed transport cases retains the old
        # task-zero behavior. Its partial patch is evidence, not a successful
        # harness completion; transient failures below remain native masks.
        if saved is not None and (saved.metadata or {}).get("runner_exception") == "true" and harness_error is None:
            error_kind = error_kind or "agent"
        if patch.strip() and error_kind is None:
            grading_started_at = time.time()
            update_metrics(params.metrics_fpath, {"grading_started_at": grading_started_at})
            try:
                resolved, error_kind, error_detail = await self._grade_sandbox_patch(params, patch)
            except SandboxCreateError as exc:
                error_kind = "sandbox"
                error_detail = _grading_exception_detail(params, exc)
                (params.persistent_dir / "eval_error.txt").write_text(repr(exc))
            except (asyncio.TimeoutError, TimeoutError) as exc:
                error_kind = "eval_timeout"
                error_detail = _grading_exception_detail(params, exc)
                (params.persistent_dir / "eval_error.txt").write_text(repr(exc))
            except Exception as exc:
                error_kind = "grader"
                error_detail = _grading_exception_detail(params, exc)
                (params.persistent_dir / "eval_error.txt").write_text(repr(exc))

            finally:
                grading_completed_at = time.time()
                update_metrics(params.metrics_fpath, {"grading_completed_at": grading_completed_at})
                receipt_path = params.persistent_dir / "grader_result.json"
                receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
                receipt.update(started_at=grading_started_at, completed_at=grading_completed_at)
                receipt_path.write_text(json.dumps(receipt))

        if harness_error is not None and error_kind is None:
            error_kind = "harness"
            error_detail = harness_error.message
        if integrity_metrics.get("reward_hack_detected"):
            resolved = False
        eval_timed_out = error_kind == "eval_timeout"
        mask_sample = _should_mask_sample(resolved, agent_error_kind, agent_timed_out, error_kind, agent_stop_reason)
        update_metrics(
            params.metrics_fpath,
            {
                "resolved": resolved,
                "patch_exists": bool(patch.strip()),
                "model_patch": patch or None,
                "agent_timed_out": agent_timed_out,
                "agent_error_kind": agent_error_kind,
                "agent_stop_reason": agent_stop_reason,
                "eval_timed_out": eval_timed_out,
                "error_kind": error_kind,
                "error_detail": error_detail,
                "mask_sample": mask_sample,
                "agent_run_time": time.time() - started,
                **memory_events,
                **integrity_metrics,
            },
        )

        response = NeMoGymResponse(
            id=f"anyswe-{params.instance_id}",
            created_at=int(time.time()),
            model=params.body.model or (saved.model if saved is not None else params.name),
            object="response",
            output=saved.output if saved is not None else [],
            parallel_tool_calls=params.body.parallel_tool_calls,
            tool_choice=params.body.tool_choice,
            tools=(saved.tools or []) if saved is not None else [],
            usage=saved.usage if saved is not None else None,
            status="failed" if harness_error is not None else None,
            error=harness_error,
            metadata={
                "input": json.dumps([]),
                "metrics": params.metrics_fpath.read_text(),
                "instance_config": _safe_config_json(params),
            },
        )
        if observations is None or params.rollout_id is None:
            return response
        return response.model_copy(update={_INTERNAL_OBSERVATIONS_KEY: observations.model_dump(mode="json")})


async def run_swe_episode(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Stable worker entry point returning the canonical JSON response shape."""

    task_started_at = time.time()
    envelope = AnySweEpisodePayload.model_validate(payload)
    params = envelope.unpack()
    update_metrics(params.metrics_fpath, {"task_started_at": task_started_at})
    response = await AnySweSandboxRunner(params).run()
    return response.model_dump(mode="json")


class AnySweAgent(_AnySweAgentServer):
    """Owns the AnySWE API and delegates complete episodes to one runner."""

    _sandbox_spec = staticmethod(AnySweSandboxRunner._sandbox_spec)
    _sandbox_agent_env = staticmethod(AnySweSandboxRunner._sandbox_agent_env)
    _instance_dict = staticmethod(AnySweSandboxRunner._instance_dict)
    _private_grader_plan = staticmethod(AnySweSandboxRunner._private_grader_plan)
    _apply_patch_script = staticmethod(AnySweSandboxRunner._apply_patch_script)
    _grading_sandbox = staticmethod(AnySweSandboxRunner._grading_sandbox)
    _grade_sandbox_patch = AnySweSandboxRunner._grade_sandbox_patch
    _grade_private_patch = AnySweSandboxRunner._grade_private_patch
    _grade_r2e_patch = AnySweSandboxRunner._grade_r2e_patch

    async def _run_agent_in_sandbox(self, params: AnySweInstanceConfig) -> NeMoGymResponse:
        worker = importlib.import_module("responses_api_agents.anyswe_agent.app").run_swe_episode
        update_metrics(params.metrics_fpath, {"submitted_at": time.time()})
        result = await run_sandbox_episode(
            worker,
            AnySweEpisodePayload.worker_payload(params),
            self._sandbox_spec(params).resources,
            episode_id=params.rollout_id,
        )
        return NeMoGymResponse.model_validate(result)

    def _setup_params(
        self, body: NeMoGymResponseCreateParamsNonStreaming, rollout_id: Optional[str] = None
    ) -> AnySweInstanceConfig:
        metadata = dict(body.metadata or {})
        private_grader_payload = None
        if self.config.private_grader is not None:
            metadata_key = self.config.private_grader.metadata_key
            if metadata_key not in metadata:
                raise ValueError(f"private grader metadata is missing: {metadata_key}")
            private_grader_payload = metadata.pop(metadata_key)
            if isinstance(private_grader_payload, str):
                try:
                    private_grader_payload = json.loads(private_grader_payload)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"private grader metadata is not valid JSON: {metadata_key}") from exc
            if not isinstance(private_grader_payload, dict):
                raise ValueError(f"private grader metadata must be an object: {metadata_key}")
            body = body.model_copy(update={"metadata": metadata})
        problem_info = metadata | {"container_formatter": self.config.container_formatter}
        instance_id = problem_info.get("instance_id", "unknown")
        persistent_dir = self._server.base_results_dir / (
            f"{instance_id}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        )
        persistent_dir.mkdir(parents=True, exist_ok=True)
        server_config = self._server.model_dump()
        params = AnySweInstanceConfig(
            **self.config.model_dump(),
            **server_config,
            problem_info=problem_info,
            body=body,
            persistent_dir=persistent_dir,
            metrics_fpath=persistent_dir / "nemo_gym_metrics.json",
            container=self._sandbox_image(problem_info),
            rollout_id=rollout_id,
            private_grader_payload=private_grader_payload,
        )
        params.metrics_fpath.write_text("{}")
        GymAgentHarnessProcessor(config=params).write_runner()
        return params

    async def _responses(
        self, body: NeMoGymResponseCreateParamsNonStreaming, rollout_id: Optional[str] = None
    ) -> NeMoGymResponse:
        params = self._setup_params(body, rollout_id)
        (params.persistent_dir / "params.json").write_text(_safe_config_json(params, indent=2))
        try:
            return await self._run_agent_in_sandbox(params)
        except Exception:
            traceback_path = params.persistent_dir / "traceback.err"
            traceback_path.write_text(format_exc())
            print(f"[{params.instance_id}] exception: see {traceback_path}", flush=True)
            raise

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        return await self._responses(body)

    async def run(self, body: AnySweRunRequest) -> AnySweVerifyResponse:
        async with self._sem if self._sem is not None else nullcontext():
            response = await self._responses(body.responses_create_params, self.rollout_id_from_run(body))

            meta, response.metadata = response.metadata, None
            response_payload = response.model_dump(mode="json")
            raw_observations = response_payload.pop(_INTERNAL_OBSERVATIONS_KEY, None)
            observations = (
                AgentObservationBundle.model_validate(raw_observations) if isinstance(raw_observations, dict) else None
            )
            response = NeMoGymResponse.model_validate(response_payload)
            metrics = SWEBenchMetrics.model_validate_json(meta["metrics"])
            instance_config = AnySweInstanceConfig.model_validate_json(meta["instance_config"])

            return AnySweVerifyResponse(
                responses_create_params=instance_config.body.model_dump()
                | {"input": json.loads(meta["input"]), "tools": [t.model_dump() for t in (response.tools or [])]},
                response=response,
                reward=1.0 if metrics.resolved else 0.0,
                done_reason="context_exhausted" if metrics.agent_stop_reason == "context_exhausted" else None,
                **metrics.model_dump(),
                instance_config=instance_config.model_dump(),
                ng_agent_observations=observations,
            )


if __name__ == "__main__":
    AnySweAgent.run_webserver()
