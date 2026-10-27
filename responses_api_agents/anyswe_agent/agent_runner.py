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
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import Request


sys.path.insert(0, "/nemo_gym_mount")
agent_deps_dir = os.environ.get("NGSWE_AGENT_DEPS_DIR", "/agent_deps_mount")
os.environ["PATH"] = os.environ.get("PATH", "") + f":{agent_deps_dir}/bin"

_REPO_CANDIDATES = ("/testbed", "/workspace/repo", "/app", "/root/repo")


def _json_env(name: str) -> dict:
    encoded = os.environ.get(f"{name}_B64")
    if encoded:
        return json.loads(base64.b64decode(encoded).decode())
    return json.loads(os.environ.get(name, "{}"))


def _find_repo(task_cwd: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if task_cwd is not None:
        candidates.append(Path(task_cwd))
    candidates.extend(Path(candidate) for candidate in _REPO_CANDIDATES)
    seen: set[str] = set()
    for repo in candidates:
        key = str(repo)
        if key in seen:
            continue
        seen.add(key)
        if repo.exists() and (repo / ".git").exists():
            return repo
    return None


def _baseline_git_dir(index_path: Path) -> Path:
    return index_path.with_name(index_path.name + ".git")


def _alternate_index_env(index_path: Path) -> dict[str, str]:
    # A caller's Git context must not redirect the task's repository or object store.
    context = {
        "GIT_DIR",
        "GIT_COMMON_DIR",
        "GIT_WORK_TREE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_INDEX_FILE",
    }
    return {
        **{key: value for key, value in os.environ.items() if key not in context},
        "GIT_INDEX_FILE": str(index_path),
    }


def _snapshot_repo(repo: Path, index_path: Path) -> str:
    """Write the task image's initial working tree to an alternate Git index."""
    index_path.unlink(missing_ok=True)
    env = _alternate_index_env(index_path)
    subprocess.run(["git", "read-tree", "HEAD"], check=True, cwd=repo, env=env)
    subprocess.run(["git", "add", "-A"], check=True, cwd=repo, env=env)
    baseline_tree = subprocess.run(
        ["git", "write-tree"],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo,
        env=env,
    ).stdout.strip()
    # Keep only the snapshot's reachable trees/blobs outside the agent worktree.
    # No alternates or hardlinks may depend on .git surviving the agent's tools.
    git_dir = _baseline_git_dir(index_path)
    git_dir.mkdir(mode=0o700)
    object_format = subprocess.run(
        ["git", "rev-parse", "--show-object-format"],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "init", "--bare", "-q", f"--object-format={object_format}", str(git_dir)], check=True, env=env
    )
    for name in ("config", "info/exclude", "info/attributes"):
        source = subprocess.run(
            ["git", "rev-parse", "--git-path", name], cwd=repo, env=env, check=True, capture_output=True, text=True
        ).stdout.strip()
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = repo / source_path
        if source_path.is_file():
            shutil.copyfile(source_path, git_dir / name)
    # Stream directly onto the archive filesystem; Git's pack temporary file
    # otherwise lives in the task repository and rename crosses mount points.
    pack_path = git_dir / "objects/pack/baseline.pack"
    with pack_path.open("wb") as stream:
        subprocess.run(
            ["git", "pack-objects", "--revs", "--stdout"],
            input=(baseline_tree + "\n").encode(),
            stdout=stream,
            stderr=subprocess.PIPE,
            check=True,
            cwd=repo,
            env=env,
        )
    subprocess.run(
        ["git", "index-pack", "-o", str(pack_path.with_suffix(".idx")), str(pack_path)],
        capture_output=True,
        check=True,
        cwd=repo,
        env=env,
    )
    return baseline_tree


def _empty_unborn_repositories(worktree: Path, env: dict[str, str]) -> list[str]:
    """Return untracked repository boundaries that contain no agent files."""
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z", "--"],
        capture_output=True,
        check=True,
        cwd=worktree,
        env=env,
    ).stdout
    clean_env = _alternate_index_env(Path(os.devnull))
    clean_env.pop("GIT_INDEX_FILE")
    empty: list[str] = []
    for encoded in untracked.split(b"\0"):
        relative = os.fsdecode(encoded)
        if not relative.endswith("/"):
            continue
        relative = relative[:-1]
        nested = worktree / relative
        metadata = nested / ".git"
        try:
            entries = list(nested.iterdir())
        except OSError:
            continue
        if nested.is_symlink() or metadata.is_symlink() or not metadata.is_dir() or entries != [metadata]:
            continue
        identity = subprocess.run(
            ["git", "-C", str(nested), "rev-parse", "--is-inside-work-tree", "--absolute-git-dir"],
            capture_output=True,
            text=True,
            check=False,
            env=clean_env,
        )
        if identity.returncode != 0 or identity.stdout.splitlines() != ["true", str(metadata.absolute())]:
            continue
        status = subprocess.run(
            ["git", "-C", str(nested), "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            capture_output=True,
            check=False,
            env=clean_env,
        )
        symbolic_head = subprocess.run(
            ["git", "-C", str(nested), "symbolic-ref", "-q", "HEAD"],
            capture_output=True,
            check=False,
            env=clean_env,
        )
        committed_head = subprocess.run(
            ["git", "-C", str(nested), "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            check=False,
            env=clean_env,
        )
        if (
            status.returncode == 0
            and not status.stdout
            and symbolic_head.returncode == 0
            and committed_head.returncode != 0
        ):
            empty.append(relative)
    return empty


def _extract_patch(repo: Path, index_path: Path, baseline_tree: str) -> str:
    """Return only changes made after the agent started, including untracked files."""
    # A removed task directory is an empty final worktree: emit its deletions,
    # without recreating it or treating a model action as a collector crash.
    with TemporaryDirectory(dir=index_path.parent) as empty:
        worktree = repo if repo.is_dir() else Path(empty)
        env = {
            **_alternate_index_env(index_path),
            "GIT_DIR": str(_baseline_git_dir(index_path)),
            "GIT_WORK_TREE": str(worktree),
        }
        empty_repositories = _empty_unborn_repositories(worktree, env)
        add = ["git", "add", "-A"]
        if empty_repositories:
            add.extend(["--", ".", *(f":(exclude,literal){path}" for path in empty_repositories)])
        subprocess.run(add, check=True, cwd=worktree, env=env)
        command = [
            "git",
            "-c",
            "core.quotePath=true",
            "diff",
            "--binary",
            "--no-color",
            "--no-ext-diff",
            "--no-textconv",
            "--cached",
            baseline_tree,
        ]
        patch = subprocess.run(command, capture_output=True, check=True, cwd=worktree, env=env).stdout
        try:
            # Decode without subprocess universal-newline translation.
            return patch.decode("utf-8")
        except UnicodeDecodeError:
            # Git may treat non-NUL compressed data as text. Emit binary patches
            # instead of replacing its bytes; the grader's existing API is UTF-8.
            attributes = _baseline_git_dir(index_path) / "info" / "attributes"
            previous = attributes.read_bytes() if attributes.exists() else None
            try:
                attributes.write_bytes((previous or b"") + b"\n* -diff\n")
                return subprocess.run(
                    command,
                    capture_output=True,
                    check=True,
                    cwd=worktree,
                    env=env,
                ).stdout.decode("utf-8")
            finally:
                if previous is None:
                    attributes.unlink()
                else:
                    attributes.write_bytes(previous)


def _agent_request(rollout_id: str | None) -> Request:
    path_params = {"rollout_id": rollout_id} if rollout_id else {}
    return Request({"type": "http", "path_params": path_params})


def _exception_response(exc: Exception, body):
    """Retain typed model transport failures without serializing request secrets."""
    from aiohttp import ClientConnectionError
    from openai import APIConnectionError

    from nemo_gym.openai_utils import NeMoGymResponse

    cause = exc
    seen = set()
    chain = []
    while id(cause) not in seen:
        seen.add(id(cause))
        chain.append(cause)
        nested = getattr(cause, "original_exception", None) or cause.__cause__
        if not isinstance(nested, Exception):
            break
        cause = nested
    status = next(
        (candidate.status_code for candidate in chain if isinstance(getattr(candidate, "status_code", None), int)),
        None,
    )
    transport = any(
        isinstance(getattr(candidate, "status_code", None), int)
        and (candidate.status_code in (401, 403, 404, 408, 429) or 500 <= candidate.status_code < 600)
        or isinstance(candidate, (APIConnectionError, ClientConnectionError, ConnectionError, TimeoutError))
        for candidate in chain
    )
    message = f"Harness exception: {type(cause).__name__}"
    if isinstance(status, int):
        message = f"Error code: {status}; {message}"
    return NeMoGymResponse(
        id="anyswe-runner-exception",
        created_at=0,
        model=body.model,
        object="response",
        output=[],
        tool_choice=body.tool_choice,
        tools=body.tools,
        parallel_tool_calls=body.parallel_tool_calls,
        status="failed",
        error={"code": "server_error", "message": message},
        metadata={"runner_exception": "true", "model_transport_error": "true" if transport else "false"},
    )


def _run_and_capture(agent, request, body, repo, index_path, baseline_tree, output_dir: Path) -> None:
    """Write response and patch even when a harness raises after editing files."""
    try:
        try:
            response = asyncio.run(agent.responses(request=request, body=body))
        except Exception as exc:
            response = _exception_response(exc, body)
        (output_dir / "response.json").write_text(response.model_dump_json())
        print(f"agent finished: {len(response.output)} output items", flush=True)
    finally:
        try:
            patch = _extract_patch(repo, index_path, baseline_tree) if repo and baseline_tree else ""
            (output_dir / "patch.diff").write_text(patch)
            if repo:
                print(f"patch: {len(patch)} chars from {repo}", flush=True)
        finally:
            index_path.unlink(missing_ok=True)
            if repo:
                shutil.rmtree(_baseline_git_dir(index_path))


def main() -> None:
    model_url = os.environ.get("NGSWE_MODEL_URL", "")
    model_name = os.environ["NGSWE_MODEL_NAME"]
    instruction = Path("/trajectories_mount/instruction.txt").read_text()
    agent_kwargs = _json_env("NGSWE_AGENT_KWARGS")
    sampling = _json_env("NGSWE_SAMPLING")

    task_cwd = Path.cwd()
    os.chdir(agent_deps_dir)
    from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
    from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_utils import ServerClient

    module = importlib.import_module(os.environ["NGSWE_AGENT_MODULE"])
    agent_class = getattr(module, os.environ["NGSWE_AGENT_CLASS"])
    config_class = getattr(module, os.environ["NGSWE_AGENT_CONFIG_CLASS"])
    os.chdir(task_cwd)

    server_name = "policy_model"
    global_config = (
        {server_name: {"responses_api_models": {"model": {"host": "0.0.0.0", "port": 0}}}} if model_url else {}
    )
    client = ServerClient.model_construct(global_config_dict=global_config)
    client._build_server_base_url = lambda config: model_url
    config_sampling = {key: value for key, value in sampling.items() if key in config_class.model_fields}
    model_server = ModelServerRef(name=server_name, type="responses_api_models") if model_url else None
    if "working_directory" in config_class.model_fields:
        agent_kwargs.setdefault("working_directory", str(Path.cwd()))
    config = config_class(
        host="0.0.0.0",
        port=0,
        name=agent_class.__name__.lower(),
        entrypoint="app.py",
        model_server=model_server,
        resources_server=ResourcesServerRef(name="anyswe", type="resources_servers"),
        **{**agent_kwargs, **config_sampling},
    )
    agent = agent_class(config=config, server_client=client)

    # Some published task images contain pre-existing working-tree changes. Snapshot that exact
    # state through an alternate index so the submitted patch contains only the agent's edits and
    # does not mutate the repository index the agent sees.
    repo = _find_repo(task_cwd)
    index_path = Path("/trajectories_mount/anyswe-baseline.index")
    baseline_tree = _snapshot_repo(repo, index_path) if repo else None

    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[NeMoGymEasyInputMessage(role="user", content=instruction)],
        model=model_name,
        **sampling,
    )
    request = _agent_request(os.environ.get("NGSWE_ROLLOUT_ID"))
    _run_and_capture(agent, request, body, repo, index_path, baseline_tree, Path("/trajectories_mount"))


if __name__ == "__main__":
    main()
