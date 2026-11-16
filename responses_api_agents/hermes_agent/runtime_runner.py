# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""One HermesAgent episode in a prepared Hermes runtime (``HermesAgentConfig.hermes_runtime``).

    <runtime>/bin/python -m responses_api_agents.hermes_agent.runtime_runner <job dir>

``<job dir>/request.json`` carries the agent config, the request body, the episode's model URL,
and the resources server URL and session cookie for request tools; ``response.json`` returns the
agent's response and, when asked, its observation bundle. One episode per process keeps Hermes'
process-global registry, ``HERMES_HOME`` and SIGTERM handling private to the episode.
"""

import asyncio
import json
import sys
from pathlib import Path

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.hermes_agent.app import HermesAgent, HermesAgentConfig


def main(job_dir: Path) -> None:
    job = json.loads((job_dir / "request.json").read_text(encoding="utf-8"))
    agent = HermesAgent(
        config=HermesAgentConfig.model_validate(job["config"]),
        server_client=ServerClient.model_construct(global_config_dict={}),
    )
    bundle = []
    response = asyncio.run(
        agent._run_hermes(
            NeMoGymResponseCreateParamsNonStreaming.model_validate(job["body"]),
            model_base_url=job["model_base_url"],
            observation_collector=bundle.append if job.get("observe") else None,
            cookie=job.get("cookie") or "",
            resources_url=job.get("resources_url"),
        )
    )
    reply = {
        "response": response.model_dump(mode="json"),
        "observations": bundle[0].model_dump(mode="json") if bundle else None,
    }
    (job_dir / "response.json").write_text(json.dumps(reply), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
