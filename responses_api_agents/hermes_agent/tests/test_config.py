# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
from pathlib import Path

import pytest
import yaml

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.hermes_agent.app import HermesAgent, HermesAgentConfig


def test_module_parses():
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    src = app_path.read_text()
    compile(src, str(app_path), "exec")


def test_config_yaml_parses():
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "hermes_agent.yaml"
    data = yaml.safe_load(cfg_path.read_text())
    assert "hermes_agent" in data
    inner = data["hermes_agent"]["responses_api_agents"]["hermes_agent"]
    assert inner["entrypoint"] == "app.py"
    assert inner["max_turns"] == 30
    assert inner["enabled_toolsets"] is None
    assert inner["system_prompt"] is None
    assert inner["terminal_backend"] == "local"
    assert inner["terminal_timeout"] == 60
    assert "disabled_toolsets" not in inner


def _config(**overrides) -> HermesAgentConfig:
    return HermesAgentConfig(
        host="127.0.0.1",
        port=0,
        name="hermes",
        entrypoint="app.py",
        model_server=ModelServerRef(type="responses_api_models", name="policy"),
        resources_server=ResourcesServerRef(type="resources_servers", name="resource"),
        model="fixture-policy",
        **overrides,
    )


@pytest.mark.parametrize(
    "mcp_servers",
    [None, {}, {"bench": {"command": "/opt/bench/bin/python", "args": ["-m", "bench_tools"], "env": {"A": "b"}}}],
)
def test_mcp_servers_reach_hermes_config_unchanged(tmp_path, monkeypatch, mcp_servers):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "unused"))
    agent = HermesAgent(config=_config(mcp_servers=mcp_servers), server_client=ServerClient.model_construct())
    generated = yaml.safe_load(agent._build_config())
    written = yaml.safe_load((Path(os.environ["HERMES_HOME"]) / "config.yaml").read_text())
    assert written == generated
    assert generated.get("mcp_servers") == (mcp_servers or None)


_SERVER = """
import asyncio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

server = Server("gym_probe")

@server.list_tools()
async def list_tools():
    return [types.Tool(name="probe_echo", description="echo", inputSchema={"type": "object"})]

@server.call_tool()
async def call_tool(name, arguments):
    return [types.TextContent(type="text", text="ok")]

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

asyncio.run(main())
"""


def test_mcp_server_tools_register_when_config_is_written_after_hermes_import(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    import sys

    from tools.registry import registry  # pyright: ignore[reportMissingImports]

    script = tmp_path / "server.py"
    script.write_text(_SERVER)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "unused"))
    HermesAgent(
        config=_config(mcp_servers={"gym_probe": {"command": sys.executable, "args": [str(script)]}}),
        server_client=ServerClient.model_construct(),
    )
    assert "mcp_gym_probe_probe_echo" in registry.get_all_tool_names()


if __name__ == "__main__":
    test_module_parses()
    test_config_yaml_parses()
    print("OK")
