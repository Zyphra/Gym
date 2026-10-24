# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
from omegaconf import OmegaConf

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.hermes_agent.app import HermesAgent, HermesAgentConfig, _bind_policy_summary


def test_native_summary_reuses_the_main_client_model_and_rollout_url(monkeypatch, record_property, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    import run_agent

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            text = "Main fixture response." if len(requests) == 1 else "Verified fixture summary."
            payload = json.dumps(
                {
                    "id": f"fixture-{len(requests)}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": body["model"],
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original = run_agent.AIAgent
    native_agents = []

    def record_native_agent(**kwargs):
        native = original(**kwargs)
        native_agents.append(native)
        return native

    monkeypatch.setattr(run_agent, "AIAgent", record_native_agent)
    try:
        client = ServerClient.model_construct(
            global_config_dict=OmegaConf.create(
                {
                    "policy": {
                        "responses_api_models": {"openai_model": {"host": "127.0.0.1", "port": server.server_port}}
                    },
                }
            )
        )
        client._build_server_base_url = lambda _cfg: f"http://127.0.0.1:{server.server_port}"
        app = HermesAgent(
            config=HermesAgentConfig(
                host="127.0.0.1",
                port=0,
                name="hermes",
                entrypoint="app.py",
                model_server=ModelServerRef(type="responses_api_models", name="policy"),
                resources_server=ResourcesServerRef(type="resources_servers", name="resource"),
                model="fixture-policy",
                api_key="local-control",
                max_turns=2,
                max_tokens=8192,
                context_window=32768,
                enabled_toolsets=[],
                compression_enabled=True,
                terminal_backend="local",
            ),
            server_client=client,
        )

        async def exercise():
            await app._create_response(
                NeMoGymResponseCreateParamsNonStreaming(input="Return the local fixture response."),
                rollout_id="summary-session",
            )
            return await asyncio.to_thread(
                native_agents[0].context_compressor._generate_summary,
                [{"role": "user", "content": "The local fixture was checked."}],
            )

        summary = asyncio.run(exercise())
        compressor = native_agents[0].context_compressor
        assert compressor.context_length == 32768
        assert compressor.threshold_tokens == int((32768 - 8192) * 0.85)
        assert "Verified fixture summary." in summary
        assert len(requests) == 2
        assert [path for path, _ in requests] == ["/ng-rollout/summary-session/v1/chat/completions"] * 2
        assert [body["model"] for _, body in requests] == ["fixture-policy"] * 2
        assert requests[1][1]["tool_choice"] == "none"
        assert requests[1][1]["max_tokens"] <= 8192
        record_property("native_runtime", str(Path(run_agent.__file__).resolve()))
        record_property(
            "captured_routes", json.dumps([{"path": path, "model": body["model"]} for path, body in requests])
        )
    finally:
        for native in native_agents:
            native.client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_older_native_compressor_refuses_before_any_policy_call():
    from types import SimpleNamespace

    with pytest.raises(RuntimeError, match="managed summary callback"):
        _bind_policy_summary(SimpleNamespace(context_compressor=SimpleNamespace()))
