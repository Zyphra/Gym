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
from typing import Any

import httpx
from harbor.llms.base import ContextLengthExceededError

from responses_api_agents.harbor_agent.custom_agents.llms.nemo_gym_llm import (
    _CONTEXT_LENGTH_ERROR_PHRASES,
    _THINK_CLOSE,
    _THINK_OPEN,
)
from responses_api_agents.harbor_agent.custom_agents.llms.nemo_gym_llm import (
    NemoGymLLM as HarborNemoGymLLM,
)


class NemoGymLLM(HarborNemoGymLLM):
    """Standalone Terminus dialect over the shared Gym LLM implementation."""

    model_transport_error: str | None = None

    def _parse_response_content(self, message: Any) -> tuple[Any, Any]:
        content = message.get("content", "") if isinstance(message, dict) else ""
        if content is None:
            content = ""
        reasoning_content = message.get("reasoning_content") if isinstance(message, dict) else None

        # Raw reasoning is never executable, including unclosed think blocks.
        # The generation prompt may supply the opener, so honor a close alone.
        if isinstance(content, str):
            if _THINK_CLOSE in content:
                tagged_reasoning, content = content.rsplit(_THINK_CLOSE, 1)
                if _THINK_OPEN in content:
                    tagged_reasoning += _THINK_CLOSE + content
                    content = ""
                if reasoning_content is None:
                    reasoning_content = tagged_reasoning.replace(_THINK_OPEN, "").replace(_THINK_CLOSE, "")
            elif _THINK_OPEN in content:
                if reasoning_content is None:
                    reasoning_content = content.split(_THINK_OPEN, 1)[1]
                content = ""

        return content, reasoning_content

    async def _post_chat_completions(
        self, payload: dict[str, Any], timeout_sec: float | None = None
    ) -> dict[str, Any]:
        endpoint = self._chat_completions_endpoint()
        timeout = timeout_sec if timeout_sec is not None else self._timeout_sec
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=timeout)
        self.model_transport_error = None
        try:
            response = await self._http_client.post(endpoint, json=payload, timeout=timeout)
        except httpx.RequestError as error:
            self.model_transport_error = f"{type(error).__name__}: {error}"[:2000]
            raise

        if response.status_code >= 400:
            error_text = response.text.lower()
            if any(phrase in error_text for phrase in _CONTEXT_LENGTH_ERROR_PHRASES):
                self.context_length_exceeded = True
                raise ContextLengthExceededError(f"Model {self._model_name} context length exceeded: {response.text}")
            if response.status_code in {401, 403, 404, 408, 429} or 500 <= response.status_code < 600:
                self.model_transport_error = f"Model endpoint HTTP {response.status_code}"
            response.raise_for_status()

        return response.json()
