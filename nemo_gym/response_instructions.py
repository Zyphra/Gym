# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional


def fold_instructions_into_messages(
    instructions: Optional[str], messages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Represent Responses instructions and leading system/developer text as one system message."""
    if not instructions:
        return messages
    parts = [instructions]
    index = 0
    while index < len(messages) and messages[index].get("role") in ("system", "developer"):
        content = messages[index].get("content")
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.append("".join(part.get("text", "") for part in content or [] if isinstance(part, dict)))
        index += 1
    return [{"role": "system", "content": "\n\n".join(parts)}, *messages[index:]]
