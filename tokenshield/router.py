"""Task-aware model routing — pick cheapest model by request complexity."""

from __future__ import annotations

import json
from dataclasses import dataclass

CHARS_PER_TOKEN = 4.0
SIMPLE_TOKEN_LIMIT = 500
COMPLEX_TOKEN_LIMIT = 3000
COMPLEX_TOOL_THRESHOLD = 5
SIMPLE_MAX_MESSAGES = 2


def _estimate(text: str | list) -> int:
    if isinstance(text, list):
        # sum text from content blocks
        text = " ".join(b.get("text", "") for b in text if isinstance(b, dict))
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def _total_tokens(messages: list[dict], tools: list[dict] | None) -> int:
    total = sum(_estimate(m.get("content", "")) for m in messages)
    if tools:
        total += _estimate(json.dumps(tools, separators=(",", ":")))
    return total


@dataclass
class CostRouter:
    simple: str = "gpt-4o-mini"
    medium: str = "gpt-4o"
    complex: str = "gpt-4o"

    def route(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model_override: str | None = None,
    ) -> str:
        if model_override is not None:
            return model_override

        tokens = _total_tokens(messages, tools)
        tool_count = len(tools) if tools else 0
        convo_msgs = [m for m in messages if m.get("role") != "system"]

        if tokens > COMPLEX_TOKEN_LIMIT or tool_count > COMPLEX_TOOL_THRESHOLD:
            return self.complex

        if (
            tokens < SIMPLE_TOKEN_LIMIT
            and tool_count == 0
            and len(convo_msgs) <= SIMPLE_MAX_MESSAGES
        ):
            return self.simple

        return self.medium
