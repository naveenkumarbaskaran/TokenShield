"""Token estimation utilities."""

from __future__ import annotations

# Average chars per token for GPT-family tokenizers
CHARS_PER_TOKEN = 4.0

# Overhead per message (role, formatting tokens)
MESSAGE_OVERHEAD = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def estimate_message_tokens(messages: list[dict[str, str]]) -> int:
    """Estimate total tokens for a list of chat messages."""
    total = 0
    for msg in messages:
        total += MESSAGE_OVERHEAD
        total += estimate_tokens(msg.get("content", ""))
        total += estimate_tokens(msg.get("role", ""))
        if "name" in msg:
            total += estimate_tokens(msg["name"])
    total += 2  # priming tokens
    return total
