"""Input compression — reduce tokens before sending to LLM."""

from __future__ import annotations

from dataclasses import dataclass

CHARS_PER_TOKEN = 4.0


def _estimate(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN))


@dataclass
class Compressor:
    max_history_turns: int = 20
    max_tools: int = 10
    max_system_tokens: int = 2000

    def compress(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> tuple[list[dict], list[dict] | None]:
        messages = self._window_history(messages)
        messages = self._truncate_system(messages)
        tools = self._prune_tools(tools)
        return messages, tools

    def _window_history(self, messages: list[dict]) -> list[dict]:
        system = [m for m in messages if m.get("role") == "system"]
        convo = [m for m in messages if m.get("role") != "system"]
        # each turn = one user + one assistant message
        keep = self.max_history_turns * 2
        return system + convo[-keep:] if len(convo) > keep else system + convo

    def _truncate_system(self, messages: list[dict]) -> list[dict]:
        result = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if _estimate(content) > self.max_system_tokens:
                    limit = self.max_system_tokens * int(CHARS_PER_TOKEN)
                    msg = {**msg, "content": content[:limit] + " [truncated]"}
            result.append(msg)
        return result

    def _prune_tools(self, tools: list[dict] | None) -> list[dict] | None:
        if tools is None:
            return None
        return tools[: self.max_tools]
