"""Exact-match response cache with pluggable backends."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class CacheBackend(Protocol):
    def get(self, key: str) -> dict | None: ...
    def set(self, key: str, value: dict) -> None: ...


@dataclass
class MemoryBackend:
    _store: dict[str, dict] = field(default_factory=dict)

    def get(self, key: str) -> dict | None:
        return self._store.get(key)

    def set(self, key: str, value: dict) -> None:
        self._store[key] = value


@dataclass
class DiskBackend:
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".tokenshield" / "cache")

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def set(self, key: str, value: dict) -> None:
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(value))
        tmp.replace(self._path(key))


def _cache_key(model: str, messages: list[dict]) -> str:
    payload = json.dumps({"model": model, "messages": messages}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class ResponseCache:
    backend: CacheBackend = field(default_factory=MemoryBackend)
    ttl_seconds: int = 3600
    enabled: bool = True

    def get(self, model: str, messages: list[dict]) -> dict | None:
        if not self.enabled:
            return None
        key = _cache_key(model, messages)
        entry = self.backend.get(key)
        if entry is None:
            return None
        if self.ttl_seconds > 0:
            age = time.time() - entry.get("_cached_at", 0)
            if age > self.ttl_seconds:
                return None
        return {k: v for k, v in entry.items() if k != "_cached_at"}

    def set(self, model: str, messages: list[dict], response: dict) -> None:
        if not self.enabled:
            return
        key = _cache_key(model, messages)
        self.backend.set(key, {**response, "_cached_at": time.time()})
