# Smart Routing + Response Cache — Design Spec

**Date:** 2026-06-01
**Status:** Approved

---

## Problem

TokenShield tracks and gates costs but doesn't reduce them at the source. Every request goes to the same model regardless of complexity, and identical (or near-identical) requests are re-executed every time. For multi-agent systems running hundreds of requests per session, this is the biggest remaining cost lever.

---

## Goals

- Route requests to the cheapest model capable of handling the task
- Cache responses to skip redundant LLM calls entirely
- Compress inputs (history windowing, tool pruning, system prompt truncation) before routing
- Compose cleanly with existing `Shield` — no breaking changes
- No new required dependencies (optional extras for Redis/semantic cache)

---

## Architecture

### Full pipeline

```
request
  → Compressor   (trim history, tools, system prompt)
  → CostRouter   (classify complexity → pick cheapest model)
  → ResponseCache (exact-match hit → return immediately)
  → LLM call
  → ResponseCache (store result)
  → CostTracker
```

### New files

| File | Responsibility |
|------|---------------|
| `tokenshield/compressor.py` | Compress messages before send: window history, prune tools, truncate system prompt |
| `tokenshield/router.py` | Classify request complexity, map to cheapest capable model |
| `tokenshield/cache.py` | Exact-match response cache; pluggable backend (memory / disk / Redis) |

### Modified files

| File | Change |
|------|--------|
| `tokenshield/shield.py` | Accept `router`, `cache`, `compressor` params; wire into `call()` pipeline |
| `tokenshield/__init__.py` | Export `CostRouter`, `ResponseCache`, `Compressor` |

---

## Component Specs

### `Compressor`

Applies three transformations in order before a request is sent:

1. **History windowing** — keep only the last N `user`/`assistant` turns (default: 20). System messages always kept.
2. **Tool pruning** — if tools list exceeds `max_tools` (default: 10), truncate to first N tools and log a warning.
3. **System prompt truncation** — if system message exceeds `max_system_tokens` (default: 2000 estimated tokens), truncate to that limit with a `[truncated]` marker.

```python
@dataclass
class Compressor:
    max_history_turns: int = 20       # user+assistant pairs to keep
    max_tools: int = 10               # max tool schemas to send
    max_system_tokens: int = 2000     # estimated token limit for system prompt

    def compress(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> tuple[list[dict], list[dict] | None]:
        ...  # returns (compressed_messages, compressed_tools)
```

### `CostRouter`

Classifies each request as `simple`, `medium`, or `complex` using a fast heuristic (no LLM call):

- **simple**: total estimated input tokens < 500 AND no tools AND ≤ 2 messages
- **complex**: total estimated input tokens > 3000 OR tools count > 5 OR any message > 1000 tokens
- **medium**: everything else

Maps complexity to user-configured model tiers:

```python
@dataclass
class CostRouter:
    simple: str = "gpt-4o-mini"
    medium: str = "gpt-4o"
    complex: str = "gpt-4o"

    def route(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        ...  # returns model name
```

### `ResponseCache`

Exact-match cache keyed by `sha256(model + json(messages))`. Pluggable backend.

```python
class CacheBackend(Protocol):
    def get(self, key: str) -> dict | None: ...
    def set(self, key: str, value: dict) -> None: ...

@dataclass
class ResponseCache:
    backend: CacheBackend = field(default_factory=MemoryBackend)
    ttl_seconds: int = 3600    # 0 = no expiry
    enabled: bool = True

    def get(self, model: str, messages: list[dict]) -> dict | None: ...
    def set(self, model: str, messages: list[dict], response: dict) -> None: ...
```

Built-in backends: `MemoryBackend` (default, in-process dict), `DiskBackend` (JSON files in `~/.tokenshield/cache/`).

### `Shield.call()` updated signature

```python
def call(
    self,
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str | None = None,   # explicit override — skips router
    dry_run: bool = False,
    skip_cache: bool = False,   # force re-execution even on cache hit
    **kwargs,
) -> dict:
```

Response dict gains two new fields:
- `"cache_hit": bool` — whether response came from cache
- `"routed_model": str` — model selected by router (may differ from `model` field if routing was used)

---

## Usage Examples

### Drop-in with routing + cache

```python
from tokenshield import Shield, BudgetPolicy, CostRouter, ResponseCache

shield = Shield(
    model="gpt-4o",
    policy=BudgetPolicy(max_cost_per_day=20.00),
    router=CostRouter(simple="gpt-4o-mini", medium="gpt-4o", complex="claude-3-opus"),
    cache=ResponseCache(ttl_seconds=3600),
)

result = shield.call(messages=[{"role": "user", "content": "What is 2+2?"}])
# routes to gpt-4o-mini (simple), caches result

result2 = shield.call(messages=[{"role": "user", "content": "What is 2+2?"}])
# cache_hit=True, $0 cost
```

### Compression only

```python
from tokenshield import Shield
from tokenshield.compressor import Compressor

shield = Shield(
    model="gpt-4o",
    compressor=Compressor(max_history_turns=10, max_tools=5),
)
```

### Disk cache (survives restarts)

```python
from tokenshield.cache import ResponseCache, DiskBackend

shield = Shield(
    model="gpt-4o",
    cache=ResponseCache(backend=DiskBackend(), ttl_seconds=86400),
)
```

---

## Testing

- `tests/test_compressor.py` — history windowing, tool pruning, system prompt truncation, passthrough when under limits
- `tests/test_router.py` — classification thresholds, model mapping, explicit override
- `tests/test_cache.py` — cache hit/miss, TTL expiry, MemoryBackend, DiskBackend
- `tests/test_shield.py` (extend) — full pipeline: compress → route → cache hit → cache miss → cost tracked

---

## Non-goals

- No semantic/embedding-based similarity cache (out of scope — adds heavy dependency)
- No Redis backend (can be added later as optional extra)
- No ML-based complexity classifier (heuristic is sufficient and zero-cost)
