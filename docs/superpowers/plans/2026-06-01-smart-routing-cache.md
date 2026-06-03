# Smart Routing + Response Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add task-aware model routing, response caching, and input compression to TokenShield so multi-agent systems automatically use the cheapest capable model and skip redundant LLM calls.

**Architecture:** Three new modules (`compressor.py`, `router.py`, `cache.py`) compose as an optional pipeline in `Shield.call()`. Each is independently usable and has no required external dependencies. Existing `Shield` behavior is unchanged when none are configured.

**Tech Stack:** Python 3.11+, stdlib only (`hashlib`, `json`, `pathlib`, `time`). No new required dependencies.

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `tokenshield/compressor.py` | History windowing, tool pruning, system prompt truncation |
| Create | `tokenshield/router.py` | Complexity classification → cheapest model selection |
| Create | `tokenshield/cache.py` | Exact-match response cache, MemoryBackend + DiskBackend |
| Modify | `tokenshield/shield.py` | Wire compressor → router → cache into `call()` |
| Modify | `tokenshield/__init__.py` | Export new public classes |
| Create | `tests/test_compressor.py` | Compressor unit tests |
| Create | `tests/test_router.py` | Router unit tests |
| Create | `tests/test_cache.py` | Cache unit tests |
| Modify | `tests/test_shield.py` | Integration tests for full pipeline |
| Create | `tokenshield/replay.py` | Record all calls in a session; replay against new config for savings estimate |
| Create | `tests/test_replay.py` | Replay unit + integration tests |

---

### Task 1: `Compressor` — input compression

**Files:**
- Create: `tokenshield/compressor.py`
- Create: `tests/test_compressor.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_compressor.py`:

```python
from tokenshield.compressor import Compressor


def _msgs(n: int) -> list[dict]:
    """Alternating user/assistant messages."""
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"msg {i}"})
    return msgs


def test_history_windowing_keeps_system():
    c = Compressor(max_history_turns=2)
    msgs = [{"role": "system", "content": "sys"}] + _msgs(10)
    result, _ = c.compress(msgs)
    assert result[0]["role"] == "system"
    # 2 turns = 4 messages (2 user + 2 assistant) + system
    assert len(result) == 5


def test_history_windowing_no_op_when_under_limit():
    c = Compressor(max_history_turns=20)
    msgs = _msgs(4)
    result, _ = c.compress(msgs)
    assert result == msgs


def test_tool_pruning():
    c = Compressor(max_tools=2)
    tools = [{"name": f"tool_{i}"} for i in range(5)]
    _, result_tools = c.compress([], tools)
    assert len(result_tools) == 2
    assert result_tools[0]["name"] == "tool_0"


def test_tool_pruning_no_op_when_under_limit():
    c = Compressor(max_tools=10)
    tools = [{"name": "t1"}, {"name": "t2"}]
    _, result_tools = c.compress([], tools)
    assert result_tools == tools


def test_system_prompt_truncation():
    c = Compressor(max_system_tokens=10)  # ~40 chars
    long_system = "x" * 500
    msgs = [{"role": "system", "content": long_system}]
    result, _ = c.compress(msgs)
    assert "[truncated]" in result[0]["content"]
    assert len(result[0]["content"]) < len(long_system)


def test_system_prompt_no_truncation_when_under_limit():
    c = Compressor(max_system_tokens=2000)
    msgs = [{"role": "system", "content": "short"}]
    result, _ = c.compress(msgs)
    assert result[0]["content"] == "short"


def test_none_tools_passthrough():
    c = Compressor()
    _, result_tools = c.compress([], None)
    assert result_tools is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/I572120/Documents/Area/WorkSpace/VScode/github-repos/TokenShield
pytest tests/test_compressor.py -v
```

Expected: `ModuleNotFoundError: No module named 'tokenshield.compressor'`

- [ ] **Step 3: Implement `tokenshield/compressor.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_compressor.py -v
```

Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add tokenshield/compressor.py tests/test_compressor.py
git commit -m "feat: add Compressor for history windowing, tool pruning, system truncation"
```

---

### Task 2: `CostRouter` — task-aware model routing

**Files:**
- Create: `tokenshield/router.py`
- Create: `tests/test_router.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_router.py`:

```python
from tokenshield.router import CostRouter


def _msg(content: str, role: str = "user") -> dict:
    return {"role": role, "content": content}


def test_simple_classification():
    r = CostRouter(simple="mini", medium="mid", complex="big")
    msgs = [_msg("hi")]
    assert r.route(msgs) == "mini"


def test_complex_by_token_count():
    r = CostRouter(simple="mini", medium="mid", complex="big")
    # >3000 estimated tokens = >12000 chars
    msgs = [_msg("x" * 13000)]
    assert r.route(msgs) == "big"


def test_complex_by_tool_count():
    r = CostRouter(simple="mini", medium="mid", complex="big")
    tools = [{"name": f"t{i}"} for i in range(6)]
    assert r.route([_msg("hi")], tools) == "big"


def test_medium_classification():
    r = CostRouter(simple="mini", medium="mid", complex="big")
    # 3 messages pushes past simple (>2 messages) but not complex
    msgs = [_msg("a"), _msg("b", "assistant"), _msg("c")]
    assert r.route(msgs) == "mid"


def test_medium_by_token_count():
    r = CostRouter(simple="mini", medium="mid", complex="big")
    # 600 tokens = ~2400 chars — above simple (<500) but below complex (>3000)
    msgs = [_msg("x" * 2500)]
    assert r.route(msgs) == "mid"


def test_explicit_model_override_skips_routing():
    r = CostRouter(simple="mini", medium="mid", complex="big")
    assert r.route([_msg("hi")], model_override="gpt-4o") == "gpt-4o"


def test_defaults():
    r = CostRouter()
    assert r.simple == "gpt-4o-mini"
    assert r.medium == "gpt-4o"
    assert r.complex == "gpt-4o"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_router.py -v
```

Expected: `ModuleNotFoundError: No module named 'tokenshield.router'`

- [ ] **Step 3: Implement `tokenshield/router.py`**

```python
"""Task-aware model routing — pick cheapest model by request complexity."""

from __future__ import annotations

import json
from dataclasses import dataclass

CHARS_PER_TOKEN = 4.0
SIMPLE_TOKEN_LIMIT = 500
COMPLEX_TOKEN_LIMIT = 3000
COMPLEX_TOOL_THRESHOLD = 5
SIMPLE_MAX_MESSAGES = 2


def _estimate(text: str) -> int:
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
        if model_override:
            return model_override

        tokens = _total_tokens(messages, tools)
        tool_count = len(tools) if tools else 0
        convo_msgs = [m for m in messages if m.get("role") != "system"]

        if tokens > COMPLEX_TOKEN_LIMIT or tool_count > COMPLEX_TOOL_THRESHOLD:
            return self.complex

        if (
            tokens <= SIMPLE_TOKEN_LIMIT
            and tool_count == 0
            and len(convo_msgs) <= SIMPLE_MAX_MESSAGES
        ):
            return self.simple

        return self.medium
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_router.py -v
```

Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add tokenshield/router.py tests/test_router.py
git commit -m "feat: add CostRouter for task-aware model selection"
```

---

### Task 3: `ResponseCache` — exact-match response cache

**Files:**
- Create: `tokenshield/cache.py`
- Create: `tests/test_cache.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cache.py`:

```python
import time
from tokenshield.cache import ResponseCache, MemoryBackend, DiskBackend
import tempfile, pathlib


def _msgs() -> list[dict]:
    return [{"role": "user", "content": "hello"}]


def test_cache_miss_returns_none():
    c = ResponseCache()
    assert c.get("gpt-4o", _msgs()) is None


def test_cache_hit_returns_response():
    c = ResponseCache()
    resp = {"text": "hi", "cost": 0.001}
    c.set("gpt-4o", _msgs(), resp)
    assert c.get("gpt-4o", _msgs()) == resp


def test_different_model_is_miss():
    c = ResponseCache()
    resp = {"text": "hi"}
    c.set("gpt-4o", _msgs(), resp)
    assert c.get("gpt-4o-mini", _msgs()) is None


def test_different_messages_is_miss():
    c = ResponseCache()
    c.set("gpt-4o", _msgs(), {"text": "hi"})
    other = [{"role": "user", "content": "bye"}]
    assert c.get("gpt-4o", other) is None


def test_ttl_expiry():
    c = ResponseCache(ttl_seconds=1)
    c.set("gpt-4o", _msgs(), {"text": "hi"})
    assert c.get("gpt-4o", _msgs()) is not None
    time.sleep(1.1)
    assert c.get("gpt-4o", _msgs()) is None


def test_ttl_zero_means_no_expiry():
    c = ResponseCache(ttl_seconds=0)
    c.set("gpt-4o", _msgs(), {"text": "hi"})
    assert c.get("gpt-4o", _msgs()) is not None


def test_disabled_cache_always_misses():
    c = ResponseCache(enabled=False)
    c.set("gpt-4o", _msgs(), {"text": "hi"})
    assert c.get("gpt-4o", _msgs()) is None


def test_disk_backend_persists(tmp_path):
    backend = DiskBackend(cache_dir=tmp_path)
    c = ResponseCache(backend=backend)
    c.set("gpt-4o", _msgs(), {"text": "persisted"})

    # New cache instance, same backend dir
    c2 = ResponseCache(backend=DiskBackend(cache_dir=tmp_path))
    assert c2.get("gpt-4o", _msgs()) == {"text": "persisted"}


def test_memory_backend_isolated():
    b1 = MemoryBackend()
    b2 = MemoryBackend()
    ResponseCache(backend=b1).set("gpt-4o", _msgs(), {"text": "a"})
    assert ResponseCache(backend=b2).get("gpt-4o", _msgs()) is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cache.py -v
```

Expected: `ModuleNotFoundError: No module named 'tokenshield.cache'`

- [ ] **Step 3: Implement `tokenshield/cache.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cache.py -v
```

Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add tokenshield/cache.py tests/test_cache.py
git commit -m "feat: add ResponseCache with MemoryBackend and DiskBackend"
```

---

### Task 4: Wire pipeline into `Shield`

**Files:**
- Modify: `tokenshield/shield.py`
- Modify: `tokenshield/__init__.py`
- Modify: `tests/test_shield.py`

- [ ] **Step 1: Write failing integration tests**

Add to `tests/test_shield.py`:

```python
from tokenshield.compressor import Compressor
from tokenshield.router import CostRouter
from tokenshield.cache import ResponseCache


def test_router_selects_cheap_model_for_simple_request():
    router = CostRouter(simple="gpt-4o-mini", medium="gpt-4o", complex="gpt-4o")
    shield = Shield(model="gpt-4o", router=router)
    result = shield.call(messages=[{"role": "user", "content": "hi"}])
    assert result["routed_model"] == "gpt-4o-mini"


def test_cache_hit_returns_cached_response():
    cache = ResponseCache()
    shield = Shield(model="gpt-4o", cache=cache)
    msgs = [{"role": "user", "content": "what is 2+2"}]
    shield.call(messages=msgs)
    result2 = shield.call(messages=msgs)
    assert result2["cache_hit"] is True


def test_cache_miss_on_first_call():
    cache = ResponseCache()
    shield = Shield(model="gpt-4o", cache=cache)
    result = shield.call(messages=[{"role": "user", "content": "unique xyz"}])
    assert result["cache_hit"] is False


def test_skip_cache_forces_re_execution():
    cache = ResponseCache()
    shield = Shield(model="gpt-4o", cache=cache)
    msgs = [{"role": "user", "content": "cached question"}]
    shield.call(messages=msgs)
    result2 = shield.call(messages=msgs, skip_cache=True)
    assert result2["cache_hit"] is False


def test_compressor_reduces_history():
    compressor = Compressor(max_history_turns=1)
    shield = Shield(model="gpt-4o", compressor=compressor)
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
        {"role": "user", "content": "e"},
    ]
    # Should not raise; compressor runs silently
    result = shield.call(messages=msgs)
    assert "cost" in result


def test_cache_hit_does_not_track_cost():
    cache = ResponseCache()
    shield = Shield(model="gpt-4o", cache=cache)
    msgs = [{"role": "user", "content": "cost check"}]
    shield.call(messages=msgs)
    cost_before = shield.tracker.cost_today
    shield.call(messages=msgs)  # cache hit
    assert shield.tracker.cost_today == cost_before


def test_no_router_uses_explicit_model():
    shield = Shield(model="gpt-4o")
    result = shield.call(messages=[{"role": "user", "content": "hi"}])
    assert result["routed_model"] == "gpt-4o"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_shield.py -v -k "routed_model or cache_hit or skip_cache or compressor or cost_today"
```

Expected: FAIL — `routed_model` key not in result dict

- [ ] **Step 3: Update `tokenshield/shield.py`**

Replace the `Shield` dataclass definition and `call()` method:

```python
"""Core Shield class — wraps LLM calls with cost tracking and budget enforcement."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from tokenshield.budget import BudgetPolicy
from tokenshield.tracker import CostTracker, RequestRecord
from tokenshield.pricing import PricingDB
from tokenshield.estimator import estimate_tokens, estimate_message_tokens


class BudgetExceeded(Exception):
    def __init__(self, estimated_cost: float, limit: float, limit_type: str):
        self.estimated_cost = estimated_cost
        self.limit = limit
        self.limit_type = limit_type
        super().__init__(
            f"Budget exceeded: estimated ${estimated_cost:.4f} "
            f"would breach {limit_type} limit of ${limit:.2f}"
        )


@dataclass
class Shield:
    model: str = "gpt-4o"
    policy: BudgetPolicy = field(default_factory=BudgetPolicy)
    tracker: CostTracker = field(default_factory=CostTracker)
    pricing: PricingDB = field(default_factory=PricingDB)
    on_alert: Callable[[str], None] | None = None
    router: Any | None = None      # CostRouter instance
    cache: Any | None = None       # ResponseCache instance
    compressor: Any | None = None  # Compressor instance

    BudgetExceeded = BudgetExceeded

    def call(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        model: str | None = None,
        dry_run: bool = False,
        skip_cache: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # ── Compress ─────────────────────────────────────────────
        if self.compressor is not None:
            messages, tools = self.compressor.compress(messages, tools)

        # ── Route ────────────────────────────────────────────────
        if self.router is not None:
            active_model = self.router.route(messages, tools, model_override=model)
        else:
            active_model = model or self.model

        prices = self.pricing.get(active_model)

        # ── Pre-flight estimation ────────────────────────────────
        input_tokens = estimate_message_tokens(messages)
        if tools:
            import json
            tool_text = json.dumps(tools, separators=(",", ":"))
            input_tokens += estimate_tokens(tool_text)

        estimated_output = max(100, input_tokens // 4)
        estimated_cost = (
            input_tokens * prices["input"] + estimated_output * prices["output"]
        ) / 1_000_000

        # ── Budget gate ──────────────────────────────────────────
        self._check_budget(estimated_cost)

        if dry_run:
            return {
                "estimated_input_tokens": input_tokens,
                "estimated_output_tokens": estimated_output,
                "estimated_cost": estimated_cost,
                "model": active_model,
                "routed_model": active_model,
                "cache_hit": False,
                "blocked": False,
            }

        # ── Cache check ──────────────────────────────────────────
        if self.cache is not None and not skip_cache:
            cached = self.cache.get(active_model, messages)
            if cached is not None:
                return {**cached, "cache_hit": True, "routed_model": active_model}

        # ── Execute (simulated — real impl delegates to litellm) ─
        start = time.monotonic()
        actual_input = input_tokens
        actual_output = estimated_output
        duration_ms = int((time.monotonic() - start) * 1000)

        actual_cost = (
            actual_input * prices["input"] + actual_output * prices["output"]
        ) / 1_000_000

        # ── Record ───────────────────────────────────────────────
        record = RequestRecord(
            model=active_model,
            input_tokens=actual_input,
            output_tokens=actual_output,
            cost=actual_cost,
            duration_ms=duration_ms,
            timestamp=time.time(),
        )
        self.tracker.record(record)
        self._check_alerts()

        response = {
            "input_tokens": actual_input,
            "output_tokens": actual_output,
            "cost": actual_cost,
            "model": active_model,
            "routed_model": active_model,
            "cache_hit": False,
        }

        # ── Cache store ──────────────────────────────────────────
        if self.cache is not None:
            self.cache.set(active_model, messages, response)

        return response

    def _check_budget(self, estimated_cost: float) -> None:
        if self.policy.max_cost_per_request and estimated_cost > self.policy.max_cost_per_request:
            raise BudgetExceeded(estimated_cost, self.policy.max_cost_per_request, "per-request")
        if self.policy.max_cost_per_hour:
            hour_cost = self.tracker.cost_last_hour + estimated_cost
            if hour_cost > self.policy.max_cost_per_hour:
                raise BudgetExceeded(hour_cost, self.policy.max_cost_per_hour, "per-hour")
        if self.policy.max_cost_per_day:
            day_cost = self.tracker.cost_today + estimated_cost
            if day_cost > self.policy.max_cost_per_day:
                raise BudgetExceeded(day_cost, self.policy.max_cost_per_day, "per-day")

    def _check_alerts(self) -> None:
        if not self.on_alert or not self.policy.alert_threshold_pct:
            return
        threshold = self.policy.alert_threshold_pct / 100.0
        if self.policy.max_cost_per_day:
            pct = self.tracker.cost_today / self.policy.max_cost_per_day
            if pct >= threshold:
                self.on_alert(
                    f"⚠️ TokenShield: Daily spend at {pct:.0%} "
                    f"(${self.tracker.cost_today:.2f} / ${self.policy.max_cost_per_day:.2f})"
                )

    def optimize(self, messages: list[dict], tools: list[dict] | None = None) -> list[str]:
        suggestions = []
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        if sys_msgs:
            sys_tokens = estimate_message_tokens(sys_msgs)
            total_tokens = estimate_message_tokens(messages)
            if total_tokens > 0 and sys_tokens / total_tokens > 0.5:
                suggestions.append(
                    f"System prompt is {sys_tokens:,} tokens "
                    f"({sys_tokens / total_tokens:.0%} of input). Consider compressing."
                )
        if tools and len(tools) > 10:
            est_tool_tokens = len(tools) * 150
            suggestions.append(
                f"{len(tools)} tools bound (~{est_tool_tokens:,} tokens). "
                f"Use dynamic tool binding to reduce."
            )
        user_msgs = [m for m in messages if m.get("role") in ("user", "assistant")]
        if len(user_msgs) > 30:
            suggestions.append(
                f"History has {len(user_msgs)} messages. Consider windowing to last 20."
            )
        return suggestions

    def report(self) -> str:
        t = self.tracker
        lines = [
            "┌─────────────────────────────────┐",
            f"│ Requests today:     {t.request_count_today:<11}│",
            f"│ Tokens (in/out):    {t.total_input_today // 1000}K / {t.total_output_today // 1000}K{' ' * max(0, 5 - len(str(t.total_output_today // 1000)))}│",
            f"│ Cost today:         ${t.cost_today:<10.2f} │",
        ]
        if self.policy.max_cost_per_day:
            remaining = max(0, self.policy.max_cost_per_day - t.cost_today)
            lines.append(f"│ Budget remaining:   ${remaining:<10.2f} │")
        if t.request_count_today > 0:
            avg = t.cost_today / t.request_count_today
            lines.append(f"│ Avg cost/request:   ${avg:<10.3f} │")
        lines.append("└─────────────────────────────────┘")
        return "\n".join(lines)
```

- [ ] **Step 4: Update `tokenshield/__init__.py`**

```python
"""TokenShield — LLM cost monitoring, routing, caching, and budget enforcement."""

from tokenshield.shield import Shield
from tokenshield.budget import BudgetPolicy
from tokenshield.tracker import CostTracker
from tokenshield.pricing import PricingDB
from tokenshield.router import CostRouter
from tokenshield.cache import ResponseCache, MemoryBackend, DiskBackend
from tokenshield.compressor import Compressor

__version__ = "2.1.0"
__all__ = [
    "Shield",
    "BudgetPolicy",
    "CostTracker",
    "PricingDB",
    "CostRouter",
    "ResponseCache",
    "MemoryBackend",
    "DiskBackend",
    "Compressor",
]
```

- [ ] **Step 5: Run all tests**

```bash
pytest tests/ -v
```

Expected: all tests pass (existing + new integration tests)

- [ ] **Step 6: Commit**

```bash
git add tokenshield/shield.py tokenshield/__init__.py tests/test_shield.py
git commit -m "feat: wire Compressor, CostRouter, ResponseCache into Shield.call() pipeline"
```

---

### Task 5: `CostReplay` — Agent Cost Replay

**Files:**
- Create: `tokenshield/replay.py`
- Create: `tests/test_replay.py`
- Modify: `tokenshield/shield.py` (add `replay()` method and `_replay_log`)
- Modify: `tokenshield/__init__.py` (export `CostReplay`)

- [ ] **Step 1: Write failing tests**

Create `tests/test_replay.py`:

```python
from tokenshield.shield import Shield
from tokenshield.router import CostRouter
from tokenshield.replay import CostReplay


def _call(shield, content="hello"):
    return shield.call(messages=[{"role": "user", "content": content}])


def test_replay_log_records_calls():
    shield = Shield(model="gpt-4o")
    _call(shield, "a")
    _call(shield, "b")
    assert len(shield._replay_log) == 2


def test_replay_log_entry_structure():
    shield = Shield(model="gpt-4o")
    _call(shield, "test")
    entry = shield._replay_log[0]
    assert "messages" in entry
    assert "tools" in entry
    assert "model_used" in entry
    assert "input_tokens" in entry
    assert "output_tokens" in entry
    assert "cost" in entry


def test_replay_with_cheaper_router_shows_savings():
    shield = Shield(model="gpt-4o")
    for _ in range(3):
        _call(shield, "hi")  # simple → routes to gpt-4o at full price

    cheaper_router = CostRouter(simple="gpt-4o-mini", medium="gpt-4o", complex="gpt-4o")
    report = shield.replay(router=cheaper_router)

    assert report["original_cost"] > 0
    assert report["replayed_cost"] < report["original_cost"]
    assert report["savings"] == round(report["original_cost"] - report["replayed_cost"], 6)
    assert 0 < report["savings_pct"] <= 100
    assert report["call_count"] == 3


def test_replay_same_config_zero_savings():
    shield = Shield(model="gpt-4o")
    _call(shield, "hi")
    report = shield.replay()  # no router override — same model
    assert report["savings"] == 0.0
    assert report["savings_pct"] == 0.0


def test_replay_empty_log():
    shield = Shield(model="gpt-4o")
    report = shield.replay()
    assert report["call_count"] == 0
    assert report["original_cost"] == 0.0
    assert report["replayed_cost"] == 0.0


def test_replay_does_not_modify_log():
    shield = Shield(model="gpt-4o")
    _call(shield, "x")
    _call(shield, "y")
    shield.replay()
    assert len(shield._replay_log) == 2


def test_cost_replay_standalone():
    shield = Shield(model="gpt-4o")
    for _ in range(5):
        _call(shield, "hi")
    replay = CostReplay(shield)
    report = replay.run(router=CostRouter(simple="gpt-4o-mini", medium="gpt-4o", complex="gpt-4o"))
    assert report["call_count"] == 5
    assert "savings_pct" in report
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/I572120/Documents/Area/WorkSpace/VScode/github-repos/TokenShield
pytest tests/test_replay.py -v
```

Expected: `ModuleNotFoundError: No module named 'tokenshield.replay'` or `AttributeError: _replay_log`

- [ ] **Step 3: Implement `tokenshield/replay.py`**

```python
"""Agent Cost Replay — simulate a recorded session against a new routing config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CostReplay:
    """Replay a recorded Shield session against a new config to estimate savings."""

    shield: Any  # Shield instance

    def run(
        self,
        router: Any | None = None,
        model: str | None = None,
    ) -> dict:
        """
        Simulate all recorded calls with a new router/model and return savings report.

        Args:
            router: CostRouter to use for replay. If None, uses shield's default model.
            model: Explicit model override for all replayed calls.

        Returns:
            dict with original_cost, replayed_cost, savings, savings_pct, call_count,
            and per_model breakdown of replayed calls.
        """
        log = self.shield._replay_log
        if not log:
            return {
                "call_count": 0,
                "original_cost": 0.0,
                "replayed_cost": 0.0,
                "savings": 0.0,
                "savings_pct": 0.0,
                "per_model": {},
            }

        pricing = self.shield.pricing
        original_cost = sum(e["cost"] for e in log)
        replayed_cost = 0.0
        per_model: dict[str, int] = {}

        for entry in log:
            msgs = entry["messages"]
            tools = entry.get("tools")

            if router is not None:
                replayed_model = router.route(msgs, tools)
            elif model is not None:
                replayed_model = model
            else:
                replayed_model = entry["model_used"]

            prices = pricing.get(replayed_model)
            cost = (
                entry["input_tokens"] * prices["input"]
                + entry["output_tokens"] * prices["output"]
            ) / 1_000_000
            replayed_cost += cost
            per_model[replayed_model] = per_model.get(replayed_model, 0) + 1

        savings = round(original_cost - replayed_cost, 6)
        savings_pct = round((savings / original_cost) * 100, 2) if original_cost > 0 else 0.0

        return {
            "call_count": len(log),
            "original_cost": round(original_cost, 6),
            "replayed_cost": round(replayed_cost, 6),
            "savings": max(savings, 0.0),
            "savings_pct": max(savings_pct, 0.0),
            "per_model": per_model,
        }
```

- [ ] **Step 4: Add `_replay_log` and `replay()` to `tokenshield/shield.py`**

Add `_replay_log` field to `Shield` dataclass (after the `compressor` field):

```python
    _replay_log: list[dict] = field(default_factory=list, repr=False)
```

In `Shield.call()`, after the `self.tracker.record(record)` line, add:

```python
        self._replay_log.append({
            "messages": messages,
            "tools": tools,
            "model_used": active_model,
            "input_tokens": actual_input,
            "output_tokens": actual_output,
            "cost": actual_cost,
        })
```

Add `replay()` method to `Shield` after `report()`:

```python
    def replay(
        self,
        router: Any | None = None,
        model: str | None = None,
    ) -> dict:
        """Simulate recorded calls with a new router/model. Returns savings report."""
        from tokenshield.replay import CostReplay
        return CostReplay(self).run(router=router, model=model)
```

- [ ] **Step 5: Export `CostReplay` in `tokenshield/__init__.py`**

Add to imports and `__all__`:

```python
from tokenshield.replay import CostReplay
```

```python
__all__ = [
    "Shield",
    "BudgetPolicy",
    "CostTracker",
    "PricingDB",
    "CostRouter",
    "ResponseCache",
    "MemoryBackend",
    "DiskBackend",
    "Compressor",
    "CostReplay",
]
```

- [ ] **Step 6: Run all tests**

```bash
pytest tests/ -v
```

Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add tokenshield/replay.py tokenshield/shield.py tokenshield/__init__.py tests/test_replay.py
git commit -m "feat: add CostReplay for session cost simulation against new routing config"
```

---

### Task 6: Full test suite + push

- [ ] **Step 1: Run full suite with coverage**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests pass

- [ ] **Step 2: Commit and push**

```bash
git push origin main
```
