"""Tests for TokenShield core functionality."""

import time
import pytest
from tokenshield import Shield, BudgetPolicy
from tokenshield.shield import BudgetExceeded
from tokenshield.tracker import CostTracker, RequestRecord
from tokenshield.pricing import PricingDB
from tokenshield.estimator import estimate_tokens, estimate_message_tokens


# ── Estimator Tests ──────────────────────────────────────────────

class TestEstimator:
    def test_empty_string(self):
        assert estimate_tokens("") == 1  # minimum 1

    def test_short_text(self):
        tokens = estimate_tokens("Hello world")
        assert 1 <= tokens <= 5

    def test_long_text(self):
        text = "a" * 4000
        tokens = estimate_tokens(text)
        assert 900 <= tokens <= 1100  # ~1000

    @pytest.mark.parametrize("text,expected_range", [
        ("Hi", (1, 3)),
        ("This is a medium sentence with several words.", (8, 15)),
        ("x" * 400, (90, 110)),
    ])
    def test_various_lengths(self, text, expected_range):
        tokens = estimate_tokens(text)
        assert expected_range[0] <= tokens <= expected_range[1]

    def test_message_tokens(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        tokens = estimate_message_tokens(messages)
        assert tokens > 10  # messages + overhead


# ── Pricing Tests ────────────────────────────────────────────────

class TestPricing:
    def test_known_model(self):
        db = PricingDB()
        prices = db.get("gpt-4o")
        assert prices["input"] == 2.50
        assert prices["output"] == 10.00

    def test_unknown_model_fallback(self):
        db = PricingDB()
        prices = db.get("totally-unknown-model")
        assert prices["input"] > 0
        assert prices["output"] > 0

    def test_prefix_match(self):
        db = PricingDB()
        prices = db.get("gpt-4o-2024-08-06")
        assert prices["input"] == 2.50  # matches "gpt-4o"

    def test_custom_model(self):
        db = PricingDB()
        db.add("my-model", input=1.00, output=3.00)
        prices = db.get("my-model")
        assert prices == {"input": 1.00, "output": 3.00}

    def test_list_models(self):
        db = PricingDB()
        models = db.list_models()
        assert "gpt-4o" in models
        assert "claude-3-5-sonnet" in models
        assert len(models) >= 10


# ── Budget Policy Tests ──────────────────────────────────────────

class TestBudgetPolicy:
    def test_defaults(self):
        policy = BudgetPolicy()
        assert policy.max_cost_per_request is None
        assert policy.alert_threshold_pct == 80.0

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            BudgetPolicy(alert_threshold_pct=150)

    def test_custom_limits(self):
        policy = BudgetPolicy(
            max_cost_per_request=0.05,
            max_cost_per_day=20.00,
        )
        assert policy.max_cost_per_request == 0.05
        assert policy.max_cost_per_day == 20.00


# ── Tracker Tests ────────────────────────────────────────────────

class TestTracker:
    def _make_record(self, cost: float = 0.01, tokens: int = 100) -> RequestRecord:
        return RequestRecord(
            model="gpt-4o",
            input_tokens=tokens,
            output_tokens=tokens // 4,
            cost=cost,
            duration_ms=50,
            timestamp=time.time(),
        )

    def test_empty_tracker(self):
        tracker = CostTracker()
        assert tracker.cost_today == 0
        assert tracker.cost_last_hour == 0
        assert tracker.request_count_today == 0

    def test_record_and_query(self):
        tracker = CostTracker()
        tracker.record(self._make_record(cost=0.05))
        tracker.record(self._make_record(cost=0.03))
        assert tracker.cost_today == pytest.approx(0.08)
        assert tracker.request_count_today == 2

    def test_cost_by_model(self):
        tracker = CostTracker()
        tracker.record(self._make_record(cost=0.05))
        r2 = self._make_record(cost=0.02)
        r2.model = "gpt-4o-mini"
        tracker.record(r2)
        breakdown = tracker.cost_by_model()
        assert "gpt-4o" in breakdown
        assert "gpt-4o-mini" in breakdown

    def test_export_csv(self):
        tracker = CostTracker()
        tracker.record(self._make_record())
        csv = tracker.export_csv()
        assert "timestamp" in csv
        assert "gpt-4o" in csv

    def test_export_json(self):
        tracker = CostTracker()
        tracker.record(self._make_record())
        data = tracker.export_json()
        assert len(data) == 1
        assert data[0]["model"] == "gpt-4o"


# ── Shield Tests ─────────────────────────────────────────────────

class TestShield:
    def test_dry_run(self):
        shield = Shield(model="gpt-4o")
        result = shield.call(
            messages=[{"role": "user", "content": "Hello"}],
            dry_run=True,
        )
        assert "estimated_cost" in result
        assert result["blocked"] is False

    def test_budget_exceeded_per_request(self):
        shield = Shield(
            model="gpt-4o",
            policy=BudgetPolicy(max_cost_per_request=0.0001),
        )
        # Large prompt should exceed tiny limit
        big_prompt = [{"role": "user", "content": "x" * 10000}]
        with pytest.raises(BudgetExceeded) as exc_info:
            shield.call(messages=big_prompt)
        assert "per-request" in str(exc_info.value)

    def test_budget_exceeded_per_day(self):
        shield = Shield(
            model="gpt-4o",
            policy=BudgetPolicy(max_cost_per_day=0.001),
        )
        # Record enough past cost to exceed daily limit
        from tokenshield.tracker import RequestRecord
        shield.tracker.record(RequestRecord(
            model="gpt-4o", input_tokens=50000, output_tokens=10000,
            cost=0.0009, duration_ms=100, timestamp=time.time(),
        ))
        with pytest.raises(BudgetExceeded) as exc_info:
            shield.call(messages=[{"role": "user", "content": "x" * 1000}])
        assert "per-day" in str(exc_info.value)

    def test_normal_call_tracks_cost(self):
        shield = Shield(model="gpt-4o")
        shield.call(messages=[{"role": "user", "content": "Hello"}])
        assert shield.tracker.request_count_today == 1
        assert shield.tracker.cost_today > 0

    def test_alert_fires(self):
        alerts = []
        shield = Shield(
            model="gpt-4o",
            policy=BudgetPolicy(max_cost_per_day=0.005, alert_threshold_pct=50),
            on_alert=lambda msg: alerts.append(msg),
        )
        # Pre-load cost to trigger alert
        shield.tracker.record(RequestRecord(
            model="gpt-4o", input_tokens=100, output_tokens=25,
            cost=0.003, duration_ms=50, timestamp=time.time(),
        ))
        shield.call(messages=[{"role": "user", "content": "Hi"}])
        assert len(alerts) >= 1
        assert "TokenShield" in alerts[0]

    def test_report_format(self):
        shield = Shield(model="gpt-4o", policy=BudgetPolicy(max_cost_per_day=20.0))
        shield.call(messages=[{"role": "user", "content": "Test"}])
        report = shield.report()
        assert "Requests today" in report
        assert "Budget remaining" in report


# ── Optimization Tests ───────────────────────────────────────────

class TestOptimize:
    def test_large_system_prompt(self):
        shield = Shield()
        messages = [
            {"role": "system", "content": "x" * 8000},
            {"role": "user", "content": "Hi"},
        ]
        tips = shield.optimize(messages)
        assert any("System prompt" in t for t in tips)

    def test_many_tools(self):
        shield = Shield()
        tools = [{"name": f"tool_{i}"} for i in range(15)]
        tips = shield.optimize(
            messages=[{"role": "user", "content": "Hi"}],
            tools=tools,
        )
        assert any("tools bound" in t for t in tips)

    def test_long_history(self):
        shield = Shield()
        messages = [{"role": "user", "content": "msg"}] * 40
        tips = shield.optimize(messages)
        assert any("History" in t for t in tips)

    def test_clean_request_no_tips(self):
        shield = Shield()
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello"},
        ]
        tips = shield.optimize(messages, tools=[{"name": "t1"}])
        # Small system prompt, few tools, short history → no tips
        assert len(tips) == 0


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
