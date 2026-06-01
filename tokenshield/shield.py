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
    router: Any | None = None
    cache: Any | None = None
    compressor: Any | None = None
    _replay_log: list[dict] = field(default_factory=list, repr=False)

    BudgetExceeded = BudgetExceeded

    def call(
        self,
        messages: list[dict],
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

        self._replay_log.append({
            "messages": messages,
            "tools": tools,
            "model_used": active_model,
            "input_tokens": actual_input,
            "output_tokens": actual_output,
            "cost": actual_cost,
        })

        response = {
            "input_tokens": actual_input,
            "output_tokens": actual_output,
            "cost": actual_cost,
            "model": active_model,
            "routed_model": active_model,
            "cache_hit": False,
        }

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
            if total_tokens > 0 and sys_tokens > 500 and sys_tokens / total_tokens > 0.5:
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

    def replay(
        self,
        router: Any | None = None,
        model: str | None = None,
    ) -> dict:
        """Simulate recorded calls with a new router/model. Returns savings report."""
        from tokenshield.replay import CostReplay
        return CostReplay(self).run(router=router, model=model)

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
