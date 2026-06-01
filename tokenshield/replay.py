"""Agent Cost Replay — simulate a recorded session against a new routing config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CostReplay:
    """Replay a recorded Shield session against a new config to estimate savings."""

    shield: Any

    def run(
        self,
        router: Any | None = None,
        model: str | None = None,
    ) -> dict:
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

            # pricing.get() always returns a dict (falls back to {"input": 5.00, "output": 15.00} for unknown models)
            prices = pricing.get(replayed_model)
            cost = (
                entry["input_tokens"] * prices["input"]
                + entry["output_tokens"] * prices["output"]
            ) / 1_000_000
            replayed_cost += cost
            per_model[replayed_model] = per_model.get(replayed_model, 0) + 1

        raw_savings = original_cost - replayed_cost
        savings = round(max(raw_savings, 0.0), 6)
        savings_pct = round((savings / original_cost) * 100, 2) if original_cost > 0 else 0.0

        return {
            "call_count": len(log),
            "original_cost": round(original_cost, 6),
            "replayed_cost": round(replayed_cost, 6),
            "savings": savings,
            "savings_pct": savings_pct,
            "per_model": per_model,
        }
