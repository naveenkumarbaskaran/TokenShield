"""Budget policy configuration."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class BudgetPolicy:
    """
    Defines spending limits for LLM API calls.

    All costs are in USD. Set to None/0 to disable a limit.
    """

    max_cost_per_request: float | None = None
    max_cost_per_hour: float | None = None
    max_cost_per_day: float | None = None
    max_tokens_per_request: int | None = None
    alert_threshold_pct: float = 80.0  # Alert when reaching this % of any limit

    def __post_init__(self):
        if self.alert_threshold_pct < 0 or self.alert_threshold_pct > 100:
            raise ValueError("alert_threshold_pct must be between 0 and 100")
