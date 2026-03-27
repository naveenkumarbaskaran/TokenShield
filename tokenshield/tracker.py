"""Cost tracking with time-windowed aggregation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from collections import deque


@dataclass
class RequestRecord:
    """Record of a single LLM API call."""
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    duration_ms: int
    timestamp: float


@dataclass
class CostTracker:
    """
    Tracks LLM API costs with time-windowed aggregation.

    Maintains a rolling window of request records for computing
    hourly and daily costs without unbounded memory growth.
    """

    max_records: int = 10_000
    _records: deque[RequestRecord] = field(default_factory=lambda: deque(maxlen=10_000))

    def record(self, rec: RequestRecord) -> None:
        """Add a request record."""
        self._records.append(rec)

    @property
    def cost_today(self) -> float:
        """Total cost in the current calendar day."""
        cutoff = _start_of_day()
        return sum(r.cost for r in self._records if r.timestamp >= cutoff)

    @property
    def cost_last_hour(self) -> float:
        """Total cost in the last 60 minutes."""
        cutoff = time.time() - 3600
        return sum(r.cost for r in self._records if r.timestamp >= cutoff)

    @property
    def request_count_today(self) -> int:
        cutoff = _start_of_day()
        return sum(1 for r in self._records if r.timestamp >= cutoff)

    @property
    def total_input_today(self) -> int:
        cutoff = _start_of_day()
        return sum(r.input_tokens for r in self._records if r.timestamp >= cutoff)

    @property
    def total_output_today(self) -> int:
        cutoff = _start_of_day()
        return sum(r.output_tokens for r in self._records if r.timestamp >= cutoff)

    def cost_by_model(self) -> dict[str, float]:
        """Cost breakdown by model for today."""
        cutoff = _start_of_day()
        breakdown: dict[str, float] = {}
        for r in self._records:
            if r.timestamp >= cutoff:
                breakdown[r.model] = breakdown.get(r.model, 0) + r.cost
        return breakdown

    def export_csv(self) -> str:
        """Export all records as CSV."""
        lines = ["timestamp,model,input_tokens,output_tokens,cost,duration_ms"]
        for r in self._records:
            lines.append(
                f"{r.timestamp},{r.model},{r.input_tokens},"
                f"{r.output_tokens},{r.cost:.6f},{r.duration_ms}"
            )
        return "\n".join(lines)

    def export_json(self) -> list[dict]:
        """Export all records as list of dicts."""
        return [
            {
                "timestamp": r.timestamp,
                "model": r.model,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cost": r.cost,
                "duration_ms": r.duration_ms,
            }
            for r in self._records
        ]


def _start_of_day() -> float:
    """Timestamp for midnight today (local time)."""
    import datetime
    today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    return today.timestamp()
