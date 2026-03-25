"""Model pricing database."""

from __future__ import annotations

from dataclasses import dataclass, field


# Prices per 1 million tokens (USD)
DEFAULT_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Anthropic
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    # Google
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    # Mistral
    "mistral-large": {"input": 2.00, "output": 6.00},
    "mistral-small": {"input": 0.20, "output": 0.60},
}

# Fallback pricing for unknown models
FALLBACK_PRICING = {"input": 5.00, "output": 15.00}


@dataclass
class PricingDB:
    """
    Model pricing database with support for custom models.

    Prices are per 1 million tokens in USD.
    """

    _prices: dict[str, dict[str, float]] = field(
        default_factory=lambda: dict(DEFAULT_PRICING)
    )

    def get(self, model: str) -> dict[str, float]:
        """
        Get pricing for a model.

        Tries exact match, then prefix match, then fallback.
        """
        # Exact match
        if model in self._prices:
            return self._prices[model]

        # Prefix match (e.g., "gpt-4o-2024-08-06" matches "gpt-4o")
        for key in sorted(self._prices.keys(), key=len, reverse=True):
            if model.startswith(key):
                return self._prices[key]

        return FALLBACK_PRICING

    def add(self, model: str, input: float, output: float) -> None:
        """Register custom model pricing."""
        self._prices[model] = {"input": input, "output": output}

    def list_models(self) -> list[str]:
        """List all known models."""
        return sorted(self._prices.keys())
