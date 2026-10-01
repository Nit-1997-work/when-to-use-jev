"""The contract every intent classifier implements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from jevbench.common.calls import Outcome, RetryableError

__all__ = ["Classifier", "Outcome", "Prediction", "RetryableError"]


@dataclass
class Prediction:
    outcome: Outcome
    intent: str | None
    latency_ms: float
    model_reported: str | None = None
    # Confidence of the chosen intent, 0..1. Jev: native `confidence`. LLM: self-reported, only in `-conf` systems.
    confidence: float | None = None
    # Full distribution over intents. Jev only; the gateway does not expose Gemini logprobs.
    probabilities: dict[str, float] | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None  # visible output only (excludes reasoning)
    reasoning_tokens: int | None = None
    # LiteLLM gateway timing headers: the gateway's reported duration for the call, and its own overhead.
    gateway_upstream_ms: float | None = None
    gateway_overhead_ms: float | None = None
    cache_hit: bool = False
    error: str | None = None


class Classifier(Protocol):
    name: str
    model_requested: str

    async def classify(self, text: str) -> Prediction: ...

    async def aclose(self) -> None: ...
