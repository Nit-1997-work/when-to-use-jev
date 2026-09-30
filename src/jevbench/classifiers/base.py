"""The shared contract every system under test implements."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class Outcome(StrEnum):
    OK = "ok"
    # An answer that is not one of the allowed intents (or cannot be parsed). It is an answer, and it counts as wrong.
    INVALID_OUTPUT = "invalid_output"
    # Blocked: the provider's safety filter withheld the answer. Excluded from quality metrics and reported separately.
    # The stored value stays "refused" so earlier result files keep their meaning.
    REFUSED = "refused"
    # Non-retryable request failure, or retries exhausted. Excluded from quality metrics; a resumed run redoes it.
    API_ERROR = "api_error"


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


class RetryableError(Exception):
    """Transient failure (rate limit, 408/409, 5xx, timeout, connection). The runner retries and records attempts."""

    def __init__(self, message: str, *, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        # Server-requested wait (Retry-After), honoured by the runner's backoff when present.
        self.retry_after_s = retry_after_s


class Classifier(Protocol):
    name: str
    model_requested: str

    async def classify(self, text: str) -> Prediction: ...

    async def aclose(self) -> None: ...
