"""TypeSafe Jev as an intent classifier: one Choice question over the intent catalog."""

from __future__ import annotations

import time

import httpx2
import typesafe_sdk as ts

from jevbench.classifiers.base import Outcome, Prediction, RetryableError
from jevbench.intents import IntentCatalog
from jevbench.settings import Settings
from jevbench.tracing import jev_llm_span, record_jev_usage

QUESTION_ID = "intent"

_RETRYABLE = (
    ts.TypeSafeRateLimitError,
    ts.TypeSafeInternalServerError,
    ts.TypeSafeAPIConnectionError,
    ts.TypeSafeAPITimeoutError,
)
# Request timeout and conflict are transient too (the SDK's own retry policy retries 408).
_RETRYABLE_STATUS = frozenset({408, 409})


def _retry_after_s(exc: ts.TypeSafeError) -> float | None:
    """Seconds the server asked us to wait before retrying, when it said so."""
    if isinstance(exc, ts.TypeSafeRateLimitError) and exc.retry_after_ms is not None:
        return exc.retry_after_ms / 1000
    return None


def _api_error(exc: ts.TypeSafeError, started: float) -> Prediction:
    return Prediction(
        outcome=Outcome.API_ERROR,
        intent=None,
        latency_ms=(time.perf_counter() - started) * 1000,
        error=f"{type(exc).__name__}: {exc}",
    )


class JevClassifier:
    def __init__(
        self,
        settings: Settings,
        catalog: IntentCatalog,
        name: str = "jev",
        *,
        transport: httpx2.AsyncBaseTransport | None = None,  # tests inject a mock transport
    ) -> None:
        self.name = name
        self.model_requested = settings.typesafe_model
        self._catalog = catalog
        self._allowed = set(catalog.names)
        self._question = ts.Choice(instructions=catalog.instruction, criteria=catalog.criteria())
        self._client = ts.AsyncTypeSafeClient(
            api_key=settings.typesafe_api_key,
            base_url=settings.typesafe_base_url,
            model=settings.typesafe_model,
            timeout=settings.request_timeout_s,
            # No SDK retries: every measured latency is exactly one HTTP attempt. The runner retries.
            retry=ts.RetryPolicy(max_retries=0),
            transport=transport,
        )

    @staticmethod
    def state_for(text: str) -> dict[str, ts.JSONValue]:
        return {"customer_message": text}

    async def classify(self, text: str) -> Prediction:
        state = self.state_for(text)
        with jev_llm_span(self.model_requested, state) as span:
            started = time.perf_counter()
            try:
                response = await self._client.system_one(state, {QUESTION_ID: self._question})
            except _RETRYABLE as exc:
                raise RetryableError(f"{type(exc).__name__}: {exc}", retry_after_s=_retry_after_s(exc)) from exc
            except ts.TypeSafeAPIError as exc:
                if exc.status in _RETRYABLE_STATUS:
                    raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
                return _api_error(exc, started)
            except ts.TypeSafeError as exc:
                return _api_error(exc, started)
            latency_ms = (time.perf_counter() - started) * 1000

            answer = response.answers.get(QUESTION_ID)
            usage = response.usage
            answer_dump = answer.model_dump() if answer is not None else {}
            record_jev_usage(
                span,
                model_reported=response.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                answer=answer_dump,
            )

        if not isinstance(answer, ts.ChoiceAnswer) or answer.choice not in self._allowed:
            return Prediction(
                outcome=Outcome.INVALID_OUTPUT,
                intent=None,
                latency_ms=latency_ms,
                model_reported=response.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                error=f"unexpected answer: {answer_dump}",
            )

        return Prediction(
            outcome=Outcome.OK,
            intent=answer.choice,
            latency_ms=latency_ms,
            model_reported=response.model,
            confidence=answer.confidence,
            probabilities=dict(answer.probabilities),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=0,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
