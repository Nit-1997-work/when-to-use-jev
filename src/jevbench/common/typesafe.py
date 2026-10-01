"""One TypeSafe `system_one` call: errors, retry mapping, usage, and the manual Jev trace span.

Every experiment builds its own state and questions; this module sends them and reads back the answers.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field

import httpx2
import typesafe_sdk as ts

from jevbench.common.calls import RetryableError
from jevbench.common.settings import Settings
from jevbench.common.tracing import jev_llm_span, record_jev_usage

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


def build_client(settings: Settings, *, transport: httpx2.AsyncBaseTransport | None = None) -> ts.AsyncTypeSafeClient:
    return ts.AsyncTypeSafeClient(
        api_key=settings.typesafe_api_key,
        base_url=settings.typesafe_base_url,
        model=settings.typesafe_model,
        timeout=settings.request_timeout_s,
        # No SDK retries: every measured latency is exactly one HTTP attempt. The runner retries.
        retry=ts.RetryPolicy(max_retries=0),
        transport=transport,
    )


@dataclass(frozen=True)
class JevReply:
    """What one `system_one` call returned. `api_error` is set when the request failed for good."""

    latency_ms: float
    answers: Mapping[str, ts.Answer] = field(default_factory=dict)
    model_reported: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    api_error: str | None = None


async def ask(
    client: ts.AsyncTypeSafeClient, model: str, state: Mapping[str, ts.JSONValue], questions: Mapping[str, ts.Question]
) -> JevReply:
    """One call. Raises `RetryableError` for transient failures; any other failure is an API error reply."""
    with jev_llm_span(model, state) as span:
        started = time.perf_counter()
        try:
            response = await client.system_one(state, questions)
        except _RETRYABLE as exc:
            raise RetryableError(f"{type(exc).__name__}: {exc}", retry_after_s=_retry_after_s(exc)) from exc
        except ts.TypeSafeAPIError as exc:
            if exc.status in _RETRYABLE_STATUS:
                raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
            return JevReply(latency_ms=(time.perf_counter() - started) * 1000, api_error=f"{type(exc).__name__}: {exc}")
        except ts.TypeSafeError as exc:
            return JevReply(latency_ms=(time.perf_counter() - started) * 1000, api_error=f"{type(exc).__name__}: {exc}")
        latency_ms = (time.perf_counter() - started) * 1000

        answers = dict(response.answers)
        usage = response.usage
        # One question: trace its answer itself. Several: trace every answer by question name.
        if len(questions) == 1:
            only = answers.get(next(iter(questions)))
            traced: Mapping[str, object] = only.model_dump() if only is not None else {}
        else:
            traced = {name: answer.model_dump() for name, answer in answers.items()}
        record_jev_usage(
            span,
            model_reported=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            answer=traced,
        )
    return JevReply(
        latency_ms=latency_ms,
        answers=answers,
        model_reported=response.model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )
