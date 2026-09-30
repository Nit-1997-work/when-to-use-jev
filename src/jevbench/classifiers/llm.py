"""An LLM (Gemini) as an intent classifier, called through the LiteLLM-based, Chat Completions-compatible gateway."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping

import httpx2
import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from jevbench.classifiers.base import Outcome, Prediction, RetryableError
from jevbench.intents import IntentCatalog
from jevbench.settings import LLMSystemConfig, Settings

_RETRYABLE = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError)
# Request timeout and conflict are transient too (the OpenAI SDK's own retry policy retries them).
_RETRYABLE_STATUS = frozenset({408, 409})

# LiteLLM per-request cache bypass. Without it the gateway serves repeats from cache in ~1 ms.
NO_CACHE = {"cache": {"no-cache": True}}


def build_system_prompt(catalog: IntentCatalog, *, self_report_confidence: bool) -> str:
    lines = [
        "You classify customer messages sent to an online retailer's support assistant.",
        catalog.instruction,
        "Choose exactly one intent from this list. Each intent has a description (`means`) and,",
        "for intents with close neighbours, what it is not (`not`):",
        "",
        catalog.criteria_json(),
        "",
        'Reply with JSON: {"intent": "<intent name>"}.',
    ]
    if self_report_confidence:
        lines[-1] = (
            'Reply with JSON: {"intent": "<intent name>", "confidence": <number from 0 to 1>}, '
            "where confidence is the probability that your chosen intent is correct."
        )
    return "\n".join(lines)


def build_response_format(catalog: IntentCatalog, *, self_report_confidence: bool) -> dict[str, object]:
    properties: dict[str, object] = {"intent": {"type": "string", "enum": catalog.names}}
    required = ["intent"]
    if self_report_confidence:
        properties["confidence"] = {"type": "number", "minimum": 0, "maximum": 1}
        required.append("confidence")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "intent_classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def _header_float(headers: Mapping[str, str], name: str) -> float | None:
    value = headers.get(name)
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def retry_after_s(headers: Mapping[str, str] | None) -> float | None:
    """Seconds the server asked us to wait (`retry-after-ms`, or the numeric form of `retry-after`)."""
    if headers is None:
        return None
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        value = _header_float(headers, name)
        if value is not None and value >= 0:
            return value * scale
    return None


def _confidence(value: object) -> float | None:
    """A self-reported confidence, or None when it is missing or outside 0..1."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value if 0 <= value <= 1 else None


class GatewayLLMClassifier:
    def __init__(
        self,
        settings: Settings,
        catalog: IntentCatalog,
        config: LLMSystemConfig,
        *,
        name: str,
        self_report_confidence: bool = False,
        http_client: httpx2.AsyncClient | None = None,  # tests inject a mock transport
    ) -> None:
        self.name = name
        self.model_requested = config.model
        self._config = config
        self._allowed = set(catalog.names)
        self._self_report_confidence = self_report_confidence
        self._system_prompt = build_system_prompt(catalog, self_report_confidence=self_report_confidence)
        self._response_format = build_response_format(catalog, self_report_confidence=self_report_confidence)
        # max_retries=0: every measured latency is exactly one HTTP attempt. The runner retries.
        self._client = AsyncOpenAI(
            base_url=settings.gateway_base_url,
            api_key=settings.gateway_api_key,
            timeout=settings.request_timeout_s,
            max_retries=0,
            http_client=http_client,
        )

    def request_kwargs(self, text: str) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": 0,
            "response_format": self._response_format,
            "extra_body": NO_CACHE,
        }
        if self._config.reasoning_effort:
            kwargs["reasoning_effort"] = self._config.reasoning_effort
        return kwargs

    async def classify(self, text: str) -> Prediction:
        started = time.perf_counter()
        try:
            raw = await self._client.chat.completions.with_raw_response.create(**self.request_kwargs(text))  # pyright: ignore[reportCallIssue, reportArgumentType]
        except _RETRYABLE as exc:
            headers = exc.response.headers if isinstance(exc, openai.APIStatusError) else None
            raise RetryableError(f"{type(exc).__name__}: {exc}", retry_after_s=retry_after_s(headers)) from exc
        except openai.APIStatusError as exc:
            if exc.status_code in _RETRYABLE_STATUS:
                raise RetryableError(
                    f"{type(exc).__name__} {exc.status_code}: {exc}", retry_after_s=retry_after_s(exc.response.headers)
                ) from exc
            # An HTTP error is never counted as a safety block: blocks arrive as `finish_reason=content_filter`.
            return Prediction(
                outcome=Outcome.API_ERROR,
                intent=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(exc).__name__} {exc.status_code}: {str(exc)[:500]}",
            )
        completion = raw.parse()
        latency_ms = (time.perf_counter() - started) * 1000
        return self.parse_completion(completion, raw.headers, latency_ms)

    def parse_completion(self, completion: ChatCompletion, headers: Mapping[str, str], latency_ms: float) -> Prediction:
        usage = completion.usage
        reasoning = 0
        if usage is not None and usage.completion_tokens_details is not None:
            reasoning = usage.completion_tokens_details.reasoning_tokens or 0
        base = {
            "latency_ms": latency_ms,
            "model_reported": completion.model,
            "input_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": (usage.completion_tokens - reasoning) if usage else None,
            "reasoning_tokens": reasoning if usage else None,
            "gateway_upstream_ms": _header_float(headers, "x-litellm-response-duration-ms"),
            "gateway_overhead_ms": _header_float(headers, "x-litellm-overhead-duration-ms"),
            # LiteLLM only returns a cache key when the response came from its cache.
            "cache_hit": "x-litellm-cache-key" in headers,
        }

        if not completion.choices:
            return Prediction(outcome=Outcome.INVALID_OUTPUT, intent=None, error="response has no choices", **base)
        choice = completion.choices[0]
        content = choice.message.content
        if choice.finish_reason == "content_filter" or choice.message.refusal:
            error = f"finish_reason={choice.finish_reason}"
            return Prediction(outcome=Outcome.REFUSED, intent=None, error=error, **base)
        try:
            parsed = json.loads(content or "")
        except json.JSONDecodeError as exc:
            error = f"{type(exc).__name__}: {content!r:.300}"
            return Prediction(outcome=Outcome.INVALID_OUTPUT, intent=None, error=error, **base)
        intent = parsed.get("intent") if isinstance(parsed, dict) else None
        if not isinstance(intent, str) or intent not in self._allowed:
            error = f"not an allowed intent: {content!r:.300}"
            return Prediction(outcome=Outcome.INVALID_OUTPUT, intent=None, error=error, **base)

        confidence = _confidence(parsed.get("confidence")) if self._self_report_confidence else None
        return Prediction(outcome=Outcome.OK, intent=intent, confidence=confidence, **base)

    async def aclose(self) -> None:
        await self._client.close()
