"""One Chat Completions call through the LiteLLM-based gateway: errors, retry mapping, usage, and gateway headers.

Every experiment builds its own messages and response schema; this module sends them and reads back what came out.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import httpx2
import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from jevbench.common.calls import Outcome, RetryableError
from jevbench.common.settings import LLMSystemConfig, Settings

_RETRYABLE = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError)
# Request timeout and conflict are transient too (the OpenAI SDK's own retry policy retries them).
_RETRYABLE_STATUS = frozenset({408, 409})

# LiteLLM per-request cache bypass. Without it the gateway serves repeats from cache in ~1 ms.
NO_CACHE = {"cache": {"no-cache": True}}


def header_float(headers: Mapping[str, str], name: str) -> float | None:
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
        value = header_float(headers, name)
        if value is not None and value >= 0:
            return value * scale
    return None


@dataclass(frozen=True)
class GatewayReply:
    """What one gateway call returned. `problem` is set when there is no content to interpret."""

    latency_ms: float
    content: str | None = None
    # (outcome, error) when the call has no usable content: an API error, a safety block, or no choices at all.
    problem: tuple[Outcome, str] | None = None
    model_reported: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None  # visible output only (excludes reasoning)
    reasoning_tokens: int | None = None
    gateway_upstream_ms: float | None = None
    gateway_overhead_ms: float | None = None
    cache_hit: bool = False

    def measurement(self) -> dict[str, object]:
        """The measured fields every prediction record carries, whatever the answer was."""
        return {
            "latency_ms": self.latency_ms,
            "model_reported": self.model_reported,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "gateway_upstream_ms": self.gateway_upstream_ms,
            "gateway_overhead_ms": self.gateway_overhead_ms,
            "cache_hit": self.cache_hit,
        }


def read_completion(completion: ChatCompletion, headers: Mapping[str, str], latency_ms: float) -> GatewayReply:
    usage = completion.usage
    reasoning = 0
    if usage is not None and usage.completion_tokens_details is not None:
        reasoning = usage.completion_tokens_details.reasoning_tokens or 0
    measured = {
        "latency_ms": latency_ms,
        "model_reported": completion.model,
        "input_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": (usage.completion_tokens - reasoning) if usage else None,
        "reasoning_tokens": reasoning if usage else None,
        "gateway_upstream_ms": header_float(headers, "x-litellm-response-duration-ms"),
        "gateway_overhead_ms": header_float(headers, "x-litellm-overhead-duration-ms"),
        # LiteLLM only returns a cache key when the response came from its cache.
        "cache_hit": "x-litellm-cache-key" in headers,
    }
    if not completion.choices:
        return GatewayReply(problem=(Outcome.INVALID_OUTPUT, "response has no choices"), **measured)  # pyright: ignore[reportArgumentType]
    choice = completion.choices[0]
    if choice.finish_reason == "content_filter" or choice.message.refusal:
        return GatewayReply(problem=(Outcome.REFUSED, f"finish_reason={choice.finish_reason}"), **measured)  # pyright: ignore[reportArgumentType]
    return GatewayReply(content=choice.message.content, **measured)  # pyright: ignore[reportArgumentType]


class GatewayClient:
    """Sends Chat Completions requests for one model configuration. One HTTP attempt per call; the runner retries."""

    def __init__(
        self,
        settings: Settings,
        config: LLMSystemConfig,
        *,
        http_client: httpx2.AsyncClient | None = None,  # tests inject a mock transport
    ) -> None:
        self.config = config
        # max_retries=0: every measured latency is exactly one HTTP attempt. The runner retries.
        self._client = AsyncOpenAI(
            base_url=settings.gateway_base_url,
            api_key=settings.gateway_api_key,
            timeout=settings.request_timeout_s,
            max_retries=0,
            http_client=http_client,
        )

    def request_kwargs(
        self, messages: Sequence[Mapping[str, str]], response_format: Mapping[str, object]
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "model": self.config.model,
            "messages": list(messages),
            "temperature": 0,
            "response_format": response_format,
            "extra_body": NO_CACHE,
        }
        if self.config.reasoning_effort:
            kwargs["reasoning_effort"] = self.config.reasoning_effort
        return kwargs

    async def complete(
        self, messages: Sequence[Mapping[str, str]], response_format: Mapping[str, object]
    ) -> GatewayReply:
        """One call. Raises `RetryableError` for transient failures; any other HTTP error is an API error reply."""
        started = time.perf_counter()
        try:
            raw = await self._client.chat.completions.with_raw_response.create(  # pyright: ignore[reportCallIssue]
                **self.request_kwargs(messages, response_format)  # pyright: ignore[reportArgumentType]
            )
        except _RETRYABLE as exc:
            headers = exc.response.headers if isinstance(exc, openai.APIStatusError) else None
            raise RetryableError(f"{type(exc).__name__}: {exc}", retry_after_s=retry_after_s(headers)) from exc
        except openai.APIStatusError as exc:
            if exc.status_code in _RETRYABLE_STATUS:
                raise RetryableError(
                    f"{type(exc).__name__} {exc.status_code}: {exc}", retry_after_s=retry_after_s(exc.response.headers)
                ) from exc
            # An HTTP error is never counted as a safety block: blocks arrive as `finish_reason=content_filter`.
            return GatewayReply(
                latency_ms=(time.perf_counter() - started) * 1000,
                problem=(Outcome.API_ERROR, f"{type(exc).__name__} {exc.status_code}: {str(exc)[:500]}"),
            )
        completion = raw.parse()
        latency_ms = (time.perf_counter() - started) * 1000
        return read_completion(completion, raw.headers, latency_ms)

    async def aclose(self) -> None:
        await self._client.close()
