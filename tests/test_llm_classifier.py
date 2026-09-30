from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace

import httpx2
import pytest
from openai.types.chat import ChatCompletion

from jevbench.classifiers.base import Outcome, RetryableError
from jevbench.classifiers.llm import (
    NO_CACHE,
    GatewayLLMClassifier,
    build_response_format,
    build_system_prompt,
    retry_after_s,
)
from jevbench.intents import load_catalog
from jevbench.settings import Settings

Handler = Callable[[httpx2.Request], httpx2.Response]


def _classifier(settings: Settings, *, conf: bool = False, handler: Handler | None = None) -> GatewayLLMClassifier:
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) if handler else None
    return GatewayLLMClassifier(
        settings,
        load_catalog(),
        settings.baseline,
        name="baseline",
        self_report_confidence=conf,
        http_client=http_client,
    )


def _body(content: str | None, *, finish_reason: str = "stop", reasoning: int = 0, choices: bool = True) -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "google/test-model",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}]
        if choices
        else [],
        "usage": {
            "prompt_tokens": 1200,
            "completion_tokens": 12 + reasoning,
            "total_tokens": 1212 + reasoning,
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        },
    }


def _completion(content: str | None, **kwargs: object) -> ChatCompletion:
    return ChatCompletion.model_validate(_body(content, **kwargs))  # pyright: ignore[reportArgumentType]


def test_prompt_and_schema_carry_every_intent() -> None:
    catalog = load_catalog()
    prompt = build_system_prompt(catalog, self_report_confidence=False)
    schema = build_response_format(catalog, self_report_confidence=False)
    # The LLM sees exactly the label text Jev gets as Choice criteria.
    assert catalog.criteria_json() in prompt
    enum = schema["json_schema"]["schema"]["properties"]["intent"]["enum"]  # pyright: ignore[reportIndexIssue]
    assert enum == catalog.names
    confident = build_response_format(catalog, self_report_confidence=True)
    assert confident["json_schema"]["schema"]["required"] == ["intent", "confidence"]  # pyright: ignore[reportIndexIssue]
    assert "confidence" in build_system_prompt(catalog, self_report_confidence=True)


def test_every_request_bypasses_the_gateway_cache(settings: Settings) -> None:
    kwargs = _classifier(settings).request_kwargs("hi")
    assert kwargs["extra_body"] == NO_CACHE
    assert kwargs["temperature"] == 0
    assert "reasoning_effort" not in kwargs


def test_reasoning_effort_is_sent_only_when_configured(settings: Settings) -> None:
    low = replace(settings, baseline=replace(settings.baseline, reasoning_effort="low"))
    assert _classifier(low).request_kwargs("hi")["reasoning_effort"] == "low"


def test_parse_ok_splits_reasoning_from_output_tokens(settings: Settings) -> None:
    p = _classifier(settings).parse_completion(_completion('{"intent": "track_order"}', reasoning=30), {}, 123.0)
    assert p.outcome == Outcome.OK
    assert p.intent == "track_order"
    assert (p.input_tokens, p.output_tokens, p.reasoning_tokens) == (1200, 12, 30)
    assert p.confidence is None


@pytest.mark.parametrize(
    ("content", "finish_reason", "outcome"),
    [
        ('{"intent": "not_a_real_intent"}', "stop", Outcome.INVALID_OUTPUT),
        ('{"intent": ["track_order"]}', "stop", Outcome.INVALID_OUTPUT),  # not a string: must not crash the run
        ('["track_order"]', "stop", Outcome.INVALID_OUTPUT),
        ("not json", "stop", Outcome.INVALID_OUTPUT),
        (None, "content_filter", Outcome.REFUSED),
    ],
)
def test_parse_failures_are_classified(
    settings: Settings, content: str | None, finish_reason: str, outcome: Outcome
) -> None:
    p = _classifier(settings).parse_completion(_completion(content, finish_reason=finish_reason), {}, 1.0)
    assert p.outcome == outcome
    assert p.intent is None


def test_a_response_without_choices_is_an_invalid_output(settings: Settings) -> None:
    p = _classifier(settings).parse_completion(_completion(None, choices=False), {}, 1.0)
    assert (p.outcome, p.error) == (Outcome.INVALID_OUTPUT, "response has no choices")


def test_cache_hit_header_is_flagged(settings: Settings) -> None:
    headers = {"x-litellm-cache-key": "abc", "x-litellm-response-duration-ms": "1.2"}
    p = _classifier(settings).parse_completion(_completion('{"intent": "track_order"}'), headers, 5.0)
    assert p.cache_hit is True
    assert p.gateway_upstream_ms == 1.2


def test_self_reported_confidence_is_validated(settings: Settings) -> None:
    classifier = _classifier(settings, conf=True)
    ok = classifier.parse_completion(_completion('{"intent": "track_order", "confidence": 0.8}'), {}, 1.0)
    bad = classifier.parse_completion(_completion('{"intent": "track_order", "confidence": 7}'), {}, 1.0)
    flag = classifier.parse_completion(_completion('{"intent": "track_order", "confidence": true}'), {}, 1.0)
    assert ok.confidence == 0.8
    assert bad.confidence is None
    assert flag.confidence is None


async def test_classify_sends_the_request_and_reads_the_gateway_headers(settings: Settings) -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(json.loads(request.content))
        headers = {"x-litellm-response-duration-ms": "650", "x-litellm-overhead-duration-ms": "4"}
        return httpx2.Response(200, json=_body('{"intent": "track_order"}'), headers=headers)

    classifier = _classifier(settings, handler=handler)
    prediction = await classifier.classify("where is my order")
    await classifier.aclose()

    assert seen["cache"] == {"no-cache": True}
    assert seen["temperature"] == 0
    assert seen["messages"][1] == {"role": "user", "content": "where is my order"}  # pyright: ignore[reportIndexIssue]
    assert prediction.outcome == Outcome.OK
    assert (prediction.gateway_upstream_ms, prediction.gateway_overhead_ms, prediction.cache_hit) == (650.0, 4.0, False)
    assert prediction.latency_ms > 0


async def test_a_safety_block_is_recorded_as_blocked(settings: Settings) -> None:
    classifier = _classifier(
        settings, handler=lambda _: httpx2.Response(200, json=_body(None, finish_reason="content_filter"))
    )
    prediction = await classifier.classify("hi")
    await classifier.aclose()
    assert prediction.outcome == Outcome.REFUSED
    assert prediction.error == "finish_reason=content_filter"


async def test_an_http_error_is_an_api_error_never_a_block(settings: Settings) -> None:
    body = {"error": {"message": "Request blocked: invalid safety_settings"}}
    classifier = _classifier(settings, handler=lambda _: httpx2.Response(400, json=body))
    prediction = await classifier.classify("hi")
    await classifier.aclose()
    assert prediction.outcome == Outcome.API_ERROR
    assert prediction.error is not None
    assert prediction.error.startswith("BadRequestError 400")


@pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
async def test_transient_statuses_are_retryable(settings: Settings, status: int) -> None:
    classifier = _classifier(settings, handler=lambda _: httpx2.Response(status, json={"error": "try again"}))
    with pytest.raises(RetryableError):
        await classifier.classify("hi")
    await classifier.aclose()


async def test_retry_after_is_passed_to_the_runner(settings: Settings) -> None:
    headers = {"retry-after": "2"}
    classifier = _classifier(settings, handler=lambda _: httpx2.Response(429, json={}, headers=headers))
    with pytest.raises(RetryableError) as caught:
        await classifier.classify("hi")
    await classifier.aclose()
    assert caught.value.retry_after_s == 2.0


async def test_timeouts_are_retryable(settings: Settings) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("timed out", request=request)

    classifier = _classifier(settings, handler=handler)
    with pytest.raises(RetryableError, match="APITimeoutError"):
        await classifier.classify("hi")
    await classifier.aclose()


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after-ms": "250", "retry-after": "9"}, 0.25),
        ({"retry-after": "3"}, 3.0),
        ({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, None),  # date form: fall back to our own backoff
        ({"retry-after": "-1"}, None),
        ({"retry-after": "inf"}, None),
        ({}, None),
        (None, None),
    ],
)
def test_retry_after_parsing(headers: dict[str, str] | None, expected: float | None) -> None:
    assert retry_after_s(headers) == expected
