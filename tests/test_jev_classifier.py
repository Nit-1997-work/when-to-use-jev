from __future__ import annotations

import json

import httpx2
import pytest

from jevbench.classifiers.base import Outcome, RetryableError
from jevbench.classifiers.jev import JevClassifier
from jevbench.intents import load_catalog
from jevbench.settings import Settings


def _jev(settings: Settings, handler: httpx2.MockTransport) -> JevClassifier:
    return JevClassifier(settings, load_catalog(), transport=handler)


def _answer(choice: str, confidence: float) -> dict[str, object]:
    return {
        "model": "jev-1.13.0",
        "usage": {"input_tokens": 2100, "output_tokens": 43},
        "answers": {
            "intent": {
                "type": "choice",
                "choice": choice,
                "confidence": confidence,
                "probabilities": {"track_order": 0.9, "track_delivery": 0.1},
            }
        },
    }


async def test_request_carries_state_and_every_intent(settings: Settings) -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(json.loads(request.content))
        return httpx2.Response(200, json=_answer("track_order", 0.87))

    jev = _jev(settings, httpx2.MockTransport(handler))
    prediction = await jev.classify("where is my order")
    await jev.aclose()

    assert seen["state"] == {"customer_message": "where is my order"}
    question = seen["questions"]["intent"]  # pyright: ignore[reportIndexIssue]
    assert question["type"] == "choice"
    assert list(question["criteria"]) == load_catalog().names
    assert prediction.outcome == Outcome.OK
    assert prediction.intent == "track_order"
    assert prediction.confidence == 0.87
    assert prediction.probabilities == {"track_order": 0.9, "track_delivery": 0.1}
    assert (prediction.input_tokens, prediction.output_tokens, prediction.reasoning_tokens) == (2100, 43, 0)
    assert prediction.model_reported == "jev-1.13.0"


@pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
async def test_transient_statuses_are_retryable_not_wrong_answers(settings: Settings, status: int) -> None:
    jev = _jev(settings, httpx2.MockTransport(lambda _: httpx2.Response(status, json={"error": "slow down"})))
    with pytest.raises(RetryableError):
        await jev.classify("hi")
    await jev.aclose()


async def test_rate_limit_retry_after_is_passed_to_the_runner(settings: Settings) -> None:
    response = httpx2.Response(429, json={"error": "slow down"}, headers={"retry-after": "3"})
    jev = _jev(settings, httpx2.MockTransport(lambda _: response))
    with pytest.raises(RetryableError) as caught:
        await jev.classify("hi")
    await jev.aclose()
    assert caught.value.retry_after_s == 3.0


async def test_timeouts_are_retryable(settings: Settings) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("timed out", request=request)

    jev = _jev(settings, httpx2.MockTransport(handler))
    with pytest.raises(RetryableError):
        await jev.classify("hi")
    await jev.aclose()


async def test_bad_request_is_an_api_error(settings: Settings) -> None:
    jev = _jev(settings, httpx2.MockTransport(lambda _: httpx2.Response(400, json={"error": "bad"})))
    prediction = await jev.classify("hi")
    await jev.aclose()
    assert prediction.outcome == Outcome.API_ERROR
    assert prediction.error is not None
    assert prediction.error.startswith("TypeSafeBadRequestError")


async def test_a_choice_outside_the_intent_list_is_an_invalid_output(settings: Settings) -> None:
    jev = _jev(settings, httpx2.MockTransport(lambda _: httpx2.Response(200, json=_answer("not_an_intent", 0.9))))
    prediction = await jev.classify("hi")
    await jev.aclose()
    assert prediction.outcome == Outcome.INVALID_OUTPUT
    assert prediction.intent is None
    assert prediction.input_tokens == 2100
