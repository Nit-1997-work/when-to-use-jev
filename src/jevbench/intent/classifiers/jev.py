"""TypeSafe Jev as an intent classifier: one Choice question over the intent catalog."""

from __future__ import annotations

import httpx2
import typesafe_sdk as ts

from jevbench.common.settings import Settings
from jevbench.common.typesafe import ask, build_client
from jevbench.intent.classifiers.base import Outcome, Prediction
from jevbench.intent.intents import IntentCatalog

QUESTION_ID = "intent"


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
        self._allowed = set(catalog.names)
        self._question = ts.Choice(instructions=catalog.instruction, criteria=catalog.criteria())
        self._client = build_client(settings, transport=transport)

    @staticmethod
    def state_for(text: str) -> dict[str, ts.JSONValue]:
        return {"customer_message": text}

    async def classify(self, text: str) -> Prediction:
        reply = await ask(self._client, self.model_requested, self.state_for(text), {QUESTION_ID: self._question})
        if reply.api_error is not None:
            error = reply.api_error
            return Prediction(outcome=Outcome.API_ERROR, intent=None, latency_ms=reply.latency_ms, error=error)

        answer = reply.answers.get(QUESTION_ID)
        if not isinstance(answer, ts.ChoiceAnswer) or answer.choice not in self._allowed:
            answer_dump = answer.model_dump() if answer is not None else {}
            return Prediction(
                outcome=Outcome.INVALID_OUTPUT,
                intent=None,
                latency_ms=reply.latency_ms,
                model_reported=reply.model_reported,
                input_tokens=reply.input_tokens,
                output_tokens=reply.output_tokens,
                error=f"unexpected answer: {answer_dump}",
            )

        return Prediction(
            outcome=Outcome.OK,
            intent=answer.choice,
            latency_ms=reply.latency_ms,
            model_reported=reply.model_reported,
            confidence=answer.confidence,
            probabilities=dict(answer.probabilities),
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            reasoning_tokens=0,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
