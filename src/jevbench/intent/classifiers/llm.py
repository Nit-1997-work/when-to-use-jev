"""An LLM (Gemini) as an intent classifier, called through the LiteLLM-based, Chat Completions-compatible gateway."""

from __future__ import annotations

import json
from collections.abc import Mapping

import httpx2
from openai.types.chat import ChatCompletion

from jevbench.common.gateway import NO_CACHE, GatewayClient, GatewayReply, read_completion, retry_after_s
from jevbench.common.settings import LLMSystemConfig, Settings
from jevbench.intent.classifiers.base import Outcome, Prediction
from jevbench.intent.intents import IntentCatalog

__all__ = ["NO_CACHE", "GatewayLLMClassifier", "build_response_format", "build_system_prompt", "retry_after_s"]


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
        self._allowed = set(catalog.names)
        self._self_report_confidence = self_report_confidence
        self._system_prompt = build_system_prompt(catalog, self_report_confidence=self_report_confidence)
        self._response_format = build_response_format(catalog, self_report_confidence=self_report_confidence)
        self._gateway = GatewayClient(settings, config, http_client=http_client)

    def _messages(self, text: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": self._system_prompt}, {"role": "user", "content": text}]

    def request_kwargs(self, text: str) -> dict[str, object]:
        return self._gateway.request_kwargs(self._messages(text), self._response_format)

    async def classify(self, text: str) -> Prediction:
        return self.interpret(await self._gateway.complete(self._messages(text), self._response_format))

    def parse_completion(self, completion: ChatCompletion, headers: Mapping[str, str], latency_ms: float) -> Prediction:
        return self.interpret(read_completion(completion, headers, latency_ms))

    def interpret(self, reply: GatewayReply) -> Prediction:
        if reply.problem is not None:
            outcome, error = reply.problem
            if outcome == Outcome.API_ERROR:
                return Prediction(outcome=outcome, intent=None, latency_ms=reply.latency_ms, error=error)
            return Prediction(outcome=outcome, intent=None, error=error, **reply.measurement())  # pyright: ignore[reportArgumentType]
        base = reply.measurement()
        content = reply.content
        try:
            parsed = json.loads(content or "")
        except json.JSONDecodeError as exc:
            error = f"{type(exc).__name__}: {content!r:.300}"
            return Prediction(outcome=Outcome.INVALID_OUTPUT, intent=None, error=error, **base)  # pyright: ignore[reportArgumentType]
        intent = parsed.get("intent") if isinstance(parsed, dict) else None
        if not isinstance(intent, str) or intent not in self._allowed:
            error = f"not an allowed intent: {content!r:.300}"
            return Prediction(outcome=Outcome.INVALID_OUTPUT, intent=None, error=error, **base)  # pyright: ignore[reportArgumentType]

        confidence = _confidence(parsed.get("confidence")) if self._self_report_confidence else None
        return Prediction(outcome=Outcome.OK, intent=intent, confidence=confidence, **base)  # pyright: ignore[reportArgumentType]

    async def aclose(self) -> None:
        await self._gateway.aclose()
