"""Intent classifiers under test, by name. The concrete model behind each LLM system comes from `.env`."""

from __future__ import annotations

from jevbench.common.settings import Prices, Settings
from jevbench.intent.classifiers.base import Classifier, Outcome, Prediction, RetryableError
from jevbench.intent.classifiers.jev import JevClassifier
from jevbench.intent.classifiers.llm import GatewayLLMClassifier
from jevbench.intent.intents import IntentCatalog

SYSTEMS: dict[str, str] = {
    "jev": "TypeSafe Jev, one Choice question over all intents (TYPESAFE_DEFAULT_MODEL).",
    "baseline": "BASELINE_MODEL via the gateway, structured output: intent only.",
    "baseline-conf": "BASELINE_MODEL via the gateway, structured output: intent + self-reported confidence.",
    "baseline-2": "BASELINE_MODEL_2 via the gateway, structured output: intent only.",
}


def build_classifier(name: str, settings: Settings, catalog: IntentCatalog) -> Classifier:
    if name == "jev":
        return JevClassifier(settings, catalog)
    if name == "baseline":
        return GatewayLLMClassifier(settings, catalog, settings.baseline, name=name)
    if name == "baseline-conf":
        return GatewayLLMClassifier(settings, catalog, settings.baseline, name=name, self_report_confidence=True)
    if name == "baseline-2":
        if settings.baseline_2 is None:
            raise SystemExit("baseline-2 needs BASELINE_MODEL_2 in .env.")
        return GatewayLLMClassifier(settings, catalog, settings.baseline_2, name=name)
    raise SystemExit(f"Unknown system {name!r}. Choose from: {', '.join(SYSTEMS)}.")


def prices_for(settings: Settings, system: str) -> Prices | None:
    """Token prices for a system under test, or None for an unknown system."""
    if system == "jev":
        return settings.jev_prices
    if system in ("baseline", "baseline-conf"):
        return settings.baseline.prices
    if system == "baseline-2" and settings.baseline_2 is not None:
        return settings.baseline_2.prices
    return None


__all__ = ["SYSTEMS", "Classifier", "Outcome", "Prediction", "RetryableError", "build_classifier", "prices_for"]
