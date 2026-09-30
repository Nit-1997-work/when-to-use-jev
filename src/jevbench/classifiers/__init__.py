"""Systems under test, by name. The concrete model behind each LLM system comes from `.env`."""

from __future__ import annotations

from jevbench.classifiers.base import Classifier, Outcome, Prediction, RetryableError
from jevbench.classifiers.jev import JevClassifier
from jevbench.classifiers.llm import GatewayLLMClassifier
from jevbench.intents import IntentCatalog
from jevbench.settings import Settings

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


__all__ = ["SYSTEMS", "Classifier", "Outcome", "Prediction", "RetryableError", "build_classifier"]
