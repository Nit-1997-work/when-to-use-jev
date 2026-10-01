"""Re-ranking systems under test, by name. The concrete Gemini models come from `.env` (BASELINE_MODEL, _2)."""

from __future__ import annotations

from jevbench.common.settings import LLMSystemConfig, Prices, Settings
from jevbench.rerank.labels import LabelSet
from jevbench.rerank.ranking import Reranker
from jevbench.rerank.systems.baselines import Lexical, PresentedOrder
from jevbench.rerank.systems.gemini import GeminiLabel, GeminiListwise, GeminiPointwise
from jevbench.rerank.systems.jev import JevChoice, JevNoulBatch, JevNoulPair, JevScore

SYSTEMS: dict[str, str] = {
    "gemini-listwise": "BASELINE_MODEL (Flash-Lite): one call ranks every candidate; flags when nothing matches.",
    "gemini-flash-listwise": "BASELINE_MODEL_2 (Flash): the same listwise call on the larger model.",
    "gemini-label": "BASELINE_MODEL: one call labels every candidate E/S/C/I; ranked by label.",
    "gemini-pointwise": "BASELINE_MODEL: one call per candidate (in parallel) labels it E/S/C/I.",
    "jev-score": "Jev: one request, a Score question per candidate; ranked by expected gain.",
    "jev-noul-batch": "Jev: one request, an exact-match Noul per candidate; ranked by probability.",
    "jev-noul-pair": "Jev: one request per candidate (in parallel), an exact-match Noul each.",
    "jev-choice": "Jev: one request, one Choice over the candidate ids plus none_of_these.",
    "lexical": "BM25 over title and brand within the shortlist (no model call).",
    "random": "The presented order, a seeded per-search shuffle (no model call).",
}
NO_MODEL = frozenset({"lexical", "random"})


def _baseline_2(settings: Settings, name: str) -> LLMSystemConfig:
    if settings.baseline_2 is None:
        raise SystemExit(f"{name} needs BASELINE_MODEL_2 in .env.")
    return settings.baseline_2


def build_reranker(name: str, settings: Settings, labels: LabelSet) -> Reranker:
    match name:
        case "gemini-listwise":
            return GeminiListwise(settings, settings.baseline, labels, name=name)
        case "gemini-flash-listwise":
            return GeminiListwise(settings, _baseline_2(settings, name), labels, name=name)
        case "gemini-label":
            return GeminiLabel(settings, settings.baseline, labels, name=name)
        case "gemini-pointwise":
            return GeminiPointwise(settings, settings.baseline, labels, name=name)
        case "jev-score":
            return JevScore(settings, labels, name=name)
        case "jev-noul-batch":
            return JevNoulBatch(settings, labels, name=name)
        case "jev-noul-pair":
            return JevNoulPair(settings, labels, name=name)
        case "jev-choice":
            return JevChoice(settings, labels, name=name)
        case "lexical":
            return Lexical()
        case "random":
            return PresentedOrder()
    raise SystemExit(f"Unknown system {name!r}. Choose from: {', '.join(SYSTEMS)}.")


def prices_for(settings: Settings, system: str) -> Prices | None:
    """Token prices for a system under test; systems without a model call cost nothing."""
    if system in NO_MODEL:
        return Prices(input=0.0, output=0.0, thinking=0.0)
    if system.startswith("jev-"):
        return settings.jev_prices
    if system == "gemini-flash-listwise":
        return settings.baseline_2.prices if settings.baseline_2 is not None else None
    if system in SYSTEMS:
        return settings.baseline.prices
    return None


def llm_config_key(system: str) -> str | None:
    """Which manifest settings entry describes the system's LLM, if it has one."""
    if system == "gemini-flash-listwise":
        return "baseline_2"
    if system.startswith("gemini-"):
        return "baseline"
    return None
