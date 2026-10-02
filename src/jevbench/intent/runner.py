"""Runs one intent classifier over one split, sequentially, writing one JSONL record per classification.

Records are the source of truth. Phoenix spans and experiments are views derived from the same calls.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from jevbench.common.paths import results_dir
from jevbench.common.runner import RunTarget, base_manifest, call_with_retries, now, run_records
from jevbench.common.settings import Settings
from jevbench.common.tracing import current_trace_id, example_span, record_result, tracer
from jevbench.intent.classifiers import Classifier, Outcome, Prediction, prices_for
from jevbench.intent.data import SPLITS_DIR, Example, file_sha256, load_split
from jevbench.intent.intents import IntentCatalog

RESULTS_DIR = results_dir("intent")
# Manifest fields that must not change while one results file is filled across several sessions.
RESUME_KEYS = ("system", "split", "repeat", "model_requested", "intents_sha256", "split_sha256", "settings")


def target(run_id: str, system: str, split: str, repeat: int) -> RunTarget:
    return RunTarget(root=RESULTS_DIR, run_id=run_id, system=system, split=split, repeat=repeat)


def build_manifest(
    target: RunTarget, settings: Settings, catalog: IntentCatalog, classifier: Classifier
) -> dict[str, object]:
    return base_manifest(
        target,
        model_requested=classifier.model_requested,
        data_hashes={
            "intents_sha256": catalog.sha256,
            "split_sha256": file_sha256(SPLITS_DIR / f"{target.split}.jsonl"),
        },
        prices=prices_for(settings, target.system),
        settings=settings,
    )


async def classify_with_retries(classifier: Classifier, text: str) -> tuple[Prediction, int, float]:
    """Returns (prediction, attempts, total backoff seconds). Backoff is never part of `latency_ms`."""
    return await call_with_retries(
        lambda: classifier.classify(text),
        lambda error: Prediction(outcome=Outcome.API_ERROR, intent=None, latency_ms=0.0, error=error),
    )


def build_record(
    target: RunTarget,
    classifier: Classifier,
    catalog: IntentCatalog,
    example: Example,
    prediction: Prediction,
    attempts: int,
    backoff_s: float,
    started_at: str,
    trace_id: str | None,
) -> dict[str, object]:
    predicted_category = catalog.category_of(prediction.intent) if prediction.intent else None
    top2 = None
    if prediction.probabilities:
        ranked = sorted(prediction.probabilities.items(), key=lambda kv: kv[1], reverse=True)
        top2 = [name for name, _ in ranked[:2]]
    measured = {k: (v.value if isinstance(v, Outcome) else v) for k, v in asdict(prediction).items() if k != "intent"}
    return {
        "run_id": target.run_id,
        "system": target.system,
        "split": target.split,
        "repeat": target.repeat,
        "example_id": example.id,
        "text": example.text,
        "expected_intent": example.intent,
        "expected_category": example.category,
        "tags": example.tags,
        "has_profanity": example.has_profanity,
        "has_typos": example.has_typos,
        "predicted_intent": prediction.intent,
        "predicted_category": predicted_category,
        "correct": prediction.intent == example.intent,
        "category_correct": predicted_category == example.category,
        "top2": top2,
        "model_requested": classifier.model_requested,
        "attempts": attempts,
        "backoff_s": round(backoff_s, 3),
        "started_at": started_at,
        "trace_id": trace_id,
        **measured,
    }


async def warm_up(classifier: Classifier, count: int) -> None:
    """Discarded calls that open connections before timing.

    Traced under a `warmup` span so Phoenix dashboards can exclude them.
    """
    with tracer().start_as_current_span("warmup", attributes={"jevbench.warmup": True}):
        for example in load_split("warmup")[:count]:
            await classify_with_retries(classifier, example.text)


async def run_target(
    target: RunTarget,
    classifier: Classifier,
    settings: Settings,
    catalog: IntentCatalog,
    *,
    limit: int | None,
    warmup: int,
    progress_every: int = 50,
) -> Path:
    examples = load_split(target.split)
    if limit is not None:
        examples = examples[:limit]

    async def run_one(example: Example) -> dict[str, object]:
        started_at = now()
        with example_span("classify", target.system, example.id, target.split, target.repeat, example.text) as span:
            prediction, attempts, backoff_s = await classify_with_retries(classifier, example.text)
            record = build_record(
                target, classifier, catalog, example, prediction, attempts, backoff_s, started_at, current_trace_id()
            )
            record_result(
                span,
                output=prediction.intent or "",
                outcome=str(record["outcome"]),
                attributes={"jevbench.expected_intent": example.intent, "jevbench.correct": bool(record["correct"])},
            )
        return record

    def progress(records: list[dict[str, object]]) -> str:
        return f"acc={sum(bool(r['correct']) for r in records) / len(records):.3f}"

    return await run_records(
        target,
        examples,
        example_id=lambda e: e.id,
        manifest=lambda: build_manifest(target, settings, catalog, classifier),
        resume_keys=RESUME_KEYS,
        warm_up=(lambda: warm_up(classifier, warmup)) if warmup else None,
        run_one=run_one,
        progress=progress,
        progress_every=progress_every,
    )
