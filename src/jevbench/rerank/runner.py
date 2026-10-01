"""Runs one re-ranking system over one split, one search at a time, writing one JSONL record per search.

Records are the source of truth: each carries the search's gold labels and the system's full ranking, so the report
never needs anything but the run directory.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from jevbench.common.calls import Outcome
from jevbench.common.paths import results_dir
from jevbench.common.runner import RunTarget, base_manifest, call_with_retries, now, run_records
from jevbench.common.settings import Settings
from jevbench.common.tracing import current_trace_id, example_span, record_result, tracer
from jevbench.rerank.data import SPLITS_DIR, Search, file_sha256, load_split
from jevbench.rerank.labels import LabelSet
from jevbench.rerank.metrics import ndcg_at_k
from jevbench.rerank.ranking import Ranking, Reranker
from jevbench.rerank.systems import prices_for

RESULTS_DIR = results_dir("rerank")
# Manifest fields that must not change while one results file is filled across several sessions.
RESUME_KEYS = ("system", "split", "repeat", "model_requested", "labels_sha256", "split_sha256", "settings")


def target(run_id: str, system: str, split: str, repeat: int) -> RunTarget:
    return RunTarget(root=RESULTS_DIR, run_id=run_id, system=system, split=split, repeat=repeat)


def build_manifest(target: RunTarget, settings: Settings, labels: LabelSet, reranker: Reranker) -> dict[str, object]:
    return base_manifest(
        target,
        model_requested=reranker.model_requested,
        data_hashes={
            "labels_sha256": labels.sha256,
            "split_sha256": file_sha256(SPLITS_DIR / f"{target.split}.jsonl"),
        },
        prices=prices_for(settings, target.system),
        settings=settings,
    )


async def rerank_with_retries(reranker: Reranker, search: Search) -> tuple[Ranking, int, float]:
    """Returns (ranking, attempts, total backoff seconds). Backoff is never part of `latency_ms`."""
    return await call_with_retries(
        lambda: reranker.rerank(search),
        lambda error: Ranking(outcome=Outcome.API_ERROR, latency_ms=0.0, calls=0, error=error),
    )


def build_record(
    target: RunTarget,
    reranker: Reranker,
    search: Search,
    ranking: Ranking,
    attempts: int,
    backoff_s: float,
    started_at: str,
    trace_id: str | None,
) -> dict[str, object]:
    measured = asdict(ranking)
    return {
        "run_id": target.run_id,
        "system": target.system,
        "split": target.split,
        "repeat": target.repeat,
        "example_id": search.id,
        "query_id": search.query_id,
        "query": search.query,
        "source": search.source,
        "grocery": search.grocery,
        "hard": search.hard,
        "should_abstain": search.should_abstain,
        "candidate_ids": [c.cid for c in search.candidates],
        "product_ids": [c.product_id for c in search.candidates],
        "gold_labels": search.labels,
        "model_requested": reranker.model_requested,
        "attempts": attempts,
        "backoff_s": round(backoff_s, 3),
        "started_at": started_at,
        "trace_id": trace_id,
        "outcome": ranking.outcome.value,
        "latency_ms": ranking.latency_ms,
        "calls": ranking.calls,
        "order": ranking.order,
        "scores": ranking.scores,
        "predicted_labels": ranking.labels,
        "p_exact": ranking.p_exact,
        "abstain_flag": ranking.abstain_flag,
        "exact_confidence": ranking.exact_confidence,
        "explanation": ranking.explanation,
        "invented_ids": ranking.invented_ids,
        "duplicate_ids": ranking.duplicate_ids,
        "missing_ids": ranking.missing_ids,
        "blocked_calls": ranking.blocked_calls,
        **{
            key: measured[key]
            for key in (
                "model_reported",
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "gateway_upstream_ms",
                "gateway_overhead_ms",
                "cache_hit",
                "error",
            )
        },
    }


def _top_label(record: dict[str, object]) -> str:
    order = record["order"]
    if not isinstance(order, list) or not order:
        return ""
    gold = dict(zip(record["candidate_ids"], record["gold_labels"], strict=True))  # pyright: ignore[reportArgumentType, reportCallIssue]
    return str(gold.get(order[0], ""))


async def warm_up(reranker: Reranker, count: int) -> None:
    """Discarded searches that open connections before timing. Traced under a `warmup` span."""
    with tracer().start_as_current_span("warmup", attributes={"jevbench.warmup": True}):
        for search in load_split("warmup")[:count]:
            await rerank_with_retries(reranker, search)


async def run_target(
    target: RunTarget,
    reranker: Reranker,
    settings: Settings,
    labels: LabelSet,
    *,
    limit: int | None,
    warmup: int,
    progress_every: int = 50,
) -> Path:
    searches = load_split(target.split)
    if limit is not None:
        searches = searches[:limit]

    async def run_one(search: Search) -> dict[str, object]:
        started_at = now()
        with example_span("rerank", target.system, search.id, target.split, target.repeat, search.query) as span:
            ranking, attempts, backoff_s = await rerank_with_retries(reranker, search)
            record = build_record(
                target, reranker, search, ranking, attempts, backoff_s, started_at, current_trace_id()
            )
            record_result(
                span,
                output=",".join(ranking.order[:3]),
                outcome=ranking.outcome.value,
                attributes={"jevbench.top_label": _top_label(record), "jevbench.calls": ranking.calls},
            )
        return record

    def progress(records: list[dict[str, object]]) -> str:
        ranked = [r for r in records if r["outcome"] in ("ok", "invalid_output") and not r["should_abstain"]]
        if not ranked:
            return f"answered={sum(r['outcome'] in ('ok', 'invalid_output') for r in records)}"
        scores = []
        for record in ranked:
            gold = dict(zip(record["candidate_ids"], record["gold_labels"], strict=True))  # pyright: ignore[reportArgumentType, reportCallIssue]
            scores.append(ndcg_at_k([str(gold[cid]) for cid in record["order"]], 3))  # pyright: ignore[reportGeneralTypeIssues, reportOptionalIterable]
        valid = [s for s in scores if s == s]  # drop NaN
        return f"ndcg@3={sum(valid) / len(valid):.3f}" if valid else "ndcg@3=-"

    return await run_records(
        target,
        searches,
        example_id=lambda s: s.id,
        manifest=lambda: build_manifest(target, settings, labels, reranker),
        resume_keys=RESUME_KEYS,
        warm_up=(lambda: warm_up(reranker, warmup)) if warmup else None,
        run_one=run_one,
        progress=progress,
        progress_every=progress_every,
    )
