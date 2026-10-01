"""Record, search, and run-directory factories shared by the re-ranking tests."""

from __future__ import annotations

import json
from pathlib import Path

from jevbench.rerank.data import Candidate, Search


def rerank_record(
    *,
    system: str = "jev-score",
    split: str = "main",
    repeat: int = 1,
    example_id: str = "q1",
    gold: list[str] | None = None,
    order: list[str] | None = None,
    outcome: str = "ok",
    should_abstain: bool = False,
    product_ids: list[str] | None = None,
    **overrides: object,
) -> dict[str, object]:
    """One re-ranking result record as the runner writes it: by default a perfect ranking of E, S, I."""
    gold = gold or ["E", "S", "I"]
    cids = [f"c{i}" for i in range(1, len(gold) + 1)]
    record: dict[str, object] = {
        "run_id": "run",
        "system": system,
        "split": split,
        "repeat": repeat,
        "example_id": example_id,
        "query_id": 1,
        "query": "sesame chips",
        "source": "other",
        "grocery": False,
        "hard": False,
        "should_abstain": should_abstain,
        "candidate_ids": cids,
        "product_ids": product_ids or [f"P{i}" for i in range(1, len(gold) + 1)],
        "gold_labels": gold,
        "model_requested": "model-x",
        "attempts": 1,
        "backoff_s": 0.0,
        "started_at": "2026-10-01T20:00:00.000+00:00",
        "trace_id": None,
        "outcome": outcome,
        "latency_ms": 100.0,
        "calls": 1,
        "order": cids if order is None else order,
        "scores": None,
        "predicted_labels": None,
        "p_exact": None,
        "abstain_flag": None,
        "exact_confidence": None,
        "explanation": None,
        "invented_ids": 0,
        "duplicate_ids": 0,
        "missing_ids": 0,
        "blocked_calls": 0,
        "model_reported": "model-x",
        "input_tokens": 1000,
        "output_tokens": 10,
        "reasoning_tokens": 0,
        "gateway_upstream_ms": None,
        "gateway_overhead_ms": None,
        "cache_hit": False,
        "error": None,
    }
    record.update(overrides)
    return record


def make_search(labels: str = "ESCI", *, query: str = "sesame chips", search_id: str = "q1") -> Search:
    candidates = tuple(
        Candidate(cid=f"c{i}", product_id=f"P{i}", title=f"Product {i} {label}", brand="Brand", color=None, label=label)
        for i, label in enumerate(labels, start=1)
    )
    return Search(
        id=search_id,
        query_id=int(search_id.lstrip("q") or 1),
        query=query,
        source="other",
        grocery=False,
        hard=False,
        should_abstain="E" not in labels,
        exact_count=labels.count("E"),
        split="practice",
        candidates=candidates,
    )


def write_rerank_run(root: Path, records: list[dict[str, object]], run_id: str = "run") -> Path:
    """Records plus one manifest per (system, split, repeat), as a run directory."""
    run_dir = root / run_id
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for record in records:
        groups.setdefault((str(record["system"]), str(record["split"]), int(str(record["repeat"]))), []).append(record)
    for (system, split, repeat), rows in groups.items():
        path = run_dir / system / split / f"repeat-{repeat}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        manifest = {
            "run_id": run_id,
            "system": system,
            "split": split,
            "repeat": repeat,
            "model_requested": "model-x",
            "written_at": "2026-10-01T20:00:00.000+00:00",
            "labels_sha256": "l" * 64,
            "split_sha256": "s" * 64,
            "prices_per_mtok": {"input": 1.0, "output": 2.0, "thinking": 2.0},
            "settings": {"client_location": "test bench", "request_timeout_s": 30.0},
            "git": {"commit": "0123456789abcdef", "dirty": False},
            "python": "3.12.0",
            "packages": {"openai": "1.0"},
        }
        path.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir
