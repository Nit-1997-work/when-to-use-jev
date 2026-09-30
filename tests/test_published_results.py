"""The published results must be exactly what the harness computes from the published records.

These run in CI without API keys: they read only files committed under results/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jevbench.data import load_split
from jevbench.metrics import load_results, summarize
from jevbench.report import build_report, load_manifests, system_prices
from jevbench.settings import REPO_ROOT

# USD per 1M tokens (input, output, thinking), copied from each run's price sheet on its run date. Kept here on
# purpose, separate from the manifests, so the cost check below does not trust the harness's own price lookup.
PUBLISHED_RUNS = {
    "pilot-20260929T205311Z": {
        "jev": (0.042, 0.0, 0.0),
        "baseline": (0.30, 2.50, 2.50),
        "baseline-conf": (0.30, 2.50, 2.50),
        "baseline-2": (0.75, 3.75, 3.75),
    },
}


def _run_dir(run_id: str) -> Path:
    path = REPO_ROOT / "results" / run_id
    if not (path / "report.md").exists():
        pytest.skip(f"published run {run_id} is not in this checkout")
    return path


@pytest.mark.parametrize("run_id", PUBLISHED_RUNS)
def test_published_report_matches_its_records(run_id: str) -> None:
    run_dir = _run_dir(run_id)
    assert build_report(run_dir) == (run_dir / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("run_id", PUBLISHED_RUNS)
def test_published_records_are_complete_and_match_the_splits(run_id: str) -> None:
    frame = load_results(_run_dir(run_id))
    assert len(frame) == len(frame.drop_duplicates(["system", "split", "repeat", "example_id"]))
    for (_, split), group in frame.groupby(["system", "split"]):
        gold = {example.id: example for example in load_split(str(split))}
        assert sorted(group["example_id"]) == sorted(gold)
        for example_id, text, intent in zip(group["example_id"], group["text"], group["expected_intent"], strict=True):
            assert (text, intent) == (gold[example_id].text, gold[example_id].intent)
        assert (group["correct"] == (group["predicted_intent"] == group["expected_intent"])).all()


@pytest.mark.parametrize(("run_id", "prices"), PUBLISHED_RUNS.items())
def test_headline_numbers_recomputed_independently(run_id: str, prices: dict[str, tuple[float, ...]]) -> None:
    run_dir = _run_dir(run_id)
    frame = load_results(run_dir)
    manifests = load_manifests(run_dir)
    for (system, split), group in frame.groupby(["system", "split"]):
        summary = summarize(group, system_prices(manifests, str(system), str(split)))
        rows = group.to_dict("records")
        answers = [r for r in rows if r["outcome"] in ("ok", "invalid_output")]
        timed = [float(r["latency_ms"]) for r in rows if r["outcome"] == "ok" and r["cache_hit"] is not True]
        price_in, price_out, price_think = prices[str(system)]
        costs = [
            (
                (r["input_tokens"] or 0) * price_in
                + (r["output_tokens"] or 0) * price_out
                + (r["reasoning_tokens"] or 0) * price_think
            )
            / 1e6
            for r in rows
        ]
        assert summary.accuracy == pytest.approx(sum(bool(r["correct"]) for r in answers) / len(answers))
        assert summary.blocked_rate == pytest.approx(sum(r["outcome"] == "refused" for r in rows) / len(rows))
        assert summary.latency_p50 == pytest.approx(float(np.percentile(timed, 50)))
        assert summary.latency_p95 == pytest.approx(float(np.percentile(timed, 95)))
        assert summary.cost_per_1k_usd == pytest.approx(1000 * sum(costs) / len(costs))
