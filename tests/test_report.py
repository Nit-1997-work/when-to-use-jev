from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from jevbench.report import ReportError, build_report, prices_from_manifest, system_prices
from jevbench.settings import LLM_SAFETY_SETTINGS, Prices

if TYPE_CHECKING:
    from tests.conftest import RecordFactory, RunWriter

JEV_PROBABILITIES = {"track_order": 0.9, "track_delivery": 0.1}
WRONG = {"predicted_intent": "track_delivery", "predicted_category": "DELIVERY", "correct": False}
BLOCKED = {
    "outcome": "refused",
    "predicted_intent": None,
    "predicted_category": None,
    "correct": False,
    "category_correct": False,
    "error": "finish_reason=content_filter",
}


def _run(make_record: RecordFactory, write_run: RunWriter, *, drop_last_baseline: bool = False) -> Path:
    """Jev gets 20 clean messages right; the baseline gets them wrong, and its safety filter blocks a profane one."""
    records: list[dict[str, object]] = []
    for i in range(20):
        records.append(
            make_record(system="jev", example_id=f"e{i}", probabilities=JEV_PROBABILITIES, top2=list(JEV_PROBABILITIES))
        )
        records.append(make_record(system="baseline", example_id=f"e{i}", latency_ms=800.0, **WRONG))
    records.append(make_record(system="jev", example_id="p1", has_profanity=True, probabilities=JEV_PROBABILITIES))
    if not drop_last_baseline:
        records.append(make_record(system="baseline", example_id="p1", has_profanity=True, **BLOCKED))
    return write_run(records)


def test_blocked_messages_are_excluded_and_noted(make_record: RecordFactory, write_run: RunWriter) -> None:
    report = build_report(_run(make_record, write_run))
    assert "| jev | 21 of 21 | 0.0% | 0.0% | 100.0%" in report
    assert "| baseline | 20 of 21 | 4.8% | 0.0% | 0.0%" in report  # accuracy over the 20 answered messages
    assert "| jev | 0 of 20 (0.0%) | 0 of 1 (0.0%) |" in report
    assert "| baseline | 0 of 20 (0.0%) | 1 of 1 (100.0%) |" in report
    assert "| jev vs baseline | 20 | 100.0% | 0.0% | 20 | 0 | <0.0001 |" in report
    assert "nan" not in report


def test_costs_use_the_prices_recorded_in_the_manifests(make_record: RecordFactory, write_run: RunWriter) -> None:
    records = [make_record(system="jev"), make_record(system="baseline")]
    manifests = {
        ("jev", "main", 1): {
            "system": "jev",
            "split": "main",
            "repeat": 1,
            "model_requested": "jev-x",
            "prices_per_mtok": {"input": 0.042, "output": 0.0, "thinking": 0.0},
        },
        ("baseline", "main", 1): {
            "system": "baseline",
            "split": "main",
            "repeat": 1,
            "model_requested": "llm-x",
            "prices_per_mtok": {"input": 1.0, "output": 2.0, "thinking": 2.0},
        },
    }
    report = build_report(write_run(records, manifests=manifests))
    assert "$0.0420 | $42.00 |" in report  # 1,000 input tokens at $0.042 per 1M; output is free
    assert "$1.0200 | $1,020.00 |" in report  # 1,000 in at $1 + 10 out at $2 per 1M
    assert "| jev | jev-x | model-x | - | $0.042 / $0.00 / $0.00 |" in report
    assert "| baseline | llm-x | model-x | provider default | $1.00 / $2.00 / $2.00 |" in report


def test_run_conditions_are_reported(make_record: RecordFactory, write_run: RunWriter) -> None:
    report = build_report(_run(make_record, write_run))
    assert "## Run conditions" in report
    assert f"- LLM safety settings: {LLM_SAFETY_SETTINGS}." in report
    assert "- Client location: test bench. Request timeout: 30.0 s." in report
    assert "- Code: commit 0123456." in report
    assert "| jev | main | 21 | 2026-09-29 20:00 to 2026-09-29 20:00 |" in report
    assert "runs overlapped in time" in report  # both systems share one timestamp in this fixture


def test_uncommitted_code_is_flagged(make_record: RecordFactory, write_run: RunWriter) -> None:
    manifest = {"system": "jev", "split": "main", "repeat": 1, "git": {"commit": "abcdef0123", "dirty": True}}
    report = build_report(write_run([make_record()], manifests={("jev", "main", 1): manifest}))
    assert "- Code: commit abcdef0 (with uncommitted changes)." in report


def test_systems_that_ran_different_examples_are_not_compared(make_record: RecordFactory, write_run: RunWriter) -> None:
    run_dir = _run(make_record, write_run, drop_last_baseline=True)
    with pytest.raises(ReportError, match="ran different examples"):
        build_report(run_dir)
    assert "> Warning: partial run. main repeat 1" in build_report(run_dir, allow_partial=True)


def test_repeats_are_flagged(make_record: RecordFactory, write_run: RunWriter) -> None:
    report = build_report(write_run([make_record(repeat=1), make_record(repeat=2)]))
    assert "> Note: this run has up to 2 repeats" in report


def test_a_system_with_no_answers_prints_dashes(make_record: RecordFactory, write_run: RunWriter) -> None:
    report = build_report(write_run([make_record(system="jev"), make_record(system="baseline", **BLOCKED)]))
    assert "| baseline | 0 of 1 | 100.0% | 0.0% | - |" in report
    assert "nan" not in report


def test_report_is_deterministic(make_record: RecordFactory, write_run: RunWriter) -> None:
    run_dir = _run(make_record, write_run)
    assert build_report(run_dir) == build_report(run_dir)


@pytest.mark.parametrize(
    ("manifest", "expected"),
    [
        ({"system": "jev", "prices_per_mtok": {"input": 0.042, "output": 0, "thinking": 0}}, Prices(0.042, 0.0, 0.0)),
        # Older manifests only kept prices inside `settings`.
        ({"system": "jev", "settings": {"jev_input_price_per_mtok": 0.042}}, Prices(0.042, 0.0, 0.0)),
        (
            {"system": "baseline-conf", "settings": {"baseline": {"price_per_mtok": {"input": 0.3, "output": 2.5}}}},
            Prices(0.3, 2.5, None),
        ),
        (
            {"system": "baseline-2", "settings": {"baseline_2": {"price_per_mtok": {"input": 0.75, "output": 3.75}}}},
            Prices(0.75, 3.75, None),
        ),
        ({"system": "baseline", "settings": {}}, None),
    ],
)
def test_prices_from_manifest(manifest: dict[str, object], expected: Prices | None) -> None:
    assert prices_from_manifest(manifest) == expected


def test_repeats_with_different_prices_are_rejected() -> None:
    manifests: dict[tuple[str, str, int], dict[str, object]] = {
        ("jev", "main", 1): {"system": "jev", "prices_per_mtok": {"input": 0.042}},
        ("jev", "main", 2): {"system": "jev", "prices_per_mtok": {"input": 0.05}},
    }
    with pytest.raises(ReportError, match="different prices"):
        system_prices(manifests, "jev", "main")
