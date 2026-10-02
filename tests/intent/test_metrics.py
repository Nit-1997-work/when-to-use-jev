from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from jevbench.common.settings import Prices
from jevbench.common.stats import (
    call_cost_usd,
    count,
    expected_calibration_error,
    load_results,
    macro_f1,
    mcnemar_exact,
    num,
)
from jevbench.intent.metrics import (
    blocked_counts,
    confidence_signal,
    coverage_at_accuracy,
    paired_comparison,
    summarize,
    top_confusions,
)

if TYPE_CHECKING:
    from tests.conftest import RecordFactory


def _frame(make_record: RecordFactory, *records: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame([make_record(example_id=f"e{i}", **r) for i, r in enumerate(records)])


def test_num_and_count_coerce_safely() -> None:
    assert num("1.5") == 1.5
    assert math.isnan(num(None))
    assert math.isnan(num("x"))
    assert count(np.int64(3)) == 3
    assert count(None) == 0


def test_macro_f1_perfect_and_partial() -> None:
    expected = pd.Series(["a", "a", "b", "b"])
    assert macro_f1(expected, expected) == 1.0
    predicted = pd.Series(["a", "b", "b", "b"])
    # a: tp=1 fp=0 fn=1 -> 2/3 ; b: tp=2 fp=1 fn=0 -> 4/5
    assert macro_f1(expected, predicted) == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_macro_f1_scores_only_intents_present_in_the_sample() -> None:
    # "c" is predicted but never expected: it adds a false positive nowhere, and gets no F1 of its own.
    assert macro_f1(pd.Series(["a", "b"]), pd.Series(["a", "c"])) == pytest.approx((1.0 + 0.0) / 2)


def test_ece_is_zero_when_confidence_matches_accuracy() -> None:
    confidence = np.array([0.75] * 4)
    correct = np.array([True, True, True, False])
    assert expected_calibration_error(confidence, correct) == pytest.approx(0.0)


def test_ece_detects_overconfidence() -> None:
    confidence = np.array([1.0] * 4)
    correct = np.array([True, False, True, False])
    assert expected_calibration_error(confidence, correct) == pytest.approx(0.5)


def test_coverage_takes_the_most_confident_answers_first() -> None:
    confidence = np.array([0.99, 0.95, 0.9, 0.5])
    correct = np.array([True, True, True, False])
    assert coverage_at_accuracy(confidence, correct, 0.99) == pytest.approx(0.75)
    assert coverage_at_accuracy(confidence, correct, 0.5) == pytest.approx(1.0)


def test_coverage_never_splits_a_group_of_equal_confidence() -> None:
    # Three answers say 0.95; one of them is wrong. A threshold cannot keep two of them and drop the third,
    # so reaching 90% accuracy is only possible at the 1.0 group: 1 of 5 answers.
    confidence = np.array([1.0, 0.95, 0.95, 0.95, 0.9])
    correct = np.array([True, True, True, False, True])
    assert coverage_at_accuracy(confidence, correct, 0.9) == pytest.approx(0.2)


def test_coverage_counts_answers_without_confidence_in_the_total() -> None:
    confidence = np.array([0.9, 0.8])
    correct = np.array([True, True])
    assert coverage_at_accuracy(confidence, correct, 0.95, total=4) == pytest.approx(0.5)
    assert coverage_at_accuracy(np.array([]), np.array([], dtype=bool), 0.95, total=3) == 0.0
    assert math.isnan(coverage_at_accuracy(np.array([]), np.array([], dtype=bool), 0.95))


def test_confidence_signal_prefers_the_top_probability(make_record: RecordFactory) -> None:
    jev = _frame(make_record, {"probabilities": {"a": 0.7, "b": 0.3}, "confidence": 0.5})
    signal = confidence_signal(jev)
    assert signal is not None
    assert signal[1] == "top probability"
    assert signal[0].tolist() == [0.7]

    llm = _frame(make_record, {"confidence": 0.9})
    signal = confidence_signal(llm)
    assert signal is not None
    assert signal[1] == "self-reported"
    assert confidence_signal(_frame(make_record, {})) is None


def test_mcnemar_counts_discordant_pairs() -> None:
    a = np.array([True, True, True, False, True])
    b = np.array([True, False, False, False, True])
    result = mcnemar_exact(a, b)
    assert (result["a_only"], result["b_only"]) == (2, 0)
    assert result["p_value"] == pytest.approx(0.5)


def test_cost_is_none_when_prices_missing(make_record: RecordFactory) -> None:
    assert call_cost_usd(_frame(make_record, {}), Prices(input=None, output=None, thinking=None)) is None


def test_cost_counts_input_output_and_reasoning(make_record: RecordFactory) -> None:
    frame = _frame(make_record, {"reasoning_tokens": 100})
    cost = call_cost_usd(frame, Prices(input=0.1, output=0.4, thinking=None))  # thinking billed at output rate
    assert cost is not None
    assert cost.iloc[0] == pytest.approx((1000 * 0.1 + 10 * 0.4 + 100 * 0.4) / 1_000_000)


def test_blocked_calls_are_left_out_of_quality_metrics_but_billed(make_record: RecordFactory) -> None:
    frame = _frame(
        make_record,
        {},
        {"outcome": "refused", "predicted_intent": None, "predicted_category": None, "correct": False,
         "category_correct": False, "input_tokens": 3000, "output_tokens": 0, "latency_ms": 5.0},
    )  # fmt: skip
    summary = summarize(frame, Prices(input=1.0, output=0.0, thinking=None))
    assert (summary.n, summary.answered) == (2, 1)
    assert summary.accuracy == 1.0
    assert summary.macro_f1 == 1.0
    assert summary.category_accuracy == 1.0
    assert summary.blocked_rate == 0.5
    assert summary.latency_n == 1
    assert summary.latency_p50 == 100.0  # the blocked call is not timed
    # A block still bills the prompt it read, so cost averages over every call.
    assert summary.cost_per_1k_usd == pytest.approx((1000 + 3000) / 2 / 1_000_000 * 1000)


def test_api_errors_are_left_out_and_invalid_answers_count_as_wrong(make_record: RecordFactory) -> None:
    frame = _frame(
        make_record,
        {},
        {"outcome": "api_error", "predicted_intent": None, "correct": False, "input_tokens": None},
        {"outcome": "invalid_output", "predicted_intent": None, "predicted_category": None, "correct": False,
         "category_correct": False},
    )  # fmt: skip
    summary = summarize(frame, None)
    assert summary.answered == 2
    assert summary.accuracy == 0.5
    assert summary.api_error_rate == pytest.approx(1 / 3)
    assert summary.invalid_rate == 0.5
    assert summary.cost_per_1k_usd is None


def test_missing_cache_flag_is_not_a_cache_hit(make_record: RecordFactory) -> None:
    frame = _frame(make_record, {"cache_hit": None, "latency_ms": 100.0}, {"cache_hit": True, "latency_ms": 1.0})
    summary = summarize(frame, None)
    assert summary.cache_hits == 1
    assert summary.latency_n == 1
    assert summary.latency_p50 == 100.0


def test_summary_confidence_uses_answers_without_a_signal_in_the_denominator(make_record: RecordFactory) -> None:
    frame = _frame(
        make_record,
        {"confidence": 1.0},
        {"confidence": None},
        {"outcome": "refused", "predicted_intent": None, "correct": False},
    )
    summary = summarize(frame, None)
    assert summary.confidence_signal == "self-reported"
    assert summary.confidence_n == 1
    assert summary.coverage_at_98 == pytest.approx(0.5)  # 1 of 2 answered; the block is not an answer


def test_top2_accuracy_uses_answered_calls(make_record: RecordFactory) -> None:
    frame = _frame(
        make_record,
        {"top2": ["track_order", "track_delivery"]},
        {"top2": ["track_delivery", "cancel_order"], "predicted_intent": "track_delivery", "correct": False},
    )
    assert summarize(frame, None).top2_accuracy == 0.5


def test_load_results_keeps_the_last_record_per_example(tmp_path: Path, make_record: RecordFactory) -> None:
    path = tmp_path / "run" / "jev" / "main" / "repeat-1.jsonl"
    path.parent.mkdir(parents=True)
    rows = [make_record(outcome="api_error", correct=False), make_record(), '{"torn']
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows), encoding="utf-8")
    frame = load_results(tmp_path / "run")
    assert len(frame) == 1
    assert frame["outcome"].tolist() == ["ok"]


def test_load_results_rejects_an_empty_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path)


def test_top_confusions_skip_blocks_and_sort_ties_by_name(make_record: RecordFactory) -> None:
    wrong = {"correct": False}
    frame = _frame(
        make_record,
        {**wrong, "expected_intent": "b", "predicted_intent": "a"},
        {**wrong, "expected_intent": "a", "predicted_intent": "b"},
        {**wrong, "expected_intent": "a", "predicted_intent": None, "outcome": "invalid_output"},
        {**wrong, "expected_intent": "a", "predicted_intent": None, "outcome": "refused"},
    )
    confusions = top_confusions(frame)
    assert list(zip(confusions["expected_intent"], confusions["predicted"], strict=True)) == [
        ("a", "__invalid__"),
        ("a", "b"),
        ("b", "a"),
    ]


def test_blocked_counts_split_clean_and_profane(make_record: RecordFactory) -> None:
    frame = _frame(
        make_record,
        {"has_profanity": False, "outcome": "refused"},
        {"has_profanity": False},
        {"has_profanity": True, "outcome": "refused"},
    )
    assert blocked_counts(frame) == {False: (1, 2), True: (1, 1)}


def test_paired_comparison_uses_messages_both_systems_answered(make_record: RecordFactory) -> None:
    records = [
        make_record(system="jev", example_id="e1"),
        make_record(system="jev", example_id="e2"),
        make_record(system="gemini", example_id="e1", correct=False),
        make_record(system="gemini", example_id="e2", outcome="refused", correct=False),
    ]
    result = paired_comparison(pd.DataFrame(records), "jev", "gemini", "main")
    assert result is not None
    assert result["n"] == 1
    assert (result["a_only"], result["b_only"]) == (1, 0)
    assert paired_comparison(pd.DataFrame(records), "jev", "absent", "main") is None
