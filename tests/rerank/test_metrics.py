from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from jevbench.common.settings import Prices
from jevbench.rerank.labels import GAIN, GENTLE_GAIN
from jevbench.rerank.metrics import (
    abstained,
    abstention,
    calibration,
    cascade,
    consistency,
    dcg,
    head_to_head,
    label_quality,
    ndcg_at_k,
    order_sensitivity,
    paired_permutation_p,
    quality,
    reciprocal_rank,
    summarize,
    tune_threshold,
)
from tests.rerank.helpers import rerank_record

# "sesame chips" from ESCI: 1 Exact, 9 Substitutes, 6 Irrelevant. Ideal top 3: E, S, S.
SESAME = ["E"] + ["S"] * 9 + ["I"] * 6


def _ranked(top: list[str]) -> list[str]:
    rest = list(SESAME)
    for label in top:
        rest.remove(label)
    return top + rest


@pytest.mark.parametrize(
    ("top", "expected"),
    [
        (["E", "S", "S"], 1.0),
        (["S", "E", "S"], (0.1 + 1 / math.log2(3) + 0.05) / (1 + 0.1 / math.log2(3) + 0.05)),
        (["S", "S", "S"], (0.1 + 0.1 / math.log2(3) + 0.05) / (1 + 0.1 / math.log2(3) + 0.05)),
        (["I", "I", "S"], 0.05 / (1 + 0.1 / math.log2(3) + 0.05)),
    ],
)
def test_ndcg_at_3_matches_hand_computed_sesame_chips(top: list[str], expected: float) -> None:
    assert ndcg_at_k(_ranked(top), 3) == pytest.approx(expected)


def test_ndcg_hand_computed_lavazza_coffee_machine() -> None:
    # 3 Exact, 5 Substitute, 8 Complement. System A: E, C, S. System B: E, E, S. Ideal: E, E, E (2.131).
    ideal = 1 + 1 / math.log2(3) + 0.5
    assert ndcg_at_k(["E", "C", "S"] + ["E", "E"] + ["S"] * 4 + ["C"] * 7, 3) == pytest.approx(
        (1 + 0.01 / math.log2(3) + 0.05) / ideal
    )
    assert ndcg_at_k(["E", "E", "S"] + ["E"] + ["S"] * 4 + ["C"] * 8, 3) == pytest.approx(
        (1 + 1 / math.log2(3) + 0.05) / ideal
    )


def test_gains_and_edge_cases() -> None:
    assert GAIN == {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
    assert math.isnan(ndcg_at_k(["I", "I"], 3))  # nothing to gain: undefined, never 0 or 1
    assert dcg(["E"], 3) == 1.0
    assert ndcg_at_k(["S", "E"], 3, GENTLE_GAIN) < ndcg_at_k(["E", "S"], 3, GENTLE_GAIN)
    assert reciprocal_rank(["S", "C", "E"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["S"]) == 0.0


def _frame(*records: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(records))


def test_quality_uses_answered_searches_with_an_exact_candidate() -> None:
    frame = _frame(
        rerank_record(example_id="a", order=["c1", "c2", "c3"], gold=["E", "S", "I"]),
        rerank_record(example_id="b", order=["c3", "c2", "c1"], gold=["E", "S", "I"]),
        rerank_record(example_id="nomatch", gold=["S", "I", "I"], should_abstain=True),
        rerank_record(example_id="blocked", outcome="refused", order=[]),
    )
    q = quality(frame)
    assert list(q["example_id"]) == ["a", "b"]
    assert list(q["exact1"]) == [True, False]
    assert q.loc[0, "ndcg3"] == pytest.approx(1.0)


def test_abstention_uses_flag_or_threshold() -> None:
    assert abstained({"abstain_flag": True, "exact_confidence": 0.99}, 0.5) is True
    assert abstained({"abstain_flag": None, "exact_confidence": 0.2}, 0.5) is True
    assert abstained({"abstain_flag": None, "exact_confidence": 0.8}, 0.5) is False
    assert abstained({"abstain_flag": None, "exact_confidence": None}, 0.5) is False

    frame = _frame(
        rerank_record(example_id="n1", should_abstain=True, exact_confidence=0.1, gold=["S", "I", "I"]),
        rerank_record(example_id="n2", should_abstain=True, exact_confidence=0.7, gold=["S", "I", "I"]),
        rerank_record(example_id="m1", exact_confidence=0.9),
        rerank_record(example_id="m2", exact_confidence=0.3),
    )
    result = abstention(frame, 0.5)
    assert (result["precision"], result["recall"], result["false_rate"]) == (0.5, 0.5, 0.5)
    threshold, f1 = tune_threshold(frame)  # pyright: ignore[reportGeneralTypeIssues]
    assert f1 == pytest.approx(abstention(frame, threshold)["f1"])
    assert f1 >= result["f1"]


def test_tuning_needs_searches_that_should_abstain() -> None:
    assert tune_threshold(_frame(rerank_record(exact_confidence=0.4))) is None


def test_label_quality_and_calibration() -> None:
    frame = _frame(
        rerank_record(
            gold=["E", "S", "I"],
            predicted_labels={"c1": "E", "c2": "E", "c3": "I"},
            p_exact={"c1": 0.9, "c2": 0.6, "c3": 0.1},
        )
    )
    result = label_quality(frame)
    assert result is not None
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["substitute_as_exact"] == 1.0
    n, ece = calibration(frame)  # pyright: ignore[reportGeneralTypeIssues]
    assert n == 3
    assert ece == pytest.approx((0.1 + 0.6 + 0.1) / 3)
    assert label_quality(_frame(rerank_record())) is None


def test_consistency_and_order_sensitivity() -> None:
    frame = _frame(
        rerank_record(split="consistency", repeat=1, order=["c1", "c2", "c3"], product_ids=["A", "B", "C"]),
        rerank_record(split="consistency", repeat=2, order=["c1", "c3", "c2"], product_ids=["A", "B", "C"]),
        rerank_record(split="consistency", repeat=3, order=["c2", "c1", "c3"], product_ids=["A", "B", "C"]),
        rerank_record(split="reversed", repeat=1, order=["c3", "c2", "c1"], product_ids=["C", "B", "A"]),
    )
    stable = consistency(frame)
    assert stable is not None
    assert stable["pairs"] == 3
    assert stable["top1_agreement"] == pytest.approx(1 / 3)
    reversed_ = order_sensitivity(frame)
    assert reversed_ == {"searches": 1, "top1_same": 1.0}  # c3 in reversed order is product A
    assert consistency(frame[frame["split"] == "reversed"]) is None


def test_head_to_head_and_permutation_test() -> None:
    a = quality(_frame(*[rerank_record(example_id=f"q{i}", order=["c1", "c2", "c3"]) for i in range(5)]))
    b = quality(_frame(*[rerank_record(example_id=f"q{i}", order=["c3", "c2", "c1"]) for i in range(5)]))
    result = head_to_head(a, b)
    assert result is not None
    assert result["ndcg3_diff"] > 0
    assert (result["a_only"], result["b_only"]) == (5, 0)
    assert paired_permutation_p(np.ones(3), np.ones(3)) == 1.0
    assert paired_permutation_p(np.ones(30), np.zeros(30)) < 0.001


def test_cascade_passes_least_confident_searches_to_the_llm() -> None:
    jev = _frame(
        rerank_record(example_id="sure", order=["c1", "c2", "c3"], scores={"c1": 0.9, "c2": 0.1, "c3": 0.0}),
        rerank_record(example_id="unsure", order=["c3", "c1", "c2"], scores={"c3": 0.5, "c1": 0.49, "c2": 0.0}),
    )
    llm = _frame(
        rerank_record(example_id="sure", order=["c1", "c2", "c3"]),
        rerank_record(example_id="unsure", order=["c1", "c2", "c3"]),
    )
    points = cascade(jev, quality(jev), quality(llm), pd.Series([0.001, 0.001]), pd.Series([0.01, 0.01]), (0, 0.5, 1))
    assert [p["share"] for p in points] == [0, 0.5, 1]
    assert points[0]["ndcg3"] < points[1]["ndcg3"] == points[2]["ndcg3"] == pytest.approx(1.0)
    assert points[1]["cost_per_1k"] == pytest.approx((0.001 + 0.5 * 0.01) * 1000)


def test_summary_counts_reliability_and_cost() -> None:
    frame = _frame(
        rerank_record(example_id="a", input_tokens=1000, output_tokens=100, invented_ids=1),
        rerank_record(example_id="b", outcome="invalid_output", order=["c1", "c2", "c3"]),
        rerank_record(example_id="c", outcome="api_error", order=[]),
    )
    s = summarize(frame, Prices(1.0, 2.0, 2.0))
    assert (s.n, s.answered, s.ranked) == (3, 2, 2)
    assert s.invalid_rate == 0.5
    assert s.api_error_rate == pytest.approx(1 / 3)
    assert s.invented_ids == 1
    assert s.cost_per_1k_usd is not None
    assert summarize(frame, None).cost_per_1k_usd is None
