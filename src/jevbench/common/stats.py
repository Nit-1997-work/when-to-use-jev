"""Statistics shared by every experiment's report. Pure functions on numpy arrays and pandas frames.

Conventions:
- A call is answered when the system returned an answer, even a malformed one (`invalid_output`, which counts as
  wrong). Blocked calls (the provider's safety filter withheld the answer) and API errors are not answers.
- Latency uses successful calls that were not served from a cache.
- Cost averages over every call made, because providers bill blocked calls too.
- Numbers are coerced through `num()`/`count()`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from jevbench.common.settings import Prices

ANSWERED_OUTCOMES = ("ok", "invalid_output")
BLOCKED = "refused"  # stored outcome value for a safety-filter block
API_ERROR = "api_error"
RECORD_KEY = ("run_id", "system", "split", "repeat", "example_id")


def num(value: object) -> float:
    """The one place numbers are coerced to Python floats. Non-numeric or missing values become NaN."""
    try:
        return float(value)  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError):
        return math.nan


def count(value: object) -> int:
    """Integer counts; anything non-numeric counts as 0."""
    number = num(value)
    return 0 if math.isnan(number) else round(number)


def load_results(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(run_dir.rglob("repeat-*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # torn last line from an interrupted run
    if not rows:
        raise FileNotFoundError(f"No result records under {run_dir}.")
    frame = pd.DataFrame(rows)
    # A resumed run appends a new record for each example it redid (API errors); the last record is the final one.
    key = [column for column in RECORD_KEY if column in frame.columns]
    return frame.drop_duplicates(subset=key, keep="last").reset_index(drop=True)


def answered(frame: pd.DataFrame) -> pd.Series:
    """Calls where the system returned an answer (right, wrong, or malformed)."""
    return frame["outcome"].isin(ANSWERED_OUTCOMES)


def served_from_cache(frame: pd.DataFrame) -> pd.Series:
    """Cache hits. A missing flag is not a cache hit."""
    return frame["cache_hit"].eq(True)


def bootstrap_ci(values: np.ndarray, *, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of 0/1 (or real) values."""
    if len(values) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return (num(np.quantile(means, alpha / 2)), num(np.quantile(means, 1 - alpha / 2)))


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width bins over [0, 1]."""
    if len(confidence) == 0:
        return math.nan
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += num(mask.mean()) * abs(num(correct[mask].mean()) - num(confidence[mask].mean()))
    return ece


def macro_f1(expected: pd.Series, predicted: pd.Series) -> float:
    """Unweighted mean of per-label F1 over the labels that occur in `expected`.

    Labels absent from the evaluated set are skipped: they have no support, so their F1 is undefined.
    """
    scores: list[float] = []
    for label in sorted(set(expected)):
        tp = count(((expected == label) & (predicted == label)).sum())
        fp = count(((expected != label) & (predicted == label)).sum())
        fn = count(((expected == label) & (predicted != label)).sum())
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return sum(scores) / len(scores) if scores else math.nan


def mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray) -> dict[str, float]:
    """Paired test on the same examples: do A and B differ in accuracy?"""
    a_only = count(np.sum(a_correct & ~b_correct))
    b_only = count(np.sum(~a_correct & b_correct))
    n = a_only + b_only
    p = binomtest(a_only, n, 0.5).pvalue if n else 1.0
    return {"a_only": a_only, "b_only": b_only, "p_value": num(p)}


def call_cost_usd(frame: pd.DataFrame, prices: Prices) -> pd.Series | None:
    """Per-call USD cost, or None when a needed price is not configured (never silently 0)."""
    needs_output = num(frame["output_tokens"].fillna(0).sum()) > 0
    needs_thinking = num(frame["reasoning_tokens"].fillna(0).sum()) > 0
    if prices.input is None or (needs_output and prices.output is None):
        return None
    if needs_thinking and prices.effective_thinking is None:
        return None
    return (
        frame["input_tokens"].fillna(0) * prices.input
        + frame["output_tokens"].fillna(0) * (prices.output or 0.0)
        + frame["reasoning_tokens"].fillna(0) * (prices.effective_thinking or 0.0)
    ) / 1_000_000
