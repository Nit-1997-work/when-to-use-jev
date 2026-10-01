"""Intent-classification metrics over result records. Pure functions on pandas frames, unit-testable offline.

Conventions:
- Quality metrics (accuracy, macro-F1, category accuracy, top-2, calibration, coverage) use answered calls only.
  Blocked calls (the provider's safety filter withheld the answer) and API errors are excluded and counted
  separately. An answer outside the intent list (`invalid_output`) is an answer, and it counts as wrong.
- Latency uses successful calls that were not served from a cache.
- Cost averages over every call made, because providers bill blocked calls too.
- Numbers are coerced through `num()`/`count()`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from jevbench.common.settings import Prices
from jevbench.common.stats import (
    API_ERROR,
    BLOCKED,
    answered,
    bootstrap_ci,
    call_cost_usd,
    count,
    expected_calibration_error,
    macro_f1,
    mcnemar_exact,
    num,
    served_from_cache,
)

INVALID_LABEL = "__invalid__"  # stands in for an answer outside the intent list


# ---------- building blocks ----------


def coverage_at_accuracy(
    confidence: np.ndarray, correct: np.ndarray, target: float, *, total: int | None = None
) -> float:
    """Largest share of `total` answers a system can act on automatically while keeping accuracy >= target.

    Acting means taking every answer whose confidence is at or above one threshold. Answers with equal confidence
    cannot be told apart, so a threshold only ever falls between groups of equal confidence. `total` defaults to the
    number of answers given; pass a larger total when some answers have no confidence and so can never be acted on.
    """
    total = len(confidence) if total is None else total
    if total == 0:
        return math.nan
    if len(confidence) == 0:
        return 0.0
    order = np.argsort(-confidence, kind="stable")
    ranked, hits = confidence[order], correct[order].astype(np.float64)
    group_ends = np.flatnonzero(np.r_[ranked[1:] != ranked[:-1], True])  # last index of each equal-confidence group
    acted_on = group_ends + 1
    accuracy = np.cumsum(hits)[group_ends] / acted_on
    reachable = acted_on[accuracy >= target]
    return num(reachable.max()) / total if len(reachable) else 0.0


def confidence_signal(frame: pd.DataFrame) -> tuple[pd.Series, str] | None:
    """The per-call confidence used for calibration and coverage, with its label.

    Jev returns a probability for every intent, so its signal is the probability of the chosen intent (the standard
    top-label definition). LLM systems get no probabilities through the gateway; `-conf` systems self-report one.
    """
    if "probabilities" in frame.columns and frame["probabilities"].notna().any():
        top = frame["probabilities"].map(lambda p: max(p.values()) if isinstance(p, dict) and p else math.nan)
        return top.astype(np.float64), "top probability"
    if "confidence" in frame.columns and frame["confidence"].notna().any():
        return frame["confidence"].astype(np.float64), "self-reported"
    return None


# ---------- per-system summary ----------


@dataclass
class Summary:
    system: str
    split: str
    n: int  # calls made
    answered: int
    repeats: int
    accuracy: float  # over answered calls
    accuracy_ci: tuple[float, float]
    macro_f1: float
    category_accuracy: float
    top2_accuracy: float | None
    blocked_rate: float  # of all calls
    api_error_rate: float  # of all calls
    invalid_rate: float  # of answered calls
    cache_hits: int
    latency_n: int
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_mean: float
    gateway_p50: float | None
    input_tokens_mean: float
    output_tokens_mean: float
    reasoning_tokens_mean: float
    cost_per_1k_usd: float | None
    confidence_signal: str | None
    confidence_n: int  # answered calls with a usable confidence
    ece: float | None
    coverage_at_95: float | None
    coverage_at_98: float | None


def _rate(mask: pd.Series) -> float:
    return num(mask.mean()) if len(mask) else math.nan


def summarize(frame: pd.DataFrame, prices: Prices | None) -> Summary:
    is_answered = answered(frame)
    answers = frame[is_answered]
    ok = frame["outcome"] == "ok"
    timed = frame[ok & ~served_from_cache(frame)]
    latency = timed["latency_ms"].to_numpy(dtype=np.float64)
    hits = answers["correct"].to_numpy(dtype=bool)

    top2_accuracy = None
    if "top2" in answers.columns and answers["top2"].notna().any():
        pairs = zip(answers["top2"], answers["expected_intent"], strict=True)
        top2_accuracy = num(np.mean([isinstance(top2, list) and expected in top2 for top2, expected in pairs]))

    signal_name, confidence_n = None, 0
    ece = cov95 = cov98 = None
    signal = confidence_signal(answers)
    if signal is not None:
        values, signal_name = signal
        usable = values.notna().to_numpy()
        conf = values.to_numpy(dtype=np.float64)[usable]
        hit = hits[usable]
        confidence_n = len(conf)
        ece = expected_calibration_error(conf, hit)
        # Answers without a usable confidence can never be acted on automatically, but they still count.
        cov95 = coverage_at_accuracy(conf, hit, 0.95, total=len(answers))
        cov98 = coverage_at_accuracy(conf, hit, 0.98, total=len(answers))

    # Every call is billed for the tokens it used, including safety blocks, so cost averages over all calls.
    cost = call_cost_usd(frame, prices) if prices is not None else None
    upstream = timed["gateway_upstream_ms"].dropna() if "gateway_upstream_ms" in timed.columns else pd.Series()

    def pct(q: float) -> float:
        return num(np.percentile(latency, q)) if len(latency) else math.nan

    return Summary(
        system=str(frame["system"].iloc[0]),
        split=str(frame["split"].iloc[0]),
        n=len(frame),
        answered=len(answers),
        repeats=count(frame["repeat"].nunique()),
        accuracy=num(hits.mean()) if len(hits) else math.nan,
        accuracy_ci=bootstrap_ci(hits.astype(np.float64)),
        macro_f1=macro_f1(answers["expected_intent"], answers["predicted_intent"].fillna(INVALID_LABEL)),
        category_accuracy=_rate(answers["category_correct"].astype(bool)),
        top2_accuracy=top2_accuracy,
        blocked_rate=_rate(frame["outcome"] == BLOCKED),
        api_error_rate=_rate(frame["outcome"] == API_ERROR),
        invalid_rate=_rate(answers["outcome"] == "invalid_output"),
        cache_hits=count(served_from_cache(frame).sum()),
        latency_n=len(latency),
        latency_p50=pct(50),
        latency_p95=pct(95),
        latency_p99=pct(99),
        latency_mean=num(latency.mean()) if len(latency) else math.nan,
        gateway_p50=num(upstream.median()) if len(upstream) else None,
        input_tokens_mean=num(frame.loc[ok, "input_tokens"].mean()),
        output_tokens_mean=num(frame.loc[ok, "output_tokens"].mean()),
        reasoning_tokens_mean=num(frame.loc[ok, "reasoning_tokens"].fillna(0).mean()),
        cost_per_1k_usd=num(cost.mean()) * 1000 if cost is not None and len(cost) else None,
        confidence_signal=signal_name,
        confidence_n=confidence_n,
        ece=ece,
        coverage_at_95=cov95,
        coverage_at_98=cov98,
    )


def top_confusions(frame: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The most frequent wrong answers (blocked calls and API errors are not answers)."""
    wrong = frame[answered(frame) & ~frame["correct"].astype(bool)]
    counts = (
        wrong.assign(predicted=wrong["predicted_intent"].fillna(INVALID_LABEL))
        .groupby(["expected_intent", "predicted"])
        .size()
        .rename("count")
        .reset_index()
    )
    # Ties are broken by name so the report is identical on every machine.
    return counts.sort_values(["count", "expected_intent", "predicted"], ascending=[False, True, True]).head(n)


def blocked_counts(frame: pd.DataFrame) -> dict[bool, tuple[int, int]]:
    """{has_profanity: (blocked, calls)} for one system and split."""
    out: dict[bool, tuple[int, int]] = {}
    for profane in (False, True):
        group = frame[frame["has_profanity"].astype(bool) == profane]
        out[profane] = (count((group["outcome"] == BLOCKED).sum()), len(group))
    return out


def paired_comparison(
    frame: pd.DataFrame, system_a: str, system_b: str, split: str, repeat: int = 1
) -> dict[str, float] | None:
    """McNemar on examples both systems answered in the same split and repeat."""
    sel = frame[(frame["split"] == split) & (frame["repeat"] == repeat) & answered(frame)]
    a = sel[sel["system"] == system_a].set_index("example_id")["correct"]
    b = sel[sel["system"] == system_b].set_index("example_id")["correct"]
    shared = a.index.intersection(b.index)
    if len(shared) == 0:
        return None
    a_c = a.loc[shared].to_numpy(dtype=bool)
    b_c = b.loc[shared].to_numpy(dtype=bool)
    return {
        "n": len(shared),
        "a_accuracy": num(a_c.mean()),
        "b_accuracy": num(b_c.mean()),
        **mcnemar_exact(a_c, b_c),
    }
