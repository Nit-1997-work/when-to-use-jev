"""Re-ranking metrics over result records. Pure functions on pandas frames, unit-testable offline.

Conventions:
- Every record carries the search's gold labels in presented order (`gold_labels`, aligned with `candidate_ids`) and
  the system's ranking (`order`), so every metric is computed from records alone.
- Quality metrics use answered searches (`ok` and `invalid_output`; an unusable answer is ranked in presented order).
  Blocked searches and API errors are excluded and counted separately.
- Ranking quality uses searches that have at least one Exact candidate; searches built with no Exact candidate are
  scored only on abstention.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

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
from jevbench.rerank.labels import GAIN, GENTLE_GAIN

DEFAULT_THRESHOLD = 0.5
LABEL_CODES = ("E", "S", "C", "I")


# ---------- one ranking ----------


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows as plain dicts with string keys (record files always have string keys)."""
    return cast(list[dict[str, Any]], frame.to_dict(orient="records"))


def ranked_labels(record: Mapping[str, Any]) -> list[str]:
    """Gold labels in the order the system ranked the candidates."""
    gold = {str(cid): str(label) for cid, label in zip(record["candidate_ids"], record["gold_labels"], strict=True)}
    return [gold[str(cid)] for cid in record["order"]]


def dcg(labels: Sequence[str], k: int, gain: Mapping[str, float] = GAIN) -> float:
    return sum(gain[label] / math.log2(position + 2) for position, label in enumerate(labels[:k]))


def ndcg_at_k(ranked: Sequence[str], k: int, gain: Mapping[str, float] = GAIN) -> float:
    """DCG of the top k divided by the best possible DCG for the same labels; NaN when every gain is 0."""
    ideal = dcg(sorted(ranked, key=lambda label: -gain[label]), k, gain)
    return dcg(ranked, k, gain) / ideal if ideal > 0 else math.nan


def reciprocal_rank(ranked: Sequence[str], target: str = "E") -> float:
    """1 / position of the first `target` label; 0 when there is none."""
    return next((1.0 / (i + 1) for i, label in enumerate(ranked) if label == target), 0.0)


def quality(frame: pd.DataFrame) -> pd.DataFrame:
    """Per answered search with an Exact candidate: NDCG@3/@10, gentle NDCG@3, Exact@1, Usable@1, reciprocal rank."""
    rows = frame[answered(frame) & ~frame["should_abstain"].astype(bool)]
    out = []
    for record in _records(rows):
        ranked = ranked_labels(record)
        out.append(
            {
                "example_id": record["example_id"],
                "repeat": record["repeat"],
                "grocery": bool(record["grocery"]),
                "hard": bool(record["hard"]),
                "source": record["source"],
                "ndcg3": ndcg_at_k(ranked, 3),
                "ndcg10": ndcg_at_k(ranked, 10),
                "ndcg3_gentle": ndcg_at_k(ranked, 3, GENTLE_GAIN),
                "exact1": ranked[0] == "E",
                "usable1": ranked[0] in ("E", "S"),
                "rr": reciprocal_rank(ranked),
            }
        )
    columns = ["example_id", "repeat", "grocery", "hard", "source", "ndcg3", "ndcg10", "ndcg3_gentle", "exact1"]
    return pd.DataFrame(out, columns=[*columns, "usable1", "rr"])


# ---------- abstention ----------


def abstained(record: Mapping[str, Any], threshold: float) -> bool:
    """The system's own flag when it gives one; otherwise "nothing is exact" below the threshold; else never."""
    flag = record.get("abstain_flag")
    if isinstance(flag, bool):
        return flag
    confidence = record.get("exact_confidence")
    if isinstance(confidence, int | float) and not math.isnan(confidence):
        return confidence < threshold
    return False


def abstention(frame: pd.DataFrame, threshold: float) -> dict[str, float]:
    """Precision, recall, F1 of abstaining over answered searches; false abstentions on searches with an Exact."""
    rows = frame[answered(frame)]
    if rows.empty:
        return {
            "n": 0,
            "positives": 0,
            "precision": math.nan,
            "recall": math.nan,
            "f1": math.nan,
            "false_rate": math.nan,
        }
    predicted = np.array([abstained(r, threshold) for r in _records(rows)], dtype=bool)
    truth = rows["should_abstain"].astype(bool).to_numpy()
    tp = count((predicted & truth).sum())
    fp = count((predicted & ~truth).sum())
    fn = count((~predicted & truth).sum())
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2 * precision * recall / (precision + recall) if tp else (0.0 if tp + fp + fn else math.nan)
    negatives = count((~truth).sum())
    return {
        "n": len(rows),
        "positives": count(truth.sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_rate": fp / negatives if negatives else math.nan,
    }


def tune_threshold(frame: pd.DataFrame) -> tuple[float, float] | None:
    """The `exact_confidence` threshold with the best abstention F1, and that F1. None when there is nothing to tune.

    Candidate thresholds sit halfway between neighbouring observed confidences (plus one below all and one above all),
    so every distinct cut is tried once and a threshold never lands on an observed value.
    """
    rows = frame[answered(frame) & frame["exact_confidence"].notna()]
    if rows.empty or not rows["should_abstain"].astype(bool).any():
        return None
    values = sorted({float(v) for v in rows["exact_confidence"]})
    midpoints = [(low + high) / 2 for low, high in pairwise(values)]
    candidates = [values[0] / 2, *midpoints, (values[-1] + 1.0) / 2 if values[-1] < 1.0 else math.nextafter(1.0, 2)]
    best: tuple[float, float] | None = None
    for threshold in sorted(set(candidates)):
        f1 = abstention(rows, threshold)["f1"]
        if not math.isnan(f1) and (best is None or f1 > best[1]):
            best = (threshold, f1)
    return best


# ---------- labels and calibration ----------


def candidate_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per candidate the system labeled: gold and predicted ESCI code."""
    rows = []
    for record in _records(frame[answered(frame)]):
        predicted = record.get("predicted_labels")
        if not isinstance(predicted, dict):
            continue
        for cid, gold in zip(record["candidate_ids"], record["gold_labels"], strict=True):
            rows.append({"gold": gold, "predicted": predicted.get(cid, "missing")})
    return pd.DataFrame(rows, columns=["gold", "predicted"])


def label_quality(frame: pd.DataFrame) -> dict[str, float] | None:
    pairs = candidate_labels(frame)
    if pairs.empty:
        return None
    return {
        "n": len(pairs),
        "accuracy": num((pairs["gold"] == pairs["predicted"]).mean()),
        "macro_f1": macro_f1(pairs["gold"], pairs["predicted"]),
        "exact_as_substitute": num(
            ((pairs["gold"] == "E") & (pairs["predicted"] == "S")).sum() / max(1, (pairs["gold"] == "E").sum())
        ),
        "substitute_as_exact": num(
            ((pairs["gold"] == "S") & (pairs["predicted"] == "E")).sum() / max(1, (pairs["gold"] == "S").sum())
        ),
    }


def confusion(frame: pd.DataFrame) -> pd.DataFrame:
    pairs = candidate_labels(frame)
    return pd.crosstab(pairs["gold"], pairs["predicted"]).reindex(index=list(LABEL_CODES), fill_value=0)


def calibration(frame: pd.DataFrame) -> tuple[int, float] | None:
    """(candidates, expected calibration error) of per-candidate P(exact) against the Exact label."""
    probabilities: list[float] = []
    exact: list[bool] = []
    for record in _records(frame[answered(frame)]):
        p_exact = record.get("p_exact")
        if not isinstance(p_exact, dict):
            continue
        for cid, gold in zip(record["candidate_ids"], record["gold_labels"], strict=True):
            if cid in p_exact:
                probabilities.append(float(p_exact[cid]))
                exact.append(gold == "E")
    if not probabilities:
        return None
    return len(probabilities), expected_calibration_error(np.array(probabilities), np.array(exact))


# ---------- consistency ----------


def consistency(frame: pd.DataFrame) -> dict[str, float] | None:
    """Across repeats of the `consistency` split: top-1 agreement and mean Kendall's tau over repeat pairs."""
    rows = frame[(frame["split"] == "consistency") & answered(frame)]
    if rows["repeat"].nunique() < 2:
        return None
    orders = {(r["example_id"], int(r["repeat"])): list(r["order"]) for r in _records(rows)}
    repeats = sorted(rows["repeat"].unique())
    agree: list[bool] = []
    taus: list[float] = []
    for example_id in sorted(rows["example_id"].unique()):
        for i, a in enumerate(repeats):
            for b in repeats[i + 1 :]:
                first, second = orders.get((example_id, int(a))), orders.get((example_id, int(b)))
                if first is None or second is None:
                    continue
                agree.append(first[0] == second[0])
                position = {cid: p for p, cid in enumerate(second)}
                tau = kendalltau(range(len(first)), [position[cid] for cid in first]).statistic
                if not math.isnan(tau):
                    taus.append(float(tau))
    if not agree:
        return None
    return {
        "pairs": len(agree),
        "top1_agreement": num(np.mean(agree)),
        "kendall_tau": num(np.mean(taus)) if taus else math.nan,
    }


def order_sensitivity(frame: pd.DataFrame) -> dict[str, float] | None:
    """Same candidates in reversed order (`reversed` vs `consistency` repeat 1): how often is #1 the same product?"""
    rows = frame[answered(frame)]
    forward = rows[(rows["split"] == "consistency") & (rows["repeat"] == 1)]
    backward = rows[rows["split"] == "reversed"]
    if forward.empty or backward.empty:
        return None

    def top_products(part: pd.DataFrame) -> dict[str, str]:
        tops: dict[str, str] = {}
        for record in _records(part):
            products = dict(zip(record["candidate_ids"], record["product_ids"], strict=True))
            tops[str(record["example_id"])] = products[record["order"][0]]
        return tops

    a, b = top_products(forward), top_products(backward)
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    return {"searches": len(shared), "top1_same": num(np.mean([a[s] == b[s] for s in shared]))}


# ---------- head-to-head ----------


def paired_permutation_p(a: np.ndarray, b: np.ndarray, *, n_perm: int = 10000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation test on the mean difference."""
    diff = a - b
    if len(diff) == 0 or np.allclose(diff, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    observed = abs(diff.mean())
    return num((np.abs((signs * diff).mean(axis=1)) >= observed - 1e-12).mean())


def head_to_head(quality_a: pd.DataFrame, quality_b: pd.DataFrame) -> dict[str, float] | None:
    """System A vs B on the searches both answered (repeat 1): NDCG@3 difference and Exact@1 McNemar."""
    a = quality_a[quality_a["repeat"] == 1].set_index("example_id")
    b = quality_b[quality_b["repeat"] == 1].set_index("example_id")
    shared = a.index.intersection(b.index)
    if len(shared) == 0:
        return None
    a_ndcg, b_ndcg = a.loc[shared, "ndcg3"].to_numpy(float), b.loc[shared, "ndcg3"].to_numpy(float)
    diff = a_ndcg - b_ndcg
    return {
        "n": len(shared),
        "ndcg3_a": num(a_ndcg.mean()),
        "ndcg3_b": num(b_ndcg.mean()),
        "ndcg3_diff": num(diff.mean()),
        "ndcg3_diff_ci": bootstrap_ci(diff),  # pyright: ignore[reportReturnType]
        "ndcg3_p": paired_permutation_p(a_ndcg, b_ndcg),
        **mcnemar_exact(a.loc[shared, "exact1"].to_numpy(bool), b.loc[shared, "exact1"].to_numpy(bool)),
    }


def cascade(
    jev: pd.DataFrame,
    jev_quality: pd.DataFrame,
    llm_quality: pd.DataFrame,
    jev_cost: pd.Series | None,
    llm_cost: pd.Series | None,
    shares: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 1.0),
) -> list[dict[str, float]]:
    """Jev keeps the searches where it is most confident and passes the rest to the LLM.

    Confidence is the gap between Jev's top two candidate scores. For each share passed on, the mixed NDCG@3 and the
    cost per search (Jev on every search, plus the LLM on the share passed on), on searches both systems answered.
    """
    margins: dict[str, float] = {}
    for record in _records(jev[(jev["repeat"] == 1) & answered(jev)]):
        scores = record.get("scores")
        if isinstance(scores, dict) and scores:
            top = sorted((float(v) for v in scores.values()), reverse=True)
            margins[str(record["example_id"])] = top[0] - (top[1] if len(top) > 1 else 0.0)
    a = jev_quality[jev_quality["repeat"] == 1].set_index("example_id")["ndcg3"]
    b = llm_quality[llm_quality["repeat"] == 1].set_index("example_id")["ndcg3"]
    shared = [e for e in sorted(margins, key=lambda e: margins[e]) if e in a.index and e in b.index]
    if not shared:
        return []
    jev_unit = num(jev_cost.mean()) if jev_cost is not None and len(jev_cost) else math.nan
    llm_unit = num(llm_cost.mean()) if llm_cost is not None and len(llm_cost) else math.nan
    out = []
    for share in shares:
        passed = set(shared[: round(share * len(shared))])
        ndcg = [b[e] if e in passed else a[e] for e in shared]
        out.append(
            {
                "share": share,
                "n": len(shared),
                "ndcg3": num(np.mean(ndcg)),
                "cost_per_1k": (jev_unit + share * llm_unit) * 1000,
            }
        )
    return out


# ---------- per-system summary ----------


@dataclass
class Summary:
    system: str
    split: str
    n: int  # searches attempted
    answered: int
    ranked: int  # answered searches with an Exact candidate (the quality denominator)
    ndcg3: float
    ndcg3_ci: tuple[float, float]
    ndcg10: float
    ndcg3_gentle: float
    exact1: float
    exact1_ci: tuple[float, float]
    usable1: float
    mrr: float
    blocked_rate: float
    api_error_rate: float
    invalid_rate: float  # of answered searches
    invented_ids: int
    duplicate_ids: int
    missing_ids: int
    blocked_calls: int  # per-candidate calls withheld by the safety filter, inside answered searches
    cache_hits: int
    calls_mean: float
    latency_n: int
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_mean: float
    input_tokens_mean: float
    output_tokens_mean: float
    reasoning_tokens_mean: float
    cost_per_1k_usd: float | None


def _rate(mask: pd.Series) -> float:
    return num(mask.mean()) if len(mask) else math.nan


def search_cost_usd(frame: pd.DataFrame, prices: Prices | None) -> pd.Series | None:
    """Per-search USD cost from summed tokens; None when a needed price is not configured."""
    if prices is None:
        return None
    return call_cost_usd(frame, prices)


def summarize(frame: pd.DataFrame, prices: Prices | None) -> Summary:
    is_answered = answered(frame)
    answers = frame[is_answered]
    q = quality(frame)
    ok = frame["outcome"] == "ok"
    timed = frame[ok & ~served_from_cache(frame)]
    latency = timed["latency_ms"].to_numpy(dtype=np.float64)
    ndcg3 = q["ndcg3"].dropna().to_numpy(float)
    exact1 = q["exact1"].to_numpy(float)
    cost = search_cost_usd(frame, prices)

    def pct(p: float) -> float:
        return num(np.percentile(latency, p)) if len(latency) else math.nan

    return Summary(
        system=str(frame["system"].iloc[0]),
        split=str(frame["split"].iloc[0]),
        n=len(frame),
        answered=len(answers),
        ranked=len(q),
        ndcg3=num(ndcg3.mean()) if len(ndcg3) else math.nan,
        ndcg3_ci=bootstrap_ci(ndcg3),
        ndcg10=num(q["ndcg10"].mean()),
        ndcg3_gentle=num(q["ndcg3_gentle"].mean()),
        exact1=num(exact1.mean()) if len(exact1) else math.nan,
        exact1_ci=bootstrap_ci(exact1),
        usable1=num(q["usable1"].mean()),
        mrr=num(q["rr"].mean()),
        blocked_rate=_rate(frame["outcome"] == BLOCKED),
        api_error_rate=_rate(frame["outcome"] == API_ERROR),
        invalid_rate=_rate(answers["outcome"] == "invalid_output"),
        invented_ids=count(answers["invented_ids"].sum()),
        duplicate_ids=count(answers["duplicate_ids"].sum()),
        missing_ids=count(answers["missing_ids"].sum()),
        blocked_calls=count(answers["blocked_calls"].sum()) if "blocked_calls" in answers else 0,
        cache_hits=count(served_from_cache(frame).sum()),
        calls_mean=num(frame["calls"].mean()),
        latency_n=len(latency),
        latency_p50=pct(50),
        latency_p95=pct(95),
        latency_p99=pct(99),
        latency_mean=num(latency.mean()) if len(latency) else math.nan,
        input_tokens_mean=num(frame.loc[ok, "input_tokens"].mean()),
        output_tokens_mean=num(frame.loc[ok, "output_tokens"].mean()),
        reasoning_tokens_mean=num(frame.loc[ok, "reasoning_tokens"].fillna(0).mean()),
        cost_per_1k_usd=num(cost.mean()) * 1000 if cost is not None and len(cost) else None,
    )
