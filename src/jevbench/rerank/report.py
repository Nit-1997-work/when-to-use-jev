"""Turn a re-ranking run directory into a Markdown report.

The report needs nothing but the run directory and the frozen thresholds file. Prices, models, and settings come from
the run's own manifests, so anyone can rebuild it without API keys: `jevbench rerank report --run-id <run>`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import yaml

from jevbench.common.paths import file_sha256
from jevbench.common.report_format import (
    Manifests,
    ReportError,
    incomparable,
    load_manifests,
    manifest_settings,
    md_table,
    missing,
    ms,
    number,
    ordered,
    p_value,
    pct,
    run_conditions,
    usd,
)
from jevbench.common.settings import Prices
from jevbench.common.stats import count, load_results, served_from_cache
from jevbench.rerank.metrics import (
    DEFAULT_THRESHOLD,
    LABEL_CODES,
    Summary,
    abstention,
    calibration,
    cascade,
    confusion,
    consistency,
    head_to_head,
    label_quality,
    order_sensitivity,
    quality,
    search_cost_usd,
    summarize,
)
from jevbench.rerank.systems import SYSTEMS, llm_config_key
from jevbench.rerank.thresholds import THRESHOLDS_PATH

SYSTEM_ORDER = tuple(SYSTEMS)
QUALITY_SPLITS = ("main", "practice", "consistency", "reversed")
SPLIT_ORDER = ("main", "nomatch", "practice", "practice-nomatch", "consistency", "reversed")
# Abstention is measured over a normal split together with its constructed no-match counterpart.
ABSTENTION_FAMILIES = (("main", "nomatch"), ("practice", "practice-nomatch"))
REFERENCE = "gemini-listwise"

NOTES = (
    "- Ranking quality uses answered searches that have at least one Exact candidate. Gains follow the KDD Cup 2022"
    " values: Exact 1.0, Substitute 0.1, Complement 0.01, Irrelevant 0. NDCG@k is the ranking's discounted gain over"
    " the top k divided by the best possible for that search. The gentle column uses Exact 1, Substitute 0.5,"
    " Complement 0.1, Irrelevant 0.",
    "- Exact@1: the top product is an Exact match. Usable@1: Exact or Substitute. MRR: 1 / position of the first"
    " Exact product. `random` (the presented order) is the floor.",
    "- An unusable answer (`invalid_output`) counts as answered and is ranked in presented order. Blocked searches"
    " (the provider's safety filter) and API errors are excluded and counted separately.",
    "- Abstention: a system abstains by its own flag (Gemini listwise; label systems when no candidate is labeled"
    " Exact) or when its confidence that some candidate is Exact falls below the frozen threshold in"
    " data/rerank/thresholds.yaml (Jev systems).",
    "- Latency: client wall-clock time for one search, from sending the request to a parsed ranking; for"
    " per-candidate systems, the whole parallel fan-out. One clean attempt; retries and backoff are not included."
    " Cost: summed tokens priced at the rates recorded in the run's manifests.",
)


# ---------- inputs ----------


def load_thresholds(path: Path = THRESHOLDS_PATH) -> tuple[dict[str, float], str | None]:
    """({system: threshold}, file SHA-256) from the frozen thresholds file; empty when it does not exist yet."""
    if not path.exists():
        return {}, None
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    thresholds = raw.get("thresholds", {}) if isinstance(raw, dict) else {}
    return {str(k): float(v) for k, v in thresholds.items()}, file_sha256(path)


def prices_from_manifest(manifest: Mapping[str, object]) -> Prices | None:
    recorded = manifest.get("prices_per_mtok")
    return Prices.from_dict(recorded) if isinstance(recorded, dict) else None


def _reasoning_effort(system: str, manifest: Mapping[str, object]) -> str:
    key = llm_config_key(system)
    if key is None:
        return "-"
    config = manifest_settings(manifest).get(key)
    effort = config.get("reasoning_effort") if isinstance(config, dict) else None
    return str(effort or "provider default")


def system_prices(manifests: Manifests, system: str) -> Prices | None:
    recorded = {prices_from_manifest(m) for (s, _, _), m in manifests.items() if s == system}
    if len(recorded) > 1:
        raise ReportError(f"{system}: splits were recorded with different prices; report them separately.")
    return next(iter(recorded), None)


def _ci(value: float, ci: tuple[float, float], fmt: str) -> str:
    if missing(value):
        return "-"
    text = f"{value:.3f}" if fmt == "num" else pct(value)
    lo, hi = ci
    if missing(lo):
        return text
    bounds = f"{lo:.3f}-{hi:.3f}" if fmt == "num" else f"{pct(lo)}-{pct(hi)}"
    return f"{text} ({bounds})"


# ---------- sections ----------


def quality_table(summaries: list[Summary]) -> str:
    rows = [
        [
            s.system,
            f"{s.ranked:,}",
            _ci(s.ndcg3, s.ndcg3_ci, "num"),
            _ci(s.exact1, s.exact1_ci, "pct"),
            pct(s.usable1),
            number(s.mrr, 3),
            number(s.ndcg10, 3),
            number(s.ndcg3_gentle, 3),
        ]
        for s in summaries
    ]
    columns = [
        "System",
        "Searches",
        "NDCG@3 (95% CI)",
        "Exact@1 (95% CI)",
        "Usable@1",
        "MRR",
        "NDCG@10",
        "NDCG@3 gentle",
    ]
    return md_table(columns, rows)


def slice_table(frame: pd.DataFrame, systems: list[str]) -> str | None:
    rows = []
    for system in systems:
        q = quality(frame[frame["system"] == system])
        q = q[q["repeat"] == 1]
        if q.empty:
            continue
        slices = {
            "all": q,
            "general": q[~q["grocery"]],
            "grocery": q[q["grocery"]],
            "hard (3 or fewer Exact)": q[q["hard"]],
            "negations": q[q["source"] == "negations"],
        }
        for name, part in slices.items():
            if len(part):
                rows.append(
                    [system, name, f"{len(part):,}", number(part["ndcg3"].mean(), 3), pct(part["exact1"].mean())]
                )
    return md_table(["System", "Slice", "Searches", "NDCG@3", "Exact@1"], rows) if rows else None


def head_to_head_table(frame: pd.DataFrame, systems: list[str]) -> str | None:
    if REFERENCE not in systems:
        return None
    reference = quality(frame[frame["system"] == REFERENCE])
    rows = []
    for system in (s for s in systems if s != REFERENCE):
        result = head_to_head(quality(frame[frame["system"] == system]), reference)
        if result is None:
            continue
        lo, hi = result["ndcg3_diff_ci"]  # pyright: ignore[reportGeneralTypeIssues]
        rows.append(
            [
                f"{system} vs {REFERENCE}",
                f"{count(result['n']):,}",
                f"{result['ndcg3_diff']:+.3f} ({lo:+.3f} to {hi:+.3f})",
                p_value(result["ndcg3_p"]),
                str(count(result["a_only"])),
                str(count(result["b_only"])),
                p_value(result["p_value"]),
            ]
        )
    if not rows:
        return None
    columns = [
        "Pair",
        "Searches",
        "NDCG@3 difference (95% CI)",
        "p (permutation)",
        "Only it Exact@1",
        "Only reference Exact@1",
        "p (McNemar)",
    ]
    return md_table(columns, rows)


def abstention_table(frame: pd.DataFrame, systems: list[str], thresholds: Mapping[str, float]) -> str | None:
    rows = []
    for normal, nomatch in ABSTENTION_FAMILIES:
        family = frame[frame["split"].isin([normal, nomatch]) & (frame["repeat"] == 1)]
        if family.empty or not family["should_abstain"].astype(bool).any():
            continue
        for system in systems:
            part = family[family["system"] == system]
            if part.empty:
                continue
            uses_threshold = part["abstain_flag"].isna().all() and part["exact_confidence"].notna().any()
            threshold = thresholds.get(system, DEFAULT_THRESHOLD)
            result = abstention(part, threshold)
            if not uses_threshold and part["abstain_flag"].isna().all():
                continue  # the system never abstains (no flag, no confidence)
            rows.append(
                [
                    system,
                    f"{normal} + {nomatch}",
                    f"{threshold:.3f}" if uses_threshold else "own flag",
                    f"{count(result['positives']):,} of {count(result['n']):,}",
                    pct(result["precision"]),
                    pct(result["recall"]),
                    pct(result["f1"]),
                    pct(result["false_rate"]),
                ]
            )
    if not rows:
        return None
    columns = ["System", "Splits", "Threshold", "Should abstain", "Precision", "Recall", "F1", "False abstentions"]
    return md_table(columns, rows)


def label_table(frame: pd.DataFrame, systems: list[str]) -> tuple[str, list[str]] | None:
    rows = []
    confusions = []
    for system in systems:
        part = frame[(frame["system"] == system) & frame["split"].isin(QUALITY_SPLITS) & (frame["repeat"] == 1)]
        result = label_quality(part)
        if result is None:
            continue
        rows.append(
            [
                system,
                f"{count(result['n']):,}",
                pct(result["accuracy"]),
                pct(result["macro_f1"]),
                pct(result["exact_as_substitute"]),
                pct(result["substitute_as_exact"]),
            ]
        )
        table = confusion(part)
        columns = [c for c in [*LABEL_CODES, "missing"] if c in table.columns]
        confusion_rows = [[f"gold {gold}", *[str(count(table.loc[gold, c])) for c in columns]] for gold in table.index]
        confusions += [
            "",
            f"Confusion, {system} (rows: gold label; columns: predicted):",
            "",
            md_table(["", *columns], confusion_rows),
        ]
    if not rows:
        return None
    columns = ["System", "Candidates", "Accuracy", "Macro-F1", "Exact labeled Substitute", "Substitute labeled Exact"]
    return md_table(columns, rows), confusions


def latency_cost_table(summaries: list[Summary]) -> str:
    rows = []
    for s in summaries:
        per_million = None if s.cost_per_1k_usd is None else s.cost_per_1k_usd * 1000
        rows.append(
            [
                s.system,
                f"{s.latency_n:,}",
                number(s.calls_mean, 1),
                ms(s.latency_p50),
                ms(s.latency_p95),
                ms(s.latency_p99),
                ms(s.latency_mean),
                number(s.input_tokens_mean, 0),
                number(s.output_tokens_mean, 1),
                number(s.reasoning_tokens_mean, 1),
                usd(s.cost_per_1k_usd),
                usd(per_million, 2),
            ]
        )
    columns = [
        "System",
        "Timed searches",
        "Calls / search",
        "p50 ms",
        "p95 ms",
        "p99 ms",
        "Mean ms",
        "Input tok",
        "Output tok",
        "Reasoning tok",
        "Cost / 1k searches",
        "Cost / 1M searches",
    ]
    return md_table(columns, rows)


def reliability_table(summaries: list[Summary]) -> str:
    rows = [
        [
            s.system,
            f"{s.answered:,} of {s.n:,}",
            pct(s.blocked_rate),
            pct(s.api_error_rate),
            pct(s.invalid_rate),
            str(s.invented_ids),
            str(s.duplicate_ids),
            str(s.missing_ids),
            str(s.blocked_calls),
        ]
        for s in summaries
    ]
    columns = [
        "System",
        "Answered",
        "Blocked",
        "API errors",
        "Invalid",
        "Invented ids",
        "Duplicate ids",
        "Missing ids",
        "Blocked calls",
    ]
    return md_table(columns, rows)


def consistency_table(frame: pd.DataFrame, systems: list[str]) -> str | None:
    rows = []
    for system in systems:
        part = frame[frame["system"] == system]
        repeated, reversed_ = consistency(part), order_sensitivity(part)
        if repeated is None and reversed_ is None:
            continue
        rows.append(
            [
                system,
                pct(repeated["top1_agreement"]) if repeated else "-",
                number(repeated["kendall_tau"], 3) if repeated else "-",
                pct(reversed_["top1_same"]) if reversed_ else "-",
            ]
        )
    columns = ["System", "Same #1 across repeats", "Kendall's tau across repeats", "Same #1 when order is reversed"]
    return md_table(columns, rows) if rows else None


def calibration_table(frame: pd.DataFrame, systems: list[str]) -> str | None:
    rows = []
    for system in systems:
        part = frame[(frame["system"] == system) & frame["split"].isin(QUALITY_SPLITS) & (frame["repeat"] == 1)]
        result = calibration(part)
        if result is not None:
            rows.append([system, f"{result[0]:,}", f"{result[1]:.3f}"])
    return md_table(["System", "Candidates", "Calibration error of P(exact)"], rows) if rows else None


def cascade_table(frame: pd.DataFrame, manifests: Manifests, systems: list[str], split: str) -> str | None:
    if REFERENCE not in systems:
        return None
    reference = frame[(frame["system"] == REFERENCE) & (frame["split"] == split)]
    reference_cost = search_cost_usd(reference, system_prices(manifests, REFERENCE))
    rows = []
    for system in (s for s in systems if s.startswith("jev-")):
        jev = frame[(frame["system"] == system) & (frame["split"] == split)]
        jev_cost = search_cost_usd(jev, system_prices(manifests, system))
        for point in cascade(jev, quality(jev), quality(reference), jev_cost, reference_cost):
            rows.append([system, pct(point["share"]), number(point["ndcg3"], 3), usd(point["cost_per_1k"])])
    columns = ["Jev system", f"Share passed to {REFERENCE}", "NDCG@3", "Cost / 1k searches"]
    return md_table(columns, rows) if rows else None


def build_report(run_dir: Path, *, allow_partial: bool = False, thresholds_path: Path = THRESHOLDS_PATH) -> str:
    frame = load_results(run_dir)
    manifests = load_manifests(run_dir)
    problems = incomparable(frame)
    if problems and not allow_partial:
        raise ReportError(
            "Systems in this run cannot be compared, because they ran different searches:\n  - "
            + "\n  - ".join(problems)
            + "\nFinish (resume) the run, or pass --allow-partial to report it anyway."
        )
    thresholds, thresholds_sha = load_thresholds(thresholds_path)
    systems = ordered([str(s) for s in frame["system"].unique()], SYSTEM_ORDER)

    out: list[str] = [f"# Results: run `{run_dir.name}`", "", *NOTES]
    out += [f"\n> Warning: partial run. {problem}." for problem in problems]
    cache_hits = count(served_from_cache(frame).sum())
    if cache_hits:
        out += ["", f"> Warning: {cache_hits} gateway cache hits were recorded and excluded from latency."]
    labels = sorted({str(m.get("labels_sha256", ""))[:8] for m in manifests.values()})
    if thresholds_sha:
        threshold_note = f"data/rerank/thresholds.yaml, SHA-256 {thresholds_sha[:8]}..."
    else:
        threshold_note = f"not tuned yet; {DEFAULT_THRESHOLD} for every system"
    extra = [
        f"- Label definitions: data/rerank/labels.yaml, SHA-256 {', '.join(labels)}...",
        f"- Abstention thresholds: {threshold_note}.",
    ]
    conditions = run_conditions(
        frame,
        manifests,
        system_order=SYSTEM_ORDER,
        split_order=SPLIT_ORDER,
        reasoning_effort=_reasoning_effort,
        prices=prices_from_manifest,
        calls_column="Searches",
        pacing="- Each system re-ranked one search at a time.",
        extra_bullets=extra,
    )
    out += ["", "## Run conditions", "", conditions]

    for split in ordered([str(s) for s in frame["split"].unique()], SPLIT_ORDER):
        split_frame = frame[(frame["split"] == split) & (frame["repeat"] == 1)]
        split_systems = ordered([str(s) for s in split_frame["system"].unique()], SYSTEM_ORDER)
        summaries = [
            summarize(split_frame[split_frame["system"] == s], system_prices(manifests, s)) for s in split_systems
        ]
        out += ["", f"## Split: {split}"]
        if any(s.ranked for s in summaries):
            out += ["", "### Ranking quality", "", quality_table(summaries)]
            if split == "main" and (slices := slice_table(split_frame, split_systems)):
                out += ["", "### Ranking quality by slice", "", slices]
            if (pairs := head_to_head_table(split_frame, split_systems)) is not None:
                out += ["", f"### Head-to-head against {REFERENCE}", "", pairs]
        out += ["", "### Latency and cost", "", latency_cost_table(summaries)]
        out += ["", "### Reliability", "", reliability_table(summaries)]
        if split in ("main", "practice") and (cascaded := cascade_table(split_frame, manifests, split_systems, split)):
            out += ["", f"### Cascade: Jev keeps confident searches, {REFERENCE} gets the rest", "", cascaded]

    if (abstain := abstention_table(frame, systems, thresholds)) is not None:
        out += ["", "## Abstention: does the system notice when nothing matches?", "", abstain]
    if (label_result := label_table(frame, systems)) is not None:
        table, confusions = label_result
        out += ["", "## Label quality (systems that label each candidate)", "", table, *confusions]
    if (stable := consistency_table(frame, systems)) is not None:
        out += ["", "## Consistency", "", stable]
    if (calibrated := calibration_table(frame, systems)) is not None:
        out += ["", "## Calibration of Jev's P(exact)", "", calibrated]
    return "\n".join(out) + "\n"
