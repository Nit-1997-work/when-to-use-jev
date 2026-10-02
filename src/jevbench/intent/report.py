"""Turn a run directory into a Markdown report: run conditions, headline tables, paired tests, slices, confusions.

The report needs nothing but the run directory. Prices, models, and settings come from the run's own manifests,
so anyone can rebuild it from published results without API keys: `jevbench intent report --run-id <run>`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

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
    share,
    usd,
)
from jevbench.common.settings import Prices
from jevbench.common.stats import BLOCKED, answered, count, load_results, num, served_from_cache
from jevbench.intent.metrics import Summary, blocked_counts, paired_comparison, summarize, top_confusions

SPLIT_ORDER = ("main", "profanity", "practice")
SYSTEM_ORDER = ("jev", "baseline", "baseline-conf", "baseline-2")
# Manifests record LLM settings per configuration; these systems share or own one.
_LLM_CONFIG_KEY = {"baseline": "baseline", "baseline-conf": "baseline", "baseline-2": "baseline_2"}

NOTES = (
    "- Quality metrics (accuracy, macro-F1, category accuracy, top-2, slices, confusions, calibration, coverage) use"
    " answered messages. A message is answered when the system returned an intent; an intent outside the list"
    " counts as wrong.",
    "- Blocked: the provider's safety filter withheld the answer (`finish_reason=content_filter` through the"
    " gateway). Blocked messages are excluded from quality metrics and counted in the Blocked column and table."
    " API errors (failures after retries) are excluded and counted the same way.",
    "- Latency: client wall-clock time for one HTTP attempt, from sending the request to a parsed answer, over"
    " answered calls not served from a cache. Retries and their backoff are not included.",
    "- Gateway-reported time: the gateway's own duration header for the call (`x-litellm-response-duration-ms`);"
    " the rest of the client latency is network and gateway ingress.",
    "- Cost: the tokens reported by each response, priced at the rates recorded in the run's manifests and averaged"
    " over every call made, including blocked ones (they are billed).",
    "- Confidence: Jev uses the probability of its chosen intent; `baseline-conf` uses the model's self-reported"
    " confidence. Calibration error is the expected calibration error over 15 equal-width bins. Coverage at X% is"
    " the largest share of answered messages a system can handle on its own, by acting only on answers at or above"
    " one confidence threshold, while staying at least X% accurate on them; thresholds fall only between groups of"
    " equal confidence.",
    "- Head-to-head: McNemar's exact test on the messages both systems answered (repeat 1).",
)


# ---------- manifests ----------


def _llm_config(manifest: Mapping[str, object]) -> Mapping[str, object]:
    config = manifest_settings(manifest).get(_LLM_CONFIG_KEY.get(str(manifest.get("system")), ""))
    return config if isinstance(config, dict) else {}


def _reasoning_effort(system: str, manifest: Mapping[str, object]) -> str:
    return "-" if system == "jev" else str(_llm_config(manifest).get("reasoning_effort") or "provider default")


def prices_from_manifest(manifest: Mapping[str, object]) -> Prices | None:
    """The system's token prices as recorded when it ran (older manifests keep them only inside `settings`)."""
    recorded = manifest.get("prices_per_mtok")
    if isinstance(recorded, dict):
        return Prices.from_dict(recorded)
    if manifest.get("system") == "jev":
        price = manifest_settings(manifest).get("jev_input_price_per_mtok")
        return Prices.from_dict({"input": price, "output": 0.0, "thinking": 0.0}) if price is not None else None
    llm_prices = _llm_config(manifest).get("price_per_mtok")
    return Prices.from_dict(llm_prices) if isinstance(llm_prices, dict) else None


def system_prices(manifests: Manifests, system: str, split: str) -> Prices | None:
    recorded = {prices_from_manifest(m) for (s, sp, _), m in manifests.items() if s == system and sp == split}
    if len(recorded) > 1:
        raise ReportError(f"{system}/{split}: repeats were recorded with different prices; report them separately.")
    return next(iter(recorded), None)


# ---------- sections ----------


def accuracy_table(summaries: list[Summary]) -> str:
    rows = []
    for s in summaries:
        lo, hi = s.accuracy_ci
        ci = "" if missing(lo) else f" ({pct(lo)}-{pct(hi)})"
        rows.append(
            [
                s.system,
                f"{s.answered:,} of {s.n:,}",
                pct(s.blocked_rate),
                pct(s.api_error_rate),
                f"{pct(s.accuracy)}{ci}",
                pct(s.macro_f1),
                pct(s.category_accuracy),
                pct(s.top2_accuracy),
                pct(s.invalid_rate),
            ]
        )
    columns = [
        "System",
        "Answered",
        "Blocked",
        "API errors",
        "Accuracy (95% CI)",
        "Macro-F1",
        "Category acc",
        "Top-2 acc",
        "Invalid",
    ]
    return md_table(columns, rows)


def latency_cost_table(summaries: list[Summary]) -> str:
    rows = []
    for s in summaries:
        per_million = None if s.cost_per_1k_usd is None else s.cost_per_1k_usd * 1000
        rows.append(
            [
                s.system,
                f"{s.latency_n:,}",
                ms(s.latency_p50),
                ms(s.latency_p95),
                ms(s.latency_p99),
                ms(s.latency_mean),
                ms(s.gateway_p50),
                number(s.input_tokens_mean, 0),
                number(s.output_tokens_mean, 1),
                number(s.reasoning_tokens_mean, 1),
                usd(s.cost_per_1k_usd),
                usd(per_million, 2),
            ]
        )
    columns = [
        "System",
        "Timed calls",
        "p50 ms",
        "p95 ms",
        "p99 ms",
        "Mean ms",
        "Gateway-reported p50 ms",
        "Input tok",
        "Output tok",
        "Reasoning tok",
        "Cost / 1k",
        "Cost / 1M",
    ]
    return md_table(columns, rows)


def confidence_table(summaries: list[Summary]) -> str | None:
    rows = [
        [
            s.system,
            str(s.confidence_signal),
            f"{s.confidence_n:,}",
            "-" if missing(s.ece) else f"{s.ece:.3f}",
            pct(s.coverage_at_95),
            pct(s.coverage_at_98),
        ]
        for s in summaries
        if s.confidence_signal is not None
    ]
    if not rows:
        return None
    columns = [
        "System",
        "Signal",
        "Answers with a signal",
        "Calibration error",
        "Coverage at 95% acc",
        "Coverage at 98% acc",
    ]
    return md_table(columns, rows)


def blocked_table(frame: pd.DataFrame, systems: list[str]) -> str | None:
    if not (frame["outcome"] == BLOCKED).any():
        return None
    rows = []
    for system in systems:
        counts = blocked_counts(frame[frame["system"] == system])
        rows.append([system, share(*counts[False]), share(*counts[True])])
    return md_table(["System", "Clean messages blocked", "Profane messages blocked"], rows)


def paired_table(frame: pd.DataFrame, systems: list[str], split: str) -> str | None:
    rows = []
    for other in (s for s in systems if s != "jev"):
        result = paired_comparison(frame, "jev", other, split)
        if result is None:
            continue
        rows.append(
            [
                f"jev vs {other}",
                f"{count(result['n']):,}",
                pct(result["a_accuracy"]),
                pct(result["b_accuracy"]),
                str(count(result["a_only"])),
                str(count(result["b_only"])),
                p_value(result["p_value"]),
            ]
        )
    if not rows:
        return None
    columns = ["Pair", "Both answered", "Jev acc", "Other acc", "Only Jev right", "Only other right", "p-value"]
    return md_table(columns, rows)


def slice_table(frame: pd.DataFrame, systems: list[str]) -> str:
    rows = []
    answers = frame[answered(frame)]
    for system in systems:
        for has_typos, label in ((False, "no typos"), (True, "typos")):
            group = answers[(answers["system"] == system) & (answers["has_typos"].astype(bool) == has_typos)]
            if len(group):
                rows.append([system, label, f"{len(group):,}", pct(num(group["correct"].mean()))])
    return md_table(["System", "Slice", "Answered", "Accuracy"], rows) if rows else "_No data._"


def confusion_table(frame: pd.DataFrame) -> str | None:
    confusions = top_confusions(frame, n=8)
    if confusions.empty:
        return None
    rows = [
        [str(expected), str(predicted), str(n)]
        for expected, predicted, n in zip(
            confusions["expected_intent"], confusions["predicted"], confusions["count"], strict=True
        )
    ]
    return md_table(["Expected", "Predicted", "Count"], rows)


def build_report(run_dir: Path, *, allow_partial: bool = False) -> str:
    frame = load_results(run_dir)
    manifests = load_manifests(run_dir)
    problems = incomparable(frame)
    if problems and not allow_partial:
        raise ReportError(
            "Systems in this run cannot be compared, because they ran different examples:\n  - "
            + "\n  - ".join(problems)
            + "\nFinish (resume) the run, or pass --allow-partial to report it anyway."
        )

    out: list[str] = [f"# Results: run `{run_dir.name}`", "", *NOTES]
    out += [f"\n> Warning: partial run. {problem}." for problem in problems]
    cache_hits = count(served_from_cache(frame).sum())
    if cache_hits:
        out += ["", f"> Warning: {cache_hits} gateway cache hits were recorded and excluded from latency."]
    repeats = count(frame.groupby(["system", "split"])["repeat"].nunique().max())
    if repeats > 1:
        out += [
            "",
            f"> Note: this run has up to {repeats} repeats of the same messages. Margins of error treat every repeat"
            " as a separate message, so they are narrower than they should be; the head-to-head test uses repeat 1.",
        ]
    conditions = run_conditions(
        frame,
        manifests,
        system_order=SYSTEM_ORDER,
        split_order=SPLIT_ORDER,
        reasoning_effort=_reasoning_effort,
        prices=prices_from_manifest,
    )
    out += ["", "## Run conditions", "", conditions]

    for split in ordered([str(s) for s in frame["split"].unique()], SPLIT_ORDER):
        split_frame = frame[frame["split"] == split]
        systems = ordered([str(s) for s in split_frame["system"].unique()], SYSTEM_ORDER)
        summaries = [
            summarize(split_frame[split_frame["system"] == system], system_prices(manifests, system, split))
            for system in systems
        ]
        out += ["", f"## Split: {split}", "", "### Accuracy", "", accuracy_table(summaries)]
        blocked = blocked_table(split_frame, systems)
        if blocked:
            out += [
                "",
                "### Blocked messages",
                "",
                "The provider's safety filter withheld these answers; they are excluded from the quality metrics.",
                "",
                blocked,
            ]
        out += ["", "### Latency and cost", "", latency_cost_table(summaries)]
        confidence = confidence_table(summaries)
        if confidence:
            out += ["", "### Confidence", "", confidence]
        paired = paired_table(frame, systems, split)
        if paired:
            out += ["", "### Head-to-head on messages both systems answered", "", paired]
        out += ["", "### Accuracy by typo slice", "", slice_table(split_frame, systems)]
        for system in systems:
            table = confusion_table(split_frame[split_frame["system"] == system])
            if table:
                out += ["", f"### Top confusions: {system}", "", table]
    return "\n".join(out) + "\n"
