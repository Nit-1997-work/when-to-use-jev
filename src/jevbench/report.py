"""Turn a run directory into a Markdown report: run conditions, headline tables, paired tests, slices, confusions.

The report needs nothing but the run directory. Prices, models, and settings come from the run's own manifests,
so anyone can rebuild it from published results without API keys: `jevbench report --run-id <run>`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from jevbench.metrics import (
    BLOCKED,
    Summary,
    answered,
    blocked_counts,
    count,
    load_results,
    num,
    paired_comparison,
    served_from_cache,
    summarize,
    top_confusions,
)
from jevbench.settings import LLM_SAFETY_SETTINGS, Prices

SPLIT_ORDER = ("main", "profanity", "practice")
SYSTEM_ORDER = ("jev", "baseline", "baseline-conf", "baseline-2")
# Manifests record LLM settings per configuration; these systems share or own one.
_LLM_CONFIG_KEY = {"baseline": "baseline", "baseline-conf": "baseline", "baseline-2": "baseline_2"}

# Run manifests keyed by (system, split, repeat).
type Manifests = Mapping[tuple[str, str, int], Mapping[str, object]]

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


class ReportError(RuntimeError):
    pass


# ---------- formatting ----------


def _missing(value: float | None) -> bool:
    return value is None or math.isnan(value)


def _pct(value: float | None) -> str:
    return "-" if value is None or math.isnan(value) else f"{value * 100:.1f}%"


def _ms(value: float | None) -> str:
    return "-" if value is None or math.isnan(value) else f"{value:,.0f}"


def _number(value: float | None, decimals: int) -> str:
    return "-" if value is None or math.isnan(value) else f"{value:,.{decimals}f}"


def _usd(value: float | None, decimals: int = 4) -> str:
    return "n/a (no price recorded)" if value is None or math.isnan(value) else f"${value:,.{decimals}f}"


def _price(value: float | None) -> str:
    if value is None:
        return "not set"
    text = f"{value:.4f}".rstrip("0")
    return f"${text}" if len(text.split(".")[1]) >= 2 else f"${value:.2f}"


def _p_value(value: float) -> str:
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def _share(part: int, whole: int) -> str:
    return "-" if whole == 0 else f"{part:,} of {whole:,} ({part / whole:.1%})"


def _ordered(values: list[str], order: tuple[str, ...]) -> list[str]:
    return sorted(values, key=lambda v: (order.index(v) if v in order else len(order), v))


def _table(columns: tuple[str, ...] | list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + " --- |" * len(columns)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


# ---------- manifests ----------


def load_manifests(run_dir: Path) -> dict[tuple[str, str, int], dict[str, object]]:
    manifests: dict[tuple[str, str, int], dict[str, object]] = {}
    for path in sorted(run_dir.glob("*/*/repeat-*.manifest.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ReportError(f"{path} is not valid JSON: {exc}") from exc
        manifests[(str(data["system"]), str(data["split"]), count(data["repeat"]))] = data
    return manifests


def _settings(manifest: Mapping[str, object]) -> Mapping[str, object]:
    settings = manifest.get("settings")
    return settings if isinstance(settings, dict) else {}


def _llm_config(manifest: Mapping[str, object]) -> Mapping[str, object]:
    config = _settings(manifest).get(_LLM_CONFIG_KEY.get(str(manifest.get("system")), ""))
    return config if isinstance(config, dict) else {}


def prices_from_manifest(manifest: Mapping[str, object]) -> Prices | None:
    """The system's token prices as recorded when it ran (older manifests keep them only inside `settings`)."""
    recorded = manifest.get("prices_per_mtok")
    if isinstance(recorded, dict):
        return Prices.from_dict(recorded)
    if manifest.get("system") == "jev":
        price = _settings(manifest).get("jev_input_price_per_mtok")
        return Prices.from_dict({"input": price, "output": 0.0, "thinking": 0.0}) if price is not None else None
    llm_prices = _llm_config(manifest).get("price_per_mtok")
    return Prices.from_dict(llm_prices) if isinstance(llm_prices, dict) else None


def system_prices(manifests: Manifests, system: str, split: str) -> Prices | None:
    recorded = {prices_from_manifest(m) for (s, sp, _), m in manifests.items() if s == system and sp == split}
    if len(recorded) > 1:
        raise ReportError(f"{system}/{split}: repeats were recorded with different prices; report them separately.")
    return next(iter(recorded), None)


def _values(manifests: Manifests, key: str) -> list[str]:
    """Distinct non-empty settings values across manifests, in a stable order."""
    return sorted({str(v) for m in manifests.values() if (v := _settings(m).get(key)) not in (None, "")})


# ---------- checks ----------


def incomparable(frame: pd.DataFrame) -> list[str]:
    """Why systems cannot be compared: within a split and repeat, every system must have run the same examples."""
    problems: list[str] = []
    for (split, repeat), group in frame.groupby(["split", "repeat"], sort=True):
        examples = {str(system): frozenset(g["example_id"]) for system, g in group.groupby("system")}
        if len(set(examples.values())) > 1:
            sizes = ", ".join(f"{system} {len(ids):,}" for system, ids in sorted(examples.items()))
            problems.append(f"{split} repeat {repeat}: systems ran different examples ({sizes})")
    return problems


# ---------- sections ----------


def run_conditions(frame: pd.DataFrame, manifests: Manifests) -> str:
    systems = _ordered([str(s) for s in frame["system"].unique()], SYSTEM_ORDER)
    by_system = {system: [m for (s, _, _), m in sorted(manifests.items()) if s == system] for system in systems}

    model_rows = []
    for system in systems:
        first = by_system[system][0] if by_system[system] else {}
        reported = sorted({str(m) for m in frame.loc[frame["system"] == system, "model_reported"].dropna()})
        effort = "-" if system == "jev" else str(_llm_config(first).get("reasoning_effort") or "provider default")
        prices = prices_from_manifest(first) if first else None
        price = (
            " / ".join(_price(p) for p in (prices.input, prices.output, prices.effective_thinking))
            if prices
            else "not recorded"
        )
        model_rows.append([system, str(first.get("model_requested", "-")), ", ".join(reported) or "-", effort, price])
    model_columns = [
        "System",
        "Model requested",
        "Model reported",
        "Reasoning effort",
        "Price per 1M tokens (in / out / thinking)",
    ]

    started = pd.to_datetime(frame["started_at"], utc=True)
    window_rows: list[list[str]] = []
    spans: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for system in systems:
        is_system = frame["system"] == system
        for split in _ordered([str(s) for s in frame.loc[is_system, "split"].unique()], SPLIT_ORDER):
            times = started[is_system & (frame["split"] == split)]
            first_call, last_call = pd.Timestamp(times.min()), pd.Timestamp(times.max())
            window = f"{first_call:%Y-%m-%d %H:%M} to {last_call:%Y-%m-%d %H:%M}"
            window_rows.append([system, split, f"{len(times):,}", window])
            low, high = spans.get(system, (first_call, last_call))
            spans[system] = (min(low, first_call), max(high, last_call))
    intervals = list(spans.values())
    overlapping = any(a[0] <= b[1] and b[0] <= a[1] for i, a in enumerate(intervals) for b in intervals[i + 1 :])

    git = sorted(
        {
            f"{str(g.get('commit') or 'unknown')[:7]}{' (with uncommitted changes)' if g.get('dirty') else ''}"
            for m in manifests.values()
            if isinstance(g := m.get("git"), dict)
        }
    )
    packages = sorted(
        {
            f"Python {m.get('python')}; " + ", ".join(f"{name} {ver}" for name, ver in p.items() if ver)
            for m in manifests.values()
            if isinstance(p := m.get("packages"), dict)
        }
    )
    bullets = [
        f"- LLM safety settings: {', '.join(_values(manifests, 'llm_safety_settings')) or LLM_SAFETY_SETTINGS}.",
        f"- LLM prices source: {'; '.join(_values(manifests, 'price_source_url')) or 'not recorded'}.",
    ]
    if jev_source := _values(manifests, "jev_price_source_url"):
        bullets.append(f"- Jev price source: {'; '.join(jev_source)}.")
    bullets += [
        f"- Client location: {'; '.join(_values(manifests, 'client_location')) or 'not recorded'}."
        f" Request timeout: {'; '.join(_values(manifests, 'request_timeout_s')) or 'not recorded'} s.",
        f"- Code: commit {'; '.join(git) or 'not recorded'}. {'; '.join(packages)}.",
        "- Each system sent one request at a time."
        + (
            " The systems' runs overlapped in time (see the windows above), so they shared the client machine and"
            " network."
            if overlapping
            else ""
        ),
    ]
    return "\n".join(
        [
            _table(model_columns, model_rows),
            "",
            _table(["System", "Split", "Calls", "First to last call (UTC)"], window_rows),
            "",
            *bullets,
        ]
    )


def accuracy_table(summaries: list[Summary]) -> str:
    rows = []
    for s in summaries:
        lo, hi = s.accuracy_ci
        ci = "" if _missing(lo) else f" ({_pct(lo)}-{_pct(hi)})"
        rows.append(
            [
                s.system,
                f"{s.answered:,} of {s.n:,}",
                _pct(s.blocked_rate),
                _pct(s.api_error_rate),
                f"{_pct(s.accuracy)}{ci}",
                _pct(s.macro_f1),
                _pct(s.category_accuracy),
                _pct(s.top2_accuracy),
                _pct(s.invalid_rate),
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
    return _table(columns, rows)


def latency_cost_table(summaries: list[Summary]) -> str:
    rows = []
    for s in summaries:
        per_million = None if s.cost_per_1k_usd is None else s.cost_per_1k_usd * 1000
        rows.append(
            [
                s.system,
                f"{s.latency_n:,}",
                _ms(s.latency_p50),
                _ms(s.latency_p95),
                _ms(s.latency_p99),
                _ms(s.latency_mean),
                _ms(s.gateway_p50),
                _number(s.input_tokens_mean, 0),
                _number(s.output_tokens_mean, 1),
                _number(s.reasoning_tokens_mean, 1),
                _usd(s.cost_per_1k_usd),
                _usd(per_million, 2),
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
    return _table(columns, rows)


def confidence_table(summaries: list[Summary]) -> str | None:
    rows = [
        [
            s.system,
            str(s.confidence_signal),
            f"{s.confidence_n:,}",
            "-" if _missing(s.ece) else f"{s.ece:.3f}",
            _pct(s.coverage_at_95),
            _pct(s.coverage_at_98),
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
    return _table(columns, rows)


def blocked_table(frame: pd.DataFrame, systems: list[str]) -> str | None:
    if not (frame["outcome"] == BLOCKED).any():
        return None
    rows = []
    for system in systems:
        counts = blocked_counts(frame[frame["system"] == system])
        rows.append([system, _share(*counts[False]), _share(*counts[True])])
    return _table(["System", "Clean messages blocked", "Profane messages blocked"], rows)


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
                _pct(result["a_accuracy"]),
                _pct(result["b_accuracy"]),
                str(count(result["a_only"])),
                str(count(result["b_only"])),
                _p_value(result["p_value"]),
            ]
        )
    if not rows:
        return None
    columns = ["Pair", "Both answered", "Jev acc", "Other acc", "Only Jev right", "Only other right", "p-value"]
    return _table(columns, rows)


def slice_table(frame: pd.DataFrame, systems: list[str]) -> str:
    rows = []
    answers = frame[answered(frame)]
    for system in systems:
        for has_typos, label in ((False, "no typos"), (True, "typos")):
            group = answers[(answers["system"] == system) & (answers["has_typos"].astype(bool) == has_typos)]
            if len(group):
                rows.append([system, label, f"{len(group):,}", _pct(num(group["correct"].mean()))])
    return _table(["System", "Slice", "Answered", "Accuracy"], rows) if rows else "_No data._"


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
    return _table(["Expected", "Predicted", "Count"], rows)


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
    out += ["", "## Run conditions", "", run_conditions(frame, manifests)]

    for split in _ordered([str(s) for s in frame["split"].unique()], SPLIT_ORDER):
        split_frame = frame[frame["split"] == split]
        systems = _ordered([str(s) for s in split_frame["system"].unique()], SYSTEM_ORDER)
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
