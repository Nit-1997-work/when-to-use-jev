"""Markdown building blocks shared by every experiment's report: number formats, tables, manifests, run conditions."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pandas as pd

from jevbench.common.settings import LLM_SAFETY_SETTINGS, Prices
from jevbench.common.stats import count

# Run manifests keyed by (system, split, repeat).
type Manifests = Mapping[tuple[str, str, int], Mapping[str, object]]


class ReportError(RuntimeError):
    pass


# ---------- formatting ----------


def missing(value: float | None) -> bool:
    return value is None or math.isnan(value)


def pct(value: float | None) -> str:
    return "-" if value is None or math.isnan(value) else f"{value * 100:.1f}%"


def ms(value: float | None) -> str:
    return "-" if value is None or math.isnan(value) else f"{value:,.0f}"


def number(value: float | None, decimals: int) -> str:
    return "-" if value is None or math.isnan(value) else f"{value:,.{decimals}f}"


def usd(value: float | None, decimals: int = 4) -> str:
    return "n/a (no price recorded)" if value is None or math.isnan(value) else f"${value:,.{decimals}f}"


def price_text(value: float | None) -> str:
    if value is None:
        return "not set"
    text = f"{value:.4f}".rstrip("0")
    return f"${text}" if len(text.split(".")[1]) >= 2 else f"${value:.2f}"


def p_value(value: float) -> str:
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def share(part: int, whole: int) -> str:
    return "-" if whole == 0 else f"{part:,} of {whole:,} ({part / whole:.1%})"


def ordered(values: list[str], order: tuple[str, ...]) -> list[str]:
    return sorted(values, key=lambda v: (order.index(v) if v in order else len(order), v))


def md_table(columns: Sequence[str], rows: list[list[str]]) -> str:
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


def manifest_settings(manifest: Mapping[str, object]) -> Mapping[str, object]:
    settings = manifest.get("settings")
    return settings if isinstance(settings, dict) else {}


def setting_values(manifests: Manifests, key: str) -> list[str]:
    """Distinct non-empty settings values across manifests, in a stable order."""
    return sorted({str(v) for m in manifests.values() if (v := manifest_settings(m).get(key)) not in (None, "")})


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


# ---------- run conditions ----------


def run_conditions(
    frame: pd.DataFrame,
    manifests: Manifests,
    *,
    system_order: tuple[str, ...],
    split_order: tuple[str, ...],
    reasoning_effort: Callable[[str, Mapping[str, object]], str],
    prices: Callable[[Mapping[str, object]], Prices | None],
    calls_column: str = "Calls",
    pacing: str = "- Each system sent one request at a time.",
    extra_bullets: Sequence[str] = (),
) -> str:
    """Models, prices, timing windows, and environment, all read from the run's own records and manifests."""
    systems = ordered([str(s) for s in frame["system"].unique()], system_order)
    by_system = {system: [m for (s, _, _), m in sorted(manifests.items()) if s == system] for system in systems}

    model_rows = []
    for system in systems:
        first = by_system[system][0] if by_system[system] else {}
        reported = sorted({str(m) for m in frame.loc[frame["system"] == system, "model_reported"].dropna()})
        recorded = prices(first) if first else None
        price = (
            " / ".join(price_text(p) for p in (recorded.input, recorded.output, recorded.effective_thinking))
            if recorded
            else "not recorded"
        )
        model_rows.append(
            [
                system,
                str(first.get("model_requested", "-")),
                ", ".join(reported) or "-",
                reasoning_effort(system, first),
                price,
            ]
        )
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
        for split in ordered([str(s) for s in frame.loc[is_system, "split"].unique()], split_order):
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
        f"- LLM safety settings: {', '.join(setting_values(manifests, 'llm_safety_settings')) or LLM_SAFETY_SETTINGS}.",
        f"- LLM prices source: {'; '.join(setting_values(manifests, 'price_source_url')) or 'not recorded'}.",
    ]
    if jev_source := setting_values(manifests, "jev_price_source_url"):
        bullets.append(f"- Jev price source: {'; '.join(jev_source)}.")
    bullets += [
        f"- Client location: {'; '.join(setting_values(manifests, 'client_location')) or 'not recorded'}."
        f" Request timeout: {'; '.join(setting_values(manifests, 'request_timeout_s')) or 'not recorded'} s.",
        f"- Code: commit {'; '.join(git) or 'not recorded'}. {'; '.join(packages)}.",
        *extra_bullets,
        pacing
        + (
            " The systems' runs overlapped in time (see the windows above), so they shared the client machine and"
            " network."
            if overlapping
            else ""
        ),
    ]
    return "\n".join(
        [
            md_table(model_columns, model_rows),
            "",
            md_table(["System", "Split", calls_column, "First to last call (UTC)"], window_rows),
            "",
            *bullets,
        ]
    )
