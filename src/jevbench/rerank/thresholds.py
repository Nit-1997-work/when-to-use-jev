"""The frozen abstention thresholds (data/rerank/thresholds.yaml), tuned on the practice splits only."""

from __future__ import annotations

from pathlib import Path

import yaml

from jevbench.common.paths import data_dir

THRESHOLDS_PATH = data_dir("rerank") / "thresholds.yaml"

HEADER = """\
# Abstention thresholds: a system abstains ("nothing here is an exact match") when its confidence that some candidate
# is Exact is below its threshold. Written by `jevbench rerank tune-thresholds` from a practice run (the threshold
# with the best abstention F1 on practice + practice-nomatch), then frozen. Systems that decide with their own flag
# (Gemini) have no threshold. The report records this file's SHA-256.
"""


def write_thresholds(tuned: dict[str, tuple[float, float]], run_id: str, path: Path | None = None) -> Path:
    path = path or THRESHOLDS_PATH
    body = {
        "tuned_on_run": run_id,
        "thresholds": {system: round(threshold, 6) for system, (threshold, _) in sorted(tuned.items())},
        "practice_f1": {system: round(f1, 4) for system, (_, f1) in sorted(tuned.items())},
    }
    path.write_text(HEADER + "---\n" + yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path
