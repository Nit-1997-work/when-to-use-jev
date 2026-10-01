"""Helpers shared by every experiment's command line."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from jevbench.common.paths import REPO_ROOT

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def run_id_arg(value: str) -> str:
    """Run ids become directory names under results/, so they must be a single safe path segment."""
    if not _RUN_ID.match(value) or ".." in value:
        raise argparse.ArgumentTypeError("use letters, digits, '.', '_' or '-' (max 64 chars)")
    return value


def use_system_trust_store() -> None:
    """Trust the OS certificate store (e.g. corporate TLS inspection roots) unless disabled."""
    if os.environ.get("JEVBENCH_TRUSTSTORE", "1").strip() != "0":
        import truststore

        truststore.inject_into_ssl()


def display(path: Path) -> str:
    """A path relative to the repository when it is inside it."""
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def resolve_run_dir(results_dir: Path, run_id: str | None, run_hint: str) -> Path:
    """The named run, or the most recent one. `run_hint` is the command that creates runs, for the error message."""
    from jevbench.common.runner import latest_run_dir

    if run_id:
        path = results_dir / run_id
        if not path.is_dir():
            raise SystemExit(f"No such run: {path}")
        return path
    latest = latest_run_dir(results_dir)
    if latest is None:
        raise SystemExit(f"No runs under {display(results_dir)}/. Run `{run_hint}` first.")
    return latest


def refuse_dirty_full_run(limit: int | None, allow_dirty: bool) -> None:
    """Full runs produce publishable numbers, so they must trace to a commit. Smoke runs (--limit) are exempt."""
    from jevbench.common import runner

    if limit is None and not allow_dirty and not runner.tree_is_clean():
        raise SystemExit(
            "Refusing a full run: the working tree has uncommitted changes (or git is unavailable), so the results "
            "would not trace to a commit. Commit first, or pass --allow-dirty for a scratch run."
        )
