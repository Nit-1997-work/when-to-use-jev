"""The run machinery every experiment shares: result locations, manifests, resume, retries, and the run loop.

One run writes `results/<experiment>/<run_id>/<system>/<split>/repeat-<n>.jsonl`, one JSONL record per example, next
to a manifest that records everything needed to reproduce or audit the numbers. Records are the source of truth.
"""

from __future__ import annotations

import asyncio
import json
import platform
import random
import subprocess
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from jevbench.common.calls import Outcome, RetryableError
from jevbench.common.paths import REPO_ROOT
from jevbench.common.settings import Prices, Settings

MAX_ATTEMPTS = 5
BACKOFF_BASE_S = 1.0
BACKOFF_MAX_S = 20.0
RETRY_AFTER_MAX_S = 120.0  # never wait longer than this for a server-requested Retry-After
PACKAGES = ("typesafe-sdk", "openai", "httpx2", "arize-phoenix-otel")


@dataclass(frozen=True)
class RunTarget:
    """One (system, split, repeat) of a run, under an experiment's results directory."""

    root: Path
    run_id: str
    system: str
    split: str
    repeat: int

    @property
    def path(self) -> Path:
        return self.root / self.run_id / self.system / self.split / f"repeat-{self.repeat}.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.path.with_suffix(".manifest.json")

    @property
    def label(self) -> str:
        return f"{self.system}/{self.split}/r{self.repeat}"


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def git_state() -> dict[str, object]:
    """Commit and whether the working tree has uncommitted changes (`None` when git is unavailable)."""

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def tree_is_clean() -> bool:
    """True only when git confirms there are no uncommitted changes; an unknown state does not count as clean."""
    dirty = git_state()["dirty"]
    return isinstance(dirty, bool) and not dirty


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def base_manifest(
    target: RunTarget,
    *,
    model_requested: str,
    data_hashes: Mapping[str, str],
    prices: Prices | None,
    settings: Settings,
) -> dict[str, object]:
    """Everything needed to reproduce or audit the numbers of one (system, split, repeat)."""
    return {
        "run_id": target.run_id,
        "system": target.system,
        "split": target.split,
        "repeat": target.repeat,
        "model_requested": model_requested,
        "written_at": now(),
        **data_hashes,
        "prices_per_mtok": prices.as_dict() if prices is not None else None,
        "settings": settings.public_snapshot(),
        "git": git_state(),
        "python": platform.python_version(),
        "packages": {p: _package_version(p) for p in PACKAGES},
    }


def read_manifest(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON ({exc}). Fix or remove it before resuming.") from None
    if not isinstance(data, dict):
        raise SystemExit(f"{path} is not a manifest object. Fix or remove it before resuming.")
    return data


def ensure_manifest(
    target: RunTarget, manifest: Mapping[str, object], *, has_records: bool, resume_keys: Sequence[str]
) -> None:
    """Write the manifest when a target starts. When resuming, keep the original and refuse a changed configuration."""
    existing = read_manifest(target.manifest_path)
    if has_records:
        if existing is None:
            raise SystemExit(f"{target.path} has records but no manifest. Start a new --run-id.")
        changed = [key for key in resume_keys if existing.get(key) != manifest.get(key)]
        if changed:
            raise SystemExit(
                f"Cannot resume {target.label}: {', '.join(changed)} changed since its records were written. "
                "Restore the original configuration or start a new --run-id."
            )
        return
    target.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    target.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def repair_torn_tail(path: Path) -> bool:
    """Drop a half-written last line (from a killed run), so the next record starts on a line of its own."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb+") as fh:
        fh.seek(-1, 2)
        if fh.read(1) == b"\n":
            return False
        fh.seek(0)
        keep = fh.read().rfind(b"\n") + 1  # 0 when there is no complete line at all
        fh.truncate(keep)
    return True


def completed_ids(path: Path) -> set[str]:
    """Examples that already have a final record. An API error is not final: a resumed run redoes it."""
    if not path.exists():
        return set()
    latest: dict[str, object] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                record = json.loads(line)
                latest[str(record["example_id"])] = record.get("outcome")
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                continue
    return {example_id for example_id, outcome in latest.items() if outcome != Outcome.API_ERROR}


def latest_run_dir(results_dir: Path) -> Path | None:
    """The run with the most recently written manifest. Run ids are free-form, so their names do not sort by time."""
    if not results_dir.is_dir():
        return None
    newest: tuple[str, Path] | None = None
    for run_dir in (p for p in results_dir.iterdir() if p.is_dir()):
        for manifest_path in run_dir.glob("*/*/repeat-*.manifest.json"):
            try:
                written_at = str(json.loads(manifest_path.read_text(encoding="utf-8")).get("written_at", ""))
            except (OSError, json.JSONDecodeError, AttributeError):
                continue
            if newest is None or written_at > newest[0]:
                newest = (written_at, run_dir)
    return newest[1] if newest else None


async def call_with_retries[T](call: Callable[[], Awaitable[T]], exhausted: Callable[[str], T]) -> tuple[T, int, float]:
    """Returns (result, attempts, total backoff seconds). Backoff is never part of a measured latency.

    `call` raises `RetryableError` for transient failures; after `MAX_ATTEMPTS` the error message goes to `exhausted`,
    which builds the API-error result.
    """
    waited = 0.0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return await call(), attempt, waited
        except RetryableError as exc:
            if attempt == MAX_ATTEMPTS:
                return exhausted(str(exc)), attempt, waited
            delay = min(BACKOFF_MAX_S, BACKOFF_BASE_S * 2 ** (attempt - 1)) * (1 + random.random() * 0.25)
            if exc.retry_after_s is not None:
                delay = max(delay, min(exc.retry_after_s, RETRY_AFTER_MAX_S))
            await asyncio.sleep(delay)
            waited += delay
    raise AssertionError("unreachable")


async def run_records[E](
    target: RunTarget,
    examples: Sequence[E],
    *,
    example_id: Callable[[E], str],
    manifest: Callable[[], Mapping[str, object]],
    resume_keys: Sequence[str],
    warm_up: Callable[[], Awaitable[None]] | None,
    run_one: Callable[[E], Awaitable[dict[str, object]]],
    progress: Callable[[list[dict[str, object]]], str],
    progress_every: int = 50,
) -> Path:
    """Run every example that has no final record yet, appending one record per example, in order.

    Resumable: a killed run's half-written line is dropped, finished examples are skipped, API errors are redone,
    and a changed configuration (`resume_keys`) is refused. `progress` summarizes the records written so far.
    """
    if repair_torn_tail(target.path):
        print(f"[{target.label}] removed a half-written last line; that example will be redone")
    done = completed_ids(target.path)
    pending = [e for e in examples if example_id(e) not in done]
    print(f"[{target.label}] {len(pending)} to run, {len(done)} already done")
    if not pending:
        return target.path

    has_records = target.path.exists() and target.path.stat().st_size > 0
    ensure_manifest(target, manifest(), has_records=has_records, resume_keys=resume_keys)

    if warm_up is not None:
        await warm_up()

    written: list[dict[str, object]] = []
    t0 = time.perf_counter()
    target.path.parent.mkdir(parents=True, exist_ok=True)
    with target.path.open("a", encoding="utf-8") as out:
        for i, example in enumerate(pending, start=1):
            record = await run_one(example)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            written.append(record)
            if i % progress_every == 0 or i == len(pending):
                elapsed = time.perf_counter() - t0
                print(f"  {i}/{len(pending)}  {progress(written)}  elapsed={elapsed:.0f}s")
    return target.path
