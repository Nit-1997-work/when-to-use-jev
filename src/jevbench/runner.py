"""Runs one system over one split, sequentially, writing one JSONL record per classification.

Records are the source of truth. Phoenix spans and experiments are views derived from the same calls.
"""

from __future__ import annotations

import asyncio
import json
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from jevbench.classifiers import Classifier, Outcome, Prediction, RetryableError
from jevbench.data import SPLITS_DIR, Example, file_sha256, load_split
from jevbench.intents import IntentCatalog
from jevbench.settings import REPO_ROOT, Settings
from jevbench.tracing import classification_span, current_trace_id, record_outcome, tracer

RESULTS_DIR = REPO_ROOT / "results"
MAX_ATTEMPTS = 5
BACKOFF_BASE_S = 1.0
BACKOFF_MAX_S = 20.0
RETRY_AFTER_MAX_S = 120.0  # never wait longer than this for a server-requested Retry-After
# Manifest fields that must not change while one results file is filled across several sessions.
RESUME_KEYS = ("system", "split", "repeat", "model_requested", "intents_sha256", "split_sha256", "settings")
PACKAGES = ("typesafe-sdk", "openai", "httpx2", "arize-phoenix-otel")


@dataclass(frozen=True)
class RunTarget:
    run_id: str
    system: str
    split: str
    repeat: int

    @property
    def path(self) -> Path:
        return RESULTS_DIR / self.run_id / self.system / self.split / f"repeat-{self.repeat}.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.path.with_suffix(".manifest.json")

    @property
    def label(self) -> str:
        return f"{self.system}/{self.split}/r{self.repeat}"


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _now() -> str:
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


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def build_manifest(
    target: RunTarget, settings: Settings, catalog: IntentCatalog, classifier: Classifier
) -> dict[str, object]:
    """Everything needed to reproduce or audit the numbers of one (system, split, repeat)."""
    prices = settings.prices_for(target.system)
    return {
        "run_id": target.run_id,
        "system": target.system,
        "split": target.split,
        "repeat": target.repeat,
        "model_requested": classifier.model_requested,
        "written_at": _now(),
        "intents_sha256": catalog.sha256,
        "split_sha256": file_sha256(SPLITS_DIR / f"{target.split}.jsonl"),
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


def ensure_manifest(target: RunTarget, manifest: dict[str, object], *, has_records: bool) -> None:
    """Write the manifest when a target starts. When resuming, keep the original and refuse a changed configuration."""
    existing = read_manifest(target.manifest_path)
    if has_records:
        if existing is None:
            raise SystemExit(f"{target.path} has records but no manifest. Start a new --run-id.")
        changed = [key for key in RESUME_KEYS if existing.get(key) != manifest.get(key)]
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


async def classify_with_retries(classifier: Classifier, text: str) -> tuple[Prediction, int, float]:
    """Returns (prediction, attempts, total backoff seconds). Backoff is never part of `latency_ms`."""
    waited = 0.0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return await classifier.classify(text), attempt, waited
        except RetryableError as exc:
            if attempt == MAX_ATTEMPTS:
                failed = Prediction(outcome=Outcome.API_ERROR, intent=None, latency_ms=0.0, error=str(exc))
                return failed, attempt, waited
            delay = min(BACKOFF_MAX_S, BACKOFF_BASE_S * 2 ** (attempt - 1)) * (1 + random.random() * 0.25)
            if exc.retry_after_s is not None:
                delay = max(delay, min(exc.retry_after_s, RETRY_AFTER_MAX_S))
            await asyncio.sleep(delay)
            waited += delay
    raise AssertionError("unreachable")


def build_record(
    target: RunTarget,
    classifier: Classifier,
    catalog: IntentCatalog,
    example: Example,
    prediction: Prediction,
    attempts: int,
    backoff_s: float,
    started_at: str,
    trace_id: str | None,
) -> dict[str, object]:
    predicted_category = catalog.category_of(prediction.intent) if prediction.intent else None
    top2 = None
    if prediction.probabilities:
        ranked = sorted(prediction.probabilities.items(), key=lambda kv: kv[1], reverse=True)
        top2 = [name for name, _ in ranked[:2]]
    measured = {k: (v.value if isinstance(v, Outcome) else v) for k, v in asdict(prediction).items() if k != "intent"}
    return {
        "run_id": target.run_id,
        "system": target.system,
        "split": target.split,
        "repeat": target.repeat,
        "example_id": example.id,
        "text": example.text,
        "expected_intent": example.intent,
        "expected_category": example.category,
        "tags": example.tags,
        "has_profanity": example.has_profanity,
        "has_typos": example.has_typos,
        "predicted_intent": prediction.intent,
        "predicted_category": predicted_category,
        "correct": prediction.intent == example.intent,
        "category_correct": predicted_category == example.category,
        "top2": top2,
        "model_requested": classifier.model_requested,
        "attempts": attempts,
        "backoff_s": round(backoff_s, 3),
        "started_at": started_at,
        "trace_id": trace_id,
        **measured,
    }


async def warm_up(classifier: Classifier, count: int) -> None:
    """Discarded calls that open connections before timing.

    Traced under a `warmup` span so Phoenix dashboards can exclude them.
    """
    with tracer().start_as_current_span("warmup", attributes={"jevbench.warmup": True}):
        for example in load_split("warmup")[:count]:
            await classify_with_retries(classifier, example.text)


async def run_target(
    target: RunTarget,
    classifier: Classifier,
    settings: Settings,
    catalog: IntentCatalog,
    *,
    limit: int | None,
    warmup: int,
    progress_every: int = 50,
) -> Path:
    examples = load_split(target.split)
    if limit is not None:
        examples = examples[:limit]
    if repair_torn_tail(target.path):
        print(f"[{target.label}] removed a half-written last line; that example will be redone")
    done = completed_ids(target.path)
    pending = [e for e in examples if e.id not in done]
    print(f"[{target.label}] {len(pending)} to run, {len(done)} already done")
    if not pending:
        return target.path

    has_records = target.path.exists() and target.path.stat().st_size > 0
    ensure_manifest(target, build_manifest(target, settings, catalog, classifier), has_records=has_records)

    if warmup:
        await warm_up(classifier, warmup)

    correct = 0
    t0 = time.perf_counter()
    target.path.parent.mkdir(parents=True, exist_ok=True)
    with target.path.open("a", encoding="utf-8") as out:
        for i, example in enumerate(pending, start=1):
            started_at = _now()
            with classification_span(target.system, example.id, target.split, target.repeat, example.text) as span:
                prediction, attempts, backoff_s = await classify_with_retries(classifier, example.text)
                trace_id = current_trace_id()
                record = build_record(
                    target, classifier, catalog, example, prediction, attempts, backoff_s, started_at, trace_id
                )
                record_outcome(
                    span,
                    expected=example.intent,
                    predicted=prediction.intent,
                    outcome=str(record["outcome"]),
                    correct=bool(record["correct"]),
                )
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            correct += bool(record["correct"])
            if i % progress_every == 0 or i == len(pending):
                elapsed = time.perf_counter() - t0
                print(f"  {i}/{len(pending)}  acc={correct / i:.3f}  elapsed={elapsed:.0f}s")
    return target.path
