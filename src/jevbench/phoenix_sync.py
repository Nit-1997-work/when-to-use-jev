"""Mirror splits and finished runs into Phoenix as datasets and experiments.

The JSONL records stay the source of truth. This makes them browsable side by side in the Phoenix UI,
linked to the traces recorded during the run (`trace_id`).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from phoenix.client import Client

from jevbench.data import SPLITS_DIR, file_sha256, load_split
from jevbench.metrics import count, num
from jevbench.runner import RunTarget

EVALUATIONS = ("intent_correct", "category_correct")
_REPEAT_FILE = re.compile(r"^repeat-(\d+)$")


class SyncError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SyncError(f"Cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SyncError(f"{path} is not valid JSON: {exc}") from exc


def _opt_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _read_records(path: Path) -> list[dict[str, object]]:
    """Result records; a torn last line from an interrupted run is skipped, like the runner does."""
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _client(base_url: str | None) -> Client:
    return Client(base_url=base_url)


_CREATE_MODEL = """
mutation($input: CreateModelMutationInput!) { createModel(input: $input) { model { name } } }
"""


def register_jev_price(base_url: str, model: str, input_price_per_mtok: float) -> str:
    """Teach Phoenix Jev's price so its cost dashboards include Jev.

    Phoenix prices spans when they arrive, so run this before a run, not after. Idempotent. The gateway's Gemini
    model names (`google/gemini-*`) are not in Phoenix's built-in price table; add them in Settings > Models.
    """
    variables = {
        "input": {
            "name": model,
            "provider": "typesafe",
            "namePattern": f"^{re.escape(model)}$",
            "costs": [
                {"tokenType": "input", "kind": "PROMPT", "costPerMillionTokens": input_price_per_mtok},
                {"tokenType": "output", "kind": "COMPLETION", "costPerMillionTokens": 0.0},  # Jev output is free
            ],
        }
    }
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/graphql", json={"query": _CREATE_MODEL, "variables": variables}, timeout=30
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise SyncError(f"Cannot reach Phoenix at {base_url}: {exc}") from exc
    errors = response.json().get("errors")
    if not errors:
        return f"registered {model} at ${input_price_per_mtok}/M input tokens"
    message = "; ".join(str(e.get("message")) for e in errors)
    if "already exists" in message.lower() or "unique" in message.lower():
        return f"{model} price already registered"
    raise SyncError(f"Phoenix rejected the {model} price: {message}")


def dataset_name(split: str) -> str:
    """Content-addressed: a changed split file becomes a new dataset, never a silent overwrite."""
    return f"bitext-retail-{split}-{file_sha256(SPLITS_DIR / f'{split}.jsonl')[:8]}"


def ensure_dataset(split: str, base_url: str | None) -> tuple[str, dict[str, str]]:
    """Create the Phoenix dataset for a split if needed. Returns (dataset_id, {our example id: phoenix example id})."""
    client = _client(base_url)
    name = dataset_name(split)
    try:
        dataset = client.datasets.get_dataset(dataset=name, timeout=60)
    except (httpx.HTTPStatusError, ValueError):
        examples = load_split(split)
        dataset = client.datasets.create_dataset(
            name=name,
            dataset_description=f"Bitext Retail eCommerce intents, `{split}` split built by `jevbench prepare`.",
            inputs=[{"customer_message": e.text} for e in examples],
            outputs=[{"intent": e.intent, "category": e.category} for e in examples],
            metadata=[
                {"example_id": e.id, "tags": e.tags, "has_profanity": e.has_profanity, "has_typos": e.has_typos}
                for e in examples
            ],
            timeout=120,
        )
    id_map = {str(ex["metadata"]["example_id"]): str(ex["id"]) for ex in dataset.examples}
    return dataset.id, id_map


def _git_commit(manifest: dict[str, object]) -> str | None:
    git = manifest.get("git")
    return _opt_str(git.get("commit")) if isinstance(git, dict) else None


def log_experiment(target: RunTarget, base_url: str | None) -> str:
    """Log one finished (system, split, repeat) as a Phoenix experiment with per-example evaluations."""
    if not target.path.exists():
        raise FileNotFoundError(f"No results at {target.path}.")
    manifest_path = target.path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise SyncError(f"Missing manifest {manifest_path}; results without a manifest are not publishable.")
    manifest = _read_json(manifest_path)
    # Example ids are stable across split versions, so records from another version would link to the wrong rows.
    recorded_split = _opt_str(manifest.get("split_sha256"))
    if recorded_split is not None and recorded_split != file_sha256(SPLITS_DIR / f"{target.split}.jsonl"):
        raise SyncError(f"{target.path} was recorded on a different version of the {target.split} split.")
    dataset_id, id_map = ensure_dataset(target.split, base_url)

    client = _client(base_url)
    experiment = client.experiments.create(
        dataset_id=dataset_id,
        experiment_name=f"{target.system} | {target.split} | r{target.repeat} | {target.run_id}",
        experiment_description=f"jevbench run {target.run_id}: {target.system} ({manifest['model_requested']}).",
        experiment_metadata={
            "run_id": target.run_id,
            "system": target.system,
            "model_requested": manifest["model_requested"],
            "repeat": target.repeat,
            "intents_sha256": manifest["intents_sha256"],
            "git_commit": _git_commit(manifest),
        },
    )

    for record in _read_records(target.path):
        example_id = id_map.get(str(record["example_id"]))
        if example_id is None:
            continue  # result from a different split version; skip rather than mislink
        start = datetime.fromisoformat(str(record["started_at"]))
        latency_ms = num(record["latency_ms"])
        end = start + timedelta(milliseconds=0.0 if latency_ms != latency_ms else latency_ms)  # NaN -> 0
        run = client.experiments.log_run(
            experiment_id=experiment["id"],
            dataset_example_id=example_id,
            output={
                "intent": record["predicted_intent"],
                "confidence": record["confidence"],
                "outcome": record["outcome"],
                "latency_ms": record["latency_ms"],
                "input_tokens": record["input_tokens"],
                "output_tokens": record["output_tokens"],
                "reasoning_tokens": record["reasoning_tokens"],
            },
            start_time=start,
            end_time=end,
            trace_id=_opt_str(record["trace_id"]),
            error=_opt_str(record["error"]) if record["outcome"] != "ok" else None,
        )
        for name, field in zip(EVALUATIONS, ("correct", "category_correct"), strict=True):
            client.experiments.log_evaluation(
                experiment_run_id=run["id"],
                name=name,
                score=1.0 if record[field] else 0.0,
                label="correct" if record[field] else "wrong",
            )
    return client.experiments.get_experiment_url(dataset_id=dataset_id, experiment_id=experiment["id"])


def targets_in(run_dir: Path) -> list[RunTarget]:
    """Every results/<run>/<system>/<split>/repeat-<n>.jsonl under a run directory."""
    targets: list[RunTarget] = []
    for path in sorted(run_dir.glob("*/*/repeat-*.jsonl")):
        match = _REPEAT_FILE.match(path.stem)
        if match is None:
            continue
        targets.append(
            RunTarget(
                run_id=run_dir.name,
                system=path.parent.parent.name,
                split=path.parent.name,
                repeat=count(match.group(1)),  # the regex guarantees digits
            )
        )
    return targets
