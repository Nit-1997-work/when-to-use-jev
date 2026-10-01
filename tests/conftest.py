from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from jevbench.common.settings import LLMSystemConfig, Prices, Settings

RecordFactory = Callable[..., dict[str, object]]
RunWriter = Callable[..., Path]


@pytest.fixture
def settings() -> Settings:
    """Offline settings: fake endpoints and keys, no LLM prices, tracing off."""
    return Settings(
        gateway_base_url="http://gateway.invalid/v1",
        gateway_api_key="test",
        baseline=LLMSystemConfig(model="google/test-model", reasoning_effort=None, prices=Prices(None, None, None)),
        baseline_2=None,
        typesafe_api_key="test",
        typesafe_base_url="http://typesafe.invalid",
        typesafe_model="jev-test",
        jev_input_price=0.042,
        jev_price_source_url=None,
        price_source_url=None,
        request_timeout_s=5,
        client_location=None,
        phoenix_collector_endpoint=None,
        phoenix_base_url=None,
        phoenix_project_prefix="test",
    )


def _record(**overrides: object) -> dict[str, object]:
    """One result record as the runner writes it: by default a correct, timed answer."""
    record: dict[str, object] = {
        "run_id": "run",
        "system": "jev",
        "split": "main",
        "repeat": 1,
        "example_id": "e1",
        "text": "where is my order",
        "expected_intent": "track_order",
        "expected_category": "ORDER",
        "tags": "B",
        "has_profanity": False,
        "has_typos": False,
        "predicted_intent": "track_order",
        "predicted_category": "ORDER",
        "correct": True,
        "category_correct": True,
        "top2": None,
        "model_requested": "model-x",
        "attempts": 1,
        "backoff_s": 0.0,
        "started_at": "2026-09-29T20:00:00.000+00:00",
        "trace_id": None,
        "outcome": "ok",
        "latency_ms": 100.0,
        "model_reported": "model-x",
        "confidence": None,
        "probabilities": None,
        "input_tokens": 1000,
        "output_tokens": 10,
        "reasoning_tokens": 0,
        "gateway_upstream_ms": None,
        "gateway_overhead_ms": None,
        "cache_hit": False,
        "error": None,
    }
    record.update(overrides)
    return record


@pytest.fixture
def make_record() -> RecordFactory:
    return _record


def _manifest(system: str, split: str, repeat: int, **overrides: object) -> dict[str, object]:
    manifest: dict[str, object] = {
        "run_id": "run",
        "system": system,
        "split": split,
        "repeat": repeat,
        "model_requested": "model-x",
        "written_at": "2026-09-29T20:00:00.000+00:00",
        "intents_sha256": "i" * 64,
        "split_sha256": "s" * 64,
        "prices_per_mtok": {"input": 1.0, "output": 2.0, "thinking": 2.0},
        "settings": {"client_location": "test bench", "request_timeout_s": 30.0, "price_source_url": "https://prices"},
        "git": {"commit": "0123456789abcdef", "dirty": False},
        "python": "3.12.0",
        "packages": {"openai": "1.0"},
    }
    manifest.update(overrides)
    return manifest


@pytest.fixture
def write_run(tmp_path: Path) -> RunWriter:
    """Write records (and one manifest per system/split/repeat) as a run directory; returns its path."""

    def write(
        records: list[dict[str, object]],
        *,
        run_id: str = "run",
        manifests: dict[tuple[str, str, int], dict[str, object]] | None = None,
        root: Path | None = None,
    ) -> Path:
        run_dir = (root or tmp_path) / run_id
        groups: dict[tuple[str, str, int], list[dict[str, object]]] = {}
        for record in records:
            key = (str(record["system"]), str(record["split"]), int(str(record["repeat"])))
            groups.setdefault(key, []).append(record)
        for (system, split, repeat), rows in groups.items():
            path = run_dir / system / split / f"repeat-{repeat}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            manifest = (manifests or {}).get((system, split, repeat)) or _manifest(system, split, repeat)
            path.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return run_dir

    return write
