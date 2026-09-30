from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import jevbench.phoenix_sync as phoenix_sync
import jevbench.runner as runner
from jevbench.data import SPLITS_DIR, file_sha256, load_split
from jevbench.phoenix_sync import SyncError, dataset_name, log_experiment, register_jev_price, targets_in
from jevbench.runner import RunTarget


def _graphql(monkeypatch: pytest.MonkeyPatch, body: dict[str, object], status: int = 200) -> list[dict[str, object]]:
    sent: list[dict[str, object]] = []

    def post(url: str, *, json: dict[str, object], timeout: float) -> httpx.Response:
        sent.append(json)
        return httpx.Response(status, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(phoenix_sync.httpx, "post", post)
    return sent


def test_register_jev_price_matches_exactly_that_model(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _graphql(monkeypatch, {"data": {"createModel": {"model": {"name": "jev-1.13.0"}}}})
    result = register_jev_price("http://phoenix/", "jev-1.13.0", 0.042)
    assert result == "registered jev-1.13.0 at $0.042/M input tokens"
    pattern = sent[0]["variables"]["input"]["namePattern"]  # pyright: ignore[reportIndexIssue]
    assert re.fullmatch(pattern, "jev-1.13.0")
    assert not re.fullmatch(pattern, "jev-1x13y0")


def test_register_jev_price_is_idempotent_and_reports_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    _graphql(monkeypatch, {"errors": [{"message": "Model already exists"}]})
    assert register_jev_price("http://phoenix", "jev-1", 0.042) == "jev-1 price already registered"
    _graphql(monkeypatch, {"errors": [{"message": "forbidden"}]})
    with pytest.raises(SyncError, match="forbidden"):
        register_jev_price("http://phoenix", "jev-1", 0.042)
    _graphql(monkeypatch, {}, status=503)
    with pytest.raises(SyncError, match="Cannot reach Phoenix"):
        register_jev_price("http://phoenix", "jev-1", 0.042)


def test_dataset_names_are_content_addressed() -> None:
    assert dataset_name("practice") == f"bitext-retail-practice-{file_sha256(SPLITS_DIR / 'practice.jsonl')[:8]}"


def test_targets_in_a_run_directory(tmp_path: Path) -> None:
    for relative in ("jev/main/repeat-1.jsonl", "jev/main/repeat-2.jsonl", "baseline/main/repeat-x.jsonl"):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text("", encoding="utf-8")
    assert targets_in(tmp_path) == [
        RunTarget(tmp_path.name, "jev", "main", 1),
        RunTarget(tmp_path.name, "jev", "main", 2),
    ]


class _FakeClient:
    def __init__(self) -> None:
        self.runs: list[dict[str, object]] = []
        self.evaluations: list[dict[str, object]] = []
        self.datasets = SimpleNamespace(get_dataset=self._missing, create_dataset=self._create)
        self.experiments = SimpleNamespace(
            create=lambda **_: {"id": "exp"},
            log_run=self._log_run,
            log_evaluation=lambda **kwargs: self.evaluations.append(kwargs),
            get_experiment_url=lambda dataset_id, experiment_id: f"http://phoenix/{dataset_id}/{experiment_id}",
        )

    @staticmethod
    def _missing(**_: object) -> None:
        raise ValueError("no such dataset")

    @staticmethod
    def _create(**kwargs: object) -> SimpleNamespace:
        metadata = kwargs["metadata"]
        assert isinstance(metadata, list)
        return SimpleNamespace(id="ds", examples=[{"id": f"px-{m['example_id']}", "metadata": m} for m in metadata])

    def _log_run(self, **kwargs: object) -> dict[str, object]:
        self.runs.append(kwargs)
        return {"id": f"run-{len(self.runs)}"}


@pytest.fixture
def target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunTarget:
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    return RunTarget("run", "jev", "practice", 1)


def _write_target(target: RunTarget, records: list[dict[str, object]], manifest: dict[str, object] | None) -> None:
    target.path.parent.mkdir(parents=True, exist_ok=True)
    target.path.write_text("".join(json.dumps(r) + "\n" for r in records) + "\n", encoding="utf-8")
    if manifest is not None:
        target.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_log_experiment_links_records_to_dataset_examples(target: RunTarget, monkeypatch: pytest.MonkeyPatch) -> None:
    example = load_split("practice")[0]
    record = {
        "example_id": example.id,
        "predicted_intent": example.intent,
        "confidence": 0.9,
        "outcome": "ok",
        "latency_ms": 120.0,
        "input_tokens": 2000,
        "output_tokens": 400,
        "reasoning_tokens": 0,
        "started_at": "2026-09-29T20:00:00.000+00:00",
        "trace_id": "abc",
        "error": None,
        "correct": True,
        "category_correct": True,
    }
    stale = {**record, "example_id": "not-in-this-split"}
    manifest = {
        "model_requested": "jev-test",
        "intents_sha256": "i",
        "split_sha256": file_sha256(SPLITS_DIR / "practice.jsonl"),
        "git": {"commit": "abc"},
    }
    _write_target(target, [record, stale], manifest)
    client = _FakeClient()
    monkeypatch.setattr(phoenix_sync, "_client", lambda _: client)

    assert log_experiment(target, "http://phoenix") == "http://phoenix/ds/exp"
    assert [run["dataset_example_id"] for run in client.runs] == [f"px-{example.id}"]  # the stale record is skipped
    assert [(e["name"], e["label"]) for e in client.evaluations] == [
        ("intent_correct", "correct"),
        ("category_correct", "correct"),
    ]


def test_log_experiment_refuses_unpublishable_or_stale_results(target: RunTarget) -> None:
    with pytest.raises(FileNotFoundError):
        log_experiment(target, None)
    _write_target(target, [], manifest=None)
    with pytest.raises(SyncError, match="Missing manifest"):
        log_experiment(target, None)
    _write_target(target, [], manifest={"model_requested": "jev-test", "split_sha256": "0" * 64})
    with pytest.raises(SyncError, match="different version of the practice split"):
        log_experiment(target, None)
    target.manifest_path.write_text("{broken", encoding="utf-8")
    with pytest.raises(SyncError, match="not valid JSON"):
        log_experiment(target, None)
