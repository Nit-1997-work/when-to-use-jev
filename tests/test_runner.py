from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import jevbench.runner as runner
from jevbench.classifiers import Outcome, Prediction, RetryableError
from jevbench.data import load_split
from jevbench.intents import load_catalog
from jevbench.settings import Settings

if TYPE_CHECKING:
    from tests.conftest import RecordFactory, RunWriter

PRACTICE = {example.text: example.intent for example in load_split("practice")}


class FakeClassifier:
    """Answers correctly unless a scripted failure is queued."""

    def __init__(self, *failures: Exception, wrong: bool = False) -> None:
        self.name = "jev"
        self.model_requested = "jev-test"
        self.failures = list(failures)
        self.wrong = wrong
        self.calls = 0

    async def classify(self, text: str) -> Prediction:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        intent = "not_an_intent" if self.wrong else PRACTICE.get(text, "track_order")
        return Prediction(outcome=Outcome.OK, intent=intent, latency_ms=10.0, input_tokens=100, output_tokens=5)

    async def aclose(self) -> None:
        return None


@pytest.fixture
def results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record backoff sleeps instead of waiting."""
    recorded: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)
    return recorded


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


async def test_retries_are_counted_and_backoff_is_kept_out_of_latency(sleeps: list[float]) -> None:
    classifier = FakeClassifier(RetryableError("503"))
    prediction, attempts, waited = await runner.classify_with_retries(classifier, "hi")
    assert prediction.outcome == Outcome.OK
    assert prediction.latency_ms == 10.0
    assert attempts == 2
    assert waited == pytest.approx(sum(sleeps))
    assert 1.0 <= sleeps[0] <= 1.25  # first backoff: 1 s plus up to 25% jitter


async def test_retry_after_is_honoured_and_capped(sleeps: list[float]) -> None:
    classifier = FakeClassifier(RetryableError("429", retry_after_s=7.0), RetryableError("429", retry_after_s=1e6))
    await runner.classify_with_retries(classifier, "hi")
    assert sleeps[0] == 7.0
    assert sleeps[1] == runner.RETRY_AFTER_MAX_S


async def test_exhausted_retries_become_an_api_error(sleeps: list[float]) -> None:
    classifier = FakeClassifier(*[RetryableError("down") for _ in range(runner.MAX_ATTEMPTS)])
    prediction, attempts, _ = await runner.classify_with_retries(classifier, "hi")
    assert (prediction.outcome, attempts) == (Outcome.API_ERROR, runner.MAX_ATTEMPTS)
    assert prediction.error == "down"
    assert len(sleeps) == runner.MAX_ATTEMPTS - 1


def _target(repeat: int = 1) -> runner.RunTarget:
    return runner.RunTarget(run_id="run", system="jev", split="practice", repeat=repeat)


async def _run(settings: Settings, classifier: FakeClassifier, *, limit: int = 3, warmup: int = 0) -> Path:
    return await runner.run_target(_target(), classifier, settings, load_catalog(), limit=limit, warmup=warmup)


async def test_run_writes_records_and_a_manifest(results_dir: Path, settings: Settings) -> None:
    private = replace(
        settings,
        gateway_base_url="https://private-gateway.invalid/v1",
        gateway_api_key="gateway-secret-value",
        typesafe_api_key="typesafe-secret-value",
    )
    path = await _run(private, FakeClassifier(), warmup=2)
    records = _records(path)
    assert [r["correct"] for r in records] == [True, True, True]
    assert "intent" not in records[0]  # `predicted_intent` is the one copy of the answer
    assert records[0]["outcome"] == "ok"
    manifest_text = _target().manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["model_requested"] == "jev-test"
    assert manifest["prices_per_mtok"] == {"input": 0.042, "output": 0.0, "thinking": 0.0}
    assert manifest["settings"]["llm_safety_settings"] == "provider defaults (not overridden)"
    for private_value in ("private-gateway", "gateway-secret-value", "typesafe-secret-value"):
        assert private_value not in manifest_text


async def test_resume_skips_finished_examples_and_keeps_the_manifest(results_dir: Path, settings: Settings) -> None:
    await _run(settings, FakeClassifier(), limit=2)
    manifest_before = _target().manifest_path.read_text(encoding="utf-8")
    classifier = FakeClassifier()
    path = await _run(settings, classifier, limit=3)
    assert classifier.calls == 1
    assert len(_records(path)) == 3
    assert _target().manifest_path.read_text(encoding="utf-8") == manifest_before

    await _run(settings, classifier, limit=3)  # nothing left to do
    assert classifier.calls == 1
    assert _target().manifest_path.read_text(encoding="utf-8") == manifest_before


async def test_resume_redoes_api_errors(results_dir: Path, settings: Settings, sleeps: list[float]) -> None:
    down = FakeClassifier(*[RetryableError("down") for _ in range(runner.MAX_ATTEMPTS)])
    await _run(settings, down, limit=1)
    assert _records(_target().path)[0]["outcome"] == "api_error"
    path = await _run(settings, FakeClassifier(), limit=1)
    assert [r["outcome"] for r in _records(path)] == ["api_error", "ok"]  # the report keeps the last record


async def test_resume_repairs_a_half_written_line(results_dir: Path, settings: Settings) -> None:
    path = await _run(settings, FakeClassifier(), limit=2)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"example_id": "torn')  # killed mid-write
    await _run(settings, FakeClassifier(), limit=3)
    assert len(_records(path)) == 3  # every line parses: the torn fragment is gone


def test_repair_torn_tail(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    assert runner.repair_torn_tail(path) is False
    path.write_text('{"a": 1}\n', encoding="utf-8")
    assert runner.repair_torn_tail(path) is False
    path.write_text('{"a": 1}\n{"b"', encoding="utf-8")
    assert runner.repair_torn_tail(path) is True
    assert path.read_text(encoding="utf-8") == '{"a": 1}\n'
    path.write_text('{"only half', encoding="utf-8")
    assert runner.repair_torn_tail(path) is True
    assert path.read_text(encoding="utf-8") == ""


async def test_resume_refuses_a_changed_configuration(results_dir: Path, settings: Settings) -> None:
    await _run(settings, FakeClassifier(), limit=1)
    changed = FakeClassifier()
    changed.model_requested = "jev-other"
    with pytest.raises(SystemExit, match="model_requested changed"):
        await _run(settings, changed, limit=2)


async def test_records_without_a_manifest_are_refused(results_dir: Path, settings: Settings) -> None:
    await _run(settings, FakeClassifier(), limit=1)
    _target().manifest_path.unlink()
    with pytest.raises(SystemExit, match="no manifest"):
        await _run(settings, FakeClassifier(), limit=2)


def test_an_unreadable_manifest_stops_the_run(tmp_path: Path) -> None:
    path = tmp_path / "repeat-1.manifest.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="not valid JSON"):
        runner.read_manifest(path)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit, match="not a manifest"):
        runner.read_manifest(path)


def test_completed_ids_ignore_api_errors_and_junk(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    lines = [
        {"example_id": "a", "outcome": "ok"},
        {"example_id": "b", "outcome": "api_error"},
        {"example_id": "c", "outcome": "api_error"},
        {"example_id": "c", "outcome": "refused"},
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n[1]\nnot json\n", encoding="utf-8")
    assert runner.completed_ids(path) == {"a", "c"}


def test_latest_run_is_chosen_by_time_not_name(
    tmp_path: Path, make_record: RecordFactory, write_run: RunWriter
) -> None:
    results = tmp_path / "results"
    runs = (("zzz-older", "2026-01-01T00:00:00.000+00:00"), ("aaa-newer", "2026-09-29T00:00:00.000+00:00"))
    for run_id, written_at in runs:
        manifest: dict[str, object] = {"system": "jev", "split": "main", "repeat": 1, "written_at": written_at}
        write_run([make_record()], run_id=run_id, manifests={("jev", "main", 1): manifest}, root=results)
    (results / "no-manifests").mkdir()
    assert runner.latest_run_dir(results) == results / "aaa-newer"
    assert runner.latest_run_dir(tmp_path / "missing") is None


def test_git_state_reports_commit_and_dirty_flag() -> None:
    state = runner.git_state()
    assert set(state) == {"commit", "dirty"}
    assert state["dirty"] in (True, False, None)
