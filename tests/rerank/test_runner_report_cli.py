from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import jevbench.cli as cli
import jevbench.common.runner as core
import jevbench.rerank.data as rerank_data
import jevbench.rerank.runner as runner
import jevbench.rerank.thresholds as thresholds_module
from jevbench.common import settings as settings_module
from jevbench.common.calls import Outcome, RetryableError
from jevbench.common.report_format import ReportError
from jevbench.common.settings import Settings
from jevbench.rerank import systems as rerank_systems
from jevbench.rerank.data import Search
from jevbench.rerank.labels import load_labels
from jevbench.rerank.ranking import Ranking
from jevbench.rerank.report import build_report
from tests.rerank.helpers import make_search, rerank_record, write_rerank_run


class FakeReranker:
    """Ranks by gold label (a perfect system) or reversed; can fail first; abstains when nothing is Exact."""

    def __init__(self, name: str = "jev-score", *failures: Exception, perfect: bool = True) -> None:
        self.name = name
        self.model_requested = "jev-test"
        self.failures = list(failures)
        self.perfect = perfect
        self.calls = 0

    async def rerank(self, search: Search) -> Ranking:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        rank = {"E": 0, "S": 1, "C": 2, "I": 3}
        order = sorted(search.candidates, key=lambda c: rank[c.label] if self.perfect else -rank[c.label])
        exact = 0.9 if "E" in search.labels else 0.2
        return Ranking(
            outcome=Outcome.OK,
            latency_ms=12.0,
            order=[c.cid for c in order],
            scores={c.cid: float(-rank[c.label]) for c in search.candidates},
            p_exact={c.cid: 0.9 if c.label == "E" else 0.1 for c in search.candidates},
            exact_confidence=exact,
            input_tokens=500,
            output_tokens=0,
            reasoning_tokens=0,
            model_reported="jev-test",
        )

    async def aclose(self) -> None:
        return None


@pytest.fixture
def splits_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Tiny practice, practice-nomatch, and warm-up splits; results under a temp directory."""
    directory = tmp_path / "splits"
    rerank_data.write_splits(
        {
            "practice": [make_search("ESCI", search_id="q1"), make_search("SEI", search_id="q2")],
            "practice-nomatch": [make_search("SCI", search_id="q3")],
            "warmup": [make_search("ESI", search_id="q4")],
        },
        directory,
    )
    original = rerank_data.load_split
    monkeypatch.setattr(runner, "load_split", lambda name, splits_dir=directory: original(name, splits_dir))
    monkeypatch.setattr(runner, "SPLITS_DIR", directory)
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path / "results")
    return directory


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(core.asyncio, "sleep", fake_sleep)


async def _run(settings: Settings, reranker: FakeReranker, split: str = "practice", **kwargs: object) -> Path:
    target = runner.target("run", reranker.name, split, 1)
    return await runner.run_target(target, reranker, settings, load_labels(), limit=None, warmup=1, **kwargs)  # pyright: ignore[reportArgumentType]


@pytest.mark.usefixtures("splits_dir", "no_sleep")
async def test_run_writes_one_complete_record_per_search(settings: Settings) -> None:
    reranker = FakeReranker("jev-score", RetryableError("503"))
    path = await _run(settings, reranker)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["example_id"] for r in records] == ["q1", "q2"]
    first = records[0]
    assert first["attempts"] == 1 and records[1]["attempts"] == 1  # the warm-up search absorbed the retry
    assert first["gold_labels"] == ["E", "S", "C", "I"]
    assert first["order"] == ["c1", "c2", "c3", "c4"]
    assert first["outcome"] == "ok"
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["labels_sha256"] == load_labels().sha256
    assert manifest["prices_per_mtok"] == {"input": 0.042, "output": 0.0, "thinking": 0.0}

    again = FakeReranker("jev-score")
    await _run(settings, again)
    assert again.calls == 0  # resumable: nothing left to do


@pytest.mark.usefixtures("splits_dir", "no_sleep")
async def test_exhausted_retries_are_api_errors(settings: Settings) -> None:
    down = FakeReranker("jev-score", *[RetryableError("down") for _ in range(3 * core.MAX_ATTEMPTS)])  # warm-up + 2
    path = await _run(settings, down)
    assert {json.loads(line)["outcome"] for line in path.read_text(encoding="utf-8").splitlines()} == {"api_error"}


@pytest.mark.usefixtures("splits_dir")
async def test_end_to_end_report_and_threshold_tuning(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, perfect in (("gemini-listwise", False), ("jev-score", True), ("random", False)):
        for split in ("practice", "practice-nomatch"):
            await _run(settings, FakeReranker(name, perfect=perfect), split)
    run_dir = runner.RESULTS_DIR / "run"

    monkeypatch.setattr(thresholds_module, "THRESHOLDS_PATH", tmp_path / "thresholds.yaml")
    monkeypatch.setattr(cli, "load_env", lambda: None)
    monkeypatch.setattr(cli, "use_system_trust_store", lambda: None)
    cli.main(["rerank", "tune-thresholds", "--run-id", "run"])
    tuned = (tmp_path / "thresholds.yaml").read_text(encoding="utf-8")
    assert "jev-score" in tuned and "tuned_on_run: run" in tuned

    report = build_report(run_dir, thresholds_path=tmp_path / "thresholds.yaml")
    assert report.startswith("# Results: run `run`")
    for section in (
        "## Run conditions",
        "### Ranking quality",
        "Head-to-head against gemini-listwise",
        "## Abstention",
    ):
        assert section in report
    assert "| jev-score | 2 | 1.000" in report  # perfect ranking on both practice searches
    assert "## Calibration of Jev's P(exact)" in report


def test_report_refuses_runs_with_different_searches(tmp_path: Path) -> None:
    run_dir = write_rerank_run(
        tmp_path, [rerank_record(system="jev-score"), rerank_record(system="random", example_id="q2")]
    )
    with pytest.raises(ReportError, match="ran different searches"):
        build_report(run_dir, thresholds_path=tmp_path / "none.yaml")
    assert "Warning: partial run" in build_report(run_dir, allow_partial=True, thresholds_path=tmp_path / "none.yaml")


def test_report_covers_main_slices_consistency_labels_and_cascade(tmp_path: Path) -> None:
    records = []
    for system, order in (("gemini-listwise", ["c2", "c1", "c3"]), ("jev-score", ["c1", "c2", "c3"])):
        labels = {"c1": "E", "c2": "S", "c3": "I"}
        scores = {"c1": 0.9, "c2": 0.5, "c3": 0.1}
        for i in range(3):
            extra: dict[str, Any] = (
                {"predicted_labels": labels, "scores": scores} if system == "jev-score" else {"abstain_flag": False}
            )
            records.append(
                rerank_record(system=system, example_id=f"m{i}", order=order, grocery=i == 0, hard=True, **extra)
            )
            records.append(
                rerank_record(
                    system=system,
                    split="nomatch",
                    example_id=f"n{i}",
                    gold=["S", "I", "I"],
                    should_abstain=True,
                    exact_confidence=0.1,
                    **extra,
                )
            )
        for repeat in (1, 2):
            records.append(rerank_record(system=system, split="consistency", repeat=repeat, order=order))
        records.append(
            rerank_record(system=system, split="reversed", order=list(reversed(order)), product_ids=["P3", "P2", "P1"])
        )
    report = build_report(write_rerank_run(tmp_path, records), thresholds_path=tmp_path / "none.yaml")
    for section in (
        "### Ranking quality by slice",
        "| jev-score | grocery | 1 |",
        "## Label quality",
        "Confusion, jev-score",
        "## Consistency",
        "### Cascade",
        "own flag",
    ):
        assert section in report, section


def test_cli_lists_and_guards(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "load_env", lambda: None)
    monkeypatch.setattr(cli, "use_system_trust_store", lambda: None)
    cli.main(["rerank", "systems"])
    assert "jev-score" in capsys.readouterr().out
    monkeypatch.setattr(core, "tree_is_clean", lambda: False)
    with pytest.raises(SystemExit, match="uncommitted changes"):
        cli.main(["rerank", "run", "--system", "random"])


@pytest.mark.usefixtures("splits_dir")
def test_cli_smoke_run_check_and_report(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_env", lambda: None)
    monkeypatch.setattr(cli, "use_system_trust_store", lambda: None)
    monkeypatch.setattr(settings_module, "load_settings", lambda *_, **__: settings)
    monkeypatch.setattr(rerank_systems, "build_reranker", lambda name, *_: FakeReranker(name))
    monkeypatch.setattr(rerank_data, "load_split", runner.load_split)

    cli.main(
        [
            "rerank",
            "run",
            "--system",
            "jev-score",
            "--split",
            "practice",
            "--limit",
            "1",
            "--warmup",
            "0",
            "--run-id",
            "smoke",
        ]
    )
    assert "Next: jevbench rerank report --run-id smoke" in capsys.readouterr().out
    cli.main(["rerank", "report", "--run-id", "smoke"])
    assert (runner.RESULTS_DIR / "smoke" / "report.md").exists()
    with pytest.raises(SystemExit) as done:
        cli.main(["rerank", "check", "--systems", "jev-score", "random"])
    assert done.value.code == 0
    assert "[OK ] jev-score" in capsys.readouterr().out
