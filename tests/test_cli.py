from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import jevbench.classifiers as classifiers
import jevbench.cli as cli
import jevbench.phoenix_sync as phoenix_sync
import jevbench.runner as runner
import jevbench.settings as settings_module
from jevbench.classifiers import Outcome, Prediction
from jevbench.settings import Settings

if TYPE_CHECKING:
    from tests.conftest import RecordFactory, RunWriter


class FakeClassifier:
    name = "jev"
    model_requested = "jev-test"

    async def classify(self, text: str) -> Prediction:
        return Prediction(outcome=Outcome.OK, intent="track_order", latency_ms=12.0, confidence=0.9)

    async def aclose(self) -> None:
        return None


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """No real .env, no trust-store injection, and results under a temp directory."""
    monkeypatch.setattr(cli, "load_env", lambda: None)
    monkeypatch.setattr(cli, "_use_system_trust_store", lambda: None)
    results = tmp_path / "results"
    monkeypatch.setattr(runner, "RESULTS_DIR", results)
    return results


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    monkeypatch.setattr(settings_module, "load_settings", lambda *_, **__: settings)
    monkeypatch.setattr(classifiers, "build_classifier", lambda *_: FakeClassifier())


@pytest.mark.parametrize("value", ["run-1", "20260929T205311Z", "a.b_c"])
def test_valid_run_ids(value: str) -> None:
    assert cli.run_id_arg(value) == value


@pytest.mark.parametrize("value", ["", "../x", "a/b", "-x", "a" * 65, "a..b"])
def test_run_ids_must_be_one_safe_path_segment(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        cli.run_id_arg(value)


def test_full_runs_from_uncommitted_code_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_tree_is_clean", lambda: False)
    with pytest.raises(SystemExit, match="uncommitted changes"):
        cli.main(["run", "--system", "jev"])


@pytest.mark.parametrize(("dirty", "clean"), [(False, True), (True, False), (None, False)])
def test_an_unknown_git_state_is_not_clean(monkeypatch: pytest.MonkeyPatch, dirty: bool | None, clean: bool) -> None:
    monkeypatch.setattr(runner, "git_state", lambda: {"commit": None, "dirty": dirty})
    assert cli._tree_is_clean() is clean  # pyright: ignore[reportPrivateUsage]


@pytest.mark.usefixtures("fakes")
def test_a_smoke_run_writes_results(offline: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["run", "--system", "jev", "--split", "practice", "--limit", "2", "--warmup", "0", "--run-id", "smoke"])
    assert len((offline / "smoke" / "jev" / "practice" / "repeat-1.jsonl").read_text().splitlines()) == 2
    assert "Next: jevbench report --run-id smoke" in capsys.readouterr().out


def test_report_writes_the_latest_run(
    offline: Path, make_record: RecordFactory, write_run: RunWriter, capsys: pytest.CaptureFixture[str]
) -> None:
    write_run([make_record()], root=offline)
    cli.main(["report"])
    assert (offline / "run" / "report.md").read_text().startswith("# Results: run `run`")
    assert "Written to" in capsys.readouterr().out


def test_report_refuses_incomparable_systems(offline: Path, make_record: RecordFactory, write_run: RunWriter) -> None:
    write_run([make_record(system="jev"), make_record(system="baseline", example_id="other")], root=offline)
    with pytest.raises(SystemExit, match="ran different examples"):
        cli.main(["report", "--run-id", "run"])
    cli.main(["report", "--run-id", "run", "--allow-partial"])
    assert "Warning: partial run" in (offline / "run" / "report.md").read_text()


def test_report_needs_an_existing_run(offline: Path) -> None:
    with pytest.raises(SystemExit, match="No runs"):
        cli.main(["report"])
    with pytest.raises(SystemExit, match="No such run"):
        cli.main(["report", "--run-id", "missing"])


def test_audit_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("OPENAI_API_BASE", "https://llm-gateway.internal.example/v1")
    clean, leaky = tmp_path / "clean.md", tmp_path / "leaky.md"
    clean.write_text("nothing to see\n", encoding="utf-8")
    leaky.write_text("see llm-gateway.internal.example\n", encoding="utf-8")
    with pytest.raises(SystemExit) as passed:
        cli.main(["audit", str(clean)])
    assert passed.value.code == 0
    with pytest.raises(SystemExit) as failed:
        cli.main(["audit", str(leaky)])
    assert failed.value.code == 1
    assert "audit: 1 files scanned, 1 finding(s)" in capsys.readouterr().out


def test_systems_lists_every_system(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["systems"])
    out = capsys.readouterr().out
    assert all(name in out for name in ("jev", "baseline", "baseline-conf", "baseline-2"))


@pytest.mark.usefixtures("fakes")
def test_check_reports_each_system(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as done:
        cli.main(["check", "--systems", "jev"])
    assert done.value.code == 0
    assert "[OK ] jev" in capsys.readouterr().out


def test_phoenix_commands_need_no_model_credentials(
    offline: Path,
    make_record: RecordFactory,
    write_run: RunWriter,
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        settings_module,
        "load_settings",
        lambda *_, **kwargs: calls.append(kwargs) or replace(settings, phoenix_base_url="http://phoenix"),
    )
    monkeypatch.setattr(phoenix_sync, "register_jev_price", lambda url, model, price: f"{url} {model} {price}")
    monkeypatch.setattr(phoenix_sync, "log_experiment", lambda target, url: f"{url}/{target.system}")
    write_run([make_record(system="jev"), make_record(system="baseline")], root=offline)

    cli.main(["phoenix-setup"])
    cli.main(["phoenix-sync", "--systems", "jev"])
    out = capsys.readouterr().out
    assert "http://phoenix jev-test 0.042" in out
    assert "jev/main/r1: http://phoenix/jev" in out
    assert "baseline" not in out
    assert calls == [{"require_credentials": False}, {"require_credentials": False}]
