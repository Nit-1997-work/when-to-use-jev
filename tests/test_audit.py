from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from jevbench.audit import DENYLIST_FILE, committable_files, deny_rules, expand, scan

GATEWAY_RULE = "the LLM gateway host (OPENAI_API_BASE)"
# Built at run time, so this file never contains a secret-shaped literal: the repository's own audit scans it.
FAKE_KEY = "-".join(("sk", "gateway", "secret", "000000"))
PRIVATE_KEY_HEADER = " ".join(("-----BEGIN", "RSA", "PRIVATE", "KEY-----"))


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_BASE", "OPENAI_API_KEY", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_the_gateway_host_and_key_values_are_denied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_BASE", "https://llm-gateway.internal.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("TYPESAFE_API_KEY", "changeme")  # the .env.example placeholder is not a secret
    rules = deny_rules(tmp_path)
    names = [rule.name for rule in rules]
    assert GATEWAY_RULE in names
    assert "the value of OPENAI_API_KEY" in names
    assert "the value of TYPESAFE_API_KEY" not in names

    report = tmp_path / "report.md"
    report.write_text(f"fine\ncalled https://LLM-GATEWAY.internal.example/v1\nkey {FAKE_KEY}\n")
    findings = scan([report], rules)
    assert [(f.line, f.rule) for f in findings] == [
        (2, GATEWAY_RULE),
        (3, "an OpenAI-style API key"),
        (3, "the value of OPENAI_API_KEY"),
    ]
    assert not any(FAKE_KEY in str(f) for f in findings)  # findings never echo the secret


def test_short_key_values_are_not_used_as_patterns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "abc")
    assert "the value of OPENAI_API_KEY" not in [rule.name for rule in deny_rules(tmp_path)]


def test_denylist_file_patterns(tmp_path: Path) -> None:
    (tmp_path / DENYLIST_FILE).write_text("# internal names\n\ninternal-team\nacme\\.corp\n", encoding="utf-8")
    rules = deny_rules(tmp_path)
    notes = tmp_path / "notes.md"
    notes.write_text("Internal-Team wrote this\nacme-corp is fine\nACME.CORP is not\n", encoding="utf-8")
    assert [(f.line, f.rule) for f in scan([notes], rules)] == [
        (1, f"{DENYLIST_FILE} pattern on line 3"),
        (3, f"{DENYLIST_FILE} pattern on line 4"),
    ]


def test_an_invalid_denylist_pattern_stops_the_audit(tmp_path: Path) -> None:
    (tmp_path / DENYLIST_FILE).write_text("(unclosed\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="line 1"):
        deny_rules(tmp_path)


def test_binary_files_are_skipped_and_directories_walked(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "key.pem").write_text(f"{PRIVATE_KEY_HEADER}\n", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00" + PRIVATE_KEY_HEADER.encode())
    files = expand([tmp_path])
    assert sorted(p.name for p in files) == ["blob.bin", "key.pem"]
    assert [f.path.name for f in scan(files, deny_rules(tmp_path))] == ["key.pem"]
    with pytest.raises(SystemExit, match="No such file"):
        expand([tmp_path / "missing"])


def test_committable_files_follow_gitignore(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("hi\n", encoding="utf-8")
    (tmp_path / "debug.log").write_text("secret\n", encoding="utf-8")
    assert [p.name for p in committable_files(tmp_path)] == [".gitignore", "notes.md"]


def test_committable_files_needs_a_repository(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="Cannot list"):
        committable_files(tmp_path)
