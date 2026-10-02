"""Pre-publish scan: nothing that could be committed may name the private LLM gateway or contain a secret.

The deny-list is built at scan time, so it never has to be committed itself:
- the host of `OPENAI_API_BASE` (the LLM gateway) and the values of the API keys, from the environment or `.env`;
- well-known secret shapes (private keys, OpenAI-style and Google API keys);
- extra regular expressions, one per line, from a gitignored `.audit-denylist` file (internal names or hosts).
Findings name the file, line, and rule, never the matched text, so secrets are not echoed to the terminal.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

DENYLIST_FILE = ".audit-denylist"
_SECRET_SETTINGS = ("OPENAI_API_KEY", "TYPESAFE_API_KEY")
_SECRET_SHAPES = {
    "a private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "an OpenAI-style API key": r"\bsk-[A-Za-z0-9_-]{20,}",
    "a Google API key": r"\bAIza[0-9A-Za-z_-]{35}\b",
}


@dataclass(frozen=True)
class Rule:
    name: str  # shown in findings; never the secret itself
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    rule: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: contains {self.rule}"


def _setting(name: str) -> str | None:
    value = os.environ.get(name, "").strip().strip("'\"")
    return value if value and value != "changeme" else None


def deny_rules(repo_root: Path) -> list[Rule]:
    rules = [Rule(name, re.compile(pattern)) for name, pattern in _SECRET_SHAPES.items()]
    gateway = _setting("OPENAI_API_BASE")
    host = urlparse(gateway).hostname if gateway else None
    if host:
        rules.append(Rule("the LLM gateway host (OPENAI_API_BASE)", re.compile(re.escape(host), re.IGNORECASE)))
    for name in _SECRET_SETTINGS:
        value = _setting(name)
        if value is not None and len(value) >= 8:  # shorter values would match ordinary text
            rules.append(Rule(f"the value of {name}", re.compile(re.escape(value))))
    denylist = repo_root / DENYLIST_FILE
    if denylist.exists():
        for number, raw in enumerate(denylist.read_text(encoding="utf-8").splitlines(), start=1):
            entry = raw.strip()
            if not entry or entry.startswith("#"):
                continue
            try:
                rules.append(Rule(f"{DENYLIST_FILE} pattern on line {number}", re.compile(entry, re.IGNORECASE)))
            except re.error as exc:
                raise SystemExit(f"{DENYLIST_FILE} line {number}: invalid pattern ({exc}).") from None
    return rules


def committable_files(repo_root: Path) -> list[Path]:
    """Tracked files plus untracked files that are not ignored: everything a commit of this tree could include."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"Cannot list the files git would commit ({exc}). Pass paths to scan explicitly.") from None
    names = {name for name in out.stdout.decode("utf-8").split("\0") if name}
    return sorted(path for path in (repo_root / name for name in names) if path.is_file())


def expand(paths: list[Path]) -> list[Path]:
    """Files under the given paths (directories are walked)."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(p for p in sorted(path.rglob("*")) if p.is_file())
        elif path.is_file():
            files.append(path)
        else:
            raise SystemExit(f"No such file or directory: {path}")
    return files


def scan(files: list[Path], rules: list[Rule]) -> list[Finding]:
    findings: list[Finding] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable files cannot carry the text we look for
        for number, line in enumerate(text.splitlines(), start=1):
            findings.extend(Finding(path, number, rule.name) for rule in rules if rule.pattern.search(line))
    return findings
