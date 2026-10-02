"""Repository locations shared by every experiment."""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = REPO_ROOT / ".env"


def data_dir(experiment: str) -> Path:
    """`data/<experiment>/`: label files, prepared splits, and the gitignored raw downloads."""
    return REPO_ROOT / "data" / experiment


def results_dir(experiment: str) -> Path:
    """`results/<experiment>/`: one directory per run."""
    return REPO_ROOT / "results" / experiment


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
