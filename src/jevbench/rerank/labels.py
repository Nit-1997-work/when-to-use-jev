"""The shared label definitions (data/rerank/labels.yaml): one source of label text for every re-ranking system."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from jevbench.common.paths import data_dir

DEFAULT_LABELS_PATH = data_dir("rerank") / "labels.yaml"

# ESCI's four levels, best first, with the KDD Cup 2022 Task 1 gains (Reddy et al., 2022, section 3.1).
LEVELS = ("exact", "substitute", "complement", "irrelevant")
ESCI_CODE = {"exact": "E", "substitute": "S", "complement": "C", "irrelevant": "I"}
CODE_LEVEL = {code: level for level, code in ESCI_CODE.items()}
GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
# A gentler secondary scale that values substitutes more, to show whether the winner depends on the official gains.
GENTLE_GAIN = {"E": 1.0, "S": 0.5, "C": 0.1, "I": 0.0}


class LabelError(ValueError):
    pass


@dataclass(frozen=True)
class Level:
    name: str
    means: str
    not_: str
    example: str

    def criterion(self) -> dict[str, str]:
        """The description object sent to every model for this level."""
        return {"means": self.means, "not": self.not_, "example": self.example}


@dataclass(frozen=True)
class LabelSet:
    instruction: str
    levels: tuple[Level, ...]  # best first: exact, substitute, complement, irrelevant
    exact_question: str
    exact_true: str
    exact_false: str
    none_of_these: str
    sha256: str

    def level(self, name: str) -> Level:
        return next(level for level in self.levels if level.name == name)

    def criteria_json(self) -> str:
        """The label definitions as the LLM sees them, in the same words Jev gets."""
        return json.dumps({level.name: level.criterion() for level in self.levels}, indent=2, ensure_ascii=False)


def _text(raw: dict[str, object], key: str, path: Path) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LabelError(f"{path}: `{key}` must be a non-empty string.")
    return " ".join(value.split())


def load_labels(path: Path = DEFAULT_LABELS_PATH) -> LabelSet:
    if not path.exists():
        raise LabelError(f"{path} not found.")
    content = path.read_bytes()
    raw = yaml.safe_load(content)
    if not isinstance(raw, dict):
        raise LabelError(f"{path}: expected a mapping at the top level.")
    labels = raw.get("labels")
    if not isinstance(labels, dict) or list(labels) != list(LEVELS):
        raise LabelError(f"{path}: `labels` must define exactly {', '.join(LEVELS)}, in that order.")
    levels = []
    for name in LEVELS:
        entry = labels[name]
        if not isinstance(entry, dict):
            raise LabelError(f"{path}: label `{name}` must be a mapping.")
        levels.append(
            Level(
                name=name,
                means=_text(entry, "means", path),
                not_=_text(entry, "not", path),
                example=_text(entry, "example", path),
            )
        )
    exact = raw.get("exact_match")
    if not isinstance(exact, dict):
        raise LabelError(f"{path}: `exact_match` must be a mapping.")
    return LabelSet(
        instruction=_text(raw, "instruction", path),
        levels=tuple(levels),
        exact_question=_text(exact, "question", path),
        exact_true=_text(exact, "match", path),
        exact_false=_text(exact, "no_match", path),
        none_of_these=_text(raw, "none_of_these", path),
        sha256=hashlib.sha256(content).hexdigest(),
    )
