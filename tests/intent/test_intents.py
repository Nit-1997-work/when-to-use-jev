from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from jevbench.intent.intents import DEFAULT_INTENTS_PATH, CatalogError, load_catalog

VALID = """
instruction: "Which intent?"
intents:
  - name: a
    category: X
    means: Means a.
    not: Means b (b).
  - name: b
    category: Y
    means: Means b.
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "intents.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_frozen_catalog() -> None:
    catalog = load_catalog()
    assert len(catalog.intents) == 46
    assert len({i.category for i in catalog.intents}) == 13
    assert catalog.sha256 == hashlib.sha256(DEFAULT_INTENTS_PATH.read_bytes()).hexdigest()


def test_criteria_include_not_only_when_given(tmp_path: Path) -> None:
    catalog = load_catalog(_write(tmp_path, VALID))
    assert catalog.instruction == "Which intent?"
    assert catalog.names == ["a", "b"]
    assert catalog.criteria() == {"a": {"means": "Means a.", "not": "Means b (b)."}, "b": {"means": "Means b."}}
    assert catalog.category_of("b") == "Y"
    assert catalog.category_of("zzz") is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[1, 2]", "must define"),
        ("instruction: x\n", "must define"),
        ("instruction: x\nintents: {}\n", "non-empty list"),
        ("instruction: x\nintents: []\n", "non-empty list"),
        ("instruction: x\nintents:\n  - just a string\n", "must be a mapping"),
        ("instruction: x\nintents:\n  - {name: a, category: X}\n", "non-empty text for means"),
        ("instruction: x\nintents:\n  - {name: a, category: X, means: m, not: [1]}\n", "`not` must be text"),
        (
            "instruction: x\nintents:\n  - {name: a, category: X, means: m}\n  - {name: a, category: Y, means: n}\n",
            "Duplicate",
        ),
        ("instruction: [unclosed\n", "Invalid YAML"),
    ],
)
def test_invalid_catalogs_are_rejected(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(CatalogError, match=message):
        load_catalog(_write(tmp_path, text))


def test_choice_questions_are_limited_to_255_options(tmp_path: Path) -> None:
    items = "".join(f"  - {{name: i{n}, category: X, means: m}}\n" for n in range(256))
    with pytest.raises(CatalogError, match="255"):
        load_catalog(_write(tmp_path, "instruction: x\nintents:\n" + items))


def test_an_unreadable_catalog_is_reported(tmp_path: Path) -> None:
    with pytest.raises(CatalogError, match="Cannot read"):
        load_catalog(tmp_path / "missing.yaml")
