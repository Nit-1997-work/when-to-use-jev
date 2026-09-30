from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from jevbench.data import (
    SPLITS,
    WARMUP_SIZE,
    DataError,
    SplitSpec,
    build_splits,
    load_raw,
    load_split,
    write_splits,
)
from jevbench.intents import IntentCatalog, load_catalog

INTENTS = 46


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    try:
        return load_raw()
    except DataError:
        pytest.skip("raw CSV not downloaded (see data/SOURCES.md)")


@pytest.fixture(scope="module")
def splits(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return build_splits(raw, load_catalog())


# ---------- the committed split files (no download needed) ----------


@pytest.mark.parametrize("spec", SPLITS, ids=lambda spec: spec.name)
def test_committed_split_sizes_and_composition(spec: SplitSpec) -> None:
    examples = load_split(spec.name)
    per_intent = spec.clean_per_intent + spec.profane_per_intent
    assert len(examples) == INTENTS * per_intent
    assert Counter(e.intent for e in examples) == dict.fromkeys(load_catalog().names, per_intent)
    profane = Counter(e.intent for e in examples if e.has_profanity)
    assert all(profane[name] == spec.profane_per_intent for name in load_catalog().names)
    assert {e.split for e in examples} == {spec.name}


def test_committed_splits_never_share_a_message() -> None:
    names = [spec.name for spec in SPLITS] + ["warmup"]
    ids = [e.id for name in names for e in load_split(name)]
    assert len(ids) == len(set(ids))
    texts = [e.text for name in names for e in load_split(name)]
    assert len(texts) == len(set(texts))


def test_committed_warmup_is_clean() -> None:
    warmup = load_split("warmup")
    assert len(warmup) == WARMUP_SIZE
    assert not any(e.has_profanity for e in warmup)


def test_committed_categories_match_the_catalog() -> None:
    catalog = load_catalog()
    for spec in SPLITS:
        assert all(catalog.category_of(e.intent) == e.category for e in load_split(spec.name))


def test_invalid_split_lines_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "bad.jsonl").write_text('{"id": "x"}\n', encoding="utf-8")
    with pytest.raises(DataError, match=r"bad\.jsonl:1"):
        load_split("bad", tmp_path)
    with pytest.raises(DataError, match="not found"):
        load_split("missing", tmp_path)


def test_a_different_csv_is_rejected_by_its_checksum(tmp_path: Path) -> None:
    csv = tmp_path / "other.csv"
    csv.write_text("instruction,intent,category,tags,response\nhi,pay,PAYMENT,B,ok\n", encoding="utf-8")
    with pytest.raises(DataError, match="SHA-256"):
        load_raw(csv)
    assert load_raw(csv, expected_sha256=None)["id"].tolist() == ["bitext-00000"]
    with pytest.raises(DataError, match="not found"):
        load_raw(tmp_path / "missing.csv")


def _dataset(intents: dict[str, tuple[str, int, int]]) -> pd.DataFrame:
    """{intent: (category, clean rows, profane rows)}, shaped like `load_raw` output."""
    rows = [
        {
            "id": f"{intent}-{i}",
            "text": f"{intent} message {i}",
            "intent": intent,
            "category": category,
            "tags": "W" if i >= clean else "B",
            "has_profanity": i >= clean,
            "has_typos": False,
        }
        for intent, (category, clean, profane) in intents.items()
        for i in range(clean + profane)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def two_intents(tmp_path: Path) -> IntentCatalog:
    path = tmp_path / "intents.yaml"
    path.write_text(
        "instruction: x\nintents:\n  - {name: a, category: X, means: m}\n  - {name: b, category: Y, means: n}\n"
    )
    return load_catalog(path)


def test_split_builder_allocates_disjoint_sets(two_intents: IntentCatalog, tmp_path: Path) -> None:
    splits = build_splits(_dataset({"a": ("X", 60, 16), "b": ("Y", 60, 16)}), two_intents)
    assert {name: len(frame) for name, frame in splits.items()} == {
        "practice": 10,
        "main": 100,
        "profanity": 20,
        "warmup": WARMUP_SIZE,
    }
    assert not pd.concat(splits.values())["id"].duplicated().any()
    assert splits["profanity"]["has_profanity"].all()
    write_splits(splits, tmp_path / "splits")
    assert [e.id for e in load_split("main", tmp_path / "splits")] == splits["main"]["id"].tolist()


@pytest.mark.parametrize(
    ("intents", "message"),
    [
        ({"a": ("X", 60, 16), "c": ("Y", 60, 16)}, "does not match the dataset"),
        ({"a": ("X", 60, 16), "b": ("Z", 60, 16)}, "catalog says Y"),
        ({"a": ("X", 30, 16), "b": ("Y", 60, 16)}, "not enough clean rows for split main"),
        ({"a": ("X", 60, 12), "b": ("Y", 60, 16)}, "not enough profane rows for split profanity"),
    ],
)
def test_split_builder_guards(
    two_intents: IntentCatalog, intents: dict[str, tuple[str, int, int]], message: str
) -> None:
    with pytest.raises(DataError, match=message):
        build_splits(_dataset(intents), two_intents)


# ---------- rebuilding the splits from the pinned CSV ----------


def test_catalog_matches_dataset(raw: pd.DataFrame) -> None:
    catalog = load_catalog()
    assert set(catalog.names) == set(raw["intent"])
    assert len(catalog.names) == INTENTS


def test_split_sizes_and_composition(splits: dict[str, pd.DataFrame]) -> None:
    for spec in SPLITS:
        frame = splits[spec.name]
        per_intent = spec.clean_per_intent + spec.profane_per_intent
        assert len(frame) == INTENTS * per_intent
        counts = frame.groupby("intent")["has_profanity"].agg(["size", "sum"])
        assert (counts["size"] == per_intent).all()
        assert (counts["sum"] == spec.profane_per_intent).all()


def test_rebuilt_splits_match_the_committed_files(splits: dict[str, pd.DataFrame]) -> None:
    for name, frame in splits.items():
        assert frame["id"].tolist() == [e.id for e in load_split(name)]


def test_no_row_in_two_splits(splits: dict[str, pd.DataFrame]) -> None:
    ids = pd.concat(splits.values())["id"]
    assert not ids.duplicated().any()


def test_warmup_is_clean_and_unused(splits: dict[str, pd.DataFrame]) -> None:
    warmup = splits["warmup"]
    assert len(warmup) == WARMUP_SIZE
    assert not warmup["has_profanity"].any()


def test_splits_are_deterministic(raw: pd.DataFrame) -> None:
    catalog = load_catalog()
    first = build_splits(raw, catalog)
    second = build_splits(raw, catalog)
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])


def test_run_order_is_shuffled(splits: dict[str, pd.DataFrame]) -> None:
    # Guard against running all examples of one intent back to back (time-drift confound).
    first_50 = splits["main"]["intent"].head(50)
    assert first_50.nunique() > 20
