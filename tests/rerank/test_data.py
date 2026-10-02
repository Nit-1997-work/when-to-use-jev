from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from jevbench.rerank import data
from jevbench.rerank.labels import LEVELS, LabelError, load_labels
from tests.rerank.helpers import make_search


def _frame(queries: dict[int, tuple[str, str, str]]) -> pd.DataFrame:
    """ESCI-shaped rows: {query_id: (query, labels as a string like "EESI", ESCI split)}."""
    rows = []
    for query_id, (query, labels, split) in queries.items():
        for i, label in enumerate(labels):
            rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "product_id": f"B{query_id:04d}{i:02d}",
                    "esci_label": label,
                    "split": split,
                    "product_title": f"  Title {query_id}-{i}   with  spaces ",
                    "product_brand": "Brand" if i % 2 else None,
                    "product_color": None,
                    "source": "negations" if query_id % 2 else "other",
                }
            )
    return pd.DataFrame(rows)


def test_make_search_is_deterministic_shuffled_and_relabels_ids() -> None:
    rows = _frame({7: ("milk", "EESSIICC", "test")})
    first = data.make_search(rows, split="main", grocery=True)
    again = data.make_search(rows.sample(frac=1.0, random_state=1), split="main", grocery=True)
    assert first == again  # input row order does not matter
    assert [c.cid for c in first.candidates] == [f"c{i}" for i in range(1, 9)]
    assert sorted(first.labels) == sorted("EESSIICC")
    assert first.candidates[0].title.startswith("Title 7-")  # whitespace collapsed
    assert (first.exact_count, first.hard, first.should_abstain, first.grocery) == (2, True, False, True)

    nomatch = data.make_search(rows, split="nomatch", grocery=False, remove_exact=True)
    assert "E" not in nomatch.labels and len(nomatch.candidates) == 6
    assert (nomatch.should_abstain, nomatch.hard, nomatch.exact_count) == (True, False, 2)

    backwards = data.make_search(rows, split="reversed", grocery=False, reverse=True)
    assert [c.product_id for c in backwards.candidates] == [c.product_id for c in reversed(first.candidates)]
    assert backwards.candidates[0].cid == "c1"  # ids follow presented order, so they give no hint


def test_eligibility_keeps_8_to_40_candidates() -> None:
    frame = _frame(
        {1: ("a", "E" * 7, "test"), 2: ("b", "E" * 8, "test"), 3: ("c", "E" * 41, "test"), 4: ("d", "E" * 9, "train")}
    )
    assert data.eligible_query_ids(frame, "test") == [2]
    assert data.eligible_query_ids(frame, "train") == [4]


def test_build_splits_fills_every_split_without_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in (("MAIN_SIZE", 6), ("MAIN_GROCERY", 2), ("NOMATCH", 3), ("CONSISTENCY", 2)):
        monkeypatch.setattr(data, name, value)
    monkeypatch.setattr(data, "TRAIN_SPLITS", (("practice", 3), ("practice-nomatch", 2), ("warmup", 1)))
    queries = {q: (f"query {q}", "EESSSIIC", "train") for q in range(1, 9)}
    queries.update({q: (f"query {q}", "EESSSIIC", "test") for q in range(100, 120)})
    reviewed, approved = {100, 101, 102}, {100, 101}
    splits = data.build_splits(_frame(queries), reviewed=reviewed, approved=approved)

    sizes = {name: len(searches) for name, searches in splits.items()}
    assert sizes == {
        "practice": 3,
        "practice-nomatch": 2,
        "main": 6,
        "nomatch": 3,
        "consistency": 2,
        "reversed": 2,
        "warmup": 1,
    }
    assert sum(s.grocery for s in splits["main"]) == 2
    assert 102 not in {s.query_id for s in splits["main"]}  # reviewed but not grocery: kept out of the random part
    assert all(s.should_abstain for s in splits["nomatch"] + splits["practice-nomatch"])
    assert [s.query_id for s in splits["reversed"]] == [s.query_id for s in splits["consistency"]]
    again = data.build_splits(_frame(queries), reviewed=reviewed, approved=approved)
    assert again == splits


def test_splits_round_trip_through_jsonl(tmp_path: Path) -> None:
    search = make_search("ESI")
    paths = data.write_splits({"practice": [search]}, tmp_path)
    assert data.load_split("practice", tmp_path) == [search]
    assert "candidates" in json.loads(paths["practice"].read_text(encoding="utf-8").splitlines()[0])
    with pytest.raises(data.DataError, match="not found"):
        data.load_split("missing", tmp_path)
    (tmp_path / "broken.jsonl").write_text("{not json\n", encoding="utf-8")
    with pytest.raises(data.DataError, match="invalid search"):
        data.load_split("broken", tmp_path)


def test_grocery_candidates_and_review(tmp_path: Path) -> None:
    frame = _frame(
        {200 + q: (query, "EESSSIIC", "test") for q, query in enumerate(["olive oil", "oil filter", "desk lamp"])}
    )
    terms = data.load_grocery_terms()
    pool = data.grocery_candidates(frame, terms)
    assert [p["query"] for p in pool] == ["olive oil", "oil filter"]  # keyword hits only; a person decides
    path = data.write_grocery_review(pool, tmp_path / "review.yaml")
    with pytest.raises(data.DataError, match="no grocery decision"):
        data.load_grocery_review(path)
    text = path.read_text(encoding="utf-8").replace("grocery: null", "grocery: false", 2)
    path.write_text(text.replace("grocery: false", "grocery: true", 1), encoding="utf-8")
    assert data.load_grocery_review(path) == ({200, 201}, {200})
    path.write_text(text.replace("grocery: false", "grocery: maybe", 1), encoding="utf-8")
    with pytest.raises(data.DataError, match="use true, false, or null"):
        data.load_grocery_review(path)
    with pytest.raises(data.DataError, match="not found"):
        data.load_grocery_review(tmp_path / "missing.yaml")


def test_raw_files_are_checked(tmp_path: Path) -> None:
    with pytest.raises(data.DataError, match="not found"):
        data.verify_raw(tmp_path)
    for name, _ in data.RAW_FILES.values():
        (tmp_path / name).write_text("not the pinned file", encoding="utf-8")
    with pytest.raises(data.DataError, match="SHA-256"):
        data.verify_raw(tmp_path)


def test_committed_splits_match_their_profile() -> None:
    """The committed splits are the ones the design describes (rebuilding them needs the 1.1 GB ESCI download)."""
    splits = {name: data.load_split(name) for name in data.SPLIT_NAMES}
    assert {name: len(s) for name, s in splits.items()} == {
        "practice": 100,
        "practice-nomatch": 50,
        "main": 1000,
        "nomatch": 300,
        "consistency": 100,
        "reversed": 100,
        "warmup": 20,
    }
    assert all(data.MIN_CANDIDATES <= len(s.candidates) <= data.MAX_CANDIDATES for s in splits["main"])
    assert all(s.should_abstain for s in splits["nomatch"]) and not any(s.should_abstain for s in splits["main"])
    used = [s.query_id for name, searches in splits.items() if name != "reversed" for s in searches]
    assert len(used) == len(set(used))


def test_rebuilding_from_the_pinned_files_is_byte_identical() -> None:
    if not all((data.RAW_DIR / name).exists() for name, _ in data.RAW_FILES.values()):
        pytest.skip("ESCI files not downloaded (see data/SOURCES.md)")
    reviewed, approved = data.load_grocery_review()
    splits = data.build_splits(data.load_esci(), reviewed=reviewed, approved=approved)
    for name, searches in splits.items():
        text = "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in (data.asdict(x) for x in searches))
        committed = (data.SPLITS_DIR / f"{name}.jsonl").read_bytes()
        assert hashlib.sha256(text.encode()).hexdigest() == hashlib.sha256(committed).hexdigest(), name


def test_labels_file_defines_every_level(tmp_path: Path) -> None:
    labels = load_labels()
    assert [level.name for level in labels.levels] == list(LEVELS)
    assert json.loads(labels.criteria_json())["exact"]["means"] == labels.level("exact").means
    assert labels.exact_true and labels.exact_false and labels.none_of_these
    bad = tmp_path / "labels.yaml"
    bad.write_text("labels: {exact: {}}\n", encoding="utf-8")
    with pytest.raises(LabelError, match="exactly exact"):
        load_labels(bad)
    with pytest.raises(LabelError, match="not found"):
        load_labels(tmp_path / "missing.yaml")
