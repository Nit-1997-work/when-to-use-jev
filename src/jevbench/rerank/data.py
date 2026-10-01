"""Build the re-ranking splits from the Amazon Shopping Queries Dataset (ESCI).

Deterministic: the same pinned ESCI files, grocery review, and seed always produce byte-identical split files.
No search appears in more than one split, except that `reversed` repeats `consistency` with the order reversed.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from jevbench.common.paths import data_dir, file_sha256

__all__ = ["SPLITS_DIR", "Candidate", "DataError", "Search", "file_sha256", "load_split"]

DATA_DIR = data_dir("rerank")
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
GROCERY_TERMS_PATH = DATA_DIR / "grocery_terms.yaml"
GROCERY_REVIEW_PATH = DATA_DIR / "grocery_review.yaml"

# Pinned in data/SOURCES.md: amazon-science/esci-data at this commit. A different file would silently change splits.
ESCI_COMMIT = "7916cdf6ab75a462e77f20ab40428a10923998d5"
RAW_FILES = {
    "examples": (
        "shopping_queries_dataset_examples.parquet",
        "4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a",
    ),
    "products": (
        "shopping_queries_dataset_products.parquet",
        "25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265",
    ),
    "sources": (
        "shopping_queries_dataset_sources.csv",
        "a5fed8ecc016443de40bf3c63098f0e3f23bbe4daa4236f1c38b8c3184778c50",
    ),
}

SEED = 20261001
MIN_CANDIDATES = 8
MAX_CANDIDATES = 40  # keeps prompts and Jev Choice questions bounded; drops 2.2% of test searches
TITLE_MAX_CHARS = 300  # the 99th percentile title is 200 characters
HARD_MAX_EXACT = 3  # a search with this many Exact candidates or fewer is in the "hard" slice
NOMATCH_MIN_CANDIDATES = 5  # a constructed no-match search keeps at least this many candidates

# (split, ESCI split it is drawn from, number of searches)
TRAIN_SPLITS = (("practice", 100), ("practice-nomatch", 50), ("warmup", 20))
MAIN_SIZE = 1000
MAIN_GROCERY = 300  # at most; random test searches fill the rest of MAIN_SIZE
NOMATCH = 300
CONSISTENCY = 100
SPLIT_NAMES = ("practice", "practice-nomatch", "main", "nomatch", "consistency", "reversed", "warmup")


class DataError(ValueError):
    pass


# ---------- raw ESCI ----------


def verify_raw(raw_dir: Path = RAW_DIR) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for key, (name, expected) in RAW_FILES.items():
        path = raw_dir / name
        if not path.exists():
            raise DataError(f"{path} not found. Download it with the commands in data/SOURCES.md.")
        if (actual := file_sha256(path)) != expected:
            raise DataError(
                f"{path} has SHA-256 {actual}, expected {expected}. "
                "Download the pinned revision with the commands in data/SOURCES.md."
            )
        paths[key] = path
    return paths


def _clean(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    return text or None


def load_esci(raw_dir: Path = RAW_DIR, *, verify: bool = True) -> pd.DataFrame:
    """One row per (search, product) in the reduced English set, with product text and the search's source."""
    paths = verify_raw(raw_dir) if verify else {k: raw_dir / name for k, (name, _) in RAW_FILES.items()}
    examples = pq.read_table(
        paths["examples"],
        columns=["query_id", "query", "product_id", "esci_label", "split"],
        filters=[("small_version", "=", 1), ("product_locale", "=", "us")],
    ).to_pandas()
    products = pq.read_table(
        paths["products"],
        columns=["product_id", "product_title", "product_brand", "product_color"],
        filters=[("product_locale", "=", "us"), ("product_id", "in", sorted(set(examples["product_id"])))],
    ).to_pandas()
    sources = pd.read_csv(paths["sources"], dtype={"query_id": "int64", "source": str})
    frame = examples.merge(products, on="product_id", how="left").merge(sources, on="query_id", how="left")
    if frame["product_title"].isna().any():
        raise DataError("Some ESCI examples have no product title; the pinned files should not.")
    return frame


# ---------- searches ----------


@dataclass(frozen=True)
class Candidate:
    cid: str  # the short id systems see: c1, c2, ... in presented order
    product_id: str  # ESCI's product id (an ASIN)
    title: str
    brand: str | None
    color: str | None
    label: str  # gold ESCI label: E, S, C, or I

    def shown(self) -> dict[str, str]:
        """The product text every system sees: title, brand, and color, never the label or the ASIN."""
        product = {"title": self.title}
        if self.brand:
            product["brand"] = self.brand
        if self.color:
            product["color"] = self.color
        return product


@dataclass(frozen=True)
class Search:
    id: str
    query_id: int
    query: str
    source: str
    grocery: bool
    hard: bool  # few Exact candidates (see HARD_MAX_EXACT); never set on constructed no-match searches
    should_abstain: bool  # no candidate is Exact
    exact_count: int  # Exact candidates in ESCI's original list (before any were removed)
    split: str
    candidates: tuple[Candidate, ...] = field(default_factory=tuple)

    @property
    def labels(self) -> list[str]:
        return [c.label for c in self.candidates]


def make_search(
    rows: pd.DataFrame,
    *,
    split: str,
    grocery: bool,
    seed: int = SEED,
    remove_exact: bool = False,
    reverse: bool = False,
) -> Search:
    """One search: its candidates in a seeded per-search shuffle (reversed if asked), with short ids in that order."""
    rows = rows.sort_values("product_id", kind="stable")
    first = rows.iloc[0]
    query_id = int(first["query_id"])
    exact_count = int((rows["esci_label"] == "E").sum())
    order = np.random.default_rng([seed, query_id]).permutation(len(rows))
    shuffled = rows.iloc[order]
    if remove_exact:
        shuffled = shuffled[shuffled["esci_label"] != "E"]
    if reverse:
        shuffled = shuffled.iloc[::-1]
    candidates = tuple(
        Candidate(
            cid=f"c{i}",
            product_id=str(row.product_id),
            title=(_clean(row.product_title) or "")[:TITLE_MAX_CHARS],
            brand=_clean(row.product_brand),
            color=_clean(row.product_color),
            label=str(row.esci_label),
        )
        for i, row in enumerate(shuffled.itertuples(index=False), start=1)
    )
    remaining_exact = sum(c.label == "E" for c in candidates)
    return Search(
        id=f"q{query_id}",
        query_id=query_id,
        query=_clean(first["query"]) or "",
        source=str(first["source"]) if isinstance(first["source"], str) else "unknown",
        grocery=grocery,
        hard=not remove_exact and 0 < exact_count <= HARD_MAX_EXACT,
        should_abstain=remaining_exact == 0,
        exact_count=exact_count,
        split=split,
        candidates=candidates,
    )


def eligible_query_ids(frame: pd.DataFrame, esci_split: str) -> list[int]:
    """Searches in one ESCI split with MIN_CANDIDATES to MAX_CANDIDATES candidates, sorted by id."""
    part = frame[frame["split"] == esci_split]
    sizes = part.groupby("query_id").size()
    return sorted(int(q) for q in sizes[(sizes >= MIN_CANDIDATES) & (sizes <= MAX_CANDIDATES)].index)


def _can_construct_nomatch(rows: pd.DataFrame) -> bool:
    return int((rows["esci_label"] != "E").sum()) >= NOMATCH_MIN_CANDIDATES and bool((rows["esci_label"] == "E").any())


# ---------- grocery review ----------


@dataclass(frozen=True)
class GroceryTerms:
    review_pool_size: int
    min_food_title_share: float
    query_pattern: re.Pattern[str]
    title_pattern: re.Pattern[str]
    size_pattern: re.Pattern[str]


def load_grocery_terms(path: Path = GROCERY_TERMS_PATH) -> GroceryTerms:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    def words(key: str) -> re.Pattern[str]:
        return re.compile(r"\b(" + "|".join(raw[key]) + r")\b", re.IGNORECASE)

    return GroceryTerms(
        review_pool_size=int(raw["review_pool_size"]),
        min_food_title_share=float(raw["min_food_title_share"]),
        query_pattern=words("query_terms"),
        title_pattern=words("title_terms"),
        size_pattern=re.compile(raw["title_size"], re.IGNORECASE),
    )


def grocery_candidates(frame: pd.DataFrame, terms: GroceryTerms, seed: int = SEED) -> list[dict[str, object]]:
    """Test searches that might be grocery, for a person to review. Deterministic sample of `review_pool_size`."""
    pool: list[dict[str, object]] = []
    eligible = set(eligible_query_ids(frame, "test"))
    test = frame[frame["query_id"].isin(eligible)]
    for query_id, rows in test.groupby("query_id", sort=True):
        query = _clean(rows["query"].iloc[0]) or ""
        titles = [str(t) for t in rows.sort_values("product_id")["product_title"]]
        food_titles = [t for t in titles if terms.title_pattern.search(t) and terms.size_pattern.search(t)]
        share = len(food_titles) / len(titles)
        by_query = bool(terms.query_pattern.search(query))
        by_products = share >= terms.min_food_title_share
        if not (by_query or by_products):
            continue
        exact_titles = rows[rows["esci_label"] == "E"].sort_values("product_id")["product_title"]
        pool.append(
            {
                "query_id": int(query_id),  # pyright: ignore[reportArgumentType]
                "query": query,
                "signal": "both" if by_query and by_products else ("query" if by_query else "products"),
                "food_title_share": round(share, 2),
                "exact_product": (_clean(exact_titles.iloc[0]) or "")[:120] if len(exact_titles) else None,
                "grocery": None,
            }
        )
    if len(pool) > terms.review_pool_size:
        keep = sorted(np.random.default_rng(seed).choice(len(pool), terms.review_pool_size, replace=False))
        pool = [pool[i] for i in keep]
    return pool


REVIEW_HEADER = """\
# Grocery review: one decision per candidate search from data/rerank/grocery_terms.yaml.
# `grocery: true` when the shopper wants something to eat or drink; `false` otherwise (see grocery_terms.yaml).
# `jevbench rerank prepare` refuses to build splits while any decision is still null.
"""


class _IndentedDumper(yaml.SafeDumper):
    """Indents list items under their key, as yamllint expects."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        super().increase_indent(flow, False)


def dump_review(searches: list[dict[str, object]], header: str = REVIEW_HEADER) -> str:
    body = yaml.dump({"searches": searches}, Dumper=_IndentedDumper, sort_keys=False, allow_unicode=True, width=1000)
    return header + "---\n" + body


def write_grocery_review(pool: list[dict[str, object]], path: Path = GROCERY_REVIEW_PATH) -> Path:
    path.write_text(dump_review(pool), encoding="utf-8")
    return path


def load_grocery_review(path: Path = GROCERY_REVIEW_PATH) -> tuple[set[int], set[int]]:
    """(every reviewed search, the searches judged grocery)."""
    if not path.exists():
        raise DataError(f"{path} not found. Run `jevbench rerank prepare --grocery-candidates` and review it first.")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    reviewed: set[int] = set()
    approved: set[int] = set()
    undecided: list[int] = []
    for entry in raw.get("searches", []):
        query_id = int(entry["query_id"])
        reviewed.add(query_id)
        decision = entry.get("grocery")
        if decision is None:
            undecided.append(query_id)
        elif not isinstance(decision, bool):
            raise DataError(f"{path}: search {query_id} has grocery={decision!r}; use true, false, or null.")
        elif decision:
            approved.add(query_id)
    if undecided:
        raise DataError(f"{path}: {len(undecided)} searches have no grocery decision yet (e.g. {undecided[0]}).")
    return reviewed, approved


# ---------- splits ----------


def build_splits(
    frame: pd.DataFrame, *, reviewed: set[int], approved: set[int], seed: int = SEED
) -> dict[str, list[Search]]:
    by_query = {int(q): rows for q, rows in frame.groupby("query_id", sort=True)}  # pyright: ignore[reportArgumentType]
    rng = np.random.default_rng(seed)
    splits: dict[str, list[Search]] = {name: [] for name in SPLIT_NAMES}

    # Train: practice, practice-nomatch, warm-up.
    train = list(rng.permutation(eligible_query_ids(frame, "train")))
    cursor = 0
    for name, size in TRAIN_SPLITS:
        while len(splits[name]) < size:
            if cursor >= len(train):
                raise DataError(f"Not enough eligible ESCI train searches for split {name}.")
            query_id = int(train[cursor])
            cursor += 1
            rows = by_query[query_id]
            if name == "practice-nomatch":
                if not _can_construct_nomatch(rows):
                    continue
                splits[name].append(make_search(rows, split=name, grocery=False, seed=seed, remove_exact=True))
            else:
                splits[name].append(make_search(rows, split=name, grocery=False, seed=seed))

    # Test: main (grocery + random), nomatch, consistency, reversed.
    test_ids = eligible_query_ids(frame, "test")
    grocery = [int(q) for q in rng.permutation(sorted(approved & set(test_ids)))][:MAIN_GROCERY]
    others = [int(q) for q in rng.permutation([q for q in test_ids if q not in reviewed])]
    random_size = MAIN_SIZE - len(grocery)
    main_random, rest = others[:random_size], others[random_size:]
    if len(main_random) < random_size:
        raise DataError("Not enough eligible ESCI test searches for the main split.")
    main = [make_search(by_query[q], split="main", grocery=True, seed=seed) for q in grocery]
    main += [make_search(by_query[q], split="main", grocery=False, seed=seed) for q in main_random]
    # Shuffle run order so time-of-run drift cannot line up with the grocery slice.
    splits["main"] = [main[i] for i in rng.permutation(len(main))]

    cursor = 0
    while len(splits["nomatch"]) < NOMATCH:
        if cursor >= len(rest):
            raise DataError("Not enough eligible ESCI test searches for the nomatch split.")
        rows = by_query[rest[cursor]]
        cursor += 1
        if _can_construct_nomatch(rows):
            splits["nomatch"].append(make_search(rows, split="nomatch", grocery=False, seed=seed, remove_exact=True))
    consistency_ids = rest[cursor : cursor + CONSISTENCY]
    if len(consistency_ids) < CONSISTENCY:
        raise DataError("Not enough eligible ESCI test searches for the consistency split.")
    splits["consistency"] = [
        make_search(by_query[q], split="consistency", grocery=False, seed=seed) for q in consistency_ids
    ]
    splits["reversed"] = [
        make_search(by_query[q], split="reversed", grocery=False, seed=seed, reverse=True) for q in consistency_ids
    ]

    used = [s.query_id for name, searches in splits.items() if name != "reversed" for s in searches]
    if len(used) != len(set(used)):
        raise DataError("A search was assigned to more than one split.")
    return splits


def write_splits(splits: dict[str, list[Search]], out_dir: Path = SPLITS_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, searches in splits.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for search in searches:
                fh.write(json.dumps(asdict(search), ensure_ascii=False) + "\n")
        paths[name] = path
    return paths


def search_from_dict(raw: dict[str, object]) -> Search:
    candidates = tuple(Candidate(**c) for c in raw.pop("candidates"))  # pyright: ignore[reportArgumentType, reportCallIssue, reportGeneralTypeIssues]
    return Search(**raw, candidates=candidates)  # pyright: ignore[reportArgumentType]


def load_split(name: str, splits_dir: Path = SPLITS_DIR) -> list[Search]:
    path = splits_dir / f"{name}.jsonl"
    if not path.exists():
        raise DataError(f"{path} not found. Run `jevbench rerank prepare` first.")
    searches: list[Search] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                searches.append(search_from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                raise DataError(f"{path}:{line_no}: invalid search ({exc}). Re-run `jevbench rerank prepare`.") from exc
    return searches


def profile(splits: dict[str, list[Search]]) -> list[str]:
    """One line per split: searches, candidates, labels, and slices."""
    lines = []
    for name, searches in splits.items():
        labels = [label for s in searches for label in s.labels]
        sizes = [len(s.candidates) for s in searches]
        counts = {code: labels.count(code) for code in "ESCI"}
        lines.append(
            f"{name:17s} {len(searches):5d} searches  candidates {min(sizes)}-{max(sizes)} (median "
            f"{int(np.median(sizes))})  labels E/S/C/I {counts['E']}/{counts['S']}/{counts['C']}/{counts['I']}  "
            f"grocery={sum(s.grocery for s in searches)}  hard={sum(s.hard for s in searches)}  "
            f"abstain={sum(s.should_abstain for s in searches)}"
        )
    return lines
