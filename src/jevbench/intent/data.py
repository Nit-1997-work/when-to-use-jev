"""Build the practice / main / profanity / warmup splits from the Bitext Retail CSV.

Deterministic: the same CSV and seed always produce byte-identical split files.
No source row appears in more than one split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from jevbench.common.paths import data_dir, file_sha256
from jevbench.intent.intents import IntentCatalog

__all__ = [
    "SPLITS_DIR",
    "DataError",
    "Example",
    "build_splits",
    "file_sha256",
    "load_raw",
    "load_split",
    "write_splits",
]

RAW_CSV = data_dir("intent") / "raw" / "bitext-retail-ecommerce.csv"
# Pinned revision's checksum (data/SOURCES.md). A different file would silently produce different splits.
RAW_CSV_SHA256 = "13a988266fed4e2b2c1ff947a89ef220ce09b5b13ac83c4a1496c0d7b81e8127"
SPLITS_DIR = data_dir("intent") / "splits"
SEED = 20260929
PROFANITY_TAG = "W"
TYPO_TAG = "Z"


@dataclass(frozen=True)
class SplitSpec:
    name: str
    clean_per_intent: int
    profane_per_intent: int


# Order matters: rows are allocated to splits in this order from one shuffled pool per intent.
SPLITS: tuple[SplitSpec, ...] = (
    SplitSpec("practice", clean_per_intent=4, profane_per_intent=1),
    SplitSpec("main", clean_per_intent=45, profane_per_intent=5),
    SplitSpec("profanity", clean_per_intent=0, profane_per_intent=10),
)
WARMUP_SIZE = 20


class DataError(ValueError):
    pass


def load_raw(path: Path = RAW_CSV, *, expected_sha256: str | None = RAW_CSV_SHA256) -> pd.DataFrame:
    if not path.exists():
        raise DataError(f"{path} not found. Download it with the command in data/SOURCES.md.")
    if expected_sha256 is not None and (actual := file_sha256(path)) != expected_sha256:
        raise DataError(
            f"{path} has SHA-256 {actual}, expected {expected_sha256}. "
            "Download the pinned revision with the command in data/SOURCES.md."
        )
    df = pd.read_csv(path, usecols=["instruction", "intent", "category", "tags"], dtype=str, keep_default_na=False)
    df.insert(0, "id", [f"bitext-{i:05d}" for i in range(len(df))])
    df = df.rename(columns={"instruction": "text"})
    df["has_profanity"] = df["tags"].str.contains(PROFANITY_TAG, regex=False)
    df["has_typos"] = df["tags"].str.contains(TYPO_TAG, regex=False)
    return df


def build_splits(df: pd.DataFrame, catalog: IntentCatalog, seed: int = SEED) -> dict[str, pd.DataFrame]:
    dataset_intents = set(df["intent"])
    catalog_intents = set(catalog.names)
    if dataset_intents != catalog_intents:
        raise DataError(
            "Intent catalog does not match the dataset. "
            f"Missing from catalog: {sorted(dataset_intents - catalog_intents)}. "
            f"Not in dataset: {sorted(catalog_intents - dataset_intents)}."
        )
    for intent in catalog.intents:
        categories = set(df.loc[df["intent"] == intent.name, "category"])
        if categories != {intent.category}:
            raise DataError(f"{intent.name}: catalog says {intent.category}, dataset says {sorted(categories)}.")

    parts: dict[str, list[pd.DataFrame]] = {spec.name: [] for spec in SPLITS}
    leftovers: list[pd.DataFrame] = []
    for intent in catalog.names:  # catalog order, so output does not depend on pandas groupby order
        rows: pd.DataFrame = df.loc[df["intent"] == intent]
        profane_mask = rows["has_profanity"].astype(bool)
        pools: dict[bool, pd.DataFrame] = {
            False: rows.loc[~profane_mask].sample(frac=1.0, random_state=seed),
            True: rows.loc[profane_mask].sample(frac=1.0, random_state=seed),
        }
        cursor = {False: 0, True: 0}
        for spec in SPLITS:
            for profane, count in ((False, spec.clean_per_intent), (True, spec.profane_per_intent)):
                pool = pools[profane]
                if cursor[profane] + count > len(pool):
                    kind = "profane" if profane else "clean"
                    raise DataError(f"{intent}: not enough {kind} rows for split {spec.name}.")
                parts[spec.name].append(pool.iloc[cursor[profane] : cursor[profane] + count])
                cursor[profane] += count
        leftovers.append(pools[False].iloc[cursor[False] :])

    splits: dict[str, pd.DataFrame] = {}
    for spec in SPLITS:
        combined = pd.concat(parts[spec.name])
        # Shuffle run order so time-of-run drift (network, rate limits) cannot line up with intents.
        splits[spec.name] = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Warm-up calls use unused clean rows; their results are discarded.
    splits["warmup"] = pd.concat(leftovers).sample(n=WARMUP_SIZE, random_state=seed).reset_index(drop=True)

    used = pd.concat(splits.values())["id"]
    if used.duplicated().any():
        raise DataError("A source row was assigned to more than one split.")
    return splits


def write_splits(splits: dict[str, pd.DataFrame], out_dir: Path = SPLITS_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in splits.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for record in frame.to_dict(orient="records"):
                fh.write(json.dumps({**record, "split": name}, ensure_ascii=False) + "\n")
        paths[name] = path
    return paths


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    intent: str
    category: str
    tags: str
    has_profanity: bool
    has_typos: bool
    split: str


def load_split(name: str, splits_dir: Path = SPLITS_DIR) -> list[Example]:
    path = splits_dir / f"{name}.jsonl"
    if not path.exists():
        raise DataError(f"{path} not found. Run `jevbench intent prepare` first.")
    examples: list[Example] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                examples.append(Example(**json.loads(line)))
            except (json.JSONDecodeError, TypeError) as exc:
                raise DataError(
                    f"{path}:{line_no}: invalid example ({exc}). Re-run `jevbench intent prepare`."
                ) from exc
    return examples
