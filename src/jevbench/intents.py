"""The intent catalog: one source of label text for every classifier, so no system gets extra help."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from jevbench.settings import REPO_ROOT

DEFAULT_INTENTS_PATH = REPO_ROOT / "data" / "intents.yaml"


@dataclass(frozen=True)
class Intent:
    name: str
    category: str
    means: str
    not_: str | None

    def criterion(self) -> dict[str, str]:
        """The description object sent to every model for this option."""
        described = {"means": self.means}
        if self.not_:
            described["not"] = self.not_
        return described


@dataclass(frozen=True)
class IntentCatalog:
    instruction: str
    intents: tuple[Intent, ...]
    sha256: str

    @property
    def names(self) -> list[str]:
        return [i.name for i in self.intents]

    def category_of(self, intent: str) -> str | None:
        return next((i.category for i in self.intents if i.name == intent), None)

    def criteria(self) -> dict[str, dict[str, str]]:
        """{intent_name: {"means": ..., "not": ...}} in catalog order."""
        return {i.name: i.criterion() for i in self.intents}

    def criteria_json(self) -> str:
        """The exact serialized label text shown to the LLM."""
        return json.dumps(self.criteria(), indent=2, ensure_ascii=False)


class CatalogError(ValueError):
    pass


def _intent(item: object, position: int, path: Path) -> Intent:
    """One catalog entry, validated: `name`, `category`, and `means` are required text; `not` is optional text."""
    if not isinstance(item, dict):
        raise CatalogError(f"{path}: intent #{position} must be a mapping.")
    fields = {key: item.get(key) for key in ("name", "category", "means", "not")}
    missing = [key for key in ("name", "category", "means") if not isinstance(fields[key], str) or not fields[key]]
    if missing:
        raise CatalogError(f"{path}: intent #{position} needs non-empty text for {', '.join(missing)}.")
    if fields["not"] is not None and not isinstance(fields["not"], str):
        raise CatalogError(f"{path}: intent #{position} `not` must be text.")
    return Intent(
        name=str(fields["name"]),
        category=str(fields["category"]),
        means=str(fields["means"]),
        not_=str(fields["not"]) if fields["not"] is not None else None,
    )


def load_catalog(path: Path = DEFAULT_INTENTS_PATH) -> IntentCatalog:
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise CatalogError(f"Cannot read intent catalog at {path}: {exc}") from exc
    try:
        raw = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise CatalogError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict) or "instruction" not in raw or "intents" not in raw:
        raise CatalogError(f"{path} must define `instruction` and `intents`.")
    if not isinstance(raw["intents"], list) or not raw["intents"]:
        raise CatalogError(f"{path}: `intents` must be a non-empty list.")

    intents = tuple(_intent(item, position, path) for position, item in enumerate(raw["intents"], start=1))
    names = [i.name for i in intents]
    if len(names) != len(set(names)):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise CatalogError(f"Duplicate intents in {path}: {duplicates}")
    if len(intents) > 255:
        raise CatalogError("Jev Choice questions accept at most 255 options.")

    return IntentCatalog(
        instruction=str(raw["instruction"]),
        intents=intents,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
