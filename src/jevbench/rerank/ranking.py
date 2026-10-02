"""What every re-ranking system returns, and the code that turns model answers into a complete ranking.

A system never authors product data: it returns candidate ids (or per-candidate scores), and code here completes the
ranking the same way for every system. Invented and duplicate ids are dropped and counted, candidates a system left out
are appended in presented order, and ties are broken by presented order.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from jevbench.common.calls import Outcome, RetryableError
from jevbench.rerank.data import Search
from jevbench.rerank.labels import CODE_LEVEL, ESCI_CODE, GAIN, LEVELS


@dataclass
class Ranking:
    outcome: Outcome
    latency_ms: float  # the whole search: for per-candidate systems, the parallel fan-out end to end
    order: list[str] = field(default_factory=list)  # candidate ids, best first, always complete when answered
    calls: int = 1  # HTTP calls made for this search (one attempt each)
    scores: dict[str, float] | None = None  # the per-candidate sort key, when the system produces one
    labels: dict[str, str] | None = None  # predicted ESCI code per candidate (E/S/C/I), for label-producing systems
    p_exact: dict[str, float] | None = None  # probability each candidate is an exact match (Jev systems)
    abstain_flag: bool | None = None  # the system's own "nothing matches" decision, when it makes one
    exact_confidence: float | None = None  # how likely some candidate is exact; the report thresholds it
    explanation: str | None = None
    invented_ids: int = 0
    duplicate_ids: int = 0
    missing_ids: int = 0
    blocked_calls: int = 0  # per-candidate calls the provider's safety filter withheld
    model_reported: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    gateway_upstream_ms: float | None = None
    gateway_overhead_ms: float | None = None
    cache_hit: bool = False
    error: str | None = None


class Reranker(Protocol):
    name: str
    model_requested: str

    async def rerank(self, search: Search) -> Ranking: ...

    async def aclose(self) -> None: ...


def complete_order(returned: Sequence[object], cids: Sequence[str]) -> tuple[list[str], int, int, int]:
    """(order, invented, duplicates, missing): the returned ids that exist, once each, then the rest in given order."""
    known = set(cids)
    order: list[str] = []
    seen: set[str] = set()
    invented = duplicates = 0
    for item in returned:
        cid = item if isinstance(item, str) else None
        if cid is None or cid not in known:
            invented += 1
        elif cid in seen:
            duplicates += 1
        else:
            order.append(cid)
            seen.add(cid)
    missing = [cid for cid in cids if cid not in seen]
    return order + missing, invented, duplicates, len(missing)


def order_by_scores(scores: Mapping[str, float], cids: Sequence[str]) -> list[str]:
    """Highest score first; ties, and candidates without a score, keep their presented order."""
    position = {cid: i for i, cid in enumerate(cids)}
    return sorted(cids, key=lambda cid: (-scores.get(cid, float("-inf")), position[cid]))


LABEL_RANK = {"E": 3.0, "S": 2.0, "C": 1.0, "I": 0.0}


def label_code(value: object) -> str | None:
    """An ESCI code from a level name ("exact") or a code ("E"); None for anything else."""
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if text in ESCI_CODE:
        return ESCI_CODE[text]
    upper = text.upper()
    return upper if upper in CODE_LEVEL else None


def expected_gain(probabilities: Mapping[int, float]) -> float:
    """Expected KDD gain under Score probabilities over levels 0..3 (irrelevant, complement, substitute, exact)."""
    by_level = dict(zip(reversed(LEVELS), range(len(LEVELS)), strict=True))  # irrelevant=0 ... exact=3
    return sum(probabilities.get(by_level[level], 0.0) * GAIN[ESCI_CODE[level]] for level in LEVELS)


async def fan_out[T](calls: Sequence[Awaitable[T]]) -> list[T]:
    """Run calls in parallel. If any hits a retryable error, cancel the rest and raise one `RetryableError`, so the
    runner retries the whole search and latency always covers one clean attempt of the full fan-out."""
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(_awaited(call)) for call in calls]
    except ExceptionGroup as failures:
        retryable, others = failures.split(RetryableError)
        if others is not None or retryable is None:
            raise
        errors = [e for e in retryable.exceptions if isinstance(e, RetryableError)]
        waits = [e.retry_after_s for e in errors if e.retry_after_s is not None]
        message = f"{len(errors)} of {len(calls)} calls: {errors[0]}"
        raise RetryableError(message, retry_after_s=max(waits, default=None)) from None
    return [task.result() for task in tasks]


async def _awaited[T](call: Awaitable[T]) -> T:
    return await call
