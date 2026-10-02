"""Reference re-rankers that call no model: BM25 within the shortlist, and the presented (seeded random) order."""

from __future__ import annotations

import math
import re
import time
from collections import Counter

from jevbench.common.calls import Outcome
from jevbench.rerank.data import Search
from jevbench.rerank.ranking import Ranking, order_by_scores

_TOKEN = re.compile(r"[a-z0-9]+")
K1 = 1.5
B = 0.75


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def bm25_scores(query: str, documents: dict[str, str]) -> dict[str, float]:
    """Okapi BM25 of `query` against each document, with the shortlist itself as the corpus."""
    docs = {key: tokens(text) for key, text in documents.items()}
    n = len(docs)
    avg_len = sum(len(d) for d in docs.values()) / n if n else 0.0
    frequency = Counter(term for d in docs.values() for term in set(d))
    scores: dict[str, float] = {}
    for key, doc in docs.items():
        counts = Counter(doc)
        score = 0.0
        for term in set(tokens(query)):
            if term not in counts:
                continue
            idf = math.log(1 + (n - frequency[term] + 0.5) / (frequency[term] + 0.5))
            tf = counts[term]
            norm = tf + K1 * (1 - B + B * len(doc) / avg_len) if avg_len else tf + K1
            score += idf * tf * (K1 + 1) / norm
        scores[key] = score
    return scores


class Lexical:
    """BM25 over title and brand within the shortlist: the "fast search only" baseline."""

    name = "lexical"
    model_requested = "bm25"

    async def rerank(self, search: Search) -> Ranking:
        started = time.perf_counter()
        documents = {c.cid: f"{c.title} {c.brand or ''}" for c in search.candidates}
        scores = bm25_scores(search.query, documents)
        order = order_by_scores(scores, [c.cid for c in search.candidates])
        latency_ms = (time.perf_counter() - started) * 1000
        return Ranking(outcome=Outcome.OK, latency_ms=latency_ms, order=order, calls=0, scores=scores)

    async def aclose(self) -> None:
        return None


class PresentedOrder:
    """The presented order, which is a seeded per-search shuffle: the floor every re-ranker must beat."""

    name = "random"
    model_requested = "none"

    async def rerank(self, search: Search) -> Ranking:
        return Ranking(outcome=Outcome.OK, latency_ms=0.0, order=[c.cid for c in search.candidates], calls=0)

    async def aclose(self) -> None:
        return None
