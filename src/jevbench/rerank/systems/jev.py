"""Jev re-rankers: per-candidate Score or Noul questions in one request, one Noul per request, or one Choice.

Every request carries the same information Gemini gets: the query and each candidate's title, brand, and color under
a short id, plus the label definitions from data/rerank/labels.yaml.
"""

from __future__ import annotations

import time

import httpx2
import typesafe_sdk as ts

from jevbench.common.calls import Outcome
from jevbench.common.settings import Settings
from jevbench.common.typesafe import JevReply, ask, build_client
from jevbench.rerank.data import Search
from jevbench.rerank.labels import ESCI_CODE, LEVELS, LabelSet
from jevbench.rerank.ranking import Ranking, expected_gain, fan_out, order_by_scores

NONE_OF_THESE = "none_of_these"


def candidates_state(search: Search) -> dict[str, ts.JSONValue]:
    """The query and every candidate keyed by its short id, so a question can name one directly."""
    return {"query": search.query, "candidates": {c.cid: c.shown() for c in search.candidates}}


def score_question(labels: LabelSet, cid: str) -> ts.Score:
    """Score levels run from 0 (worst) to 3 (best): irrelevant, complement, substitute, exact."""
    return ts.Score(
        instructions=f"{labels.instruction} How well does the product `candidates.{cid}` match `query`?",
        criteria=[labels.level(name).criterion() for name in reversed(LEVELS)],
    )


def exact_noul(labels: LabelSet, product: str) -> ts.Noul:
    return ts.Noul(
        instructions=f"{labels.instruction} {labels.exact_question} The product is `{product}`; the query is `query`.",
        criteria={"true": labels.exact_true, "false": labels.exact_false},
    )


def choice_question(labels: LabelSet, search: Search) -> ts.Choice:
    criteria: dict[str, dict[str, str] | str] = {c.cid: c.shown() for c in search.candidates}
    criteria[NONE_OF_THESE] = labels.none_of_these
    return ts.Choice(
        instructions=f"{labels.instruction} Which product is the best exact match for `query`?",
        criteria=criteria,
    )


def _measured(reply: JevReply) -> dict[str, object]:
    return {
        "latency_ms": reply.latency_ms,
        "model_reported": reply.model_reported,
        "input_tokens": reply.input_tokens,
        "output_tokens": reply.output_tokens,
        "reasoning_tokens": 0,
    }


def _api_error(reply: JevReply, calls: int = 1) -> Ranking:
    return Ranking(outcome=Outcome.API_ERROR, latency_ms=reply.latency_ms, calls=calls, error=reply.api_error)


def _scored(
    search: Search,
    measured: dict[str, object],
    scores: dict[str, float],
    p_exact: dict[str, float],
    *,
    calls: int = 1,
) -> Ranking:
    """Rank by score; a candidate without an answer goes last and counts as missing."""
    cids = [c.cid for c in search.candidates]
    missing = len(cids) - len(scores)
    if not scores:
        return Ranking(
            outcome=Outcome.INVALID_OUTPUT,
            order=cids,
            calls=calls,
            missing_ids=missing,
            error="no usable answers",
            **measured,  # pyright: ignore[reportArgumentType]
        )
    return Ranking(
        outcome=Outcome.OK,
        order=order_by_scores(scores, cids),
        calls=calls,
        scores=scores,
        p_exact=p_exact,
        exact_confidence=max(p_exact.values()) if p_exact else None,
        missing_ids=missing,
        **measured,  # pyright: ignore[reportArgumentType]
    )


class _JevReranker:
    def __init__(
        self,
        settings: Settings,
        labels: LabelSet,
        *,
        name: str,
        transport: httpx2.AsyncBaseTransport | None = None,  # tests inject a mock transport
    ) -> None:
        self.name = name
        self.model_requested = settings.typesafe_model
        self.labels = labels
        self.client = build_client(settings, transport=transport)

    async def aclose(self) -> None:
        await self.client.aclose()


class JevScore(_JevReranker):
    """One request: a Score question per candidate over the four ESCI levels; rank by expected gain."""

    async def rerank(self, search: Search) -> Ranking:
        questions = {c.cid: score_question(self.labels, c.cid) for c in search.candidates}
        reply = await ask(self.client, self.model_requested, candidates_state(search), questions)
        if reply.api_error is not None:
            return _api_error(reply)
        scores: dict[str, float] = {}
        p_exact: dict[str, float] = {}
        labels: dict[str, str] = {}
        for c in search.candidates:
            answer = reply.answers.get(c.cid)
            if isinstance(answer, ts.ScoreAnswer):
                probabilities = dict(answer.probabilities)
                scores[c.cid] = expected_gain(probabilities)
                p_exact[c.cid] = probabilities.get(3, 0.0)
                best = max(probabilities, key=lambda level: probabilities[level])
                labels[c.cid] = ESCI_CODE[LEVELS[len(LEVELS) - 1 - best]]
        ranking = _scored(search, _measured(reply), scores, p_exact)
        ranking.labels = labels or None
        return ranking


class JevNoulBatch(_JevReranker):
    """One request: an "is this an exact match?" Noul per candidate; rank by probability of yes."""

    async def rerank(self, search: Search) -> Ranking:
        questions = {c.cid: exact_noul(self.labels, f"candidates.{c.cid}") for c in search.candidates}
        reply = await ask(self.client, self.model_requested, candidates_state(search), questions)
        if reply.api_error is not None:
            return _api_error(reply)
        nouls = {
            c.cid: answer.noul
            for c in search.candidates
            if isinstance(answer := reply.answers.get(c.cid), ts.NoulAnswer)
        }
        return _scored(search, _measured(reply), nouls, nouls)


class JevNoulPair(_JevReranker):
    """One request per candidate, all in parallel: the query and one product, one Noul. TypeSafe's cookbook design."""

    async def rerank(self, search: Search) -> Ranking:
        question = {"exact": exact_noul(self.labels, "product")}
        started = time.perf_counter()
        replies = await fan_out(
            [
                ask(self.client, self.model_requested, {"query": search.query, "product": c.shown()}, question)
                for c in search.candidates
            ]
        )
        latency_ms = (time.perf_counter() - started) * 1000
        for reply in replies:
            if reply.api_error is not None:
                return Ranking(
                    outcome=Outcome.API_ERROR, latency_ms=latency_ms, calls=len(replies), error=reply.api_error
                )
        nouls = {
            c.cid: answer.noul
            for c, reply in zip(search.candidates, replies, strict=True)
            if isinstance(answer := reply.answers.get("exact"), ts.NoulAnswer)
        }
        tokens = [r.input_tokens for r in replies if r.input_tokens is not None]
        out_tokens = [r.output_tokens for r in replies if r.output_tokens is not None]
        measured: dict[str, object] = {
            "latency_ms": latency_ms,
            "model_reported": next((r.model_reported for r in replies if r.model_reported), None),
            "input_tokens": sum(tokens) if tokens else None,
            "output_tokens": sum(out_tokens) if out_tokens else None,
            "reasoning_tokens": 0,
        }
        return _scored(search, measured, nouls, nouls, calls=len(replies))


class JevChoice(_JevReranker):
    """One request: a single Choice over the candidate ids plus "none of these"; rank by option probability."""

    async def rerank(self, search: Search) -> Ranking:
        question = {"best": choice_question(self.labels, search)}
        reply = await ask(self.client, self.model_requested, {"query": search.query}, question)
        if reply.api_error is not None:
            return _api_error(reply)
        answer = reply.answers.get("best")
        if not isinstance(answer, ts.ChoiceAnswer):
            return _scored(search, _measured(reply), {}, {})
        probabilities = dict(answer.probabilities)
        scores = {c.cid: probabilities[c.cid] for c in search.candidates if c.cid in probabilities}
        ranking = _scored(search, _measured(reply), scores, {})
        ranking.p_exact = None  # Choice probabilities share one unit of mass; they are not per-candidate P(exact)
        ranking.exact_confidence = 1.0 - probabilities.get(NONE_OF_THESE, 0.0)
        return ranking
