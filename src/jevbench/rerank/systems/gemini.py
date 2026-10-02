"""Gemini re-rankers through the LiteLLM-based gateway: rank the whole list, label every candidate, or one per call.

Every request carries the same information Jev gets: the query and each candidate's title, brand, and color under a
short id, plus the label definitions from data/rerank/labels.yaml.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping

import httpx2

from jevbench.common.calls import Outcome
from jevbench.common.gateway import GatewayClient, GatewayReply
from jevbench.common.settings import LLMSystemConfig, Settings
from jevbench.rerank.data import Search
from jevbench.rerank.labels import LEVELS, LabelSet
from jevbench.rerank.ranking import LABEL_RANK, Ranking, complete_order, fan_out, label_code, order_by_scores


def products_json(search: Search) -> str:
    """The user message: the query and every candidate, in presented order."""
    products = [{"id": c.cid, **c.shown()} for c in search.candidates]
    return json.dumps({"query": search.query, "products": products}, ensure_ascii=False, indent=1)


def _schema(name: str, properties: Mapping[str, object]) -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": dict(properties),
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


def _intro(labels: LabelSet) -> list[str]:
    return [
        "You judge products returned by an online store's search engine for a shopper's query.",
        labels.instruction,
        "Each label has a description (`means`), what it is not (`not`), and an example:",
        "",
        labels.criteria_json(),
        "",
    ]


def listwise_prompt(labels: LabelSet) -> str:
    return "\n".join(
        [
            *_intro(labels),
            "Rank every product, best match first: exact matches first, then substitutes, then complements, then"
            " irrelevant products. Return every product id exactly once in `ranked_ids`.",
            "If no product is an exact match, set `no_relevant_results` to true and write a one-sentence"
            " `explanation` for the shopper; still return the full ranking. Otherwise set it to false and leave"
            " `explanation` empty.",
        ]
    )


def label_prompt(labels: LabelSet) -> str:
    return "\n".join(
        [*_intro(labels), f"Label every product with exactly one of: {', '.join(LEVELS)}. Answer for every id."]
    )


def pointwise_prompt(labels: LabelSet) -> str:
    return "\n".join([*_intro(labels), f"Label the one product with exactly one of: {', '.join(LEVELS)}."])


def listwise_format(search: Search) -> dict[str, object]:
    ids = [c.cid for c in search.candidates]
    return _schema(
        "rerank",
        {
            "ranked_ids": {"type": "array", "items": {"type": "string", "enum": ids}},
            "no_relevant_results": {"type": "boolean"},
            "explanation": {"type": "string"},
        },
    )


def label_format(search: Search) -> dict[str, object]:
    return _schema("labels", {c.cid: {"type": "string", "enum": list(LEVELS)} for c in search.candidates})


POINTWISE_FORMAT = _schema("label", {"label": {"type": "string", "enum": list(LEVELS)}})


def _failed(reply: GatewayReply, search: Search) -> Ranking | None:
    """A ranking for a reply with no content to interpret (API error, safety block, no choices), else None."""
    if reply.problem is None:
        return None
    outcome, error = reply.problem
    if outcome == Outcome.API_ERROR:
        return Ranking(outcome=outcome, latency_ms=reply.latency_ms, error=error)
    fallback = [c.cid for c in search.candidates] if outcome == Outcome.INVALID_OUTPUT else []
    return Ranking(outcome=outcome, order=fallback, error=error, **reply.measurement())  # pyright: ignore[reportArgumentType]


def _parse_object(reply: GatewayReply) -> dict[str, object] | None:
    try:
        parsed = json.loads(reply.content or "")
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _invalid(reply: GatewayReply, search: Search, why: str) -> Ranking:
    """An answer we cannot use: it counts as answered, ranked in presented order."""
    return Ranking(
        outcome=Outcome.INVALID_OUTPUT,
        order=[c.cid for c in search.candidates],
        error=f"{why}: {reply.content!r:.300}",
        **reply.measurement(),  # pyright: ignore[reportArgumentType]
    )


class _GeminiReranker:
    def __init__(
        self,
        settings: Settings,
        config: LLMSystemConfig,
        labels: LabelSet,
        *,
        name: str,
        http_client: httpx2.AsyncClient | None = None,  # tests inject a mock transport
    ) -> None:
        self.name = name
        self.model_requested = config.model
        self.labels = labels
        self.gateway = GatewayClient(settings, config, http_client=http_client)

    async def aclose(self) -> None:
        await self.gateway.aclose()


class GeminiListwise(_GeminiReranker):
    """The common production setup: one call sees every candidate and returns an ordering."""

    async def rerank(self, search: Search) -> Ranking:
        messages = [
            {"role": "system", "content": listwise_prompt(self.labels)},
            {"role": "user", "content": products_json(search)},
        ]
        reply = await self.gateway.complete(messages, listwise_format(search))
        return self.interpret(reply, search)

    def interpret(self, reply: GatewayReply, search: Search) -> Ranking:
        if (failed := _failed(reply, search)) is not None:
            return failed
        parsed = _parse_object(reply)
        if parsed is None or not isinstance(parsed.get("ranked_ids"), list):
            return _invalid(reply, search, "no ranked_ids list")
        order, invented, duplicates, missing = complete_order(
            parsed["ranked_ids"],  # pyright: ignore[reportArgumentType]
            [c.cid for c in search.candidates],
        )
        flag = parsed.get("no_relevant_results")
        explanation = parsed.get("explanation")
        return Ranking(
            outcome=Outcome.OK,
            order=order,
            abstain_flag=flag if isinstance(flag, bool) else None,
            explanation=explanation.strip() or None if isinstance(explanation, str) else None,
            invented_ids=invented,
            duplicate_ids=duplicates,
            missing_ids=missing,
            **reply.measurement(),  # pyright: ignore[reportArgumentType]
        )


class GeminiLabel(_GeminiReranker):
    """One call labels every candidate E/S/C/I; code ranks by label. The same task Jev's Score design does."""

    async def rerank(self, search: Search) -> Ranking:
        messages = [
            {"role": "system", "content": label_prompt(self.labels)},
            {"role": "user", "content": products_json(search)},
        ]
        reply = await self.gateway.complete(messages, label_format(search))
        return self.interpret(reply, search)

    def interpret(self, reply: GatewayReply, search: Search) -> Ranking:
        if (failed := _failed(reply, search)) is not None:
            return failed
        parsed = _parse_object(reply)
        if parsed is None:
            return _invalid(reply, search, "not a JSON object")
        cids = [c.cid for c in search.candidates]
        labels = {cid: code for cid in cids if (code := label_code(parsed.get(cid))) is not None}
        return _ranked_by_labels(labels, cids, reply)


class GeminiPointwise(_GeminiReranker):
    """One call per candidate, all in parallel, each labeling one product E/S/C/I."""

    async def rerank(self, search: Search) -> Ranking:
        system = {"role": "system", "content": pointwise_prompt(self.labels)}
        started = time.perf_counter()
        replies = await fan_out(
            [
                self.gateway.complete(
                    [
                        system,
                        {
                            "role": "user",
                            "content": json.dumps({"query": search.query, "product": c.shown()}, ensure_ascii=False),
                        },
                    ],
                    POINTWISE_FORMAT,
                )
                for c in search.candidates
            ]
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return self.interpret(replies, search, latency_ms)

    def interpret(self, replies: list[GatewayReply], search: Search, latency_ms: float) -> Ranking:
        cids = [c.cid for c in search.candidates]
        totals = _summed(replies)
        problems = [reply.problem for reply in replies if reply.problem is not None]
        # An API error fails the whole search (a resumed run redoes it). A safety block withholds one candidate's
        # label: that candidate goes last, like a missing one, and the search is blocked only if every call was.
        api_error = next((p for p in problems if p[0] == Outcome.API_ERROR), None)
        blocked = sum(p[0] == Outcome.REFUSED for p in problems)
        if api_error is not None or blocked == len(replies):
            outcome, error = api_error or problems[0]
            return Ranking(outcome=outcome, latency_ms=latency_ms, calls=len(replies), error=error, **totals)  # pyright: ignore[reportArgumentType]
        labels: dict[str, str] = {}
        for cid, reply in zip(cids, replies, strict=True):
            parsed = _parse_object(reply) if reply.problem is None else None
            code = label_code(parsed.get("label")) if parsed else None
            if code is not None:
                labels[cid] = code
        ranking = _ranked_by_labels(labels, cids, None, latency_ms=latency_ms, calls=len(replies), **totals)
        ranking.blocked_calls = blocked
        if blocked:
            ranking.error = f"{blocked} of {len(replies)} calls blocked by the safety filter"
        return ranking


def _summed(replies: list[GatewayReply]) -> dict[str, object]:
    """Token totals over a fan-out; the model id and cache flag from any call that has them."""

    def total(field: str) -> int | None:
        values = [getattr(r, field) for r in replies if getattr(r, field) is not None]
        return sum(values) if values else None

    return {
        "model_reported": next((r.model_reported for r in replies if r.model_reported), None),
        "input_tokens": total("input_tokens"),
        "output_tokens": total("output_tokens"),
        "reasoning_tokens": total("reasoning_tokens"),
        "cache_hit": any(r.cache_hit for r in replies),
    }


def _ranked_by_labels(
    labels: dict[str, str], cids: list[str], reply: GatewayReply | None, **measured: object
) -> Ranking:
    """Rank by label (exact first); an unlabeled candidate goes last and counts as missing."""
    missing = len(cids) - len(labels)
    if reply is not None:
        measured = reply.measurement()
    if not labels:
        content = reply.content if reply is not None else None
        return Ranking(
            outcome=Outcome.INVALID_OUTPUT,
            order=list(cids),
            missing_ids=missing,
            error=f"no usable labels: {content!r:.300}",
            **measured,  # pyright: ignore[reportArgumentType]
        )
    scores = {cid: LABEL_RANK[code] for cid, code in labels.items()}
    return Ranking(
        outcome=Outcome.OK,
        order=order_by_scores(scores, cids),
        scores=scores,
        labels=labels,
        abstain_flag="E" not in labels.values(),
        missing_ids=missing,
        **measured,  # pyright: ignore[reportArgumentType]
    )
