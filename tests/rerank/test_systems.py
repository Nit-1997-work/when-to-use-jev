from __future__ import annotations

import json
from collections.abc import Callable, Mapping

import httpx2
import pytest

from jevbench.common.calls import Outcome, RetryableError
from jevbench.common.settings import Settings
from jevbench.rerank.data import Search
from jevbench.rerank.labels import load_labels
from jevbench.rerank.ranking import complete_order, expected_gain, fan_out, label_code, order_by_scores
from jevbench.rerank.systems import NO_MODEL, SYSTEMS, build_reranker, llm_config_key, prices_for
from jevbench.rerank.systems.baselines import Lexical, PresentedOrder, bm25_scores
from jevbench.rerank.systems.gemini import GeminiLabel, GeminiListwise, GeminiPointwise
from jevbench.rerank.systems.jev import NONE_OF_THESE, JevChoice, JevNoulBatch, JevNoulPair, JevScore
from tests.rerank.helpers import make_search

Handler = Callable[[httpx2.Request], httpx2.Response]
LABELS = load_labels()


# ---------- shared ranking code ----------


def test_complete_order_drops_invented_and_duplicate_ids_and_appends_missing() -> None:
    order, invented, duplicates, missing = complete_order(["c2", "c9", "c2", 7], ["c1", "c2", "c3"])
    assert order == ["c2", "c1", "c3"]
    assert (invented, duplicates, missing) == (2, 1, 2)


def test_scores_rank_highest_first_and_ties_keep_presented_order() -> None:
    assert order_by_scores({"c1": 0.5, "c2": 0.9, "c3": 0.5}, ["c1", "c2", "c3", "c4"]) == ["c2", "c1", "c3", "c4"]


def test_label_codes_and_expected_gain() -> None:
    assert [label_code(v) for v in ("exact", "Substitute", "C", "i", "maybe", 3)] == ["E", "S", "C", "I", None, None]
    assert expected_gain({3: 1.0}) == 1.0
    assert expected_gain({0: 0.5, 2: 0.5}) == pytest.approx(0.05)


async def test_fan_out_turns_any_retryable_failure_into_one_retry() -> None:
    async def ok() -> int:
        return 1

    async def slow_down() -> int:
        raise RetryableError("429", retry_after_s=2.0)

    async def broken() -> int:
        raise ValueError("bug")

    assert await fan_out([ok(), ok()]) == [1, 1]
    with pytest.raises(RetryableError) as caught:
        await fan_out([ok(), slow_down(), slow_down()])
    assert caught.value.retry_after_s == 2.0
    assert "2 of 3 calls" in str(caught.value)
    with pytest.raises(ExceptionGroup):
        await fan_out([ok(), broken()])


# ---------- Gemini ----------


def _completion(content: str | None, *, finish_reason: str = "stop") -> dict[str, object]:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "google/test-model",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 900, "completion_tokens": 40, "total_tokens": 940},
    }


def _client(handler: Handler) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


async def test_listwise_sends_every_candidate_and_completes_the_ranking(settings: Settings, search: Search) -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(json.loads(request.content))
        answer = {"ranked_ids": ["c2", "c9", "c2"], "no_relevant_results": False, "explanation": ""}
        return httpx2.Response(200, json=_completion(json.dumps(answer)))

    reranker = GeminiListwise(settings, settings.baseline, LABELS, name="gemini-listwise", http_client=_client(handler))
    ranking = await reranker.rerank(search)
    await reranker.aclose()

    user = json.loads(seen["messages"][1]["content"])  # pyright: ignore[reportIndexIssue]
    assert user["query"] == "sesame chips"
    assert [p["id"] for p in user["products"]] == ["c1", "c2", "c3", "c4"]
    assert "label" not in json.dumps(user)  # the gold label never reaches the model
    schema = seen["response_format"]["json_schema"]["schema"]  # pyright: ignore[reportIndexIssue]
    assert schema["properties"]["ranked_ids"]["items"]["enum"] == ["c1", "c2", "c3", "c4"]
    assert seen["extra_body"] if "extra_body" in seen else seen["cache"] == {"no-cache": True}
    assert ranking.outcome == Outcome.OK
    assert ranking.order == ["c2", "c1", "c3", "c4"]
    assert (ranking.invented_ids, ranking.duplicate_ids, ranking.missing_ids) == (1, 1, 3)
    assert ranking.abstain_flag is False
    assert ranking.explanation is None
    assert (ranking.input_tokens, ranking.output_tokens) == (900, 40)


async def test_listwise_failures(settings: Settings, search: Search) -> None:
    def reranker(handler: Handler) -> GeminiListwise:
        return GeminiListwise(settings, settings.baseline, LABELS, name="g", http_client=_client(handler))

    unusable = await reranker(lambda _: httpx2.Response(200, json=_completion("not json"))).rerank(search)
    assert unusable.outcome == Outcome.INVALID_OUTPUT
    assert unusable.order == ["c1", "c2", "c3", "c4"]  # ranked in presented order
    blocked = await reranker(
        lambda _: httpx2.Response(200, json=_completion(None, finish_reason="content_filter"))
    ).rerank(search)
    assert blocked.outcome == Outcome.REFUSED
    bad = await reranker(lambda _: httpx2.Response(400, json={"error": {"message": "bad"}})).rerank(search)
    assert bad.outcome == Outcome.API_ERROR
    with pytest.raises(RetryableError):
        await reranker(lambda _: httpx2.Response(503, json={"error": {"message": "down"}})).rerank(search)


async def test_listwise_keeps_the_models_abstention(settings: Settings) -> None:
    answer = {"ranked_ids": ["c1", "c2"], "no_relevant_results": True, "explanation": " None of these are chips. "}
    client = _client(lambda _: httpx2.Response(200, json=_completion(json.dumps(answer))))
    ranking = await GeminiListwise(settings, settings.baseline, LABELS, name="g", http_client=client).rerank(
        make_search("SI")
    )
    assert ranking.abstain_flag is True
    assert ranking.explanation == "None of these are chips."


async def test_label_system_ranks_by_label(settings: Settings, search: Search) -> None:
    answer = {"c1": "complement", "c2": "exact", "c3": "irrelevant", "c4": "nonsense"}
    seen: dict[str, object] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(json.loads(request.content))
        return httpx2.Response(200, json=_completion(json.dumps(answer)))

    ranking = await GeminiLabel(settings, settings.baseline, LABELS, name="g", http_client=_client(handler)).rerank(
        search
    )
    schema = seen["response_format"]["json_schema"]["schema"]  # pyright: ignore[reportIndexIssue]
    assert schema["required"] == ["c1", "c2", "c3", "c4"]
    assert ranking.order == ["c2", "c1", "c3", "c4"]
    assert ranking.labels == {"c1": "C", "c2": "E", "c3": "I"}
    assert ranking.missing_ids == 1
    assert ranking.abstain_flag is False
    nothing = await GeminiLabel(
        settings,
        settings.baseline,
        LABELS,
        name="g",
        http_client=_client(lambda _: httpx2.Response(200, json=_completion("{}"))),
    ).rerank(search)
    assert nothing.outcome == Outcome.INVALID_OUTPUT


async def test_pointwise_makes_one_call_per_candidate(settings: Settings, search: Search) -> None:
    labels_by_title = {"Product 1 E": "exact", "Product 2 S": "substitute", "Product 3 C": "complement"}
    calls: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        product = json.loads(json.loads(request.content)["messages"][1]["content"])["product"]
        calls.append(product["title"])
        label = labels_by_title.get(product["title"], "irrelevant")
        return httpx2.Response(200, json=_completion(json.dumps({"label": label})))

    ranking = await GeminiPointwise(settings, settings.baseline, LABELS, name="g", http_client=_client(handler)).rerank(
        search
    )
    assert sorted(calls) == ["Product 1 E", "Product 2 S", "Product 3 C", "Product 4 I"]
    assert ranking.calls == 4
    assert ranking.order == ["c1", "c2", "c3", "c4"]
    assert ranking.input_tokens == 4 * 900  # summed over the fan-out

    def block_one(request: httpx2.Request) -> httpx2.Response:
        product = json.loads(json.loads(request.content)["messages"][1]["content"])["product"]
        finish = "content_filter" if product["title"] == "Product 3 C" else "stop"
        return httpx2.Response(200, json=_completion(json.dumps({"label": "exact"}), finish_reason=finish))

    one_blocked = await GeminiPointwise(
        settings, settings.baseline, LABELS, name="g", http_client=_client(block_one)
    ).rerank(search)
    assert one_blocked.outcome == Outcome.OK  # one blocked candidate goes last; the search still counts
    assert one_blocked.blocked_calls == 1
    assert one_blocked.order[-1] == "c3"
    assert one_blocked.missing_ids == 1

    def block_all(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_completion(None, finish_reason="content_filter"))

    all_blocked = await GeminiPointwise(
        settings, settings.baseline, LABELS, name="g", http_client=_client(block_all)
    ).rerank(search)
    assert all_blocked.outcome == Outcome.REFUSED


# ---------- Jev ----------


def _jev_response(answers: Mapping[str, object]) -> dict[str, object]:
    return {"model": "jev-1.13.0", "usage": {"input_tokens": 3000, "output_tokens": 50}, "answers": answers}


def _score(probabilities: dict[int, float]) -> dict[str, object]:
    return {
        "type": "score",
        "score": sum(k * v for k, v in probabilities.items()),
        "confidence": 0.9,
        "legend": {"0": "irrelevant", "1": "complement", "2": "substitute", "3": "exact"},
        "probabilities": {str(k): v for k, v in probabilities.items()},
    }


async def test_jev_score_asks_one_score_per_candidate(settings: Settings, search: Search) -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(json.loads(request.content))
        return httpx2.Response(
            200,
            json=_jev_response(
                {
                    "c1": _score({3: 0.2, 2: 0.8}),
                    "c2": _score({3: 0.9, 2: 0.1}),
                    "c3": _score({0: 1.0}),
                }
            ),
        )

    reranker = JevScore(settings, LABELS, name="jev-score", transport=httpx2.MockTransport(handler))
    ranking = await reranker.rerank(search)
    await reranker.aclose()

    assert seen["state"] == {
        "query": "sesame chips",
        "candidates": {
            f"c{i}": {"title": f"Product {i} {label}", "brand": "Brand"} for i, label in enumerate("ESCI", 1)
        },
    }
    question = seen["questions"]["c2"]  # pyright: ignore[reportIndexIssue]
    assert question["type"] == "score"
    assert [level["means"] for level in question["criteria"]] == [
        LABELS.level(name).means for name in ("irrelevant", "complement", "substitute", "exact")
    ]
    assert "candidates.c2" in question["instructions"]
    assert ranking.order == ["c2", "c1", "c3", "c4"]
    assert ranking.p_exact == {"c1": 0.2, "c2": 0.9, "c3": 0.0}
    assert ranking.labels == {"c1": "S", "c2": "E", "c3": "I"}
    assert ranking.exact_confidence == 0.9
    assert ranking.missing_ids == 1
    assert (ranking.input_tokens, ranking.reasoning_tokens) == (3000, 0)


async def test_jev_noul_batch_and_pair(settings: Settings, search: Search) -> None:
    batch_answers = {"c1": {"type": "noul", "noul": 0.3}, "c2": {"type": "noul", "noul": 0.8}}
    batch = JevNoulBatch(
        settings,
        LABELS,
        name="jev-noul-batch",
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json=_jev_response(batch_answers))),
    )
    ranking = await batch.rerank(search)
    assert ranking.order[:2] == ["c2", "c1"]
    assert ranking.exact_confidence == 0.8

    states: list[dict[str, object]] = []

    def pair_handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        states.append(body["state"])
        noul = 0.95 if body["state"]["product"]["title"] == "Product 4 I" else 0.1
        return httpx2.Response(200, json=_jev_response({"exact": {"type": "noul", "noul": noul}}))

    pair = JevNoulPair(settings, LABELS, name="jev-noul-pair", transport=httpx2.MockTransport(pair_handler))
    paired = await pair.rerank(search)
    assert len(states) == 4 and all(set(s) == {"query", "product"} for s in states)
    assert paired.order[0] == "c4"
    assert paired.calls == 4
    assert paired.input_tokens == 4 * 3000


async def test_jev_choice_ranks_by_probability_and_reads_none_of_these(settings: Settings, search: Search) -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(json.loads(request.content))
        probabilities = {"c1": 0.1, "c2": 0.2, "c3": 0.05, "c4": 0.05, NONE_OF_THESE: 0.6}
        answer = {"type": "choice", "choice": NONE_OF_THESE, "confidence": 0.6, "probabilities": probabilities}
        return httpx2.Response(200, json=_jev_response({"best": answer}))

    ranking = await JevChoice(settings, LABELS, name="jev-choice", transport=httpx2.MockTransport(handler)).rerank(
        search
    )
    assert list(seen["questions"]["best"]["criteria"]) == ["c1", "c2", "c3", "c4", NONE_OF_THESE]  # pyright: ignore[reportIndexIssue]
    assert ranking.order == ["c2", "c1", "c3", "c4"]
    assert ranking.exact_confidence == pytest.approx(0.4)
    assert ranking.p_exact is None


async def test_jev_failures(settings: Settings, search: Search) -> None:
    def jev(status: int) -> JevScore:
        return JevScore(
            settings,
            LABELS,
            name="j",
            transport=httpx2.MockTransport(lambda _: httpx2.Response(status, json={"error": "x"})),
        )

    assert (await jev(400).rerank(search)).outcome == Outcome.API_ERROR
    with pytest.raises(RetryableError):
        await jev(429).rerank(search)
    empty = JevNoulBatch(
        settings,
        LABELS,
        name="j",
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json=_jev_response({}))),
    )
    assert (await empty.rerank(search)).outcome == Outcome.INVALID_OUTPUT


# ---------- references and registry ----------


async def test_lexical_and_presented_order(search: Search) -> None:
    scores = bm25_scores("sesame chips", {"a": "Sesame Seed Chips", "b": "Trail Mix", "c": "Pita Chips"})
    assert scores["a"] > scores["c"] > scores["b"] == 0.0
    lexical = await Lexical().rerank(make_search("SIE", query="product 3"))
    assert lexical.order[0] == "c3"
    assert lexical.calls == 0
    assert (await PresentedOrder().rerank(search)).order == ["c1", "c2", "c3", "c4"]


def test_every_system_builds_and_has_prices(settings: Settings) -> None:
    for name in SYSTEMS:
        if name == "gemini-flash-listwise":
            with pytest.raises(SystemExit, match="BASELINE_MODEL_2"):
                build_reranker(name, settings, LABELS)
            continue
        reranker = build_reranker(name, settings, LABELS)
        assert reranker.name == name
    with pytest.raises(SystemExit, match="Unknown system"):
        build_reranker("nope", settings, LABELS)
    assert prices_for(settings, "jev-score") == settings.jev_prices
    assert prices_for(settings, "gemini-label") == settings.baseline.prices
    assert prices_for(settings, "gemini-flash-listwise") is None  # not configured in the test settings
    assert all(prices_for(settings, name) is not None for name in NO_MODEL)
    assert prices_for(settings, "nope") is None
    assert (llm_config_key("gemini-flash-listwise"), llm_config_key("gemini-label"), llm_config_key("jev-score")) == (
        "baseline_2",
        "baseline",
        None,
    )
