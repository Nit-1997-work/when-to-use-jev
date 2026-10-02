# Experiment Design: Jev vs Gemini Flash-Lite for Product Search Re-ranking

Status: v0.2, harness implemented and smoke-tested; practice and full runs not yet done.
This is Experiment 2. Experiment 1 (intent classification) is described in [../intent/experiment-design.md](../intent/experiment-design.md).

## 1. Question

A product search engine returns a shortlist of about 10 to 40 candidates for a query.
A re-ranker reorders that shortlist so the best match comes first, and says when nothing on the list really matches.
For this job, how does TypeSafe's Jev compare with a typical production setup (Gemini Flash-Lite ranking the whole list in one structured-output call) on:

1. **Ranking quality** - is the best product first, and are the top few good?
2. **Latency** - how long does it take to re-rank one shortlist?
3. **Cost** - what does it cost to re-rank one shortlist?

Secondary questions:

- **Abstention** - how well does each system detect that no candidate is an exact match?
- **Consistency** - does the same shortlist get the same ranking on a repeat run, and when the candidate order is reversed?
- **Approach vs model** - how much of any difference comes from Jev itself, and how much from scoring each candidate separately instead of ranking the whole list at once?

## 2. Background: a typical LLM re-ranking setup

Product search in shopping apps and shopping assistants is often built in two steps:

1. **Retrieval:** a fast search engine (keyword, embedding, or both) returns a shortlist of a few dozen candidate products for a query.
2. **Re-ranking:** a small, fast LLM sees the query and the whole shortlist in one structured-output call.
   It returns the candidate ids in order, or a flag plus a short explanation when nothing on the list really matches.

The application usually acts on the first result, for example adding it to a cart, and shows the next few as alternatives.
So top-1 matters most, then the top 3; the order further down hardly matters.

This experiment tests step 2 only, on public data.

## 3. Scope

In scope:

- Re-ranking a fixed shortlist of real product candidates for a short search query.
- Detecting that no candidate is an exact match.
- Quality, latency, cost, and consistency of each system.

Out of scope for this experiment:

- Query rewriting, query expansion, and category tagging.
- First-stage retrieval: every system re-ranks the same shortlist.
- Personalization (purchase history, preferences).
- Conversation or session context beyond the query itself.
- Price-aware ranking (the public data has no prices, and price comparisons belong in code anyway).
- Writing the user-facing "couldn't find it" message (generation stays with an LLM; see the hybrid note in section 4).

## 4. Can Jev do this job?

Re-ranking is a set of short judgments: "how well does this product match this query?"
That is the kind of narrow, structured decision Jev is built for, and TypeSafe publishes a [re-ranking cookbook](https://docs.typesafe.ai/cookbooks/rerank_typesafe.md) for legal passages.
That cookbook is a different domain, so it is a reason to test, not evidence of the outcome.

Jev's documented weak spots that matter here ([jev-1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md)):

| Weak spot | Where it bites in re-ranking | Mitigation in this design |
| --- | --- | --- |
| Literal reading | "pineapple" vs pineapple juice; "gluten-free bread" vs regular bread | Write the exact/substitute boundary into the criteria, with grocery-style examples |
| Indirection | A question that names one candidate must find it in a long list | Candidates are keyed by short id (`candidates.c7`), and a one-candidate-per-request variant is tested too |
| Large, irrelevant state | Long product descriptions | Send title, brand, and color only; descriptions are a later ablation |
| Numbers | "under $5", "family size" | Out of scope; keep in code in production |
| Generation | The friendly "couldn't find white rum" message | Not measured; a production hybrid would call an LLM only when Jev abstains |

## 5. Dataset

**Primary: Amazon Shopping Queries Dataset (ESCI)**, from the KDD Cup 2022 ([`amazon-science/esci-data`](https://github.com/amazon-science/esci-data), pinned at commit `7916cdf`; see `data/SOURCES.md`).

| Property | Value |
| --- | --- |
| Content | Real, difficult Amazon search queries, each with a list of candidate products |
| Labels | Every query-product pair is human-labeled **E**xact, **S**ubstitute, **C**omplement, or **I**rrelevant |
| Version used | The reduced "Task 1" version (`small_version == 1`), which drops queries Amazon judged easy |
| Size (English, reduced) | 29,844 queries and 601,354 judgments; 20,888 train and 8,956 test queries; median 16 candidates per query (mean 20, range 8 to 188) |
| Label shares (English, reduced) | Exact 43.5%, Substitute 35.1%, Irrelevant 16.9%, Complement 4.5% |
| Product fields | `product_title`, `product_brand`, `product_color`, `product_description`, `product_bullet_point` |
| Extras | `shopping_queries_dataset_sources.csv` gives a `source` per query (`other`, `parse_pattern`, `negations`, `behavioral`, `nlqec`) |
| License | Apache-2.0 (the repository's LICENSE, saved as `data/rerank/raw/LICENSE`) |

Why this one:

- The task is exactly the re-ranking step: a query, a real shortlist, and a re-ranker in between.
- The labels match the decisions a product re-ranker has to make:
  - Exact vs Substitute is "gluten-free bread" vs plain bread, or shredded vs block cheddar.
  - Complement is "rum extract when you asked for rum", or coffee pods for a coffee machine.
  - A shortlist with no Exact product is the "the catalog doesn't carry it" case.
- Graded labels support graded metrics (NDCG), not just "right or wrong".
- It is large, human-labeled, openly licensed, and widely used, so results are reproducible.

Known limitations, which the write-up will state:

- It is Amazon's general catalog, not a grocery catalog.
  There is no category field, so the grocery slice (section 6) comes from a keyword filter plus a review of each search, not from a catalog taxonomy.
- Queries come without user or session context, which many real re-rankers also see.
- There are no prices and no original search-engine order, so the "no re-ranking" baselines are a lexical ranking and the seeded random order (section 7).
- Only 1 of the 29,844 reduced English searches has no Exact candidate, so every "nothing matches" search is constructed (section 6).
- The labels are noisy: nearly identical products sometimes carry different labels.
  The noise applies to every system equally, so it lowers every score without favoring one system.
- It is public, so both models may have seen it during training.
  This affects every system equally, which is acceptable for a head-to-head comparison, but it can inflate absolute numbers.

**Optional second dataset: WANDS** (Wayfair, about 480 queries, labels Exact / Partial / Irrelevant).
Add it only to check that the conclusion holds in a second domain.

Considered and not chosen:

- TREC Product Search 2023: deeper labels, but heavier setup and overlaps ESCI.
- Home Depot Product Search Relevance (Kaggle): competition rules restrict reuse.

## 6. Data splits

`jevbench rerank prepare` builds every split deterministically (fixed seed `20261001`) from the pinned ESCI files, after checking their SHA-256.
All splits use English queries from the reduced version with **8 to 40 candidates**, which drops 2.2% of test searches (198 of 8,956) and keeps every Jev Choice question far below its 255-option limit.
Sampling is by search, and no search appears in more than one split (the `reversed` split repeats `consistency` on purpose).
The practice, practice-nomatch, and warm-up splits come from ESCI's `train` split; every reported split comes from ESCI's `test` split.

| Split | Searches | Built from | Use |
| --- | --- | --- | --- |
| `practice` | 100 | ESCI train | Tune label definitions, prompts, and Jev criteria. Never reported as a result. |
| `practice-nomatch` | 50 | ESCI train, Exact candidates removed | Tune each Jev system's abstention threshold. |
| **`main`** | **1,000** | ESCI test: 171 reviewed grocery searches + 829 random searches | Headline ranking quality, latency, and cost; false abstentions. |
| `nomatch` | 300 | ESCI test, Exact candidates removed | Abstention: shortlists where nothing is an Exact match. |
| `consistency` | 100 | ESCI test | Run 3 times: does the same shortlist get the same ranking? |
| `reversed` | 100 | The same 100 searches, candidate order reversed | Run once: does the top product depend on candidate order? |
| `warmup` | 20 | ESCI train | Discarded warm-up searches. |

**Candidates:** every labeled candidate for the search, in a seeded per-search shuffle, with short ids `c1`...`cN` in presented order.
Every system sees the same candidates in the same order, under the same ids.
ESCI has no original search order, and a random order keeps systems from inheriting a useful hint.
In `reversed`, ids are reassigned in the new order, so the ids themselves give no hint either.
Product text is the title (capped at 300 characters; the 99th percentile is 200), brand, and color.

**Grocery slice:** a candidate search either has a grocery keyword in its query or has mostly food-like candidate titles with pack sizes (both lists frozen in `data/rerank/grocery_terms.yaml`).
That gives 499 candidate test searches; each one was judged grocery or not from its query and first Exact product, and the decisions are committed in `data/rerank/grocery_review.yaml`.
171 of the 499 are grocery (food or drink for people; not pills, pet food, kitchenware, books, or products that borrow food words).
The decisions were proposed by the coding agent and are pending the authors' review; changing one changes `main`.
Keyword matches that are not grocery (for example "oil filter", "coffee table") are kept out of the random part of `main` too, so they cannot dilute the general slice.
`main` keeps its 1,000 searches: 300 grocery searches were planned, and random searches fill the 129 that do not exist.

**Hard slice:** searches with 3 or fewer Exact candidates (218 of the 1,000 in `main`).
About 42% of `main` candidates are Exact, so most searches have many acceptable answers; the hard slice is where systems differ most.

**No-match searches:** other test searches with every Exact candidate removed, leaving Substitutes, Complements, and Irrelevants, with at least 5 candidates left.
This mirrors the common case where the catalog does not carry the exact item but carries near alternatives.
Every search in `nomatch` and `practice-nomatch` should be abstained on; every search in `main` should not, so `main` also measures false abstentions.
No-match searches are scored on abstention only, never on NDCG (with no Exact candidate, the best possible top 3 can be all Irrelevant, which leaves NDCG undefined).

**Size:** 1,000 `main` searches give a 95% confidence interval on mean NDCG@3 of roughly ±0.02, and on Exact@1 of about ±3 percentage points.
The practice run measures the actual spread; `main` grows to 2,000 searches if the intervals are too wide.

## 7. Systems compared

Every system gets the same query, the same candidates in the same order under the same short ids, and the same product fields: title, brand, and color.
Every system returns a full ordering of the candidate ids.
Code completes every ordering the same way: invented and duplicate ids are dropped and counted, candidates a system left out are appended in presented order, and ties are broken by presented order.
An answer that cannot be used at all counts as answered (`invalid_output`) and is ranked in presented order.

### Gemini systems (through the LiteLLM-based gateway)

| ID | Model | Design |
| --- | --- | --- |
| **`gemini-listwise`** | Gemini Flash-Lite (`BASELINE_MODEL`) | **The typical production setup.** One call per search with every candidate. Structured output: `ranked_ids` (an array of this search's ids), `no_relevant_results`, `explanation`. When nothing is an exact match, the model sets the flag and explains, and still returns the full ranking, so ranking quality and abstention are both measured. |
| `gemini-flash-listwise` | Gemini Flash (`BASELINE_MODEL_2`) | The same call on the larger model, with the lowest reasoning effort the gateway accepts. A quality ceiling for the LLM approach. |
| `gemini-label` | Gemini Flash-Lite | One call per search that labels every candidate exact / substitute / complement / irrelevant (one required field per id). Ranked by label. The same task as `jev-score`, so it isolates the model from the approach. |
| `gemini-pointwise` | Gemini Flash-Lite | One call per candidate, all candidates of a search in parallel, one label each. Mirrors `jev-noul-pair`. |

All Gemini calls use temperature 0, the provider's default thinking level unless stated, and the provider's default safety settings.
The label systems abstain when no candidate is labeled exact.

### Jev systems

| ID | Request | Ranking | Abstains when |
| --- | --- | --- | --- |
| **`jev-score`** | One request per search. State: the query and the candidates keyed by id. One `Score` question per candidate, with criteria ordered from the lowest level (index 0, irrelevant) to the highest (index 3, exact). | Expected gain under the level probabilities (section 8) | The highest P(exact) is below its threshold |
| **`jev-noul-batch`** | One request per search, same state. One `Noul` per candidate: "is `candidates.cK` an exact match for `query`?", with match / no-match criteria. | Probability of yes | The highest probability is below its threshold |
| `jev-noul-pair` | One request per candidate, all in parallel. State: the query and one product. One `Noul`, same criteria. TypeSafe's cookbook design; no list lookup. | Probability of yes | Same |
| `jev-choice` | One request per search. One `Choice` whose options are the candidate ids (described by their product text) plus `none_of_these`. | Option probability | 1 - P(`none_of_these`) is below its threshold |

`jev-score` and `jev-noul-batch` are the main Jev designs.
`jev-noul-pair` and `jev-choice` go into the full run only if the practice run shows them competitive, or if their cost and latency tell a different story.
Jev is pinned to one version (`TYPESAFE_DEFAULT_MODEL`), and the returned model id is logged with every response.

The main Jev request, as built in `src/jevbench/rerank/systems/jev.py`:

```python
from typesafe_sdk import Score

state = {"query": search.query, "candidates": {"c1": {"title": ..., "brand": ..., "color": ...}, ...}}
questions = {
    cid: Score(
        instructions=f"{labels.instruction} How well does the product `candidates.{cid}` match `query`?",
        criteria=[irrelevant, complement, substitute, exact],  # each {"means": ..., "not": ..., "example": ...}
    )
    for cid in state["candidates"]
}
response = await client.system_one(state, questions)
# response.answers[cid].probabilities -> {0: p_irrelevant, 1: p_complement, 2: p_substitute, 3: p_exact}
```

### Reference systems (no model call)

| ID | Design |
| --- | --- |
| `lexical` | BM25 over candidate title and brand within each shortlist. The "fast search only" baseline. |
| `random` | The presented order, a seeded per-search shuffle. The floor. |

An open-source cross-encoder re-ranker (for example `bge-reranker-v2-m3`) is an optional quality reference only.
It runs on different hardware, so its latency and cost would not be comparable.

### Fairness rules

- Every system uses the same label definitions, written once in `data/rerank/labels.yaml` and frozen after the practice run.
  The definitions follow ESCI's labels, rewritten for a shopper's search with grocery examples.
  Gemini sees them as JSON in its prompt; Jev sees the same objects as Score criteria, and their exact-match wording as Noul criteria.
- Zero-shot only: no labeled examples in prompts. Jev cannot be fine-tuned, so zero-shot is the fair setup.
- Prompts, questions, and thresholds change only on the practice splits.
- Abstention thresholds are chosen on `practice` + `practice-nomatch` to maximize abstention F1 (`jevbench rerank tune-thresholds`), frozen in `data/rerank/thresholds.yaml`, and applied by the report; the report records the file's SHA-256.

## 8. Metrics

### Ranking quality (`main`)

ESCI labels become gains with the KDD Cup 2022 Task 1 values (Reddy et al., 2022, section 3.1): Exact 1.0, Substitute 0.1, Complement 0.01, Irrelevant 0.
Each step is ten times the one below, so the ordering of labels nearly always dominates: any Exact in the top 3 outweighs three Substitutes there.

| Metric | Why it matters in a shopping app |
| --- | --- |
| **NDCG@3 (headline)** | The product the app acts on and the few alternatives shown next to it |
| Exact@1: the top product is an Exact match | The product the app acts on; always shown next to `random`, which already scores about 42% on `main` because 42.5% of its candidates are Exact |
| Usable@1: the top product is Exact or Substitute | Whether the top product is at least acceptable |
| MRR of the first Exact product | How far down the first right answer is |
| NDCG@10 | Overall list quality; comparable in spirit with published ESCI results, which use the full list |
| NDCG@3 with gentler gains (Exact 1, Substitute 0.5, Complement 0.1, Irrelevant 0) | Whether the winner depends on how much a Substitute is worth |

NDCG@k divides the ranking's discounted gain over the top k (position weights 1, 0.63, 0.5, ...) by the best possible for the same search, so every search scores between 0 and 1.
It is deterministic and computed from the human labels alone; no model judges the rankings.
Reported overall and for the general, grocery, hard, and negations slices.
Statistics: 95% bootstrap confidence intervals over searches; against `gemini-listwise`, a paired permutation test for the NDCG@3 difference and McNemar's exact test for Exact@1.

### Label quality (systems that label candidates)

For `jev-score` (the most likely level), `gemini-label`, and `gemini-pointwise`: 4-way label accuracy, macro-F1, how often Exact is labeled Substitute and Substitute labeled Exact, and the confusion matrix.

### Abstention (`main` + `nomatch`)

- Precision, recall, and F1 of abstaining.
- False-abstention rate on `main` searches, which all have an Exact candidate.

### Latency

- Wall-clock time to re-rank one search, from sending the request to having a parsed ordering.
  For per-candidate designs this is the whole parallel fan-out.
- Report p50, p95, p99, and mean per search, and HTTP calls per search.
- Searches are processed one at a time per system, after discarded warm-up searches, in the split's shuffled order, from one machine whose location is recorded.
- Same rules as Experiment 1: one clean attempt timed, SDK retries off, runner retries and backoff recorded separately, gateway cache bypassed and cache hits flagged and excluded, gateway timing headers recorded.
  If any call in a fan-out hits a retryable error, the whole search is retried, so a timed latency is always one clean attempt of the full fan-out.

### Cost

- Token usage from every response, summed over a search's calls, including Gemini reasoning tokens.
- Cost per search = tokens x list price on the run date, recorded in the run manifest.
- Report cost per 1,000 and per 1 million searches, and tokens per search.
- Blocked responses are billed and included in cost.

Estimate before the smoke run, from Experiment 1's prices and about 20 candidates of about 40 tokens each:

| System | Cost per search | x 1,700 searches |
| --- | --- | --- |
| `gemini-listwise` | about $0.0006 | about $1.00 |
| `gemini-label` | about $0.0009 | about $1.40 |
| `gemini-pointwise` | about $0.0026 | about $4.20 |
| `gemini-flash-listwise` | about $0.0025 | about $4.00 |
| `jev-score`, `jev-noul-batch` | $0.0002 to $0.0008 each | $0.30 to $1.30 each |
| `jev-noul-pair` | about $0.00025 | about $0.40 |
| `jev-choice` | about $0.00006 | about $0.10 |

One full run is about $12-15; with practice tuning and smoke runs, budget about $50.
The smoke run replaces these estimates with measured token counts (section 12).

### Consistency (`consistency` + `reversed`)

- Three repeat runs: top-1 agreement and Kendall's tau between runs.
- One run with the candidate order reversed: how often the top product is the same as in repeat 1.

### Calibration (secondary)

- For `jev-score`, `jev-noul-batch`, and `jev-noul-pair`: expected calibration error of each candidate's P(exact) against the Exact label.
- Gemini gives no native probabilities through the gateway, and Choice probabilities share one unit across candidates, so neither is part of the calibration analysis.

### Cascade

Computed from existing records with no extra calls: Jev keeps the searches where the gap between its top two scores is largest, and `gemini-listwise` re-ranks the rest.
Reported as NDCG@3 and cost per 1,000 searches at 0%, 10%, 25%, 50%, and 100% passed on.

### Operational

- Invalid outputs, invented / duplicate / missing ids, blocked searches, and API errors per system.
  Blocked searches and API errors are excluded from quality metrics and reported separately, as in Experiment 1.

## 9. What "good" looks like

Jev is a strong replacement candidate for an LLM re-ranker if `jev-score` or `jev-noul-batch`, compared with `gemini-listwise`:

- Is within 0.02 on NDCG@3 and within 2 percentage points on Exact@1 (or better), **and**
- Has p95 latency at least 3x lower, **and**
- Costs at least 5x less per 1,000 searches.

A strong result adds:

- Higher abstention F1, or the same abstention recall with fewer false abstentions.
- Higher repeat and order consistency.

If Jev is less accurate, the write-up still answers useful questions:

- Where the gap is: Exact vs Substitute, the grocery or hard slices, or negations.
- Whether the gap is the model or the approach (`jev-score` vs `gemini-label`, `jev-noul-pair` vs `gemini-pointwise`).
- Whether the cascade recovers it, and at what cost.

## 10. Reading the results for a real application

| Experiment result | What it means for a shopping app or assistant |
| --- | --- |
| NDCG@3 | Quality of the product the app acts on and the alternatives shown next to it |
| Exact@1 | Quality of the product the app acts on |
| Abstention F1 | How often "couldn't find X" is shown when, and only when, it should be |
| Latency per search | The re-ranking step's share of search latency; several queries often run in parallel, so p95 matters |
| Cost per search | Re-ranking cost per search query |
| Order consistency | Whether the same search gives the same top product |

What this experiment cannot tell us, and the follow-up that would:

- Effect of user context and personalization: a follow-up arm with user or session context.
- Domain-specific naming and catalog quirks: a validation pass on a labeled sample from the target catalog.
- The abstention message: a hybrid with an LLM writing the message only when Jev abstains, measured on latency and cost.

## 11. Harness

Experiment 2 is the `jevbench.rerank` module, built on the shared `jevbench.common` core it shares with Experiment 1: resumable runs, one JSONL record per search, run manifests with model ids, prices, git commit, and file hashes, tracing, the pre-publish audit, and a report computed only from records.

```text
data/rerank/labels.yaml            # E / S / C / I definitions every system sees (frozen after practice)
data/rerank/grocery_terms.yaml     # the frozen keyword lists that pick grocery candidates
data/rerank/grocery_review.yaml    # one grocery decision per candidate search
data/rerank/thresholds.yaml        # abstention thresholds, written by tune-thresholds from a practice run
data/rerank/splits/*.jsonl         # one search per line: query, slices, candidates with gold labels
data/rerank/raw/                   # the pinned ESCI files (gitignored) and their LICENSE
src/jevbench/rerank/data.py        # deterministic split builder (checks the ESCI files' SHA-256)
src/jevbench/rerank/labels.py      # loads labels.yaml; the KDD gains
src/jevbench/rerank/ranking.py     # the Ranking every system returns; order completion; parallel fan-out
src/jevbench/rerank/systems/       # gemini.py, jev.py, baselines.py (BM25, presented order)
src/jevbench/rerank/runner.py      # one record per search -> results/rerank/<run>/<system>/<split>/repeat-<n>.jsonl
src/jevbench/rerank/metrics.py     # NDCG, Exact@1, MRR, abstention, labels, consistency, calibration, cascade
src/jevbench/rerank/report.py      # results/rerank/<run>/report.md
src/jevbench/rerank/cli.py         # jevbench rerank prepare | systems | check | run | report | tune-thresholds
```

One record is one search: the search's gold labels and candidate ids in presented order, the returned ordering, per-candidate scores, labels, and P(exact) where the system gives them, the abstain flag or confidence, invented / duplicate / missing ids, HTTP calls, summed tokens, latency, attempts, backoff, and model id.

Make targets: `rerank-grocery-candidates`, `rerank-prepare`, `rerank-check`, `rerank-smoke`, `rerank-practice`, `rerank-tune`, `rerank-run`, `rerank-report`.

## 12. Steps

1. Done: ESCI pinned (commit `7916cdf`, SHA-256 of every file in `data/SOURCES.md`); Apache-2.0 license saved with the data.
2. Done: `labels.yaml` and `grocery_terms.yaml` drafted; grocery review proposed (171 of 499); splits built and committed.
3. Done: every system implemented and tested offline (mocked APIs); smoke run over 5 practice searches per system (results in section 14).
4. Done: the authors accepted the grocery review as proposed.
5. Done: practice run `practice-20261001` (every system x 100 `practice` + 50 `practice-nomatch` searches), thresholds tuned and frozen in `data/rerank/thresholds.yaml` (section 15). Label definitions, prompts, and Jev criteria were not changed.
6. Freeze `labels.yaml`, prompts, and `thresholds.yaml`; commit.
7. Full run from committed code (`make rerank-run`), then the report and write-up.

## 13. Decisions and open items

- Decided: query rewriting and expansion are out of scope.
  Writing new search terms is generation, which stays with an LLM.
  Tagging queries with categories is a good Jev candidate for a later experiment (TypeSafe's [hierarchical classification cookbook](https://docs.typesafe.ai/cookbooks/hierarchical_classification.md) covers this shape).
- Decided: product text is title, brand, and color; descriptions and bullet points are a possible ablation, not part of the main run.
- Decided: zero-shot, frozen definitions, thresholds tuned only on practice data and applied by the report.
- Decided: every no-match search is constructed, because ESCI's reduced set has almost none.
- Decided: Gemini listwise always returns the full ranking, even when it flags that nothing matches.
- Open: whether shorter Score criteria keep `jev-score`'s quality while cutting its per-question tokens (practice run).
- Open: the source and date for Jev's price per million input tokens (carried over from Experiment 1).
- Open: whether TypeSafe's Master Customer Agreement allows publishing benchmark results (carried over from Experiment 1).

## 14. Smoke run

Run `smoke-20261001-rerank`, 2026-10-01, from a laptop on home Wi-Fi: every system x 5 `practice` + 5 `practice-nomatch` searches, after 2 warm-up searches each (`make rerank-smoke`).
Before it, `jevbench rerank check` ran every system once on the largest practice search (40 candidates).
Five searches say nothing about quality; the run checks that every system, record, and report section works, and it measures tokens, latency, and cost.

What it showed:

- Every system answered every search: no invalid outputs, invented ids, duplicate ids, or API errors.
- Gemini's strict structured outputs work through the gateway, both the array of ids and the one-required-field-per-id shape.
- Jev accepted 40 questions in one request, and a 40-option Choice.
- The provider's safety filter withholds some per-candidate calls for harmless products: in `check`, a juice drink variety pack was blocked every time it was asked about alone, and the smoke run had 2 blocked calls.
  `gemini-pointwise` therefore ranks a blocked candidate last (counted as missing and as a blocked call) instead of losing the whole search; a search counts as blocked only when every call is.
- **Jev bills input tokens per question, not once per request.**
  The shared state is counted again for every question, so `jev-score` used about 9,300 input tokens per practice search (about 500 per candidate) and about 20,500 on the 40-candidate search, against about 1,850 and 3,330 for `gemini-listwise`.
  `jev-score` is still cheaper, but by about 2.4x, below the 5x bar in section 9; `jev-noul-batch` is about 5.6x cheaper and `jev-choice` about 15x.
  Shorter Score criteria (for example, dropping the examples from each level) are the main cost lever to try in the practice run.

Measured on the 5 practice searches (1 HTTP attempt each; Jev output tokens are free):

| System | Calls per search | p50 latency | Input tokens per search | Cost per 1,000 searches |
|---|---|---|---|---|
| `gemini-listwise` | 1 | 1,152 ms | 1,845 | $0.93 |
| `gemini-flash-listwise` | 1 | 3,172 ms | 1,845 (plus 64 reasoning) | $2.19 |
| `gemini-label` | 1 | 1,191 ms | 2,499 | $1.23 |
| `gemini-pointwise` | 18 | 858 ms | 11,061 | $3.70 |
| `jev-score` | 1 | 165 ms | 9,324 | $0.39 |
| `jev-noul-batch` | 1 | 110 ms | 3,960 | $0.17 |
| `jev-noul-pair` | 18 | 219 ms | 8,484 | $0.36 |
| `jev-choice` | 1 | 133 ms | 1,439 | $0.06 |

At these rates one full run (about 1,700 searches per system) costs about $15, in line with the estimate in section 8.

## 15. Practice run

Run `practice-20261001`, 2026-10-01, from a laptop on home Wi-Fi: every system x 100 `practice` + 50 `practice-nomatch` searches, one search at a time per system, systems one after another.
Practice searches come from ESCI's train split and tune the setup; these numbers are not results.
Two Gemini Flash searches failed on connection errors after every retry and were resumed; 1 to 3 searches per Gemini system were withheld by the provider's safety filter (queries about antibiotics and adult products), and are excluded from quality metrics.

| System | NDCG@3 (95% CI) | Exact@1 | p50 / p95 latency | Cost / 1k searches |
|---|---|---|---|---|
| `gemini-listwise` | 0.750 (0.684-0.809) | 79.0% | 1,072 / 1,537 ms | $1.09 |
| `gemini-flash-listwise` | 0.769 (0.707-0.829) | 78.8% | 2,755 / 8,502 ms | $2.69 |
| `gemini-label` | 0.701 (0.637-0.764) | 73.2% | 1,211 / 1,847 ms | $1.44 |
| `gemini-pointwise` | 0.675 (0.615-0.735) | 70.0% | 908 / 1,615 ms | $4.36 |
| `jev-score` | 0.767 (0.706-0.825) | 81.0% | 169 / 275 ms | $0.46 |
| `jev-noul-batch` | 0.783 (0.720-0.843) | 80.0% | 134 / 196 ms | $0.20 |
| `jev-noul-pair` | 0.771 (0.711-0.830) | 79.0% | 189 / 579 ms | $0.42 |
| `jev-choice` | 0.741 (0.679-0.802) | 76.0% | 116 / 179 ms | $0.07 |
| `lexical` | 0.570 (0.506-0.636) | 56.0% | - | - |
| `random` | 0.426 (0.365-0.488) | 36.0% | - | - |

Abstention thresholds tuned on this run (best F1 on practice + practice-nomatch): `jev-score` 0.785, `jev-noul-batch` 0.565, `jev-noul-pair` 0.59, `jev-choice` 0.795.
Gemini decides with its own flag; its abstention F1 on the same searches was 50% to 55%, against 66% to 68% for the tuned Jev systems (tuned on these same searches, so the Jev numbers are optimistic here).

