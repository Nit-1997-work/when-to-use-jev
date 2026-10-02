# Experiment Design: Jev vs Gemini Flash for E-commerce Intent Classification

Status: v1.0.
The published results come from run `pilot-20260929T205311Z`: see [the report](../../results/intent/pilot-20260929T205311Z/report.md) and section 12.

## 1. Question

For e-commerce intent classification, how do TypeSafe's Jev and a typical production setup (a low-latency Gemini Flash-Lite model with structured output) compare on:

1. **Accuracy** - does it pick the right intent?
2. **Latency** - how long does one classification take?
3. **Token cost** - what does one classification cost?

Secondary: how useful is each system's confidence for deciding when not to trust the answer?
This is Jev's main selling point, and it comes almost free from the same runs.

## 2. Dataset

**Primary: Bitext Retail (eCommerce)** - `bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset` on Hugging Face.

| Property | Value |
| ---------- | ------- |
| Size | 44,884 customer utterances |
| Labels | **46 intents** grouped into **13 categories** |
| Categories | ACCOUNT, APP_WEBSITE, CART, CONTACT, DELIVERY, FEEDBACK, ORDER, PAYMENT, PRODUCT, RETURNS, SALES, STORE, USER |
| Balance | 721-1,000 utterances per intent |
| Fields | `instruction` (user text), `intent`, `category`, `tags`, `response` (we ignore `response`) |
| Extras | `tags` codes mark language variation such as colloquial, offensive, and typos, so we can slice results by messiness |
| License | CDLA-Sharing-1.0 (publishable; derived data must keep the same license) |
| Published | 2024 |

Sample intents: `add_product`, `remove_product`, `track_order`, `cancel_order`, `change_order`, `missing_item`, `wrong_item`, `damaged_delivery`, `delivery_time`, `request_refund`, `refund_status`, `return_product_online`, `return_product_in_store`, `availability_online`, `availability_in_store`, `payment_issue`, `human_agent`.

Example rows:

- "I got to add an item to the cart" -> `add_product` (CART)
- "i have to add products to the basket i ned help" -> `add_product` (CART, with typos)

Why this one:

- It is squarely e-commerce: cart, orders, delivery, returns, refunds, payment, and store.
- 46 labels is a realistic production routing size, and it includes close pairs (`return_product` vs `return_product_online` vs `return_product_in_store`) that test fine distinctions.
- It is large, balanced, openly licensed, and widely used, so the results are reproducible for the blog.

Known limitations, which we will state in the blog:

- It is synthetic (generated from templates), so it is cleaner than real traffic.
- It is public, so both models may have seen it during training. This affects both models equally, which is acceptable for a head-to-head comparison, but it inflates the absolute accuracy numbers.

**Optional second dataset: CLINC150 `clinc/clinc_oos`** (CC-BY-3.0, 150 intents + an out-of-scope label, human-written).
Add it only if we want a human-written check or an out-of-scope test.
It is not needed for the core question.

Other candidates considered and rejected: Bitext Customer Support (27 intents, less retail-specific), Banking77 (not e-commerce), MASSIVE (general voice assistant), and the many small 2025-2026 "ecommerce-intent" sets on Hugging Face (unclear provenance, mostly LLM-generated, rarely used).

## 3. Systems compared

| ID | System | How it is called |
| ---- | -------- | ------------------ |
| **`baseline`** | **Gemini Flash-Lite** (`google/gemini-3.5-flash-lite`, a common low-latency choice for LLM intent classifiers) | Chat Completions call through the LiteLLM-based gateway, structured output with an `enum` of the 46 intents, temperature 0, provider-default thinking (`minimal`) and provider-default safety settings |
| `baseline-conf` | Same model | Same call, plus a self-reported `confidence` field in the schema. Needed because the gateway rejects `logprobs` for Gemini (verified: HTTP 400 "Logprobs is not supported for this model"). Used only for the confidence analysis; `baseline` stays the accuracy/latency/cost reference |
| **`jev`** | **Jev** (`jev-1.13.0`, pinned rather than the moving `jev-latest` alias) | One `Choice` question with 46 options, each with a description. The response gives `choice`, `probabilities`, and `confidence` |
| `baseline-2` (optional) | Gemini Flash (`google/gemini-3.7-flash`, a larger general-purpose model) | Same as `baseline`, with `reasoning_effort=low`, the lowest level the gateway accepts for this model. It still reasons: 24-148 reasoning tokens per call in the published run (5th to 95th percentile) |

Both systems get **the same information**:

- The same 46 intent names.
- The same description per intent. There is no official description list, so we write it once and freeze it before the test run.
  Following the TypeSafe skill guidance, each description is a small object rather than a bare string, so the close sibling intents get explicit contrasts:
  `{"means": "Wants to return an item by mail or online", "not": "Returning in a physical store (return_product_in_store)"}`.
  Gemini receives the exact same objects rendered into its prompt, so neither system gets extra help.
- The option names are the Bitext intent names as-is. Jev sees option names as well as descriptions (only question IDs are hidden from the model), so both systems see identical label text.
- The same instruction: "Which intent best describes this customer message?"
- Zero-shot: no training examples. Jev cannot be fine-tuned, so zero-shot is the fair setup.
- A closed label set with no "none of the above" option, because every Bitext message has exactly one gold intent.
  The skill recommends a no-match option when nothing may fit; that matters for real traffic, and is covered by the optional CLINC out-of-scope run, where both systems get an `other` option.

Jev request shape:

```python
# uv add typesafe-sdk   (import name: typesafe_sdk; reads TYPESAFE_API_KEY / TYPESAFE_BASE_URL / TYPESAFE_DEFAULT_MODEL)
from typesafe_sdk import AsyncTypeSafeClient, Choice

async with AsyncTypeSafeClient() as client:  # model pinned via TYPESAFE_DEFAULT_MODEL=jev-1.13.0
    result = await client.system_one(
        state={"customer_message": text},
        questions={
            "intent": Choice(
                instructions="Which intent best describes this customer message?",
                criteria=INTENT_DESCRIPTIONS,  # {"add_product": {"means": ..., "not": ...}, ...}
            )
        },
    )
answer = result.answers["intent"]  # .choice, .probabilities, .confidence
```

Gemini request shape: a Chat Completions call through the LiteLLM-based gateway, with a system message that lists the same 46 intents and descriptions, the customer message as the user turn, and `response_format` set to a JSON schema whose `intent` field is an enum of the 46 intents.

## 4. Data splits

The source has 44,884 messages. About 48% contain profanity (tag `W`) and about 51% contain typos (tag `Z`); see `data/SOURCES.md`.
Real traffic has far less profanity, so profanity gets its own set instead of dominating the headline numbers.

Sampling is per intent with a fixed random seed, and no message appears in more than one set.

| Set | Per intent | Total | Composition | Use |
| --- | ---------- | ----- | ----------- | --- |
| Practice | 5 | 230 | 4 clean + 1 profane (20%) | Tune the 46 intent descriptions. Frozen after this. Never reported. |
| **Main test** | 50 | **2,300** | 45 clean + 5 profane (10%). Typos at their natural rate (~50%). | Headline accuracy, latency, and cost. |
| Profanity test | 10 | 460 | All profane | "What happens when users swear?" Wrong answers and safety-filter blocks are counted separately. |

Every intent has at least 310 clean and 363 profane source messages, so every set can be filled without reuse.
A fourth file, `warmup` (20 unused clean messages), feeds the discarded warm-up calls.
The run order inside each split is shuffled, so time-of-run drift (network, rate limits) cannot line up with specific intents.
A main test of 2,300 detects accuracy differences of about 2 percentage points.
Both systems are cheap, so we can raise it to 100 per intent (4,600) if the confidence intervals are too wide.

## 5. Metrics

### Accuracy

- **Intent accuracy** (46-way) and **macro-F1**.
- **Category accuracy** (13-way): count a prediction correct when its category is right, even if the specific intent is wrong. This separates "wrong area" from "close sibling" mistakes.
- The top confused pairs (expected vs predicted intent).
- Accuracy by typo slice (the `Z` tag); the profanity split covers offensive language.
- **Invalid-output rate**: Gemini responses that fail to parse or fall outside the label set, which Jev cannot produce. Count these as wrong.
- **Blocked messages**: when the provider's safety filter withholds an answer (`finish_reason=content_filter`), the message has no answer.
  Blocked messages are excluded from every quality metric (accuracy, macro-F1, category accuracy, confidence) and reported separately, split into clean and profane messages.
  API errors that survive the runner's retries are handled the same way.
- Statistics: 95% bootstrap confidence intervals, and McNemar's exact test on the messages both systems answered.

### Latency

- Measured on the client as wall-clock time from sending the request to having a parsed answer.
- Report **p50, p95, and p99**, plus the mean.
- Each system sends one request at a time, so queueing does not distort the numbers, after 20 discarded warm-up calls.
  In the published run the four systems ran at the same time from one machine, each still one request at a time.
  Where this could be checked, the effect was small: Gemini Flash's median latency was 2,932 ms while the other systems were running and 2,892 ms after they finished.
- Run from the same machine and network for both systems, and record where that machine is (TypeSafe is hosted on the US West Coast).
- All Gemini calls go through a **LiteLLM-based gateway compatible with OpenAI's Chat Completions API**. There are no direct Gemini API calls. Gemini latency therefore includes the gateway hop, which is how most production LLM classifiers are deployed. The gateway's own timing headers (`x-litellm-response-duration-ms`, `x-litellm-overhead-duration-ms`) are recorded per call.
- The gateway caches identical requests: a repeat returns in about 90 ms with an `x-litellm-cache-key` header. Every request therefore sends LiteLLM's per-request bypass (`{"cache": {"no-cache": true}}`), and any response that still carries a cache key is flagged and excluded from latency.
- The published results come from one run of every message.
  Repeating the run at other times of day would show day-to-day spread; the harness supports repeats, but the report's margins of error assume one run per message.
- Log retries (HTTP 429/529 and similar) separately, so rate limiting does not look like model latency.

### Token cost

- Record actual token usage from each response:
  - Jev: `usage.input_tokens` (output is free).
  - Gemini: `prompt_tokens`, `completion_tokens`, and `completion_tokens_details.reasoning_tokens` from the Chat Completions `usage` object returned by the gateway.
- Cost per call = tokens x list price on the run date.
  - Jev: $0.042 per million input tokens.
  - Gemini, as published for Vertex AI's global endpoint (the one the gateway calls) and identically on the Gemini Developer API price page, read 2026-09-29:
    Flash-Lite costs $0.30 per million input tokens and $2.50 per million output tokens.
    Flash costs $0.75 and $3.75, an introductory price through 2026-12-31 ($1.50 and $7.50 from 2027-01-01).
    Thinking tokens are billed at the output rate.
    Sources: `cloud.google.com/vertex-ai/generative-ai/pricing` and `ai.google.dev/gemini-api/docs/pricing`.
  - Blocked responses are billed (Vertex charges every request that returns HTTP 200), so cost averages over every call, blocked ones included.
- Report **cost per 1,000 classifications** and **cost per 1 million classifications**.
- Also report tokens per call for each system. The 46 intent descriptions dominate the input, so the prompt size is similar for both. The difference comes from the per-token price and from Gemini's output tokens.

### Confidence (secondary)

- Jev: the probability of its chosen intent (the top `probabilities` value).
  Its `confidence` field tracks it closely (correlation 0.999 in the published run).
- Gemini: the gateway does not support `logprobs` for these models, so Gemini's confidence is **self-reported** (`baseline-conf`). The blog must label it that way; it is a weaker signal than a native probability.
- Jev's `confidence` measures how concentrated the probabilities are, not whether the answer is correct.
  Between close siblings (for example the three return intents), probability can legitimately split, so we also report accuracy on the top-2 options and look at what the low-confidence cases actually are.
- Report:
  - **Calibration**: expected calibration error (ECE, 15 equal-width bins). This shows whether "80% confident" means "right 80% of the time".
  - **Accuracy vs coverage**: if we only act on answers above a confidence threshold, what accuracy do we get and what share of messages do we act on? For example: "Jev acts on 85% of messages at 98% accuracy; Gemini acts on 70%."
    Coverage is a share of answered messages.
    Thresholds fall only between groups of equal confidence: answers that all say 0.95 cannot be split into acted-on and not.

## 6. Measurement tooling

Use **Arize Phoenix** (open source, runs locally; `arize-phoenix-otel` with OpenInference instrumentation) for traces, dashboards, and side-by-side browsing.
The harness itself drives every call and writes one JSONL record per classification; those records are the source of truth, and `jevbench intent report` computes the statistics from them.

| Need | Tool | Hand-written? |
| ---- | ---- | ------------- |
| Per-call latency, tokens (incl. reasoning tokens), errors | OpenInference spans, one Phoenix project per system (`jevbench-<system>`). Gemini via `openinference-instrumentation-openai` on the OpenAI client pointed at the LiteLLM-based gateway (automatic). Jev via one manual `LLM` span per call with `llm.model_name`, `llm.provider`, and `llm.token_count.*`. Each call sits under a `classify` span tagged with example id, split, repeat, and correctness; warm-up calls sit under a `warmup` span | Jev span only |
| Cost in USD | Computed by `jevbench intent report` from recorded tokens and the prices recorded in each run's manifests. `make phoenix` registers Jev's price for Phoenix cost dashboards; the gateway's `google/gemini-*` names must be added in Settings > Models | Config only |
| The test set | `jevbench intent phoenix-sync` uploads each split as a Phoenix **dataset** (content-addressed name, so a changed split becomes a new dataset) | No |
| Side-by-side comparison | `jevbench intent phoenix-sync` logs each (system, split, repeat) as a Phoenix **experiment** from the JSONL records, with `intent_correct` and `category_correct` evaluations, linked to the run's traces | No |
| Accuracy, macro-F1, CIs, McNemar, latency percentiles, cost, ECE, coverage, slices, confusions | `jevbench intent report` -> `results/intent/<run_id>/report.md` | Yes (tested offline) |

Why the harness drives the calls instead of Phoenix's `run_experiment`: it keeps the timed path free of experiment bookkeeping, makes runs resumable after an interruption, and keeps one local, auditable record per call even when Phoenix is not running.

Rules that keep the numbers honest:

- Register Phoenix with `batch=True`, so exporting spans never adds to the measured latency (the default synchronous exporter sends spans on the request path).
- Latency is timed around exactly one HTTP attempt to a parsed answer. SDK retries are off; the runner's retries and backoff time are recorded separately.
- Keep our own recorded prices and token counts in the run output, so the blog does not depend on Phoenix's pricing table for Gemini 3.x names.
- Compute the report only from the committed records and manifests: `jevbench intent report` needs no API keys, and CI checks that the committed report matches the records.
- Refuse full runs from uncommitted code, and refuse to resume a run whose configuration changed.
- A local Phoenix in Docker means readers can reproduce the blog with `docker run` and no account.

Alternatives considered:

- **Langfuse** also has datasets, experiments, and custom model pricing. It is a fine choice, but a hosted or shared instance is harder for blog readers to reproduce than a local Phoenix.
- **Promptfoo** reports pass/fail, latency, and cost well, but it treats each output as pass/fail text and has no place for Jev's probability distribution, which the calibration analysis needs.

## 7. Controls

- Temperature 0 for Gemini. Jev has no sampling setting.
- Pin both model versions and log the model ID returned with every response.
  Through the gateway, Gemini's returned model ID is the gateway alias (for example `google/gemini-3.5-flash-lite`), not a dated version.
- Keep the provider's default safety settings (the harness sends none).
- Save every request and response to JSONL (text, prediction, probabilities, tokens, latency, timestamp), so every number in the blog can be recomputed.
- Do not change descriptions or prompts after looking at test results.

## 8. What "good" looks like

Jev is a strong replacement candidate for an LLM intent classifier if it:

- Is **within 2 percentage points** of Gemini Flash-Lite on intent accuracy (inside the confidence interval), **and**
- Has **p95 latency at least 5x lower**, **and**
- Costs **at least 10x less** per 1,000 classifications.

A strong result adds: better calibration, meaning more messages handled automatically at the same accuracy.

If Jev is clearly less accurate, the blog still has a finding: where the gap is (close sibling intents, typos, or specific categories) and whether confidence thresholds can recover it.

## 9. Deliverables

```text
when-to-use-jev/
  data/intent/intents.yaml        # the 46 intent descriptions both systems see (frozen after practice tuning)
  data/intent/splits/*.jsonl      # practice / main / profanity / warmup (built by `jevbench intent prepare`)
  src/jevbench/common/            # shared by every experiment: settings, gateway and Jev calls, runner, stats, audit
  src/jevbench/intent/data.py     # deterministic split builder
  src/jevbench/intent/classifiers/ # jev.py (Choice call) and llm.py (gateway, structured output), same Prediction shape
  src/jevbench/intent/runner.py   # sequential, resumable runs -> results/intent/<run>/<system>/<split>/repeat-<n>.jsonl
  src/jevbench/intent/metrics.py  # accuracy, macro-F1, top-2, ECE, coverage, confusions
  src/jevbench/intent/report.py   # results/intent/<run>/report.md
  src/jevbench/intent/phoenix_sync.py # datasets + experiments in Phoenix
  src/jevbench/cli.py             # `jevbench intent prepare | systems | check | run | report | phoenix-sync`, `jevbench audit`
  results/intent/<run>/           # published runs: report.md committed; records and manifests kept locally
  scripts/run_matrix.sh           # systems x splits x repeats under one run id
  tests/                          # offline unit tests
  docs/intent/experiment-design.md
```

Stack: Python 3.12 with `uv`, `pandas`, `typesafe-sdk`, `openai` (the client for the LiteLLM-based, Chat Completions-compatible gateway; used for all LLM calls), `arize-phoenix-otel`, `arize-phoenix-client`, `openinference-instrumentation-openai`, `scipy`, and `truststore`. Phoenix itself runs in Docker (`arizephoenix/phoenix:version-20.16.0`).

Cost of one full run (every system, main and profanity splits): about $10.
Gemini Flash is most of it ($5.47); Jev's share was $0.26.

## 10. Steps

1. Done: Jev and gateway keys validated (`make intent-check`), splits built (`make intent-prepare`), harness smoke-tested (`make intent-smoke`).
2. Done: `data/intent/intents.yaml` frozen (its SHA-256, `d71d9547...`, is recorded in every run manifest).
3. Done: Gemini prices set in `.env` and recorded in every run manifest.
4. Done: full run `pilot-20260929T205311Z` (every system x main + profanity, one run).
5. Write up from `results/intent/pilot-20260929T205311Z/report.md`.

## 11. Decisions and open items

- Decided: the run keeps the provider's default safety settings, because that is what a production classifier gets.
  The blocks come from the provider's configurable safety filter.
  Blocked messages are excluded from quality metrics and reported separately; in the published run, Flash-Lite blocked 3.7% of clean and 71.3% of profane main-split messages.
- Decided: one run of every message (no repeats); repeats of the same messages would make the report's margins of error too narrow.
- Open: cite the source and date for Jev's $0.042 per million input tokens (set `JEV_PRICE_SOURCE_URL` for future runs).
- Open: check TypeSafe's Master Customer Agreement (`typesafe.ai/legal/mca`) for rules on publishing benchmark results before the blog goes out.

## 12. Published run

Run `pilot-20260929T205311Z`, 2026-09-29 from 20:53 to 23:17 UTC; full results in [the report](../../results/intent/pilot-20260929T205311Z/report.md).

- Every system classified all 2,300 main and 460 profanity messages once, from one client machine on home Wi-Fi.
- Each system sent one request at a time, but the four systems ran at the same time (see section 5, Latency).
- Prompt size is nearly identical: Jev used 2,235 input tokens per call and Gemini 2,281.
- Jev also reports about 400 output tokens per call, which are free.
- The run's manifests record commit `383854f` with uncommitted changes, because the harness was not yet committed when it ran.
  The later fixes in this repository change only situations the run never hit (it had no API errors, rate limits, HTTP-error blocks, invalid outputs, crashes, or resumes), plus dropping a duplicate field from new records, so no number from this run depends on them.
  The report was rebuilt from the records with the final code, and CI checks that it still matches them.
