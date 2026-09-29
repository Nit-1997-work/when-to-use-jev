# Experiment Design: Jev vs Gemini Flash for E-commerce Intent Classification

Status: draft v0.2.

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
| **G** | **Gemini Flash-Lite** (`google/gemini-3.5-flash-lite`, a common low-latency choice for LLM intent classifiers) | Chat Completions call through the LiteLLM-based gateway, structured output with an `enum` of the 46 intents, temperature 0, thinking off or at the minimum setting |
| **J** | **Jev** (`jev-1.13.0`, pinned rather than the moving `jev-latest` alias) | One `Choice` question with 46 options, each with a one-line description. The response gives `choice`, `probabilities`, and `confidence` |
| G2 (optional) | Gemini Flash (`google/gemini-3.7-flash`, a larger general-purpose model) | Same as G. Shows whether the bigger Gemini changes the picture |

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
| Practice | 5 | 230 | Same mix as the main test | Tune the 46 intent descriptions. Frozen after this. Never reported. |
| **Main test** | 50 | **2,300** | 45 clean + 5 profane (10%). Typos at their natural rate (~50%). | Headline accuracy, latency, and cost. |
| Profanity test | 10 | 460 | All profane | "What happens when users swear?" Wrong answers and safety refusals are counted separately. |

Each profanity-free intent has at least ~360 source messages, so every set can be filled without reuse.
A main test of 2,300 detects accuracy differences of about 2 percentage points.
Both systems are cheap, so we can raise it to 100 per intent (4,600) if the confidence intervals are too wide.

## 5. Metrics

### Accuracy

- **Intent accuracy** (46-way) and **macro-F1**.
- **Category accuracy** (13-way): count a prediction correct when its category is right, even if the specific intent is wrong. This separates "wrong area" from "close sibling" mistakes.
- A confusion matrix, plus the top confused pairs.
- Accuracy by `tags` slice (typos, colloquial, offensive, and so on).
- **Invalid-output rate**: Gemini responses that fail to parse or fall outside the label set, which Jev cannot produce. Count these as wrong.
- Statistics: 95% bootstrap confidence intervals, and McNemar's test on the paired predictions.

### Latency

- Measured on the client as wall-clock time from sending the request to having a parsed answer.
- Report **p50, p95, and p99**, plus the mean.
- Run sequentially (one request at a time) so queueing does not distort the numbers. Send 20 warm-up calls first and discard them.
- Run from the same machine and network for both systems, and record where that machine is (TypeSafe is hosted on the US West Coast).
- All Gemini calls go through a **LiteLLM-based gateway compatible with OpenAI's Chat Completions API**. There are no direct Gemini API calls. Gemini latency therefore includes the gateway hop, which is how most production LLM classifiers are deployed. Record the gateway's own overhead separately if it reports it.
- Repeat the whole test run 3 times, at different times of day. Report the spread.
- Log retries (HTTP 429/529 and similar) separately, so rate limiting does not look like model latency.

### Token cost

- Record actual token usage from each response:
  - Jev: `usage.input_tokens` (output is free).
  - Gemini: `prompt_tokens`, `completion_tokens`, and `completion_tokens_details.reasoning_tokens` from the Chat Completions `usage` object returned by the gateway.
- Cost per call = tokens x list price on the run date.
  - Jev: $0.042 per million input tokens.
  - Gemini: take the current published prices for input, output, and thinking tokens. Record the date and the price sheet URL.
- Report **cost per 1,000 classifications** and **cost per 1 million classifications**.
- Also report tokens per call for each system. The 46 intent descriptions dominate the input, so the prompt size is similar for both. The difference comes from the per-token price and from Gemini's output tokens.

### Confidence (secondary)

- Jev: `confidence` and the top `probabilities` value.
- Gemini: the probability of the chosen label from token `logprobs` if the gateway passes them through. Otherwise, ask for a 0-1 confidence in the schema and label it as "self-reported".
- Jev's `confidence` measures how concentrated the probabilities are, not whether the answer is correct.
  Between close siblings (for example the three return intents), probability can legitimately split, so we also report accuracy on the top-2 options and look at what the low-confidence cases actually are.
- Report:
  - **Calibration**: expected calibration error (ECE) and a reliability plot. This shows whether "80% confident" means "right 80% of the time".
  - **Accuracy vs coverage**: if we only act on answers above a confidence threshold, what accuracy do we get and what share of messages do we act on? For example: "Jev acts on 85% of messages at 98% accuracy; Gemini acts on 70%."

## 6. Measurement tooling

Use **Arize Phoenix** (open source, runs locally; `arize-phoenix-otel` with OpenInference instrumentation) as the system of record, and keep a small analysis notebook for the statistics Phoenix does not compute.

| Need | Tool | Hand-written? |
| ---- | ---- | ------------- |
| Per-call latency, tokens (incl. reasoning tokens), errors | OpenInference spans. Gemini via `openinference-instrumentation-openai` on the OpenAI client pointed at the LiteLLM-based gateway (automatic). Jev via one manual `LLM` span per call, with `llm.model_name`, `llm.provider`, `llm.token_count.*`, and the full `probabilities` as span attributes | Jev span only (~20 lines) |
| Cost in USD | Phoenix cost tracking. Add custom prices in Settings > Models for `jev-1.13.0` (input $0.042/M, output $0) and for the gateway's `google/gemini-*` names if the built-in table does not match them | Config only |
| The test set | A Phoenix **dataset**: input = message, expected output = intent, metadata = category and tags | No |
| Running both systems on it | Phoenix **experiments**: one experiment per system x repeat, with code evaluators `intent_correct` and `category_correct`. The UI compares runs side by side per example | Task functions + 2 tiny evaluators |
| Latency percentiles, cost and token dashboards | Phoenix project dashboards (one project per system) | No |
| p50/p95/p99 per system, bootstrap CIs, McNemar, ECE, reliability plots, accuracy-vs-coverage, slice tables | Notebook that pulls experiment runs and spans from Phoenix (`arize-phoenix-client`) into pandas, then `scikit-learn`, `statsmodels`, `numpy` | Yes, but only analysis, no measurement |

Rules that keep the numbers honest:

- Register Phoenix with `batch=True`, so exporting spans never adds to the measured latency (the default synchronous exporter sends spans on the request path).
- Latency is the duration of the provider-call span, not the whole experiment task, so evaluator and bookkeeping time is excluded.
- Keep our own recorded prices and token counts in the run output too, so the blog does not depend on Phoenix's pricing table for Gemini 3.x names.
- A local Phoenix in Docker means readers can reproduce the blog with `docker run` and no account.

Alternatives considered:

- **Langfuse** also has datasets, experiments, and custom model pricing. It is a fine choice, but a hosted or shared instance is harder for blog readers to reproduce than a local Phoenix.
- **Promptfoo** reports pass/fail, latency, and cost well, but it treats each output as pass/fail text and has no place for Jev's probability distribution, which the calibration analysis needs.

## 7. Controls

- Temperature 0 for Gemini. Jev has no sampling setting.
- Pin both model versions and log the model ID returned with every response.
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
  data/prepare.py          # download Bitext, stratified dev/test split, upload as Phoenix datasets
  data/intents.yaml        # the 46 intent descriptions (frozen after dev)
  classifiers/jev.py       # Choice call + manual OpenInference span -> {intent, probabilities, confidence}
  classifiers/gemini.py    # structured output via instrumented OpenAI client -> same shape
  run.py                   # Phoenix experiments: classifier x split x repeat
  analyze.ipynb            # pulls runs/spans from Phoenix; percentiles, CIs, calibration, plots
  docs/experiment-design.md
```

Stack: Python with `uv`, `datasets`, `typesafe-sdk`, `openai` (the client for the LiteLLM-based, Chat Completions-compatible gateway; used for all LLM calls), `arize-phoenix` (local server), `arize-phoenix-otel`, `arize-phoenix-client`, `openinference-instrumentation-openai`, `pandas`, `scikit-learn`, and `statsmodels`.

Expected cost of the whole experiment: a few dollars at most. About 2,500 calls x 3 repeats x 2-3 systems, at roughly 1-2k input tokens per call.

## 10. Steps

1. Get a Jev API key, and try a handful of messages by hand in `console.typesafe.ai`.
2. Build the splits and write the 46 intent descriptions.
3. Tune the descriptions on the dev split, then freeze them.
4. Run G, J, and optionally G2 on the test split, 3 times.
5. Analyze and write up.

## 11. Open items

- Access: Jev API key, and a key for the LiteLLM-based gateway with token usage in the `usage` object.
- Does the gateway pass through `logprobs` for Gemini? If not, the confidence comparison uses self-reported confidence.
- Check TypeSafe's Master Customer Agreement (`typesafe.ai/legal/mca`) for rules on publishing benchmark results before the blog goes out.
