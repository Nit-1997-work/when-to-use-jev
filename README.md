# when-to-use-jev

Benchmarking Jev against an LLM for e-commerce intent classification on a public dataset: accuracy, latency, cost, and when each one is the right choice.

- Results: [results/intent/pilot-20260929T205311Z/report.md](results/intent/pilot-20260929T205311Z/report.md).
- Experiment design: [docs/intent/experiment-design.md](docs/intent/experiment-design.md).

All LLM calls go through a LiteLLM-based gateway compatible with OpenAI's Chat Completions API; there are no direct Gemini API calls.

## Rebuild the published report

The report is computed only from the committed records and their manifests, so it needs no API keys:

```bash
make install
uv run jevbench intent report --run-id pilot-20260929T205311Z
```

CI checks on every change that the committed report is exactly what the code computes from the committed records.

## Run the experiment yourself

Requirements: [uv](https://docs.astral.sh/uv/), a TypeSafe API key, an OpenAI-compatible LLM gateway (for example a LiteLLM proxy in front of Vertex AI) with its key, and optionally Docker for Phoenix.
Keep the provider's default safety settings to reproduce the published conditions.

```bash
make install                 # uv sync
cp .env.example .env         # fill in keys, models, prices, and CLIENT_LOCATION
curl -L -o data/intent/raw/bitext-retail-ecommerce.csv "<URL in data/SOURCES.md>"
make intent-prepare          # rebuild data/intent/splits/*.jsonl (checks the CSV's SHA-256; output is byte-identical)
make phoenix                 # optional: tracing UI + collector
make intent-check            # one live call per system
make intent-smoke            # 5 practice examples x every system, then a report
make intent-run              # full run: every system x main + profanity splits (needs committed code)
make audit                   # before publishing: scan for the gateway host, API keys, and .audit-denylist patterns
```

`make intent-sync RUN_ID=<run_id>` mirrors a run into Phoenix as datasets and experiments.

## Systems under test

| Name | What it is |
| --- | --- |
| `jev` | TypeSafe Jev, one Choice question over all 46 intents. |
| `baseline` | `BASELINE_MODEL` (Gemini Flash-Lite) through the gateway, structured output: intent only. |
| `baseline-conf` | Same model, structured output with a self-reported confidence (the gateway does not expose logprobs). |
| `baseline-2` | `BASELINE_MODEL_2` (Gemini Flash) through the gateway, intent only. |

Both systems receive identical label text from [data/intent/intents.yaml](data/intent/intents.yaml).

## How the harness measures

- Each system sends one request at a time, after discarded warm-up calls, in a shuffled order.
- Latency is client wall-clock time for exactly one HTTP attempt; SDK retries are off, and the runner's retries and backoff are recorded separately.
- Every gateway request bypasses the LiteLLM response cache; any cache hit is flagged and excluded from latency.
- The provider's default safety settings are used.
  When the safety filter withholds an answer, the message counts as blocked: it is excluded from quality metrics and reported separately, split into clean and profane messages.
- Token usage comes from each response, including reasoning tokens.
  Cost uses the prices recorded in the run's manifests and includes blocked calls, which are billed.
- Every classification is one JSONL record under `results/<experiment>/<run_id>/<system>/<split>/repeat-<n>.jsonl`, next to a manifest with model IDs, prices, git commit, package versions, and hashes of the intents and split files.
- Runs are resumable: re-running the same command skips examples that already have a final record, redoes API errors, and refuses to continue if the configuration changed.
- Full runs refuse to start from uncommitted code, so every published number traces to a commit.

## Layout

```text
data/intent/intents.yaml   # the 46 intent descriptions both systems see
data/intent/splits/*.jsonl # practice / main / profanity / warmup splits
results/intent/<run_id>/   # published runs: records, manifests, and report.md
src/jevbench/common/       # shared harness: settings, gateway and Jev calls, runner, stats, report formatting, audit
src/jevbench/intent/       # intent classification: data, classifiers, metrics, report, Phoenix sync
src/jevbench/cli.py        # `jevbench intent ...`, `jevbench audit`, `jevbench phoenix-setup`
scripts/                 # run matrix and Phoenix startup
tests/                   # offline tests (make validate), including the published-results check
```

## License

The code is licensed under Apache-2.0 ([LICENSE](LICENSE)).
The data in `data/intent/splits/` and the published result records are derived from the Bitext Retail (eCommerce) dataset and are licensed under CDLA-Sharing-1.0 ([data/LICENSE.md](data/LICENSE.md)).
