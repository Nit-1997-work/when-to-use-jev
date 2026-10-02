# Data sources

## Bitext Retail (eCommerce)

- Dataset: `bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset` on Hugging Face.
- Pinned revision: `12dd624ddcd3057382b2faad661bcda1fa869491`.
- File: `data/intent/raw/bitext-retail-ecommerce.csv` (gitignored, 42.6 MB).
- SHA-256: `13a988266fed4e2b2c1ff947a89ef220ce09b5b13ac83c4a1496c0d7b81e8127`.
- License: CDLA-Sharing-1.0. The dataset card is saved as `data/intent/raw/README.md`.
- Downloaded: 2026-09-29.

Re-fetch:

```bash
curl -L -o data/intent/raw/bitext-retail-ecommerce.csv \
  "https://huggingface.co/datasets/bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset/resolve/12dd624ddcd3057382b2faad661bcda1fa869491/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv"
```

Profile at download time:

- 44,884 rows. Columns: `instruction`, `intent`, `category`, `tags`, `response`. No nulls.
- 46 intents in 13 categories, with 721-1,000 rows per intent.
- No duplicate messages, and no message appears with two different intents.
- Messages are short: median 12 words, p95 17, max 24.
- The `W` tag (49.7% of rows) marks profanity. 97% of `W` rows contain profanity, and no other rows do.
- The `Z` tag (51.1% of rows) marks typos and noise.

## Amazon Shopping Queries Dataset (ESCI)

- Repository: `amazon-science/esci-data` on GitHub, folder `shopping_queries_dataset/`.
- Pinned commit: `7916cdf6ab75a462e77f20ab40428a10923998d5` (2024-10-07).
- Files (all gitignored under `data/rerank/raw/`):

| File | Size | SHA-256 |
|---|---|---|
| `shopping_queries_dataset_examples.parquet` | 51.3 MB | `4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a` |
| `shopping_queries_dataset_products.parquet` | 1.1 GB | `25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265` |
| `shopping_queries_dataset_sources.csv` | 1.7 MB | `a5fed8ecc016443de40bf3c63098f0e3f23bbe4daa4236f1c38b8c3184778c50` |

- License: Apache-2.0. The repository's LICENSE is saved as `data/rerank/raw/LICENSE`.
- Paper: Reddy et al., "Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search", 2022 (arXiv 2206.06588).
- Downloaded: 2026-10-01.

Re-fetch:

```bash
for f in shopping_queries_dataset_examples.parquet shopping_queries_dataset_products.parquet \
         shopping_queries_dataset_sources.csv; do
  curl -L -o "data/rerank/raw/$f" \
    "https://github.com/amazon-science/esci-data/raw/7916cdf6ab75a462e77f20ab40428a10923998d5/shopping_queries_dataset/$f"
done
```

Profile at download time (reduced "Task 1" version, `small_version == 1`, English `us` locale):

- 29,844 searches and 601,354 judgments: 20,888 train and 8,956 test searches.
- Labels: Exact 43.5%, Substitute 35.1%, Irrelevant 16.9%, Complement 4.5%.
- Candidates per search: median 16, mean 20.1, minimum 8, maximum 188. 2.2% of test searches have more than 40.
- Only 1 search has no Exact candidate. Every product has a title.
- Query sources: `other` 24,107, `parse_pattern` 2,558, `negations` 2,318, `behavioral` 762, `nlqec` 99.

Splits built by `jevbench rerank prepare` (searches with 8 to 40 candidates; seed 20261001):

| Split | Searches | Candidates (median) | Exact / Substitute / Complement / Irrelevant | Grocery | Hard | Should abstain |
|---|---|---|---|---|---|---|
| `practice` | 100 | 11-40 (16) | 919 / 694 / 86 / 405 | 0 | 19 | 0 |
| `practice-nomatch` | 50 | 5-36 (11) | 0 / 363 / 58 / 214 | 0 | 0 | 50 |
| `main` | 1,000 | 8-40 (16) | 8,824 / 7,699 / 903 / 3,352 | 171 | 218 | 0 |
| `nomatch` | 300 | 5-39 (11) | 0 / 2,336 / 355 / 908 | 0 | 0 | 300 |
| `consistency` | 100 | 9-40 (16) | 822 / 715 / 88 / 340 | 0 | 28 | 0 |
| `reversed` | 100 | 9-40 (16) | 822 / 715 / 88 / 340 | 0 | 28 | 0 |
| `warmup` | 20 | 14-40 (16) | 178 / 130 / 12 / 72 | 0 | 5 | 0 |
