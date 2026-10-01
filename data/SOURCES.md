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
