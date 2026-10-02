# Results: run `main-20261002`

- Ranking quality uses answered searches that have at least one Exact candidate. Gains follow the KDD Cup 2022 values: Exact 1.0, Substitute 0.1, Complement 0.01, Irrelevant 0. NDCG@k is the ranking's discounted gain over the top k divided by the best possible for that search. The gentle column uses Exact 1, Substitute 0.5, Complement 0.1, Irrelevant 0.
- Exact@1: the top product is an Exact match. Usable@1: Exact or Substitute. MRR: 1 / position of the first Exact product. `random` (the presented order) is the floor.
- An unusable answer (`invalid_output`) counts as answered and is ranked in presented order. Blocked searches (the provider's safety filter) and API errors are excluded and counted separately.
- Abstention: a system abstains by its own flag (Gemini listwise; label systems when no candidate is labeled Exact) or when its confidence that some candidate is Exact falls below the frozen threshold in data/rerank/thresholds.yaml (Jev systems).
- Latency: client wall-clock time for one search, from sending the request to a parsed ranking; for per-candidate systems, the whole parallel fan-out. One clean attempt; retries and backoff are not included. Cost: summed tokens priced at the rates recorded in the run's manifests.

## Run conditions

| System | Model requested | Model reported | Reasoning effort | Price per 1M tokens (in / out / thinking) |
| --- | --- | --- | --- | --- |
| gemini-listwise | google/gemini-3.5-flash-lite | google/gemini-3.5-flash-lite | provider default | $0.30 / $2.50 / $2.50 |
| gemini-flash-listwise | google/gemini-3.7-flash | google/gemini-3.7-flash | low | $0.75 / $3.75 / $3.75 |
| gemini-label | google/gemini-3.5-flash-lite | google/gemini-3.5-flash-lite | provider default | $0.30 / $2.50 / $2.50 |
| gemini-pointwise | google/gemini-3.5-flash-lite | google/gemini-3.5-flash-lite | provider default | $0.30 / $2.50 / $2.50 |
| jev-score | jev-1.13.0 | jev-1.13.0 | - | $0.042 / $0.00 / $0.00 |
| jev-noul-batch | jev-1.13.0 | jev-1.13.0 | - | $0.042 / $0.00 / $0.00 |
| jev-noul-pair | jev-1.13.0 | jev-1.13.0 | - | $0.042 / $0.00 / $0.00 |
| jev-choice | jev-1.13.0 | jev-1.13.0 | - | $0.042 / $0.00 / $0.00 |
| lexical | bm25 | - | - | $0.00 / $0.00 / $0.00 |
| random | none | - | - | $0.00 / $0.00 / $0.00 |

| System | Split | Searches | First to last call (UTC) |
| --- | --- | --- | --- |
| gemini-listwise | main | 1,000 | 2026-10-02 02:12 to 2026-10-02 02:30 |
| gemini-listwise | nomatch | 300 | 2026-10-02 02:31 to 2026-10-02 02:36 |
| gemini-listwise | consistency | 300 | 2026-10-02 02:39 to 2026-10-02 02:45 |
| gemini-listwise | reversed | 100 | 2026-10-02 02:37 to 2026-10-02 02:39 |
| gemini-flash-listwise | main | 1,000 | 2026-10-02 02:13 to 2026-10-02 03:21 |
| gemini-flash-listwise | nomatch | 300 | 2026-10-02 02:13 to 2026-10-02 02:39 |
| gemini-flash-listwise | consistency | 300 | 2026-10-02 02:13 to 2026-10-02 02:38 |
| gemini-flash-listwise | reversed | 100 | 2026-10-02 02:13 to 2026-10-02 02:21 |
| gemini-label | main | 1,000 | 2026-10-02 02:12 to 2026-10-02 02:33 |
| gemini-label | nomatch | 300 | 2026-10-02 02:33 to 2026-10-02 02:38 |
| gemini-label | consistency | 300 | 2026-10-02 02:41 to 2026-10-02 02:48 |
| gemini-label | reversed | 100 | 2026-10-02 02:39 to 2026-10-02 02:41 |
| gemini-pointwise | main | 1,000 | 2026-10-02 02:12 to 2026-10-02 02:28 |
| gemini-pointwise | nomatch | 300 | 2026-10-02 02:29 to 2026-10-02 02:33 |
| gemini-pointwise | consistency | 300 | 2026-10-02 02:36 to 2026-10-02 02:42 |
| gemini-pointwise | reversed | 100 | 2026-10-02 02:33 to 2026-10-02 02:35 |
| jev-score | main | 1,000 | 2026-10-02 02:12 to 2026-10-02 02:16 |
| jev-score | nomatch | 300 | 2026-10-02 02:16 to 2026-10-02 02:18 |
| jev-score | consistency | 300 | 2026-10-02 02:18 to 2026-10-02 02:20 |
| jev-score | reversed | 100 | 2026-10-02 02:18 to 2026-10-02 02:18 |
| jev-noul-batch | main | 1,000 | 2026-10-02 02:12 to 2026-10-02 02:15 |
| jev-noul-batch | nomatch | 300 | 2026-10-02 02:15 to 2026-10-02 02:16 |
| jev-noul-batch | consistency | 300 | 2026-10-02 02:16 to 2026-10-02 02:17 |
| jev-noul-batch | reversed | 100 | 2026-10-02 02:16 to 2026-10-02 02:16 |
| jev-noul-pair | main | 1,000 | 2026-10-02 02:12 to 2026-10-02 02:19 |
| jev-noul-pair | nomatch | 300 | 2026-10-02 02:19 to 2026-10-02 02:20 |
| jev-noul-pair | consistency | 300 | 2026-10-02 02:21 to 2026-10-02 02:23 |
| jev-noul-pair | reversed | 100 | 2026-10-02 02:20 to 2026-10-02 02:21 |
| jev-choice | main | 1,000 | 2026-10-02 02:12 to 2026-10-02 02:14 |
| jev-choice | nomatch | 300 | 2026-10-02 02:14 to 2026-10-02 02:15 |
| jev-choice | consistency | 300 | 2026-10-02 02:16 to 2026-10-02 02:17 |
| jev-choice | reversed | 100 | 2026-10-02 02:15 to 2026-10-02 02:16 |
| lexical | main | 1,000 | 2026-10-02 02:12 to 2026-10-02 02:12 |
| lexical | nomatch | 300 | 2026-10-02 02:12 to 2026-10-02 02:12 |
| lexical | consistency | 300 | 2026-10-02 02:12 to 2026-10-02 02:12 |
| lexical | reversed | 100 | 2026-10-02 02:12 to 2026-10-02 02:12 |
| random | main | 1,000 | 2026-10-02 02:12 to 2026-10-02 02:12 |
| random | nomatch | 300 | 2026-10-02 02:12 to 2026-10-02 02:12 |
| random | consistency | 300 | 2026-10-02 02:12 to 2026-10-02 02:12 |
| random | reversed | 100 | 2026-10-02 02:12 to 2026-10-02 02:12 |

- LLM safety settings: provider defaults (not overridden).
- LLM prices source: https://ai.google.dev/gemini-api/docs/pricing (Standard paid tier, read 2026-09-29; 3.7 Flash promo price through 2026-12-31).
- Client location: home wifi. Request timeout: 30.0 s.
- Code: commit f00502f. Python 3.12.14; typesafe-sdk 0.7.2, openai 3.22.0, httpx2 2.13.1, arize-phoenix-otel 0.17.2.
- Label definitions: data/rerank/labels.yaml, SHA-256 7b99d837...
- Abstention thresholds: data/rerank/thresholds.yaml, SHA-256 afcb6eaf....
- Each system re-ranked one search at a time. The systems' runs overlapped in time (see the windows above), so they shared the client machine and network.

## Split: main

### Ranking quality

| System | Searches | NDCG@3 (95% CI) | Exact@1 (95% CI) | Usable@1 | MRR | NDCG@10 | NDCG@3 gentle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-listwise | 991 | 0.768 (0.750-0.786) | 77.4% (74.9%-79.9%) | 92.4% | 0.853 | 0.785 | 0.839 |
| gemini-flash-listwise | 994 | 0.793 (0.776-0.810) | 80.2% (77.7%-82.6%) | 94.1% | 0.872 | 0.807 | 0.860 |
| gemini-label | 984 | 0.723 (0.705-0.742) | 73.5% (70.7%-76.2%) | 92.6% | 0.829 | 0.754 | 0.813 |
| gemini-pointwise | 997 | 0.700 (0.680-0.720) | 69.4% (66.6%-72.3%) | 90.0% | 0.803 | 0.737 | 0.794 |
| jev-score | 1,000 | 0.789 (0.772-0.808) | 81.0% (78.6%-83.3%) | 94.5% | 0.874 | 0.806 | 0.857 |
| jev-noul-batch | 1,000 | 0.786 (0.768-0.804) | 80.9% (78.4%-83.3%) | 94.4% | 0.873 | 0.808 | 0.854 |
| jev-noul-pair | 1,000 | 0.774 (0.756-0.791) | 78.1% (75.5%-80.6%) | 93.0% | 0.861 | 0.795 | 0.845 |
| jev-choice | 1,000 | 0.781 (0.764-0.798) | 81.0% (78.5%-83.3%) | 93.9% | 0.880 | 0.773 | 0.846 |
| lexical | 1,000 | 0.627 (0.607-0.646) | 61.6% (58.5%-64.5%) | 85.6% | 0.750 | 0.687 | 0.731 |
| random | 1,000 | 0.472 (0.452-0.493) | 41.6% (38.5%-44.6%) | 80.3% | 0.599 | 0.565 | 0.629 |

### Ranking quality by slice

| System | Slice | Searches | NDCG@3 | Exact@1 |
| --- | --- | --- | --- | --- |
| gemini-listwise | all | 991 | 0.768 | 77.4% |
| gemini-listwise | general | 820 | 0.765 | 77.4% |
| gemini-listwise | grocery | 171 | 0.783 | 77.2% |
| gemini-listwise | hard (3 or fewer Exact) | 215 | 0.587 | 56.3% |
| gemini-listwise | negations | 80 | 0.713 | 77.5% |
| gemini-flash-listwise | all | 994 | 0.793 | 80.2% |
| gemini-flash-listwise | general | 823 | 0.786 | 79.3% |
| gemini-flash-listwise | grocery | 171 | 0.829 | 84.2% |
| gemini-flash-listwise | hard (3 or fewer Exact) | 216 | 0.644 | 59.7% |
| gemini-flash-listwise | negations | 81 | 0.762 | 82.7% |
| gemini-label | all | 984 | 0.723 | 73.5% |
| gemini-label | general | 814 | 0.722 | 72.9% |
| gemini-label | grocery | 170 | 0.728 | 76.5% |
| gemini-label | hard (3 or fewer Exact) | 212 | 0.533 | 52.8% |
| gemini-label | negations | 77 | 0.691 | 72.7% |
| gemini-pointwise | all | 997 | 0.700 | 69.4% |
| gemini-pointwise | general | 826 | 0.701 | 69.1% |
| gemini-pointwise | grocery | 171 | 0.695 | 70.8% |
| gemini-pointwise | hard (3 or fewer Exact) | 218 | 0.491 | 44.0% |
| gemini-pointwise | negations | 80 | 0.646 | 70.0% |
| jev-score | all | 1,000 | 0.789 | 81.0% |
| jev-score | general | 829 | 0.783 | 80.5% |
| jev-score | grocery | 171 | 0.818 | 83.6% |
| jev-score | hard (3 or fewer Exact) | 218 | 0.608 | 61.0% |
| jev-score | negations | 81 | 0.760 | 82.7% |
| jev-noul-batch | all | 1,000 | 0.786 | 80.9% |
| jev-noul-batch | general | 829 | 0.781 | 80.8% |
| jev-noul-batch | grocery | 171 | 0.809 | 81.3% |
| jev-noul-batch | hard (3 or fewer Exact) | 218 | 0.611 | 62.4% |
| jev-noul-batch | negations | 81 | 0.756 | 82.7% |
| jev-noul-pair | all | 1,000 | 0.774 | 78.1% |
| jev-noul-pair | general | 829 | 0.770 | 77.3% |
| jev-noul-pair | grocery | 171 | 0.791 | 81.9% |
| jev-noul-pair | hard (3 or fewer Exact) | 218 | 0.595 | 57.3% |
| jev-noul-pair | negations | 81 | 0.757 | 84.0% |
| jev-choice | all | 1,000 | 0.781 | 81.0% |
| jev-choice | general | 829 | 0.779 | 81.3% |
| jev-choice | grocery | 171 | 0.789 | 79.5% |
| jev-choice | hard (3 or fewer Exact) | 218 | 0.614 | 61.0% |
| jev-choice | negations | 81 | 0.748 | 80.2% |
| lexical | all | 1,000 | 0.627 | 61.6% |
| lexical | general | 829 | 0.629 | 62.4% |
| lexical | grocery | 171 | 0.617 | 57.9% |
| lexical | hard (3 or fewer Exact) | 218 | 0.482 | 44.5% |
| lexical | negations | 81 | 0.536 | 55.6% |
| random | all | 1,000 | 0.472 | 41.6% |
| random | general | 829 | 0.482 | 43.5% |
| random | grocery | 171 | 0.424 | 32.2% |
| random | hard (3 or fewer Exact) | 218 | 0.227 | 11.9% |
| random | negations | 81 | 0.435 | 39.5% |

### Head-to-head against gemini-listwise

| Pair | Searches | NDCG@3 difference (95% CI) | p (permutation) | Only it Exact@1 | Only reference Exact@1 | p (McNemar) |
| --- | --- | --- | --- | --- | --- | --- |
| gemini-flash-listwise vs gemini-listwise | 991 | +0.025 (+0.013 to +0.037) | <0.0001 | 52 | 25 | 0.0028 |
| gemini-label vs gemini-listwise | 983 | -0.045 (-0.057 to -0.033) | <0.0001 | 30 | 69 | 0.0001 |
| gemini-pointwise vs gemini-listwise | 990 | -0.066 (-0.080 to -0.052) | <0.0001 | 38 | 115 | <0.0001 |
| jev-score vs gemini-listwise | 991 | +0.021 (+0.007 to +0.035) | 0.0030 | 94 | 58 | 0.0044 |
| jev-noul-batch vs gemini-listwise | 991 | +0.017 (+0.003 to +0.031) | 0.0152 | 94 | 61 | 0.0099 |
| jev-noul-pair vs gemini-listwise | 991 | +0.006 (-0.009 to +0.021) | 0.4428 | 97 | 89 | 0.6079 |
| jev-choice vs gemini-listwise | 991 | +0.013 (-0.001 to +0.027) | 0.0739 | 95 | 58 | 0.0035 |
| lexical vs gemini-listwise | 991 | -0.141 (-0.162 to -0.121) | <0.0001 | 88 | 244 | <0.0001 |
| random vs gemini-listwise | 991 | -0.294 (-0.316 to -0.273) | <0.0001 | 27 | 381 | <0.0001 |

### Latency and cost

| System | Timed searches | Calls / search | p50 ms | p95 ms | p99 ms | Mean ms | Input tok | Output tok | Reasoning tok | Cost / 1k searches | Cost / 1M searches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-listwise | 991 | 1.0 | 1,038 | 1,523 | 1,651 | 1,110 | 2,188 | 171.0 | 0.0 | $1.0820 | $1,081.96 |
| gemini-flash-listwise | 994 | 1.0 | 2,989 | 8,748 | 11,030 | 4,054 | 2,186 | 169.1 | 89.6 | $2.6056 | $2,605.64 |
| gemini-label | 984 | 1.0 | 1,127 | 1,826 | 1,963 | 1,250 | 2,972 | 216.2 | 0.0 | $1.4295 | $1,429.45 |
| gemini-pointwise | 997 | 20.8 | 875 | 1,263 | 2,813 | 980 | 12,897 | 176.1 | 0.0 | $4.3047 | $4,304.65 |
| jev-score | 1,000 | 1.0 | 214 | 425 | 529 | 245 | 10,883 | 306.7 | 0.0 | $0.4571 | $457.07 |
| jev-noul-batch | 1,000 | 1.0 | 154 | 369 | 476 | 182 | 4,691 | 369.0 | 0.0 | $0.1970 | $197.01 |
| jev-noul-pair | 1,000 | 20.8 | 323 | 718 | 849 | 377 | 9,909 | 415.6 | 0.0 | $0.4162 | $416.19 |
| jev-choice | 1,000 | 1.0 | 138 | 306 | 421 | 158 | 1,761 | 206.8 | 0.0 | $0.0740 | $73.98 |
| lexical | 1,000 | 0.0 | 0 | 0 | 0 | 0 | - | - | 0.0 | $0.0000 | $0.00 |
| random | 1,000 | 0.0 | 0 | 0 | 0 | 0 | - | - | 0.0 | $0.0000 | $0.00 |

### Reliability

| System | Answered | Blocked | API errors | Invalid | Invented ids | Duplicate ids | Missing ids | Blocked calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-listwise | 991 of 1,000 | 0.9% | 0.0% | 0.0% | 0 | 2 | 1 | 0 |
| gemini-flash-listwise | 994 of 1,000 | 0.6% | 0.0% | 0.0% | 0 | 0 | 4 | 0 |
| gemini-label | 984 of 1,000 | 1.6% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| gemini-pointwise | 997 of 1,000 | 0.3% | 0.0% | 0.0% | 0 | 0 | 892 | 892 |
| jev-score | 1,000 of 1,000 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| jev-noul-batch | 1,000 of 1,000 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| jev-noul-pair | 1,000 of 1,000 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| jev-choice | 1,000 of 1,000 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| lexical | 1,000 of 1,000 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| random | 1,000 of 1,000 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |

### Cascade: Jev keeps confident searches, gemini-listwise gets the rest

| Jev system | Share passed to gemini-listwise | NDCG@3 | Cost / 1k searches |
| --- | --- | --- | --- |
| jev-score | 0.0% | 0.789 | $0.4571 |
| jev-score | 10.0% | 0.785 | $0.5653 |
| jev-score | 25.0% | 0.781 | $0.7276 |
| jev-score | 50.0% | 0.775 | $0.9980 |
| jev-score | 100.0% | 0.768 | $1.5390 |
| jev-noul-batch | 0.0% | 0.785 | $0.1970 |
| jev-noul-batch | 10.0% | 0.781 | $0.3052 |
| jev-noul-batch | 25.0% | 0.776 | $0.4675 |
| jev-noul-batch | 50.0% | 0.771 | $0.7380 |
| jev-noul-batch | 100.0% | 0.768 | $1.2790 |
| jev-noul-pair | 0.0% | 0.774 | $0.4162 |
| jev-noul-pair | 10.0% | 0.776 | $0.5244 |
| jev-noul-pair | 25.0% | 0.779 | $0.6867 |
| jev-noul-pair | 50.0% | 0.777 | $0.9572 |
| jev-noul-pair | 100.0% | 0.768 | $1.4982 |
| jev-choice | 0.0% | 0.781 | $0.0740 |
| jev-choice | 10.0% | 0.781 | $0.1822 |
| jev-choice | 25.0% | 0.778 | $0.3445 |
| jev-choice | 50.0% | 0.773 | $0.6150 |
| jev-choice | 100.0% | 0.768 | $1.1559 |

## Split: nomatch

### Latency and cost

| System | Timed searches | Calls / search | p50 ms | p95 ms | p99 ms | Mean ms | Input tok | Output tok | Reasoning tok | Cost / 1k searches | Cost / 1M searches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-listwise | 298 | 1.0 | 971 | 1,290 | 1,550 | 1,002 | 1,554 | 118.6 | 0.0 | $0.7624 | $762.39 |
| gemini-flash-listwise | 300 | 1.0 | 5,449 | 9,513 | 10,640 | 5,234 | 1,554 | 116.5 | 188.8 | $2.3104 | $2,310.41 |
| gemini-label | 295 | 1.0 | 992 | 1,454 | 1,800 | 1,029 | 1,942 | 124.8 | 0.0 | $0.8898 | $889.80 |
| gemini-pointwise | 300 | 12.0 | 838 | 1,075 | 1,402 | 878 | 7,444 | 102.5 | 0.0 | $2.4896 | $2,489.60 |
| jev-score | 300 | 1.0 | 229 | 405 | 468 | 251 | 6,397 | 175.5 | 0.0 | $0.2687 | $268.66 |
| jev-noul-batch | 300 | 1.0 | 158 | 367 | 427 | 197 | 2,822 | 211.5 | 0.0 | $0.1185 | $118.51 |
| jev-noul-pair | 300 | 12.0 | 268 | 591 | 743 | 301 | 5,729 | 239.9 | 0.0 | $0.2406 | $240.60 |
| jev-choice | 300 | 1.0 | 159 | 383 | 420 | 199 | 1,182 | 128.8 | 0.0 | $0.0497 | $49.66 |
| lexical | 300 | 0.0 | 0 | 0 | 0 | 0 | - | - | 0.0 | $0.0000 | $0.00 |
| random | 300 | 0.0 | 0 | 0 | 0 | 0 | - | - | 0.0 | $0.0000 | $0.00 |

### Reliability

| System | Answered | Blocked | API errors | Invalid | Invented ids | Duplicate ids | Missing ids | Blocked calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-listwise | 298 of 300 | 0.7% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| gemini-flash-listwise | 300 of 300 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| gemini-label | 295 of 300 | 1.7% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| gemini-pointwise | 300 of 300 | 0.0% | 0.0% | 0.0% | 0 | 0 | 131 | 131 |
| jev-score | 300 of 300 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| jev-noul-batch | 300 of 300 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| jev-noul-pair | 300 of 300 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| jev-choice | 300 of 300 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| lexical | 300 of 300 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| random | 300 of 300 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |

## Split: consistency

### Ranking quality

| System | Searches | NDCG@3 (95% CI) | Exact@1 (95% CI) | Usable@1 | MRR | NDCG@10 | NDCG@3 gentle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-listwise | 99 | 0.776 (0.713-0.833) | 76.8% (67.7%-84.8%) | 90.9% | 0.843 | 0.789 | 0.845 |
| gemini-flash-listwise | 100 | 0.804 (0.747-0.855) | 81.0% (73.0%-88.0%) | 97.0% | 0.875 | 0.820 | 0.870 |
| gemini-label | 98 | 0.739 (0.674-0.801) | 75.5% (66.3%-82.7%) | 93.9% | 0.832 | 0.775 | 0.830 |
| gemini-pointwise | 100 | 0.685 (0.626-0.741) | 70.0% (60.0%-78.0%) | 92.0% | 0.804 | 0.749 | 0.788 |
| jev-score | 100 | 0.804 (0.749-0.859) | 82.0% (74.0%-89.0%) | 96.0% | 0.877 | 0.831 | 0.878 |
| jev-noul-batch | 100 | 0.805 (0.750-0.859) | 82.0% (75.0%-89.0%) | 96.0% | 0.881 | 0.829 | 0.876 |
| jev-noul-pair | 100 | 0.799 (0.743-0.853) | 84.0% (76.0%-91.0%) | 97.0% | 0.887 | 0.829 | 0.872 |
| jev-choice | 100 | 0.769 (0.719-0.817) | 79.0% (71.0%-86.0%) | 92.0% | 0.870 | 0.781 | 0.841 |
| lexical | 100 | 0.626 (0.563-0.689) | 58.0% (49.0%-67.0%) | 82.0% | 0.722 | 0.694 | 0.737 |
| random | 100 | 0.453 (0.390-0.521) | 40.0% (31.0%-50.0%) | 75.0% | 0.566 | 0.568 | 0.604 |

### Head-to-head against gemini-listwise

| Pair | Searches | NDCG@3 difference (95% CI) | p (permutation) | Only it Exact@1 | Only reference Exact@1 | p (McNemar) |
| --- | --- | --- | --- | --- | --- | --- |
| gemini-flash-listwise vs gemini-listwise | 99 | +0.027 (-0.014 to +0.068) | 0.2175 | 7 | 3 | 0.3438 |
| gemini-label vs gemini-listwise | 98 | -0.037 (-0.076 to +0.002) | 0.0641 | 5 | 6 | 1.0000 |
| gemini-pointwise vs gemini-listwise | 99 | -0.089 (-0.132 to -0.047) | 0.0001 | 4 | 11 | 0.1185 |
| jev-score vs gemini-listwise | 99 | +0.029 (-0.019 to +0.078) | 0.2544 | 10 | 5 | 0.3018 |
| jev-noul-batch vs gemini-listwise | 99 | +0.027 (-0.014 to +0.074) | 0.2557 | 9 | 4 | 0.2668 |
| jev-noul-pair vs gemini-listwise | 99 | +0.022 (-0.027 to +0.074) | 0.4330 | 13 | 6 | 0.1671 |
| jev-choice vs gemini-listwise | 99 | -0.006 (-0.055 to +0.042) | 0.8022 | 10 | 8 | 0.8145 |
| lexical vs gemini-listwise | 99 | -0.149 (-0.210 to -0.090) | <0.0001 | 6 | 24 | 0.0014 |
| random vs gemini-listwise | 99 | -0.326 (-0.390 to -0.262) | <0.0001 | 3 | 40 | <0.0001 |

### Latency and cost

| System | Timed searches | Calls / search | p50 ms | p95 ms | p99 ms | Mean ms | Input tok | Output tok | Reasoning tok | Cost / 1k searches | Cost / 1M searches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-listwise | 99 | 1.0 | 1,069 | 1,508 | 1,586 | 1,107 | 2,138 | 164.0 | 0.0 | $1.0493 | $1,049.30 |
| gemini-flash-listwise | 100 | 1.0 | 2,851 | 8,345 | 10,436 | 4,105 | 2,134 | 161.5 | 103.8 | $2.5954 | $2,595.37 |
| gemini-label | 98 | 1.0 | 1,145 | 1,772 | 1,814 | 1,230 | 2,878 | 205.0 | 0.0 | $1.3697 | $1,369.74 |
| gemini-pointwise | 100 | 19.6 | 873 | 1,163 | 1,617 | 916 | 12,209 | 166.1 | 0.0 | $4.0779 | $4,077.90 |
| jev-score | 100 | 1.0 | 256 | 505 | 579 | 283 | 10,340 | 289.8 | 0.0 | $0.4343 | $434.28 |
| jev-noul-batch | 100 | 1.0 | 197 | 370 | 400 | 220 | 4,484 | 348.7 | 0.0 | $0.1883 | $188.35 |
| jev-noul-pair | 100 | 19.6 | 230 | 713 | 858 | 393 | 9,400 | 393.0 | 0.0 | $0.3948 | $394.80 |
| jev-choice | 100 | 1.0 | 206 | 419 | 476 | 234 | 1,721 | 196.6 | 0.0 | $0.0723 | $72.28 |
| lexical | 100 | 0.0 | 0 | 0 | 0 | 0 | - | - | 0.0 | $0.0000 | $0.00 |
| random | 100 | 0.0 | 0 | 0 | 0 | 0 | - | - | 0.0 | $0.0000 | $0.00 |

### Reliability

| System | Answered | Blocked | API errors | Invalid | Invented ids | Duplicate ids | Missing ids | Blocked calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-listwise | 99 of 100 | 1.0% | 0.0% | 0.0% | 0 | 1 | 0 | 0 |
| gemini-flash-listwise | 100 of 100 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| gemini-label | 98 of 100 | 2.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| gemini-pointwise | 100 of 100 | 0.0% | 0.0% | 0.0% | 0 | 0 | 103 | 103 |
| jev-score | 100 of 100 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| jev-noul-batch | 100 of 100 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| jev-noul-pair | 100 of 100 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| jev-choice | 100 of 100 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| lexical | 100 of 100 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| random | 100 of 100 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |

## Split: reversed

### Ranking quality

| System | Searches | NDCG@3 (95% CI) | Exact@1 (95% CI) | Usable@1 | MRR | NDCG@10 | NDCG@3 gentle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-listwise | 100 | 0.803 (0.751-0.852) | 80.0% (72.0%-87.0%) | 97.0% | 0.872 | 0.804 | 0.876 |
| gemini-flash-listwise | 100 | 0.820 (0.767-0.869) | 83.0% (75.0%-90.0%) | 99.0% | 0.884 | 0.831 | 0.890 |
| gemini-label | 100 | 0.727 (0.666-0.787) | 75.0% (66.0%-83.0%) | 95.0% | 0.833 | 0.763 | 0.821 |
| gemini-pointwise | 100 | 0.742 (0.686-0.794) | 77.0% (69.0%-85.0%) | 95.0% | 0.846 | 0.770 | 0.835 |
| jev-score | 100 | 0.807 (0.758-0.858) | 83.0% (76.0%-90.0%) | 96.0% | 0.886 | 0.820 | 0.876 |
| jev-noul-batch | 100 | 0.822 (0.768-0.873) | 84.0% (77.0%-91.0%) | 96.0% | 0.889 | 0.833 | 0.884 |
| jev-noul-pair | 100 | 0.809 (0.753-0.863) | 86.0% (79.0%-93.0%) | 97.0% | 0.898 | 0.830 | 0.878 |
| jev-choice | 100 | 0.773 (0.719-0.822) | 83.0% (76.0%-90.0%) | 95.0% | 0.884 | 0.793 | 0.846 |
| lexical | 100 | 0.627 (0.564-0.690) | 59.0% (50.0%-68.0%) | 82.0% | 0.726 | 0.695 | 0.737 |
| random | 100 | 0.480 (0.413-0.543) | 44.0% (34.0%-54.0%) | 78.0% | 0.607 | 0.586 | 0.633 |

### Head-to-head against gemini-listwise

| Pair | Searches | NDCG@3 difference (95% CI) | p (permutation) | Only it Exact@1 | Only reference Exact@1 | p (McNemar) |
| --- | --- | --- | --- | --- | --- | --- |
| gemini-flash-listwise vs gemini-listwise | 100 | +0.017 (-0.017 to +0.052) | 0.3590 | 5 | 2 | 0.4531 |
| gemini-label vs gemini-listwise | 100 | -0.076 (-0.118 to -0.035) | 0.0001 | 5 | 10 | 0.3018 |
| gemini-pointwise vs gemini-listwise | 100 | -0.061 (-0.107 to -0.012) | 0.0148 | 7 | 10 | 0.6291 |
| jev-score vs gemini-listwise | 100 | +0.003 (-0.033 to +0.042) | 0.8561 | 7 | 4 | 0.5488 |
| jev-noul-batch vs gemini-listwise | 100 | +0.019 (-0.025 to +0.063) | 0.4003 | 8 | 4 | 0.3877 |
| jev-noul-pair vs gemini-listwise | 100 | +0.006 (-0.043 to +0.054) | 0.8254 | 10 | 4 | 0.1796 |
| jev-choice vs gemini-listwise | 100 | -0.030 (-0.073 to +0.009) | 0.1472 | 9 | 6 | 0.6072 |
| lexical vs gemini-listwise | 100 | -0.176 (-0.229 to -0.123) | <0.0001 | 3 | 24 | <0.0001 |
| random vs gemini-listwise | 100 | -0.324 (-0.392 to -0.258) | <0.0001 | 7 | 43 | <0.0001 |

### Latency and cost

| System | Timed searches | Calls / search | p50 ms | p95 ms | p99 ms | Mean ms | Input tok | Output tok | Reasoning tok | Cost / 1k searches | Cost / 1M searches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-listwise | 100 | 1.0 | 1,049 | 1,484 | 1,655 | 1,096 | 2,134 | 163.0 | 0.0 | $1.0477 | $1,047.72 |
| gemini-flash-listwise | 100 | 1.0 | 2,970 | 9,660 | 10,904 | 4,302 | 2,134 | 161.9 | 112.2 | $2.6281 | $2,628.08 |
| gemini-label | 100 | 1.0 | 1,117 | 1,764 | 1,895 | 1,205 | 2,864 | 203.9 | 0.0 | $1.3689 | $1,368.92 |
| gemini-pointwise | 100 | 19.6 | 918 | 1,259 | 1,980 | 983 | 12,209 | 166.4 | 0.0 | $4.0787 | $4,078.65 |
| jev-score | 100 | 1.0 | 216 | 367 | 444 | 238 | 10,340 | 289.8 | 0.0 | $0.4343 | $434.28 |
| jev-noul-batch | 100 | 1.0 | 184 | 373 | 427 | 229 | 4,484 | 348.7 | 0.0 | $0.1883 | $188.35 |
| jev-noul-pair | 100 | 19.6 | 222 | 606 | 881 | 306 | 9,400 | 393.0 | 0.0 | $0.3948 | $394.80 |
| jev-choice | 100 | 1.0 | 256 | 393 | 490 | 243 | 1,721 | 196.6 | 0.0 | $0.0723 | $72.28 |
| lexical | 100 | 0.0 | 0 | 0 | 0 | 0 | - | - | 0.0 | $0.0000 | $0.00 |
| random | 100 | 0.0 | 0 | 0 | 0 | 0 | - | - | 0.0 | $0.0000 | $0.00 |

### Reliability

| System | Answered | Blocked | API errors | Invalid | Invented ids | Duplicate ids | Missing ids | Blocked calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-listwise | 100 of 100 | 0.0% | 0.0% | 0.0% | 0 | 1 | 1 | 0 |
| gemini-flash-listwise | 100 of 100 | 0.0% | 0.0% | 0.0% | 0 | 0 | 1 | 0 |
| gemini-label | 100 of 100 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| gemini-pointwise | 100 of 100 | 0.0% | 0.0% | 0.0% | 0 | 0 | 105 | 105 |
| jev-score | 100 of 100 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| jev-noul-batch | 100 of 100 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| jev-noul-pair | 100 of 100 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| jev-choice | 100 of 100 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| lexical | 100 of 100 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| random | 100 of 100 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 |

## Abstention: does the system notice when nothing matches?

| System | Splits | Threshold | Should abstain | Precision | Recall | F1 | False abstentions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-listwise | main + nomatch | own flag | 298 of 1,289 | 49.3% | 48.3% | 48.8% | 14.9% |
| gemini-flash-listwise | main + nomatch | own flag | 300 of 1,294 | 58.5% | 46.0% | 51.5% | 9.9% |
| gemini-label | main + nomatch | own flag | 295 of 1,279 | 52.4% | 51.5% | 52.0% | 14.0% |
| gemini-pointwise | main + nomatch | own flag | 300 of 1,297 | 52.8% | 51.0% | 51.9% | 13.7% |
| jev-score | main + nomatch | 0.785 | 300 of 1,300 | 40.3% | 78.0% | 53.1% | 34.7% |
| jev-noul-batch | main + nomatch | 0.565 | 300 of 1,300 | 43.8% | 65.3% | 52.4% | 25.2% |
| jev-noul-pair | main + nomatch | 0.590 | 300 of 1,300 | 46.2% | 62.0% | 52.9% | 21.7% |
| jev-choice | main + nomatch | 0.795 | 300 of 1,300 | 43.4% | 71.0% | 53.9% | 27.8% |

## Label quality (systems that label each candidate)

| System | Candidates | Accuracy | Macro-F1 | Exact labeled Substitute | Substitute labeled Exact |
| --- | --- | --- | --- | --- | --- |
| gemini-label | 24,379 | 57.9% | 54.2% | 39.3% | 14.8% |
| gemini-pointwise | 24,668 | 55.8% | 53.2% | 33.3% | 16.0% |
| jev-score | 24,708 | 55.3% | 53.2% | 31.3% | 14.3% |

Confusion, gemini-label (rows: gold label; columns: predicted):

|  | E | S | C | I |
| --- | --- | --- | --- | --- |
| gold E | 5335 | 4071 | 169 | 784 |
| gold S | 1329 | 6055 | 90 | 1533 |
| gold C | 103 | 288 | 426 | 259 |
| gold I | 218 | 1241 | 169 | 2309 |

Confusion, gemini-pointwise (rows: gold label; columns: predicted):

|  | E | S | C | I | missing |
| --- | --- | --- | --- | --- | --- |
| gold E | 5491 | 3480 | 236 | 836 | 403 |
| gold S | 1455 | 5548 | 119 | 1514 | 477 |
| gold C | 139 | 265 | 444 | 199 | 32 |
| gold I | 286 | 1098 | 184 | 2274 | 188 |

Confusion, jev-score (rows: gold label; columns: predicted):

|  | E | S | C | I |
| --- | --- | --- | --- | --- |
| gold E | 5269 | 3280 | 274 | 1645 |
| gold S | 1302 | 4924 | 126 | 2777 |
| gold C | 113 | 100 | 525 | 341 |
| gold I | 223 | 683 | 191 | 2935 |

## Consistency

| System | Same #1 across repeats | Kendall's tau across repeats | Same #1 when order is reversed |
| --- | --- | --- | --- |
| gemini-listwise | 95.3% | 0.895 | 26.3% |
| gemini-flash-listwise | 91.9% | 0.890 | 43.0% |
| gemini-label | 86.7% | 0.900 | 14.3% |
| gemini-pointwise | 82.0% | 0.846 | 22.0% |
| jev-score | 79.7% | 0.881 | 54.0% |
| jev-noul-batch | 85.7% | 0.883 | 57.0% |
| jev-noul-pair | 80.7% | 0.893 | 75.0% |
| jev-choice | 94.0% | 0.960 | 61.0% |
| lexical | 100.0% | 1.000 | 94.0% |
| random | 100.0% | 1.000 | 0.0% |

## Calibration of Jev's P(exact)

| System | Candidates | Calibration error of P(exact) |
| --- | --- | --- |
| jev-score | 24,708 | 0.179 |
| jev-noul-batch | 24,708 | 0.159 |
| jev-noul-pair | 24,708 | 0.125 |
