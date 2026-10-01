# Results: run `pilot-20260929T205311Z`

- Quality metrics (accuracy, macro-F1, category accuracy, top-2, slices, confusions, calibration, coverage) use answered messages. A message is answered when the system returned an intent; an intent outside the list counts as wrong.
- Blocked: the provider's safety filter withheld the answer (`finish_reason=content_filter` through the gateway). Blocked messages are excluded from quality metrics and counted in the Blocked column and table. API errors (failures after retries) are excluded and counted the same way.
- Latency: client wall-clock time for one HTTP attempt, from sending the request to a parsed answer, over answered calls not served from a cache. Retries and their backoff are not included.
- Gateway-reported time: the gateway's own duration header for the call (`x-litellm-response-duration-ms`); the rest of the client latency is network and gateway ingress.
- Cost: the tokens reported by each response, priced at the rates recorded in the run's manifests and averaged over every call made, including blocked ones (they are billed).
- Confidence: Jev uses the probability of its chosen intent; `baseline-conf` uses the model's self-reported confidence. Calibration error is the expected calibration error over 15 equal-width bins. Coverage at X% is the largest share of answered messages a system can handle on its own, by acting only on answers at or above one confidence threshold, while staying at least X% accurate on them; thresholds fall only between groups of equal confidence.
- Head-to-head: McNemar's exact test on the messages both systems answered (repeat 1).

## Run conditions

| System | Model requested | Model reported | Reasoning effort | Price per 1M tokens (in / out / thinking) |
| --- | --- | --- | --- | --- |
| jev | jev-1.13.0 | jev-1.13.0 | - | $0.042 / $0.00 / $0.00 |
| baseline | google/gemini-3.5-flash-lite | google/gemini-3.5-flash-lite | provider default | $0.30 / $2.50 / $2.50 |
| baseline-conf | google/gemini-3.5-flash-lite | google/gemini-3.5-flash-lite | provider default | $0.30 / $2.50 / $2.50 |
| baseline-2 | google/gemini-3.7-flash | google/gemini-3.7-flash | low | $0.75 / $3.75 / $3.75 |

| System | Split | Calls | First to last call (UTC) |
| --- | --- | --- | --- |
| jev | main | 2,300 | 2026-09-29 20:53 to 2026-09-29 20:59 |
| jev | profanity | 460 | 2026-09-29 20:59 to 2026-09-29 21:00 |
| baseline | main | 2,300 | 2026-09-29 20:53 to 2026-09-29 21:27 |
| baseline | profanity | 460 | 2026-09-29 21:27 to 2026-09-29 21:34 |
| baseline-conf | main | 2,300 | 2026-09-29 20:53 to 2026-09-29 21:28 |
| baseline-conf | profanity | 460 | 2026-09-29 21:29 to 2026-09-29 21:35 |
| baseline-2 | main | 2,300 | 2026-09-29 20:54 to 2026-09-29 22:53 |
| baseline-2 | profanity | 460 | 2026-09-29 22:53 to 2026-09-29 23:17 |

- LLM safety settings: provider defaults (not overridden).
- LLM prices source: https://ai.google.dev/gemini-api/docs/pricing (Standard paid tier, read 2026-09-29; 3.7 Flash promo price through 2026-12-31).
- Client location: home wifi. Request timeout: 30.0 s.
- Code: commit 383854f (with uncommitted changes). Python 3.12.14; typesafe-sdk 0.7.2, openai 3.22.0, arize-phoenix-otel 0.17.2.
- Each system sent one request at a time. The systems' runs overlapped in time (see the windows above), so they shared the client machine and network.

## Split: main

### Accuracy

| System | Answered | Blocked | API errors | Accuracy (95% CI) | Macro-F1 | Category acc | Top-2 acc | Invalid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jev | 2,300 of 2,300 | 0.0% | 0.0% | 96.6% (95.9%-97.3%) | 96.6% | 97.8% | 99.0% | 0.0% |
| baseline | 2,060 of 2,300 | 10.4% | 0.0% | 96.7% (95.9%-97.5%) | 96.7% | 98.1% | - | 0.0% |
| baseline-conf | 2,146 of 2,300 | 6.7% | 0.0% | 96.4% (95.6%-97.2%) | 96.2% | 97.8% | - | 0.0% |
| baseline-2 | 2,258 of 2,300 | 1.8% | 0.0% | 96.4% (95.7%-97.1%) | 96.4% | 98.2% | - | 0.0% |

### Blocked messages

The provider's safety filter withheld these answers; they are excluded from the quality metrics.

| System | Clean messages blocked | Profane messages blocked |
| --- | --- | --- |
| jev | 0 of 2,070 (0.0%) | 0 of 230 (0.0%) |
| baseline | 76 of 2,070 (3.7%) | 164 of 230 (71.3%) |
| baseline-conf | 87 of 2,070 (4.2%) | 67 of 230 (29.1%) |
| baseline-2 | 4 of 2,070 (0.2%) | 38 of 230 (16.5%) |

### Latency and cost

| System | Timed calls | p50 ms | p95 ms | p99 ms | Mean ms | Gateway-reported p50 ms | Input tok | Output tok | Reasoning tok | Cost / 1k | Cost / 1M |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jev | 2,300 | 145 | 238 | 565 | 162 | - | 2,235 | 403.3 | 0.0 | $0.0939 | $93.87 |
| baseline | 2,060 | 831 | 1,274 | 1,501 | 887 | 677 | 2,281 | 13.0 | 0.0 | $0.7168 | $716.83 |
| baseline-conf | 2,146 | 857 | 1,306 | 1,632 | 911 | 705 | 2,340 | 22.5 | 0.0 | $0.7583 | $758.32 |
| baseline-2 | 2,258 | 2,883 | 4,785 | 6,628 | 3,099 | 2,675 | 2,281 | 8.4 | 65.2 | $1.9824 | $1,982.36 |

### Confidence

| System | Signal | Answers with a signal | Calibration error | Coverage at 95% acc | Coverage at 98% acc |
| --- | --- | --- | --- | --- | --- |
| jev | top probability | 2,300 | 0.013 | 100.0% | 97.8% |
| baseline-conf | self-reported | 2,146 | 0.010 | 100.0% | 69.4% |

### Head-to-head on messages both systems answered

| Pair | Both answered | Jev acc | Other acc | Only Jev right | Only other right | p-value |
| --- | --- | --- | --- | --- | --- | --- |
| jev vs baseline | 2,060 | 96.7% | 96.7% | 25 | 25 | 1.0000 |
| jev vs baseline-conf | 2,146 | 96.7% | 96.4% | 30 | 24 | 0.4966 |
| jev vs baseline-2 | 2,258 | 96.5% | 96.4% | 37 | 34 | 0.8126 |

### Accuracy by typo slice

| System | Slice | Answered | Accuracy |
| --- | --- | --- | --- |
| jev | no typos | 1,108 | 96.7% |
| jev | typos | 1,192 | 96.6% |
| baseline | no typos | 982 | 96.7% |
| baseline | typos | 1,078 | 96.8% |
| baseline-conf | no typos | 1,034 | 96.3% |
| baseline-conf | typos | 1,112 | 96.5% |
| baseline-2 | no typos | 1,088 | 96.8% |
| baseline-2 | typos | 1,170 | 96.1% |

### Top confusions: jev

| Expected | Predicted | Count |
| --- | --- | --- |
| request_right_to_rectification | change_account | 11 |
| track_order | track_delivery | 10 |
| damaged_delivery | product_issue | 9 |
| customer_service | human_agent | 8 |
| return_policy | return_product_in_store | 8 |
| payment_methods | pay | 6 |
| return_product_in_store | product_information | 5 |
| exchange_product | exchange_product_in_store | 3 |

### Top confusions: baseline

| Expected | Predicted | Count |
| --- | --- | --- |
| request_right_to_rectification | change_account | 14 |
| return_policy | return_product_in_store | 8 |
| track_order | track_delivery | 8 |
| return_product_in_store | product_information | 5 |
| damaged_delivery | product_issue | 4 |
| missing_item | delivery_issue | 4 |
| payment_methods | pay | 3 |
| refund_policy | return_policy | 3 |

### Top confusions: baseline-conf

| Expected | Predicted | Count |
| --- | --- | --- |
| request_right_to_rectification | change_account | 13 |
| return_policy | return_product_in_store | 8 |
| track_order | track_delivery | 8 |
| damaged_delivery | product_issue | 6 |
| refund_policy | return_policy | 3 |
| return_product | return_policy | 3 |
| return_product_in_store | product_information | 3 |
| track_order | refund_status | 3 |

### Top confusions: baseline-2

| Expected | Predicted | Count |
| --- | --- | --- |
| request_right_to_rectification | change_account | 18 |
| return_policy | return_product_in_store | 8 |
| return_product | return_policy | 8 |
| return_product_in_store | product_information | 8 |
| track_order | track_delivery | 7 |
| request_refund | refund_status | 6 |
| payment_methods | pay | 5 |
| submit_product_feedback | submit_feedback | 4 |

## Split: profanity

### Accuracy

| System | Answered | Blocked | API errors | Accuracy (95% CI) | Macro-F1 | Category acc | Top-2 acc | Invalid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jev | 460 of 460 | 0.0% | 0.0% | 95.7% (93.7%-97.6%) | 95.6% | 98.3% | 99.1% | 0.0% |
| baseline | 132 of 460 | 71.3% | 0.0% | 97.7% (94.7%-100.0%) | 95.1% | 100.0% | - | 0.0% |
| baseline-conf | 360 of 460 | 21.7% | 0.0% | 95.6% (93.3%-97.8%) | 94.9% | 97.5% | - | 0.0% |
| baseline-2 | 403 of 460 | 12.4% | 0.0% | 95.3% (93.1%-97.3%) | 95.3% | 97.8% | - | 0.0% |

### Blocked messages

The provider's safety filter withheld these answers; they are excluded from the quality metrics.

| System | Clean messages blocked | Profane messages blocked |
| --- | --- | --- |
| jev | - | 0 of 460 (0.0%) |
| baseline | - | 328 of 460 (71.3%) |
| baseline-conf | - | 100 of 460 (21.7%) |
| baseline-2 | - | 57 of 460 (12.4%) |

### Latency and cost

| System | Timed calls | p50 ms | p95 ms | p99 ms | Mean ms | Gateway-reported p50 ms | Input tok | Output tok | Reasoning tok | Cost / 1k | Cost / 1M |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jev | 460 | 156 | 268 | 795 | 176 | - | 2,236 | 403.3 | 0.0 | $0.0939 | $93.91 |
| baseline | 132 | 823 | 1,270 | 1,364 | 879 | 710 | 2,282 | 13.4 | 0.0 | $0.7171 | $717.11 |
| baseline-conf | 360 | 847 | 1,352 | 1,655 | 902 | 730 | 2,341 | 22.6 | 0.0 | $0.7585 | $758.49 |
| baseline-2 | 403 | 2,951 | 4,871 | 6,090 | 3,188 | 2,745 | 2,282 | 8.4 | 68.8 | $1.9690 | $1,969.02 |

### Confidence

| System | Signal | Answers with a signal | Calibration error | Coverage at 95% acc | Coverage at 98% acc |
| --- | --- | --- | --- | --- | --- |
| jev | top probability | 460 | 0.018 | 100.0% | 96.1% |
| baseline-conf | self-reported | 360 | 0.015 | 100.0% | 64.2% |

### Head-to-head on messages both systems answered

| Pair | Both answered | Jev acc | Other acc | Only Jev right | Only other right | p-value |
| --- | --- | --- | --- | --- | --- | --- |
| jev vs baseline | 132 | 97.0% | 97.7% | 1 | 2 | 1.0000 |
| jev vs baseline-conf | 360 | 95.0% | 95.6% | 4 | 6 | 0.7539 |
| jev vs baseline-2 | 403 | 95.0% | 95.3% | 8 | 9 | 1.0000 |

### Accuracy by typo slice

| System | Slice | Answered | Accuracy |
| --- | --- | --- | --- |
| jev | no typos | 225 | 96.0% |
| jev | typos | 235 | 95.3% |
| baseline | no typos | 59 | 98.3% |
| baseline | typos | 73 | 97.3% |
| baseline-conf | no typos | 173 | 96.0% |
| baseline-conf | typos | 187 | 95.2% |
| baseline-2 | no typos | 202 | 97.0% |
| baseline-2 | typos | 201 | 93.5% |

### Top confusions: jev

| Expected | Predicted | Count |
| --- | --- | --- |
| customer_service | human_agent | 4 |
| request_right_to_rectification | change_account | 3 |
| exchange_product | exchange_product_in_store | 2 |
| payment_methods | pay | 2 |
| return_policy | return_product_in_store | 2 |
| return_product_in_store | availability_in_store | 2 |
| damaged_delivery | product_issue | 1 |
| return_product | return_product_online | 1 |

### Top confusions: baseline

| Expected | Predicted | Count |
| --- | --- | --- |
| return_policy | return_product_in_store | 2 |
| missing_item | delivery_issue | 1 |

### Top confusions: baseline-conf

| Expected | Predicted | Count |
| --- | --- | --- |
| request_right_to_rectification | change_account | 3 |
| customer_service | human_agent | 2 |
| return_policy | return_product_in_store | 2 |
| return_product_in_store | store_location | 2 |
| damaged_delivery | product_issue | 1 |
| missing_item | delivery_issue | 1 |
| payment_methods | pay | 1 |
| return_product | return_product_online | 1 |

### Top confusions: baseline-2

| Expected | Predicted | Count |
| --- | --- | --- |
| request_right_to_rectification | change_account | 5 |
| return_product_in_store | product_information | 3 |
| submit_product_idea | submit_feedback | 3 |
| return_policy | return_product_in_store | 2 |
| pay | payment_methods | 1 |
| return_policy | return_product | 1 |
| return_product | return_policy | 1 |
| return_product | return_product_online | 1 |
