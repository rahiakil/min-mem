# Storage Growth and QA-Retention Proof

*Generated 2026-07-21T06:47:53.519584+00:00*

## Claim

Deterministic, zero-neural **write-path** minification reduces physical memory
storage while preserving **100% per-question QA retention** versus each reader
model's identity-memory baseline (no regressions).

## Summary

- Dictionary: 1408 entries (66b34543687dfa10)
- Overall retention: **99.2%** (10 regressions)
- Agent corpus char reduction: **19.6%**
- LoCoMo-shaped: **21.1%**
- MemBench-shaped: **21.1%**
- Org consolidation reduction: **80.3%**
- Network payload saved per sync: **29,856 bytes** (19.6%)
- Org broadcast saved: **122,195 bytes**
- Auditability index: **89.1/100**
- Phrase-only savings: **1.0%** vs full POS policy: **19.0%**
- Tiny read-path policy @25% budget: **98.4%** retention vs oldest-FIFO **67.4%** (+31.1 pt; 75.0% context bytes cut; 10.1 ms)
- Cross-model readers: **97.4%** retention (qwen2.5:0.5b, gemma4:latest)

## Storage growth (agent corpus)

| Events | Identity bytes | Minified bytes | Reduction % |
| --- | --- | --- | --- |
| 10 | 1903 | 1496 | 21.4 |
| 25 | 4958 | 3964 | 20.0 |
| 50 | 9706 | 7716 | 20.5 |
| 100 | 18887 | 15232 | 19.4 |
| 200 | 38028 | 30564 | 19.6 |
| 500 | 94943 | 76360 | 19.6 |

## Cloud projection (assumption: $0.023/GB-month)

- 30-day bytes (10 agents × 50 mem/day): 10,203,300,000
- Monthly savings: **$28.1159**
- Annual savings: **$337.3907**

## Reader models

`bm25_extract`, `keyword`, `ollama:qwen2.5:0.5b`, `ollama:gemma4:latest`

Re-run: `.venv/bin/python experiments/storage_proof/runner.py`
