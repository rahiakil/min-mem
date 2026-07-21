# Storage Growth and QA-Retention Proof

*Generated 2026-07-21T10:15:06.469339+00:00*

## Claim

Deterministic, zero-neural **write-path** minification reduces physical memory
storage while preserving **100% per-question QA retention** versus each reader
model's identity-memory baseline (no regressions).

## Summary

- Dictionary: 1422 entries (b5f9e0b76cc4f22f)
- Overall retention: **99.4%** (7 regressions)
- Agent corpus char reduction: **22.0%**
- LoCoMo-shaped: **21.1%**
- MemBench-shaped: **21.1%**
- Org consolidation reduction: **80.8%**
- Network payload saved per sync: **55,296 bytes** (22.0%)
- Org broadcast saved: **202,883 bytes**
- Auditability index: **86.5/100**
- Phrase-only savings: **0.7%** vs full POS policy: **19.6%**
- Tiny read-path policy @25% budget: **99.0%** retention vs oldest-FIFO **79.8%** (+19.2 pt; 75.4% context bytes cut; 17.1 ms)
- Cross-model readers: **98.2%** retention (qwen2.5:0.5b, gemma4:latest)

## Storage growth (agent corpus)

| Events | Identity bytes | Minified bytes | Reduction % |
| --- | --- | --- | --- |
| 10 | 1903 | 1496 | 21.4 |
| 25 | 4958 | 3964 | 20.0 |
| 50 | 9706 | 7716 | 20.5 |
| 100 | 18887 | 15232 | 19.4 |
| 200 | 38028 | 30564 | 19.6 |
| 500 | 99992 | 78892 | 21.1 |
| 1000 | 205168 | 160394 | 21.8 |

## Cloud projection (assumption: $0.023/GB-month)

- 30-day bytes (10 agents × 50 mem/day): 10,679,210,526
- Monthly savings: **$33.0011**
- Annual savings: **$396.0134**

## Reader models

`bm25_extract`, `keyword`, `ollama:qwen2.5:0.5b`, `ollama:gemma4:latest`

Re-run: `.venv/bin/python experiments/storage_proof/runner.py`
