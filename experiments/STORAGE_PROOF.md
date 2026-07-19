# Storage Growth and QA-Retention Proof

*Generated 2026-07-19T02:40:50.858742+00:00*

## Claim

Deterministic, zero-neural **write-path** minification reduces physical memory
storage while preserving **100% per-question QA retention** versus each reader
model's identity-memory baseline (no regressions).

## Summary

- Dictionary: 873 entries (eb141c022109e6fa)
- Overall retention: **97.4%** (2 regressions)
- Agent corpus char reduction: **21.3%**
- LoCoMo-shaped: **21.3%**
- MemBench-shaped: **21.3%**
- Org consolidation reduction: **80.3%**
- Network payload saved per sync: **2,568 bytes** (21.3%)
- Org broadcast saved: **9,693 bytes**
- Auditability index: **89.0/100**
- Phrase-only savings: **1.7%** vs full POS policy: **20.5%**
- Tiny read-path policy @25% budget: **100.0%** retention vs oldest-FIFO **37.5%** (+62.5 pt; 73.0% context bytes cut; 0.7 ms)
- Cross-model readers: **87.5%** retention (qwen2.5:0.5b, gemma4:latest)

## Storage growth (agent corpus)

| Events | Identity bytes | Minified bytes | Reduction % |
| --- | --- | --- | --- |
| 10 | 1903 | 1492 | 21.6 |
| 25 | 4920 | 3867 | 21.4 |
| 50 | 10091 | 7909 | 21.6 |

## Cloud projection (assumption: $0.023/GB-month)

- 30-day bytes (10 agents × 50 mem/day): 5,237,000
- Monthly savings: **$0.0003**
- Annual savings: **$0.0035**

## Reader models

`bm25_extract`, `keyword`, `ollama:qwen2.5:0.5b`, `ollama:gemma4:latest`

Re-run: `.venv/bin/python experiments/storage_proof/runner.py`
