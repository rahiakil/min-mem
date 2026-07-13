# Storage Growth and QA-Retention Proof

*Generated 2026-07-13T00:27:40.239798+00:00*

## Claim

Deterministic, zero-neural **write-path** minification reduces physical memory
storage while preserving **100% per-question QA retention** versus each reader
model's identity-memory baseline (no regressions).

## Summary

- Dictionary: 200 entries (33230ab11655b8e2)
- Overall retention: **100.0%** (0 regressions)
- Agent corpus char reduction: **20.4%**
- LoCoMo-shaped: **20.4%**
- MemBench-shaped: **20.4%**
- Org consolidation reduction: **80.1%**
- Cross-model readers: **100.0%** retention (`qwen2.5:0.5b`, `gemma4:latest`; 16 comparisons, 0 regressions)

## Storage growth (agent corpus)

| Events | Identity bytes | Minified bytes | Reduction % |
| --- | --- | --- | --- |
| 10 | 1903 | 1516 | 20.3 |
| 25 | 4920 | 3919 | 20.3 |
| 50 | 10091 | 8007 | 20.7 |

## Cloud projection (assumption: $0.023/GB-month)

- 30-day bytes (10 agents × 50 mem/day): 5,237,000
- Monthly savings: **$0.0003**
- Annual savings: **$0.0033**

## Reader models

`bm25_extract`, `keyword`, `ollama:qwen2.5:0.5b`, `ollama:gemma4:latest`

Re-run: `.venv/bin/python experiments/storage_proof/runner.py`
