# Tiered compression value waterfall

*Generated 2026-07-21T20:12:05.204209+00:00*

Corpus: **304** passages × **1** = **304** memory blocks · Dictionary: **1422** entries

## Waterfall (cumulative char reduction)

| Stage | Char ↓ | Δ vs prev | Noun% | Throughput | Inference |
| --- | --- | --- | --- | --- | --- |
| `identity` | 0.0% | +0.0% | 100.0% | 53 pass/s | none |
| `min-mem (pass 1)` | 21.9% | +21.9% | 87.5% | 50 pass/s | none |
| `min-mem (pass 2)` | 22.0% | +0.1% | 87.5% | 38 pass/s | none |
| `llmlingua-2@0.5 (alone)` | 45.9% | +23.9% | 82.2% | 1 pass/s | small_lm_mbert |
| `min-mem → llmlingua-2@0.5` | 55.0% | +9.0% | 77.1% | 1 pass/s | small_lm |
| `min-mem×2 → llmlingua-2@0.5` | 55.0% | +0.1% | 77.0% | 1 pass/s | small_lm |
| `gpt2-selective@0.5 (alone)` | 33.9% | +-21.2% | 65.4% | 3 pass/s | small_lm_gpt2 |
| `min-mem → gpt2-selective@0.5` | 45.2% | +11.3% | 60.4% | 3 pass/s | small_lm |
| `min-mem×2 → gpt2-selective@0.5` | 45.2% | +0.1% | 60.4% | 3 pass/s | small_lm |

## min-mem value-add before deletion

- **min-mem pre-step before LLMLingua**
  - min_mem_alone_pct: 21.95
  - llmlingua_alone_pct: 45.93
  - stacked_pct: 54.95
  - min_mem_lift_on_stack: 9.02
- **2nd min-mem pass before LLMLingua**
  - pass2_alone_pct: 22.01
  - stacked_pct: 55.01
  - pass2_lift_on_stack: 0.06

## Reproduce

```bash
.venv/bin/python experiments/tiered_value_benchmark.py
```
