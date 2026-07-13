# Tiered compression value waterfall

*Generated 2026-07-12T23:22:04.681780+00:00*

Corpus: **15** passages × **4** = **60** memory blocks · Dictionary: **200** entries

## Waterfall (cumulative char reduction)

| Stage | Char ↓ | Δ vs prev | Noun% | Throughput | Inference |
| --- | --- | --- | --- | --- | --- |
| `identity` | 0.0% | +0.0% | 100.0% | 46 pass/s | none |
| `min-mem (pass 1)` | 20.1% | +20.1% | 96.1% | 70 pass/s | none |
| `min-mem (pass 2)` | 20.4% | +0.2% | 96.1% | 60 pass/s | none |
| `llmlingua-2@0.5 (alone)` | 42.5% | +22.2% | 82.0% | 1 pass/s | small_lm_mbert |
| `min-mem → llmlingua-2@0.5` | 52.5% | +10.0% | 81.1% | 1 pass/s | small_lm |
| `min-mem×2 → llmlingua-2@0.5` | 52.6% | +0.1% | 81.1% | 1 pass/s | small_lm |
| `gpt2-selective@0.5 (alone)` | 32.1% | +-20.5% | 66.3% | 3 pass/s | small_lm_gpt2 |
| `min-mem → gpt2-selective@0.5` | 42.7% | +10.6% | 66.2% | 3 pass/s | small_lm |
| `min-mem×2 → gpt2-selective@0.5` | 43.0% | +0.2% | 66.2% | 3 pass/s | small_lm |

## min-mem value-add before deletion

- **min-mem pre-step before LLMLingua**
  - min_mem_alone_pct: 20.12
  - llmlingua_alone_pct: 42.53
  - stacked_pct: 52.54
  - min_mem_lift_on_stack: 10.01
- **2nd min-mem pass before LLMLingua**
  - pass2_alone_pct: 20.35
  - stacked_pct: 52.6
  - pass2_lift_on_stack: 0.06

## Reproduce

```bash
.venv/bin/python experiments/tiered_value_benchmark.py
```
