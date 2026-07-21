# Tiered Compression Baseline Study

*Generated from benchmark run — 2026-07-21T19:20:58.902595+00:00*

## Research question

Can **auditable, zero-inference lexical minification** (min-mem) serve as a
first tier that improves or matches downstream retrieval when combined with
LM-based deletion compressors (Selective Context, LLMLingua-2)?

## Contribution framing

| Tier | Method | Inference cost | Auditable |
|------|--------|----------------|-----------|
| 0 | identity | none | yes |
| 1 | **min-mem** | **none** | **yes** (deterministic dict) |
| 2 | gpt2-selective / llmlingua-2 | small LM on CPU | partial |
| 3 | min-mem → tier 2 | small LM on CPU | pre-step yes |

## Primary comparison

Corpus: **304** agent-memory passages · Dictionary: **1422** entries · Deletion rate / keep ratio: **0.5**

| Method | Char% | Token% | Noun% | Jaccard | Syn% | Inference |
| --- | --- | --- | --- | --- | --- | --- |
| `min-mem×2+llmlingua-2@0.5` | 54.8 | 55.1 | 77.0 | 0.42 | 59.9 | small_lm |
| `min-mem+llmlingua-2@0.5` | 54.8 | 55.0 | 77.1 | 0.42 | 60.0 | small_lm |
| `llmlingua-2@0.5` | 45.8 | 54.0 | 82.2 | 0.67 | 66.9 | small_lm_mbert |
| `min-mem×2+gpt2-selective@0.5` | 44.8 | 31.5 | 60.4 | 0.29 | 60.9 | small_lm |
| `min-mem+gpt2-selective@0.5` | 44.7 | 31.6 | 60.4 | 0.29 | 61.0 | small_lm |
| `gpt2-selective@0.5` | 33.6 | 34.6 | 65.4 | 0.66 | 66.4 | small_lm_gpt2 |
| `min-mem×2` | 21.3 | 1.2 | 87.5 | 0.40 | 86.6 | none |
| `min-mem` | 21.3 | 1.2 | 87.5 | 0.40 | 86.7 | none |
| `identity` | 0.0 | 0.0 | 100.0 | 1.00 | 100.0 | none |

## Pareto frontier (primary)

| Method | Char% | Noun% | Jaccard |
| --- | --- | --- | --- |
| `min-mem×2+llmlingua-2@0.5` | 54.8 | 77.0 | 0.42 |
| `min-mem+llmlingua-2@0.5` | 54.8 | 77.1 | 0.42 |
| `llmlingua-2@0.5` | 45.8 | 82.2 | 0.67 |
| `min-mem×2` | 21.3 | 87.5 | 0.40 |
| `min-mem` | 21.3 | 87.5 | 0.40 |
| `identity` | 0.0 | 100.0 | 1.00 |

## Key learnings

1. min-mem alone achieves 21.3% char reduction with 87.5% noun retention at **zero inference cost**.
2. llmlingua-2@0.5 reaches 45.8% char savings but noun retention drops to 82.2% (jaccard 0.67).
3. gpt2-selective@0.5 reaches 33.6% char savings but noun retention drops to 65.4% (jaccard 0.66).
4. Tiered min-mem → LLMLingua-2 compounds to **54.8%** char savings vs 45.8% for deletion alone.
5. min-mem → GPT-2 selective stacking yields higher total compression than either stage alone at the same keep ratio.
6. **Contribution framing:** min-mem is the auditable, zero-cost normalization tier; LM deletion methods are optional second stages for byte-starved deployments.
7. **Retrieval win:** `min-mem+llmlingua-2@0.5` keeps 100% QA accuracy while cutting memory 53% (6352 chars) — best compression at full fidelity.

## Retrieval fidelity (Ollama QA)

Model: `qwen2.5:0.5b` · Probes: 5 · Memory blocks: 60

| Mode | Accuracy | Tokens | Chars saved |
| --- | --- | --- | --- |
| `identity` | 100% | 1843 | 0 |
| `min-mem` | 100% | 1807 | 2540 |
| `min-mem×2` | 100% | 1807 | 2552 |
| `gpt2-selective@0.5` | 80% | 1275 | 3872 |
| `min-mem+gpt2-selective@0.5` | 40% | 1331 | 5036 |
| `min-mem×2+gpt2-selective@0.5` | 40% | 1331 | 5048 |
| `llmlingua-2@0.5` | 100% | 967 | 5132 |
| `min-mem+llmlingua-2@0.5` | 100% | 935 | 6352 |
| `min-mem×2+llmlingua-2@0.5` | 100% | 935 | 6352 |

## Reproduce

```bash
pip install -e ".[dev,baselines]"
python experiments/benchmark_baselines.py
python experiments/retrieval_fidelity.py   # optional: needs Ollama
```

## Paper repo

Copy `baseline_results.json` and `BASELINE_STUDY.md` to `min-mem-paper/experiments/`
for LaTeX integration.
