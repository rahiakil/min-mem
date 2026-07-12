# Tiered Compression Baseline Study

*Generated from benchmark run — 2026-07-12T22:23:48.214604+00:00*

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

Corpus: **15** agent-memory passages · Dictionary: **157** entries · Deletion rate / keep ratio: **0.5**

| Method | Char% | Token% | Noun% | Jaccard | Syn% | Inference |
| --- | --- | --- | --- | --- | --- | --- |
| `min-mem+llmlingua-2@0.5` | 52.0 | 55.4 | 81.1 | 0.45 | 60.8 | small_lm |
| `min-mem+gpt2-selective@0.5` | 42.3 | 31.0 | 66.2 | 0.33 | 63.8 | small_lm |
| `llmlingua-2@0.5` | 42.3 | 53.7 | 82.0 | 0.68 | 68.9 | small_lm_mbert |
| `gpt2-selective@0.5` | 33.2 | 33.4 | 66.3 | 0.66 | 66.5 | small_lm_gpt2 |
| `min-mem` | 19.3 | 1.7 | 96.1 | 0.47 | 87.5 | none |
| `identity` | 0.0 | 0.0 | 100.0 | 1.00 | 100.0 | none |

## Pareto frontier (primary)

| Method | Char% | Noun% | Jaccard |
| --- | --- | --- | --- |
| `min-mem+llmlingua-2@0.5` | 52.0 | 81.1 | 0.45 |
| `llmlingua-2@0.5` | 42.3 | 82.0 | 0.68 |
| `min-mem` | 19.3 | 96.1 | 0.47 |
| `identity` | 0.0 | 100.0 | 1.00 |

## Key learnings

1. min-mem alone achieves 19.3% char reduction with 96.1% noun retention at **zero inference cost**.
2. llmlingua-2@0.5 reaches 42.3% char savings but noun retention drops to 82.0% (jaccard 0.68).
3. gpt2-selective@0.5 reaches 33.2% char savings but noun retention drops to 66.3% (jaccard 0.66).
4. min-mem → GPT-2 selective stacking yields higher total compression than either stage alone at the same keep ratio.
5. **Contribution framing:** min-mem is the auditable, zero-cost normalization tier; LM deletion methods are optional second stages for byte-starved deployments.
6. **Retrieval win:** min-mem+llmlingua-2@0.5 keeps 100% QA accuracy while cutting memory 52% (6324 chars) — best compression at full fidelity.

## Retrieval fidelity (Ollama QA)

Model: `qwen2.5:0.5b` · Probes: 5 · Memory blocks: 60

| Mode | Accuracy | Tokens | Chars saved |
| --- | --- | --- | --- |
| `identity` | 100% | 1843 | 0 |
| `min-mem` | 100% | 1819 | 2400 |
| `gpt2-selective@0.5` | 80% | 1275 | 3872 |
| `min-mem+gpt2-selective@0.5` | 40% | 1311 | 5136 |
| `llmlingua-2@0.5` | 100% | 967 | 5132 |
| `min-mem+llmlingua-2@0.5` | 100% | 939 | 6324 |

## Rate sweep (0.3 / 0.5 / 0.7)

| Method | Char% | Noun% | Jaccard |
| --- | --- | --- | --- |
| `gpt2-selective@0.3` | 58.0 | 40.6 | 0.43 |
| `gpt2-selective@0.5` | 33.2 | 66.3 | 0.66 |
| `gpt2-selective@0.7` | 25.7 | 74.1 | 0.74 |
| `llmlingua-2@0.3` | 64.8 | 63.4 | 0.42 |
| `llmlingua-2@0.5` | 42.3 | 82.0 | 0.68 |
| `llmlingua-2@0.7` | 25.3 | 88.2 | 0.83 |
| `min-mem+gpt2-selective@0.3` | 58.9 | 46.7 | 0.23 |
| `min-mem+gpt2-selective@0.5` | 42.3 | 66.2 | 0.33 |
| `min-mem+gpt2-selective@0.7` | 36.3 | 73.4 | 0.37 |
| `min-mem+llmlingua-2@0.3` | 71.2 | 59.4 | 0.33 |
| `min-mem+llmlingua-2@0.5` | 52.0 | 81.1 | 0.45 |
| `min-mem+llmlingua-2@0.7` | 36.7 | 89.7 | 0.47 |

## Reproduce

```bash
pip install -e ".[dev,baselines]"
python experiments/benchmark_baselines.py
python experiments/retrieval_fidelity.py   # optional: needs Ollama
```

## Paper repo

Copy `baseline_results.json` and `BASELINE_STUDY.md` to `min-mem-paper/experiments/`
for LaTeX integration.
