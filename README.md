<div align="center">

# min-mem

**Shrink LLM agent memory without losing meaning.**

[![CI](https://github.com/rahiakil/min-mem/actions/workflows/ci.yml/badge.svg)](https://github.com/rahiakil/min-mem/actions/workflows/ci.yml)
[![GitHub Pages](https://github.com/rahiakil/min-mem/actions/workflows/pages.yml/badge.svg)](https://rahiakil.github.io/min-mem/)
[![PyPI](https://img.shields.io/pypi/v/min-mem?label=PyPI&color=blue)](https://pypi.org/project/min-mem/)
[![Python](https://img.shields.io/pypi/pyversions/min-mem)](https://pypi.org/project/min-mem/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Live demo](https://rahiakil.github.io/min-mem/) · [Quick start](#quick-start) · [Benchmarks](#benchmarks) · [Contributing](CONTRIBUTING.md)

</div>

---

Agent memory is natural language — and it is often **needlessly long**. `utilize`, `in order to`, `nevertheless` carry the same facts as `use`, `to`, `yet`. **min-mem** rewrites memory with a curated synonym dictionary while **POS tagging guarantees nouns are never touched**.

> Not summarization. Not gzip. **Lexical minification** — shorter words, same entities, human-readable text.

---

## At a glance

| | Before | After min-mem |
|---|--------|---------------|
| **Chars** | 148 avg / passage | **−19.3%** |
| **Noun tags preserved** | 100% baseline | **96.1%** |
| **Naive substitute-all** | −27.0% chars | **65.8%** noun retention ⚠️ |
| **Dictionary** | — | **157 entries** (7 phrases + 150 words) |
| **Agent context (60 bullets)** | baseline prompt | **−2,400 chars**, retrieval parity |

---

## How it works

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    LLM Agent Memory Write Path                              │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────┐    ┌────────────────────┐    ┌─────────────────────────┐
│  sessionStart │───▶│  min_dict.json      │───▶│  Phrase pass (longest    │
│  Cursor hook  │    │  ~/.config/min-mem  │    │  first) → word pass      │
└───────────────┘    └────────────────────┘    └────────────┬────────────────┘
                                                            │
                                                            ▼
                                               ┌─────────────────────────┐
                                               │  NLTK POS tagger         │
                                               │  NN/NNS/NNP/NNPS → SKIP  │
                                               └────────────┬────────────────┘
                                                            │
        ┌───────────────────────────────────────────────────┘
        ▼
┌───────────────┐    ┌────────────────────┐    ┌─────────────────────────┐
│  Persist T′   │───▶│  Smaller LLM       │───▶│  min-mem measure        │
│  (minified)   │    │  context window    │    │  chars · tokens · swaps  │
└───────────────┘    └────────────────────┘    └─────────────────────────┘
```

**Pipeline (deterministic):**

1. **Phrases** — multi-word keys applied longest-first (`in order to` → `to`)
2. **POS gate** — nouns and named entities never substituted
3. **Inflection** — `required` → `needed`, `utilize` → `use`
4. **Persist** — plain English string, auditable in git

---

## Quick start

### Install from PyPI

```bash
pip install min-mem
min-mem init          # downloads dictionary + NLTK data + Cursor hook
min-mem doctor        # verify everything is ready
```

### One-liner

```bash
min-mem minify "The user prefers to utilize Python in order to accomplish data tasks."
# → The user prefers to use Python to do data tasks.
```

### Agent hook (Cursor)

`min-mem init` installs a **sessionStart** hook that loads the dictionary at the beginning of every agent session:

```
.cursor/hooks.json          → sessionStart → min_mem_session.sh
~/.config/min-mem/          → min_dict.json (157 entries)
~/.config/min-mem/metrics.json → cumulative savings across sessions
```

Re-run init in any project:

```bash
cd your-agent-project
min-mem init
```

### Python API

```python
from min_mem import MinMemConverter, MemoryMinifier

# Single block
converter = MinMemConverter()
result = converter.minify(
    "The user previously utilized Python to investigate numerous libraries."
)
print(result.minified)       # shorter text
print(result.savings_ratio)  # 0.19

# Agent memory loop with cumulative metrics
minifier = MemoryMinifier()
blocks = [
    "User prefers to utilize TypeScript in order to build APIs.",
    "Project uses PostgreSQL for persistent entity storage.",
]
stored = minifier.minify_many(blocks)
print(minifier.report())     # chars_saved, savings_pct
```

### Measure savings

```bash
min-mem measure -f memory.txt
min-mem measure -f memory.txt --json   # for CI / dashboards
```

---

## Example

<table>
<tr><th>Input</th><th>Output</th></tr>
<tr>
<td>

```
The user prefers to utilize Python in order to
accomplish data analysis tasks. However, they
previously required additional libraries to
facilitate the workflow.
```

</td>
<td>

```
The user prefers to use Python to do data
analysis tasks. But, they before needed also
libraries to aid the workflow.
```

</td>
</tr>
</table>

**Preserved:** `Python`, `libraries`, `workflow` (nouns)  
**Shortened:** `utilize→use`, `in order to→to`, `However→But`, `facilitate→aid`

---

## Benchmarks

Reproducible on a **15-passage agent-memory corpus** (prefs, sessions, facts, instructions). Run locally:

```bash
git clone https://github.com/rahiakil/min-mem.git && cd min-mem
pip install -e ".[dev,experiments]"
make figures    # benchmark + charts → experiments/results.json
```

### Method comparison

| Method | Char reduction | Token reduction | Noun retention | Synonym-aware |
|--------|---------------|-----------------|----------------|---------------|
| **min-mem (full)** | **19.3%** | 1.7% | **96.1%** | 87.5% |
| naive-dict (no POS) | 27.0% | 1.1% | 65.8% ⚠️ | 87.2% |
| phrase-only | 1.7% | 2.3% | 98.2% | 97.3% |
| no-inflection | 17.2% | 2.5% | 97.7% | 93.1% |
| gzip (bytes only) | 19.8% | — | — | — |

**Takeaway:** POS gating trades ~8 pts of byte savings for **30 pts of noun safety**. Byte savings beat token savings under `cl100k_base` — storage-priced backends benefit most.

### By memory category

| Category | Char savings |
|----------|-------------|
| High verbosity | **29.0%** |
| User preferences | 26.2% |
| Project context | 19.1% |
| Entity-heavy | 14.8% |

### Agent context demo

60 memory bullets + skills block on CPU (Ollama `qwen2.5:0.5b`):

- **2,400 characters** removed from memory section
- **24 prompt tokens** saved
- **2/2** factual probes correct (verbose vs minified)

Charts and interactive demo: **[rahiakil.github.io/min-mem](https://rahiakil.github.io/min-mem/)**

### Tiered compression vs external baselines

min-mem is the **zero-inference, auditable first tier**. Optional second tiers use small CPU LMs (GPT-2 selective pruning, LLMLingua-2):

```bash
pip install -e ".[dev,baselines]"
make baselines          # → experiments/baseline_results.json
make retrieval          # Ollama QA fidelity across modes
```

| Mode | Char ↓ | QA accuracy (60 blocks) | Inference |
|------|--------|-------------------------|-----------|
| min-mem | 19% | **100%** | none |
| llmlingua-2@0.5 | 42% | **100%** | small LM |
| **min-mem → llmlingua-2** | **52%** | **100%** | small LM |

Full study: [`experiments/BASELINE_STUDY.md`](experiments/BASELINE_STUDY.md)

---

## CLI reference

| Command | Description |
|---------|-------------|
| `min-mem init` | Install dictionary, NLTK data, Cursor hook |
| `min-mem doctor` | Health check (`--json` for automation) |
| `min-mem minify` | Minify text, file, or stdin (`-v`, `--json`) |
| `min-mem stats` | Compression stats only |
| `min-mem measure` | Benchmark a memory file |
| `min-mem dict` | Inspect dictionary (`--count`) |

```bash
# Custom dictionary
export MIN_MEM_DICT=/path/to/my_dict.json
min-mem minify -f agent_memory.md
```

---

## Dictionary

`min_dict.json` — **157 entries**, grown via corpus-driven curation loop.

| Long | → | Short |
|------|---|-------|
| utilize | → | use |
| in order to | → | to |
| nevertheless | → | yet |
| investigate | → | check |
| numerous | → | many |
| facilitate | → | aid |

**Rules:** phrases before words · longest-first · **nouns never replaced** · conservative synonyms only

Extend the dictionary via PR — see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Project layout

```
min-mem/
├── src/min_mem/
│   ├── converter.py      # POS-gated minification
│   ├── dictionary.py     # Dictionary loader (bundled + user)
│   ├── bootstrap.py      # init, doctor, metrics
│   ├── agent.py          # MemoryMinifier for agent loops
│   ├── hooks/            # Cursor sessionStart integration
│   └── data/min_dict.json
├── .cursor/hooks.json    # Agent hook (installed by init)
├── experiments/          # Public benchmarks + GitHub Pages data
├── agents/               # Minimal Ollama demo agent
├── docs/                 # GitHub Pages site
└── tests/
```

---

## Development

```bash
pip install -e ".[dev,experiments]"
pytest -v
make benchmark    # run comparative study
make agent        # CPU LLM context demo (needs Ollama)
```

### Publish to PyPI

```bash
pip install build twine
python -m build
twine upload dist/*
```

---

## Limits

- English only (NLTK POS tagging)
- Lexical swaps only — no sentence deletion or restructuring
- Register may shift slightly (`however` → `but`)
- `expand()` is best-effort inverse

---

## Contribute

We welcome dictionary entries, benchmarks, and integrations.

1. Fork → branch → PR (branch protection on `main`)
2. Add synonyms to `min_dict.json` — **non-nouns only**
3. `pytest` must pass

[CONTRIBUTING.md](CONTRIBUTING.md) · [Code of Conduct](CODE_OF_CONDUCT.md)

---

<div align="center">

**Built for agents that remember too much.**

[⭐ Star on GitHub](https://github.com/rahiakil/min-mem) · [Report an issue](https://github.com/rahiakil/min-mem/issues)

</div>
