# min-mem

[![CI](https://github.com/rahiakil/min-mem/actions/workflows/ci.yml/badge.svg)](https://github.com/rahiakil/min-mem/actions/workflows/ci.yml)
[![GitHub Pages](https://github.com/rahiakil/min-mem/actions/workflows/pages.yml/badge.svg)](https://github.com/rahiakil/min-mem/actions/workflows/pages.yml)

**🌐 [Live demo & experiments →](https://rahiakil.github.io/min-mem/)**

Shrink agent memory for LLM storage without losing meaning — replace longer words with shorter synonyms from a minimal dictionary. **Nouns are never changed.**

> **Contributing welcome** — we use PRs and branch protection on `main`. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Idea

LLM memory often repeats verbose phrasing (`utilize`, `in order to`, `nevertheless`). Many of these have shorter equivalents (`use`, `to`, `yet`) that carry the same semantic knowledge. This project:

1. Maintains a **minimal dictionary** (`min_dict.json`) of longer → shorter forms
2. **Minifies** arbitrary memory text using that dictionary
3. Uses **POS tagging** to skip nouns, so entities and concepts stay intact

This is not lossy compression — it is lexical normalization toward shorter, equivalent wording.

## Install

```bash
pip install -e ".[dev]"
```

On first run, NLTK tokenizer/tagger data is downloaded automatically.

## Usage

### CLI

```bash
# Minify inline text
min-mem minify "The user prefers to utilize Python in order to accomplish data tasks."

# From a file, with stats on stderr
min-mem minify -f memory.txt -v

# JSON output (for pipelines)
min-mem minify --json -f memory.txt

# Inspect the dictionary
min-mem dict

# Compression stats only
min-mem stats -f memory.txt
```

### Python API

```python
from min_mem import MinMemConverter

converter = MinMemConverter()
result = converter.minify(
    "The user previously utilized Python to investigate numerous libraries."
)

print(result.minified)
# The user before used Python to check many libraries.

print(f"Saved {result.chars_saved} chars ({result.savings_ratio:.1%})")
for r in result.replacements:
    print(f"  {r.original} -> {r.replacement}")
```

## Dictionary

`min_dict.json` maps longer forms to their shortest practical equivalent:

| Long | Min |
|------|-----|
| utilize | use |
| in order to | to |
| nevertheless | yet |
| investigate | check |
| numerous | many |

Phrases are applied before single words (longest first). You can extend the dictionary without code changes.

**Rule:** nouns (`NN`, `NNS`, `NNP`, `NNPS`) are never replaced, even if they appear in the dictionary.

## Example

**Input:**
```
The user prefers to utilize Python in order to accomplish data analysis tasks.
However, they previously required additional libraries to facilitate the workflow.
```

**Output:**
```
The user prefers to use Python to do data analysis tasks.
But, they before needed also libraries to aid the workflow.
```

## Limits

- English only (NLTK POS tagging)
- Synonym choice is conservative — shortest word that preserves typical meaning
- Does not rewrite sentence structure or remove redundancy beyond dictionary swaps
- `expand()` provides a best-effort inverse (short → first long form in dict)

## Experiments

Reproducible benchmarks and CPU LLM agent demo:

```bash
pip install -e ".[dev,experiments]"

# Comparative study (5 methods, 15-sample corpus)
python experiments/run_benchmark.py
python experiments/generate_figures.py

# Agent + skills context reduction (requires Ollama + qwen2.5:0.5b)
python experiments/agent_context_demo.py
```

Results power the [GitHub Pages site](https://rahiakil.github.io/min-mem/).

## Project layout

```
min_dict.json          # minimal synonym dictionary
src/min_mem/           # converter, CLI
experiments/           # benchmarks + CPU LLM agent demo
docs/                  # GitHub Pages site
tests/
```

## Contribute

1. Fork → branch → PR
2. Dictionary entries: edit `min_dict.json` (non-nouns only)
3. `pytest` must pass — CI runs on every PR

See [CONTRIBUTING.md](CONTRIBUTING.md) · [Code of Conduct](CODE_OF_CONDUCT.md)

## Test

```bash
pytest
```
