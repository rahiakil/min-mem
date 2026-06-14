# Contributing to Min-Mem

Thanks for helping shrink agent memory for everyone. We use **pull requests** and **branch protection** on `main` — direct pushes are blocked.

## Quick start

```bash
git clone https://github.com/rahiakil/min-mem.git
cd min-mem
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,experiments]"
pytest
```

## How to contribute

### 1. Dictionary entries (most valuable)

Add long → short synonyms to `min_dict.json`:

- **Non-nouns only** (verbs, adjectives, adverbs, connectors, phrases)
- Shortest word that preserves typical meaning
- Include common phrases (`in order to` → `to`) for biggest wins

```json
"nevertheless": "yet"
```

### 2. Code improvements

- POS tagging accuracy
- Inflection handling
- New languages
- Token-optimized synonym selection

### 3. Experiments

- Extend `experiments/corpus.json` with real agent memory samples (anonymized)
- Add baselines to `experiments/run_benchmark.py`
- Improve `experiments/agent_context_demo.py`

### 4. Documentation & website

- `docs/` is the GitHub Pages site — PRs welcome for clarity and demos

## Pull request workflow

1. **Fork** the repo
2. **Branch** from `main`: `git checkout -b feat/my-contribution`
3. **Change** — keep diffs focused
4. **Test**: `pytest` and, if touching experiments, re-run benchmarks
5. **Open PR** against `main` — fill out the PR template
6. Wait for review — at least one approval required

## PR checklist

- [ ] `pytest` passes
- [ ] Dictionary changes don't add noun → noun swaps
- [ ] No secrets or private memory in corpus samples
- [ ] README or docs updated if user-facing behavior changed

## Code style

- Match existing Python style in `src/min_mem/`
- Minimal diffs — one concern per PR
- No unnecessary abstractions

## Community

- Be respectful (see CODE_OF_CONDUCT.md)
- Issues welcome for bugs, ideas, and dictionary proposals
- Ask in issues before large architectural changes

## Recognition

Contributors who add dictionary entries or experiment improvements will be credited in release notes. Help us make agent memory smaller — together.
