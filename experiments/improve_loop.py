#!/usr/bin/env python3
"""Iteratively grow min_dict.json until benchmarks plateau.

Each iteration:
  1. Mine corpus for high-value candidate synonyms not yet in dictionary
  2. Add batch that passes safety checks (noun preservation, no regression)
  3. Re-run benchmark metrics + minimal agent context stats
  4. Log to experiments/improve_history.json
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "agents"))

from min_mem import MinMemConverter  # noqa: E402
from min_mem.dictionary import NOUN_TAGS, MinDictionary  # noqa: E402
from dictionary_tiers import load_entries  # noqa: E402

try:
    from nltk import pos_tag, word_tokenize
    from min_mem.converter import _ensure_nltk_data
except ImportError:
    pos_tag = None  # type: ignore

CORPUS = ROOT / "experiments" / "corpus.json"
DICT_PATH = ROOT / "min_dict.json"
CANDIDATES = ROOT / "experiments" / "candidates.json"
HISTORY = ROOT / "experiments" / "improve_history.json"
RESULTS = ROOT / "experiments" / "results.json"

MAX_ITERATIONS = 12
BATCH_SIZE = 8
MIN_CHAR_IMPROVEMENT = 0.15  # stop if gain below this % for 2 rounds
PLATEAU_ROUNDS = 2
MIN_NOUN_PRESERVATION = 94.0

# Entries that cause bad inflections — never add via loop
BLOCKLIST = {"implement", "implementation", "implementations", "establish", "contains"}


@dataclass
class IterationResult:
    iteration: int
    dict_size: int
    char_savings_pct: float
    token_savings_pct: float
    noun_preservation_pct: float
    memory_reduction_pct: float
    agent_tokens_saved: int
    added_entries: list[str]
    timestamp: str


def load_corpus() -> list[str]:
    return [s["text"] for s in json.loads(CORPUS.read_text())["samples"]]


def save_dict(entries: dict[str, str]) -> None:
    data = {
        "_meta": {
            "description": "Minimal synonym map: longer form -> shortest equivalent. Nouns are never replaced at runtime (POS-gated).",
            "version": "auto-improved",
            "last_improved": datetime.now(timezone.utc).isoformat(),
        },
        "entries": dict(sorted(entries.items())),
    }
    DICT_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def noun_preservation_score(corpus: list[str], converter: MinMemConverter) -> float:
    _ensure_nltk_data()
    total, preserved = 0, 0
    for text in corpus:
        r = converter.minify(text)
        orig = [t.lower() for t, tag in pos_tag(word_tokenize(text)) if tag in NOUN_TAGS]
        new = [t.lower() for t, tag in pos_tag(word_tokenize(r.minified)) if tag in NOUN_TAGS]
        total += len(orig)
        preserved += sum(1 for n in orig if n in new)
    return 100.0 * preserved / total if total else 100.0


def benchmark_quick(corpus: list[str], entries: dict[str, str]) -> dict:
    converter = MinMemConverter(MinDictionary.from_dict(entries))
    char_savings, token_savings = [], []
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tok = lambda s: len(enc.encode(s))
    except ImportError:
        tok = lambda s: len(s.split())

    for text in corpus:
        r = converter.minify(text)
        char_savings.append(100.0 * r.chars_saved / len(text) if text else 0)
        token_savings.append(100.0 * (tok(text) - tok(r.minified)) / tok(text) if tok(text) else 0)

    n = len(corpus)
    mem_texts = corpus
    min_texts = [converter.minify(t).minified for t in corpus]
    raw_len = sum(len(t) for t in mem_texts)
    min_len = sum(len(t) for t in min_texts)

    from minimal_agent import MinimalAgent, AgentConfig

    agent = MinimalAgent(memories=corpus, config=AgentConfig(minify_memory=True))
    agent_v = MinimalAgent(memories=corpus, config=AgentConfig(minify_memory=False))
    cmp = agent.compare_context()

    return {
        "char_savings_pct_mean": sum(char_savings) / n,
        "token_savings_pct_mean": sum(token_savings) / n,
        "noun_preservation_pct": noun_preservation_score(corpus, converter),
        "memory_reduction_pct": 100.0 * (raw_len - min_len) / raw_len if raw_len else 0,
        "agent_tokens_saved": cmp["tokens_saved"],
        "agent_chars_saved": cmp["chars_saved"],
        "dict_size": len(entries),
    }


def score_candidate(word: str, short: str, corpus: list[str], entries: dict[str, str]) -> float:
    """Expected char savings: frequency in corpus × char delta."""
    if word in entries or word in BLOCKLIST:
        return 0.0
    delta = max(len(word) - len(short), 0)
    if delta <= 0:
        return 0.0
    freq = 0
    pattern = re.compile(re.escape(word), re.IGNORECASE)
    for text in corpus:
        freq += len(pattern.findall(text))
    return freq * delta


def mine_corpus_words(corpus: list[str], entries: dict[str, str]) -> dict[str, str]:
    """Discover long non-noun tokens in corpus; propose crude shorts."""
    _ensure_nltk_data()
    from collections import Counter

    freq: Counter[str] = Counter()
    for text in corpus:
        for token, tag in pos_tag(word_tokenize(text)):
            w = token.lower()
            if len(w) < 8 or tag in NOUN_TAGS or w in entries or w in BLOCKLIST:
                continue
            if not w.isalpha():
                continue
            freq[w] += 1

    mined: dict[str, str] = {}
    suffix_cuts = [
        ("ization", "ing"),
        ("isation", "ing"),
        ("ation", "ing"),
        ("ment", ""),
        ("ness", ""),
        ("ingly", "ing"),
        ("ally", "ly"),
        ("ously", "ly"),
        ("fully", "ly"),
    ]
    for word, count in freq.most_common(40):
        if count < 1:
            continue
        for suf, rep in suffix_cuts:
            if word.endswith(suf) and len(word) - len(suf) + len(rep) >= 4:
                short = word[: -len(suf)] + rep
                if short != word and len(short) < len(word):
                    mined[word] = short
                break
    return mined


def pick_candidates(corpus: list[str], entries: dict[str, str], n: int) -> list[tuple[str, str]]:
    pool = json.loads(CANDIDATES.read_text())["candidates"]
    pool.update(mine_corpus_words(corpus, entries))
    scored: list[tuple[float, str, str]] = []
    for long, short in pool.items():
        if long in BLOCKLIST:
            continue
        s = score_candidate(long, short, corpus, entries)
        if s > 0:
            scored.append((s, long, short))
    scored.sort(reverse=True)
    return [(l, sh) for _, l, sh in scored[: n * 3]]  # try more than batch


def try_add_entries(
    entries: dict[str, str],
    candidates: list[tuple[str, str]],
    corpus: list[str],
    batch_size: int,
) -> tuple[dict[str, str], list[str]]:
    baseline = benchmark_quick(corpus, entries)
    added: list[str] = []
    current = deepcopy(entries)

    for long, short in candidates:
        if len(added) >= batch_size:
            break
        if long in current:
            continue
        trial = deepcopy(current)
        trial[long] = short
        metrics = benchmark_quick(corpus, trial)
        if (
            metrics["char_savings_pct_mean"] >= baseline["char_savings_pct_mean"]
            and metrics["noun_preservation_pct"] >= MIN_NOUN_PRESERVATION
        ):
            current = trial
            added.append(f"{long} → {short}")
            baseline = metrics

    return current, added


def run_full_benchmark() -> None:
    subprocess.run([sys.executable, str(ROOT / "experiments" / "run_benchmark.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "experiments" / "generate_figures.py")], check=True)


def main() -> None:
    corpus = load_corpus()
    entries = load_entries(DICT_PATH)
    history: list[dict] = []
    if HISTORY.exists():
        history = json.loads(HISTORY.read_text())

    plateau = 0
    prev_char = 0.0

    print(f"Starting dictionary: {len(entries)} entries")
    baseline = benchmark_quick(corpus, entries)
    print(json.dumps(baseline, indent=2))

    for i in range(1, MAX_ITERATIONS + 1):
        candidates = pick_candidates(corpus, entries, BATCH_SIZE)
        if not candidates:
            print("No more candidates.")
            break

        new_entries, added = try_add_entries(entries, candidates, corpus, BATCH_SIZE)
        if not added:
            print(f"Iter {i}: no safe additions.")
            plateau += 1
            if plateau >= PLATEAU_ROUNDS:
                break
            continue

        entries = new_entries
        save_dict(entries)
        metrics = benchmark_quick(corpus, entries)

        result = IterationResult(
            iteration=i,
            dict_size=metrics["dict_size"],
            char_savings_pct=round(metrics["char_savings_pct_mean"], 2),
            token_savings_pct=round(metrics["token_savings_pct_mean"], 2),
            noun_preservation_pct=round(metrics["noun_preservation_pct"], 2),
            memory_reduction_pct=round(metrics["memory_reduction_pct"], 2),
            agent_tokens_saved=metrics["agent_tokens_saved"],
            added_entries=added,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        history.append(asdict(result))

        gain = metrics["char_savings_pct_mean"] - prev_char
        print(f"\n=== Iteration {i} ===")
        print(f"Added: {added}")
        print(json.dumps(asdict(result), indent=2))

        if gain < MIN_CHAR_IMPROVEMENT:
            plateau += 1
        else:
            plateau = 0
        prev_char = metrics["char_savings_pct_mean"]

        if plateau >= PLATEAU_ROUNDS:
            print(f"Plateau reached ({PLATEAU_ROUNDS} rounds < {MIN_CHAR_IMPROVEMENT}% gain).")
            break

    HISTORY.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nFinal dictionary: {len(entries)} entries")
    print(f"History: {HISTORY}")

    run_full_benchmark()

    # Agent demo if ollama up
    from minimal_agent import MinimalAgent, AgentConfig, ollama_available

    if ollama_available():
        agent = MinimalAgent(
            memories=corpus * 4,
            config=AgentConfig(minify_memory=True),
        )
        cmp = agent.compare_context()
        qa = []
        tests = [
            ("What language does the user prefer?", ["python"]),
            ("Where is the organization?", ["berlin"]),
        ]
        for q, kw in tests:
            try:
                t = agent.ask(q)
                ok = any(k in t.answer.lower() for k in kw)
                qa.append({"q": q, "a": t.answer[:100], "ok": ok})
            except Exception as e:
                qa.append({"q": q, "error": str(e)})

        agent_report = {
            "context": cmp,
            "qa": qa,
            "dict_size": len(entries),
            "history_iterations": len(history),
        }
        out = ROOT / "experiments" / "agent_loop_result.json"
        out.write_text(json.dumps(agent_report, indent=2), encoding="utf-8")
        print(f"Agent report: {out}")
    else:
        print("Ollama not available — skipped live agent QA.")


if __name__ == "__main__":
    main()
