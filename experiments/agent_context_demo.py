#!/usr/bin/env python3
"""Prove min-mem reduces agent context using a local CPU LLM (Ollama).

Builds a realistic agent prompt (system + skills + memory blocks), minifies
memory, measures context reduction, and runs QA parity checks.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from min_mem import MinMemConverter  # noqa: E402

try:
    import tiktoken
except ImportError:
    tiktoken = None

OUTPUT = ROOT / "docs" / "data" / "agent-demo.json"
CORPUS = ROOT / "experiments" / "corpus.json"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen2.5:0.5b"
SCALE_FACTOR = 4  # repeat corpus to simulate mature agent memory store

SKILLS = """## Agent Skills (always active)
- code_review: Review diffs for bugs, style, and security issues.
- explore: Search codebase semantically before making changes.
- shell: Run terminal commands; prefer non-interactive flags.
- memory_write: Persist user preferences and project facts to memory store.
"""

SYSTEM = """You are a coding agent with persistent memory. Answer questions using ONLY the memory block below. Be concise — one sentence max."""

# Smoke test on a controlled 15-passage subset (x4 = 60 memory bullets) so the
# 0.5B reader can still retrieve reliably; the full 304-passage corpus is
# exercised by the benchmark and storage-proof runs.
_ALL = json.loads(CORPUS.read_text())["samples"]
MEMORIES = [m["text"] for m in _ALL[:15]] * SCALE_FACTOR

QUESTIONS = [
    ("What language does the user prefer?", ["python"]),
    ("Where is the user's organization located?", ["berlin"]),
    ("What note-taking tool does the user use?", ["obsidian"]),
    ("Does the user prefer strict typing?", ["typescript", "typing", "yes"]),
]


@dataclass
class ContextStats:
    chars: int
    tokens: int
    memory_chars: int
    memory_tokens: int
    pct_of_8k: float
    pct_of_32k: float


def count_tokens(text: str) -> int:
    if tiktoken is None:
        return len(text.split())
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def build_prompt(memories: list[str]) -> str:
    memory_block = "\n".join(f"- {m}" for m in memories)
    return f"{SYSTEM}\n\n{SKILLS}\n\n## Memory\n{memory_block}\n\n## Question\n"


def stats(prompt: str, memories: list[str]) -> ContextStats:
    mem_text = "\n".join(memories)
    chars = len(prompt)
    tokens = count_tokens(prompt)
    mc = len(mem_text)
    mt = count_tokens(mem_text)
    return ContextStats(
        chars=chars,
        tokens=tokens,
        memory_chars=mc,
        memory_tokens=mt,
        pct_of_8k=100.0 * tokens / 8192,
        pct_of_32k=100.0 * tokens / 32768,
    )


def ollama_ask(prompt: str, question: str) -> str:
    full = prompt + question
    payload = json.dumps({
        "model": MODEL,
        "prompt": full,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 60},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data.get("response", "").strip()


def answers_match(verbose: str, minified: str, keywords: list[str]) -> bool:
    blob = (verbose + " " + minified).lower()
    return all(k in blob for k in keywords)


def main() -> None:
    converter = MinMemConverter()
    minified_memories = [converter.minify(m).minified for m in MEMORIES]

    prompt_verbose = build_prompt(MEMORIES)
    prompt_min = build_prompt(minified_memories)

    s_verbose = stats(prompt_verbose, MEMORIES)
    s_min = stats(prompt_min, minified_memories)

    char_saved = s_verbose.chars - s_min.chars
    token_saved = s_verbose.tokens - s_min.tokens
    char_pct = 100.0 * char_saved / s_verbose.chars if s_verbose.chars else 0
    token_pct = 100.0 * token_saved / s_verbose.tokens if s_verbose.tokens else 0
    mem_char_pct = (
        100.0 * (s_verbose.memory_chars - s_min.memory_chars) / s_verbose.memory_chars
        if s_verbose.memory_chars else 0
    )

    # Per-memory minification trace
    traces = []
    for orig, mini in zip(MEMORIES, minified_memories):
        r = converter.minify(orig)
        traces.append({
            "original": orig,
            "minified": mini,
            "chars_saved": r.chars_saved,
            "savings_pct": round(r.savings_ratio * 100, 1),
            "swaps": len(r.replacements),
        })

    # CPU LLM QA parity
    qa_results = []
    ollama_ok = True
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
    except (urllib.error.URLError, TimeoutError):
        ollama_ok = False

    if ollama_ok:
        for q, keywords in QUESTIONS:
            t0 = time.time()
            ans_v = ollama_ask(prompt_verbose, q)
            t1 = time.time()
            ans_m = ollama_ask(prompt_min, q)
            t2 = time.time()
            qa_results.append({
                "question": q,
                "answer_verbose": ans_v,
                "answer_minified": ans_m,
                "keywords_expected": keywords,
                "match": answers_match(ans_v, ans_m, keywords),
                "latency_verbose_ms": int((t1 - t0) * 1000),
                "latency_minified_ms": int((t2 - t1) * 1000),
            })

    # Project savings at scale (50, 100, 200 memory blocks)
    base_mem = json.loads(CORPUS.read_text())["samples"]
    base_texts = [m["text"] for m in base_mem]
    projections = []
    for n in [15, 50, 100, 200]:
        texts = (base_texts * ((n // len(base_texts)) + 1))[:n]
        mini = [converter.minify(t).minified for t in texts]
        pv = build_prompt(texts)
        pm = build_prompt(mini)
        projections.append({
            "memory_blocks": n,
            "verbose_tokens": count_tokens(pv),
            "minified_tokens": count_tokens(pm),
            "tokens_saved": count_tokens(pv) - count_tokens(pm),
            "pct_of_8k_verbose": round(100.0 * count_tokens(pv) / 8192, 1),
            "pct_of_8k_minified": round(100.0 * count_tokens(pm) / 8192, 1),
        })

    payload = {
        "model": MODEL,
        "ollama_available": ollama_ok,
        "memory_count": len(MEMORIES),
        "skills_chars": len(SKILLS),
        "context": {
            "verbose": asdict(s_verbose),
            "minified": asdict(s_min),
            "chars_saved": char_saved,
            "tokens_saved": token_saved,
            "char_reduction_pct": round(char_pct, 1),
            "token_reduction_pct": round(token_pct, 1),
            "memory_char_reduction_pct": round(mem_char_pct, 1),
            "context_8k_freed_pct": round(s_verbose.pct_of_8k - s_min.pct_of_8k, 2),
        },
        "memory_traces": traces,
        "qa_parity": qa_results,
        "qa_match_rate": (
            round(100.0 * sum(1 for r in qa_results if r["match"]) / len(qa_results), 1)
            if qa_results else None
        ),
        "scale_projections": projections,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(json.dumps(payload["context"], indent=2))
    if qa_results:
        print(f"QA match rate: {payload['qa_match_rate']}%")


if __name__ == "__main__":
    main()
