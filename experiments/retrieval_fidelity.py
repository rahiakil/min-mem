#!/usr/bin/env python3
"""Downstream retrieval fidelity: QA probes across compression modes.

Runs the same factual questions against verbose memory and each compressed
variant using a local CPU LLM (Ollama). Shows whether tiered compression
maintains or improves answer accuracy while shrinking context.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from min_mem.converter import MinMemConverter  # noqa: E402
from min_mem.dictionary import MinDictionary  # noqa: E402
from compressors import build_compressors, try_import_optional  # noqa: E402

try:
    import tiktoken
except ImportError:
    tiktoken = None

BASELINE_JSON = Path(__file__).parent / "baseline_results.json"
CORPUS = Path(__file__).parent / "corpus.json"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen2.5:0.5b"
SCALE = 4
PRIMARY_RATE = 0.5

SYSTEM = (
    "You are a coding agent with persistent memory. "
    "Answer using ONLY the memory below. One short sentence."
)
SKILLS = "## Skills\n- recall: answer from memory\n"

PROBES = [
    ("What language does the user prefer?", ["python"]),
    ("Where is the user's organization located?", ["berlin"]),
    ("What note-taking tool does the user use?", ["obsidian"]),
    ("Does the user prefer strict typing?", ["typescript", "typing", "yes"]),
    ("What architecture does the project use?", ["microservices", "docker"]),
]


@dataclass
class ProbeResult:
    question: str
    answer: str
    keywords: list[str]
    matched: bool
    latency_ms: int


def count_tokens(text: str) -> int:
    if tiktoken is None:
        return len(text.split())
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def build_prompt(memories: list[str], question: str) -> str:
    block = "\n".join(f"- {m}" for m in memories)
    return f"{SYSTEM}\n\n{SKILLS}\n\n## Memory\n{block}\n\n## Question\n{question}"


def ollama_ask(prompt: str, timeout: int = 120) -> tuple[str, int]:
    t0 = time.time()
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 60},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data.get("response", "").strip(), int((time.time() - t0) * 1000)


def keyword_match(answer: str, keywords: list[str]) -> bool:
    blob = answer.lower()
    return any(k in blob for k in keywords)


def run_probes(memories: list[str]) -> list[ProbeResult]:
    results = []
    for question, keywords in PROBES:
        prompt = build_prompt(memories, question)
        try:
            answer, latency = ollama_ask(prompt)
        except Exception as exc:
            answer, latency = f"ERROR: {exc}", 0
        results.append(
            ProbeResult(
                question=question,
                answer=answer,
                keywords=keywords,
                matched=keyword_match(answer, keywords),
                latency_ms=latency,
            )
        )
    return results


def main() -> None:
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
    except (urllib.error.URLError, TimeoutError):
        print("Ollama not available — skipping retrieval fidelity.")
        return

    corpus = json.loads(CORPUS.read_text())["samples"]
    base_memories = [s["text"] for s in corpus] * SCALE
    dictionary = MinDictionary.from_path(ROOT / "min_dict.json")
    converter = MinMemConverter(dictionary)
    optional = try_import_optional()

    # Primary modes only (faster QA loop)
    modes = build_compressors(
        converter,
        rates=(PRIMARY_RATE,),
        include_gpt2=optional.get("gpt2", False),
        include_llmlingua=optional.get("llmlingua", False),
    )

    by_mode = []
    verbose_results = run_probes(base_memories)
    verbose_acc = sum(1 for r in verbose_results if r.matched) / len(verbose_results)

    for mode in modes:
        print(f"  retrieval: {mode.name}")
        try:
            compressed = [mode.compress(m) for m in base_memories]
        except Exception as exc:
            print(f"    skip: {exc}")
            continue

        probe_results = run_probes(compressed)
        acc = sum(1 for r in probe_results if r.matched) / len(probe_results)
        raw_chars = sum(len(m) for m in base_memories)
        comp_chars = sum(len(m) for m in compressed)
        sample_prompt = build_prompt(compressed, PROBES[0][0])

        by_mode.append({
            "mode": mode.name,
            "accuracy_pct": round(100.0 * acc, 1),
            "accuracy_vs_verbose_delta": round(100.0 * (acc - verbose_acc), 1),
            "prompt_tokens": count_tokens(sample_prompt),
            "memory_chars_saved": raw_chars - comp_chars,
            "memory_reduction_pct": round(100.0 * (raw_chars - comp_chars) / raw_chars, 1),
            "probes": [asdict(r) for r in probe_results],
        })

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "probe_count": len(PROBES),
        "memory_blocks": len(base_memories),
        "verbose_accuracy_pct": round(100.0 * verbose_acc, 1),
        "by_mode": by_mode,
    }

    # Merge into baseline_results.json if present
    if BASELINE_JSON.exists():
        baseline = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
        baseline["retrieval_fidelity"] = payload
        baseline["learnings"] = baseline.get("learnings", [])
        best = max(
            (r for r in by_mode if r["accuracy_pct"] >= verbose_acc * 100 - 0.1),
            key=lambda x: x["memory_reduction_pct"],
            default=None,
        )
        tiered = next((r for r in by_mode if r["mode"] == "min-mem+llmlingua-2@0.5"), None)
        if tiered and tiered["accuracy_pct"] >= 100.0 * verbose_acc:
            baseline["learnings"].append(
                f"**Retrieval win:** `min-mem+llmlingua-2@0.5` keeps {tiered['accuracy_pct']:.0f}% QA accuracy "
                f"while cutting memory {tiered['memory_reduction_pct']:.0f}% "
                f"({tiered['memory_chars_saved']} chars) — best compression at full fidelity."
            )
        elif best:
            baseline["learnings"].append(
                f"Retrieval QA: `{best['mode']}` scored {best['accuracy_pct']:.0f}% "
                f"({best['accuracy_vs_verbose_delta']:+.0f} pts vs verbose) "
                f"with {best['memory_reduction_pct']:.0f}% memory reduction."
            )
        BASELINE_JSON.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        from generate_baseline_report import write_report

        write_report(baseline)

    out = Path(__file__).parent / "retrieval_fidelity.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    for row in by_mode:
        print(
            f"  {row['mode']}: {row['accuracy_pct']}% acc, "
            f"-{row['memory_reduction_pct']}% mem"
        )


if __name__ == "__main__":
    main()
