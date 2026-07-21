#!/usr/bin/env python3
"""Waterfall benchmark: show incremental value of min-mem passes then LM tiers.

Outputs experiments/tiered_value.json and experiments/TIERED_VALUE.md
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from min_mem.converter import MinMemConverter  # noqa: E402
from min_mem.dictionary import MinDictionary  # noqa: E402
from compressors import (  # noqa: E402
    GPT2SelectiveContext,
    IdentityCompressor,
    LLMLingua2Compressor,
    MinMemCompressor,
    ChainedCompressor,
    try_import_optional,
)
from metrics import aggregate, evaluate  # noqa: E402

CORPUS = Path(__file__).parent / "corpus.json"
OUT_JSON = Path(__file__).parent / "tiered_value.json"
OUT_MD = Path(__file__).parent / "TIERED_VALUE.md"
PRIMARY_RATE = 0.5
SCALE = int(os.environ.get("TIERED_SCALE", "1"))  # agent-scale memory blocks (waterfall is scale-invariant)


def load_corpus() -> list[dict]:
    return json.loads(CORPUS.read_text(encoding="utf-8"))["samples"]


def scaled_texts(samples: list[dict]) -> list[str]:
    base = [s["text"] for s in samples]
    return base * SCALE


def bench_mode(name: str, compress_fn, texts: list[str], dictionary: MinDictionary) -> dict:
    t0 = time.perf_counter()
    results = []
    for i, text in enumerate(texts):
        sample_id = f"block-{i % len(texts) // SCALE}-{i}"
        try:
            out = compress_fn(text)
        except Exception as exc:
            out = text
            name = f"{name} (failed: {exc})"
        results.append(evaluate(name, sample_id, text, out, -1, dictionary))
    elapsed = time.perf_counter() - t0
    summary_map = aggregate(results)
    summary = summary_map.get(name, next(iter(summary_map.values()), {}))
    raw_chars = sum(len(t) for t in texts)
    comp_chars = sum(r.minified_chars for r in results)
    return {
        "mode": name,
        "summary": summary,
        "total_chars": raw_chars,
        "compressed_chars": comp_chars,
        "chars_saved": raw_chars - comp_chars,
        "char_reduction_pct": round(100.0 * (raw_chars - comp_chars) / raw_chars, 2) if raw_chars else 0,
        "elapsed_ms": round(elapsed * 1000, 1),
        "throughput_passages_per_sec": round(len(texts) / elapsed, 1) if elapsed else 0,
    }


def main() -> None:
    samples = load_corpus()
    texts = scaled_texts(samples)
    dictionary = MinDictionary.from_path(ROOT / "min_dict.json")
    converter = MinMemConverter(dictionary)
    optional = try_import_optional()

    mm1 = MinMemCompressor(converter, name="min-mem", passes=1)
    mm2 = MinMemCompressor(converter, name="min-mem×2", passes=2)

    ladder: list[tuple[str, object]] = [
        ("identity", IdentityCompressor()),
        ("min-mem (pass 1)", mm1),
        ("min-mem (pass 2)", mm2),
    ]

    if optional.get("llmlingua"):
        ll2 = LLMLingua2Compressor(rate=PRIMARY_RATE)
        ll2.name = f"llmlingua-2@{PRIMARY_RATE}"
        ladder.extend([
            (f"llmlingua-2@{PRIMARY_RATE} (alone)", ll2),
            (f"min-mem → llmlingua-2@{PRIMARY_RATE}", ChainedCompressor(mm1, ll2)),
            (f"min-mem×2 → llmlingua-2@{PRIMARY_RATE}", ChainedCompressor(mm2, ll2)),
        ])

    if optional.get("gpt2"):
        gpt = GPT2SelectiveContext(keep_ratio=PRIMARY_RATE)
        gpt.name = f"gpt2-selective@{PRIMARY_RATE}"
        ladder.extend([
            (f"gpt2-selective@{PRIMARY_RATE} (alone)", gpt),
            (f"min-mem → gpt2-selective@{PRIMARY_RATE}", ChainedCompressor(mm1, gpt)),
            (f"min-mem×2 → gpt2-selective@{PRIMARY_RATE}", ChainedCompressor(mm2, gpt)),
        ])

    rows = []
    prev_reduction = 0.0
    for label, compressor in ladder:
        row = bench_mode(label, compressor.compress, texts, dictionary)
        row["incremental_pct"] = round(row["char_reduction_pct"] - prev_reduction, 2)
        row["inference_cost"] = getattr(compressor, "inference_cost", "none")
        rows.append(row)
        prev_reduction = row["char_reduction_pct"]

    # Value-add: min-mem contribution before LLMLingua
    mm_row = next((r for r in rows if r["mode"] == "min-mem (pass 1)"), None)
    ll_solo = next((r for r in rows if "llmlingua-2" in r["mode"] and "alone" in r["mode"]), None)
    ll_stack = next((r for r in rows if r["mode"].startswith("min-mem → llmlingua")), None)
    mm2_stack = next((r for r in rows if r["mode"].startswith("min-mem×2 → llmlingua")), None)

    value_add = []
    if mm_row and ll_solo:
        value_add.append({
            "comparison": "min-mem pre-step before LLMLingua",
            "min_mem_alone_pct": mm_row["char_reduction_pct"],
            "llmlingua_alone_pct": ll_solo["char_reduction_pct"],
            "stacked_pct": ll_stack["char_reduction_pct"] if ll_stack else None,
            "min_mem_lift_on_stack": (
                round(ll_stack["char_reduction_pct"] - ll_solo["char_reduction_pct"], 2)
                if ll_stack else None
            ),
        })
    if mm_row and mm2_stack:
        value_add.append({
            "comparison": "2nd min-mem pass before LLMLingua",
            "pass2_alone_pct": next(
                (r["char_reduction_pct"] for r in rows if r["mode"] == "min-mem (pass 2)"), 0
            ),
            "stacked_pct": mm2_stack["char_reduction_pct"],
            "pass2_lift_on_stack": round(
                mm2_stack["char_reduction_pct"] - (ll_stack["char_reduction_pct"] if ll_stack else 0),
                2,
            ) if ll_stack else None,
        })

    # Single-passage trace across tiers
    example = samples[0]["text"]
    example_id = samples[0]["id"]
    traces = [{"stage": "identity", "text": example, "chars": len(example)}]
    for label, compressor in ladder[1:]:
        if "gpt2" in label and "alone" in label:
            continue  # skip duplicate-style rows in trace
        try:
            out = compressor.compress(example)
        except Exception:
            out = traces[-1]["text"]
        traces.append({"stage": label, "text": out, "chars": len(out)})

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus_passages": len(samples),
        "memory_blocks": len(texts),
        "dictionary_size": len(dictionary),
        "optional_backends": optional,
        "waterfall": rows,
        "value_add": value_add,
        "example_trace": {"sample_id": example_id, "stages": traces},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Tiered compression value waterfall",
        "",
        f"*Generated {payload['timestamp']}*",
        "",
        f"Corpus: **{len(samples)}** passages × **{SCALE}** = **{len(texts)}** memory blocks · "
        f"Dictionary: **{len(dictionary)}** entries",
        "",
        "## Waterfall (cumulative char reduction)",
        "",
        "| Stage | Char ↓ | Δ vs prev | Noun% | Throughput | Inference |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        s = r["summary"]
        lines.append(
            f"| `{r['mode']}` | {r['char_reduction_pct']:.1f}% | "
            f"+{r['incremental_pct']:.1f}% | "
            f"{s['nouns_preserved_pct_mean']:.1f}% | "
            f"{r['throughput_passages_per_sec']:.0f} pass/s | {r['inference_cost']} |"
        )

    if value_add:
        lines.extend(["", "## min-mem value-add before deletion", ""])
        for v in value_add:
            lines.append(f"- **{v['comparison']}**")
            for k, val in v.items():
                if k != "comparison" and val is not None:
                    lines.append(f"  - {k}: {val}")

    lines.extend([
        "",
        "## Reproduce",
        "",
        "```bash",
        ".venv/bin/python experiments/tiered_value_benchmark.py",
        "```",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON} and {OUT_MD}")
    for r in rows:
        print(
            f"  {r['mode']}: {r['char_reduction_pct']}% "
            f"(+{r['incremental_pct']}%), {r['throughput_passages_per_sec']} pass/s"
        )


if __name__ == "__main__":
    main()
