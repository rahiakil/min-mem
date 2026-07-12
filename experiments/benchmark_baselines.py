#!/usr/bin/env python3
"""External baseline benchmark: min-mem vs LM-deletion vs tiered chains.

Compares rule-based POS-gated substitution (zero inference) against
Selective-Context-style GPT-2 pruning and LLMLingua-2 token deletion,
including pre→post tiered pipelines.

Outputs:
  experiments/baseline_results.json   — full metrics + sweep
  experiments/BASELINE_STUDY.md       — human-readable report (via generate script)
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from min_mem.converter import MinMemConverter  # noqa: E402
from min_mem.dictionary import MinDictionary  # noqa: E402
from compressors import build_compressors, try_import_optional  # noqa: E402
from metrics import aggregate, evaluate, results_to_dict  # noqa: E402

CORPUS_PATH = Path(__file__).parent / "corpus.json"
OUTPUT_JSON = Path(__file__).parent / "baseline_results.json"
SWEEP_RATES = (0.3, 0.5, 0.7)
PRIMARY_RATE = 0.5


def pareto_points(summary: dict) -> list[dict]:
    """Modes where no other mode dominates on compression AND noun retention."""
    points = []
    for name, s in summary.items():
        points.append({
            "method": name,
            "char_savings": s["char_savings_pct_mean"],
            "noun_retention": s["nouns_preserved_pct_mean"],
            "jaccard": s["content_jaccard_mean"],
            "inference_cost": s.get("inference_cost", "unknown"),
        })

    frontier = []
    for p in points:
        dominated = False
        for q in points:
            if q["method"] == p["method"]:
                continue
            if (
                q["char_savings"] >= p["char_savings"]
                and q["noun_retention"] >= p["noun_retention"]
                and (
                    q["char_savings"] > p["char_savings"]
                    or q["noun_retention"] > p["noun_retention"]
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(p)
    return sorted(frontier, key=lambda x: -x["char_savings"])


def derive_learnings(summary: dict, optional: dict[str, bool]) -> list[str]:
    learnings = []

    mm = summary.get("min-mem", {})
    if mm:
        learnings.append(
            f"min-mem alone achieves {mm['char_savings_pct_mean']:.1f}% char reduction "
            f"with {mm['nouns_preserved_pct_mean']:.1f}% noun retention at **zero inference cost**."
        )

    for key in (f"llmlingua-2@{PRIMARY_RATE}", f"gpt2-selective@{PRIMARY_RATE}"):
        if key not in summary:
            continue
        s = summary[key]
        learnings.append(
            f"{key} reaches {s['char_savings_pct_mean']:.1f}% char savings but "
            f"noun retention drops to {s['nouns_preserved_pct_mean']:.1f}% "
            f"(jaccard {s['content_jaccard_mean']:.2f})."
        )

    chain_ll = summary.get(f"min-mem+llmlingua-2@{PRIMARY_RATE}")
    solo_ll = summary.get(f"llmlingua-2@{PRIMARY_RATE}")
    if chain_ll and solo_ll:
        if chain_ll["char_savings_pct_mean"] > solo_ll["char_savings_pct_mean"]:
            learnings.append(
                f"Tiered min-mem → LLMLingua-2 compounds to **{chain_ll['char_savings_pct_mean']:.1f}%** "
                f"char savings vs {solo_ll['char_savings_pct_mean']:.1f}% for deletion alone."
            )

    chain_gpt = summary.get(f"min-mem+gpt2-selective@{PRIMARY_RATE}")
    solo_gpt = summary.get(f"gpt2-selective@{PRIMARY_RATE}")
    if chain_gpt and solo_gpt:
        if chain_gpt["char_savings_pct_mean"] > solo_gpt["char_savings_pct_mean"]:
            learnings.append(
                "min-mem → GPT-2 selective stacking yields higher total compression than "
                "either stage alone at the same keep ratio."
            )

    if not optional.get("llmlingua"):
        learnings.append(
            "LLMLingua-2 not installed — run `pip install -e \".[baselines]\"` for full sweep."
        )
    if not optional.get("gpt2"):
        learnings.append(
            "GPT-2 selective baseline skipped — install torch + transformers."
        )

    learnings.append(
        "**Contribution framing:** min-mem is the auditable, zero-cost normalization tier; "
        "LM deletion methods are optional second stages for byte-starved deployments."
    )
    return learnings


def run_mode_benchmark(
    corpus: list[dict],
    dictionary: MinDictionary,
    rates: tuple[float, ...],
) -> tuple[list, dict, dict]:
    optional = try_import_optional()
    converter = MinMemConverter(dictionary)
    modes = build_compressors(
        converter,
        rates=rates,
        include_gpt2=optional.get("gpt2", False),
        include_llmlingua=optional.get("llmlingua", False),
    )

    results = []
    errors: dict[str, str] = {}
    t0 = time.time()

    for mode in modes:
        print(f"  mode: {mode.name}")
        for sample in corpus:
            sid = sample["id"]
            text = sample["text"]
            try:
                transformed = mode.compress(text)
                results.append(
                    evaluate(
                        mode.name,
                        sid,
                        text,
                        transformed,
                        replacements=-1,
                        dictionary=dictionary,
                        params=getattr(mode, "rate", None) or getattr(mode, "keep_ratio", None),
                        inference_cost=getattr(mode, "inference_cost", "none"),
                    )
                )
            except Exception as exc:
                errors[f"{mode.name}:{sid}"] = str(exc)

    summary = aggregate(results)
    elapsed = time.time() - t0
    return results, summary, {"optional_deps": optional, "errors": errors, "elapsed_sec": elapsed}


def main() -> None:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))["samples"]
    dictionary = MinDictionary.from_path(ROOT / "min_dict.json")

    print("=== Primary comparison (rate=0.5) ===")
    primary_results, primary_summary, meta = run_mode_benchmark(
        corpus, dictionary, rates=(PRIMARY_RATE,)
    )

    print("\n=== Full sweep (0.3, 0.5, 0.7) ===")
    sweep_results, sweep_summary, sweep_meta = run_mode_benchmark(
        corpus, dictionary, rates=SWEEP_RATES
    )

    # Load internal ablation summary for context
    internal_path = Path(__file__).parent / "results.json"
    internal_summary = {}
    if internal_path.exists():
        internal_summary = json.loads(internal_path.read_text()).get("summary", {})

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "study": "tiered_compression_baselines",
        "corpus_size": len(corpus),
        "dictionary_size": len(dictionary),
        "sweep_rates": list(SWEEP_RATES),
        "primary_rate": PRIMARY_RATE,
        "optional_deps": meta["optional_deps"],
        "primary_summary": primary_summary,
        "sweep_summary": sweep_summary,
        "internal_ablation_summary": internal_summary,
        "pareto_primary": pareto_points(primary_summary),
        "pareto_sweep": pareto_points(sweep_summary),
        "learnings": derive_learnings(primary_summary, meta["optional_deps"]),
        "per_sample_primary": results_to_dict(primary_results),
        "per_sample_sweep": results_to_dict(sweep_results),
        "errors": {**meta.get("errors", {}), **sweep_meta.get("errors", {})},
        "elapsed_sec": meta["elapsed_sec"] + sweep_meta["elapsed_sec"],
    }

    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {OUTPUT_JSON}")
    print(json.dumps(primary_summary, indent=2))

    # Auto-generate markdown report
    from generate_baseline_report import write_report

    write_report(payload)


if __name__ == "__main__":
    main()
