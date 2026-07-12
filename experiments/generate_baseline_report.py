#!/usr/bin/env python3
"""Generate BASELINE_STUDY.md from baseline_results.json."""

from __future__ import annotations

import json
from pathlib import Path

INPUT = Path(__file__).parent / "baseline_results.json"
OUTPUT = Path(__file__).parent / "BASELINE_STUDY.md"


def _table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def write_report(payload: dict, output: Path = OUTPUT) -> Path:
    ps = payload.get("primary_summary", {})
    learnings = payload.get("learnings", [])
    pareto = payload.get("pareto_primary", [])
    retrieval = payload.get("retrieval_fidelity", {})

    lines = [
        "# Tiered Compression Baseline Study",
        "",
        f"*Generated from benchmark run — {payload.get('timestamp', 'unknown')}*",
        "",
        "## Research question",
        "",
        "Can **auditable, zero-inference lexical minification** (min-mem) serve as a",
        "first tier that improves or matches downstream retrieval when combined with",
        "LM-based deletion compressors (Selective Context, LLMLingua-2)?",
        "",
        "## Contribution framing",
        "",
        "| Tier | Method | Inference cost | Auditable |",
        "|------|--------|----------------|-----------|",
        "| 0 | identity | none | yes |",
        "| 1 | **min-mem** | **none** | **yes** (deterministic dict) |",
        "| 2 | gpt2-selective / llmlingua-2 | small LM on CPU | partial |",
        "| 3 | min-mem → tier 2 | small LM on CPU | pre-step yes |",
        "",
        "## Primary comparison",
        "",
        f"Corpus: **{payload.get('corpus_size', '?')}** agent-memory passages · "
        f"Dictionary: **{payload.get('dictionary_size', '?')}** entries · "
        f"Deletion rate / keep ratio: **{payload.get('primary_rate', 0.5)}**",
        "",
        _table_row(["Method", "Char%", "Token%", "Noun%", "Jaccard", "Syn%", "Inference"]),
        _table_row(["---"] * 7),
    ]

    order = sorted(ps.keys(), key=lambda k: -ps[k]["char_savings_pct_mean"])
    for method in order:
        s = ps[method]
        lines.append(
            _table_row([
                f"`{method}`",
                f"{s['char_savings_pct_mean']:.1f}",
                f"{s['token_savings_pct_mean']:.1f}",
                f"{s['nouns_preserved_pct_mean']:.1f}",
                f"{s['content_jaccard_mean']:.2f}",
                f"{s['synonym_aware_pct_mean']:.1f}",
                s.get("inference_cost", "?"),
            ])
        )

    lines.extend(["", "## Pareto frontier (primary)", ""])
    if pareto:
        lines.append(_table_row(["Method", "Char%", "Noun%", "Jaccard"]))
        lines.append(_table_row(["---"] * 4))
        for p in pareto:
            lines.append(
                _table_row([
                    f"`{p['method']}`",
                    f"{p['char_savings']:.1f}",
                    f"{p['noun_retention']:.1f}",
                    f"{p['jaccard']:.2f}",
                ])
            )
    else:
        lines.append("_No frontier computed._")

    lines.extend(["", "## Key learnings", ""])
    for i, L in enumerate(learnings, 1):
        lines.append(f"{i}. {L}")

    if retrieval:
        lines.extend([
            "",
            "## Retrieval fidelity (Ollama QA)",
            "",
            f"Model: `{retrieval.get('model', '?')}` · "
            f"Probes: {retrieval.get('probe_count', '?')} · "
            f"Memory blocks: {retrieval.get('memory_blocks', '?')}",
            "",
            _table_row(["Mode", "Accuracy", "Tokens", "Chars saved"]),
            _table_row(["---"] * 4),
        ])
        for row in retrieval.get("by_mode", []):
            lines.append(
                _table_row([
                    f"`{row['mode']}`",
                    f"{row.get('accuracy_pct', 0):.0f}%",
                    str(row.get("prompt_tokens", "")),
                    str(row.get("memory_chars_saved", "")),
                ])
            )

    sweep = payload.get("sweep_summary", {})
    if sweep:
        lines.extend(["", "## Rate sweep (0.3 / 0.5 / 0.7)", ""])
        llm_modes = [k for k in sweep if "llmlingua" in k or "gpt2" in k]
        if llm_modes:
            lines.append(_table_row(["Method", "Char%", "Noun%", "Jaccard"]))
            lines.append(_table_row(["---"] * 4))
            for method in sorted(llm_modes):
                s = sweep[method]
                lines.append(
                    _table_row([
                        f"`{method}`",
                        f"{s['char_savings_pct_mean']:.1f}",
                        f"{s['nouns_preserved_pct_mean']:.1f}",
                        f"{s['content_jaccard_mean']:.2f}",
                    ])
                )

    lines.extend([
        "",
        "## Reproduce",
        "",
        "```bash",
        "pip install -e \".[dev,baselines]\"",
        "python experiments/benchmark_baselines.py",
        "python experiments/retrieval_fidelity.py   # optional: needs Ollama",
        "```",
        "",
        "## Paper repo",
        "",
        "Copy `baseline_results.json` and `BASELINE_STUDY.md` to `min-mem-paper/experiments/`",
        "for LaTeX integration.",
        "",
    ])

    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")
    return output


def main() -> None:
    if not INPUT.exists():
        raise SystemExit(f"Missing {INPUT} — run benchmark_baselines.py first")
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    write_report(payload)


if __name__ == "__main__":
    main()
