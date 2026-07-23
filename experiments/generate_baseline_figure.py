#!/usr/bin/env python3
"""Pareto chart: char savings vs noun retention across compression modes."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = Path(__file__).parent / "baseline_results.json"
OUT_DIR = Path(__file__).parent / "figures"

# Short legend labels (avoid long inline annotations that bleed past the axes).
SHORT = {
    "identity": "identity",
    "min-mem": "Min-Mem",
    "min-mem×2": "Min-Mem×2",
    "gpt2-selective@0.5": "GPT-2 sel.",
    "min-mem+gpt2-selective@0.5": "Min-Mem+GPT-2",
    "min-mem×2+gpt2-selective@0.5": "Min-Mem×2+GPT-2",
    "llmlingua-2@0.5": "LLMLingua-2",
    "min-mem+llmlingua-2@0.5": "Min-Mem+LL2",
    "min-mem×2+llmlingua-2@0.5": "Min-Mem×2+LL2",
}

COLORS = {
    "identity": "#888888",
    "min-mem": "#2ecc71",
    "min-mem×2": "#27ae60",
    "gpt2-selective@0.5": "#e67e22",
    "min-mem+gpt2-selective@0.5": "#d35400",
    "min-mem×2+gpt2-selective@0.5": "#a04000",
    "llmlingua-2@0.5": "#3498db",
    "min-mem+llmlingua-2@0.5": "#e74c3c",
    "min-mem×2+llmlingua-2@0.5": "#c0392b",
}


def main() -> None:
    import matplotlib.pyplot as plt

    data = json.loads(INPUT.read_text(encoding="utf-8"))
    summary = data["primary_summary"]
    retrieval = data.get("retrieval_fidelity", {}).get("by_mode", [])
    acc_by_mode = {r["mode"]: r["accuracy_pct"] for r in retrieval}

    OUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    # Plot in a stable order so the legend reads identity -> stacked.
    order = [
        "identity", "min-mem", "min-mem×2",
        "gpt2-selective@0.5", "min-mem+gpt2-selective@0.5", "min-mem×2+gpt2-selective@0.5",
        "llmlingua-2@0.5", "min-mem+llmlingua-2@0.5", "min-mem×2+llmlingua-2@0.5",
    ]
    for method in order:
        if method not in summary:
            continue
        s = summary[method]
        x = s["char_savings_pct_mean"]
        y = s["nouns_preserved_pct_mean"]
        c = COLORS.get(method, "#333333")
        label = SHORT.get(method, method)
        if method in acc_by_mode:
            label += f" ({acc_by_mode[method]:.0f}% QA)"
        ax.scatter(x, y, s=90, c=c, edgecolors="white", linewidths=0.8,
                   zorder=3, label=label)

    ax.set_xlabel("Mean character savings (%)")
    ax.set_ylabel("Noun tag retention (%)")
    ax.set_title("Tiered compression: savings vs entity safety (primary rate=0.5)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 58)
    ax.set_ylim(35, 102)

    # Drop the bounding-box spines so nothing appears to "bleed" past a box.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.legend(fontsize=7, loc="lower left", ncol=2, frameon=False,
              handletextpad=0.4, columnspacing=1.0)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"fig_baseline_pareto.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
