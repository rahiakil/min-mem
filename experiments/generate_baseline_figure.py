#!/usr/bin/env python3
"""Pareto chart: char savings vs noun retention across compression modes."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = Path(__file__).parent / "baseline_results.json"
OUT_DIR = Path(__file__).parent / "figures"


def main() -> None:
    import matplotlib.pyplot as plt

    data = json.loads(INPUT.read_text(encoding="utf-8"))
    summary = data["primary_summary"]
    retrieval = data.get("retrieval_fidelity", {}).get("by_mode", [])
    acc_by_mode = {r["mode"]: r["accuracy_pct"] for r in retrieval}

    OUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = {
        "identity": "#888888",
        "min-mem": "#2ecc71",
        "gpt2-selective@0.5": "#e67e22",
        "llmlingua-2@0.5": "#3498db",
        "min-mem+gpt2-selective@0.5": "#9b59b6",
        "min-mem+llmlingua-2@0.5": "#e74c3c",
    }

    for method, s in summary.items():
        x = s["char_savings_pct_mean"]
        y = s["nouns_preserved_pct_mean"]
        c = colors.get(method, "#333333")
        ax.scatter(x, y, s=120, c=c, edgecolors="white", linewidths=0.8, zorder=3)
        label = method.replace("@0.5", "")
        if method in acc_by_mode:
            label += f"\n{acc_by_mode[method]:.0f}% QA"
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7)

    ax.set_xlabel("Mean character savings (%)")
    ax.set_ylabel("Noun tag retention (%)")
    ax.set_title("Tiered compression: savings vs entity safety (primary rate=0.5)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 58)
    ax.set_ylim(35, 102)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"fig_baseline_pareto.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
