#!/usr/bin/env python3
"""Generate storage growth and org consolidation figures for the paper."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

RESULTS = EXPERIMENTS / "storage_proof_results.json"
FIG_DIR = EXPERIMENTS / "figures"


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skip figures")
        return

    if not RESULTS.exists():
        print(f"Missing {RESULTS}; run storage_proof/runner.py first")
        return

    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    sg = data["storage_growth"]["checkpoints"]
    events = [c["event_count"] for c in sg]
    id_bytes = [c["identity_payload_bytes"] / 1024 for c in sg]
    min_bytes = [c["minified_payload_bytes"] / 1024 for c in sg]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(events, id_bytes, "o-", label="Identity store", color="#4C72B0", linewidth=2)
    ax.plot(events, min_bytes, "s-", label="Min-mem store", color="#55A868", linewidth=2)
    ax.set_xlabel("Memory events")
    ax.set_ylabel("Payload (KB)")
    ax.set_title("Storage growth: identity vs min-mem")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ymax = max(id_bytes + min_bytes) * 1.12
    ax.set_ylim(0, ymax)
    for xs, ys, color in ((events, id_bytes, "#4C72B0"), (events, min_bytes, "#55A868")):
        ax.annotate(
            f"{ys[-1]:.1f}",
            xy=(xs[-1], ys[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=7,
            color=color,
        )
    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.14, top=0.90)
    fig.savefig(FIG_DIR / "fig_storage_growth.pdf")
    fig.savefig(FIG_DIR / "fig_storage_growth.png", dpi=150)
    plt.close(fig)

    org = data["org_simulation"]["consolidation"]
    labels = ["Naive\nreplicated", "Shared\nconsolidated"]
    vals = [org["naive_replicated_bytes"] / 1024, org["shared_consolidated_bytes"] / 1024]
    fig2, ax2 = plt.subplots(figsize=(4.8, 3.2))
    ymax = max(vals) * 1.22
    bars = ax2.bar(labels, vals, color=["#C44E52", "#55A868"])
    ax2.set_ylabel("KB")
    ax2.set_title("Organizational memory growth (3 writers)")
    ax2.set_ylim(0, ymax)
    ax2.margins(x=0.15)
    for bar, val in zip(bars, vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + ymax * 0.03,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig2.subplots_adjust(bottom=0.18)
    fig2.savefig(FIG_DIR / "fig_org_consolidation.pdf")
    fig2.savefig(FIG_DIR / "fig_org_consolidation.png", dpi=150)
    plt.close(fig2)

    benchmarks = data["benchmarks"]
    names = ["agent_corpus", "locomo", "membench"]
    reductions = [benchmarks[n]["char_reduction_pct"] for n in names]
    fig3, ax3 = plt.subplots(figsize=(4.5, 3))
    ax3.bar([n.replace("_", " ") for n in names], reductions, color="#8172B2")
    ax3.set_ylabel("Char reduction %")
    ax3.set_title("Storage reduction by benchmark")
    ax3.set_ylim(0, max(reductions) * 1.2 if reductions else 25)
    fig3.tight_layout()
    fig3.savefig(FIG_DIR / "fig_benchmark_reduction.pdf")
    fig3.savefig(FIG_DIR / "fig_benchmark_reduction.png", dpi=150)
    plt.close(fig3)

    policy = data.get("tiny_policy_validation", {})
    rows = policy.get("agent_unique", policy).get("results", [])
    if rows:
        budgets = sorted({r["budget_ratio"] for r in rows})
        fig4, ax4 = plt.subplots(figsize=(5.4, 3.4))
        for policy_name, color, marker in (
            ("fifo_oldest", "#C44E52", "o"),
            ("fifo_newest", "#E17C72", "v"),
            ("bm25_only", "#4C72B0", "s"),
            ("tiny_linear", "#55A868", "^"),
        ):
            try:
                ys = [
                    next(
                        r["retention_pct"]
                        for r in rows
                        if r["policy"] == policy_name and abs(r["budget_ratio"] - b) < 1e-9
                    )
                    for b in budgets
                ]
            except StopIteration:
                continue
            ax4.plot(
                budgets,
                ys,
                marker=marker,
                color=color,
                label=policy_name.replace("_", " "),
                linewidth=2,
            )
        ax4.set_xlabel("Keep budget ratio")
        ax4.set_ylabel("QA retention vs full minified store (%)")
        ax4.set_title("Read-path tiny policy vs FIFO/BM25")
        ax4.set_ylim(0, 105)
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=7, framealpha=0.9)
        fig4.tight_layout()
        fig4.savefig(FIG_DIR / "fig_tiny_policy.pdf")
        fig4.savefig(FIG_DIR / "fig_tiny_policy.png", dpi=150)
        plt.close(fig4)

    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
