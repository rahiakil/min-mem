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

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(events, id_bytes, "o-", label="Identity store", color="#4C72B0")
    ax.plot(events, min_bytes, "s-", label="Min-mem store", color="#55A868")
    ax.set_xlabel("Memory events")
    ax.set_ylabel("Payload (KB)")
    ax.set_title("Memory growth: identity vs min-mem")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_storage_growth.pdf")
    fig.savefig(FIG_DIR / "fig_storage_growth.png", dpi=150)
    plt.close(fig)

    org = data["org_simulation"]["consolidation"]
    labels = ["Naive replicated", "Shared consolidated"]
    vals = [org["naive_replicated_bytes"] / 1024, org["shared_consolidated_bytes"] / 1024]
    fig2, ax2 = plt.subplots(figsize=(4.5, 3))
    ax2.bar(labels, vals, color=["#C44E52", "#55A868"])
    ax2.set_ylabel("KB")
    ax2.set_title("Organizational consolidation (3 writers)")
    fig2.tight_layout()
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

    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
