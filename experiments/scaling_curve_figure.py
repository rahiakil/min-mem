"""Generate the scaling-curve figure from scaling_curve_results.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "scaling_curve_results.json"
OUT = ROOT / "min-mem-paper" / "figures" / "fig_scaling_curve.pdf"
PAPER_FIG = Path("/home/papa/repos/min-mem-paper/figures/fig_scaling_curve.pdf")

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 8,
                      "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})

data = json.loads(RES.read_text())
pts = data["scales"]
scales = [p["scale"] for p in pts]
bytes_saved = [p["bytes_saved"] / 1024 for p in pts]  # KiB
char_pct = [p["char_savings_pct"] for p in pts]
tok_pct = [p["token_savings_pct"] for p in pts]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.4))

ax1.plot(scales, bytes_saved, "o-", color="#1f77b4", lw=1.5, ms=4)
ax1.set_xscale("log")
ax1.set_xlabel("Corpus scale (x)")
ax1.set_ylabel("Bytes saved (KiB)")
ax1.set_title("(a) Absolute savings scale linearly")
ax1.grid(True, alpha=0.3, which="both")

ax2.plot(scales, char_pct, "s-", color="#2ca02c", lw=1.5, ms=4, label="Char savings %")
ax2.plot(scales, tok_pct, "d--", color="#d62728", lw=1.5, ms=4, label="Token savings %")
ax2.set_xscale("log")
ax2.set_xlabel("Corpus scale (x)")
ax2.set_ylabel("Savings rate (%)")
ax2.set_title("(b) Savings rate is scale-invariant")
ax2.set_ylim(0, max(char_pct) + 4)
ax2.legend(loc="upper right", frameon=False)
ax2.grid(True, alpha=0.3, which="both")

fig.tight_layout()
PAPER_FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(PAPER_FIG, bbox_inches="tight")
print(f"wrote {PAPER_FIG}")
