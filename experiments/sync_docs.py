#!/usr/bin/env python3
"""Copy benchmark artifacts into docs/ for GitHub Pages."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results.json"
FIG_SRC = ROOT / "experiments" / "figures"
DOCS_DATA = ROOT / "docs" / "data"
DOCS_IMG = ROOT / "docs" / "assets" / "img"


def main() -> None:
    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    DOCS_IMG.mkdir(parents=True, exist_ok=True)

    if RESULTS.exists():
        shutil.copy2(RESULTS, DOCS_DATA / "benchmark.json")

    copies = {
        "improve_history.json": "improve-history.json",
        "agent_loop_result.json": "agent-loop.json",
    }
    for src_name, dst_name in copies.items():
        src = ROOT / "experiments" / src_name
        if src.exists():
            shutil.copy2(src, DOCS_DATA / dst_name)

    if FIG_SRC.exists():
        for ext in ("png", "pdf"):
            for fig in FIG_SRC.glob(f"fig_*.{ext}"):
                shutil.copy2(fig, DOCS_IMG / fig.name)

    print(f"Synced docs/data and docs/assets/img from experiments/")


if __name__ == "__main__":
    main()
