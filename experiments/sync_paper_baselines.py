#!/usr/bin/env python3
"""Sync baseline study artifacts to private paper repo."""

from __future__ import annotations

import shutil
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1].parent / "min-mem-paper" / "experiments"
SRC = Path(__file__).resolve().parent

FILES = (
    "baseline_results.json",
    "BASELINE_STUDY.md",
    "retrieval_fidelity.json",
    "storage_proof_results.json",
)


def main() -> None:
    if not PAPER.parent.exists():
        print(f"Paper repo not found at {PAPER.parent}")
        return
    PAPER.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        src = SRC / name
        if src.exists():
            shutil.copy2(src, PAPER / name)
            print(f"  → {PAPER / name}")


if __name__ == "__main__":
    main()
