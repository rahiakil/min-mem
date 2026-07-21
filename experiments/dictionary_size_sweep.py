#!/usr/bin/env python3
"""Dictionary-size sweep: measure char savings and entity-noun retention as the
released dictionary grows from a small seed to its full size.

This is the "improvements with larger dictionaries" experiment: it shows that
char savings rise with dictionary size while entity-noun retention stays high,
and that the marginal gain per added entry diminishes (diminishing returns).

Entries are added in priority order: char-savings potential (len(src)-len(dst))
weighted by corpus frequency, so the curve rises steeply then plateaus.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from min_mem.converter import MinMemConverter
from min_mem.dictionary import MinDictionary

from run_benchmark import evaluate, noun_preservation  # noqa: E402

CORPUS = ROOT / "experiments" / "corpus.json"
DICT_PATH = ROOT / "min_dict.json"
OUT_JSON = ROOT / "experiments" / "dictionary_size_sweep_results.json"
FIGURES = ROOT / "experiments" / "figures"
TARGET_SIZES = [50, 100, 200, 400, 800, 1422]


def _corpus_freq(samples: list[dict]) -> dict[str, int]:
    freq: dict[str, int] = {}
    for s in samples:
        for tok in s["text"].lower().split():
            freq[tok] = freq.get(tok, 0) + 1
    return freq


def _priority_order(entries: dict[str, str], freq: dict[str, int]) -> list[str]:
    def score(k: str) -> float:
        gain = max(len(k) - len(entries[k]), 0)
        f = freq.get(k.lower(), 1)
        return gain * (1 + f)

    return sorted(entries.keys(), key=score, reverse=True)


def main() -> None:
    samples = json.loads(CORPUS.read_text(encoding="utf-8"))["samples"]
    full = json.loads(DICT_PATH.read_text(encoding="utf-8"))
    entries = full.get("entries", full)
    noun_abbrev = full.get("noun_abbreviations", {})
    freq = _corpus_freq(samples)
    order = _priority_order(entries, freq)

    rows = []
    for size in TARGET_SIZES:
        keys = order[:size]
        sub = {k: entries[k] for k in keys if k in entries}
        dictionary = MinDictionary.from_dict(sub)
        # Attach the noun-abbreviation allowlist so retention is measured
        # against the same entity-noun set at every size.
        dictionary._noun_abbreviations = dict(noun_abbrev)
        converter = MinMemConverter(dictionary)
        char_pcts, ent_pcts = [], []
        for s in samples:
            text = s["text"]
            r = converter.minify(text)
            ev = evaluate(f"size-{size}", s["id"], text, r.minified, len(r.replacements), dictionary)
            char_pcts.append(ev.char_savings_pct)
            ent_pcts.append(ev.entity_nouns_preserved_pct)
        n = len(samples)
        rows.append({
            "dict_size": len(sub),
            "char_savings_pct": round(sum(char_pcts) / n, 2),
            "entity_nouns_preserved_pct": round(sum(ent_pcts) / n, 2),
        })
        print(f"size={len(sub):5d} char={rows[-1]['char_savings_pct']:.2f}% "
              f"ent_noun={rows[-1]['entity_nouns_preserved_pct']:.2f}%")

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus_size": len(samples),
        "target_sizes": TARGET_SIZES,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")

    # Figure: dual-axis, char savings (left) and entity-noun retention (right)
    # vs dictionary size.
    sizes = [r["dict_size"] for r in rows]
    char = [r["char_savings_pct"] for r in rows]
    ent = [r["entity_nouns_preserved_pct"] for r in rows]
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    ax.plot(sizes, char, "o-", color="#1f77b4", label="Char savings (%)")
    ax.set_xlabel("Dictionary size (entries)")
    ax.set_ylabel("Char savings (%)", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot(sizes, ent, "s--", color="#d62728", label="Entity-noun retention (%)")
    ax2.set_ylabel("Entity-noun retention (%)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(80, 102)
    ax.grid(True, alpha=0.3)
    ax.set_title("Savings vs dictionary size")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_dict_size_sweep.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_dict_size_sweep.png", bbox_inches="tight")
    print(f"Wrote figures/fig_dict_size_sweep.{{pdf,png}}")


if __name__ == "__main__":
    main()
