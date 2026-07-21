#!/usr/bin/env python3
"""Add a curated, auditable noun-abbreviation allowlist + safe missing
verb/adverb swaps to min_dict.json.

The noun-abbreviation allowlist is a small set of *unambiguous, standard*
common-noun abbreviations (television->tv, kilometers->km, laboratory->lab,
mathematics->math, influenza->flu, refrigerator->fridge, ...). These are
applied by the converter BEFORE the POS noun-gate (see converter.py), so the
entity-noun protection guarantee is preserved for every other noun; only this
explicit allowlist bypasses it. Each entry is attested by Lexpanded-PPDB-S or
is a well-known standard abbreviation.

Also fixes a handful of low-quality legacy entries and adds clearly
meaning-preserving verb/adverb swaps that were missing.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DICT_PATH = ROOT / "min_dict.json"

# --- Safe missing verb / adverb swaps (meaning-preserving, common) ---
SAFE_SWAPS = {
    "quickly": "fast",
    "swiftly": "fast",
    "although": "though",
    "whereas": "but",
    "definitely": "sure",
    "highly": "very",
    "utilization": "use",
    "rarely": "seldom",
    "gradually": "slow",
    "truly": "really",
    "really": "very",
    "hence": "thus",
    "amongst": "among",
    "amidst": "amid",
    "firstly": "first",
    "lastly": "last",
    "thirdly": "third",
}

# --- Fixes for low-quality legacy entries ---
LEGACY_FIXES = {
    "slowly": "slow",       # was: slowly -> easy (wrong)
    "primarily": "mostly",  # was: primarily -> main (wrong)
    "usually": "often",     # was: usually -> oft (archaic/odd)
}
LEGACY_REMOVE = {"ensure", "specifically", "steadily"}
# ensure -> see (wrong); specifically -> namely (wrong); steadily -> steady (no-op)

# --- Noun-abbreviation allowlist (unambiguous, standard) ---
# Applied before the POS noun-gate; entity nouns are still protected.
NOUN_ABBREVIATIONS = {
    "television": "tv",
    "kilometers": "km",
    "kilometres": "km",
    "kilograms": "kg",
    "laboratory": "lab",
    "laboratories": "labs",
    "mathematics": "math",
    "microphone": "mic",
    "influenza": "flu",
    "refrigerator": "fridge",
    "gymnasium": "gym",
    "motorcycle": "bike",
    "synchronization": "sync",
    "synchronize": "sync",
    "prohibition": "ban",
    "prohibitions": "bans",
    "telephone": "phone",
    "photograph": "photo",
    "automobile": "car",
    "gasoline": "gas",
}


def main() -> None:
    data = json.loads(DICT_PATH.read_text(encoding="utf-8"))
    entries = data.setdefault("entries", {})
    if not isinstance(entries, dict):
        raise SystemExit("Unexpected entries shape")
    abbrev = data.setdefault("noun_abbreviations", {})

    fixed = 0
    for k, v in LEGACY_FIXES.items():
        if k in entries and entries[k] != v:
            entries[k] = v
            fixed += 1
    removed = 0
    for k in LEGACY_REMOVE:
        if k in entries:
            del entries[k]
            removed += 1

    added_swaps = 0
    for k, v in SAFE_SWAPS.items():
        if k not in entries:
            entries[k] = v
            added_swaps += 1

    added_abbrev = 0
    for k, v in NOUN_ABBREVIATIONS.items():
        # Don't shadow an existing verb/adj entry; noun abbreviations are a separate map.
        if k in entries:
            del entries[k]
        if abbrev.get(k) != v:
            abbrev[k] = v
            added_abbrev += 1

    meta = data.setdefault("_meta", {})
    meta["noun_abbreviation_count"] = len(abbrev)
    meta["noun_abbreviation_policy"] = (
        "Unambiguous standard common-noun abbreviations applied before the "
        "POS noun-gate; entity nouns remain protected."
    )
    meta["entry_count"] = len(entries)
    meta["phrase_count"] = sum(1 for k in entries if " " in k)

    DICT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Legacy fixes: {fixed}, removes: {removed}")
    print(f"Added safe swaps: {added_swaps}")
    print(f"Noun abbreviations: {added_abbrev} (total {len(abbrev)})")
    print(f"Entries: {len(entries)}; phrases: {meta['phrase_count']}")
    print(f"Wrote {DICT_PATH}")


if __name__ == "__main__":
    main()
