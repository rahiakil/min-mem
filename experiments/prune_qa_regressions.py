#!/usr/bin/env python3
"""Prune dictionary entries that cause QA regressions, recovering 100% retention.

Strategy
--------
1. Build the agent_corpus bundle (records + qa_items).
2. Run the deterministic QA retention gate (bm25_extract + keyword models,
   no LLM) with the full dictionary; collect the regressing QA items.
3. For each regressing QA, look at the pinned retrieval records, minify them,
   and collect the source keys of the substitutions applied to those records
   --- those are the candidate culprit entries.
4. Greedy: remove candidate culprits one at a time, keeping a removal if it
   reduces the regression count, until regressions == 0 (or no improvement).
5. Save the pruned dictionary (base + safe new entries) to min_dict.json and
   the bundled copy.

Fast: each retention check is keyword/BM25 only (~ms), so the greedy sweep
over the small candidate set is cheap.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from min_mem import MinMemConverter  # noqa: E402
from min_mem.dictionary import MinDictionary  # noqa: E402

from dictionary_tiers import load_entries  # noqa: E402
from storage_proof.adapters.fixtures import build_agent_corpus_bundle  # noqa: E402
from storage_proof.bm25 import MemoryRecord  # noqa: E402
from storage_proof.quality_gate import compare_retention, pinned_retrieval_ids  # noqa: E402

DICT_PATH = ROOT / "min_dict.json"
BUNDLED_PATH = ROOT / "src" / "min_mem" / "data" / "min_dict.json"
BASELINE_SIZE = 873  # the committed 873-entry dict had 100% retention
MODELS = ["bm25_extract", "keyword"]


def log(msg: str) -> None:
    print(msg, flush=True)


def regressions_for(entries: dict[str, str], bundle, pinned) -> tuple[int, list]:
    converter = MinMemConverter(MinDictionary.from_dict(entries))
    minified = []
    for r in bundle.records:
        m = converter.minify_passes(r.text, passes=2).minified
        minified.append(MemoryRecord(record_id=r.record_id, text=m,
                                     agent_id=r.agent_id, namespace=r.namespace,
                                     timestamp=r.timestamp, session_id=r.session_id))
    rep = compare_retention(bundle.records, minified, bundle.qa_items, MODELS, bundle.name)
    return rep.regressions, rep.items


def substitutions_on_records(entries: dict[str, str], records: list[MemoryRecord]) -> set[str]:
    converter = MinMemConverter(MinDictionary.from_dict(entries))
    sources: set[str] = set()
    for r in records:
        res = converter.minify_passes(r.text, passes=2)
        for rep in res.replacements:
            sources.add(rep.original.lower())
    return sources


def main() -> None:
    entries = load_entries(DICT_PATH)
    new_keys = [k for k in entries if k not in _baseline_keys()]
    log(f"Loaded dict: {len(entries)} entries ({len(new_keys)} new vs {BASELINE_SIZE}-entry baseline)")

    bundle = build_agent_corpus_bundle(scale=4)
    pinned = pinned_retrieval_ids(bundle.records, bundle.qa_items)
    log(f"Bundle: {len(bundle.records)} records, {len(bundle.qa_items)} qa items")

    reg, items = regressions_for(entries, bundle, pinned)
    log(f"Full dict regressions: {reg}")
    if reg == 0:
        log("Already zero regressions; nothing to prune.")
        return

    # Identify candidate culprit entries from the regressing QA items' pinned records
    by_id = {r.record_id: r for r in bundle.records}
    candidates: set[str] = set()
    for it in items:
        if it.retained:
            continue
        idx = next((i for i, qa in enumerate(bundle.qa_items) if qa.qa_id == it.qa_id), None)
        if idx is None or idx >= len(pinned):
            continue
        recs = [by_id[rid] for rid in pinned[idx] if rid in by_id]
        candidates |= substitutions_on_records(entries, recs)
    # Only consider culprits that are NEW entries (base is known-safe at 100%)
    candidates &= set(new_keys)
    log(f"Candidate culprit entries (new, on regressing contexts): {len(candidates)}")

    if not candidates:
        log("No new-entry culprits found on regressing contexts; cannot prune further.")
        return

    # Greedy removal: remove the candidate that most reduces regressions, repeat.
    current = dict(entries)
    removed: list[str] = []
    while True:
        cur_reg, _ = regressions_for(current, bundle, pinned)
        if cur_reg == 0:
            break
        best_key, best_reg = None, cur_reg
        for key in candidates:
            if key not in current:
                continue
            trial = {k: v for k, v in current.items() if k != key}
            r, _ = regressions_for(trial, bundle, pinned)
            if r < best_reg:
                best_reg, best_key = r, key
        if best_key is None:
            log(f"No single removal reduces regressions below {cur_reg}; stopping.")
            break
        del current[best_key]
        removed.append(best_key)
        log(f"  removed '{best_key}' -> regressions {best_reg}")

    final_reg, _ = regressions_for(current, bundle, pinned)
    log(f"Final: {len(current)} entries, regressions={final_reg}")
    log(f"Removed {len(removed)} culprit entries: {removed}")

    if final_reg != 0:
        log("WARNING: could not reach zero regressions via single-entry removal; "
            "consider pairwise search.")

    # Save
    _save(current)
    log(f"Saved pruned dictionary: {len(current)} entries.")


def _baseline_keys() -> set[str]:
    """Keys of the committed 873-entry baseline, recovered from git."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "show", "HEAD:min_dict.json"],
            cwd=str(ROOT), capture_output=True, text=True, check=True,
        )
        data = json.loads(out.stdout)
        return set((data.get("entries") or data).keys())
    except Exception:
        return set()


def _save(entries: dict[str, str]) -> None:
    from datetime import datetime, timezone
    payload = {
        "_meta": {
            "description": "Minimal synonym map: longer form -> shortest equivalent. Nouns are never replaced at runtime (POS-gated).",
            "version": "expanded-corpora-2026-07-pruned",
            "runtime_inference": "none",
            "build_time_compute": "offline_wordnet_corpora_qa_pruned",
            "entry_count": len(entries),
            "phrase_count": sum(1 for k in entries if " " in k),
            "last_improved": datetime.now(timezone.utc).isoformat(),
        },
        "entries": dict(sorted(entries.items())),
    }
    text = json.dumps(payload, indent=2) + "\n"
    DICT_PATH.write_text(text, encoding="utf-8")
    BUNDLED_PATH.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
