#!/usr/bin/env python3
"""Diagnose which dictionary substitutions were applied to the pinned retrieval
contexts for the regressing QA items (qa-01, qa-02)."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from min_mem import MinMemConverter
from min_mem.dictionary import MinDictionary
from dictionary_tiers import load_entries
from storage_proof.adapters.fixtures import build_agent_corpus_bundle
from storage_proof.quality_gate import pinned_retrieval_ids

entries = load_entries(ROOT / "min_dict.json")
converter = MinMemConverter(MinDictionary.from_dict(entries))
bundle = build_agent_corpus_bundle(scale=4)
pinned = pinned_retrieval_ids(bundle.records, bundle.qa_items)
by_id = {r.record_id: r for r in bundle.records}

for idx, qa in enumerate(bundle.qa_items):
    if qa.qa_id not in ("qa-01", "qa-02"):
        continue
    print(f"=== {qa.qa_id}: {qa.question}")
    print(f"  answer: {qa.answer} | keywords: {qa.keywords}")
    ids = pinned[idx]
    for rid in ids:
        r = by_id[rid]
        res = converter.minify_passes(r.text, passes=2)
        if res.replacements:
            print(f"  record {rid} substitutions:")
            for rep in res.replacements:
                print(f"    {rep.original!r} -> {rep.replacement!r}")
            print("  ORIG:", repr(r.text[:200]))
            print("  MIN :", repr(res.minified[:200]))
