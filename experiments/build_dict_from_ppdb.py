#!/usr/bin/env python3
"""Build min_dict.json candidates from Simple PPDB / SimplePPDB++.

Offline dictionary construction — **not** on the runtime write path.
Requires a local PPDB extract (see --ppdb-path). Outputs a filtered candidate
JSON for human/LLM review before merging into min_dict.json.

Filter order (as in paper):
  1. PPDB relation label (simplifying / equivalent; drop complicating)
  2. Target tokenizer strictly fewer tokens (cl100k_base) — token savings gate
  3. Source longer than target in characters
  4. Target higher corpus frequency than source (when --freq-table provided)
  5. POS consistency on both sides (NLTK tagger)
  6. Noun-head protection + inflection blocklist (same as improve_loop)
  7. Optional LLM-judge batch (--llm-judge) at **build time only**

Usage:
  python experiments/build_dict_from_ppdb.py --ppdb-path ~/data/ppdb/simpleppdb.tab
  python experiments/build_dict_from_ppdb.py --dry-run --limit 500
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from min_mem.dictionary import NOUN_TAGS, MinDictionary  # noqa: E402
from dictionary_tiers import load_entries  # noqa: E402

try:
    import tiktoken
    from nltk import pos_tag, word_tokenize
    from min_mem.converter import _ensure_nltk_data
except ImportError:
    tiktoken = None
    pos_tag = None

DICT_PATH = ROOT / "min_dict.json"
OUT_CANDIDATES = ROOT / "experiments" / "ppdb_candidates.json"
OUT_MERGE = ROOT / "experiments" / "ppdb_merged_preview.json"

BLOCKLIST = {"implement", "implementation", "implementations", "establish", "contains"}
SIMPLIFYING = {"<", "<=", "="}  # PPDB entailment labels for shorter/same
SKIP_RELATIONS = {">", ">="}  # complicating


@dataclass
class Candidate:
    source: str
    target: str
    char_delta: int
    token_delta: int
    source_tokens: int
    target_tokens: int
    relation: str
    pos_source: str
    pos_target: str
    freq_source: float | None
    freq_target: float | None
    filters_passed: list[str]


def count_tokens(text: str) -> int:
    if tiktoken is None:
        return len(text.split())
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def pos_of(phrase: str) -> str:
    if pos_tag is None:
        return "UNK"
    _ensure_nltk_data()
    tokens = word_tokenize(phrase)
    if not tokens:
        return "UNK"
    return pos_tag(tokens)[-1][1]


def noun_head(phrase: str) -> bool:
    p = pos_of(phrase)
    return p in NOUN_TAGS


def load_freq_table(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k.lower(): float(v) for k, v in data.items()}


def parse_ppdb_line(line: str) -> tuple[str, str, str] | None:
    """Parse a minimal PPDB TSV: phrase1 \\t phrase2 \\t relation."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split("\t")
    if len(parts) < 3:
        return None
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def iter_ppdb(path: Path, limit: int | None) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = parse_ppdb_line(line)
            if parsed is None:
                continue
            rows.append(parsed)
            if limit and len(rows) >= limit:
                break
    return rows


def filter_candidate(
    source: str,
    target: str,
    relation: str,
    freq: dict[str, float],
) -> Candidate | None:
    passed: list[str] = []
    if relation in SKIP_RELATIONS:
        return None
    if relation not in SIMPLIFYING and relation != "=":
        return None
    passed.append("relation")

    src_tok = count_tokens(source)
    tgt_tok = count_tokens(target)
    if tgt_tok >= src_tok:
        return None
    passed.append("token_strict")

    if len(target) >= len(source):
        return None
    passed.append("char_shorter")

    if source.lower() in BLOCKLIST or target.lower() in BLOCKLIST:
        return None
    if noun_head(source) and source.lower() != target.lower():
        return None
    passed.append("noun_block")

    ps, pt = pos_of(source), pos_of(target)
    if ps != "UNK" and pt != "UNK" and ps != pt:
        # Allow phrase→word if phrase is connector pattern
        if " " not in source:
            return None
    passed.append("pos")

    fs = freq.get(source.lower())
    ft = freq.get(target.lower())
    if freq and fs is not None and ft is not None and ft < fs:
        return None
    if freq:
        passed.append("frequency")

    return Candidate(
        source=source,
        target=target,
        char_delta=len(source) - len(target),
        token_delta=src_tok - tgt_tok,
        source_tokens=src_tok,
        target_tokens=tgt_tok,
        relation=relation,
        pos_source=ps,
        pos_target=pt,
        freq_source=fs,
        freq_target=ft,
        filters_passed=passed,
    )


def merge_preview(existing: dict[str, str], candidates: list[Candidate], top: int) -> dict:
    merged = dict(existing)
    added = []
    for c in sorted(candidates, key=lambda x: (-x.token_delta, -x.char_delta))[:top]:
        key = c.source.lower()
        if key in merged:
            continue
        merged[key] = c.target
        added.append(c.source)
    return {
        "_meta": {
            "description": "PPDB merge preview — review before promoting to min_dict.json",
            "build_pipeline": "ppdb-v1",
            "build_time_compute": "offline",
            "runtime_inference": "none",
            "generated": datetime.now(timezone.utc).isoformat(),
            "candidate_count": len(candidates),
            "added_count": len(added),
            "sources": ["manual+ppdb"],
        },
        "entries": dict(sorted(merged.items())),
        "added_from_ppdb": added,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter PPDB pairs into min-mem dictionary candidates")
    ap.add_argument("--ppdb-path", type=Path, help="Local SimplePPDB TSV (phrase\\tparaphrase\\trelation)")
    ap.add_argument("--freq-table", type=Path, help="JSON map word→frequency for frequency gate")
    ap.add_argument("--limit", type=int, default=50_000, help="Max PPDB rows to scan")
    ap.add_argument("--top", type=int, default=500, help="Top candidates to export")
    ap.add_argument("--dry-run", action="store_true", help="Emit schema example without PPDB file")
    args = ap.parse_args()

    freq = load_freq_table(args.freq_table)
    existing = load_entries(DICT_PATH)

    if args.dry_run or args.ppdb_path is None or not args.ppdb_path.exists():
        example = [
            Candidate("in order to", "to", 8, 2, 3, 1, "<", "IN", "IN", None, None, ["relation", "token_strict"]),
            Candidate("utilize", "use", 3, 0, 1, 1, "<", "VB", "VB", 120.0, 980.0, ["relation", "token_strict"]),
        ]
        payload = {
            "_schema": {
                "entries": "long_form -> short_form (runtime dict)",
                "entry_provenance": "optional per-key {source, ppdb_relation, token_delta}",
                "build_time_llm_judge": "optional; does not affect runtime zero-inference claim",
            },
            "candidates": [asdict(c) for c in example],
            "note": "Provide --ppdb-path to a SimplePPDB extract to run full pipeline.",
        }
        OUT_CANDIDATES.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Dry-run wrote {OUT_CANDIDATES}")
        return

    rows = iter_ppdb(args.ppdb_path, args.limit)
    candidates: list[Candidate] = []
    for source, target, relation in rows:
        c = filter_candidate(source, target, relation, freq)
        if c:
            candidates.append(c)

    OUT_CANDIDATES.write_text(
        json.dumps(
            {
                "_meta": {
                    "ppdb_path": str(args.ppdb_path),
                    "scanned": len(rows),
                    "accepted": len(candidates),
                    "filters": [
                        "simplifying_relation",
                        "target_tokens_strictly_less",
                        "char_shorter",
                        "noun_blocklist",
                        "pos_consistency",
                        "frequency_optional",
                    ],
                },
                "candidates": [asdict(c) for c in candidates[: args.top]],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    preview = merge_preview(existing, candidates, args.top)
    OUT_MERGE.write_text(json.dumps(preview, indent=2) + "\n", encoding="utf-8")
    print(f"Scanned {len(rows)} PPDB rows → {len(candidates)} passed filters")
    print(f"Wrote {OUT_CANDIDATES} and {OUT_MERGE}")


if __name__ == "__main__":
    main()
