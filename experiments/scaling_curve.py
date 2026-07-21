"""Scaling-curve experiment: savings and QA retention at 1x/5x/25x/100x corpus,
with extrapolation to a local-LLM user.

Minification is per-record, so character-savings *rate* is scale-invariant; what
scales is absolute bytes and context tokens. This script measures both and
extrapolates disk/network/context savings for a hypothetical local-model user
running a 4B-class model on commodity hardware.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from min_mem.converter import MinMemConverter
from min_mem.dictionary import MinDictionary
from storage_proof.adapters.fixtures import build_agent_corpus_records, _load_agent_qa
from storage_proof.bm25 import MemoryRecord
from storage_proof.quality_gate import compare_retention

DICT = ROOT / "min_dict.json"
CORPUS = ROOT / "experiments" / "corpus.json"
OUT_JSON = ROOT / "experiments" / "scaling_curve_results.json"

SCALES = [1, 5, 25, 100]
# Local-model user extrapolation scenario.
LOCAL_MODEL_TOKENS_PER_SEC = 50.0   # 4B-class model on a consumer GPU
LOCAL_MODEL_CONTEXT_TOKENS = 8000  # typical context budget
BYTES_PER_RECORD_AVG = 200          # ~mean record size


def tok_count(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


def minify_records(records, converter):
    out = []
    for r in records:
        m = converter.minify_passes(r.text, passes=2).minified
        out.append(MemoryRecord(record_id=r.record_id, text=m, agent_id=r.agent_id,
                                 namespace=r.namespace, timestamp=r.timestamp,
                                 session_id=r.session_id))
    return out


def measure_scale(scale: int, converter, qa_items) -> dict:
    records = build_agent_corpus_records(scale=scale)
    minified = minify_records(records, converter)

    id_bytes = sum(len(r.text.encode("utf-8")) for r in records)
    mf_bytes = sum(len(r.text.encode("utf-8")) for r in minified)
    id_tokens = sum(tok_count(r.text) for r in records)
    mf_tokens = sum(tok_count(r.text) for r in minified)

    # Deterministic QA retention on a capped QA sample (QA items are passage-level,
    # so they do not multiply with scale; measure once on the base set).
    if scale == 1 and qa_items:
        rep = compare_retention(records, minified, qa_items,
                                ["bm25_extract"], "agent_corpus")
        det_retention = round(100.0 * rep.retention_ratio, 1)
        det_regressions = rep.regressions
    else:
        det_retention = None
        det_regressions = None

    return {
        "scale": scale,
        "records": len(records),
        "identity_bytes": id_bytes,
        "minified_bytes": mf_bytes,
        "bytes_saved": id_bytes - mf_bytes,
        "char_savings_pct": round(100.0 * (1 - mf_bytes / id_bytes), 2),
        "identity_tokens": id_tokens,
        "minified_tokens": mf_tokens,
        "tokens_saved": id_tokens - mf_tokens,
        "token_savings_pct": round(100.0 * (1 - mf_tokens / max(id_tokens, 1)), 2),
        "det_qa_retention_pct": det_retention,
        "det_qa_regressions": det_regressions,
    }


def extrapolate_local_user(base: dict) -> dict:
    """A local-model user with N memories: project disk, context, and latency savings."""
    rate = base["char_savings_pct"] / 100.0
    tok_rate = base["token_savings_pct"] / 100.0
    scenarios = []
    for n_records in [1_000, 10_000, 100_000, 1_000_000]:
        disk_identity_mb = n_records * BYTES_PER_RECORD_AVG / 1_048_576
        disk_saved_mb = disk_identity_mb * rate
        ctx_identity_tokens = n_records * 38  # ~mean tokens per record
        ctx_saved_tokens = ctx_identity_tokens * tok_rate
        # Full-context read latency on a local 4B model at LOCAL_MODEL_TOKENS_PER_SEC
        latency_identity_s = ctx_identity_tokens / LOCAL_MODEL_TOKENS_PER_SEC
        latency_saved_s = ctx_saved_tokens / LOCAL_MODEL_TOKENS_PER_SEC
        scenarios.append({
            "n_records": n_records,
            "disk_identity_mb": round(disk_identity_mb, 1),
            "disk_saved_mb": round(disk_saved_mb, 1),
            "disk_saved_pct": round(100 * rate, 2),
            "ctx_identity_tokens": int(ctx_identity_tokens),
            "ctx_saved_tokens": int(ctx_saved_tokens),
            "ctx_saved_pct": round(100 * tok_rate, 2),
            "latency_identity_s": round(latency_identity_s, 1),
            "latency_saved_s": round(latency_saved_s, 1),
        })
    return {
        "local_model_tokens_per_sec": LOCAL_MODEL_TOKENS_PER_SEC,
        "local_model_context_tokens": LOCAL_MODEL_CONTEXT_TOKENS,
        "scenarios": scenarios,
    }


def main() -> None:
    converter = MinMemConverter(MinDictionary.from_path(DICT))
    qa_items = _load_agent_qa()

    points = []
    for s in SCALES:
        print(f"measuring scale {s}x ...", flush=True)
        points.append(measure_scale(s, converter, qa_items))

    base = points[0]
    extrapolation = extrapolate_local_user(base)

    result = {
        "scales": points,
        "extrapolation": extrapolation,
        "note": ("Character-savings rate is per-record and scale-invariant; absolute "
                 "bytes and context tokens scale linearly with the record count."),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
