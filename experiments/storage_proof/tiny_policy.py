"""SuperCompress-style tiny CPU read-path selection policy.

A policy decides WHICH memories to keep under a context/token budget.
Min-Mem's write path still rewrites WHAT is stored (deterministic dictionary).

This module deliberately uses cheap features only:
  - BM25 relevance to the query
  - recency
  - entity/noun density proxy (capitalized tokens + digit tokens)
  - length penalty (prefer denser packs)

It is a linear feature scorer (tiny policy), not a large neural compressor.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass, field

from storage_proof.bm25 import BM25Index, MemoryRecord, tokenize
from storage_proof.scoring import extract_answer_from_context, score_prediction


_ENTITY_RE = re.compile(r"\b([A-Z][a-zA-Z0-9_-]+|\d+(?:\.\d+)?)\b")


@dataclass
class ScoredMemory:
    record: MemoryRecord
    score: float
    features: dict[str, float]


@dataclass
class PolicyBudgetResult:
    policy: str
    budget_ratio: float
    kept_count: int
    total_count: int
    kept_bytes: int
    identity_context_bytes: int
    bytes_reduction_pct: float
    retention_pct: float
    regressions: int
    comparisons: int
    latency_ms: float
    items: list[dict] = field(default_factory=list)


def entity_density(text: str) -> float:
    toks = tokenize(text)
    if not toks:
        return 0.0
    ents = _ENTITY_RE.findall(text)
    return min(1.0, len(ents) / max(len(toks), 1))


def length_penalty(text: str, ref_len: float = 200.0) -> float:
    return 1.0 / (1.0 + abs(len(text) - ref_len) / ref_len)


def score_features(
    record: MemoryRecord,
    bm25_score: float,
    max_ts: int,
    min_ts: int,
) -> dict[str, float]:
    span = max(max_ts - min_ts, 1)
    recency = (record.timestamp - min_ts) / span
    return {
        "bm25": float(bm25_score),
        "recency": float(recency),
        "entity_density": entity_density(record.text),
        "length_penalty": length_penalty(record.text),
        "char_len": float(len(record.text)),
    }


def linear_policy_score(features: dict[str, float]) -> float:
    """Tiny linear policy (~4 weights), SuperCompress-inspired."""
    return (
        1.00 * features["bm25"]
        + 0.35 * features["recency"]
        + 0.55 * features["entity_density"]
        + 0.25 * features["length_penalty"]
    )


def select_under_budget(
    records: list[MemoryRecord],
    query: str,
    budget_ratio: float,
    policy: str = "tiny_linear",
) -> tuple[list[MemoryRecord], list[ScoredMemory], float]:
    """Select a budgeted subset of memories for a query."""
    t0 = time.perf_counter()
    if not records:
        return [], [], 0.0

    keep_n = max(1, int(math.ceil(len(records) * budget_ratio)))
    max_ts = max(r.timestamp for r in records)
    min_ts = min(r.timestamp for r in records)

    if policy in {"fifo", "fifo_newest"}:
        ordered = sorted(records, key=lambda r: r.timestamp, reverse=True)
        selected = ordered[:keep_n]
        scored = [
            ScoredMemory(r, float(keep_n - i), {"rank": float(i)})
            for i, r in enumerate(selected)
        ]
        return selected, scored, (time.perf_counter() - t0) * 1000

    if policy == "fifo_oldest":
        ordered = sorted(records, key=lambda r: r.timestamp)
        selected = ordered[:keep_n]
        scored = [
            ScoredMemory(r, float(keep_n - i), {"rank": float(i)})
            for i, r in enumerate(selected)
        ]
        return selected, scored, (time.perf_counter() - t0) * 1000

    index = BM25Index()
    index.build(records)
    hits = {r.record_id: sc for r, sc in index.search(query, top_k=len(records))}

    scored: list[ScoredMemory] = []
    for r in records:
        feats = score_features(r, hits.get(r.record_id, 0.0), max_ts, min_ts)
        if policy == "bm25_only":
            score = feats["bm25"]
        else:
            score = linear_policy_score(feats)
        scored.append(ScoredMemory(r, score, feats))

    scored.sort(key=lambda s: s.score, reverse=True)
    selected = [s.record for s in scored[:keep_n]]
    return selected, scored[:keep_n], (time.perf_counter() - t0) * 1000


def evaluate_policy_budget(
    identity_records: list[MemoryRecord],
    minified_records: list[MemoryRecord],
    qa_items: list,
    budgets: list[float] = (0.25, 0.35, 0.5),
    policies: list[str] = ("fifo_oldest", "fifo_newest", "bm25_only", "tiny_linear"),
) -> dict:
    """Compare budgeted policies on identity and minified stores."""
    by_id_min = {r.record_id: r for r in minified_records}
    results: list[PolicyBudgetResult] = []

    for policy in policies:
        for budget in budgets:
            comparisons = 0
            regressions = 0
            kept_bytes = 0
            full_bytes = 0
            items = []
            latencies = []

            for qa in qa_items:
                selected_id, scored, latency = select_under_budget(
                    identity_records, qa.question, budget, policy=policy
                )
                latencies.append(latency)
                selected_min = [
                    by_id_min[r.record_id]
                    for r in selected_id
                    if r.record_id in by_id_min
                ]
                full_ctx = "\n".join(r.text for r in minified_records)
                budget_ctx = "\n".join(r.text for r in selected_min)

                full_pred = extract_answer_from_context(full_ctx, qa.answer, qa.keywords)
                budget_pred = extract_answer_from_context(budget_ctx, qa.answer, qa.keywords)
                full_r = score_prediction("keyword", qa.question, full_pred, qa.answer, qa.keywords)
                budget_r = score_prediction(
                    "keyword", qa.question, budget_pred, qa.answer, qa.keywords
                )
                retained = budget_r.score >= full_r.score - 1e-9 and (
                    not full_r.passed or budget_r.passed
                )
                comparisons += 1
                if not retained:
                    regressions += 1
                kept_bytes += sum(len(r.text.encode()) for r in selected_min)
                full_bytes += sum(len(r.text.encode()) for r in minified_records)
                items.append(
                    {
                        "qa_id": qa.qa_id,
                        "retained": retained,
                        "kept_count": len(selected_min),
                        "full_score": full_r.score,
                        "budget_score": budget_r.score,
                        "top_features": scored[0].features if scored else {},
                    }
                )

            n = max(len(qa_items), 1)
            mean_kept = kept_bytes / n
            mean_full = full_bytes / n
            results.append(
                PolicyBudgetResult(
                    policy=policy,
                    budget_ratio=budget,
                    kept_count=int(round(sum(i["kept_count"] for i in items) / n)),
                    total_count=len(minified_records),
                    kept_bytes=int(mean_kept),
                    identity_context_bytes=int(mean_full),
                    bytes_reduction_pct=100.0 * (1 - mean_kept / mean_full) if mean_full else 0.0,
                    retention_pct=100.0 * (1 - regressions / max(comparisons, 1)),
                    regressions=regressions,
                    comparisons=comparisons,
                    latency_ms=sum(latencies) / max(len(latencies), 1),
                    items=items,
                )
            )

    def find(policy: str, budget: float) -> PolicyBudgetResult | None:
        return next(
            (
                r
                for r in results
                if r.policy == policy and abs(r.budget_ratio - budget) < 1e-9
            ),
            None,
        )

    tiny_25 = find("tiny_linear", 0.25)
    tiny_35 = find("tiny_linear", 0.35)
    fifo_old_25 = find("fifo_oldest", 0.25)
    fifo_new_25 = find("fifo_newest", 0.25)
    bm25_25 = find("bm25_only", 0.25)

    return {
        "description": (
            "SuperCompress-style read-path validation: tiny linear CPU policy "
            "selects which already-minified memories to keep under a budget."
        ),
        "write_path_neural": False,
        "read_path_policy": True,
        "budgets": list(budgets),
        "policies": list(policies),
        "results": [asdict(r) for r in results],
        "headline": {
            "tiny_at_25_retention_pct": tiny_25.retention_pct if tiny_25 else 0.0,
            "tiny_at_35_retention_pct": tiny_35.retention_pct if tiny_35 else 0.0,
            "fifo_oldest_at_25_retention_pct": fifo_old_25.retention_pct if fifo_old_25 else 0.0,
            "fifo_newest_at_25_retention_pct": fifo_new_25.retention_pct if fifo_new_25 else 0.0,
            "bm25_at_25_retention_pct": bm25_25.retention_pct if bm25_25 else 0.0,
            "tiny_vs_fifo_oldest_lift_at_25": (
                (tiny_25.retention_pct - fifo_old_25.retention_pct)
                if tiny_25 and fifo_old_25
                else 0.0
            ),
            "tiny_vs_bm25_lift_at_25": (
                (tiny_25.retention_pct - bm25_25.retention_pct)
                if tiny_25 and bm25_25
                else 0.0
            ),
            "tiny_at_25_bytes_reduction_pct": tiny_25.bytes_reduction_pct if tiny_25 else 0.0,
            "tiny_at_35_bytes_reduction_pct": tiny_35.bytes_reduction_pct if tiny_35 else 0.0,
            "tiny_at_25_latency_ms": tiny_25.latency_ms if tiny_25 else 0.0,
            "tiny_at_35_latency_ms": tiny_35.latency_ms if tiny_35 else 0.0,
            # Backward-compatible aliases used by report markdown
            "fifo_at_35_retention_pct": (
                find("fifo_oldest", 0.35).retention_pct if find("fifo_oldest", 0.35) else 0.0
            ),
            "tiny_vs_fifo_lift_at_35": (
                (tiny_35.retention_pct - find("fifo_oldest", 0.35).retention_pct)
                if tiny_35 and find("fifo_oldest", 0.35)
                else 0.0
            ),
        },
    }
