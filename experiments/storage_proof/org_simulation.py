"""Multi-agent organizational memory consolidation (deterministic dreaming)."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Callable

from storage_proof.bm25 import BM25Index, MemoryRecord


@dataclass
class OrgConsolidationResult:
    writer_agents: int
    private_agent_totals_bytes: dict[str, int]
    naive_replicated_bytes: int
    shared_consolidated_bytes: int
    dedup_removed_count: int
    consolidation_reduction_pct: float
    cross_agent_record_count: int
    nightly_batches: int = 1


def _text_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def normalize_for_dedup(text: str) -> str:
    return " ".join(text.lower().split())


def nightly_consolidate(
    records: list[MemoryRecord],
    minify_fn: Callable[[str], str],
    writer_count: int = 3,
) -> tuple[list[MemoryRecord], OrgConsolidationResult]:
    """Deterministic org batch: minify + exact dedup on normalized text."""
    by_agent: dict[str, list[MemoryRecord]] = {}
    for r in records:
        by_agent.setdefault(r.agent_id, []).append(r)

    private_totals = {
        agent: sum(_text_bytes(r.text) for r in recs)
        for agent, recs in by_agent.items()
    }
    naive_replicated = sum(private_totals.values())

    seen_hashes: set[str] = set()
    consolidated: list[MemoryRecord] = []
    dedup_removed = 0

    # Process in timestamp order across all agents
    sorted_records = sorted(records, key=lambda r: (r.timestamp, r.record_id))
    for r in sorted_records:
        minified_text = minify_fn(r.text)
        norm = normalize_for_dedup(minified_text)
        h = hashlib.sha256(norm.encode()).hexdigest()
        if h in seen_hashes:
            dedup_removed += 1
            continue
        seen_hashes.add(h)
        consolidated.append(
            MemoryRecord(
                record_id=f"org-{r.record_id}",
                text=minified_text,
                agent_id="org_reader",
                namespace="org/consolidated",
                timestamp=r.timestamp,
                session_id=r.session_id,
            )
        )

    shared_bytes = sum(_text_bytes(r.text) for r in consolidated)
    reduction = 100.0 * (1 - shared_bytes / naive_replicated) if naive_replicated else 0.0

    result = OrgConsolidationResult(
        writer_agents=writer_count,
        private_agent_totals_bytes=private_totals,
        naive_replicated_bytes=naive_replicated,
        shared_consolidated_bytes=shared_bytes,
        dedup_removed_count=dedup_removed,
        consolidation_reduction_pct=reduction,
        cross_agent_record_count=len(consolidated),
    )
    return consolidated, result


def cross_agent_retrieve(
    consolidated: list[MemoryRecord],
    question: str,
    top_k: int = 5,
) -> str:
    index = BM25Index()
    index.build(consolidated)
    hits = index.search(question, top_k=top_k)
    return "\n".join(r.text for r, _ in hits)


def org_result_to_dict(result: OrgConsolidationResult) -> dict:
    return asdict(result)
