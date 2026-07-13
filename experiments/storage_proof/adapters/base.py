"""Benchmark adapter base types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from storage_proof.bm25 import MemoryRecord


@dataclass
class QAItem:
    qa_id: str
    question: str
    answer: str
    keywords: list[str] = field(default_factory=list)
    category: str = "single_hop"
    evidence_ids: list[str] = field(default_factory=list)
    benchmark: str = ""


@dataclass
class BenchmarkBundle:
    name: str
    version: str
    data_hash: str
    records: list[MemoryRecord]
    qa_items: list[QAItem]
    source_path: str = "fixture"


class BenchmarkAdapter:
    name: str = "base"
    version: str = "1.0"

    def load(self, path: Path | None = None) -> BenchmarkBundle:
        raise NotImplementedError
