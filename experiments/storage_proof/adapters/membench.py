"""MemBench adapter: official data when available, fixture fallback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from storage_proof.adapters.base import BenchmarkAdapter, BenchmarkBundle, QAItem
from storage_proof.adapters.fixtures import build_membench_fixture
from storage_proof.bm25 import MemoryRecord

VERSION = "membench-1.0"


class MemBenchAdapter(BenchmarkAdapter):
    name = "membench"
    version = VERSION

    def load(self, path: Path | None = None) -> BenchmarkBundle:
        if path is None or not path.exists():
            return build_membench_fixture()

        data = json.loads(path.read_text(encoding="utf-8"))
        records: list[MemoryRecord] = []
        qa_items: list[QAItem] = []
        ts = 1_720_000_000

        items = data if isinstance(data, list) else data.get("samples", data.get("data", []))
        for item in items[:50]:
            mem_text = item.get("memory") or item.get("content") or item.get("text", "")
            if mem_text:
                records.append(
                    MemoryRecord(
                        record_id=str(item.get("id", len(records))),
                        text=mem_text,
                        agent_id=item.get("agent_id", "observer"),
                        namespace=item.get("scenario", "membench"),
                        timestamp=ts,
                        session_id=str(item.get("session", "")),
                    )
                )
                ts += 120
            q = item.get("question")
            a = item.get("answer")
            if q and a:
                qa_items.append(
                    QAItem(
                        qa_id=str(item.get("id", len(qa_items))),
                        question=q,
                        answer=str(a),
                        keywords=item.get("keywords", []),
                        category=item.get("category", "factual"),
                        benchmark="membench",
                    )
                )

        if not records:
            return build_membench_fixture()

        h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        return BenchmarkBundle(self.name, VERSION, h, records, qa_items, str(path))
