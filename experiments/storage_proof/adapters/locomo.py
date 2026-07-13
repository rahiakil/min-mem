"""LoCoMo adapter: official JSON when available, fixture fallback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from storage_proof.adapters.base import BenchmarkAdapter, BenchmarkBundle, QAItem
from storage_proof.adapters.fixtures import build_locomo_fixture
from storage_proof.bm25 import MemoryRecord

VERSION = "locomo-1.0"


class LoCoMoAdapter(BenchmarkAdapter):
    name = "locomo"
    version = VERSION

    def load(self, path: Path | None = None) -> BenchmarkBundle:
        if path is None or not path.exists():
            return build_locomo_fixture()

        data = json.loads(path.read_text(encoding="utf-8"))
        records: list[MemoryRecord] = []
        qa_items: list[QAItem] = []
        ts = 1_700_000_000

        samples = data if isinstance(data, list) else [data]
        for sample in samples[:3]:  # subset for CI speed
            conv = sample.get("conversation", {})
            sid = sample.get("sample_id", "conv")
            for key, val in conv.items():
                if not key.startswith("session_") or key.endswith("_date_time"):
                    continue
                if not isinstance(val, list):
                    continue
                for turn in val:
                    text = turn.get("text", "")
                    if not text:
                        continue
                    records.append(
                        MemoryRecord(
                            record_id=turn.get("dia_id", f"{sid}-{key}"),
                            text=text,
                            agent_id=turn.get("speaker", "unknown"),
                            namespace=f"locomo/{sid}",
                            timestamp=ts,
                            session_id=key,
                        )
                    )
                    ts += 60

            for qa in sample.get("qa", [])[:20]:
                qa_items.append(
                    QAItem(
                        qa_id=str(qa.get("question_id", len(qa_items))),
                        question=qa.get("question", ""),
                        answer=str(qa.get("answer", "")),
                        keywords=[],
                        category=str(qa.get("category", "single_hop")),
                        evidence_ids=qa.get("evidence", []),
                        benchmark="locomo",
                    )
                )

        if not records:
            return build_locomo_fixture()

        h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        return BenchmarkBundle(self.name, VERSION, h, records, qa_items, str(path))
