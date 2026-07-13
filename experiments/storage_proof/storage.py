"""Physical storage measurement: JSONL, SQLite, gzip, growth checkpoints."""

from __future__ import annotations

import gzip
import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from storage_proof.bm25 import MemoryRecord


@dataclass
class StorageCheckpoint:
    event_count: int
    identity_payload_bytes: int
    minified_payload_bytes: int
    identity_file_bytes: int
    minified_file_bytes: int
    identity_gzip_bytes: int
    minified_gzip_bytes: int
    reduction_pct: float
    minify_latency_ms: float


@dataclass
class StorageReport:
    checkpoints: list[StorageCheckpoint] = field(default_factory=list)
    identity_sqlite_bytes: int = 0
    minified_sqlite_bytes: int = 0
    identity_jsonl_bytes: int = 0
    minified_jsonl_bytes: int = 0
    bytes_per_memory_identity: float = 0.0
    bytes_per_memory_minified: float = 0.0
    growth_slope_bytes_per_event: float = 0.0


def _payload_bytes(records: list[MemoryRecord], minified: bool) -> int:
    total = 0
    for r in records:
        text = r.text
        total += len(text.encode("utf-8"))
    return total


def write_jsonl(records: list[MemoryRecord], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in records:
        lines.append(
            json.dumps(
                {
                    "id": r.record_id,
                    "agent_id": r.agent_id,
                    "namespace": r.namespace,
                    "timestamp": r.timestamp,
                    "session_id": r.session_id,
                    "text": r.text,
                },
                ensure_ascii=False,
            )
        )
    content = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(content, encoding="utf-8")
    return path.stat().st_size


def write_sqlite(records: list[MemoryRecord], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            agent_id TEXT,
            namespace TEXT,
            timestamp INTEGER,
            session_id TEXT,
            text TEXT NOT NULL
        )
        """
    )
    for r in records:
        conn.execute(
            "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?)",
            (r.record_id, r.agent_id, r.namespace, r.timestamp, r.session_id, r.text),
        )
    conn.commit()
    conn.close()
    return path.stat().st_size


def gzip_bytes(data: bytes) -> int:
    return len(gzip.compress(data, compresslevel=6))


def measure_growth(
    identity_records: list[MemoryRecord],
    minified_records: list[MemoryRecord],
    checkpoint_counts: list[int],
    minify_latencies_ms: list[float],
    work_dir: Path,
) -> StorageReport:
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoints: list[StorageCheckpoint] = []

    for i, count in enumerate(checkpoint_counts):
        id_slice = identity_records[:count]
        min_slice = minified_records[:count]
        id_payload = _payload_bytes(id_slice, False)
        min_payload = _payload_bytes(min_slice, False)

        id_jsonl = work_dir / f"identity_{count}.jsonl"
        min_jsonl = work_dir / f"minified_{count}.jsonl"
        id_file = write_jsonl(id_slice, id_jsonl)
        min_file = write_jsonl(min_slice, min_jsonl)

        id_blob = id_jsonl.read_bytes()
        min_blob = min_jsonl.read_bytes()
        id_gz = gzip_bytes(id_blob)
        min_gz = gzip_bytes(min_blob)

        lat = minify_latencies_ms[i] if i < len(minify_latencies_ms) else 0.0
        reduction = 100.0 * (1 - min_payload / id_payload) if id_payload else 0.0
        checkpoints.append(
            StorageCheckpoint(
                event_count=count,
                identity_payload_bytes=id_payload,
                minified_payload_bytes=min_payload,
                identity_file_bytes=id_file,
                minified_file_bytes=min_file,
                identity_gzip_bytes=id_gz,
                minified_gzip_bytes=min_gz,
                reduction_pct=reduction,
                minify_latency_ms=lat,
            )
        )

    full_id_jsonl = work_dir / "identity_full.jsonl"
    full_min_jsonl = work_dir / "minified_full.jsonl"
    full_id_sql = work_dir / "identity_full.sqlite"
    full_min_sql = work_dir / "minified_full.sqlite"

    id_jsonl_size = write_jsonl(identity_records, full_id_jsonl)
    min_jsonl_size = write_jsonl(minified_records, full_min_jsonl)
    id_sql_size = write_sqlite(identity_records, full_id_sql)
    min_sql_size = write_sqlite(minified_records, full_min_sql)

    n = len(identity_records)
    slope = 0.0
    if len(checkpoints) >= 2:
        c0, c1 = checkpoints[0], checkpoints[-1]
        de = c1.event_count - c0.event_count
        if de > 0:
            slope = (c1.identity_payload_bytes - c0.identity_payload_bytes) / de

    return StorageReport(
        checkpoints=checkpoints,
        identity_sqlite_bytes=id_sql_size,
        minified_sqlite_bytes=min_sql_size,
        identity_jsonl_bytes=id_jsonl_size,
        minified_jsonl_bytes=min_jsonl_size,
        bytes_per_memory_identity=id_jsonl_size / max(n, 1),
        bytes_per_memory_minified=min_jsonl_size / max(n, 1),
        growth_slope_bytes_per_event=slope,
    )


def report_to_dict(report: StorageReport) -> dict:
    return {
        "checkpoints": [asdict(c) for c in report.checkpoints],
        "identity_sqlite_bytes": report.identity_sqlite_bytes,
        "minified_sqlite_bytes": report.minified_sqlite_bytes,
        "identity_jsonl_bytes": report.identity_jsonl_bytes,
        "minified_jsonl_bytes": report.minified_jsonl_bytes,
        "bytes_per_memory_identity": report.bytes_per_memory_identity,
        "bytes_per_memory_minified": report.bytes_per_memory_minified,
        "growth_slope_bytes_per_event": report.growth_slope_bytes_per_event,
        "sqlite_reduction_pct": (
            100.0
            * (1 - report.minified_sqlite_bytes / report.identity_sqlite_bytes)
            if report.identity_sqlite_bytes
            else 0.0
        ),
        "jsonl_reduction_pct": (
            100.0
            * (1 - report.minified_jsonl_bytes / report.identity_jsonl_bytes)
            if report.identity_jsonl_bytes
            else 0.0
        ),
    }
