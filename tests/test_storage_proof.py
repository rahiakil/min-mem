"""Tests for storage growth and QA-retention proof harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from min_mem.converter import MinMemConverter  # noqa: E402
from storage_proof.adapters.fixtures import build_agent_corpus_bundle  # noqa: E402
from storage_proof.bm25 import BM25Index, MemoryRecord  # noqa: E402
from storage_proof.org_simulation import nightly_consolidate, normalize_for_dedup  # noqa: E402
from storage_proof.quality_gate import compare_retention  # noqa: E402
from storage_proof.scoring import token_f1  # noqa: E402
from storage_proof.storage import measure_growth  # noqa: E402


def test_bm25_retrieves_relevant_memory():
    index = BM25Index()
    index.build([
        MemoryRecord("1", "The user prefers Python for data analysis."),
        MemoryRecord("2", "The organization is located in Berlin."),
    ])
    hits = index.search("What language does the user prefer?", top_k=1)
    assert hits
    assert "python" in hits[0][0].text.lower()


def test_token_f1_exact_match():
    assert token_f1("Python", "Python") == 1.0


def test_storage_checkpoints_shrink():
    records = [
        MemoryRecord(f"id-{i}", "The user prefers to utilize Python in order to accomplish tasks.")
        for i in range(20)
    ]
    converter = MinMemConverter()
    minified = [
        MemoryRecord(r.record_id, converter.minify(r.text).minified, r.agent_id, r.namespace, r.timestamp)
        for r in records
    ]
    report = measure_growth(records, minified, [10, 20], [1.0, 1.0], Path("/tmp/min_mem_storage_test"))
    assert report.checkpoints[0].minified_payload_bytes < report.checkpoints[0].identity_payload_bytes
    assert report.checkpoints[0].reduction_pct > 0


def test_org_dedup_reduces_bytes():
    text = "The user prefers to utilize Python in order to accomplish tasks."
    records = [
        MemoryRecord("a1", text, "writer_0", "org", 1),
        MemoryRecord("a2", text, "writer_1", "org", 2),
        MemoryRecord("a3", text, "writer_2", "org", 3),
    ]
    converter = MinMemConverter()
    _, result = nightly_consolidate(records, lambda t: converter.minify(t).minified)
    assert result.dedup_removed_count >= 2
    assert result.shared_consolidated_bytes < result.naive_replicated_bytes


def test_retention_no_regression_on_agent_corpus():
    bundle = build_agent_corpus_bundle(scale=2)
    converter = MinMemConverter()
    identity = bundle.records
    minified = [
        MemoryRecord(r.record_id, converter.minify_passes(r.text).minified, r.agent_id, r.namespace, r.timestamp)
        for r in identity
    ]
    report = compare_retention(identity, minified, bundle.qa_items, ["bm25_extract", "keyword"], "agent_corpus")
    assert report.passed, f"Regressions: {report.regressions}"


def test_tiny_policy_beats_fifo_under_budget():
    from storage_proof.tiny_policy import evaluate_policy_budget

    bundle = build_agent_corpus_bundle(scale=1)
    converter = MinMemConverter()
    minified = [
        MemoryRecord(
            r.record_id,
            converter.minify(r.text).minified,
            r.agent_id,
            r.namespace,
            r.timestamp,
            r.session_id,
        )
        for r in bundle.records
    ]
    report = evaluate_policy_budget(
        bundle.records,
        minified,
        bundle.qa_items,
        budgets=[0.25],
        policies=["fifo_oldest", "tiny_linear"],
    )
    headline = report["headline"]
    assert headline["tiny_at_25_retention_pct"] >= headline["fifo_oldest_at_25_retention_pct"]
    assert headline["tiny_at_25_bytes_reduction_pct"] > 0


def test_normalize_for_dedup():
    assert normalize_for_dedup("  Hello   World ") == "hello world"
