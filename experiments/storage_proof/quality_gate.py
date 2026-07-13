"""Strict QA retention gate: zero per-question regressions vs identity baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

from storage_proof.bm25 import BM25Index, MemoryRecord
from storage_proof.scoring import ReaderResult, extract_answer_from_context, score_prediction


@dataclass
class QARetentionItem:
    qa_id: str
    benchmark: str
    model: str
    question: str
    identity_score: float
    minified_score: float
    retained: bool
    identity_passed: bool
    minified_passed: bool
    category: str = ""


@dataclass
class RetentionReport:
    items: list[QARetentionItem] = field(default_factory=list)
    retention_ratio: float = 1.0
    regressions: int = 0
    models: list[str] = field(default_factory=list)
    benchmarks: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.regressions == 0


def evaluate_store(
    records: list[MemoryRecord],
    qa_items: list,
    model: str,
    store_label: str,
    pinned_ids: list[list[str]] | None = None,
) -> list[ReaderResult]:
    index = BM25Index()
    index.build(records)
    by_id = {r.record_id: r for r in records}
    results: list[ReaderResult] = []
    for i, qa in enumerate(qa_items):
        if pinned_ids and i < len(pinned_ids):
            ctx_records = [by_id[rid] for rid in pinned_ids[i] if rid in by_id]
            context = "\n".join(r.text for r in ctx_records)
        else:
            hits = index.search(qa.question, top_k=5)
            context = "\n".join(r.text for r, _ in hits)
        if model == "bm25_extract":
            pred = extract_answer_from_context(context, qa.answer, qa.keywords)
        elif model == "keyword":
            pred = qa.answer if extract_answer_from_context(context, qa.answer, qa.keywords) else ""
        else:
            pred = extract_answer_from_context(context, qa.answer, qa.keywords)
        results.append(
            score_prediction(model, qa.question, pred, qa.answer, qa.keywords)
        )
    return results


def pinned_retrieval_ids(records: list[MemoryRecord], qa_items: list, top_k: int = 5) -> list[list[str]]:
    index = BM25Index()
    index.build(records)
    pinned: list[list[str]] = []
    for qa in qa_items:
        hits = index.search(qa.question, top_k=top_k)
        pinned.append([r.record_id for r, _ in hits])
    return pinned


def compare_retention(
    identity_records: list[MemoryRecord],
    minified_records: list[MemoryRecord],
    qa_items: list,
    models: list[str],
    benchmark_name: str,
) -> RetentionReport:
    report = RetentionReport()
    report.benchmarks = [benchmark_name]
    report.models = models

    # Pin retrieval to identity-ranked memory units so QA tests content preservation,
    # not BM25 score drift from lexical edits.
    pinned = pinned_retrieval_ids(identity_records, qa_items)

    for model in models:
        id_results = evaluate_store(identity_records, qa_items, model, "identity", pinned)
        min_results = evaluate_store(minified_records, qa_items, model, "minified", pinned)

        for qa, id_r, min_r in zip(qa_items, id_results, min_results):
            retained = min_r.score >= id_r.score - 1e-9
            if id_r.passed and not min_r.passed:
                retained = False

            if not retained:
                report.regressions += 1

            report.items.append(
                QARetentionItem(
                    qa_id=qa.qa_id,
                    benchmark=benchmark_name,
                    model=model,
                    question=qa.question,
                    identity_score=id_r.score,
                    minified_score=min_r.score,
                    retained=retained,
                    identity_passed=id_r.passed,
                    minified_passed=min_r.passed,
                    category=qa.category,
                )
            )

    total = len(report.items)
    kept = sum(1 for i in report.items if i.retained)
    report.retention_ratio = kept / total if total else 1.0
    return report


def prune_unsafe_dictionary_entries(
    failures: list[QARetentionItem],
    replacement_map: dict[str, list[str]],
) -> set[str]:
    """Return dictionary keys linked to QA regressions for optional pruning."""
    unsafe: set[str] = set()
    for item in failures:
        if item.retained:
            continue
        for src, tgts in replacement_map.items():
            for t in tgts:
                if t.lower() in item.question.lower():
                    unsafe.add(src)
    return unsafe


def retention_to_dict(report: RetentionReport) -> dict:
    return {
        "retention_ratio": report.retention_ratio,
        "retention_pct": report.retention_ratio * 100.0,
        "regressions": report.regressions,
        "passed": report.passed,
        "models": report.models,
        "benchmarks": report.benchmarks,
        "items": [asdict(i) for i in report.items],
    }
