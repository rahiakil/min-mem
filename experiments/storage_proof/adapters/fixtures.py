"""Fixture builders from corpus.json and LoCoMo/MemBench-shaped subsets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from storage_proof.adapters.base import BenchmarkBundle, QAItem
from storage_proof.bm25 import MemoryRecord

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "experiments" / "corpus.json"
FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _hash_payload(obj: object) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_corpus_samples() -> list[dict]:
    return json.loads(CORPUS.read_text(encoding="utf-8"))["samples"]


def build_agent_corpus_records(scale: int = 4) -> list[MemoryRecord]:
    samples = load_corpus_samples()
    records: list[MemoryRecord] = []
    ts = 1_700_000_000
    for rep in range(scale):
        for s in samples:
            records.append(
                MemoryRecord(
                    record_id=f"{s['id']}-r{rep}",
                    text=s["text"],
                    agent_id=f"writer_{rep % 3}",
                    namespace="org/shared",
                    timestamp=ts,
                    session_id=f"session-{rep}-{s['id']}",
                )
            )
            ts += 3600
    return records


AGENT_QA: list[QAItem] = [
    QAItem("qa-01", "What language does the user prefer?", "Python", ["python"], "preference", ["prefs-01"], "agent_corpus"),
    QAItem("qa-02", "Where is the user's organization located?", "Berlin", ["berlin"], "factual", ["facts-01"], "agent_corpus"),
    QAItem("qa-03", "What note-taking tool does the user use?", "Obsidian", ["obsidian"], "preference", ["mixed-01"], "agent_corpus"),
    QAItem("qa-04", "Does the user prefer strict typing?", "TypeScript", ["typescript", "typing"], "preference", ["prefs-02"], "agent_corpus"),
    QAItem("qa-05", "What architecture does the project use?", "microservices", ["microservices", "docker"], "project", ["project-01"], "agent_corpus"),
    QAItem("qa-06", "What database does the application use?", "PostgreSQL", ["postgresql", "postgres"], "factual", ["facts-02"], "agent_corpus"),
    QAItem("qa-07", "What does min-mem implement?", "lexical normalization", ["lexical", "normalization", "min-mem"], "project", ["mixed-02"], "agent_corpus"),
    QAItem("qa-08", "What CI tool was previously used?", "GitHub Actions", ["github", "actions"], "project", ["entities-01"], "agent_corpus"),
]


def build_locomo_fixture(scale: int = 8) -> BenchmarkBundle:
    """LoCoMo-shaped multi-session conversational memory from agent corpus."""
    path = FIXTURES / "locomo_subset.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        records = [
            MemoryRecord(
                record_id=r.get("record_id", r.get("id", f"rec-{i}")),
                text=r["text"],
                agent_id=r.get("agent_id", "speaker_a"),
                namespace=r.get("namespace", "conversation/conv-01"),
                timestamp=r.get("timestamp", 0),
                session_id=r.get("session_id", ""),
            )
            for i, r in enumerate(data["records"])
        ]
        qa = [
            QAItem(
                q.get("qa_id", q.get("id", f"qa-{i}")),
                q["question"],
                q["answer"],
                q.get("keywords", []),
                q.get("category", "single_hop"),
                q.get("evidence_ids", []),
                "locomo",
            )
            for i, q in enumerate(data["qa"])
        ]
        return BenchmarkBundle("locomo", "fixture-v1", _hash_payload(data), records, qa, str(path))

    samples = load_corpus_samples()
    records: list[MemoryRecord] = []
    qa_items: list[QAItem] = []
    ts = 1_710_000_000
    for sess in range(scale):
        for i, s in enumerate(samples):
            rid = f"locomo-s{sess}-t{i}"
            records.append(
                MemoryRecord(
                    record_id=rid,
                    text=s["text"],
                    agent_id="speaker_a" if sess % 2 == 0 else "speaker_b",
                    namespace="conversation/conv-01",
                    timestamp=ts,
                    session_id=f"session_{sess + 1}",
                )
            )
            ts += 1800

    # LoCoMo-style QA across categories
    qa_templates = [
        ("locomo-sh-1", "What language does the user prefer for data analysis?", "Python", ["python"], "single_hop"),
        ("locomo-sh-2", "Where is the organization located?", "Berlin", ["berlin"], "single_hop"),
        ("locomo-temp-1", "What did the assistant help construct in the previous session?", "API endpoint", ["api", "endpoint"], "temporal"),
        ("locomo-mh-1", "What tools does the user combine for notes and planning?", "Obsidian markdown", ["obsidian", "markdown"], "multi_hop"),
        ("locomo-od-1", "What container technology does the project use?", "Docker", ["docker"], "open_domain"),
        ("locomo-pref-1", "Does the user prefer TypeScript?", "yes TypeScript", ["typescript"], "preference"),
        ("locomo-db-1", "Which database version is in use?", "PostgreSQL 15", ["postgresql", "15"], "factual"),
        ("locomo-proj-1", "What does min-mem facilitate?", "storage reduction", ["storage", "reduction"], "project"),
    ]
    for qid, q, a, kw, cat in qa_templates:
        qa_items.append(QAItem(qid, q, a, kw, cat, [], "locomo"))

    payload = {"records": [r.__dict__ for r in records], "qa": [q.__dict__ for q in qa_items]}
    FIXTURES.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return BenchmarkBundle("locomo", "fixture-v1", _hash_payload(payload), records, qa_items, str(path))


def build_membench_fixture(scale: int = 6) -> BenchmarkBundle:
    """MemBench-shaped factual + reflective memory at 10k-scale simulation."""
    path = FIXTURES / "membench_subset.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        records = [
            MemoryRecord(
                record_id=r.get("record_id", r.get("id", f"rec-{i}")),
                text=r["text"],
                agent_id=r.get("agent_id", "observer"),
                namespace=r.get("namespace", "membench/observation"),
                timestamp=r.get("timestamp", 0),
                session_id=r.get("session_id", ""),
            )
            for i, r in enumerate(data["records"])
        ]
        qa = [
            QAItem(
                q.get("qa_id", q.get("id", f"qa-{i}")),
                q["question"],
                q["answer"],
                q.get("keywords", []),
                q.get("category", "factual"),
                q.get("evidence_ids", []),
                "membench",
            )
            for i, q in enumerate(data["qa"])
        ]
        return BenchmarkBundle("membench", "fixture-v1", _hash_payload(data), records, qa, str(path))

    samples = load_corpus_samples()
    records: list[MemoryRecord] = []
    ts = 1_720_000_000
    # Simulate 10k-token scale by repeating corpus with observation/participation namespaces
    for rep in range(scale):
        for i, s in enumerate(samples):
            scenario = "participation" if rep % 2 == 0 else "observation"
            level = "factual" if i % 2 == 0 else "reflective"
            records.append(
                MemoryRecord(
                    record_id=f"mb-{scenario}-{level}-r{rep}-i{i}",
                    text=s["text"],
                    agent_id="participant" if scenario == "participation" else "observer",
                    namespace=f"membench/{scenario}/{level}",
                    timestamp=ts,
                    session_id=f"day-{rep}",
                )
            )
            ts += 600

    qa_items = [
        QAItem("mb-fact-1", "What is the user's preferred language?", "Python", ["python"], "factual", [], "membench"),
        QAItem("mb-fact-2", "What city is the organization in?", "Berlin", ["berlin"], "factual", [], "membench"),
        QAItem("mb-refl-1", "What workflow does the user use for notes?", "Second Brain Obsidian", ["obsidian", "second brain"], "reflective", [], "membench"),
        QAItem("mb-part-1", "What did the user request regarding documentation?", "additional documentation", ["documentation"], "participation", [], "membench"),
        QAItem("mb-obs-1", "What architecture pattern is implemented?", "microservices", ["microservices"], "observation", [], "membench"),
        QAItem("mb-upd-1", "What database does the application utilize?", "PostgreSQL", ["postgresql"], "knowledge_update", [], "membench"),
    ]

    payload = {"records": [r.__dict__ for r in records], "qa": [q.__dict__ for q in qa_items]}
    FIXTURES.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return BenchmarkBundle("membench", "fixture-v1", _hash_payload(payload), records, qa_items, str(path))


def _load_agent_qa() -> list[QAItem]:
    """Load agent-corpus QA from experiments/agent_qa.json if present, else AGENT_QA."""
    qa_path = ROOT / "experiments" / "agent_qa.json"
    if qa_path.exists():
        data = json.loads(qa_path.read_text(encoding="utf-8"))
        return [
            QAItem(q.get("qa_id", f"qa-{i}"), q["question"], q["answer"],
                   q.get("keywords", []), q.get("category", ""), q.get("evidence_ids", []),
                   "agent_corpus")
            for i, q in enumerate(data.get("qa", []))
        ]
    return list(AGENT_QA)


def build_agent_corpus_bundle(scale: int = 4) -> BenchmarkBundle:
    records = build_agent_corpus_records(scale)
    qa = _load_agent_qa()
    payload = {"records": [r.__dict__ for r in records], "qa": [q.__dict__ for q in qa]}
    return BenchmarkBundle("agent_corpus", "corpus-v1", _hash_payload(payload), records, qa, str(CORPUS))
