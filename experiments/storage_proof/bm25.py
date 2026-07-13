"""Deterministic BM25 retrieval over memory records (no neural models)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class MemoryRecord:
    record_id: str
    text: str
    agent_id: str = "default"
    namespace: str = "org"
    timestamp: int = 0
    session_id: str = ""


class BM25Index:
    """Okapi BM25 with fixed hyperparameters for reproducibility."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[MemoryRecord] = []
        self._doc_tokens: list[list[str]] = []
        self._df: dict[str, int] = {}
        self._avgdl = 0.0

    def add(self, record: MemoryRecord) -> None:
        toks = tokenize(record.text)
        self._docs.append(record)
        self._doc_tokens.append(toks)
        for t in set(toks):
            self._df[t] = self._df.get(t, 0) + 1
        self._avgdl = sum(len(d) for d in self._doc_tokens) / max(len(self._doc_tokens), 1)

    def build(self, records: list[MemoryRecord]) -> None:
        self._docs.clear()
        self._doc_tokens.clear()
        self._df.clear()
        for rec in records:
            self.add(rec)

    def search(self, query: str, top_k: int = 5) -> list[tuple[MemoryRecord, float]]:
        q_tokens = tokenize(query)
        if not q_tokens or not self._docs:
            return []

        n = len(self._docs)
        scores: list[tuple[int, float]] = []
        for i, doc_toks in enumerate(self._doc_tokens):
            dl = len(doc_toks)
            tf_map: dict[str, int] = {}
            for t in doc_toks:
                tf_map[t] = tf_map.get(t, 0) + 1
            score = 0.0
            for qt in q_tokens:
                if qt not in tf_map:
                    continue
                df = self._df.get(qt, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                tf = tf_map[qt]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                score += idf * (tf * (self.k1 + 1)) / denom
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        out: list[tuple[MemoryRecord, float]] = []
        for idx, sc in scores[:top_k]:
            if sc > 0:
                out.append((self._docs[idx], sc))
        return out
