"""Shared evaluation metrics for min-mem experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from min_mem.converter import _ensure_nltk_data
from min_mem.dictionary import NOUN_TAGS, MinDictionary

try:
    import tiktoken
    from nltk import pos_tag, word_tokenize
except ImportError:
    tiktoken = None  # type: ignore[assignment]
    pos_tag = None  # type: ignore[assignment]
    word_tokenize = None  # type: ignore[assignment]


@dataclass
class MethodResult:
    method: str
    sample_id: str
    original_chars: int
    minified_chars: int
    char_savings_pct: float
    original_tokens: int
    minified_tokens: int
    token_savings_pct: float
    replacements: int
    nouns_preserved_pct: float
    content_jaccard: float
    synonym_aware_pct: float
    params: dict | None = None
    inference_cost: str = "none"


def count_tokens(text: str) -> int:
    if tiktoken is None:
        return len(text.split())
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def extract_nouns(text: str) -> list[str]:
    _ensure_nltk_data()
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    return [t.lower() for t, tag in tagged if tag in NOUN_TAGS]


def content_words(text: str) -> set[str]:
    _ensure_nltk_data()
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "and", "or", "but", "if", "that", "this", "it", "they", "we", "you",
    }
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    return {
        t.lower()
        for t, tag in tagged
        if tag not in {".", ",", ":", ";", "!", "?", "''", "``"}
        and t.lower() not in stop
        and len(t) > 2
    }


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def synonym_aware_overlap(original: str, transformed: str, dictionary: MinDictionary) -> float:
    orig_words = content_words(original)
    trans_words = content_words(transformed)
    if not orig_words:
        return 1.0

    reverse: dict[str, set[str]] = {}
    for entry in dictionary.word_entries():
        reverse.setdefault(entry.target.lower(), set()).add(entry.source.lower())
        reverse.setdefault(entry.source.lower(), set()).add(entry.target.lower())

    preserved = 0
    for w in orig_words:
        if w in trans_words:
            preserved += 1
            continue
        if reverse.get(w, set()) & trans_words:
            preserved += 1
    return preserved / len(orig_words)


def noun_preservation(original: str, transformed: str) -> float:
    orig = extract_nouns(original)
    new = extract_nouns(transformed)
    if not orig:
        return 100.0
    preserved = sum(1 for n in orig if n in new)
    return 100.0 * preserved / len(orig)


def evaluate(
    method: str,
    sample_id: str,
    original: str,
    transformed: str,
    replacements: int,
    dictionary: MinDictionary,
    *,
    params: dict | None = None,
    inference_cost: str = "none",
) -> MethodResult:
    oc, mc = len(original), len(transformed)
    ot, mt = count_tokens(original), count_tokens(transformed)
    return MethodResult(
        method=method,
        sample_id=sample_id,
        original_chars=oc,
        minified_chars=mc,
        char_savings_pct=100.0 * (oc - mc) / oc if oc else 0.0,
        original_tokens=ot,
        minified_tokens=mt,
        token_savings_pct=100.0 * (ot - mt) / ot if ot else 0.0,
        replacements=replacements,
        nouns_preserved_pct=noun_preservation(original, transformed),
        content_jaccard=jaccard(content_words(original), content_words(transformed)),
        synonym_aware_pct=100.0 * synonym_aware_overlap(original, transformed, dictionary),
        params=params,
        inference_cost=inference_cost,
    )


def aggregate(results: list[MethodResult]) -> dict:
    by_method: dict[str, list[MethodResult]] = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r)

    summary = {}
    for method, rows in by_method.items():
        n = len(rows)
        summary[method] = {
            "n": n,
            "char_savings_pct_mean": sum(r.char_savings_pct for r in rows) / n,
            "token_savings_pct_mean": sum(r.token_savings_pct for r in rows) / n,
            "replacements_mean": sum(r.replacements for r in rows) / n,
            "nouns_preserved_pct_mean": sum(r.nouns_preserved_pct for r in rows) / n,
            "content_jaccard_mean": sum(r.content_jaccard for r in rows) / n,
            "synonym_aware_pct_mean": sum(r.synonym_aware_pct for r in rows) / n,
            "inference_cost": rows[0].inference_cost,
            "params": rows[0].params,
        }
    return summary


def results_to_dict(results: list[MethodResult]) -> list[dict]:
    return [asdict(r) for r in results]
