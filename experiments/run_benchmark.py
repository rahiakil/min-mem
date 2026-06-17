#!/usr/bin/env python3
"""Comparative benchmark for min-mem and baselines. Writes results JSON + LaTeX table rows."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from min_mem.converter import MinMemConverter, _detokenize, _preserve_case  # noqa: E402
from min_mem.dictionary import NOUN_TAGS, MinDictionary  # noqa: E402
from dictionary_tiers import build_tiers, load_entries  # noqa: E402

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    from nltk import pos_tag, word_tokenize
    from min_mem.converter import _ensure_nltk_data
except ImportError:
    pos_tag = None  # type: ignore[assignment]

CORPUS_PATH = Path(__file__).parent / "corpus.json"
OUTPUT_PATH = Path(__file__).parent / "results.json"


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
    """Fraction of original content words preserved or replaced by known synonym."""
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
        aliases = reverse.get(w, set())
        if aliases & trans_words:
            preserved += 1
    return preserved / len(orig_words)


def noun_preservation(original: str, transformed: str) -> float:
    orig = extract_nouns(original)
    new = extract_nouns(transformed)
    if not orig:
        return 100.0
    preserved = sum(1 for n in orig if n in new)
    return 100.0 * preserved / len(orig)


class NaiveConverter:
    """Dictionary replacement without POS gating — risks noun corruption."""

    def __init__(self, dictionary: MinDictionary) -> None:
        self.dictionary = dictionary

    def minify(self, text: str) -> str:
        import re as _re

        working = text
        for entry in self.dictionary.phrase_entries():
            pattern = _re.compile(_re.escape(entry.source), _re.IGNORECASE)

            def sub(m: _re.Match[str], target: str = entry.target) -> str:
                return _preserve_case(m.group(0), target)

            working = pattern.sub(sub, working)

        tokens = word_tokenize(working)
        out: list[str] = []
        for token in tokens:
            repl = self.dictionary.lookup_inflected(token)
            if repl and repl.lower() != token.lower():
                out.append(_preserve_case(token, repl))
            else:
                out.append(token)
        return _detokenize(out)


class PhraseOnlyConverter:
    def __init__(self, dictionary: MinDictionary) -> None:
        self.dictionary = dictionary

    def minify(self, text: str) -> str:
        working = text
        for entry in self.dictionary.phrase_entries():
            pattern = re.compile(re.escape(entry.source), re.IGNORECASE)
            working = pattern.sub(
                lambda m, t=entry.target: _preserve_case(m.group(0), t),
                working,
            )
        return working


class NoInflectionConverter:
    def __init__(self, dictionary: MinDictionary) -> None:
        self.dictionary = dictionary
        self._full = MinMemConverter(dictionary)

    def minify(self, text: str) -> str:
        _ensure_nltk_data()
        working = text
        for entry in self.dictionary.phrase_entries():
            pattern = re.compile(re.escape(entry.source), re.IGNORECASE)
            working = pattern.sub(
                lambda m, t=entry.target: _preserve_case(m.group(0), t),
                working,
            )
        tokens = word_tokenize(working)
        tagged = pos_tag(tokens)
        out: list[str] = []
        for token, tag in tagged:
            if tag in NOUN_TAGS:
                out.append(token)
                continue
            repl = self.dictionary.lookup_word(token)
            if repl and repl.lower() != token.lower():
                out.append(_preserve_case(token, repl))
            else:
                out.append(token)
        return _detokenize(out)


def gzip_ratio(text: str) -> float:
    import gzip

    raw = len(text.encode("utf-8"))
    compressed = len(gzip.compress(text.encode("utf-8"), compresslevel=9))
    return 100.0 * (1 - compressed / raw) if raw else 0.0


def evaluate(
    method: str,
    sample_id: str,
    original: str,
    transformed: str,
    replacements: int,
    dictionary: MinDictionary,
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
        }
    return summary


def run_dictionary_ablation(corpus: list[dict], full_entries: dict[str, str]) -> dict:
    """Benchmark progressive min dictionary tiers."""
    tiers = build_tiers(full_entries)
    ablation = []

    for tier in tiers:
        dictionary = MinDictionary.from_dict(tier["entries"])
        converter = MinMemConverter(dictionary)
        char_savings, token_savings, noun_pres, swaps = [], [], [], []

        for sample in corpus:
            text = sample["text"]
            if not tier["entries"]:
                transformed = text
                n_swaps = 0
            else:
                r = converter.minify(text)
                transformed = r.minified
                n_swaps = len(r.replacements)
            ev = evaluate(
                tier["name"], sample["id"], text, transformed, n_swaps, dictionary
            )
            char_savings.append(ev.char_savings_pct)
            token_savings.append(ev.token_savings_pct)
            noun_pres.append(ev.nouns_preserved_pct)
            swaps.append(max(n_swaps, 0))

        n = len(corpus)
        ablation.append({
            "tier": tier["name"],
            "label": tier["label"],
            "dict_size": tier.get("size", 0),
            "char_savings_pct_mean": sum(char_savings) / n,
            "token_savings_pct_mean": sum(token_savings) / n,
            "nouns_preserved_pct_mean": sum(noun_pres) / n,
            "replacements_mean": sum(swaps) / n,
        })

    return {"tiers": ablation}


def top_dictionary_hits(corpus: list[dict], dictionary: MinDictionary) -> list[dict]:
    """Most frequent replacements across corpus."""
    from collections import Counter

    hits: Counter[str] = Counter()
    converter = MinMemConverter(dictionary)
    for sample in corpus:
        r = converter.minify(sample["text"])
        for rep in r.replacements:
            key = f"{rep.original} → {rep.replacement}"
            hits[key] += 1
    return [{"swap": k, "count": v} for k, v in hits.most_common(15)]


def main() -> None:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    dictionary = MinDictionary.from_path(ROOT / "min_dict.json")
    full = MinMemConverter(dictionary)
    naive = NaiveConverter(dictionary)
    phrase = PhraseOnlyConverter(dictionary)
    no_infl = NoInflectionConverter(dictionary)

    results: list[MethodResult] = []
    gzip_ratios: list[dict] = []

    for sample in corpus["samples"]:
        sid = sample["id"]
        text = sample["text"]

        r = full.minify(text)
        results.append(
            evaluate("min-mem (full)", sid, text, r.minified, len(r.replacements), dictionary)
        )

        naive_out = naive.minify(text)
        results.append(
            evaluate("naive-dict", sid, text, naive_out, -1, dictionary)
        )

        phrase_out = phrase.minify(text)
        results.append(
            evaluate("phrase-only", sid, text, phrase_out, -1, dictionary)
        )

        no_infl_out = no_infl.minify(text)
        results.append(
            evaluate("no-inflection", sid, text, no_infl_out, -1, dictionary)
        )

        results.append(
            evaluate("identity", sid, text, text, 0, dictionary)
        )

        gzip_ratios.append({
            "sample_id": sid,
            "gzip_savings_pct": gzip_ratio(text),
        })

    summary = aggregate(results)
    by_category: dict[str, list[float]] = {}
    for sample in corpus["samples"]:
        r = full.minify(sample["text"])
        cat = sample["category"]
        by_category.setdefault(cat, []).append(r.savings_ratio * 100)

    full_entries = load_entries(ROOT / "min_dict.json")
    dict_ablation = run_dictionary_ablation(corpus["samples"], full_entries)
    top_hits = top_dictionary_hits(corpus["samples"], dictionary)

    # Per-sample min-mem only (for scatter chart)
    per_sample_minmem = [
        asdict(r) for r in results if r.method == "min-mem (full)"
    ]

    payload = {
        "corpus_size": len(corpus["samples"]),
        "dictionary_size": len(dictionary),
        "summary": summary,
        "gzip_baseline": {
            "mean_savings_pct": sum(g["gzip_savings_pct"] for g in gzip_ratios) / len(gzip_ratios),
        },
        "by_category_char_savings": {
            k: sum(v) / len(v) for k, v in by_category.items()
        },
        "dictionary_ablation": dict_ablation,
        "top_swaps": top_hits,
        "per_sample_minmem": per_sample_minmem,
        "per_sample": [asdict(r) for r in results],
        "gzip_per_sample": gzip_ratios,
    }

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
