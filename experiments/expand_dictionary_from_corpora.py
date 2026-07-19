#!/usr/bin/env python3
"""Expand min_dict.json from large offline corpora + WordNet (build-time only).

Fast path:
  1. Brown tagged frequencies + agent corpus
  2. WordNet same-synset shortenings (v/a/r only)
  3. Curated phrases + candidates.json
  4. Score × accept top entries, validate noun/char gates once then prune

Usage:
  python -u experiments/expand_dictionary_from_corpora.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from dictionary_tiers import load_entries  # noqa: E402
from min_mem import MinMemConverter  # noqa: E402
from min_mem.converter import _ensure_nltk_data  # noqa: E402
from min_mem.dictionary import NOUN_TAGS, MinDictionary  # noqa: E402

from nltk import pos_tag, word_tokenize
from nltk.corpus import brown, wordnet as wn
from wordfreq import zipf_frequency

DICT_PATH = ROOT / "min_dict.json"
BUNDLED_PATH = ROOT / "src" / "min_mem" / "data" / "min_dict.json"
CORPUS_PATH = ROOT / "experiments" / "corpus.json"
CANDIDATES_PATH = ROOT / "experiments" / "candidates.json"
REPORT_PATH = ROOT / "experiments" / "dictionary_expansion_report.json"

BLOCKLIST = {
    "implement", "implementation", "implementations", "establish", "contains",
    "containing", "directly", "liberal", "improving", "tending", "actual",
    "particular", "flushed", "readable", "estimable", "longsighted",
    "featherbrained", "befuddled", "decreased",
}

CURATED_PHRASES: dict[str, str] = {
    "a large number of": "many", "a majority of": "most", "a number of": "some",
    "a significant number of": "many", "a variety of": "many",
    "as a consequence of": "from", "as a result of": "from", "as long as": "if",
    "as soon as": "once", "as well as": "and", "at a later date": "later",
    "at an early date": "soon", "at all times": "always", "at present": "now",
    "at the present time": "now", "at this point in time": "now",
    "by means of": "by", "by virtue of": "by", "due to the fact that": "because",
    "during the course of": "during", "for the duration of": "during",
    "for the purpose of": "for", "for this reason": "so",
    "in a timely manner": "soon", "in accordance with": "per",
    "in addition to": "besides", "in close proximity to": "near",
    "in excess of": "over", "in light of": "given", "in order that": "so",
    "in order to": "to", "in regard to": "on", "in spite of": "despite",
    "in the absence of": "without", "in the event that": "if",
    "in the near future": "soon", "in the vicinity of": "near",
    "on a daily basis": "daily", "on account of": "because",
    "on the basis of": "from", "owing to the fact that": "because",
    "prior to": "before", "subsequent to": "after", "taking into account": "given",
    "the majority of": "most", "until such time as": "until",
    "with regard to": "on", "with respect to": "on",
    "with the exception of": "except",
}

MIN_NOUN_PRESERVATION = 94.0


def log(msg: str) -> None:
    print(msg, flush=True)


def zipf(word: str) -> float:
    return float(zipf_frequency(word.split()[0], "en"))


def brown_frequencies() -> Counter[str]:
    freq: Counter[str] = Counter()
    for sent in brown.tagged_sents():
        for token, tag in sent:
            w = token.lower()
            if tag in NOUN_TAGS or not w.isalpha() or len(w) < 6 or w in BLOCKLIST:
                continue
            freq[w] += 1
    if CORPUS_PATH.exists():
        for sample in json.loads(CORPUS_PATH.read_text())["samples"]:
            for token, tag in pos_tag(word_tokenize(sample["text"])):
                w = token.lower()
                if tag in NOUN_TAGS or not w.isalpha() or len(w) < 6 or w in BLOCKLIST:
                    continue
                freq[w] += 2  # boost agent-memory domain
    return freq


def lemma_clean(name: str) -> str | None:
    w = name.replace("_", " ").lower().strip()
    if not w or " " in w or not w.isalpha() or w in BLOCKLIST:
        return None
    return w


def harvest_wordnet(
    freq: Counter[str],
    *,
    min_delta: int,
    min_zipf_gain: float,
    min_target_zipf: float,
    min_corpus_freq: int,
) -> dict[str, tuple[str, float]]:
    best: dict[str, tuple[str, float]] = {}
    for pos in (wn.VERB, wn.ADJ, wn.ADV):
        for syn in wn.all_synsets(pos):
            lemmas = [lemma_clean(l.name()) for l in syn.lemmas()]
            lemmas = [x for x in lemmas if x is not None]
            if len(lemmas) < 2:
                continue
            zmap = {w: zipf(w) for w in lemmas}
            for src in lemmas:
                c = freq.get(src, 0)
                if c < min_corpus_freq:
                    continue
                z_src = zmap[src]
                for tgt in lemmas:
                    if tgt == src:
                        continue
                    delta = len(src) - len(tgt)
                    if delta < min_delta:
                        continue
                    z_tgt = zmap[tgt]
                    if z_tgt < min_target_zipf or z_tgt < z_src + min_zipf_gain:
                        continue
                    if len(tgt) <= 2 and tgt not in {"do", "so", "to", "on", "by", "if"}:
                        continue
                    score = c * delta * (1.0 + z_tgt - z_src)
                    prev = best.get(src)
                    if prev is None or score > prev[1] or (abs(score - prev[1]) < 1e-9 and len(tgt) < len(prev[0])):
                        best[src] = (tgt, score)
    return best


def evaluate(entries: dict[str, str], texts: list[str], *, nouns: bool = True) -> dict[str, float]:
    converter = MinMemConverter(MinDictionary.from_dict(entries))
    char_ratios: list[float] = []
    total_nouns = preserved = 0
    if nouns:
        _ensure_nltk_data()
    for text in texts:
        r = converter.minify(text)
        char_ratios.append(100.0 * r.chars_saved / len(text) if text else 0.0)
        if nouns:
            orig = [t.lower() for t, tag in pos_tag(word_tokenize(text)) if tag in NOUN_TAGS]
            new = {t.lower() for t, tag in pos_tag(word_tokenize(r.minified)) if tag in NOUN_TAGS}
            total_nouns += len(orig)
            preserved += sum(1 for n in orig if n in new)
    return {
        "char_savings_pct": sum(char_ratios) / len(char_ratios) if char_ratios else 0.0,
        "noun_preservation_pct": (100.0 * preserved / total_nouns) if total_nouns else 100.0,
        "dict_size": float(len(entries)),
    }


def prune_to_safe(
    base: dict[str, str],
    additions: list[tuple[str, str, float]],
    agent_texts: list[str],
    brown_texts: list[str],
) -> tuple[dict[str, str], list[dict]]:
    """Binary-ish prune: keep largest prefix of ranked additions that passes gates."""
    baseline_agent = evaluate(base, agent_texts, nouns=True)
    baseline_brown = evaluate(base, brown_texts, nouns=False)

    def ok(trial: dict[str, str]) -> tuple[bool, dict, dict]:
        m_agent = evaluate(trial, agent_texts, nouns=True)
        m_brown = evaluate(trial, brown_texts, nouns=False)
        good = (
            m_agent["noun_preservation_pct"] >= MIN_NOUN_PRESERVATION
            and m_agent["char_savings_pct"] + 1e-9 >= baseline_agent["char_savings_pct"]
            and m_brown["char_savings_pct"] + 1e-9 >= baseline_brown["char_savings_pct"]
        )
        return good, m_agent, m_brown

    lo, hi = 0, len(additions)
    best_n = 0
    best_metrics = (baseline_agent, baseline_brown)
    while lo <= hi:
        mid = (lo + hi) // 2
        trial = dict(base)
        for src, tgt, _ in additions[:mid]:
            trial[src] = tgt
        good, m_a, m_b = ok(trial)
        log(f"  probe n={mid}: agent={m_a['char_savings_pct']:.2f}% brown={m_b['char_savings_pct']:.2f}% nouns={m_a['noun_preservation_pct']:.1f}% ok={good}")
        if good:
            best_n = mid
            best_metrics = (m_a, m_b)
            lo = mid + 1
        else:
            hi = mid - 1

    # Fine-tune: walk forward from best_n while singles still help.
    current = dict(base)
    for src, tgt, _ in additions[:best_n]:
        current[src] = tgt
    accepted = [
        {"source": s, "target": t, "score": round(sc, 3), "char_delta": len(s) - len(t)}
        for s, t, sc in additions[:best_n]
    ]
    for src, tgt, score in additions[best_n:]:
        trial = deepcopy(current)
        trial[src] = tgt
        good, m_a, m_b = ok(trial)
        if not good:
            continue
        current = trial
        accepted.append(
            {"source": src, "target": tgt, "score": round(score, 3), "char_delta": len(src) - len(tgt)}
        )
        best_metrics = (m_a, m_b)
        log(f"  +{src}→{tgt} size={len(current)}")
    return current, accepted, best_metrics


def save_dictionary(entries: dict[str, str], sources: list[str]) -> None:
    payload = {
        "_meta": {
            "description": "Minimal synonym map: longer form -> shortest equivalent. Nouns are never replaced at runtime (POS-gated).",
            "version": "expanded-corpora-2026-07",
            "runtime_inference": "none",
            "build_time_compute": "offline_wordnet_corpora",
            "entry_count": len(entries),
            "phrase_count": sum(1 for k in entries if " " in k),
            "sources": sources,
            "last_improved": datetime.now(timezone.utc).isoformat(),
        },
        "entries": dict(sorted(entries.items())),
    }
    text = json.dumps(payload, indent=2) + "\n"
    DICT_PATH.write_text(text, encoding="utf-8")
    BUNDLED_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--min-delta", type=int, default=2)
    ap.add_argument("--min-zipf-gain", type=float, default=0.45)
    ap.add_argument("--min-target-zipf", type=float, default=4.0)
    ap.add_argument("--min-corpus-freq", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    _ensure_nltk_data()
    log("Brown+agent frequencies…")
    freq = brown_frequencies()
    log(f"  tracked tokens: {len(freq)}")

    log("WordNet harvest…")
    wn_pairs = harvest_wordnet(
        freq,
        min_delta=args.min_delta,
        min_zipf_gain=args.min_zipf_gain,
        min_target_zipf=args.min_target_zipf,
        min_corpus_freq=args.min_corpus_freq,
    )
    log(f"  candidates: {len(wn_pairs)}")

    existing = load_entries(DICT_PATH)
    seeds = {}
    if CANDIDATES_PATH.exists():
        seeds = {
            k.lower(): v
            for k, v in json.loads(CANDIDATES_PATH.read_text())["candidates"].items()
        }

    brown_blob = "\n".join(" ".join(s) for s in brown.sents()[:2000])
    pool: dict[str, tuple[str, float]] = {}
    for src, (tgt, score) in wn_pairs.items():
        pool[src] = (tgt, score)
    for src, tgt in {**CURATED_PHRASES, **seeds}.items():
        s, t = src.lower(), tgt.lower()
        if s in existing or s in BLOCKLIST or len(t) >= len(s):
            continue
        hits = len(re.findall(re.escape(s), brown_blob, flags=re.I))
        score = (hits + 10) * (len(s) - len(t)) * 5.0
        if s not in pool or score > pool[s][1]:
            pool[s] = (t, score)

    ranked = [(s, t, sc) for s, (t, sc) in pool.items() if s not in existing]
    ranked.sort(key=lambda x: (0 if " " in x[0] else 1, -x[2], -len(x[0])))
    additions = ranked[: args.max_new]
    log(f"Proposed additions: {len(additions)} (from {len(ranked)} ranked)")

    agent_texts = [s["text"] for s in json.loads(CORPUS_PATH.read_text())["samples"]]
    brown_texts = [" ".join(s) for s in brown.sents()[:300]]

    base_agent = evaluate(existing, agent_texts, nouns=True)
    base_brown = evaluate(existing, brown_texts, nouns=False)
    log(
        f"Baseline agent={base_agent['char_savings_pct']:.2f}% "
        f"brown={base_brown['char_savings_pct']:.2f}% "
        f"nouns={base_agent['noun_preservation_pct']:.1f}%"
    )

    log("Validating / pruning…")
    expanded, accepted, (m_agent, m_brown) = prune_to_safe(
        existing, additions, agent_texts, brown_texts
    )

    sources = [
        "wordnet",
        "nltk:brown",
        "experiments/corpus.json",
        "experiments/candidates.json",
        "curated_phrases",
    ]
    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "baseline_agent": base_agent,
        "baseline_brown": base_brown,
        "final_agent": m_agent,
        "final_brown": m_brown,
        "proposed": len(additions),
        "accepted_count": len(accepted),
        "accepted": accepted,
        "note": "Simple PPDB mirrors 404; used WordNet + Brown/agent corpora.",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log(f"Wrote {REPORT_PATH}")
    log(
        f"Result size={len(expanded)} (+{len(accepted)}) "
        f"agent={m_agent['char_savings_pct']:.2f}% "
        f"brown={m_brown['char_savings_pct']:.2f}% "
        f"nouns={m_agent['noun_preservation_pct']:.1f}%"
    )

    if args.dry_run:
        log("Dry-run: dictionary not updated.")
        return
    save_dictionary(expanded, sources)
    log(f"Updated {DICT_PATH} and {BUNDLED_PATH}")


if __name__ == "__main__":
    main()
