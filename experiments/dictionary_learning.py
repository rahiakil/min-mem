#!/usr/bin/env python3
"""Advanced streaming dictionary-learning experiment.

Demonstrates the right way to keep adding keywords to min_dict.json:
a streaming, multi-feature scorer with token-aware acceptance.

Build-time only — never on the runtime write path.

Pipeline
--------
1. Stream memory batches (Brown chunks + agent corpus replayed).
2. Mine candidate long->short pairs from WordNet (v/a/r same-synset)
   plus curated high-ROI phrases.
3. Score each candidate with a multi-feature linear scorer:
     score = w_freq * freq * char_delta
           + w_tok  * token_delta (cl100k_base)
           + w_zipf * (zipf_tgt - zipf_src)
           - w_risk * noun_risk
4. Accept only if ALL gates hold:
     - target strictly fewer tokens (cl100k_base)
     - target shorter in characters
     - zipf(target) >= zipf(source) + gain
     - noun preservation on eval corpus stays >= 94%
     - char savings on eval corpus does not regress
5. Emit a learning curve: dict size vs char savings vs noun retention
   per batch, plus a plateau point and per-batch marginal gain.

Outputs
--------
experiments/dictionary_learning_results.json
experiments/figures/fig_dict_learning.pdf
"""

from __future__ import annotations

import json
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
CORPUS_PATH = ROOT / "experiments" / "corpus.json"
RESULTS_PATH = ROOT / "experiments" / "dictionary_learning_results.json"
FIG_DIR = ROOT / "experiments" / "figures"

BLOCKLIST = {
    "implement", "implementation", "implementations", "establish",
    "contains", "containing", "directly", "liberal", "improving",
    "tending", "actual", "particular", "flushed", "readable", "estimable",
    "longsighted", "featherbrained", "befuddled", "decreased",
}

CURATED_PHRASES = {
    "in order to": "to", "in spite of": "despite", "in the event that": "if",
    "prior to": "before", "as well as": "and", "due to the fact that": "because",
    "for the purpose of": "for", "in the near future": "soon",
    "at this point in time": "now", "a large number of": "many",
    "on account of": "because", "subsequent to": "after",
    "with respect to": "on", "in accordance with": "per",
}

MIN_NOUN_PRESERVATION = 94.0
SAFE_NOUNISH = {"help", "use", "need", "end", "start", "try", "show", "give", "make"}

# Scorer weights (advanced multi-feature)
W_FREQ = 1.0
W_TOK = 4.0
W_ZIPF = 2.0
W_RISK = 8.0


def log(msg: str) -> None:
    print(msg, flush=True)


def zipf(word: str) -> float:
    return float(zipf_frequency(word.split()[0], "en"))


def count_tokens(text: str) -> int:
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return len(text.split())


def lemma_clean(name: str) -> str | None:
    w = name.replace("_", " ").lower().strip()
    if not w or " " in w or not w.isalpha() or w in BLOCKLIST:
        return None
    return w


def isolated_pos(word: str, cache: dict[str, str]) -> str:
    if word in cache:
        return cache[word]
    _ensure_nltk_data()
    tag = pos_tag(word_tokenize(word))[-1][1] if word else "UNK"
    cache[word] = tag
    return tag


def harvest_candidates(freq: Counter[str], pos_cache: dict[str, str]) -> dict[str, tuple[str, float]]:
    """WordNet same-synset shortenings + curated phrases."""
    best: dict[str, tuple[str, float]] = {}
    for pos in (wn.VERB, wn.ADJ, wn.ADV):
        for syn in wn.all_synsets(pos):
            lemmas = [lemma_clean(l.name()) for l in syn.lemmas()]
            lemmas = [x for x in lemmas if x is not None]
            if len(lemmas) < 2:
                continue
            zmap = {w: zipf(w) for w in lemmas}
            for src in lemmas:
                if isolated_pos(src, pos_cache) in NOUN_TAGS:
                    continue
                for tgt in lemmas:
                    if tgt == src or len(tgt) >= len(src):
                        continue
                    if len(tgt) <= 2 and tgt not in {"do", "so", "to", "on", "by", "if"}:
                        continue
                    best[src] = (tgt, zmap[src], zmap[tgt])
    # Curated phrases (high ROI)
    for src, tgt in CURATED_PHRASES.items():
        best[src] = (tgt, zipf(src), zipf(tgt))
    return best


def advanced_score(src: str, tgt: str, z_src: float, z_tgt: float,
                   freq: Counter[str], tok_src: int, tok_dst: int) -> float:
    char_delta = len(src) - len(tgt)
    tok_delta = tok_src - tok_dst
    zipf_gain = z_tgt - z_src
    noun_risk = 1.0 if (z_tgt < 3.5 and tgt not in SAFE_NOUNISH) else 0.0
    return (
        W_FREQ * freq.get(src, 0) * max(char_delta, 0)
        + W_TOK * max(tok_delta, 0)
        + W_ZIPF * max(zipf_gain, 0)
        - W_RISK * noun_risk
    )


def evaluate(entries: dict[str, str], texts: list[str], *, nouns: bool = True) -> dict[str, float]:
    converter = MinMemConverter(MinDictionary.from_dict(entries))
    char_ratios = []
    total_nouns = preserved = 0
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


def stream_batches(agent_texts, brown_texts, n_batches: int = 6) -> list[list[str]]:
    """Simulate streaming memory batches."""
    batches = []
    chunk = max(1, len(brown_texts) // n_batches)
    for i in range(n_batches):
        b = brown_texts[i * chunk:(i + 1) * chunk]
        # Each batch also re-sees agent memories (they recur in production).
        batches.append(agent_texts + b)
    return batches


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--max-new-per-batch", type=int, default=40)
    ap.add_argument("--seed-size", type=int, default=0,
                    help="If >0, start from a random subset of the base dict of this size "
                         "to expose the learning curve shape.")
    args = ap.parse_args()

    _ensure_nltk_data()
    log("Loading base dictionary and corpora...")
    base_full = load_entries(DICT_PATH)
    if args.seed_size and args.seed_size < len(base_full):
        import random
        random.seed(13)
        keys = random.sample(list(base_full.keys()), args.seed_size)
        base = {k: base_full[k] for k in keys}
        log(f"Seeded with {len(base)} of {len(base_full)} base entries to expose curve.")
    else:
        base = dict(base_full)
    agent_texts = [s["text"] for s in json.loads(CORPUS_PATH.read_text())["samples"]]
    brown_texts = [" ".join(s) for s in brown.sents()[:400]]
    eval_texts = agent_texts + brown_texts[:100]

    # Global frequency from all seen text (streaming accumulation)
    global_freq: Counter[str] = Counter()
    pos_cache: dict[str, str] = {}

    log("Harvesting WordNet + curated candidates...")
    candidates = harvest_candidates(global_freq, pos_cache)
    log(f"  raw candidates: {len(candidates)}")

    # Precompute token counts for candidates
    tok_cache: dict[str, int] = {}

    def tok(s: str) -> int:
        if s not in tok_cache:
            tok_cache[s] = count_tokens(s)
        return tok_cache[s]

    current = dict(base)
    baseline = evaluate(current, eval_texts, nouns=True)
    log(f"Baseline: size={len(current)} char={baseline['char_savings_pct']:.2f}% "
        f"nouns={baseline['noun_preservation_pct']:.1f}%")

    batches = stream_batches(agent_texts, brown_texts, args.batches)
    curve = [{
        "batch": 0,
        "dict_size": len(current),
        "char_savings_pct": round(baseline["char_savings_pct"], 3),
        "noun_preservation_pct": round(baseline["noun_preservation_pct"], 3),
        "added": 0,
        "marginal_char_gain": 0.0,
    }]

    prev_char = baseline["char_savings_pct"]
    plateau_count = 0

    for bi, batch_texts in enumerate(batches, 1):
        # Update streaming frequency with this batch
        for text in batch_texts:
            try:
                tagged = pos_tag(word_tokenize(text[:8000]))
            except Exception:
                continue
            for token, tag in tagged:
                w = token.lower()
                if tag in NOUN_TAGS or not w.isalpha() or len(w) < 6 or w in BLOCKLIST:
                    continue
                global_freq[w] += 1

        # Score and rank candidates
        scored = []
        for src, (tgt, z_src, z_tgt) in candidates.items():
            if src in current or src in BLOCKLIST or tgt in BLOCKLIST:
                continue
            s = advanced_score(src, tgt, z_src, z_tgt, global_freq, tok(src), tok(tgt))
            if s > 0:
                scored.append((s, src, tgt, z_src, z_tgt))
        scored.sort(reverse=True)

        # Token-aware acceptance: try top-K, accept if all gates hold
        added_this_batch = 0
        trial = deepcopy(current)
        for s, src, tgt, z_src, z_tgt in scored[:args.max_new_per_batch * 4]:
            if added_this_batch >= args.max_new_per_batch:
                break
            # Gates
            if tok(tgt) >= tok(src):
                continue
            if len(tgt) >= len(src):
                continue
            if z_tgt < z_src + 0.3:
                continue
            trial[src] = tgt
            added_this_batch += 1

        if added_this_batch == 0:
            log(f"Batch {bi}: no candidates passed gates.")
            curve.append({
                "batch": bi, "dict_size": len(current),
                "char_savings_pct": round(prev_char, 3),
                "noun_preservation_pct": round(baseline["noun_preservation_pct"], 3),
                "added": 0, "marginal_char_gain": 0.0,
            })
            continue

        # Validate the whole batch at once; if it fails, drop back to singleton trials
        m = evaluate(trial, eval_texts, nouns=True)
        if (m["char_savings_pct"] + 1e-9 >= prev_char
                and m["noun_preservation_pct"] >= MIN_NOUN_PRESERVATION):
            current = trial
        else:
            for s, src, tgt, z_src, z_tgt in scored[:args.max_new_per_batch * 4]:
                if len(current) >= len(base) + bi * args.max_new_per_batch:
                    break
                if src in current:
                    continue
                if tok(tgt) >= tok(src) or len(tgt) >= len(src) or z_tgt < z_src + 0.3:
                    continue
                t2 = deepcopy(current)
                t2[src] = tgt
                m2 = evaluate(t2, eval_texts, nouns=True)
                if (m2["char_savings_pct"] + 1e-9 >= prev_char
                        and m2["noun_preservation_pct"] >= MIN_NOUN_PRESERVATION):
                    current = t2

        m = evaluate(current, eval_texts, nouns=True)
        gain = m["char_savings_pct"] - prev_char
        curve.append({
            "batch": bi,
            "dict_size": len(current),
            "char_savings_pct": round(m["char_savings_pct"], 3),
            "noun_preservation_pct": round(m["noun_preservation_pct"], 3),
            "added": len(current) - len(base),
            "marginal_char_gain": round(gain, 4),
        })
        log(f"Batch {bi}: size={len(current)} char={m['char_savings_pct']:.2f}% "
            f"gain={gain:+.4f} nouns={m['noun_preservation_pct']:.1f}%")
        if gain < 0.05:
            plateau_count += 1
        else:
            plateau_count = 0
        prev_char = m["char_savings_pct"]
        if plateau_count >= 2:
            log("Plateau reached; stopping early.")
            break

    final = evaluate(current, eval_texts, nouns=True)
    agent_final = evaluate(current, agent_texts, nouns=True)

    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "algorithm": "streaming_multi_feature_token_aware",
        "scorer_weights": {"freq": W_FREQ, "token": W_TOK, "zipf": W_ZIPF, "risk": W_RISK},
        "acceptance_gates": [
            "target strictly fewer cl100k_base tokens",
            "target shorter in characters",
            "zipf(target) >= zipf(source) + 0.3",
            "noun preservation >= 94% on eval corpus",
            "char savings do not regress",
        ],
        "base_dict_size": len(base),
        "final_dict_size": len(current),
        "batches_run": len(curve) - 1,
        "baseline": baseline,
        "final_eval_mix": final,
        "final_agent": agent_final,
        "learning_curve": curve,
        "plateau_batch": next((c["batch"] for c in curve[1:] if c["marginal_char_gain"] < 0.05), None),
    }
    RESULTS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log(f"Wrote {RESULTS_PATH}")

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        FIG_DIR.mkdir(parents=True, exist_ok=True)
        xs = [c["batch"] for c in curve]
        char = [c["char_savings_pct"] for c in curve]
        size = [c["dict_size"] for c in curve]
        fig, ax1 = plt.subplots(figsize=(4.2, 2.8))
        ax1.plot(xs, char, "o-", color="#1f77b4", label="Char savings %")
        ax1.set_xlabel("Streaming batch")
        ax1.set_ylabel("Char savings %", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax2 = ax1.twinx()
        ax2.plot(xs, size, "s--", color="#d62728", label="Dictionary size")
        ax2.set_ylabel("Dictionary size", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax1.set_title("Dictionary learning curve (streaming, token-aware)")
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_dict_learning.pdf")
        fig.savefig(FIG_DIR / "fig_dict_learning.png", dpi=150)
        log(f"Wrote figure {FIG_DIR / 'fig_dict_learning.pdf'}")
    except Exception as e:
        log(f"Figure skipped: {e}")

    log(f"Done: {len(base)} -> {len(current)} entries, "
        f"char {baseline['char_savings_pct']:.2f}% -> {final['char_savings_pct']:.2f}%")


if __name__ == "__main__":
    main()
