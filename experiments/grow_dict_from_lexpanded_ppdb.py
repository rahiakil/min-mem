"""Grow min_dict.json from Lexpanded-PPDB (English S) — offline, build-time only.

Lexpanded-PPDB ships "just pairs" as 2-column TSV (source \\t target) with no
relation label. This script streams the gz, orients each pair longer->shorter,
and applies the paper's filter order (minus the PPDB relation gate, which is
absent here; we treat all pairs as candidate equivalences and let POS + length
+ noun-head gates carry quality):

  1. Orient longer -> shorter (skip if equal length or target not shorter)
  2. Target strictly fewer tokens (cl100k_base) and strictly fewer chars
  3. Both sides single coarse POS, and not noun -> verb (noun-head protection)
  4. Source not a bare noun (avoid noun source entries per the paper's guarantee)
  5. Inflection blocklist; drop pairs that only differ by inflection
  6. Score by char_delta * frequency_gain (wordfreq); keep top-N
  7. Merge into min_dict.json without clobbering existing entries
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DICT_PATH = ROOT / "min_dict.json"
OUT_REPORT = ROOT / "experiments" / "ppdb_growth_report.json"

BLOCKLIST = {
    "implement", "implementation", "implementations", "establish", "contains",
    "deployment", "deployments",  # noun-corrupting; protected by converter too
}

try:
    import nltk
    from nltk import pos_tag, word_tokenize
    from min_mem.converter import _ensure_nltk_data
    _ensure_nltk_data()
    HAVE_NLTK = True
except Exception:
    HAVE_NLTK = False

try:
    from wordfreq import zipf_frequency
    HAVE_FREQ = True
except Exception:
    HAVE_FREQ = False

try:
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")
    HAVE_TOK = True
except Exception:
    HAVE_TOK = False

NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
# Coarse POS buckets for consistency check.
def coarse(tag: str) -> str:
    if tag in NOUN_TAGS:
        return "N"
    if tag.startswith("VB"):
        return "V"
    if tag.startswith("JJ") or tag == "CD":
        return "A"
    if tag.startswith("RB"):
        return "R"
    return "X"


WORD_RE = re.compile(r"^[A-Za-z][A-Za-z\-'\.]*$")


def tok_count(text: str) -> int:
    if HAVE_TOK:
        return len(ENC.encode(text))
    return len(text.split())


def tok_count_cheap(text: str) -> int:
    """Fast split-based token count for the streaming pass."""
    return len(text.split())


def pos_of(phrase: str) -> str:
    toks = word_tokenize(phrase)
    if not toks:
        return ""
    return pos_tag(toks)[0][1]


def is_bare_noun(phrase: str) -> bool:
    """True if the phrase is a single token that tags as a noun."""
    if " " in phrase:
        return False
    if not HAVE_NLTK:
        return False
    return pos_of(phrase) in NOUN_TAGS


def noun_head(phrase: str) -> str | None:
    """Return the last noun token of a phrase, if any."""
    if not HAVE_NLTK:
        return None
    toks = pos_tag(word_tokenize(phrase))
    nouns = [w for w, t in toks if t in NOUN_TAGS]
    return nouns[-1].lower() if nouns else None


def orient(source: str, target: str) -> tuple[str, str] | None:
    """Orient longer -> shorter. Return None if not a strict simplification."""
    s, t = source.strip(), target.strip()
    if not s or not t or s.lower() == t.lower():
        return None
    if len(t) < len(s):
        return s, t
    if len(s) < len(t):
        return t, s
    return None  # equal length -> not a simplification


def passes_cheap_filters(src: str, tgt: str) -> bool:
    if len(tgt) >= len(src):
        return False
    if tok_count_cheap(tgt) >= tok_count_cheap(src):
        return False
    for w in (src, tgt):
        if any(ch.isdigit() for ch in w):
            return False
    if src.lower() in BLOCKLIST or tgt.lower() in BLOCKLIST:
        return False
    # Reject pure inflection (same lemma, differ only by short suffix)
    if src.lower().startswith(tgt.lower()) or tgt.lower().startswith(src.lower()):
        stem = min(len(src), len(tgt))
        if abs(len(src) - len(tgt)) <= 3 and src[:stem].lower() == tgt[:stem].lower():
            return False
    return True


def passes_pos_filters(src: str, tgt: str) -> bool:
    """Expensive POS-based gates; run only on the final top-N candidates."""
    if not HAVE_NLTK:
        return True
    # Single-word source must not be a bare noun (noun-preservation guarantee)
    if " " not in src and is_bare_noun(src):
        return False
    ps, pt = pos_of(src), pos_of(tgt)
    cs, ct = coarse(ps), coarse(pt)
    if cs == "N" and ct == "V":
        return False
    if " " not in src and " " not in tgt and cs != ct:
        return False
    sh, th = noun_head(src), noun_head(tgt)
    if sh and th and sh != th:
        return False
    return True


def passes_filters(src: str, tgt: str) -> bool:
    return passes_cheap_filters(src, tgt) and passes_pos_filters(src, tgt)


def score(src: str, tgt: str) -> float:
    char_delta = len(src) - len(tgt)
    if HAVE_FREQ:
        z_gain = zipf_frequency(tgt.lower(), "en") - zipf_frequency(src.lower(), "en")
    else:
        z_gain = 0.0
    return char_delta * (1.0 + max(z_gain, 0.0))


def stream_pairs(path: Path, keys_file: Path, single_word_only: bool = True):
    """Stream oriented (longer->shorter) candidate pairs, prefiltering with mawk
    for C-speed row rejection. mawk orients, enforces char-strict + letter-start +
    no-digit + not-already-a-dictionary-key (and optionally single-word-only), so
    Python only sees high-quality survivors. The token-strict gate is NOT applied
    here: single-word swaps save characters, not tokens, and char-strict is the
    right gate for them."""
    import subprocess
    if str(path).endswith(".gz"):
        decomp = ["gzip", "-dc", str(path)]
    else:
        decomp = ["cat", str(path)]
    sw = "  if (src ~ / / || tgt ~ / /) next;\n" if single_word_only else ""
    awk_script = r"""
FNR==NR { keys[tolower($0)]=1; next }
NF<2 { next }
{
  s=$1; t=$2;
  if (s=="" || t=="") next;
  if (length(s) > length(t)) { src=s; tgt=t }
  else if (length(t) > length(s)) { src=t; tgt=s }
  else next;
  c1=substr(src,1,1); c2=substr(tgt,1,1);
  if (c1 !~ /[A-Za-z]/ || c2 !~ /[A-Za-z]/) next;
  if (src ~ /[0-9]/ || tgt ~ /[0-9]/) next;
  __SW__
  lk=tolower(src);
  if (keys[lk]) next;
  print src "\t" tgt;
}
""".replace("__SW__", sw)
    gz = subprocess.Popen(decomp, stdout=subprocess.PIPE, bufsize=128 * 1024)
    aw = subprocess.Popen(
        ["mawk", "-F\t", awk_script, str(keys_file), "-"],
        stdin=gz.stdout, stdout=subprocess.PIPE, bufsize=128 * 1024, text=True,
    )
    gz.stdout.close()
    assert aw.stdout is not None
    for line in aw.stdout:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            yield parts[0], parts[1]
    aw.wait()
    gz.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppdb-path", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="Max rows to scan (0 = all)")
    ap.add_argument("--top", type=int, default=4000, help="Top candidates to merge")
    ap.add_argument("--max-source-len", type=int, default=40)
    ap.add_argument("--allow-multiword", action="store_true",
                    help="Allow multi-word source phrases (else single-word only).")
    ap.add_argument("--max-source-words", type=int, default=3,
                    help="Max words in source phrase when --allow-multiword.")
    ap.add_argument("--max-target-words", type=int, default=2,
                    help="Max words in target phrase when --allow-multiword.")
    ap.add_argument("--min-target-zipf", type=float, default=3.5,
                    help="Target must be at least this common (Zipf).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    single_word_only = not args.allow_multiword

    existing = json.loads(DICT_PATH.read_text(encoding="utf-8"))
    entries = existing.get("entries", existing) if isinstance(existing, dict) else existing
    existing_keys = {k.lower() for k in entries}

    # Write existing keys for the awk prefilter.
    import tempfile
    keys_file = Path(tempfile.mkstemp(suffix=".keys")[1])
    keys_file.write_text("\n".join(sorted(existing_keys)) + "\n", encoding="utf-8")

    # Dedup by source key, keeping the shortest target seen (cheap, no per-row scoring).
    best: dict[str, str] = {}
    scanned = 0
    for source, target in stream_pairs(args.ppdb_path, keys_file, single_word_only):
        scanned += 1
        if args.limit and scanned > args.limit:
            break
        if scanned % 5_000_000 == 0:
            print(f"  scanned {scanned:,} survivors, {len(best):,} unique", flush=True)
        src, tgt = source, target
        if len(src) > args.max_source_len or len(tgt) < 2 or len(src) < 3:
            continue
        src_words = src.split()
        tgt_words = tgt.split()
        if single_word_only:
            if len(src_words) > 1 or len(tgt_words) > 1:
                continue
        else:
            if len(src_words) > args.max_source_words or len(tgt_words) > args.max_target_words:
                continue
            if len(tgt_words) >= len(src_words):  # target must be fewer words
                continue
        if not WORD_RE.match(src_words[0]) or not WORD_RE.match(tgt_words[0]):
            continue
        if src.lower() in BLOCKLIST or tgt.lower() in BLOCKLIST:
            continue
        # Reject pure inflection (same lemma, differ only by short suffix)
        sl, tl = src.lower(), tgt.lower()
        if sl.startswith(tl) or tl.startswith(sl):
            stem = min(len(src), len(tgt))
            if abs(len(src) - len(tgt)) <= 3 and src[:stem].lower() == tgt[:stem].lower():
                continue
        key = sl
        if key not in best or len(tgt) < len(best[key]):
            best[key] = tgt
    keys_file.unlink(missing_ok=True)

    # Pre-rank by char_delta (cheap), then apply the freq gate + score only to the top pool.
    # This avoids wordfreq calls on the full unique set (which can be tens of millions).
    pre = [(k, t, len(k) - len(t)) for k, t in best.items()]
    pre.sort(key=lambda x: x[2], reverse=True)
    candidates = pre[: 200_000]
    kept = []
    for k, t, _ in candidates:
        if HAVE_FREQ:
            twords = t.split()
            if any(zipf_frequency(w, "en") < 3.0 for w in twords):
                continue
            zt = zipf_frequency(twords[0], "en")
            if zt < args.min_target_zipf:
                continue
            zs = zipf_frequency(k.split()[0], "en")
            if zt < zs - 0.3:
                continue
        kept.append((k, t))
    pool = kept[: 50_000]
    scored = [(k, t, score(k, t)) for k, t in pool]
    scored.sort(key=lambda x: x[2], reverse=True)
    print(f"Scanned {scanned:,} survivors; {len(best):,} unique; pre-ranked {len(candidates):,} by char_delta; "
          f"{len(kept):,} passed freq gate. Applying POS gates to top {args.top * 3}...")

    # POS-gate only the top candidates (expensive), then keep the best args.top.
    # The token-strict gate is skipped for single-word swaps (they save chars, not tokens).
    pos_pool = scored[: args.top * 3]
    survivors = []
    for k, t, s in pos_pool:
        if not passes_pos_filters(k, t):
            continue
        if HAVE_TOK and not single_word_only and tok_count(t) >= tok_count(k):
            continue
        survivors.append((k, t, s))
    top = survivors[: args.top]
    print(f"  {len(pos_pool)} -> {len(survivors)} after POS -> merging {len(top)}")

    if args.dry_run:
        for k, t, s in top[:30]:
            print(f"  {k} -> {t}  (score {s:.2f})")
        return

    added = 0
    for k, t, _s in top:
        if k in existing_keys:
            continue
        entries[k] = t
        existing_keys.add(k)
        added += 1

    if isinstance(existing, dict) and "entries" in existing:
        existing["entries"] = entries
        out = existing
    else:
        out = entries
    DICT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "scanned_survivors": scanned,
        "unique_sources": len(best),
        "pos_pool": len(pos_pool),
        "merged": added,
        "new_dict_size": len(entries),
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {DICT_PATH} (now {len(entries)} entries). Report: {OUT_REPORT}")


if __name__ == "__main__":
    main()
