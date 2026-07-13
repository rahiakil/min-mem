#!/usr/bin/env python3
"""Generate publication figures from benchmark results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = Path(__file__).parent / "results.json"
FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def fig_method_comparison(data: dict) -> None:
    summary = data["summary"]
    methods = ["min-mem (full)", "phrase-only", "no-inflection", "naive-dict"]
    labels = ["Min-Mem", "Phrase-only", "No inflection", "Naive dict"]
    char_sav = [summary[m]["char_savings_pct_mean"] for m in methods]
    token_sav = [summary[m]["token_savings_pct_mean"] for m in methods]
    noun_pres = [summary[m]["nouns_preserved_pct_mean"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))

    ax = axes[0]
    ymax = max(char_sav + token_sav) * 1.35
    bars_char = ax.bar(x - width / 2, char_sav, width, label="Characters", color="#2c5f8a")
    bars_tok = ax.bar(x + width / 2, token_sav, width, label="Tokens (GPT-4)", color="#6baed6")
    ax.set_ylabel("Mean reduction (%)")
    ax.set_title("(a) Size reduction by method")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, ymax)
    ax.margins(x=0.08)
    for bars in (bars_char, bars_tok):
        for bar in bars:
            h = bar.get_height()
            if h <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + ymax * 0.02,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )
    ax.axhline(
        data["gzip_baseline"]["mean_savings_pct"],
        color="#c44e52",
        linestyle="--",
        linewidth=1,
        label="gzip (bytes)",
    )
    ax.legend(loc="upper right", fontsize=6, framealpha=0.9)

    ax = axes[1]
    colors = ["#2ca02c" if v >= 99 else "#ff7f0e" if v >= 95 else "#d62728" for v in noun_pres]
    bars = ax.bar(x, noun_pres, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Noun preservation (%)")
    ax.set_title("(b) Entity safety (noun tags)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 112)
    ax.margins(x=0.08)
    for bar, val in zip(bars, noun_pres):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(val + 2, 108),
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.subplots_adjust(bottom=0.28, wspace=0.28)
    fig.savefig(FIGURES / "fig_method_comparison.pdf")
    fig.savefig(FIGURES / "fig_method_comparison.png")
    plt.close(fig)


def fig_category_breakdown(data: dict) -> None:
    cats = data["by_category_char_savings"]
    names = [k.replace("_", "\n") for k in cats]
    values = list(cats.values())
    colors = plt.cm.Blues(np.linspace(0.45, 0.85, len(values)))

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    bars = ax.barh(names, values, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Character reduction (%)")
    ax.set_title("Min-Mem savings by memory category")
    ax.set_xlim(0, max(values) * 1.2)
    for bar, val in zip(bars, values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_category_breakdown.pdf")
    fig.savefig(FIGURES / "fig_category_breakdown.png")
    plt.close(fig)


def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.3, 1.5, 1.6, 1.0, "Input\nmemory text", "#e8f4fc"),
        (2.3, 1.5, 1.6, 1.0, "Phrase\nreplacement", "#b3d9f2"),
        (4.3, 1.5, 1.6, 1.0, "Tokenize\n+ POS tag", "#7ec8e3"),
        (6.3, 1.5, 1.6, 1.0, "Dict lookup\n+ inflect", "#4aa3cc"),
        (8.3, 1.5, 1.6, 1.0, "Minified\noutput", "#2c5f8a"),
    ]
    for x, y, w, h, label, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7, color="black")

    for x1, x2 in [(1.9, 2.3), (3.9, 4.3), (5.9, 6.3), (7.9, 8.3)]:
        ax.annotate("", xy=(x2, 2.0), xytext=(x1, 2.0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1))

    ax.text(6.3, 0.7, "Skip if NN / NNP", fontsize=7, style="italic", ha="center")
    ax.annotate("", xy=(7.1, 1.45), xytext=(7.1, 0.95),
                arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=0.8))

    ax.set_title("Min-Mem processing pipeline", fontsize=10, pad=8)
    fig.savefig(FIGURES / "fig_architecture.pdf")
    fig.savefig(FIGURES / "fig_architecture.png")
    plt.close(fig)


def fig_semantic_proxy(data: dict) -> None:
    summary = data["summary"]
    methods = ["min-mem (full)", "phrase-only", "no-inflection", "naive-dict"]
    labels = ["Min-Mem", "Phrase", "No infl.", "Naive"]
    syn = [summary[m]["synonym_aware_pct_mean"] for m in methods]

    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    ax.bar(labels, syn, color=["#2ca02c", "#98df8a", "#aec7e8", "#ffbb78"], edgecolor="black", linewidth=0.4)
    ax.set_ylabel("Synonym-aware retention (%)")
    ax.set_title("Semantic preservation proxy")
    ax.set_ylim(0, 105)
    for i, v in enumerate(syn):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_semantic_proxy.pdf")
    fig.savefig(FIGURES / "fig_semantic_proxy.png")
    plt.close(fig)


def fig_dictionary_ablation(data: dict) -> None:
    abl = data.get("dictionary_ablation", {}).get("tiers", [])
    if not abl:
        return
    labels = [t["label"] for t in abl]
    sizes = [t["dict_size"] for t in abl]
    chars = [t["char_savings_pct_mean"] for t in abl]

    fig, ax1 = plt.subplots(figsize=(5.8, 3.0))
    color = "#2c5f8a"
    ymax = max(chars) * 1.28 if chars else 25
    bars = ax1.bar(range(len(labels)), chars, color=color, edgecolor="black", linewidth=0.4)
    ax1.set_ylabel("Character reduction (%)")
    short_labels = ["None", "Phrases", "+Conn.", "+Verbs", "+Adj", "Full"]
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(
        [f"{short_labels[i]}\n(n={sizes[i]})" for i in range(len(labels))],
        fontsize=7,
    )
    ax1.set_title("Dictionary ablation: savings vs dictionary size")
    ax1.set_ylim(0, ymax)
    ax1.margins(x=0.05)
    for bar, v in zip(bars, chars):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            v + ymax * 0.03,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(FIGURES / "fig_dictionary_ablation.pdf")
    fig.savefig(FIGURES / "fig_dictionary_ablation.png")
    plt.close(fig)


def fig_top_swaps(data: dict) -> None:
    hits = data.get("top_swaps", [])[:10]
    if not hits:
        return
    labels = [h["swap"] for h in hits][::-1]
    counts = [h["count"] for h in hits][::-1]
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ax.barh(labels, counts, color="#6366f1", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Occurrences across corpus")
    ax.set_title("Top min dictionary replacements")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_top_swaps.pdf")
    fig.savefig(FIGURES / "fig_top_swaps.png")
    plt.close(fig)


def fig_per_sample(data: dict) -> None:
    samples = data.get("per_sample_minmem", [])
    if not samples:
        return
    chars = [s["char_savings_pct"] for s in samples]
    tokens = [s["token_savings_pct"] for s in samples]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.scatter(chars, tokens, c="#3b9eff", edgecolors="black", linewidths=0.4, s=40, alpha=0.85)
    ax.set_xlabel("Character reduction (%)")
    ax.set_ylabel("Token reduction (%)")
    ax.set_title("Per-sample char vs token savings")
    ax.axhline(0, color="#666", linewidth=0.5)
    ax.axvline(0, color="#666", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_per_sample.pdf")
    fig.savefig(FIGURES / "fig_per_sample.png")
    plt.close(fig)


def fig_improve_history() -> None:
    hist_path = Path(__file__).parent / "improve_history.json"
    if not hist_path.exists():
        return
    hist = json.loads(hist_path.read_text())
    iters = [h["iteration"] for h in hist]
    chars = [h["char_savings_pct"] for h in hist]
    sizes = [h["dict_size"] for h in hist]
    tokens = [h["agent_tokens_saved"] for h in hist]

    fig, ax1 = plt.subplots(figsize=(5.0, 2.6))
    ax1.plot(iters, chars, "o-", color="#2c5f8a", label="Char savings %")
    ax1.set_xlabel("Improvement iteration")
    ax1.set_ylabel("Char reduction (%)", color="#2c5f8a")
    ax2 = ax1.twinx()
    ax2.plot(iters, sizes, "s--", color="#22d3a6", label="Dict size")
    ax2.set_ylabel("Dictionary entries", color="#22d3a6")
    ax1.set_title("Dictionary growth loop convergence")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_improve_history.pdf")
    fig.savefig(FIGURES / "fig_improve_history.png")
    plt.close(fig)


def fig_value_decomposition(data: dict) -> None:
    """Phrase rules vs full POS-gated policy pipeline."""
    abl = {t["tier"]: t for t in data.get("dictionary_ablation", {}).get("tiers", [])}
    summary = data["summary"]
    if not abl:
        return

    labels = ["Phrases\nonly", "Verbs +\nPOS policy", "Full\npipeline", "Naive\n(no POS)"]
    chars = [
        abl["phrases"]["char_savings_pct_mean"],
        abl["+verbs"]["char_savings_pct_mean"],
        abl["full"]["char_savings_pct_mean"],
        summary["naive-dict"]["char_savings_pct_mean"],
    ]
    nouns = [
        abl["phrases"]["nouns_preserved_pct_mean"],
        abl["+verbs"]["nouns_preserved_pct_mean"],
        abl["full"]["nouns_preserved_pct_mean"],
        summary["naive-dict"]["nouns_preserved_pct_mean"],
    ]
    colors = ["#9ecae1", "#6baed6", "#2c5f8a", "#d62728"]

    fig, ax1 = plt.subplots(figsize=(5.2, 3.0))
    x = np.arange(len(labels))
    ymax = max(chars) * 1.3
    bars = ax1.bar(x, chars, color=colors, edgecolor="black", linewidth=0.4)
    ax1.set_ylabel("Mean char reduction (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_title("Where savings come from: phrases vs POS policy")
    ax1.set_ylim(0, ymax)
    ax1.margins(x=0.08)
    for bar, val, noun in zip(bars, chars, nouns):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + ymax * 0.03,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            ymax * 0.12,
            f"noun {noun:.0f}%",
            ha="center",
            va="bottom",
            fontsize=6,
            color="#444444",
        )
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(FIGURES / "fig_value_decomposition.pdf")
    fig.savefig(FIGURES / "fig_value_decomposition.png")
    plt.close(fig)


def main() -> None:
    data = load()
    fig_method_comparison(data)
    fig_category_breakdown(data)
    fig_architecture()
    fig_semantic_proxy(data)
    fig_dictionary_ablation(data)
    fig_value_decomposition(data)
    fig_top_swaps(data)
    fig_per_sample(data)
    fig_improve_history()
    print(f"Figures written to {FIGURES}")


if __name__ == "__main__":
    main()
