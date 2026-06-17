#!/usr/bin/env python3
"""Build dictionary tiers for ablation benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

PHRASE_HINTS = {"in order to", "in spite of", "in the event that", "prior to", "relating to"}
CONNECTOR_HINTS = {
    "however", "therefore", "nevertheless", "nonetheless", "furthermore",
    "moreover", "accordingly", "consequently", "thus", "despite",
}
VERB_HINTS = {
    "utilize", "accomplish", "investigate", "facilitate", "require", "demonstrate",
    "obtain", "commence", "terminate", "endeavor", "possess", "indicate",
    "establish", "maintain", "determine", "recognize", "understand", "communicate",
    "construct", "discover", "eliminate", "examine", "implement", "incorporate",
    "inform", "inquire", "modify", "participate", "purchase", "remove", "respond",
    "retain", "assist", "attempt", "comprehend", "communicate", "reside",
}


def load_entries(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("entries", data)


def classify(entries: dict[str, str]) -> dict[str, list[str]]:
    phrases, connectors, verbs, other = [], [], [], []
    for k in entries:
        if " " in k or k in PHRASE_HINTS:
            phrases.append(k)
        elif k in CONNECTOR_HINTS:
            connectors.append(k)
        elif k in VERB_HINTS:
            verbs.append(k)
        else:
            other.append(k)
    return {
        "phrases": sorted(phrases, key=len, reverse=True),
        "connectors": sorted(connectors),
        "verbs": sorted(verbs),
        "other": sorted(other),
    }


def build_tiers(full: dict[str, str]) -> list[dict]:
    """Progressive dictionary tiers for ablation."""
    groups = classify(full)
    tiers: list[dict] = [
        {"name": "identity", "label": "No dictionary", "entries": {}},
    ]

    def subset(keys: list[str]) -> dict[str, str]:
        return {k: full[k] for k in keys if k in full}

    cumulative: dict[str, str] = {}
    steps = [
        ("phrases", "Phrases only", groups["phrases"]),
        ("+connectors", "+ Connectors", groups["connectors"]),
        ("+verbs", "+ Verbs", groups["verbs"]),
        ("+adj/other", "+ Adj/Other", groups["other"]),
    ]
    for slug, label, keys in steps:
        cumulative.update(subset(keys))
        tiers.append({
            "name": slug,
            "label": label,
            "entries": dict(cumulative),
            "size": len(cumulative),
        })

    tiers.append({"name": "full", "label": "Full min dictionary", "entries": full, "size": len(full)})
    return tiers


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    tiers = build_tiers(load_entries(root / "min_dict.json"))
    for t in tiers:
        print(f"{t['label']}: {t.get('size', 0)} entries")
