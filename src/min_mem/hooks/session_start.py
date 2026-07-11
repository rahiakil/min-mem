#!/usr/bin/env python3
"""Cursor sessionStart hook — bootstrap dictionary and expose stats to the agent."""

from __future__ import annotations

import json
import sys

from min_mem.bootstrap import doctor_report, init_project


def main() -> int:
    _ = sys.stdin.read()  # Cursor hook payload (session metadata)
    try:
        init_project(cursor_hook=False, force_dict=False)
    except Exception:
        pass
    report = doctor_report()
    payload = {
        "additional_context": (
            f"Min-Mem active: {report['dictionary_entries']} synonym entries loaded "
            f"from {report['dictionary_path']}. "
            "Use min-mem minify on agent memory before persistence to reduce context size "
            "while preserving nouns."
        )
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
