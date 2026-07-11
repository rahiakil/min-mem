"""First-run setup: dictionary install, NLTK data, optional Cursor hooks."""

from __future__ import annotations

import json
import os
import shutil
import stat
from importlib import resources
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "min-mem"
USER_DICT_PATH = CONFIG_DIR / "min_dict.json"
METRICS_PATH = CONFIG_DIR / "metrics.json"

CURSOR_HOOKS_JSON = """{
  "version": 1,
  "hooks": {
    "sessionStart": [
      {
        "command": ".cursor/hooks/min_mem_session.sh"
      }
    ]
  }
}
"""

CURSOR_HOOK_SCRIPT = """#!/usr/bin/env bash
# Min-Mem session bootstrap — loads dictionary and records session metrics.
set -euo pipefail
input=$(cat)
python3 -m min_mem.hooks.session_start <<<"$input" 2>/dev/null || echo '{}'
exit 0
"""


def bundled_dict_text() -> str:
    ref = resources.files("min_mem.data").joinpath("min_dict.json")
    return ref.read_text(encoding="utf-8")


def bundled_entry_count() -> int:
    data = json.loads(bundled_dict_text())
    entries = data.get("entries", data)
    return len(entries)


def install_user_dictionary(force: bool = False) -> Path:
    """Copy bundled dictionary to ~/.config/min-mem/min_dict.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if USER_DICT_PATH.exists() and not force:
        return USER_DICT_PATH
    USER_DICT_PATH.write_text(bundled_dict_text(), encoding="utf-8")
    return USER_DICT_PATH


def ensure_nltk() -> None:
    from min_mem.converter import _ensure_nltk_data

    _ensure_nltk_data()


def install_cursor_hook(project_root: Path | None = None) -> Path:
    """Write Cursor sessionStart hook into a project."""
    root = Path(project_root or Path.cwd())
    hooks_dir = root / ".cursor" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    hook_script = hooks_dir / "min_mem_session.sh"
    hook_script.write_text(CURSOR_HOOK_SCRIPT, encoding="utf-8")
    hook_script.chmod(hook_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    hooks_json = root / ".cursor" / "hooks.json"
    if hooks_json.exists():
        existing = json.loads(hooks_json.read_text(encoding="utf-8"))
        hooks = existing.setdefault("hooks", {})
        entries = hooks.setdefault("sessionStart", [])
        cmd = ".cursor/hooks/min_mem_session.sh"
        if not any(e.get("command") == cmd for e in entries):
            entries.append({"command": cmd})
        hooks_json.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    else:
        hooks_json.write_text(CURSOR_HOOKS_JSON, encoding="utf-8")

    return hook_script


def record_metric(chars_saved: int, tokens_saved: int = 0) -> dict:
    """Accumulate savings across sessions (used by hooks and agents)."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {"sessions": 0, "total_chars_saved": 0, "total_tokens_saved": 0, "minifications": 0}
    if METRICS_PATH.exists():
        metrics.update(json.loads(METRICS_PATH.read_text(encoding="utf-8")))
    metrics["sessions"] = metrics.get("sessions", 0) + 1
    metrics["total_chars_saved"] = metrics.get("total_chars_saved", 0) + chars_saved
    metrics["total_tokens_saved"] = metrics.get("total_tokens_saved", 0) + tokens_saved
    metrics["minifications"] = metrics.get("minifications", 0) + (1 if chars_saved else 0)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def doctor_report() -> dict:
    """Health check for install, dictionary, and NLTK."""
    dict_path = USER_DICT_PATH if USER_DICT_PATH.exists() else None
    dict_entries = 0
    dict_source = "bundled"
    if dict_path:
        data = json.loads(dict_path.read_text(encoding="utf-8"))
        dict_entries = len(data.get("entries", data))
        dict_source = str(dict_path)
    else:
        dict_entries = bundled_entry_count()

    nltk_ok = True
    nltk_error = ""
    try:
        ensure_nltk()
    except Exception as exc:  # pragma: no cover
        nltk_ok = False
        nltk_error = str(exc)

    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    return {
        "ok": nltk_ok and dict_entries > 0,
        "version": __import__("min_mem").__version__,
        "dictionary_entries": dict_entries,
        "dictionary_path": dict_source,
        "user_dict_installed": USER_DICT_PATH.exists(),
        "nltk_ok": nltk_ok,
        "nltk_error": nltk_error,
        "metrics": metrics,
        "env_dict": os.environ.get("MIN_MEM_DICT", ""),
    }


def init_project(cursor_hook: bool = True, force_dict: bool = False) -> dict:
    """Full first-run setup."""
    dict_path = install_user_dictionary(force=force_dict)
    os.environ.setdefault("MIN_MEM_DICT", str(dict_path))
    ensure_nltk()
    hook_path = None
    if cursor_hook:
        hook_path = str(install_cursor_hook())
    return {
        "dictionary": str(dict_path),
        "entries": len(json.loads(dict_path.read_text(encoding="utf-8")).get("entries", {})),
        "cursor_hook": hook_path,
        "metrics_file": str(METRICS_PATH),
    }
