from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Callable, Iterator

# Penn Treebank tags treated as nouns — never minified.
NOUN_TAGS = frozenset({"NN", "NNS", "NNP", "NNPS"})

# Dev checkout: repo-root dictionary. PyPI install: bundled package data.
_REPO_DICT_PATH = Path(__file__).resolve().parents[2] / "min_dict.json"
_USER_DICT_PATH = Path.home() / ".config" / "min-mem" / "min_dict.json"


def resolve_dict_path(explicit: Path | str | None = None) -> Path:
    """Resolve dictionary path: explicit > env > user config > repo > bundled."""
    if explicit is not None:
        return Path(explicit)
    if env := os.environ.get("MIN_MEM_DICT"):
        return Path(env)
    if _USER_DICT_PATH.exists():
        return _USER_DICT_PATH
    if _REPO_DICT_PATH.exists():
        return _REPO_DICT_PATH
    return _USER_DICT_PATH  # triggers bundled fallback in from_path


def _load_dict_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    ref = resources.files("min_mem.data").joinpath("min_dict.json")
    return json.loads(ref.read_text(encoding="utf-8"))


DEFAULT_DICT_PATH = resolve_dict_path()

_Inflector = Callable[[str], str]

_SUFFIX_RULES: list[tuple[str, Callable[[str], list[str]], _Inflector]] = [
    (
        "ing",
        lambda stem: [stem, stem + "e"] if not stem.endswith("e") else [stem[:-1], stem],
        lambda target: target[:-1] + "ing" if target.endswith("e") else target + "ing",
    ),
    (
        "ed",
        lambda stem: [stem, stem + "e"] if not stem.endswith("e") else [stem[:-1], stem],
        lambda target: target + "d" if target.endswith("e") else target + "ed",
    ),
    (
        "ly",
        lambda stem: [stem],
        lambda target: target + "ly",
    ),
    (
        "es",
        lambda stem: [stem],
        lambda target: (
            target + "es"
            if target.endswith(("s", "x", "z", "ch", "sh"))
            else target + "s"
        ),
    ),
    (
        "s",
        lambda stem: [stem],
        lambda target: target + "s",
    ),
]


def _match_inflected(word: str, words: dict[str, str]) -> str | None:
    for suffix, stems_for, inflect in _SUFFIX_RULES:
        if not word.endswith(suffix) or len(word) <= len(suffix) + 1:
            continue
        stem = word[: -len(suffix)]
        for candidate in stems_for(stem):
            target = words.get(candidate)
            if target is not None:
                return inflect(target)
    return None


@dataclass(frozen=True)
class DictionaryEntry:
    source: str
    target: str
    is_phrase: bool


class MinDictionary:
    """Loads and indexes the minimal synonym dictionary."""

    def __init__(self, entries: dict[str, str], noun_abbreviations: dict[str, str] | None = None) -> None:
        self._entries = {k.lower(): v for k, v in entries.items()}
        self._phrases = sorted(
            (k for k in self._entries if " " in k),
            key=len,
            reverse=True,
        )
        self._words = {k: v for k, v in self._entries.items() if " " not in k}
        # Unambiguous common-noun abbreviations applied BEFORE the POS noun-gate
        # (entity nouns remain protected). Keys are single words.
        self._noun_abbreviations = {k.lower(): v for k, v in (noun_abbreviations or {}).items()}

    @classmethod
    def from_path(cls, path: Path | str | None = None) -> MinDictionary:
        resolved = resolve_dict_path(path)
        data = _load_dict_json(resolved)
        entries = data.get("entries", data)
        noun_abbrev = data.get("noun_abbreviations", {}) if isinstance(data, dict) else {}
        return cls(entries, noun_abbrev)

    @classmethod
    def from_dict(cls, entries: dict[str, str]) -> MinDictionary:
        return cls(entries)

    def lookup_word(self, word: str) -> str | None:
        return self._words.get(word.lower())

    def lookup_inflected(self, word: str) -> str | None:
        """Resolve base or inflected forms to a min target."""
        direct = self.lookup_word(word)
        if direct is not None:
            return direct
        return _match_inflected(word.lower(), self._words)

    def phrase_entries(self) -> Iterator[DictionaryEntry]:
        for source in self._phrases:
            yield DictionaryEntry(source, self._entries[source], is_phrase=True)

    def noun_abbreviation_entries(self) -> Iterator[DictionaryEntry]:
        """Unambiguous common-noun abbreviations; applied before the POS gate."""
        for source, target in self._noun_abbreviations.items():
            yield DictionaryEntry(source, target, is_phrase=True)

    def noun_abbreviation_sources(self) -> frozenset[str]:
        """Lowercased source nouns in the abbreviation allowlist."""
        return frozenset(self._noun_abbreviations.keys())

    def word_entries(self) -> Iterator[DictionaryEntry]:
        for source, target in self._words.items():
            yield DictionaryEntry(source, target, is_phrase=False)

    def __len__(self) -> int:
        return len(self._entries)
