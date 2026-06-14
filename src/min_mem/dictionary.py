from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

# Penn Treebank tags treated as nouns — never minified.
NOUN_TAGS = frozenset({"NN", "NNS", "NNP", "NNPS"})

DEFAULT_DICT_PATH = Path(__file__).resolve().parents[2] / "min_dict.json"

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

    def __init__(self, entries: dict[str, str]) -> None:
        self._entries = {k.lower(): v for k, v in entries.items()}
        self._phrases = sorted(
            (k for k in self._entries if " " in k),
            key=len,
            reverse=True,
        )
        self._words = {k: v for k, v in self._entries.items() if " " not in k}

    @classmethod
    def from_path(cls, path: Path | str = DEFAULT_DICT_PATH) -> MinDictionary:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = data.get("entries", data)
        return cls(entries)

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

    def word_entries(self) -> Iterator[DictionaryEntry]:
        for source, target in self._words.items():
            yield DictionaryEntry(source, target, is_phrase=False)

    def __len__(self) -> int:
        return len(self._entries)
