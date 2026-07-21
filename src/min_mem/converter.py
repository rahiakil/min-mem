from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from min_mem.dictionary import DEFAULT_DICT_PATH, NOUN_TAGS, MinDictionary

try:
    import nltk
    from nltk import pos_tag, word_tokenize
    from nltk.data import find as nltk_find
except ImportError as exc:  # pragma: no cover - exercised via lazy import guard
    nltk = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _ensure_nltk_data() -> None:
    if nltk is None:
        raise ImportError(
            "nltk is required for POS-aware minification. "
            "Install with: pip install nltk"
        ) from _IMPORT_ERROR

    packages = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for resource_path, package in packages:
        try:
            nltk_find(resource_path)
        except LookupError:
            nltk.download(package, quiet=True)


@dataclass
class Replacement:
    original: str
    replacement: str
    position: int


@dataclass
class MinifyResult:
    original: str
    minified: str
    replacements: list[Replacement] = field(default_factory=list)

    @property
    def original_chars(self) -> int:
        return len(self.original)

    @property
    def minified_chars(self) -> int:
        return len(self.minified)

    @property
    def chars_saved(self) -> int:
        return self.original_chars - self.minified_chars

    @property
    def savings_ratio(self) -> float:
        if self.original_chars == 0:
            return 0.0
        return self.chars_saved / self.original_chars


class MinMemConverter:
    """Convert memory text to a shorter equivalent using the min dictionary.

    Nouns are never replaced. Multi-word phrases are handled before single words.
    Case of the first letter is preserved when possible.
    """

    def __init__(self, dictionary: MinDictionary | None = None) -> None:
        self.dictionary = dictionary or MinDictionary.from_path()

    @classmethod
    def from_dict_path(cls, path: Path | str) -> MinMemConverter:
        return cls(MinDictionary.from_path(path))

    def minify(self, text: str) -> MinifyResult:
        return self._minify_once(text)

    def minify_passes(self, text: str, passes: int = 2) -> MinifyResult:
        """Apply minification repeatedly until stable or *passes* exhausted."""
        if passes < 1:
            passes = 1
        # Nouns identified in the original text are protected across all passes:
        # re-tagging already-minified text can mis-tag a noun as a verb and let a
        # noun source entry fire, corrupting an entity. Protecting by the original
        # tagging preserves the noun-preservation guarantee across passes.
        protected = self._noun_tokens(text)
        result = self._minify_once(text, protected=protected)
        for _ in range(passes - 1):
            if not result.replacements:
                break
            nxt = self._minify_once(result.minified, protected=protected)
            if nxt.minified == result.minified:
                break
            result = MinifyResult(
                original=text,
                minified=nxt.minified,
                replacements=result.replacements + nxt.replacements,
            )
        return result

    def _noun_tokens(self, text: str) -> frozenset[str]:
        _ensure_nltk_data()
        return frozenset(
            tok.lower()
            for tok, tag in pos_tag(word_tokenize(text))
            if tag in NOUN_TAGS
        )

    def _minify_once(self, text: str, protected: frozenset[str] | None = None) -> MinifyResult:
        _ensure_nltk_data()

        replacements: list[Replacement] = []
        working = text

        for entry in self.dictionary.phrase_entries():
            pattern = re.compile(re.escape(entry.source), re.IGNORECASE)

            def _phrase_sub(match: re.Match[str], target: str = entry.target) -> str:
                original = match.group(0)
                replacements.append(
                    Replacement(original=original, replacement=target, position=match.start())
                )
                return _preserve_case(original, target)

            working = pattern.sub(_phrase_sub, working)

        # Unambiguous common-noun abbreviations: applied before the POS noun-gate
        # so the entity-noun protection guarantee is preserved for every other noun.
        # Only the explicit, auditable allowlist in the dictionary bypasses the gate.
        for entry in self.dictionary.noun_abbreviation_entries():
            pattern = re.compile(re.escape(entry.source), re.IGNORECASE)

            def _abbrev_sub(match: re.Match[str], target: str = entry.target) -> str:
                original = match.group(0)
                replacements.append(
                    Replacement(original=original, replacement=target, position=match.start())
                )
                return _preserve_case(original, target)

            working = pattern.sub(_abbrev_sub, working)

        tokens = word_tokenize(working)
        tagged = pos_tag(tokens)

        minified_tokens: list[str] = []
        cursor = 0

        for token, tag in tagged:
            pos = cursor
            cursor = working.find(token, cursor)
            if cursor == -1:
                cursor = pos

            replacement: str | None = None
            if tag not in NOUN_TAGS and (protected is None or token.lower() not in protected):
                replacement = self.dictionary.lookup_inflected(token)

            if replacement is not None and replacement.lower() != token.lower():
                cased = _preserve_case(token, replacement)
                replacements.append(
                    Replacement(original=token, replacement=cased, position=cursor)
                )
                minified_tokens.append(cased)
            else:
                minified_tokens.append(token)

            cursor += len(token)

        minified = _detokenize(minified_tokens)
        return MinifyResult(original=text, minified=minified, replacements=replacements)

    def expand(self, text: str) -> str:
        """Best-effort inverse: map short forms back to longer dictionary keys."""
        reverse = {}
        for entry in self.dictionary.word_entries():
            reverse.setdefault(entry.target.lower(), entry.source)

        tokens = word_tokenize(text)
        expanded: list[str] = []
        for token in tokens:
            expanded.append(reverse.get(token.lower(), token))
        return _detokenize(expanded)


def _preserve_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _detokenize(tokens: list[str]) -> str:
    if not tokens:
        return ""

    no_space_before = {".", ",", "!", "?", ":", ";", ")", "]", "}", "'s", "n't", "'re", "'ve", "'ll", "'d", "'m"}
    no_space_after = {"(", "[", "{"}

    parts: list[str] = []
    for i, token in enumerate(tokens):
        if i == 0:
            parts.append(token)
            continue

        prev = tokens[i - 1]
        if token in no_space_before:
            parts.append(token)
        elif prev in no_space_after:
            parts.append(token)
        else:
            parts.append(" " + token)

    return "".join(parts)
