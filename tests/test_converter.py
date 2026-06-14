import pytest

from min_mem.converter import MinMemConverter
from min_mem.dictionary import MinDictionary


@pytest.fixture
def converter() -> MinMemConverter:
    return MinMemConverter(
        MinDictionary.from_dict(
            {
                "utilize": "use",
                "in order to": "to",
                "nevertheless": "yet",
                "investigate": "check",
                "numerous": "many",
                "previously": "before",
                "however": "but",
                "require": "need",
            }
        )
    )


def test_minify_replaces_verbs_and_phrases(converter: MinMemConverter) -> None:
    text = "We utilize Python in order to investigate numerous issues."
    result = converter.minify(text)

    assert "use" in result.minified
    assert "to" in result.minified
    assert "check" in result.minified
    assert "many" in result.minified
    assert "utilize" not in result.minified.lower()
    assert result.chars_saved > 0


def test_minify_preserves_nouns(converter: MinMemConverter) -> None:
    text = "Python is a language."
    result = converter.minify(text)
    assert result.minified == text
    assert result.replacements == []


def test_minify_preserves_case(converter: MinMemConverter) -> None:
    text = "Utilize the tool."
    result = converter.minify(text)
    assert result.minified.startswith("Use")


def test_phrase_replacement_before_words(converter: MinMemConverter) -> None:
    text = "Do this in order to finish."
    result = converter.minify(text)
    assert "in order to" not in result.minified.lower()
    assert " to " in result.minified or result.minified.endswith(" to finish.")


def test_stats_ratio(converter: MinMemConverter) -> None:
    text = "However, we previously utilized numerous tools."
    result = converter.minify(text)
    assert 0 < result.savings_ratio < 1
    assert len(result.replacements) >= 3


def test_minify_handles_inflections(converter: MinMemConverter) -> None:
    text = "They required utilizing optimized tools."
    result = converter.minify(text)
    assert "need" in result.minified.lower() or "needed" in result.minified.lower()


def test_expand_roundtrip(converter: MinMemConverter) -> None:
    text = "We use Python to check many issues."
    expanded = converter.expand(text)
    assert "utilize" in expanded or "investigate" in expanded
