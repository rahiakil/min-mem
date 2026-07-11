"""Tests for bootstrap and bundled dictionary."""

from __future__ import annotations

import json
from importlib import resources

from min_mem.bootstrap import bundled_entry_count, doctor_report
from min_mem.dictionary import MinDictionary, resolve_dict_path


def test_bundled_dictionary_loads() -> None:
    d = MinDictionary.from_path()
    assert len(d) >= 100


def test_bundled_data_file_exists() -> None:
    ref = resources.files("min_mem.data").joinpath("min_dict.json")
    data = json.loads(ref.read_text(encoding="utf-8"))
    assert "entries" in data
    assert len(data["entries"]) == bundled_entry_count()


def test_doctor_report() -> None:
    report = doctor_report()
    assert report["dictionary_entries"] > 0
    assert "version" in report


def test_resolve_dict_path_explicit(tmp_path) -> None:
    custom = tmp_path / "custom.json"
    custom.write_text('{"entries": {"utilize": "use"}}', encoding="utf-8")
    assert resolve_dict_path(custom) == custom
