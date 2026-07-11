"""Drop-in memory minifier for agent frameworks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from min_mem.converter import MinMemConverter
from min_mem.bootstrap import METRICS_PATH, record_metric


@dataclass
class SessionStats:
    blocks_minified: int = 0
    chars_before: int = 0
    chars_after: int = 0

    @property
    def chars_saved(self) -> int:
        return self.chars_before - self.chars_after

    @property
    def savings_pct(self) -> float:
        if self.chars_before == 0:
            return 0.0
        return 100.0 * self.chars_saved / self.chars_before

    def to_dict(self) -> dict:
        return {
            "blocks_minified": self.blocks_minified,
            "chars_before": self.chars_before,
            "chars_after": self.chars_after,
            "chars_saved": self.chars_saved,
            "savings_pct": round(self.savings_pct, 2),
        }


@dataclass
class MemoryMinifier:
    """Wrap agent memory writes with minification and cumulative metrics."""

    converter: MinMemConverter = field(default_factory=MinMemConverter)
    stats: SessionStats = field(default_factory=SessionStats)

    def minify_block(self, text: str) -> str:
        result = self.converter.minify(text)
        self.stats.blocks_minified += 1
        self.stats.chars_before += result.original_chars
        self.stats.chars_after += result.minified_chars
        if result.chars_saved:
            record_metric(result.chars_saved)
        return result.minified

    def minify_many(self, blocks: list[str]) -> list[str]:
        return [self.minify_block(b) for b in blocks]

    def report(self) -> dict:
        return self.stats.to_dict()

    def save_report(self, path: Path | str = METRICS_PATH.parent / "last_session.json") -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.report(), indent=2), encoding="utf-8")
        return out
