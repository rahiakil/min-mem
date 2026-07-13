"""Compression modes for baseline comparison: rule-based, LM-deletion, and tiered chains."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Protocol


class Compressor(Protocol):
    name: str
    inference_cost: str

    def compress(self, text: str) -> str: ...


@dataclass
class IdentityCompressor:
    name: str = "identity"
    inference_cost: str = "none"

    def compress(self, text: str) -> str:
        return text


@dataclass
class MinMemCompressor:
    converter: object
    name: str = "min-mem"
    inference_cost: str = "none"
    passes: int = 1

    def compress(self, text: str) -> str:
        if self.passes <= 1:
            return self.converter.minify(text).minified
        return self.converter.minify_passes(text, self.passes).minified


@dataclass
class ChainedCompressor:
    first: Compressor
    second: Compressor
    name: str = ""
    inference_cost: str = "small_lm"

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.first.name}+{self.second.name}"
        costs = {self.first.inference_cost, self.second.inference_cost}
        if costs == {"none"}:
            self.inference_cost = "none"
        elif "none" in costs:
            self.inference_cost = "small_lm"
        else:
            self.inference_cost = "small_lm"

    def compress(self, text: str) -> str:
        return self.second.compress(self.first.compress(text))


class GPT2SelectiveContext:
    """Phrase-level self-information pruning via GPT-2 (Selective Context family).

    Lightweight stand-in when `selective-context` pip package fails to build.
    """

    name = "gpt2-selective"
    inference_cost = "small_lm_gpt2"

    def __init__(self, keep_ratio: float = 0.5) -> None:
        self.keep_ratio = keep_ratio
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        self._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self._model = GPT2LMHeadModel.from_pretrained("gpt2")
        self._model.eval()
        self._torch = torch

    def _unit_info(self, unit: str) -> float:
        self._load()
        enc = self._tokenizer(unit, return_tensors="pt", truncation=True, max_length=512)
        with self._torch.no_grad():
            out = self._model(**enc, labels=enc["input_ids"])
        return float(out.loss.item())

    @staticmethod
    def _split_units(text: str) -> list[str]:
        parts = re.split(r"(?<=[.,;])\s+|\s+(?=(?:However|Nevertheless|But|And)\b)", text)
        units = [p.strip() for p in parts if p.strip()]
        return units or [text]

    def compress(self, text: str) -> str:
        units = self._split_units(text)
        if len(units) <= 1:
            return text
        scored = [(u, self._unit_info(u)) for u in units]
        keep_n = max(1, round(len(units) * self.keep_ratio))
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        keep_set = {u for u, _ in ranked[:keep_n]}
        kept = [u for u in units if u in keep_set]
        return " ".join(kept)


class LLMLingua2Compressor:
    name = "llmlingua-2"
    inference_cost = "small_lm_mbert"

    def __init__(self, rate: float = 0.5) -> None:
        self.rate = rate
        self._compressor = None

    def _load(self) -> None:
        if self._compressor is not None:
            return
        from llmlingua import PromptCompressor

        self._compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map="cpu",
        )

    def compress(self, text: str) -> str:
        self._load()
        out = self._compressor.compress_prompt(
            text, rate=self.rate, force_tokens=["\n", "?"]
        )
        return out["compressed_prompt"]


def build_compressors(
    min_mem_converter,
    *,
    rates: tuple[float, ...] = (0.5,),
    include_gpt2: bool = True,
    include_llmlingua: bool = True,
    include_two_pass: bool = True,
) -> list[Compressor]:
    """Build mode list for benchmark sweep."""
    mm1 = MinMemCompressor(min_mem_converter, name="min-mem", passes=1)
    modes: list[Compressor] = [
        IdentityCompressor(),
        mm1,
    ]

    if include_two_pass:
        mm2 = MinMemCompressor(min_mem_converter, name="min-mem×2", passes=2)
        modes.append(mm2)

    if include_gpt2:
        for keep in rates:
            gpt = GPT2SelectiveContext(keep_ratio=keep)
            gpt.name = f"gpt2-selective@{keep}"
            modes.append(gpt)
            modes.append(
                ChainedCompressor(
                    mm1,
                    gpt,
                    name=f"min-mem+gpt2-selective@{keep}",
                )
            )
            if include_two_pass:
                modes.append(
                    ChainedCompressor(
                        mm2,
                        gpt,
                        name=f"min-mem×2+gpt2-selective@{keep}",
                    )
                )

    if include_llmlingua:
        for rate in rates:
            ll2 = LLMLingua2Compressor(rate=rate)
            ll2.name = f"llmlingua-2@{rate}"
            modes.append(ll2)
            modes.append(
                ChainedCompressor(
                    mm1,
                    ll2,
                    name=f"min-mem+llmlingua-2@{rate}",
                )
            )
            if include_two_pass:
                modes.append(
                    ChainedCompressor(
                        mm2,
                        ll2,
                        name=f"min-mem×2+llmlingua-2@{rate}",
                    )
                )

    return modes


def try_import_optional() -> dict[str, bool]:
    flags = {"gpt2": False, "llmlingua": False, "torch": False}
    try:
        import torch  # noqa: F401

        flags["torch"] = True
        flags["gpt2"] = True
    except ImportError:
        pass
    try:
        import llmlingua  # noqa: F401

        flags["llmlingua"] = True
    except ImportError:
        pass
    return flags
