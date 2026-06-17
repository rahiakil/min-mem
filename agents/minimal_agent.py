#!/usr/bin/env python3
"""Minimal coding agent with persistent memory and optional min-mem."""

from __future__ import annotations

import json
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from min_mem import MinMemConverter  # noqa: E402

try:
    import tiktoken
except ImportError:
    tiktoken = None

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:0.5b"


@dataclass
class AgentConfig:
    model: str = DEFAULT_MODEL
    minify_memory: bool = True
    system: str = (
        "You are a minimal coding agent. Answer using ONLY the memory below. "
        "One short sentence."
    )
    skills: str = (
        "## Skills\n"
        "- recall: answer from memory\n"
        "- min_mem: memory is lexically minified\n"
    )


@dataclass
class AgentTurn:
    question: str
    answer: str
    prompt_chars: int
    prompt_tokens: int
    memory_chars: int


@dataclass
class MinimalAgent:
    """Tiny agent: system + skills + memory blocks → Ollama."""

    config: AgentConfig = field(default_factory=AgentConfig)
    memories: list[str] = field(default_factory=list)
    _converter: MinMemConverter = field(default_factory=MinMemConverter)

    def load_memories(self, texts: list[str]) -> None:
        self.memories = list(texts)

    def _active_memories(self) -> list[str]:
        if not self.config.minify_memory:
            return self.memories
        return [self._converter.minify(m).minified for m in self.memories]

    def build_prompt(self, question: str) -> str:
        mems = self._active_memories()
        block = "\n".join(f"- {m}" for m in mems)
        return f"{self.config.system}\n\n{self.config.skills}\n\n## Memory\n{block}\n\n## Question\n{question}"

    def count_tokens(self, text: str) -> int:
        if tiktoken is None:
            return len(text.split())
        return len(tiktoken.get_encoding("cl100k_base").encode(text))

    def context_stats(self) -> dict:
        mems = self._active_memories()
        prompt = self.build_prompt("x")
        raw_mem = "\n".join(self.memories)
        min_mem = "\n".join(mems)
        return {
            "memory_blocks": len(self.memories),
            "memory_chars_verbose": len(raw_mem),
            "memory_chars_active": len(min_mem),
            "memory_chars_saved": len(raw_mem) - len(min_mem),
            "memory_reduction_pct": round(
                100.0 * (len(raw_mem) - len(min_mem)) / len(raw_mem) if raw_mem else 0, 2
            ),
            "prompt_tokens": self.count_tokens(prompt),
            "pct_of_8k": round(100.0 * self.count_tokens(prompt) / 8192, 2),
        }

    def ask(self, question: str, timeout: int = 120) -> AgentTurn:
        prompt = self.build_prompt(question)
        payload = json.dumps({
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 80},
        }).encode()
        req = urllib.request.Request(
            OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        answer = data.get("response", "").strip()
        mems = self._active_memories()
        return AgentTurn(
            question=question,
            answer=answer,
            prompt_chars=len(prompt),
            prompt_tokens=self.count_tokens(prompt),
            memory_chars=sum(len(m) for m in mems),
        )

    def compare_context(self) -> dict:
        """Verbose vs minified side-by-side."""
        cfg_v = AgentConfig(minify_memory=False, model=self.config.model)
        cfg_m = AgentConfig(minify_memory=True, model=self.config.model)
        av = MinimalAgent(config=cfg_v, memories=self.memories)
        am = MinimalAgent(config=cfg_m, memories=self.memories)
        sv, sm = av.context_stats(), am.context_stats()
        return {
            "verbose": sv,
            "minified": sm,
            "tokens_saved": sv["prompt_tokens"] - sm["prompt_tokens"],
            "chars_saved": sv["memory_chars_verbose"] - sm["memory_chars_active"],
        }


def ollama_available() -> bool:
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False
