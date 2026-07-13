"""Optional Ollama cross-model reader (benchmark evaluation only, not write path)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODELS = ["qwen2.5:0.5b"]


@dataclass
class OllamaStatus:
    available: bool
    models: list[str]
    error: str = ""


def check_ollama(models: list[str] | None = None) -> OllamaStatus:
    models = models or DEFAULT_MODELS
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        available_names = {m["name"].split(":")[0] for m in data.get("models", [])}
        found = []
        for m in models:
            base = m.split(":")[0]
            if any(base in n for n in available_names):
                found.append(m)
        return OllamaStatus(available=bool(found), models=found)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return OllamaStatus(available=False, models=[], error=str(exc))


def ollama_answer(model: str, context: str, question: str, timeout: int = 60) -> tuple[str, int]:
    prompt = (
        "Answer using ONLY the memory below. One short phrase.\n\n"
        f"## Memory\n{context}\n\n## Question\n{question}\n\nAnswer:"
    )
    t0 = time.time()
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 40},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    latency = int((time.time() - t0) * 1000)
    return data.get("response", "").strip(), latency
