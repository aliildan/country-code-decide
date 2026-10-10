"""LLM baselines (plan-02 Task 7): zero-shot via the Ollama chat API, batched, cached, logged.

Only public, non-personal strings are sent (CLAUDE.md §2.3); hosted (`:cloud`) models are opt-in.
Malformed answers become abstentions, never guesses.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx

from countrycode import labels
from countrycode.evaluation.harness import Prediction
from countrycode.real.gold import GoldItem

PROMPT_PATH = Path("configs/prompts/llm_baseline.txt")
CACHE_DIR = Path("data/raw/llm")
LOG_PATH = Path("runs/llm/log.jsonl")
OLLAMA = "http://127.0.0.1:11434"
ABSTAIN = Prediction(None, (), 0.0)

PostFn = Callable[[str, str], str]


def ollama_post(
    model: str,
    prompt: str,
    host: str = OLLAMA,
    timeout: float = 300,
    max_tokens: int = 1500,
    sampling: dict | None = None,
) -> str:
    """Greedy by default (baselines); generation passes seeded sampling against loops."""
    options = {"temperature": 0, "num_predict": max_tokens, **(sampling or {})}
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "format": "json",
        "options": options,
        "messages": [{"role": "user", "content": prompt}],
    }
    r = httpx.post(f"{host}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]


def _conf(x: object) -> float:
    try:
        return min(1.0, max(0.0, float(x)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.5


def parse_answers(raw: str, n: int) -> list[Prediction]:
    out = [ABSTAIN] * n
    try:
        answers = json.loads(raw).get("answers", [])
    except (json.JSONDecodeError, AttributeError):
        return out
    for a in answers if isinstance(answers, list) else []:
        if not isinstance(a, dict):
            continue
        try:
            i = int(a.get("i", -1))
        except (TypeError, ValueError):
            continue
        code = str(a.get("code", "")).strip().upper()
        if not 0 <= i < n or code == "ABSTAIN":
            continue
        code = labels.RESERVED_TEXT_CODES.get(code, code)
        if labels.is_target(code):
            out[i] = Prediction(code, (code,), _conf(a.get("confidence")))
    return out


def render(items: Sequence[GoldItem], instructions: str) -> str:
    lines = [instructions.strip(), "", "Items:"]
    for i, g in enumerate(items):
        ctx = f"field={g.field_kind}; register={g.register_country}"
        lines.append(f"- [{i}] {ctx}; text={json.dumps(g.text, ensure_ascii=False)}")
    return "\n".join(lines)


class LLMPredictor:
    def __init__(
        self,
        model: str,
        post_fn: PostFn | None = None,
        batch: int = 25,
        cache_dir: Path = CACHE_DIR,
        log_path: Path | None = LOG_PATH,
    ) -> None:
        self.model = model
        self.name = "llm_" + model.replace(":", "_").replace("/", "_")
        self.post = post_fn or ollama_post
        self.batch = batch
        self.cache_dir = Path(cache_dir) / self.name
        self.log_path = log_path
        self.instructions = PROMPT_PATH.read_text(encoding="utf-8")

    def _call(self, prompt: str) -> str:
        key = hashlib.sha256(f"{self.model}\x1f{prompt}".encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["raw"]
        t0 = time.perf_counter()
        raw = self.post(self.model, prompt)
        rec = {
            "model": self.model,
            "prompt_sha256": key,
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "seconds": round(time.perf_counter() - t0, 3),
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**rec, "raw": raw}, ensure_ascii=False), encoding="utf-8")
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        return raw

    def predict(self, items: Sequence[GoldItem]) -> list[Prediction]:
        out: list[Prediction] = []
        for i in range(0, len(items), self.batch):
            chunk = items[i : i + self.batch]
            out += parse_answers(self._call(render(chunk, self.instructions)), len(chunk))
        return out
