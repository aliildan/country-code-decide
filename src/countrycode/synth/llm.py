"""LLM variant generators `llm@1` and `llmjunk@1` (plan-03 Task 7).

One prompt per target (250 countries + 51 US states) asks a local LLM for realistic ways people
write the place; one prompt asks for non-place strings (→ NONE). Only target names and codes are
sent (no register data). Responses are cached and logged; every variant still passes G1, and the
owner audits a random sample (CLAUDE.md §3).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from countrycode import labels
from countrycode.paths import CONFIGS
from countrycode.synth.framework import SHARED_ISLAND, BaseAlias, is_latin

MODEL = "qwen3.5:35b-a3b-q4_K_M"  # Apache-2.0 (Qwen); local Ollama
CACHE_PATH = Path("data/interim/synth/llm_variants.jsonl")
LOG_PATH = Path("runs/llm/generation_log.jsonl")
PROMPT_TARGET = CONFIGS / "prompts" / "llm_variants.txt"
PROMPT_JUNK = CONFIGS / "prompts" / "llm_junk.txt"
JUNK_KEY = "__junk__"
# Seeded sampling: greedy decoding loops on long lists ("Pan-Pan-Pan-…"); the seed keeps it
# reproducible.
SAMPLING = {"temperature": 0.6, "top_p": 0.9, "repeat_penalty": 1.15, "seed": 13}

PostFn = Callable[[str, str], str]


def parse_variants(raw: str) -> list[str]:
    try:
        vs = json.loads(raw).get("variants", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    out: list[str] = []
    for v in vs if isinstance(vs, list) else []:
        if isinstance(v, str) and v.strip() and v.strip() not in out:
            out.append(v.strip())
    return out


def target_name(code: str) -> str:
    return labels.us_state_name(code) if code.startswith("US-") else labels.country_name(code)


def prompt_for(code: str) -> str:
    if code == JUNK_KEY:
        return PROMPT_JUNK.read_text(encoding="utf-8")
    extra = (
        " — a US state; also as written in a US 'state of incorporation' field"
        if code.startswith("US-")
        else ""
    )
    return PROMPT_TARGET.read_text(encoding="utf-8").format(
        name=target_name(code), code=code, extra=extra
    )


def read_cache(path: Path = CACHE_PATH) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    out: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["target"]] = d["variants"]
    return out


def build_llm_cache(
    model: str,
    targets: Sequence[str],
    post_fn: PostFn | None = None,
    path: Path = CACHE_PATH,
    log_path: Path | None = LOG_PATH,
    include_junk: bool = True,
) -> int:
    from countrycode.evaluation.llm import ollama_post

    done = {t for t, v in read_cache(path).items() if v}  # empty answers are retried
    todo = [t for t in [*targets, *([JUNK_KEY] if include_junk else [])] if t not in done]
    path.parent.mkdir(parents=True, exist_ok=True)
    for t in todo:
        prompt = prompt_for(t)
        budget = 4000 if t == JUNK_KEY else 3000  # long multilingual lists need room
        try:
            raw = (
                post_fn(model, prompt)
                if post_fn
                else ollama_post(model, prompt, max_tokens=budget, sampling=SAMPLING)
            )
        except Exception as e:  # noqa: BLE001 — a failed prompt is skipped, retried next run
            print(f"llm-cache: {t} failed: {type(e).__name__}", flush=True)
            continue
        rec = {
            "target": t,
            "model": model,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "sampling": None if post_fn else SAMPLING,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**rec, "variants": parse_variants(raw)}, ensure_ascii=False) + "\n")
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
    return len(todo)


COMPANY_WORDS = frozenset({"holding", "holdings", "company", "corporation", "corp", "inc",
                           "incorporated", "ltd", "limited", "llc", "plc", "gmbh", "group",
                           "bank"})  # fmt: skip


def _company_like(s: str) -> bool:
    return bool(COMPANY_WORDS & {w for w in re.split(r"[^\w]+", s.lower()) if w})


def _foreign_code(s: str, own: tuple[str, ...]) -> bool:
    """A two-letter country code of another place ("Argentina (US)"); the own code, the own
    state's USPS code and the own country (US for a state) are fine."""
    home = {labels.country_of(t) for t in own}
    for tok in re.findall(r"\b[A-Z]{2}\b", s):
        code = "GB" if tok == "UK" else tok
        if f"US-{tok}" in own or code in own or code in home:
            continue
        if labels.is_target(code):
            return True
    return False


class LlmVariants:
    name = "llm"
    version = "1"

    def __init__(self, path: Path = CACHE_PATH, model: str = MODEL, version: str = "1") -> None:
        self.version = version  # 2 (plan-08): company names dropped ("Brunei Darussalam Holdings")
        self.cache = read_cache(path)
        self.config = {"model": model, "prompt": PROMPT_TARGET.name, "targets": len(self.cache)}
        self.allow_relabel = version != "2"  # v2: a variant keeps its own label or is dropped
        if version == "2":
            self.config |= {"drop_company_words": sorted(COMPANY_WORDS), "relabel": False,
                            "skip_targets": sorted(SHARED_ISLAND)}  # fmt: skip

    def bases(self) -> list[BaseAlias]:
        return [
            BaseAlias(target_name(t), (t,), "llm", "en")
            for t in sorted(self.cache)
            if t != JUNK_KEY
        ]

    def generate(self, row: BaseAlias, rng: np.random.Generator) -> list[str]:
        if self.version == "2" and SHARED_ISLAND & set(row.targets):
            return []
        # Latin script with at least two letters: auditable by the owner; bare numbers are noise
        return [
            v
            for v in self.cache.get(row.targets[0], [])
            if v != row.alias
            and is_latin(v)
            and sum(c.isalpha() for c in v) >= 2
            and not (self.version == "2" and (_company_like(v) or _foreign_code(v, row.targets)))
        ]


class LlmJunk:
    name = "llmjunk"
    version = "1"
    near_check = False  # NONE by construction; only exact collisions with a place matter

    def __init__(self, path: Path = CACHE_PATH, model: str = MODEL) -> None:
        self.cache = read_cache(path)
        self.config = {"model": model, "prompt": PROMPT_JUNK.name}

    def bases(self) -> list[BaseAlias]:
        return [BaseAlias(JUNK_KEY, (labels.NONE,), "junk", "zxx")]

    def generate(self, row: BaseAlias, rng: np.random.Generator) -> list[str]:
        return list(self.cache.get(JUNK_KEY, []))
