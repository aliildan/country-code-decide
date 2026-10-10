"""Training examples (plan-04 Task 1, design-01 §3/§6).

A training string with a target set becomes one or more `TrainExample`s: a field context (field,
register country) plus soft label distributions for the two heads — `country` over 250 countries
+ NONE, and `us_state` over 51 states + NO_STATE (only when the country label puts mass on US).
Real strings never enter training in v1; from v1.1 the train-role register strings do, with
their own context (field, register — e.g. "US-CO" for the Colorado register).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

import numpy as np
import yaml

from countrycode import labels
from countrycode.paths import CONFIGS
from countrycode.text import compact_key

RULES_PATH = CONFIGS / "context_rules.yaml"
NO_STATE = "NO_STATE"
COUNTRY_CLASSES: tuple[str, ...] = (*labels.countries(), labels.NONE)
STATE_CLASSES: tuple[str, ...] = (*labels.us_states(), NO_STATE)
DATASET_DIR = Path("data/processed/train")


@dataclass(frozen=True)
class TrainExample:
    text: str
    field: str
    register: str
    country: dict[str, float]
    state: dict[str, float] | None
    source: str  # "lexicon" or "synthetic:<gen>@<ver>"


def context_text(field: str, register: str) -> str:
    return f"field: {field} | register: {register}"


@cache
def rules() -> dict:
    return yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))


def label_dists(targets: Sequence[str]) -> tuple[dict[str, float], dict[str, float] | None]:
    w = 1.0 / len(targets)
    country: dict[str, float] = {}
    state: dict[str, float] = {}
    for t in targets:
        c = labels.country_of(t)
        country[c] = country.get(c, 0.0) + w
        if c == "US":
            k = t if t.startswith("US-") else NO_STATE
            state[k] = state.get(k, 0.0) + w
    us = country.get("US", 0.0)
    state_dist = {k: v / us for k, v in state.items()} if us > 0 else None
    return dict(country), state_dist


def _register(reg: str, rng: np.random.Generator) -> str:
    if reg == "US":
        share = float(rules().get("us_state_register_share", 0.0))
        return str(rng.choice(labels.us_states())) if rng.random() < share else reg
    if reg != "XX":
        return reg
    return str(rng.choice([r for r in rules()["register_countries"] if r != "US"]))


def _sample_context(rng: np.random.Generator) -> tuple[str, str]:
    ctx = rules()["contexts"]
    w = np.array([c[2] for c in ctx], dtype=float)
    field, reg, _ = ctx[int(rng.choice(len(ctx), p=w / w.sum()))]
    return field, _register(reg, rng)


def _context_override(text: str, field: str, reg: str) -> tuple[str, ...] | None:
    """A code that names a foreign subdivision in a US jurisdiction field ("ON", "NL" → Canada;
    owner labelling policy, `us_jurisdiction_codes` in context_rules.yaml)."""
    if field != "jurisdiction" or not (reg == "US" or reg.startswith("US-")):
        return None
    key = compact_key(text).upper()
    for country, codes in rules().get("us_jurisdiction_codes", {}).items():
        if key in codes:
            return (country,)
    return None


def _collision(targets: Sequence[str]) -> tuple[list[str], list[str]] | None:
    states = [t for t in targets if t.startswith("US-")]
    countries = [t for t in targets if not t.startswith("US-") and t != labels.NONE]
    return (countries, states) if states and countries and "US" not in countries else None


def expand(
    text: str,
    targets: Sequence[str],
    source: str,
    rng: np.random.Generator,
    place: bool = False,
) -> list[TrainExample]:
    """`place`: a register that is one of the targets takes that target (place-name policy)."""

    def example(field: str, reg: str, subset: Sequence[str]) -> TrainExample:
        chosen = (reg,) if place and reg in targets else subset
        c, s = label_dists(_context_override(text, field, reg) or chosen)
        return TrainExample(text, field, reg, c, s, source)

    col = _collision(targets)
    if col is None:
        return [example(*_sample_context(rng), targets)]
    countries, states = col
    r = rules()["us_collision"]
    out = []
    for subset, contexts in ((states, r["state"]), (countries, r["country"]), (targets, r["soft"])):
        for ctx in contexts:
            field, reg, times = ctx[0], ctx[1], int(ctx[2]) if len(ctx) > 2 else 1
            for _ in range(times):
                out.append(example(field, _register(reg, rng), subset))
    return out


def expand_place(
    text: str, targets: Sequence[str], source: str, rng: np.random.Generator
) -> list[TrainExample]:
    """Place names: the sampled context (own-register rule), and for a shared name also the
    context-free set and one example per own register (country field / US jurisdiction field)."""
    out = expand(text, targets, source, rng, place=True)
    if len(targets) > 1:
        c, s = label_dists(targets)
        out.append(TrainExample(text, "unknown", "unknown", c, s, source))
        for t in targets:
            c, s = label_dists((t,))
            field = "jurisdiction" if t.startswith("US-") else "country"
            out.append(TrainExample(text, field, t, c, s, source))
    return out


def _seed(seed: int, text: str, source: str) -> int:
    h = hashlib.sha256(f"{seed}\x1f{source}\x1f{text}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def expand_all_contexts(text: str, targets: Sequence[str], source: str, rng) -> list[TrainExample]:
    """Codes are short and frequent in real fields: one example per configured context."""
    if _collision(targets) is not None:
        return expand(text, targets, source, rng)
    out = []
    for field, reg, _ in rules()["contexts"]:
        r = _register(reg, rng)
        c, s = label_dists(_context_override(text, field, r) or targets)
        out.append(TrainExample(text, field, r, c, s, source))
    return out


def build_examples(items: Iterable[tuple], seed: int = 13) -> list[TrainExample]:
    """items: (text, targets, source) — context sampled from (seed, source, text); sources ending
    in ":code" get every context — or (text, targets, source, field, register) for real strings
    that carry their own context."""
    out: list[TrainExample] = []
    for it in items:
        text, targets, source = it[0], tuple(it[1]), it[2]
        if len(it) == 5:
            c, s = label_dists(targets)
            out.append(TrainExample(text, it[3], it[4], c, s, source))
            continue
        rng = np.random.default_rng(_seed(seed, text, source))
        fixed = rules().get("source_contexts", {}).get(source.split("@")[0])
        if fixed:  # context-bound strings ("ON" is Canada only in a US jurisdiction field)
            for ctx in fixed:
                for _ in range(int(ctx[2]) if len(ctx) > 2 else 1):
                    r = _register(ctx[1], rng)
                    c, s = label_dists(_context_override(text, ctx[0], r) or targets)
                    out.append(TrainExample(text, ctx[0], r, c, s, source))
        elif source.endswith(":code"):
            out += expand_all_contexts(text, targets, source, rng)
        elif source in rules().get("place_sources", ()):
            out += expand_place(text, targets, source, rng)
        else:
            out += expand(text, targets, source, rng)
    return out


def write_dataset(examples: Sequence[TrainExample], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with path.open("w", encoding="utf-8") as f:
        for e in examples:
            line = json.dumps(asdict(e), ensure_ascii=False, sort_keys=True) + "\n"
            h.update(line.encode())
            f.write(line)
    return h.hexdigest()


def read_dataset(path: Path) -> list[TrainExample]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(TrainExample(**json.loads(line)))
    return out


def data_check(
    examples: Sequence[TrainExample], real_keys: set[str], eval_keys: set[str] | None = None
) -> dict:
    """G4 for training data: no synthetic string equals a real string; no real training string
    equals a string of an evaluation role (select/calib/test/ood, `eval_keys`); labels valid."""
    from countrycode.text import compact_key

    syn_real = sum(
        1
        for e in examples
        if e.source.startswith("synthetic:") and compact_key(e.text) in real_keys
    )
    lex_real = sum(
        1 for e in examples if e.source == "lexicon" and compact_key(e.text) in real_keys
    )
    real_eval = sum(
        1
        for e in examples
        if e.source.startswith("real:") and compact_key(e.text) in (eval_keys or set())
    )
    bad_labels = sum(
        1
        for e in examples
        if any(k not in COUNTRY_CLASSES for k in e.country)
        or (e.state is not None and any(k not in STATE_CLASSES for k in e.state))
        or abs(sum(e.country.values()) - 1) > 1e-6
    )
    return {
        "examples": len(examples),
        "by_source": dict(Counter(e.source for e in examples)),
        "synthetic_equal_to_real": syn_real,
        "lexicon_equal_to_real": lex_real,
        "real_train_in_eval": real_eval,
        "bad_labels": bad_labels,
        "ok": syn_real == 0 and real_eval == 0 and bad_labels == 0,
    }
