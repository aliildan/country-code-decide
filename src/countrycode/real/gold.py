"""Gold labels for real items (plan-02 Task 4, design-01 §4).

Three label routes, never mixed up:
- `reference_list` / `preset`: labels a curated source ships with the item (GBIF);
- `reference_list` / `auto_exact`: exact, context-free and *raw*-unique lexicon hits (no
  resolution rule needed) — a random sample is also queued for the owner;
- `human` / `reviewed_proposal`: the owner read the item (with proposals) and answered.
The owner's answer always wins. Claude never writes answers (CLAUDE.md §2.8).
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz, process

from countrycode import labels
from countrycode.lexicon.build import Lexicon
from countrycode.real.items import RealItem
from countrycode.text import compact_key

OWNER_PATH = Path("data/gold/real-v1/owner.jsonl")
PRESET_PATH = Path("data/interim/real/preset_labels.jsonl")

REVIEW_GUIDE = """Label what a careful KYB analyst would accept as the place this field names.
ISO alpha-2 for countries (PR, VI, GU… are countries); US-XX for US states and DC (in US
jurisdiction/state fields abbreviations like CA, DE mean states). Several codes = ambiguous set
(e.g. CD,CG for "Congo"). NONE = not a country or US state (junk, dates, company names,
continents and large regions like "Europe", "Scandinavia"). A named group of up to four
jurisdictions where a company sits in exactly one of them is their set ("Channel Islands" →
GG,JE; "Caroline Islands" → FM,PW; owner practice 2026-10-10). XK = Kosovo.
Cities and regions (owner policy 2026-10-10): if the name points to ONE country, answer that
country ("Wien" → AT, "Bavaria" → DE, "Scotland" → GB); a US city or county in a US field →
its state if you know it, else US; if the name exists in several countries, answer the set
("London" → GB,CA,US; "Paris" → FR,US) — unless the register is one of them: then that one
("Liverpool" in a GB register → GB; "Adams County" in the Colorado register → US-CO)."""


@dataclass(frozen=True)
class GoldItem:
    item_id: str
    source: str
    field: str
    field_kind: str
    register_country: str
    text: str
    count: int
    split: str
    targets: tuple[str, ...]
    label_source: str
    annotation_mode: str
    # "country": the label source only resolves countries (GBIF), so US-TX matches gold US.
    granularity: str = "full"


@dataclass(frozen=True)
class QueueItem:
    item: RealItem
    split: str
    proposals: tuple[tuple[tuple[str, ...], str], ...] | None  # None = compute when served
    auto: tuple[str, ...] | None  # auto/curated label being audited, if any


def auto_label(item: RealItem, lexicon: Lexicon) -> list[str] | None:
    raw = lexicon.raw_lookup(item.text)
    if len(raw) == 1 and lexicon.lookup(item.text) == raw:
        return sorted(raw)
    return None


class _FuzzIndex:
    """Compact lexicon keys for fuzzy proposals; long keys also get a partial-match pass."""

    def __init__(self, lexicon: Lexicon) -> None:
        self.keys = sorted(lexicon.index_keys())
        self.long_keys = [k for k in self.keys if len(k) >= 5]


_FUZZ_CACHE: dict[int, _FuzzIndex] = {}


def _fuzz_index(lexicon: Lexicon) -> _FuzzIndex:
    if id(lexicon) not in _FUZZ_CACHE:
        _FUZZ_CACHE[id(lexicon)] = _FuzzIndex(lexicon)
    return _FUZZ_CACHE[id(lexicon)]


def fuzzy_candidates(text: str, lexicon: Lexicon, limit: int = 10) -> list[tuple[str, float]]:
    """(compact key, score 0–100): full Levenshtein ratio, plus partial ratio for long keys."""
    q = compact_key(text)
    if not q:
        return []
    idx = _fuzz_index(lexicon)
    scores: dict[str, float] = {}
    for key, score, _ in process.extract(q, idx.keys, scorer=fuzz.ratio, limit=limit):
        scores[key] = max(scores.get(key, 0.0), score)
    if len(q) >= 5:
        for key, score, _ in process.extract(
            q, idx.long_keys, scorer=fuzz.partial_ratio, limit=limit
        ):
            scores[key] = max(scores.get(key, 0.0), score * 0.95)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]


def proposals(item: RealItem, lexicon: Lexicon, k: int = 3) -> list[tuple[tuple[str, ...], str]]:
    out: list[tuple[tuple[str, ...], str]] = []
    exact = lexicon.lookup(item.text)
    if exact:
        out.append((tuple(sorted(exact)), "exact"))
    for key, score in fuzzy_candidates(item.text, lexicon, limit=k * 4):
        t = tuple(sorted(lexicon.lookup(key)))
        if t and all(t != o[0] for o in out):
            out.append((t, f"fuzzy {key!r} {score:.0f}"))
        if len(out) >= k + (1 if exact else 0):
            break
    return out


def read_presets(path: Path = PRESET_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {r["item_id"]: r for r in rows}


def read_owner(path: Path = OWNER_PATH) -> dict[str, dict]:
    if not Path(path).exists():
        return {}
    answers: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            a = json.loads(line)
            answers[a["item_id"]] = a  # last answer wins
    return {k: v for k, v in answers.items() if v.get("targets")}


def validate_targets(targets: Sequence[str]) -> tuple[str, ...]:
    ts = tuple(sorted({t.strip().upper() for t in targets if t.strip()}))
    bad = [t for t in ts if not labels.is_target(t)]
    if not ts or bad or (labels.NONE in ts and len(ts) > 1):
        raise ValueError(f"invalid targets: {bad or ts}")
    return ts


def build_queue(
    items: Sequence[RealItem],
    manifest: Mapping[str, str],
    lexicon: Lexicon,
    presets: Mapping[str, dict],
    audit_auto: int = 100,
    audit_preset: int = 50,
    seed: int = 13,
) -> list[QueueItem]:
    rng = random.Random(seed)
    need, autos, preset_items = [], [], []
    for it in items:
        if it.item_id in presets:
            preset_items.append(it)
        elif (a := auto_label(it, lexicon)) is not None:
            autos.append((it, tuple(a)))
        else:
            need.append(it)
    queue = [
        QueueItem(it, manifest[it.item_id], tuple(proposals(it, lexicon)), None) for it in need
    ]
    for it, a in rng.sample(autos, min(audit_auto, len(autos))):
        queue.append(QueueItem(it, manifest[it.item_id], ((a, "auto label (audit)"),), a))
    for it in rng.sample(preset_items, min(audit_preset, len(preset_items))):
        a = tuple(presets[it.item_id]["targets"])
        queue.append(QueueItem(it, manifest[it.item_id], ((a, "curated label (audit)"),), a))
    order = {"dev": 0, "calib": 0, "test": 1}
    return sorted(queue, key=lambda q: (order[q.split], -q.item.count, q.item.item_id))


def load_gold(
    items: Sequence[RealItem],
    manifest: Mapping[str, str],
    lexicon: Lexicon,
    presets: Mapping[str, dict] | None = None,
    owner_path: Path = OWNER_PATH,
) -> list[GoldItem]:
    presets = presets if presets is not None else read_presets()
    owner = read_owner(owner_path)
    out: list[GoldItem] = []
    for it in items:
        base = (it.item_id, it.source, it.field, it.field_kind, it.register_country, it.text)
        split = manifest[it.item_id]
        if it.item_id in owner:
            ts = validate_targets(owner[it.item_id]["targets"])
            out.append(GoldItem(*base, it.count, split, ts, "human", "reviewed_proposal"))
        elif it.item_id in presets:
            pr = presets[it.item_id]
            ts = tuple(pr["targets"])
            src = pr.get("label_source", "reference_list")
            gran = pr.get("granularity", "country")  # GBIF presets resolve countries only
            out.append(GoldItem(*base, it.count, split, ts, src, "preset", gran))
        elif (a := auto_label(it, lexicon)) is not None:
            out.append(GoldItem(*base, it.count, split, tuple(a), "reference_list", "auto_exact"))
    return out


def gold_summary(gold: Sequence[GoldItem], pending: int) -> dict:
    from collections import Counter

    return {
        "labelled": len(gold),
        "pending_owner": pending,
        "by_mode": dict(Counter(g.annotation_mode for g in gold)),
        "by_source_split": dict(Counter(f"{g.source}/{g.split}" for g in gold)),
        "ambiguous_gold": sum(1 for g in gold if len(g.targets) > 1),
        "none_gold": sum(1 for g in gold if g.targets == (labels.NONE,)),
        "keys": len({compact_key(g.text) for g in gold}),
    }
