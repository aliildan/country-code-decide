"""Lexicon build and lookup (plan-01 Task 8, design-01 §2/§6).

A key (`compact_key` of an alias) maps to a *set* of targets, so ambiguity is explicit:
"CA" → {CA, US-CA}, "Georgia" → {GE, US-GA}, "Yugoslavia" → its successor states.

Resolution rules per key, in order (found in the ambiguity review, plan-01 Task 8):
1. codes first — if the key is a code (alpha-2, USPS, reserved, Kosovo codes), only code rows count
   ("SK" is Slovakia even though Wikidata lists "SK" as an alias of South Korea);
2. current before historical — historical rows drop out when a current name has the key
   ("Serbia" is RS, not the Yugoslav successor set);
3. weak support drops out — a target backed by < 10 % of the strongest target's (source, language)
   support, or by a single row against ≥ 3, is removed ("China" → CN, not TW; "Jersey" → JE);
4. reviewed force sets win ("Congo" → {CD, CG}).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import cast

from countrycode import labels
from countrycode.lexicon.types import AliasKind, AliasRow, Overrides
from countrycode.text import compact_key, match_key

LEXICON_PATH = Path("data/processed/lexicon/lexicon-v1.json")
STATS_PATH = Path("results/lexicon/stats.json")


def code_rows() -> list[AliasRow]:
    rows = [AliasRow(c, c, "code", "zxx", "codes") for c in labels.countries()]
    rows += [AliasRow(labels.usps(s), s, "code", "zxx", "codes") for s in labels.us_states()]
    rows += [AliasRow(s, s, "code", "zxx", "codes") for s in labels.us_states()]  # "US-CA"
    rows += [AliasRow(r, t, "code", "zxx", "codes") for r, t in labels.RESERVED_TEXT_CODES.items()]
    rows += [
        AliasRow(c, labels.NONE, "non_country", "zxx", "codes") for c in labels.NONE_TEXT_CODES
    ]
    return rows


CODE_SOURCES = frozenset({"codes", "overrides"})
SHORT_KEY = 3
WEAK_SHARE = 0.1


def _resolve(rows: list[AliasRow]) -> frozenset[str]:
    codes = [r for r in rows if r.kind == "code" and r.source in CODE_SOURCES]
    if codes:
        return frozenset(r.target for r in codes)
    current = [r for r in rows if r.kind != "historical"]
    if current and len({r.target for r in current}) < len({r.target for r in rows}):
        rows = current
    support: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for r in rows:
        support[r.target].add((r.source, r.lang))
    top = max(len(v) for v in support.values())
    return frozenset(
        t
        for t, v in support.items()
        if not (len(v) < WEAK_SHARE * top or (len(v) == 1 and top >= 3))
    )


class Lexicon:
    def __init__(
        self,
        rows: Sequence[AliasRow],
        meta: dict | None = None,
        force_sets: dict[str, frozenset[str]] | None = None,
    ) -> None:
        self.rows: tuple[AliasRow, ...] = tuple(rows)
        self.meta = meta or {}
        self.force_sets = dict(force_sets or {})
        by_key: dict[str, list[AliasRow]] = defaultdict(list)
        short_forms: dict[str, set[str]] = defaultdict(set)
        for r in self.rows:
            key = compact_key(r.alias)
            if key:
                by_key[key].append(r)
                if len(key) <= SHORT_KEY:
                    short_forms[key].add(match_key(r.alias))
        # "N/A" and "NA" share the compact key "na": short keys must also match a spaced form
        self._short_forms = dict(short_forms)
        self._raw_targets = {k: frozenset(r.target for r in v) for k, v in by_key.items()}
        self._index = {k: _resolve(v) for k, v in by_key.items()}
        self._index.update(self.force_sets)

    def _key(self, text: str) -> str:
        key = compact_key(text)
        short = len(key) <= SHORT_KEY and key in self._short_forms
        return "" if short and match_key(text) not in self._short_forms[key] else key

    def lookup(self, text: str) -> frozenset[str]:
        return self._index.get(self._key(text), frozenset())

    def raw_lookup(self, text: str) -> frozenset[str]:
        """Targets before the resolution rules (every source's opinion)."""
        key = self._key(text)
        return self._raw_targets.get(key, frozenset()) | self.force_sets.get(key, frozenset())

    def is_unique(self, text: str) -> bool:
        return len(self.lookup(text)) == 1

    def index_keys(self) -> list[str]:
        return list(self._index)

    def keys(self) -> int:
        return len(self._index)

    def ambiguous(self) -> dict[str, frozenset[str]]:
        return {k: v for k, v in self._index.items() if len(v) > 1}

    def stats(self) -> dict:
        return {
            "rows": len(self.rows),
            "keys": self.keys(),
            "ambiguous_keys": len(self.ambiguous()),
            "keys_resolved_by_rules": sum(
                1 for k, v in self._index.items() if v != self._raw_targets.get(k, v)
            ),
            "force_sets": len(self.force_sets),
            "by_source": dict(Counter(r.source for r in self.rows)),
            "by_kind": dict(Counter(r.kind for r in self.rows)),
            "languages": len({r.lang for r in self.rows}),
            "targets_covered": len({r.target for r in self.rows}),
            **{k: v for k, v in self.meta.items() if k != "rows"},
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            **self.meta,
            "force_sets": {k: sorted(v) for k, v in sorted(self.force_sets.items())},
            "rows": [[r.alias, r.target, r.kind, r.lang, r.source] for r in self.rows],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Lexicon:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = [AliasRow(a, t, cast(AliasKind, k), lang, s) for a, t, k, lang, s in data["rows"]]
        meta = {k: v for k, v in data.items() if k not in ("rows", "version", "force_sets")}
        force = {k: frozenset(v) for k, v in data.get("force_sets", {}).items()}
        return cls(rows, meta, force_sets=force)


def build_lexicon(
    rows: Iterable[AliasRow], overrides: Overrides, meta: dict | None = None
) -> Lexicon:
    kept = {
        r
        for r in rows
        if (compact_key(r.alias), r.target) not in overrides.remove and labels.is_target(r.target)
    }
    kept |= {r for r in overrides.add if labels.is_target(r.target)}
    rows_sorted = sorted(kept, key=lambda r: (r.target, r.lang, r.alias, r.kind))
    return Lexicon(rows_sorted, meta, force_sets=overrides.force_sets)
