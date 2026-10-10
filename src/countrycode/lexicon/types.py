"""Shared lexicon types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import yaml

from countrycode.text import compact_key

AliasKind = Literal["name", "code", "demonym", "non_country", "historical"]


@dataclass(frozen=True)
class AliasRow:
    alias: str
    target: str
    kind: AliasKind
    lang: str
    source: str


@dataclass(frozen=True)
class Overrides:
    add: list[AliasRow] = field(default_factory=list)
    remove: frozenset[tuple[str, str]] = frozenset()
    historical: dict[str, list[str]] = field(default_factory=dict)
    extra_items: dict[str, str] = field(default_factory=dict)
    # compact_key → target set kept as genuinely ambiguous, whatever the sources say ("Congo").
    force_sets: dict[str, frozenset[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Overrides:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        add = [
            AliasRow(str(a), str(t), cast(AliasKind, k), str(lang), "overrides")
            for a, t, k, lang in raw.get("add") or []
        ]
        xk = raw.get("xk_alternatives") or {}
        add += [AliasRow(str(c), "XK", "code", "zxx", "overrides") for c in xk.get("codes", [])]
        add += [AliasRow(str(n), "XK", "name", "mul", "overrides") for n in xk.get("names", [])]
        remove = frozenset((compact_key(str(a)), str(t)) for a, t in raw.get("remove") or [])
        historical = {str(k): [str(x) for x in v] for k, v in (raw.get("historical") or {}).items()}
        extra = {str(k): str(v) for k, v in (raw.get("extra_items") or {}).items()}
        force = {
            compact_key(str(k)): frozenset(str(x) for x in v)
            for k, v in (raw.get("force_sets") or {}).items()
        }
        return cls(
            add=add, remove=remove, historical=historical, extra_items=extra, force_sets=force
        )
