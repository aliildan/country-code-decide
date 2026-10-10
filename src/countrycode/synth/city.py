"""City names → country (or US state) under the owner's place policy (decisions 2026-10-10):
a name that points to one place is that country (a US city: its state); a name shared by several
places — cities ≥ 15k and US counties (`place_index`) — is the set of their codes; in a register of
one of them it is that one (applied when examples get their context, `train.dataset.expand_place`).
Source: GeoNames cities15000 + admin2Codes (CC BY 4.0) — the local name, its ASCII form and a few
Latin-script alternate names per city ≥ 50k. Country names win: a city name that is a lexicon key
of another target is dropped by the G1 exact check.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from countrycode import labels
from countrycode.synth.framework import BaseAlias, is_latin
from countrycode.synth.geo import GEONAMES_DIR
from countrycode.text import compact_key


def _target(cc: str, admin1: str) -> str | None:
    if cc == "US":
        t = f"US-{admin1}"
        return t if labels.is_target(t) else "US"
    return cc if labels.is_target(cc) else None


def _rows(root: Path):
    for line in (root / "cities15000.txt").read_text(encoding="utf-8").splitlines():
        p = line.split("\t")
        if len(p) >= 15:
            yield p


def _names(p: list[str]) -> list[str]:
    # real name forms only: Latin script, starting with a capital ("München", "Gladbėki");
    # lowercase romanisations of other scripts ("mwnykh") are not something a clerk writes
    alts = [a for a in p[3].split(",") if len(a) >= 3 and is_latin(a) and a[0].isupper()]
    return list(dict.fromkeys([p[1], p[2], *alts]))


def city_index(root: Path = GEONAMES_DIR) -> dict[str, set[str]]:
    """compact name → every target that has a city of that name."""
    idx: dict[str, set[str]] = defaultdict(set)
    for p in _rows(root):
        t = _target(p[8], p[10])
        if t is None:
            continue
        for n in _names(p):
            k = compact_key(n)
            if k:
                idx[k].add(t)
    return dict(idx)


def place_index(root: Path = GEONAMES_DIR) -> dict[str, set[str]]:
    """City ∪ US county names → every target that has a place of that name, so one name gets one
    set whichever generator emits it ("Geneva" → CH, US-AL, US-IL)."""
    from countrycode.synth.county import county_index

    parts = []
    if (root / "cities15000.txt").exists():
        parts.append(city_index(root))
    if (root / "admin2Codes.txt").exists():
        parts.append(county_index(root))
    idx: dict[str, set[str]] = defaultdict(set)
    for part in parts:
        for k, v in part.items():
            idx[k] |= v
    return dict(idx)


class CityNames:
    name = "city"
    version = "2"  # 2: shared place index with US counties
    near_check = False  # labels come from GeoNames by construction; exact collisions checked

    def __init__(
        self, root: Path = GEONAMES_DIR, alternates: int = 3, min_population: int = 50_000
    ) -> None:
        self.root = root
        self.alternates = alternates
        self.index = place_index(root)  # every city ≥ 15k and every US county
        self.cities = {p[0]: p for p in _rows(root) if int(p[14] or 0) >= min_population}
        self.config = {
            "source": "GeoNames cities15000",
            "index": "cities15000 + admin2Codes (US)",
            "alternates": alternates,
            "min_population": min_population,
        }

    def bases(self) -> list[BaseAlias]:
        return [BaseAlias(f"__city_{gid}", ("NONE",), "city", "mul") for gid in sorted(self.cities)]

    def generate(
        self, row: BaseAlias, rng: np.random.Generator
    ) -> list[tuple[str, tuple[str, ...]]]:
        p = self.cities[row.alias.removeprefix("__city_")]
        names = _names(p)
        head, alts = names[:2], names[2:]
        if alts:
            k = min(self.alternates, len(alts))
            alts = [alts[int(i)] for i in rng.choice(len(alts), size=k, replace=False)]
        out = []
        for n in dict.fromkeys([*head, *alts]):
            targets = self.index.get(compact_key(n))
            if targets:
                out.append((n, tuple(sorted(targets))))
        return out
