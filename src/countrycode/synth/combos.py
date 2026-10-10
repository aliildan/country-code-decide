"""Combination generators (plan-03, added after the dev error analysis of small-synth):

- `pairs@1`: two countries in one string ("Austria/Germany", "INDIA NEPAL") → the set of both
  (the string does not identify one country; the model should abstain or return the set).
- `affix@1`: a country with a compass or region qualifier ("NORTHERN KENYA", "AUSTRALIA N W",
  "EAST AFRICA KENYA") → that country. A qualifier that forms another country's name
  ("SOUTH SUDAN") is caught by the G1 exact check.
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from countrycode import labels
from countrycode.lexicon.build import Lexicon
from countrycode.synth.framework import PRIORITY_LANGS, SHARED_ISLAND, BaseAlias, is_latin


def _country_names(lexicon: Lexicon) -> dict[str, list[str]]:
    names: dict[str, set[str]] = defaultdict(set)
    for r in lexicon.rows:
        if (
            r.kind == "name"
            and not r.target.startswith("US-")
            and r.target != labels.NONE
            and r.lang.split("_")[0] in PRIORITY_LANGS
            and is_latin(r.alias)
            and lexicon.lookup(r.alias) == {r.target}
        ):
            names[r.target].add(r.alias)
    return {t: sorted(v) for t, v in names.items()}


COMPASS = frozenset({"north", "south", "east", "west", "northern", "southern", "eastern",
                     "western", "central", "upper", "lower"})  # fmt: skip


def _words(s: str) -> list[str]:
    return [w for w in re.split(r"[^\w]+", s.lower()) if w]


def _english_cldr_names(lexicon: Lexicon) -> dict[str, list[str]]:
    """v2+ (plan-08): only English CLDR names, unique in the lexicon, no shared-island codes."""
    names: dict[str, set[str]] = defaultdict(set)
    for r in lexicon.rows:
        if (
            r.kind == "name"
            and r.source == "cldr"
            and r.lang.split("_")[0] == "en"
            and not r.target.startswith("US-")
            and r.target not in (labels.NONE, *SHARED_ISLAND)
            and is_latin(r.alias)
            and lexicon.lookup(r.alias) == {r.target}
        ):
            names[r.target].add(r.alias)
    return {t: sorted(v) for t, v in names.items()}


def nested_targets(lexicon: Lexicon, names: dict[str, list[str]]) -> set[str]:
    """Targets a compass qualifier could turn into another place: a name with a compass word
    ("South Africa"), or a name found as whole words inside another target's alias ("Ireland" in
    "Northern Ireland", "Sudan" in "South Sudan")."""
    seqs: dict[str, str] = {}
    out: set[str] = set()
    for t, ns in names.items():
        for n in ns:
            w = _words(n)
            if COMPASS & set(w):
                out.add(t)
            seqs[" ".join(w)] = t
    longest = max((len(k.split()) for k in seqs), default=0)
    for r in lexicon.rows:
        w = _words(r.alias)
        for n in range(1, min(longest, len(w) - 1) + 1):  # strictly inside a longer alias
            for i in range(len(w) - n + 1):
                t = seqs.get(" ".join(w[i : i + n]))
                if t is not None and t != r.target:
                    out.add(t)
    return out


SEPARATORS = (" ", ", ", " / ", "/", " - ", " & ", " and ", " oder ", " ou ", " og ", " ve ")


class CountryPairs:
    name = "pairs"
    version = "1"
    near_check = False  # labels are the two named countries; exact collisions still checked

    def __init__(self, lexicon: Lexicon, n: int = 4000, version: str = "1") -> None:
        self.version = version  # 2 (v1.1): n = 12,000; 3 (plan-08): English CLDR names
        self.names = _english_cldr_names(lexicon) if version == "3" else _country_names(lexicon)
        self.codes = sorted(self.names)
        self.n = n
        self.config: dict[str, int | str] = {"n": n, "separators": len(SEPARATORS)}
        if version == "3":
            self.config["names"] = "english_cldr"

    def bases(self) -> list[BaseAlias]:
        return [BaseAlias(f"__pair_{i}", ("NONE",), "pair", "zxx") for i in range(self.n)]

    def generate(
        self, row: BaseAlias, rng: np.random.Generator
    ) -> list[tuple[str, tuple[str, ...]]]:
        a, b = (str(x) for x in rng.choice(self.codes, size=2, replace=False))
        s = f"{rng.choice(self.names[a])}{rng.choice(SEPARATORS)}{rng.choice(self.names[b])}"
        if rng.random() < 0.5:
            s = s.upper()
        return [(s, tuple(sorted((a, b))))]


PREFIXES = ("North", "South", "East", "West", "N", "S", "E", "W", "Northern", "Southern",
            "Eastern", "Western", "Central", "Upper", "Lower", "Mainland", "Nord", "Süd", "Ost",
            "West-", "Northwest", "Northeast", "Southwest", "Southeast")  # fmt: skip
SUFFIXES = ("N", "S", "E", "W", "NE", "NW", "SE", "SW", "N W", "S E", "mainland", "north",
            "south", "east", "west", "interior", "coast")  # fmt: skip
REGIONS = ("Africa", "East Africa", "West Africa", "Southern Africa", "Asia", "SE Asia",
           "Europe", "W Europe", "South America", "S America", "Central America", "West Indies",
           "Oceania", "Middle East", "Scandinavia", "Balkans", "Caribbean")  # fmt: skip


class Affixes:
    name = "affix"
    version = "1"

    def __init__(self, lexicon: Lexicon, per_country: int = 8, version: str = "1") -> None:
        self.version = version  # 2 (plan-08): English CLDR names; no nested or compass targets
        self.names = _country_names(lexicon)
        self.per_country = per_country
        self.config = {"per_country": per_country, "prefixes": len(PREFIXES),
                       "suffixes": len(SUFFIXES), "regions": len(REGIONS)}  # fmt: skip
        if version == "2":
            names = _english_cldr_names(lexicon)
            skip = nested_targets(lexicon, names)
            self.names = {t: ns for t, ns in names.items() if t not in skip}
            self.config |= {"names": "english_cldr", "skip_nested": True}

    def bases(self) -> list[BaseAlias]:
        return [BaseAlias(n[0], (t,), "affix", "en") for t, n in sorted(self.names.items())]

    def generate(self, row: BaseAlias, rng: np.random.Generator) -> list[str]:
        names = self.names[row.targets[0]]
        out: list[str] = []
        for _ in range(self.per_country * 3):
            if len(out) >= self.per_country:
                break
            name = str(rng.choice(names))
            kind = int(rng.integers(0, 3))
            if kind == 0:
                s = f"{rng.choice(PREFIXES)} {name}"
            elif kind == 1:
                s = f"{name} {rng.choice(SUFFIXES)}"
            else:
                region = str(rng.choice(REGIONS))
                s = f"{region} {name}" if rng.random() < 0.5 else f"{name}, {region}"
            if rng.random() < 0.6:
                s = s.upper()
            if s not in out:
                out.append(s)
        return out
