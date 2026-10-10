"""Admin regions → country (plan-09 Task 2): "Ontario" → CA, "Bali" → ID, "Azores" → PT.

Source: GeoNames admin1CodesASCII (CC BY 4.0), the name and its ASCII form, as written and in
capitals. US admin1 are states (targets of their own) and are skipped. A name shared by up to
three places — admin1 regions of other countries, cities ≥ 15k, US counties (`place_index`) — is
the set of their codes (owner place policy; the own-register rule applies through
`place_sources`); a name shared more widely ("Central", "Eastern") carries no information and is
dropped. Disputed regions are skipped. Country and state names win through the G1 exact check
("Florida", a department of Uruguay, is US-FL's name). No containment check: a region may hold
another country's name ("New South Wales").
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from countrycode import labels
from countrycode.synth.framework import BaseAlias
from countrycode.synth.geo import GEONAMES_DIR
from countrycode.text import compact_key

# Crimea, Sevastopol, Abkhazia, Transnistria, and the two Moroccan regions in Western Sahara.
DISPUTED = frozenset({"UA.11", "UA.20", "GE.02", "MD.58", "MA.11", "MA.12"})
MAX_SHARED = 3


def _admin1(root: Path) -> list[tuple[str, str, list[str]]]:
    """(admin1 code, country code, name forms) for every non-US, non-disputed region."""
    out = []
    for line in (root / "admin1CodesASCII.txt").read_text(encoding="utf-8").splitlines():
        p = line.split("\t")
        if len(p) < 3:
            continue
        code, cc = p[0], p[0].split(".")[0]
        if cc == "US" or code in DISPUTED or not labels.is_target(cc):
            continue
        out.append((code, cc, [n for n in dict.fromkeys([p[1], p[2]]) if n]))
    return out


class RegionNames:
    name = "region"
    version = "1"
    near_check = False  # region names are not variants of lexicon names
    containment_check = False

    def __init__(self, root: Path = GEONAMES_DIR) -> None:
        from countrycode.synth.city import place_index

        self.regions = {code: (cc, names) for code, cc, names in _admin1(root)}
        index: dict[str, set[str]] = defaultdict(set)
        for cc, names in self.regions.values():
            for n in names:
                index[compact_key(n)].add(cc)
        for k, ts in place_index(root).items():
            if k in index:
                index[k] |= ts
        self.index = dict(index)
        self.config = {"source": "GeoNames admin1CodesASCII", "index": "admin1 + place_index",
                       "max_shared": MAX_SHARED, "disputed": sorted(DISPUTED)}  # fmt: skip

    def bases(self) -> list[BaseAlias]:
        return [
            BaseAlias(f"__region_{c}", ("NONE",), "region", "mul") for c in sorted(self.regions)
        ]

    def generate(
        self, row: BaseAlias, rng: np.random.Generator
    ) -> list[tuple[str, tuple[str, ...]]]:
        _, names = self.regions[row.alias.removeprefix("__region_")]
        out = []
        for n in names:
            targets = self.index.get(compact_key(n), set())
            if 0 < len(targets) <= MAX_SHARED:
                ts = tuple(sorted(targets))
                out += [(v, ts) for v in dict.fromkeys([n, n.upper()])]
        return out
