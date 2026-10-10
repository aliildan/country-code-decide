"""US county names → state under the owner's place policy (decisions 2026-10-10): a county name
that exists in one state is that state; a name shared by several states is the set of their
codes (in a register of one of them, that one — applied when examples get their context, see
`train.dataset.expand_place`). Source: GeoNames admin2Codes.txt (CC BY 4.0). Forms per county:
the name ("Adams County"), its upper-case form and the bare name ("Adams"). State and country
names win: a form that is a lexicon key of another target is dropped by the G1 exact check.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from countrycode import labels
from countrycode.synth.city import place_index
from countrycode.synth.framework import BaseAlias
from countrycode.synth.geo import GEONAMES_DIR
from countrycode.text import compact_key

SUFFIXES = (" City and Borough", " Census Area", " Municipality", " Borough", " County", " Parish")


def _rows(root: Path):
    for line in (root / "admin2Codes.txt").read_text(encoding="utf-8").splitlines():
        p = line.split("\t")
        if len(p) < 3 or not p[0].startswith("US."):
            continue
        state = f"US-{p[0].split('.')[1]}"
        if labels.is_target(state):
            yield p[0], p[1], state


def bare(name: str) -> str:
    for s in SUFFIXES:
        if name.endswith(s) and len(name) > len(s):
            return name[: -len(s)]
    return name


def _forms(name: str) -> list[str]:
    return list(dict.fromkeys([name, name.upper(), bare(name)]))


def county_index(root: Path = GEONAMES_DIR) -> dict[str, set[str]]:
    """compact form → every state that has a county of that name."""
    idx: dict[str, set[str]] = defaultdict(set)
    for _, name, state in _rows(root):
        for f in _forms(name):
            if k := compact_key(f):
                idx[k].add(state)
    return dict(idx)


class CountyNames:
    name = "county"
    version = "1"
    near_check = False  # labels come from GeoNames by construction; exact collisions checked

    def __init__(self, root: Path = GEONAMES_DIR) -> None:
        self.index = place_index(root)  # shared with city@2
        self.counties = {code: name for code, name, _ in _rows(root)}
        self.config = {
            "source": "GeoNames admin2Codes (US)",
            "index": "cities15000 + admin2Codes (US)",
            "forms": "name, upper, bare",
        }

    def bases(self) -> list[BaseAlias]:
        return [
            BaseAlias(f"__county_{c}", ("NONE",), "county", "en") for c in sorted(self.counties)
        ]

    def generate(
        self, row: BaseAlias, rng: np.random.Generator
    ) -> list[tuple[str, tuple[str, ...]]]:
        name = self.counties[row.alias.removeprefix("__county_")]
        return [(f, tuple(sorted(self.index[compact_key(f)]))) for f in _forms(name)]
