"""Place-context generator `geo@1`: a country (or US state) written together with one of its
regions or cities, as in address-like fields and curated variant lists ("ALAGOAS BRAZIL",
"Innsbruck, Österreich", "Austin, TX"). Region and city names come from GeoNames
(CC BY 4.0: admin1CodesASCII, cities15000). The label is certain by construction because the
country or state name is written in the string, so only exact collisions are checked (G1).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from countrycode import labels
from countrycode.lexicon.build import Lexicon
from countrycode.synth.framework import PRIORITY_LANGS, BaseAlias, base_info, is_latin

GEONAMES_DIR = Path("data/raw/geonames")


def load_geonames(root: Path = GEONAMES_DIR) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """regions[cc] = admin1 names; cities[cc or US-XX] = city names by population (desc)."""
    regions: dict[str, list[str]] = defaultdict(list)
    for line in (root / "admin1CodesASCII.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            cc = parts[0].split(".")[0]
            regions[cc].append(parts[2] or parts[1])
    rows: list[tuple[str, str, int]] = []
    for line in (root / "cities15000.txt").read_text(encoding="utf-8").splitlines():
        p = line.split("\t")
        if len(p) < 15:
            continue
        cc, admin1, name, pop = p[8], p[10], p[2] or p[1], int(p[14] or 0)
        key = f"US-{admin1}" if cc == "US" and labels.is_target(f"US-{admin1}") else cc
        rows.append((key, name, pop))
    cities: dict[str, list[str]] = defaultdict(list)
    for key, name, _ in sorted(rows, key=lambda r: -r[2]):
        cities[key].append(name)
    return dict(regions), dict(cities)


# Moroccan admin1 regions in Western Sahara: a place there is EH or MA (owner audit, plan-08).
DISPUTED_REGIONS = {"MA": frozenset({"Laayoune-Sakia El Hamra", "Dakhla-Oued Ed-Dahab"})}


class GeoContext:
    name = "geo"
    version = "1"
    near_check = False  # the country/state name is part of the string

    def __init__(
        self, lexicon: Lexicon, root: Path = GEONAMES_DIR, per_target: int = 25, version: str = "1"
    ) -> None:
        self.version = version  # 2 (plan-08): no disputed regions; CLDR or anchored names only
        self.regions, self.cities = load_geonames(root)
        self.per_target = per_target
        self.config = {"source": "GeoNames admin1 + cities15000", "per_target": per_target}
        anchored = None
        if version == "2":
            self.regions = {
                cc: [r for r in rs if r not in DISPUTED_REGIONS.get(cc, ())]
                for cc, rs in self.regions.items()
            }
            anchored = base_info(lexicon)  # no nicknames ("Lone Star State") or obscure aliases
            self.config |= {"skip_disputed": True, "names": "cldr_or_anchored+alpha3",
                            "containment": False}  # fmt: skip
            # a GeoNames place of the own target may hold another country's name ("West Jordan")
            self.containment_check = False
        names: dict[str, set[str]] = defaultdict(set)
        for r in lexicon.rows:
            if (
                anchored is not None
                and r.kind == "code"
                and r.source == "wikidata"  # ISO alpha-3 (P298), not override codes like "UVK"
                and len(r.alias) == 3
                and r.alias.isupper()
                and lexicon.lookup(r.alias) == {r.target}
            ):
                names[r.target].add(r.alias)  # alpha-3 ("Alberta, CAN"); not "AUS" (AU or US)
            if (
                r.kind == "name"
                and r.lang.split("_")[0] in PRIORITY_LANGS
                and is_latin(r.alias)
                and lexicon.lookup(r.alias) == {r.target}
                and (anchored is None or r.source == "cldr" or anchored[r.alias].anchored)
            ):
                names[r.target].add(r.alias)
        self.names = {t: sorted(v) for t, v in names.items()}

    def bases(self) -> list[BaseAlias]:
        out = []
        for t in (*labels.countries(), *labels.us_states()):
            if t != "US" and self.names.get(t):
                out.append(BaseAlias(f"__geo_{t}", (t,), "geo", "en"))
        return out

    def generate(self, row: BaseAlias, rng: np.random.Generator) -> list[str]:
        t = row.targets[0]
        own = self.names[t]
        places = [*(self.cities.get(t, [])[:20])]
        if not t.startswith("US-"):
            places += self.regions.get(t, [])
        places = [p for p in places if is_latin(p)]
        if not places:
            return []
        state_codes = [labels.usps(t)] if t.startswith("US-") else []
        out: list[str] = []
        for _ in range(self.per_target * 3):
            if len(out) >= self.per_target:
                break
            place = str(rng.choice(places))
            name = str(rng.choice(own + state_codes)) if state_codes else str(rng.choice(own))
            pattern = int(rng.integers(0, 5))
            s = [f"{place} {name}", f"{place}, {name}", f"{name} ({place})", f"{name}, {place}",
                 f"{place} - {name}"][pattern]  # fmt: skip
            if t.startswith("US-") and rng.random() < 0.3:
                s = f"{s}, USA"
            if rng.random() < 0.5:
                s = s.upper()
            if s not in out:
                out.append(s)
        return out
