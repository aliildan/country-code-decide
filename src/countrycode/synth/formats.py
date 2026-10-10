"""Format, transliteration and junk generators (plan-03 Tasks 4–5).

- `format@1`: case, spacing, comma inversion ("Korea, Republic of"), postcode noise
  ("10250 THAILAND"), field-length truncation (30 characters) and abbreviation-style prefixes
  ("BERM", "AUST") labelled with *every* target whose name starts that way ({AT, AU}).
- `translit@1`: romanisations (anyascii) and the German/Nordic umlaut conventions.
- `junk@1`: strings that are not a place (dates, numbers, placeholders) → NONE.
"""

from __future__ import annotations

import re

import numpy as np
from anyascii import anyascii

from countrycode import labels
from countrycode.lexicon.build import Lexicon
from countrycode.synth.framework import (
    PRIORITY_LANGS,
    BaseAlias,
    BaseInfo,
    is_latin,
    seedable,
    unsafe_reason,
)
from countrycode.text import compact_key

Variant = str | tuple[str, tuple[str, ...]]


def _lang(lang: str) -> str:
    return lang.split("_")[0]


class FormatVariants:
    name = "format"
    version = "3"  # 2: case variants of codes ("tx", "Tx"); 3: every code alias gets them

    def __init__(
        self,
        lexicon: Lexicon,
        zip_prob: float = 0.3,
        max_field: int = 30,
        info: dict[str, BaseInfo] | None = None,
    ) -> None:
        self.zip_prob = zip_prob
        self.max_field = max_field
        self.config = {"zip_prob": zip_prob, "max_field": max_field, "prefix_len": [3, 5]}
        self.info = info  # v4 (plan-08): seedable names only; no prefix truncations
        self.prefix_lens: tuple[int, ...] = (3, 4, 5)
        if info is not None:
            self.version = "4"
            self.prefix_lens = ()  # "SAHRA" ← Sahrawi, "ROMAN" ← Romanian: truncations make words
            self.config = {**self.config, "prefix_len": [], "bases": "seedable"}
        prefixes: dict[str, set[str]] = {}
        for r in lexicon.rows:
            if r.kind not in ("name", "historical") or _lang(r.lang) not in PRIORITY_LANGS:
                continue
            key = compact_key(r.alias)
            for n in (3, 4, 5):
                if len(key) > n:
                    prefixes.setdefault(key[:n], set()).update(lexicon.lookup(key))
        self.prefixes = prefixes

    def generate(self, row: BaseAlias, rng: np.random.Generator) -> list[Variant]:
        a = row.alias
        if row.kind == "code":
            return (
                [v for v in dict.fromkeys([a.lower(), a.title()]) if v != a] if is_latin(a) else []
            )
        if not is_latin(a):
            return []
        if self.info is not None and not seedable(row, self.info):
            return []
        out: list[Variant] = [a.upper(), a.lower(), a.title()]
        if " " in a:
            out += [a.replace(" ", ""), a.replace(" ", "-")]
        if "-" in a:
            out.append(a.replace("-", " "))
        if " of " in a and "," not in a:
            head, tail = a.split(" of ", 1)
            out.append(f"{tail}, {head} of")
        if ", " in a and a.rstrip().endswith(" of"):
            tail, head = a.split(", ", 1)
            out.append(f"{head} {tail}")
        if rng.random() < self.zip_prob:
            z = "".join(str(d) for d in rng.integers(0, 10, int(rng.integers(4, 6))))
            out.append(f"{z} {a.upper()}" if rng.random() < 0.5 else f"{a.upper()} {z}")
        if len(a) > self.max_field:
            out.append(a[: self.max_field])
        if is_latin(a) and _lang(row.lang) in PRIORITY_LANGS and len(compact_key(a)) >= 6:
            key = compact_key(a)
            for n in self.prefix_lens:
                ts = self.prefixes.get(key[:n], set()) - {labels.NONE}
                if ts and set(row.targets) & ts:
                    out.append((key[:n].upper(), tuple(sorted(ts))))
        seen: set[str] = set()
        uniq: list[Variant] = []
        for v in out:
            t = v if isinstance(v, str) else v[0]
            if t != a and t not in seen:
                seen.add(t)
                uniq.append(v)
        return uniq


UMLAUT_DE = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue",
                           "ß": "ss"})  # fmt: skip
NORDIC = str.maketrans({"ø": "oe", "Ø": "Oe", "å": "aa", "Å": "Aa", "æ": "ae", "Æ": "Ae"})


# Scripts whose names are romanised in registers (transliteration of Russian, Arabic, Chinese …);
# rarer scripts and Wikidata aliases produced unrecognisable or wrong strings (owner audit 2).
TRANSLIT_LANGS = frozenset({"ru", "uk", "bg", "sr", "mk", "be", "el", "ar", "fa", "he", "zh",
                            "ja", "ko", "hi", "th"})  # fmt: skip
MIN_TRANSLIT_KEY = 4


class Transliterate:
    name = "translit"
    version = "1"
    config: dict = {"methods": ["anyascii", "umlaut_de", "nordic"]}

    def __init__(
        self,
        info: dict[str, BaseInfo] | None = None,
        version: str = "2",
        places: dict[str, set[str]] | None = None,
    ) -> None:
        self.info = info  # v2 (plan-08): Latin sources typed and safe; other scripts safe
        if info is not None:
            self.version = version  # v3 (owner audit 2): other scripts only as CLDR names of
            self.config = {**self.config, "bases": "latin_typed_safe+other_safe"}  # major scripts
            if version in ("3", "4"):
                self.config |= {"other_scripts": "cldr", "langs": sorted(TRANSLIT_LANGS),
                                "min_key": MIN_TRANSLIT_KEY}  # fmt: skip
        self.places = places  # v4 (owner audit 3): "HeiShan" is Montenegro and a county in China
        if version == "4":
            if places is None:
                from countrycode.synth.city import place_index

                self.places = place_index()
            self.config["no_foreign_place_names"] = "GeoNames cities ≥ 15k + US counties"

    def generate(self, row: BaseAlias, rng: np.random.Generator) -> list[Variant]:
        a = row.alias
        if row.kind == "code":
            return []
        if self.info is not None and (
            not seedable(row, self.info) if is_latin(a) else unsafe_reason(row, self.info)
        ):
            return []
        if self.version in ("3", "4") and not is_latin(a):
            i = self.info[a] if self.info is not None else None
            if i is None or not (i.cldr_langs & TRANSLIT_LANGS):
                return []  # Wikidata aliases, rare scripts: unrecognisable or wrong (audit 2)
        out = [anyascii(a)]
        latin = self.info is None or is_latin(a)  # v2: no half-converted "Хуссар Дакотae"
        if latin and any(c in a for c in "äöüÄÖÜß"):
            out.append(a.translate(UMLAUT_DE))
        if latin and any(c in a for c in "øØåÅæÆ"):
            out.append(a.translate(NORDIC))
        kept: list[Variant] = [
            v for v in dict.fromkeys(out) if v and v != a and any(c.isalpha() for c in v)
        ]
        if self.version in ("3", "4"):  # not "JU"
            kept = [v for v in kept if len(compact_key(str(v))) >= MIN_TRANSLIT_KEY]
        if self.version == "4" and self.places is not None:
            own = set(row.targets)
            kept = [v for v in kept if not (self.places.get(compact_key(str(v)), set()) - own)]
        return kept


PLACEHOLDERS = (
    "n/a", "N/A", "N.A.", "none", "NONE", "unknown", "Unknown", "UNK", "tbd", "TBD", "null",
    "NULL", "nil", "-", "--", "---", ".", "?", "??", "*", "x", "xx", "xxx", "XXX", "0", "00",
    "not applicable", "Not Applicable", "same", "same as above", "see above", "various",
    "multiple", "other", "foreign", "FOREIGN", "test", "unbekannt", "ukjent", "ukendt", "okänd",
    "inconnu", "desconocido", "sconosciuto", "bilinmiyor", "keine Angabe", "ei tiedossa",
)  # fmt: skip


ZIP_SHAPED = re.compile(r"(?<!\d)\d{5}(?!\d)")


class JunkStrings:
    """NONE examples. Not tied to lexicon aliases: `bases()` yields n placeholder bases."""

    name = "junk"
    version = "1"
    near_check = False  # labels are NONE by construction; only exact collisions matter

    def __init__(self, n: int = 3000, version: str = "1") -> None:
        self.n = n
        self.version = version  # 2 (plan-09): no ZIP-shaped strings (a US ZIP names a state)
        self.config = {"n": n, "placeholders": len(PLACEHOLDERS)}
        if version == "2":
            self.config["no_zip_shaped"] = True

    def bases(self) -> list[BaseAlias]:
        return [BaseAlias(f"__junk_{i}", (labels.NONE,), "junk", "zxx") for i in range(self.n)]

    def generate(self, row: BaseAlias, rng: np.random.Generator) -> list[Variant]:
        out = self._generate(rng)
        if self.version == "2":
            out = [v for v in out if not isinstance(v, str) or not ZIP_SHAPED.search(v)]
        return out

    def _generate(self, rng: np.random.Generator) -> list[Variant]:
        kind = int(rng.integers(0, 4))
        if kind == 0:
            return [str(rng.choice(PLACEHOLDERS))]
        if kind == 1:
            y, m, d = (
                int(rng.integers(1900, 2027)),
                int(rng.integers(1, 13)),
                int(rng.integers(1, 29)),
            )
            fmt = int(rng.integers(0, 3))
            return [[f"{m:02d}/{d:02d}/{y}", f"{y}-{m:02d}-{d:02d}", f"{d}.{m}.{y}"][fmt]]
        if kind == 2:
            return ["".join(str(x) for x in rng.integers(0, 10, int(rng.integers(3, 9))))]
        return [f"{rng.choice(PLACEHOLDERS)} {''.join(str(x) for x in rng.integers(0, 10, 5))}"]
