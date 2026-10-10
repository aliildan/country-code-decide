"""GBIF parsers' curated country-name variants (Apache-2.0): text → ISO code, `ZZ` = unknown."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from countrycode import labels
from countrycode.lexicon.types import Overrides

OVERRIDES_PATH = Path("configs/lexicon/overrides.yaml")
# Former ISO 3166-1 alpha-2 codes → their ISO 3166-3 entry (successor sets in overrides.yaml).
FORMER_ALPHA2 = {
    "YU": "YUCS",
    "SU": "SUHH",
    "CS": "CSHH",
    "DD": "DDDE",
    "ZR": "ZRCD",
    "BU": "BUMM",
    "TP": "TPTL",
    "AN": "ANHH",
}


def map_code(code: str, historical: dict[str, list[str]]) -> tuple[str, ...] | None:
    if code in ("ZZ", "XZ") or code in labels.NONE_TEXT_CODES:  # XZ: GBIF international waters
        return (labels.NONE,)
    code = labels.RESERVED_TEXT_CODES.get(code, code)
    if code in labels.countries():
        return (code,)
    if code in FORMER_ALPHA2:
        return tuple(sorted(historical[FORMER_ALPHA2[code]]))
    return None


def parse_gbif_tsv(
    text: str, overrides: Overrides | None = None
) -> tuple[list[tuple[str, tuple[str, ...]]], Counter]:
    historical = (overrides or Overrides.load(OVERRIDES_PATH)).historical
    pairs: list[tuple[str, tuple[str, ...]]] = []
    dropped: Counter = Counter()
    for line in text.splitlines():
        if not line.strip() or line.startswith("#") or "\t" not in line:
            continue
        name, code = line.rsplit("\t", 1)
        targets = map_code(code.strip(), historical)
        if targets is None:
            dropped[code.strip()] += 1
        else:
            pairs.append((name, targets))
    return pairs, dropped
