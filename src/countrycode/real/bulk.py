"""Country fields from open register bulk files (plan-06 Task 3). Only distinct field values with
counts leave these functions — no record, name or address is kept.

- Brønnøysund Enhetsregisteret `enheter_alle.json.gz` (NLOD): business and postal address
  `land` (+ the register's own `landkode` on the same record) and the foreign-register address.
- Companies House Basic Company Data (OGL v3): `RegAddress.Country`, `CountryOfOrigin`.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from countrycode import labels

BRREG_FIELDS = ("forretningsadresse", "postadresse", "utenlandskRegisterAdresse")
CH_FIELDS = ("RegAddress.Country", "CountryOfOrigin")


def brreg_country_values(path: Path) -> dict[tuple[str, str], dict]:
    """(field, text) → {"count": n, "codes": {landkode: n}}."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    out: dict[tuple[str, str], dict] = defaultdict(lambda: {"count": 0, "codes": Counter()})
    for rec in data:
        for fld in BRREG_FIELDS:
            a = rec.get(fld)
            if not isinstance(a, dict) or not a.get("land"):
                continue
            v = out[(f"{fld}.land", str(a["land"]))]
            v["count"] += 1
            if a.get("landkode"):
                v["codes"][str(a["landkode"])] += 1
    return {k: {"count": v["count"], "codes": dict(v["codes"])} for k, v in out.items()}


def register_code_labels(
    values: dict[tuple[str, str], dict], min_share: float = 0.99
) -> dict[tuple[str, str], str]:
    """A text whose records carry one register code in ≥ `min_share` of cases is labelled with
    that code (`label_source = register_code`)."""
    out = {}
    for key, v in values.items():
        codes = v["codes"]
        total = sum(codes.values())
        if not total:
            continue
        code, n = max(codes.items(), key=lambda kv: kv[1])
        code = labels.RESERVED_TEXT_CODES.get(code, code)
        if n / total >= min_share and code in labels.countries():
            out[key] = code
    return out


def ch_country_values(path: Path) -> dict[tuple[str, str], int]:
    out: Counter = Counter()
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(name) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace"))
            cols = {c.strip(): c for c in (reader.fieldnames or [])}
            for row in reader:
                for fld in CH_FIELDS:
                    v = (row.get(cols.get(fld, fld)) or "").strip()
                    if v:
                        out[(fld, v)] += 1
    return dict(out)
