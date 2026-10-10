"""Territory display names from Unicode CLDR (via Babel) in every available locale."""

from __future__ import annotations

from babel import Locale
from babel.core import get_cldr_version
from babel.localedata import locale_identifiers

from countrycode import labels
from countrycode.lexicon.types import AliasRow

# Regions, unions and "unknown region": names of places that are not a country of the label space.
CLDR_NONE_CODES = frozenset({"EU", "EZ", "UN", "ZZ", "QO"})


def _is_none_code(code: str) -> bool:
    return code in CLDR_NONE_CODES or (len(code) == 3 and code.isdigit())


def load_cldr() -> list[AliasRow]:
    countries = frozenset(labels.countries())
    rows: set[AliasRow] = set()
    for loc in locale_identifiers():
        try:
            territories = Locale.parse(loc).territories
        except (ValueError, KeyError):
            continue
        for code, name in territories.items():
            code, name = str(code), str(name)
            if code in countries:
                rows.add(AliasRow(name, code, "name", loc, "cldr"))
            elif _is_none_code(code):
                rows.add(AliasRow(name, labels.NONE, "non_country", loc, "cldr"))
    return sorted(rows, key=lambda r: (r.target, r.lang, r.alias))


def cldr_version() -> str:
    return str(get_cldr_version())
