"""Label space: 250 countries (249 ISO 3166-1 alpha-2 + XK), 51 US states (ISO 3166-2:US), NONE.

Design-01 §2. The lists live in `configs/labels/*.yaml` (reviewed, committed);
`python -m countrycode.labels build-countries` regenerates the country list from CLDR (Babel).
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Final

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "labels"
COUNTRIES_YAML = CONFIG_DIR / "countries.yaml"
US_STATES_YAML = CONFIG_DIR / "us_states.yaml"

NONE: Final[str] = "NONE"
NON_ISO: Final[frozenset[str]] = frozenset({"XK"})
# Exceptionally reserved codes that people write as if they were country codes.
RESERVED_TEXT_CODES: Final[Mapping[str, str]] = {
    "UK": "GB",
    "EL": "GR",
    "AC": "SH",
    "TA": "SH",
    "DG": "IO",
    "IC": "ES",
    "EA": "ES",
    "CP": "FR",
    "FX": "FR",
    "CQ": "GG",
}
NONE_TEXT_CODES: Final[frozenset[str]] = frozenset({"EU", "EZ", "UN", "ZZ"})

# CLDR two-letter territories that are not countries of the label space (XK is kept).
_CLDR_EXCLUDE = frozenset(
    {"AC", "CP", "CQ", "DG", "EA", "EU", "EZ", "IC", "TA", "UN", "QO", "ZZ"}
    | {"AA"}
    | {f"Q{c}" for c in "MNOPQRSTUVWXYZ"}
    | ({f"X{c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"} - {"XK"})
)


@cache
def _countries() -> tuple[tuple[str, str, bool], ...]:
    rows = yaml.safe_load(COUNTRIES_YAML.read_text(encoding="utf-8"))
    return tuple((str(r["code"]), str(r["name"]), bool(r.get("non_iso", False))) for r in rows)


@cache
def _us_states() -> tuple[tuple[str, str], ...]:
    rows = yaml.safe_load(US_STATES_YAML.read_text(encoding="utf-8"))
    return tuple((str(r["code"]), str(r["name"])) for r in rows)


def countries() -> tuple[str, ...]:
    return tuple(c for c, _, _ in _countries())


def country_name(code: str) -> str:
    return next(n for c, n, _ in _countries() if c == code)


def us_states() -> tuple[str, ...]:
    return tuple(c for c, _ in _us_states())


def us_state_name(code: str) -> str:
    return next(n for c, n in _us_states() if c == code)


def usps(state: str) -> str:
    return state.removeprefix("US-")


@cache
def targets() -> tuple[str, ...]:
    return (*countries(), *us_states(), NONE)


@cache
def _target_set() -> frozenset[str]:
    return frozenset(targets())


def is_target(t: str) -> bool:
    return t in _target_set()


def country_of(t: str) -> str:
    return "US" if t.startswith("US-") else t


def build_countries() -> list[dict]:
    from babel import Locale

    names = {str(k): str(v) for k, v in Locale("en").territories.items()}
    codes = sorted(
        k for k in names if len(k) == 2 and k.isalpha() and k.isupper() and k not in _CLDR_EXCLUDE
    )
    rows = [
        {"code": c, "name": names[c], **({"non_iso": True} if c in NON_ISO else {})} for c in codes
    ]
    assert len(rows) == 250, len(rows)
    return rows


def _write_countries(rows: list[dict]) -> None:
    lines = [
        "# 249 ISO 3166-1 alpha-2 codes + XK (user-assigned). Generated from CLDR (Babel);",
        "# reviewed. Codes are quoted: unquoted NO would parse as False in YAML 1.1.",
    ]
    for r in rows:
        extra = ", non_iso: true" if r.get("non_iso") else ""
        name = r["name"].replace('"', '\\"')
        lines.append(f'- {{code: "{r["code"]}", name: "{name}"{extra}}}')
    COUNTRIES_YAML.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:] == ["build-countries"]:
        _write_countries(build_countries())
        print(f"wrote {COUNTRIES_YAML}")
    else:
        sys.exit("usage: python -m countrycode.labels build-countries")
