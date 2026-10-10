"""Labels, aliases, alpha-3 codes and demonyms from Wikidata (CC0) via SPARQL.

Keys of the fetched bindings: `countries` (current ISO 3166-1 items), `us_states` (ISO 3166-2:US),
`extra` (items such as England mapped to a target in overrides.yaml) and `historical`
(ISO 3166-3 items, mapped to successor sets in overrides.yaml).
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from countrycode import labels
from countrycode.http import CachedClient
from countrycode.lexicon.types import AliasKind, AliasRow, Overrides

ENDPOINT = "https://query.wikidata.org/sparql"
CACHE_DIR = Path("data/raw/wikidata")
OVERRIDES_PATH = Path("configs/lexicon/overrides.yaml")
_HEADERS = {"Accept": "application/sparql-results+json"}

_NAME_UNION = """
  { ?item rdfs:label ?label . BIND("name" AS ?kind) }
  UNION { ?item skos:altLabel ?label . BIND("name" AS ?kind) }"""


def _values(codes: Sequence[str]) -> str:
    return " ".join(f'"{c}"' for c in codes)


def build_queries(codes: Sequence[str], batch: int = 25) -> list[str]:
    """Current countries: P297 statements without an end time, items not dissolved."""
    out = []
    for i in range(0, len(codes), batch):
        out.append(
            f"""SELECT ?code ?label ?kind WHERE {{
  VALUES ?code {{ {_values(codes[i : i + batch])} }}
  ?item p:P297 ?st . ?st ps:P297 ?code .
  FILTER NOT EXISTS {{ ?st pq:P582 ?end }}
  FILTER NOT EXISTS {{ ?item wdt:P576 ?dissolved }}
  {{ ?item rdfs:label ?label . BIND("name" AS ?kind) }}
  UNION {{ ?item skos:altLabel ?label . BIND("name" AS ?kind) }}
  UNION {{ ?item wdt:P298 ?label . BIND("code" AS ?kind) }}
  UNION {{ ?item wdt:P1549 ?label . BIND("demonym" AS ?kind) }}
}}"""
        )
    return out


def _us_state_queries(codes: Sequence[str], batch: int = 25) -> list[str]:
    return [
        f"""SELECT ?code ?label ?kind WHERE {{
  VALUES ?code {{ {_values(codes[i : i + batch])} }}
  ?item wdt:P300 ?code .{_NAME_UNION}
}}"""
        for i in range(0, len(codes), batch)
    ]


def _extra_query(items: Sequence[str]) -> str:
    vals = " ".join(f"wd:{q}" for q in items)
    return f"""SELECT ?item ?label ?kind WHERE {{
  VALUES ?item {{ {vals} }}{_NAME_UNION}
}}"""


def _historical_query(codes: Sequence[str]) -> str:
    return f"""SELECT ?code ?label ?kind WHERE {{
  VALUES ?code {{ {_values(codes)} }}
  ?item wdt:P773 ?code .{_NAME_UNION}
}}"""


def _run(client: CachedClient, query: str) -> list[dict]:
    data = client.get_json(ENDPOINT, {"query": query, "format": "json"}, _HEADERS)
    return data["results"]["bindings"]


def fetch_bindings(
    client: CachedClient, extra_items: Sequence[str], historical_codes: Sequence[str] = ()
) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"countries": [], "us_states": [], "extra": [], "historical": []}
    for q in build_queries(labels.countries()):
        out["countries"] += _run(client, q)
    for q in _us_state_queries(labels.us_states()):
        out["us_states"] += _run(client, q)
    if extra_items:
        out["extra"] = _run(client, _extra_query(extra_items))
    if historical_codes:
        out["historical"] = _run(client, _historical_query(historical_codes))
    return out


def _lang(b: Mapping) -> str:
    return b["label"].get("xml:lang") or "zxx"


def parse_bindings(
    bindings: Mapping[str, list[dict]],
    historical: Mapping[str, list[str]],
    extra: Mapping[str, str],
) -> list[AliasRow]:
    rows: set[AliasRow] = set()
    for key in ("countries", "us_states"):
        for b in bindings.get(key, []):
            target = b["code"]["value"]
            if not labels.is_target(target):
                continue
            kind = cast(AliasKind, b["kind"]["value"])
            rows.add(AliasRow(b["label"]["value"], target, kind, _lang(b), "wikidata"))
    for b in bindings.get("extra", []):
        qid = b["item"]["value"].rsplit("/", 1)[-1]
        if qid in extra:
            rows.add(AliasRow(b["label"]["value"], extra[qid], "name", _lang(b), "wikidata"))
    for b in bindings.get("historical", []):
        for target in historical.get(b["code"]["value"], []):
            rows.add(AliasRow(b["label"]["value"], target, "historical", _lang(b), "wikidata"))
    return sorted(rows, key=lambda r: (r.target, r.lang, r.alias))


def load_wikidata(
    client: CachedClient | None = None, overrides: Overrides | None = None
) -> list[AliasRow]:
    ov = overrides or Overrides.load(OVERRIDES_PATH)
    cl = client or CachedClient(CACHE_DIR, max_rps=1.0)
    raw = fetch_bindings(cl, list(ov.extra_items), list(ov.historical))
    return parse_bindings(raw, ov.historical, ov.extra_items)


if __name__ == "__main__":
    if sys.argv[1:] != ["fetch"]:
        sys.exit("usage: python -m countrycode.lexicon.wikidata fetch")
    ov = Overrides.load(OVERRIDES_PATH)
    raw = fetch_bindings(
        CachedClient(CACHE_DIR, max_rps=1.0), list(ov.extra_items), list(ov.historical)
    )
    for k, v in raw.items():
        print(f"{k}: {len(v)} bindings")
    print(f"rows: {len(parse_bindings(raw, ov.historical, ov.extra_items))}")
