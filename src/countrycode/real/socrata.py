"""Socrata aggregate queries: distinct values of one column with counts (no records, no names)."""

from __future__ import annotations

from countrycode.http import CachedClient


def fetch_socrata_values(
    client: CachedClient, domain: str, dataset: str, column: str, page_size: int = 50_000
) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    offset = 0
    while True:
        params = {
            "$select": f"{column}, count(*) as n",
            "$group": column,
            "$order": column,
            "$limit": str(page_size),
            "$offset": str(offset),
        }
        rows = client.get_json(f"https://{domain}/resource/{dataset}.json", params)
        for r in rows:
            if r.get(column) is not None:
                out.append((str(r[column]), int(r["n"])))
        if len(rows) < page_size:
            return out
        offset += page_size
