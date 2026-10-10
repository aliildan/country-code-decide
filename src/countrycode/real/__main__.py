"""python -m countrycode.real ingest — distinct values of the configured real sources."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import cast

import yaml

from countrycode.http import CachedClient
from countrycode.real.bulk import brreg_country_values, ch_country_values, register_code_labels
from countrycode.real.gbif import parse_gbif_tsv
from countrycode.real.items import ITEMS_PATH, FieldKind, RealItem, read_items, write_items
from countrycode.real.socrata import fetch_socrata_values
from countrycode.real.splits import check_splits, read_manifest, write_manifest

CONFIG = Path("configs/real_sources.yaml")
CACHE = Path("data/raw/real")
STATS = Path("results/real/ingest.json")
PRESET_LABELS = Path("data/interim/real/preset_labels.jsonl")


def ingest_gbif(cfg: dict, client: CachedClient) -> tuple[list[RealItem], list[dict], dict]:
    pairs, dropped = parse_gbif_tsv(client.get(cfg["url"]).decode("utf-8"))
    items, labels_ = [], []
    for text, targets in pairs:
        it = RealItem(
            cfg["source"],
            "countryName",
            cast(FieldKind, cfg["field_kind"]),
            cfg["register_country"],
            text,
            1,
        )
        items.append(it)
        labels_.append(
            {"item_id": it.item_id, "targets": list(targets), "label_source": "reference_list"}
        )
    return items, labels_, dict(dropped)


BULK = Path("data/raw/bulk")


def ingest_bulk() -> tuple[list[RealItem], list[dict]]:
    """Brreg (labels from the register's own codes) and Companies House (no codes)."""
    items: list[RealItem] = []
    presets: list[dict] = []
    brreg = BULK / "enheter_alle.json.gz"
    if brreg.exists():
        vals = brreg_country_values(brreg)
        codes = register_code_labels(vals)
        for (field, text), v in sorted(vals.items()):
            it = RealItem("no-brreg", field, "country", "NO", text, v["count"])
            items.append(it)
            if (field, text) in codes:
                presets.append(
                    {
                        "item_id": it.item_id,
                        "targets": [codes[(field, text)]],
                        "label_source": "register_code",
                        "granularity": "full",
                    }
                )
    ch = next(iter(sorted(BULK.glob("BasicCompanyDataAsOneFile-*.zip"))), None)
    if ch is not None:
        for (field, text), n in sorted(ch_country_values(ch).items()):
            items.append(RealItem("gb-ch", field, "country", "GB", text, n))
    return items, presets


def ingest() -> list[RealItem]:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    client = CachedClient(CACHE, max_rps=2.0)
    items: list[RealItem] = []
    for src in cfg["socrata"]:
        for column, kind in src["fields"].items():
            for text, n in fetch_socrata_values(client, src["domain"], src["dataset"], column):
                items.append(
                    RealItem(
                        src["source"],
                        column,
                        cast(FieldKind, kind),
                        src["register_country"],
                        text,
                        n,
                    )
                )
    return items


def split() -> None:
    items = read_items(ITEMS_PATH)
    manifest_path = write_manifest(items)
    report = check_splits(items, read_manifest(manifest_path))
    out = Path("results/real/splits.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if report["within_source_leaks"]:
        sys.exit("split check FAILED: within-source leaks")


def main() -> None:
    cmd = sys.argv[1:2]
    if cmd == ["split"]:
        split()
        return
    if cmd != ["ingest"]:
        sys.exit("usage: python -m countrycode.real ingest|split")
    items = ingest()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    g_items, g_labels, g_dropped = ingest_gbif(cfg["gbif"], CachedClient(CACHE, max_rps=2.0))
    items += g_items
    b_items, b_labels = ingest_bulk()
    items += b_items
    write_items(items, ITEMS_PATH)
    PRESET_LABELS.parent.mkdir(parents=True, exist_ok=True)
    PRESET_LABELS.write_text(
        "".join(json.dumps(x) + "\n" for x in [*g_labels, *b_labels]), encoding="utf-8"
    )
    stats = {
        "items": len(items),
        "rows_covered": sum(i.count for i in items),
        "gbif_dropped_codes": g_dropped,
        "by_source_field": {
            f"{s}/{f}": n
            for (s, f), n in sorted(Counter((i.source, i.field) for i in items).items())
        },
    }
    STATS.parent.mkdir(parents=True, exist_ok=True)
    STATS.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
