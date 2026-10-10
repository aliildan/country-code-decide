"""Real-data splits (design-01 §4): per source, by hash of compact_key(text) — dev 30 / calib 30 /
test 40 %. The same normalised string of one source never lands in two parts."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from countrycode.real.items import RealItem
from countrycode.text import compact_key

Split = Literal["dev", "calib", "test"]
MANIFEST_PATH = Path("data/splits/real-v1.json")


def assign_split(source: str, text: str) -> Split:
    h = hashlib.sha256(f"{source}\x1f{compact_key(text)}".encode()).digest()
    bucket = int.from_bytes(h[:8], "big") % 100
    return "dev" if bucket < 30 else "calib" if bucket < 60 else "test"


def build_manifest(items: list[RealItem]) -> dict[str, Split]:
    return {it.item_id: assign_split(it.source, it.text) for it in items}


def write_manifest(items: list[RealItem], path: Path = MANIFEST_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_manifest(items), sort_keys=True), encoding="utf-8")
    return path


def read_manifest(path: Path = MANIFEST_PATH) -> dict[str, Split]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_splits(items: list[RealItem], manifest: Mapping[str, str]) -> dict:
    parts_by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
    cross: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    for it in items:
        part = manifest[it.item_id]
        counts[f"{it.source}/{part}"] += 1
        parts_by_key[(it.source, compact_key(it.text))].add(part)
        cross[compact_key(it.text)].add(part)
    return {
        "within_source_leaks": sum(1 for v in parts_by_key.values() if len(v) > 1),
        "cross_source_keys_in_several_parts": sum(1 for v in cross.values() if len(v) > 1),
        "items_by_source_part": dict(sorted(counts.items())),
    }
