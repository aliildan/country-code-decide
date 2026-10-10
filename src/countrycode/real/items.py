"""Real item type: one distinct string of one field of one source, with its count."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

FieldKind = Literal["country", "jurisdiction", "nationality", "unknown"]
ITEMS_PATH = Path("data/interim/real/items.jsonl")


@dataclass(frozen=True)
class RealItem:
    source: str
    field: str
    field_kind: FieldKind
    register_country: str
    text: str
    count: int

    @property
    def item_id(self) -> str:
        key = f"{self.source}\x1f{self.field}\x1f{self.text}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def write_items(items: list[RealItem], path: Path = ITEMS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


def read_items(path: Path = ITEMS_PATH) -> list[RealItem]:
    with path.open(encoding="utf-8") as f:
        return [RealItem(**json.loads(line)) for line in f if line.strip()]
