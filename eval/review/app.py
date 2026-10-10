"""Owner review tool for real gold labels (plan-02 Task 4). Local only: http://127.0.0.1:8310

    uv run python eval/review/app.py [--port 8310]

Shows one real string at a time with its context and proposals; the owner accepts, picks or types
the answer. Answers are appended to data/gold/real-v1/owner.jsonl (last answer per item wins).
Claude never writes answers here (CLAUDE.md §2.8).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from countrycode import labels
from countrycode.lexicon.build import LEXICON_PATH, Lexicon
from countrycode.real.gold import (
    OWNER_PATH,
    REVIEW_GUIDE,
    QueueItem,
    build_queue,
    proposals,
    read_presets,
    validate_targets,
)
from countrycode.real.items import ITEMS_PATH, RealItem, read_items
from countrycode.real.splits import read_manifest


class Answer(BaseModel):
    item_id: str
    targets: list[str] = []
    skip: bool = False
    note: str = ""


PAGE = (Path(__file__).parent / "page.html").read_text(encoding="utf-8")


def create_app(
    items: Sequence[RealItem],
    manifest: Mapping[str, str],
    lexicon: Lexicon,
    presets: Mapping[str, dict],
    owner_path: Path = OWNER_PATH,
    audit_auto: int = 100,
    audit_preset: int = 50,
) -> FastAPI:
    queue: list[QueueItem] = build_queue(
        items, manifest, lexicon, presets, audit_auto=audit_auto, audit_preset=audit_preset
    )
    by_id = {q.item.item_id: q for q in queue}
    skipped: set[str] = set()
    props_cache: dict[str, list] = {}

    def props(q: QueueItem) -> list:
        if q.item.item_id not in props_cache:
            ps = q.proposals if q.proposals is not None else proposals(q.item, lexicon)
            props_cache[q.item.item_id] = [[list(t), why] for t, why in ps]
        return props_cache[q.item.item_id]

    app = FastAPI(title="Country review")

    def answered() -> set[str]:
        if not owner_path.exists():
            return set()
        done = {}
        for line in owner_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                a = json.loads(line)
                done[a["item_id"]] = bool(a.get("targets"))
        return {k for k, v in done.items() if v}

    def payload(q: QueueItem | None, done: int) -> dict:
        prog = {"done": done, "total": len(queue)}
        if q is None:
            return {"item": None, "progress": prog}
        it = q.item
        return {
            "item": {
                "item_id": it.item_id,
                "source": it.source,
                "field": it.field,
                "field_kind": it.field_kind,
                "register_country": it.register_country,
                "text": it.text,
                "count": it.count,
            },
            "split": q.split,
            "proposals": props(q),
            "progress": prog,
        }

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE.replace("__GUIDE__", REVIEW_GUIDE)

    @app.get("/api/next")
    def next_item() -> dict:
        done = answered()
        pending = [q for q in queue if q.item.item_id not in done]
        fresh = [q for q in pending if q.item.item_id not in skipped]
        return payload((fresh or pending or [None])[0], len(done & by_id.keys()))

    @app.get("/api/item/{item_id}")
    def get_item(item_id: str) -> dict:
        if item_id not in by_id:
            raise HTTPException(404, "unknown item")
        return payload(by_id[item_id], len(answered() & by_id.keys()))

    @app.post("/api/answer")
    def post_answer(a: Answer) -> dict:
        if a.item_id not in by_id:
            raise HTTPException(404, "unknown item")
        if a.skip:
            skipped.add(a.item_id)
            return {"ok": True}
        try:
            targets = validate_targets(a.targets)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        skipped.discard(a.item_id)
        rec = {
            "item_id": a.item_id,
            "targets": list(targets),
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "note": a.note,
        }
        owner_path.parent.mkdir(parents=True, exist_ok=True)
        with owner_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {"ok": True}

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8310)
    ap.add_argument(
        "--host", default="127.0.0.1", help="0.0.0.0 = reachable from the LAN (no auth)"
    )
    args = ap.parse_args()
    items = read_items(ITEMS_PATH)
    app = create_app(items, read_manifest(), Lexicon.load(LEXICON_PATH), read_presets())
    print(f"review: http://127.0.0.1:{args.port}  ({len(labels.targets())} targets)")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
