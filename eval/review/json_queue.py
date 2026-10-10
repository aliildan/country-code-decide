"""Owner review as a JSON file — the same queue as the review tool, edited in any text editor.

    uv run python eval/review/json_queue.py export   # → data/gold/real-v1/review.json
    uv run python eval/review/json_queue.py import   # owner's answers → owner.jsonl

`export` writes every unanswered queue item with an empty `answer`, one item per line; it never
overwrites an existing file (use --force after importing). `import` validates each answer and
appends changed ones to data/gold/real-v1/owner.jsonl (note "json"; last answer per item wins);
invalid answers are reported, never written. The answers are the owner's (CLAUDE.md §2.8).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from countrycode.lexicon.build import LEXICON_PATH, Lexicon
from countrycode.real.gold import (
    OWNER_PATH,
    REVIEW_GUIDE,
    build_queue,
    proposals,
    read_owner,
    read_presets,
    validate_targets,
)
from countrycode.real.items import ITEMS_PATH, RealItem, read_items
from countrycode.real.splits import read_manifest

JSON_PATH = Path("data/gold/real-v1/review.json")

ANSWER_HELP = (
    'Fill "answer": a proposal number ("1" = first proposal), or codes separated by comma/space '
    '("US-CA", "GB,CA,US", "NONE"). Leave "" (or "?") to skip. Then ask Claude to import.'
)


def _last_answers(owner_path: Path) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    if owner_path.exists():
        for line in owner_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                a = json.loads(line)
                out[a["item_id"]] = tuple(a.get("targets") or ())
    return out


def export_queue(
    items: Sequence[RealItem],
    manifest: Mapping[str, str],
    lexicon: Lexicon,
    presets: Mapping[str, dict],
    owner_path: Path,
    out: Path,
    audit_auto: int = 100,
    audit_preset: int = 50,
    sources: set[str] | None = None,
) -> int:
    """`sources`: only items of these sources (e.g. the fresh-test registers)."""
    done = set(read_owner(owner_path))
    queue = build_queue(
        items, manifest, lexicon, presets, audit_auto=audit_auto, audit_preset=audit_preset
    )
    rows = []
    for q in queue:
        it = q.item
        if it.item_id in done or (sources is not None and it.source not in sources):
            continue
        ps = q.proposals if q.proposals is not None else proposals(it, lexicon)
        rows.append(
            {
                "text": it.text,
                "field": it.field,
                "kind": it.field_kind,
                "reg": it.register_country,
                "src": it.source,
                "n": it.count,
                "split": q.split,
                "proposals": [f"{','.join(t)}  ({why})" for t, why in ps],
                "answer": "",
                "id": it.item_id,
            }
        )
    head = json.dumps({"guide": REVIEW_GUIDE, "how": ANSWER_HELP}, ensure_ascii=False, indent=1)
    body = ",\n".join("  " + json.dumps(r, ensure_ascii=False) for r in rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(head[:-2] + ',\n "items": [\n' + body + "\n ]\n}\n", encoding="utf-8")
    return len(rows)


def parse_answer(answer: str, props: Sequence[str]) -> tuple[str, ...] | None:
    """None = skipped; raises ValueError for an answer that is not a valid label."""
    a = answer.strip()
    if not a or a == "?":
        return None
    if a.isdigit():
        i = int(a)
        if not 1 <= i <= len(props):
            raise ValueError(f"no proposal {i}")
        a = props[i - 1].split("  (")[0]
    return validate_targets(re.split(r"[,;\s]+", a))


def import_answers(path: Path, owner_path: Path) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    last = _last_answers(owner_path)
    res: dict = {"imported": 0, "unchanged": 0, "empty": 0, "invalid": []}
    recs = []
    for r in doc["items"]:
        try:
            ts = parse_answer(str(r.get("answer", "")), r.get("proposals", []))
        except ValueError:
            res["invalid"].append({"id": r["id"], "answer": r.get("answer")})
            continue
        if ts is None:
            res["empty"] += 1
        elif last.get(r["id"]) == ts:
            res["unchanged"] += 1
        else:
            ts_now = datetime.now(UTC).isoformat(timespec="seconds")
            recs.append({"item_id": r["id"], "targets": list(ts), "ts": ts_now, "note": "json"})
    if recs:
        owner_path.parent.mkdir(parents=True, exist_ok=True)
        with owner_path.open("a", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    res["imported"] = len(recs)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "import"])
    ap.add_argument("--path", type=Path, default=JSON_PATH)
    ap.add_argument("--force", action="store_true", help="export over an existing file")
    ap.add_argument("--sources", help="comma-separated sources to export (default: all)")
    args = ap.parse_args()
    if args.cmd == "import":
        print(json.dumps(import_answers(args.path, OWNER_PATH), ensure_ascii=False, indent=1))
        return
    if args.path.exists() and not args.force:
        sys.exit(f"{args.path} exists — import it first, then export with --force")
    items = read_items(ITEMS_PATH)
    lx = Lexicon.load(LEXICON_PATH)
    sources = set(args.sources.split(",")) if args.sources else None
    n = export_queue(
        items, read_manifest(), lx, read_presets(), OWNER_PATH, args.path, sources=sources
    )
    print(f"{n} items → {args.path}")


if __name__ == "__main__":
    main()
