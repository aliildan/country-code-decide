"""Owner G1 audit as a JSON file — the same sample as the audit tool, edited in any text editor.

    uv run python eval/review/audit_json.py export   # → data/gold/synth-v1/audit.json
    uv run python eval/review/audit_json.py import   # owner's verdicts → audit.jsonl

Each line is one generated string with its label. Fill "verdict": "ok" when the label is right,
"x" when it is wrong (optionally "fix": the right codes). `import` appends the verdicts to
data/gold/synth-v1/audit.jsonl exactly as the audit tool does (last verdict per item wins);
anything else is reported, never written. The verdicts are the owner's (CLAUDE.md §3 G1).
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from countrycode import labels
from countrycode.synth.audit import AUDIT_PATH, file_sha256, read_verdicts, sample_for_audit
from countrycode.synth.framework import SYNTH_DIR, SynthItem

JSON_PATH = Path("data/gold/synth-v1/audit.json")
HOW = (
    'Is the label right for the text? "verdict": "ok" or "x" (wrong; optionally "fix": the right '
    'codes, e.g. "AU" or "CH,US-AL"). Several codes = the text may mean any of them. NONE = not '
    'a country or US state. Leave "" to skip. Then ask Claude to import.'
)
VERDICTS = {"ok": "ok", "x": "wrong", "wrong": "wrong"}


def _name(t: str) -> str:
    if t == labels.NONE:
        return "not a country / US state"
    return labels.us_state_name(t) if t.startswith("US-") else labels.country_name(t)


def _from(alias: str) -> str:
    """The base a string came from; internal ids are hidden, except a region's GeoNames code, which
    lets the reader check an unfamiliar region against its source."""
    if alias.startswith("__region_"):
        return f"GeoNames admin1 {alias.removeprefix('__region_')}"
    return "" if alias.startswith("__") else alias


def _context(generator: str) -> str:
    from countrycode.train.dataset import rules

    fixed = rules().get("source_contexts", {}).get(f"synthetic:{generator}", [])
    return "; ".join(f"{f} field of a {r} register" for f, r, *_ in fixed)


def export_sample(
    sample: Sequence[SynthItem], audit_path: Path, out: Path, tags: set[str] | None = None
) -> int:
    """`tags`: only these generator releases (e.g. the ones a candidate model trains on)."""
    done = set(read_verdicts(audit_path))
    rows = []
    for s in sample:
        tag = f"{s.generator}@{s.version}"
        if (tag, s.text) in done or (tags is not None and tag not in tags):
            continue
        row = {
            "text": s.text,
            "label": " | ".join(f"{t} {_name(t)}" for t in s.targets),
            "from": _from(s.source_alias),
            "verdict": "",
            "fix": "",
            "tag": tag,
        }
        if ctx := _context(s.generator):
            row["context"] = ctx  # the label holds only there ("NL" → CA)
        rows.append(row)
    body = ",\n".join("  " + json.dumps(r, ensure_ascii=False) for r in rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        '{\n "how": ' + json.dumps(HOW) + ',\n "items": [\n' + body + "\n ]\n}\n", encoding="utf-8"
    )
    return len(rows)


def import_verdicts(
    path: Path,
    audit_path: Path,
    sample: Sequence[SynthItem] | None = None,
    synth_dir: Path = SYNTH_DIR,
) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    keys = {(f"{s.generator}@{s.version}", s.text): s for s in (sample or [])}
    last = read_verdicts(audit_path)
    res: dict = {"imported": 0, "unchanged": 0, "empty": 0, "invalid": []}
    recs = []
    shas: dict[str, str] = {}
    for r in doc["items"]:
        v = str(r.get("verdict", "")).strip().lower()
        if not v:
            res["empty"] += 1
            continue
        fix = [t.upper() for t in re.split(r"[,;\s]+", str(r.get("fix", "")).strip()) if t]
        if v not in VERDICTS or any(not labels.is_target(t) for t in fix):
            res["invalid"].append({"text": r["text"], "verdict": r.get("verdict")})
            continue
        k = (r["tag"], r["text"])
        prev = last.get(k)
        if prev and prev["verdict"] == VERDICTS[v] and prev.get("owner_targets", []) == fix:
            res["unchanged"] += 1
            continue
        targets = list(keys[k].targets) if k in keys else _targets_from_label(r["label"])
        ts = datetime.now(UTC).isoformat(timespec="seconds")
        recs.append(
            {
                "tag": r["tag"],
                "text": r["text"],
                "targets": targets,
                "verdict": VERDICTS[v],
                "owner_targets": fix,
                "ts": ts,
                "note": "json",
                "file_sha256": shas.setdefault(
                    r["tag"], file_sha256(Path(synth_dir) / f"{r['tag']}.jsonl")
                ),
            }
        )
    if recs:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("a", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    res["imported"] = len(recs)
    return res


def _targets_from_label(label: str) -> list[str]:
    return [part.strip().split(" ", 1)[0] for part in label.split(" | ")]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "import"])
    ap.add_argument("--path", type=Path, default=JSON_PATH)
    ap.add_argument("--force", action="store_true", help="export over an existing file")
    ap.add_argument("--run", help="only the generators of results/train/<run>.json")
    args = ap.parse_args()
    sample = sample_for_audit()
    if args.cmd == "import":
        print(json.dumps(import_verdicts(args.path, AUDIT_PATH, sample), indent=1))
        return
    if args.path.exists() and not args.force:
        sys.exit(f"{args.path} exists — import it first, then export with --force")
    tags = None
    if args.run:
        tags = set(json.loads(Path(f"results/train/{args.run}.json").read_text())["gens"])
    print(f"{export_sample(sample, AUDIT_PATH, args.path, tags)} items → {args.path}")


if __name__ == "__main__":
    main()
