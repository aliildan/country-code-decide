"""Blind re-labelling of the register calibration role (design-02 §2: gold quality).

    uv run python eval/review/blind_json.py export [--path …]  # unanswered items → blind.json
    uv run python eval/review/blind_json.py import [--path …]  # owner's answers → blind.jsonl
    uv run python eval/review/blind_json.py report      # → results/real/gold_quality.json
    uv run python eval/review/blind_json.py adjudicate  # disagreements → adjudicate.json

A seeded, stratified sample of the register calib role is shown **without proposals or gold**:
15 % first (89 items), then the whole role (decisions.md 2026-10-10: 89 items can never bound gold
error below 2 %), exported as a second file with only the items not yet answered. The
blind answers are stored apart from owner.jsonl and never become gold by themselves; the owner
adjudicates each disagreement in review-JSON format (`json_queue.py import --path …`), and the
owner's answer wins (CLAUDE.md §2.8).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from scipy.stats import beta

from countrycode.real.gold import GoldItem, validate_targets

JSON_PATH = Path("data/gold/real-v1/blind.json")
BLIND_PATH = Path("data/gold/real-v1/blind.jsonl")
ADJ_PATH = Path("data/gold/real-v1/adjudicate.json")
REPORT_PATH = Path("results/real/gold_quality.json")
BLIND_FRAC = 1.0  # the whole register calib role (first batch: 0.15)
HOW = (
    "Answer each string from scratch (no proposals are shown on purpose): codes separated by "
    'comma/space ("US-CA", "GB,CA,US", "NONE"). Leave "" or "?" to skip. Then ask Claude to import.'
)


def _rank(seed: int, g: GoldItem) -> str:
    return hashlib.sha256(f"blind\x1f{seed}\x1f{g.item_id}".encode()).hexdigest()


def sample_blind(items: Sequence[GoldItem], frac: float = 0.15, seed: int = 13) -> list[GoldItem]:
    """`frac` of each annotation mode, chosen by a seeded hash (stable as items are added)."""
    by_mode: dict[str, list[GoldItem]] = defaultdict(list)
    for g in items:
        by_mode[g.annotation_mode].append(g)
    out: list[GoldItem] = []
    for mode in sorted(by_mode):
        group = sorted(by_mode[mode], key=lambda g: _rank(seed, g))
        out += group[: round(frac * len(group))]
    return out


def _write(rows: list[dict], out: Path, how: str = HOW) -> None:
    body = ",\n".join("  " + json.dumps(r, ensure_ascii=False) for r in rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('{\n "how": ' + json.dumps(how) + ',\n "items": [\n' + body + "\n ]\n}\n",
                   encoding="utf-8")  # fmt: skip


def export_blind(items: Sequence[GoldItem], out: Path) -> int:
    rows = [{"text": g.text, "field": g.field, "kind": g.field_kind, "reg": g.register_country,
             "src": g.source, "answer": "", "id": g.item_id} for g in items]  # fmt: skip
    _write(rows, out)
    return len(rows)


def _read_blind(path: Path) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["item_id"]] = tuple(sorted(r["targets"]))
    return out


def unanswered(items: Sequence[GoldItem], blind_path: Path) -> list[GoldItem]:
    """Items without a blind answer yet (a later, larger sample exports only the new ones)."""
    done = _read_blind(blind_path)
    return [g for g in items if g.item_id not in done]


def import_blind(path: Path, blind_path: Path) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    last = _read_blind(blind_path)
    res: dict = {"imported": 0, "unchanged": 0, "empty": 0, "invalid": []}
    recs = []
    for r in doc["items"]:
        a = str(r.get("answer", "")).strip()
        if not a or a == "?":
            res["empty"] += 1
            continue
        try:
            ts = validate_targets(re.split(r"[,;\s]+", a))
        except ValueError:
            res["invalid"].append({"id": r["id"], "answer": r.get("answer")})
            continue
        if last.get(r["id"]) == ts:
            res["unchanged"] += 1
            continue
        recs.append({"item_id": r["id"], "targets": list(ts),
                     "ts": datetime.now(UTC).isoformat(timespec="seconds")})  # fmt: skip
    if recs:
        blind_path.parent.mkdir(parents=True, exist_ok=True)
        with Path(blind_path).open("a", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    res["imported"] = len(recs)
    return res


def _first_gold(adjudication: Path | None) -> dict[str, tuple[str, ...]]:
    """item id → the gold label before adjudication (proposal 1 of the adjudication file)."""
    if adjudication is None or not Path(adjudication).exists():
        return {}
    out = {}
    for r in json.loads(Path(adjudication).read_text(encoding="utf-8"))["items"]:
        codes = r["proposals"][0].split("(")[0].strip()
        out[r["id"]] = tuple(sorted(c for c in re.split(r"[,\s]+", codes) if c))
    return out


def report(gold: Sequence[GoldItem], blind_path: Path, adjudication: Path | None = None) -> dict:
    """Agreement of blind answers with gold *before adjudication* on the items answered blind;
    one-sided 95 % Clopper–Pearson upper bound on the disagreement rate (an upper bound on gold
    error only if the blind answer is assumed right — reported as such)."""
    blind = _read_blind(blind_path)
    first = _first_gold(adjudication)
    pairs = [(g, blind[g.item_id]) for g in gold if g.item_id in blind]
    before = {g.item_id: first.get(g.item_id, tuple(sorted(g.targets))) for g, _ in pairs}
    dis = [g for g, b in pairs if before[g.item_id] != b]
    n, k = len(pairs), len(dis)
    to_blind = sum(1 for g, b in pairs if g.item_id in first and tuple(sorted(g.targets)) == b)
    by_mode = Counter(g.annotation_mode for g, _ in pairs)
    dis_mode = Counter(g.annotation_mode for g in dis)
    return {
        "n": n,
        "agree": n - k,
        "disagree": k,
        "agreement": (n - k) / n if n else None,
        "gold_error_upper_95": float(beta.ppf(0.95, k + 1, n - k)) if n else None,
        "by_mode": {
            m: {"n": c, "disagree": dis_mode.get(m, 0)} for m, c in sorted(by_mode.items())
        },
        "adjudicated": len(first),
        "adjudicated_to_blind": to_blind,
    }


def export_adjudication(gold: Sequence[GoldItem], blind_path: Path, out: Path) -> int:
    """Disagreements in review-JSON format: proposal 1 = gold, 2 = blind; the owner answers."""
    blind = _read_blind(blind_path)
    rows = []
    for g in gold:
        b = blind.get(g.item_id)
        if b is None or tuple(sorted(g.targets)) == b:
            continue
        rows.append({"text": g.text, "field": g.field, "kind": g.field_kind,
                     "reg": g.register_country, "src": g.source, "n": g.count, "split": g.split,
                     "proposals": [f"{','.join(sorted(g.targets))}  (gold)",
                                   f"{','.join(b)}  (blind)"],
                     "answer": "", "id": g.item_id})  # fmt: skip
    _write(rows, out, 'Final answer per string: "1" = gold, "2" = blind, or codes. Claude imports.')
    return len(rows)


def _calib_register() -> list[GoldItem]:
    from countrycode.calib.calibration import domain_of
    from countrycode.evaluation.__main__ import gold_and_lexicon
    from countrycode.real.roles import with_roles

    gold, _ = gold_and_lexicon()
    return [g for g in with_roles(gold) if g.split == "calib" and domain_of(g.source) == "register"]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "import", "report", "adjudicate"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--frac", type=float, default=BLIND_FRAC)
    ap.add_argument("--path", type=Path, default=JSON_PATH)
    args = ap.parse_args()
    sample = sample_blind(_calib_register(), frac=args.frac)
    if args.cmd == "export":
        if args.path.exists() and not args.force:
            sys.exit(f"{args.path} exists — import it first, then export with --force")
        print(f"{export_blind(unanswered(sample, BLIND_PATH), args.path)} items → {args.path}")
    elif args.cmd == "import":
        print(json.dumps(import_blind(args.path, BLIND_PATH), indent=1))
    elif args.cmd == "report":
        rep = report(sample, BLIND_PATH, adjudication=ADJ_PATH)
        REPORT_PATH.write_text(json.dumps(rep, indent=1) + "\n")
        print(json.dumps(rep))
    else:
        print(f"{export_adjudication(sample, BLIND_PATH, ADJ_PATH)} disagreements → {ADJ_PATH}")


if __name__ == "__main__":
    main()
