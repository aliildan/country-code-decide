"""G1 human audit of synthetic releases (plan-03 Task 9, CLAUDE.md §3).

A stable random sample of up to `n` items per generator file is read by the owner in the review
tool (`eval/review/audit_app.py`); verdicts go to `data/gold/synth-v1/audit.jsonl`. A generator
passes G1 with >= 300 audited items and 0 errors (label error < 1 % at 95 % confidence).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path

from scipy.stats import beta

from countrycode.synth.framework import SYNTH_DIR, SynthItem, read_synth

AUDIT_PATH = Path("data/gold/synth-v1/audit.jsonl")
# Files audited before verdicts carried a hash (audits 1–4, 2026-10-10): tag → sha256 of the file
# the owner audited (each re-release was checked byte-identical against it; decisions.md).
PINS_PATH = Path("data/gold/synth-v1/audit_pins.json")
MIN_AUDIT = 300


def sample_for_audit(
    release_dir: Path = SYNTH_DIR, n: int = MIN_AUDIT, seed: int = 13
) -> list[SynthItem]:
    out: list[SynthItem] = []
    for path in sorted(Path(release_dir).glob("*@*.jsonl")):
        items = read_synth(path)
        rng = random.Random(f"{seed}:{path.name}")
        out += rng.sample(items, min(n, len(items)))
    return out


def upper_bound_95(n: int, errors: int) -> float:
    """One-sided 95 % Clopper–Pearson upper bound of the error rate."""
    if n == 0:
        return 1.0
    return 1.0 if errors >= n else float(beta.ppf(0.95, errors + 1, n - errors))


def read_verdicts(path: Path = AUDIT_PATH) -> dict[tuple[str, str], dict]:
    if not Path(path).exists():
        return {}
    out: dict[tuple[str, str], dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            v = json.loads(line)
            out[(v["tag"], v["text"])] = v  # last verdict wins
    return out


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_status(
    sample: Sequence[SynthItem],
    path: Path = AUDIT_PATH,
    synth_dir: Path = SYNTH_DIR,
    pins: dict[str, str] | None = None,
) -> dict[str, dict]:
    """G1 counts per release. A verdict counts only for the file content it was given for: its
    `file_sha256` (or, for verdicts imported before hashes, the pinned hash of the release) must
    equal the current file's hash; otherwise it is `stale` (the release changed since)."""
    verdicts = read_verdicts(path)
    pinned: dict[str, str] = (
        pins
        if pins is not None
        else (json.loads(PINS_PATH.read_text()) if PINS_PATH.exists() else {})
    )
    current: dict[str, str | None] = {}
    status: dict[str, dict] = {}
    for it in sample:
        tag = f"{it.generator}@{it.version}"
        if tag not in current:
            f = Path(synth_dir) / f"{tag}.jsonl"
            current[tag] = file_sha256(f) if f.exists() else None
        st = status.setdefault(
            tag, {"sampled": 0, "audited": 0, "errors": 0, "stale": 0, "file_sha256": current[tag]}
        )
        st["sampled"] += 1
        v = verdicts.get((tag, it.text))
        if v is None:
            continue
        if v.get("file_sha256", pinned.get(tag)) != current[tag]:
            st["stale"] += 1
            continue
        st["audited"] += 1
        st["errors"] += int(v["verdict"] != "ok")
    for st in status.values():
        st["upper_bound_95"] = upper_bound_95(st["audited"], st["errors"])
        st["g1_pass"] = st["errors"] == 0 and st["audited"] >= min(MIN_AUDIT, st["sampled"])
    return status


GATE_REPORT = Path("results/synthetic/synth-v2.json")


def g1_status(gens: Sequence[str], report: Path = GATE_REPORT) -> dict:
    """G1 per release of a training run, from the gate report; a release never audited fails."""
    rep = json.loads(Path(report).read_text(encoding="utf-8")).get("generators", {})
    releases = {}
    for tag in gens:
        a = rep.get(tag, {}).get("G1_audit", {})
        releases[tag] = {"audited": a.get("audited", 0), "errors": a.get("errors", 0),
                         "pass": bool(a.get("g1_pass", False)),
                         "file_sha256": a.get("file_sha256")}  # fmt: skip
    return {"releases": releases, "all_pass": all(r["pass"] for r in releases.values())}
