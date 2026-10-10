"""Evaluation harness (plan-02 Task 5, design-01 §7).

A predictor returns, per gold item, a `Prediction`: one target (or None = abstain), its candidate
set and a confidence. Correct = the target equals a single gold target, or — for ambiguous gold
(a set) — the predictor abstained and returned exactly that set. Selective metrics answer items
by confidence; an abstention is never "answered".
"""

from __future__ import annotations

import hashlib
import json
import time
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from countrycode import labels
from countrycode.evaluation.metrics import aurc, bootstrap_ci, coverage_at_risk, ece
from countrycode.real.gold import GoldItem
from countrycode.text import compact_key

ALPHAS = (0.01, 0.02, 0.05)
RESULTS_DIR = Path("results")
PREREG_PATH = Path("docs/preregistration.md")


@dataclass(frozen=True)
class Prediction:
    target: str | None
    candidates: tuple[str, ...]
    confidence: float


class Predictor(Protocol):
    name: str

    def predict(self, items: Sequence[GoldItem]) -> list[Prediction]: ...


def _same(pred: str, gold: str, granularity: str) -> bool:
    if granularity == "country":
        return labels.country_of(pred) == labels.country_of(gold)
    return pred == gold


def item_correct(p: Prediction, g: GoldItem) -> bool:
    """Full-coverage correctness. NONE gold: no code (abstain or NONE) is right. Ambiguous gold:
    abstaining with exactly that candidate set is right."""
    if g.targets == (labels.NONE,):
        return p.target in (None, labels.NONE)
    if len(g.targets) == 1:
        return p.target is not None and _same(p.target, g.targets[0], g.granularity)
    return p.target is None and set(p.candidates) == set(g.targets)


def answered_correct(p: Prediction, g: GoldItem) -> bool:
    """Correctness of an answered item (a single code on ambiguous gold is an error)."""
    return (
        len(g.targets) == 1
        and p.target is not None
        and _same(p.target, g.targets[0], g.granularity)
    )


def gold_kind(g: GoldItem) -> str:
    if len(g.targets) > 1:
        return "ambiguous"
    t = g.targets[0] if g.targets else labels.NONE
    return "none" if t == labels.NONE else "us_state" if t.startswith("US-") else "country"


def script_of(text: str) -> str:
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            for s in ("LATIN", "CYRILLIC", "GREEK", "ARABIC", "HEBREW", "HANGUL", "THAI"):
                if name.startswith(s):
                    return s
            if name.startswith(("CJK", "HIRAGANA", "KATAKANA")):
                return "CJK"
            if name.startswith(("DEVANAGARI", "BENGALI", "TAMIL", "TELUGU", "GURMUKHI")):
                return "INDIC"
            return "OTHER"
    return "NONE"


def _stats(preds: Sequence[Prediction], gold: Sequence[GoldItem], ci: bool = False) -> dict:
    n = len(gold)
    if n == 0:
        return {"n": 0}
    w = np.array([g.count for g in gold], dtype=float)
    corr = np.array([item_correct(p, g) for p, g in zip(preds, gold, strict=True)])
    ans = np.array([p.target is not None for p in preds])
    ans_corr = np.array([answered_correct(p, g) for p, g in zip(preds, gold, strict=True)])
    conf = np.array([p.confidence if p.target is not None else -1.0 for p in preds])
    none_gold = np.array([g.targets == (labels.NONE,) for g in gold])
    false_accept = np.array([p.target not in (None, labels.NONE) for p in preds])
    single = np.array([len(g.targets) == 1 and g.targets != (labels.NONE,) for g in gold])
    country_ok = np.array(
        [
            len(g.targets) == 1
            and p.target is not None
            and labels.country_of(p.target) == labels.country_of(g.targets[0])
            for p, g in zip(preds, gold, strict=True)
        ]
    )
    compat = np.array(
        [
            c or (len(g.targets) > 1 and p.target in g.targets)
            for c, p, g in zip(corr, preds, gold, strict=True)
        ]
    )
    out = {
        "n": n,
        "weight": float(w.sum()),
        "accuracy": float(corr.mean()),
        "set_compatible_accuracy": float(compat.mean()),
        "country_accuracy": float(country_ok[single].mean()) if single.any() else float("nan"),
        "accuracy_weighted": float((corr * w).sum() / w.sum()),
        "answered": float(ans.mean()),
        "risk_answered": float((~ans_corr[ans]).mean()) if ans.any() else float("nan"),
        "aurc": aurc(conf, ans_corr),
        "aurc_weighted": aurc(conf, ans_corr, w),
        "coverage_at_risk": {str(a): coverage_at_risk(conf, ans_corr, a) for a in ALPHAS},
        "coverage_at_risk_weighted": {
            str(a): coverage_at_risk(conf, ans_corr, a, w) for a in ALPHAS
        },
        "ece": ece(conf[ans], ans_corr[ans]) if ans.any() else float("nan"),
        "none_items": int(none_gold.sum()),
        "none_false_accept": float(false_accept[none_gold].mean()) if none_gold.any() else None,
        "ambiguous_items": int(sum(len(g.targets) > 1 for g in gold)),
    }
    if ci:
        out["accuracy_ci95"] = bootstrap_ci(lambda idx: float(corr[idx].mean()), n)
        out["coverage_at_1pct_ci95"] = bootstrap_ci(
            lambda idx: coverage_at_risk(conf[idx], ans_corr[idx], 0.01), n, n_boot=300
        )
    return out


def _prereg_hash(path: Path = PREREG_PATH) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    if "Status: frozen" not in text:
        return None
    return hashlib.sha256(text.encode()).hexdigest()


def evaluate(
    predictor: Predictor,
    gold: Sequence[GoldItem],
    part: str,
    out_dir: Path | None = RESULTS_DIR,
    allow_test: bool = False,
    seen_keys: set[str] | None = None,
    prereg_path: Path = PREREG_PATH,
    kinds: Mapping[str, str] | None = None,
) -> dict:
    """`kinds`: item_id → noise kind (evaluation.slices) for `noise=<kind>` slices."""
    gold = [g for g in gold if g.split == part]
    prereg = _prereg_hash(prereg_path)
    if part.startswith("test") and not (allow_test and prereg):
        raise PermissionError("test is sealed: needs allow_test=True and a frozen preregistration")
    t0 = time.perf_counter()
    preds = predictor.predict(gold)
    elapsed = time.perf_counter() - t0
    slices: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(gold):
        slices[f"source={g.source}"].append(i)
        slices[f"field_kind={g.field_kind}"].append(i)
        slices[f"script={script_of(g.text)}"].append(i)
        slices[f"gold={gold_kind(g)}"].append(i)
        slices[f"label={g.annotation_mode}"].append(i)
        if kinds is not None and g.item_id in kinds:
            slices[f"noise={kinds[g.item_id]}"].append(i)
        if seen_keys is not None:
            slices["lexicon=" + ("seen" if compact_key(g.text) in seen_keys else "novel")].append(i)
    report = {
        "predictor": predictor.name,
        "part": part,
        "overall": _stats(preds, gold, ci=True),
        "slices": {
            k: _stats([preds[i] for i in v], [gold[i] for i in v])
            for k, v in sorted(slices.items())
        },
        "latency_ms_per_item": 1000 * elapsed / max(len(gold), 1),
        "preregistration_sha256": prereg if part.startswith("test") else None,
    }
    if out_dir is not None:
        path = Path(out_dir) / part / f"{predictor.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=1, default=float) + "\n")
    return report
