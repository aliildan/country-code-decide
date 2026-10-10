"""Fit a run's calibration on real calib items (plan-04 Task 5).

    uv run python -m countrycode.calib.fit --run runs/<name>

Writes runs/<name>/calibration.json (temperatures, λ_α for α in 1/2/5 %, conformal quantile).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from countrycode import labels
from countrycode.calib.calibration import (
    ALPHAS,
    Calibration,
    CalibrationSet,
    conformal_quantile,
    domain_of,
    fit_temperature,
    fst_threshold,
    fst_threshold_units,
    ltt_threshold,
    softmax_t,
    threshold_order,
)
from countrycode.evaluation.harness import Prediction, answered_correct
from countrycode.real.gold import GoldItem
from countrycode.train.dataset import COUNTRY_CLASSES, NO_STATE, STATE_CLASSES, context_text
from countrycode.train.model import combine

TARGETS = list(labels.targets())
_T_INDEX = {t: i for i, t in enumerate(TARGETS)}


def head_labels(g: GoldItem) -> tuple[str | None, str | None]:
    """Gold class per head (None = not usable for that head's temperature)."""
    if len(g.targets) != 1:
        return None, None
    t = next(iter(g.targets))
    if t.startswith("US-"):
        return "US", t
    if t == "US" and g.granularity == "full":
        return "US", NO_STATE
    return t, None


def gold_indices(g: GoldItem, row: np.ndarray) -> set[int]:
    """Targets that must be in a candidate set: all members of ambiguous gold; for
    country-granularity gold the most probable acceptable target."""
    if len(g.targets) > 1:
        return {_T_INDEX[t] for t in g.targets}
    t = next(iter(g.targets))
    if g.granularity == "country":
        ok = [i for i, x in enumerate(TARGETS) if labels.country_of(x) == labels.country_of(t)]
        return {max(ok, key=lambda i: row[i])}
    return {_T_INDEX[t]}


@torch.inference_mode()
def logits(model, tok, items: Sequence[GoldItem], max_len: int, batch: int = 512):
    from countrycode.train.trainer import encode

    model.eval()
    cs, ss = [], []
    for i in range(0, len(items), batch):
        chunk = items[i : i + batch]
        ctx = [context_text(g.field_kind, g.register_country) for g in chunk]
        ids, mask = encode(tok, ctx, [g.text for g in chunk], max_len, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            c, s = model(ids, mask)
        cs.append(c.float().cpu().numpy())
        ss.append(s.float().cpu().numpy())
    return np.concatenate(cs), np.concatenate(ss)


def _temperatures(
    c_logits: np.ndarray, s_logits: np.ndarray, gold: Sequence[GoldItem], min_examples: int = 0
) -> tuple[float, float, int]:
    """One temperature per head; a head with fewer than `min_examples` gold examples keeps T = 1
    (v1.2.1: the curated domain had one US-state example)."""
    c_idx = {c: i for i, c in enumerate(COUNTRY_CLASSES)}
    s_idx = {s: i for i, s in enumerate(STATE_CLASSES)}
    hc = [(i, c_idx[c]) for i, g in enumerate(gold) if (c := head_labels(g)[0]) is not None]
    hs = [(i, s_idx[s]) for i, g in enumerate(gold) if (s := head_labels(g)[1]) is not None]
    tc = (fit_temperature(c_logits[[i for i, _ in hc]], np.array([y for _, y in hc]))
          if len(hc) >= max(min_examples, 1) else 1.0)  # fmt: skip
    ts = (fit_temperature(s_logits[[i for i, _ in hs]], np.array([y for _, y in hs]))
          if len(hs) >= max(min_examples, 1) else 1.0)  # fmt: skip
    return tc, ts, len(hs)


def _scored(
    c_logits: np.ndarray, s_logits: np.ndarray, gold: Sequence[GoldItem], tc: float, ts: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibrated probabilities, max-probability score and answered-correctness per item."""
    probs = combine(softmax_t(c_logits, tc), softmax_t(s_logits, ts))
    top = probs.argmax(1)
    conf = probs.max(1)
    correct = np.array(
        [
            answered_correct(Prediction(TARGETS[int(j)], (), float(p)), g)
            for j, p, g in zip(top, conf, gold, strict=True)
        ]
    )
    return probs, conf, correct


def fit(c_logits: np.ndarray, s_logits: np.ndarray, gold: Sequence[GoldItem]) -> Calibration:
    """v1/v1.1 procedure: temperatures and Bonferroni-LTT thresholds on the same items."""
    tc, ts, n_state = _temperatures(c_logits, s_logits, gold)
    probs, conf, correct = _scored(c_logits, s_logits, gold, tc, ts)
    thresholds = {str(a): ltt_threshold(conf, correct, a) for a in ALPHAS}
    q = conformal_quantile(probs, [gold_indices(g, r) for g, r in zip(gold, probs, strict=True)])
    return Calibration(
        temps=(tc, ts),
        thresholds=thresholds,
        conformal_q=q,
        n_calib=len(gold),
        notes={"calib_accuracy_answer_all": float(correct.mean()), "n_state_temp": n_state},
    )


def fit_split(
    fit_c: np.ndarray,
    fit_s: np.ndarray,
    fit_gold: Sequence[GoldItem],
    cert_c: np.ndarray,
    cert_s: np.ndarray,
    cert_gold: Sequence[GoldItem],
    units: Sequence[Sequence[int]] | None = None,
    min_temp_examples: int = 0,
) -> Calibration:
    """v1.2 procedure (design-02 §1): temperatures and the fixed-sequence order from the fitting
    split; thresholds certified by fixed-sequence testing on the certification split only.
    v1.2.1: with `units` (index groups of one source + normalised key) the test counts units, not
    items (fst_threshold_units)."""
    tc, ts, n_state = _temperatures(fit_c, fit_s, fit_gold, min_temp_examples)
    _, fit_conf, _ = _scored(fit_c, fit_s, fit_gold, tc, ts)
    probs, conf, correct = _scored(cert_c, cert_s, cert_gold, tc, ts)
    n_cert = len(units) if units is not None else len(cert_gold)
    orders = {str(a): threshold_order(fit_conf, n_cert, a) for a in ALPHAS}
    if units is None:
        thresholds = {str(a): fst_threshold(conf, correct, a, orders[str(a)]) for a in ALPHAS}
    else:
        thresholds = {str(a): fst_threshold_units(conf, correct, units, a, orders[str(a)])
                      for a in ALPHAS}  # fmt: skip
    golds = [gold_indices(g, r) for g, r in zip(cert_gold, probs, strict=True)]
    return Calibration(
        temps=(tc, ts),
        thresholds=thresholds,
        conformal_q=conformal_quantile(probs, golds),
        n_calib=len(cert_gold),
        candidate_rule="min_p",
        notes={
            "procedure": "fst",
            "n_fit": len(fit_gold),
            "n_cert": len(cert_gold),
            "n_state_temp": n_state,
            "orders": orders,
            "cert_accuracy_answer_all": float(correct.mean()),
            **(
                {"cert_unit": "source+normalised key", "n_cert_units": len(units)}
                if units is not None
                else {}
            ),
        },
    )


def fit_domains_split(
    fit_c: np.ndarray,
    fit_s: np.ndarray,
    fit_gold: Sequence[GoldItem],
    cert_c: np.ndarray,
    cert_s: np.ndarray,
    cert_gold: Sequence[GoldItem],
    min_items: int = 300,
    units: bool = False,
    min_temp_examples: int = 0,
) -> CalibrationSet:
    """`fit_split` per domain with at least `min_items` certification items and some fitting
    items; a pooled policy when no domain qualifies. `units`: certify over (source, normalised
    key) units (v1.2.1)."""
    from countrycode.text import compact_key

    def unit_groups(gold: Sequence[GoldItem]) -> list[list[int]]:
        out: dict[tuple[str, str], list[int]] = {}
        for i, g in enumerate(gold):
            out.setdefault((g.source, compact_key(g.text)), []).append(i)
        return list(out.values())

    def groups(gold: Sequence[GoldItem]) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for i, g in enumerate(gold):
            if (d := domain_of(g.source)) is not None:
                out.setdefault(d, []).append(i)
        return out

    gf, gc = groups(fit_gold), groups(cert_gold)
    policies = {}
    for d, ci in sorted(gc.items()):
        fi = gf.get(d, [])
        if len(ci) >= min_items and fi:
            cg = [cert_gold[i] for i in ci]
            policies[d] = fit_split(fit_c[fi], fit_s[fi], [fit_gold[i] for i in fi],
                                    cert_c[ci], cert_s[ci], cg,
                                    units=unit_groups(cg) if units else None,
                                    min_temp_examples=min_temp_examples)  # fmt: skip
    if not policies:
        policies["pooled"] = fit_split(fit_c, fit_s, fit_gold, cert_c, cert_s, cert_gold)
    return CalibrationSet(policies)


def fit_domains(
    c_logits: np.ndarray, s_logits: np.ndarray, gold: Sequence[GoldItem], min_items: int = 300
) -> CalibrationSet:
    """One policy per domain with at least `min_items` calib items (others fall back to the
    strictest merge at serving time)."""
    groups: dict[str, list[int]] = {}
    for i, g in enumerate(gold):
        d = domain_of(g.source)
        if d is not None:
            groups.setdefault(d, []).append(i)
    policies = {}
    for d, idx in sorted(groups.items()):
        if len(idx) >= min_items:
            policies[d] = fit(c_logits[idx], s_logits[idx], [gold[i] for i in idx])
    if not policies:
        policies["pooled"] = fit(c_logits, s_logits, gold)
    return CalibrationSet(policies)


def bundle_logits(bundle: Path, items: Sequence[GoldItem], batch: int = 256):
    from countrycode.infer.pipeline import CountryCoder

    coder = CountryCoder.load(bundle)
    cs, ss = [], []
    for i in range(0, len(items), batch):
        chunk = items[i : i + batch]
        c, s = coder.logits(
            [g.text for g in chunk],
            [g.field_kind for g in chunk],
            [g.register_country for g in chunk],
        )
        cs.append(c)
        ss.append(s)
    return np.concatenate(cs), np.concatenate(ss)


def main() -> None:
    from countrycode.evaluation.__main__ import gold_and_lexicon

    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", type=Path, help="PyTorch run (diagnostics)")
    src.add_argument("--bundle", type=Path, help="shipped ONNX bundle (release)")
    ap.add_argument("--pooled", action="store_true", help="v1 behaviour: one pooled policy")
    ap.add_argument("--min-items", type=int, default=300)
    ap.add_argument("--v11", action="store_true", help="fit on the v1.1 calib role")
    ap.add_argument(
        "--v12", action="store_true", help="v1.2: fit on role select, certify on role calib (FST)"
    )
    ap.add_argument("--units", action="store_true",
                    help="v1.2.1: certify over (source, normalised key) units; heads with < 30 "
                    "gold examples keep T = 1")  # fmt: skip
    args = ap.parse_args()
    gold, _ = gold_and_lexicon()
    if args.v12:
        from countrycode.real.roles import with_roles

        roles = with_roles(gold)
        fit_items = [g for g in roles if g.split == "select"]
        cert_items = [g for g in roles if g.split == "calib"]
        if args.bundle:
            fc, fs = bundle_logits(args.bundle, fit_items)
            cc, cs = bundle_logits(args.bundle, cert_items)
            out = args.bundle / "calibration.json"
        else:
            from countrycode.train.trainer import load_run

            model, tok, meta = load_run(args.run)
            fc, fs = logits(model, tok, fit_items, meta["config"]["max_len"])
            cc, cs = logits(model, tok, cert_items, meta["config"]["max_len"])
            out = args.run / "calibration.json"
        cset = fit_domains_split(
            fc, fs, fit_items, cc, cs, cert_items, args.min_items,
            units=args.units, min_temp_examples=30 if args.units else 0,
        )  # fmt: skip
        cset.save(out)
        for d, cal in cset.policies.items():
            print(json.dumps({"domain": d, "n_fit": cal.notes["n_fit"], "n_cert": cal.n_calib,
                              "n_units": cal.notes.get("n_cert_units"), "temps": cal.temps,
                              "thresholds": cal.thresholds}))  # fmt: skip
        return
    if args.v11:
        from countrycode.real.roles import with_roles

        calib = [g for g in with_roles(gold) if g.split == "calib"]
    else:
        calib = [g for g in gold if g.split == "calib"]
    if args.bundle:
        c, s = bundle_logits(args.bundle, calib)
        out = args.bundle / "calibration.json"
    else:
        from countrycode.train.trainer import load_run

        model, tok, meta = load_run(args.run)
        c, s = logits(model, tok, calib, meta["config"]["max_len"])
        out = args.run / "calibration.json"
    if args.pooled:
        cal = fit(c, s, calib)
        cal.save(out)
        print(json.dumps({"temps": cal.temps, "thresholds": cal.thresholds, **cal.notes}))
        return
    cs = fit_domains(c, s, calib, min_items=args.min_items)
    cs.save(out)
    for d, cal in cs.policies.items():
        print(json.dumps({"domain": d, "n": cal.n_calib, "temps": cal.temps,
                          "thresholds": cal.thresholds, "q": cal.conformal_q}))  # fmt: skip


if __name__ == "__main__":
    main()
