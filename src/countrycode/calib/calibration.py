"""Calibration and selective prediction (plan-04 Task 5, design-01 §6).

- one temperature per head, fitted by NLL on real *calib* items (never test);
- Learn-then-Test thresholds λ_α on the final max probability: answer only when conf >= λ_α,
  with risk <= α certified at confidence 1-δ (Bonferroni over a fixed grid; ported from
  SignRule-Decide `calib/calibration.py`);
- split-conformal candidate sets for the review queue: the smallest top-probability set whose
  mass reaches the calibrated quantile.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path

import numpy as np

from countrycode.evaluation.harness import Prediction
from countrycode.paths import CONFIGS

ALPHAS = (0.01, 0.02, 0.05)
LAMBDA_GRID = tuple(
    [round(0.5 + 0.05 * i, 3) for i in range(9)]
    + [0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999]
)


def softmax_t(logits: np.ndarray, t: float) -> np.ndarray:
    z = logits / t
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return 1.0

    def loss(log_t: float) -> float:
        p = softmax_t(logits, math.exp(log_t))
        return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())

    from scipy.optimize import minimize_scalar  # fitting only: serving stays scipy-free

    res = minimize_scalar(loss, bounds=(math.log(0.05), math.log(20.0)), method="bounded")
    return float(math.exp(res.x))


def _binom_cdf(k: int, n: int, p: float) -> float:
    from scipy.stats import binom  # fitting only: serving stays scipy-free

    return float(binom.cdf(k, n, p))


def ltt_threshold(
    conf: np.ndarray,
    correct: np.ndarray,
    alpha: float,
    delta: float = 0.1,
    grid: Sequence[float] = LAMBDA_GRID,
) -> float | None:
    """Lowest λ whose H0 'risk(λ) > α' is rejected at level δ/|grid| (binomial tail)."""
    level = delta / len(grid)
    certified = []
    for lam in grid:
        answered = conf >= lam
        n = int(answered.sum())
        if n == 0:
            continue
        errors = int((~correct[answered]).sum())
        if _binom_cdf(errors, n, alpha) <= level:
            certified.append(lam)
    return min(certified) if certified else None


def min_answered(alpha: float, delta: float = 0.1, errors: int = 0) -> int:
    """Smallest number of answered items with `errors` errors that rejects 'risk > α' at δ."""
    n = errors + 1
    while _binom_cdf(errors, n, alpha) > delta:
        n += 1
    return n


def threshold_order(
    fit_conf: np.ndarray, n_cert: int, alpha: float, delta: float = 0.1, steps: int = 40
) -> list[float]:
    """Fixed-sequence order, strict → loose, built from scores of the fitting split only: λ at
    coverages 0.20 … 0.99 of `fit_conf`, starting at the first λ whose expected answered count on
    the certification split (coverage × n_cert) reaches the zero-error floor."""
    floor = min_answered(alpha, delta)
    out: list[float] = []
    for cov in np.linspace(0.20, 0.99, steps):
        if cov * n_cert < floor:
            continue
        lam = float(np.quantile(fit_conf, 1.0 - cov))
        if not out or lam < out[-1]:
            out.append(lam)
    return out


def fst_threshold(
    conf: np.ndarray,
    correct: np.ndarray,
    alpha: float,
    order: Sequence[float],
    delta: float = 0.1,
) -> float | None:
    """Fixed-sequence testing (LTT): walk `order` strict → loose, reject H0 'risk(λ) > α' with
    the exact binomial tail at level δ, stop at the first non-rejection; the loosest rejected λ."""
    best = None
    for lam in order:
        answered = conf >= lam
        n = int(answered.sum())
        errors = int((~correct[answered]).sum())
        if n == 0 or _binom_cdf(errors, n, alpha) > delta:
            break
        best = float(lam)
    return best


def fst_threshold_units(
    conf: np.ndarray,
    correct: np.ndarray,
    units: Sequence[Sequence[int]],
    alpha: float,
    order: Sequence[float],
    delta: float = 0.1,
) -> float | None:
    """Fixed-sequence testing over units instead of items (v1.2.1). Items sharing a source and a
    normalised key are one unit: the binomial test needs independent draws, and the same string in
    several fields of a register is not one. A unit is answered when any member is answered and
    wrong when any answered member is wrong."""
    best = None
    for lam in order:
        answered = conf >= lam
        n = errors = 0
        for u in units:
            hit = [i for i in u if answered[i]]
            if hit:
                n += 1
                errors += int(not all(correct[i] for i in hit))
        if n == 0 or _binom_cdf(errors, n, alpha) > delta:
            break
        best = float(lam)
    return best


def _mass_to_cover(row: np.ndarray, gold: set[int]) -> float:
    order = np.argsort(-row)
    mass = 0.0
    need = set(gold)
    for j in order:
        mass += float(row[j])
        need.discard(int(j))
        if not need:
            return mass
    return 1.0


def conformal_quantile(
    probs: np.ndarray, gold: Sequence[set[int]], coverage: float = 0.95
) -> float:
    scores = np.array([_mass_to_cover(r, g) for r, g in zip(probs, gold, strict=True)])
    n = len(scores)
    if n == 0:
        return 1.0
    k = min(n, math.ceil((n + 1) * coverage))
    return float(np.sort(scores)[k - 1])


def prediction_set(row: np.ndarray, q: float, max_size: int = 5) -> list[int]:
    order = np.argsort(-row)
    out, mass = [], 0.0
    for j in order[:max_size]:
        out.append(int(j))
        mass += float(row[j])
        if mass >= q - 1e-12:
            break
    return out


@dataclass
class Calibration:
    temps: tuple[float, float] = (1.0, 1.0)  # country head, state head
    thresholds: dict[str, float | None] = field(default_factory=dict)  # str(alpha) -> λ
    conformal_q: float = 0.95
    delta: float = 0.1
    n_calib: int = 0
    notes: dict = field(default_factory=dict)
    # v1.2: "min_p" = codes with probability ≥ candidate_min_p, at most candidate_max
    # (uncertified suggestions); v1/v1.1 files have no such key and keep "conformal"
    candidate_rule: str = "conformal"
    candidate_min_p: float = 0.05
    candidate_max: int = 3

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: Path) -> Calibration:
        d = json.loads(path.read_text())
        d["temps"] = tuple(d["temps"])
        return cls(**d)


def candidates(row: np.ndarray, cal: Calibration) -> list[int]:
    """Candidate indices: the conformal set (v1/v1.1) or, for `min_p`, the codes with
    probability ≥ candidate_min_p (at most candidate_max; never empty)."""
    if cal.candidate_rule != "min_p":
        return prediction_set(row, cal.conformal_q)
    order = np.argsort(-row)[: cal.candidate_max]
    out = [int(j) for j in order if row[j] >= cal.candidate_min_p]
    return out or [int(order[0])]


def decide(row: np.ndarray, cal: Calibration, alpha: float, targets: Sequence[str]) -> Prediction:
    """Answer the top target when its probability clears λ_α, else abstain with candidates."""
    j = int(np.argmax(row))
    conf = float(row[j])
    lam = cal.thresholds.get(str(alpha))
    cands = tuple(targets[i] for i in candidates(row, cal))
    if lam is not None and conf >= lam:
        return Prediction(targets[j], cands, conf)
    return Prediction(None, cands, conf)


DOMAINS_PATH = CONFIGS / "domains.yaml"


@cache
def _domain_map() -> dict[str, str]:
    import yaml

    raw = yaml.safe_load(DOMAINS_PATH.read_text(encoding="utf-8")) if DOMAINS_PATH.exists() else {}
    return {src: dom for dom, sources in (raw or {}).items() for src in sources}


def domain_of(source: str | None) -> str | None:
    """Calibration domain of a source ("register", "curated"); None when unknown."""
    return _domain_map().get(source) if source else None


def _strictest(policies: list[Calibration]) -> Calibration:
    """For an unknown domain: softest temperatures, highest thresholds (a threshold one policy
    could not certify stays uncertified), widest conformal quantile."""
    temps = (max(p.temps[0] for p in policies), max(p.temps[1] for p in policies))
    alphas = {a for p in policies for a in p.thresholds}
    thresholds: dict[str, float | None] = {}
    for a in sorted(alphas):
        vals = [p.thresholds.get(a) for p in policies]
        thresholds[a] = (
            None if any(v is None for v in vals) else max(v for v in vals if v is not None)
        )
    rules = {(p.candidate_rule, p.candidate_min_p, p.candidate_max) for p in policies}
    if len(rules) != 1:
        raise ValueError(f"policies differ in their candidate rule: {sorted(map(str, rules))}")
    rule, min_p, cmax = rules.pop()
    return Calibration(
        temps=temps,
        thresholds=thresholds,
        conformal_q=max(p.conformal_q for p in policies),
        n_calib=sum(p.n_calib for p in policies),
        notes={"strictest_of": len(policies)},
        candidate_rule=rule,  # the policies' shared candidate rule (v1.2: min_p)
        candidate_min_p=min_p,
        candidate_max=cmax,
    )


@dataclass
class CalibrationSet:
    policies: dict[str, Calibration]

    def for_domain(self, domain: str | None) -> Calibration:
        if domain in self.policies:
            return self.policies[domain]  # type: ignore[index]
        return _strictest(list(self.policies.values()))

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"domains": {k: asdict(v) for k, v in self.policies.items()}}, indent=1)
        )

    @classmethod
    def load(cls, path: Path) -> CalibrationSet:
        d = json.loads(path.read_text())
        if "domains" not in d:  # v1 file: one policy for every domain
            cal = Calibration.load(path)
            return cls({"register": cal, "curated": cal})
        out = {}
        for k, v in d["domains"].items():
            v["temps"] = tuple(v["temps"])
            out[k] = Calibration(**v)
        return cls(out)
