"""Selective-prediction metrics with optional item weights (type-level = weight 1, frequency-
weighted = string count). Ported from SignRule-Decide `evaluation/metrics.py`. Pure numpy."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _w(weights: np.ndarray | None, n: int) -> np.ndarray:
    return np.ones(n) if weights is None else np.asarray(weights, dtype=float)


def risk_coverage(
    conf: np.ndarray, correct: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Coverage and selective risk when answering the k most confident items, k = 1..n."""
    w = _w(weights, len(conf))
    order = np.argsort(-conf, kind="stable")
    wo, co = w[order], correct[order]
    answered = np.cumsum(wo)
    errs = np.cumsum(wo * ~co)
    return answered / w.sum(), errs / answered


def aurc(conf: np.ndarray, correct: np.ndarray, weights: np.ndarray | None = None) -> float:
    if len(conf) == 0:
        return float("nan")
    cov, risk = risk_coverage(conf, correct, weights)
    dc = np.diff(np.concatenate([[0.0], cov]))
    return float((risk * dc).sum())


def coverage_at_risk(
    conf: np.ndarray, correct: np.ndarray, alpha: float, weights: np.ndarray | None = None
) -> float:
    """Largest coverage whose empirical selective risk is <= alpha (0 if none). Ties in
    confidence are answered together (a threshold cannot split them)."""
    if len(conf) == 0:
        return float("nan")
    cov, risk = risk_coverage(conf, correct, weights)
    order = np.argsort(-conf, kind="stable")
    cs = conf[order]
    last_of_tie = np.append(cs[1:] != cs[:-1], True)
    ok = np.where((risk <= alpha + 1e-12) & last_of_tie)[0]
    return float(cov[ok].max()) if len(ok) else 0.0


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    n = len(conf)
    if n == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        g = idx == b
        if g.any():
            total += g.sum() / n * abs(correct[g].mean() - conf[g].mean())
    return float(total)


def bootstrap_ci(
    stat: Callable[[np.ndarray], float], n: int, n_boot: int = 1000, seed: int = 13
) -> tuple[float, float]:
    """95 % percentile CI of `stat(indices)` over item resamples."""
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = [stat(rng.integers(0, n, n)) for _ in range(n_boot)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
