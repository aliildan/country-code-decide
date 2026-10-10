import numpy as np
import pytest

from countrycode import labels
from countrycode.calib.calibration import (
    fit_temperature,
    fst_threshold,
    min_answered,
    threshold_order,
)
from countrycode.calib.fit import fit_split
from countrycode.real.gold import GoldItem
from countrycode.train.dataset import COUNTRY_CLASSES, STATE_CLASSES


def test_min_answered_floors():
    assert min_answered(0.01) == 230
    assert min_answered(0.02) == 114
    assert min_answered(0.05) == 45
    assert min_answered(0.01, errors=1) == 388


def test_order_starts_where_floor_is_reachable():
    fit_conf = np.random.default_rng(0).uniform(size=10_000)
    order = threshold_order(fit_conf, n_cert=592, alpha=0.01)
    assert 0 < len(order) <= 40
    assert all(a > b for a, b in zip(order, order[1:], strict=False))
    first_cov = float(np.mean(fit_conf >= order[0]))
    assert 592 * first_cov >= 230  # reachable floor at the start
    assert first_cov < 230 / 592 + 0.79 / 39 + 0.005  # and no later than one grid step


def test_fst_stops_at_first_failure():
    conf = np.r_[np.full(60, 0.95), np.full(30, 0.85), np.full(600, 0.75)]
    correct = np.r_[np.ones(60), np.r_[np.zeros(10), np.ones(20)], np.ones(600)].astype(bool)
    assert fst_threshold(conf, correct, 0.05, [0.9, 0.8, 0.7]) == 0.9


def test_fst_returns_none_when_floor_unreachable():
    conf = np.linspace(0.5, 1.0, 100)
    correct = np.ones(100, dtype=bool)
    order = threshold_order(conf, n_cert=100, alpha=0.01)
    assert fst_threshold(conf, correct, 0.01, order) is None
    assert fst_threshold(conf, correct, 0.01, [0.99, 0.9]) is None


def test_fst_matches_mapie():
    risk_control = pytest.importorskip("mapie.risk_control")
    rng = np.random.default_rng(7)
    conf = rng.uniform(0.3, 1.0, size=3000)
    correct = ~((conf > 0.8) & (rng.uniform(size=3000) < 0.005)) & ~(
        (conf <= 0.8) & (rng.uniform(size=3000) < 0.15)
    )
    order = threshold_order(rng.uniform(0.3, 1.0, size=5000), n_cert=3000, alpha=0.02)
    ours = fst_threshold(conf, correct, 0.02, order)
    bcc = risk_control.BinaryClassificationController(
        predict_function=lambda x: np.c_[1 - x[:, 0], x[:, 0]],
        risk=risk_control.precision,
        target_level=0.98,
        confidence_level=0.9,
        list_predict_params=np.array(order),
        fwer_method="fixed_sequence",
    )
    bcc.calibrate(conf.reshape(-1, 1), correct.astype(int))
    valid = [float(v) for v in np.atleast_1d(bcc.valid_predict_params)]
    assert ours is not None and valid  # a non-trivial certificate on both sides
    assert ours == pytest.approx(min(valid))


def _item(i: int, target: str) -> GoldItem:
    return GoldItem(f"i{i}", "us-co", "f", "country", "US-CO", f"x{i}", 1, "calib", (target,),
                    "human", "reviewed_proposal")  # fmt: skip


def test_fit_split_never_uses_cert_items_for_temperature():
    rng = np.random.default_rng(3)
    countries = [c for c in COUNTRY_CLASSES if c != labels.NONE][:20]
    fit_gold = [_item(i, countries[i % 20]) for i in range(80)]
    cert_gold = [_item(100 + i, countries[i % 20]) for i in range(80)]
    fit_c = rng.normal(size=(80, len(COUNTRY_CLASSES)))
    cert_c = 5 * rng.normal(size=(80, len(COUNTRY_CLASSES)))
    s = np.zeros((80, len(STATE_CLASSES)))
    cal = fit_split(fit_c, s, fit_gold, cert_c, s, cert_gold)
    y = np.array([COUNTRY_CLASSES.index(g.targets[0]) for g in fit_gold])
    assert cal.temps[0] == pytest.approx(fit_temperature(fit_c, y))
    assert cal.notes["n_fit"] == 80 and cal.notes["n_cert"] == 80
    assert cal.notes["procedure"] == "fst"
