import numpy as np
import pytest

from countrycode import labels
from countrycode.calib.calibration import (
    fit_temperature,
    fst_threshold,
    fst_threshold_units,
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


def test_units_count_a_repeated_key_once():
    # 200 distinct keys answered correctly; one wrong key repeated in 4 fields; the item-level test
    # sees 4 errors in 204, the unit-level one 1 error in 201
    conf = np.full(204, 0.9)
    correct = np.r_[np.ones(200), np.zeros(4)].astype(bool)
    units = [[i] for i in range(200)] + [[200, 201, 202, 203]]
    lam = fst_threshold_units(conf, correct, units, 0.05, [0.85])
    assert lam == 0.85
    assert fst_threshold(conf, correct, 0.05, [0.85]) == 0.85


def test_units_remove_the_certificate_that_duplicates_create():
    # 60 correct keys each repeated 3 times (180 items) + 2 wrong singletons: item level 2/182
    # certifies 5 %; unit level 2/62 cannot
    conf = np.full(182, 0.9)
    correct = np.r_[np.ones(180), np.zeros(2)].astype(bool)
    units = [[3 * k, 3 * k + 1, 3 * k + 2] for k in range(60)] + [[180], [181]]
    assert fst_threshold(conf, correct, 0.05, [0.85]) == 0.85
    assert fst_threshold_units(conf, correct, units, 0.05, [0.85]) is None


def test_a_unit_is_answered_when_any_member_is_and_wrong_when_any_answered_member_is():
    # unit 0: one member answered and right, one unanswered and wrong -> answered, right
    conf = np.r_[0.95, 0.10, np.full(120, 0.95)]
    correct = np.r_[True, False, np.ones(120, dtype=bool)]
    units = [[0, 1]] + [[i] for i in range(2, 122)]
    assert fst_threshold_units(conf, correct, units, 0.05, [0.9]) == 0.9


def test_fit_split_with_units_records_them():
    rng = np.random.default_rng(3)
    countries = [c for c in COUNTRY_CLASSES if c != labels.NONE][:20]
    fit_gold = [_item(i, countries[i % 20]) for i in range(80)]
    cert_gold = [_item(100 + i, countries[i % 20]) for i in range(80)]
    c = rng.normal(size=(80, len(COUNTRY_CLASSES)))
    s = np.zeros((80, len(STATE_CLASSES)))
    units = [list(range(k, k + 4)) for k in range(0, 80, 4)]
    cal = fit_split(c, s, fit_gold, c, s, cert_gold, units=units)
    assert cal.notes["cert_unit"] == "source+normalised key"
    assert cal.notes["n_cert"] == 80 and cal.notes["n_cert_units"] == 20


def test_a_head_with_few_gold_examples_keeps_temperature_one():
    rng = np.random.default_rng(4)
    countries = [c for c in COUNTRY_CLASSES if c not in (labels.NONE, "US")][:20]
    gold = [_item(i, countries[i % 20]) for i in range(80)] + [_item(80, "US-CA")]
    c = rng.normal(size=(81, len(COUNTRY_CLASSES)))
    s = 3 * rng.normal(size=(81, len(STATE_CLASSES)))
    assert fit_split(c, s, gold, c, s, gold).temps[1] != 1.0  # v1.2 behaviour, reproducible
    cal = fit_split(c, s, gold, c, s, gold, min_temp_examples=30)
    assert cal.temps[1] == 1.0  # one US-state example cannot fit a temperature


def test_fit_domains_split_groups_units_by_source_and_normalised_key():
    from countrycode.calib.fit import fit_domains_split

    rng = np.random.default_rng(5)
    countries = [c for c in COUNTRY_CLASSES if c != labels.NONE][:20]
    fit_gold = [_item(i, countries[i % 20]) for i in range(320)]
    # 320 cert items, 80 distinct texts ("Österreich" and "OSTERREICH " share a key) in us-co
    cert_gold = [GoldItem(f"c{i}", "us-co", f"f{i % 4}", "country", "US-CO",
                          ("Österreich" if i % 4 else "OSTERREICH ") + f" {i // 4}", 1, "calib",
                          (countries[(i // 4) % 20],), "human", "reviewed_proposal")
                 for i in range(320)]  # fmt: skip
    c = rng.normal(size=(320, len(COUNTRY_CLASSES)))
    s = np.zeros((320, len(STATE_CLASSES)))
    cset = fit_domains_split(c, s, fit_gold, c, s, cert_gold, units=True)
    reg = cset.policies["register"]
    assert reg.notes["n_cert"] == 320 and reg.notes["n_cert_units"] == 80
