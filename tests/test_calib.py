import numpy as np

from countrycode.calib.calibration import (
    Calibration,
    conformal_quantile,
    decide,
    fit_temperature,
    ltt_threshold,
    prediction_set,
)


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    n, k = 2000, 5
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += 1.0
    t = fit_temperature(logits * 4.0, y)  # sharpened 4x → temperature near 4
    assert 2.5 < t < 6.0


def test_ltt_threshold_controls_risk():
    rng = np.random.default_rng(1)
    conf = rng.uniform(0, 1, 50_000)
    correct = rng.uniform(0, 1, 50_000) < conf  # well calibrated (5k is too few to certify 5 %)
    lam = ltt_threshold(conf, correct, alpha=0.05, delta=0.1)
    assert lam is not None and lam >= 0.9
    ans = conf >= lam
    assert (~correct[ans]).mean() <= 0.05


def test_ltt_none_when_impossible():
    conf = np.full(100, 0.99)
    correct = np.zeros(100, dtype=bool)
    assert ltt_threshold(conf, correct, alpha=0.01) is None


def test_conformal_set_and_decide():
    probs = np.array([[0.6, 0.3, 0.1], [0.9, 0.05, 0.05]])
    q = conformal_quantile(probs, [{0}, {1}], coverage=0.9)
    assert 0 < q <= 1.0
    assert prediction_set(np.array([0.5, 0.45, 0.05]), 0.9) == [0, 1]
    cal = Calibration(temps=(1.0, 1.0), thresholds={"0.05": 0.8}, conformal_q=0.9)
    targets = ("CD", "CG", "AT")
    p = decide(np.array([0.5, 0.45, 0.05]), cal, 0.05, targets)
    assert p.target is None and set(p.candidates) == {"CD", "CG"}
    p = decide(np.array([0.05, 0.05, 0.9]), cal, 0.05, targets)
    assert p.target == "AT"


def test_min_p_candidates():
    from countrycode.calib.calibration import candidates

    cal = Calibration(candidate_rule="min_p")
    row = np.array([0.6, 0.3, 0.06, 0.04])
    assert candidates(row, cal) == [0, 1, 2]
    assert candidates(np.array([0.5, 0.2, 0.15, 0.1, 0.05]), cal) == [0, 1, 2]  # at most 3


def test_candidates_never_empty():
    from countrycode.calib.calibration import candidates

    cal = Calibration(candidate_rule="min_p")
    row = np.full(30, 1 / 30)
    row[7] += 0.001
    assert candidates(row, cal) == [7]


def test_old_calibration_keeps_conformal_candidates(tmp_path):
    import json

    old = {"temps": [1.0, 1.0], "thresholds": {"0.05": 0.99}, "conformal_q": 0.95,
           "delta": 0.1, "n_calib": 10, "notes": {}}  # fmt: skip
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(old))
    cal = Calibration.load(path)
    assert cal.candidate_rule == "conformal"
    row = np.array([0.5, 0.3, 0.1, 0.06, 0.04])
    pred = decide(row, cal, 0.05, ["A", "B", "C", "D", "E"])
    assert pred.target is None and pred.candidates == ("A", "B", "C", "D")  # mass ≥ 0.95


def test_strictest_policy_keeps_the_v12_candidate_rule():
    from countrycode.calib.calibration import Calibration, CalibrationSet

    a = Calibration(temps=(1.0, 1.0), thresholds={"0.05": 0.5}, candidate_rule="min_p")
    b = Calibration(temps=(2.0, 1.0), thresholds={"0.05": 0.7}, candidate_rule="min_p")
    s = CalibrationSet({"register": a, "curated": b}).for_domain(None)
    assert (s.candidate_rule, s.candidate_min_p, s.candidate_max) == ("min_p", 0.05, 3)
    assert s.thresholds["0.05"] == 0.7 and s.temps == (2.0, 1.0)


def test_strictest_refuses_policies_with_different_candidate_rules():
    import pytest

    from countrycode.calib.calibration import Calibration, CalibrationSet

    a = Calibration(temps=(1.0, 1.0), thresholds={"0.05": 0.5}, candidate_rule="min_p")
    b = Calibration(temps=(1.0, 1.0), thresholds={"0.05": 0.5})  # conformal
    with pytest.raises(ValueError, match="candidate rule"):
        CalibrationSet({"register": a, "curated": b}).for_domain(None)
