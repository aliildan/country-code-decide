import json

from countrycode.calib.calibration import Calibration, CalibrationSet, domain_of


def test_domain_comes_from_source():
    assert domain_of("us-co") == "register" and domain_of("gbif") == "curated"
    assert domain_of("something-else") is None and domain_of(None) is None


def test_for_domain_and_strictest(tmp_path):
    reg = Calibration(temps=(1.2, 1.0), thresholds={"0.01": 0.9, "0.05": 0.6}, conformal_q=0.95)
    cur = Calibration(temps=(2.4, 0.5), thresholds={"0.01": None, "0.05": 0.85}, conformal_q=0.99)
    cs = CalibrationSet({"register": reg, "curated": cur})
    assert cs.for_domain("register") is reg
    s = cs.for_domain(None)
    assert s.thresholds == {"0.01": None, "0.05": 0.85}
    assert s.temps == (2.4, 1.0) and s.conformal_q == 0.99
    p = tmp_path / "c.json"
    cs.save(p)
    back = CalibrationSet.load(p)
    assert back.for_domain("register").thresholds["0.01"] == 0.9


def test_load_v1_single_calibration(tmp_path):
    p = tmp_path / "c.json"
    Calibration(temps=(2.0, 1.0), thresholds={"0.05": 0.8}).save(p)
    cs = CalibrationSet.load(p)
    assert cs.for_domain("register").thresholds == {"0.05": 0.8}
    assert json.loads(p.read_text())["temps"] == [2.0, 1.0]
