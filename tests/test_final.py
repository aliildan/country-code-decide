import numpy as np

from countrycode import labels
from countrycode.calib.calibration import Calibration
from countrycode.evaluation.final import CalibratedPredictor, judge_hypotheses, judge_v11
from countrycode.real.gold import GoldItem

T = list(labels.targets())
G = GoldItem("i", "s", "f", "country", "US", "x", 1, "dev", ("AT",), "x", "y")


class FakeProbs:
    def __init__(self, rows):
        self.rows = rows

    def probs(self, items):
        return np.array(self.rows[: len(items)])


def test_calibrated_predictor_abstains_below_threshold():
    r1 = np.zeros(len(T))
    r1[T.index("AT")] = 0.99
    r1[T.index("DE")] = 0.01
    r2 = np.zeros(len(T))
    r2[T.index("CD")] = 0.5
    r2[T.index("CG")] = 0.5
    cal = Calibration(thresholds={"0.01": 0.95}, conformal_q=0.95)
    p = CalibratedPredictor(FakeProbs([r1, r2]), cal, alpha=0.01, name="m@1%").predict([G, G])
    assert p[0].target == "AT" and p[1].target is None and set(p[1].candidates) == {"CD", "CG"}
    allp = CalibratedPredictor(FakeProbs([r1, r2]), cal, alpha=None, name="m").predict([G, G])
    assert allp[1].target in ("CD", "CG")


def test_judge_hypotheses():
    reports = {
        "model": {"overall": {"accuracy": 0.80}, "latency_ms_per_item": 2.0},
        "model@0.01": {
            "overall": {"answered": 0.6, "risk_answered": 0.015, "none_false_accept": 0.01}
        },
        "lexicon_fuzzy": {"overall": {"accuracy": 0.70}},
        "countrynames": {"overall": {"accuracy": 0.50}},
        "llm_x": {"overall": {"accuracy": 0.81}, "latency_ms_per_item": 100.0},
    }
    h = judge_hypotheses(reports, cpu_p50_ms=4.0, h4_dev_gain=0.061)
    assert h["H1"]["pass"] and h["H2"]["pass"] and h["H3"]["pass"] and h["H4"]["pass"]
    assert h["H5"]["pass"] and h["H6"]["pass"]
    reports["model@0.01"]["overall"]["risk_answered"] = 0.03
    assert not judge_hypotheses(reports, 4.0, 0.061)["H2"]["pass"]


def test_h3_is_vacuous_when_nothing_is_answered():
    reports = {
        "model": {"overall": {"accuracy": 0.6}, "latency_ms_per_item": 2.0},
        "model@0.01": {
            "overall": {"answered": 0.0, "risk_answered": float("nan"), "none_false_accept": 0.0}
        },
        "lexicon_fuzzy": {"overall": {"accuracy": 0.6}},
    }
    h = judge_hypotheses(reports, cpu_p50_ms=4.0, h4_dev_gain=0.1)
    assert h["H3"]["pass"] is False and h["H3"]["vacuous"] is True


def v11_reports(none_items=40):
    return {
        "model": {"overall": {"accuracy": 0.90}},
        "model@0.01": {
            "overall": {"answered": 0.7, "risk_answered": 0.012, "none_items": none_items,
                        "none_false_accept": 0.0}
        },
        "v1_model": {"overall": {"accuracy": 0.85}},
        "lexicon_fuzzy": {"overall": {"accuracy": 0.84}},
        "llm_x": {"overall": {"accuracy": 0.91}, "latency_ms_per_item": 200.0},
    }  # fmt: skip


def test_judge_v11_register_criteria():
    h = judge_v11(v11_reports(), cpu_p50_ms=4.0)
    assert all(h[k]["pass"] for k in ("R1", "R2", "R3", "R4", "R5", "R6"))
    r = v11_reports()
    r["model@0.01"]["overall"]["answered"] = 0.55  # R2 needs ≥ 60 % answered
    r["v1_model"]["overall"]["accuracy"] = 0.88  # R4 needs v1 + 3 pp
    h = judge_v11(r, cpu_p50_ms=4.0)
    assert not h["R2"]["pass"] and not h["R4"]["pass"] and h["R3"]["pass"]


def test_judge_v11_r3_not_testable_without_enough_none_items():
    h = judge_v11(v11_reports(none_items=12), cpu_p50_ms=4.0)
    assert h["R3"]["pass"] is None and h["R3"]["none_items"] == 12
    r = v11_reports()
    del r["v1_model"]
    assert judge_v11(r, cpu_p50_ms=4.0)["R4"]["pass"] is None


def test_v1_gets_its_own_register_convention():
    from dataclasses import replace

    from countrycode.evaluation.final import LegacyRegister

    seen = []

    class Spy:
        def probs(self, items):
            seen.extend(g.register_country for g in items)
            return np.zeros((len(items), len(T)))

    items = [replace(G, register_country="US-CO"), replace(G, register_country="GB")]
    LegacyRegister(Spy()).probs(items)
    assert seen == ["US", "GB"]


def v12_reports(answered=0.7, risk=0.01, n=300):
    return {
        "model": {"overall": {"accuracy": 0.90, "n": n}},
        "model@0.02": {"overall": {"answered": answered, "risk_answered": risk, "n": n}},
        "model@0.01": {"overall": {"answered": 0.4, "risk_answered": 0.005, "n": n}},
        "lexicon_fuzzy": {"overall": {"accuracy": 0.80}},
        "e5_knn": {"overall": {"accuracy": 0.84}},
    }


TRIM = {
    "size_mb": 60.0,
    "tokens_identical": 1.0,
    "select_accuracy_untrimmed": 0.656,
    "select_accuracy": 0.655,
}
CERT = {"0.01": None, "0.02": 0.8, "0.05": 0.6}


def test_judge_v12():
    from countrycode.evaluation.final import judge_v12

    h = judge_v12(v12_reports(), cpu_p50_ms=4.0, trim=TRIM, cert=CERT)
    assert all(h[k]["pass"] for k in ("P1", "P2", "P3", "P4"))
    assert h["A1"]["pass"] is None and h["A1"]["certified"] is False
    # 10 errors among 200 answered contradict a 2 % certificate
    h = judge_v12(v12_reports(answered=200 / 300, risk=10 / 200), 4.0, TRIM, CERT)
    assert h["P1"]["pass"] is False and h["P1"]["contradicted"] is True
    assert judge_v12(v12_reports(answered=0.55), 4.0, TRIM, CERT)["P2"]["pass"] is False
    assert (
        judge_v12(v12_reports(), 4.0, {**TRIM, "select_accuracy": 0.65}, CERT)["P3"]["pass"]
        is False
    )
    assert judge_v12(v12_reports(), 4.0, {**TRIM, "size_mb": 81.0}, CERT)["P3"]["pass"] is False
    assert judge_v12(v12_reports(), 4.0, TRIM, {**CERT, "0.02": None})["P1"]["pass"] is False
    worse = v12_reports()
    worse["model"]["overall"]["accuracy"] = 0.88
    assert judge_v12(worse, 4.0, TRIM, CERT)["P4"]["pass"] is False
