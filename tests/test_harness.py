import pytest

from countrycode.evaluation.harness import Prediction, evaluate, item_correct
from countrycode.real.gold import GoldItem


def g(text, targets, count=1, source="us-co", kind="country", split="dev"):
    return GoldItem(
        f"id-{text}",
        source,
        "f",
        kind,
        "US",
        text,
        count,
        split,
        tuple(targets),
        "reference_list",
        "auto_exact",
    )


class Fixed:
    name = "fixed"

    def __init__(self, preds):
        self.preds = preds

    def predict(self, items):
        return [self.preds[i.text] for i in items]


GOLD = [
    g("Austria", ["AT"], count=10),
    g("Autria", ["AT"], count=1),
    g("Congo", ["CD", "CG"]),
    g("n/a", ["NONE"], count=5),
    g("Норвегия", ["NO"]),
]
PREDS = {
    "Austria": Prediction("AT", ("AT",), 0.99),
    "Autria": Prediction("AU", ("AU",), 0.6),
    "Congo": Prediction(None, ("CD", "CG"), 0.5),
    "n/a": Prediction("AT", ("AT",), 0.7),
    "Норвегия": Prediction("NO", ("NO",), 0.95),
}


def test_item_correct_rules():
    assert item_correct(PREDS["Austria"], GOLD[0])
    assert not item_correct(PREDS["Autria"], GOLD[1])
    assert item_correct(PREDS["Congo"], GOLD[2])  # abstain with the exact set
    assert not item_correct(Prediction("CD", ("CD",), 0.9), GOLD[2])  # single code on a set


def test_evaluate_type_and_weighted(tmp_path):
    rep = evaluate(Fixed(PREDS), GOLD, "dev", out_dir=tmp_path)
    assert rep["overall"]["accuracy"] == pytest.approx(3 / 5)
    assert rep["overall"]["accuracy_weighted"] == pytest.approx(12 / 18)
    assert rep["overall"]["none_false_accept"] == pytest.approx(1.0)
    assert rep["overall"]["n"] == 5
    assert "script=CYRILLIC" in rep["slices"] and "source=us-co" in rep["slices"]
    assert (tmp_path / "dev" / "fixed.json").exists()


def test_coverage_at_risk():
    rep = evaluate(Fixed(PREDS), GOLD, "dev", out_dir=None)
    # answered by confidence: Austria .99 ok, Норвегия .95 ok, n/a .7 wrong → coverage 2/5 at 0 risk
    assert rep["overall"]["coverage_at_risk"]["0.01"] == pytest.approx(2 / 5)


def test_refuses_test_without_flag(tmp_path):
    with pytest.raises(PermissionError):
        evaluate(Fixed(PREDS), [g("Austria", ["AT"], split="test")], "test", out_dir=tmp_path)


def test_country_level_accuracy_credits_country_only_answers():
    gold = [g("Deleware", ["US-DE"], kind="jurisdiction")]
    rep = evaluate(Fixed({"Deleware": Prediction("US", ("US",), 0.9)}), gold, "dev", out_dir=None)
    assert rep["overall"]["accuracy"] == 0.0 and rep["overall"]["country_accuracy"] == 1.0


def test_country_granularity_gold_accepts_states():
    gold = GoldItem(
        "i",
        "gbif",
        "f",
        "country",
        "unknown",
        "TEXAS",
        1,
        "dev",
        ("US",),
        "reference_list",
        "preset",
        "country",
    )
    assert item_correct(Prediction("US-TX", ("US-TX",), 0.9), gold)
    assert not item_correct(Prediction("MX", ("MX",), 0.9), gold)


def test_none_gold_accepts_abstain_or_none():
    gold = g("n/a", ["NONE"])
    assert item_correct(Prediction(None, (), 0.0), gold)
    assert item_correct(Prediction("NONE", ("NONE",), 0.9), gold)
    assert not item_correct(Prediction("AT", ("AT",), 0.9), gold)


def test_set_compatible_accuracy():
    gold = [g("Congo", ["CD", "CG"])]
    rep = evaluate(Fixed({"Congo": Prediction("CG", ("CG",), 0.9)}), gold, "dev", out_dir=None)
    assert rep["overall"]["accuracy"] == 0.0
    assert rep["overall"]["set_compatible_accuracy"] == 1.0


def test_noise_slices_present():
    gold = [g("Autria", ["AT"])]
    pred = Fixed({"Autria": Prediction("AT", ("AT",), 0.9)})
    rep = evaluate(pred, gold, "dev", out_dir=None, kinds={"id-Autria": "near"})
    assert rep["slices"]["noise=near"]["n"] == 1
