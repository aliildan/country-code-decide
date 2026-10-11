import importlib.util
import sys
from pathlib import Path

from countrycode.evaluation.harness import Prediction
from countrycode.real.gold import GoldItem

_spec = importlib.util.spec_from_file_location("cascade", Path("scripts/cascade.py"))
assert _spec and _spec.loader
cascade = importlib.util.module_from_spec(_spec)
sys.modules["cascade"] = cascade
_spec.loader.exec_module(cascade)


def g(i, target, source="us-ny"):
    return GoldItem(f"i{i}", source, "f", "country", "US-NY", f"t{i}", 1, "test_v12", (target,),
                    "x", "y")  # fmt: skip


def P(target, conf=0.9):
    return Prediction(target, (target,) if target else (), conf)


def test_cascade_counts_each_stage_and_the_residual_by_source():
    items = [g(0, "AT"), g(1, "DE"), g(2, "FR", "us-or"), g(3, "GB", "us-or"), g(4, "IT")]
    lexicon = {"i0": P("AT"), "i1": P("FR"), "i2": P(None), "i3": P(None), "i4": P(None)}
    model = {"i0": P("AT"), "i1": P("DE"), "i2": P("FR"), "i3": P("US"), "i4": P(None)}
    r = cascade.cascade(items, lexicon, model)
    assert r["n"] == 5
    assert r["lexicon"] == {"answered": 2, "errors": 1}
    assert r["residual"] == {"n": 3, "model_answered": 2, "model_errors": 1}
    assert r["cascade"] == {"answered": 4, "errors": 2}
    assert r["model_alone"] == {"answered": 4, "errors": 1}
    assert r["residual_by_source"]["us-or"] == {"n": 2, "model_answered": 2, "model_errors": 1}
    assert r["residual_by_source"]["us-ny"] == {"n": 1, "model_answered": 0, "model_errors": 0}
