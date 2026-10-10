import importlib.util
import sys
from pathlib import Path

import onnx

from countrycode.calib.calibration import Calibration
from countrycode.infer.pipeline import CountryCoder, make_bundle
from tests.test_pipeline import tiny_model, tiny_tokenizer

_spec = importlib.util.spec_from_file_location("web_parity", Path("scripts/web_parity.py"))
assert _spec and _spec.loader
web_parity = importlib.util.module_from_spec(_spec)
sys.modules["web_parity"] = web_parity
_spec.loader.exec_module(web_parity)


def _bundle(tmp_path):
    tiny_tokenizer(tmp_path / "tokenizer.json")
    cal = Calibration(temps=(1.0, 1.0), thresholds={"0.02": 0.0, "0.05": 0.0}, conformal_q=0.9)
    return make_bundle(
        tiny_model(), tmp_path / "tokenizer.json", cal, tmp_path / "b", meta={"max_len": 32}
    )


def test_browser_copy_computes_matmuls_in_fp32(tmp_path):
    b = _bundle(tmp_path)
    out = web_parity.browser_copy(b, tmp_path / "web")
    levels = [
        a.i
        for n in onnx.load(str(out / "model.int8.onnx")).graph.node
        if n.op_type == "MatMulNBits"
        for a in n.attribute
        if a.name == "accuracy_level"
    ]
    assert levels and set(levels) == {0}
    assert (out / "tokenizer.json").exists() and (out / "calibration.json").exists()


def test_compare_reports_agreement_and_probability_gaps(tmp_path):
    b = _bundle(tmp_path)
    a = CountryCoder.load(b)
    same = web_parity.compare(a, a, ["Austria", "x"], ["country"] * 2, ["unknown"] * 2, [None] * 2)
    assert same["n"] == 2 and same["top1_agreement"] == 1.0 and same["max_abs_dp"] == 0.0
    assert same["decision_agreement"] == {"0.02": 1.0, "0.05": 1.0}
    web = CountryCoder.load(web_parity.browser_copy(b, tmp_path / "web"))
    r = web_parity.compare(a, web, ["Austria", "x"], ["country"] * 2, ["unknown"] * 2, [None] * 2)
    assert 0.0 <= r["top1_agreement"] <= 1.0 and r["max_abs_dp"] >= 0.0
