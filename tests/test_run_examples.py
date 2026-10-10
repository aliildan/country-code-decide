import importlib.util
import json
import sys
from pathlib import Path

from countrycode.calib.calibration import Calibration
from countrycode.infer.pipeline import make_bundle
from tests.test_pipeline import tiny_model, tiny_tokenizer

_spec = importlib.util.spec_from_file_location("run_examples", Path("scripts/run_examples.py"))
assert _spec and _spec.loader
run_examples = importlib.util.module_from_spec(_spec)
sys.modules["run_examples"] = run_examples
_spec.loader.exec_module(run_examples)


def test_examples_file_records_decisions_and_names(tmp_path):
    tiny_tokenizer(tmp_path / "tokenizer.json")
    cal = Calibration(temps=(1.0, 1.0), thresholds={"0.02": 0.0, "0.01": 0.99}, conformal_q=0.9)
    b = make_bundle(
        tiny_model(), tmp_path / "tokenizer.json", cal, tmp_path / "countrycode-small-vX",
        meta={"max_len": 32},
    )  # fmt: skip
    cfg = tmp_path / "examples.yaml"
    cfg.write_text("- {text: Austria, field: country, register_country: NO}\n- {text: Deleware}\n")
    out = run_examples.main(["--bundle", str(b), "--config", str(cfg), "--alpha", "0.02",
                             "--domain", "register",
                             "--results", str(tmp_path / "res")])  # fmt: skip
    assert out == tmp_path / "res" / "examples-small-vX.json"
    d = json.loads(out.read_text())
    assert d["alpha"] == 0.02 and d["domain"] == "register" and len(d["examples"]) == 2
    e = d["examples"][0]
    assert e["text"] == "Austria" and e["field"] == "country" and e["register_country"] == "NO"
    assert (
        not e["abstain"]
        and e["name"] == json.loads((b / "meta.json").read_text())["names"][e["target"]]
    )
    assert (
        d["examples"][1]["field"] == "unknown" and d["examples"][1]["register_country"] == "unknown"
    )
    strict = run_examples.main(["--bundle", str(b), "--config", str(cfg), "--alpha", "0.01",
                                "--results", str(tmp_path / "r2")])  # fmt: skip
    s = json.loads(strict.read_text())["examples"][0]
    assert s["abstain"] and s["target"] is None and s["name"] is None and s["candidates"]
