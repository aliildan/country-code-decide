import json

from countrycode.calib.calibration import Calibration
from countrycode.cli import main
from countrycode.infer.pipeline import make_bundle
from tests.test_pipeline import tiny_model, tiny_tokenizer


def test_cli_prints_json_decisions(tmp_path, capsys):
    tiny_tokenizer(tmp_path / "tokenizer.json")
    cal = Calibration(temps=(1.0, 1.0), thresholds={"0.05": 0.0}, conformal_q=0.9)
    b = make_bundle(
        tiny_model(), tmp_path / "tokenizer.json", cal, tmp_path / "b", meta={"max_len": 32}
    )
    assert (
        main(["--bundle", str(b), "--alpha", "0.05", "--field", "country", "Austria", "Deleware"])
        == 0
    )
    out = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [d["text"] for d in out] == ["Austria", "Deleware"] and all("target" in d for d in out)
