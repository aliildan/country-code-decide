"""Serving needs only the light stack (plan-10 Task 1): no torch, scipy or transformers."""

import json
import subprocess
import sys
from pathlib import Path

from countrycode.calib.calibration import Calibration
from countrycode.infer.pipeline import CountryCoder, make_bundle
from tests.test_pipeline import tiny_model, tiny_tokenizer

BLOCK = """
import importlib.abc, sys
class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in {"torch", "scipy", "transformers"}:
            raise ImportError("blocked heavy dependency: " + name)
sys.meta_path.insert(0, _Block())
"""


def _run(code: str) -> str:
    out = subprocess.run(
        [sys.executable, "-c", BLOCK + code], capture_output=True, text=True, cwd=Path.cwd()
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def _bundle(tmp_path: Path) -> Path:
    tiny_tokenizer(tmp_path / "tokenizer.json")
    cal = Calibration(temps=(1.3, 0.8), thresholds={"0.05": 0.0}, conformal_q=0.9)
    return make_bundle(
        tiny_model(), tmp_path / "tokenizer.json", cal, tmp_path / "b", meta={"max_len": 32}
    )


def test_pipeline_runs_without_torch_scipy_transformers(tmp_path):
    b = _bundle(tmp_path)
    texts = ["Austria", "Deleware", "n/a"]
    expected = CountryCoder.load(b).probs(texts, ["country"] * 3, ["unknown", "US", "unknown"])
    got = _run(
        "import json\n"
        "from countrycode.infer.pipeline import CountryCoder\n"
        f"c = CountryCoder.load({str(b)!r})\n"
        f"p = c.probs({texts!r}, ['country'] * 3, ['unknown', 'US', 'unknown'])\n"
        "d = c.code('Austria', alpha=0.05)\n"
        "print(json.dumps({'p': p.tolist(), 'target': d.target}))\n"
    )
    res = json.loads(got)
    assert res["target"] is not None
    diffs = [
        abs(a - b)
        for ra, rb in zip(res["p"], expected.tolist(), strict=True)
        for a, b in zip(ra, rb, strict=True)
    ]
    assert max(diffs) < 1e-9


def test_server_imports_without_torch_scipy_transformers(tmp_path):
    b = _bundle(tmp_path)
    got = _run(
        "import importlib.util, sys\n"
        "from countrycode.infer.pipeline import CountryCoder\n"
        "spec = importlib.util.spec_from_file_location('server_app', 'server/app.py')\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['server_app'] = m\n"
        "spec.loader.exec_module(m)\n"
        "from fastapi.testclient import TestClient\n"
        f"cl = TestClient(m.create_app(CountryCoder.load({str(b)!r})))\n"
        "r = cl.post('/v1/countrycode', json={'items': [{'text': 'Austria'}]})\n"
        "print(r.status_code)\n"
    )
    assert got.strip() == "200"
