import importlib.util
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from countrycode.calib.calibration import Calibration
from countrycode.infer.pipeline import CountryCoder, make_bundle
from tests.test_pipeline import tiny_model, tiny_tokenizer

_spec = importlib.util.spec_from_file_location("server_app", Path("server/app.py"))
assert _spec and _spec.loader
server_app = importlib.util.module_from_spec(_spec)
sys.modules["server_app"] = server_app
_spec.loader.exec_module(server_app)


def client(tmp_path):
    tiny_tokenizer(tmp_path / "tokenizer.json")
    cal = Calibration(temps=(1.0, 1.0), thresholds={"0.01": 0.99, "0.05": 0.0}, conformal_q=0.9)
    b = make_bundle(
        tiny_model(), tmp_path / "tokenizer.json", cal, tmp_path / "b", meta={"max_len": 32}
    )
    return TestClient(server_app.create_app(CountryCoder.load(b)))


def test_batch_endpoint_keeps_per_item_context(tmp_path):
    c = client(tmp_path)
    r = c.post(
        "/v1/countrycode",
        json={
            "alpha": 0.05,
            "items": [
                {"text": "Austria", "field": "country"},
                {"text": "Deleware", "field": "jurisdiction", "register_country": "us"},
                {"text": "x", "field": "weird-field", "register_country": "??", "name": "John Doe"},
            ],
        },
    )
    assert r.status_code == 200
    out = r.json()["results"]
    assert len(out) == 3 and all("target" in d and "candidates" in d for d in out)
    assert "John Doe" not in r.text  # unknown fields are ignored, never echoed


def test_systemone_typed_answers(tmp_path):
    c = client(tmp_path)
    r = c.post(
        "/v1/systemone",
        json={
            "state": {"text": "Austria", "field": "country"},
            "questions": {
                "country": {"type": "choice"},
                "us_state": {"type": "choice"},
                "is_country": {"type": "noul"},
            },
            "policy": {"alpha": 0.01},
        },
    )
    assert r.status_code == 200
    a = r.json()["answers"]
    assert set(a) == {"country", "us_state", "is_country"}
    assert (
        "probs" in a["country"] and "abstain" in a["country"] and 0 <= a["is_country"]["p_yes"] <= 1
    )


def test_health(tmp_path):
    assert client(tmp_path).get("/health").json()["status"] == "ok"


def test_items_choose_a_domain_and_keep_us_state_registers(tmp_path):
    from countrycode.calib.calibration import CalibrationSet

    tiny_tokenizer(tmp_path / "tokenizer.json")
    loose = Calibration(temps=(1.0, 1.0), thresholds={"0.05": 0.0}, conformal_q=0.9)
    strict = Calibration(temps=(1.0, 1.0), thresholds={"0.05": None}, conformal_q=0.9)
    b = make_bundle(tiny_model(), tmp_path / "tokenizer.json",
                    CalibrationSet({"register": loose, "curated": strict}), tmp_path / "b",
                    meta={"max_len": 32})  # fmt: skip
    c = TestClient(server_app.create_app(CountryCoder.load(b)))
    items = [{"text": "Austria", "domain": "register"}, {"text": "Austria", "domain": "curated"}]
    r = c.post("/v1/countrycode", json={"alpha": 0.05, "items": items})
    assert [d["abstain"] for d in r.json()["results"]] == [False, True]
    item = server_app.Item(text="Adams County", field="jurisdiction", register_country="us-co")
    assert server_app._norm(item)[2] == "US-CO"


def test_empty_text_is_rejected_and_is_country_follows_the_country_answer(tmp_path):
    c = client(tmp_path)
    for blank in ("", "   "):
        r = c.post("/v1/systemone", json={"state": {"text": blank}})
        assert r.status_code == 422
    strict = c.post("/v1/systemone", json={"state": {"text": "Austria"}, "policy": {"alpha": 0.01}})
    a = strict.json()["answers"]
    assert not (a["country"]["abstain"] and not a["is_country"]["abstain"])


def test_unknown_domain_is_rejected(tmp_path):
    c = client(tmp_path)
    r = c.post("/v1/countrycode", json={"items": [{"text": "Austria", "domain": "Register"}]})
    assert r.status_code == 422
