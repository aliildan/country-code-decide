import importlib.util
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from countrycode.synth.framework import SynthItem

_spec = importlib.util.spec_from_file_location("audit_app", Path("eval/review/audit_app.py"))
assert _spec and _spec.loader
audit_app = importlib.util.module_from_spec(_spec)
sys.modules["audit_app"] = audit_app
_spec.loader.exec_module(audit_app)


def test_audit_flow(tmp_path):
    sample = [
        SynthItem("Austira", ("AT",), "keyboard", "1", 1, "Austria", "h"),
        SynthItem("AUST", ("AT", "AU"), "format", "1", 2, "Austria", "h"),
    ]
    path = tmp_path / "audit.jsonl"
    synth = tmp_path / "synth"
    synth.mkdir()
    for tag in ("keyboard@1", "format@1"):
        (synth / f"{tag}.jsonl").write_text("{}\n", encoding="utf-8")
    client = TestClient(audit_app.create_app(sample, path, synth_dir=synth))
    first = client.get("/api/next").json()
    assert first["item"]["text"] == "Austira" and first["item"]["names"] == ["Austria"]
    assert (
        client.post(
            "/api/verdict", json={"tag": "keyboard@1", "text": "Austira", "verdict": "ok"}
        ).status_code
        == 200
    )
    second = client.get("/api/next").json()
    assert second["item"]["text"] == "AUST" and second["progress"]["keyboard@1"]["audited"] == 1
    r = client.post(
        "/api/verdict",
        json={"tag": "format@1", "text": "AUST", "verdict": "wrong", "targets": ["AU"]},
    )
    assert r.status_code == 200
    rec = [json.loads(x) for x in path.read_text().splitlines()]
    assert rec[1]["verdict"] == "wrong" and rec[1]["owner_targets"] == ["AU"]
    assert client.get("/api/next").json()["item"] is None
    assert "Synthetic audit" in client.get("/").text
