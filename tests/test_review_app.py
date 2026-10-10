import importlib.util
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from countrycode.lexicon.build import build_lexicon, code_rows
from countrycode.lexicon.types import AliasRow, Overrides
from countrycode.real.items import RealItem

_spec = importlib.util.spec_from_file_location("review_app", Path("eval/review/app.py"))
assert _spec and _spec.loader
review_app = importlib.util.module_from_spec(_spec)
sys.modules["review_app"] = review_app
_spec.loader.exec_module(review_app)

LX = build_lexicon([AliasRow("Delaware", "US-DE", "name", "en", "t"), *code_rows()], Overrides())


def make(tmp_path):
    items = [
        RealItem("us-co", "f", "jurisdiction", "US", t, n) for t, n in [("Deleware", 5), ("CA", 9)]
    ]
    manifest = {i.item_id: "dev" for i in items}
    owner = tmp_path / "owner.jsonl"
    app = review_app.create_app(items, manifest, LX, presets={}, owner_path=owner, audit_auto=0)
    return TestClient(app), items, owner


def test_next_answer_progress(tmp_path):
    client, items, owner = make(tmp_path)
    first = client.get("/api/next").json()
    assert first["item"]["text"] == "CA" and first["progress"] == {"done": 0, "total": 2}
    r = client.post("/api/answer", json={"item_id": first["item"]["item_id"], "targets": ["us-ca"]})
    assert r.status_code == 200
    line = json.loads(owner.read_text().splitlines()[0])
    assert line["targets"] == ["US-CA"] and line["item_id"] == first["item"]["item_id"]
    second = client.get("/api/next").json()
    assert second["item"]["text"] == "Deleware" and second["progress"]["done"] == 1


def test_invalid_code_rejected(tmp_path):
    client, items, _ = make(tmp_path)
    r = client.post("/api/answer", json={"item_id": items[0].item_id, "targets": ["XX"]})
    assert r.status_code == 422


def test_skip_keeps_item_pending(tmp_path):
    client, items, _ = make(tmp_path)
    first = client.get("/api/next").json()["item"]["item_id"]
    client.post("/api/answer", json={"item_id": first, "targets": [], "skip": True})
    nxt = client.get("/api/next").json()
    assert nxt["item"]["item_id"] != first and nxt["progress"]["done"] == 0


def test_index_page(tmp_path):
    client, _, _ = make(tmp_path)
    assert "Country review" in client.get("/").text
