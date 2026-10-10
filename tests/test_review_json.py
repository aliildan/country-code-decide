import importlib.util
import json
import sys
from pathlib import Path

from countrycode.lexicon.build import build_lexicon, code_rows
from countrycode.lexicon.types import AliasRow, Overrides
from countrycode.real.items import RealItem

_spec = importlib.util.spec_from_file_location("review_json", Path("eval/review/json_queue.py"))
assert _spec and _spec.loader
review_json = importlib.util.module_from_spec(_spec)
sys.modules["review_json"] = review_json
_spec.loader.exec_module(review_json)

LX = build_lexicon([AliasRow("Delaware", "US-DE", "name", "en", "t"), *code_rows()], Overrides())


def setup(tmp_path):
    items = [
        RealItem("us-co", "f", "jurisdiction", "US", t, n)
        for t, n in [("Deleware", 5), ("CA", 9), ("zzq", 1)]
    ]
    manifest = {i.item_id: "dev" for i in items}
    owner = tmp_path / "owner.jsonl"
    owner.write_text(
        json.dumps({"item_id": items[1].item_id, "targets": ["US-CA"]}) + "\n", encoding="utf-8"
    )
    return items, manifest, owner


def test_export_lists_pending_items_with_empty_answers(tmp_path):
    items, manifest, owner = setup(tmp_path)
    out = tmp_path / "review.json"
    n = review_json.export_queue(items, manifest, LX, {}, owner, out, audit_auto=0)
    doc = json.loads(out.read_text(encoding="utf-8"))
    texts = [r["text"] for r in doc["items"]]
    assert n == 2 and texts == ["Deleware", "zzq"]  # answered "CA" left out
    assert all(r["answer"] == "" for r in doc["items"])  # never pre-filled
    assert doc["items"][0]["proposals"][0].startswith("US-DE")
    assert "guide" in doc


def test_import_appends_valid_answers_and_reports_bad_ones(tmp_path):
    items, manifest, owner = setup(tmp_path)
    out = tmp_path / "review.json"
    review_json.export_queue(items, manifest, LX, {}, owner, out, audit_auto=0)
    doc = json.loads(out.read_text(encoding="utf-8"))
    doc["items"][0]["answer"] = "1"  # first proposal
    doc["items"][1]["answer"] = "none"
    out.write_text(json.dumps(doc), encoding="utf-8")
    res = review_json.import_answers(out, owner)
    assert res == {"imported": 2, "unchanged": 0, "empty": 0, "invalid": []}
    rows = [json.loads(line) for line in owner.read_text(encoding="utf-8").splitlines()]
    assert rows[-2]["targets"] == ["US-DE"] and rows[-1]["targets"] == ["NONE"]
    assert rows[-1]["note"] == "json"
    # re-import is idempotent; bad codes are reported, not written
    doc["items"][1]["answer"] = "XX, US"
    out.write_text(json.dumps(doc), encoding="utf-8")
    res = review_json.import_answers(out, owner)
    assert res["imported"] == 0 and res["unchanged"] == 1
    assert res["invalid"] == [{"id": doc["items"][1]["id"], "answer": "XX, US"}]


def test_parse_answer_forms():
    props = ["US-DE  (exact)", "CA,GB,US  (fuzzy 'x' 90)"]
    assert review_json.parse_answer("2", props) == ("CA", "GB", "US")
    assert review_json.parse_answer("gb ca;us", props) == ("CA", "GB", "US")
    assert review_json.parse_answer("", props) is None
    assert review_json.parse_answer("?", props) is None


def test_export_filters_sources(tmp_path):
    items = [
        RealItem("us-co", "f", "jurisdiction", "US-CO", "Deleware", 5),
        RealItem("us-ny", "f", "jurisdiction", "US-NY", "Delawre", 2),
    ]
    manifest = {i.item_id: "dev" for i in items}
    out = tmp_path / "review.json"
    n = review_json.export_queue(
        items, manifest, LX, {}, tmp_path / "owner.jsonl", out, audit_auto=0, sources={"us-ny"}
    )
    rows = json.loads(out.read_text(encoding="utf-8"))["items"]
    assert n == 1 and rows[0]["src"] == "us-ny"
