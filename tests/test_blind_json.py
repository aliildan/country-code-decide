import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from countrycode.real.gold import GoldItem

_spec = importlib.util.spec_from_file_location("blind_json", Path("eval/review/blind_json.py"))
assert _spec and _spec.loader
blind_json = importlib.util.module_from_spec(_spec)
sys.modules["blind_json"] = blind_json
_spec.loader.exec_module(blind_json)


def item(i: int, mode: str, targets=("US-CO",)) -> GoldItem:
    return GoldItem(f"id{i}", "us-co", "jurisdictonofformation", "jurisdiction", "US-CO",
                    f"text {i}", 1, "calib", tuple(targets), "human", mode)  # fmt: skip


ITEMS = [item(i, "auto_exact") for i in range(200)] + [
    item(1000 + i, "reviewed_proposal") for i in range(100)
]


def test_sample_is_stratified_and_seeded():
    a = blind_json.sample_blind(ITEMS, frac=0.15, seed=13)
    b = blind_json.sample_blind(ITEMS, frac=0.15, seed=13)
    assert [g.item_id for g in a] == [g.item_id for g in b]
    modes = Counter(g.annotation_mode for g in a)
    assert modes == {"auto_exact": 30, "reviewed_proposal": 15}


def test_export_has_no_proposals_or_gold(tmp_path):
    out = tmp_path / "blind.json"
    assert blind_json.export_blind(ITEMS[:3], out) == 3
    rows = json.loads(out.read_text(encoding="utf-8"))["items"]
    assert set(rows[0]) == {"text", "field", "kind", "reg", "src", "answer", "id"}
    assert all(r["answer"] == "" for r in rows)


def test_report_bounds_gold_error(tmp_path):
    sample = ITEMS[:90]
    blind = tmp_path / "blind.jsonl"
    blind.write_text("".join(json.dumps({"item_id": g.item_id, "targets": list(g.targets)}) + "\n"
                             for g in sample), encoding="utf-8")  # fmt: skip
    rep = blind_json.report(sample, blind)
    assert rep["n"] == 90 and rep["disagree"] == 0
    assert rep["gold_error_upper_95"] == pytest.approx(0.0327, abs=1e-3)


def test_import_validates_and_adjudication_lists_both_answers(tmp_path):
    out, blind = tmp_path / "blind.json", tmp_path / "blind.jsonl"
    blind_json.export_blind(ITEMS[:2], out)
    doc = json.loads(out.read_text(encoding="utf-8"))
    doc["items"][0]["answer"] = "US-CA"
    doc["items"][1]["answer"] = "XX"
    out.write_text(json.dumps(doc), encoding="utf-8")
    res = blind_json.import_blind(out, blind)
    assert res["imported"] == 1 and res["invalid"] == [{"id": "id1", "answer": "XX"}]
    adj = tmp_path / "adjudicate.json"
    assert blind_json.export_adjudication(ITEMS[:2], blind, adj) == 1
    row = json.loads(adj.read_text(encoding="utf-8"))["items"][0]
    assert row["proposals"] == ["US-CO  (gold)", "US-CA  (blind)"] and row["answer"] == ""


def test_unanswered_skips_items_already_answered_blind(tmp_path):
    blind = tmp_path / "blind.jsonl"
    blind.write_text(json.dumps({"item_id": "id0", "targets": ["US-CO"]}) + "\n", encoding="utf-8")
    left = blind_json.unanswered(ITEMS[:3], blind)
    assert [g.item_id for g in left] == ["id1", "id2"]


def test_report_measures_agreement_before_adjudication(tmp_path):
    blind = tmp_path / "blind.jsonl"
    blind.write_text("".join(json.dumps({"item_id": g.item_id, "targets": ["US-CO"]}) + "\n"
                             for g in ITEMS[:10]), encoding="utf-8")  # fmt: skip
    adj = tmp_path / "adjudicate.json"
    blind_json._write([{"text": "text 0", "proposals": ["US-CA  (gold)", "US-CO  (blind)"],
                        "answer": "2", "id": "id0"}], adj)  # fmt: skip
    rep = blind_json.report(ITEMS[:10], blind, adjudication=adj)  # gold now equals blind
    assert rep["n"] == 10 and rep["disagree"] == 1 and rep["adjudicated"] == 1
    assert rep["adjudicated_to_blind"] == 1
