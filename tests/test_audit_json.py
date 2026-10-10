import importlib.util
import json
import sys
from pathlib import Path

from countrycode.synth.framework import SynthItem

_spec = importlib.util.spec_from_file_location("audit_json", Path("eval/review/audit_json.py"))
assert _spec and _spec.loader
audit_json = importlib.util.module_from_spec(_spec)
sys.modules["audit_json"] = audit_json
_spec.loader.exec_module(audit_json)

SAMPLE = [
    SynthItem("Austira", ("AT",), "keyboard", "1", 1, "Austria", "h"),
    SynthItem("AUST", ("AT", "AU"), "format", "1", 2, "Austria", "h"),
    SynthItem("Geneva", ("CH", "US-AL", "US-IL"), "city", "2", 3, "__city_1", "h"),
]


def test_export_lists_unaudited_items_with_names(tmp_path):
    audit = tmp_path / "audit.jsonl"
    audit.write_text(
        json.dumps({"tag": "keyboard@1", "text": "Austira", "verdict": "ok"}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "audit.json"
    assert audit_json.export_sample(SAMPLE, audit, out) == 2
    rows = json.loads(out.read_text(encoding="utf-8"))["items"]
    assert [r["text"] for r in rows] == ["AUST", "Geneva"]
    assert rows[0]["label"] == "AT Austria | AU Australia" and rows[0]["verdict"] == ""
    assert rows[1]["from"] == ""  # internal base ids are not shown


def test_import_appends_verdicts_and_reports_bad_ones(tmp_path):
    audit = tmp_path / "audit.jsonl"
    out = tmp_path / "audit.json"
    audit_json.export_sample(SAMPLE, audit, out)
    doc = json.loads(out.read_text(encoding="utf-8"))
    doc["items"][0]["verdict"] = "ok"
    doc["items"][1]["verdict"] = "x"
    doc["items"][1]["fix"] = "CH"
    doc["items"][2]["verdict"] = "maybe"
    out.write_text(json.dumps(doc), encoding="utf-8")
    synth = tmp_path / "synth"
    synth.mkdir()
    for tag in ("keyboard@1", "format@1", "city@2"):
        (synth / f"{tag}.jsonl").write_text("{}\n", encoding="utf-8")
    res = audit_json.import_verdicts(out, audit, synth_dir=synth)
    assert res["imported"] == 2 and res["invalid"] == [{"text": "Geneva", "verdict": "maybe"}]
    recs = [json.loads(x) for x in audit.read_text(encoding="utf-8").splitlines()]
    assert recs[0]["verdict"] == "ok" and recs[0]["targets"] == ["AT"]
    assert recs[1]["verdict"] == "wrong" and recs[1]["owner_targets"] == ["CH"]
    assert audit_json.import_verdicts(out, audit, synth_dir=synth)["unchanged"] == 2


def test_export_can_be_limited_to_a_runs_generators(tmp_path):
    out = tmp_path / "audit.json"
    n = audit_json.export_sample(SAMPLE, tmp_path / "a.jsonl", out, tags={"keyboard@1"})
    rows = json.loads(out.read_text(encoding="utf-8"))["items"]
    assert n == 1 and rows[0]["tag"] == "keyboard@1"


def test_context_bound_items_show_their_context(tmp_path):
    out = tmp_path / "audit.json"
    item = SynthItem("NL", ("CA",), "caprov", "1", 1, "__caprov_NL", "h")
    audit_json.export_sample([item], tmp_path / "audit.jsonl", out)
    row = json.loads(out.read_text(encoding="utf-8"))["items"][0]
    assert row["context"] == "jurisdiction field of a US register"


def test_region_items_show_their_geonames_source(tmp_path):
    out = tmp_path / "audit.json"
    item = SynthItem("Paktia", ("AF",), "region", "1", 1, "__region_AF.33", "h")
    audit_json.export_sample([item], tmp_path / "audit.jsonl", out)
    assert (
        json.loads(out.read_text(encoding="utf-8"))["items"][0]["from"] == "GeoNames admin1 AF.33"
    )


def test_import_records_the_audited_files_hash(tmp_path):
    from countrycode.synth.audit import file_sha256

    synth = tmp_path / "synth"
    synth.mkdir()
    (synth / "keyboard@1.jsonl").write_text('{"text": "Austira"}\n', encoding="utf-8")
    out, audit = tmp_path / "audit.json", tmp_path / "audit.jsonl"
    audit_json.export_sample(SAMPLE[:1], audit, out)
    doc = json.loads(out.read_text(encoding="utf-8"))
    doc["items"][0]["verdict"] = "ok"
    out.write_text(json.dumps(doc), encoding="utf-8")
    audit_json.import_verdicts(out, audit, SAMPLE[:1], synth_dir=synth)
    rec = json.loads(audit.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["file_sha256"] == file_sha256(synth / "keyboard@1.jsonl")
