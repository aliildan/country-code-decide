import json

from countrycode.synth.audit import audit_status, file_sha256
from countrycode.synth.framework import SynthItem

ITEM = SynthItem("A", ("AT",), "gen", "1", 1, "src", "h")


def _setup(tmp_path, verdict: dict):
    synth = tmp_path / "synth"
    synth.mkdir()
    f = synth / "gen@1.jsonl"
    f.write_text('{"text": "A"}\n', encoding="utf-8")
    audit = tmp_path / "audit.jsonl"
    audit.write_text(json.dumps({"tag": "gen@1", "text": "A", "verdict": "ok", **verdict}) + "\n")
    return synth, f, audit


def test_a_verdict_counts_only_for_the_file_it_was_given_for(tmp_path):
    synth, f, audit = _setup(tmp_path, {})
    sha = file_sha256(f)
    audit.write_text(json.dumps({"tag": "gen@1", "text": "A", "verdict": "ok",
                                 "file_sha256": sha}) + "\n")  # fmt: skip
    assert audit_status([ITEM], audit, synth_dir=synth, pins={})["gen@1"]["audited"] == 1
    f.write_text('{"text": "A", "targets": ["AU"]}\n', encoding="utf-8")  # same text, new label
    st = audit_status([ITEM], audit, synth_dir=synth, pins={})["gen@1"]
    assert st["audited"] == 0 and st["stale"] == 1 and st["g1_pass"] is False


def test_verdicts_without_a_hash_count_only_through_a_pin(tmp_path):
    synth, f, audit = _setup(tmp_path, {})
    assert audit_status([ITEM], audit, synth_dir=synth, pins={})["gen@1"]["stale"] == 1
    pins = {"gen@1": file_sha256(f)}
    assert audit_status([ITEM], audit, synth_dir=synth, pins=pins)["gen@1"]["audited"] == 1
