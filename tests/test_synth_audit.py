import json

from countrycode.synth.audit import audit_status, file_sha256, sample_for_audit, upper_bound_95
from countrycode.synth.framework import SynthItem, write_items


def mk(tmp_path, gen, n):
    items = [SynthItem(f"{gen}{i}", ("AT",), gen, "1", i, "Austria", "h") for i in range(n)]
    write_items(items, tmp_path / f"{gen}@1.jsonl")


def test_sample_is_stable_and_per_generator(tmp_path):
    mk(tmp_path, "keyboard", 1000)
    mk(tmp_path, "junk", 50)
    a = sample_for_audit(tmp_path, n=300, seed=1)
    b = sample_for_audit(tmp_path, n=300, seed=1)
    assert a == b
    by = {}
    for it in a:
        by.setdefault(it.generator, 0)
        by[it.generator] += 1
    assert by == {"keyboard": 300, "junk": 50}


def test_upper_bound():
    assert abs(upper_bound_95(300, 0) - 0.00994) < 1e-4  # ≈ rule of three
    assert upper_bound_95(300, 1) > 0.01


def test_audit_status(tmp_path):
    sample = [SynthItem(f"k{i}", ("AT",), "keyboard", "1", i, "Austria", "h") for i in range(300)]
    path = tmp_path / "audit.jsonl"
    with path.open("w") as f:
        for i, it in enumerate(sample):
            verdict = "wrong" if i == 0 else "ok"
            f.write(json.dumps({"tag": "keyboard@1", "text": it.text, "verdict": verdict}) + "\n")
    synth = tmp_path / "synth"
    synth.mkdir()
    (synth / "keyboard@1.jsonl").write_text("{}\n", encoding="utf-8")
    pins = {"keyboard@1": file_sha256(synth / "keyboard@1.jsonl")}  # verdicts without a hash
    st = audit_status(sample, path, synth_dir=synth, pins=pins)["keyboard@1"]
    assert st["audited"] == 300 and st["errors"] == 1 and st["g1_pass"] is False
