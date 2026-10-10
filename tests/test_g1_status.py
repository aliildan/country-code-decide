import json

from countrycode.synth.audit import g1_status


def _report(tmp_path, **gens):
    rep = {"generators": {t: {"G1_audit": {"audited": a, "errors": e, "g1_pass": p}}
                          for t, (a, e, p) in gens.items()}}  # fmt: skip
    path = tmp_path / "synth.json"
    path.write_text(json.dumps(rep))
    return path


def test_all_releases_of_a_run_must_pass(tmp_path):
    rep = _report(tmp_path, **{"junk@2": (300, 0, True), "translit@4": (300, 0, True)})
    st = g1_status(["junk@2", "translit@4"], rep)
    want = {"audited": 300, "errors": 0, "pass": True, "file_sha256": None}
    assert st["all_pass"] is True and st["releases"]["translit@4"] == want
    rep = _report(tmp_path, **{"junk@2": (300, 0, True), "translit@3": (300, 1, False)})
    assert g1_status(["junk@2", "translit@3"], rep)["all_pass"] is False
    assert g1_status(["junk@2", "missing@1"], rep)["all_pass"] is False  # never audited


def test_stamp_writes_the_status_into_bundle_meta(tmp_path):
    from countrycode.infer.__main__ import stamp_g1

    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "meta.json").write_text(json.dumps({"version": "x", "pending_g1": True}))
    (tmp_path / "train.json").write_text(json.dumps({"gens": ["junk@2"]}))
    rep = _report(tmp_path, **{"junk@2": (300, 0, True)})
    stamp_g1(tmp_path / "b", tmp_path / "train.json", rep)
    meta = json.loads((tmp_path / "b" / "meta.json").read_text())
    assert meta["pending_g1"] is False and meta["g1"]["all_pass"] is True
