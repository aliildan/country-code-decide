import importlib.util
import json
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("hf_upload", Path("scripts/hf_upload.py"))
assert _spec and _spec.loader
hf_upload = importlib.util.module_from_spec(_spec)
sys.modules["hf_upload"] = hf_upload
_spec.loader.exec_module(hf_upload)


def _setup(tmp_path, pending=False, release_pass=True, card="plain card\n"):
    root = tmp_path / "repo"
    (root / "configs" / "docs").mkdir(parents=True)
    (root / "results").mkdir()
    (root / "configs" / "docs" / "model_card.md.tmpl").write_text("plain card\n")
    (root / "model_card.md").write_text(card)
    (root / "results" / "release.json").write_text(json.dumps({"version": "small-v9"}))
    for f in ("LICENSE", "NOTICE"):
        (root / f).write_text(f)
    b = tmp_path / "dist" / "countrycode-small-v9"
    b.mkdir(parents=True)
    for f in hf_upload.FILES:
        (b / f).write_text("x")
    meta = {"pending_g1": pending, "g1": {"releases": {"keyboard@2": {"pass": release_pass}}}}
    (b / "meta.json").write_text(json.dumps(meta))
    return root, b


def test_a_clean_release_bundle_passes_the_checks(tmp_path):
    root, b = _setup(tmp_path)
    assert hf_upload.check(b, root) == []


def test_checks_refuse_unaudited_failed_stale_or_mismatched_bundles(tmp_path):
    root, b = _setup(tmp_path / "a", pending=True)
    assert any("G1" in p for p in hf_upload.check(b, root))
    root, b = _setup(tmp_path / "b", release_pass=False)
    assert any("keyboard@2" in p for p in hf_upload.check(b, root))
    root, b = _setup(tmp_path / "c", card="old card\n")
    assert any("model_card.md" in p for p in hf_upload.check(b, root))
    root, b = _setup(tmp_path / "d")
    (b / "model.onnx").unlink()
    assert any("model.onnx" in p for p in hf_upload.check(b, root))
    root, b = _setup(tmp_path / "e")
    (root / "results" / "release.json").write_text(json.dumps({"version": "small-v8"}))
    assert any("release" in p for p in hf_upload.check(b, root))


def test_stage_writes_the_model_card_as_readme_with_licence_files(tmp_path):
    root, b = _setup(tmp_path)
    out = hf_upload.stage(b, root, tmp_path / "stage")
    names = sorted(p.name for p in out.iterdir())
    assert names == sorted([*hf_upload.FILES, "README.md", "LICENSE", "NOTICE"])
    assert (out / "README.md").read_text() == "plain card\n"
