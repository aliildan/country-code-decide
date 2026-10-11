import importlib.util
import json
import sys
from pathlib import Path

from tests.test_make_docs import _v121

_spec = importlib.util.spec_from_file_location(
    "linkedin_assets", Path("scripts/linkedin_assets.py")
)
assert _spec and _spec.loader
la = importlib.util.module_from_spec(_spec)
sys.modules["linkedin_assets"] = la
_spec.loader.exec_module(la)

LINKS = {"repo": "https://github.com/x/y", "hf_model": "a/b", "paper_doi": "10.5281/zenodo.1"}


def test_post_numbers_come_from_the_release_results(tmp_path):
    r = _v121(tmp_path)
    en = la.post_text(r, LINKS, "en")
    for s in ("75 MB", "4 ms", "656", "82.6%", "73.8%", "35.8%", "29×", "87.8%", "2.5%", "89.5%",
              "88.6%", "https://github.com/x/y", "https://huggingface.co/a/b",
              "https://doi.org/10.5281/zenodo.1"):  # fmt: skip
        assert s in en, s
    tr = la.post_text(r, LINKS, "tr")
    for s in ("75 MB", "%82,6", "%73,8", "%35,8", "%87,8", "%2,5", "%89,5", "%88,6", "29 kat"):
        assert s in tr, s


def test_where_it_helps_series_and_chart(tmp_path):
    r = _v121(tmp_path)
    groups = la.where_it_helps(r)
    assert [g["n"] for g in groups] == [100, 656]
    assert groups[1]["values"] == {"model": 0.826, "llm": 0.738, "lexicon_context": 0.358}
    out = tmp_path / "chart.png"
    la.plot_where_it_helps(groups, out)
    assert out.stat().st_size > 10_000


def test_examples_card_renders(tmp_path):
    ex = {"alpha": 0.05, "examples": [
        {"text": "Untied Kingdon", "field": "country", "register_country": "unknown",
         "target": "GB", "name": "United Kingdom", "confidence": 0.98, "candidates": ["GB"],
         "abstain": False},
        {"text": "n/a", "field": "country", "register_country": "unknown", "target": None,
         "name": None, "confidence": 0.12, "candidates": ["NE", "NONE"],
         "abstain": True}]}  # fmt: skip
    p = tmp_path / "ex.json"
    p.write_text(json.dumps(ex))
    out = tmp_path / "card.png"
    la.plot_examples(json.loads(p.read_text()), out)
    assert out.stat().st_size > 10_000
