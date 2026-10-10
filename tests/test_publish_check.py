import importlib.util
import pathlib
import sys

_spec = importlib.util.spec_from_file_location("pc", pathlib.Path("scripts/publish_check.py"))
assert _spec and _spec.loader
pc = importlib.util.module_from_spec(_spec)
sys.modules["pc"] = pc  # dataclasses resolve their module via sys.modules
_spec.loader.exec_module(pc)


def rules(vs):
    return {v.rule for v in vs}


def test_private_paths_forbidden():
    for p in ("CLAUDE.md", "docs/x.md", "data/a.json", "runs/r.log", "notes/n.md", "secrets/k.env"):
        assert "forbidden-path" in rules(pc.check_path(p))


def test_public_paths_ok():
    assert pc.check_path("configs/labels/countries.yaml") == []
    assert pc.check_path("results/lexicon/stats.json") == []


def test_data_suffix_forbidden():
    assert "forbidden-suffix" in rules(pc.check_path("src/x.parquet"))


def test_secret_detected():
    line = "t = 'ghp_" + "A" * 36 + "'"
    assert "secret" in rules(pc.check_content("a.py", line.encode()))


def test_public_allow_list():
    assert pc.is_public("NOTICE") and not pc.is_public("web/index.html")  # browser demo removed
    assert not pc.is_public("docs/a.md")
