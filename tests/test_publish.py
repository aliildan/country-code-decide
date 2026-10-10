import importlib.util
import subprocess
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("publish", Path("scripts/publish.py"))
assert _spec and _spec.loader
publish = importlib.util.module_from_spec(_spec)
sys.modules["publish"] = publish
_spec.loader.exec_module(publish)


def test_build_tree_copies_only_public_files(tmp_path):
    src = tmp_path / "src"
    (src / "src").mkdir(parents=True)
    (src / "docs").mkdir()
    (src / "src" / "a.py").write_text("x = 1\n")
    (src / "docs" / "private.md").write_text("secret plan\n")
    (src / "README.md").write_text("# hi\n")
    subprocess.run(["git", "init", "-q"], cwd=src, check=True)
    subprocess.run(["git", "add", "-A"], cwd=src, check=True)
    files = publish.build_tree(src, tmp_path / "out")
    assert files == ["README.md", "src/a.py"]
    assert not (tmp_path / "out" / "docs").exists()
