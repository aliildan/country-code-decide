"""Configs resolve from the repository, not the working directory (external review)."""

import subprocess
import sys
from pathlib import Path


def test_domains_and_context_rules_load_outside_the_repository(tmp_path):
    code = (
        "from countrycode.calib.calibration import domain_of\n"
        "from countrycode.train.dataset import rules\n"
        "from countrycode.real.roles import HELD_OUT_PATH\n"
        "print(domain_of('us-co'), domain_of('gbif'), 'us_collision' in rules(),"
        " HELD_OUT_PATH.exists())\n"
    )
    src = Path(__file__).resolve().parents[1] / "src"
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True,
                         text=True, env={"PYTHONPATH": str(src), "PATH": ""})  # fmt: skip
    assert out.returncode == 0, out.stderr[-1500:]
    assert out.stdout.split() == ["register", "curated", "True", "True"]
