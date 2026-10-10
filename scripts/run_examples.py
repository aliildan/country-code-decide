"""Run the README's hand-picked examples through a release bundle and record the decisions in
results/examples-<version>.json, so the README shows the model's real output, never typed text.

    uv run python scripts/run_examples.py --bundle dist/countrycode-small-v1.2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> Path:
    from countrycode.infer.pipeline import CountryCoder

    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs/docs/examples.yaml"))
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--domain", default="register")
    ap.add_argument("--results", default=str(ROOT / "results"))
    args = ap.parse_args(argv)
    coder = CountryCoder.load(args.bundle)
    names = coder.meta["names"]
    rows = []
    # BaseLoader: every value stays a string (YAML 1.1 would read the register `NO` as False)
    for ex in yaml.load(Path(args.config).read_text(encoding="utf-8"), Loader=yaml.BaseLoader):
        field = ex.get("field", "unknown")
        reg = ex.get("register_country", "unknown")
        d = coder.code(ex["text"], field, reg, alpha=args.alpha, domain=args.domain)
        rows.append({
            "text": ex["text"], "field": field, "register_country": reg, "target": d.target,
            "name": names[d.target] if d.target else None, "confidence": d.confidence,
            "candidates": list(d.candidates), "abstain": d.abstain,
        })  # fmt: skip
    version = Path(args.bundle).name.removeprefix("countrycode-")
    out = Path(args.results) / f"examples-{version}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"bundle": Path(args.bundle).name, "alpha": args.alpha, "domain": args.domain,
             "examples": rows}, indent=1, ensure_ascii=False,
        ) + "\n", encoding="utf-8",
    )  # fmt: skip
    return out


if __name__ == "__main__":
    print(main(sys.argv[1:]))
