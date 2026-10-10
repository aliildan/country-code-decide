"""Risk–coverage curve points of a final evaluation part, from the private per-item predictions
(data/processed/eval/<part>_predictions.jsonl) to the published aggregate
results/<part>/risk_coverage.json (60 points per predictor, no strings).

    uv run python scripts/risk_coverage.py --part test_v12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from countrycode.evaluation.__main__ import gold_and_lexicon
from countrycode.evaluation.plots import curves, load_predictions
from countrycode.real.roles import with_roles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="test_v12")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()
    gold, _ = gold_and_lexicon()
    gold = [g for g in with_roles(gold) if g.split == args.part]
    prefix = "" if args.results_dir == Path("results") else f"{args.results_dir.name}_"
    preds = load_predictions(Path("data/processed/eval") / f"{prefix}{args.part}_predictions.jsonl")
    out = args.results_dir / args.part / "risk_coverage.json"
    out.write_text(json.dumps(curves(preds, gold), indent=1) + "\n")
    print(out, len(gold))


if __name__ == "__main__":
    main()
