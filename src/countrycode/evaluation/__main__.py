"""python -m countrycode.evaluation baselines [--part dev] [--only a,b] → results/<part>/."""

from __future__ import annotations

import argparse
import json

from countrycode.evaluation.harness import Predictor, evaluate
from countrycode.lexicon.build import LEXICON_PATH, Lexicon
from countrycode.real.gold import load_gold, read_presets
from countrycode.real.items import read_items
from countrycode.real.splits import read_manifest


def make_baselines(lx: Lexicon, only: set[str] | None) -> list[Predictor]:
    from countrycode.evaluation import baselines as b

    factories = {
        "lexicon_exact": lambda: b.LexiconExact(lx),
        "lexicon_fuzzy": lambda: b.LexiconFuzzy(lx),
        "lexicon_context": lambda: b.LexiconContext(lx),
        "ons_style": lambda: b.OnsStyle(lx),
        "countrynames": b.CountryNames,
        "pycountry": b.PyCountry,
        "coco": b.Coco,
        "e5_knn": lambda: b.E5Knn(lx),
    }
    preds: list[Predictor] = [f() for name, f in factories.items() if only is None or name in only]
    if only:
        from countrycode.evaluation.llm import LLMPredictor

        preds += [LLMPredictor(o.removeprefix("llm:")) for o in only if o.startswith("llm:")]
    return preds


def gold_and_lexicon():
    lx = Lexicon.load(LEXICON_PATH)
    gold = load_gold(read_items(), read_manifest(), lx, read_presets())
    return gold, lx


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("baselines")
    s.add_argument("--part", default="dev", choices=["dev", "calib"])
    s.add_argument("--only", default="")
    args = ap.parse_args()
    gold, lx = gold_and_lexicon()
    seen = set(lx.index_keys())
    only = set(args.only.split(",")) if args.only else None
    for pred in make_baselines(lx, only):
        rep = evaluate(pred, gold, args.part, seen_keys=seen)
        o = rep["overall"]
        print(
            json.dumps(
                {
                    "predictor": pred.name,
                    "n": o["n"],
                    "acc": round(o["accuracy"], 4),
                    "acc_w": round(o["accuracy_weighted"], 4),
                    "country_acc": round(o["country_accuracy"], 4),
                    "cov@1%": round(o["coverage_at_risk"]["0.01"], 4),
                    "none_fa": o["none_false_accept"],
                    "ms": round(rep["latency_ms_per_item"], 3),
                }
            )
        )


if __name__ == "__main__":
    main()
