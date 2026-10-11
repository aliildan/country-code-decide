"""Lexicon-first cascade, measured (post-hoc, after an external question): an exact lexicon
with the training context rules answers what it can; the model at its certified 5 % threshold gets
the residual. Reads the private per-item predictions of the v1.2.1 post-hoc run and writes the
aggregate counts to results/posthoc-v1.2.1/cascade.json (no strings).

    uv run python scripts/cascade.py
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from countrycode.evaluation.harness import Prediction, answered_correct
from countrycode.real.gold import GoldItem

PARTS = ("test_v12", "test_v11", "ood")
RESULTS = Path("results/posthoc-v1.2.1")


def _stage(items: Sequence[GoldItem], preds: Mapping[str, Prediction]) -> tuple[int, int]:
    answered = [g for g in items if preds[g.item_id].target is not None]
    return len(answered), sum(not answered_correct(preds[g.item_id], g) for g in answered)


def cascade(
    items: Sequence[GoldItem], lexicon: Mapping[str, Prediction], model: Mapping[str, Prediction]
) -> dict:
    """Counts per stage: the lexicon, the model on the lexicon's residual, both together, and the
    model alone for comparison; the residual also per source."""
    residual = [g for g in items if lexicon[g.item_id].target is None]
    la, le = _stage(items, lexicon)
    ra, re_ = _stage(residual, model)
    ma, me = _stage(items, model)
    by_source: dict[str, dict] = {}
    for src in sorted({g.source for g in items}):
        r = [g for g in residual if g.source == src]
        a, e = _stage(r, model)
        by_source[src] = {"n": len(r), "model_answered": a, "model_errors": e}
    return {
        "n": len(items),
        "lexicon": {"answered": la, "errors": le},
        "residual": {"n": len(residual), "model_answered": ra, "model_errors": re_},
        "cascade": {"answered": la + ra, "errors": le + re_},
        "model_alone": {"answered": ma, "errors": me},
        "residual_by_source": by_source,
    }


def main() -> None:
    from countrycode.evaluation.__main__ import gold_and_lexicon
    from countrycode.evaluation.plots import load_predictions
    from countrycode.real.roles import with_roles

    gold = with_roles(gold_and_lexicon()[0])
    out = {"lexicon": "lexicon_context", "model": "model@0.05", "parts": {}}
    for part in PARTS:
        items = [g for g in gold if g.split == part]
        preds = load_predictions(
            Path(f"data/processed/eval/posthoc-v1.2.1_{part}_predictions.jsonl")
        )
        out["parts"][part] = cascade(items, preds["lexicon_context"], preds["model@0.05"])
    (RESULTS / "cascade.json").write_text(json.dumps(out, indent=1) + "\n")
    for part, r in out["parts"].items():
        res = r["residual"]
        print(f"{part}: lexicon {r['lexicon']}, residual {res}, cascade {r['cascade']}, "
              f"model alone {r['model_alone']}")  # fmt: skip


if __name__ == "__main__":
    main()
