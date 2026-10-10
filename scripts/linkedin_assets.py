"""LinkedIn / release assets from the final evaluation (plan-05 Task 6).

    uv run python scripts/linkedin_assets.py --part test

Writes to docs/linkedin/ (private — the owner decides what to post): risk_coverage.png,
examples.md (real strings: where the model is right or abstains and the baselines are not),
post_draft.md (numbers pulled from results/). Writes the aggregate curve points to
results/<part>/risk_coverage.json (published).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from countrycode.evaluation.__main__ import gold_and_lexicon
from countrycode.evaluation.harness import item_correct
from countrycode.evaluation.plots import SERIES, curves, load_predictions, plot_risk_coverage

OUT = Path("docs/linkedin")
BASELINES = ("countrynames", "lexicon_fuzzy", "llm_qwen3.5_35b-a3b-q4_K_M")


def examples(preds, gold, n: int = 14) -> str:
    head = "| text (real) | field | truth | CountryCode-Decide @5 % | countrynames | fuzzy | LLM |"
    rows = [head, "|---|---|---|---|---|---|---|"]
    model = preds.get("model@0.05", {})
    picked = 0
    for g in sorted(gold, key=lambda g: (g.source == "gbif", -g.count)):
        m = model.get(g.item_id)
        # honest examples only: a real code as truth, the model answers it, baselines answer
        # a *different* code (abstentions and NONE-gold items are not counted as wins)
        if len(g.targets) != 1 or g.targets == ("NONE",) or m is None or m.target is None:
            continue
        if not item_correct(m, g):
            continue
        wrong = [
            b for b in BASELINES
            if b in preds and preds[b][g.item_id].target not in (None, g.targets[0])
            and not item_correct(preds[b][g.item_id], g)
        ]  # fmt: skip
        if len(wrong) < 2:
            continue

        def show(p) -> str:
            return p.target if p.target else "abstain " + "/".join(p.candidates[:2])

        cells = [show(preds[b][g.item_id]) if b in preds else "–" for b in BASELINES]
        rows.append(f"| `{g.text}` | {g.field_kind} | {'/'.join(g.targets)} | {show(m)} | "
                    + " | ".join(cells) + " |")  # fmt: skip
        picked += 1
        if picked >= n:
            break
    return "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="test")
    args = ap.parse_args()
    gold, _ = gold_and_lexicon()
    gold = [g for g in gold if g.split == args.part]
    preds = load_predictions(Path("data/processed/eval") / f"{args.part}_predictions.jsonl")
    data = curves(preds, gold)
    res = Path("results") / args.part / "risk_coverage.json"
    res.write_text(json.dumps(data, indent=1) + "\n")
    OUT.mkdir(parents=True, exist_ok=True)
    title = f"Real register & curated strings ({args.part}): answer more, err less"
    note = (
        f"n = {len(gold):,} real strings never used in training (GBIF curated variants + "
        "Colorado/Connecticut registry fields); error = strict match to the gold code."
    )
    plot_risk_coverage(data, OUT / "risk_coverage.png", title, note)
    (OUT / "examples.md").write_text(examples(preds, gold))
    hyp = json.loads((Path("results") / args.part / "hypotheses.json").read_text())
    lat = json.loads(Path("results/latency.json").read_text())
    model = json.loads((Path("results") / args.part / "model.json").read_text())["overall"]
    op = json.loads((Path("results") / args.part / "model@0.05.json").read_text())["overall"]
    draft = f"""# Draft — owner to edit (numbers from results/{args.part})

Country fields in company registers are a mess: "Deleware", "England and Whales", "Østerrike",
OCR'd handwriting, "n/a". I trained a small open model ({lat["int8_bytes"] / 1e6:.0f} MB, \
{lat["p50_ms"]:.1f} ms per string on a CPU) that turns them into ISO 3166 codes — and says
"not sure" instead of guessing.

- Accuracy on real strings it never saw: {model["accuracy"]:.1%} (answer everything)
- At a certified 5 % error target it answers {op["answered"]:.0%} of strings automatically \
(realised error {op["risk_answered"]:.1%}); the rest go to a human with 1–3 candidates.
- Trained on curated names + quality-gated synthetic noise (keyboard typos, real OCR
  misreadings, transliterations…), zero real register strings.
- Runs in the browser. Apache-2.0.

Pre-registered hypotheses: {
        ", ".join(
            f"{k} {'✓' if v.get('pass') else '✗'}" for k, v in hyp.items() if k.startswith("H")
        )
    }

See {OUT / "risk_coverage.png"} and {OUT / "examples.md"}.
"""
    (OUT / "post_draft.md").write_text(draft)
    print(f"assets in {OUT}; curve points in {res}; series {list(data)}; legend {list(SERIES)[:1]}")


if __name__ == "__main__":
    main()
