"""Figures for the release notes and LinkedIn (plan-05 Task 6), from the private per-item
predictions of the final evaluation. Writes PNGs to docs/linkedin/ (private, the owner decides what
to post) and the aggregate curve points to results/<part>/risk_coverage.json (published).

Palette: reference categorical slots 1–5 (validated: CVD ΔE ≥ 9.1, normal ≥ 19.6); slots 3–5 are
below 3:1 on the surface, so every line carries a direct label.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from countrycode.evaluation.harness import Prediction, answered_correct
from countrycode.evaluation.metrics import risk_coverage
from countrycode.real.gold import GoldItem

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = {  # fixed order: the model first, never re-coloured by rank
    "model": ("#2a78d6", "CountryCode-Decide (small)"),
    "llm_qwen3.5_35b-a3b-q4_K_M": ("#eb6834", "Qwen3.5-35B LLM (zero-shot)"),
    "lexicon_fuzzy": ("#1baf7a", "CLDR lexicon + fuzzy match"),
    "e5_knn": ("#eda100", "multilingual-e5-small nearest name"),
    "lexicon_context": ("#7f5fb8", "lexicon + context rules"),
    "countrynames": ("#e87ba4", "countrynames (OpenSanctions)"),
}


def load_predictions(path: Path) -> dict[str, dict[str, Prediction]]:
    out: dict[str, dict[str, Prediction]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        out[d["predictor"]][d["item_id"]] = Prediction(
            d["target"], tuple(d["candidates"]), float(d["confidence"])
        )
    return out


def curves(preds: dict[str, dict[str, Prediction]], gold: list[GoldItem]) -> dict[str, dict]:
    res = {}
    for name in SERIES:
        if name not in preds:
            continue
        p = [preds[name][g.item_id] for g in gold]
        conf = np.array([x.confidence if x.target is not None else -1.0 for x in p])
        ok = np.array([answered_correct(x, g) for x, g in zip(p, gold, strict=True)])
        answered = conf >= 0
        cov, risk = risk_coverage(conf[answered], ok[answered])
        cov = cov * answered.mean()
        idx = np.unique(np.linspace(0, len(cov) - 1, 60).astype(int)) if len(cov) else []
        res[name] = {
            "coverage": [float(cov[i]) for i in idx],
            "risk": [float(risk[i]) for i in idx],
        }
    return res


def plot_risk_coverage(data: dict[str, dict], out: Path, title: str, note: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ends = []
    for name, (color, label) in SERIES.items():
        if name not in data or not data[name]["coverage"]:
            continue
        c, r = np.array(data[name]["coverage"]), 100 * np.array(data[name]["risk"])
        ax.plot(c * 100, r, color=color, lw=2, label=label, solid_capstyle="round")
        ends.append([float(r[-1]), float(c[-1] * 100), label.split(" (")[0]])
    ymax = max([e[0] for e in ends] + [1.0])
    gap = 0.045 * ymax  # de-collide end labels vertically
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, x, text in ends:
        ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK2)  # fmt: skip
    ax.set_xlabel("strings answered automatically (%)", color=INK2)
    ax.set_ylabel("error rate among answered (%)", color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=12)
    ax.grid(color=GRID, lw=0.8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    if note:
        fig.text(0.01, 0.01, note, fontsize=7, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
