"""LinkedIn assets for the documented release (v1.2.1), from results/ only.

    uv run python scripts/linkedin_assets.py

Writes to docs/linkedin/ (private — the owner decides what to post):
- post_en.md, post_tr.md: drafts whose numbers come from results/ (never typed);
- where_it_helps.png: the model, the zero-shot LLM and the context lexicon on the hard strings of
  the fresh registers and on held-out messy strings of the training registers;
- examples.png: hand-picked typo strings with the release model's real answers
  (docs/linkedin/examples-<version>.json, written by scripts/run_examples.py with
  configs/docs/linkedin_examples.yaml).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "linkedin"
LLM = "llm_qwen3.5_35b-a3b-q4_K_M"
# Validated categorical slots 1–3 (scripts/validate_palette.js, light surface): every bar carries
# a direct value label because slot 3 is below 3:1 contrast.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = {
    "model": ("#2a78d6", "CountryCode-Decide (75 MB, CPU)"),
    "llm": ("#eb6834", "35B LLM, zero-shot (GPU)"),
    "lexicon_context": ("#1baf7a", "Name lexicon + context rules"),
}


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _release(res: Path) -> dict:
    return {"version": None, "part": "test", "preregistered": None, **_load(res / "release.json")}


def where_it_helps(res: Path) -> list[dict]:
    """Two groups: the hand-labelled strings of the fresh registers, and the held-out strings of
    the training registers."""
    part = res / _release(res)["part"]
    v11 = part.parent / "test_v11"

    def acc(rep: dict, key: str | None) -> float:
        return rep["slices"][key]["accuracy"] if key else rep["overall"]["accuracy"]

    groups = []
    for label, d, key in (
        ("Fresh registers:\nstrings a person had to label", part, "label=reviewed_proposal"),
        ("Messy strings of the\nregisters it was trained on", v11, None),
    ):
        reps = {"model": _load(d / "model.json"), "llm": _load(d / f"{LLM}.json"),
                "lexicon_context": _load(d / "lexicon_context.json")}  # fmt: skip
        m = reps["model"]
        n = m["slices"][key]["n"] if key else m["overall"]["n"]
        groups.append({"label": label, "n": n,
                       "values": {k: round(acc(v, key), 4) for k, v in reps.items()}})  # fmt: skip
    return groups


def plot_where_it_helps(groups: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    width, gap = 0.17, 0.03
    for gi, g in enumerate(groups):
        for si, (key, (color, _)) in enumerate(SERIES.items()):
            x = gi + (si - 1) * (width + gap)
            v = 100 * g["values"][key]
            ax.bar(x, v, width=width, color=color, zorder=2)
            ax.annotate(f"{v:.1f} %", (x, v), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=12, color=INK)  # fmt: skip
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g['label']}\n(n = {g['n']:,})" for g in groups], fontsize=12,
                       color=INK2)  # fmt: skip
    ax.set_ylim(0, 100)
    ax.set_ylabel("accuracy, every string answered (%)", color=INK2, fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2, length=0)
    from matplotlib.patches import Rectangle

    handles = [Rectangle((0, 0), 1, 1, color=c) for c, _ in SERIES.values()]
    ax.legend(handles, [lab for _, lab in SERIES.values()], frameon=False, fontsize=11,
              labelcolor=INK2, loc="lower left", ncols=3, bbox_to_anchor=(0, 1.02))  # fmt: skip
    fig.suptitle("Where a small model earns its place — and where a lexicon is enough",
                 x=0.06, ha="left", y=0.95, fontsize=16, color=INK)  # fmt: skip
    fig.text(0.06, 0.045, "Country and jurisdiction fields of company registers, distinct strings. "
             "Fresh registers: New York, Oregon. Trained on: Colorado, Connecticut, UK, Norway.",
             fontsize=9, color=INK2)  # fmt: skip
    fig.text(0.06, 0.018, "Method, data and all numbers: github.com/aliildan/country-code-decide",
             fontsize=9, color=INK2)  # fmt: skip
    ax.set_xlim(-0.55, len(groups) - 0.45)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_examples(ex: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = ex["examples"]
    fig = plt.figure(figsize=(12, 6.75), dpi=150, facecolor=SURFACE)
    fig.text(0.06, 0.9, "Messy in, ISO code out — or “not sure”", fontsize=20, color=INK)
    fig.text(0.06, 0.845, f"Real answers of the released model at a certified "
             f"{round(ex['alpha'] * 100)} % error target", fontsize=12, color=INK2)  # fmt: skip
    y0, dy = 0.74, 0.083
    for i, e in enumerate(rows):
        y = y0 - i * dy
        ctx = "US jurisdiction" if e["field"] == "jurisdiction" else f"{e['field']} field"
        fig.text(0.06, y, e["text"], fontsize=15, color=INK, family="DejaVu Sans Mono")
        fig.text(0.40, y, ctx, fontsize=11, color=INK2)
        if e["abstain"]:
            ans = f"not sure → review   ({', '.join(e['candidates'][:3])})"
            fig.text(0.58, y, ans, fontsize=14, color=INK2)
        else:
            fig.text(0.58, y, e["target"], fontsize=15, color=INK, weight="bold",
                     family="DejaVu Sans Mono")  # fmt: skip
            fig.text(0.67, y, f"{e['name']}  ·  {e['confidence']:.2f}", fontsize=13, color=INK2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def _num(res: Path) -> dict:
    """Every number the post uses, from the release's results."""
    rel = _release(res)
    part = res / rel["part"]
    v11 = part.parent / "test_v11"
    lat = _load(res / f"latency-{rel['version']}.json")
    llm_ms = max(_load(part / f"{LLM}.json").get("latency_ms_per_item") or 0.0,
                 _load(res / (rel["preregistered"] or rel["part"]) / f"{LLM}.json")
                 .get("latency_ms_per_item") or 0.0)  # fmt: skip
    op = _load(part / "model@0.05.json")["overall"]
    return {
        "mb": lat["int8_bytes"] / 1e6, "ms": lat["p50_ms"], "slower": llm_ms / lat["p50_ms"],
        "n11": _load(v11 / "model.json")["overall"]["n"],
        "m11": _load(v11 / "model.json")["overall"]["accuracy"],
        "l11": _load(v11 / f"{LLM}.json")["overall"]["accuracy"],
        "c11": _load(v11 / "lexicon_context.json")["overall"]["accuracy"],
        "n12": _load(part / "model.json")["overall"]["n"],
        "m12": _load(part / "model.json")["overall"]["accuracy"],
        "c12": _load(part / "lexicon_context.json")["overall"]["accuracy"],
        "ans": op["answered"], "risk": op["risk_answered"],
    }  # fmt: skip


def post_text(res: Path, links: dict, lang: str) -> str:
    """configs/docs/linkedin_post_<lang>.md.tmpl filled with the release's numbers."""
    v = _num(res)
    if lang == "tr":
        pct = lambda x: f"%{100 * x:.1f}".replace(".", ",")  # noqa: E731
    else:
        pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    fields = {k: pct(v[k]) for k in ("m11", "l11", "c11", "m12", "c12", "ans", "risk")}
    fields |= {"mb": f"{v['mb']:.0f}", "ms": f"{v['ms']:.0f}", "slower": f"{v['slower']:.0f}",
               "n11": f"{v['n11']:,}", "repo": links["repo"],
               "hf": f"https://huggingface.co/{links['hf_model']}",
               "paper": f"https://doi.org/{links['paper_doi']}" if links.get("paper_doi")
               else "(DOI to follow)"}  # fmt: skip
    tmpl = (ROOT / "configs" / "docs" / f"linkedin_post_{lang}.md.tmpl").read_text(encoding="utf-8")
    return tmpl.format(**fields)


def main() -> int:
    res = ROOT / "results"
    links = yaml.safe_load((ROOT / "configs/docs/links.yaml").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    for lang in ("en", "tr"):
        (OUT / f"post_{lang}.md").write_text(post_text(res, links, lang), encoding="utf-8")
    plot_where_it_helps(where_it_helps(res), OUT / "where_it_helps.png")
    ex = OUT / f"examples-{_release(res)['version']}.json"
    if ex.exists():
        plot_examples(_load(ex), OUT / "examples.png")
    else:
        print(f"no {ex.name}: run scripts/run_examples.py --config configs/docs/"
              f"linkedin_examples.yaml --results docs/linkedin first", file=sys.stderr)  # fmt: skip
    print(f"assets in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
