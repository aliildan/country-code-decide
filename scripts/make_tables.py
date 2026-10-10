"""LaTeX tables, number macros and figure data for the paper, generated from results/ (CLAUDE.md §6:
every number in the paper comes from results/; tables are generated, never typed).

    uv run python scripts/make_tables.py            # results/tables/*.tex
    uv run python scripts/make_tables.py --check    # CI: fail if they are stale
    uv run python scripts/make_tables.py --bundle dist/countrycode-small-v1.2   # refresh the
                                                    # release calibration copy first

The paper (docs/paper/main.tex) \\input{}s the tables and uses the macros for every number in its
text, so a changed result changes the paper.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LABELS = {
    "model": "CountryCode-Decide (always answers)",
    "model@0.02": "CountryCode-Decide at $\\alpha = 2\\%$",
    "model@0.05": "CountryCode-Decide at $\\alpha = 5\\%$",
    "v1_model": "CountryCode-Decide v1 (previous release)",
    "llm_qwen3.5_35b-a3b-q4_K_M": "Qwen3.5-35B-A3B, zero-shot (GPU)",
    "e5_knn": "multilingual-e5 nearest name",
    "lexicon_fuzzy": "Lexicon + RapidFuzz",
    "lexicon_exact": "Lexicon, exact match",
    "ons_style": "ONS-style index + Levenshtein",
    "countrynames": "countrynames (OpenSanctions)",
    "pycountry": "pycountry",
    "coco": "country\\_converter",
}
ORDER = list(LABELS)
SLICES = (
    ("lexicon=seen", "name in the lexicon (exact key)"),
    ("lexicon=novel", "no exact lexicon match"),
    ("field_kind=country", "country fields"),
    ("field_kind=jurisdiction", "jurisdiction fields"),
    ("gold=country", "gold: a country"),
    ("gold=us_state", "gold: a US state"),
    ("gold=none", "gold: not a place"),
    ("gold=ambiguous", "gold: ambiguous set"),
    ("source=us-ny", "New York"),
    ("source=us-or", "Oregon"),
)
SHORT = {
    "model": "Model",
    "llm_qwen3.5_35b-a3b-q4_K_M": "LLM",
    "e5_knn": "E5 NN",
    "lexicon_fuzzy": "Fuzzy",
    "countrynames": "countrynames",
}
SLICE_COLS = ("model", "llm_qwen3.5_35b-a3b-q4_K_M", "e5_knn", "lexicon_fuzzy", "countrynames")
PREREG = {
    "P1": "The register 2\\,\\% certificate exists and holds on unseen registers",
    "P2": "The 2\\,\\% operating point answers $\\geq 60\\,\\%$ of the fresh strings",
    "P3": "Vocabulary trimming: $\\leq 80$\\,MB, CPU p50 $\\leq 5$\\,ms, identical tokens, "
    "select accuracy within 0.3\\,pp",
    "P4": "Accuracy $\\geq$ best non-LLM baseline + 5\\,pp on unseen registers",
    "A1": "$\\alpha = 1\\,\\%$ (reported, not judged)",
}
# The owner's first G1 audit (decisions.md 2026-10-10): the releases the v1/v1.1 models used.
FIRST_AUDIT = ("affix@1", "format@3", "geo@1", "junk@1", "keyboard@1", "llm@1", "llmjunk@1",
               "ocr@1", "pairs@1", "translit@1")  # fmt: skip
CURVES = ("model", "llm_qwen3.5_35b-a3b-q4_K_M", "e5_knn", "lexicon_fuzzy", "countrynames")


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _pct(x: float | None, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{100 * x:.{nd}f}"


def _num(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def _release(res: Path) -> dict:
    return {"version": None, "part": "test", **_load(res / "release.json")}


def _reports(res: Path, part: str) -> dict[str, dict]:
    return {p.stem: _load(p) for p in sorted((res / part).glob("*.json"))
            if "overall" in _load(p)}  # fmt: skip


def _cmd(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def _lam(cal: dict, domain: str, alpha: str) -> str:
    lam = cal.get(domain, {}).get("thresholds", {}).get(alpha)
    return "--" if lam is None else f"{lam:.3f}"


def macros(res: Path) -> str:
    """\\newcommand per number used in the paper's text."""
    rel = _release(res)
    part = rel["part"]
    reps = _reports(res, part)
    m = reps["model"]["overall"]
    h = _load(res / part / "hypotheses.json")
    op2, op5 = reps["model@0.02"]["overall"], reps["model@0.05"]["overall"]
    lat = _load(res / f"latency-{rel['version']}.json")
    trim = _load(res / f"trim-{rel['version']}.json")
    gold = _load(res / "real" / "gold_quality.json")
    web = _load(res / f"web-parity-{rel['version']}.json")
    tr = _load(res / "train" / f"{rel.get('train_run')}.json")
    cal = _load(res / f"calibration-{rel['version']}.json").get("domains", {})
    llm = next(k for k in reps if k.startswith("llm_"))
    base = {k: v["overall"]["accuracy"] for k, v in reps.items()
            if k in LABELS and not k.startswith(("model", "llm_", "v1_"))}  # fmt: skip
    best = max(base, key=lambda k: base[k])
    seen = h.get("seen", {}).get("by_source", {})
    by = tr.get("by_source", {})
    n_syn = sum(v for k, v in by.items() if k.startswith("synthetic:"))
    n_real = sum(v for k, v in by.items() if k.startswith("real:"))
    n_lex = sum(v for k, v in by.items() if k.split(":")[0] == "lexicon")
    total = sum(by.values()) or 1
    ci = m.get("accuracy_ci95", [math.nan, math.nan])
    sl = reps["model"].get("slices", {})
    lines = [
        _cmd("TestN", _num(m["n"])),
        _cmd("TestNY", _num(seen.get("us-ny", {}).get("items", 0))),
        _cmd("TestOR", _num(seen.get("us-or", {}).get("items", 0))),
        _cmd("TestSeen", _num(sum(s.get("seen", 0) for s in seen.values()))),
        _cmd("TestNovelN", _num(sl.get("lexicon=novel", {}).get("n", 0))),
        _cmd("AccNovel", _pct(sl.get("lexicon=novel", {}).get("accuracy"))),
        _cmd("TestNoneN", _num(m.get("none_items", 0))),
        _cmd("TestAmbigN", _num(m.get("ambiguous_items", 0))),
        _cmd("ModelReleases", str(len(tr.get("gens", [])))),
        _cmd("AccModel", _pct(m["accuracy"])),
        _cmd("AccCILo", _pct(ci[0])),
        _cmd("AccCIHi", _pct(ci[1])),
        _cmd("CtryAccModel", _pct(m["country_accuracy"])),
        _cmd("AurcModel", f"{m['aurc']:.3f}"),
        _cmd("EceModel", f"{m['ece']:.3f}"),
        _cmd("OpTwoAnswered", _pct(op2["answered"])),
        _cmd("OpTwoRisk", _pct(op2["risk_answered"])),
        _cmd("OpTwoNoneFA", _pct(op2.get("none_false_accept"))),
        _cmd("OpFiveAnswered", _pct(op5["answered"])),
        _cmd("OpFiveRisk", _pct(op5["risk_answered"])),
        _cmd("POneAnsweredN", _num(h["P1"]["answered_n"])),
        _cmd("POneErrors", _num(h["P1"]["errors"])),
        _cmd("POnePValue", f"{h['P1']['p_value_risk_le_2pct']:.3f}"),
        _cmd("AccLLM", _pct(reps[llm]["overall"]["accuracy"])),
        _cmd("LLMCovFive", _pct(reps[llm]["overall"]["coverage_at_risk"]["0.05"], 0)),
        _cmd("LLMms", f"{reps[llm]['latency_ms_per_item']:.0f}"),
        _cmd("LLMSlower", f"{reps[llm]['latency_ms_per_item'] / lat['p50_ms']:.0f}"),
        _cmd("BestBaseline", LABELS[best]),
        _cmd("AccBest", _pct(base[best])),
        _cmd("AccFuzzy", _pct(base.get("lexicon_fuzzy"))),
        _cmd("AccExact", _pct(base.get("lexicon_exact"))),
        _cmd("AccCountrynames", _pct(base.get("countrynames"))),
        _cmd("AccPycountry", _pct(base.get("pycountry"))),
        _cmd("AccCoco", _pct(base.get("coco"))),
        _cmd("AccVOne", _pct(reps.get("v1_model", {}).get("overall", {}).get("accuracy"))),
        _cmd("SizeMB", f"{lat['int8_bytes'] / 1e6:.1f}"),
        _cmd("PFifty", f"{lat['p50_ms']:.1f}"),
        _cmd("PNinetyFive", f"{lat['p95_ms']:.1f}"),
        _cmd("BatchMs", f"{lat['batch_ms_per_item']:.1f}"),
        _cmd("UntrimmedMB", f"{trim.get('size_mb_a', math.nan):.1f}"),
        _cmd("VocabOld", _num(trim.get("vocab_old", 0))),
        _cmd("VocabNew", _num(trim.get("vocab_new", 0))),
        _cmd("TokensChecked", _num(trim.get("tokens_checked", 0))),
        _cmd("TrimDrop", f"{100 * h['P3']['select_accuracy_drop']:.2f}"),
        _cmd("GoldN", _num(gold.get("n", 0))),
        _cmd("GoldAgree", _pct(gold.get("agreement"))),
        _cmd("GoldDisagree", _num(gold.get("disagree", 0))),
        _cmd("GoldUpper", _pct(gold.get("gold_error_upper_95"))),
        _cmd("TrainN", _num(total)),
        _cmd("TrainLex", _num(n_lex)),
        _cmd("TrainSyn", _num(n_syn)),
        _cmd("TrainReal", _num(n_real)),
        _cmd("SynShare", _pct(n_syn / total)),
        _cmd("WebN", _num(web.get("n", 0))),
        _cmd("WebAgreeTwo", _pct(web.get("decision_agreement", {}).get("0.02"))),
        _cmd("WebTop", _pct(web.get("top1_agreement"))),
        _cmd("WebMeanDp", f"{web.get('mean_abs_dp_top', math.nan):.3f}"),
        _cmd("WebMaxDp", f"{web.get('max_abs_dp', math.nan):.2f}"),
        _cmd("RegCalibN", _num(cal.get("register", {}).get("n_calib", 0))),
        _cmd("CurCalibN", _num(cal.get("curated", {}).get("n_calib", 0))),
        _cmd("RegLambdaTwo", _lam(cal, "register", "0.02")),
        _cmd("RegLambdaFive", _lam(cal, "register", "0.05")),
        _cmd("CurLambdaFive", _lam(cal, "curated", "0.05")),
    ]  # fmt: skip
    gens = _load(res / "synthetic" / "synth-v2.json").get("generators", {})
    first = {t: gens[t]["G1_audit"] for t in FIRST_AUDIT if t in gens}
    lines += [
        _cmd("FirstAuditReleases", str(len(first))),
        _cmd("FirstAuditFailed", str(sum(1 for a in first.values() if a.get("errors", 0)))),
        _cmd("FirstAuditN", _num(sum(a.get("audited", 0) + a.get("stale", 0)
                                     for a in first.values()))),
        _cmd("FirstAuditErrors", _num(sum(a.get("errors", 0) for a in first.values()))),
    ]  # fmt: skip
    lex = _load(res / "lexicon" / "stats.json")
    ing = _load(res / "real" / "ingest.json")
    lines += [
        _cmd("LexRows", _num(lex.get("rows", 0))),
        _cmd("LexLangs", _num(lex.get("languages", 0))),
        _cmd("LexKeys", _num(lex.get("keys", 0))),
        _cmd("LexAmbig", _num(lex.get("ambiguous_keys", 0))),
        _cmd("CldrVersion", str(lex.get("cldr", "?"))),
        _cmd("IngestItems", _num(ing.get("items", 0))),
        _cmd("IngestRows", f"{ing.get('rows_covered', 0) / 1e6:.1f}"),  # millions
    ]
    for part_key, prefix in (("test_v11", "VEleven"), ("ood", "Ood"), ("select", "Select")):
        r = _reports(res, part_key)
        if "model" not in r:
            continue
        lines.append(_cmd(f"{prefix}N", _num(r["model"]["overall"]["n"])))
        lines.append(_cmd(f"{prefix}Acc", _pct(r["model"]["overall"]["accuracy"])))
        if llm in r:
            lines.append(_cmd(f"{prefix}LLM", _pct(r[llm]["overall"]["accuracy"])))
    return "\n".join(lines) + "\n"


def table_predictors(res: Path, part: str) -> str:
    reps = _reports(res, part)
    rows = [k for k in ORDER if k in reps and k != "model@0.01"]
    out = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Predictor & Acc. & Country acc. & Cov.\\ @1\\,\\% & Cov.\\ @5\\,\\% & ECE & "
        "ms/string \\\\",
        "\\midrule",
    ]
    for k in rows:
        o = reps[k]["overall"]
        cov = o.get("coverage_at_risk", {})
        ms = reps[k].get("latency_ms_per_item")
        ms_s = ("cached" if k.startswith("llm_") and (ms or 0) < 1.0
                else "--" if ms is None else f"{ms:.1f}")  # fmt: skip
        ece = o.get("ece")
        out.append(
            f"{LABELS[k]} & {_pct(o['accuracy'])} & {_pct(o.get('country_accuracy'))} & "
            f"{_pct(cov.get('0.01'))} & {_pct(cov.get('0.05'))} & "
            f"{'--' if ece is None or ece != ece else f'{ece:.3f}'} & {ms_s} \\\\"
        )
        if k == "v1_model":
            out.append("\\midrule")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def table_slices(res: Path, part: str) -> str:
    reps = _reports(res, part)
    cols = [c for c in SLICE_COLS if c in reps]
    head = " & ".join(["Slice", "$n$"] + [SHORT[c] for c in cols])
    out = ["\\begin{tabular}{l r " + "r" * len(cols) + "}", "\\toprule", head + " \\\\",
           "\\midrule"]  # fmt: skip
    for key, label in SLICES:
        s = reps["model"].get("slices", {}).get(key)
        if not s or not s.get("n"):
            continue
        cells = [_pct(reps[c].get("slices", {}).get(key, {}).get("accuracy")) for c in cols]
        out.append(f"{label} & {_num(s['n'])} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def table_releases(res: Path) -> str:
    rel = _release(res)
    gens = _load(res / "synthetic" / "synth-v2.json").get("generators", {})
    used = set(_load(res / "train" / f"{rel.get('train_run')}.json").get("gens", []))
    out = ["\\begin{tabular}{lrrrll}", "\\toprule",
           "Release & Items & Audited & Errors & G1 & In the model \\\\", "\\midrule"]  # fmt: skip
    for tag, g in gens.items():
        a = g["G1_audit"]
        audited, errors = a.get("audited", 0), a.get("errors", 0)
        if a.get("g1_pass"):
            g1 = "pass"
        elif audited:
            g1 = "fail"
        elif a.get("stale"):
            g1 = "stale"
        else:
            g1 = "not audited"
        out.append(f"{tag.replace('@', '@')} & {_num(g['items'])} & {_num(audited)} & "
                   f"{_num(errors)} & {g1} & {'yes' if tag in used else ''} \\\\")  # fmt: skip
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def table_prereg(res: Path) -> str:
    h = _load(res / _release(res)["part"] / "hypotheses.json")
    out = ["\\begin{tabular}{l >{\\raggedright\\arraybackslash}p{6.2cm} "
           ">{\\raggedright\\arraybackslash}p{4.6cm} l}", "\\toprule",
           "& Hypothesis & Observed & Result \\\\", "\\midrule"]  # fmt: skip
    p1, p2, p3, p4, a1 = (h.get(k, {}) for k in ("P1", "P2", "P3", "P4", "A1"))
    obs = {
        "P1": f"$\\lambda = {p1.get('lambda', math.nan):.3f}$; {p1.get('errors')} errors in "
        f"{p1.get('answered_n')} answered; $p = {p1.get('p_value_risk_le_2pct', math.nan):.3f}$",
        "P2": f"{_pct(p2.get('answered'))}\\,\\% answered",
        "P3": f"{p3.get('size_mb', math.nan):.1f}\\,MB, {p3.get('cpu_p50_ms', math.nan):.1f}\\,ms, "
        f"tokens {_pct(p3.get('tokens_identical'))}\\,\\% identical, "
        f"$-{100 * p3.get('select_accuracy_drop', math.nan):.2f}$\\,pp",
        "P4": f"{_pct(p4.get('model_accuracy'))} vs {_pct(p4.get('best_non_llm_accuracy'))}\\,\\% "
        f"({LABELS.get(p4.get('best_non_llm', ''), p4.get('best_non_llm', ''))})",
        "A1": "not certified" if not a1.get("certified") else "certified",
    }
    for k, text in PREREG.items():
        v = h.get(k, {})
        res_s = {True: "met", False: "\\textbf{not met}", None: "reported"}[v.get("pass")]
        out.append(f"{k} & {text} & {obs[k]} & {res_s} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def table_calibration(res: Path) -> str:
    cal = _load(res / f"calibration-{_release(res)['version']}.json").get("domains", {})
    out = ["\\begin{tabular}{lrrrrr}", "\\toprule",
           "Domain & $n_{\\text{calib}}$ & $T_{\\text{country}}$ & $\\lambda_{1\\%}$ & "
           "$\\lambda_{2\\%}$ & $\\lambda_{5\\%}$ \\\\", "\\midrule"]  # fmt: skip
    for dom, c in sorted(cal.items()):
        t = c.get("thresholds", {})
        lam = [("--" if t.get(a) is None else f"{t[a]:.3f}") for a in ("0.01", "0.02", "0.05")]
        out.append(f"{dom} & {_num(c.get('n_calib', 0))} & {c['temps'][0]:.2f} & "
                   + " & ".join(lam) + " \\\\")  # fmt: skip
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def table_training(res: Path) -> str:
    rel = _release(res)
    by = _load(res / "train" / f"{rel.get('train_run')}.json").get("by_source", {})
    total = sum(by.values()) or 1
    groups: dict[str, int] = {}
    for k, v in by.items():
        if k.startswith("synthetic:"):
            groups[f"generated: {k.removeprefix('synthetic:')}"] = v
        elif k.startswith("real:"):
            groups["real register strings"] = groups.get("real register strings", 0) + v
        else:
            groups["lexicon names and codes"] = groups.get("lexicon names and codes", 0) + v
    out = ["\\begin{tabular}{lrr}", "\\toprule", "Source & Examples & Share (\\%) \\\\",
           "\\midrule"]  # fmt: skip
    for k, v in groups.items():
        out.append(f"{k} & {_num(v)} & {_pct(v / total)} \\\\")
    out += ["\\midrule", f"total & {_num(total)} & 100.0 \\\\", "\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def curve_csvs(res: Path, part: str) -> dict[str, str]:
    rc = _load(res / part / "risk_coverage.json")
    out = {}
    for name in CURVES:
        c = rc.get(name)
        if not c or not c.get("coverage"):
            continue
        rows = "".join(f"{100 * x:.2f},{100 * y:.3f}\n"
                       for x, y in zip(c["coverage"], c["risk"], strict=True))  # fmt: skip
        out[f"{part}-{name.replace('.', '_')}.csv"] = "coverage,risk\n" + rows
    return out


def outputs(res: Path) -> dict[str, str]:
    part = _release(res)["part"]
    files = {
        "tables/macros.tex": macros(res),
        "tables/test.tex": table_predictors(res, part),
        "tables/slices.tex": table_slices(res, part),
        "tables/releases.tex": table_releases(res),
        "tables/prereg.tex": table_prereg(res),
        "tables/calibration.tex": table_calibration(res),
        "tables/training.tex": table_training(res),
    }
    for p in ("test_v11", "ood"):
        if (res / p / "model.json").exists():
            files[f"tables/{p}.tex"] = table_predictors(res, p)
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bundle", type=Path, default=None)
    ap.add_argument("--figures", type=Path, default=None, help="write the curve CSVs here")
    args = ap.parse_args(argv)
    res = ROOT / "results"
    if args.bundle:
        version = args.bundle.name.removeprefix("countrycode-")
        shutil.copy(args.bundle / "calibration.json", res / f"calibration-{version}.json")
    stale = []
    for rel_path, text in outputs(res).items():
        target = res / rel_path
        if args.check:
            if not target.exists() or target.read_text(encoding="utf-8") != text:
                stale.append(rel_path)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
    if args.figures:  # pgfplots input for the (private) paper build; never in results/
        args.figures.mkdir(parents=True, exist_ok=True)
        for name, text in curve_csvs(res, _release(res)["part"]).items():
            (args.figures / name).write_text(text, encoding="utf-8")
    if stale:
        print(f"make_tables: stale {stale}; run scripts/make_tables.py", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
