"""Render README.md and model_card.md from configs/docs/*.tmpl and results/ (CLAUDE.md §6:
every number comes from results/, tables are generated, never typed).

    uv run python scripts/make_docs.py          # write
    uv run python scripts/make_docs.py --check  # CI: fail if the docs are stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = {
    "README.md": "configs/docs/README.md.tmpl",
    "model_card.md": "configs/docs/model_card.md.tmpl",
}
PLACEHOLDER = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


HYPOTHESIS_KEYS = {
    "v1": ("H1", "H2", "H3", "H4", "H5", "H6"),
    "v11": ("R1", "R2", "R3", "R4", "R5", "R6"),
    "v12": ("P1", "P2", "P3", "P4", "A1"),
}


def _release(res: Path) -> dict:
    """The documented release (results/release.json); v1 (part `test`, H1–H6) by default."""
    return {"version": None, "part": "test", "hypotheses": "v1", **_load(res / "release.json")}


def _latency_record(res: Path) -> dict:
    """The documented release's own measurement; only an unversioned release (v1) reads
    results/latency.json — another bundle's numbers are never shown."""
    version = _release(res)["version"]
    return _load(res / f"latency-{version}.json") if version else _load(res / "latency.json")


def _latency(res: Path) -> str:
    lat = _latency_record(res)
    if not lat:
        return "_latency not measured yet_"
    return (
        f"p50 {lat['p50_ms']:.1f} ms, p95 {lat['p95_ms']:.1f} ms per string on CPU "
        f"({lat['threads']} threads, ONNX Runtime, tokenisation included); "
        f"model file {lat['int8_bytes'] / 1e6:.0f} MB"
    )


def _table(res: Path, part: str) -> str:
    p = res / part / "TABLE.md"
    if not p.exists():
        return f"_no {part} results yet_"
    lines = p.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if line.startswith("|"))


def _hypotheses(res: Path) -> str:
    rel = _release(res)
    h = _load(res / rel["part"] / "hypotheses.json")
    if not h:
        return "_test not evaluated yet_"
    rows = ["| hypothesis | result | evidence |", "|---|---|---|"]
    for k in HYPOTHESIS_KEYS[rel["hypotheses"]]:
        v = h.get(k, {})
        ok = {True: "met", False: "**not met**", None: "not evaluated"}[v.get("pass")]
        if v.get("pass") is None and k.startswith("A") and v:
            ok = "reported (not judged)"  # A-hypotheses are pre-registered as reported only
        ev = ", ".join(f"{a}={b:.3f}" if isinstance(b, float) else f"{a}={b}"
                       for a, b in v.items() if a != "pass")  # fmt: skip
        rows.append(f"| {k} | {ok} | {ev} |")
    return "\n".join(rows)


def _synthetic(res: Path) -> str:
    """Training data of the documented model (results/train/<run>.json) with the owner's G1
    audit per generator; falls back to the whole synthetic release when no run is documented."""
    r = _load(res / "synthetic" / "synth-v2.json")
    if not r:
        return "_no synthetic report_"
    rel = _release(res)  # a trimmed bundle's run is a copy: the release names the trained run
    run = (
        rel.get("train_run")
        or Path(_load(res / rel["part"] / "hypotheses.json").get("run") or "").name
    )
    tr = _load(res / "train" / f"{run}.json") if run else {}
    if not tr:
        rows = ["| generator | items | human audit (errors / audited) |", "|---|---|---|"]
        for tag, g in r["generators"].items():
            a = g["G1_audit"]
            rows.append(
                f"| {tag} | {g['items']:,} | {a.get('errors', 0)} / {a.get('audited', 0)} |"
            )
        return "\n".join(rows)
    by = tr["by_source"]
    rows = ["| generator | training examples | human audit (errors / audited) |", "|---|---|---|"]
    for tag in tr["gens"]:
        a = r["generators"].get(tag, {}).get("G1_audit", {})
        n = by.get(f"synthetic:{tag}", 0)
        rows.append(f"| {tag} | {n:,} | {a.get('errors', 0)} / {a.get('audited', 0)} |")
    n_real = sum(v for k, v in by.items() if k.startswith("real:"))
    n_lex = sum(v for k, v in by.items() if k == "lexicon" or k.startswith("lexicon:"))
    n_syn = sum(v for k, v in by.items() if k.startswith("synthetic:"))
    total = sum(by.values())
    rows.append("")
    rows.append(
        f"- {total:,} training examples: {n_lex:,} lexicon names and codes "
        f"({100 * n_lex / total:.1f} %), {n_syn:,} generated ({100 * n_syn / total:.1f} %), "
        f"{n_real:,} real register strings ({100 * n_real / total:.1f} %)."
    )
    if n_real:
        rows.append(
            "- Real register strings are labelled by a person or by the register's own code; "
            "none shares a normalised key with any evaluation part."
        )
    else:
        rows.append("- No real register string in training.")
    return "\n".join(rows)


def _lexicon(res: Path) -> str:
    s = _load(res / "lexicon" / "stats.json")
    if not s:
        return "_no lexicon stats_"
    return (f"{s['rows']:,} names in {s['languages']:,} languages (CLDR {s.get('cldr', '?')}, "
            f"Wikidata, codes, reviewed overrides); {s['keys']:,} distinct keys, "
            f"{s['ambiguous_keys']:,} kept as genuinely ambiguous")  # fmt: skip


def _operating_point(res: Path) -> tuple[str, str]:
    """(alpha file suffix, sentence about the certificate) for the documented release: v1.2
    quotes 2 % only when P1 holds on the fresh registers; earlier releases certified 5 %."""
    rel = _release(res)
    if rel["hypotheses"] != "v12":
        return "0.05", (" The 1 % and 2 % targets could not be certified on this calibration "
                        "set.")  # fmt: skip
    h = _load(res / rel["part"] / "hypotheses.json")
    one = ("" if h.get("A1", {}).get("certified")
           else " The 1 % target could not be certified.")  # fmt: skip
    g = _load(res / "real" / "gold_quality.json")
    bound = g.get("gold_error_upper_95")
    if bound is not None and bound > 0.02:  # pre-registered caveat (preregistration-v1.2)
        one += (f" Caveat: the calibration strings' gold-label error is bounded only at "
                f"{100 * bound:.1f} % (blind re-labelling), above the 2 % target.")  # fmt: skip
    if h.get("P1", {}).get("pass"):
        return "0.02", (" The 2 % certificate (fixed-sequence test on the register calibration "
                        "strings) is not contradicted on the fresh registers." + one)  # fmt: skip
    return "0.05", (
        " A 2 % target was certified on the calibration strings but not confirmed "
        "on the fresh registers, so only the 5 % operating point is quoted." + one
    )


def _predictors(t: Path) -> dict[str, dict]:
    """Every predictor report of a test part except the model's own and its operating points."""
    skip = ("model", "hypotheses", "risk_coverage")
    return {
        p.stem: _load(p)
        for p in sorted(t.glob("*.json"))
        if p.stem not in skip and not p.stem.startswith("model@")
    }


BASELINE_LABELS = {
    "e5_knn": "multilingual-e5 nearest name",
    "lexicon_fuzzy": "lexicon + RapidFuzz",
    "lexicon_exact": "exact lexicon match",
    "ons_style": "ONS-style index + Levenshtein",
    "countrynames": "countrynames (OpenSanctions)",
    "pycountry": "pycountry",
    "coco": "country_converter",
}


def _pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def _headline(res: Path) -> str:
    t = res / _release(res)["part"]
    alpha, cert_note = _operating_point(res)
    m, op = _load(t / "model.json"), _load(t / f"model@{alpha}.json")
    if not m or not op:
        return "_test not evaluated yet_"
    others = _predictors(t)
    ctry = {k: v["overall"].get("country_accuracy") for k, v in others.items() if "overall" in v}
    ranked = sorted((k for k in ctry if ctry[k] is not None), key=lambda k: -ctry[k])
    lines = []
    if ranked:
        mc, top = m["overall"]["country_accuracy"], ranked[0]
        where = (f", best of all predictors (next: {top} {ctry[top]:.3f})" if mc >= ctry[top]
                 else f" (highest: {top} {ctry[top]:.3f})")  # fmt: skip
        lines.append(
            f"- Country-level accuracy (right country; a US state counts as US): {mc:.3f}{where}."
        )
    novel = m.get("slices", {}).get("lexicon=novel")
    if novel and novel.get("n"):
        lines.append(
            f"- On the {novel['n']} strings not in the lexicon (no exact name match) the model's "
            f"accuracy is {novel['accuracy']:.3f}."
        )
    o = op["overall"]
    n_none = f" (n = {o['none_items']})" if o.get("none_items") else ""
    lines.append(
        f"- At the certified {round(float(alpha) * 100)} % error target the model answers "
        f"{o['answered']:.0%} of the strings automatically with a realised error of "
        f"{100 * o['risk_answered']:.1f} %; non-places get a code "
        f"{100 * o['none_false_accept']:.1f} % of the time{n_none}.{cert_note}"
    )
    p50 = _latency_record(res).get("p50_ms", 1.0)
    for name, r in others.items():
        if not name.startswith("llm_"):
            continue
        cov = r["overall"].get("coverage_at_risk", {}).get("0.05", 0.0)
        la, ma = r["overall"]["accuracy"], m["overall"]["accuracy"]
        acc = (f"is more accurate when every string must be answered ({la:.3f} vs {ma:.3f})"
               if la > ma else f"is less accurate when every string must be answered ({la:.3f} "
               f"vs the model's {ma:.3f})")  # fmt: skip
        conf = (", but its confidence cannot be thresholded: coverage at 5 % risk is 0%" if not cov
                else f"; its own confidence covers {cov:.0%} of the strings at 5 % empirical "
                "risk, without a certificate")  # fmt: skip
        lines.append(
            f"- The zero-shot LLM ({name.removeprefix('llm_')}) {acc}{conf}. "
            f"It is ~{r['latency_ms_per_item'] / max(p50, 1e-9):.0f}× slower (the LLM on a GPU, "
            "this model on a CPU)."
        )
    return "\n".join(lines) + "\n"


def _gold_quality(res: Path) -> str:
    g = _load(res / "real" / "gold_quality.json")
    if not g or not g.get("n"):
        return "_no blind re-labelling yet_"
    return (f"The register calibration strings were labelled a second time, blind (no proposal and "
            f"no first label shown): agreement {100 * g['agreement']:.1f} % on {g['n']:,} strings "
            f"({g['disagree']} disagreements); one-sided 95 % upper bound on disagreement "
            f"{100 * g['gold_error_upper_95']:.1f} %.")  # fmt: skip


def _certified(res: Path) -> str:
    """Certified error-rate targets per calibration domain of the documented release."""
    rel = _release(res)
    cal = _load(res / rel["part"] / "hypotheses.json").get("calibration", {})
    if not cal:
        return "_not calibrated yet_"
    parts = []
    for dom, c in sorted(cal.items()):
        ok = sorted((a for a, lam in c.get("thresholds", {}).items() if lam is not None), key=float)
        levels = ", ".join(f"{round(float(a) * 100)} %" for a in ok) or "none"
        parts.append(f"{dom}: {levels}")
    return "certified targets — " + "; ".join(parts)


def _at_a_glance(res: Path) -> str:
    """Headline table of the documented release: fresh-test accuracy, the certified operating
    point, the best non-LLM baseline, the LLM and size/latency (all from results/)."""
    rel = _release(res)
    t = res / rel["part"]
    alpha, _ = _operating_point(res)
    m, op = _load(t / "model.json"), _load(t / f"model@{alpha}.json")
    if not m or not op:
        return "_test not evaluated yet_"
    o, oo = m["overall"], op["overall"]
    ci = o.get("accuracy_ci95")
    acc = _pct(o["accuracy"]) + (f" (95 % CI {100 * ci[0]:.1f}–{100 * ci[1]:.1f})" if ci else "")
    name = "CountryCode-Decide " + (rel["version"] or "").replace("-v", " v")
    rows = [
        f"| Fresh registers, {o['n']:,} distinct strings, read once | {name.strip()} |",
        "|---|---|",
        f"| Accuracy, every string answered | {acc} |",
        f"| Right country (a US state counts as US) | {_pct(o['country_accuracy'])} |",
        f"| At the certified {round(float(alpha) * 100)} % error target | answers "
        f"{round(100 * oo['answered'])} % automatically, realised error "
        f"{_pct(oo['risk_answered'])} |",
    ]
    others = {k: v for k, v in _predictors(t).items() if "overall" in v}
    base = [k for k in others if not k.startswith("llm_") and k != "v1_model"]
    if base:
        best = max(base, key=lambda k: others[k]["overall"]["accuracy"])
        rows.append(f"| Best non-LLM baseline ({BASELINE_LABELS.get(best, best)}) | "
                    f"{_pct(others[best]['overall']['accuracy'])} |")  # fmt: skip
    p50 = _latency_record(res).get("p50_ms")
    for k in (k for k in others if k.startswith("llm_")):
        slow = (f", ~{others[k]['latency_ms_per_item'] / p50:.0f}× slower (LLM on a GPU, "
                "this model on a CPU)"
                if p50 and others[k].get("latency_ms_per_item") else "")  # fmt: skip
        rows.append(f"| Zero-shot LLM ({k.removeprefix('llm_')}) | "
                    f"{_pct(others[k]['overall']['accuracy'])}{slow} |")  # fmt: skip
    lat = _latency_record(res)
    if lat:
        rows.append(f"| Model file · CPU latency | {lat['int8_bytes'] / 1e6:.0f} MB (int8 ONNX) · "
                    f"p50 {lat['p50_ms']:.1f} ms per string |")  # fmt: skip
    return "\n".join(rows)


def _examples(res: Path) -> str:
    """Hand-picked inputs run through the release bundle (scripts/run_examples.py)."""
    rel = _release(res)
    ex = _load(res / f"examples-{rel['version']}.json") if rel["version"] else {}
    if not ex.get("examples"):
        return "_no examples yet_"
    a = round(float(ex["alpha"]) * 100)
    rows = [
        f"| Input | Context | Answer at the {a} % error target | Confidence |",
        "|---|---|---|---|",
    ]
    for e in ex["examples"]:
        reg = e["register_country"]
        ctx = [f"{e['field']} field" if e["field"] != "unknown" else "",
               f"{reg} register" if reg != "unknown" else ""]  # fmt: skip
        if e["abstain"]:
            ans = f"abstains → review (candidates {', '.join(e['candidates'])})"
        else:
            ans = f"**{e['target']}** {e['name']}"
        rows.append(f"| `{e['text']}` | {', '.join(c for c in ctx if c) or 'none'} | {ans} | "
                    f"{e['confidence']:.2f} |")  # fmt: skip
    return "\n".join(rows)


def _links(res: Path) -> dict:
    import yaml

    p = res.parent / "configs" / "docs" / "links.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


def _badges(res: Path) -> str:
    links = _links(res)
    b = [
        "[![Licence: Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-blue.svg)](LICENSE)",
        "[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg?logo=python&logoColor"
        "=white)](pyproject.toml)",
    ]
    if hf := links.get("hf_model"):
        tag = hf.split("/")[-1].replace("-", "--")
        b.append(f"[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20model-{tag}"
                 f"-FFD21E.svg)](https://huggingface.co/{hf})")  # fmt: skip
    if d := links.get("paper_doi"):
        tag = d.replace("/", "%2F")
        b.append(f"[![Paper](https://img.shields.io/badge/paper-{tag}-1682D4.svg)]"
                 f"(https://doi.org/{d})")  # fmt: skip
    if d := links.get("software_doi"):
        b.append(f"[![Software DOI](https://zenodo.org/badge/DOI/{d}.svg)]"
                 f"(https://doi.org/{d})")  # fmt: skip
    return "\n".join(b)


def _paper_link(res: Path) -> str:
    d = _links(res).get("paper_doi")
    return f" · [Paper](https://doi.org/{d})" if d else ""


def _citation(res: Path) -> str:
    links = _links(res)
    if not links:
        return "_no citation metadata_"
    authors = " and ".join(f"{a['family']}, {a['given']}" for a in links["authors"])
    key = f"{links['authors'][0]['family'].lower()}{links['year']}countrycode"
    out = ["```bibtex"]
    if d := links.get("paper_doi"):
        out += [f"@misc{{{key},", f"  title = {{{links['paper_title']}}},",
                f"  author = {{{authors}}},", f"  year = {{{links['year']}}},",
                "  publisher = {Zenodo},", f"  doi = {{{d}}}", "}", ""]  # fmt: skip
    out += [
        f"@software{{{key}_software,",
        f"  title = {{{links['title']}}},",
        f"  author = {{{authors}}},",
        f"  year = {{{links['year']}}},",
        f"  url = {{{links['repo']}}}" + ("," if links.get("software_doi") else ""),
    ]
    if d := links.get("software_doi"):
        out.append(f"  doi = {{{d}}}")
    out += ["}", "```"]
    return "\n".join(out)


def _web_parity(res: Path) -> str:
    rel = _release(res)
    w = _load(res / f"web-parity-{rel['version']}.json") if rel["version"] else {}
    if not w:
        return "_browser parity not measured_"
    alpha, _ = _operating_point(res)
    a, top = w["decision_agreement"][alpha], w["top1_agreement"]
    target = round(float(alpha) * 100)
    return (f"onnxruntime-web computes the 8-bit matrix products in fp32 where the CPU server "
            f"uses int8, so probabilities differ slightly: on {w['n']:,} held-out strings the "
            f"browser makes the same decision as the server for {100 * a:.1f} % at the "
            f"{target} % target (same top code {100 * top:.1f} %; top probability off by "
            f"{w['mean_abs_dp_top']:.3f} on average, at most {w['max_abs_dp']:.2f}). "
            "The certificate applies to the server and Python path")  # fmt: skip


def citation_cff(res: Path) -> str:
    """CITATION.cff from configs/docs/links.yaml (Zenodo's GitHub integration reads it)."""
    import yaml

    links = _links(res)
    authors = [{"family-names": a["family"], "given-names": a["given"]} for a in links["authors"]]
    cff: dict = {
        "cff-version": "1.2.0",
        "message": "If you use CountryCode-Decide, please cite it as below.",
        "title": links["title"],
        "authors": authors,
        "version": links["version"],
        "date-released": links["date_released"],
        "license": "Apache-2.0",
        "repository-code": links["repo"],
        "url": f"https://huggingface.co/{links['hf_model']}",
        "abstract": links["abstract"],
        "keywords": links["keywords"],
    }
    if d := links.get("software_doi"):
        cff["doi"] = d
        cff["identifiers"] = [{"type": "doi", "value": d, "description": "Zenodo software DOI"}]
    if d := links.get("paper_doi"):
        cff["preferred-citation"] = {
            "type": "article", "title": links["paper_title"], "authors": authors,
            "year": links["year"], "doi": d, "publisher": {"name": "Zenodo"},
            "url": f"https://doi.org/{d}",
        }  # fmt: skip
    return yaml.safe_dump(cff, sort_keys=False, allow_unicode=True, width=100)


GENERATED = {"CITATION.cff": citation_cff}


def _hero(res: Path) -> str:
    lat = _latency_record(res)
    if not lat:
        return "_latency not measured yet_"
    return f"{lat['int8_bytes'] / 1e6:.0f} MB, about {lat['p50_ms']:.0f} ms per string on a CPU"


BLOCKS = {
    "HEADLINE": _headline,
    "LATENCY": _latency,
    "TABLE_TEST": lambda r: _table(r, _release(r)["part"]),
    "TABLE_DEV": lambda r: _table(r, "dev"),
    "TABLE_V11": lambda r: _table(r, "test_v11"),
    "TABLE_OOD": lambda r: _table(r, "ood"),
    "HYPOTHESES": _hypotheses,
    "SYNTHETIC": _synthetic,
    "LEXICON": _lexicon,
    "GOLD_QUALITY": _gold_quality,
    "CERTIFIED": _certified,
    "AT_A_GLANCE": _at_a_glance,
    "EXAMPLES": _examples,
    "BADGES": _badges,
    "PAPER_LINK": _paper_link,
    "CITATION": _citation,
    "HERO": _hero,
    "WEB_PARITY": _web_parity,
    "REPO": lambda r: _links(r).get("repo", "_no repository link_"),
}


def render(template: str, results: Path) -> str:
    def sub(m: re.Match) -> str:
        return BLOCKS[m.group(1)](results)  # KeyError on unknown placeholders

    return PLACEHOLDER.sub(sub, template)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    stale = []
    for out, tmpl in TEMPLATES.items():
        t = ROOT / tmpl
        if not t.exists():
            continue
        text = render(t.read_text(encoding="utf-8"), ROOT / "results")
        target = ROOT / out
        if args.check:
            if not target.exists() or target.read_text(encoding="utf-8") != text:
                stale.append(out)
        else:
            target.write_text(text, encoding="utf-8")
    for out, make in GENERATED.items():
        text = make(ROOT / "results")
        target = ROOT / out
        if args.check:
            if not target.exists() or target.read_text(encoding="utf-8") != text:
                stale.append(out)
        else:
            target.write_text(text, encoding="utf-8")
    if stale:
        print(f"make_docs: stale {stale}; run scripts/make_docs.py", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
