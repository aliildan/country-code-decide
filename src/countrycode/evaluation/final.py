"""Final evaluation under the pre-registration (plan-04 Task 6).

    uv run --group eval python -m countrycode.evaluation.final --bundle dist/<bundle> \
        --part test --allow-test --h4-dev-gain <gain>

Predictors: the model answering everything (calibrated probabilities), the model at each certified
operating point α ∈ {1, 2, 5} %, and the baselines. Writes results/<part>/*.json, TABLE.md and
hypotheses.json (H1–H6 against docs/preregistration.md, or with --v11 R1–R6 against
docs/preregistration-v1.1.md; --v1-bundle adds the v1 model for R4).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import numpy as np

from countrycode import labels
from countrycode.calib.calibration import ALPHAS, Calibration, decide
from countrycode.evaluation.harness import Prediction, evaluate
from countrycode.real.gold import GoldItem

TARGETS = list(labels.targets())
NON_LLM_BASELINES = (
    "lexicon_exact", "lexicon_fuzzy", "ons_style", "countrynames", "pycountry", "coco", "e5_knn",
)  # fmt: skip


class ProbSource(Protocol):
    def probs(self, items: Sequence[GoldItem]) -> np.ndarray: ...


class BundleProbs:
    """Probabilities of the shipped ONNX bundle (claims are about the downloadable file)."""

    def __init__(self, bundle: Path, batch: int = 256) -> None:
        from countrycode.infer.pipeline import CountryCoder

        self.coder = CountryCoder.load(bundle)
        self.batch = batch
        self.policy = self.coder.policy

    def probs(self, items: Sequence[GoldItem]) -> np.ndarray:
        out = []
        for i in range(0, len(items), self.batch):
            c = items[i : i + self.batch]
            out.append(
                self.coder.probs(
                    [g.text for g in c],
                    [g.field_kind for g in c],
                    [g.register_country for g in c],
                    [g.source for g in c],
                )
            )
        return np.concatenate(out) if out else np.zeros((0, len(TARGETS)))


class LegacyRegister:
    """The v1 bundle as it was trained and served: a US state register ("US-CO") is "US"."""

    def __init__(self, source: ProbSource) -> None:
        self.source = source
        self.policy = getattr(source, "policy", None)

    def probs(self, items: Sequence[GoldItem]) -> np.ndarray:
        def v1(g: GoldItem) -> GoldItem:
            us_state = g.register_country.startswith("US-")
            return replace(g, register_country="US") if us_state else g

        return self.source.probs([v1(g) for g in items])


def output_paths(results_dir: Path, part: str) -> tuple[Path, Path, bool]:
    """(report dir, private per-item file, whether this run documents the release). Only the
    default results/ run writes results/release.json; a post-hoc run (v1.2.1) writes its own
    directory and never touches the pre-registered read."""
    default = results_dir == Path("results")
    prefix = "" if default else f"{results_dir.name}_"
    per_item = Path("data/processed/eval") / f"{prefix}{part}_predictions.jsonl"
    return results_dir / part, per_item, default


class CalibratedPredictor:
    def __init__(
        self, source: ProbSource, cal: Calibration, alpha: float | None, name: str
    ) -> None:
        self.source, self.cal, self.alpha, self.name = source, cal, alpha, name

    def predict(self, items: Sequence[GoldItem]) -> list[Prediction]:
        rows = self.source.probs(items)
        policy = getattr(self.source, "policy", None)
        out = []
        for row, g in zip(rows, items, strict=True):
            cal = policy(g.source) if policy else self.cal
            if self.alpha is None:
                j = int(np.argmax(row))
                order = np.argsort(-row)[:3]
                out.append(
                    Prediction(TARGETS[j], tuple(TARGETS[int(i)] for i in order), float(row[j]))
                )
            else:
                out.append(decide(row, cal, self.alpha, TARGETS))
        return out


def judge_hypotheses(
    reports: dict[str, dict], cpu_p50_ms: float | None, h4_dev_gain: float
) -> dict:
    model = reports["model"]["overall"]
    op = reports.get("model@0.01", {}).get("overall", {})
    base = {k: v["overall"]["accuracy"] for k, v in reports.items() if k in NON_LLM_BASELINES}
    best_name = max(base, key=lambda k: base[k]) if base else None
    llm = {k: v for k, v in reports.items() if k.startswith("llm_")}
    h: dict[str, dict] = {}
    h["H1"] = {
        "model_accuracy": model["accuracy"], "best_non_llm": best_name,
        "best_non_llm_accuracy": base.get(best_name) if best_name else None,
        "pass": best_name is not None and model["accuracy"] >= base[best_name] + 0.05,
    }  # fmt: skip
    risk, ans = op.get("risk_answered"), op.get("answered")
    h["H2"] = {
        "realised_risk": risk, "answered": ans,
        "pass": risk is not None and ans is not None and risk <= 0.02 and ans >= 0.5,
    }  # fmt: skip
    fa = op.get("none_false_accept")
    vacuous = not ans  # nothing answered at α = 1 %: a 0 % false-accept rate proves nothing
    h["H3"] = {"none_false_accept": fa, "vacuous": vacuous,
               "pass": fa is not None and fa <= 0.02 and not vacuous}  # fmt: skip
    h["H4"] = {"dev_gain": h4_dev_gain, "pass": h4_dev_gain >= 0.02}
    h["H5"] = {"cpu_p50_ms": cpu_p50_ms, "pass": cpu_p50_ms is not None and cpu_p50_ms <= 5.0}
    if llm:
        name, rep = max(llm.items(), key=lambda kv: kv[1]["overall"]["accuracy"])
        speed = rep.get("latency_ms_per_item", 0) / max(cpu_p50_ms or 1e9, 1e-9)
        h["H6"] = {
            "llm": name, "llm_accuracy": rep["overall"]["accuracy"], "latency_ratio": speed,
            "pass": model["accuracy"] >= rep["overall"]["accuracy"] - 0.02 and speed >= 20,
        }  # fmt: skip
    else:
        h["H6"] = {"pass": None, "note": "no LLM baseline evaluated"}
    return h


R3_MIN_NONE_ITEMS = 20  # fewer NONE-gold register items → R3 not testable


def judge_v11(reports: dict[str, dict], cpu_p50_ms: float | None) -> dict:
    """R1–R6 against docs/preregistration-v1.1.md (register test items only)."""
    model = reports["model"]["overall"]
    op = reports.get("model@0.01", {}).get("overall", {})
    base = {k: v["overall"]["accuracy"] for k, v in reports.items() if k in NON_LLM_BASELINES}
    best_name = max(base, key=lambda k: base[k]) if base else None
    h: dict[str, dict] = {}
    h["R1"] = {
        "model_accuracy": model["accuracy"], "best_non_llm": best_name,
        "best_non_llm_accuracy": base.get(best_name) if best_name else None,
        "pass": best_name is not None and model["accuracy"] >= base[best_name] + 0.05,
    }  # fmt: skip
    risk, ans = op.get("risk_answered"), op.get("answered")
    h["R2"] = {
        "realised_risk": risk, "answered": ans,
        "pass": risk is not None and ans is not None and risk <= 0.02 and ans >= 0.6,
    }  # fmt: skip
    fa, n_none = op.get("none_false_accept"), op.get("none_items", 0)
    r3 = {"none_items": n_none, "none_false_accept": fa, "answered": ans}
    if n_none < R3_MIN_NONE_ITEMS:
        h["R3"] = r3 | {"pass": None, "note": f"< {R3_MIN_NONE_ITEMS} NONE-gold items"}
    else:
        h["R3"] = r3 | {"pass": fa is not None and fa <= 0.02 and (ans or 0) >= 0.3}
    v1 = reports.get("v1_model", {}).get("overall", {}).get("accuracy")
    h["R4"] = {"v1_accuracy": v1, "model_accuracy": model["accuracy"],
               "pass": None if v1 is None else model["accuracy"] >= v1 + 0.03}  # fmt: skip
    h["R5"] = {"cpu_p50_ms": cpu_p50_ms, "pass": cpu_p50_ms is not None and cpu_p50_ms <= 5.0}
    llm = {k: v for k, v in reports.items() if k.startswith("llm_")}
    if llm:
        name, rep = max(llm.items(), key=lambda kv: kv[1]["overall"]["accuracy"])
        speed = rep.get("latency_ms_per_item", 0) / max(cpu_p50_ms or 1e9, 1e-9)
        h["R6"] = {
            "llm": name, "llm_accuracy": rep["overall"]["accuracy"], "latency_ratio": speed,
            "pass": model["accuracy"] >= rep["overall"]["accuracy"] - 0.02 and speed >= 20,
        }  # fmt: skip
    else:
        h["R6"] = {"pass": None, "note": "no LLM baseline evaluated"}
    return h


def _answered_errors(op: dict) -> tuple[int, int]:
    n_ans = round((op.get("answered") or 0.0) * op.get("n", 0))
    risk = op.get("risk_answered")
    k = round(risk * n_ans) if n_ans and risk == risk and risk is not None else 0
    return n_ans, k


def judge_v12(reports: dict[str, dict], cpu_p50_ms: float | None, trim: dict, cert: dict) -> dict:
    """P1–P4 against docs/preregistration-v1.2.md (test_v12: unseen registers New York, Oregon);
    `cert` = register thresholds of the shipped calibration; A1 (α = 1 %) reported, not judged."""
    from scipy.stats import binom

    model = reports["model"]["overall"]
    h: dict[str, dict] = {}
    op = reports.get("model@0.02", {}).get("overall", {})
    n_ans, k = _answered_errors(op)
    certified = cert.get("0.02") is not None
    p_value = float(binom.sf(k - 1, n_ans, 0.02)) if n_ans else None
    contradicted = p_value is not None and p_value <= 0.05
    h["P1"] = {"certified": certified, "lambda": cert.get("0.02"), "answered_n": n_ans,
               "errors": k, "p_value_risk_le_2pct": p_value, "contradicted": contradicted,
               "pass": certified and n_ans > 0 and not contradicted}  # fmt: skip
    h["P2"] = {"answered": op.get("answered"), "pass": (op.get("answered") or 0) >= 0.60}
    acc_before, acc_after = trim.get("select_accuracy_untrimmed"), trim.get("select_accuracy")
    drop = acc_before - acc_after if acc_before is not None and acc_after is not None else None
    h["P3"] = {"size_mb": trim.get("size_mb"), "cpu_p50_ms": cpu_p50_ms,
               "tokens_identical": trim.get("tokens_identical"), "select_accuracy_drop": drop,
               "pass": (trim.get("size_mb") or 1e9) <= 80 and cpu_p50_ms is not None
               and cpu_p50_ms <= 5.0 and (trim.get("tokens_identical") or 0) == 1.0
               and drop is not None and drop <= 0.003}  # fmt: skip
    base = {k_: v["overall"]["accuracy"] for k_, v in reports.items() if k_ in NON_LLM_BASELINES}
    best = max(base, key=lambda k_: base[k_]) if base else None
    h["P4"] = {"model_accuracy": model["accuracy"], "best_non_llm": best,
               "best_non_llm_accuracy": base.get(best) if best else None,
               "pass": best is not None and model["accuracy"] >= base[best] + 0.05}  # fmt: skip
    op1 = reports.get("model@0.01", {}).get("overall", {})
    h["A1"] = {"certified": cert.get("0.01") is not None, "answered": op1.get("answered"),
               "risk_answered": op1.get("risk_answered"), "pass": None}  # fmt: skip
    return h


def rejudge(part: str, h4_dev_gain: float, v11: bool = False) -> dict:
    """Recompute hypotheses.json from the stored aggregate reports (no new predictions)."""
    out = Path("results") / part
    reports = {p.stem: json.loads(p.read_text()) for p in out.glob("*.json")
               if p.stem not in ("hypotheses", "risk_coverage")}  # fmt: skip
    lat = json.loads(Path("results/latency.json").read_text())
    old = json.loads((out / "hypotheses.json").read_text())
    hyp = (
        judge_v11(reports, lat["p50_ms"])
        if v11
        else judge_hypotheses(reports, lat["p50_ms"], h4_dev_gain)
    )
    hyp |= {k: old[k] for k in ("bundle", "run", "pending_g1", "calibration") if k in old}
    (out / "hypotheses.json").write_text(json.dumps(hyp, indent=1, default=float) + "\n")
    return hyp


def main() -> None:
    from countrycode.evaluation.__main__ import gold_and_lexicon, make_baselines
    from countrycode.evaluation.llm import LLMPredictor

    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--rejudge", action="store_true", help="recompute hypotheses from results/")
    ap.add_argument(
        "--part",
        default="dev",
        choices=["dev", "calib", "test", "select", "test_v11", "ood", "test_seen_v1", "test_v12"],
    )
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--llm", default="qwen3.5:35b-a3b-q4_K_M")
    ap.add_argument("--cpu-p50-ms", type=float, default=None)
    ap.add_argument("--h4-dev-gain", type=float, default=0.0)
    ap.add_argument("--v11", action="store_true", help="v1.1 roles; test part = test_v11")
    ap.add_argument("--v1-bundle", type=Path, help="v1 bundle evaluated as `v1_model` (R4)")
    ap.add_argument("--v12", action="store_true", help="v1.2 roles; judged part = test_v12")
    ap.add_argument("--trim", type=Path, help="v1.2 trim record (results/trim-<version>.json)")
    ap.add_argument("--results-dir", type=Path, default=Path("results"),
                    help="post-hoc runs write here (e.g. results/posthoc-v1.2.1)")  # fmt: skip
    args = ap.parse_args()
    if args.rejudge:
        hyp = rejudge(args.part, args.h4_dev_gain, args.v11)
        print(json.dumps({k: v.get("pass") for k, v in hyp.items() if isinstance(v, dict)}))
        return
    gold, lx = gold_and_lexicon()
    prereg_path = Path("docs/preregistration.md")
    if args.v11 or args.v12:
        from countrycode.real.roles import with_roles

        gold = with_roles(gold)
        prereg_path = Path(f"docs/preregistration-v1.{2 if args.v12 else 1}.md")
    seen = set(lx.index_keys())
    src = BundleProbs(args.bundle)
    cal = src.coder.cal
    meta = src.coder.meta
    preds = [CalibratedPredictor(src, cal, None, "model")]
    preds += [CalibratedPredictor(src, cal, a, f"model@{a}") for a in ALPHAS]
    if args.v1_bundle:
        v1 = BundleProbs(args.v1_bundle)
        legacy = LegacyRegister(v1)
        preds.append(CalibratedPredictor(legacy, v1.coder.cal, None, "v1_model"))
    preds += make_baselines(lx, None)
    if args.llm:
        preds.append(LLMPredictor(args.llm))
    out, per_item, documents_release = output_paths(args.results_dir, args.part)
    items = [g for g in gold if g.split == args.part]
    from countrycode.evaluation.slices import noise_kind

    kinds = {g.item_id: noise_kind(g.text, g.targets, lx) for g in items}
    reports = {}
    per_item.parent.mkdir(parents=True, exist_ok=True)
    with per_item.open("w", encoding="utf-8") as f:
        for p in preds:
            reports[p.name] = evaluate(
                p, gold, args.part, allow_test=args.allow_test, seen_keys=seen,
                prereg_path=prereg_path, kinds=kinds, out_dir=args.results_dir,
            )  # fmt: skip
            # private per-item record for plots and examples (data/ is never published)
            for g, pr in zip(items, p.predict(items), strict=True):
                f.write(json.dumps({"predictor": p.name, "item_id": g.item_id, "target": pr.target,
                                    "candidates": list(pr.candidates),
                                    "confidence": pr.confidence}) + "\n")  # fmt: skip
            print(p.name, round(reports[p.name]["overall"]["accuracy"], 4), flush=True)
    if args.v11 and args.part != "test_v11":
        return  # R1–R6 are judged on test_v11 only; other parts are reported, not judged
    if args.v12 and args.part != "test_v12":
        return  # P1–P4 are judged on test_v12 only
    if args.v12:
        trim = json.loads(args.trim.read_text()) if args.trim else {}
        reg = src.coder.cals.policies.get("register")
        hyp = judge_v12(reports, args.cpu_p50_ms, trim, reg.thresholds if reg else {})
        for name, path in (("gold_quality", "results/real/gold_quality.json"),
                           ("seen", "results/real/test_v12_seen.json")):  # fmt: skip
            if Path(path).exists():
                hyp[name] = json.loads(Path(path).read_text())
        version = Path(args.bundle).name.removeprefix("countrycode-")  # names latency-<v>.json
        release = {"version": version, "part": "test_v12", "hypotheses": "v12",
                   "train_run": meta.get("train_run")}  # fmt: skip
        if documents_release:
            Path("results/release.json").write_text(json.dumps(release, indent=1) + "\n")
    elif args.v11:
        hyp = judge_v11(reports, args.cpu_p50_ms)
    else:
        hyp = judge_hypotheses(reports, args.cpu_p50_ms, args.h4_dev_gain)
    hyp["bundle"] = str(args.bundle)
    hyp["run"] = meta.get("run")
    hyp["pending_g1"] = meta.get("pending_g1")
    hyp["calibration"] = {
        d: {
            "temps": c.temps,
            "thresholds": c.thresholds,
            "q": c.conformal_q,
            "candidate_rule": c.candidate_rule,
            "procedure": c.notes.get("procedure", "bonferroni"),
        }
        for d, c in src.coder.cals.policies.items()
    }
    (out / "hypotheses.json").write_text(json.dumps(hyp, indent=1, default=float) + "\n")
    print(json.dumps({k: v.get("pass") for k, v in hyp.items() if isinstance(v, dict)}))


if __name__ == "__main__":
    main()
