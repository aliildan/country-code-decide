import importlib.util
import json
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("make_tables", Path("scripts/make_tables.py"))
assert _spec and _spec.loader
make_tables = importlib.util.module_from_spec(_spec)
sys.modules["make_tables"] = make_tables
_spec.loader.exec_module(make_tables)


def _w(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))


def _results(tmp_path: Path) -> Path:
    r = tmp_path / "results"
    t = r / "test_v12"
    _w(r / "release.json", {"version": "small-v9", "part": "test_v12", "hypotheses": "v12",
                            "train_run": "run-x"})  # fmt: skip
    slices = {"lexicon=novel": {"n": 33, "accuracy": 0.212}, "gold=country": {"n": 300,
              "accuracy": 0.9}}  # fmt: skip
    _w(t / "model.json", {"predictor": "model", "latency_ms_per_item": 4.2, "slices": slices,
       "overall": {"n": 361, "accuracy": 0.8864, "country_accuracy": 0.912, "aurc": 0.0095,
                   "ece": 0.058, "none_items": 4, "accuracy_ci95": [0.853, 0.917],
                   "coverage_at_risk": {"0.01": 0.85, "0.05": 0.91}}})  # fmt: skip
    for name, ans, risk in (("model@0.02", 0.856, 0.0097), ("model@0.05", 0.903, 0.02)):
        _w(t / f"{name}.json", {"predictor": name, "latency_ms_per_item": 4.1, "overall": {
            "n": 361, "accuracy": 0.85, "answered": ans, "risk_answered": risk,
            "none_false_accept": 0.0}})  # fmt: skip
    for name, acc, ms in (("llm_qwen3.5_35b-a3b-q4_K_M", 0.85, 116.9), ("e5_knn", 0.825, 0.6),
                          ("lexicon_fuzzy", 0.795, 0.9), ("countrynames", 0.634, 2.5),
                          ("v1_model", 0.95, 3.7)):  # fmt: skip
        _w(t / f"{name}.json", {"predictor": name, "latency_ms_per_item": ms, "slices": {
            "gold=country": {"n": 300, "accuracy": acc}}, "overall": {
            "n": 361, "accuracy": acc, "country_accuracy": acc,
            "coverage_at_risk": {"0.05": 0.709}}})  # fmt: skip
    _w(t / "hypotheses.json", {
        "P1": {"lambda": 0.827, "answered_n": 309, "errors": 3, "p_value_risk_le_2pct": 0.947,
               "pass": True},
        "P2": {"answered": 0.856, "pass": True},
        "P3": {"size_mb": 74.7, "cpu_p50_ms": 4.09, "tokens_identical": 1.0,
               "select_accuracy_drop": 0.0028, "pass": True},
        "P4": {"model_accuracy": 0.886, "best_non_llm": "e5_knn",
               "best_non_llm_accuracy": 0.825, "pass": True},
        "A1": {"certified": False, "pass": None},
        "seen": {"by_source": {"us-ny": {"items": 246, "seen": 228},
                               "us-or": {"items": 115, "seen": 99}}}})  # fmt: skip
    _w(r / "latency-small-v9.json", {"p50_ms": 4.09, "p95_ms": 5.07, "batch_ms_per_item": 5.9,
                                     "int8_bytes": 74_731_597})  # fmt: skip
    _w(r / "trim-small-v9.json", {"size_mb_a": 146.7, "vocab_old": 256000, "vocab_new": 68525,
                                  "tokens_checked": 511980})  # fmt: skip
    _w(r / "real" / "gold_quality.json", {"n": 592, "agreement": 0.9916, "disagree": 5,
                                          "gold_error_upper_95": 0.0177})  # fmt: skip
    _w(
        r / "train" / "run-x.json",
        {
            "gens": ["keyboard@2"],
            "by_source": {
                "lexicon": 90,
                "lexicon:code": 10,
                "synthetic:keyboard@2": 50,
                "real:human": 10,
            },
        },
    )
    _w(r / "calibration-small-v9.json", {"domains": {
        "register": {"temps": [1.33, 1.67], "n_calib": 592,
                     "thresholds": {"0.01": None, "0.02": 0.8274, "0.05": 0.5756}},
        "curated": {"temps": [2.29, 0.21], "n_calib": 2915,
                    "thresholds": {"0.01": None, "0.02": None, "0.05": 0.7327}}}})  # fmt: skip
    _w(r / "synthetic" / "synth-v2.json", {"generators": {
        "keyboard@1": {"items": 80170, "G1_audit": {"audited": 300, "errors": 11}},
        "keyboard@2": {"items": 9806, "G1_audit": {"audited": 300, "errors": 0, "g1_pass": True}},
        "junk@1": {"items": 2260, "G1_audit": {"audited": 0, "stale": 300}},
        "ocr@2": {"items": 715, "G1_audit": {"audited": 0}}}})  # fmt: skip
    _w(r / "lexicon" / "stats.json", {"rows": 372683, "languages": 1274, "keys": 86406,
                                      "ambiguous_keys": 1947, "cldr": "47"})  # fmt: skip
    _w(r / "real" / "ingest.json", {"items": 14011, "rows_covered": 47671185})
    return r


def test_lexicon_and_ingest_macros(tmp_path):
    m = make_tables.macros(_results(tmp_path))
    for expected in ("\\newcommand{\\LexRows}{372{,}683}", "\\newcommand{\\LexLangs}{1{,}274}",
                     "\\newcommand{\\LexKeys}{86{,}406}", "\\newcommand{\\LexAmbig}{1{,}947}",
                     "\\newcommand{\\CldrVersion}{47}", "\\newcommand{\\IngestItems}{14{,}011}",
                     "\\newcommand{\\IngestRows}{47.7}"):  # fmt: skip
        assert expected in m, expected


def test_macros_carry_the_papers_numbers(tmp_path):
    m = make_tables.macros(_results(tmp_path))
    for expected in ("\\newcommand{\\TestN}{361}", "\\newcommand{\\AccModel}{88.6}",
                     "\\newcommand{\\AccCILo}{85.3}", "\\newcommand{\\OpTwoAnswered}{85.6}",
                     "\\newcommand{\\OpTwoRisk}{1.0}", "\\newcommand{\\POneErrors}{3}",
                     "\\newcommand{\\TestSeen}{327}", "\\newcommand{\\AccNovel}{21.2}",
                     "\\newcommand{\\AccBest}{82.5}", "\\newcommand{\\LLMSlower}{29}",
                     "\\newcommand{\\SizeMB}{74.7}", "\\newcommand{\\VocabNew}{68{,}525}",
                     "\\newcommand{\\SynShare}{31.2}",
                     "\\newcommand{\\RegLambdaTwo}{0.827}",
                     "\\newcommand{\\GoldUpper}{1.8}"):  # fmt: skip
        assert expected in m, expected
    assert "multilingual-e5-small nearest name" in m and "v1_model" not in m


def test_predictor_table_orders_rows_and_marks_cached_llm_latency(tmp_path):
    r = _results(tmp_path)
    t = make_tables.table_predictors(r, "test_v12")
    lines = [line for line in t.splitlines() if line.endswith("\\\\")][1:]
    assert lines[0].startswith("CountryCode-Decide (always answers) & 88.6 & 91.2")
    assert "Qwen3.5-35B-A3B" in t and "116.9" in t
    p = r / "test_v12" / "llm_qwen3.5_35b-a3b-q4_K_M.json"
    d = json.loads(p.read_text())
    d["latency_ms_per_item"] = 0.004
    p.write_text(json.dumps(d))
    assert "cached" in make_tables.table_predictors(r, "test_v12")


def test_release_table_marks_audit_outcome_and_use(tmp_path):
    t = make_tables.table_releases(_results(tmp_path))
    assert "keyboard@1 & 80{,}170 & 300 & 11 & fail &  \\\\" in t
    assert "keyboard@2 & 9{,}806 & 300 & 0 & pass & yes \\\\" in t
    assert "junk@1 & 2{,}260 & 0 & 0 & stale" in t and "ocr@2 & 715 & 0 & 0 & not audited" in t


def test_prereg_calibration_training_tables(tmp_path):
    r = _results(tmp_path)
    p = make_tables.table_prereg(r)
    assert "P1 &" in p and "3 errors in 309 answered" in p and "met" in p and "A1" in p
    c = make_tables.table_calibration(r)
    assert "register & 592 & 1.33 & -- & 0.827 & 0.576" in c
    assert "curated & 2{,}915 & 2.29 & -- & -- & 0.733" in c
    tr = make_tables.table_training(r)
    assert "lexicon names and codes & 100 & 62.5" in tr and "generated: keyboard@2 & 50" in tr


def test_outputs_include_curve_csvs_and_every_table(tmp_path):
    r = _results(tmp_path)
    (r / "test_v12" / "risk_coverage.json").write_text(json.dumps(
        {"model": {"coverage": [0.1, 0.5], "risk": [0.0, 0.01]}}))  # fmt: skip
    csv = make_tables.curve_csvs(r, "test_v12")
    assert csv["test_v12-model.csv"] == "coverage,risk\n10.00,0.000\n50.00,1.000\n"
    out = make_tables.outputs(r)
    assert {"tables/macros.tex", "tables/test.tex", "tables/releases.tex"} <= set(out)
    assert not any(k.endswith(".csv") for k in out)  # data files never go to results/


def test_first_audit_macros_sum_the_first_audits_releases(tmp_path):
    m = make_tables.macros(_results(tmp_path))
    # keyboard@1 (11 errors / 300) and junk@1 (300 stale verdicts, 0 errors) were in the first
    # audit; keyboard@2 is a repair and ocr@2 was never audited
    assert "\\newcommand{\\FirstAuditN}{600}" in m
    assert "\\newcommand{\\FirstAuditErrors}{11}" in m
    assert "\\newcommand{\\FirstAuditReleases}{2}" in m
    assert "\\newcommand{\\FirstAuditFailed}{1}" in m


def test_slice_size_and_release_count_macros(tmp_path):
    r = _results(tmp_path)
    m = json.loads((r / "test_v12" / "model.json").read_text())
    m["overall"]["ambiguous_items"] = 4
    (r / "test_v12" / "model.json").write_text(json.dumps(m))
    out = make_tables.macros(r)
    assert "\\newcommand{\\TestAmbigN}{4}" in out and "\\newcommand{\\TestNoneN}{4}" in out
    assert "\\newcommand{\\ModelReleases}{1}" in out  # releases used by the documented run


def _posthoc(tmp_path):
    """v1.2.1 layout: the pre-registered read in test_v12, the release run in posthoc-v1.2.1/."""
    r = _results(tmp_path)
    _w(r / "release.json", {"version": "small-v1.2.1", "part": "posthoc-v1.2.1/test_v12",
                            "preregistered": "test_v12", "hypotheses": "v12",
                            "train_run": "run-x", "model_version": "small-v9"})  # fmt: skip
    pre = r / "test_v12"
    m2 = json.loads((pre / "model@0.02.json").read_text())
    m2["slices"] = {"label=auto_exact": {"n": 261, "answered": 255 / 261, "risk_answered": 0.0},
                    "label=reviewed_proposal": {"n": 100, "answered": 0.54,
                                                "risk_answered": 3 / 54}}  # fmt: skip
    (pre / "model@0.02.json").write_text(json.dumps(m2))
    ph = r / "posthoc-v1.2.1" / "test_v12"
    ph.mkdir(parents=True)
    lab = {"label=auto_exact": {"n": 261, "accuracy": 259 / 261, "answered": 1.0},
           "label=reviewed_proposal": {"n": 100, "accuracy": 0.61, "answered": 1.0}}  # fmt: skip
    for f in pre.glob("*.json"):
        d = json.loads(f.read_text())
        d.setdefault("slices", {}).update(lab)
        (ph / f.name).write_text(json.dumps(d))
    _w(ph / "model@0.05.json", {"predictor": "model@0.05", "latency_ms_per_item": 4.1,
       "slices": {"label=reviewed_proposal": {"n": 100, "answered": 0.61, "risk_answered": 5 / 61,
                                              "accuracy": 0.56}},
       "overall": {"n": 361, "accuracy": 0.864, "answered": 0.878, "risk_answered": 0.025,
                   "none_false_accept": 0.0}})  # fmt: skip
    _w(
        ph / "lexicon_context.json",
        {
            "predictor": "lexicon_context",
            "latency_ms_per_item": 0.0,
            "slices": {
                "label=auto_exact": {"n": 261, "accuracy": 260 / 261, "answered": 1.0},
                "label=reviewed_proposal": {"n": 100, "accuracy": 0.63, "answered": 0.62},
            },
            "overall": {"n": 361, "accuracy": 0.895, "answered": 0.895, "risk_answered": 0.015},
        },
    )
    _w(r / "calibration-small-v1.2.1.json", {"domains": {
        "register": {"temps": [1.33, 1.67], "n_calib": 592, "notes": {"n_cert_units": 318},
                     "thresholds": {"0.01": None, "0.02": None, "0.05": 0.7621}},
        "curated": {"temps": [2.29, 1.0], "n_calib": 2915, "notes": {"n_cert_units": 2796},
                    "thresholds": {"0.01": None, "0.02": None, "0.05": 0.7327}}}})  # fmt: skip
    _w(r / "latency-small-v1.2.1.json", {"p50_ms": 4.09, "p95_ms": 5.07, "batch_ms_per_item": 5.9,
                                         "int8_bytes": 74_731_597})  # fmt: skip
    _w(
        r / "real" / "ingest.json",
        {
            "items": 14011,
            "rows_covered": 47671185,
            "by_source_field": {
                "us-ny/country": 156,
                "us-ny/jurisdiction": 95,
                "us-or/jurisdiction": 117,
            },
        },
    )
    return r


def test_posthoc_macros_keep_preregistered_and_release_numbers_apart(tmp_path):
    m = make_tables.macros(_posthoc(tmp_path))
    for expected in ("\\newcommand{\\POneAnsweredN}{309}",
                     "\\newcommand{\\PreTwoHandAnswered}{54}",
                     "\\newcommand{\\PreTwoHandRisk}{5.6}",
                     "\\newcommand{\\PreTwoAutoAnswered}{255}",
                     "\\newcommand{\\RegLambdaTwoPre}{0.827}",
                     "\\newcommand{\\RegLambdaFive}{0.762}",
                     "\\newcommand{\\RegUnits}{318}", "\\newcommand{\\OpFiveAnswered}{87.8}",
                     "\\newcommand{\\OpFiveRisk}{2.5}", "\\newcommand{\\HandFiveRisk}{8.2}",
                     "\\newcommand{\\AccContext}{89.5}", "\\newcommand{\\HandModel}{61.0}",
                     "\\newcommand{\\HandContext}{63.0}", "\\newcommand{\\TestUnscored}{7}",
                     "\\newcommand{\\POneUpper}{2.5}"):  # fmt: skip
        assert expected in m, expected


def test_label_type_and_calibration_tables_for_a_posthoc_release(tmp_path):
    r = _posthoc(tmp_path)
    lt = make_tables.table_label_types(r)
    assert "CountryCode-Decide (always answers) & 99.2 & 61.0 & 100" in lt
    assert "Lexicon + context rules & 99.6 & 63.0 & 62" in lt
    cal = make_tables.table_calibration(r)
    assert "register & items (v1.2) & 592 & -- & 0.827 & 0.576" in cal
    assert "register & units (v1.2.1) & 318 & -- & -- & 0.762" in cal
    assert "tables/labels.tex" in make_tables.outputs(r)


def test_operating_points_that_answer_nothing_are_left_out(tmp_path):
    r = _posthoc(tmp_path)
    _w(r / "posthoc-v1.2.1" / "test_v12" / "model@0.02.json", {
        "predictor": "model@0.02", "latency_ms_per_item": 4.1,
        "overall": {"n": 361, "accuracy": 0.011, "answered": 0.0}})  # fmt: skip
    t = make_tables.table_predictors(r, "posthoc-v1.2.1/test_v12")
    assert "alpha = 2" not in t and "alpha = 5" in t


def _cascade_results(tmp_path):
    r = _posthoc(tmp_path)
    parts = {
        "test_v12": {"n": 361, "lexicon": {"answered": 323, "errors": 5},
                     "residual": {"n": 38, "model_answered": 8, "model_errors": 5},
                     "cascade": {"answered": 331, "errors": 10}},
        "test_v11": {"n": 656, "lexicon": {"answered": 163, "errors": 3},
                     "residual": {"n": 493, "model_answered": 355, "model_errors": 22},
                     "cascade": {"answered": 518, "errors": 25}},
    }  # fmt: skip
    _w(r / "posthoc-v1.2.1" / "cascade.json", {"parts": parts})
    return r


def test_cascade_macros_and_table(tmp_path):
    r = _cascade_results(tmp_path)
    m = make_tables.macros(r)
    for expected in ("\\newcommand{\\CasFreshResN}{38}", "\\newcommand{\\CasFreshResAns}{8}",
                     "\\newcommand{\\CasFreshResErr}{5}", "\\newcommand{\\CasFreshAnsPct}{91.7}",
                     "\\newcommand{\\CasFreshErrPct}{3.0}", "\\newcommand{\\CasMessyLexPct}{24.8}",
                     "\\newcommand{\\CasMessyAnsPct}{79.0}", "\\newcommand{\\CasMessyErrPct}{4.8}",
                     "\\newcommand{\\CasMessyResRisk}{6.2}"):  # fmt: skip
        assert expected in m, expected
    t = make_tables.table_cascade(r)
    assert "Fresh registers & 361 & 89.5 & 1.5 & 91.7 & 3.0 & 8 / 38 & 5 \\\\" in t
    assert "tables/cascade.tex" in make_tables.outputs(r)
