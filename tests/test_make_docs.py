import importlib.util
import json
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("make_docs", Path("scripts/make_docs.py"))
assert _spec and _spec.loader
make_docs = importlib.util.module_from_spec(_spec)
sys.modules["make_docs"] = make_docs
_spec.loader.exec_module(make_docs)


def test_render_fills_known_blocks_and_fails_on_unknown(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "latency.json").write_text(
        json.dumps({"p50_ms": 3.1, "p95_ms": 3.7, "threads": 4, "int8_bytes": 147_000_000})
    )
    out = make_docs.render("A {{LATENCY}} B", tmp_path / "results")
    assert "3.1 ms" in out and "{{" not in out
    try:
        make_docs.render("{{NOPE}}", tmp_path / "results")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown placeholder must fail")


def test_headline_reads_results(tmp_path):
    t = tmp_path / "results" / "test"
    t.mkdir(parents=True)

    def rep(name, **o):
        (t / f"{name}.json").write_text(
            json.dumps({"predictor": name, "overall": o, "latency_ms_per_item": 100.0})
        )

    rep("model", accuracy=0.64, country_accuracy=0.75, ece=0.07)
    rep("model@0.05", answered=0.33, risk_answered=0.037, none_false_accept=0.04)
    rep("llm_x", accuracy=0.72, country_accuracy=0.70, ece=0.2, coverage_at_risk={"0.05": 0.0})
    rep("lexicon_fuzzy", accuracy=0.62, country_accuracy=0.68)
    (tmp_path / "results" / "latency.json").write_text(
        json.dumps({"p50_ms": 4.0, "p95_ms": 5, "threads": 4, "int8_bytes": 1})
    )
    out = make_docs.render("{{HEADLINE}}", tmp_path / "results")
    assert "0.750" in out and "3.7 %" in out and "33%" in out
    assert "25× slower (the LLM on a GPU, this model on a CPU)" in out  # 100 ms vs p50 4 ms


def _synth_results(tmp_path, by_source):
    res = tmp_path / "results"
    for d in ("synthetic", "test", "train"):
        (res / d).mkdir(parents=True)
    audit = {"errors": 1, "audited": 300}
    (res / "synthetic" / "synth-v2.json").write_text(json.dumps({"generators": {
        "keyboard@1": {"items": 80, "G1_audit": audit},
        "city@1": {"items": 44, "G1_audit": audit},
    }}))  # fmt: skip
    (res / "test" / "hypotheses.json").write_text(json.dumps({"run": "runs/small-x"}))
    (res / "train" / "small-x.json").write_text(
        json.dumps({"run": "small-x", "gens": ["keyboard@1"], "by_source": by_source})
    )
    return res


def test_synthetic_block_lists_the_documented_runs_training_data(tmp_path):
    res = _synth_results(tmp_path, {"lexicon": 9, "synthetic:keyboard@1": 1234})
    out = make_docs.render("{{SYNTHETIC}}", res)
    assert "| keyboard@1 | 1,234 | 1 / 300 |" in out
    assert "city@1" not in out  # released later, not used by this model
    assert "No real register string" in out


def test_synthetic_block_reports_real_register_training_strings(tmp_path):
    res = _synth_results(
        tmp_path, {"synthetic:keyboard@1": 5, "real:human": 700, "real:register_code": 120}
    )
    out = make_docs.render("{{SYNTHETIC}}", res)
    assert "820 real register strings" in out
    assert "No real register string" not in out
    assert out.count("820") == 1  # the count is stated once
    assert "human audit (errors / audited)" in out and "owner" not in out


def test_release_selects_part_hypotheses_and_latency(tmp_path):
    res = tmp_path / "results"
    (res / "test_v12").mkdir(parents=True)
    (res / "release.json").write_text(
        json.dumps({"version": "small-v1.2", "part": "test_v12", "hypotheses": "v12"})
    )
    (res / "test_v12" / "hypotheses.json").write_text(json.dumps({
        "P1": {"pass": True, "lambda": 0.886}, "P2": {"pass": False, "answered": 0.55},
        "P3": {"pass": True}, "P4": {"pass": True}, "A1": {"pass": None},
    }))  # fmt: skip
    (res / "test_v12" / "TABLE.md").write_text("| predictor | n |\n|---|---|\n| model | 368 |\n")
    (res / "latency-small-v1.2.json").write_text(
        json.dumps({"p50_ms": 4.1, "p95_ms": 5.1, "threads": 4, "int8_bytes": 71_200_000})
    )
    out = make_docs.render("{{HYPOTHESES}}\n{{TABLE_TEST}}\n{{LATENCY}}", res)
    assert "| P1 | met |" in out and "| P2 | **not met** |" in out
    assert "| A1 | reported (not judged) |" in out
    assert "H1" not in out and "| model | 368 |" in out and "71 MB" in out


def _v12(tmp_path, p1_pass: bool):
    r = tmp_path / "results"
    t = r / "test_v12"
    t.mkdir(parents=True)
    (r / "release.json").write_text(json.dumps({"version": "small-v1.2", "part": "test_v12",
                                                "hypotheses": "v12"}))  # fmt: skip
    for name, o in {"model": {"accuracy": 0.9, "country_accuracy": 0.95},
                    "model@0.02": {"answered": 0.81, "risk_answered": 0.012,
                                   "none_false_accept": 0.01},
                    "model@0.05": {"answered": 0.9, "risk_answered": 0.03,
                                   "none_false_accept": 0.02}}.items():  # fmt: skip
        (t / f"{name}.json").write_text(json.dumps({"predictor": name, "overall": o}))
    (t / "hypotheses.json").write_text(
        json.dumps(
            {"P1": {"pass": p1_pass, "p_value_risk_le_2pct": 0.4}, "A1": {"certified": False}}
        )
    )
    return r


def test_v12_headline_quotes_the_2pct_point_only_when_p1_holds(tmp_path):
    held = make_docs._headline(_v12(tmp_path / "a", True))
    assert "certified 2 % error target" in held and "81%" in held
    assert "1 % target could not be certified" in held
    failed = make_docs._headline(_v12(tmp_path / "b", False))
    assert "certified 5 % error target" in failed and "not confirmed" in failed
    assert "could not be certified on this calibration set" not in held + failed  # v1 wording


def test_gold_quality_block(tmp_path):
    r = tmp_path / "results"
    (r / "real").mkdir(parents=True)
    (r / "real" / "gold_quality.json").write_text(json.dumps(
        {"n": 592, "agree": 585, "disagree": 7, "agreement": 585 / 592,
         "gold_error_upper_95": 0.021}))  # fmt: skip
    out = make_docs.render("{{GOLD_QUALITY}}", r)
    assert "592" in out and "98.8 %" in out and "2.1 %" in out


def test_v12_headline_reports_strings_new_to_the_lexicon(tmp_path):
    r = _v12(tmp_path, True)
    m = json.loads((r / "test_v12" / "model.json").read_text())
    m["slices"] = {
        "lexicon=novel": {"n": 34, "accuracy": 0.62},
        "lexicon=seen": {"n": 327, "accuracy": 0.97},
    }
    (r / "test_v12" / "model.json").write_text(json.dumps(m))
    out = make_docs._headline(r)
    assert "34 strings not in the lexicon" in out and "0.620" in out


def test_latency_of_a_versioned_release_never_falls_back_to_another_bundle(tmp_path):
    r = tmp_path / "results"
    r.mkdir()
    (r / "release.json").write_text(json.dumps({"version": "small-v1.2", "part": "test_v12",
                                                "hypotheses": "v12"}))  # fmt: skip
    (r / "latency.json").write_text(json.dumps({"p50_ms": 3.0, "p95_ms": 3.5, "threads": 4,
                                                "int8_bytes": 147_000_000}))  # fmt: skip
    assert "not measured yet" in make_docs.render("{{LATENCY}}", r)


def test_synthetic_block_follows_the_releases_training_run(tmp_path):
    r = tmp_path / "results"
    (r / "test_v12").mkdir(parents=True)
    (r / "train").mkdir()
    (r / "synthetic").mkdir()
    rel = {"version": "small-v1.2", "part": "test_v12", "hypotheses": "v12",
           "train_run": "small-v12f"}  # fmt: skip
    (r / "release.json").write_text(json.dumps(rel))
    (r / "test_v12" / "hypotheses.json").write_text(json.dumps({"run": "runs/r-trim-C"}))
    (r / "train" / "small-v12f.json").write_text(
        json.dumps({"gens": ["junk@2"], "by_source": {"synthetic:junk@2": 1375, "real:human": 500}})
    )
    (r / "synthetic" / "synth-v2.json").write_text(
        json.dumps(
            {
                "generators": {
                    "junk@2": {"items": 1375, "G1_audit": {"errors": 0, "audited": 300}},
                    "ocr@1": {"items": 7538, "G1_audit": {"errors": 31, "audited": 300}},
                }
            }
        )
    )
    out = make_docs.render("{{SYNTHETIC}}", r)
    assert "| junk@2 | 1,375 | 0 / 300 |" in out and "ocr@1" not in out


def test_headline_never_claims_best_or_llm_superiority_against_the_numbers(tmp_path):
    t = tmp_path / "results" / "test"
    t.mkdir(parents=True)

    def rep(name, **o):
        (t / f"{name}.json").write_text(
            json.dumps({"predictor": name, "overall": o, "latency_ms_per_item": 100.0})
        )

    rep("model", accuracy=0.886, country_accuracy=0.912)
    rep("model@0.05", answered=0.9, risk_answered=0.043, none_false_accept=0.0, none_items=4)
    rep("v1_model", accuracy=0.884, country_accuracy=0.921)
    rep("llm_x", accuracy=0.850, country_accuracy=0.858, coverage_at_risk={"0.05": 0.709})
    out = make_docs.render("{{HEADLINE}}", tmp_path / "results")
    assert "best of all predictors" not in out and "highest: v1_model 0.921" in out
    assert "more accurate when every string" not in out and "0.850 vs the model's 0.886" in out
    assert "(n = 4)" in out


def test_certified_levels_are_rendered_from_the_release_calibration(tmp_path):
    r = _v12(tmp_path, True)
    h = json.loads((r / "test_v12" / "hypotheses.json").read_text())
    reg = {"0.01": None, "0.02": 0.83, "0.05": 0.58}
    cur = {"0.01": None, "0.02": None, "0.05": 0.73}
    h["calibration"] = {"register": {"thresholds": reg}, "curated": {"thresholds": cur}}
    (r / "test_v12" / "hypotheses.json").write_text(json.dumps(h))
    out = make_docs.render("{{CERTIFIED}}", r)
    assert "register: 2 %, 5 %" in out and "curated: 5 %" in out and "1 %" not in out


def test_headline_adds_the_gold_caveat_when_the_bound_exceeds_2pct(tmp_path):
    r = _v12(tmp_path, True)
    (r / "real").mkdir()
    (r / "real" / "gold_quality.json").write_text(
        json.dumps(
            {
                "n": 89,
                "agree": 88,
                "disagree": 1,
                "agreement": 88 / 89,
                "gold_error_upper_95": 0.052,
            }
        )
    )
    assert "gold-label error" in make_docs._headline(r)


def _glance(tmp_path, p1_pass=True):
    r = _v12(tmp_path, p1_pass)
    t = r / "test_v12"
    m = json.loads((t / "model.json").read_text())
    m["overall"].update(n=361, accuracy_ci95=[0.85, 0.92])
    (t / "model.json").write_text(json.dumps(m))
    for name, acc, ms in [("e5_knn", 0.825, 0.6), ("countrynames", 0.634, 2.5),
                          ("v1_model", 0.95, 3.7), ("llm_qwen3.5_35b", 0.85, 117.0)]:  # fmt: skip
        (t / f"{name}.json").write_text(json.dumps({
            "predictor": name, "overall": {"accuracy": acc, "country_accuracy": acc},
            "latency_ms_per_item": ms}))  # fmt: skip
    (r / "latency-small-v1.2.json").write_text(json.dumps(
        {"p50_ms": 4.09, "p95_ms": 5.07, "threads": 4, "int8_bytes": 74_731_597}))  # fmt: skip
    return r


def test_at_a_glance_is_rendered_from_the_release_results(tmp_path):
    out = make_docs._at_a_glance(_glance(tmp_path / "a", True))
    assert out.startswith("| ") and "{{" not in out
    assert "361 distinct strings" in out
    assert "90.0 %" in out and "85.0–92.0" in out  # accuracy with its 95 % interval
    assert "95.0 %" in out  # right country
    assert "2 % error target" in out and "81 %" in out and "1.2 %" in out
    assert "multilingual-e5-small nearest name" in out and "82.5 %" in out  # readable label
    assert "v1_model" not in out
    assert "qwen3.5_35b" in out and "29× slower (LLM on a GPU, this model on a CPU)" in out
    assert "75 MB" in out and "4.1 ms" in out
    failed = make_docs._at_a_glance(_glance(tmp_path / "b", False))
    assert "5 % error target" in failed and "90 %" in failed and "3.0 %" in failed


def test_examples_are_rendered_from_the_release_examples_file(tmp_path):
    r = _v12(tmp_path, True)
    assert make_docs._examples(r) == "_no examples yet_"
    (r / "examples-small-v1.2.json").write_text(json.dumps({
        "alpha": 0.02, "domain": "register", "examples": [
            {"text": "Østerrike", "field": "country", "register_country": "NO", "target": "AT",
             "name": "Austria", "confidence": 0.97, "candidates": ["AT"], "abstain": False},
            {"text": "Georgia", "field": "unknown", "register_country": "unknown",
             "target": None, "name": None, "confidence": 0.75, "candidates": ["GE", "US-GA"],
             "abstain": True}]}))  # fmt: skip
    out = make_docs._examples(r)
    assert "`Østerrike`" in out and "**AT** Austria" in out and "0.97" in out
    assert "country field, NO register" in out
    assert "abstains" in out and "GE, US-GA" in out


def _links(tmp_path, **over):
    import yaml

    d = tmp_path / "configs" / "docs"
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"repo": "https://github.com/aliildan/country-code-decide",
           "hf_model": "aildan/country-code-decide-small", "paper_doi": None,
           "software_doi": None, "title": "CountryCode-Decide", "paper_title": "Paper T",
           "authors": [{"given": "Ali", "family": "Ildan"}], "year": 2026, **over}  # fmt: skip
    (d / "links.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True))
    (tmp_path / "results").mkdir(exist_ok=True)
    return tmp_path / "results"


def test_badges_and_citation_show_dois_only_once_they_exist(tmp_path):
    r = _links(tmp_path / "a")
    b, c = make_docs._badges(r), make_docs._citation(r)
    assert "huggingface.co/aildan/country-code-decide-small" in b and "zenodo" not in b
    assert make_docs._paper_link(r) == ""
    assert make_docs.render("{{REPO}}", r) == "https://github.com/aliildan/country-code-decide"
    assert "@software" in c and "github.com/aliildan/country-code-decide" in c and "doi" not in c
    r = _links(tmp_path / "b", paper_doi="10.5281/zenodo.11", software_doi="10.5281/zenodo.22")
    b, c = make_docs._badges(r), make_docs._citation(r)
    assert "https://doi.org/10.5281/zenodo.11" in b and "https://doi.org/10.5281/zenodo.22" in b
    assert "[Paper](https://doi.org/10.5281/zenodo.11)" in make_docs._paper_link(r)
    assert "Paper T" in c and "doi = {10.5281/zenodo.11}" in c and "doi = {10.5281/zenodo.22}" in c


def test_hero_line_from_the_release_latency(tmp_path):
    out = make_docs._hero(_glance(tmp_path, True))
    assert "75 MB" in out and "4 ms" in out


def test_other_parts_tables_render_from_their_own_results(tmp_path):
    r = tmp_path / "results"
    for part in ("test_v11", "ood"):
        (r / part).mkdir(parents=True)
        (r / part / "TABLE.md").write_text(
            f"# {part}\n\n| predictor | acc |\n|---|---|\n| {part} | 1 |\n"
        )
    out = make_docs.render("{{TABLE_V11}}\n{{TABLE_OOD}}", r)
    assert "| test_v11 | 1 |" in out and "| ood | 1 |" in out and "#" not in out


def test_synthetic_block_states_the_generated_share_of_training(tmp_path):
    res = _synth_results(
        tmp_path,
        {"lexicon": 90, "lexicon:code": 10, "synthetic:keyboard@1": 50, "real:human": 10},
    )
    out = make_docs.render("{{SYNTHETIC}}", res)
    assert (
        "160 training examples: 100 lexicon names and codes (62.5 %), 50 generated (31.2 %), "
        "10 real register strings (6.2 %)" in out
    )


def test_citation_cff_is_generated_from_links_and_gains_dois_once_minted(tmp_path):
    import yaml

    r = _links(tmp_path / "a", abstract="A small model.", keywords=["iso-3166"], version="1.2.0",
               date_released="2026-10-10")  # fmt: skip
    cff = yaml.safe_load(make_docs.citation_cff(r))
    assert cff["cff-version"] == "1.2.0" and cff["title"] == "CountryCode-Decide"
    assert cff["authors"] == [{"family-names": "Ildan", "given-names": "Ali"}]
    assert cff["version"] == "1.2.0" and str(cff["date-released"]) == "2026-10-10"
    assert cff["license"] == "Apache-2.0" and cff["abstract"] == "A small model."
    assert cff["repository-code"] == "https://github.com/aliildan/country-code-decide"
    assert cff["url"] == "https://huggingface.co/aildan/country-code-decide-small"
    assert "doi" not in cff and "preferred-citation" not in cff
    r = _links(tmp_path / "b", abstract="A.", keywords=[], version="1.2.0",
               date_released="2026-10-10", paper_doi="10.5281/zenodo.11",
               software_doi="10.5281/zenodo.22")  # fmt: skip
    cff = yaml.safe_load(make_docs.citation_cff(r))
    assert cff["doi"] == "10.5281/zenodo.22"
    pc = cff["preferred-citation"]
    assert pc["doi"] == "10.5281/zenodo.11" and pc["title"] == "Paper T" and pc["type"] == "article"
    text = make_docs.citation_cff(r)
    assert "&id" not in text and "*id" not in text  # no YAML aliases: plain CFF for every reader


def _v121(tmp_path):
    """A post-hoc release (v1.2.1): the pre-registered read in test_v12, the release's own
    evaluation in posthoc-v1.2.1/<part>."""
    r = _glance(tmp_path, True)  # results/test_v12 = the pre-registered read (P1 met)
    (r / "test_v12" / "hypotheses.json").write_text(
        json.dumps({"P1": {"pass": True, "p_value_risk_le_2pct": 0.9}, "A1": {"certified": False}})
    )
    (r / "release.json").write_text(json.dumps({
        "version": "small-v1.2.1", "part": "posthoc-v1.2.1/test_v12", "preregistered": "test_v12",
        "hypotheses": "v12", "model_version": "small-v1.2"}))  # fmt: skip
    ph = r / "posthoc-v1.2.1"

    def rep(part, name, acc, slices=None, **o):
        n = {"test_v12": 361, "test_v11": 656, "ood": 3980}[part]
        (ph / part).mkdir(parents=True, exist_ok=True)
        (ph / part / f"{name}.json").write_text(json.dumps({
            "predictor": name, "latency_ms_per_item": 117.0 if name.startswith("llm") else 4.2,
            "slices": slices or {}, "overall": {"n": n, "accuracy": acc,
                                                "country_accuracy": acc, **o}}))  # fmt: skip

    def lab(auto, hand, answered):
        return {"label=auto_exact": {"n": 261, "accuracy": auto, "answered": 1.0},
                "label=reviewed_proposal": {"n": 100, "accuracy": hand, "answered": answered},
                "script=LATIN": {"n": 361}}  # fmt: skip

    rep("test_v12", "model", 0.886, lab(259 / 261, 0.61, 1.0), accuracy_ci95=[0.85, 0.92])
    rep(
        "test_v12",
        "model@0.05",
        0.864,
        lab(256 / 261, 0.56, 0.61),
        answered=0.878,
        risk_answered=0.025,
        none_false_accept=0.0,
    )
    rep("test_v12", "lexicon_context", 0.895, lab(260 / 261, 0.63, 0.62))
    rep("test_v12", "llm_qwen3.5_35b-a3b-q4_K_M", 0.85, lab(246 / 261, 0.61, 0.97))
    rep(
        "test_v11",
        "model",
        0.826,
        {
            "script=LATIN": {"n": 640},
            "script=NONE": {"n": 16},
            "lexicon=novel": {"n": 472, "accuracy": 0.811},
        },
    )
    rep(
        "test_v11",
        "llm_qwen3.5_35b-a3b-q4_K_M",
        0.738,
        {"lexicon=novel": {"n": 472, "accuracy": 0.731}},
    )
    rep("test_v11", "lexicon_context", 0.358)
    rep("ood", "model", 0.589, {"script=LATIN": {"n": 3975}, "script=CYRILLIC": {"n": 5}})
    rep("ood", "llm_qwen3.5_35b-a3b-q4_K_M", 0.696)
    for v, two, five, units in (("small-v1.2", 0.827, 0.576, None),
                                ("small-v1.2.1", None, 0.762, 318)):  # fmt: skip
        notes = {"n_cert_units": units} if units else {}
        (r / f"calibration-{v}.json").write_text(json.dumps({"domains": {"register": {
            "n_calib": 592, "thresholds": {"0.01": None, "0.02": two, "0.05": five},
            "notes": notes}}}))  # fmt: skip
    (r / "latency-small-v1.2.1.json").write_text(
        json.dumps({"p50_ms": 4.09, "p95_ms": 5.07, "threads": 4, "int8_bytes": 74_731_597})
    )
    (r / "real").mkdir(exist_ok=True)
    (r / "real" / "gold_quality.json").write_text(json.dumps({
        "n": 592, "agreement": 0.9916, "disagree": 5, "gold_error_upper_95": 0.0177,
        "by_mode": {"auto_exact": {"n": 337, "disagree": 0}, "preset": {"n": 37, "disagree": 0},
                    "reviewed_proposal": {"n": 218, "disagree": 5}}}))  # fmt: skip
    return r


def test_posthoc_release_shows_the_preregistered_verdicts_as_read(tmp_path):
    r = _v121(tmp_path)
    assert "| P1 | met |" in make_docs._hypotheses(r)
    alpha, note = make_docs._operating_point(r)
    assert alpha == "0.05" and "318" in note and "2 %" in note


def test_certificate_note_compares_item_and_unit_level_certificates(tmp_path):
    out = make_docs._certificate_note(_v121(tmp_path))
    for s in ("592", "318", "not certified", "0.762", "after"):
        assert s in out, s


def test_label_type_table_from_slices(tmp_path):
    out = make_docs._label_types(_v121(tmp_path))
    assert "| CountryCode-Decide (always answers) | 99.2 % | 61.0 % | 100 |" in out
    assert "| Lexicon + context rules | 99.6 % | 63.0 % | 62 |" in out
    assert "261" in out and "100" in out


def test_where_the_model_helps_from_the_other_sets(tmp_path):
    out = make_docs._where_it_helps(_v121(tmp_path))
    for s in ("656", "82.6 %", "73.8 %", "35.8 %", "81.1 %", "69.6 %", "58.9 %"):
        assert s in out, s


def test_gold_quality_reports_hand_labels_and_single_pass_test_labels(tmp_path):
    out = make_docs._gold_quality(_v121(tmp_path))
    assert "218 hand-labelled" in out and "2.3 %" in out and "once" in out


def test_script_scope_counts_non_latin_evaluation_strings(tmp_path):
    out = make_docs._script_scope(_v121(tmp_path))
    assert "4,997" in out and "5 " in out and "not measured" in out


def test_hard_risk_shows_where_the_errors_of_the_operating_point_are(tmp_path):
    r = _v121(tmp_path)
    p = r / "posthoc-v1.2.1" / "test_v12" / "model@0.05.json"
    d = json.loads(p.read_text())
    d["slices"] = {"label=auto_exact": {"n": 261, "answered": 0.98, "risk_answered": 0.0},
                   "label=reviewed_proposal": {"n": 100, "answered": 0.61,
                                               "risk_answered": 0.082}}  # fmt: skip
    p.write_text(json.dumps(d))
    out = make_docs._hard_risk(r)
    assert "8.2 %" in out and "0.0 %" in out and "61" in out


def test_other_set_tables_follow_the_release_run(tmp_path):
    r = _v121(tmp_path)
    (r / "posthoc-v1.2.1" / "test_v11" / "TABLE.md").write_text("| posthoc | 1 |\n")
    (r / "test_v11").mkdir()
    (r / "test_v11" / "TABLE.md").write_text("| preregistered era | 1 |\n")
    assert "posthoc" in make_docs.render("{{TABLE_V11}}", r)


def test_llm_latency_comes_from_a_run_that_called_the_llm(tmp_path):
    r = _v121(tmp_path)
    p = r / "posthoc-v1.2.1" / "test_v12" / "llm_qwen3.5_35b-a3b-q4_K_M.json"
    d = json.loads(p.read_text())
    d["latency_ms_per_item"] = 0.004  # answers read from the cache in the post-hoc run
    p.write_text(json.dumps(d))
    d["latency_ms_per_item"] = 117.0  # the pre-registered read called the LLM
    (r / "test_v12" / "llm_qwen3.5_35b-a3b-q4_K_M.json").write_text(json.dumps(d))
    assert "~29× slower" in make_docs._at_a_glance(r)  # 117 ms in the pre-registered read
    assert "~29× slower" in make_docs._headline(r)


def _with_cascade(tmp_path):
    r = _v121(tmp_path)
    parts = {
        "test_v12": {"n": 361, "lexicon": {"answered": 323, "errors": 5},
                     "residual": {"n": 38, "model_answered": 8, "model_errors": 5},
                     "cascade": {"answered": 331, "errors": 10}},
        "test_v11": {"n": 656, "lexicon": {"answered": 163, "errors": 3},
                     "residual": {"n": 493, "model_answered": 355, "model_errors": 22},
                     "cascade": {"answered": 518, "errors": 25}},
    }  # fmt: skip
    (r / "posthoc-v1.2.1" / "cascade.json").write_text(json.dumps({"parts": parts}))
    return r


def test_cascade_table_from_results(tmp_path):
    out = make_docs._cascade(_with_cascade(tmp_path))
    assert "| Fresh registers (New York, Oregon), 361 strings | 89.5 %, error 1.5 % |" in out
    assert "91.7 %, error 3.0 %" in out and "8 of 38, 5 wrong (62.5 %)" in out
    assert "24.8 %, error 1.8 %" in out and "79.0 %, error 4.8 %" in out
    assert "355 of 493, 22 wrong (6.2 %)" in out
    assert make_docs._cascade(_v121(tmp_path / "none")) == ""


def test_at_a_glance_shows_what_the_model_adds_on_messy_strings(tmp_path):
    out = make_docs._at_a_glance(_with_cascade(tmp_path))
    assert "| Messy register strings (656): lexicon alone → lexicon, then model |" in out
    assert "answers 24.8 % → 79.0 % (error 4.8 %)" in out
