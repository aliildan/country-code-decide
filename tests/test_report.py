import json

from countrycode.evaluation.report import markdown_table


def test_markdown_table(tmp_path):
    for name, acc in (("a", 0.5), ("b", 0.75)):
        rep = {
            "predictor": name,
            "overall": {
                "n": 10,
                "accuracy": acc,
                "accuracy_weighted": acc,
                "country_accuracy": acc,
                "coverage_at_risk": {"0.01": 0.1, "0.05": 0.2},
                "none_false_accept": 0.1,
                "ece": 0.05,
                "aurc": 0.2,
            },
            "latency_ms_per_item": 1.5,
        }
        (tmp_path / f"{name}.json").write_text(json.dumps(rep))
    md = markdown_table(tmp_path)
    lines = md.splitlines()
    assert lines[0].startswith("| predictor") and "b" in lines[2]  # sorted by accuracy desc
    assert "0.750" in md and "1.5" in md


def test_llm_answers_served_from_the_cache_show_no_latency(tmp_path):
    for name, ms in (("llm_x", 0.004), ("lexicon_exact", 0.004)):
        rep = {"predictor": name, "overall": {"n": 1, "accuracy": 1.0}, "latency_ms_per_item": ms}
        (tmp_path / f"{name}.json").write_text(json.dumps(rep))
    rows = {line.split(" | ")[0]: line for line in markdown_table(tmp_path).splitlines()[2:]}
    assert rows["| llm_x"].endswith("| cached |")  # a 35B LLM never answers in 0.004 ms
    assert rows["| lexicon_exact"].endswith("| 0.0 |")
