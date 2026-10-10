import json

from countrycode.train.__main__ import write_train_summary


def test_train_summary_is_aggregates_only(tmp_path):
    cfg = {"name": "small-x", "pending_g1": True}
    check = {"dataset": "data/processed/train/x.jsonl", "sha256": "abc", "seed": 13,
             "gens": ["keyboard@1"], "examples": 10, "by_source": {"lexicon": 10},
             "synthetic_equal_to_real": 0, "ok": True}  # fmt: skip
    path = write_train_summary(cfg, check, tmp_path)
    s = json.loads(path.read_text())
    assert path.name == "small-x.json"
    assert s == {"run": "small-x", "dataset_sha256": "abc", "seed": 13, "gens": ["keyboard@1"],
                 "examples": 10, "by_source": {"lexicon": 10}, "synthetic_equal_to_real": 0,
                 "real_train_in_eval": 0, "pending_g1": True}  # fmt: skip
