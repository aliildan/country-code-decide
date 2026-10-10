import pytest

from countrycode.synth.realism import edit_profile, js_divergence


def test_edit_profile_counts_operations():
    pairs = [
        ("deleware", "delaware"),
        ("austira", "austria"),
        ("unitedkingdon", "unitedkingdom"),
        ("brmuda", "bermuda"),
        ("canadda", "canada"),
    ]
    p = edit_profile(pairs)
    assert p["n_pairs"] == 5
    ops = p["op_share"]
    assert ops["replace"] > 0 and ops["delete"] > 0 and ops["insert"] > 0 and ops["transpose"] > 0
    assert sum(ops.values()) == pytest.approx(1.0)
    assert p["edits_share"]["1"] == pytest.approx(1.0)  # all five are single edits


def test_js_divergence():
    assert js_divergence({"a": 1.0}, {"a": 1.0}) == pytest.approx(0.0)
    assert js_divergence({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)


def test_typo_pairs_exclude_extra_words():
    from countrycode.lexicon.build import build_lexicon
    from countrycode.lexicon.types import AliasRow, Overrides
    from countrycode.synth.realism import real_noisy_pairs

    lx = build_lexicon([AliasRow("Japan", "JP", "name", "en", "t")], Overrides())
    pairs = real_noisy_pairs([("Japna", ("JP",)), ("AICHI JAPAN", ("JP",))], lx)
    assert pairs == [("japna", "japan")]
