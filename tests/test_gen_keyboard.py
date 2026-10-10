import numpy as np

from countrycode.synth.framework import BaseAlias
from countrycode.synth.keyboard import LAYOUTS, KeyboardTypos, neighbours


def test_neighbours_are_physical():
    n = neighbours(LAYOUTS["us"])
    assert {"q", "w", "s", "z"} <= n["a"] and "p" not in n["a"]
    assert "ü" in neighbours(LAYOUTS["de"])["p"]


def test_variants_differ_by_few_edits_and_keep_case():
    gen = KeyboardTypos(per_alias=5)
    row = BaseAlias("Austria", ("AT",), "name", "de")
    out = gen.generate(row, np.random.default_rng(1))
    assert 1 <= len(out) <= 5 and all(v != "Austria" for v in out)
    from rapidfuzz.distance import OSA

    assert all(OSA.distance(v.lower(), "austria") <= 3 for v in out)
    assert all(v[0].isupper() or v[0] != "A" for v in out)


def test_deterministic():
    gen = KeyboardTypos(per_alias=3)
    row = BaseAlias("Norway", ("NO",), "name", "en")
    a = gen.generate(row, np.random.default_rng(5))
    b = gen.generate(row, np.random.default_rng(5))
    assert a == b


def test_skips_codes_and_short_aliases():
    gen = KeyboardTypos()
    assert gen.generate(BaseAlias("AT", ("AT",), "code", "zxx"), np.random.default_rng(0)) == []
    assert gen.generate(BaseAlias("Iran", ("IR",), "name", "en"), np.random.default_rng(0)) != []
    assert gen.generate(BaseAlias("Mali", ("ML",), "name", "en"), np.random.default_rng(0)) != []
    assert gen.generate(BaseAlias("Chad", ("TD",), "name", "en"), np.random.default_rng(0)) != []


def test_only_latin_aliases_get_typos_and_priority_languages_get_more():
    gen = KeyboardTypos(per_alias=3)
    rng = np.random.default_rng(0)
    assert gen.generate(BaseAlias("Норвегия", ("NO",), "name", "ru"), rng) == []
    assert gen.generate(BaseAlias("ৰোৱাণ্ডা", ("RW",), "name", "as"), rng) == []
    assert len(gen.generate(BaseAlias("Norwegen", ("NO",), "name", "de"), rng)) == 3
    assert len(gen.generate(BaseAlias("Noorwe", ("NO",), "name", "af"), rng)) == 1


def _info(**langs: str):
    from countrycode.synth.framework import BaseInfo

    out = {}
    for a, v in langs.items():
        lg, t = v.split(":")
        out[a] = BaseInfo(frozenset({lg}), frozenset({"name"}), frozenset({t}))
    return out


def test_v2_types_only_register_names_of_six_letters_or_more():
    info = _info(Austria="en:AT", Austrija="lv:AT", Peru="en:PE")
    gen = KeyboardTypos(per_alias=5, info=info)
    assert gen.version == "2"
    rng = np.random.default_rng(0)
    assert gen.generate(BaseAlias("Austrija", ("AT",), "name", "lv"), rng) == []
    assert gen.generate(BaseAlias("Peru", ("PE",), "name", "en"), rng) == []


def test_v2_keeps_variants_recoverable():
    from rapidfuzz.distance import OSA

    from countrycode.text import compact_key

    gen = KeyboardTypos(per_alias=5, info=_info(Austria="en:AT"))
    for seed in range(20):
        out = gen.generate(BaseAlias("Austria", ("AT",), "name", "en"), np.random.default_rng(seed))
        assert all(OSA.distance(compact_key(v), "austria") <= 1 for v in out)  # 0.2 × 7 letters


def test_v2_writes_some_typos_in_capitals_and_some_with_a_shift_slip():
    gen = KeyboardTypos(per_alias=5, info=_info(Australia="en:AU"))
    row = BaseAlias("Australia", ("AU",), "name", "en")
    out = [v for s in range(60) for v in gen.generate(row, np.random.default_rng(s))]
    assert any(v.isupper() for v in out)
    assert any(v[:2].isupper() and v[2:].islower() for v in out)  # "AUstralia"
