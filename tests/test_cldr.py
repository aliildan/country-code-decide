from countrycode import labels as L
from countrycode.lexicon.cldr import cldr_version, load_cldr

ROWS = load_cldr()
S = {(r.alias, r.target, r.lang) for r in ROWS}


def test_expected_names():
    assert ("Österreich", "AT", "de") in S
    assert ("Østerrike", "AT", "nb") in S
    assert ("Almanya", "DE", "tr") in S
    assert ("ノルウェー", "NO", "ja") in S


def test_regions_are_none():
    assert any(r.alias == "Europa" and r.target == L.NONE and r.kind == "non_country" for r in ROWS)


def test_targets_valid():
    assert all(L.is_target(r.target) for r in ROWS)
    assert cldr_version()
