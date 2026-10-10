from countrycode.lexicon.build import Lexicon, build_lexicon, code_rows
from countrycode.lexicon.types import AliasKind, AliasRow, Overrides


def R(a: str, t: str, kind: AliasKind = "name", lang: str = "xx") -> AliasRow:
    return AliasRow(a, t, kind, lang, "test")


BASE = [
    R("Østerrike", "AT"),
    R("Georgia", "GE"),
    R("Georgia", "US-GA"),
    R("Yugoslavia", "RS", "historical"),
    R("Yugoslavia", "ME", "historical"),
]
NO_OV = Overrides()


def test_lookup_and_unique():
    lx = build_lexicon(BASE, NO_OV)
    assert lx.lookup("østerrike ") == {"AT"} and lx.is_unique("Østerrike")
    assert lx.lookup("GEORGIA") == {"GE", "US-GA"} and not lx.is_unique("Georgia")
    assert lx.lookup("Yugoslavia") == {"RS", "ME"}


def test_code_collisions_are_sets():
    lx = build_lexicon(code_rows(), NO_OV)
    for code in ("CA", "DE", "IN", "ME", "AL", "PA", "GA"):
        assert lx.lookup(code) == {code, f"US-{code}"}
    assert lx.lookup("UK") == {"GB"} and lx.lookup("EU") == {"NONE"}


def test_blank_lookup_is_empty():
    lx = build_lexicon(BASE, NO_OV)
    assert lx.lookup("") == frozenset() and lx.lookup(" .- ") == frozenset()


def test_remove_override():
    ov = Overrides(remove=frozenset({("georgia", "GE")}))
    assert build_lexicon(BASE, ov).lookup("Georgia") == {"US-GA"}


def test_save_load_roundtrip(tmp_path):
    lx = build_lexicon(BASE, NO_OV)
    p = tmp_path / "lx.json"
    lx.save(p)
    assert Lexicon.load(p).lookup("Georgia") == {"GE", "US-GA"}


def test_flag_emoji_resolves_via_code():
    lx = build_lexicon(code_rows(), NO_OV)
    assert lx.lookup("🇦🇹") == {"AT"}


def test_historical_yields_to_current():
    rows = [R("Serbia", "RS"), R("Serbia", "BA", "historical"), R("Serbia", "SI", "historical")]
    assert build_lexicon(rows, NO_OV).lookup("Serbia") == {"RS"}


def test_codes_take_precedence_over_name_aliases():
    rows = [*code_rows(), R("SK", "KR"), R("PA", "PS")]
    lx = build_lexicon(rows, NO_OV)
    assert lx.lookup("SK") == {"SK"} and lx.lookup("PA") == {"PA", "US-PA"}


def test_weak_support_dropped():
    rows = [R("China", "CN", lang=f"l{i}") for i in range(20)] + [R("China", "TW", lang="x")]
    rows += [R("Georgia", "GE", lang=f"l{i}") for i in range(10)]
    rows += [R("Georgia", "US-GA", lang=f"l{i}") for i in range(5)]
    lx = build_lexicon(rows, NO_OV)
    assert lx.lookup("China") == {"CN"} and lx.lookup("Georgia") == {"GE", "US-GA"}


def test_force_sets_keep_real_ambiguity():
    rows = [R("Congo", "CG", lang=f"l{i}") for i in range(20)] + [R("Congo", "CD")]
    ov = Overrides(force_sets={"congo": frozenset({"CD", "CG"})})
    lx = build_lexicon(rows, ov)
    assert lx.lookup("Congo") == {"CD", "CG"}
    p_lx = Lexicon(lx.rows, lx.meta, force_sets=lx.force_sets)
    assert p_lx.lookup("Congo") == {"CD", "CG"}


def test_raw_lookup_keeps_pre_rule_targets():
    rows = [R("Jersey", "JE", lang=f"l{i}") for i in range(20)] + [R("Jersey", "US-NJ")]
    lx = build_lexicon(rows, NO_OV)
    assert lx.lookup("Jersey") == {"JE"} and lx.raw_lookup("Jersey") == {"JE", "US-NJ"}


def test_short_keys_need_the_same_spaced_form():
    rows = [*code_rows(), R("U.S.", "US")]
    lx = build_lexicon(rows, NO_OV)
    assert lx.lookup("N/A") == frozenset() and lx.raw_lookup("N/A") == frozenset()
    assert lx.lookup("NA") == {"NA"} and lx.lookup("na") == {"NA"}
    assert lx.lookup("U.S.") == {"US"} and lx.lookup("US") == {"US"}
    assert lx.lookup("Austria") == frozenset()  # unrelated: unchanged behaviour
