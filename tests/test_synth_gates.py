from countrycode.lexicon.build import build_lexicon, code_rows
from countrycode.lexicon.types import AliasKind, AliasRow, Overrides
from countrycode.synth.gates import CollisionChecker


def R(a: str, t: str, lang: str = "en", kind: AliasKind = "name") -> AliasRow:
    return AliasRow(a, t, kind, lang, "t")


LX = build_lexicon(
    [
        R("Austria", "AT"),
        R("Australia", "AU"),
        R("Iran", "IR"),
        R("Iraq", "IQ"),
        R("Niger", "NE"),
        R("Nigeria", "NG"),
        R("Georgia", "GE"),
        R("Georgia", "US-GA"),
        R("Gruzia", "GE"),
        *code_rows(),
    ],
    Overrides(),
)
CHK = CollisionChecker(LX)


def test_typo_into_another_country_is_dropped():
    keep, _, reason = CHK("Iraq", ("IR",), "Iran")
    assert not keep and reason == "exact_other"


def test_tie_with_other_country_is_dropped():
    keep, _, reason = CHK("Irak", ("IR",), "Iran")  # as close to Iraq as to Iran
    assert not keep and reason == "near_other"


def test_clear_own_variant_kept():
    assert CHK("Austira", ("AT",), "Austria") == (True, ("AT",), "ok")
    assert CHK("Nigerria", ("NG",), "Nigeria")[0]


def test_exact_hit_takes_lexicon_set():
    assert CHK("GEORGIA", ("GE", "US-GA"), "Georgia") == (True, ("GE", "US-GA"), "ok")
    assert CHK("GRUZIA", ("GE", "US-GA"), "Georgia") == (True, ("GE",), "relabel_exact")


def test_zero_similarity_to_source_does_not_crash():
    keep, _, _ = CHK("Ostriya", ("AT",), "Австрия")  # romanisation shares no letters
    assert keep in (True, False)


def test_romanisation_compared_with_romanised_source():
    lx = build_lexicon([R("Австрия", "AT", "ru"), R("Australia", "AU"), *code_rows()], Overrides())
    assert CollisionChecker(lx)("Avstriya", ("AT",), "Австрия")[0]


def test_own_score_uses_any_alias_of_the_target():
    lx = build_lexicon(
        [R("Germany", "DE"), R("Deutschland", "DE", "de"), R("Dominica", "DM"), *code_rows()],
        Overrides(),
    )
    keep, targets, _ = CollisionChecker(lx)("Doitschland", ("DE",), "Germany")
    assert keep and targets == ("DE",)


LX2 = build_lexicon(
    [
        R("Washington", "US-WA"),
        R("Washington DC", "US-DC"),
        R("Niger", "NE"),
        R("Republic of Niger", "NE"),
        R("Nigerian", "NG", kind="demonym"),
        R("Sudan", "SD"),
        R("South Sudan", "SS"),
        R("Australia", "AU"),
        R("Micronesia", "FM"),
        R("Micronesia", "NONE", kind="non_country"),
        *code_rows(),
    ],
    Overrides(),
)
STRICT = CollisionChecker(LX2, margin=10, containment=True, reject_none_mix=True)


def test_margin_drops_a_variant_almost_as_close_to_another_code():
    assert CollisionChecker(LX2)("Wasington", ("US-WA",), "Washington")[0]
    keep, _, reason = STRICT("Wasington", ("US-WA",), "Washington")
    assert not keep and reason == "near_other"


def test_containment_drops_another_codes_name_inside_the_variant():
    keep, _, reason = STRICT("Nigerian Republic", ("NE",), "Niger")
    assert not keep and reason == "contains_other"


def test_containment_keeps_names_that_nest_another_name():
    assert STRICT("SOUTH SUDAN REPUBLIC", ("SS",), "South Sudan") == (True, ("SS",), "ok")


def test_containment_ignores_short_tokens():
    assert STRICT("AUSTRALIA NE", ("AU",), "Australia")[0]  # "NE" is Niger's code, not a name


def test_none_mixed_with_a_code_is_rejected():
    keep, _, reason = STRICT("MICRONESIA", ("FM",), "Micronesia")  # lexicon set {FM, NONE}
    assert not keep and reason == "none_mix"


def test_containment_drops_a_name_whose_set_is_wider_than_the_label():
    lx = build_lexicon(
        [R("Czechia", "CZ"), R("Czechoslovakia", "CZ", kind="historical"),
         R("Czechoslovakia", "SK", kind="historical"), R("Georgia", "GE"), R("Georgia", "US-GA"),
         R("Republic of Georgia", "GE"), *code_rows()],
        Overrides(),
    )  # fmt: skip
    chk = CollisionChecker(lx, margin=10, containment=True, reject_none_mix=True)
    keep, _, reason = chk("Czechoslovakia (former)", ("CZ",), "Czechia")
    assert not keep and reason == "contains_other"
    assert chk("REPUBLIC OF GEORGIA (TBILISI)", ("GE",), "Republic of Georgia")[0]  # covered
