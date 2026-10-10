from countrycode.lexicon.build import build_lexicon, code_rows
from countrycode.lexicon.types import AliasRow, Overrides
from countrycode.synth.combos import Affixes, CountryPairs
from countrycode.synth.framework import generate_items
from countrycode.synth.gates import CollisionChecker

LX = build_lexicon(
    [
        AliasRow(a, t, "name", "en", "t")
        for a, t in [
            ("India", "IN"),
            ("Nepal", "NP"),
            ("Austria", "AT"),
            ("Germany", "DE"),
            ("Sudan", "SD"),
            ("South Sudan", "SS"),
            ("Kenya", "KE"),
            ("Australia", "AU"),
        ]
    ]
    + code_rows(),
    Overrides(),
)


def test_pairs_are_sets_of_both():
    gen = CountryPairs(LX, n=50)
    items, _ = generate_items(gen, gen.bases(), set(), seed=1)
    assert items and all(len(i.targets) == 2 for i in items)
    assert all(i.targets[0] != i.targets[1] for i in items)


def test_affixes_keep_country_and_g1_catches_new_countries():
    gen = Affixes(LX)
    items, dropped = generate_items(gen, gen.bases(), set(), seed=2, checker=CollisionChecker(LX))
    texts = {i.text.upper(): i.targets for i in items}
    assert any(
        t.startswith(("NORTH", "N ", "EAST", "E ")) and v == ("KE",) for t, v in texts.items()
    )
    assert texts.get("SOUTH SUDAN") in (None, ("SS",))  # never labelled SD


def C(a: str, t: str, lang: str = "en", source: str = "cldr") -> AliasRow:
    return AliasRow(a, t, "name", lang, source)


LXC = build_lexicon(
    [
        C("Ireland", "IE"),
        C("Northern Ireland", "GB", source="wikidata"),
        C("United Kingdom", "GB"),
        C("Kenya", "KE"),
        C("Kenia", "KE", "de"),
        C("Austria", "AT"),
        C("South Africa", "ZA"),
        C("St. Martin", "MF"),
        C("Sint Maarten", "SX"),
        C("Niue", "NU"),
        C("Savage", "NU", "fr", "wikidata"),
        *code_rows(),
    ],
    Overrides(),
)


def test_affix_v2_uses_english_cldr_names_and_skips_nested_or_shared_targets():
    gen = Affixes(LXC, version="2")
    assert gen.version == "2"
    targets = {b.targets[0] for b in gen.bases()}
    assert {"KE", "AT", "GB"} <= targets
    assert not targets & {"IE", "MF", "SX", "ZA"}  # in "Northern Ireland"; shared island; compass
    items, _ = generate_items(gen, gen.bases(), set(), seed=3)
    assert not any("KENIA" in i.text.upper() for i in items)


def test_pairs_v3_use_english_cldr_names_without_shared_islands():
    gen = CountryPairs(LXC, n=300, version="3")
    items, _ = generate_items(gen, gen.bases(), set(), seed=4)
    texts = " ".join(i.text.upper() for i in items)
    assert items and "SAVAGE" not in texts and "KENIA" not in texts
    assert not any({"MF", "SX"} & set(i.targets) for i in items)
