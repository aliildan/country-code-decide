from countrycode.lexicon.build import build_lexicon, code_rows
from countrycode.lexicon.types import AliasKind, AliasRow, Overrides
from countrycode.synth.framework import base_aliases, base_info, typed_name, unsafe_reason


def R(a: str, t: str, lang: str = "en", kind: AliasKind = "name") -> AliasRow:
    return AliasRow(a, t, kind, lang, "t")


LX = build_lexicon(
    [
        R("Austria", "AT"),
        R("Österreich", "AT", "de"),
        R("Austrija", "AT", "lv"),
        R("Gabonese", "GA", kind="demonym"),
        R("Gabonaise", "GA", "fr", "demonym"),
        R("Micronesia", "FM"),
        R("Micronesia", "NONE", "en", "non_country"),
        *(R("Yugoslavia", t, "en", "historical") for t in ("BA", "HR", "ME")),
        R("burundeză", "BI", "ro", "demonym"),
        R("burundeza", "FI", "vec", "demonym"),
        R("Brasil", "BR", "pt_BR"),
        *code_rows(),
    ],
    Overrides(),
)
BASES = {b.alias: b for b in base_aliases(LX)}
INFO = base_info(LX)


def test_unsafe_bases_are_named():
    assert unsafe_reason(BASES["Micronesia"], INFO) == "none_mix"
    assert unsafe_reason(BASES["Yugoslavia"], INFO) == "multi_successor"
    assert unsafe_reason(BASES["burundeză"], INFO) == "key_collision"


def test_ordinary_names_and_codes_are_safe():
    assert unsafe_reason(BASES["Austria"], INFO) is None
    assert unsafe_reason(BASES["AT"], INFO) is None


def test_typed_names_are_register_languages_and_english_demonyms():
    assert typed_name(BASES["Austria"], INFO) and typed_name(BASES["Österreich"], INFO)
    assert typed_name(BASES["Brasil"], INFO)  # pt_BR counts as pt
    assert not typed_name(BASES["Austrija"], INFO)
    assert typed_name(BASES["Gabonese"], INFO) and not typed_name(BASES["Gabonaise"], INFO)
    assert not typed_name(BASES["AT"], INFO)  # codes are not names


def test_noise_seeds_are_cldr_names_or_aliases_containing_a_standard_name():
    from countrycode.synth.framework import seedable

    lx = build_lexicon(
        [
            AliasRow("Tunesië", "TN", "name", "nl", "cldr"),
            AliasRow("Republiek Tunesië", "TN", "name", "nl", "wikidata"),
            AliasRow("Staat New York", "US-NY", "name", "de", "wikidata"),
            AliasRow("Palmetto State", "US-SC", "name", "en", "wikidata"),
            AliasRow("Southern Provinces", "EH", "name", "en", "wikidata"),
            *code_rows(),
        ],
        Overrides(),
    )
    bases, info = {b.alias: b for b in base_aliases(lx)}, base_info(lx)
    assert seedable(bases["Tunesië"], info) and seedable(bases["Republiek Tunesië"], info)
    assert seedable(bases["Staat New York"], info)  # contains the English name "New York"
    assert not seedable(bases["Palmetto State"], info)  # a nickname
    assert not seedable(bases["Southern Provinces"], info)


def test_noisy_variant_must_keep_a_core_word_of_its_target():
    from countrycode.synth.framework import recoverable

    lx = build_lexicon(
        [
            AliasRow("Iémen", "YE", "name", "pt", "cldr"),
            AliasRow("Iémen do Sul", "YE", "historical", "pt", "cldr"),
            AliasRow("Suíça", "CH", "name", "pt", "cldr"),
            AliasRow("Confederação Suíça", "CH", "name", "pt", "wikidata"),
            *code_rows(),
        ],
        Overrides(),
    )
    info = base_info(lx)
    ye, ch = info["Iémen do Sul"].core, info["Confederação Suíça"].core
    assert recoverable("Iémen do Sul", "lemen do Sul", 0.15, ye)
    assert not recoverable("Iémen do Sul", "remendo Sul", 0.15, ye)  # the country word is gone
    assert recoverable("Confederação Suíça", "confederatio suica", 0.15, ch)  # diacritics fold


def test_regions_named_like_a_country_are_unsafe():
    lx = build_lexicon(
        [
            AliasRow("Polynésie française", "PF", "name", "fr", "cldr"),
            AliasRow("Polynésie", "NONE", "non_country", "fr", "cldr"),
            AliasRow("Östafrika", "NONE", "non_country", "sv", "cldr"),
            *code_rows(),
        ],
        Overrides(),
    )
    bases, info = {b.alias: b for b in base_aliases(lx)}, base_info(lx)
    assert unsafe_reason(bases["Polynésie"], info) == "country_like_region"
    assert unsafe_reason(bases["Östafrika"], info) is None


def test_shared_island_codes_are_unsafe_seeds():
    from countrycode.synth.framework import BaseAlias

    b = BaseAlias("Saint-Martin", ("MF",), "name", "fr")
    assert unsafe_reason(b, INFO) == "shared_island"  # one island, two codes (owner audit)


def test_acronyms_of_the_standard_name_are_anchored():
    rows = [AliasRow(a, t, "name", "en", "wikidata") for a, t in
            (("UAE", "AE"), ("UK", "GB"), ("Emirates", "AE"))]  # fmt: skip
    lx = build_lexicon([*rows, *code_rows()], Overrides())
    info = base_info(lx)
    assert info["UAE"].anchored and info["UK"].anchored  # United Arab Emirates, United Kingdom
    assert not info["Emirates"].anchored


def test_country_like_regions_are_found_in_other_scripts_too():
    lx = build_lexicon(
        [AliasRow("Zuid-Afrika", "ZA", "name", "nl", "cldr"),
         AliasRow("Паўднёвая Афрыка", "NONE", "non_country", "be", "cldr"), *code_rows()],
        Overrides(),
    )  # fmt: skip
    b = {x.alias: x for x in base_aliases(lx)}["Паўднёвая Афрыка"]
    assert unsafe_reason(b, base_info(lx)) == "country_like_region"  # "Afryka" ~ "Afrika"
