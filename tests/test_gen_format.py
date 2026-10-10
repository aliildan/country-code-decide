import numpy as np

from countrycode.lexicon.build import build_lexicon, code_rows
from countrycode.lexicon.types import AliasKind, AliasRow, Overrides
from countrycode.synth.formats import FormatVariants, JunkStrings, Transliterate
from countrycode.synth.framework import BaseAlias, generate_items


def R(a: str, t: str, lang: str = "en", kind: AliasKind = "name") -> AliasRow:
    return AliasRow(a, t, kind, lang, "t")


LX = build_lexicon(
    [
        R("Austria", "AT"),
        R("Australia", "AU"),
        R("Bermuda", "BM"),
        R("Korea, Republic of", "KR"),
        R("Republic of Korea", "KR"),
        R("Норвегия", "NO", "ru"),
        R("Österreich", "AT", "de"),
        *code_rows(),
    ],
    Overrides(),
)


def texts(gen, row, seed=0):
    return [
        v if isinstance(v, str) else v[0] for v in gen.generate(row, np.random.default_rng(seed))
    ]


def test_truncation_labels_are_prefix_sets():
    gen = FormatVariants(LX)
    out = dict(
        v
        for v in gen.generate(BaseAlias("Austria", ("AT",), "name", "en"), np.random.default_rng(0))
        if not isinstance(v, str)
    )
    assert out.get("AUST") == ("AT", "AU")
    assert all(len(k) >= 3 for k in out)


def test_format_variants_include_case_and_zip_noise():
    out = texts(FormatVariants(LX, zip_prob=1.0), BaseAlias("Bermuda", ("BM",), "name", "en"))
    assert "BERMUDA" in out and any(any(ch.isdigit() for ch in v) for v in out)


def test_comma_inversion():
    out = texts(FormatVariants(LX), BaseAlias("Republic of Korea", ("KR",), "name", "en"))
    assert "Korea, Republic of" in out or "KOREA, REPUBLIC OF" in out


def test_transliteration():
    assert "Norvegiya" in texts(Transliterate(), BaseAlias("Норвегия", ("NO",), "name", "ru"))
    out = texts(Transliterate(), BaseAlias("Österreich", ("AT",), "name", "de"))
    assert "Oesterreich" in out and "Osterreich" in out


def test_junk_is_none_and_never_a_country():
    gen = JunkStrings(n=50)
    items, _ = generate_items(gen, gen.bases(), real_keys=set(), seed=1)
    assert len(items) >= 30 and all(i.targets == ("NONE",) for i in items)
    assert not any(LX.lookup(i.text) - {"NONE"} for i in items)


def test_junk_survives_g1_exact_only():
    from countrycode.synth.gates import CollisionChecker

    gen = JunkStrings(n=40)
    items, dropped = generate_items(gen, gen.bases(), set(), seed=2, checker=CollisionChecker(LX))
    assert len(items) >= 20 and dropped.get("near_other", 0) == 0


def test_format_only_latin():
    assert (
        FormatVariants(LX).generate(
            BaseAlias("Норвегия", ("NO",), "name", "ru"), np.random.default_rng(0)
        )
        == []
    )


def test_codes_get_case_variants():
    out = texts(FormatVariants(LX), BaseAlias("TX", ("US-TX",), "code", "zxx"))
    assert set(out) == {"tx", "Tx"}
    assert FormatVariants(LX).version == "3"


def CL(a: str, t: str, lang: str = "en") -> AliasRow:
    return AliasRow(a, t, "name", lang, "cldr")


LX4 = build_lexicon(
    [
        CL("Austria", "AT"),
        CL("Australia", "AU"),
        CL("Austrija", "AT", "lv"),
        CL("Österreich", "AT", "de"),
        R("Vašingtons", "US-WA", "sgs"),
        R("Норвегия", "NO", "ru"),
        *code_rows(),
    ],
    Overrides(),
)


def _info():
    from countrycode.synth.framework import base_info

    return base_info(LX4)


def test_format_v4_keeps_code_case_variants():
    gen = FormatVariants(LX4, info=_info())
    assert gen.version == "4"
    out = texts(gen, BaseAlias("TX", ("US-TX",), "code", "zxx"))
    assert "tx" in out and "Tx" in out


def test_format_v4_skips_untyped_names_and_has_no_prefix_truncations():
    gen = FormatVariants(LX4, info=_info())
    assert texts(gen, BaseAlias("Austrija", ("AT",), "name", "lv")) == []
    for alias, lang in (("Austria", "en"), ("Österreich", "de")):
        out = gen.generate(BaseAlias(alias, ("AT",), "name", lang), np.random.default_rng(0))
        assert out and all(isinstance(v, str) for v in out)  # "SAHRA", "ROMAN" were words


def test_translit_v2_latin_sources_typed_only():
    gen = Transliterate(info=_info())
    assert gen.version == "2"
    assert texts(gen, BaseAlias("Vašingtons", ("US-WA",), "name", "sgs")) == []
    assert "Oesterreich" in texts(gen, BaseAlias("Österreich", ("AT",), "name", "de"))
    assert "Norvegiya" in texts(gen, BaseAlias("Норвегия", ("NO",), "name", "ru"))


def test_junk_v2_has_no_zip_shaped_strings():
    import re

    gen = JunkStrings(n=400, version="2")
    assert gen.version == "2"
    out = [v for b in gen.bases() for v in texts(gen, b, seed=int(b.alias.split("_")[-1]))]
    assert out and not any(re.search(r"(?<!\d)\d{5}(?!\d)", v) for v in out)


def test_translit_v2_never_leaves_other_scripts_half_converted():
    lx = build_lexicon([R("Хуссар Дакотæ", "US-SD", "os"), *code_rows()], Overrides())
    from countrycode.synth.framework import base_info

    out = texts(
        Transliterate(info=base_info(lx)), BaseAlias("Хуссар Дакотæ", ("US-SD",), "name", "os")
    )
    assert out and all(not any(ord(c) >= 0x250 and c.isalpha() for c in v) for v in out)


def test_translit_v3_romanises_only_cldr_names_of_major_scripts():
    from countrycode.synth.framework import base_info

    rows = [
        AliasRow("Норвегия", "NO", "name", "ru", "cldr"),
        AliasRow("واشینگتن", "US-WA", "name", "fa", "wikidata"),  # not CLDR
        AliasRow("Ангол", "AO", "name", "mn", "cldr"),  # Mongolian: not a major register script
        AliasRow("Gyɔgyea", "GE", "name", "ak", "cldr"),  # IPA letters, Akan
        AliasRow("ҶУ", "UZ", "name", "tg", "wikidata"),
        AliasRow("Österreich", "AT", "name", "de", "cldr"),
        *code_rows(),
    ]
    lx = build_lexicon(rows, Overrides())
    gen = Transliterate(info=base_info(lx), version="3")
    assert gen.version == "3"
    got = {r.alias: texts(gen, BaseAlias(r.alias, (r.target,), "name", r.lang)) for r in rows[:6]}
    assert "Norvegiya" in got["Норвегия"] and "Oesterreich" in got["Österreich"]
    assert got["واشینگتن"] == got["Ангол"] == got["Gyɔgyea"] == got["ҶУ"] == []


def test_translit_v4_drops_romanisations_that_name_a_place_of_another_country():
    from countrycode.synth.framework import base_info

    rows = [AliasRow("黑山", "ME", "name", "zh", "cldr"),
            AliasRow("Норвегия", "NO", "name", "ru", "cldr"), *code_rows()]  # fmt: skip
    gen = Transliterate(info=base_info(build_lexicon(rows, Overrides())), version="4",
                        places={"heishan": {"CN"}, "norvegiya": {"NO"}})  # fmt: skip
    assert texts(gen, BaseAlias("黑山", ("ME",), "name", "zh")) == []  # a county in Liaoning
    assert "Norvegiya" in texts(gen, BaseAlias("Норвегия", ("NO",), "name", "ru"))  # own place
