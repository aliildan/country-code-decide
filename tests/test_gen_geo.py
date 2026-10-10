import numpy as np

from countrycode.lexicon.build import build_lexicon, code_rows
from countrycode.lexicon.types import AliasRow, Overrides
from countrycode.synth.geo import GeoContext, load_geonames

LX = build_lexicon(
    [
        AliasRow("Austria", "AT", "name", "en", "t"),
        AliasRow("Österreich", "AT", "name", "de", "t"),
        AliasRow("Texas", "US-TX", "name", "en", "t"),
        *code_rows(),
    ],
    Overrides(),
)


def write_geo(tmp_path):
    (tmp_path / "admin1CodesASCII.txt").write_text(
        "AT.07\tTyrol\tTyrol\t1\nAT.09\tVienna\tVienna\t2\nUS.TX\tTexas\tTexas\t3\n",
        encoding="utf-8",
    )
    rows = [
        [
            "10",
            "Innsbruck",
            "Innsbruck",
            "",
            "0",
            "0",
            "P",
            "PPLA",
            "AT",
            "",
            "07",
            "",
            "",
            "",
            "130000",
        ],
        [
            "11",
            "Vienna",
            "Vienna",
            "",
            "0",
            "0",
            "P",
            "PPLC",
            "AT",
            "",
            "09",
            "",
            "",
            "",
            "1900000",
        ],
        ["12", "Austin", "Austin", "", "0", "0", "P", "PPLA", "US", "", "TX", "", "", "", "960000"],
    ]
    (tmp_path / "cities15000.txt").write_text(
        "\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8"
    )


def test_load_geonames(tmp_path):
    write_geo(tmp_path)
    regions, cities = load_geonames(tmp_path)
    assert regions["AT"] == ["Tyrol", "Vienna"] and cities["AT"][0] == "Vienna"
    assert cities["US-TX"] == ["Austin"]


def test_country_and_state_variants(tmp_path):
    write_geo(tmp_path)
    gen = GeoContext(LX, tmp_path, per_target=40)
    at = [
        v.lower()
        for v in gen.generate(
            next(b for b in gen.bases() if b.targets == ("AT",)), np.random.default_rng(0)
        )
    ]
    assert any("tyrol" in v and ("austria" in v or "österreich" in v) for v in at)
    tx = [
        v.lower()
        for v in gen.generate(
            next(b for b in gen.bases() if b.targets == ("US-TX",)), np.random.default_rng(0)
        )
    ]
    assert any("austin" in v and ("texas" in v or "tx" in v) for v in tx)
    assert gen.near_check is False


def test_geo_v2_skips_regions_in_western_sahara(tmp_path):
    (tmp_path / "admin1CodesASCII.txt").write_text(
        "MA.06\tCasablanca-Settat\tCasablanca-Settat\t1\n"
        "MA.12\tDakhla-Oued Ed-Dahab\tDakhla-Oued Ed-Dahab\t2\n",
        encoding="utf-8",
    )
    (tmp_path / "cities15000.txt").write_text("", encoding="utf-8")
    lx = build_lexicon([AliasRow("Morocco", "MA", "name", "en", "t"), *code_rows()], Overrides())
    gen = GeoContext(lx, tmp_path, per_target=10, version="2")
    assert gen.version == "2"
    out = gen.generate(
        next(b for b in gen.bases() if b.targets == ("MA",)), np.random.default_rng(0)
    )
    assert out and not any("DAKHLA" in s.upper() for s in out)


def test_geo_v2_uses_standard_names_not_nicknames(tmp_path):
    write_geo(tmp_path)
    lx = build_lexicon(
        [
            AliasRow("Texas", "US-TX", "name", "en", "wikidata"),
            AliasRow("Lone Star State", "US-TX", "name", "en", "wikidata"),
            AliasRow("Austria", "AT", "name", "en", "cldr"),
            *code_rows(),
        ],
        Overrides(),
    )
    gen = GeoContext(lx, tmp_path, per_target=40, version="2")
    out = [s for b in gen.bases() for s in gen.generate(b, np.random.default_rng(1))]
    assert out and not any("LONE STAR" in s.upper() for s in out)


def test_geo_v2_uses_alpha3_codes_and_skips_containment(tmp_path):
    (tmp_path / "admin1CodesASCII.txt").write_text("CA.01\tAlberta\tAlberta\t1\n", encoding="utf-8")
    (tmp_path / "cities15000.txt").write_text("", encoding="utf-8")
    rows = [AliasRow("Canada", "CA", "name", "en", "cldr"),
            AliasRow("CAN", "CA", "code", "zxx", "wikidata")]  # fmt: skip
    lx = build_lexicon([*rows, *code_rows()], Overrides())
    gen = GeoContext(lx, tmp_path, per_target=40, version="2")
    assert gen.containment_check is False
    base = next(b for b in gen.bases() if b.targets == ("CA",))
    out = gen.generate(base, np.random.default_rng(0))
    assert any("CAN" in s.upper().replace("CANADA", "") for s in out)


def test_geo_v2_alpha3_codes_come_from_wikidata_only(tmp_path):
    admin1 = tmp_path / "admin1CodesASCII.txt"
    admin1.write_text("XK.01\tPristina\tPristina\t1\n", encoding="utf-8")
    (tmp_path / "cities15000.txt").write_text("", encoding="utf-8")
    rows = [AliasRow("Kosovo", "XK", "name", "en", "cldr"),
            AliasRow("UVK", "XK", "code", "zxx", "overrides")]  # fmt: skip
    gen = GeoContext(build_lexicon([*rows, *code_rows()], Overrides()), tmp_path, 60, version="2")
    out = [s for b in gen.bases() for s in gen.generate(b, np.random.default_rng(2))]
    assert out and not any("UVK" in s.upper() for s in out)  # an override code, not ISO alpha-3
