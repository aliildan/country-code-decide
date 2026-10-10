from countrycode.lexicon.build import build_lexicon, code_rows
from countrycode.lexicon.types import AliasRow, Overrides
from countrycode.synth.city import CityNames, city_index
from countrycode.synth.framework import generate_items
from countrycode.synth.gates import CollisionChecker


def write_cities(tmp_path):
    rows = [
        [
            "1",
            "Vienna",
            "Vienna",
            "Wien,Vienne,Bécs",
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
        [
            "2",
            "Paris",
            "Paris",
            "Parigi",
            "0",
            "0",
            "P",
            "PPLC",
            "FR",
            "",
            "11",
            "",
            "",
            "",
            "2100000",
        ],
        ["3", "Paris", "Paris", "", "0", "0", "P", "PPL", "US", "", "TX", "", "", "", "25000"],
        [
            "4",
            "Springfield",
            "Springfield",
            "",
            "0",
            "0",
            "P",
            "PPL",
            "US",
            "",
            "IL",
            "",
            "",
            "",
            "110000",
        ],
        [
            "5",
            "Springfield",
            "Springfield",
            "",
            "0",
            "0",
            "P",
            "PPL",
            "US",
            "",
            "MA",
            "",
            "",
            "",
            "150000",
        ],
        ["6", "Lebanon", "Lebanon", "", "0", "0", "P", "PPL", "US", "", "PA", "", "", "", "26000"],
    ]
    (tmp_path / "cities15000.txt").write_text(
        "\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8"
    )


def test_city_index_sets(tmp_path):
    write_cities(tmp_path)
    idx = city_index(tmp_path)
    assert idx["wien"] == {"AT"} and idx["paris"] == {"FR", "US-TX"}
    assert idx["springfield"] == {"US-IL", "US-MA"}


def test_city_items_and_g1(tmp_path):
    write_cities(tmp_path)
    lx = build_lexicon([AliasRow("Lebanon", "LB", "name", "en", "t"), *code_rows()], Overrides())
    gen = CityNames(tmp_path, alternates=3)
    items, dropped = generate_items(gen, gen.bases(), set(), seed=1, checker=CollisionChecker(lx))
    got = {i.text: i.targets for i in items}
    assert got["Wien"] == ("AT",) and got["Paris"] == ("FR", "US-TX")
    assert "Lebanon" not in got  # a country name wins: exact collision dropped by G1
    assert gen.near_check is False


def test_filters_population_and_lowercase_romanisations(tmp_path):
    rows = [
        [
            "1",
            "Munich",
            "Munich",
            "Muenchen,München,mwnykh,Monaco di Baviera",
            "0",
            "0",
            "P",
            "PPLA",
            "DE",
            "",
            "02",
            "",
            "",
            "",
            "1500000",
        ],
        [
            "2",
            "Tinytown",
            "Tinytown",
            "",
            "0",
            "0",
            "P",
            "PPL",
            "DE",
            "",
            "02",
            "",
            "",
            "",
            "20000",
        ],
    ]
    (tmp_path / "cities15000.txt").write_text(
        "\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8"
    )
    from countrycode.synth.city import _names

    gen = CityNames(tmp_path, alternates=5, min_population=50_000)
    assert [b.alias for b in gen.bases()] == ["__city_1"]
    assert "mwnykh" not in _names(rows[0]) and "München" in _names(rows[0])
