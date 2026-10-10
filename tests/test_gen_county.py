from countrycode.lexicon.build import build_lexicon, code_rows
from countrycode.lexicon.types import AliasRow, Overrides
from countrycode.synth.county import CountyNames, county_index
from countrycode.synth.framework import generate_items
from countrycode.synth.gates import CollisionChecker

ROWS = [
    ("US.CO.001", "Adams County"),
    ("US.IL.001", "Adams County"),
    ("US.CO.123", "Weld County"),
    ("US.LA.033", "East Baton Rouge Parish"),
    ("US.OR.067", "Washington County"),
    ("US.XX.001", "Nowhere County"),  # not a target state: ignored
    ("CA.08.3501", "Ottawa"),  # not US: ignored
]


def write_admin2(tmp_path):
    lines = [f"{code}\t{name}\t{name}\t1" for code, name in ROWS]
    (tmp_path / "admin2Codes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_county_index_sets_full_and_bare_names(tmp_path):
    write_admin2(tmp_path)
    idx = county_index(tmp_path)
    assert idx["adamscounty"] == {"US-CO", "US-IL"} and idx["adams"] == {"US-CO", "US-IL"}
    assert idx["weldcounty"] == {"US-CO"} and idx["eastbatonrouge"] == {"US-LA"}
    assert "nowherecounty" not in idx and "ottawa" not in idx


def test_county_items_and_g1(tmp_path):
    write_admin2(tmp_path)
    lx = build_lexicon(
        [AliasRow("Washington", "US-WA", "name", "en", "t"), *code_rows()], Overrides()
    )
    gen = CountyNames(tmp_path)
    items, dropped = generate_items(gen, gen.bases(), set(), seed=1, checker=CollisionChecker(lx))
    got = {i.text: i.targets for i in items}
    assert got["Weld County"] == ("US-CO",) and got["WELD COUNTY"] == ("US-CO",)
    assert got["Adams County"] == ("US-CO", "US-IL") and got["Adams"] == ("US-CO", "US-IL")
    assert "Washington" not in got  # a state name wins: exact collision dropped by G1
    assert got["Washington County"] == ("US-OR",)
    assert gen.near_check is False


def test_city_and_county_share_one_place_index(tmp_path):
    from countrycode.synth.city import CityNames

    write_admin2(tmp_path)
    with (tmp_path / "admin2Codes.txt").open("a", encoding="utf-8") as f:
        f.write("US.AL.061\tGeneva County\tGeneva County\t1\n")
    city = ["1", "Geneva", "Geneva", "", "0", "0", "P", "PPL", "CH", "", "GE", "", "", "", "200000"]
    city2 = ["2", "Geneva", "Geneva", "", "0", "0", "P", "PPL", "US", "", "IL", "", "", "", "60000"]
    (tmp_path / "cities15000.txt").write_text(
        "\n".join("\t".join(r) for r in (city, city2)) + "\n", encoding="utf-8"
    )
    lx = build_lexicon(code_rows(), Overrides())
    want = ("CH", "US-AL", "US-IL")
    for gen in (CityNames(tmp_path), CountyNames(tmp_path)):
        items, _ = generate_items(gen, gen.bases(), set(), seed=1, checker=CollisionChecker(lx))
        assert {i.text: i.targets for i in items}["Geneva"] == want
