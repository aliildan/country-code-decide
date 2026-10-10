import numpy as np

from countrycode.lexicon.build import build_lexicon, code_rows
from countrycode.lexicon.types import AliasRow, Overrides
from countrycode.synth.framework import generate_items
from countrycode.synth.gates import CollisionChecker
from countrycode.synth.region import RegionNames

ADMIN1 = [
    ("CA.08", "Ontario"), ("ID.02", "Bali"), ("PT.23", "Azores"), ("BR.07", "Distrito Federal"),
    ("MX.09", "Distrito Federal"), ("VE.25", "Distrito Federal"), ("GH.02", "Central"),
    ("KE.01", "Central"), ("ZM.02", "Central"), ("LK.01", "Central"), ("UY.07", "Florida"),
    ("UA.11", "Crimea"), ("AU.02", "New South Wales"), ("US.TX", "Texas"),
]  # fmt: skip
LX = build_lexicon(
    [AliasRow("Florida", "US-FL", "name", "en", "t"), AliasRow("Wales", "GB", "name", "en", "t"),
     *code_rows()],
    Overrides(),
)  # fmt: skip


def write(tmp_path):
    lines = [f"{c}\t{n}\t{n}\t{i}" for i, (c, n) in enumerate(ADMIN1)]
    lines.append("FR.11\tÎle-de-France\tIle-de-France\t99")
    (tmp_path / "admin1CodesASCII.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / "cities15000.txt").write_text("", encoding="utf-8")


def released(tmp_path):
    write(tmp_path)
    gen = RegionNames(tmp_path)
    chk = CollisionChecker(LX, margin=10, containment=True, reject_none_mix=True)
    items, _ = generate_items(gen, gen.bases(), set(), seed=1, checker=chk)
    return gen, {i.text: i.targets for i in items}


def test_regions_name_their_country_in_any_case(tmp_path):
    gen, got = released(tmp_path)
    assert gen.near_check is False and gen.containment_check is False
    assert got["Ontario"] == ("CA",) and got["ONTARIO"] == ("CA",) and got["Bali"] == ("ID",)
    assert got["New South Wales"] == ("AU",)  # "Wales" inside is not GB
    assert got["Ile-de-France"] == ("FR",) and got["Île-de-France"] == ("FR",)


def test_shared_names_become_sets_up_to_three_and_country_names_win(tmp_path):
    _, got = released(tmp_path)
    assert got["Distrito Federal"] == ("BR", "MX", "VE")
    assert "Central" not in got  # four countries: no information
    assert "Florida" not in got  # a US state's name (G1 exact check)


def test_us_states_and_disputed_regions_are_skipped(tmp_path):
    _, got = released(tmp_path)
    assert "Texas" not in got and "Crimea" not in got
    assert np.random.default_rng(0) is not None
