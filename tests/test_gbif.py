from countrycode.real.gbif import parse_gbif_tsv

TSV = """# A command used to sort this file:
# (sed -u 3q; ...)
# Note
A\tZZ
AALST BELGIUM\tBE
UNITED KINGDOM\tUK
YUGOSLAVIA\tYU
KOSOVO\tXK
ATLANTIS\tQQ
"""


def test_parse_gbif():
    pairs, dropped = parse_gbif_tsv(TSV)
    d = dict(pairs)
    assert d["A"] == ("NONE",) and d["AALST BELGIUM"] == ("BE",)
    assert d["UNITED KINGDOM"] == ("GB",) and d["KOSOVO"] == ("XK",)
    assert d["YUGOSLAVIA"] == ("BA", "HR", "ME", "MK", "RS", "SI", "XK")  # former code → set
    assert dropped == {"QQ": 1}


def test_international_waters_is_none():
    pairs, _ = parse_gbif_tsv("HIGH SEAS\tXZ\n")
    assert dict(pairs)["HIGH SEAS"] == ("NONE",)
