from countrycode.lexicon.build import build_lexicon, code_rows
from countrycode.lexicon.types import Overrides
from countrycode.synth.caprov import CanadianProvinceCodes
from countrycode.synth.framework import generate_items
from countrycode.synth.gates import CollisionChecker


def test_province_codes_and_abbreviations_are_canada_and_skip_the_lexicon_check():
    lx = build_lexicon(code_rows(), Overrides())  # "NL" is the Netherlands' code
    gen = CanadianProvinceCodes()
    chk = CollisionChecker(lx, margin=10, containment=True, reject_none_mix=True)
    items, _ = generate_items(gen, gen.bases(), set(), seed=1, checker=chk)
    got = {i.text: i.targets for i in items}
    assert got["NL"] == ("CA",) and got["ON"] == ("CA",) and got["P.E.I."] == ("CA",)
    assert got["AB, CAN"] == ("CA",) and got["Ont."] == ("CA",)
    assert len(got) >= 13 * 3 and gen.checker_exempt is True
