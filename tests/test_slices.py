from countrycode.evaluation.slices import noise_kind
from countrycode.lexicon.build import build_lexicon, code_rows
from countrycode.lexicon.types import AliasRow, Overrides

LX = build_lexicon(
    [AliasRow("Germany", "DE", "name", "en", "t"), AliasRow("Norway", "NO", "name", "en", "t"),
     *code_rows()],
    Overrides(),
)  # fmt: skip


def test_noise_kinds():
    assert noise_kind("Germany", ("DE",), LX) == "exact"
    assert noise_kind("Germny", ("DE",), LX) == "near"
    assert noise_kind("06830", ("US-CT",), LX) == "digits"
    assert noise_kind("Норвегия", ("NO",), LX) == "non_latin"
    assert noise_kind("xqzt", ("NONE",), LX) == "other"


def test_noise_kind_precedence():
    assert noise_kind("Germany 2", ("DE",), LX) == "digits"  # not an exact key → digits first
    assert noise_kind("Norwey", ("DE",), LX) == "other"  # near another target is not "near"
