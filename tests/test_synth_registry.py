from countrycode.lexicon.build import build_lexicon, code_rows
from countrycode.lexicon.types import AliasRow, Overrides
from countrycode.synth.__main__ import make_generators, release_checker

LX = build_lexicon(
    [AliasRow("Austria", "AT", "name", "en", "cldr"), AliasRow("Kenya", "KE", "name", "en", "cldr"),
     *code_rows()],
    Overrides(),
)  # fmt: skip


def test_new_releases_use_the_repaired_versions():
    gens = make_generators(["keyboard", "format", "translit", "pairs", "affix"], LX)
    assert [f"{g.name}@{g.version}" for g in gens] == [
        "keyboard@2", "format@4", "translit@4", "pairs@3", "affix@2",
    ]  # fmt: skip


def test_release_checker_is_strict():
    chk = release_checker(LX)
    assert (chk.margin, chk.containment, chk.reject_none_mix) == (10, True, True)


def test_audited_releases_keep_the_checker_they_were_released_with():
    from countrycode.synth.__main__ import checker_groups
    from countrycode.synth.formats import JunkStrings

    gens = [JunkStrings(version="1"), *make_generators(["junk", "keyboard"], LX)]
    groups = checker_groups(gens, LX)
    assert [([f"{g.name}@{g.version}" for g in gs], c.containment) for gs, c in groups] == [
        (["junk@1"], False), (["junk@2", "keyboard@2"], True),
    ]  # fmt: skip
