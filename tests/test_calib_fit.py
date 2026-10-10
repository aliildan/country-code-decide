import numpy as np

from countrycode import labels
from countrycode.calib.fit import gold_indices, head_labels
from countrycode.real.gold import GoldItem

T = list(labels.targets())


def g(targets, granularity="full"):
    return GoldItem(
        "i", "s", "f", "country", "US", "x", 1, "calib", tuple(targets), "x", "y", granularity
    )


def test_gold_indices_modes():
    row = np.zeros(len(T))
    row[T.index("US-TX")] = 0.7
    row[T.index("US")] = 0.2
    assert gold_indices(g(["US"], "country"), row) == {T.index("US-TX")}  # best acceptable target
    assert gold_indices(g(["CD", "CG"]), row) == {T.index("CD"), T.index("CG")}
    assert gold_indices(g(["AT"]), row) == {T.index("AT")}


def test_head_labels():
    assert head_labels(g(["US-TX"])) == ("US", "US-TX")
    assert head_labels(g(["US"], "country")) == ("US", None)
    assert head_labels(g(["AT"])) == ("AT", None)
    assert head_labels(g(["CD", "CG"])) == (None, None)


def test_fit_domains_skips_small_domains():
    from countrycode.calib.fit import fit_domains

    rng = np.random.default_rng(0)
    n = 40
    c = rng.normal(0, 1, (n, 251))
    s = rng.normal(0, 1, (n, 52))
    gold = [g(["AT"]) for _ in range(n)]
    gold = [GoldItem("i", "us-co" if i < 30 else "gbif", "f", "country", "US", "x", 1, "calib",
                     ("AT",), "x", "y") for i in range(n)]  # fmt: skip
    cs = fit_domains(c, s, gold, min_items=25)
    assert set(cs.policies) == {"register"}
