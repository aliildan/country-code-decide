from countrycode.train.__main__ import seen_counts


def test_seen_counts_by_source():
    keys = {"us-ny": ["delaware", "delawre", "x"], "us-or": ["oregon"]}
    rep = seen_counts(keys, {"delaware", "oregon"})
    assert rep == {"us-ny": {"items": 3, "seen": 1}, "us-or": {"items": 1, "seen": 1}}
