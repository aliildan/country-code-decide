from countrycode import labels as L


def test_country_count_and_xk():
    cs = L.countries()
    assert len(cs) == 250 and "XK" in cs and "XK" in L.NON_ISO
    assert all(len(c) == 2 and c.isupper() for c in cs)
    # NO/NA: YAML-1.1 traps (unquoted NO parses as False)
    assert {"PR", "VI", "GU", "AS", "MP", "UM", "AT", "TR", "NA", "NO"} <= set(cs)
    assert not {"UK", "EU", "AN", "YU", "ZZ"} & set(cs)


def test_us_states():
    ss = L.us_states()
    assert len(ss) == 51 and "US-DC" in ss and "US-CA" in ss and "US-PR" not in ss
    assert L.usps("US-CA") == "CA"


def test_targets():
    assert len(L.targets()) == 302 and L.NONE in L.targets()
    assert L.country_of("US-CA") == "US" and L.country_of("AT") == "AT"
    assert L.country_of("NONE") == "NONE"
    assert L.is_target("US-NY") and not L.is_target("UK")


def test_names():
    assert L.country_name("AT") == "Austria"


def test_reserved_maps():
    assert L.RESERVED_TEXT_CODES["UK"] == "GB" and L.RESERVED_TEXT_CODES["EL"] == "GR"
    assert {"EU", "EZ", "UN", "ZZ"} == set(L.NONE_TEXT_CODES)
