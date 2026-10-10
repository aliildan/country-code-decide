from countrycode.text import compact_key, match_key


def test_latin_folding():
    assert match_key("Österreich") == "osterreich"
    assert match_key("Østerrike") == "osterrike"
    assert match_key("Straße") == "strasse"
    assert match_key("Türkiye") == "turkiye"
    assert match_key("Łotwa") == "lotwa"


def test_spacing_and_punctuation():
    assert match_key("  Untied   Kingdom ") == "untied kingdom"
    assert match_key("U.S.A.") == "u s a" and compact_key("U.S.A.") == "usa"
    assert match_key("Korea, Republic of") == "korea republic of"


def test_greek_cyrillic():
    assert match_key("Ελλάδα") == "ελλαδα"
    assert match_key("НОРВЕГИЯ") == "норвегия"


def test_non_latin_marks_kept():
    for s in ("भारत", "ประเทศไทย", "ノルウェー", "مِصر"):
        assert match_key(s) == s


def test_blank():
    assert match_key("  -- ") == "" and compact_key("") == ""


def test_flag_emoji():
    assert match_key("🇦🇹") == "at" and compact_key("🇽🇰 Kosovo") == "xkkosovo"
