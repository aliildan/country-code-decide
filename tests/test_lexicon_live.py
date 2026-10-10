import pytest

from countrycode.lexicon.build import LEXICON_PATH, Lexicon

pytestmark = pytest.mark.skipif(not LEXICON_PATH.exists(), reason="lexicon not built")


def test_live_lexicon():
    lx = Lexicon.load(LEXICON_PATH)
    assert len(lx.rows) >= 30_000
    assert lx.lookup("Østerrike") == {"AT"} and lx.lookup("ノルウェー") == {"NO"}
    assert lx.lookup("Kalifornien") == {"US-CA"} and {"GE", "US-GA"} <= lx.lookup("Georgia")


def test_kosovo_alternatives():
    lx = Lexicon.load(LEXICON_PATH)
    for s in ("XK", "XKX", "XKS", "RKS", "KOS", "Kosova", "Republika e Kosovës", "RS-KM", "🇽🇰"):
        assert lx.lookup(s) == {"XK"}, s
    assert lx.lookup("KS") == {"XK", "US-KS"}
