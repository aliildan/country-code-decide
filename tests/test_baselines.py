import numpy as np

from countrycode.evaluation.baselines import (
    Coco,
    CountryNames,
    E5Knn,
    LexiconExact,
    LexiconFuzzy,
    OnsStyle,
    PyCountry,
)
from countrycode.lexicon.build import build_lexicon, code_rows
from countrycode.lexicon.types import AliasRow, Overrides
from countrycode.real.gold import GoldItem


def R(a, t):
    return AliasRow(a, t, "name", "en", "t")


LX = build_lexicon(
    [R("Delaware", "US-DE"), R("Austria", "AT"), R("Australia", "AU"), *code_rows()], Overrides()
)


def g(text):
    return GoldItem("i", "s", "f", "country", "US", text, 1, "dev", ("AT",), "x", "y")


def test_lexicon_exact():
    p = LexiconExact(LX).predict([g("austria"), g("CA"), g("Autria")])
    assert (p[0].target, p[0].confidence) == ("AT", 1.0)
    assert p[1].target is None and set(p[1].candidates) == {"CA", "US-CA"}
    assert p[2].target is None


def test_lexicon_fuzzy():
    p = LexiconFuzzy(LX).predict([g("Deleware"), g("zzzzzz")])
    assert p[0].target == "US-DE" and 0.8 <= p[0].confidence < 1.0
    assert p[1].target is None


def test_ons_style_collapses_double_letters():
    p = OnsStyle(LX).predict([g("Austrriaa"), g("Delawaree")])
    assert p[0].target == "AT" and p[1].target == "US-DE"


def test_library_baselines_never_raise():
    items = [g("Österreich"), g("n/a"), g("Untied Kingdom"), g("")]
    for b in (CountryNames(), PyCountry(), Coco()):
        preds = b.predict(items)
        assert len(preds) == 4 and preds[1].target is None and preds[3].target is None
    assert CountryNames().predict([g("Österreich")])[0].target == "AT"


def test_e5_knn_with_fake_embedder():
    vocab = {"austria": [1.0, 0.0], "australia": [0.0, 1.0], "delaware": [0.7, 0.7]}

    def embed(texts):
        out = []
        for t in texts:
            k = t.lower().removeprefix("query: ")
            default = [0.95, 0.31] if k.startswith("austr") else [-1.0, 0.0]
            out.append(vocab.get(k, default))
        v = np.array(out, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    p = E5Knn(LX, embed_fn=embed, cache=None).predict([g("Austria"), g("Austrija")])
    assert p[0].target == "AT" and p[1].target == "AT" and p[1].confidence < 1.0
