import re

import numpy as np

from countrycode.lexicon.build import build_lexicon, code_rows
from countrycode.lexicon.types import AliasRow, Overrides
from countrycode.synth.framework import generate_items
from countrycode.synth.gates import CollisionChecker
from countrycode.synth.regjunk import RegisterJunk


def test_register_junk_is_none_and_looks_like_register_noise():
    gen = RegisterJunk(n=600)
    lx = build_lexicon([AliasRow("Summit", "JO", "name", "en", "t"), *code_rows()], Overrides())
    chk = CollisionChecker(lx, margin=10, containment=True, reject_none_mix=True)
    items, _ = generate_items(gen, gen.bases(), set(), seed=2, checker=chk)
    texts = [i.text for i in items]
    assert items and all(i.targets == ("NONE",) for i in items)
    assert any(re.search(r"\b(LLC|L\.L\.C\.|Inc\.?|INC|Corp\.?)\b", t) for t in texts)
    assert any(re.match(r"\d+ \w+", t) for t in texts)  # street line
    assert any(t.upper().startswith("ATTN") for t in texts)
    assert not any("SUMMIT" in t.upper() for t in texts)  # a place name inside → dropped
    a = gen.generate(gen.bases()[0], np.random.default_rng(1))
    assert a == gen.generate(gen.bases()[0], np.random.default_rng(1))
