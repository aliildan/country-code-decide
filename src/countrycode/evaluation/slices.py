"""Noise-kind slices of real items (design-02 §2). Heuristic and documented, never a label:

- `exact`: the string's key is a lexicon key (of any target);
- `digits`: contains a digit (postcodes, dates, "2nd floor");
- `non_latin`: has letters outside Latin script;
- `near`: within edit distance 2 (OSA, on compact keys) of a lexicon key of a gold target;
- `other`: everything else (heavier noise, unknown words, non-places).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from rapidfuzz.distance import OSA

from countrycode.lexicon.build import Lexicon
from countrycode.synth.framework import is_latin
from countrycode.text import compact_key

_KEYS: dict[int, dict[str, list[str]]] = {}


def _keys_by_target(lexicon: Lexicon) -> dict[str, list[str]]:
    if id(lexicon) not in _KEYS:
        m: dict[str, list[str]] = defaultdict(list)
        for key in lexicon.index_keys():
            for t in lexicon.lookup(key):
                m[t].append(key)
        _KEYS[id(lexicon)] = dict(m)
    return _KEYS[id(lexicon)]


def noise_kind(text: str, targets: Sequence[str], lexicon: Lexicon) -> str:
    key = compact_key(text)
    if key and lexicon.lookup(text):
        return "exact"
    if any(c.isdigit() for c in text):
        return "digits"
    if any(c.isalpha() for c in text) and not is_latin(text):
        return "non_latin"
    keys = _keys_by_target(lexicon)
    if key and any(
        OSA.distance(key, k, score_cutoff=2) <= 2 for t in targets for k in keys.get(t, ())
    ):
        return "near"
    return "other"
