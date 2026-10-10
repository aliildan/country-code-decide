"""Joint distribution of the two heads, numpy only (serving must not import torch).

p(US-CA) = p(US)·p(US-CA | US), p(US) = p(US)·p(NO_STATE); every other target is the country
head's probability.
"""

from __future__ import annotations

import numpy as np

from countrycode import labels
from countrycode.train.dataset import COUNTRY_CLASSES, NO_STATE, STATE_CLASSES

_TARGETS = labels.targets()
_C_INDEX = {c: i for i, c in enumerate(COUNTRY_CLASSES)}
_S_INDEX = {s: i for i, s in enumerate(STATE_CLASSES)}


def combine(country_probs: np.ndarray, state_probs: np.ndarray) -> np.ndarray:
    """(n, 251), (n, 52) → (n, 302) over labels.targets()."""
    n = country_probs.shape[0]
    out = np.zeros((n, len(_TARGETS)))
    p_us = country_probs[:, _C_INDEX["US"]]
    for j, t in enumerate(_TARGETS):
        if t.startswith("US-"):
            out[:, j] = p_us * state_probs[:, _S_INDEX[t]]
        elif t == "US":
            out[:, j] = p_us * state_probs[:, _S_INDEX[NO_STATE]]
        else:
            out[:, j] = country_probs[:, _C_INDEX[t]]
    return out
