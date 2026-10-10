"""Canadian province and territory codes → CA in a US register's jurisdiction field (plan-09
Task 3): "ON", "B.C.", "P.E.I.", "AB, CAN".

The codes are ISO 3166-2:CA; the abbreviations are the usual Canadian forms. The strings are bound
to that context (`source_contexts` in context_rules.yaml): elsewhere "NL", "PE", "SK", "NU", "YT"
are country codes, so the context-free lexicon check of G1 does not apply (`checker_exempt`); the
owner audits every item with its context shown.
"""

from __future__ import annotations

import numpy as np

from countrycode.synth.framework import BaseAlias

ABBREVIATIONS = {
    "AB": ["Alta."], "BC": ["B.C."], "MB": ["Man."], "NB": ["N.B."], "NL": ["Nfld.", "N.L."],
    "NS": ["N.S."], "NT": ["N.W.T.", "NWT"], "NU": ["Nvt."], "ON": ["Ont."],
    "PE": ["P.E.I.", "PEI"], "QC": ["Que.", "P.Q.", "PQ"], "SK": ["Sask."], "YT": ["Y.T.", "Yuk."],
}  # fmt: skip


class CanadianProvinceCodes:
    name = "caprov"
    version = "1"
    near_check = False
    containment_check = False
    checker_exempt = True  # context-bound labels; the lexicon check is context-free
    config = {"source": "ISO 3166-2:CA codes + usual abbreviations",
              "context": "US register, jurisdiction field"}  # fmt: skip

    def bases(self) -> list[BaseAlias]:
        return [BaseAlias(f"__caprov_{c}", ("CA",), "caprov", "en") for c in sorted(ABBREVIATIONS)]

    def generate(self, row: BaseAlias, rng: np.random.Generator) -> list[str]:
        code = row.alias.removeprefix("__caprov_")
        return list(dict.fromkeys([code, *ABBREVIATIONS[code], f"{code}, CAN", f"{code}, Canada"]))
