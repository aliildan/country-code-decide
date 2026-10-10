"""Quality gates for synthetic data (CLAUDE.md §3). G1 (label correctness) automatic check:
a variant must never carry a code when it is, or is at least as close to, another target's name.

The strict settings (plan-08, after the owner's audit of the v1 releases) add a margin over the
nearest other code, a whole-word containment check and rejection of NONE mixed with codes.
"""

from __future__ import annotations

import re

from anyascii import anyascii
from rapidfuzz import fuzz, process

from countrycode import labels
from countrycode.lexicon.build import Lexicon
from countrycode.text import compact_key

MAX_GRAM = 4
MIN_GRAM_KEY = 4  # shorter tokens are codes or compass letters ("NE", "SE"), not names


class CollisionChecker:
    """Called as checker(text, own_targets, source_alias) -> (keep, targets, reason).

    - exact key hit with no target in common → drop ("exact_other");
    - exact key hit sharing a target → keep with the lexicon's set (consistent with the fast
      path; "relabel_exact" when it differs from the base set);
    - otherwise: if any lexicon key of a disjoint target set is at least as similar (Levenshtein
      ratio on compact keys) as the closest name of the own targets (or the source alias and its
      romanisation) → drop ("near_other"); ties drop too. With `margin`, a disjoint key within
      `margin` points of the own score drops too;
    - `containment`: a word n-gram whose lexicon set is not within the own targets and that lies
      inside no longer own-target n-gram → drop ("contains_other": "Nigerian Republic" labelled
      NE, "Czechoslovakia (former)" labelled CZ);
    - `reject_none_mix`: a final set holding NONE and a code → drop ("none_mix").
    """

    def __init__(
        self,
        lexicon: Lexicon,
        margin: float = 0.0,
        containment: bool = False,
        reject_none_mix: bool = False,
    ) -> None:
        self.lexicon = lexicon
        self.margin = margin
        self.containment = containment
        self.reject_none_mix = reject_none_mix
        self.keys = lexicon.index_keys()
        self.keys_by_target: dict[str, list[str]] = {}
        for k in self.keys:
            for t in lexicon.lookup(k):
                self.keys_by_target.setdefault(t, []).append(k)

    def _own_score(self, key: str, own: tuple[str, ...], source_alias: str) -> float:
        best = max(
            fuzz.ratio(key, compact_key(source_alias)),
            fuzz.ratio(key, compact_key(anyascii(source_alias))),
        )
        for t in own:
            hit = process.extractOne(key, self.keys_by_target.get(t, []), scorer=fuzz.ratio)
            if hit:
                best = max(best, hit[1])
        return best

    def _contains_other(self, text: str, own_set: set[str]) -> bool:
        words = [w for w in re.split(r"[^\w]+", text) if w]
        own_spans: list[tuple[int, int]] = []
        other_spans: list[tuple[int, int]] = []
        for n in range(1, MAX_GRAM + 1):
            for i in range(len(words) - n + 1):
                key = compact_key(" ".join(words[i : i + n]))
                if len(key) < MIN_GRAM_KEY:
                    continue
                hit = self.lexicon.lookup(key)
                if hit and hit <= own_set:
                    own_spans.append((i, i + n))
                elif hit:  # another code, or a wider set ("Czechoslovakia" for CZ alone)
                    other_spans.append((i, i + n))
        return any(
            not any(a <= i and j <= b and b - a > j - i for a, b in own_spans)
            for i, j in other_spans
        )

    def __call__(
        self,
        text: str,
        own: tuple[str, ...],
        source_alias: str,
        exact_only: bool = False,
        containment: bool | None = None,
    ) -> tuple[bool, tuple[str, ...], str]:
        contain = self.containment if containment is None else containment
        keep, targets, reason = self._check(text, own, source_alias, exact_only, contain)
        if keep and self.reject_none_mix and labels.NONE in targets and len(targets) > 1:
            return False, own, "none_mix"
        return keep, targets, reason

    def _check(
        self, text: str, own: tuple[str, ...], source_alias: str, exact_only: bool, contain: bool
    ) -> tuple[bool, tuple[str, ...], str]:
        key = compact_key(text)
        own_set = set(own)
        hit = self.lexicon.lookup(key)
        if hit:
            if not hit & own_set:
                return False, own, "exact_other"
            ts = tuple(sorted(hit))
            return True, ts, "ok" if ts == tuple(sorted(own)) else "relabel_exact"
        if contain and self._contains_other(text, own_set):
            return False, own, "contains_other"
        if exact_only:
            return True, tuple(sorted(own)), "ok"
        own_score = self._own_score(key, own, source_alias)
        bar = max(0.0, own_score - self.margin - 1e-9)
        for other, score, _ in process.extract(
            key, self.keys, scorer=fuzz.ratio, limit=8, score_cutoff=bar
        ):
            if score >= own_score - self.margin and not self.lexicon.lookup(other) & own_set:
                return False, own, "near_other"
        return True, tuple(sorted(own)), "ok"
