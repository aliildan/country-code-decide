"""G2 realism: edit profiles of real noisy strings vs synthetic variants (CLAUDE.md §3).

A pair is (noisy, canonical) in compact-key form; operations describe what the writer did to the
canonical name (insert = extra letter, delete = dropped letter, replace, transpose). Real pairs
come from real *dev* strings only (never calib/test, G4).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from rapidfuzz import fuzz, process
from rapidfuzz.distance import OSA, Levenshtein

from countrycode.lexicon.build import Lexicon
from countrycode.text import compact_key

OPS = ("insert", "delete", "replace", "transpose")


def _ops(noisy: str, canonical: str) -> tuple[Counter, list[float], list[tuple[str, str]]]:
    ops = Levenshtein.editops(canonical, noisy)
    counts = Counter(op.tag for op in ops)
    t = Levenshtein.distance(canonical, noisy) - OSA.distance(canonical, noisy)
    for _ in range(t):  # a swap appears as insert+delete or two replaces in Levenshtein
        if counts["insert"] and counts["delete"]:
            counts["insert"] -= 1
            counts["delete"] -= 1
        elif counts["replace"] >= 2:
            counts["replace"] -= 2
        counts["transpose"] += 1
    n = max(len(canonical), 1)
    positions = [op.src_pos / n for op in ops]
    conf = [(canonical[op.src_pos], noisy[op.dest_pos]) for op in ops if op.tag == "replace"]
    return +counts, positions, conf


def edit_profile(pairs: Iterable[tuple[str, str]]) -> dict:
    op_counts: Counter = Counter()
    edits: Counter = Counter()
    pos: Counter = Counter()
    confusions: Counter = Counter()
    n = 0
    for noisy, canonical in pairs:
        c, positions, conf = _ops(noisy, canonical)
        k = sum(c.values())
        if k == 0:
            continue
        n += 1
        op_counts.update(c)
        edits["3+" if k >= 3 else str(k)] += 1
        for p in positions:
            pos["start" if p < 0.2 else "end" if p > 0.8 else "middle"] += 1
        confusions.update(conf)
    tot_ops = sum(op_counts.values()) or 1
    tot_pos = sum(pos.values()) or 1
    return {
        "n_pairs": n,
        "op_share": {o: op_counts[o] / tot_ops for o in OPS},
        "edits_share": {k: edits[k] / max(n, 1) for k in ("1", "2", "3+")},
        "position_share": {k: pos[k] / tot_pos for k in ("start", "middle", "end")},
        "top_confusions": [f"{a}>{b}:{c}" for (a, b), c in confusions.most_common(20)],
    }


def js_divergence(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Jensen–Shannon divergence (base 2, in [0, 1]) of two discrete distributions."""
    keys = set(p) | set(q)
    sp, sq = sum(p.values()) or 1.0, sum(q.values()) or 1.0
    pp = {k: p.get(k, 0.0) / sp for k in keys}
    qq = {k: q.get(k, 0.0) / sq for k in keys}
    m = {k: (pp[k] + qq[k]) / 2 for k in keys}

    def kl(a: Mapping[str, float]) -> float:
        return sum(a[k] * math.log2(a[k] / m[k]) for k in keys if a[k] > 0)

    return 0.5 * kl(pp) + 0.5 * kl(qq)


def real_noisy_pairs(
    texts_and_targets: Sequence[tuple[str, tuple[str, ...]]],
    lexicon: Lexicon,
    min_ratio: float = 70.0,
    max_edits: int = 3,
    max_len_diff: int = 2,
) -> list[tuple[str, str]]:
    """(noisy, nearest canonical alias of the gold target) for real strings that are not exact
    lexicon keys and look like a misspelling: ratio >= min_ratio, at most `max_edits` OSA edits
    and a length difference of at most `max_len_diff` (extra words such as "AICHI JAPAN" are
    format noise, not typing noise)."""
    by_target: dict[str, list[str]] = {}
    for key in lexicon.index_keys():
        targets = lexicon.lookup(key)
        if len(targets) == 1:
            by_target.setdefault(next(iter(targets)), []).append(key)
    out = []
    for text, targets in texts_and_targets:
        key = compact_key(text)
        if len(targets) != 1 or not key or lexicon.lookup(key) or targets[0] not in by_target:
            continue
        best = process.extractOne(key, by_target[targets[0]], scorer=fuzz.ratio)
        if (
            best
            and best[1] >= min_ratio
            and OSA.distance(best[0], key) <= max_edits
            and abs(len(best[0]) - len(key)) <= max_len_diff
        ):
            out.append((key, best[0]))
    return out


def divergence_report(real: dict, synth: dict) -> dict:
    return {
        "js_op_share": js_divergence(real["op_share"], synth["op_share"]),
        "js_edits_share": js_divergence(real["edits_share"], synth["edits_share"]),
        "js_position_share": js_divergence(real["position_share"], synth["position_share"]),
    }
