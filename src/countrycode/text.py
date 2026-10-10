"""Matching keys for country strings (plan-01 Task 4).

`match_key` is for comparison only — model inputs keep the original text. It folds case, Latin /
Greek / Cyrillic diacritics and special letters, punctuation and spacing, turns flag emoji into
their two letters, and leaves the combining marks of other scripts (Devanagari, Thai, Arabic, …)
untouched, because there they are part of the letters.
"""

from __future__ import annotations

import unicodedata

_FOLD = str.maketrans(
    {"ø": "o", "æ": "ae", "œ": "oe", "ß": "ss", "ł": "l", "đ": "d", "ð": "d", "þ": "th", "ı": "i"}
)
_REGIONAL_A = 0x1F1E6  # REGIONAL INDICATOR SYMBOL LETTER A
_FOLDABLE_SCRIPTS = ("LATIN", "GREEK", "CYRILLIC")


def _flags_to_letters(s: str) -> str:
    out = []
    for ch in s:
        cp = ord(ch)
        if _REGIONAL_A <= cp <= _REGIONAL_A + 25:
            out.append(chr(ord("a") + cp - _REGIONAL_A))
        else:
            out.append(ch)
    return "".join(out)


def _foldable(ch: str) -> bool:
    try:
        return unicodedata.name(ch).startswith(_FOLDABLE_SCRIPTS)
    except ValueError:
        return False


def _strip_marks(s: str) -> str:
    out: list[str] = []
    base_foldable = False
    for ch in unicodedata.normalize("NFKD", s):
        if unicodedata.category(ch) == "Mn":
            if base_foldable:
                continue
        else:
            base_foldable = _foldable(ch)
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def match_key(s: str) -> str:
    s = _flags_to_letters(unicodedata.normalize("NFKC", s)).casefold().translate(_FOLD)
    s = _strip_marks(s)
    chars = [" " if unicodedata.category(ch)[0] in "PSZC" else ch for ch in s]
    return " ".join("".join(chars).split())


def compact_key(s: str) -> str:
    return match_key(s).replace(" ", "")
