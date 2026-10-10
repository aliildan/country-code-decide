"""Keyboard typo generator `keyboard@1` (plan-03 Task 3).

Edits happen on real keyboard layouts chosen by the alias language. The number of edits and the
mix of operations follow the real dev typo profile (G2, `configs/synth/keyboard.yaml`): writers
mostly add letters (doubling, a neighbouring key), sometimes drop or swap one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from countrycode.synth.framework import (
    MIN_NOISE_KEY,
    PRIORITY_LANGS,
    BaseAlias,
    BaseInfo,
    is_latin,
    recoverable,
    seedable,
)
from countrycode.text import compact_key

CONFIG_PATH = Path("configs/synth/keyboard.yaml")

LAYOUTS: dict[str, tuple[str, ...]] = {
    "us": ("qwertyuiop", "asdfghjkl", "zxcvbnm"),
    "de": ("qwertzuiopü", "asdfghjklöä", "yxcvbnm"),
    "fr": ("azertyuiop", "qsdfghjklm", "wxcvbn"),
    "nordic": ("qwertyuiopå", "asdfghjkløæ", "zxcvbnm"),
    "sv": ("qwertyuiopå", "asdfghjklöä", "zxcvbnm"),
    "tr": ("qwertyuıopğü", "asdfghjklşi", "zxcvbnmöç"),
}
LANG_LAYOUT = {
    "de": "de", "fr": "fr", "nb": "nordic", "nn": "nordic", "no": "nordic", "da": "nordic",
    "sv": "sv", "fi": "sv", "et": "sv", "tr": "tr",
}  # fmt: skip


def neighbours(rows: tuple[str, ...]) -> dict[str, set[str]]:
    """Physical neighbours: same row ±1, row above (same/right), row below (left/same)."""
    out: dict[str, set[str]] = {}
    for r, row in enumerate(rows):
        for i, ch in enumerate(row):
            near = set()
            for rr, offs in ((r, (-1, 1)), (r - 1, (0, 1)), (r + 1, (-1, 0))):
                if 0 <= rr < len(rows):
                    for o in offs:
                        j = i + o
                        if 0 <= j < len(rows[rr]):
                            near.add(rows[rr][j])
            near.discard(ch)
            out[ch] = near
    return out


_NEIGHBOURS = {name: neighbours(rows) for name, rows in LAYOUTS.items()}


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        "edits": {"1": 0.56, "2": 0.36, "3": 0.08},
        "ops": {"insert": 0.64, "delete": 0.175, "replace": 0.167, "transpose": 0.018},
        "insert_double_share": 0.5,
        "min_letters": 4,
    }


def _match_case(new: str, like: str) -> str:
    return new.upper() if like.isupper() else new


class KeyboardTypos:
    name = "keyboard"
    version = "1"

    MAX_EDIT_RATIO = 0.2  # v2: at most one edit per five letters
    UPPER_SHARE = 0.3  # v2: registers write in capitals ("UNITED KINGDON")
    SHIFT_SHARE = 0.05  # v2: shift held one letter too long ("CAlifornia")

    def __init__(
        self,
        per_alias: int = 3,
        config: dict | None = None,
        info: dict[str, BaseInfo] | None = None,
    ) -> None:
        cfg = config or _load_config()
        self.per_alias = per_alias
        self.edits = cfg["edits"]
        self.ops = cfg["ops"]
        self.double_share = float(cfg["insert_double_share"])
        self.min_letters = int(cfg["min_letters"])
        self.config = {**cfg, "per_alias": per_alias}
        self.info = info  # v2 (plan-08): typed safe names only, recoverable variants only
        if info is not None:
            self.version = "2"
            self.config |= {"bases": "typed_safe", "max_edit_ratio": self.MAX_EDIT_RATIO,
                            "min_key": MIN_NOISE_KEY, "upper_share": self.UPPER_SHARE,
                            "shift_share": self.SHIFT_SHARE}  # fmt: skip

    def _one_edit(self, s: list[str], nb: dict[str, set[str]], rng: np.random.Generator) -> None:
        letters = [i for i, c in enumerate(s) if c.isalpha()]
        if not letters:
            return
        op = rng.choice(
            list(self.ops), p=np.array(list(self.ops.values())) / sum(self.ops.values())
        )
        i = int(rng.choice(letters))
        c = s[i]
        near = sorted(nb.get(c.lower(), ()))
        if op == "insert":
            if rng.random() < self.double_share or not near:
                s.insert(i, c)  # doubled letter
            else:
                s.insert(i + int(rng.integers(0, 2)), _match_case(str(rng.choice(near)), c))
        elif op == "delete" and len(letters) > 3:
            del s[i]
        elif op == "replace" and near:
            s[i] = _match_case(str(rng.choice(near)), c)
        elif op == "transpose" and i + 1 < len(s) and s[i + 1].isalpha() and s[i + 1] != c:
            s[i], s[i + 1] = s[i + 1], s[i]
        else:  # operation impossible here: fall back to a doubled letter
            s.insert(i, c)

    def _case(self, v: str, rng: np.random.Generator) -> str:
        r = rng.random()
        if r < self.UPPER_SHARE:
            return v.upper()
        if r < self.UPPER_SHARE + self.SHIFT_SHARE and v[:2].isalpha() and v[0].isupper():
            return v[:2].upper() + v[2:]
        return v

    def generate(self, row: BaseAlias, rng: np.random.Generator) -> list[str]:
        if (
            row.kind == "code"
            or sum(c.isalpha() for c in row.alias) < self.min_letters
            or not is_latin(row.alias)
        ):
            return []
        if self.info is not None and not (
            seedable(row, self.info) and len(compact_key(row.alias)) >= MIN_NOISE_KEY
        ):
            return []
        per_alias = self.per_alias if row.lang.split("_")[0] in PRIORITY_LANGS else 1
        if self.info is not None:
            per_alias = self.per_alias
        nb = _NEIGHBOURS[LANG_LAYOUT.get(row.lang.split("_")[0], "us")]
        out: list[str] = []
        keys = list(self.edits)
        probs = np.array([self.edits[k] for k in keys], dtype=float)
        for _ in range(per_alias * (6 if self.info is not None else 3)):
            if len(out) >= per_alias:
                break
            n = int(rng.choice(keys, p=probs / probs.sum()))
            s = list(row.alias)
            for _ in range(n):
                self._one_edit(s, nb, rng)
            v = "".join(s)
            if self.info is not None:
                v = self._case(v, rng)
            if self.info is not None and not recoverable(
                row.alias, v, self.MAX_EDIT_RATIO, self.info[row.alias].core
            ):
                continue
            if v != row.alias and v not in out:
                out.append(v)
        return out
