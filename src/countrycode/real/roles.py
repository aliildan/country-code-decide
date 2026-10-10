"""v1.1 roles of real items (plan-06 Task 4). Splits stay the v1 hash splits; roles reuse them so
no v1 decision is re-used as fresh evidence:

- register sources: dev → train (real noise for training); calib → half select / half calib
  (stable hash); test → test if the label came from the owner (never seen by a v1 decision),
  else test_seen_v1 (auto-labelled items read in the v1 test);
- curated sources (GBIF): dev → select; calib → calib; test → ood (read in v1, reported apart).
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from functools import cache

import yaml

from countrycode.calib.calibration import domain_of
from countrycode.paths import CONFIGS
from countrycode.real.gold import GoldItem
from countrycode.text import compact_key

ROLES = ("train", "select", "calib", "test", "test_seen_v1", "ood", "test_v12")
HELD_OUT_PATH = CONFIGS / "held_out_sources.yaml"


@cache
def held_out() -> dict[str, str]:
    """Source → role for sources held out in full (configs/held_out_sources.yaml)."""
    raw = (
        yaml.safe_load(HELD_OUT_PATH.read_text(encoding="utf-8")) if HELD_OUT_PATH.exists() else {}
    )
    return {src: role for role, sources in (raw or {}).items() for src in sources}


def _half(item: GoldItem) -> str:
    h = hashlib.sha256(f"half\x1f{item.source}\x1f{compact_key(item.text)}".encode()).digest()
    return "select" if h[0] % 2 == 0 else "calib"


def role_of(item: GoldItem) -> str:
    if (role := held_out().get(item.source)) is not None:
        return role
    register = domain_of(item.source) == "register"
    if item.split == "dev":
        return "train" if register else "select"
    if item.split == "calib":
        return _half(item) if register else "calib"
    if register:
        return "test" if item.annotation_mode == "reviewed_proposal" else "test_seen_v1"
    return "ood"


def real_training_items(gold: list[GoldItem]) -> list[GoldItem]:
    """Train-role items whose normalised key appears in no other role (any source)."""
    other = {compact_key(g.text) for g in gold if role_of(g) != "train"}
    return [g for g in gold if role_of(g) == "train" and compact_key(g.text) not in other]


ROLE_SPLIT = {"test": "test_v11"}  # the fresh v1.1 test part is sealed like v1's test


def with_roles(gold: list[GoldItem]) -> list[GoldItem]:
    """Copies whose `split` is the v1.1 role (so the harness can select parts by role)."""
    return [replace(g, split=ROLE_SPLIT.get(r, r)) for g in gold for r in [role_of(g)]]
