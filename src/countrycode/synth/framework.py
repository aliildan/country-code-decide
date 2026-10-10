"""Synthetic data framework (plan-03 Task 1; CLAUDE.md §3 gates G4, G5).

A generator turns one base alias (lexicon name + its resolved target set) into variant strings.
`run_release` applies generators to every base alias with a deterministic per-alias seed, removes
variants that equal a real evaluation string (G4), runs the G1 checker when given, de-duplicates,
and writes one JSONL per generator plus a manifest (counts, drop reasons, config hashes).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from anyascii import anyascii
from rapidfuzz.distance import OSA

from countrycode import labels
from countrycode.lexicon.build import Lexicon
from countrycode.text import compact_key

SYNTH_DIR = Path("data/processed/synth")

PRIORITY_LANGS = frozenset({  # languages whose names are typed in European/US registers
    "en", "de", "fr", "es", "it", "pt", "nl", "sv", "nb", "nn", "no", "da", "fi", "et", "tr",
    "pl", "cs", "sk", "sl", "hr", "sr", "bs", "hu", "ro", "bg", "el", "ru", "uk", "lt", "lv",
    "is", "ga", "mt", "ca",
})  # fmt: skip


# One island, two codes: "St. Martin" alone may be either side (owner audit, plan-08).
SHARED_ISLAND = frozenset({"MF", "SX"})

# Languages whose names a register clerk types (plan-08): noise and format variants only start
# from these names; the owner's audit found rare-language noise unrecognisable.
NOISE_LANGS = frozenset({"en", "de", "fr", "es", "it", "pt", "nl", "sv", "nb", "nn", "no", "da"})


def is_latin(s: str) -> bool:
    """Every letter is Latin (Basic Latin to Latin Extended-B, incl. IPA-free diacritics)."""
    return any(c.isalpha() for c in s) and all(ord(c) < 0x250 for c in s if c.isalpha())


@dataclass(frozen=True)
class BaseAlias:
    alias: str
    targets: tuple[str, ...]
    kind: str
    lang: str


@dataclass(frozen=True)
class SynthItem:
    text: str
    targets: tuple[str, ...]
    generator: str
    version: str
    seed: int
    source_alias: str
    config_hash: str

    @property
    def label_source(self) -> str:
        return f"synthetic:{self.generator}@{self.version}"


class Generator(Protocol):
    name: str
    version: str
    config: dict

    def generate(
        self, row: BaseAlias, rng: np.random.Generator
    ) -> Sequence[str | tuple[str, tuple[str, ...]]]: ...


class Checker(Protocol):
    """(text, own targets, source alias, exact_only, containment) -> (keep, targets, reason)."""

    def __call__(
        self,
        text: str,
        own: tuple[str, ...],
        source_alias: str,
        exact_only: bool = False,
        containment: bool | None = None,
    ) -> tuple[bool, tuple[str, ...], str]: ...


@dataclass(frozen=True)
class BaseInfo:
    """What the lexicon rows of one alias string say (languages without region, kinds, targets)."""

    langs: frozenset[str]
    kinds: frozenset[str]
    row_targets: frozenset[str]
    # a CLDR name in NOISE_LANGS, or an alias holding a standard name of its target as whole
    # words ("Republiek Tunesië", "Staat New York"); not nicknames ("Palmetto State")
    anchored: bool = True
    # folded words (>= 4 letters, not shared by 3+ targets) of the standard names of its targets;
    # a noisy variant must keep one of them within one edit
    core: frozenset[str] = frozenset()
    # a NONE region with a word within one edit of a country's core word ("Polynésie" ~ PF)
    country_like: bool = False
    # base languages of the alias's CLDR rows (standard names, any language)
    cldr_langs: frozenset[str] = frozenset()


def _base_lang(lang: str) -> str:
    return lang.split("_")[0].split("-")[0]


GENERIC_SPREAD = 3  # a word in the standard names of 3+ targets ("republic", "islands")
ACRONYM_STOP = frozenset({"of", "the", "and"})  # "UAE", "UK", "DRC" are standard-name initials


def _words(s: str) -> tuple[str, ...]:
    return tuple(w for w in re.split(r"[^\w]+", s.casefold()) if w)


def _holds(words: tuple[str, ...], name: tuple[str, ...]) -> bool:
    n = len(name)
    return n > 0 and any(words[i : i + n] == name for i in range(len(words) - n + 1))


def base_info(lexicon: Lexicon) -> dict[str, BaseInfo]:
    langs: dict[str, set[str]] = {}
    kinds: dict[str, set[str]] = {}
    targets: dict[str, set[str]] = {}
    cldr: set[str] = set()
    core: dict[str, set[tuple[str, ...]]] = {}
    for t in labels.targets():
        if t != labels.NONE:
            name = labels.us_state_name(t) if t.startswith("US-") else labels.country_name(t)
            core.setdefault(t, set()).add(_words(name))
    cldr_langs: dict[str, set[str]] = {}
    for r in lexicon.rows:
        if r.source == "cldr":
            cldr_langs.setdefault(r.alias, set()).add(_base_lang(r.lang))
        langs.setdefault(r.alias, set()).add(_base_lang(r.lang))
        kinds.setdefault(r.alias, set()).add(r.kind)
        targets.setdefault(r.alias, set()).add(r.target)
        if r.source == "cldr" and _base_lang(r.lang) in NOISE_LANGS:
            cldr.add(r.alias)
            if r.target != labels.NONE:  # region names are not standard names of a target
                core.setdefault(r.target, set()).add(_words(r.alias))
    words: dict[str, set[str]] = {}
    for t, names in core.items():
        words[t] = {k for n in names for w in n if len(k := compact_key(w)) >= 4}
    spread: dict[str, int] = {}
    for ws in words.values():
        for x in ws:
            spread[x] = spread.get(x, 0) + 1
    country_words = {x for ws in words.values() for x in ws if spread[x] < GENERIC_SPREAD}
    acronyms: dict[str, set[str]] = {}
    for t, names in core.items():
        for n in names:
            initials = "".join(x[0] for x in n if x not in ACRONYM_STOP)
            if len(initials) >= 2:
                acronyms.setdefault(t, set()).add(initials)
    out = {}
    for a in langs:
        w = _words(a)
        flat = re.sub(r"[.\s]", "", a)
        anchored = (
            a in cldr
            or any(_holds(w, n) for t in targets[a] for n in core.get(t, ()))
            or (flat.isupper() and any(flat.lower() in acronyms.get(t, ()) for t in targets[a]))
        )
        kept = frozenset(
            x for t in targets[a] for x in words.get(t, ()) if spread[x] < GENERIC_SPREAD
        )
        like = targets[a] == {labels.NONE} and any(
            OSA.distance(k, c) <= 1
            for x in w
            if len(k := compact_key(anyascii(x))) >= 4  # romanised: "Афрыка" ~ "Afrika"
            for c in country_words
            if abs(len(c) - len(k)) <= 1
        )
        out[a] = BaseInfo(
            frozenset(langs[a]),
            frozenset(kinds[a]),
            frozenset(targets[a]),
            anchored,
            kept,
            like,
            frozenset(cldr_langs.get(a, ())),
        )
    return out


def unsafe_reason(b: BaseAlias, info: dict[str, BaseInfo]) -> str | None:
    """Why a base alias must not seed generated variants (plan-08), or None. Codes are safe."""
    if b.kind == "code":
        return None
    if labels.NONE in b.targets and len(b.targets) > 1:
        return "none_mix"
    if SHARED_ISLAND & set(b.targets):
        return "shared_island"
    if b.kind == "historical" and len(b.targets) >= 3:
        return "multi_successor"
    i = info.get(b.alias)
    if i is not None and i.country_like:
        return "country_like_region"
    if i is not None and set(b.targets) != i.row_targets:
        return "key_collision"  # another alias shares the key ("burundeză" BI, "burundeza" FI)
    return None


def typed_name(b: BaseAlias, info: dict[str, BaseInfo]) -> bool:
    """A name a register clerk would type: a NOISE_LANGS name; demonyms only in English."""
    i = info.get(b.alias)
    if b.kind == "code" or i is None:
        return False
    if "demonym" in i.kinds:
        return "en" in i.langs
    return bool(i.langs & NOISE_LANGS)


def seedable(b: BaseAlias, info: dict[str, BaseInfo]) -> bool:
    """A base that may seed noisy variants: a typed, safe, anchored name."""
    i = info.get(b.alias)
    return i is not None and i.anchored and typed_name(b, info) and unsafe_reason(b, info) is None


MIN_NOISE_KEY = 6  # shorter names (and codes) get no noise: "FRG" → "frog", "Roum." → "room"


def recoverable(
    source: str, variant: str, max_ratio: float, core: frozenset[str] = frozenset()
) -> bool:
    """A careful reader can still recover the source: enough letters, few edits and, when the
    target has core words, one of them kept within one edit ("Iémen do Sul" → "remendo Sul" is
    lost) (plan-08)."""
    s = compact_key(source)
    if len(s) < MIN_NOISE_KEY or OSA.distance(s, compact_key(variant)) > max_ratio * len(s):
        return False
    if not core:
        return True
    vw = {k for w in _words(variant) if (k := compact_key(w))}
    return any(OSA.distance(w, c) <= 1 for w in vw for c in core)


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def base_aliases(lexicon: Lexicon) -> list[BaseAlias]:
    """One base per distinct alias string, labelled with its key's resolved target set."""
    seen: dict[str, BaseAlias] = {}
    for r in lexicon.rows:
        prev = seen.get(r.alias)
        if prev is not None and not (r.kind == "code" and prev.kind != "code"):
            continue  # first row wins, except that a code row makes the alias a code
        targets = tuple(sorted(lexicon.lookup(r.alias)))
        if targets:
            seen[r.alias] = BaseAlias(r.alias, targets, r.kind, r.lang)
    return sorted(seen.values(), key=lambda b: (b.targets, b.alias))


def _alias_seed(seed: int, gen: str, alias: str) -> int:
    h = hashlib.sha256(f"{seed}\x1f{gen}\x1f{alias}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def generate_items(
    gen: Generator,
    bases: Iterable[BaseAlias],
    real_keys: set[str],
    seed: int,
    checker: Checker | None = None,
) -> tuple[list[SynthItem], Counter]:
    chash = config_hash(gen.config)
    dropped: Counter = Counter()
    out: dict[tuple[str, tuple[str, ...]], SynthItem] = {}
    for b in bases:
        s = _alias_seed(seed, gen.name, b.alias)
        for variant in gen.generate(b, np.random.default_rng(s)):
            text, own = (variant, b.targets) if isinstance(variant, str) else variant
            key = compact_key(text)
            if not key or text == b.alias:
                dropped["empty_or_identity"] += 1
                continue
            if key in real_keys:
                dropped["leak"] += 1
                continue
            targets = own
            if checker is not None and not getattr(gen, "checker_exempt", False):
                exact_only = not getattr(gen, "near_check", True)
                contain = getattr(gen, "containment_check", None)
                if contain is None:
                    keep, targets, reason = checker(text, own, b.alias, exact_only)
                else:  # e.g. geo: a GeoNames place of the own target may hold a country name
                    keep, targets, reason = checker(text, own, b.alias, exact_only, contain)
                if keep and reason == "relabel_exact" and not getattr(gen, "allow_relabel", True):
                    keep = False  # e.g. llm@2: "NK" for KP is {GB, KP} in the lexicon
                if not keep:
                    dropped[reason] += 1
                    continue
            k = (text, targets)
            if k in out:
                dropped["duplicate"] += 1
            if k not in out or b.alias < out[k].source_alias:
                out[k] = SynthItem(text, targets, gen.name, gen.version, s, b.alias, chash)
    return sorted(out.values(), key=lambda i: (i.targets, i.text)), dropped


def write_items(items: Sequence[SynthItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            row = {**asdict(it), "targets": list(it.targets), "label_source": it.label_source}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_synth(path: Path) -> list[SynthItem]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            d.pop("label_source", None)
            d["targets"] = tuple(d["targets"])
            out.append(SynthItem(**d))
    return out


_WORK: tuple = ()


def _work(chunk: list[BaseAlias]) -> tuple[list[SynthItem], Counter]:
    gen, real_keys, seed, checker = _WORK
    return generate_items(gen, chunk, real_keys, seed, checker)


def generate_parallel(
    gen: Generator,
    bases: Sequence[BaseAlias],
    real_keys: set[str],
    seed: int,
    checker: Checker | None,
    workers: int,
) -> tuple[list[SynthItem], Counter]:
    """Same output as `generate_items` (per-alias seeds do not depend on chunking)."""
    if workers <= 1:
        return generate_items(gen, bases, real_keys, seed, checker)
    import multiprocessing as mp

    global _WORK
    _WORK = (gen, real_keys, seed, checker)
    size = max(1, len(bases) // (workers * 8))
    chunks = [list(bases[i : i + size]) for i in range(0, len(bases), size)]
    with mp.get_context("fork").Pool(workers) as pool:
        parts = pool.map(_work, chunks)
    _WORK = ()
    merged: dict[tuple[str, tuple[str, ...]], SynthItem] = {}
    dropped: Counter = Counter()
    for items, d in parts:
        dropped.update(d)
        for it in items:
            k = (it.text, it.targets)
            if k not in merged or it.source_alias < merged[k].source_alias:
                if k in merged:
                    dropped["duplicate"] += 1
                merged[k] = it
            else:
                dropped["duplicate"] += 1
    return sorted(merged.values(), key=lambda i: (i.targets, i.text)), dropped


def run_release(
    generators: Sequence[Generator],
    lexicon: Lexicon,
    real_keys: set[str],
    out_dir: Path = SYNTH_DIR,
    seed: int = 13,
    checker: Checker | None = None,
    bases: Sequence[BaseAlias] | None = None,
    workers: int = 1,
) -> dict:
    default_bases = list(bases) if bases is not None else base_aliases(lexicon)
    mpath = Path(out_dir) / "manifest.json"
    previous = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest: dict = {
        "seed": seed,
        "bases": len(default_bases),
        "generators": previous.get("generators", {}) if previous.get("seed") == seed else {},
    }
    for gen in generators:
        own_bases = gen.bases() if hasattr(gen, "bases") else default_bases  # type: ignore[attr-defined]
        items, dropped = generate_parallel(gen, own_bases, real_keys, seed, checker, workers)
        tag = f"{gen.name}@{gen.version}"
        write_items(items, Path(out_dir) / f"{tag}.jsonl")
        manifest["generators"][tag] = {
            "sha256": hashlib.sha256((Path(out_dir) / f"{tag}.jsonl").read_bytes()).hexdigest(),
            "config": gen.config,
            "config_hash": config_hash(gen.config),
            "items": len(items),
            "dropped": dict(sorted(dropped.items())),
            "targets_covered": len({t for i in items for t in i.targets}),
        }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest
