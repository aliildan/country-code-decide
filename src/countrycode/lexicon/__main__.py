"""python -m countrycode.lexicon build | ambiguous [--top N]"""

from __future__ import annotations

import argparse
import json

from countrycode.lexicon.build import LEXICON_PATH, STATS_PATH, Lexicon, build_lexicon, code_rows
from countrycode.lexicon.cldr import cldr_version, load_cldr
from countrycode.lexicon.types import Overrides
from countrycode.lexicon.wikidata import OVERRIDES_PATH, load_wikidata
from countrycode.text import compact_key


def build() -> None:
    ov = Overrides.load(OVERRIDES_PATH)
    rows = [*load_cldr(), *load_wikidata(overrides=ov), *code_rows()]
    lx = build_lexicon(rows, ov, meta={"cldr": cldr_version()})
    lx.save(LEXICON_PATH)
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(lx.stats(), indent=2, sort_keys=True) + "\n")
    print(json.dumps(lx.stats(), sort_keys=True))


def ambiguous(top: int) -> None:
    lx = Lexicon.load(LEXICON_PATH)
    amb = lx.ambiguous()
    print(f"{len(amb)} ambiguous keys of {lx.keys()}")
    by_key: dict[str, list[str]] = {}
    for r in lx.rows:
        k = compact_key(r.alias)
        if k in amb:
            by_key.setdefault(k, []).append(f"{r.target}:{r.source}/{r.lang}")
    ranked = sorted(amb, key=lambda k: (-len(by_key[k]), k))[:top]
    for k in ranked:
        print(f"{k}\t{sorted(amb[k])}\t{len(by_key[k])} rows\t{sorted(set(by_key[k]))[:8]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    a = sub.add_parser("ambiguous")
    a.add_argument("--top", type=int, default=200)
    args = ap.parse_args()
    if args.cmd == "build":
        build()
    else:
        ambiguous(args.top)


if __name__ == "__main__":
    main()
