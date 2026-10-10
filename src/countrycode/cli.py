"""countrycode CLI: `countrycode [--bundle DIR] [--field F] [--register CC] [--alpha A] TEXT...`

Prints one JSON decision per text. The bundle defaults to $COUNTRYCODE_BUNDLE.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv: list[str] | None = None) -> int:
    from countrycode.infer.pipeline import CountryCoder

    ap = argparse.ArgumentParser(prog="countrycode", description=__doc__)
    ap.add_argument("texts", nargs="+")
    ap.add_argument("--bundle", default=os.environ.get("COUNTRYCODE_BUNDLE"))
    ap.add_argument("--field", default="unknown")
    ap.add_argument("--register", default="unknown")
    ap.add_argument("--alpha", default="0.05", help="0.01, 0.02, 0.05 or 'none' (always answer)")
    args = ap.parse_args(argv)
    if not args.bundle:
        ap.error("no bundle: pass --bundle or set COUNTRYCODE_BUNDLE")
    alpha = None if args.alpha == "none" else float(args.alpha)
    coder = CountryCoder.load(args.bundle)
    n = len(args.texts)
    for d in coder.code_batch(args.texts, [args.field] * n, [args.register] * n, alpha):
        print(json.dumps(d.to_dict(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
