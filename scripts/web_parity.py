"""Browser vs server parity of a release bundle (plan-10).

On CPU, onnxruntime runs the bundle's MatMulNBits with int8 activations (accuracy level 4); the
browser's wasm backend computes them in fp32. A copy with accuracy level 0 reproduces the
browser's numbers exactly on the CPU, so the gap can be measured on many real strings here and
reported as aggregates (results/web-parity-<version>.json).

    uv run python scripts/web_parity.py --bundle dist/countrycode-small-v1.2 --part select
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = ("0.02", "0.05")


def browser_copy(bundle: Path, out: Path) -> Path:
    """The bundle with every MatMulNBits at accuracy level 0 (fp32 compute, as in the browser)."""
    import onnx

    out.mkdir(parents=True, exist_ok=True)
    for f in Path(bundle).iterdir():
        if f.is_file() and not f.name.endswith(".onnx"):
            shutil.copy(f, out / f.name)
    m = onnx.load(str(Path(bundle) / "model.int8.onnx"))
    for n in m.graph.node:
        if n.op_type == "MatMulNBits":
            for a in n.attribute:
                if a.name == "accuracy_level":
                    a.i = 0
    onnx.save(m, str(out / "model.int8.onnx"))
    return out


def compare(
    server, browser, texts: Sequence[str], fields: Sequence[str], registers: Sequence[str],
    sources: Sequence[str | None],
) -> dict:  # fmt: skip
    from countrycode.calib.calibration import decide
    from countrycode.infer.pipeline import TARGETS

    def probs(coder) -> np.ndarray:  # batches of 256, as in the evaluation
        return np.concatenate([
            coder.probs(texts[i : i + 256], fields[i : i + 256], registers[i : i + 256],
                        sources[i : i + 256])
            for i in range(0, len(texts), 256)
        ])  # fmt: skip

    a, b = probs(server), probs(browser)
    agree = {}
    for alpha in ALPHAS:
        same = [
            decide(ra, server.policy(s), float(alpha), TARGETS).target
            == decide(rb, browser.policy(s), float(alpha), TARGETS).target
            for ra, rb, s in zip(a, b, sources, strict=True)
        ]
        agree[alpha] = float(np.mean(same))
    top = np.abs(a.max(axis=1) - b.max(axis=1))
    return {
        "n": len(texts),
        "top1_agreement": float(np.mean(a.argmax(axis=1) == b.argmax(axis=1))),
        "decision_agreement": agree,
        "max_abs_dp": float(np.abs(a - b).max()),
        "mean_abs_dp_top": float(top.mean()),
        "p95_abs_dp_top": float(np.quantile(top, 0.95)),
    }


def main(argv: list[str] | None = None) -> Path:
    import tempfile

    from countrycode.evaluation.__main__ import gold_and_lexicon
    from countrycode.infer.pipeline import CountryCoder
    from countrycode.real.roles import with_roles

    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--part", default="select")
    ap.add_argument("--results", default=str(ROOT / "results"))
    args = ap.parse_args(argv)
    gold, _ = gold_and_lexicon()
    items = [g for g in with_roles(gold) if g.split == args.part]
    server = CountryCoder.load(args.bundle)
    with tempfile.TemporaryDirectory() as tmp:
        browser = CountryCoder.load(browser_copy(Path(args.bundle), Path(tmp)))
        rep = compare(
            server, browser, [g.text for g in items], [g.field_kind for g in items],
            [g.register_country for g in items], [g.source for g in items],
        )  # fmt: skip
    rep = {"bundle": Path(args.bundle).name, "part": args.part, **rep}
    version = Path(args.bundle).name.removeprefix("countrycode-")
    out = Path(args.results) / f"web-parity-{version}.json"
    out.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    print(main(sys.argv[1:]))
