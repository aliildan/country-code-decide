"""python -m countrycode.infer bundle --run runs/X --out dist/Y | bench --bundle dist/Y | parity"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np


def bundle(run: Path, out: Path, version: str) -> None:
    from countrycode.calib.calibration import Calibration
    from countrycode.train.trainer import load_run

    model, tok, meta = load_run(run, device="cpu")
    # placeholder until `calib.fit --bundle` fits the shipped file (final_run.sh does that next)
    cal_path = run / "calibration.json"
    cal = Calibration.load(cal_path) if cal_path.exists() else Calibration()
    from countrycode.infer.pipeline import make_bundle

    make_bundle(
        model, run / "tokenizer" / "tokenizer.json", cal, out,
        meta={"max_len": meta["config"]["max_len"], "version": version, "run": str(run),
              "base": meta["config"]["base"], "pending_g1": meta.get("pending_g1"),
              "dataset_sha256": meta.get("dataset_sha256"), "git_commit": meta.get("git_commit")},
    )  # fmt: skip
    tc = json.loads((run / "tokenizer" / "tokenizer_config.json").read_text())
    # transformers v5 writes tokenizer_class "TokenizersBackend"; JS/older loaders need the
    # original class name (checked: transformers.js 4.2.0 then yields identical token ids).
    tc["tokenizer_class"] = "PreTrainedTokenizerFast"
    tc["additional_special_tokens"] = tc.pop("extra_special_tokens", [])
    for k in ("backend", "is_local", "local_files_only"):
        tc.pop(k, None)
    (out / "tokenizer_config.json").write_text(json.dumps(tc, indent=2))
    sizes = {p.name: p.stat().st_size for p in out.iterdir()}
    print(json.dumps(sizes))


def bench(path: Path, threads: int, n: int, out: Path = Path("results/latency.json")) -> dict:
    from countrycode.infer.pipeline import CountryCoder
    from countrycode.real.items import read_items

    coder = CountryCoder.load(path, threads=threads)
    texts = [i.text for i in read_items()][:n]
    for t in texts[:20]:
        coder.code(t)
    times = []
    for t in texts:
        t0 = time.perf_counter()
        coder.code(t)
        times.append(1000 * (time.perf_counter() - t0))
    a = np.array(times)
    t0 = time.perf_counter()
    coder.code_batch(texts)
    batch_ms = 1000 * (time.perf_counter() - t0) / len(texts)
    rep = {"bundle": str(path), "threads": threads, "n": len(times),
           "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
           "batch_ms_per_item": batch_ms,
           "int8_bytes": (path / "model.int8.onnx").stat().st_size}  # fmt: skip
    out.write_text(json.dumps(rep, indent=1) + "\n")
    return rep


def stamp_g1(bundle: Path, train_summary: Path, report: Path | None = None) -> dict:
    """Write the G1 status of the bundle's synthetic releases into meta.json: `pending_g1` is
    False only when every release passed the owner's audit (the training-time flag said
    "not yet audited")."""
    from countrycode.synth.audit import GATE_REPORT, g1_status

    gens = json.loads(Path(train_summary).read_text())["gens"]
    st = g1_status(gens, report or GATE_REPORT)
    meta_path = Path(bundle) / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["pending_g1"] = not st["all_pass"]
    meta["g1"] = st
    meta["train_run"] = Path(
        train_summary
    ).stem  # the trained run (the bundle may be a trimmed copy)
    meta_path.write_text(json.dumps(meta, indent=1))
    return st


def synth_texts(
    synth_dir: Path, gens: Sequence[str] | None = None, exclude_keys: Iterable[str] = ()
) -> set[str]:
    """The text of every generated item of the given releases (all on disk when None; labels
    unused), without any text whose key is in `exclude_keys` (evaluation strings: old release files
    predate later G4 filters — the final review found nine test_v12 keys in them)."""
    from countrycode.text import compact_key

    skip = set(exclude_keys)
    paths = ([Path(synth_dir) / f"{g}.jsonl" for g in gens] if gens is not None
             else sorted(Path(synth_dir).glob("*@*.jsonl")))  # fmt: skip
    out: set[str] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                t = json.loads(line)["text"]
                if compact_key(t) not in skip:
                    out.add(t)
    return out


def trim_texts(
    run: Path, roles: tuple[str, ...] = ("train",), synth_dir: Path | None = None
) -> list[str]:
    """Texts whose tokens a trimmed vocabulary keeps: the run's training texts, the real strings
    of `roles` (default: training only, so select/calib stay unseen and measure the cost; never
    test), every context prefix and, with `synth_dir`, the text of every generated release on
    disk minus every evaluation string (old release files predate later G4 filters)."""
    from countrycode import labels
    from countrycode.evaluation.__main__ import gold_and_lexicon
    from countrycode.real.roles import with_roles
    from countrycode.text import compact_key
    from countrycode.train.dataset import context_text, read_dataset, rules

    meta = json.loads((run / "run.json").read_text())
    ds = read_dataset(Path(meta["config"]["dataset"]))
    texts = {e.text for e in ds} | {context_text(e.field, e.register) for e in ds}
    gold, _ = gold_and_lexicon()
    texts |= {g.text for g in with_roles(gold) if g.split in roles}
    regs = {"unknown", "US", *rules()["register_countries"], *labels.us_states(),
            *{g.register_country for g in gold}}  # fmt: skip
    fields = ("country", "jurisdiction", "nationality", "unknown")
    texts |= {context_text(f, r) for f in fields for r in regs}
    if synth_dir is not None:  # variant C (decisions 2026-10-10): every release on disk
        held = {compact_key(g.text) for g in with_roles(gold) if g.split != "train"}
        texts |= synth_texts(synth_dir, exclude_keys=held)
    return sorted(texts)


def token_identity(ta, tb, texts: Sequence[str], batch: int = 8192) -> tuple[float, int]:
    """Share of texts that two tokenizers split into the same tokens — every text, no sample."""
    same = 0
    for i in range(0, len(texts), batch):
        chunk = list(texts[i : i + batch])
        ea = ta.encode_batch(chunk, add_special_tokens=False)
        eb = tb.encode_batch(chunk, add_special_tokens=False)
        same += sum(x.tokens == y.tokens for x, y in zip(ea, eb, strict=True))
    return (same / len(texts) if texts else 1.0), len(texts)


def trim_check(a: Path, b: Path, texts: list[str], out: Path) -> dict:
    """Argmax agreement of two bundles on every real item (by role; no labels used), token
    identity on the kept texts, and file sizes."""
    from tokenizers import Tokenizer

    from countrycode.evaluation.__main__ import gold_and_lexicon
    from countrycode.infer.pipeline import CountryCoder
    from countrycode.real.roles import with_roles
    from countrycode.train.model import combine

    def softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    gold, _ = gold_and_lexicon()
    items = with_roles(gold)

    def top(path: Path, int8: bool) -> np.ndarray:
        coder = CountryCoder.load(path, int8=int8)
        out = []
        for i in range(0, len(items), 256):
            chunk = items[i : i + 256]
            c, s = coder.logits([g.text for g in chunk], [g.field_kind for g in chunk],
                                [g.register_country for g in chunk])  # fmt: skip
            out.append(combine(softmax(c), softmax(s)).argmax(1))
        return np.concatenate(out)

    def agree(x: np.ndarray, y: np.ndarray) -> dict:
        by: dict[str, list[bool]] = {}
        for g, ok in zip(items, x == y, strict=True):
            by.setdefault(g.split, []).append(bool(ok))
        roles = {r: float(np.mean(v)) for r, v in sorted(by.items())}
        return {"all": float(np.mean(x == y)), **roles}

    a8, b8, a32, b32 = top(a, True), top(b, True), top(a, False), top(b, False)
    from countrycode import labels
    from countrycode.evaluation.harness import Prediction, item_correct

    targets = list(labels.targets())
    sel = [i for i, g in enumerate(items) if g.split == "select"]

    def select_accuracy(t: np.ndarray) -> float:
        return float(np.mean([item_correct(Prediction(targets[int(t[i])], (), 1.0), items[i])
                              for i in sel]))  # fmt: skip

    kept_roles = np.array([g.split == "train" for g in items])
    ta = Tokenizer.from_file(str(a / "tokenizer.json"))
    tb = Tokenizer.from_file(str(b / "tokenizer.json"))
    tok_same, tok_checked = token_identity(ta, tb, texts)
    rep = {
        "bundle_a": str(a), "bundle_b": str(b),
        "size_mb_a": (a / "model.int8.onnx").stat().st_size / 1e6,
        "size_mb": (b / "model.int8.onnx").stat().st_size / 1e6,
        "vocab_old": ta.get_vocab_size(), "vocab_new": tb.get_vocab_size(),
        "select_accuracy_untrimmed": select_accuracy(a32),
        "select_accuracy": select_accuracy(b32),
        "agreement_fp32_kept": float(np.mean((a32 == b32)[kept_roles])),
        "agreement_fp32": agree(a32, b32),
        "agreement_int8": agree(a8, b8),
        "parity_int8_vs_fp32_a": float(np.mean(a8 == a32)),
        "parity_int8_vs_fp32_b": float(np.mean(b8 == b32)),
        "tokens_identical": tok_same, "tokens_checked": tok_checked,
    }  # fmt: skip
    out.write_text(json.dumps(rep, indent=1) + "\n")
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("--run", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--version", default="dev")
    c = sub.add_parser("bench")
    c.add_argument("--bundle", type=Path, required=True)
    c.add_argument("--threads", type=int, default=4)
    c.add_argument("--n", type=int, default=1000)
    c.add_argument("--out", type=Path, default=Path("results/latency.json"))
    t = sub.add_parser("trim", help="vocabulary-trimmed copy of a training run")
    t.add_argument("--run", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--no-single-chars", action="store_true", help="keep only seen single chars")
    t.add_argument("--synth-texts", action="store_true", help="also every generated release text")
    sg = sub.add_parser("stamp-g1", help="write the G1 audit status into a bundle's meta.json")
    sg.add_argument("--bundle", type=Path, required=True)
    sg.add_argument("--train-summary", type=Path, required=True, help="results/train/<run>.json")
    tc = sub.add_parser("trimcheck", help="agreement/size record of a trimmed bundle")
    tc.add_argument("--a", type=Path, required=True)
    tc.add_argument("--b", type=Path, required=True)
    tc.add_argument("--run", type=Path, required=True, help="the untrimmed run (kept texts)")
    tc.add_argument("--out", type=Path, required=True)
    tc.add_argument("--synth-texts", action="store_true", help="kept texts include releases")
    args = ap.parse_args()
    if args.cmd == "trim":
        from countrycode.infer.trim import trim_run
        from countrycode.synth.framework import SYNTH_DIR

        texts = trim_texts(args.run, synth_dir=SYNTH_DIR if args.synth_texts else None)
        rep = trim_run(args.run, args.out, texts, not args.no_single_chars)
        meta = json.loads((args.out / "run.json").read_text())
        source = "training texts + context prefixes"
        if args.synth_texts:
            source += " + every generated release text"
        meta["trim"] = {"vocabulary_from": source, "single_chars": not args.no_single_chars,
                        **rep}  # fmt: skip
        (args.out / "run.json").write_text(json.dumps(meta, indent=1, default=float))
        print(json.dumps(rep))
        return
    if args.cmd == "stamp-g1":
        st = stamp_g1(args.bundle, args.train_summary)
        print(json.dumps(st))
        if not st["all_pass"]:
            raise SystemExit("G1 not passed for every release: the bundle stays pending_g1")
        return
    if args.cmd == "trimcheck":
        from countrycode.synth.framework import SYNTH_DIR

        texts = trim_texts(args.run, synth_dir=SYNTH_DIR if args.synth_texts else None)
        print(json.dumps(trim_check(args.a, args.b, texts, args.out)))
        return
    if args.cmd == "bundle":
        bundle(args.run, args.out, args.version)
    else:
        print(json.dumps(bench(args.bundle, args.threads, args.n, args.out)))


if __name__ == "__main__":
    main()
