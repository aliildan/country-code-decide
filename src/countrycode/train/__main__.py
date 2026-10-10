"""python -m countrycode.train build --name NAME [--gens a,b] | fit --config CFG"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from countrycode.evaluation.__main__ import gold_and_lexicon
from countrycode.real.items import read_items
from countrycode.synth.framework import SYNTH_DIR, base_aliases, read_synth
from countrycode.text import compact_key
from countrycode.train.dataset import (
    DATASET_DIR,
    build_examples,
    data_check,
    read_dataset,
    write_dataset,
)
from countrycode.train.trainer import load_config, train


def build(name: str, gens: list[str], seed: int, real_train: bool = False) -> None:
    gold, lx = gold_and_lexicon()
    items: list[tuple] = [
        (b.alias, b.targets, "lexicon:code" if b.kind == "code" else "lexicon")
        for b in base_aliases(lx)
    ]
    for tag in gens:
        for it in read_synth(SYNTH_DIR / f"{tag}.jsonl"):
            items.append((it.text, it.targets, f"synthetic:{tag}"))
    if real_train:
        from countrycode.real.roles import real_training_items

        for g in real_training_items(gold):
            items.append((g.text, g.targets, f"real:{g.label_source}", g.field_kind,
                          g.register_country))  # fmt: skip
    examples = build_examples(items, seed=seed)
    path = DATASET_DIR / f"{name}.jsonl"
    sha = write_dataset(examples, path)
    real_keys = {compact_key(i.text) for i in read_items()}
    from countrycode.real.roles import with_roles

    eval_keys = {compact_key(g.text) for g in with_roles(gold) if g.split != "train"}
    check = {"dataset": str(path), "sha256": sha, "seed": seed, "gens": gens,
             **data_check(examples, real_keys, eval_keys)}  # fmt: skip
    (DATASET_DIR / f"{name}.check.json").write_text(json.dumps(check, indent=1))
    print(json.dumps({k: v for k, v in check.items() if k != "by_source"}))
    if not check["ok"]:
        raise SystemExit("data check FAILED")


def seen_counts(keys_by_source: dict[str, list[str]], dataset_keys: set[str]) -> dict:
    """Per source: how many evaluation keys already occur (normalised) in a training dataset."""
    return {
        src: {"items": len(keys), "seen": sum(k in dataset_keys for k in keys)}
        for src, keys in sorted(keys_by_source.items())
    }


def seen(dataset: str, role: str) -> dict:
    """results/real/<role>_seen.json: evaluation keys of `role` present in `dataset`'s texts."""
    from countrycode.real.roles import with_roles

    gold, _ = gold_and_lexicon()
    keys: dict[str, list[str]] = {}
    for g in with_roles(gold):
        if g.split == role:
            keys.setdefault(g.source, []).append(compact_key(g.text))
    ds = {compact_key(e.text) for e in read_dataset(DATASET_DIR / f"{dataset}.jsonl")}
    rep = {"dataset": dataset, "role": role, "by_source": seen_counts(keys, ds)}
    out = Path("results/real") / f"{role}_seen.json"
    out.write_text(json.dumps(rep, indent=1) + "\n")
    return rep


SUMMARY_DIR = Path("results/train")
SUMMARY_KEYS = ("gens", "examples", "by_source", "synthetic_equal_to_real", "real_train_in_eval")


def write_train_summary(cfg: dict, check: dict, out_dir: Path = SUMMARY_DIR) -> Path:
    """Aggregate training-data record of a run (counts only) → results/train/<name>.json."""
    s = {
        "run": cfg["name"],
        "dataset_sha256": check["sha256"],
        "seed": check["seed"],
        **{k: check.get(k, 0) for k in SUMMARY_KEYS},
        "pending_g1": cfg.get("pending_g1", False),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cfg['name']}.json"
    path.write_text(json.dumps(s, indent=1) + "\n")
    return path


def fit(config: Path) -> None:
    cfg = load_config(config)
    check = json.loads((DATASET_DIR / f"{Path(cfg['dataset']).stem}.check.json").read_text())
    if not check["ok"]:
        raise SystemExit("refusing to train: data check not OK")
    cfg["dataset_sha256"] = check["sha256"]
    write_train_summary(cfg, check)
    gold, _ = gold_and_lexicon()
    if (
        cfg.get("selection") == "v11"
    ):  # select on the v1.1 selection role (GBIF dev + register half)
        from countrycode.real.roles import with_roles

        run = train(cfg, [g for g in with_roles(gold) if g.split == "select"], part="select")
    else:
        run = train(cfg, [g for g in gold if g.split == "dev"])
    print(f"run: {run}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--name", required=True)
    b.add_argument("--gens", default="")
    b.add_argument("--seed", type=int, default=13)
    b.add_argument("--real-train", action="store_true", help="v1.1: add real register train items")
    f = sub.add_parser("fit")
    f.add_argument("--config", type=Path, required=True)
    sn = sub.add_parser("seen", help="evaluation keys of a role already in a training dataset")
    sn.add_argument("--dataset", required=True)
    sn.add_argument("--role", required=True)
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.name, [g for g in args.gens.split(",") if g], args.seed, args.real_train)
    elif args.cmd == "seen":
        print(json.dumps(seen(args.dataset, args.role)))
    else:
        fit(args.config)


if __name__ == "__main__":
    main()
