"""python -m countrycode.synth release [--gens keyboard,format,translit,junk] [--workers N]

Writes data/processed/synth/<gen>@<ver>.jsonl + manifest.json and the G2 realism report
results/synthetic/realism.json (aggregates only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rapidfuzz.distance import OSA

from countrycode.evaluation.__main__ import gold_and_lexicon
from countrycode.real.items import read_items
from countrycode.synth.framework import SYNTH_DIR, read_synth, run_release
from countrycode.synth.gates import CollisionChecker
from countrycode.synth.realism import divergence_report, edit_profile, real_noisy_pairs
from countrycode.text import compact_key

REALISM_PATH = Path("results/synthetic/realism.json")


def make_generators(names: list[str], lexicon):
    """Current generator versions (plan-08: the repaired ones; junk@1, city@2, county@1 as
    before; llmjunk retired)."""
    from countrycode.synth.formats import FormatVariants, JunkStrings, Transliterate
    from countrycode.synth.framework import base_info
    from countrycode.synth.keyboard import KeyboardTypos

    info = base_info(lexicon)

    def mod(name: str):
        return __import__(f"countrycode.synth.{name}", fromlist=["X"])

    table = {
        "keyboard": lambda: KeyboardTypos(per_alias=3, info=info),
        "format": lambda: FormatVariants(lexicon, info=info),
        "translit": lambda: Transliterate(info=info, version="4"),
        "junk": lambda: JunkStrings(n=3000, version="2"),
        "regjunk": lambda: mod("regjunk").RegisterJunk(),
        "ocr": lambda: mod("ocr").OcrVariants(info=info),
        "llm": lambda: mod("llm").LlmVariants(version="2"),
        "geo": lambda: mod("geo").GeoContext(lexicon, version="2"),
        "pairs": lambda: mod("combos").CountryPairs(lexicon, n=4000, version="3"),
        "city": lambda: mod("city").CityNames(),
        "county": lambda: mod("county").CountyNames(),
        "affix": lambda: mod("combos").Affixes(lexicon, version="2"),
        "region": lambda: mod("region").RegionNames(),
        "caprov": lambda: mod("caprov").CanadianProvinceCodes(),
    }
    return [table[n]() for n in names]


# Releases made with the v1 checker; they keep it so their files (and audits) stay identical.
V1_CHECKER = frozenset({"junk@1", "city@2", "county@1"})


def checker_groups(gens: list, lexicon) -> list[tuple[list, CollisionChecker]]:
    old = [g for g in gens if f"{g.name}@{g.version}" in V1_CHECKER]
    new = [g for g in gens if f"{g.name}@{g.version}" not in V1_CHECKER]
    groups = [(old, CollisionChecker(lexicon)), (new, release_checker(lexicon))]
    return [(gs, c) for gs, c in groups if gs]


def release_checker(lexicon) -> CollisionChecker:
    """The G1 checker for new releases (plan-08): margin 10, containment, no NONE mixes."""
    return CollisionChecker(lexicon, margin=10, containment=True, reject_none_mix=True)


def realism(manifest: dict, lexicon) -> dict:
    gold, _ = gold_and_lexicon()
    dev = [(g.text, g.targets) for g in gold if g.split == "dev"]
    real = edit_profile(real_noisy_pairs(dev, lexicon))
    previous = json.loads(REALISM_PATH.read_text()) if REALISM_PATH.exists() else {}
    report = {"real_dev_typos": real, "generators": previous.get("generators", {})}
    for tag in manifest["generators"]:
        items = read_synth(SYNTH_DIR / f"{tag}.jsonl")
        pairs = []
        for it in items:
            a, b = compact_key(it.text), compact_key(it.source_alias)
            if a and b and OSA.distance(a, b) <= 3 and abs(len(a) - len(b)) <= 2 and a != b:
                pairs.append((a, b))
        prof = edit_profile(pairs)
        report["generators"][tag] = {
            "typo_like_items": len(pairs),
            "profile": prof,
            "divergence_vs_real": divergence_report(real, prof) if pairs else None,
        }
    REALISM_PATH.parent.mkdir(parents=True, exist_ok=True)
    REALISM_PATH.write_text(json.dumps(report, indent=1) + "\n")
    return report


def gate_report(release: str = "synth-v2") -> dict:
    """Aggregate G1–G5 status per generator → results/synthetic/<release>.json (no strings)."""
    from countrycode.synth.audit import audit_status, sample_for_audit

    manifest = json.loads((SYNTH_DIR / "manifest.json").read_text())
    realism_rep = json.loads(REALISM_PATH.read_text()) if REALISM_PATH.exists() else {}
    audit = audit_status(sample_for_audit())
    runs = {}
    for name in ("small-lex", "small-synth", "small-synth2"):
        p = Path("runs") / name / "run.json"
        if p.exists():
            ep = json.loads(p.read_text())["epochs"]
            runs[name] = max(e.get("accuracy", 0.0) for e in ep)
    gens = {}
    for tag, g in sorted(manifest["generators"].items()):
        d = g["dropped"]
        gens[tag] = {
            "items": g["items"],
            "config_hash": g["config_hash"],
            "G1_auto": {
                "exact_other": d.get("exact_other", 0),
                "near_other": d.get("near_other", 0),
            },
            "G1_audit": audit.get(tag, {"sampled": 0, "audited": 0, "errors": 0, "g1_pass": False}),
            "G2_divergence": (realism_rep.get("generators", {}).get(tag) or {}).get(
                "divergence_vs_real"
            ),
            "G4_leaks_removed": d.get("leak", 0),
            "G5": {"seed": manifest["seed"], "config_hash": g["config_hash"]},
        }
    report = {
        "release": release,
        "generators": gens,
        "G3_dev_accuracy": runs,
        "all_g1_pass": all(v["G1_audit"].get("g1_pass") for v in gens.values()),
    }
    out = Path("results/synthetic") / f"{release}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("release")
    r.add_argument("--gens", default="keyboard,format,translit,junk")
    r.add_argument("--workers", type=int, default=24)
    r.add_argument("--seed", type=int, default=13)
    sub.add_parser("ocr-cache")
    sub.add_parser("llm-cache")
    sub.add_parser("report")
    args = ap.parse_args()
    _, lexicon = gold_and_lexicon()
    if args.cmd == "report":
        print(json.dumps(gate_report(), indent=1))
        return
    if args.cmd == "llm-cache":
        from countrycode import labels
        from countrycode.synth.llm import MODEL, build_llm_cache

        n = build_llm_cache(MODEL, [*labels.countries(), *labels.us_states()])
        print(f"llm prompts answered: {n}")
        return
    if args.cmd == "ocr-cache":
        from countrycode.synth.framework import base_aliases
        from countrycode.synth.ocr import build_cache

        print(f"ocr readings written: {build_cache(base_aliases(lexicon))}")
        return
    real_keys = {compact_key(i.text) for i in read_items()}  # every split: G4
    manifest: dict = {}
    for gens, checker in checker_groups(make_generators(args.gens.split(","), lexicon), lexicon):
        manifest = run_release(
            gens, lexicon, real_keys, seed=args.seed, checker=checker, workers=args.workers
        )
    print(json.dumps({t: (g["items"], g["dropped"]) for t, g in manifest["generators"].items()}))
    rep = realism(manifest, lexicon)
    print(json.dumps({t: g["divergence_vs_real"] for t, g in rep["generators"].items()}))


if __name__ == "__main__":
    main()
