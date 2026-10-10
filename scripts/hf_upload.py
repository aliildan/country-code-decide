"""Publish a release bundle to the Hugging Face Hub (plan-10 Task 3).

Refuses a bundle whose G1 stamp is missing or failing, a stale model card, or a bundle that is not
the documented release. Dry run by default; `--push` uploads (the repo is created private first),
`--public` then makes it public.

    uv run python scripts/hf_upload.py --bundle dist/countrycode-small-v1.2            # dry run
    uv run python scripts/hf_upload.py --bundle dist/countrycode-small-v1.2 --push --public
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "model.int8.onnx",
    "model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "calibration.json",
    "meta.json",
)


def _make_docs():
    spec = importlib.util.spec_from_file_location("make_docs", ROOT / "scripts" / "make_docs.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(bundle: Path, root: Path = ROOT) -> list[str]:
    """Reasons not to publish; empty when the bundle may go out."""
    problems = [f"bundle file missing: {f}" for f in FILES if not (bundle / f).exists()]
    meta = json.loads((bundle / "meta.json").read_text()) if (bundle / "meta.json").exists() else {}
    releases = meta.get("g1", {}).get("releases", {})
    if meta.get("pending_g1", True) or not releases:
        problems.append("bundle carries no passing G1 stamp (run `countrycode.infer stamp-g1`)")
    problems += [f"G1 not passed: {tag}" for tag, r in releases.items() if not r.get("pass")]
    version = json.loads((root / "results" / "release.json").read_text()).get("version")
    if bundle.name.removeprefix("countrycode-") != version:
        problems.append(f"bundle {bundle.name} is not the documented release ({version})")
    tmpl = (root / "configs" / "docs" / "model_card.md.tmpl").read_text(encoding="utf-8")
    card = root / "model_card.md"
    rendered = _make_docs().render(tmpl, root / "results")
    if not card.exists() or card.read_text(encoding="utf-8") != rendered:
        problems.append("model_card.md is stale (run scripts/make_docs.py)")
    return problems


def stage(bundle: Path, root: Path, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        shutil.copy(bundle / f, out / f)
    shutil.copy(root / "model_card.md", out / "README.md")
    for f in ("LICENSE", "NOTICE"):
        shutil.copy(root / f, out / f)
    return out


def main(argv: list[str] | None = None) -> int:
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--repo", default=None, help="default: hf_model in configs/docs/links.yaml")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args(argv)
    repo = args.repo or yaml.safe_load((ROOT / "configs/docs/links.yaml").read_text())["hf_model"]
    problems = check(args.bundle)
    if problems:
        print("refused:\n- " + "\n- ".join(problems), file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        out = stage(args.bundle, ROOT, Path(tmp) / "stage")
        for p in sorted(out.iterdir()):
            print(f"{p.stat().st_size / 1e6:9.1f} MB  {p.name}")
        if not args.push:
            print(f"dry run: would upload to {repo} (add --push)")
            return 0
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=str(out), commit_message=f"{args.bundle.name}")
        if args.public:
            api.update_repo_settings(repo, private=False)
        print(f"uploaded to https://huggingface.co/{repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
