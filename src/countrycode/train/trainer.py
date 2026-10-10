"""Trainer and model predictor (plan-04 Task 2).

    uv run python -m countrycode.train fit --config configs/train/small-lex.yaml

Selection: the epoch with the best type-level accuracy on real dev (never calib/test). The run
directory holds weights (safetensors), the config, dataset hash, git commit, per-epoch dev
metrics and the `pending_g1` flag (synthetic data not yet audited by the owner).
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file, save_file

from countrycode import labels
from countrycode.evaluation.harness import Prediction, evaluate
from countrycode.real.gold import GoldItem
from countrycode.train.dataset import (
    COUNTRY_CLASSES,
    STATE_CLASSES,
    TrainExample,
    context_text,
    read_dataset,
)
from countrycode.train.model import CountryModel, combine, soft_ce, target_matrix

RUNS = Path("runs")
_TARGETS = labels.targets()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def encode(tok, contexts: Sequence[str], texts: Sequence[str], max_len: int, device: str):
    enc = tok(
        list(contexts), [t if t.strip() else "∅" for t in texts], padding=True, truncation=True,
        max_length=max_len, return_tensors="pt",
    )  # fmt: skip
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


class ModelPredictor:
    """Answers every item with the argmax target; confidence = its (optionally tempered) prob."""

    def __init__(
        self,
        model: CountryModel,
        tokenizer,
        max_len: int = 48,
        name: str = "model",
        temps: tuple[float, float] = (1.0, 1.0),
        batch: int = 512,
    ) -> None:
        self.model, self.tok, self.max_len, self.name = model, tokenizer, max_len, name
        self.temps, self.batch = temps, batch

    @torch.inference_mode()
    def probs(self, items: Sequence[GoldItem]) -> np.ndarray:
        device = next(self.model.parameters()).device.type
        self.model.eval()
        out = []
        for i in range(0, len(items), self.batch):
            chunk = items[i : i + self.batch]
            ctx = [context_text(g.field_kind, g.register_country) for g in chunk]
            ids, mask = encode(self.tok, ctx, [g.text for g in chunk], self.max_len, device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                c, s = self.model(ids, mask)
            pc = torch.softmax(c.float() / self.temps[0], -1).cpu().numpy()
            ps = torch.softmax(s.float() / self.temps[1], -1).cpu().numpy()
            out.append(combine(pc, ps))
        return np.concatenate(out) if out else np.zeros((0, len(_TARGETS)))

    def predict(self, items: Sequence[GoldItem]) -> list[Prediction]:
        p = self.probs(items)
        preds = []
        for row in p:
            order = np.argsort(-row)
            top = _TARGETS[int(order[0])]
            cands = tuple(_TARGETS[int(j)] for j in order[:3])
            preds.append(Prediction(top, cands, float(row[order[0]])))
        return preds


def _batches(n: int, size: int, rng: np.random.Generator):
    idx = rng.permutation(n)
    for i in range(0, n, size):
        yield idx[i : i + size]


def selection_score(rep: dict, metric: str = "accuracy") -> float:
    """`accuracy`: type-level accuracy over all items. `domain_macro` (v1.1): mean of the
    per-domain accuracies (register, curated), so the many curated-list strings do not decide
    model selection alone."""
    if metric == "accuracy":
        return rep["overall"]["accuracy"]
    from countrycode.calib.calibration import domain_of

    hits: dict[str, list[float]] = {}
    for k, v in rep["slices"].items():
        if k.startswith("source=") and v.get("n"):
            dom = domain_of(k.removeprefix("source=")) or "unknown"
            h = hits.setdefault(dom, [0.0, 0.0])
            h[0] += v["accuracy"] * v["n"]
            h[1] += v["n"]
    return sum(c / n for c, n in hits.values()) / len(hits)


def train(cfg: dict, dev_gold: Sequence[GoldItem] | None = None, part: str = "dev") -> Path:
    from transformers import AutoTokenizer

    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    run = RUNS / cfg["name"]
    run.mkdir(parents=True, exist_ok=True)
    data: list[TrainExample] = read_dataset(Path(cfg["dataset"]))
    tok = AutoTokenizer.from_pretrained(cfg["base"])
    tok.save_pretrained(run / "tokenizer")
    model = CountryModel.from_base(cfg["base"]).to("cuda")
    heads = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": cfg["lr_encoder"]}, {"params": heads, "lr": cfg["lr_head"]}],
        weight_decay=0.01,
    )
    steps = cfg["epochs"] * math.ceil(len(data) / cfg["batch"])
    warm = int(cfg.get("warmup", 0.05) * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / max(warm, 1)) * max(0.0, (steps - s) / max(steps - warm, 1)),
    )
    meta = {
        "config": cfg, "git_commit": _git_commit(), "dataset_sha256": cfg.get("dataset_sha256"),
        "classes": {"country": list(COUNTRY_CLASSES), "state": list(STATE_CLASSES)},
        "pending_g1": bool(cfg.get("pending_g1", True)), "epochs": [],
    }  # fmt: skip
    best = -1.0
    for epoch in range(cfg["epochs"]):
        model.train()
        t0, total, n = time.perf_counter(), 0.0, 0
        for idx in _batches(len(data), cfg["batch"], rng):
            b = [data[i] for i in idx]
            ids, mask = encode(
                tok, [context_text(e.field, e.register) for e in b], [e.text for e in b],
                cfg["max_len"], "cuda",
            )  # fmt: skip
            tc = target_matrix([e.country for e in b], COUNTRY_CLASSES).cuda()
            has_s = torch.tensor([e.state is not None for e in b], device="cuda")
            ts = target_matrix([e.state or {} for e in b], STATE_CLASSES).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                c, s = model(ids, mask)
            loss = soft_ce(c, tc)
            if has_s.any():
                loss = loss + cfg.get("state_weight", 1.0) * soft_ce(s[has_s], ts[has_s])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(loss) * len(b)
            n += len(b)
        rec = {
            "epoch": epoch + 1,
            "train_loss": total / max(n, 1),
            "seconds": time.perf_counter() - t0,
        }
        if dev_gold:
            pred = ModelPredictor(model, tok, cfg["max_len"], name=cfg["name"])
            rep = evaluate(pred, dev_gold, part, out_dir=None)
            rec |= {k: rep["overall"][k] for k in ("accuracy", "country_accuracy", "aurc")}
            score = selection_score(rep, cfg.get("selection_metric", "accuracy"))
            rec["selection_score"] = score
        else:
            score = -rec["train_loss"]
        meta["epochs"].append(rec)
        print(json.dumps(rec), flush=True)
        if score > best:
            best = score
            save_file(
                {k: v.contiguous() for k, v in model.state_dict().items()},
                run / "model.safetensors",
            )
            meta["best_epoch"] = epoch + 1
        (run / "run.json").write_text(json.dumps(meta, indent=1, default=float))
    return run


def _load_tokenizer(run: Path, meta: dict):
    from transformers import AutoTokenizer

    tok_dir = run / "tokenizer"
    return AutoTokenizer.from_pretrained(tok_dir if tok_dir.exists() else meta["config"]["base"])


def load_run(run: Path, device: str = "cuda") -> tuple[CountryModel, object, dict]:
    """`config_overrides` in run.json (e.g. a trimmed vocabulary) are applied to the base config."""
    from transformers import AutoConfig, AutoModel

    meta = json.loads((run / "run.json").read_text())
    cfg = AutoConfig.from_pretrained(meta["config"]["base"])
    for k, v in meta.get("config_overrides", {}).items():
        setattr(cfg, k, v)
    model = CountryModel(AutoModel.from_config(cfg))
    model.load_state_dict(load_file(str(run / "model.safetensors")))
    model.to(device).eval()
    return model, _load_tokenizer(run, meta), meta


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
