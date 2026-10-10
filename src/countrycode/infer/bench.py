"""Latency benchmark (H5): one string per call, tokenisation included, ONNX Runtime on CPU."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from countrycode.infer.onnx_export import ort_session
from countrycode.train.dataset import context_text

TokFn = Callable[[list[str], list[str]], tuple[np.ndarray, np.ndarray]]


def hf_tok_fn(tokenizer, max_len: int = 48) -> TokFn:
    def fn(contexts: list[str], texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        enc = tokenizer(contexts, texts, padding=True, truncation=True, max_length=max_len,
                        return_tensors="np")  # fmt: skip
        return enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)

    return fn


def bench_onnx(
    path: Path, tok: TokFn, texts: Sequence[str], threads: int = 4, warmup: int = 20
) -> dict:
    sess = ort_session(path, threads=threads)
    ctx = context_text("country", "unknown")

    def one(t: str) -> None:
        ids, mask = tok([ctx], [t])
        sess.run(None, {"input_ids": ids, "attention_mask": mask})

    for t in list(texts)[:warmup]:
        one(t)
    times = []
    for t in texts:
        t0 = time.perf_counter()
        one(t)
        times.append(1000 * (time.perf_counter() - t0))
    a = np.array(times)
    return {
        "n": len(times),
        "threads": threads,
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "mean_ms": float(a.mean()),
        "model_bytes": Path(path).stat().st_size,
    }
