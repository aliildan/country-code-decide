"""onnxruntime session for serving (no torch import)."""

from __future__ import annotations

from pathlib import Path


def ort_session(path: Path, threads: int = 4):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
