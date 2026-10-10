"""ONNX export (fp32) and dynamic int8 quantisation for CPU / browser serving (plan-05 T1)."""

from __future__ import annotations

from pathlib import Path

import torch

from countrycode.infer.runtime import ort_session as ort_session  # re-export
from countrycode.train.model import CountryModel


def export_onnx(model: CountryModel, path: Path, opset: int = 17) -> Path:
    model = model.float().eval().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = torch.ones((2, 12), dtype=torch.long)
    mask = torch.ones((2, 12), dtype=torch.long)
    torch.onnx.export(
        model,
        (ids, mask),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=["country_logits", "state_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "country_logits": {0: "batch"},
            "state_logits": {0: "batch"},
        },
        opset_version=opset,
        dynamo=False,
    )
    return path


def quantize_int8(src: Path, dst: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return dst


def quantize_weights_only(src: Path, dst: Path, block_size: int = 32) -> Path:
    """int8 embedding table (dynamic Gather quantisation, a weight-only lookup) and 8-bit
    MatMulNBits with accuracy_level 4: onnxruntime's CPU kernels then quantise the activations
    to int8 per block (the certificate is computed on this CPU path; a backend without int8
    kernels computes in fp32 and gives slightly different probabilities). Dynamic int8
    quantisation of every op degraded argmax agreement to ~88–92 % on dev; this keeps ~99 %."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization import matmul_nbits_quantizer as mq

    tmp = dst.with_suffix(".gather.onnx")
    quantize_dynamic(
        str(src), str(tmp), weight_type=QuantType.QInt8, op_types_to_quantize=["Gather"]
    )
    q = mq.MatMulNBitsQuantizer(
        onnx.load(str(tmp)), bits=8, block_size=block_size, is_symmetric=True, accuracy_level=4,
        op_types_to_quantize=("MatMul",),
    )  # fmt: skip
    q.process()
    q.model.save_model_to_file(str(dst), use_external_data_format=False)  # type: ignore[attr-defined]  # ONNXModel wrapper
    tmp.unlink(missing_ok=True)
    return dst
