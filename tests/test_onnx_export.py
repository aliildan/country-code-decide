import numpy as np
import torch

from countrycode.infer.onnx_export import export_onnx, ort_session, quantize_int8


def tiny_model():
    from transformers import ModernBertConfig, ModernBertModel

    from countrycode.train.model import CountryModel

    cfg = ModernBertConfig(
        vocab_size=300,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
    )
    torch.manual_seed(0)
    return CountryModel(ModernBertModel(cfg)).eval()


def test_export_parity_and_int8(tmp_path):
    m = tiny_model()
    path = export_onnx(m, tmp_path / "m.onnx")
    ids = torch.randint(5, 300, (3, 9))
    mask = torch.ones_like(ids)
    mask[0, 6:] = 0
    with torch.no_grad():
        c, s = m(ids, mask)
    sess = ort_session(path)
    oc, os_ = (
        np.asarray(x)
        for x in sess.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})
    )
    assert np.abs(oc - c.numpy()).max() < 1e-3 and np.abs(os_ - s.numpy()).max() < 1e-3
    q = quantize_int8(path, tmp_path / "m.int8.onnx")
    qc = np.asarray(
        ort_session(q).run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0]
    )
    assert qc.shape == oc.shape and q.stat().st_size < path.stat().st_size


def test_latency_bench_reports_percentiles(tmp_path):
    from countrycode.infer.bench import bench_onnx

    m = tiny_model()
    path = export_onnx(m, tmp_path / "m.onnx")

    def tok(contexts, texts):
        n = len(texts)
        return np.ones((n, 8), dtype=np.int64), np.ones((n, 8), dtype=np.int64)

    rep = bench_onnx(path, tok, ["Austria", "Deleware", "n/a"] * 10, threads=2, warmup=3)
    assert rep["n"] == 30 and 0 < rep["p50_ms"] <= rep["p95_ms"] and rep["threads"] == 2


def test_weight_only_quantisation_stays_close(tmp_path):
    from countrycode.infer.onnx_export import quantize_weights_only

    m = tiny_model()
    path = export_onnx(m, tmp_path / "m.onnx")
    q = quantize_weights_only(path, tmp_path / "m.w8.onnx")
    ids = torch.randint(5, 300, (4, 9))
    mask = torch.ones_like(ids)
    feeds = {"input_ids": ids.numpy(), "attention_mask": mask.numpy()}
    ref = np.asarray(ort_session(path).run(None, feeds)[0])
    got = np.asarray(ort_session(q).run(None, feeds)[0])
    assert np.abs(ref - got).max() < 0.1 and (ref.argmax(1) == got.argmax(1)).all()
