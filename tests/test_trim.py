import json

import numpy as np
from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE

from countrycode.infer.trim import kept_ids, trim_state_dict, trim_tokenizer

SPECIAL = ["<pad>", "<eos>", "<bos>", "<unk>"]


def toy_tokenizer() -> dict:
    vocab = {t: i for i, t in enumerate(SPECIAL)}
    for b in range(256):
        vocab[f"<0x{b:02X}>"] = len(vocab)
    for t in ["▁", "a", "b", "c", "d", "x", "y", "ab", "abc", "abcd", "▁abcd", "xy", "▁xy"]:
        vocab[t] = len(vocab)
    merges = [("a", "b"), ("ab", "c"), ("abc", "d"), ("▁", "abcd"), ("x", "y"), ("▁", "xy")]
    tok = Tokenizer(BPE(vocab, merges, unk_token="<unk>", byte_fallback=True, fuse_unk=True))
    tok.normalizer = normalizers.Replace(" ", "▁")
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
    tok.add_special_tokens(SPECIAL)
    return json.loads(tok.to_str())


def tokens(tok_json: dict, text: str) -> list[str]:
    return Tokenizer.from_str(json.dumps(tok_json)).encode(text).tokens


def test_kept_ids_include_merge_closure():
    tj = toy_tokenizer()
    keep = kept_ids(tj, ["abcd"])
    inv = {i: t for t, i in tj["model"]["vocab"].items()}
    kept = {inv[i] for i in keep}
    assert {"▁abcd", "abcd", "abc", "ab"} <= kept
    assert "xy" not in kept and "▁xy" not in kept
    assert all(f"<0x{b:02X}>" in kept for b in range(256)) and set(SPECIAL) <= kept


def test_trimmed_tokenizer_preserves_kept_texts():
    tj = toy_tokenizer()
    new, new_to_old = trim_tokenizer(tj, kept_ids(tj, ["abcd", "ab cd"]))
    for text in ["abcd", "ab cd"]:
        assert tokens(new, text) == tokens(tj, text)
    old_inv = {i: t for t, i in tj["model"]["vocab"].items()}
    assert all(old_inv[o] == t for t, n in new["model"]["vocab"].items() for o in [new_to_old[n]])


def test_trimmed_tokenizer_handles_unseen_text():
    tj = toy_tokenizer()
    new, _ = trim_tokenizer(tj, kept_ids(tj, ["abcd"]))
    for text in ["xy", "Ж", "🙂", ""]:
        out = tokens(new, text)
        assert "<unk>" not in out
    assert tokens(new, "Ж")[-2:] == ["<0xD0>", "<0x96>"]  # byte fallback


def test_trim_state_dict_slices_embeddings():
    w = np.arange(12, dtype=np.float32).reshape(6, 2)
    sd = {"encoder.embeddings.tok_embeddings.weight": w, "head.weight": np.ones((3, 2))}
    out = trim_state_dict(sd, [0, 2, 5])
    assert out["encoder.embeddings.tok_embeddings.weight"].tolist() == [[0, 1], [4, 5], [10, 11]]
    assert out["head.weight"].shape == (3, 2)


def test_load_run_applies_config_overrides(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig

    from countrycode.train import trainer
    from countrycode.train.model import CountryModel

    cfg = AutoConfig.for_model(
        "modernbert", vocab_size=300, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, pad_token_id=0, bos_token_id=2,
        eos_token_id=1, cls_token_id=2, sep_token_id=1,
    )  # fmt: skip
    monkeypatch.setattr(AutoConfig, "from_pretrained", staticmethod(lambda *a, **k: cfg))
    from transformers import AutoModel

    small = AutoConfig.for_model(**{**cfg.to_dict(), "vocab_size": 40})
    model = CountryModel(AutoModel.from_config(small))
    save_file({k: v.contiguous() for k, v in model.state_dict().items()},
              tmp_path / "model.safetensors")  # fmt: skip
    (tmp_path / "run.json").write_text(json.dumps(
        {"config": {"base": "x"}, "config_overrides": {"vocab_size": 40}}))  # fmt: skip
    monkeypatch.setattr(trainer, "_load_tokenizer", lambda run, meta: None)
    m, _, _ = trainer.load_run(tmp_path, device="cpu")
    emb = [p for n, p in m.state_dict().items() if n.endswith("tok_embeddings.weight")][0]
    assert emb.shape[0] == 40 and isinstance(emb, torch.Tensor)


def test_closure_follows_every_merge_that_builds_a_kept_token():
    # the real path for "abcd" is c+d, b+cd, a+bcd; the first-listed producer of "abcd" is
    # abc+d. A first-producer closure drops bcd, and "cd" (kept for another text) then blocks
    # the abc route: the trimmed tokenizer would stop at ["▁", "ab", "cd"].
    vocab = {t: i for i, t in enumerate(SPECIAL)}
    for b in range(256):
        vocab[f"<0x{b:02X}>"] = len(vocab)
    for t in ["▁", "a", "b", "c", "d", "cd", "bcd", "ab", "abc", "abcd", "▁abcd"]:
        vocab[t] = len(vocab)
    merges = [("c", "d"), ("b", "cd"), ("a", "b"), ("abc", "d"), ("ab", "c"), ("a", "bcd"),
              ("▁", "abcd")]  # fmt: skip
    tok = Tokenizer(BPE(vocab, merges, unk_token="<unk>", byte_fallback=True, fuse_unk=True))
    tok.normalizer = normalizers.Replace(" ", "▁")
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
    tok.add_special_tokens(SPECIAL)
    tj = json.loads(tok.to_str())
    assert tokens(tj, "abcd") == ["▁abcd"]
    new, _ = trim_tokenizer(tj, kept_ids(tj, ["abcd", "cd"]))
    assert tokens(new, "abcd") == ["▁abcd"] and tokens(new, "cd") == tokens(tj, "cd")


def test_synth_texts_reads_every_release_text(tmp_path):
    from countrycode.infer.__main__ import synth_texts

    (tmp_path / "keyboard@1.jsonl").write_text(
        json.dumps({"text": "Austira", "targets": ["AT"]}) + "\n", encoding="utf-8"
    )
    (tmp_path / "ocr@2.jsonl").write_text(
        json.dumps({"text": "Swilzerland", "targets": ["CH"]}) + "\n", encoding="utf-8"
    )
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    assert synth_texts(tmp_path) == {"Austira", "Swilzerland"}


def test_synth_texts_reads_only_the_runs_releases_and_drops_evaluation_strings(tmp_path):
    from countrycode.infer.__main__ import synth_texts

    for name, text in (("keyboard@2", "Austira"), ("format@3", "CE"), ("keyboard@1", "Nrway")):
        (tmp_path / f"{name}.jsonl").write_text(json.dumps({"text": text}) + "\n", encoding="utf-8")
    (tmp_path / "format@4.jsonl").write_text(
        json.dumps({"text": "NIR"}) + "\n" + json.dumps({"text": "Swizerland"}) + "\n",
        encoding="utf-8",
    )
    got = synth_texts(tmp_path, gens=["keyboard@2", "format@4"], exclude_keys={"nir"})
    assert got == {"Austira", "Swizerland"}


def test_token_identity_checks_every_text_not_a_sample():
    from countrycode.infer.__main__ import token_identity

    tj = toy_tokenizer()
    new, _ = trim_tokenizer(tj, kept_ids(tj, ["abcd"], single_chars=False))
    a, b = (Tokenizer.from_str(json.dumps(t)) for t in (tj, new))
    texts = ["abcd"] * 50_001
    texts[49_999] = "xy"  # odd index: a stride-2 sample would never see it
    share, checked = token_identity(a, b, texts)
    assert checked == 50_001 and share < 1.0
