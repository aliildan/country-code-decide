import json

import torch

from countrycode.calib.calibration import Calibration
from countrycode.infer.pipeline import CountryCoder, Decision, make_bundle


def tiny_tokenizer(path):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    vocab = {"[PAD]": 0, "[UNK]": 1, "<bos>": 2, "<eos>": 3}
    for w in [
        "field",
        ":",
        "country",
        "jurisdiction",
        "unknown",
        "|",
        "register",
        "us",
        "at",
        "austria",
        "deleware",
        "n/a",
    ]:
        vocab.setdefault(w, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B <eos>",
        special_tokens=[("<bos>", 2), ("<eos>", 3)],
    )
    tok.save(str(path))


def tiny_model():
    from transformers import ModernBertConfig, ModernBertModel

    from countrycode.train.model import CountryModel

    cfg = ModernBertConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
    )
    torch.manual_seed(0)
    return CountryModel(ModernBertModel(cfg)).eval()


def test_bundle_and_decisions(tmp_path):
    tiny_tokenizer(tmp_path / "tokenizer.json")
    cal = Calibration(temps=(1.0, 1.0), thresholds={"0.01": 0.99, "0.05": 0.0}, conformal_q=0.9)
    out = make_bundle(
        tiny_model(),
        tmp_path / "tokenizer.json",
        cal,
        tmp_path / "bundle",
        meta={"max_len": 32, "version": "test"},
    )
    assert (out / "model.int8.onnx").exists() and (out / "meta.json").exists()
    coder = CountryCoder.load(out)
    d = coder.code("Austria", field="country")
    assert isinstance(d, Decision) and 0 <= d.confidence <= 1 and d.candidates
    strict = coder.code("Austria", alpha=0.01)
    assert strict.abstain and strict.target is None  # tiny random model never reaches 0.99
    loose = coder.code("Austria", alpha=0.05)
    assert not loose.abstain and loose.target is not None
    assert (loose.us_state is not None) == loose.target.startswith("US-")
    batch = coder.code_batch(
        ["Austria", "Deleware", ""], fields=["country", "jurisdiction", "country"], alpha=0.05
    )
    assert len(batch) == 3
    meta = json.loads((out / "meta.json").read_text())
    assert meta["version"] == "test" and len(meta["targets"]) == 302
    assert meta["names"]["AT"] == "Austria" and meta["names"]["US-CA"] == "California"


def test_logits_shapes(tmp_path):
    tiny_tokenizer(tmp_path / "tokenizer.json")
    cal = Calibration(temps=(1.0, 1.0), thresholds={}, conformal_q=0.9)
    b = make_bundle(
        tiny_model(), tmp_path / "tokenizer.json", cal, tmp_path / "b", meta={"max_len": 32}
    )
    c, s = CountryCoder.load(b).logits(["Austria", "x"], ["country", "unknown"], ["unknown", "US"])
    assert c.shape == (2, 251) and s.shape == (2, 52)


def test_domain_policies_drive_decisions(tmp_path):
    from countrycode.calib.calibration import CalibrationSet

    tiny_tokenizer(tmp_path / "tokenizer.json")
    loose = Calibration(temps=(1.0, 1.0), thresholds={"0.05": 0.0}, conformal_q=0.9)
    strict = Calibration(temps=(1.0, 1.0), thresholds={"0.05": None}, conformal_q=0.9)
    cs = CalibrationSet({"register": loose, "curated": strict})
    b = make_bundle(
        tiny_model(), tmp_path / "tokenizer.json", cs, tmp_path / "b", meta={"max_len": 32}
    )
    coder = CountryCoder.load(b)
    out = coder.code_batch(["Austria"] * 3, sources=["us-co", "gbif", "unknown-source"], alpha=0.05)
    assert [d.abstain for d in out] == [False, True, True]  # unknown → strictest merge


def test_a_caller_can_choose_the_calibration_domain(tmp_path):
    from countrycode.calib.calibration import CalibrationSet

    tiny_tokenizer(tmp_path / "tokenizer.json")
    loose = Calibration(temps=(1.0, 1.0), thresholds={"0.05": 0.0}, conformal_q=0.9)
    strict = Calibration(temps=(1.0, 1.0), thresholds={"0.05": None}, conformal_q=0.9)
    b = make_bundle(tiny_model(), tmp_path / "tokenizer.json",
                    CalibrationSet({"register": loose, "curated": strict}), tmp_path / "b",
                    meta={"max_len": 32})  # fmt: skip
    coder = CountryCoder.load(b)
    out = coder.code_batch(["Austria"] * 2, domains=["register", "curated"], alpha=0.05)
    assert [d.abstain for d in out] == [False, True]
    assert coder.code("Austria", domain="register", alpha=0.05).abstain is False


def test_the_chosen_domain_also_sets_the_temperatures(tmp_path):
    from countrycode.calib.calibration import CalibrationSet

    tiny_tokenizer(tmp_path / "tokenizer.json")
    sharp = Calibration(temps=(0.5, 1.0), thresholds={"0.05": 0.0}, conformal_q=0.9)
    soft = Calibration(temps=(5.0, 1.0), thresholds={"0.05": 0.0}, conformal_q=0.9)
    b = make_bundle(tiny_model(), tmp_path / "tokenizer.json",
                    CalibrationSet({"register": sharp, "curated": soft}), tmp_path / "b",
                    meta={"max_len": 32})  # fmt: skip
    coder = CountryCoder.load(b)
    p = coder.probs(
        ["Austria"] * 2, ["unknown"] * 2, ["unknown"] * 2, domains=["register", "curated"]
    )
    assert p[0].max() > p[1].max()


def test_pipeline_rejects_an_unknown_domain(tmp_path):
    import pytest

    from countrycode.calib.calibration import CalibrationSet

    tiny_tokenizer(tmp_path / "tokenizer.json")
    cal = Calibration(temps=(1.0, 1.0), thresholds={"0.05": 0.0}, conformal_q=0.9)
    b = make_bundle(tiny_model(), tmp_path / "tokenizer.json",
                    CalibrationSet({"register": cal, "curated": cal}), tmp_path / "b",
                    meta={"max_len": 32})  # fmt: skip
    with pytest.raises(ValueError, match="domain"):
        CountryCoder.load(b).code("Austria", domain="regsiter")


def test_always_answer_candidates_follow_the_policys_candidate_rule(tmp_path):
    tiny_tokenizer(tmp_path / "tokenizer.json")
    cal = Calibration(temps=(1.0, 1.0), thresholds={"0.05": 0.0}, conformal_q=0.9,
                      candidate_rule="min_p", candidate_min_p=0.0065, candidate_max=3)  # fmt: skip
    b = make_bundle(
        tiny_model(), tmp_path / "tokenizer.json", cal, tmp_path / "b", meta={"max_len": 32}
    )
    coder = CountryCoder.load(b)
    row = coder.probs(["Austria"], ["country"], ["unknown"])[0]
    always = coder.code("Austria", field="country", alpha=None)
    gated = coder.code("Austria", field="country", alpha=0.05)
    assert always.candidates == gated.candidates  # same rule with or without a target
    names = list(coder.meta["targets"])
    assert all(row[names.index(c)] >= 0.0065 for c in always.candidates)
    assert len(always.candidates) < 3  # the tiny model's third code is below min_p
    assert always.target == names[int(row.argmax())]
