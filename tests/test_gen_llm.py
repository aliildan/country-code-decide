import json

import numpy as np

from countrycode.synth.framework import BaseAlias
from countrycode.synth.llm import LlmJunk, LlmVariants, build_llm_cache, parse_variants


def test_parse_variants_is_robust():
    assert parse_variants('{"variants": ["Österreich", " Austria ", 3, ""]}') == [
        "Österreich",
        "Austria",
    ]
    assert parse_variants("no json here") == []


def test_cache_and_generate(tmp_path):
    def post(model, prompt):
        if "NOT name any country" in prompt:
            return json.dumps({"variants": ["n/a", "Europe"]})
        return json.dumps({"variants": ["Österreich", "Austria", "Autriche"]})

    path = tmp_path / "llm.jsonl"
    build_llm_cache("m", ["AT"], post_fn=post, path=path, log_path=None)
    gen = LlmVariants(path=path)
    out = gen.generate(BaseAlias("Austria", ("AT",), "llm", "en"), np.random.default_rng(0))
    assert out == ["Österreich", "Autriche"]  # the base name itself is not a variant
    junk = LlmJunk(path=path)
    assert junk.generate(junk.bases()[0], np.random.default_rng(0)) == ["n/a", "Europe"]
    assert junk.near_check is False

    def boom(model: str, prompt: str) -> str:
        raise AssertionError("must not be called: everything is cached")

    build_llm_cache("m", ["AT"], post_fn=boom, path=path, log_path=None)


def test_llm_variants_keep_latin_letters_only(tmp_path):
    def post(model, prompt):
        return json.dumps({"variants": ["205", "الإمارات", "Andora", "AD", "U.A.E."]})

    path = tmp_path / "llm.jsonl"
    build_llm_cache("m", ["AD"], post_fn=post, path=path, log_path=None, include_junk=False)
    out = LlmVariants(path=path).generate(
        BaseAlias("Andorra", ("AD",), "llm", "en"), np.random.default_rng(0)
    )
    assert out == ["Andora", "AD", "U.A.E."]


def test_empty_answers_are_retried(tmp_path):
    path = tmp_path / "llm.jsonl"
    build_llm_cache(
        "m", ["AT"], post_fn=lambda m, p: "{}", path=path, log_path=None, include_junk=False
    )
    build_llm_cache(
        "m",
        ["AT"],
        post_fn=lambda m, p: json.dumps({"variants": ["Austria"]}),
        path=path,
        log_path=None,
        include_junk=False,
    )
    from countrycode.synth.llm import read_cache

    assert read_cache(path)["AT"] == ["Austria"]


def test_llm_v2_drops_company_names(tmp_path):
    def post(model, prompt):
        return json.dumps({"variants": ["Brunei Darussalam Holdings", "Brunei Co. Ltd", "Brunai"]})

    path = tmp_path / "llm.jsonl"
    build_llm_cache("m", ["BN"], post_fn=post, path=path, log_path=None, include_junk=False)
    gen = LlmVariants(path=path, version="2")
    assert gen.version == "2"
    out = gen.generate(BaseAlias("Brunei", ("BN",), "llm", "en"), np.random.default_rng(0))
    assert out == ["Brunai"]


def test_llm_v2_drops_another_countrys_code(tmp_path):
    def post(model, prompt):
        return json.dumps({"variants": ["Argentina (US)", "Argentina (AR)", "Argentyna"]})

    path = tmp_path / "llm.jsonl"
    build_llm_cache("m", ["AR"], post_fn=post, path=path, log_path=None, include_junk=False)
    gen = LlmVariants(path=path, version="2")
    out = gen.generate(BaseAlias("Argentina", ("AR",), "llm", "en"), np.random.default_rng(0))
    assert out == ["Argentina (AR)", "Argentyna"]
    rng = np.random.default_rng(0)
    from countrycode.synth.llm import _foreign_code

    assert not _foreign_code("Kansas, US", ("US-KS",)) and not _foreign_code("USA-MT", ("US-MT",))
    assert not _foreign_code("NC (North Carolina)", ("US-NC",)) and rng is not None


def test_llm_v2_skips_shared_island_and_refuses_relabels(tmp_path):
    def post(model, prompt):
        return json.dumps({"variants": ["St. Marten", "Sint Maarten (NL)"]})

    path = tmp_path / "llm.jsonl"
    build_llm_cache("m", ["SX"], post_fn=post, path=path, log_path=None, include_junk=False)
    gen = LlmVariants(path=path, version="2")
    assert (
        gen.generate(BaseAlias("Sint Maarten", ("SX",), "llm", "en"), np.random.default_rng(0))
        == []
    )
    assert gen.allow_relabel is False
