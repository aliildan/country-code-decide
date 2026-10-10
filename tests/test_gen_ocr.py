import numpy as np

from countrycode.synth.framework import BaseAlias
from countrycode.synth.ocr import OcrVariants, is_garbage, render


def test_render_makes_a_grey_line_image():
    img = render(
        "Austria", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", np.random.default_rng(0)
    )
    assert img.mode == "RGB" and img.width > img.height > 10


def test_garbage_filter():
    assert is_garbage("", "Austria") and is_garbage("#### ::", "Austria")
    assert is_garbage("Hello world peace", "Austria")
    assert not is_garbage("Austrio", "Austria")


def test_variants_come_from_cache_only_when_misread():
    cache = {"Austria": ["Austrio", "Austria", "AUSTRIA"], "Norway": ["Nonvay"]}
    gen = OcrVariants(cache=cache)
    out = gen.generate(BaseAlias("Austria", ("AT",), "name", "en"), np.random.default_rng(0))
    assert out == ["Austrio"]  # identical (case-insensitive) readings are not variants
    assert gen.generate(BaseAlias("Peru", ("PE",), "name", "en"), np.random.default_rng(0)) == []


def test_punctuation_only_readings_are_not_variants():
    gen = OcrVariants(cache={"Ilhas Cocos": ["Ilhas Cocos .", "Illhas Cocos ."]})
    out = gen.generate(BaseAlias("Ilhas Cocos", ("CC",), "name", "pt"), np.random.default_rng(0))
    assert out == ["Illhas Cocos ."]


def test_v2_reads_only_recoverable_misreadings_of_register_names():
    from countrycode.synth.framework import BaseInfo

    info = {a: BaseInfo(frozenset({lang}), frozenset({"name"}), frozenset({t}))
            for a, lang, t in [("Switzerland", "en", "CH"), ("Sviss", "is", "CH"),
                               ("Danmark", "da", "DK")]}  # fmt: skip
    cache = {"Switzerland": ["Swilzerland", "Swilzerlamd .", "suss"], "Sviss": ["Svis"],
             "Danmark": ["dans"]}  # fmt: skip
    gen = OcrVariants(cache=cache, info=info)
    assert gen.version == "2"
    rng = np.random.default_rng(0)
    out = gen.generate(BaseAlias("Switzerland", ("CH",), "name", "en"), rng)
    assert out == ["Swilzerland"]  # 1 edit of 11 letters; 2 edits exceed 0.15 × 11
    assert gen.generate(BaseAlias("Sviss", ("CH",), "name", "is"), rng) == []  # not typed
    assert gen.generate(BaseAlias("Danmark", ("DK",), "name", "da"), rng) == []  # 3 edits of 7
