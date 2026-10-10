import json

from countrycode.evaluation.llm import LLMPredictor, parse_answers
from countrycode.real.gold import GoldItem


def g(text, i=0):
    return GoldItem(f"id{i}", "s", "f", "country", "US", text, 1, "dev", ("AT",), "x", "y")


def test_parse_answers_robust():
    raw = json.dumps(
        {
            "answers": [
                {"i": 0, "code": "at", "confidence": 0.9},
                {"i": 1, "code": "ABSTAIN", "confidence": 0.4},
                {"i": 2, "code": "XX", "confidence": 0.9},
                {"i": 3, "code": "US-CA", "confidence": "high"},
            ]
        }
    )
    preds = parse_answers(raw, 5)
    assert (preds[0].target, preds[0].confidence) == ("AT", 0.9)
    assert preds[1].target is None and preds[2].target is None  # abstain / invalid code
    assert preds[3].target == "US-CA" and preds[3].confidence == 0.5  # unparsable confidence
    assert preds[4].target is None  # missing item


def test_parse_garbage_is_all_abstain():
    assert all(p.target is None for p in parse_answers("Sure! Here you go", 3))


def test_predictor_batches_and_caches(tmp_path):
    calls = []

    def post(model, prompt):
        calls.append(prompt)
        n = prompt.count("\n- [")
        return json.dumps(
            {"answers": [{"i": i, "code": "AT", "confidence": 0.8} for i in range(n)]}
        )

    p = LLMPredictor("m", post_fn=post, batch=2, cache_dir=tmp_path)
    items = [g(f"Austria {i}", i) for i in range(5)]
    assert [x.target for x in p.predict(items)] == ["AT"] * 5
    assert len(calls) == 3
    p.predict(items)
    assert len(calls) == 3  # cached
