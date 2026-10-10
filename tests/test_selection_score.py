from countrycode.train.trainer import selection_score

REP = {
    "overall": {"accuracy": 0.62},
    "slices": {
        "source=gbif": {"n": 900, "accuracy": 0.6},
        "source=us-co": {"n": 60, "accuracy": 0.9},
        "source=gb-ch": {"n": 40, "accuracy": 0.8},
        "field_kind=country": {"n": 1000, "accuracy": 0.62},
    },
}


def test_selection_score_default_is_item_accuracy():
    assert selection_score(REP, "accuracy") == 0.62


def test_domain_macro_weights_register_and_curated_equally():
    # register = (60·0.9 + 40·0.8) / 100 = 0.86; curated = 0.6 → mean 0.73
    assert abs(selection_score(REP, "domain_macro") - 0.73) < 1e-9
