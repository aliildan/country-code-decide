from countrycode.train.dataset import (
    COUNTRY_CLASSES,
    NO_STATE,
    STATE_CLASSES,
    TrainExample,
    build_examples,
    context_text,
    expand,
    label_dists,
)


def test_class_spaces():
    assert len(COUNTRY_CLASSES) == 251 and COUNTRY_CLASSES[-1] == "NONE"
    assert len(STATE_CLASSES) == 52 and STATE_CLASSES[-1] == NO_STATE


def test_label_dists():
    c, s = label_dists(("GE", "US-GA"))
    assert c == {"GE": 0.5, "US": 0.5} and s == {"US-GA": 1.0}
    c, s = label_dists(("US",))
    assert c == {"US": 1.0} and s == {NO_STATE: 1.0}
    c, s = label_dists(("AT",))
    assert c == {"AT": 1.0} and s is None
    c, s = label_dists(("US-NC", "US-SC"))
    assert c == {"US": 1.0} and s == {"US-NC": 0.5, "US-SC": 0.5}


def test_collision_expands_to_three_contexts():
    import numpy as np

    ex = expand("CA", ("CA", "US-CA"), "lexicon", np.random.default_rng(0))
    by_ctx = {(e.field, e.register): e for e in ex}
    assert by_ctx[("jurisdiction", "US")].country == {"US": 1.0}
    assert by_ctx[("jurisdiction", "US")].state == {"US-CA": 1.0}
    country_ex = [e for e in ex if e.field == "country"]
    assert all(e.country == {"CA": 1.0} for e in country_ex)
    assert {e.register == "US" for e in country_ex} == {True, False}  # US registers' country
    # fields mean countries too (Colorado principalcountry "CA" = Canada)
    soft = by_ctx[("unknown", "unknown")]
    assert soft.country == {"CA": 0.5, "US": 0.5}


def test_plain_item_gets_one_sampled_context():
    import numpy as np

    ex = expand("Østerrike", ("AT",), "lexicon", np.random.default_rng(0))
    assert len(ex) == 1 and isinstance(ex[0], TrainExample) and ex[0].country == {"AT": 1.0}


def test_context_text():
    assert context_text("jurisdiction", "US") == "field: jurisdiction | register: US"


def test_data_check_flags_real_strings_in_synthetic():
    from countrycode.train.dataset import TrainExample, data_check

    ex = [
        TrainExample(
            "Deleware", "country", "US", {"US": 1.0}, {"US-DE": 1.0}, "synthetic:keyboard@1"
        ),
        TrainExample("Delaware", "country", "US", {"US": 1.0}, {"US-DE": 1.0}, "lexicon"),
    ]
    rep = data_check(ex, real_keys={"deleware", "delaware"})
    assert rep["synthetic_equal_to_real"] == 1 and rep["ok"] is False
    assert rep["lexicon_equal_to_real"] == 1  # allowed: reference names, tagged "seen" in eval


def test_codes_get_every_context():
    from countrycode.train.dataset import build_examples, rules

    ex = build_examples([("BY", ("BY",), "lexicon:code")], seed=1)
    assert len(ex) == len(rules()["contexts"])
    assert len({(e.field, e.register) for e in ex}) >= 5 and all(
        e.country == {"BY": 1.0} for e in ex
    )


def test_collision_contexts_are_balanced_between_state_and_country():
    import numpy as np

    ex = expand("CA", ("CA", "US-CA"), "lexicon", np.random.default_rng(0))
    states = [e for e in ex if e.state == {"US-CA": 1.0} and e.country == {"US": 1.0}]
    countries = [e for e in ex if e.country == {"CA": 1.0}]
    assert len(states) == len(countries) >= 2


def test_explicit_context_items_keep_their_field():
    from countrycode.train.dataset import build_examples

    ex = build_examples([("Deleware", ("US-DE",), "real:human", "jurisdiction", "US")], seed=1)
    assert len(ex) == 1 and (ex[0].field, ex[0].register) == ("jurisdiction", "US")
    assert ex[0].country == {"US": 1.0} and ex[0].state == {"US-DE": 1.0}


def test_us_register_is_sometimes_a_state_register():
    import numpy as np

    from countrycode.train.dataset import _register

    rng = np.random.default_rng(0)
    regs = {_register("US", rng) for _ in range(200)}
    assert "US" in regs and any(r.startswith("US-") for r in regs)


def test_place_names_take_the_registers_own_place():
    # owner policy 2026-10-10: a place shared by several jurisdictions is the set, but in a
    # register of one of them it is that one ("Liverpool" in a GB register → GB)
    ex = build_examples([("Liverpool", ("AU", "CA", "GB"), "synthetic:city@2")], seed=1)
    by_reg = {(e.field, e.register): e.country for e in ex}
    assert by_reg[("country", "GB")] == {"GB": 1.0}
    assert by_reg[("country", "AU")] == {"AU": 1.0}
    assert any(set(e.country) == {"AU", "CA", "GB"} for e in ex)  # context-free reading
    for e in ex:  # never the set in one of its own registers
        if e.register in ("AU", "CA", "GB"):
            assert e.country == {e.register: 1.0}


def test_county_names_take_the_registers_own_state():
    ex = build_examples([("Adams County", ("US-CO", "US-IL"), "synthetic:county@1")], seed=1)
    co = [e for e in ex if e.register == "US-CO"]
    assert co and all(e.state == {"US-CO": 1.0} for e in co)
    assert any(e.state == {"US-CO": 0.5, "US-IL": 0.5} for e in ex)


def test_non_place_sets_ignore_the_register():
    ex = build_examples([("France/Germany", ("DE", "FR"), "synthetic:pairs@2")], seed=1)
    assert all(set(e.country) == {"DE", "FR"} for e in ex)


def test_data_check_flags_real_training_strings_in_evaluation_roles():
    from countrycode.train.dataset import TrainExample, data_check

    ex = [
        TrainExample("Colrado", "jurisdiction", "US-CO", {"US": 1.0}, {"US-CO": 1.0}, "real:human"),
        TrainExample("Denvr", "jurisdiction", "US-CO", {"US": 1.0}, {"US-CO": 1.0}, "real:human"),
    ]
    rep = data_check(ex, real_keys={"colrado", "denvr"}, eval_keys={"denvr"})
    assert rep["real_train_in_eval"] == 1 and rep["ok"] is False
    assert data_check(ex, real_keys={"colrado", "denvr"}, eval_keys=set())["ok"] is True


def test_canadian_province_codes_are_canada_in_a_us_jurisdiction_field():
    from countrycode.train.dataset import build_examples

    ex = build_examples([("NL", ("NL",), "lexicon:code")], seed=3)
    for e in ex:
        us_juris = e.field == "jurisdiction" and e.register.startswith("US")
        assert e.country == ({"CA": 1.0} if us_juris else {"NL": 1.0})
    assert any(e.field == "jurisdiction" and e.register.startswith("US") for e in ex)


def test_a_source_can_fix_its_contexts():
    from countrycode.train.dataset import build_examples

    ex = build_examples([("ON", ("CA",), "synthetic:caprov@1")], seed=5)
    assert ex and all(e.field == "jurisdiction" and e.register.startswith("US") for e in ex)
