from countrycode.real.gold import GoldItem
from countrycode.real.roles import ROLES, role_of


def g(source, split, text="x", mode="reviewed_proposal"):
    return GoldItem("i", source, "f", "country", "US", text, 1, split, ("AT",), "human", mode)


def test_roles_v11():
    assert role_of(g("us-co", "dev")) == "train"
    assert role_of(g("gbif", "dev")) == "select"
    assert role_of(g("gbif", "calib")) == "calib"
    assert role_of(g("us-co", "test")) == "test"
    assert role_of(g("us-co", "test", mode="auto_exact")) == "test_seen_v1"  # read in v1
    assert role_of(g("gbif", "test")) == "ood"
    halves = {role_of(g("us-ct", "calib", text=f"t{i}")) for i in range(40)}
    assert halves == {"select", "calib"}
    assert set(ROLES) >= {"train", "select", "calib", "test", "ood", "test_seen_v1"}


def test_real_train_items_never_share_a_key_with_other_roles():
    from countrycode.real.roles import real_training_items

    gold = [
        g("us-co", "dev", text="Deleware"),  # train
        g("us-ct", "dev", text="ENGLAND"),  # train, but the same key is a test item below
        g("gb-ch", "test", text="England"),  # test (owner)
        g("gbif", "dev", text="Austria"),  # select
    ]
    out = real_training_items(gold)
    assert [x.text for x in out] == ["Deleware"]


def test_role_split_and_v11_test_guard(tmp_path):
    import pytest

    from countrycode.evaluation.harness import Prediction, evaluate
    from countrycode.real.roles import with_roles

    gold = [g("us-co", "test", text="Deleware"), g("gbif", "dev", text="Austria")]
    rs = with_roles(gold)
    assert {x.split for x in rs} == {"test_v11", "select"}

    class P:
        name = "p"

        def predict(self, items):
            return [Prediction("AT", ("AT",), 0.9) for _ in items]

    with pytest.raises(PermissionError):
        evaluate(
            P(), rs, "test_v11", out_dir=None, allow_test=True, prereg_path=tmp_path / "missing.md"
        )
    pre = tmp_path / "pre.md"
    pre.write_text("Status: frozen\n")
    rep = evaluate(P(), rs, "test_v11", out_dir=None, allow_test=True, prereg_path=pre)
    assert rep["overall"]["n"] == 1


def test_held_out_sources_are_test_v12_whatever_the_split():
    from dataclasses import replace

    from countrycode.real.roles import role_of

    base = GoldItem("i", "us-ny", "jurisdiction", "jurisdiction", "US-NY", "DELAWARE", 3, "dev",
                    ("US-DE",), "reference_list", "auto_exact")  # fmt: skip
    for split in ("dev", "calib", "test"):
        assert role_of(replace(base, split=split)) == "test_v12"
    assert role_of(replace(base, source="us-co", register_country="US-CO")) == "train"
