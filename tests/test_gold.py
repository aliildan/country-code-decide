import json

from countrycode.lexicon.build import build_lexicon, code_rows
from countrycode.lexicon.types import AliasRow, Overrides
from countrycode.real.gold import auto_label, build_queue, load_gold, proposals
from countrycode.real.items import FieldKind, RealItem


def R(a, t, lang="en"):
    return AliasRow(a, t, "name", lang, "test")


ROWS = [R("Delaware", "US-DE"), *[R("Jersey", "JE", f"l{i}") for i in range(20)]]
LX = build_lexicon([*ROWS, R("Jersey", "US-NJ"), *code_rows()], Overrides())


def it(text: str, count: int = 1, source: str = "us-co", kind: FieldKind = "jurisdiction"):
    return RealItem(source, "f", kind, "US", text, count)


def test_auto_label_only_raw_unique():
    assert auto_label(it("Delaware"), LX) == ["US-DE"]
    assert auto_label(it("Jersey"), LX) is None  # resolved by rules → owner decides
    assert auto_label(it("CA"), LX) is None  # code collision
    assert auto_label(it("Deleware"), LX) is None


def test_proposals_include_fuzzy_match():
    props = proposals(it("Deleware"), LX)
    assert props[0][0] == ("US-DE",)


def test_queue_order_dev_calib_first_then_count():
    items = [it(f"x{i}", count=i) for i in range(30)]
    manifest = {x.item_id: ("test" if i % 2 else "dev") for i, x in enumerate(items)}
    q = build_queue(items, manifest, LX, presets={}, audit_auto=0, seed=0)
    splits = [manifest[x.item.item_id] for x in q]
    assert splits == sorted(splits, key=lambda s: s == "test")
    devs = [x.item.count for x in q if manifest[x.item.item_id] == "dev"]
    assert devs == sorted(devs, reverse=True)


def test_owner_answer_overrides_auto(tmp_path):
    items = [it("Delaware"), it("Deleware")]
    manifest = {x.item_id: "dev" for x in items}
    owner = tmp_path / "owner.jsonl"
    owner.write_text(
        json.dumps({"item_id": items[0].item_id, "targets": ["US-DE"], "ts": "t"})
        + "\n"
        + json.dumps({"item_id": items[1].item_id, "targets": ["US-DE"], "ts": "t"})
        + "\n"
    )
    gold = {g.item_id: g for g in load_gold(items, manifest, LX, presets={}, owner_path=owner)}
    assert gold[items[0].item_id].label_source == "human"
    assert gold[items[1].item_id].targets == ("US-DE",)
    assert gold[items[1].item_id].annotation_mode == "reviewed_proposal"


def test_unlabelled_items_are_pending(tmp_path):
    items = [it("Deleware")]
    gold = load_gold(items, {items[0].item_id: "dev"}, LX, presets={}, owner_path=tmp_path / "x")
    assert gold == []


def test_preset_labels_are_country_granularity(tmp_path):
    items = [RealItem("gbif", "countryName", "country", "unknown", "TEXAS", 1)]
    presets = {items[0].item_id: {"item_id": items[0].item_id, "targets": ["US"]}}
    gold = load_gold(
        items, {items[0].item_id: "dev"}, LX, presets=presets, owner_path=tmp_path / "x"
    )
    assert gold[0].granularity == "country"


def test_presets_carry_their_own_label_source(tmp_path):
    items = [RealItem("no-brreg", "forretningsadresse.land", "country", "NO", "Tyskland", 3410)]
    presets = {
        items[0].item_id: {
            "item_id": items[0].item_id,
            "targets": ["DE"],
            "label_source": "register_code",
            "granularity": "full",
        }
    }
    gold = load_gold(
        items, {items[0].item_id: "dev"}, LX, presets=presets, owner_path=tmp_path / "x"
    )
    assert gold[0].label_source == "register_code" and gold[0].granularity == "full"
    assert gold[0].annotation_mode == "preset"
