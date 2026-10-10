from countrycode.real.items import RealItem
from countrycode.real.splits import assign_split, check_splits


def it(text, source="us-co", field="f"):
    return RealItem(source, field, "country", "US", text, 1)


def test_deterministic_and_key_based():
    assert assign_split("us-co", "Delaware") == assign_split("us-co", " DELAWARE.")
    assert assign_split("us-co", "x") == assign_split("us-co", "x")


def test_proportions():
    parts = [assign_split("s", f"string {i}") for i in range(10_000)]
    for part, share in (("dev", 0.3), ("calib", 0.3), ("test", 0.4)):
        assert abs(parts.count(part) / len(parts) - share) < 0.03


def test_same_text_two_fields_same_split():
    a, b = it("CA", field="principalcountry"), it("CA", field="jurisdictonofformation")
    assert a.item_id != b.item_id
    assert assign_split(a.source, a.text) == assign_split(b.source, b.text)


def test_check_detects_leak():
    items = [it("Delaware"), it("DELAWARE")]
    ok = {items[0].item_id: "dev", items[1].item_id: "dev"}
    bad = {items[0].item_id: "dev", items[1].item_id: "test"}
    assert check_splits(items, ok)["within_source_leaks"] == 0
    assert check_splits(items, bad)["within_source_leaks"] == 1
