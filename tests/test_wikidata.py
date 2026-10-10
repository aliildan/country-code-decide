import json
from pathlib import Path

from countrycode.lexicon.types import Overrides
from countrycode.lexicon.wikidata import build_queries, parse_bindings

FIX = json.loads(Path("tests/fixtures/wikidata_bindings.json").read_text(encoding="utf-8"))


def test_batches():
    qs = build_queries([f"C{i}" for i in range(250)], batch=25)
    assert len(qs) == 10 and all("VALUES" in q for q in qs)


def test_query_filters_ended_statements():
    q = build_queries(["AT"])[0]
    assert "pq:P582" in q and "wdt:P576" in q


def test_parse_rows():
    rows = parse_bindings(FIX, historical={"YUCS": ["RS", "ME"]}, extra={"Q21": "GB"})
    s = {(r.alias, r.target, r.kind, r.lang) for r in rows}
    assert ("Österreich", "AT", "name", "de") in s
    assert ("AUT", "AT", "code", "zxx") in s
    assert ("österreichisch", "AT", "demonym", "de") in s
    assert ("Kalifornien", "US-CA", "name", "de") in s
    assert ("England", "GB", "name", "en") in s
    hist = {(r.alias, r.target) for r in rows if r.kind == "historical"}
    assert {("Jugoslawien", "RS"), ("Jugoslawien", "ME")} <= hist


def test_overrides_load():
    ov = Overrides.load(Path("configs/lexicon/overrides.yaml"))
    assert ov.extra_items["Q21"] == "GB"
    assert set(ov.historical["CSHH"]) == {"CZ", "SK"}
    xk = {(r.alias, r.target) for r in ov.add}
    assert {("XKX", "XK"), ("RKS", "XK"), ("Kosova", "XK"), ("RS-KM", "XK")} <= xk
