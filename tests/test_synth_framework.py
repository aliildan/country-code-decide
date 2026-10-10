import json

import numpy as np

from countrycode.lexicon.build import build_lexicon, code_rows
from countrycode.lexicon.types import AliasKind, AliasRow, Overrides
from countrycode.synth.framework import BaseAlias, base_aliases, generate_items, run_release


def R(a: str, t: str, lang: str = "en", kind: AliasKind = "name") -> AliasRow:
    return AliasRow(a, t, kind, lang, "t")


LX = build_lexicon(
    [
        R("Austria", "AT"),
        R("Australia", "AU"),
        R("Iran", "IR"),
        R("Iraq", "IQ"),
        R("Georgia", "GE"),
        R("Georgia", "US-GA"),
        *code_rows(),
    ],
    Overrides(),
)


class Upper:
    name, version = "upper", "1"
    config = {"x": 1}

    def generate(self, row: BaseAlias, rng: np.random.Generator) -> list[str]:
        return [row.alias.upper(), row.alias + "x"]


def test_base_aliases_use_resolved_sets():
    bases = {b.alias: b for b in base_aliases(LX)}
    assert bases["Georgia"].targets == ("GE", "US-GA")
    assert bases["Austria"].targets == ("AT",)


def test_release_is_reproducible_and_filters_leaks(tmp_path):
    real_keys = {"austriax"}  # a real test string equal to one variant → must be removed
    m1 = run_release([Upper()], LX, real_keys, tmp_path / "a", seed=7)
    m2 = run_release([Upper()], LX, real_keys, tmp_path / "b", seed=7)
    f1 = (tmp_path / "a" / "upper@1.jsonl").read_text()
    assert f1 == (tmp_path / "b" / "upper@1.jsonl").read_text()
    texts = [json.loads(line)["text"] for line in f1.splitlines()]
    assert "Austriax" not in texts and "AUSTRIA" in texts
    assert m1["generators"]["upper@1"]["dropped"]["leak"] >= 1
    item = json.loads(f1.splitlines()[0])
    assert {
        "text",
        "targets",
        "generator",
        "version",
        "seed",
        "source_alias",
        "config_hash",
    } <= set(item)
    assert item["label_source"] == "synthetic:upper@1"
    assert m1 == m2


def test_parallel_release_equals_serial(tmp_path):
    from countrycode.synth.keyboard import KeyboardTypos

    gens = [KeyboardTypos(per_alias=2)]
    run_release(gens, LX, set(), tmp_path / "s", seed=3, workers=1)
    run_release(gens, LX, set(), tmp_path / "p", seed=3, workers=3)
    a = (tmp_path / "s" / "keyboard@1.jsonl").read_text()
    assert a and a == (tmp_path / "p" / "keyboard@1.jsonl").read_text()


def test_manifest_merges_generators_across_runs(tmp_path):
    from countrycode.synth.keyboard import KeyboardTypos

    run_release([Upper()], LX, set(), tmp_path, seed=3)
    m = run_release([KeyboardTypos(per_alias=1)], LX, set(), tmp_path, seed=3)
    assert set(m["generators"]) == {"upper@1", "keyboard@1"}


def test_base_kind_prefers_code():
    lx = build_lexicon([R("AT", "AT", "en", "name"), *code_rows()], Overrides())
    bases = {b.alias: b for b in base_aliases(lx)}
    assert bases["AT"].kind == "code"


def test_generator_can_refuse_labels_changed_by_an_exact_hit():
    from countrycode.synth.gates import CollisionChecker

    class Caps:
        name, version, config = "caps", "1", {}
        allow_relabel = False

        def generate(self, row, rng):
            return [row.alias.upper()]

    items, dropped = generate_items(
        Caps(), [BaseAlias("Georgia", ("GE",), "name", "en")], set(), 1, CollisionChecker(LX)
    )
    assert items == [] and dropped["relabel_exact"] == 1  # GEORGIA → {GE, US-GA}


def test_generator_can_switch_off_containment():
    from countrycode.lexicon.types import AliasRow as AR
    from countrycode.synth.gates import CollisionChecker

    rows = [AR("Jordan", "JO", "name", "en", "t"), AR("Utah", "US-UT", "name", "en", "t")]
    lx = build_lexicon([*rows, *code_rows()], Overrides())
    chk = CollisionChecker(lx, margin=10, containment=True, reject_none_mix=True)

    class Geo:
        name, version, config = "geo", "9", {}
        near_check = False

        def generate(self, row, rng):
            return ["West Jordan, Utah"]

    base = [BaseAlias("__geo_US-UT", ("US-UT",), "geo", "en")]
    assert generate_items(Geo(), base, set(), 1, chk)[0] == []
    Geo.containment_check = False
    assert [i.text for i in generate_items(Geo(), base, set(), 1, chk)[0]] == ["West Jordan, Utah"]


def test_release_manifest_records_each_files_sha256(tmp_path):
    import hashlib

    m = run_release([Upper()], LX, set(), out_dir=tmp_path, seed=1)
    tag = "upper@1"
    assert (
        m["generators"][tag]["sha256"]
        == hashlib.sha256((tmp_path / f"{tag}.jsonl").read_bytes()).hexdigest()
    )
