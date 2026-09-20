#!/usr/bin/env python3
"""The three family-1 registries (E-082 §4) — built against a STUBBED Hub.

    python3 -m pytest -q tests/test_family1_registry.py

`ml/build_family1_registry.py` writes `family1gf.json`, `family1tf.json` and
`family09tf.json`: one entry per adapter of each family, carrying the design
row (tier, layout, cadence, channels, footprint, licence, track, sources, the
note's estimate), the probe's numbers where a probe report exists, and — for a
store that is on the Hub — that store.json's N, record span, builder commit
and per-file sha256.

NOTHING HERE TOUCHES THE NETWORK. `build_family1_registry.http_json` is the
one seam and every test replaces it, which is also how the two answers that
matter are pinned: a 404 means the store is not published (`built: false`) and
a 401 or a 403 means the read was REFUSED and must raise, because a registry
that turned "the Hub would not talk to me" into "that store is not built"
would be making a statement about the archive out of a failure to reach it.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

import build_family10_registry as reg10                         # noqa: E402
import build_family1_registry as r1                             # noqa: E402
from family1.adapters import FAMILIES, REGISTRY                 # noqa: E402

CODES = ("1gf", "1tf", "09tf")


# --------------------------------------------------------------- the stub --
class _Hub:
    """A Hub that serves exactly the store.json files a test hands it."""

    def __init__(self, files=None, code=None):
        self.files = dict(files or {})
        self.code = code
        self.asked = []

    def __call__(self, url, timeout=60):
        path = url.split("/resolve/main/", 1)[-1]
        self.asked.append(path)
        if self.code:
            raise _http_error(url, self.code)
        if path in self.files:
            return self.files[path]
        raise _http_error(url, 404)


def _http_error(url, code):
    import urllib.error
    return urllib.error.HTTPError(url, code, "stub", {}, None)


@pytest.fixture
def stub(monkeypatch):
    """Install a stubbed Hub and hand the test the object to load."""
    def install(hub):
        monkeypatch.setattr(r1, "http_json", hub)
        return hub
    return install


@pytest.fixture(scope="module")
def offline(tmp_path_factory):
    """The three registries, built with the Hub switched off entirely."""
    return r1.build_all(use_hub=False)


# =============================================================== the shape ==
def test_there_are_exactly_three_registries_and_they_are_the_three_families(
        offline):
    assert sorted(offline) == sorted(FAMILIES) == sorted(CODES)
    for code in CODES:
        r = offline[code]
        assert r["family_code"] == code
        assert r["family"] == FAMILIES[code][0]
        assert r["family_version"] == FAMILIES[code][1]
        assert r["registry"].endswith(r1.REGISTRY_NAME[code])
        assert r["hf_root"] == f"tensors/{FAMILIES[code][2]}"
        assert set(r["tiers"]) == {"P", "G", "T"}
        assert r["builder"] == "ml/build_family1_registry.py"


def test_every_registered_adapter_appears_exactly_once_with_the_right_family(
        offline):
    """The `--check` invariant, asserted directly rather than through its
    exit code: the registries and the adapter registry are the same set."""
    where = {}
    for code, r in offline.items():
        for g in r["groups"]:
            where.setdefault(g["name"], []).append(code)
            assert g["family_code"] == code, (g["name"], code)
            assert g["path"] == f"tensors/{FAMILIES[code][2]}/{g['name']}"
    assert sorted(where) == sorted(REGISTRY)
    for store, codes in where.items():
        assert codes == [getattr(REGISTRY[store], "family")], store
    assert r1.check(offline) == []


def test_every_entry_carries_the_design_row_a_reader_dispatches_on(offline):
    for code, r in offline.items():
        for g in r["groups"]:
            ad = REGISTRY[g["name"]]()
            assert g["tier"] in ("P", "G", "T")
            assert g["title"] == ad.title
            assert g["cadence"]
            assert g["layout"]
            assert g["reader"]
            assert g["C"] == len(ad.channels)
            assert [c["name"] for c in g["channels"]] == list(
                ad.channel_names)
            assert g["footprint"]["log2_fp"] == float(ad.log2_fp)
            assert g["footprint"]["log2_dt"] == float(ad.log2_dt)
            assert g["licence"] == dict(ad.licence)
            assert g["sources"] == list(ad.sources)
            assert g["credentials"] == list(ad.credentials or ())
            assert g["distribution"] in ("public", "private")
            assert g["adapter"] == f"ml/family1/adapters/{g['name']}.py"
            # the note's estimate is carried as it stands, never derived
            assert g["note_estimate"] == getattr(ad, "note_estimate", None)


def test_a_catalogue_is_tier_T_in_the_registry_and_tier_P_in_its_layout(
        offline):
    """Both are true and the registry says both: a catalogue's LAYOUT is the
    tier-P column store, and a consumer dispatching on `tier` must not be
    handed scene metadata where it expects observations."""
    tf = offline["1tf"]
    cats = [g for g in tf["groups"] if g["tier"] == "T"]
    assert len(cats) >= 8                       # the eight of wave 3, plus
    for g in cats:
        assert g["catalogue"] is True
        assert g["tier_layout"] == "P"
        assert "assets.parquet" in g["layout"]
        assert [c["name"] for c in g["channels"]][:3] == ["cloud", "valid",
                                                          "angle"]
    for g in tf["groups"]:
        if g["tier"] == "G":
            assert g["catalogue"] is False
            assert g["frames_per_bin"] >= 1
            assert g["frame_seconds"] * g["frames_per_bin"] == 432000
            assert g["dtype"] in ("float16", "uint8")


def test_09tf_inherits_1tf_by_reference_and_lists_only_its_own(offline):
    """0.9.tf is 1.0.tf with the imagery replaced; a COPY of the parent's
    rows is the thing that would go stale."""
    r = offline["09tf"]
    assert r["inherits"] == "family1tf"
    b = r["inherits_block"]
    assert b["family_code"] == "1tf"
    assert b["family_version"] == "1.0.tf"
    assert b["registry"] == "tensors/family1_tf/family1tf.json"
    assert b["groups"] == sorted(g["name"] for g in offline["1tf"]["groups"])
    # its own groups are only what 0.9.tf builds itself — today, none
    own = {g["name"] for g in r["groups"]}
    assert own & set(b["groups"]) == set()
    # and the other two registries do NOT carry an inherits key
    for code in ("1gf", "1tf"):
        assert "inherits" not in offline[code]


# ============================================================== the numbers ==
def test_the_probe_is_summarised_and_the_estimate_is_not_overwritten(offline):
    """Three numbers, kept apart: the note's guess, the probe's measurement
    and the built store (ADAPTER_CONTRACT rule 6)."""
    probed = [g for r in offline.values() for g in r["groups"] if g["probe"]]
    assert probed, "no probe report was picked up at all"
    for g in probed:
        p = g["probe"]
        assert p["file"].startswith("ml/family1/probes/")
        assert os.path.exists(os.path.join(ROOT, p["file"]))
        assert p["month"] and len(p["month"]) == 7
        assert p["tier"] in ("P", "G")
        if p["tier"] == "G":
            assert p["frames_fetched"] >= 1
        else:
            assert p["rows"] >= 1
        # the probe never replaces the design row's own estimate
        ad = REGISTRY[g["name"]]()
        assert g["note_estimate"] == getattr(ad, "note_estimate", None)


def test_the_newest_probe_wins_and_a_prefix_never_steals_one(tmp_path):
    d = str(tmp_path)
    for n in ("tide_2019-03.json", "tide_2021-08.json",
              "tide_private_2019-03.json", "tidex_2024-01.json"):
        with open(os.path.join(d, n), "w") as fh:
            json.dump({"month": n.split("_")[-1][:-5], "rows": 1}, fh)
    assert os.path.basename(r1.probe_path("tide", d)) == "tide_2021-08.json"
    assert os.path.basename(r1.probe_path("tide_private", d)) == \
        "tide_private_2019-03.json"
    assert r1.probe_path("tid", d) is None


def test_a_published_store_contributes_its_checksums_span_and_commit(stub):
    """The Hub half: a store.json that IS there fills in N, the record span,
    the builder's commit and every file's sha256."""
    meta = {"N": 1234, "bin_first": 10, "bin_last": 20, "n_bins": 11,
            "date_range": ["2019-01-01", "2020-12-31"],
            "built_at": "2026-09-01T00:00:00+00:00",
            "builder_git_sha": "abc1234", "schema_version": 2,
            "counts": {"rows_kept": 1234},
            "sha256": {"values.npy": "aa" * 32, "store.json": "bb" * 32}}
    hub = _Hub({"tensors/family1_tf/ghcnd/store.json": meta})
    stub(hub)
    r = r1.build_one("1tf")
    g = [x for x in r["groups"] if x["name"] == "ghcnd"][0]
    assert g["built"] is True
    assert g["N"] == 1234
    assert g["record_span"] == ["2019-01-01", "2020-12-31"]
    assert g["builder_git_sha"] == "abc1234"
    assert g["store_schema_version"] == 2
    assert sorted(f["name"] for f in g["files"]) == ["store.json",
                                                     "values.npy"]
    assert all(len(f["sha256"]) == 64 for f in g["files"])
    # everything else in the family is honestly unbuilt
    assert r["built"] == ["ghcnd"]
    assert r["n_built"] == 1
    assert "ghcnd" not in r["not_built"]
    unbuilt = [x for x in r["groups"] if x["name"] != "ghcnd"][0]
    assert unbuilt["built"] is False
    assert "not published" in unbuilt["built_note"]
    assert "N" not in unbuilt and "files" not in unbuilt


def test_a_private_store_is_listed_without_touching_the_private_repo(stub):
    """A public registry must not carry a private store's file hashes, and a
    private store is not therefore 'not built' — it says which it is."""
    hub = _Hub({})
    stub(hub)
    r = r1.build_one("1tf")
    priv = [g for g in r["groups"] if g["distribution"] == "private"]
    assert priv, "no private-track store in family 1.0.tf"
    for g in priv:
        assert g["repo"] == r1.PRIVATE_REPO
        assert g["built"] is False
        assert "private repository" in g["built_note"]
        assert "files" not in g
        # the design row and the probe are still there
        assert g["channels"] and g["licence"]
    # and not one request was made to the private repository
    assert all("family1_tf" in p or "family1_gf" in p for p in hub.asked)
    assert not any(r1.PRIVATE_REPO in p for p in hub.asked)


def test_the_registry_refuses_to_call_an_unreadable_store_a_missing_one(stub):
    """A 404 is an absence; a 401 or a 403 is a REFUSAL and must raise.

    Family 10's registry learned this the expensive way — it published itself
    with a store listed as missing because the Hub had answered 403 — and the
    guard is copied here deliberately (ml/CLAUDE.md §0.2).
    """
    stub(_Hub({}, code=404))
    assert r1.hub_json("x/y", "a/b.json") is None
    for code in (401, 403):
        stub(_Hub({}, code=code))
        with pytest.raises(IOError) as e:
            r1.hub_json("x/y", "a/b.json")
        assert "refused the read" in str(e.value)
        stub(_Hub({}, code=code))
        with pytest.raises(IOError):
            r1.build_one("1gf")
    stub(_Hub({}, code=500))
    with pytest.raises(Exception):
        r1.hub_json("x/y", "a/b.json")


def test_a_local_store_json_is_preferred_over_the_hub(tmp_path, stub):
    """A box that has just built a store must not have to publish it before
    the registry can describe it."""
    work = str(tmp_path)
    d = os.path.join(work, "ghcnd", "ghcnd")
    os.makedirs(d)
    with open(os.path.join(d, "store.json"), "w") as fh:
        json.dump({"N": 7, "date_range": ["2000-01-01", "2000-01-02"],
                   "sha256": {"store.json": "cc" * 32}}, fh)
    hub = _Hub({})
    stub(hub)
    r = r1.build_one("1tf", work=work)
    g = [x for x in r["groups"] if x["name"] == "ghcnd"][0]
    assert g["built"] is True and g["N"] == 7
    assert "tensors/family1_tf/ghcnd/store.json" not in hub.asked


# ==================================================================== check ==
def test_check_catches_an_adapter_that_is_in_no_registry(offline):
    regs = {k: json.loads(json.dumps(v)) for k, v in offline.items()}
    victim = regs["1tf"]["groups"].pop(0)
    bad = r1.check(regs)
    assert any(victim["name"] in m and "NO registry" in m for m in bad)


def test_check_catches_an_adapter_that_is_in_two(offline):
    regs = {k: json.loads(json.dumps(v)) for k, v in offline.items()}
    dupe = dict(regs["1tf"]["groups"][0])
    dupe["family_code"] = "1gf"
    regs["1gf"]["groups"].append(dupe)
    bad = r1.check(regs)
    assert any("2 registries" in m for m in bad)


def test_check_catches_a_store_that_may_not_be_redistributed_and_is_public(
        offline):
    regs = {k: json.loads(json.dumps(v)) for k, v in offline.items()}
    g = regs["1tf"]["groups"][0]
    g["licence"] = dict(g["licence"], redistribution="no")
    g["distribution"] = "public"
    bad = r1.check(regs)
    assert any("may not be redistributed" in m for m in bad)


def test_the_cli_writes_three_files_and_check_passes(tmp_path, monkeypatch,
                                                     capsys):
    monkeypatch.setattr(sys, "argv",
                        ["build_family1_registry.py", "--no-hub", "--check",
                         "--out-dir", str(tmp_path)])
    assert r1.main() == 0
    names = sorted(os.listdir(str(tmp_path)))
    assert names == sorted(r1.REGISTRY_NAME.values())
    for code, n in r1.REGISTRY_NAME.items():
        d = json.load(open(os.path.join(str(tmp_path), n)))
        assert d["family_code"] == code
    out = capsys.readouterr().out
    assert "CHECK OK" in out
    assert f"{len(REGISTRY)} adapter(s)" in out


# ============================================== the sibling line in family 10 =
def test_family_10_2_names_the_three_registries_family_1_really_writes():
    """The one additive key on family 10.2's side. Its three names are the
    contract between the two builders, so they are asserted against what
    `build_family1_registry` actually writes rather than against a copy."""
    want = {f"tensors/{FAMILIES[c][2]}/{r1.REGISTRY_NAME[c]}" for c in CODES}
    got = {s["registry"] for s in reg10.SIBLING_REGISTRIES}
    assert got == want
    assert {s["family_version"] for s in reg10.SIBLING_REGISTRIES} == \
        {FAMILIES[c][1] for c in CODES}
    # and the block family 1's own builder offers agrees with it
    mine = {s["registry"] for s in r1.siblings_block()["registries"]}
    assert mine == want


def test_family_10_2_registry_carries_the_siblings_block(tmp_path):
    """Built with no Hub and no stores, so the test is about the block."""
    r = reg10.build_registry(work=str(tmp_path), use_hub=False, stores=(),
                             include_argo=False)
    sib = r["siblings"]
    assert [s["registry"] for s in sib["registries"]] == \
        [s["registry"] for s in reg10.SIBLING_REGISTRIES]
    assert "ml/build_family1_registry.py" in sib["note"]
    # ADDITIVE: everything family 10's registry already carried is still there
    for k in ("family", "family_version", "schema_version", "inherits",
              "tier_g", "groups", "token_schema", "tiers", "readers"):
        assert k in r, k
