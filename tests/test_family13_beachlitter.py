#!/usr/bin/env python3
"""The `beachlitter` adapter's smoke (family 1.3, E-092 D13) — no network.

    python3 -m pytest -q tests/test_family13_beachlitter.py

The synthetic archive (`ml/family1/adapters/_beachlitter_smoke.py`) writes
every source in its real on-disk layout, mirrored by URL: the NCEI Hub CSV
with its real header, accession 0259486's workbook behind its landing page
and listings, the EMODnet beach-litter zip, the MDMAP site list and one export
zip per site, and four tiny Natural Earth layers. Every expected count below
is exact. The plan's §6 falsifiers are tested as REFUSALS on corrupted
copies of the same archive.
"""
import json
import math
import os
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

pytest.importorskip("openpyxl")
pytest.importorskip("scipy")

import build_family1_stores as b1                               # noqa: E402
import family10_store as f10                                    # noqa: E402
from family1 import adapters as fam                             # noqa: E402
from family1.adapters import _beachlitter_smoke as sm           # noqa: E402
from family1.adapters import beachlitter as bl                  # noqa: E402

# kept rows: the bulk patrols + 9 special patrols (11 written, one with no
# accession copy, one of zero minutes) + 2 sand cores + 8 EMODnet surveys
# (101, 102, 103, 104, 106, 110, 111, 112) + 10 MDMAP surveys (1001, 1002,
# 1101, 1201, 1203, 1301, 1501, 1601, 1701, 1801)
N_NURDLE = sm.NURDLE_BULK + 9
TRUTH_ROWS = N_NURDLE + 2 + 8 + 10


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("beachlitter"))
    res = b1.run_smoke("beachlitter", root=root, keep=True)
    sdir = os.path.join(root, "work", "beachlitter", "beachlitter")
    meta = json.load(open(os.path.join(sdir, "store.json")))
    st = f10.Store(sdir)
    return res, meta, st


def _ctx(root):
    return types.SimpleNamespace(source_dir=root, count_bytes=lambda n: None,
                                 a=types.SimpleNamespace(attempts=1),
                                 prog=types.SimpleNamespace(
                                     item=lambda *a: None))


def test_registered():
    assert fam.REGISTRY["beachlitter"] is bl.BeachLitterAdapter
    ad = bl.BeachLitterAdapter()
    assert ad.family == "13" and ad.C == 14 and ad.time_dtype == "int32"
    assert not ad.per_year and ad.first_year == 2001
    assert ad.channel_names == [
        "log_items_100m", "log_plastic_100m", "log_ambig_100m",
        "plastic_frac", "log_items_m2", "accum_days", "survey_len_m",
        "log_nurdles_10pmin", "log_nurdle_count", "collect_min", "volunteers",
        "effort_pmin", "new_tide_line", "log_n_m3_sand"]
    assert ad.licence["redistribution"] == "attribution"
    # MDMAP has no posted licence (plan D13): a public publish is refused
    # unless the unconfirmed licence is explicitly allowed
    assert ad.licence["redistribution_confirmed"] is False
    for who in ("NCEI", "0259486", "EMODnet", "MDMAP", "Natural Earth"):
        assert who in ad.licence["attribution"]
    lay = b1.layout_for(ad)
    assert lay.hf_root == "tensors/family1_3"
    # every qc bit is documented in the policy string
    for k, v in bl.PROTOCOLS.items():
        assert f"{k} {v}" in ad.qc_policy
    for b in ("bit 4 (16)", "bit 5 (32)", "bit 6 (64)", "bit 7 (128)"):
        assert b in ad.qc_policy
    # the effort channel stays inside float16
    assert ad.bounds()[1][bl.COL["effort_pmin"]] < 65504


def test_smoke_all_stages_and_the_probe(smoke):
    res, meta, st = smoke
    assert len(res["truth"]) == TRUTH_ROWS
    assert st.N == TRUTH_ROWS
    p = res["probe"]
    # June 2019: the 30 bulk patrols of the month and both sand cores
    assert p["rows"] == 30 + 2
    assert p["bytes_fetched"] > 0
    assert sum(p["out_of_bounds_stored"].values()) == 0


def test_the_ledger_counts_every_drop_by_name(smoke):
    _res, meta, _st = smoke
    c = meta["counts"]
    assert c["read"] == {"nurdle": N_NURDLE + 2, "ncei_sand": 2,
                         "emodnet": 11, "mdmap": 12}
    assert c["ncei_marine_setting"] == {"Nurdle Patrol": N_NURDLE + 2,
                                        "Beach": 2, "Ocean water": 1,
                                        "Ocean sediment": 1}
    assert c["nurdle_join"] == {"exact": N_NURDLE - 1,
                                "longitude_sign_differs": 1,
                                "position_rounding": 1, "unmatched": 1}
    assert c["nurdle_accession_rows_unjoined"] == 1
    assert c["nurdle_standardisation"]["zero_minutes"] == 1
    assert c["nurdle_standardisation"]["neither"] == 1
    assert c["nurdle_standardisation"]["floor_or_ceil"] == 1
    assert c["excluded_licence"] == {"LS": 1, "UN": 1}
    assert c["emodnet_dropped"] == {"no_litter_lines": 1,
                                    "survey_length_zero": 1,
                                    "survey_type Cleaning": 1}
    assert c["emodnet_length_assumed"] == 1
    assert c["dedup"] == {"emodnet > mdmap": 1}
    assert c["mdmap_dropped"] == {"duplicate_transect_row": 1,
                                  "accumulation_site_length_zero": 1}
    assert c["mdmap_site_list_vs_export"] == [
        {"site": 10, "site_list": 3, "export": 2}]
    assert c["mdmap_backbarrier_rows_excluded"] == 3
    assert c["mdmap_width_not_usable"] == {"sentinel_9999": 2, "zero": 1,
                                           "blank": 1}
    assert c["mdmap_position_from"] == {"waters_edge": 10, "one_end": 1,
                                        "back_barrier": 1}
    assert c["mdmap_accumulation_first_survey"] == 3
    assert c["rows_no_value"] == 1                 # the 0-length accumulation
    assert c["kept_by_source"] == {"nurdle": N_NURDLE, "ncei_sand": 2,
                                   "emodnet": 8, "mdmap": 10}
    assert c["freshwater_by_source"] == {"nurdle": 1, "mdmap": 1}
    assert c["far_from_any_mapped_shore_by_source"] == {"nurdle": 1}
    assert c["mdmap_material_check"]["failures"] == 0


def test_qc_bits_and_values(smoke):
    _res, _meta, st = smoke
    qc = np.asarray(st["qc"])
    v = np.asarray(st["values"], np.float32)
    proto = qc & 15
    # every row is a date placed at noon
    assert ((qc & bl.QC_NO_TIME) != 0).all()
    assert ((np.asarray(st.time_s()) % 86400) == 43200).all()
    assert int((proto == 11).sum()) == N_NURDLE
    assert int((proto == 12).sum()) == 2
    # the two flagged nurdle caveats and positions, the freshwater rows
    assert int(((proto == 11) & ((qc & bl.QC_CAVEAT) != 0)).sum()) == 1
    assert int(((proto == 11) & ((qc & bl.QC_POSITION) != 0)).sum()) == 2
    assert int(((qc & bl.QC_FRESH) != 0).sum()) == 2
    # a measured zero is 0, never NaN; a channel a protocol lacks is NaN
    nur = proto == 11
    assert (v[nur, bl.COL["log_nurdle_count"]] == 0).any()
    assert np.isnan(v[nur, bl.COL["log_items_100m"]]).all()
    lit = proto <= 10
    assert np.isnan(v[lit, bl.COL["log_nurdles_10pmin"]]).all()
    # the all-zero EMODnet survey: densities 0, plastic fraction undefined
    z = lit & (v[:, bl.COL["log_items_100m"]] == 0)
    assert int(z.sum()) == 1 and np.isnan(v[z, bl.COL["plastic_frac"]]).all()
    # the accumulation interval of 31 days survives the 0-length survey
    acc = v[:, bl.COL["accum_days"]]
    assert sorted(acc[np.isfinite(acc)].tolist()) == [31.0]
    # recomputed rate: 120 pellets, 3 people, 20 minutes = 20 per 10 p-min
    k = np.isclose(v[:, bl.COL["log_nurdle_count"]], math.log10(121),
                   atol=2e-3)
    assert k.sum() == 1
    assert v[k, bl.COL["log_nurdles_10pmin"]][0] == pytest.approx(
        math.log10(21), abs=2e-3)
    assert v[k, bl.COL["effort_pmin"]][0] == 60.0
    assert v[k, bl.COL["new_tide_line"]][0] == 1.0


def test_the_materials_table_is_whole_and_consistent():
    table, sizes = bl.load_materials()
    doc = json.load(open(bl.MATERIALS_JSON, encoding="utf-8"))
    assert sizes == {"ITA": 59, "JLIST": 183, "OSPAR": 128, "OSPAR_1000": 25,
                     "TSG-ML": 216, "UNEP": 77, "UNEP_MARLIN": 82}
    assert set(doc["references"]) == set(sizes)
    for r in doc["codes"]:
        assert r["rule"], r
    # the reference lists' own blocks
    assert table[("TSG-ML", "G1")][0] == "plastic"
    assert table[("TSG-ML", "G124")][0] == "plastic"
    assert table[("TSG-ML", "G125")][0] == "non_plastic"
    assert table[("OSPAR", "46")][0] == "plastic"
    assert table[("OSPAR", "91")][0] == "non_plastic"
    assert table[("UNEP", "FP01")][0] == "plastic"
    # what the list does not decide is never plastic
    for k in (("OSPAR", "64"), ("ITA", "IT32"), ("OSPAR", "111"),
              ("UNEP_MARLIN", "OT02")):
        assert table[k][0] == "ambiguous", k
    # every list's own cigarette-butt convention
    assert table[("TSG-ML", "G27")][0] == "plastic"
    assert table[("JLIST", "J27")][0] == "plastic"
    assert table[("UNEP", "PL11")][0] == "plastic"


@pytest.mark.parametrize("corrupt,match", [
    ("nurdle_rule", "changed its standardisation"),
    ("emodnet_lines", "type lines of list OSPAR"),
    ("emodnet_unknown_code", "is not in beachlitter_emodnet_materials"),
    ("mdmap_sum", "material totals sum to"),
])
def test_the_falsifiers_refuse(tmp_path, corrupt, match):
    root = str(tmp_path)
    sm.write_all(root, corrupt=corrupt)
    with pytest.raises(bl.FormatError, match=match):
        bl.BeachLitterAdapter().build(_ctx(root))


def test_the_shore_tests():
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "LineString",
                                         "coordinates": [[0, 0], [0, 10]]}}]}
    lake = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [
            [[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]]]}}]}
    river = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "MultiLineString",
                                         "coordinates": [[[5, 5], [5, 6]]]}}]}
    sh = bl.Shores({"sea": [gj], "lake": [lake], "river": [river]})
    lat = np.array([5.0, 0.5, 5.5, 12.0])
    lon = np.array([0.0, 1.0, 5.01, -5.0])
    fresh, far, (sea, _l, _r) = sh.flags(lat, lon)
    assert sea[0] < 0.5                          # on the coast
    assert fresh.tolist() == [False, True, True, False]
    assert far.tolist() == [False, False, False, True]


def test_dedup_only_between_litter_sources():
    r = [{"src": "nurdle", "seq": 0, "day": (2020, 1, 1), "lat": 1.0,
          "lon": 1.0},
         {"src": "emodnet", "seq": 1, "day": (2020, 1, 1), "lat": 1.0,
          "lon": 1.0},
         {"src": "mdmap", "seq": 2, "day": (2020, 1, 1), "lat": 1.005,
          "lon": 1.0},
         {"src": "mdmap", "seq": 3, "day": (2020, 1, 2), "lat": 1.0,
          "lon": 1.0}]
    c = {}
    kept = bl.dedup(r, c)
    assert [x["seq"] for x in kept] == [0, 1, 3]
    assert c["dedup"] == {"emodnet > mdmap": 1}
