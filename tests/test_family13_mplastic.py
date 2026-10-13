#!/usr/bin/env python3
"""The `mplastic` adapter's smoke (family 1.3, E-092) — no network.

    python3 -m pytest -q tests/test_family13_mplastic.py

The synthetic archive (`ml/family1/adapters/_mplastic_smoke.py`) writes every
source in its real on-disk layout, mirrored by URL: both figshare items (the
2018 .xls files with float-noise event ids and the Megaplastics sheet's
latitude/longitude swapped), the NCEI Hub CSV with its real header, three NCEI
accessions behind their landing pages and archive listings, and the EMODnet
zip with the real ODV header. Every expected count below is exact.

Needs `xlwt` (to WRITE .xls; the build itself only reads them).
"""
import collections
import datetime as dt
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

pytest.importorskip("xlwt")
pytest.importorskip("xlrd")
pytest.importorskip("openpyxl")

import build_family1_stores as b1                               # noqa: E402
import family10_store as f10                                    # noqa: E402
from family1 import adapters as fam                             # noqa: E402
from family1.adapters import _lebreton2018 as lb                # noqa: E402
from family1.adapters import _mplastic_smoke as sm              # noqa: E402
from family1.adapters import mplastic as mp                     # noqa: E402

# kept rows: 683 events + the 3,031 compiled tows that are not 'This study'
# + 16 of the 2024 rows (9 NPM3 manta with a value, 2 NPM5 large net, 3
# aircraft, 2 cleanup) + 18 NCEI (6 Law 2010, 5 Eriksen back-converted, 3
# Doyle, 3 Lusher, 1 Egger with no Ocean Cleanup copy) + 5 EMODnet samples
# (A, B, C, D, H — B since D12 counts shape-only lines)
TRUTH_ROWS = 683 + 3031 + 16 + 18 + 5


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("mplastic"))
    res = b1.run_smoke("mplastic", root=root, keep=True)
    meta = json.load(open(os.path.join(root, "work", "mplastic", "mplastic",
                                       "store.json")))
    st = f10.Store(os.path.join(root, "work", "mplastic", "mplastic"))
    return res, meta, st


def test_registered():
    assert fam.REGISTRY["mplastic"] is mp.MPlasticAdapter
    ad = mp.MPlasticAdapter()
    assert ad.family == "13" and ad.C == 12 and ad.time_dtype == "int32"
    assert not ad.per_year and ad.first_year == 1972
    assert ad.channel_names == [
        "log_n_km2", "log_g_km2", "log_n_m3", "log_n_km2_wc",
        "log_n_km2_micro", "log_n_km2_meso", "log_n_km2_macro",
        "log_n_km2_mega", "mesh_um", "net_depth_m", "sample_depth_m",
        "wind_ms"]
    assert ad.licence["redistribution"] == "attribution"
    assert ad.licence["derived_works"] == "free"
    for who in ("Lebreton et al. 2018", "Lebreton et al. 2024", "NCEI",
                "EMODnet"):
        assert who in ad.licence["attribution"]
    lay = b1.layout_for(ad)
    assert lay.hf_root == "tensors/family1_3"
    assert lay.family == "family1_3" and lay.family_version == "1.3"
    # the log channels' upper bounds ARE the linear physical bounds
    lo, hi = ad.bounds()
    assert hi[mp.COL["log_n_km2"]] == pytest.approx(math.log10(1 + 1e9))
    assert hi[mp.COL["log_n_m3"]] == pytest.approx(math.log10(1 + 1e7))


def test_smoke_all_stages_and_the_probe(smoke):
    res, meta, st = smoke
    assert len(res["truth"]) == TRUTH_ROWS
    assert st.N == TRUTH_ROWS
    p = res["probe"]
    # August 2015: the 250 odd-numbered manta events and all 151 large nets
    assert p["rows"] == 250 + 151
    assert p["counts_scope"] == "archive"
    assert p["bytes_fetched"] > 0 and p["schema_version"] == 2
    assert sum(p["out_of_bounds_stored"].values()) == 0


def test_the_ledger_counts_every_drop_by_name_and_by_pair(smoke):
    _res, meta, _st = smoke
    c = meta["counts"]
    assert c["read"] == {"lebreton2018": 683, "lebreton2018_hist": 3532,
                         "lebreton2024": 188, "ncei": 48, "emodnet": 8}
    # precedence: the compiled 'This study' rows are the 2015 manta events;
    # 25 NCEI Eriksen rows are copies of the compiled ones; one NCEI row is a
    # copy of a manta event; one EMODnet sample is a copy of an NCEI pump
    assert c["dedup"] == {"lebreton2018 > lebreton2018_hist": 501,
                          "lebreton2018_hist > ncei": 25,
                          "lebreton2018 > ncei": 1, "ncei > emodnet": 1}
    assert c["ncei_dropped_by_name"] == {"Law et al. 2014": 2}
    assert c["ncei_marine_setting"] == {"Ocean water": 48, "Beach": 1,
                                        "Nurdle Patrol": 1}
    assert c["ncei_rows_by_method"] == {"a": 6, "b": 30, "c": 7,
                                        "volumetric": 3, "dropped by name": 2}
    assert c["kept_by_source"] == {"lebreton2018": 683,
                                   "lebreton2018_hist": 3031,
                                   "lebreton2024": 16, "ncei": 18,
                                   "emodnet": 5}
    assert c["lebreton2024_excluded_by_name"] == {
        "MEGA Expedition (Manta trawl)": 120,
        "MEGA Expedition (Mega trawl)": 20,
        "AERIAL Expedition (Aerial survey)": 31}
    assert c["lebreton2024_vs_2018"]["manta_pairs"] == 120
    assert c["lebreton2024_vs_2018"]["manta_mid"] == 120
    assert c["lebreton2024_manta_is_midpoint"] == 1
    assert c["rows_no_value"] == 1                  # the NPM3 row without CD1
    assert c["emodnet_samples_dropped"] == {
        "no_plastic_line": 1, "vor_count_undefined": 1}
    # D12: the two shape-only lines (B's fragments, C's filaments) are plastic
    assert c["emodnet_line_classes"] == {"plastic": 10, "non-plastic": 2,
                                         "ambiguous": 0}
    assert c["emodnet_shape_samples"] == 1          # B; C's shape line is 0
    assert c["emodnet_date_only"] == 1
    tab = {(e["study"], e["sampling_method"]): e
           for e in c["ncei_depth_table"]}
    assert tab[("Law et al. 2010", "Neuston net")]["method"] == "a"
    assert tab[("Law et al. 2010", "Neuston net")]["depth_used_m"] == \
        pytest.approx(0.25)
    e = tab[("Eriksen et al. 2014", "Neuston net")]
    assert e["method"] == "b" and e["n_matched"] == 25
    assert e["depth_used_m"] == pytest.approx(0.08)
    assert tab[("Doyle et al. 2011", "Neuston net")]["method"] == "c"
    assert tab[("Lusher et al. 2014", "Pump")]["method"] == "volumetric"
    assert tab[("Law et al. 2014", "Neuston net")]["method"] == \
        "dropped by name"


def test_egger_rows_are_dropped_by_match_not_by_name(smoke):
    """The Ocean Cleanup's own NCEI submission: a row with an Ocean Cleanup
    copy within +-1 day and 0.06 deg (of the event's start, end or midpoint)
    is dropped; one with none is KEPT as an NCEI row."""
    _res, meta, st = smoke
    c = meta["counts"]
    # the exact copy's NEAREST Ocean Cleanup row is the compilation's copy of
    # the event (rounded to the same 4 decimals); the shifted one's is the
    # event itself, by its end
    assert c["ncei_egger_dropped_by_match"] == {
        "lebreton2018_hist > ncei (Egger match)": 1,
        "lebreton2018 > ncei (Egger match)": 1}
    assert c["ncei_egger_kept_by_year"] == {"Egger et al. 2020 2018": 1}
    assert c["ncei_egger_kept_near_toc_loose"] == []
    e = [x for x in c["ncei_depth_table"]
         if x["study"] == "Egger et al. 2020"][0]
    assert e["rows"] == 3 and e["kept"] == 1 and e["dropped_by_match"] == 2
    assert e["method"] == "c"
    v = np.asarray(st["values"], np.float32)
    lat, lon = np.asarray(st["lat"]), np.asarray(st["lon"])
    k = np.flatnonzero((np.abs(lat - 36.3) < 1e-3)
                       & (np.abs(lon + 127.7) < 1e-3))
    assert k.size == 1
    assert v[k[0], mp.COL["log_n_m3"]] == pytest.approx(math.log10(1.88),
                                                        abs=1e-3)
    # the copy a day later and 0.03 deg off the tow's end is NOT a copy by
    # the generic rule, only by the Egger match
    e0 = [x for x in sm.events() if x["type"] == "manta"][5]
    near = mp.toc_near(
        {"day": (e0["y"], e0["mo"], e0["d"] + 1), "lat": e0["lat1"] + 0.03,
         "lon": e0["lon1"] - 0.03},
        [dict(mp.cand("lebreton2018", "e", mp.t_of(e0["y"], e0["mo"],
                                                   e0["d"], 1.0)[0], 0,
                      e0["lat"], e0["lon"], "p", 1, "s", n_km2=1.0),
              alt=[(e0["lat0"], e0["lon0"]), (e0["lat1"], e0["lon1"])])],
        mp.EGGER_DAYS, mp.EGGER_DEG)
    assert len(near) == 1
    assert mp.toc_near({"day": (e0["y"], e0["mo"], e0["d"] + 3),
                        "lat": e0["lat"], "lon": e0["lon"]},
                       near, mp.EGGER_DAYS, mp.EGGER_DEG) == []


def test_per_year_counts_are_exact(smoke):
    res, meta, _st = smoke
    want = collections.Counter(
        (dt.datetime(1982, 1, 1) + dt.timedelta(seconds=r["t"])).year
        for r in res["truth"])
    got = {int(k): v for k, v in meta["per_year"].items() if v}
    assert got == dict(want)


def _rows(st):
    v = np.asarray(st["values"], np.float32)
    return v, np.asarray(st["qc"]), np.asarray(st["lat"]), \
        np.asarray(st["lon"])


def test_measured_zeros_survive_and_absences_stay_nan(smoke):
    _res, _meta, st = smoke
    v, qc, lat, lon = _rows(st)
    C = mp.COL
    # the empty manta tow (event 3): n_km2 exactly 0, not NaN
    e3 = sm.events()[sm.ZERO_MANTA]
    k = np.flatnonzero((np.abs(lat - (e3["lat0"] + 0.002)) < 1e-3)
                       & ((qc & 15) == 1))
    assert k.size == 1 and v[k[0], C["log_n_km2"]] == 0.0
    # Law 2010's empty tow (from the accession), Doyle's empty neuston (per
    # m3), EMODnet's empty volumetric sample D, the empty 2021 aircraft line
    for la, lo, ch in ((32.1, -64.95, "log_n_km2"),
                       (41.0, -130.0, "log_n_m3"),
                       (45.0, 7.0, "log_n_m3"),
                       (34.9, -145.0, "log_n_km2")):
        k = np.flatnonzero((np.abs(lat - la) < 1e-3) & (np.abs(lon - lo)
                                                       < 1e-3))
        assert k.size == 1, (la, lo)
        assert v[k[0], C[ch]] == 0.0, (la, lo, ch)
    # a per-m3-only study has no areal value, and vice versa
    k = np.flatnonzero((np.abs(lat - 41.0) < 1e-3)
                       & (np.abs(lon + 130.0) < 1e-3))
    assert np.isnan(v[k[0], C["log_n_km2"]])


def test_two_tows_at_one_station_are_paired_by_their_values(smoke):
    """Law 2010 lists two tows at 35.88 N, 58.98 W on 1991-10-18; NCEI lists
    them in the opposite order. Only the value ratio pairs them right."""
    _res, _meta, st = smoke
    v, _qc, lat, lon = _rows(st)
    k = np.flatnonzero((np.abs(lat - 35.88) < 1e-3)
                       & (np.abs(lon + 58.98) < 1e-3))
    got = sorted(float(x) for x in v[k, mp.COL["log_n_km2"]])
    assert got == pytest.approx([math.log10(17820), math.log10(18360)],
                                abs=5e-3)
    assert np.allclose(v[k, mp.COL["net_depth_m"]], 0.25, atol=1e-3)


def test_emodnet_counts_only_plastic_lines(smoke):
    _res, _meta, st = smoke
    v, qc, lat, lon = _rows(st)
    k = np.flatnonzero((np.abs(lat - 43.0) < 1e-3) & (np.abs(lon - 5.0)
                                                     < 1e-3))
    assert k.size == 1
    area = 0.6 / 1000 * 2.0
    # 10 + 20 microplastic items; the 5 non-plastic particles are excluded
    assert v[k[0], mp.COL["log_n_km2"]] == pytest.approx(
        math.log10(1 + 30 / area), abs=1e-2)
    # the two classes 0.5-1.5 and 1.5-5 mm tile the micro channel exactly
    assert v[k[0], mp.COL["log_n_km2_micro"]] == pytest.approx(
        math.log10(1 + 30 / area), abs=1e-2)
    assert np.isnan(v[k[0], mp.COL["log_n_km2_meso"]])


def test_the_megaplastics_swap_assertion_fires_on_a_corrupted_sheet(tmp_path):
    root = str(tmp_path)
    sm.write_all(root, corrupt_swap=True)
    import types
    ctx = types.SimpleNamespace(source_dir=root, count_bytes=lambda n: None,
                                a=types.SimpleNamespace(attempts=1))
    with pytest.raises(lb.FormatError, match="Megaplastics|disagrees"):
        mp.MPlasticAdapter().lebreton2018(ctx)


def test_float_event_ids_are_normalised():
    assert 1.2000000000000002 != 1.2           # the trap itself
    assert lb.event_id(1.2000000000000002) == lb.event_id(1.2) == "1.2"
    assert lb.event_id(0.30000000000000004) == "0.3"
    assert lb.event_id(447.0) == "447"
    assert lb.event_id(0.0) == "0"


def test_emodnet_classification_follows_the_vocabularies():
    c = mp.emodnet_classify
    assert c("SDN:H01::H0100001 (microplastic items)", "") == "plastic"
    assert c("SDN:H01::H0100002 (fragments)",
             "SDN:H05::H0500001 (polyethylene)") == "plastic"
    # D12 (2026-10-13): a shape line with no polymer type counts as plastic
    # by default; the strict EMODnet reading is still one argument away
    assert c("SDN:H01::H0100002 (fragments)", "") == "plastic"
    assert c("SDN:H01::H0100002 (fragments)", "",
             shapes_plastic=False) == "ambiguous"
    assert c("SDN:H01::H0100004 (filaments)",
             "SDN:H05::H0500022 (natural fibres)") == "non-plastic"
    assert c("SDN:H01::H0100007 (non-plastic man-made micro-particles)",
             "") == "non-plastic"
    assert c("SDN:H01::H0100002 (fragments)",
             "SDN:H05::H0500028 (tyre wear)") == "ambiguous"
    assert c("SDN:H01::H0100010 (Undefined micro-litter items)",
             "SDN:H05::H0500025 (undefined plastic)") == "plastic"
    # the switch reads a shape as plastic; "undefined" stays ambiguous
    assert c("SDN:H01::H0100002 (fragments)", "",
             shapes_plastic=True) == "plastic"
    assert c("SDN:H01::H0100010 (Undefined micro-litter items)", "",
             shapes_plastic=True) == "ambiguous"
    assert mp.EMODNET_SHAPES_ARE_PLASTIC is True


def test_size_classes_parse_and_tile():
    assert mp.h03_bounds("SDN:H03::H0300010 (300 micrometres to 5 "
                         "millimetres)") == (0.3, 5.0)
    assert mp.h03_bounds("SDN:H03::H0300008 (1 to 5 millimetres)") == \
        (1.0, 5.0)
    assert mp.tiles([(0.5, 1.5), (1.5, 5.0)], 0.5, 5.0)
    assert not mp.tiles([(1.0, 5.0)], 0.5, 5.0)          # a partial cover
    assert not mp.tiles([(5.0, 25.0)], 5.0, 50.0)
    assert mp.tiles([(500.0, None)], 500.0, None)


def test_dedup_rule():
    def r(src, y, m, d, la, lo):
        qc = 0 if d else mp.QC_NO_TIME | mp.QC_NO_DAY
        t = mp.t_of(y, m, d)[0] if d else mp.t_of(y, m)[0]
        return mp.cand(src, f"{src}{y}{m}{d}{la}", t, qc, la, lo, "p", 1, "s",
                       n_km2=1.0)
    rows = [r("lebreton2018", 2015, 8, 1, 30.0, -140.0),         # kept
            r("lebreton2018_hist", 2015, 8, None, 30.005, -140.005),
            r("lebreton2018_hist", 2015, 8, None, 31.0, -141.0),  # kept
            r("ncei", 2015, 8, 1, 30.009, -139.991),      # same day, inside
            r("ncei", 2015, 8, 20, 31.0, -141.0),         # month-only match
            r("ncei", 2015, 9, 1, 30.0, -140.0),          # another month
            r("ncei", 2015, 8, 1, 30.011, -140.0),        # 0.011 deg: kept
            r("ncei", 2015, 8, 1, 30.0111, -140.0),       # same source: kept
            r("emodnet", 2015, 8, 1, 30.011, -140.0)]     # copy of an ncei row
    c = {}
    kept = mp.dedup(rows, c)
    assert c["dedup"] == {"lebreton2018 > lebreton2018_hist": 1,
                          "lebreton2018 > ncei": 1,
                          "lebreton2018_hist > ncei": 1,
                          "ncei > emodnet": 1}
    assert len(kept) == 5


def test_mirror_names_and_the_hub_pending_message(tmp_path):
    assert mp.mirror_name("https://api.figshare.com/v2/articles/1") == \
        "api.figshare.com/v2/articles/1__body"
    assert mp.mirror_name("https://api.figshare.com/v2/articles/1/files") == \
        "api.figshare.com/v2/articles/1/files__body"
    assert mp.mirror_name("https://x.org/a/b/") == "x.org/a/b/index.html"
    assert mp.mirror_name("https://x.org/f.xlsx") == "x.org/f.xlsx"
    assert mp.mirror_name("https://x.org/c?a=1&b=2") == "x.org/c__body__a_1_b_2"
    p = tmp_path / "pending"
    p.write_bytes(b'{"message":"Up to date download file is being generated. '
                  b'Please check back again"}')
    assert mp.hub_pending(str(p))
    p.write_bytes(b"Unique ID,Sample Date\n")
    assert not mp.hub_pending(str(p))


def test_the_2024_description_must_define_every_column_read():
    art = {"description": sm._desc(),
           "license": {"name": "CC BY 4.0"}}
    mp.lebreton2024_check_description(art)
    bad = dict(art, description=art["description"].replace(
        "CD3</b>: numerical", "CD3</b>: something"))
    with pytest.raises(mp.FormatError, match="CD3"):
        mp.lebreton2024_check_description(bad)
    with pytest.raises(mp.FormatError, match="licence"):
        mp.lebreton2024_check_description(
            dict(art, license={"name": "CC BY-NC 4.0"}))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
