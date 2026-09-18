#!/usr/bin/env python3
"""The eight scene-catalogue adapters' smokes (family 1.0.tf, E-082 wave 3).

    python3 -m pytest -q tests/test_family1_cat_scenes.py

A catalogue store's "archive" is a web API, so its synthetic source is a
directory of CANNED RESPONSES recorded from an in-process fake producer
(`ml/family1/adapters/_cat_smoke.py`). The real paging, the real count checks,
the real refusals and the real row packing all run against them; only the
socket is replaced. Each store's fake producer is hostile in the ways its real
producer is: a page that must be turned, an empty window, a missing cloud
value, a value out of bounds, a sensor and a processing baseline the code
table does not list, a footprint across the antimeridian, and — where the real
archive does it — the SAME scene published twice under two product types or
two timeliness classes.

Every count below is exact and was derived from the descriptors, not copied
from a run.
"""
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

import build_family10_stores as b10                             # noqa: E402
import build_family1_stores as b1                               # noqa: E402
import family10_store as f10                                    # noqa: E402
from family1 import adapters as fam                             # noqa: E402
from family1.adapters import _stac as st                        # noqa: E402

pytest.importorskip("pyarrow")

# store -> (truth rows, probe month, distinct platforms)
EXPECT = {
    # 7 scenes in the two days; the eighth listing is the same scene twice
    "cat_landsat": 7,
    # 7 granules over two collections; three more are returned by the
    # next day's window (CMR matches on overlap) and dropped by their start
    "cat_hls": 7,
    "cat_viirs": 6,
    "cat_ecostress": 5,
    "cat_s2": 8,
    # 6 rows from 6 products — the three `-COG` mirrors are NOT rows
    "cat_s1": 6,
    # 7 products, 5 overpasses: two are published NR and NT
    "cat_olci": 5,
    "cat_nisar": 6,
}


@pytest.fixture(scope="module")
def smokes(tmp_path_factory):
    """Every store's smoke, run once for the whole module."""
    out = {}
    for s in sorted(EXPECT):
        root = tmp_path_factory.mktemp(s)
        out[s] = b1.run_smoke(s, root=str(root), keep=True)
    return out


@pytest.mark.parametrize("store", sorted(EXPECT))
def test_the_store_holds_exactly_the_rows_the_adapter_emitted(store, smokes):
    res = smokes[store]
    assert len(res["truth"]) == EXPECT[store], store
    p = res["probe"]
    assert p["rows"] == EXPECT[store]
    assert p["distinct_platforms"] == EXPECT[store]      # no hash collisions
    assert sum(p["rows_per_day"]) == EXPECT[store]
    assert sum(p["out_of_bounds_stored"].values()) == 0
    assert p["bytes_fetched"] > 0
    assert p["schema_version"] == 2 and p["time_dtype"] == "int32"
    assert p["counts_scope"] == "month"
    assert p["counts"]["probe_month_only"] == 1
    # every window the plan listed was walked and its rows equalled the
    # producer's own count for it (the walkers refuse otherwise)
    assert p["counts"]["producer_count"] >= EXPECT[store]


@pytest.mark.parametrize("store", sorted(EXPECT))
def test_the_sidecar_has_one_row_per_scene_and_rebuilds_the_urls(store,
                                                                 smokes):
    import pyarrow.parquet as pq
    # run_smoke's `work` is the build root; the store lives one level in
    work = os.path.join(smokes[store]["work"], store)
    path = os.path.join(work, store, st.ASSETS_PARQUET)
    assert os.path.exists(path), path
    t = pq.read_table(path)
    assert t.num_rows == EXPECT[store]
    assert list(t.column_names) == list(st.ASSET_COLS)
    plat = t.column("platform").to_pylist()
    assert len(set(plat)) == len(plat)
    sm = f10.Store(os.path.join(work, store))
    assert sorted(set(int(x) for x in sm["platform"])) == sorted(plat)
    for pid, sid, base, tmpl in zip(plat, t.column("stac_id").to_pylist(),
                                    t.column("base_url").to_pylist(),
                                    t.column("asset_set").to_pylist()):
        assert b10.platform_hash(sid) == pid
        assert base.startswith("http")
        urls = [base + "/" + n.replace("{id}", sid) for n in tmpl.split()]
        assert urls and all(u.startswith("http") for u in urls)


@pytest.mark.parametrize("store", sorted(EXPECT))
def test_store_json_carries_the_code_tables_and_the_sidecars_record(store,
                                                                    smokes):
    work = os.path.join(smokes[store]["work"], store)
    with open(os.path.join(work, store, "store.json")) as fh:
        sj = json.load(fh)
    assert sj["family"] == "family1_tf" and sj["family_code"] == "1tf"
    assert sj["tier_t_catalogue"] is True
    ad = fam.REGISTRY[store]()
    assert sj["sensor_table"] == {str(k): v
                                  for k, v in sorted(ad.SENSOR_TABLE.items())}
    assert sj["qc_table"] == {str(k): v
                              for k, v in sorted(ad.QC_TABLE.items())}
    assert sj["sensor_other_code"] == 255 and sj["qc_other_code"] == 255
    assert sj["qc_keep_not_applicable"] is True
    assert sj["assets"]["rows"] == EXPECT[store]
    assert st.ASSETS_PARQUET in sj["sha256"]
    assert "{id}" in sj["assets_note"]


# ============================================================== per store ==
def test_landsat_counts_its_two_asset_sets_and_its_unlisted_codes(smokes):
    c = smokes["cat_landsat"]["probe"]["counts"]
    assert c["scenes"] == 8 and c["rows_kept"] == 7
    assert c["rows_duplicate_scene"] == 1        # the same scene listed twice
    assert c["sensor_unlisted"] == {"LANDSAT_3": 1}
    assert c["qc_unlisted"] == {"L2XX/T9": 1}
    assert c["out_of_bounds"] == {"cloud": 1}    # a cloud cover of 140 %
    assert c["sun_elevation_missing"] == 1       # a sun elevation of 0
    p = smokes["cat_landsat"]["probe"]
    assert p["nan_fraction"]["valid"] == 1.0     # the USGS publishes none
    assert p["qc"] == {"0": 3, "1": 1, "2": 1, "3": 1, "255": 1}
    # both asset sets really are the two the adapter declares
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(smokes["cat_landsat"]["work"],
                                   "cat_landsat", "cat_landsat",
                                   st.ASSETS_PARQUET))
    sets = set(t.column("asset_set").to_pylist())
    assert len(sets) == 2
    ad = fam.REGISTRY["cat_landsat"]()
    want = {" ".join(sorted(v)) for v in ad.ASSET_SETS.values()}
    assert sets == want                          # TM/ETM+ and OLI/TIRS
    assert all(len(x.split()) == 16 for x in sets)


def test_landsat_refuses_an_asset_set_it_has_never_seen():
    from family1.adapters import cat_landsat as cl
    ad = cl.LandsatCatalogue()
    base = "https://landsatlook.usgs.gov/data/x/SCENE"
    feat = {"id": "SCENE_SR",
            "assets": {"a": {"href": f"{base}/SCENE_SR_B99.TIF"}}}
    with pytest.raises(st.Refusal) as e:
        ad._assets(feat, "SCENE", {}, {}, "t")
    assert "asset set this adapter does not know" in str(e.value)
    # and an asset directory that is not the scene's own
    feat = {"id": "SCENE_SR",
            "assets": {"a": {"href": "https://h/elsewhere/SCENE_SR_B1.TIF"}}}
    with pytest.raises(st.Refusal) as e:
        ad._assets(feat, "SCENE", {}, {}, "t")
    assert "does not end in the scene name" in str(e.value)


def test_hls_reads_the_three_numbers_only_umm_json_carries(smokes):
    p = smokes["cat_hls"]["probe"]
    c = p["counts"]
    assert c["scenes_S30"] == 4 and c["scenes_L30"] == 3
    # CMR matches on OVERLAP with both ends inclusive, so a day-window query
    # also returns the next day's granules; they are dropped by their own
    # start time, which is what keeps a granule in exactly one window
    assert c["granule_starts_outside_window"] == 3
    assert c["attr_missing_CLOUD_COVERAGE"] == 1
    assert c["attr_not_numeric_SPATIAL_COVERAGE"] == 2
    assert c["sensor_unlisted"] == {"Sentinel-2X/Sentinel-2 MSI": 2}
    assert c["out_of_bounds"] == {"cloud": 1, "angle": 1}
    # HLS is the one catalogue of the eight whose producer publishes a valid
    # fraction, so `valid` is NOT all NaN here
    assert p["nan_fraction"]["valid"] < 1.0
    assert p["nan_fraction"]["angle"] < 1.0
    assert p["qc"] == {"0": 7}                   # every row is v2.0


def test_viirs_carries_three_satellites_and_three_nan_channels(smokes):
    p = smokes["cat_viirs"]["probe"]
    c = p["counts"]
    assert c["scenes_npp"] == 4 and c["scenes_j01"] == 1 \
        and c["scenes_j02"] == 1
    assert c["granule_starts_outside_window"] == 6
    assert c["sensor_unlisted"] == {"NOAA-99/VIIRS": 2}
    for ch in ("cloud", "valid", "angle"):
        assert p["nan_fraction"][ch] == 1.0, ch
    assert p["nan_fraction"]["log2_area"] == 0.0
    assert p["qc"] == {"0": 4, "1": 2}           # v2 (NPP) and v2.1 (JPSS)
    ad = fam.REGISTRY["cat_viirs"]()
    assert set(ad.SENSOR_TABLE.values()) == {"Suomi-NPP/VIIRS",
                                             "NOAA-20/VIIRS",
                                             "NOAA-21/VIIRS"}


def test_ecostress_keeps_both_processing_versions_apart_in_qc(smokes):
    p = smokes["cat_ecostress"]["probe"]
    c = p["counts"]
    assert c["scenes_v002"] == 3 and c["scenes_v003"] == 2
    assert c["granule_starts_outside_window"] == 2
    assert p["qc"] == {"2": 3, "3": 2}
    for ch in ("cloud", "valid", "angle"):
        assert p["nan_fraction"][ch] == 1.0, ch
    # the footprint is a RECTANGLE for this producer, never a polygon
    rings = st.cmr_rings({"GranuleUR": "g", "SpatialExtent": {
        "HorizontalSpatialDomain": {"Geometry": {"BoundingRectangles": [
            {"SouthBoundingCoordinate": 50.0, "WestBoundingCoordinate": 43.0,
             "NorthBoundingCoordinate": 51.0,
             "EastBoundingCoordinate": 45.0}]}}}})
    assert len(rings) == 1 and rings[0][0] == [50.0, 50.0, 51.0, 51.0]


def test_s2_walks_hour_windows_and_names_a_new_baseline(smokes):
    p = smokes["cat_s2"]["probe"]
    c = p["counts"]
    assert c["windows"] == 720                   # 30 days x 24 hours
    assert c["windows_empty"] == 720 - 4
    assert c["scenes_l2a"] == 8
    assert c["attr_missing_cloudCover"] == 1
    assert c["sensor_unlisted"] == {"S2X": 1}
    assert c["qc_unlisted"] == {"07.77": 1}
    assert c["out_of_bounds"] == {"cloud": 1}
    assert p["qc"]["255"] == 1
    for ch in ("valid", "angle"):
        assert p["nan_fraction"][ch] == 1.0, ch
    ad = fam.REGISTRY["cat_s2"]()
    assert ad.window_step == "hour"
    assert set(ad.SENSOR_TABLE.values()) == {"S2A", "S2B", "S2C", "S2D"}


def test_s1_catalogues_a_grdh_scene_once_and_not_as_its_cog_mirror(smokes):
    """The measured trap: every GRDH scene is in CDSE twice."""
    res = smokes["cat_s1"]
    c = res["probe"]["counts"]
    assert c["scenes_grdh"] == 4 and c["scenes_slc"] == 2
    assert res["probe"]["rows"] == 6              # NOT 4 + 4 + 2 = 10
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(res["work"], "cat_s1", "cat_s1",
                                   st.ASSETS_PARQUET))
    names = t.column("stac_id").to_pylist()
    assert not any("_COG" in n for n in names), names
    assert sum(1 for n in names if "_GRDH_" in n) == 4
    assert sum(1 for n in names if "_SLC" in n) == 2
    # the filter really is on productType and not on the name
    ad = fam.REGISTRY["cat_s1"]()
    f = ad.filt(b10.parse_date("2024-06-01"), b10.parse_date("2024-06-02"),
                "IW_GRDH_1S")
    assert "'productType'" in f and "'IW_GRDH_1S'" in f
    assert "contains(Name" not in f
    assert ad.SENSOR_TABLE[11] == "S1A/IW_GRDH_1S"
    assert ad.SENSOR_TABLE[12] == "S1A/IW_SLC__1S"


def test_olci_keeps_one_row_per_overpass_and_a_coastal_fraction(smokes):
    res = smokes["cat_olci"]
    p = res["probe"]
    c = p["counts"]
    assert c["scenes_wfr"] == 7                   # products listed
    assert c["overpasses"] == 5                   # rows kept
    assert c["rows_duplicate_overpass_dropped"] == 2
    assert c["rows_non_nt_kept"] == 2             # one NR-only, one ST
    assert c["qc_unlisted"] == {"009/ST": 1}
    assert p["nan_fraction"]["coastal"] == 0.0    # the static IS in the repo
    ad = fam.REGISTRY["cat_olci"]()
    assert ad.C == 6 and ad.channel_names[5] == "coastal"
    # the coastal channel separates a coastal granule from an open-ocean one
    sm = f10.Store(os.path.join(res["work"], "cat_olci", "cat_olci"))
    coast = np.asarray(sm["values"], np.float32)[:, 5]
    assert coast.min() == 0.0                     # the mid-Pacific granule
    assert coast.max() > 0.3                      # the Iberian one
    assert 0.0 <= coast.min() <= coast.max() <= 1.0


def test_nisar_codes_the_polarimetric_mode_and_the_processing_tier(smokes):
    p = smokes["cat_nisar"]["probe"]
    c = p["counts"]
    assert c["scenes"] == 6
    assert c["pge_versions"] == {"R05.02.3": 3, "R05.00.8": 2, "R09.99.9": 1}
    assert c["sensor_unlisted"] == {"ZZZZ": 1}
    assert c["qc_unlisted"] == \
        {"NISAR_L2_GCOV_EXPERIMENTAL_V1/PR": 1}
    assert p["qc"] == {"0": 2, "2": 2, "3": 1, "255": 1}
    for ch in ("cloud", "valid", "angle"):
        assert p["nan_fraction"][ch] == 1.0, ch
    from family1.adapters import cat_nisar as cn
    ad = cn.NisarCatalogue()
    assert len(ad.SENSOR_TABLE) == 81             # 9 x 9 two-band modes
    assert ad.SENSOR_TABLE[1] == "SHSH"
    assert "QPDH" in ad.SENSOR_TABLE.values()     # measured 2026-08
    assert ad.sensor_code("DHDH", {}) in ad.SENSOR_TABLE
    assert cn.tier_of("NISAR_L2_GCOV_PROVISIONAL_V1") == "PROVISIONAL"
    assert cn.tier_of("NISAR_L2_GCOV_BETA_V1") == "BETA"
    assert cn.tier_of("NISAR_UR_L2") == "URGENT"  # measured 2026-08
    with pytest.raises(st.Refusal):
        cn.name_fields("NISAR_L2_PR")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
