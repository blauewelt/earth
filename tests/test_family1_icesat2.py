#!/usr/bin/env python3
"""The `icesat2` adapter's smoke (family 1.0.tf, E-082 wave 6) — no network.

    python3 -m pytest -q tests/test_family1_icesat2.py

ICESat-2's ATLAS instrument is a photon-counting laser; ATL08 is its
land-and-vegetation product, one record per 100-metre segment under each of
six ground tracks, reaching 88 degrees of latitude where GEDI stops at 51.6.
This store keeps LAND segments only, so these tests pin the water filter and
what it drops, the three quality conditions folded into `qc`, the size gate
that keeps phase A to one year, and the probe's own arithmetic — against a
synthetic archive written in ATL08's real group and dataset layout.

THE EXPECTED COUNTS, derived from `icesat2.make_smoke_sources` rather than
observed from it: four granules (two in 2021-12, two in 2022-01) x two ground
tracks x eight segments = 64 segments. Three segments per track-granule are
water — one by the product's inland-water mask, one classed Open_sea and one
classed Permanent_water_bodies — so 24 are dropped and 40 rows remain.
"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

import build_family10_stores as b10                             # noqa: E402
import build_family1_stores as b1                               # noqa: E402
from family1 import adapters as fam                             # noqa: E402
from family1.adapters import icesat2 as i2                      # noqa: E402

GRANULES = len(i2.SMOKE_GRANULES)              # 4
TRACKS = len(i2.SMOKE_TRACKS)                  # 2
SEGMENTS = i2.SMOKE_SEGMENTS                   # 8
SEGS_TOTAL = GRANULES * TRACKS * SEGMENTS      # 64
DROPPED = GRANULES * TRACKS * i2.DROPPED_SEGMENTS   # 24
TRUTH_ROWS = SEGS_TOTAL - DROPPED              # 40
MONTH_GRANULES = 2
PROBE_ROWS = MONTH_GRANULES * TRACKS * i2.KEPT_SEGMENTS   # 20


def test_registered_and_declared():
    assert fam.REGISTRY["icesat2"] is i2.ICESat2Adapter
    ad = i2.ICESat2Adapter()
    assert ad.C == 6 and ad.family == "1tf" and ad.distribution == "public"
    # C = 6 is load-bearing: the ledger's row is 27 + 2C = 39 bytes
    assert b10.row_bytes(ad.C, "int32") == 39
    assert ad.channel_names == [
        "h_te_best_fit", "h_canopy", "canopy_openness", "n_ca_photons",
        "n_te_photons", "segment_landcover"]
    assert ad.credentials == ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    # 100 m segments, instantaneous: the ledger's (-8.1, -12)
    assert abs(ad.log2_fp - (-8.1205)) < 1e-3
    assert ad.log2_dt == -12.0
    assert ad.platform_meta is True and ad.first_year == 2018
    assert "LAND SEGMENTS ONLY" in ad.notes and "ONE YEAR (2022)" in ad.notes
    # EVERY BOUND FITS FLOAT16, which is what `values` is stored as: a photon
    # count past 65,504 could only be stored as an infinity, which would read
    # as "never measured" instead of as a refusal
    lo, hi = ad.bounds()
    assert float(np.max(hi)) <= 60000.0
    assert float(np.max(np.abs(hi))) < 65504 and float(np.min(lo)) > -65504


def test_the_size_gate_refuses_before_its_first_byte(monkeypatch, tmp_path):
    monkeypatch.delenv("ICESAT2_ALLOW_BUILD", raising=False)
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    monkeypatch.setenv("NETRC", str(tmp_path / "absent"))

    def ctx(stage, lo, hi, src=""):
        return argparse.Namespace(
            a=argparse.Namespace(stage=stage, source_dir=src),
            source_dir=src, d_lo=b10.parse_date(lo), d_hi=b10.parse_date(hi))

    with pytest.raises(SystemExit, match="crosses a calendar-year boundary"):
        i2.ICESat2Adapter().fetch_preflight(
            ctx("all", "2022-01-01", "2023-12-31"))
    # one year passes the size gate and stops on the credential guard instead
    with pytest.raises(SystemExit, match="no netrc naming"):
        i2.ICESat2Adapter().fetch_preflight(
            ctx("all", "2022-01-01", "2022-12-31"))
    assert i2.ICESat2Adapter().fetch_preflight(
        ctx("all", "2018-01-01", "2025-12-31", src=str(tmp_path))) is None
    monkeypatch.setenv("ICESAT2_ALLOW_BUILD", "1")
    nr = tmp_path / "netrc"
    nr.write_text("machine urs.earthdata.nasa.gov login u password p\n")
    monkeypatch.setenv("NETRC", str(nr))
    assert i2.ICESat2Adapter().fetch_preflight(
        ctx("all", "2018-01-01", "2025-12-31")) is None


def test_the_granule_name_parser():
    def e(name, size="83.0"):
        return {"title": name, "granule_size": size,
                "time_start": "2022-06-01T00:00:00.000Z",
                "time_end": "2022-06-01T00:07:00.000Z",
                "links": [
                    {"href": "https://data.nsidc.earthdatacloud.nasa.gov/"
                             f"nsidc-cumulus-prod-public/{name}.iso.xml"},
                    {"href": "https://data.nsidc.earthdatacloud.nasa.gov/"
                             "nsidc-cumulus-prod-protected/ATLAS/ATL08/007/"
                             f"2022/06/01/{name}.h5"}]}
    got = i2.parse_entries([e("ATL08_20220531235826_10671511_007_01")])
    key = ("20220531235826", "1067", "15", "11")
    assert list(got) == [key]
    assert got[key]["release"] == "007" and got[key]["revision"] == "01"
    assert "protected" in got[key]["url"] and got[key]["bytes"] == 83000000
    with pytest.raises(i2.FormatError, match="two granules"):
        i2.parse_entries([e("ATL08_20220531235826_10671511_007_01"),
                          e("ATL08_20220531235826_10671511_007_02")])
    with pytest.raises(i2.FormatError, match="not ATL08 granules"):
        i2.parse_entries([{"title": "README", "links": []}])
    bad = e("ATL08_20220531235826_10671512_007_01")
    bad["links"] = [{"href": "https://x/nsidc-cumulus-prod-public/a.h5"}]
    with pytest.raises(i2.FormatError, match="no protected"):
        i2.parse_entries([bad])


def test_the_stamp_dates_a_granule():
    k = ("20220531235826", "1067", "15", "11")
    want = b10.seconds_since_epoch(dt.date(2022, 5, 31)) + 23 * 3600 + \
        58 * 60 + 26
    assert i2.stamp_seconds(k) == want
    lo = b10.seconds_since_epoch(dt.date(2022, 6, 1))
    assert i2.in_window({"t0": None, "t1": None}, k, lo, lo + 86400)
    assert not i2.in_window({"t0": None, "t1": None}, k, lo + 10 * 86400,
                            lo + 20 * 86400)


def test_the_water_classes_come_from_the_files_own_flag_table():
    class DS:
        def __init__(self, vals, means):
            self.attrs = {"flag_values": np.array(vals, np.uint8),
                          "flag_meanings": means}
    counts = {}
    water, nodata, table = i2.landcover_classes(
        DS([0, 30, 80, 90, 200, 255],
           "No_data Herbaceous Permanent_water_bodies Herbaceous_wetland "
           "Open_sea Undetermined"), counts)
    # a wetland is LAND; permanent water and open sea are not
    assert water == {80, 200}
    assert nodata == {0, 255}
    assert counts["landcover_table_source"].startswith("the segment_landcover")
    # with no flag table on the file the release-007 data dictionary is used,
    # and the fallback is RECORDED rather than silent

    class Bare:
        attrs = {}
    counts2 = {}
    water2, nodata2, table2 = i2.landcover_classes(Bare(), counts2)
    assert water2 == {80, 200} and nodata2 == {0, 255}
    assert table2[70] == "Snow_and_ice"        # land ice is KEPT
    assert counts2["landcover_table_source"].startswith("the ATL08 release")


def test_a_missing_dataset_is_a_refusal_not_a_guess(tmp_path):
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "a.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("gt1l")
        ls = g.create_group("land_segments")
        ls.create_dataset("latitude", data=np.zeros(3, np.float32))
        ls.create_group("terrain").create_dataset(
            "h_te_best_fit", data=np.zeros(3, np.float32))
    with h5py.File(p, "r") as f:
        g = f["gt1l"]
        assert i2.resolve_path(g, i2.NEEDED["lat"], "lat", "gt1l") == \
            "land_segments/latitude"
        with pytest.raises(i2.FormatError, match="land_segments/latitude"):
            i2.resolve_path(g, ("land_segments/not_here",), "x", "gt1l")


def test_smoke_all_stages_and_the_probe(tmp_path):
    res = b1.run_smoke("icesat2", root=str(tmp_path / "s"), keep=True)
    assert len(res["truth"]) == TRUTH_ROWS
    p = res["probe"]
    assert p["rows"] == PROBE_ROWS
    assert p["counts_scope"] == "month"
    assert p["distinct_platforms"] == TRACKS
    assert p["bytes_fetched"] > 0
    c = p["counts"]
    assert c["granules_wanted"] == MONTH_GRANULES
    assert c["tracks_seen"] == MONTH_GRANULES * TRACKS
    assert c["segments_in_granule"] == MONTH_GRANULES * TRACKS * SEGMENTS
    assert c["rows_kept"] == PROBE_ROWS
    # LAND ONLY — one drop by the product's own inland-water mask, one per
    # water land-cover class, each counted by name
    assert c["segments_inland_water"] == MONTH_GRANULES * TRACKS
    assert c["segments_dropped_by_landcover"] == {
        "80_Permanent_water_bodies": MONTH_GRANULES * TRACKS,
        "200_Open_sea": MONTH_GRANULES * TRACKS}
    assert c["landcover_table_source"].startswith("the segment_landcover")
    # the land-cover histogram is over EVERY segment, dropped ones included,
    # so the filter can be revisited from the probe
    assert sum(c["segment_landcover"].values()) == \
        MONTH_GRANULES * TRACKS * SEGMENTS
    # out of bounds -> NaN and COUNTED, never clipped; nothing reached the
    # writer out of bounds
    assert c["out_of_bounds"] == {"h_canopy": MONTH_GRANULES * TRACKS}
    assert p["out_of_bounds"]["h_canopy"] == MONTH_GRANULES * TRACKS
    assert sum(p["out_of_bounds_stored"].values()) == 0
    assert p["nan_fraction"]["h_canopy"] == round(
        MONTH_GRANULES * TRACKS / PROBE_ROWS, 6)
    assert p["nan_fraction"]["h_te_best_fit"] == 0.0
    # the three qc conditions, one segment each per track-granule, worst wins
    per = MONTH_GRANULES * TRACKS
    assert c["qc_terrain_flag_nonzero"] == per
    assert c["qc_msw_flag_nonzero"] == per
    assert c["qc_canopy_uncertainty_missing"] == per
    assert p["qc"] == {"0": PROBE_ROWS - 3 * per, "1": 2 * per, "2": per}
    assert c["fill_values"] == {"h_canopy_uncertainty": per}
    rd = c["resolved_datasets"]
    assert isinstance(rd, list)
    assert rd[0]["gt1l"]["h_canopy"] == "land_segments/canopy/h_canopy"
    assert rd[0]["gt1l"]["terrain_flg"] == "land_segments/terrain_flg"


def test_the_built_store_per_year_and_platforms(tmp_path):
    res = b1.run_smoke("icesat2", root=str(tmp_path / "s"), keep=True,
                       probe=False)
    root = os.path.join(res["work"], "icesat2", "icesat2")
    st = json.load(open(os.path.join(root, "store.json")))
    assert st["N"] == TRUTH_ROWS and st["C"] == 6
    assert st["per_year"] == {"2021": TRUTH_ROWS // 2, "2022": TRUTH_ROWS // 2}
    assert st["family"] == "family1_tf" and st["tier"] == "P"
    c = st["counts"]
    assert c["segments_in_granule"] == SEGS_TOTAL
    assert c["rows_kept"] == TRUTH_ROWS
    assert c["segments_inland_water"] == GRANULES * TRACKS
    assert sum(c["segments_dropped_by_landcover"].values()) == \
        DROPPED - GRANULES * TRACKS
    # THE PLATFORM TABLE: six ground tracks offered, the two the store holds
    # kept, and the beam strength read from the track group's own attribute
    plats = json.load(open(os.path.join(root, "platforms.json")))
    assert len(plats) == TRACKS
    assert st["platforms"]["source_entries"] == len(i2.TRACKS) == 6
    assert st["platforms"]["in_store"] == TRACKS
    assert st["platforms"]["in_store_without_entry"] == 0
    by_id = {v["id"]: v for v in plats.values()}
    assert set(by_id) == set(i2.SMOKE_TRACKS)
    assert by_id["gt1l"]["beam_strength"] == "weak"
    assert by_id["gt2r"]["beam_strength"] == "strong"
    assert by_id["gt2r"]["pair"] == "2" and by_id["gt2r"]["side"] == "r"
    assert "atlas_beam_type" in by_id["gt1l"]["beam_strength_source"]
    assert str(b10.platform_hash("gt1l")) in plats


def test_a_track_with_no_beam_type_says_so_rather_than_deriving_one():
    """Which beam of a pair is strong depends on the spacecraft orientation;
    deriving it from that is a RULE, not a reading, so it is left unsaid."""
    ad = i2.ICESat2Adapter()
    p = ad.platforms(None)
    assert len(p) == 6
    one = p[int(b10.platform_hash("gt3r"))]
    assert one["beam_strength"] is None
    assert "sc_orient" in one["beam_strength_source"]


def test_a_capped_probe_spreads_over_the_window():
    """Twelve granules from the head of a month is twelve passes of one
    morning; twelve spread over it is a measurement of the month."""
    assert i2.spread(list(range(4641)), 12)[:3] == [0, 386, 773]
    assert len(i2.spread(list(range(4641)), 12)) == 12
    assert i2.spread(list(range(4)), 12) == [0, 1, 2, 3]
    assert i2.spread(list(range(10)), 0) == list(range(10))


def test_storage_and_fetch_is_arithmetic_not_a_guess():
    counts = {"rows_kept": 400_000, "fetch_seconds": 50.0}
    per = [{"granule_bytes": 83_000_000, "bytes_read": 4_150_000},
           {"granule_bytes": 83_000_000, "bytes_read": 4_150_000}]
    so = i2.storage_and_fetch(counts, per, 6)
    m = so["measured"]
    assert m["granules_read"] == 2 and m["rows_per_granule"] == 200_000.0
    assert m["native_bytes_of_those_granules"] == 166_000_000
    assert m["read_fraction"] == 0.05
    assert m["bytes_per_second"] == 166_000.0
    pj = so["projected_one_year_2022"]
    assert pj["granules"] == 4641 * 12
    assert pj["rows"] == 200_000 * 4641 * 12
    assert pj["store_bytes"] == 200_000 * 4641 * 12 * 39
    assert pj["bytes_to_fetch"] == round(0.05 * i2.BYTES_MONTH * 12)
    assert so["bytes_per_row_stored"] == 39
    assert i2.storage_and_fetch({"rows_kept": 0}, per, 6) is None


def test_the_index_plan(tmp_path):
    ad = i2.ICESat2Adapter()
    src = str(tmp_path / "src")
    lo = b10.parse_date(ad.smoke_window[0])
    hi = b10.parse_date(ad.smoke_window[1])
    ad.smoke_sources(src, lo, hi)
    a = argparse.Namespace(
        store="icesat2", work=str(tmp_path / "w"), source_dir=src,
        start=str(lo), end=str(hi), stage="index", force=False, attempts=1,
        qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
        parts_from_hub=False, push_parts=False, allow_missing_years=False)
    ctx = b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))
    b10.run_stages(ctx, ["index"], stage_fn=b1.STAGE_FN, deps=b1.DEPS)
    plan = json.load(open(os.path.join(ctx.root, "plan.json")))
    assert plan["granules"] == GRANULES
    assert plan["collection"] == "C3565574177-NSIDC_CPRD"
    assert plan["version"] == "007" and plan["releases_seen"] == {"007": 4}
    assert plan["land_only"] is True and plan["one_year_only"] is True
    assert plan["tracks_expected"] == list(i2.TRACKS)
    assert plan["landcover_table"]["200"] == "Open_sea"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
