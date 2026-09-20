#!/usr/bin/env python3
"""The `burned500` adapter's smoke (family 1.0.tf, MCD64A1 v6.1) — no network.

    python3 -m pytest -q tests/test_family1_burned500.py

MCD64A1 is the MODIS monthly burned-area product: for every 500 m pixel of
land it says whether the pixel burned this month and on which day. The store
is the first MONTHLY tier-G store, and a month is coarser than the family's
five-day bin — which is the whole reason the framework grew a third skip
reason (`sharded.FRAME_NO_FRAME_IN_BIN`). What the tests pin is that reasoning:
that a month lands on exactly one bin, that the other bins write NO shard and
are counted by name, that `no_frame_in_bin` never stands in for a real hole,
that the sinusoidal grid's inverse is right, and that the SDS names are
resolved against the granule rather than transcribed.

The synthetic archive (`burned500.make_smoke_sources`) is the real layout: one
CMR `granules.umm_json` body per month and one real HDF4 granule per tile,
written with `pyhdf` on a 48 x 48 grid so a whole frame fits in a test.
"""
import datetime as dt
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

import build_family1_stores as b1                              # noqa: E402
from family1 import sharded as sh                              # noqa: E402
from family1 import adapters as fam                            # noqa: E402
from family1.adapters import burned500 as bs                   # noqa: E402

pytest.importorskip("pyhdf")
pytest.importorskip("zstandard")

ENV = (bs.SMOKE_PX_ENV, bs.SMOKE_TILES_ENV)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("burned500_smoke"))
    keep = {k: os.environ.get(k) for k in ENV}
    try:
        return b1.run_smoke("burned500", root=root, keep=True)
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def store_meta(smoke):
    p = os.path.join(smoke["work"], "burned500", "burned500", "store.json")
    return json.load(open(p))


# ================================================================== facts ===
def test_registered_and_declares_a_monthly_tier_g_store(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    assert fam.REGISTRY["burned500"] is bs.Burned500Adapter
    ad = bs.Burned500Adapter()
    assert b1.is_grid(ad)
    assert ad.family == "1tf" and ad.distribution == "public"
    assert ad.tier == "G" and ad.layout == "sharded"
    assert ad.credentials == ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    assert ad.channel_names == ["burn_doy", "qa"]
    assert ad.C == 2 and ad.dtype == "float16"
    # ONE frame per bin, the whole five days — a monthly map is one picture
    assert ad.frames_per_bin == 1
    assert ad.frame_seconds == sh.BIN_SECONDS
    assert ad.first_year == 2000
    # 463.31 m against the family's 27.83 km unit, and a month against 5 days
    assert abs(ad.log2_fp - np.log2(463.31271652777775 / 1000.0 / 27.83)) < 1e-9
    assert abs(ad.log2_dt - np.log2(365.25 / 12.0 / 5.0)) < 1e-9


def test_the_group_width_is_the_measured_compromise_not_a_preference():
    """One group per tile does not fit on the Hub; one per row does not fit
    in a runner's memory. The arithmetic is in the adapter, so a future change
    is made against numbers."""
    ad = bs.Burned500Adapter()
    groups = ad.groups()
    assert len(bs.TILES) == 268                    # CMR, 2019-08
    assert len(groups) == 46
    # 46 groups x ~310 months x 2 files, a fifth of the Hub's 100,000 limit
    assert len(groups) * 310 * 2 == 28520 < 100000
    opts = {o["tiles_per_group"]: o for o in bs.GROUP_ARITHMETIC["options"]}
    assert opts[1]["files"] > 100000 and opts[1]["why_not"]
    assert opts[36]["probe_float64_bytes"] > 3e9 and opts[36]["why_not"]
    assert opts[bs.TILES_PER_GROUP]["why_not"] is None
    assert opts[bs.TILES_PER_GROUP]["groups"] == len(groups)
    # every declared tile lands in exactly one group, and no group is empty
    seen = set()
    for t in bs.TILES:
        h, v = bs.tile_of(t)
        g = bs.group_of_tile(h, v)
        assert g in groups
        seen.add(g)
    assert seen == set(groups)


def test_a_month_lands_on_exactly_one_bin_and_the_map_is_invertible():
    """The filing rule: the bin that holds the 15th, and nothing else.

    A bin is five days and the shortest month is 28, so no bin can hold two
    anchor days — the month-to-bin map is a function with an inverse, which is
    what lets `classify_bin` answer from the bin alone.
    """
    anchored = {}
    for y in range(2000, 2027):
        for m in range(1, 13):
            b = bs.month_bin(y, m)
            assert (y, m) not in anchored
            assert b not in anchored, (y, m, b)
            anchored[b] = (y, m)
            assert bs.bin_month(b) == (y, m)
    # and five bins in six hold no month at all — the reason the skip exists
    lo = min(anchored)
    hi = max(anchored)
    empty = sum(1 for b in range(lo, hi + 1) if bs.bin_month(b) is None)
    assert empty / float(hi - lo + 1) > 0.75


def test_no_frame_in_bin_never_stands_in_for_a_real_absence():
    """The distinction the framework change rests on.

    Inside the record a bin with no anchor is `no_frame_in_bin`; outside it is
    `before_record`/`after_record`; and a MONTH the product should have is
    never any of the three — it is `absent_upstream`, which `fetch_frames`
    yields and which this classifier cannot produce at all.
    """
    first, last = (2000, 11), (2019, 8)
    assert bs.classify_bin(bs.month_bin(2019, 8), first, last) == (2019, 8)
    assert bs.classify_bin(bs.month_bin(2000, 10), first, last) == \
        "before_record"
    assert bs.classify_bin(bs.month_bin(2019, 9), first, last) == \
        "after_record"
    # a bin in the record's LAST month but after its 15th is inside the
    # record and simply anchors nothing — not `after_record`
    b = bs.month_bin(2019, 8) + 1
    assert bs.bin_month(b) is None
    assert bs.classify_bin(b, first, last) == sh.FRAME_NO_FRAME_IN_BIN
    assert sh.FRAME_NO_FRAME_IN_BIN == "no_frame_in_bin"
    assert sh.FRAME_ABSENT_UPSTREAM not in sh.FRAME_SKIP_REASONS
    for r in (bs.classify_bin(bs.month_bin(2010, 6) + k, first, last)
              for k in range(1, 6)):
        assert r == sh.FRAME_NO_FRAME_IN_BIN


def test_the_sinusoidal_inverse_is_right_and_refuses_to_invent_a_longitude():
    """No projection library: lat = y / R, lon = x / (R cos lat).

    And it is UNDEFINED off the projected globe. A nine-tile block at 70 N
    really does have corners there, so the honest answer is NaN — a clamped
    +/-180 would look like a coordinate and be none, which is the reason the
    store writes no tile-corner table at all.
    """
    # the equator's west edge is 180 W, and the grid's top is 90 N
    lat, lon = bs.sinu_to_latlon(np.array([bs.X_MIN]), np.array([0.0]))
    assert abs(float(lat[0])) < 1e-9 and abs(float(lon[0]) + 180.0) < 1e-6
    lat, _ = bs.sinu_to_latlon(np.array([0.0]), np.array([bs.Y_MAX]))
    assert abs(float(lat[0]) - 90.0) < 1e-6
    # tile h18v08's north-west corner is 10 N on the zero meridian: the MODIS
    # grid is 18 rows of 10 degrees from 90 N, so row v08 starts at 10 N, and
    # column h18 starts at the zero meridian (x = 0 at every latitude).
    g = bs.grid_of("v08h18")
    lat, lon = bs.sinu_to_latlon(np.array([g["x0"]]), np.array([g["y0"]]))
    assert abs(float(lat[0]) - 10.0) < 1e-6 and abs(float(lon[0])) < 1e-9
    # the north-west corner of the FIRST block of row v02 (70 N) is off the
    # projected globe, and answers NaN rather than -180
    g2 = bs.grid_of("v02h00")
    lat, lon = bs.sinu_to_latlon(np.array([g2["x0"]]), np.array([g2["y0"]]))
    assert abs(float(lat[0]) - 70.0) < 1e-6
    assert np.isnan(float(lon[0]))
    # the pixel is 463.31 m and the grid is 9 tiles wide, north first
    assert abs(g["dx"] - 463.31271652777775) < 1e-9
    assert g["dy"] == -g["dx"] and g["row_order"] == "north_first"
    assert g["H"] == 2400 and g["W"] == 2400 * bs.TILES_PER_GROUP


def test_the_sds_names_are_resolved_against_the_file_not_transcribed():
    got = bs.resolve_sds({"Burn Date", "Burn Date Uncertainty", "QA",
                          "First Day", "Last Day"})
    assert got == {"burn_doy": "Burn Date", "qa": "QA"}
    with pytest.raises(bs.FormatError) as e:
        bs.resolve_sds({"Something Else", "QA"})
    msg = str(e.value)
    assert "burn_doy" in msg and "Something Else" in msg   # the file's own list
    assert "SDS_CANDIDATES" in msg


def test_a_tile_name_that_is_not_one_is_a_refusal():
    assert bs.tile_of("h19v08") == (19, 8)
    for bad in ("h19v8", "H19V08", "h36v08", "h19v18", "", "v08h19"):
        with pytest.raises(bs.FormatError):
            bs.tile_of(bad)
    assert bs.group_parts("v08h18") == (8, 18)
    with pytest.raises(bs.FormatError):
        bs.group_parts("h18v08")


def test_a_cmr_body_with_no_tile_at_all_is_a_refusal():
    """The 2026-09-14 rule: an empty listing is never a measurement."""
    with pytest.raises(bs.FormatError) as e:
        bs.parse_cmr(json.dumps({"items": []}).encode(), 2019, 8)
    assert "NO tile" in str(e.value)
    with pytest.raises(bs.FormatError):
        bs.parse_cmr(b"<html>", 2019, 8)


# ============================================================== the smoke ===
def test_smoke_stores_one_frame_per_month_and_no_shard_for_the_others(smoke):
    truth = smoke["truth"]
    by_reason = {}
    for (_arr, why) in truth.values():
        by_reason[why] = by_reason.get(why, 0) + 1
    # one group, nine bins over the window: one carries August's map, four
    # carry no month at all, four are past the synthetic record's last month
    assert by_reason[None] == 1
    assert by_reason["no_frame_in_bin"] == 4
    assert by_reason["after_record"] == 4
    assert len(truth) == 9

    meta = store_meta(smoke)
    assert list(meta["groups"]) == ["v08h18"]
    g = meta["groups"]["v08h18"]
    # THE POINT OF THE WHOLE CHANGE: eight skipped bins, one shard written.
    assert g["bins"] == 1
    assert g["frames_present"] == 1 and g["frames_missing"] == 0
    assert meta["frames_missing_by_reason"] == {"no_frame_in_bin": 4,
                                                "after_record": 4}
    c = meta["counts"]
    assert c["bins_no_frame_in_bin"] == 4
    assert c["bins_outside_record"] == 4
    # and a skipped bin is counted under exactly one of the two counters
    assert c["bins_no_frame_in_bin"] + c["bins_outside_record"] == 8


def test_the_stored_frame_holds_the_products_own_classes(smoke):
    """0 is a MEASUREMENT (mapped, did not burn); unmapped and water are NaN
    and are counted apart."""
    sm = os.path.join(smoke["work"], "burned500", "burned500")
    grp = sh.ShardedGroup(os.path.join(sm, "v08h18"))
    b = int(grp.shard_index["bin"][0])
    arr = grp.read_frame(b, 0)
    px = bs.SMOKE_PX
    assert arr.shape == (px, px * bs.TILES_PER_GROUP, 2)
    burn = arr[:, :, 0]
    # the group is h18..h26 of row v08 and the smoke's two tiles are h19 and
    # h20, so the frame's slots 1 and 2 carry data and the other seven are
    # NaN — which costs no stored bytes at all
    a = burn[:, px:2 * px]                          # h19v08
    assert np.nanmin(a) == 0.0                      # mapped, did not burn
    assert np.nanmax(a) > 200                       # the burn scar's day
    assert np.isnan(burn[:, :px]).all()             # h18v08, not a land tile
    assert np.isnan(burn[:, 3 * px:]).all()         # h21..h26, likewise
    assert not np.isnan(burn[:, 2 * px:3 * px]).all()   # h20v08 is there
    # the unmapped block and the water block are NaN, not zero
    assert np.isnan(a[px - 1, 0])
    assert np.isnan(a[0, px - 1])

    c = store_meta(smoke)["counts"]
    assert c["pixels_unmapped"] > 0 and c["pixels_water"] > 0
    assert c["pixels_burned"] > 0 and c["pixels_unburned"] > 0
    # the deliberately out-of-class value in the synthetic granule
    assert c["burn_date_unknown_class"] > 0
    assert store_meta(smoke)["counts"].get("out_of_bounds", {}) in ({}, None)


def test_the_probe_measured_the_month(smoke):
    p = smoke["probe"]
    assert p["tier"] == "G" and p["layout"] == "sharded"
    assert p["frames_fetched"] == 1
    assert p["bytes_fetched"] > 0
    assert p["out_of_bounds_stored"] == 0
    assert p["inputs_not_read"] == []
    g = p["groups"]["v08h18"]
    assert g["C"] == 2 and g["dtype"] == "float16"
    assert g["frames_fetched"] == 1
    # the record's own length drives the extrapolation, and it is the number
    # of MONTHS, not of bins
    assert g["record_frames"] == (2019 - 2000) * 12 + (8 - 11) + 1


def test_the_grid_declaration_travels_with_the_store(smoke):
    sm = os.path.join(smoke["work"], "burned500", "burned500")
    spec = json.load(open(os.path.join(sm, "v08h18", "tile_grid.json")))
    assert spec["grid"]["row_order"] == "north_first"
    assert "sinu" in spec["grid"]["crs"]
    assert spec["grid"]["modis_tiles"][0] == "h18v08"
    assert spec["grid"]["modis_tiles"][-1] == "h26v08"
    assert len(spec["grid"]["modis_tiles"]) == bs.TILES_PER_GROUP
    assert spec["monthly_target"]["anchor_day"] == 15
    assert spec["monthly_target"]["frames_per_bin"] == 1
    assert spec["monthly_target"]["burn_doy_values"]["0"].startswith("mapped")
    # NO tile-corner table, and the grid says why: a nine-tile block's corners
    # near the grid's edge have no longitude, and a clamped number there would
    # look like a coordinate
    assert "tile_corners" not in spec
    assert "UNDEFINED" in spec["grid"]["inverse"]
    assert "lat = degrees(y / R)" in spec["grid"]["inverse"]


def test_the_index_reports_both_directions_of_the_tile_set_difference(
        tmp_path, monkeypatch):
    """A constant tile set is what makes the lanes assemblable; `index` must
    still MEASURE the live listing and report what differs, both ways."""
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    root = str(tmp_path)
    bs.make_smoke_sources(root, dt.date(2019, 8, 11), dt.date(2019, 8, 25))
    ad = bs.Burned500Adapter()

    class _Ctx:
        source_dir = root
        d_lo = dt.date(2019, 8, 11)
        d_hi = dt.date(2019, 8, 25)

        class a:
            attempts = 1

        def count_bytes(self, n):
            return None

    plan = ad.index(_Ctx())
    assert plan["collection"] == "C2565786756-LPCLOUD"
    assert plan["record_first_month"] == "2000-11"
    assert plan["record_last_month"] == "2019-08"
    assert plan["tiles_per_group"] == bs.TILES_PER_GROUP
    assert plan["anchor_day"] == 15
    # the synthetic listing holds a third tile the restricted build does not
    # declare, and the plan says so rather than silently ignoring it
    assert bs.SMOKE_ABSENT_TILE in plan["tiles_listed"]
    assert bs.SMOKE_ABSENT_TILE in plan["tiles_listed_not_declared"]
    assert plan["tiles_declared_not_listed"] == []
    assert plan["bin_of_first_month"] == bs.month_bin(2000, 11)
