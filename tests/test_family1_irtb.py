#!/usr/bin/env python3
"""The `irtb` adapter's smoke (family 1.gf, E-082 wave 4) — no network.

    python3 -m pytest -q tests/test_family1_irtb.py

`irtb` is the merged infrared brightness temperature: a sharded tier-G store
whose SCOPE is two settings (`IRTB_LAT_BAND`, `IRTB_EVERY`) because the whole
half-hourly 60 S-60 N record is ~7 TB and the note scopes phase A to the
tropics three-hourly. The tests that matter most here are the arithmetic ones:
the frame count a cadence implies, the refusal when it exceeds the layout's
64-frame mask, the latitude band's row arithmetic, and the mapping from a
store frame to the hourly file and sub-frame it comes from.

The synthetic archive is written in the GES DISC layout
(`<YYYY>/<DDD>/merg_<YYYYMMDDHH>_4km-pixel.nc4`), one file an hour with TWO
half-hourly time steps, on a shrunken grid.
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
from family1 import sharded as sh                               # noqa: E402
from family1 import adapters as fam                             # noqa: E402
from family1.adapters import irtb                               # noqa: E402


@pytest.fixture(autouse=True)
def _scope(monkeypatch):
    """The smoke covers the whole synthetic field at the default cadence."""
    monkeypatch.setenv("IRTB_LAT_BAND", "60")
    monkeypatch.setenv("IRTB_EVERY", "6")
    monkeypatch.delenv("IRTB_SMOKE_GRID", raising=False)


def test_registered_and_declared():
    assert fam.REGISTRY["irtb"] is irtb.IRTBAdapter
    ad = irtb.IRTBAdapter()
    assert ad.C == 1 and ad.family == "1gf" and ad.tier == "G"
    assert ad.layout == "sharded" and ad.dtype == "uint8"
    assert ad.channel_names == ["tb"]
    # THE BOUNDS ARE IN THE STORED UNIT, K - 160 — `check_sharded` compares a
    # stored tile against this table, so a table in kelvin refuses every tile
    assert ad.channels[0][2] == 0.0 and ad.channels[0][3] == 254.0
    assert ad.credentials == ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    assert abs(ad.log2_fp - (-2.78)) < 0.02   # 4.05 km at the equator
    assert abs(ad.log2_dt - (-7.9)) < 0.05      # a HALF-HOURLY measurement


def test_the_cadence_arithmetic_and_the_64_frame_wall(monkeypatch):
    monkeypatch.setenv("IRTB_EVERY", "6")
    ad = irtb.IRTBAdapter()
    assert ad.frame_seconds == 10800 and ad.frames_per_bin == 40
    assert ad.frames_per_bin * ad.frame_seconds == sh.BIN_SECONDS
    monkeypatch.setenv("IRTB_EVERY", "4")
    ad = irtb.IRTBAdapter()
    assert ad.frames_per_bin == 60           # two-hourly, the finest that fits
    # the note's own full half-hourly form is F = 240 and the layout's
    # shard-index frame_mask is a uint64 — so it REFUSES, by arithmetic, at
    # dispatch, naming the change that would be needed
    monkeypatch.setenv("IRTB_EVERY", "1")
    with pytest.raises(SystemExit, match="frame_mask"):
        irtb.IRTBAdapter()
    # a cadence that does not divide the five-day bin is refused too
    monkeypatch.setenv("IRTB_EVERY", "7")
    with pytest.raises(SystemExit, match="does not divide"):
        irtb.IRTBAdapter()


def test_the_latitude_band_is_rows_of_the_published_field():
    r0, h = irtb.band_rows(60.0)
    assert (r0, h) == (0, irtb.FULL_H)       # the whole field
    r0, h = irtb.band_rows(30.0)
    assert h == int(round(60.0 / irtb.DY)) and r0 == (irtb.FULL_H - h) // 2
    # the band really is centred, and the grid records the rows it took
    g = irtb.grid(30.0)
    assert g["band_rows"] == [r0, r0 + h]
    assert g["H"] == h and g["W"] == irtb.FULL_W
    south = g["y0"]
    assert -30.1 < south < -29.9
    assert abs((south + h * g["dy"]) - 30.0) < 0.05
    with pytest.raises(irtb.FormatError, match="60 S .. 60 N"):
        irtb.band_rows(90.0)


def test_a_store_frame_maps_to_one_hourly_file_and_sub_frame(monkeypatch):
    monkeypatch.setenv("IRTB_EVERY", "6")
    ad = irtb.IRTBAdapter()
    when, sub = ad._slot_source(2411, 0)
    assert sub == 0 and when.minute == 0
    when2, _ = ad._slot_source(2411, 1)
    assert (when2 - when) == dt.timedelta(hours=3)
    # A CADENCE ON THE HALF HOUR REALLY READS THE SECOND SUB-FRAME. Only
    # cadences that divide the bin AND give F <= 64 are legal, and
    # IRTB_EVERY 5 (2.5 h, F = 48) is the one among them whose frames fall on
    # the half hour — so it is what proves the sub-frame index is used.
    monkeypatch.setenv("IRTB_EVERY", "5")
    ad = irtb.IRTBAdapter()
    assert ad.frames_per_bin == 48 and ad.frame_seconds == 9000
    a, sa = ad._slot_source(2411, 0)
    b, sb = ad._slot_source(2411, 1)
    assert sa == 0 and sb == 1
    assert b.minute == 0 and (b - a) == dt.timedelta(hours=2)


def test_the_grid_is_a_hypothesis_every_file_falsifies():
    lat = irtb.LAT0 + (np.arange(irtb.FULL_H) + 0.5) * irtb.DY
    lon = irtb.LON0 + (np.arange(irtb.FULL_W) + 0.5) * irtb.DX
    got = irtb.grid_check(lat, lon, 30.0)
    assert got["band_rows"][1] - got["band_rows"][0] > 0
    # a DESCENDING lat axis is a refusal: the store declares row 0 south
    with pytest.raises(irtb.FormatError, match="DESCENDS"):
        irtb.grid_check(lat[::-1], lon, 30.0)
    # a shifted axis is a refusal, not a silent reprojection
    with pytest.raises(irtb.FormatError, match="differs from the declared"):
        irtb.grid_check(lat + 1.0, lon, 30.0)
    with pytest.raises(irtb.FormatError, match="declared source field"):
        irtb.grid_check(lat[:10], lon, 30.0)


def test_the_layout_is_resolved_and_a_miss_prints_the_inventory():
    inv = {"Tb": {}, "lat": {}, "lon": {}, "time": {}}
    assert irtb.resolve(inv) == {"tb": "Tb", "lat": "lat", "lon": "lon",
                                 "time": "time"}
    del inv["time"]                          # optional
    assert "time" not in irtb.resolve(inv)
    del inv["Tb"]                            # required
    with pytest.raises(irtb.FormatError) as e:
        irtb.resolve(inv)
    assert "Tb" in str(e.value) and "lat" in str(e.value)


def test_smoke_all_frames_and_the_probe(tmp_path):
    res = b1.run_smoke("irtb", root=str(tmp_path / "s"), keep=True)
    # 1991-style windows aside, the window is 2015-01-01 .. 2015-01-05 and
    # the two bins that touch it hold 40 frames each
    assert len(res["truth"]) == 80
    chk = res["check"]
    assert chk["frames_equal"] == 39
    assert chk["by_reason"] == {"before_record": 24, "absent_upstream": 1,
                                "after_record": 16}
    p = res["probe"]
    g = p["groups"]["irtb"]
    assert g["dtype"] == "uint8" and g["C"] == 1
    assert g["frames_fetched"] == 39
    assert p["out_of_bounds_stored"] == 0
    # the 600 K pixel of SMOKE_OOB_HOUR is NaN and counted, never clipped
    assert p["out_of_bounds"] == {"tb": 1}
    # the two strip rows no satellite saw are missing in every frame
    assert 0.9 < g["valid_fraction"]["tb"] < 1.0
    assert g["compression_ratio"] > 1.0
    assert p["bytes_fetched"] > 0
    c = p["counts"]
    assert c["time_steps_per_file"] == {"2": 39}
    assert c["grid_checked"] == 39
    assert c["resolution"]["tb=Tb"] == 39


def test_a_lane_asks_the_catalogue_through_its_last_owned_bin_s_end(
        tmp_path, monkeypatch):
    """A lane ending 2015-03-31 owns bin 2428 (2015-03-29 .. 2015-04-02, by
    its first day — 77dff61), so the window is widened to that bin's last
    second, and the CATALOGUE must be asked through 2015-04-02T23:59:59.

    `f10b.START` is a `date`, and `date + timedelta(seconds=...)` drops the
    time of day: the query used to end at 2015-04-02T00:00:00, the listing's
    last hour was 00Z, and the bin's 03Z .. 21Z frames — seven of them, hours
    the archive holds — were recorded `after_record` in every lane's last bin.
    """
    a = argparse.Namespace(
        store="irtb", work=str(tmp_path / "w"), source_dir="",
        start="2015-01-01", end="2015-03-31", stage="fetch", force=False,
        attempts=1, qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS,
        assemble="auto", parts_from_hub=False, push_parts=False, lanes="",
        allow_missing_years=False, allow_unconfirmed_licence=False,
        probe_month="", smoke=False)
    ad = irtb.IRTBAdapter()
    ctx = b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))
    b1.apply_lane(ctx)
    b1.prepare_grid_ctx(ctx)
    last = max(ctx.grid_bins[2015])
    assert last == 2428
    assert sh.bin_start_date(last) == dt.date(2015, 3, 29)

    # A catalogue that answers exactly the hours inside the window it is
    # asked for, from a source that holds every hour well past the lane.
    asked = []

    def fake_cmr(lo, hi, attempts=4, count=None):
        asked.append((lo, hi))
        t0 = dt.datetime.strptime(lo, "%Y-%m-%dT%H:%M:%SZ")
        t1 = dt.datetime.strptime(hi, "%Y-%m-%dT%H:%M:%SZ")
        out, h = {}, t0.replace(minute=0, second=0)
        while h <= min(t1, dt.datetime(2015, 4, 10, 23)):
            if h >= t0:
                out[h] = {"name": f"merg_{h:%Y%m%d%H}_4km-pixel.nc4",
                          "url": f"https://x/{h:%Y%m%d%H}", "bytes": 1}
            h += dt.timedelta(hours=1)
        return out

    monkeypatch.setattr(irtb, "cmr_hours", fake_cmr)
    hours = ad.hours(ctx)
    assert asked == [("2015-01-01T00:00:00Z", "2015-04-02T23:59:59Z")]
    assert max(hours) == dt.datetime(2015, 4, 2, 23)

    # Nothing inside the record is `after_record`: every frame of every owned
    # bin either reaches the download (refused here, so it is noted absent
    # and not yielded) or has another reason.
    def no_download(_ctx, entry):
        raise b10._NotFound(entry["url"])

    monkeypatch.setattr(ad, "_get", no_download)
    wanted = [("irtb", b, f) for b in ctx.grid_bins[2015]
              for f in range(ad.frames_per_bin)]
    why = [m.get("frame_missing")
           for *_k, arr, m in ad.fetch_frames(ctx, wanted)]
    assert "after_record" not in why
    assert "before_record" not in why


def test_the_scope_is_in_the_spec_and_in_the_notes(tmp_path):
    os.environ["IRTB_SMOKE_GRID"] = f"{irtb.SMOKE_W},{irtb.SMOKE_H}"
    try:
        ad = irtb.IRTBAdapter()
        spec = ad.specs()["irtb"]
        assert spec["scope"]["lat_band_deg"] == 60.0
        assert spec["scope"]["every_half_hours"] == 6
        assert spec["frames_per_bin"] == 40
        assert "kelvin = value + 160" in spec["channel_encoding"]
        assert "SCOPE:" in ad.notes and "SMOKE GRID" in ad.notes
        assert "phase-C change" in ad.notes
    finally:
        del os.environ["IRTB_SMOKE_GRID"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
