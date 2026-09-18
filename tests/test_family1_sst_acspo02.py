#!/usr/bin/env python3
"""The `sst_acspo02` adapter's smoke (family 1.gf, E-082 wave 4) — no network.

    python3 -m pytest -q tests/test_family1_sst_acspo02.py

`sst_acspo02` is the 2 km ACSPO L3S-LEO SST: a sharded tier-G store that the
note puts in PHASE C at ~2.3 TB, so the adapter is written to be probed and
built one year at a time, and `fetch_preflight` refuses a wider window before
its first byte. The tests here pin that refusal, the quality floor, the
GHRSST grid convention, the kelvin-to-Celsius conversion and the float32
bounds masking that keeps a 324-million-value frame off a float64 copy.
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
from family1.adapters import sst_acspo02 as sa                  # noqa: E402


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("SST_ACSPO02_SMOKE_GRID", raising=False)
    monkeypatch.delenv("SST_ACSPO02_COLLECTION", raising=False)
    monkeypatch.delenv("SST_ACSPO02_MAX_YEARS", raising=False)


def test_registered_and_declared():
    assert fam.REGISTRY["sst_acspo02"] is sa.SSTACSPO02Adapter
    ad = sa.SSTACSPO02Adapter()
    assert ad.C == 2 and ad.family == "1gf" and ad.tier == "G"
    assert ad.dtype == "float16" and ad.frames_per_bin == 5
    assert ad.frame_seconds == 86400
    assert ad.channel_names == ["sst", "quality_level"]
    assert ad.credentials == ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    assert abs(ad.log2_fp - (-3.644)) < 0.01     # the ledger's -3.64
    assert abs(ad.log2_dt - (-2.322)) < 0.01     # a daily field
    # DY is the default: one granule a day is one frame a day
    assert ad.which == "DY" and ad.cid == "C2805339147-POCLOUD"
    assert set(sa.COLLECTIONS) == {"DY", "PM", "AM"}
    assert ad.grid["H"] == 9000 and ad.grid["W"] == 18000


def test_a_window_wider_than_one_year_refuses_before_its_first_byte(
        monkeypatch, tmp_path):
    nr = tmp_path / "netrc"
    nr.write_text("machine urs.earthdata.nasa.gov login u password p\n")
    monkeypatch.setenv("NETRC", str(nr))
    ad = sa.SSTACSPO02Adapter()
    ctx = argparse.Namespace(a=argparse.Namespace(stage="all"),
                             source_dir="", years=[2020],
                             d_lo=dt.date(2020, 1, 1),
                             d_hi=dt.date(2020, 12, 31))
    assert ad.fetch_preflight(ctx) is None
    ctx.d_hi = dt.date(2021, 12, 31)
    with pytest.raises(SystemExit, match="SST_ACSPO02_MAX_YEARS"):
        ad.fetch_preflight(ctx)
    # and the netrc guard is separate
    monkeypatch.setenv("NETRC", str(tmp_path / "absent"))
    ctx.d_hi = dt.date(2020, 12, 31)
    with pytest.raises(SystemExit, match="no netrc naming"):
        sa.SSTACSPO02Adapter().fetch_preflight(ctx)


def test_the_grid_is_the_ghrsst_convention():
    lat = sa.LAT0 - (np.arange(sa.FULL_H) + 0.5) * sa.DY
    lon = sa.LON0 + (np.arange(sa.FULL_W) + 0.5) * sa.DX
    got = sa.grid_check(lat, lon)
    assert got["lat_first"] > got["lat_last"]      # DESCENDING
    # an ASCENDING axis is a refusal: the store declares row 0 northernmost
    with pytest.raises(sa.FormatError, match="ASCENDS"):
        sa.grid_check(lat[::-1], lon)
    with pytest.raises(sa.FormatError, match="differs from the declared"):
        sa.grid_check(lat - 1.0, lon)
    with pytest.raises(sa.FormatError, match="declared grid is"):
        sa.grid_check(lat, lon[:5])


def test_bounds_are_masked_in_float32_and_counted_not_clipped():
    a = np.zeros((4, 4, 2), np.float32)
    a[..., 0] = 15.0
    a[..., 1] = 5.0
    a[0, 0, 0] = 120.0                    # past the 40 degC bound
    a[1, 1, 1] = 9.0                      # past the grade-5 bound
    out = sa.mask_bounds_f32(a, sa.CHANNELS)
    assert out == {"sst": 1, "quality_level": 1}
    assert np.isnan(a[0, 0, 0]) and np.isnan(a[1, 1, 1])
    assert float(np.nanmax(a[..., 0])) == 15.0     # NOT clipped to 40
    assert a.dtype == np.float32                   # never widened


def test_the_layout_is_resolved_and_a_miss_prints_the_inventory():
    inv = {"sea_surface_temperature": {}, "quality_level": {}, "lat": {},
           "lon": {}, "time": {}}
    got = sa.resolve(inv)
    assert got["sst"] == "sea_surface_temperature"
    del inv["time"]                        # optional
    assert "time" not in sa.resolve(inv)
    del inv["quality_level"]               # required
    with pytest.raises(sa.FormatError) as e:
        sa.resolve(inv)
    assert "quality_level" in str(e.value) and "lat" in str(e.value)


def test_smoke_all_frames_and_the_probe(tmp_path):
    res = b1.run_smoke("sst_acspo02", root=str(tmp_path / "s"), keep=True)
    assert len(res["truth"]) == 10          # two bins x five daily frames
    chk = res["check"]
    assert chk["frames_equal"] == 4
    assert chk["by_reason"] == {"before_record": 4, "absent_upstream": 1,
                                "after_record": 1}
    p = res["probe"]
    g = p["groups"]["sst_acspo02"]
    assert g["dtype"] == "float16" and g["C"] == 2
    assert g["frames_fetched"] == 4
    assert p["out_of_bounds_stored"] == 0
    # the 400 K pixel is NaN and counted, never clipped
    assert p["out_of_bounds"] == {"sst": 1}
    # QUALITY IS A FLOOR, NOT A FILTER: grades 0 and 1 are dropped, 3 and 5
    # are stored WITH their grade
    gr = p["counts"]["quality_grade"]
    assert set(gr) == {"0", "1", "3", "5"}
    kept = gr["3"] + gr["5"]
    assert abs(g["valid_fraction"]["sst"] * 4 * 360 * 720 - kept) < 5
    # both channels are valid together — an SST without a grade is not a pixel
    assert abs(g["valid_fraction"]["sst"]
               - g["valid_fraction"]["quality_level"]) < 1e-5
    assert g["compression_ratio"] > 1.0
    assert p["bytes_fetched"] > 0
    assert p["counts"]["resolution"]["sst=sea_surface_temperature"] == 4


def test_the_spec_says_why_both_channels_are_float16(tmp_path):
    os.environ["SST_ACSPO02_SMOKE_GRID"] = f"{sa.SMOKE_W},{sa.SMOKE_H}"
    try:
        ad = sa.SSTACSPO02Adapter()
        spec = ad.specs()["sst_acspo02"]
        assert "one dtype per GROUP" in spec["channel_encoding"]
        assert spec["quality_floor"]["min_quality_level"] == sa.MIN_QUALITY
        assert spec["collection"]["key"] == "DY"
        assert "PHASE C" in ad.notes and "SMOKE GRID" in ad.notes
    finally:
        del os.environ["SST_ACSPO02_SMOKE_GRID"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
