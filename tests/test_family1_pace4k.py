#!/usr/bin/env python3
"""The `pace4k` adapter's smoke (family 1.gf, E-082 wave 6) — no network.

    python3 -m pytest -q tests/test_family1_pace4k.py

The synthetic archive (`pace4k.make_smoke_sources`) is the real shape: a
`cmr_<product>_<year>.json` holding what NASA's Common Metadata Repository
returns for each of the three products, and real netCDF-4 files under
`opendap/<L3SMI|L4SMI>/<YYYY>/<MMDD>/` with the producer's own fill values,
`scale_factor`/`add_offset` on the particulate-organic-carbon field and a
REGIONAL MOANA window placed at an exact sub-block of the global grid. One
day is listed by the apparent-optical-properties product and not by the
biogeochemical one (so the frame is `absent_upstream`); one day has both
required products and no MOANA file (a REAL frame whose last three channels
are NaN); one chlorophyll below the logarithm's floor, one cell count at
int32's minimum and one past the producer's own valid maximum. Expected
counts are exact.
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
from family1 import sharded as sh                               # noqa: E402
from family1.adapters import oc4k as oc                         # noqa: E402
from family1.adapters import pace4k as p4                       # noqa: E402

pytest.importorskip("netCDF4")

ENV = ("PACE4K_SMOKE_GRID",)


def clear_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("pace4k_smoke"))
    keep = {k: os.environ.get(k) for k in ENV}
    try:
        return b1.run_smoke("pace4k", root=root, keep=True)
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def small_archive(tmp, **kw):
    src = os.path.join(tmp, "src")
    d_lo, d_hi = (b10.parse_date(x) for x in p4.Pace4kAdapter.smoke_window)
    truth = p4.make_smoke_sources(src, d_lo, d_hi, **kw)
    return src, truth


def ctx_for(tmp, src, lo=None, hi=None, **over):
    w = p4.Pace4kAdapter.smoke_window
    d = dict(store="pace4k", work=os.path.join(tmp, "work"), source_dir=src,
             start=lo or w[0], end=hi or w[1], stage="all", force=False,
             attempts=1, qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS,
             assemble="auto", parts_from_hub=False, push_parts=False,
             allow_missing_years=False, allow_unconfirmed_licence=False,
             probe_month="", smoke=False)
    d.update(over)
    ad = p4.Pace4kAdapter()
    ctx = b10.Ctx(argparse.Namespace(**d), adapter=ad,
                  layout=b1.layout_for(ad))
    return b1.prepare_grid_ctx(ctx)


def run(ctx, stages):
    b10.run_stages(ctx, stages, stage_fn=b1.GRID_STAGE_FN, deps=b1.DEPS)


# ================================================================ facts ====
def test_registered_on_the_global_four_kilometre_grid(monkeypatch):
    clear_env(monkeypatch)
    assert fam.REGISTRY["pace4k"] is p4.Pace4kAdapter
    ad = p4.Pace4kAdapter()
    assert b1.is_grid(ad) and ad.layout == "sharded"
    assert ad.family == "1gf" and ad.distribution == "public"
    # NO credential: OB.DAAC's OPeNDAP answered without one, measured
    assert ad.credentials == ()
    assert ad.C == 7 and ad.dtype == "float16"
    assert ad.channel_names == ["log_chl", "poc", "carbon_phyto", "avw_400",
                                "pro_moana", "syn_moana", "pico_moana"]
    assert (ad.frames_per_bin, ad.frame_seconds) == (5, 86400)
    assert ad.frames_per_bin * ad.frame_seconds == sh.BIN_SECONDS
    assert (p4.H, p4.W) == (4320, 8640)
    assert ad.first_year == 2024 and str(p4.RECORD_START) == "2024-03-05"


def test_the_grid_is_oc4k_s_exactly(monkeypatch):
    """The two ocean-colour stores share one tile scheme, which is the point
    of using this grid rather than PACE's own."""
    clear_env(monkeypatch)
    a, b = p4.grid_of(), oc.grid_of()
    for k in ("H", "W", "x0", "dx", "y0", "dy", "crs", "pixel_deg"):
        assert a[k] == b[k], k
    assert p4.Pace4kAdapter.log2_fp == oc.OC4kAdapter.log2_fp
    assert "no_tile_corners" in a


def test_the_moana_window_is_an_exact_sub_block(monkeypatch):
    """MEASURED on the real files: 3360 x 2640 at global row 480, column
    2280 — 70 S..70 N, 85 W..25 E."""
    clear_env(monkeypatch)
    dy, dx = 180.0 / p4.H, 360.0 / p4.W
    lat = p4.Y0 - (np.arange(480, 480 + 3360) + 0.5) * dy
    lon = p4.X0 + (np.arange(2280, 2280 + 2640) + 0.5) * dx
    assert p4.sub_block(lat, lon) == (480, 2280)
    assert abs(lat[0] - 69.979166) < 1e-5 and abs(lat[-1] + 69.979166) < 1e-5
    assert abs(lon[0] + 84.979166) < 1e-5 and abs(lon[-1] - 24.979166) < 1e-5
    # half a cell off the grid is a REFUSAL, not a resampling
    with pytest.raises(p4.FormatError):
        p4.sub_block(lat + dy / 2.0, lon)


def test_the_three_products_and_their_names(monkeypatch):
    clear_env(monkeypatch)
    assert p4.PRODUCTS["bgc"][0] == "PACE_OCI_L3M_BGC"
    assert p4.PRODUCTS["aop"][0] == "PACE_OCI_L3M_AOP"
    assert p4.PRODUCTS["moana"][0] == "PACE_OCI_L4M_MOANA"
    assert p4.PRODUCTS["bgc"][3] == ("chlor_a", "poc", "carbon_phyto")
    assert p4.PRODUCTS["aop"][3] == ("avw",)
    assert p4.PRODUCTS["moana"][3] == ("prococcus_moana", "syncoccus_moana",
                                       "picoeuk_moana")
    # BGC and AOP are required; MOANA is not
    assert p4.PRODUCTS["bgc"][4] and p4.PRODUCTS["aop"][4]
    assert not p4.PRODUCTS["moana"][4]
    d = dt.date(2025, 7, 1)
    assert p4.object_name("bgc", d) == \
        "PACE_OCI.20250701.L3m.DAY.BGC.V3_2.4km.nc"
    assert p4.object_name("moana", d) == \
        "PACE_OCI.20250701.L4m.DAY.MOANA.V3_2.4km.nc"
    assert p4.parse_day(p4.object_name("aop", d), "aop") == d
    with pytest.raises(p4.FormatError):
        p4.parse_day("PACE_OCI.20250701.L3m.DAY.CHL.V3_2.chlor_a.4km.nc",
                     "bgc")
    # the DAP4 constraint expression carries the variables and the axes
    u = p4.opendap_url("bgc", d, p4.PRODUCTS["bgc"][3])
    assert u.startswith(p4.OPENDAP + "L3SMI/2025/0701/")
    assert ".dap.nc4?dap4.ce=" in u
    for v in ("chlor_a", "poc", "carbon_phyto", "lat", "lon"):
        assert f"/{v}" in u.replace("%2F", "/")


def test_the_derived_channels_are_flagged_in_the_spec(monkeypatch):
    clear_env(monkeypatch)
    s = p4.Pace4kAdapter().specs()["pace4k"]
    assert (s["H"], s["W"], s["C"]) == (4320, 8640, 7)
    # 4320 / 256 = 16.875 -> 17 rows; 8640 / 256 = 33.75 -> 34 columns
    assert (s["n_tiles_y"], s["n_tiles_x"]) == (17, 34)
    flags = {c["name"]: c["derived"] for c in s["channels"]}
    assert flags == {"log_chl": False, "poc": False, "carbon_phyto": False,
                     "avw_400": False, "pro_moana": True, "syn_moana": True,
                     "pico_moana": True}
    d = s["derived_channels"]
    assert d["level"] == "L4" and d["names"] == list(p4.DERIVED)
    assert "regional" in d
    assert s["record_start"] == "2024-03-05"


def test_the_channel_transforms(monkeypatch):
    """log10 for chlorophyll (oc4k's channel), minus 400 nm for the apparent
    visible wavelength, thousands of cells for the three MOANA fields."""
    clear_env(monkeypatch)
    out = p4.new_frame(4, 6)
    p4.put_channel(out, "chlor_a", np.full((4, 6), 10.0, np.float32))
    p4.put_channel(out, "avw", np.full((4, 6), 550.0, np.float32))
    p4.put_channel(out, "picoeuk_moana",
                   np.full((2, 3), 35000.0, np.float32), r0=1, c0=2)
    assert np.allclose(out[:, :, 0], 1.0)
    assert np.allclose(out[:, :, 3], 150.0)
    assert np.allclose(out[1:3, 2:5, 6], 35.0)
    assert np.isnan(out[0, 0, 6])
    # a chlorophyll at or below the floor is not a measurement of a logarithm
    counts = {}
    p4.put_channel(out, "chlor_a", np.full((4, 6), 1e-9, np.float32), counts)
    assert counts["chl_below_floor"] == 24
    assert np.isnan(out[:, :, 0]).all()


# ============================================================== the smoke ==
def test_smoke_reads_every_frame_back_exactly(smoke):
    res = smoke["check"]
    truth = smoke["truth"]
    n_real = sum(1 for a, _w in truth.values() if a is not None)
    assert res["frames_equal"] == n_real
    assert res["by_reason"]["absent_upstream"] == 1
    sm = json.load(open(os.path.join(smoke["ctx"].store, "store.json")))
    assert sm["tier"] == "G" and sm["layout"] == "sharded"
    assert sorted(sm["groups"]) == ["pace4k"]
    assert sm["C"] == 7 and sm["dtype"] == "float16"
    assert sm["frames_missing_by_reason"]["absent_upstream"] == 1
    # the producer's own oddities, counted by name
    assert sm["counts"]["chl_below_floor"] == 1
    assert sm["counts"]["negative_cells"]["prococcus_moana"] == 1
    assert sm["counts"]["above_producer_valid_max"]["picoeuk_moana"] == 1
    assert sm["counts"]["moana_not_published"] == 1
    assert "SMOKE GRID" in sm["notes"]


def test_a_day_without_moana_is_a_real_frame(smoke):
    """MOANA was published on 277 of 2024's 289 days; a day without it keeps
    its four global channels and is NOT an absent frame."""
    d = p4.SMOKE_NO_MOANA
    s = b10.seconds_since_epoch(d)
    b, f = s // sh.BIN_SECONDS, (s % sh.BIN_SECONDS) // 86400
    grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, "pace4k"))
    a = grp.read_frame(b, f)
    assert a is not None
    assert np.isfinite(a[:, :, 0]).any()          # chlorophyll is there
    assert not np.isfinite(a[:, :, 4]).any()      # the MOANA channels are not
    assert not np.isfinite(a[:, :, 6]).any()


def test_the_absent_day_is_a_zero_length_frame(smoke):
    d = p4.SMOKE_ABSENT
    s = b10.seconds_since_epoch(d)
    b, f = s // sh.BIN_SECONDS, (s % sh.BIN_SECONDS) // 86400
    grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, "pace4k"))
    assert grp.entry(b, f, 0, 0) == (sh.FRAME_ABSENT, 0)
    assert grp.read_frame(b, f) is None


def test_every_frame_reads_back_the_transformed_source(smoke):
    truth = smoke["truth"]
    grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, "pace4k"))
    for (g, b, f), (want, why) in truth.items():
        if want is None:
            continue
        got = grp.read_frame(b, f, raw=True)
        assert got is not None, (b, f)
        assert got.shape == (p4.SMOKE_H, p4.SMOKE_W, 7)
        assert np.array_equal(got, want, equal_nan=True), (b, f)


def test_the_moana_channels_only_cover_their_window(smoke):
    r0, nr, c0, nc = p4.SMOKE_MOANA
    d = dt.date(2024, 3, 6)
    s = b10.seconds_since_epoch(d)
    b, f = s // sh.BIN_SECONDS, (s % sh.BIN_SECONDS) // 86400
    grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, "pace4k"))
    a = grp.read_frame(b, f)
    for i in (4, 5, 6):
        ch = a[:, :, i]
        assert np.isfinite(ch[r0:r0 + nr, c0:c0 + nc]).any()
        outside = np.ones(ch.shape, bool)
        outside[r0:r0 + nr, c0:c0 + nc] = False
        assert not np.isfinite(ch[outside]).any()


def test_the_probe_measured_the_same_frames_and_some_bytes(smoke):
    p = smoke["probe"]
    assert p["out_of_bounds_stored"] == 0
    assert p["bytes_fetched"] > 0
    assert p["month"] == "2024-03"
    assert p["frames_per_bin"] == 5
    g = p["groups"]["pace4k"]
    assert g["C"] == 7
    assert set(g["valid_fraction"]) == set(p["groups"]["pace4k"]
                                           ["valid_fraction"])
    # the MOANA channels cover a window and so are the least valid
    vf = g["valid_fraction"]
    assert vf["pro_moana"] < vf["log_chl"]


# ============================================================== refusals ===
def test_index_refuses_an_empty_listing(tmp_path):
    src, _ = small_archive(str(tmp_path))
    for prod in p4.PRODUCTS:
        p = os.path.join(src, "pace4k", f"cmr_{prod}_2024.json")
        json.dump({"feed": {"entry": []}}, open(p, "w"))
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index"])
    assert "no BGC or no AOP" in str(e.value)


def test_a_file_on_the_wrong_grid_is_refused_not_stored(tmp_path):
    """A global file whose axes are not the declared grid is a REFUSAL."""
    src, _ = small_archive(str(tmp_path))
    d = dt.date(2024, 3, 6)
    p = os.path.join(src, "pace4k", "opendap", "L3SMI", "2024", "0306",
                     p4.object_name("bgc", d))
    bgc, _aop, _off = p4.smoke_fields(d, p4.SMOKE_H, p4.SMOKE_W)
    # write it shifted one row south — the same shape, the wrong axes
    p4.write_nc(p, p4.SMOKE_H, p4.SMOKE_W, bgc, r0=1, c0=0,
                attrs={"poc": {"scale_factor": np.float32(p4.POC_SCALE),
                               "add_offset": np.float32(p4.POC_OFFSET)}})
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index", "fetch"])
    assert "lat/lon differ" in str(e.value)


def test_a_listed_file_that_is_absent_stops_the_year(tmp_path):
    src, _ = small_archive(str(tmp_path))
    d = dt.date(2024, 3, 7)
    os.remove(os.path.join(src, "pace4k", "opendap", "L3SMI", "2024", "0307",
                           p4.object_name("bgc", d)))
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index", "fetch"])
    assert "could not be read" in str(e.value)
    assert ctx.absent, "a listed-and-absent file was not noted"
