#!/usr/bin/env python3
"""The `pheno500` adapter's smoke (family 1.0.tf, E-082 wave 6) — no network.

    python3 -m pytest -q tests/test_family1_pheno500.py

The synthetic archive (`pheno500.make_smoke_sources`) is the real shape: a
`cmr_combined_<year>.json` holding what NASA's Common Metadata Repository
returns (`feed.entry` with the granule title, its declared size and its https
`-protected/` data link) and real HDF-EOS2 granules carrying a real
`StructMetadata.0` for their tile, on a 240 x 240 grid (`PHENO500_SMOKE_PX`)
instead of the product's 2,400. Two product years, three tiles, one tile
listed in 2021 and NOT in 2020 (so one frame is `absent_upstream`), one raw
date and one raw EVI2 level outside the producer's own valid range. Expected
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
from family1.adapters import _modis_sin as ms                   # noqa: E402
from family1.adapters import pheno500 as ph                     # noqa: E402

pytest.importorskip("pyhdf")

ENV = ("PHENO500_SMOKE_PX", "PHENO500_TILES", "PHENO500_MAX_YEARS")


def clear_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("pheno500_smoke"))
    keep = {k: os.environ.get(k) for k in ENV}
    try:
        return b1.run_smoke("pheno500", root=root, keep=True)
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def small_archive(tmp, **kw):
    src = os.path.join(tmp, "src")
    d_lo, d_hi = (b10.parse_date(x)
                  for x in ph.Pheno500Adapter.smoke_window)
    truth = ph.make_smoke_sources(src, d_lo, d_hi, **kw)
    return src, truth


def ctx_for(tmp, src, lo=None, hi=None, **over):
    w = ph.Pheno500Adapter.smoke_window
    d = dict(store="pheno500", work=os.path.join(tmp, "work"), source_dir=src,
             start=lo or w[0], end=hi or w[1], stage="all", force=False,
             attempts=1, qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS,
             assemble="auto", parts_from_hub=False, push_parts=False,
             allow_missing_years=False, allow_unconfirmed_licence=False,
             probe_month="", smoke=False)
    d.update(over)
    ad = ph.Pheno500Adapter()
    ctx = b10.Ctx(argparse.Namespace(**d), adapter=ad,
                  layout=b1.layout_for(ad))
    return b1.prepare_grid_ctx(ctx)


def run(ctx, stages):
    b10.run_stages(ctx, stages, stage_fn=b1.GRID_STAGE_FN, deps=b1.DEPS)


# ================================================================ facts ====
def test_registered_with_the_measured_tile_set(monkeypatch):
    clear_env(monkeypatch)
    assert fam.REGISTRY["pheno500"] is ph.Pheno500Adapter
    ad = ph.Pheno500Adapter()
    assert b1.is_grid(ad) and ad.layout == "sharded"
    assert ad.family == "1tf" and ad.distribution == "public"
    assert ad.credentials == ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    assert ad.C == 10 and ad.dtype == "float16"
    assert ad.channel_names == [
        "greenup", "mid_greenup", "peak", "senescence", "mid_greendown",
        "dormancy", "evi_minimum", "evi_amplitude", "evi_area", "qa_overall"]
    assert (ad.frames_per_bin, ad.frame_seconds) == (1, sh.BIN_SECONDS)
    assert ad.frames_per_bin * ad.frame_seconds == sh.BIN_SECONDS
    assert abs(ad.log2_dt - 6.190) < 1e-3
    # 315 tiles a year, identical in 2001, 2010, 2020, 2024 and 2025
    assert len(ph.TILES) == len(set(ph.TILES)) == 315
    for t in ph.TILES:
        h, v = ms.tile_hv(t)
        assert 0 <= h < 36 and 0 <= v < 18
    assert ad.px == 2400
    assert len(ad.specs()) == 315


def test_the_sinusoidal_grid_is_the_producers(monkeypatch):
    clear_env(monkeypatch)
    g = ms.sin_grid(18, 4, 2400)
    assert (g["H"], g["W"]) == (2400, 2400)
    assert abs(g["pixel_m"] - 463.3127) < 1e-3
    assert abs(g["x0"] - 0.0) < 1e-6           # h18 begins at the meridian
    assert abs(g["y0"] - 5559752.598) < 1e-2   # v4's north edge
    assert g["dy"] == -g["dx"]
    # its north and south edges are 50 N and 40 N, which is what NASA's own
    # corner polygon for h18v04 says
    assert abs(g["latitude_band"][1] - 50.0) < 1e-6
    assert abs(g["latitude_band"][0] - 40.0) < 1e-6
    # and the inverse of the forward projection is the identity
    for lat, lon in ((50.0, 10.0), (-33.0, 151.0), (0.0, -60.0)):
        x, y = ms.sin_forward(lat, lon)
        la, lo = ms.sin_inverse(x, y)
        assert abs(float(la) - lat) < 1e-9 and abs(float(lo) - lon) < 1e-9
    assert "no_tile_corners" in g


def test_the_bins_are_the_ones_holding_the_first_of_july(monkeypatch):
    clear_env(monkeypatch)
    for y, want in ((2001, 1424), (2020, 2812), (2025, 3177)):
        b = ph.year_bin(y)
        assert b == want
        assert sh.bin_start_date(b) <= dt.date(y, 7, 1)
        assert sh.bin_start_date(b + 1) > dt.date(y, 7, 1)
    # two product years are 73 bins apart, so no two frames can collide
    bins = [ph.year_bin(y) for y in range(2001, 2026)]
    assert len(bins) == len(set(bins))
    assert min(np.diff(bins)) >= 72


def test_the_frame_table_gives_every_bin_its_whole_year(monkeypatch):
    clear_env(monkeypatch)
    rows = ph.frame_table(2001, 2025)
    assert len(rows) == 25
    by_year = {r[2]: r for r in rows}
    assert by_year[2020][:2] == [2812, 0]
    assert by_year[2020][3:] == ["2020-01-01", "2020-12-31", 366]
    assert by_year[2021][3:] == ["2021-01-01", "2021-12-31", 365]


def test_the_date_conversion_is_a_day_number_in_the_product_year():
    """The source stores days since 1970-01-01; the store stores a day NUMBER
    in the product year, which can fall outside 1..366 in the southern
    hemisphere."""
    base = ph.days_1970(dt.date(2020, 1, 1))
    assert base == 18262
    assert ph.days_1970(dt.date(1970, 1, 1)) == 0
    # a green-up on 2020-04-10 is day 101 of 2020
    assert ph.days_1970(dt.date(2020, 4, 10)) - base + 1 == 101
    # and one on 2019-11-15 is day -46 — a date in the PREVIOUS year
    assert ph.days_1970(dt.date(2019, 11, 15)) - base + 1 == -46


def test_the_cmr_parser_keeps_the_newest_production(monkeypatch):
    clear_env(monkeypatch)
    url = ph.LP_PREFIX + "x/x.hdf"
    recs = [("MCD12Q2.A2025001.h18v04.061.2026243175842", url, 274.7),
            ("MCD12Q2.A2025001.h18v04.061.2026253091403", url, 274.7),
            ("MCD12Q2.A2025001.h12v09.061.2026243175842", url, 274.7)]
    feed = ms.cmr_json(recs)["feed"]["entry"]
    files, counts = ms.parse_cmr(feed, ".hdf")
    assert len(files) == 2
    assert counts["granules_superseded"] == 1
    kept = files[(dt.date(2025, 1, 1), (18, 4))]
    assert kept["title"].endswith("2026253091403")
    # a title the parser cannot read is a refusal, never a skipped tile
    with pytest.raises(ms.FormatError):
        ms.granule_key("MCD12Q2.A2025001.061.2026243175842")


def test_the_spec_carries_the_date_convention_and_the_cycle(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("PHENO500_TILES", "h18v04")
    sp = ph.Pheno500Adapter().specs()
    assert list(sp) == ["h18v04"]
    s = sp["h18v04"]
    assert (s["H"], s["W"], s["C"]) == (2400, 2400, 10)
    # 2400 / 256 = 9.375 -> 10 tiles a side with a 160-pixel pad
    assert (s["n_tiles_y"], s["n_tiles_x"]) == (10, 10)
    assert s["pad_rows"] == 160 and s["pad_cols"] == 160
    am = s["annual_map"]
    assert am["cycle"] == 0
    assert "days since 1970-01-01" in am["date_convention"]
    assert am["source_scalings"]["EVI_Area"] == 0.1
    assert s["frame_table"][0][2] == ph.FIRST_YEAR
    assert s["grid"]["tile"] == "h18v04"


def test_the_layout_carries_a_PER_GROUP_affine(monkeypatch):
    """The sharded layout's `grids` is {group: grid} and each grid holds its
    own x0/y0/dx/dy and projection, which is what makes a tiled PROJECTED
    product storable at all: two tiles of the same store have different
    origins in the same metric CRS. (`seaice_asi` has two groups on two
    polar-stereographic grids; this store has 315 on one sinusoidal one.)"""
    clear_env(monkeypatch)
    monkeypatch.setenv("PHENO500_TILES", "h18v04,h12v09,h29v11")
    sp = ph.Pheno500Adapter().specs()
    assert sorted(sp) == ["h12v09", "h18v04", "h29v11"]
    origins = {g: (s["grid"]["x0"], s["grid"]["y0"]) for g, s in sp.items()}
    assert len(set(origins.values())) == 3
    # the same pixel size and the same projection in all of them
    assert len({s["grid"]["dx"] for s in sp.values()}) == 1
    assert len({s["grid"]["proj4"] for s in sp.values()}) == 1
    # and each origin is its own tile's north-west corner in metres
    for g, s in sp.items():
        h, v = ms.tile_hv(g)
        x0, _ys, _x1, y0 = ms.tile_bounds_m(h, v)
        assert abs(s["grid"]["x0"] - x0) < 1e-6
        assert abs(s["grid"]["y0"] - y0) < 1e-6


# ============================================================== the smoke ==
def test_smoke_reads_every_frame_back_exactly(smoke):
    res = smoke["check"]
    # three tiles x two years, less the one tile the archive does not publish
    # in 2020
    assert res["frames_equal"] == 5
    assert res["by_reason"]["absent_upstream"] == 1
    sm = json.load(open(os.path.join(smoke["ctx"].store, "store.json")))
    assert sm["tier"] == "G" and sm["layout"] == "sharded"
    assert sorted(sm["groups"]) == sorted(ph.SMOKE_TILES)
    assert sm["C"] == 10 and sm["dtype"] == "float16"
    assert sm["frames_missing_by_reason"]["absent_upstream"] == 1
    # every other bin of the window held no product year at all and wrote NO
    # shard — the reason the layout skips, not an absence
    assert sm["frames_missing_by_reason"][sh.FRAME_NO_FRAME_IN_BIN] > 100
    assert sm["counts"]["outside_valid_range"] == {"greenup": 1,
                                                   "evi_minimum": 1}
    assert "SMOKE GRID" in sm["notes"] and "RESTRICTED BUILD" in sm["notes"]


def test_the_bins_written_are_only_the_july_ones(smoke):
    want = {ph.year_bin(y) for y in ph.SMOKE_YEARS}
    for t in ph.SMOKE_TILES:
        arr = sh.load_shard_index(os.path.join(smoke["ctx"].store, t,
                                               "shard_index.npy"))
        assert {int(r["bin"]) for r in arr} == want, t


def test_every_frame_reads_back_the_converted_source(smoke):
    truth = smoke["truth"]
    for (g, b, f), (want, why) in truth.items():
        if want is None:
            continue
        grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, g))
        got = grp.read_frame(b, f, raw=True)
        assert got is not None, (g, b, f)
        assert got.shape == (ph.SMOKE_PX, ph.SMOKE_PX, 10)
        assert np.array_equal(got, want, equal_nan=True), (g, b, f)


def test_the_dates_land_inside_their_product_year(smoke):
    """A green-up day number of a northern tile is a real day of its year,
    and the fill patch is NaN — the conversion, read back off the store."""
    g = "h18v04"
    b = ph.year_bin(2020)
    grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, g))
    a = grp.read_frame(b, 0)
    gu = a[:, :, 0]
    good = np.isfinite(gu)
    assert good.any() and (~good).any()
    assert gu[good].min() >= 60 and gu[good].max() <= 200
    # the six date channels are integers, exactly, in float16
    assert np.all(gu[good] == np.rint(gu[good]))


def test_the_probe_measured_the_same_frames_and_some_bytes(smoke):
    p = smoke["probe"]
    assert p["frames_fetched"] == 2       # 2020's two published tiles
    assert p["out_of_bounds_stored"] == 0
    assert p["bytes_fetched"] > 0
    assert p["month"] == "2020-06"
    for g, gp in p["groups"].items():
        assert gp["C"] == 10
        assert gp["record_frames"] == ph.LAST_YEAR_MEASURED - 2001 + 1


# ============================================================== refusals ===
def test_index_refuses_a_tile_outside_the_declared_set(tmp_path):
    src, _ = small_archive(str(tmp_path))
    p = os.path.join(src, "pheno500", "cmr_combined_2020.json")
    d = json.load(open(p))
    e = dict(d["feed"]["entry"][0])
    e["title"] = e["producer_granule_id"] = \
        "MCD12Q2.A2020001.h00v00.061.2026000000001"
    d["feed"]["entry"].append(e)
    json.dump(d, open(p, "w"))
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e2:
        run(ctx, ["index"])
    assert "TILES" in str(e2.value)


def test_index_refuses_an_empty_listing(tmp_path):
    src, _ = small_archive(str(tmp_path))
    for y in ph.SMOKE_YEARS:
        p = os.path.join(src, "pheno500", f"cmr_combined_{y}.json")
        json.dump({"feed": {"entry": []}}, open(p, "w"))
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index"])
    assert "no granule" in str(e.value)


def test_a_granule_on_the_wrong_tile_is_refused_not_stored(tmp_path):
    """The tile's own StructMetadata corners are checked in EVERY file."""
    src, _ = small_archive(str(tmp_path))
    gdir = os.path.join(src, "pheno500", "granules", "combined")
    name = [n for n in sorted(os.listdir(gdir)) if "h18v04" in n
            and "A2020" in n][0]
    arrays = ph.smoke_fields("h18v04", 2020, ph.SMOKE_PX)
    # write the same pixels under h18v04's NAME with h12v09's corners
    ms.write_smoke_hdf4(os.path.join(gdir, name), arrays, 12, 9,
                        ph.SMOKE_PX,
                        {n: {"_FillValue": ph.FILL} for n in
                         ph.Pheno500Adapter.datasets})
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index", "fetch"])
    assert "StructMetadata" in str(e.value)


def test_a_listed_granule_that_is_absent_stops_the_year(tmp_path):
    """A file CMR lists and the archive will not serve is `note_absent`, which
    leaves the year unmarked — never a silently smaller store."""
    src, _ = small_archive(str(tmp_path))
    gdir = os.path.join(src, "pheno500", "granules", "combined")
    name = [n for n in sorted(os.listdir(gdir)) if "h18v04" in n
            and "A2020" in n][0]
    os.remove(os.path.join(gdir, name))
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index", "fetch"])
    assert "could not be read" in str(e.value)
    assert ctx.absent, "a listed-and-absent granule was not noted"
    assert not b10.marked(ctx.root, f"parts/{sh.bin_year(ph.year_bin(2020))}")
