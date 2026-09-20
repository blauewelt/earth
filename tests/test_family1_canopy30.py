#!/usr/bin/env python3
"""The `canopy30` adapter's smoke (family 1.0.tf, E-082 wave 6) — no network.

    python3 -m pytest -q tests/test_family1_canopy30.py

The synthetic archive (`canopy30.make_smoke_sources`) is the real shape: an
Apache directory `index.html` in the transfer server's own table format
naming the WHOLE 261-tile product (so `index`'s set check sees what it is
written to falsify), real LZW GeoTIFFs on a 512 x 512 grid
(`CANOPY30_SMOKE_PX`) for the three tiles the smoke reads, cut into the same
4 x 4 sub-tiles of 2.5 degrees the real store uses, one tile named in
`CANOPY30_TILES` that the product does not publish at all (`00N_180W`, the
mid-Pacific), and one pixel of 200 m to exercise the out-of-bounds count and
the `height_codes` breakdown. Expected counts are exact.
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
from family1.adapters import canopy30 as c30                    # noqa: E402
from family1.adapters import lossyear as ly                     # noqa: E402

pytest.importorskip("rasterio")

# The adapter reads both at CONSTRUCTION time (so the fresh instance
# `stage_probe_grid` builds for itself sees the smoke's grid); every test that
# wants the real product clears them first.
ENV = ("CANOPY30_SMOKE_PX", "CANOPY30_TILES")
SMOKE_GROUPS = [c30.group_name(t, r, c)
                for t in c30.SMOKE_TILES + (c30.SMOKE_ABSENT,)
                for r in range(c30.SUB) for c in range(c30.SUB)]
N_REAL = len(c30.SMOKE_TILES) * c30.SUB * c30.SUB      # 48
N_ABSENT = c30.SUB * c30.SUB                           # 16


def clear_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("canopy30_smoke"))
    keep = {k: os.environ.get(k) for k in ENV}
    try:
        return b1.run_smoke("canopy30", root=root, keep=True)
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def small_archive(tmp):
    src = os.path.join(tmp, "src")
    d_lo, d_hi = (b10.parse_date(x)
                  for x in c30.Canopy30Adapter.smoke_window)
    truth = c30.make_smoke_sources(src, d_lo, d_hi)
    return src, truth


def ctx_for(tmp, src, lo=None, hi=None, **over):
    w = c30.Canopy30Adapter.smoke_window
    d = dict(store="canopy30", work=os.path.join(tmp, "work"), source_dir=src,
             start=lo or w[0], end=hi or w[1], stage="all", force=False,
             attempts=1, qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS,
             assemble="auto", parts_from_hub=False, push_parts=False,
             allow_missing_years=False, allow_unconfirmed_licence=False,
             probe_month="", smoke=False)
    d.update(over)
    ad = c30.Canopy30Adapter()
    ctx = b10.Ctx(argparse.Namespace(**d), adapter=ad,
                  layout=b1.layout_for(ad))
    return b1.prepare_grid_ctx(ctx)


def run(ctx, stages):
    b10.run_stages(ctx, stages, stage_fn=b1.GRID_STAGE_FN, deps=b1.DEPS)


# ================================================================ facts ====
def test_registered_with_the_measured_tile_set(monkeypatch):
    clear_env(monkeypatch)
    assert fam.REGISTRY["canopy30"] is c30.Canopy30Adapter
    ad = c30.Canopy30Adapter()
    assert b1.is_grid(ad) and ad.layout == "sharded"
    assert ad.family == "1tf" and ad.distribution == "public"
    assert ad.licence["redistribution"] == "attribution"
    assert ad.licence["redistribution_confirmed"] is True
    assert ad.channel_names == ["canopy_height"]
    assert ad.dtype == "uint8" and ad.C == 1
    assert (ad.frames_per_bin, ad.frame_seconds) == (1, sh.BIN_SECONDS)
    assert ad.frames_per_bin * ad.frame_seconds == sh.BIN_SECONDS
    assert ad.zstd_level == 6
    assert ad.credentials == ()
    # the REAL pixel is 10 degrees / 40,000 = 0.00025 deg = 27.83 m at the
    # equator, a thousandth of the family's 27.83 km reference
    assert abs(ad.log2_fp - np.log2(0.001)) < 1e-4
    assert abs(ad.log2_dt - 6.190) < 1e-3
    # 261 tiles, MEASURED in the producer's own directory on 2026-09-20
    assert len(c30.TILES_2020) == len(set(c30.TILES_2020)) == 261
    assert set(c30.TILES_2020) < set(c30.all_tiles())
    assert len(c30.all_tiles()) == 504
    assert ad.tiles == list(c30.TILES_2020)
    assert len(ad.specs()) == 261 * 16


def test_the_tile_set_is_a_subset_of_lossyear_s(monkeypatch):
    """Every canopy tile has a Hansen loss-year twin; nineteen do not have a
    canopy tile. Measured 2026-09-20 — it is why a reader can always ask both
    stores for the same place."""
    clear_env(monkeypatch)
    canopy = set(c30.TILES_2020)
    loss = set(ly.TILES_WITH_LOSSYEAR)
    assert canopy < loss
    assert len(loss - canopy) == 19
    assert "00N_100W" in loss - canopy and "50N_000E" in canopy


def test_the_bin_is_the_one_holding_the_first_of_july(monkeypatch):
    clear_env(monkeypatch)
    b = c30.RECORD_BIN
    assert sh.bin_start_date(b) <= dt.date(2020, 7, 1)
    assert sh.bin_start_date(b + 1) > dt.date(2020, 7, 1)
    assert (b, str(sh.bin_start_date(b))) == (2812, "2020-06-30")
    assert c30.Canopy30Adapter.first_year == 2020
    # and the probe month is the month the FRAME'S SLOT begins in, which is
    # June — the bin straddles the month boundary
    assert c30.Canopy30Adapter.smoke_probe_month == "2020-06"


def test_tile_names_and_group_names_round_trip(monkeypatch):
    clear_env(monkeypatch)
    for t in c30.all_tiles():
        la, lo = c30.tile_corner(t)
        assert c30.tile_name(la, lo) == t
        assert -50 <= la <= 80 and -180 <= lo <= 170
    for bad in ("00N_060", "90N_000E_", "0N_60W", "00X_060W"):
        with pytest.raises(c30.FormatError):
            c30.tile_corner(bad)
    for r in range(4):
        for c in range(4):
            g = c30.group_name("00N_060W", r, c)
            assert c30.group_parts(g) == ("00N_060W", r, c)
    with pytest.raises(c30.FormatError):
        c30.group_parts("00N_060W")


def test_the_sub_tiles_tile_their_ten_degree_tile_exactly(monkeypatch):
    clear_env(monkeypatch)
    for tile in ("50N_000E", "00N_060W", "20S_010E"):
        la, lo = c30.tile_corner(tile)
        seen = set()
        for r in range(c30.SUB):
            for c in range(c30.SUB):
                g = c30.grid_of(c30.group_name(tile, r, c))
                assert (g["H"], g["W"]) == (10000, 10000)
                assert abs(g["dx"] - 0.00025) < 1e-12 and g["dy"] == -g["dx"]
                w, s, e, n = g["extent"]
                assert abs(w - (lo + 2.5 * c)) < 1e-9
                assert abs(n - (la - 2.5 * r)) < 1e-9
                assert abs(e - w - 2.5) < 1e-9 and abs(n - s - 2.5) < 1e-9
                seen.add((w, n))
        assert len(seen) == 16
        assert min(w for w, _ in seen) == lo
        assert max(n for _, n in seen) == la


def test_the_spec_carries_the_annual_rule_and_the_value_legend(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("CANOPY30_TILES", "50N_000E")
    sp = c30.Canopy30Adapter().specs()
    assert len(sp) == 16
    s = sp["50N_000E_r1c3"]
    assert (s["H"], s["W"], s["C"]) == (10000, 10000, 1)
    # 10000 / 256 = 39.06 -> 40 tiles a side with a 240-pixel pad
    assert (s["n_tiles_y"], s["n_tiles_x"]) == (40, 40)
    assert s["pad_rows"] == 240 and s["pad_cols"] == 240
    assert s["missing"] == 255 and s["dtype"] == "uint8"
    assert s["compression"] == {"codec": "zstd", "level": 6,
                                "unit": "one zstd frame per tile"}
    assert s["grid"]["crs"] == "EPSG:4326"
    assert s["grid"]["x0"] == 7.5 and s["grid"]["y0"] == 47.5
    assert s["grid"]["tile_10deg"] == "50N_000E"
    assert s["channels"][0]["max"] == 100.0
    am = s["annual_map"]
    assert am["bin"] == 2812 and am["period"] == ["2020-01-01", "2020-12-31"]
    assert am["bin_days"] == ["2020-06-30", "2020-07-04"]
    assert am["values"]["0"].startswith("no woody vegetation")
    # a plain geographic grid stores no corner table
    assert "tile_corners" not in s and "no_tile_corners" in s["grid"]


def test_the_listing_parser_refuses_a_duplicate_and_counts_the_rest():
    html = c30.listing_html([("2020_00N_060W.tif", "586M"),
                             ("2020_50N_000E.tif", "290M"),
                             ("README.txt", "2K")])
    files, counts = c30.parse_listing(html)
    assert sorted(files) == ["00N_060W", "50N_000E"]
    assert files["00N_060W"] == 586 * 1024 ** 2
    assert counts["files_other"] == 1
    dup = c30.listing_html([("2020_00N_060W.tif", "586M"),
                            ("2020_00N_060W.tif", "586M")])
    with pytest.raises(c30.FormatError):
        c30.parse_listing(dup)


# ============================================================== the smoke ==
def test_smoke_reads_every_frame_back_exactly(smoke):
    res = smoke["check"]
    assert res["frames_equal"] == N_REAL
    assert res["by_reason"]["absent_upstream"] == N_ABSENT
    assert res["frames_absent"] == N_ABSENT
    sm = json.load(open(os.path.join(smoke["ctx"].store, "store.json")))
    assert sm["tier"] == "G" and sm["layout"] == "sharded"
    assert sm["frames_missing_by_reason"]["absent_upstream"] == N_ABSENT
    assert sorted(sm["groups"]) == sorted(SMOKE_GROUPS)
    assert sm["per_year"] == {
        "2020": {g: (0 if g.startswith(c30.SMOKE_ABSENT) else 1)
                 for g in SMOKE_GROUPS}}
    assert sm["C"] == 1 and sm["dtype"] == "uint8"
    assert sm["counts"]["out_of_bounds"] == {"canopy_height": 1}
    assert sm["counts"]["height_codes"] == {str(c30.SMOKE_OOB_VALUE): 1}
    assert "SMOKE GRID" in sm["notes"] and "RESTRICTED BUILD" in sm["notes"]
    assert sm["licence"]["redistribution_confirmed"] is True


def test_the_unpublished_tile_is_zero_length_frames_not_a_gap(smoke):
    for r in range(c30.SUB):
        for c in range(c30.SUB):
            g = c30.group_name(c30.SMOKE_ABSENT, r, c)
            grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, g))
            arr = grp.shard_index
            assert len(arr) == 1
            row = arr[0]
            assert int(row["bin"]) == c30.RECORD_BIN
            assert int(row["frames_present"]) == 0
            assert int(row["frames_missing"]) == 1
            assert int(row["nbytes"]) == 0 and int(row["tiles_stored"]) == 0
            assert grp.entry(c30.RECORD_BIN, 0, 0, 0) == (sh.FRAME_ABSENT, 0)
            assert grp.read_frame(c30.RECORD_BIN, 0) is None


def test_every_real_sub_tile_reads_back_the_source(smoke):
    truth = smoke["truth"]
    spx = c30.SMOKE_PX // c30.SUB
    for t in c30.SMOKE_TILES:
        for r in range(c30.SUB):
            for c in range(c30.SUB):
                g = c30.group_name(t, r, c)
                want, why = truth[(g, c30.RECORD_BIN, 0)]
                assert why is None
                grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, g))
                got = grp.read_frame(c30.RECORD_BIN, 0, raw=True)
                assert got.shape == (spx, spx, 1)
                assert np.array_equal(got, want)
                v = got[:, :, 0]
                if (v != 255).any():
                    assert v[v != 255].max() <= c30.MAX_HEIGHT


def test_the_store_is_DENSE_and_zeros_are_nearly_free(smoke):
    """The product has no water code, so every pixel is a measurement: the
    valid fraction is 1.0 everywhere and no tile is empty. The all-zero
    sub-tiles nevertheless compress to a few dozen bytes, which is the whole
    reason a dense 30 m store is affordable at all."""
    small = []
    for t in c30.SMOKE_TILES:
        for r in range(c30.SUB):
            for c in range(c30.SUB):
                g = c30.group_name(t, r, c)
                arr = sh.load_shard_index(os.path.join(
                    smoke["ctx"].store, g, "shard_index.npy"))
                row = arr[0]
                assert int(row["tiles_stored"]) == 1
                # one 200 m pixel of one sub-tile is masked; everything else
                # is a measurement
                px = c30.SMOKE_PX // c30.SUB
                assert int(row["valid_pixels"][0]) in (px * px, px * px - 1)
                assert float(row["valid_fraction"][0]) > 0.999
                if int(row["nbytes"]) < 200:
                    small.append(g)
    assert small, "no all-zero sub-tile compressed under 200 bytes"


def test_the_probe_measured_the_same_frames_and_some_bytes(smoke):
    p = smoke["probe"]
    assert p["frames_fetched"] == N_REAL
    assert p["out_of_bounds_stored"] == 0
    assert p["out_of_bounds"] == {"canopy_height": 1}
    assert p["bytes_fetched"] > 0
    assert p["zstd_level"] == 6
    assert p["month"] == "2020-06"
    for t in c30.SMOKE_TILES:
        for r in range(c30.SUB):
            for c in range(c30.SUB):
                gp = p["groups"][c30.group_name(t, r, c)]
                assert gp["frames_fetched"] == 1 and gp["C"] == 1
                assert gp["record_frames"] == 1
                assert gp["tiles_empty"] == 0
                assert gp["valid_fraction"]["canopy_height"] > 0.999


def test_a_window_outside_the_record_writes_nothing(tmp_path):
    src, _ = small_archive(str(tmp_path))
    ctx = ctx_for(str(tmp_path), src, lo="2019-01-01", hi="2019-01-10")
    run(ctx, ["index", "fetch"])
    for y in ctx.years:
        c = json.load(open(os.path.join(ctx.year_dir(y), "counts.json")))
        assert sum(v["bins"] for v in c["groups"].values()) == 0
        assert c["counts"]["bins_outside_record"] > 0


def test_index_refuses_a_changed_tile_set(tmp_path):
    src, _ = small_archive(str(tmp_path))
    p = os.path.join(src, "canopy30", "index.html")
    html = open(p).read().replace('href="2020_20N_000E.tif"',
                                  'href="2020_20N_999E.tif"')
    open(p, "w").write(html)
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index"])
    msg = str(e.value)
    assert "TILES_2020" in msg or "not a Forest_height" in msg


def test_index_refuses_an_empty_listing(tmp_path):
    src, _ = small_archive(str(tmp_path))
    p = os.path.join(src, "canopy30", "index.html")
    open(p, "w").write(c30.listing_html([]))
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index"])
    assert "lists no" in str(e.value)


def test_a_relabelled_geotiff_is_refused_not_stored(tmp_path):
    """A tile whose transform does not match its name is a REFUSAL."""
    src, _ = small_archive(str(tmp_path))
    bad = os.path.join(src, "canopy30", c30.object_name("50N_000E"))
    a = c30.smoke_field("50N_000E", c30.SMOKE_PX)
    c30.write_tif(bad, "40N_000E", a, c30.SMOKE_PX)      # wrong corner
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index", "fetch"])
    assert "transform" in str(e.value)
