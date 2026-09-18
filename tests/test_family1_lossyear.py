#!/usr/bin/env python3
"""The `lossyear` adapter's smoke (family 1.0.tf, E-082 wave 2) — no network.

    python3 -m pytest -q tests/test_family1_lossyear.py

The synthetic bucket (`lossyear.make_smoke_sources`) is the real shape: a
`listing.json` holding what the Google Cloud Storage JSON API returns
(`items` of {name, size}), three land tiles with all three variables, one tile
declared in `LOSSYEAR_TILES` that has `treecover2000` and `datamask` and NO
`lossyear` — as 224 of the product's 504 tiles are — and the whole 504-tile
datamask grid plus the whole 280-tile lossyear set, so `index`'s two
set checks see the product they are written to falsify. The GeoTIFFs are real
LZW GeoTIFFs on a 512 x 512 grid (`LOSSYEAR_SMOKE_PX`), cut into the same
4 x 4 sub-tiles the real store uses, with one lossyear pixel of 40 and one
treecover of 120 to exercise the out-of-bounds count. Expected counts are
exact.
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
from family1.adapters import lossyear as ly                     # noqa: E402

pytest.importorskip("rasterio")

# The adapter reads both at CONSTRUCTION time (so the fresh instance
# `stage_probe_grid` builds for itself sees the smoke's grid); every test that
# wants the real product clears them first, and every test that wants the
# smoke's gets them from `make_smoke_sources`.
ENV = ("LOSSYEAR_SMOKE_PX", "LOSSYEAR_TILES")
SMOKE_GROUPS = [ly.group_name(t, r, c)
                for t in ly.SMOKE_TILES + (ly.SMOKE_ABSENT,)
                for r in range(ly.SUB) for c in range(ly.SUB)]
N_REAL = len(ly.SMOKE_TILES) * ly.SUB * ly.SUB      # 48
N_ABSENT = ly.SUB * ly.SUB                          # 16


def clear_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("lossyear_smoke"))
    keep = {k: os.environ.get(k) for k in ENV}
    try:
        return b1.run_smoke("lossyear", root=root, keep=True)
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def small_bucket(tmp):
    src = os.path.join(tmp, "src")
    d_lo, d_hi = (b10.parse_date(x) for x in ly.LossyearAdapter.smoke_window)
    truth = ly.make_smoke_sources(src, d_lo, d_hi)
    return src, truth


def ctx_for(tmp, src, lo=None, hi=None, **over):
    w = ly.LossyearAdapter.smoke_window
    d = dict(store="lossyear", work=os.path.join(tmp, "work"), source_dir=src,
             start=lo or w[0], end=hi or w[1], stage="all", force=False,
             attempts=1, qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS,
             assemble="auto", parts_from_hub=False, push_parts=False,
             allow_missing_years=False, allow_unconfirmed_licence=False,
             probe_month="", smoke=False)
    d.update(over)
    ad = ly.LossyearAdapter()
    ctx = b10.Ctx(argparse.Namespace(**d), adapter=ad,
                  layout=b1.layout_for(ad))
    return b1.prepare_grid_ctx(ctx)


def run(ctx, stages):
    b10.run_stages(ctx, stages, stage_fn=b1.GRID_STAGE_FN, deps=b1.DEPS)


# ================================================================ facts ====
def test_registered_with_the_real_grid_and_the_annual_bin(monkeypatch):
    clear_env(monkeypatch)
    assert fam.REGISTRY["lossyear"] is ly.LossyearAdapter
    ad = ly.LossyearAdapter()
    assert b1.is_grid(ad) and ad.layout == "sharded"
    assert ad.family == "1tf" and ad.distribution == "public"
    assert ad.licence["redistribution"] == "attribution"
    assert ad.licence["redistribution_confirmed"] is True
    assert ad.channel_names == ["lossyear", "treecover2000"]
    assert ad.dtype == "uint8" and ad.C == 2
    assert (ad.frames_per_bin, ad.frame_seconds) == (1, sh.BIN_SECONDS)
    assert ad.frames_per_bin * ad.frame_seconds == sh.BIN_SECONDS
    assert ad.zstd_level == 6
    # the REAL pixel is 10 degrees / 40,000 = 0.00025 deg = 27.83 m at the
    # equator, exactly a thousandth of the family's 27.83 km reference, so
    # log2_fp is log2(0.001) = -9.966 and not the -9.857 a nominal "30 m"
    # would give. The note's "a 30 m pixel is -9.9" is both to one decimal.
    assert abs(ad.log2_fp - (-9.9658)) < 1e-3
    assert abs(ad.log2_fp - np.log2(0.001)) < 1e-4
    assert abs(ad.log2_dt - 6.190) < 1e-3
    # the product's documented 10-degree grid, and the 280 that carry loss
    assert len(ly.all_tiles()) == len(set(ly.all_tiles())) == 504
    assert len(ly.TILES_WITH_LOSSYEAR) == 280
    assert set(ly.TILES_WITH_LOSSYEAR) < set(ly.all_tiles())
    assert ad.tiles == list(ly.TILES_WITH_LOSSYEAR)
    # the measured ocean tiles are NOT groups; the measured land ones are
    for t in ("00N_180W", "40S_010E", "80N_060W"):
        assert t not in ly.TILES_WITH_LOSSYEAR
    for t in ("20N_000E", "30S_130E", "50N_090E", "00N_060W", "00N_020E"):
        assert t in ly.TILES_WITH_LOSSYEAR
    assert len(ad.specs()) == 280 * 16


def test_the_annual_bin_is_the_first_one_that_begins_after_2025():
    b = ly.RECORD_BIN
    assert sh.bin_start_date(b) > dt.date(2025, 12, 31)
    assert sh.bin_start_date(b - 1) <= dt.date(2025, 12, 31)
    assert (b, str(sh.bin_start_date(b))) == (3215, "2026-01-05")
    assert ly.LossyearAdapter.first_year == 2026


def test_tile_names_and_group_names_round_trip(monkeypatch):
    clear_env(monkeypatch)
    for t in ly.all_tiles():
        la, lo = ly.tile_corner(t)
        assert ly.tile_name(la, lo) == t
        assert -50 <= la <= 80 and -180 <= lo <= 170
    for bad in ("00N_060", "90N_000E_", "0N_60W", "00X_060W"):
        with pytest.raises(ly.FormatError):
            ly.tile_corner(bad)
    for r in range(4):
        for c in range(4):
            g = ly.group_name("00N_060W", r, c)
            assert ly.group_parts(g) == ("00N_060W", r, c)
    with pytest.raises(ly.FormatError):
        ly.group_parts("00N_060W")


def test_the_sub_tiles_tile_their_hansen_tile_exactly(monkeypatch):
    clear_env(monkeypatch)
    # the 16 sub-tiles of one tile cover it with no gap and no overlap
    for tile in ("00N_060W", "50S_170E", "80N_180W"):
        la, lo = ly.tile_corner(tile)
        seen = set()
        for r in range(ly.SUB):
            for c in range(ly.SUB):
                g = ly.grid_of(ly.group_name(tile, r, c))
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


def test_the_spec_is_the_real_geotiff_grid_and_carries_the_annual_rule(
        monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("LOSSYEAR_TILES", "00N_060W")
    sp = ly.LossyearAdapter().specs()
    assert len(sp) == 16
    s = sp["00N_060W_r1c3"]
    assert (s["H"], s["W"], s["C"]) == (10000, 10000, 2)
    # 10000 / 256 = 39.06 -> 40 tiles a side with a 240-pixel pad
    assert (s["n_tiles_y"], s["n_tiles_x"]) == (40, 40)
    assert s["pad_rows"] == 240 and s["pad_cols"] == 240
    assert s["missing"] == 255 and s["dtype"] == "uint8"
    assert s["compression"] == {"codec": "zstd", "level": 6,
                                "unit": "one zstd frame per tile"}
    assert s["grid"]["crs"] == "EPSG:4326"
    assert s["grid"]["x0"] == -52.5 and s["grid"]["y0"] == -2.5
    assert s["grid"]["hansen_tile"] == "00N_060W"
    assert s["grid"]["pixel_window_in_tile"] == [10000, 20000, 30000, 40000]
    assert [c["max"] for c in s["channels"]] == [25.0, 100.0]
    at = s["annual_target"]
    assert at["bin"] == 3215 and at["period"][1] == "2025-12-31"
    assert at["bin_days"] == ["2026-01-05", "2026-01-09"]
    assert at["lossyear_values"]["255"].startswith("not mapped land")
    # a plain geographic grid stores no corner table
    assert "tile_corners" not in s and "no_tile_corners" in s["grid"]


def test_the_listing_parser_refuses_a_duplicate_and_counts_the_rest():
    items = [{"name": f"{ly.VERSION}/{ly.object_name(v, '00N_060W')}",
              "size": "10"} for v in ly.VARIABLES]
    items += [{"name": f"{ly.VERSION}/{ly.object_name('gain', '00N_060W')}",
               "size": "7"},
              {"name": f"{ly.VERSION}/download.html", "size": "1"}]
    by, counts = ly.parse_listing(items)
    assert by["lossyear"] == {"00N_060W": 10}
    assert counts["other"] == 1
    assert counts["other_variables"] == {"gain": 1}
    with pytest.raises(ly.FormatError):
        ly.parse_listing(items + [items[0]])


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
    # every group has a row in the record's year; the absent tile's
    # sixteen have ZERO frames present rather than no row at all
    assert sm["per_year"] == {
        "2026": {g: (0 if g.startswith(ly.SMOKE_ABSENT) else 1)
                 for g in SMOKE_GROUPS}}
    assert sm["C"] == 2 and sm["dtype"] == "uint8"
    # the out-of-bounds pixels were counted and never stored
    assert sm["counts"]["out_of_bounds"] == {"lossyear": 1,
                                             "treecover2000": 1}
    assert "SMOKE GRID" in sm["notes"] and "RESTRICTED BUILD" in sm["notes"]
    assert sm["licence"]["redistribution_confirmed"] is True


def test_the_absent_tiles_sub_tiles_are_zero_length_frames_not_gaps(smoke):
    for r in range(ly.SUB):
        for c in range(ly.SUB):
            g = ly.group_name(ly.SMOKE_ABSENT, r, c)
            grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, g))
            arr = grp.shard_index
            assert len(arr) == 1
            row = arr[0]
            assert int(row["bin"]) == ly.RECORD_BIN
            assert int(row["frames_present"]) == 0
            assert int(row["frames_missing"]) == 1
            assert int(row["nbytes"]) == 0 and int(row["tiles_stored"]) == 0
            assert grp.entry(ly.RECORD_BIN, 0, 0, 0) == (sh.FRAME_ABSENT, 0)
            assert grp.read_tile(ly.RECORD_BIN, 0, 0, 0) is None
            assert grp.read_frame(ly.RECORD_BIN, 0) is None


def test_every_real_sub_tile_reads_back_the_masked_source(smoke):
    truth = smoke["truth"]
    spx = ly.SMOKE_PX // ly.SUB
    for t in ly.SMOKE_TILES:
        for r in range(ly.SUB):
            for c in range(ly.SUB):
                g = ly.group_name(t, r, c)
                want, why = truth[(g, ly.RECORD_BIN, 0)]
                assert why is None
                grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, g))
                got = grp.read_frame(ly.RECORD_BIN, 0, raw=True)
                assert got.shape == (spx, spx, 2)
                assert np.array_equal(got, want)
                # the two channels are missing in exactly the same places,
                # because one datamask decides both — except where a value
                # of its own was out of bounds and only IT was dropped
                differ = int(((got[:, :, 0] == 255)
                              != (got[:, :, 1] == 255)).sum())
                assert differ == (2 if (t == ly.SMOKE_OOB_TILE
                                        and (r, c) == (2, 2)) else 0)
                v, w = got[:, :, 0], got[:, :, 1]
                if (v != 255).any():
                    assert v[v != 255].max() <= ly.LAST_LOSS_YEAR
                    assert w[w != 255].max() <= 100


def test_the_mask_is_the_datamask_and_ocean_costs_nothing(smoke):
    """A sub-tile of the smoke field's corner is all ocean, so it stores no
    tile at all — the sparsity the whole design rests on."""
    empty = []
    for t in ly.SMOKE_TILES:
        for r in range(ly.SUB):
            for c in range(ly.SUB):
                g = ly.group_name(t, r, c)
                arr = sh.load_shard_index(os.path.join(
                    smoke["ctx"].store, g, "shard_index.npy"))
                if int(arr[0]["tiles_stored"]) == 0:
                    empty.append(g)
                    assert int(arr[0]["frames_present"]) == 1
                    assert int(arr[0]["valid_pixels"][0]) == 0
    # the four corner sub-tiles of each tile are outside the field's disc
    assert len(empty) >= 4 * len(ly.SMOKE_TILES)


def test_the_probe_measured_the_same_frames_and_some_bytes(smoke):
    p = smoke["probe"]
    assert p["frames_fetched"] == N_REAL
    assert p["out_of_bounds_stored"] == 0
    assert p["out_of_bounds"] == {"lossyear": 1, "treecover2000": 1}
    assert p["bytes_fetched"] > 0
    assert p["zstd_level"] == 6
    n_est = 0
    for t in ly.SMOKE_TILES:
        for r in range(ly.SUB):
            for c in range(ly.SUB):
                gp = p["groups"][ly.group_name(t, r, c)]
                assert gp["frames_fetched"] == 1 and gp["C"] == 2
                assert set(gp["valid_fraction"]) == {"lossyear",
                                                     "treecover2000"}
                assert gp["record_frames"] == 1
                if gp["estimate_store_bytes"]:
                    n_est += 1
    assert n_est == N_REAL
    for r in range(ly.SUB):
        for c in range(ly.SUB):
            g = ly.group_name(ly.SMOKE_ABSENT, r, c)
            fm = p["groups"][g]["frames_missing"]
            # one absence inside the record's bin, and the probe month's
            # other bins are after it
            assert fm["absent_upstream"] == 1
            assert set(fm) <= {"absent_upstream", "after_record"}


def test_a_window_outside_the_record_writes_nothing(tmp_path):
    src, _ = small_bucket(str(tmp_path))
    ctx = ctx_for(str(tmp_path), src, lo="2024-01-01", hi="2024-01-10")
    run(ctx, ["index", "fetch"])
    for y in ctx.years:
        c = json.load(open(os.path.join(ctx.year_dir(y), "counts.json")))
        assert sum(v["bins"] for v in c["groups"].values()) == 0
        assert c["counts"]["bins_outside_record"] > 0


def test_index_refuses_a_listing_that_is_not_the_504_tile_grid(tmp_path):
    src, _ = small_bucket(str(tmp_path))
    p = os.path.join(src, "lossyear", "listing.json")
    d = json.load(open(p))
    d["items"] = [i for i in d["items"]
                  if "datamask_10N_010E" not in i["name"]]
    json.dump(d, open(p, "w"))
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index"])
    assert "504-tile grid" in str(e.value)


def test_index_refuses_a_changed_lossyear_tile_set(tmp_path):
    src, _ = small_bucket(str(tmp_path))
    p = os.path.join(src, "lossyear", "listing.json")
    d = json.load(open(p))
    d["items"] = [i for i in d["items"]
                  if "lossyear_20N_000E" not in i["name"]]
    json.dump(d, open(p, "w"))
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index"])
    assert "TILES_WITH_LOSSYEAR" in str(e.value)


def test_index_refuses_a_lossyear_tile_without_its_mask(tmp_path):
    src, _ = small_bucket(str(tmp_path))
    p = os.path.join(src, "lossyear", "listing.json")
    d = json.load(open(p))
    d["items"] = [i for i in d["items"]
                  if "treecover2000_00N_060W" not in i["name"]]
    json.dump(d, open(p, "w"))
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index"])
    assert "no treecover2000" in str(e.value)


def test_index_refuses_an_empty_listing(tmp_path):
    src, _ = small_bucket(str(tmp_path))
    p = os.path.join(src, "lossyear", "listing.json")
    json.dump({"items": []}, open(p, "w"))
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index"])
    assert "no lossyear granule" in str(e.value)


def test_a_relabelled_geotiff_is_refused_not_stored(tmp_path):
    """A tile whose transform does not match its name is a REFUSAL."""
    src, _ = small_bucket(str(tmp_path))
    bad = os.path.join(src, "lossyear", ly.VERSION,
                       ly.object_name("lossyear", "00N_060W"))
    a, _, _ = ly.smoke_fields("00N_060W", ly.SMOKE_PX)
    ly.write_tif(bad, "10N_060W", a, ly.SMOKE_PX)      # wrong corner
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index", "fetch"])
    assert "transform" in str(e.value)


def test_a_missing_file_is_an_absence_and_the_year_is_not_marked(tmp_path):
    src, _ = small_bucket(str(tmp_path))
    os.remove(os.path.join(src, "lossyear", ly.VERSION,
                           ly.object_name("datamask", "00N_020E")))
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index", "fetch"])
    assert "could not be read" in str(e.value)
    assert any("00N_020E" in a["unit"] for a in ctx.absent)
    from build_family7 import marked
    assert not marked(ctx.root, "parts/2026")
    assert not marked(ctx.root, "fetch")


def test_check_store_verifies_every_stored_tile(smoke):
    st = sh.check_store(smoke["ctx"].store)
    assert st["files"] == len(sh.store_files(smoke["ctx"].store,
                                             sorted(SMOKE_GROUPS)))
    total = sum(v["tiles_checked"] for v in st["groups"].values())
    assert total == sum(v["tiles_stored"] for v in st["groups"].values())
    assert total > 0
