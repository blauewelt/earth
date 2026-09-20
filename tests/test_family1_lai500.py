#!/usr/bin/env python3
"""The `lai500` adapter's smoke (family 1.0.tf, exception E7, E-082 wave 6) —
no network.

    python3 -m pytest -q tests/test_family1_lai500.py

The synthetic archive (`lai500.make_smoke_sources`) is the real shape: a
`cmr_terra_<year>.json` holding what NASA's Common Metadata Repository returns
and real HDF-EOS2 granules with a real `StructMetadata.0` for their tile, on a
240 x 240 grid (`LAI500_SMOKE_PX`) instead of the product's 2,400. THE WINDOW
STRADDLES NEW YEAR ON PURPOSE — composites A2020353, A2020361, A2021001 and
A2021009, where the year's last composite runs two days into the next year and
overlaps the next year's first — because placing those correctly is what the
frame_table exists for. Two tiles; one composite is published for one of them
and not the other, so one frame is `absent_upstream`; a patch of the
producer's own non-terrestrial class codes (254 water, 255 fill, 250 urban)
exercises `outside_valid_range_codes`. Expected counts are exact.
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
from family1.adapters import lai500 as la                       # noqa: E402

pytest.importorskip("pyhdf")

ENV = ("LAI500_SMOKE_PX", "LAI500_TILES", "LAI500_GROUPS",
       "LAI500_MAX_YEARS")


def clear_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("lai500_smoke"))
    keep = {k: os.environ.get(k) for k in ENV}
    try:
        return b1.run_smoke("lai500", root=root, keep=True)
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def small_archive(tmp, **kw):
    src = os.path.join(tmp, "src")
    d_lo, d_hi = (b10.parse_date(x) for x in la.Lai500Adapter.smoke_window)
    truth = la.make_smoke_sources(src, d_lo, d_hi, **kw)
    return src, truth


def ctx_for(tmp, src, lo=None, hi=None, **over):
    w = la.Lai500Adapter.smoke_window
    d = dict(store="lai500", work=os.path.join(tmp, "work"), source_dir=src,
             start=lo or w[0], end=hi or w[1], stage="all", force=False,
             attempts=1, qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS,
             assemble="auto", parts_from_hub=False, push_parts=False,
             allow_missing_years=False, allow_unconfirmed_licence=False,
             probe_month="", smoke=False)
    d.update(over)
    ad = la.Lai500Adapter()
    ctx = b10.Ctx(argparse.Namespace(**d), adapter=ad,
                  layout=b1.layout_for(ad))
    return b1.prepare_grid_ctx(ctx)


def run(ctx, stages):
    b10.run_stages(ctx, stages, stage_fn=b1.GRID_STAGE_FN, deps=b1.DEPS)


# ================================================================ facts ====
def test_registered_with_the_measured_tile_set(monkeypatch):
    clear_env(monkeypatch)
    assert fam.REGISTRY["lai500"] is la.Lai500Adapter
    ad = la.Lai500Adapter()
    assert b1.is_grid(ad) and ad.layout == "sharded"
    assert ad.family == "1tf" and ad.distribution == "public"
    assert ad.credentials == ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    assert ad.C == 4 and ad.dtype == "float16"
    assert ad.channel_names == ["lai", "fpar", "lai_stddev", "fpar_lai_qc"]
    assert (ad.frames_per_bin, ad.frame_seconds) == (1, sh.BIN_SECONDS)
    assert ad.frames_per_bin * ad.frame_seconds == sh.BIN_SECONDS
    # the footprint in TIME is the composite, not the slot: log2(8/5)
    assert abs(ad.log2_dt - np.log2(8.0 / 5.0)) < 1e-9
    assert len(la.TILES) == len(set(la.TILES)) == 290
    # the default is Terra alone, and its groups are <sensor>_<tile>
    assert ad.groups == ["terra"]
    assert ad.group_name("terra", "h18v04") == "terra_h18v04"
    assert ad.group_parts("terra_h18v04") == ("terra", "h18v04")
    assert len(ad.specs()) == 290
    # the size gate is 1 year unless raised deliberately
    assert ad.max_years == 1


def test_the_three_sensors_and_their_file_formats(monkeypatch):
    """VIIRS' granules are HDF5 and its datasets drop MODIS' `_500m` suffix —
    measured on CMR and on LP DAAC's own layer tables."""
    clear_env(monkeypatch)
    assert la.COLLECTIONS["terra"][:3] == ("MOD15A2H", "061", "hdf4")
    assert la.COLLECTIONS["aqua"][:3] == ("MYD15A2H", "061", "hdf4")
    assert la.COLLECTIONS["viirs"][:3] == ("VNP15A2H", "002", "hdf5")
    assert ms.SUFFIX["hdf5"] == ".h5" and ms.SUFFIX["hdf4"] == ".hdf"
    ad = la.Lai500Adapter()
    assert ad.datasets_of("terra")[0] == "Lai_500m"
    assert ad.datasets_of("viirs")[0] == "Lai"
    monkeypatch.setenv("LAI500_GROUPS", "terra,viirs")
    ad2 = la.Lai500Adapter()
    assert ad2.groups == ["terra", "viirs"]
    assert len(ad2.specs()) == 2 * 290
    monkeypatch.setenv("LAI500_GROUPS", "landsat")
    with pytest.raises(SystemExit):
        la.Lai500Adapter()


def test_the_composite_calendar_is_46_eight_day_periods_a_year(monkeypatch):
    clear_env(monkeypatch)
    for y in (2000, 2004, 2019, 2020, 2024, 2035):
        doys = la.composite_doys(y)
        assert len(doys) == 46
        assert doys[0] == 1 and doys[-1] == 361
        assert all(b - a == 8 for a, b in zip(doys, doys[1:]))
    # every composite is EIGHT days, including the last of a year, which runs
    # into the next — measured on the producer's own time_start/time_end
    d0, d1 = la.composite_days((2020, 361))
    assert (str(d0), str(d1)) == ("2020-12-26", "2021-01-02")
    d0, d1 = la.composite_days((2019, 361))
    assert (str(d0), str(d1)) == ("2019-12-27", "2020-01-03")
    assert la.composite_days((2021, 1))[0] == dt.date(2021, 1, 1)
    # ... so the year's last composite OVERLAPS the next year's first
    assert la.composite_days((2020, 361))[1] >= la.composite_days((2021, 1))[0]


def test_frames_per_bin_one_has_no_collision_over_the_record(monkeypatch):
    """The measurement F = 1 rests on: 1,656 composites, 1,656 distinct bins,
    minimum midpoint gap exactly 5.0 days."""
    clear_env(monkeypatch)
    assert la.slot_collisions(sh.BIN_SECONDS, 2000, 2035) == {}
    rows = la.frame_table(sh.BIN_SECONDS, 2000, 2035)
    assert len(rows) == 46 * 36 == 1656
    bins = [r[0] for r in rows]
    assert len(set(bins)) == len(bins)
    assert all(r[1] == 0 for r in rows)
    assert all(r[6] == 8 for r in rows)
    mids = []
    for r in rows:
        d0, d1 = la.composite_days((r[2], r[3]))
        s = b10.seconds_since_epoch(d0)
        e = b10.seconds_since_epoch(d1) + 86400
        mids.append((s + e) // 2)
    mids.sort()
    assert min(b - a for a, b in zip(mids, mids[1:])) == 5 * 86400


def test_the_year_boundary_lands_in_two_different_bins(monkeypatch):
    """The case the frame_table exists for: A2020361 and A2021001 overlap by
    two days, and their midpoints are six days apart, so they go in adjacent
    bins and neither is lost."""
    clear_env(monkeypatch)
    b_last = la.composite_slot((2020, 361), sh.BIN_SECONDS)
    b_first = la.composite_slot((2021, 1), sh.BIN_SECONDS)
    assert b_last != b_first
    assert b_last[1] == b_first[1] == 0
    assert b_first[0] - b_last[0] == 1
    # and the frame_table says which days each really covers
    rows = {(r[2], r[3]): r for r in la.frame_table(sh.BIN_SECONDS, 2020,
                                                    2021)}
    assert rows[(2020, 361)][:2] == list(b_last)
    assert rows[(2020, 361)][4:] == ["2020-12-26", "2021-01-02", 8]
    assert rows[(2021, 1)][:2] == list(b_first)
    assert rows[(2021, 1)][4:] == ["2021-01-01", "2021-01-08", 8]
    # a non-leap year is the tightest case and still lands in two bins
    assert (la.composite_slot((2019, 361), sh.BIN_SECONDS)[0]
            != la.composite_slot((2020, 1), sh.BIN_SECONDS)[0])


def test_an_a_date_off_the_calendar_is_a_refusal(monkeypatch):
    clear_env(monkeypatch)
    assert la.key_of(dt.date(2020, 7, 3)) == (2020, 185)
    with pytest.raises(ms.FormatError):
        la.key_of(dt.date(2020, 7, 4))


def test_the_spec_carries_the_exception_and_the_frame_table(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("LAI500_TILES", "h18v04")
    sp = la.Lai500Adapter().specs()
    assert list(sp) == ["terra_h18v04"]
    s = sp["terra_h18v04"]
    assert (s["H"], s["W"], s["C"]) == (2400, 2400, 4)
    assert (s["n_tiles_y"], s["n_tiles_x"]) == (10, 10)
    assert s["exception"]["name"] == "E7"
    assert abs(s["exception"]["log2_dt"] - np.log2(1.6)) < 1e-9
    assert s["source_scalings"]["lai"]["scale_factor"] == 0.1
    assert s["source_scalings"]["fpar"]["scale_factor"] == 0.01
    assert s["source_scalings"]["fpar"]["valid_range"] == [0, 100]
    assert s["frame_table_span_years"] == [2000, 2035]
    assert len(s["frame_table"]) == 1656


# ============================================================== the smoke ==
def test_smoke_reads_every_frame_back_exactly(smoke):
    res = smoke["check"]
    # four composites x two tiles, less the one the archive does not publish
    assert res["frames_equal"] == 7
    assert res["by_reason"]["absent_upstream"] == 1
    sm = json.load(open(os.path.join(smoke["ctx"].store, "store.json")))
    assert sm["tier"] == "G" and sm["layout"] == "sharded"
    assert sorted(sm["groups"]) == ["terra_h12v09", "terra_h18v04"]
    assert sm["C"] == 4 and sm["dtype"] == "float16"
    assert sm["frames_missing_by_reason"]["absent_upstream"] == 1
    # every bin with no composite midpoint wrote NO shard
    assert sm["frames_missing_by_reason"][sh.FRAME_NO_FRAME_IN_BIN] > 0
    # the producer's own non-terrestrial class codes, counted by raw value
    codes = sm["counts"]["outside_valid_range_codes"]
    assert codes["lai:254"] > 1000        # the water patch
    assert codes["lai:255"] == 1          # the fill pixel
    assert codes["lai:250"] == 1          # the urban pixel
    assert "SMOKE GRID" in sm["notes"] and "RESTRICTED BUILD" in sm["notes"]


def test_the_bins_written_are_exactly_the_composites(smoke):
    want = {la.composite_slot(k, sh.BIN_SECONDS)[0]
            for k in la.SMOKE_COMPOSITES}
    assert len(want) == len(la.SMOKE_COMPOSITES)
    for g in ("terra_h18v04", "terra_h12v09"):
        arr = sh.load_shard_index(os.path.join(smoke["ctx"].store, g,
                                               "shard_index.npy"))
        assert {int(r["bin"]) for r in arr} == want, g


def test_the_year_boundary_composites_are_both_stored(smoke):
    """A2020361 and A2021001 are two days apart in coverage and land in
    adjacent bins — the case the whole frame_table exists for."""
    g = "terra_h18v04"
    grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, g))
    b_last = la.composite_slot((2020, 361), sh.BIN_SECONDS)[0]
    b_first = la.composite_slot((2021, 1), sh.BIN_SECONDS)[0]
    a = grp.read_frame(b_last, 0)
    b = grp.read_frame(b_first, 0)
    assert a is not None and b is not None
    assert not np.array_equal(a, b, equal_nan=True)
    table = {(r[0], r[1]): r for r in grp.spec["frame_table"]}
    assert table[(b_last, 0)][4:] == ["2020-12-26", "2021-01-02", 8]
    assert table[(b_first, 0)][4:] == ["2021-01-01", "2021-01-08", 8]


def test_every_frame_reads_back_the_scaled_source(smoke):
    truth = smoke["truth"]
    for (g, b, f), (want, why) in truth.items():
        if want is None:
            continue
        grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, g))
        got = grp.read_frame(b, f, raw=True)
        assert got is not None, (g, b, f)
        assert got.shape == (la.SMOKE_PX, la.SMOKE_PX, 4)
        assert np.array_equal(got, want, equal_nan=True), (g, b, f)


def test_the_scalings_are_the_producers(smoke):
    """LAI in m2/m2 (raw x 0.1), FPAR a FRACTION (raw x 0.01), the quality
    bits carried as whole numbers."""
    g = "terra_h18v04"
    b = la.composite_slot((2020, 353), sh.BIN_SECONDS)[0]
    a = sh.ShardedGroup(os.path.join(smoke["ctx"].store, g)).read_frame(b, 0)
    lai, fpar, qc = a[:, :, 0], a[:, :, 1], a[:, :, 3]
    ok = np.isfinite(lai)
    assert ok.any()
    assert lai[ok].min() >= 0.0 and lai[ok].max() <= 10.0
    okf = np.isfinite(fpar)
    assert fpar[okf].min() >= 0.0 and fpar[okf].max() <= 1.0
    okq = np.isfinite(qc)
    assert np.all(qc[okq] == np.rint(qc[okq]))
    assert qc[okq].max() <= 254.0


def test_the_probe_measured_the_same_frames_and_some_bytes(smoke):
    p = smoke["probe"]
    assert p["frames_fetched"] == 4          # December's two composites x 2
    assert p["out_of_bounds_stored"] == 0
    assert p["bytes_fetched"] > 0
    assert p["month"] == "2020-12"
    for gp in p["groups"].values():
        assert gp["C"] == 4
        assert gp["record_frames"] and gp["record_frames"] > 40


# ============================================================== refusals ===
def test_the_size_gate_refuses_a_two_year_window(tmp_path, monkeypatch):
    src, _ = small_archive(str(tmp_path))
    monkeypatch.delenv("LAI500_MAX_YEARS", raising=False)
    ctx = ctx_for(str(tmp_path), src)
    assert ctx.adapter.max_years == 1
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index", "fetch"])
    assert "LAI500_MAX_YEARS" in str(e.value)


def test_index_refuses_a_tile_outside_the_declared_set(tmp_path):
    src, _ = small_archive(str(tmp_path))
    p = os.path.join(src, "lai500", "cmr_terra_2021.json")
    d = json.load(open(p))
    e = dict(d["feed"]["entry"][0])
    e["title"] = e["producer_granule_id"] = \
        "MOD15A2H.A2021001.h00v00.061.2026000000001"
    d["feed"]["entry"].append(e)
    json.dump(d, open(p, "w"))
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e2:
        run(ctx, ["index"])
    assert "TILES" in str(e2.value)


def test_index_refuses_an_empty_listing(tmp_path):
    src, _ = small_archive(str(tmp_path))
    for y in (2020, 2021):
        p = os.path.join(src, "lai500", f"cmr_terra_{y}.json")
        json.dump({"feed": {"entry": []}}, open(p, "w"))
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index"])
    assert "no granule" in str(e.value)


def test_a_granule_on_the_wrong_tile_is_refused_not_stored(tmp_path):
    src, _ = small_archive(str(tmp_path))
    gdir = os.path.join(src, "lai500", "granules", "terra")
    name = sorted(n for n in os.listdir(gdir) if "h18v04" in n)[0]
    arrays = la.smoke_fields("h18v04", (2020, 353), la.SMOKE_PX)
    ms.write_smoke_hdf4(os.path.join(gdir, name), arrays, 12, 9, la.SMOKE_PX)
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index", "fetch"])
    assert "StructMetadata" in str(e.value)
