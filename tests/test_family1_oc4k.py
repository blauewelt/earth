#!/usr/bin/env python3
"""The `oc4k` adapter's smoke (family 1.gf, E-082 wave 2) — no network.

    python3 -m pytest -q tests/test_family1_oc4k.py

The synthetic archive (`oc4k.make_smoke_sources`) is the real layout: BOTH
products (`chlor_a/daily/v6.0/` and `kd/daily/v6.0/`) with a year-directory
listing, a per-year file listing carrying the archive's own
`00README_catalogue_and_licence.txt`, and netCDF-4 days on the declared grid
with `lat` DESCENDING and the netCDF default float fill over the synthetic
land. The record is 1997-09-04 .. 1997-09-18; 1997-09-10 is published by `kd`
and NOT by `chlor_a`, so the store's record — the days both products
publish — has a hole inside it; 1997-09-08 carries one kd_490 of 99 m-1 (out
of bounds) and 1997-09-09 one chlorophyll of 1e-6 mg m-3 (below the log
floor). Expected counts are exact.
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

pytest.importorskip("netCDF4")

ENV = ("OC4K_SMOKE_GRID",)


def clear_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("oc4k_smoke"))
    keep = {k: os.environ.get(k) for k in ENV}
    try:
        return b1.run_smoke("oc4k", root=root, keep=True)
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def small_archive(tmp, **kw):
    src = os.path.join(tmp, "src")
    d_lo, d_hi = (b10.parse_date(x) for x in oc.OC4kAdapter.smoke_window)
    truth = oc.make_smoke_sources(src, d_lo, d_hi, **kw)
    return src, truth


def ctx_for(tmp, src, lo=None, hi=None, **over):
    w = oc.OC4kAdapter.smoke_window
    d = dict(store="oc4k", work=os.path.join(tmp, "work"), source_dir=src,
             start=lo or w[0], end=hi or w[1], stage="all", force=False,
             attempts=1, qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS,
             assemble="auto", parts_from_hub=False, push_parts=False,
             allow_missing_years=False, allow_unconfirmed_licence=False,
             probe_month="", smoke=False)
    d.update(over)
    ad = oc.OC4kAdapter()
    ctx = b10.Ctx(argparse.Namespace(**d), adapter=ad,
                  layout=b1.layout_for(ad))
    return b1.prepare_grid_ctx(ctx)


def run(ctx, stages):
    b10.run_stages(ctx, stages, stage_fn=b1.GRID_STAGE_FN, deps=b1.DEPS)


# ================================================================ facts ====
def test_registered_with_the_real_grid(monkeypatch):
    clear_env(monkeypatch)
    assert fam.REGISTRY["oc4k"] is oc.OC4kAdapter
    ad = oc.OC4kAdapter()
    assert b1.is_grid(ad) and ad.layout == "sharded"
    assert ad.family == "1gf" and ad.distribution == "public"
    assert ad.licence["redistribution"] == "attribution"
    assert ad.licence["redistribution_confirmed"] is True
    assert ad.channel_names == ["log_chl", "kd_490", "total_nobs"]
    assert ad.dtype == "float16" and ad.C == 3
    assert (ad.frames_per_bin, ad.frame_seconds) == (5, 86400)
    assert ad.first_year == 1997
    # 1/24 degree is 4.64 km at the equator; one day out of five
    assert abs(ad.log2_fp - (-2.585)) < 1e-3
    assert abs(ad.log2_dt - (-2.322)) < 1e-3
    sp = ad.specs()["oc4k"]
    assert (sp["H"], sp["W"], sp["C"]) == (4320, 8640, 3)
    # 8640 / 256 = 33.75 -> 34 across, 4320 / 256 = 16.875 -> 17 down
    assert (sp["n_tiles_y"], sp["n_tiles_x"]) == (17, 34)
    assert sp["pad_rows"] == 32 and sp["pad_cols"] == 64
    assert sp["dtype"] == "float16" and sp["missing"] == "NaN"
    g = sp["grid"]
    assert (g["x0"], g["y0"]) == (-180.0, 90.0)
    assert abs(g["dx"] - 1.0 / 24.0) < 1e-12 and g["dy"] == -g["dx"]
    assert g["extent"] == [-180.0, -90.0, 180.0, 90.0]
    assert "row 0 is the NORTHERNMOST" in g["row_order"]
    # the channel's UNIT says, in words, that it is a logarithm
    assert "LOGARITHM" in sp["channels"][0]["unit"]
    assert sp["channels"][0]["min"] == -4.0 and sp["channels"][0]["max"] == 2.5


def test_the_file_names_are_the_archives(monkeypatch):
    clear_env(monkeypatch)
    d = dt.date(2015, 1, 15)
    assert oc.object_name("chlor_a", d) == (
        "ESACCI-OC-L3S-CHLOR_A-MERGED-1D_DAILY_4km_GEO_PML_OCx-20150115-"
        "fv6.0.nc")
    assert oc.object_name("kd", d) == (
        "ESACCI-OC-L3S-K_490-MERGED-1D_DAILY_4km_GEO_PML_KD490_Lee-20150115-"
        "fv6.0.nc")
    # the year listing parses only its own product's files
    html = oc._listing_html("/x", [oc.object_name("chlor_a", d),
                                   oc.object_name("kd", d),
                                   "00README_catalogue_and_licence.txt"])
    files, counts = oc.parse_year_listing(html, "chlor_a", 2015)
    assert list(files) == [d]
    assert counts["files_other"] == 2
    with pytest.raises(oc.FormatError):
        oc.parse_year_listing(html, "chlor_a", 2016)


# ============================================================== the smoke ==
def test_smoke_reads_every_frame_back_exactly(smoke):
    res = smoke["check"]
    # 1997-09-04 .. 09-18 is fifteen days, one of them not in both products
    assert res["frames_equal"] == 14
    assert res["by_reason"]["absent_upstream"] == 1
    sm = json.load(open(os.path.join(smoke["ctx"].store, "store.json")))
    assert sm["tier"] == "G" and sm["layout"] == "sharded"
    assert sm["frames_missing_by_reason"]["absent_upstream"] == 1
    assert list(sm["groups"]) == ["oc4k"]
    assert sm["C"] == 3 and sm["dtype"] == "float16"
    assert sm["counts"]["out_of_bounds"] == {"kd_490": 1}
    assert sm["counts"]["chl_below_floor"] == 1
    assert "SMOKE GRID" in sm["notes"]
    assert sm["groups"]["oc4k"]["frames_present"] == 14


def test_the_absent_day_is_a_zero_length_frame(smoke):
    d = oc.SMOKE_ABSENT
    t = b10.seconds_since_epoch(d)
    b = t // sh.BIN_SECONDS
    f = (t - b * sh.BIN_SECONDS) // 86400
    grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, "oc4k"))
    assert grp.entry(b, f, 0, 0) == (sh.FRAME_ABSENT, 0)
    assert grp.read_frame(b, f) is None


def test_log_chl_is_a_logarithm_and_the_fill_is_nan(smoke):
    truth = smoke["truth"]
    grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, "oc4k"))
    n = 0
    for (g, b, f), (want, why) in sorted(truth.items()):
        if want is None:
            continue
        got = grp.read_frame(b, f, raw=True)
        assert got is not None and got.shape == want.shape
        assert np.array_equal(got, want, equal_nan=True)
        lc = got[:, :, 0]
        good = lc[np.isfinite(lc)]
        # a LOGARITHM: negative where chlorophyll is under 1 mg m-3, and
        # inside the declared [-4, 2.5]
        assert good.size and good.min() < 0.0
        assert good.min() >= -4.0 and good.max() <= 2.5
        assert np.isnan(lc).any()            # the synthetic land
        # the three channels are missing in the same places, bar the two
        # out-of-bounds / below-floor pixels
        m0 = np.isnan(got[:, :, 0])
        m1 = np.isnan(got[:, :, 1])
        assert int((m0 != m1).sum()) <= 1
        n += 1
    assert n == 14


def test_the_probe_measured_the_same_frames_and_some_bytes(smoke):
    p = smoke["probe"]
    # September 1997: 04..18 published, 09-10 not in both products
    assert p["frames_fetched"] == 14
    assert p["out_of_bounds_stored"] == 0
    assert p["out_of_bounds"] == {"kd_490": 1}
    assert p["bytes_fetched"] > 0
    gp = p["groups"]["oc4k"]
    assert gp["C"] == 3
    assert gp["frames_missing"]["absent_upstream"] == 1
    assert set(gp["valid_fraction"]) == {"log_chl", "kd_490", "total_nobs"}
    assert 0.0 < gp["valid_fraction"]["log_chl"] < 1.0
    assert gp["record_frames"] == 14
    assert gp["estimate_store_bytes"] > 0
    assert gp["bytes_per_valid_pixel"] > 0


def test_index_measures_the_record_and_names_the_ledgers_claim(tmp_path):
    src, _ = small_archive(str(tmp_path))
    ctx = ctx_for(str(tmp_path), src)
    run(ctx, ["index"])
    plan = json.load(open(os.path.join(ctx.root, "plan.json")))
    assert plan["record"] == ["1997-09-04", "1997-09-18"]
    assert plan["days_in_both_products"] == 14
    assert plan["days_absent_in_record"] == ["1997-09-10"]
    assert "2026-06-30" in plan["ledger_says"]
    assert sorted(plan["products"]) == ["chlor_a", "kd"]
    assert plan["products"]["kd"]["files"] == 15
    assert plan["products"]["chlor_a"]["files"] == 14
    ff = plan["first_file"]
    assert ff["day"] == "1997-09-04"
    assert 0.0 < ff["valid_fraction"]["log_chl"] < 1.0
    assert ff["range"]["log_chl"][0] < 0.0


def test_index_refuses_an_empty_year_listing(tmp_path):
    src, _ = small_archive(str(tmp_path))
    p = os.path.join(src, "oc4k", "chlor_a", "daily", oc.VERSION,
                     "index.html")
    open(p, "w").write("<html><body>nothing here</body></html>")
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index"])
    assert "an empty listing" in str(e.value)


def test_index_refuses_a_month_with_no_file_at_all(tmp_path):
    """A one-day hole is a counted absence; a whole EMPTY MONTH inside the
    record is a listing problem and a refusal."""
    lo, hi = dt.date(1997, 9, 4), dt.date(1997, 11, 10)
    src = os.path.join(str(tmp_path), "src")
    skip = tuple(dt.date(1997, 10, k) for k in range(1, 32))
    oc.make_smoke_sources(src, lo, hi, skip=skip)
    ctx = ctx_for(str(tmp_path), src, lo="1997-09-04", hi="1997-11-10")
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index"])
    assert "1997-10" in str(e.value)


def test_a_regridded_file_is_refused_not_stored(tmp_path):
    src, _ = small_archive(str(tmp_path))
    bad = os.path.join(src, "oc4k", "chlor_a", "daily", oc.VERSION, "1997",
                       oc.object_name("chlor_a", dt.date(1997, 9, 4)))
    chl, kd, nobs = oc.smoke_fields(dt.date(1997, 9, 4), oc.SMOKE_H,
                                    oc.SMOKE_W // 2)
    oc.write_nc(bad, oc.SMOKE_H, oc.SMOKE_W // 2,
                {"chlor_a": chl, "total_nobs": nobs})
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index"])
    assert "dimensions" in str(e.value)


def test_a_file_that_will_not_parse_is_an_absence(tmp_path):
    src, _ = small_archive(str(tmp_path))
    p = os.path.join(src, "oc4k", "kd", "daily", oc.VERSION, "1997",
                     oc.object_name("kd", dt.date(1997, 9, 6)))
    with open(p, "r+b") as fh:
        fh.truncate(128)
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index", "fetch"])
    assert "could not be read" in str(e.value)
    assert any("1997-09-06" in a["unit"] for a in ctx.absent)
    from build_family7 import marked
    assert not marked(ctx.root, "fetch")


def test_check_store_verifies_every_stored_tile(smoke):
    st = sh.check_store(smoke["ctx"].store)
    g = st["groups"]["oc4k"]
    assert g["tiles_checked"] == g["tiles_stored"] > 0
    assert g["frames_present"] == 14
