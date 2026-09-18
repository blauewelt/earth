#!/usr/bin/env python3
"""The `snow05` adapter's smoke (family 1.0.tf, MOD10C1 v61) — no network.

    python3 -m pytest -q tests/test_family1_snow05.py

The synthetic archive (`snow05.make_smoke_sources`) is the real layout: a CMR
`granules.json` the REAL parser reads, plus HDF-EOS2 granules with the real
SDS names and the uint8 fill 255, on a 600 x 300 grid
(`SNOW05_SMOKE_GRID`). It carries every kind of value the producer's key
defines — percentages, 111 polar night, 239 ocean, 252 the Antarctica mask,
255 fill — so the "the codes are kept, not collapsed" decision is tested
rather than asserted. The record is 2015-02-01 .. 2015-02-20 with 02-11 not
listed and 02-04 carrying one pixel at 254, which the producer's key does NOT
define and which the channel bound therefore catches.
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
from family1.adapters import _modis_cmg as mc                   # noqa: E402
from family1.adapters import snow05 as sn                       # noqa: E402

pytest.importorskip("pyhdf")

ENV = ("SNOW05_SMOKE_GRID", "SNOW05_GROUPS")


def clear_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("snow05_smoke"))
    keep = {k: os.environ.get(k) for k in ENV}
    try:
        return b1.run_smoke("snow05", root=root, keep=True)
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def small_archive(tmp, **kw):
    src = os.path.join(tmp, "src")
    d_lo, d_hi = (b10.parse_date(x) for x in sn.Snow05Adapter.smoke_window)
    truth = sn.make_smoke_sources(src, d_lo, d_hi, **kw)
    return src, truth


def ctx_for(tmp, src, **over):
    w = sn.Snow05Adapter.smoke_window
    d = dict(store="snow05", work=os.path.join(tmp, "work"), source_dir=src,
             start=w[0], end=w[1], stage="all", force=False,
             attempts=1, qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS,
             assemble="auto", parts_from_hub=False, push_parts=False,
             allow_missing_years=False, allow_unconfirmed_licence=False,
             probe_month="", smoke=False)
    d.update(over)
    ad = sn.Snow05Adapter()
    ctx = b10.Ctx(argparse.Namespace(**d), adapter=ad,
                  layout=b1.layout_for(ad))
    return b1.prepare_grid_ctx(ctx)


def run(ctx, stages):
    b10.run_stages(ctx, stages, stage_fn=b1.GRID_STAGE_FN, deps=b1.DEPS)


# ================================================================ facts ====
def test_registered_as_a_uint8_store(monkeypatch):
    clear_env(monkeypatch)
    assert fam.REGISTRY["snow05"] is sn.Snow05Adapter
    ad = sn.Snow05Adapter()
    assert b1.is_grid(ad) and ad.layout == "sharded"
    assert ad.family == "1tf" and ad.distribution == "public"
    assert ad.credentials == ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    assert ad.channel_names == ["snow_cover", "cloud"]
    assert ad.C == 2 and ad.dtype == "uint8"
    assert (ad.frames_per_bin, ad.frame_seconds) == (5, 86400)
    assert ad.first_year == 2000
    sp = ad.specs()["terra"]
    assert (sp["H"], sp["W"], sp["C"]) == (3600, 7200, 2)
    assert sp["dtype"] == "uint8" and sp["missing"] == sh.U8_MISSING
    assert sp["grid"]["y0"] == 90.0 and sp["grid"]["dy"] == -0.05
    # the producer's own key travels with the store
    assert sp["value_key"]["239"] == "ocean"
    assert "255" in sp["value_key"]


def test_the_bounds_keep_the_codes_and_exclude_fill(monkeypatch):
    """0..253 carries every value the producer's key defines; 254 is not in
    that key and 255 is the layout's own missing value."""
    clear_env(monkeypatch)
    ad = sn.Snow05Adapter()
    lo, hi = ad.bounds()
    assert list(lo) == [0.0, 0.0] and list(hi) == [253.0, 253.0]
    for code in (0, 100, 107, 111, 237, 239, 250, 252, 253):
        assert lo[0] <= code <= hi[0]
    assert 254 > hi[0] and sh.U8_MISSING > hi[0]


# ============================================================== the smoke ==
def test_smoke_reads_every_frame_back_exactly(smoke):
    res = smoke["check"]
    assert res["frames_equal"] == 19            # 20 listed days less 02-11
    assert res["by_reason"]["absent_upstream"] == 1
    sm = json.load(open(os.path.join(smoke["ctx"].store, "store.json")))
    assert sm["tier"] == "G" and sm["layout"] == "sharded"
    assert list(sm["groups"]) == ["terra"]
    assert sm["C"] == 2 and sm["dtype"] == "uint8"
    assert sm["counts"]["out_of_bounds"] == {"snow_cover": 1}


def test_every_class_the_producer_defines_survived_the_round_trip(smoke):
    """The store's own numbers, read back: percentages AND class codes."""
    grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, "terra"))
    arr = grp.shard_index
    b = int(arr["bin"][1])
    frame = None
    for f in range(5):
        frame = grp.read_frame(b, f, raw=True)
        if frame is not None:
            break
    assert frame is not None
    snow, cloud = frame[:, :, 0], frame[:, :, 1]
    assert (snow <= 100).any()                  # percentages
    assert (snow == 111).any()                  # polar night
    assert (snow == 239).any()                  # ocean
    assert (cloud == 252).any()                 # the Antarctica mask
    assert (snow == sh.U8_MISSING).any()        # fill, the missing value
    # nothing above 253 other than the missing value itself
    kept = snow[snow != sh.U8_MISSING]
    assert kept.max() <= 253


def test_the_counts_name_each_code(smoke):
    p = smoke["probe"]
    c = p["counts"]
    assert c["code_snow_cover"]["239"] > 0      # ocean
    assert c["code_snow_cover"]["111"] > 0      # night
    assert c["code_cloud"]["252"] > 0           # Antarctica
    assert c["percent_pixels_snow_cover"] > 0
    assert c["fill_pixels_snow_cover"] > 0
    gp = p["groups"]["terra"]
    assert gp["C"] == 2 and gp["dtype"] == "uint8"
    assert gp["frames_missing"]["absent_upstream"] == 1
    assert gp["record_frames"] == 19
    assert p["out_of_bounds"] == {"snow_cover": 1}
    assert p["out_of_bounds_stored"] == 0


def test_a_value_the_key_does_not_define_is_counted_not_stored(smoke):
    truth = smoke["truth"]
    hit = [(b, f) for (g, b, f), (a, why) in truth.items()
           if a is not None and sh.frame_day(b, f, 86400) == sn.SMOKE_OOB]
    assert len(hit) == 1
    b, f = hit[0]
    grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, "terra"))
    a = grp.read_frame(b, f, raw=True)
    assert a[sn.SMOKE_H // 2, sn.SMOKE_W // 2, 0] == sh.U8_MISSING


def test_a_listed_granule_that_is_not_there_is_an_absence(tmp_path):
    src, _ = small_archive(str(tmp_path), absent=(dt.date(2015, 2, 6),))
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index", "fetch"])
    assert "could not be read" in str(e.value)
    assert any("2015-02-06" in a["unit"] for a in ctx.absent)


def test_a_missing_fill_attribute_is_a_refusal(tmp_path):
    """The fill value is what tells 'no data' from 0 percent snow, so a file
    that does not declare it is refused rather than guessed at."""
    src, _ = small_archive(str(tmp_path))
    gdir = os.path.join(src, "snow05", "granules", "terra")
    name = sorted(n for n in os.listdir(gdir) if n.endswith(".hdf"))[-1]
    mc.set_sds_attrs(os.path.join(gdir, name),
                     {sn.SDS_SNOW: {"_FillValue": 7}})
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index"])
    assert "_FillValue" in str(e.value)


def test_check_store_verifies_every_stored_tile(smoke):
    st = sh.check_store(smoke["ctx"].store)
    g = st["groups"]["terra"]
    assert g["tiles_checked"] == g["tiles_stored"] > 0
    assert g["frames_present"] == 19


def test_the_store_is_far_smaller_than_the_float_formula(smoke):
    """Why uint8: classes compress far below the note's float arithmetic."""
    p = smoke["probe"]
    gp = p["groups"]["terra"]
    assert gp["note_formula_bytes"] > 10 * gp["estimate_store_bytes"]
    assert gp["compression_ratio"] > 3.0
    assert np.isfinite(gp["bytes_per_valid_pixel"])
