#!/usr/bin/env python3
"""The `swh` adapter's smoke (family 1.gf, E-082 wave 4) — no network.

    python3 -m pytest -q tests/test_family1_swh.py

The synthetic archive (`swh.make_smoke_sources`) is written in the real
layout — `<prefix>/<YYYY>/<MM>/ESACCI-SEASTATE-L3-SWH-MULTI_1D-<YYYYMMDD>-
fv01.nc`, one NETCDF4 trajectory file a day with the producer's own variable
names, `_FillValue` 1e+20, a `time` axis in "seconds since 1981-01-01" and a
`satellite` flag variable carrying the mission names — and every expected
count is exact.

Three source shapes the real archive has are exercised deliberately: a day
the archive does not hold at all (a real upstream gap, counted and NOT an
absence), a value outside its channel's bounds (NaN for that channel alone,
counted), and a listed file that will not open (`ctx.note_absent`, which
leaves the year unmarked).
"""
import argparse
import datetime as dt
import glob
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
import family10_store as f10                                    # noqa: E402
from family1 import adapters as fam                             # noqa: E402
from family1.adapters import swh                                # noqa: E402

# 1991-08-03 .. 1991-08-12 is ten days; 08-07 is the archive's gap, so nine
# files are written. Each holds SMOKE_PER_DAY samples, of which three are
# dropped: one with no time, one with no position, one with neither SWH.
DAYS_WRITTEN = 9
TRUTH_ROWS = DAYS_WRITTEN * (swh.SMOKE_PER_DAY - 3)
# the probe's month is 1991-08, which is the whole window
PROBE_ROWS = TRUTH_ROWS


def _window():
    return (b10.parse_date(swh.SWHAdapter.smoke_window[0]),
            b10.parse_date(swh.SWHAdapter.smoke_window[1]))


def test_registered_and_declared():
    assert fam.REGISTRY["swh"] is swh.SWHAdapter
    ad = swh.SWHAdapter()
    assert ad.C == 3 and ad.family == "1gf" and ad.distribution == "public"
    assert ad.channel_names == ["swh", "swh_denoised", "swh_uncertainty"]
    assert ad.time_dtype == "int32" and ad.first_year == 1991
    # THE STORE NEEDS NO ACCOUNT: the measurement this adapter rests on is
    # that Copernicus Marine's own object store answers anonymously.
    assert ad.credentials == ()
    assert ad.licence["redistribution"] == "attribution"
    assert abs(ad.log2_fp - np.log2(7.0 / 27.83)) < 1e-9
    assert ad.log2_dt == -4.0
    assert ad.per_year and ad.fetch_month_scope == "month"


def test_the_listing_parser_and_the_gap_finder():
    keys = [(f"{swh.PREFIX}2015/01/ESACCI-SEASTATE-L3-SWH-MULTI_1D-"
             f"2015010{i}-fv01.nc", 1000 + i) for i in (1, 2, 4)]
    keys.append((swh.PREFIX + "2015/01/README.txt", 7))
    days, counts = swh.parse_listing(keys)
    assert sorted(days) == [dt.date(2015, 1, 1), dt.date(2015, 1, 2),
                            dt.date(2015, 1, 4)]
    assert counts["files_other"] == 1
    assert swh.gaps(set(days)) == [dt.date(2015, 1, 3)]
    # two files for one day is a refusal, not a pick
    with pytest.raises(swh.FormatError, match="two files"):
        swh.parse_listing(keys + [(f"{swh.PREFIX}2015/01/ESACCI-SEASTATE-L3-"
                                   f"SWH-MULTI_1D-20150101-fv02.nc", 9)])


def test_the_mission_names_come_from_the_file():
    class V:
        flag_values = np.arange(3, dtype=np.uint8)
        flag_meanings = "cryosat-2 jason-1 saral"
    assert swh.satellite_names(V()) == {0: "cryosat-2", 1: "jason-1",
                                        2: "saral"}

    class Bad:
        flag_values = np.arange(3, dtype=np.uint8)
        flag_meanings = "cryosat-2 jason-1"
    with pytest.raises(swh.FormatError, match="flag_meanings"):
        swh.satellite_names(Bad())


def test_smoke_all_stages_and_the_probe(tmp_path):
    res = b1.run_smoke("swh", root=str(tmp_path / "s"), keep=True)
    assert len(res["truth"]) == TRUTH_ROWS
    p = res["probe"]
    assert p["rows"] == PROBE_ROWS
    assert p["distinct_platforms"] == len(swh.SMOKE_MISSIONS)
    assert p["counts_scope"] == "month"
    assert p["bytes_fetched"] > 0 and p["schema_version"] == 2
    c = p["counts"]
    assert c["days_listed"] == DAYS_WRITTEN
    assert c["days_read"] == DAYS_WRITTEN
    assert c["days_absent_upstream"] == 1           # 1991-08-07
    assert c["samples_in_file"] == DAYS_WRITTEN * swh.SMOKE_PER_DAY
    assert c["rows_no_time"] == DAYS_WRITTEN
    assert c["rows_no_position"] == DAYS_WRITTEN
    assert c["rows_no_swh"] == DAYS_WRITTEN
    assert c["rows_kept"] == TRUTH_ROWS
    # exactly one out-of-bounds swh, on SMOKE_OOB_DAY, NaN and counted
    assert c["out_of_bounds"] == {"swh": 1}
    assert sum(c["samples_per_mission"].values()) == TRUTH_ROWS
    # every row carries qc 1: the product publishes no per-sample flag
    assert p["qc"] == {"1": TRUTH_ROWS}


def test_the_store_the_plan_and_the_out_of_bounds_channel(tmp_path):
    ad = swh.SWHAdapter()
    src = str(tmp_path / "src")
    lo, hi = _window()
    truth = ad.smoke_sources(src, lo, hi)
    a = argparse.Namespace(
        store="swh", work=str(tmp_path / "w"), source_dir=src,
        start=str(lo), end=str(hi), stage="all", force=False, attempts=1,
        qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
        parts_from_hub=False, push_parts=False, allow_missing_years=False)
    ctx = b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))
    b10.run_stages(ctx, ["index", "fetch", "assemble"], stage_fn=b1.STAGE_FN,
                   deps=b1.DEPS)
    b10.check_smoke(ctx, truth)
    out = b1.stage_check(ctx)
    assert out["N"] == TRUTH_ROWS
    plan = json.load(open(os.path.join(ctx.root, "plan.json")))
    assert plan["files"] == DAYS_WRITTEN
    assert plan["missing_days"] == ["1991-08-07"]
    assert plan["record"] == ["1991-08-03", "1991-08-12"]
    assert plan["credentials"].startswith("none")
    ff = plan["first_file"]
    assert ff["n_time"] == swh.SMOKE_PER_DAY
    assert ff["time_units"] == "seconds since 1981-01-01"
    assert ff["missions"] == sorted(swh.SMOKE_MISSIONS)
    m = json.load(open(os.path.join(ctx.store, "store.json")))
    assert m["per_year"] == {"1991": TRUTH_ROWS}
    # the one out-of-bounds swh is NaN in the store and its row survived on
    # `swh_denoised` alone — never clipped to the bound
    st = f10.Store(ctx.store)
    v = np.asarray(st["values"], np.float32)
    only_den = np.isnan(v[:, 0]) & np.isfinite(v[:, 1])
    assert int(only_den.sum()) == 1
    assert float(np.nanmax(v[:, 0])) <= 25.0


def test_a_listed_file_that_will_not_open_is_an_absence(tmp_path):
    """A file the listing has and the reader cannot read leaves the year
    UNMARKED — it is never a silent skip (ADAPTER_CONTRACT rule 2)."""
    ad = swh.SWHAdapter()
    src = str(tmp_path / "src")
    lo, hi = _window()
    ad.smoke_sources(src, lo, hi)
    victim = sorted(glob.glob(os.path.join(src, "swh", "**", "*.nc"),
                              recursive=True))[2]
    with open(victim, "wb") as fh:                 # listed, and not a netCDF
        fh.write(b"not a netCDF at all")
    a = argparse.Namespace(
        store="swh", work=str(tmp_path / "w"), source_dir=src,
        start=str(lo), end=str(hi), stage="all", force=False, attempts=1,
        qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
        parts_from_hub=False, push_parts=False, allow_missing_years=False)
    ctx = b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))
    b10.run_stages(ctx, ["index"], stage_fn=b1.STAGE_FN, deps=b1.DEPS)
    with pytest.raises(SystemExit):
        b10.run_stages(ctx, ["fetch"], stage_fn=b1.STAGE_FN, deps=b1.DEPS)
    assert ctx.absent and "readable netCDF" in ctx.absent[0]["why"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
