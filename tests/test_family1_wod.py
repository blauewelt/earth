#!/usr/bin/env python3
"""The `wod` adapter's smoke (family 1.gf, E-082) — no network.

    python3 -m pytest -q tests/test_family1_wod.py

The synthetic archive (`wod.make_smoke_sources`) is written in the real
on-disk layout: the NCEI root listing, the `1800/` bundle (OSD casts of every
year before 1900 in one file) and a `1900/` directory with OSD, CTD and XBT
files plus a PFL file that must never be opened, each a NETCDF4 contiguous
ragged array with the real WOD variable names, fills and flags. The window
1899-12-30 .. 1900-01-02 crosses the bundle boundary. Expected counts are
exact.
"""
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
import build_family8_argo as f8                                 # noqa: E402
import family10_store as f10                                    # noqa: E402
from family1 import adapters as fam                             # noqa: E402
from family1.adapters import wod                                # noqa: E402

# 1899-12-30/31: two OSD casts at two hours = 8 (bundle); 1900-01-01/02:
# two OSD + one CTD + one XBT at two hours = 16
TRUTH_ROWS = 8 + 16
PROBE_ROWS = 16


def test_registered():
    assert fam.REGISTRY["wod"] is wod.WODAdapter
    ad = wod.WODAdapter()
    assert ad.C == 128 and ad.family == "1gf" and ad.time_dtype == "int64"
    assert ad.first_year == 1772 and ad.per_year
    assert ad.channel_names[:2] == ["T_10", "T_30"]
    assert ad.channel_names[-1] == "CHL_1900"
    # the levels are family 8's, imported
    assert [int(c[0].split("_")[1]) for c in ad.channels[:16]] == \
        [int(p) for p in f8.LEVELS_ARR]


def test_smoke_all_stages_and_the_probe(tmp_path):
    res = b1.run_smoke("wod", root=str(tmp_path / "s"), keep=True)
    assert len(res["truth"]) == TRUTH_ROWS
    p = res["probe"]
    assert p["rows"] == PROBE_ROWS
    assert p["rows_per_day"][:3] == [8, 8, 0]
    assert p["distinct_platforms"] == 7        # 3 cruises/names + 4 XBT casts
    c = p["counts"]
    assert c["files_listed"] == 4 and c["files_excluded_pfl"] == 1
    assert c["files"] == 3 and c["files_unknown_instrument"] == []
    assert c["casts"] == 9 + 5 + 4
    assert c["casts_kept"] == PROBE_ROWS
    assert c["casts_kept_by_instrument"] == {"OSD": 8, "CTD": 4, "XBT": 4}
    assert c["casts_other_year"] == 1
    assert c["casts_no_level"] == 1
    assert c["values_flag_removed"] == 1
    assert c["profiles_flag_removed"] == 1
    assert c["platform_from_name"] == 4
    assert c["platform_from_cast"] == 4
    # the dead oxygen sensor: every filled level 10..1500 dbar
    assert c["out_of_bounds"] == {f"O2_{q}": 1 for q in
                                  (10, 30, 50, 100, 150, 200, 300, 400, 500,
                                   700, 900, 1100, 1300, 1500)}
    assert sum(p["out_of_bounds_stored"].values()) == 0
    # high nibble = instrument (OSD 1, CTD 2, XBT 4); bit 0 value flag,
    # bit 1 profile flag
    assert p["qc"] == {"16": 7, "17": 1, "32": 3, "34": 1, "64": 4}
    assert p["nan_fraction"]["T_1700"] == 1.0
    assert p["nan_fraction"]["T_10"] == 0.0
    assert p["bytes_fetched"] > 0 and p["schema_version"] == 3


def _ctx(tmp_path, ad, src):
    import argparse
    a = argparse.Namespace(
        store="wod", work=str(tmp_path / "w"), source_dir=src,
        start=ad.smoke_window[0], end=ad.smoke_window[1], stage="all",
        force=False, attempts=1, qc_keep=2,
        check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
        parts_from_hub=False, push_parts=False, allow_missing_years=False)
    return b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))


def test_the_stages_and_the_store(tmp_path):
    ad = wod.WODAdapter()
    src = str(tmp_path / "src")
    lo, hi = (b10.parse_date(x) for x in ad.smoke_window)
    truth = ad.smoke_sources(src, lo, hi)
    ctx = _ctx(tmp_path, ad, src)
    b10.run_stages(ctx, ["index", "fetch", "assemble"], stage_fn=b1.STAGE_FN,
                   deps=b1.DEPS)
    b10.check_smoke(ctx, truth)
    out = b1.stage_check(ctx)
    assert out["N"] == TRUTH_ROWS and out["schema_version"] == 3
    plan = json.load(open(os.path.join(ctx.root, "plan.json")))
    assert plan["listing"] == {"directories": 2, "first": "1800",
                               "last": "1900"}
    assert plan["directories"]["1800"]["files"] == ["wod_osd_1800.nc"]
    assert plan["directories"]["1900"]["excluded_pfl"] == ["wod_pfl_1900.nc"]
    assert plan["first_record"]["variables"]
    m = json.load(open(os.path.join(ctx.store, "store.json")))
    assert m["per_year"] == {"1899": 8, "1900": 16}
    assert m["counts"]["casts_no_time"] == 1
    assert m["counts"]["casts_outside_window"] == 4       # 1899-12-29
    assert m["counts"]["bundle_years"] == 1
    st = f10.Store(ctx.store)
    q = sorted(set(np.asarray(st["qc"]).tolist()))
    assert 20 in q                    # OSD + a flagged depth, 1899-12-30
    t = np.asarray(st.time_s())
    assert (t < 0).all()


def test_saunders_and_the_year_split():
    # UNESCO (1983)'s check value is 10,000 dbar <-> 9,712.653 m at 30 N;
    # Saunders' closed form is good to ~0.1 % (measured 10,006.3 dbar)
    assert abs(wod.saunders_pressure(9712.653, 30.0) - 10000.0) < 10.0
    assert wod.saunders_pressure(0.0, 0.0) == 0.0
    ts = [b10.seconds_since_epoch(b10.dt.datetime(y, m, d))
          for y, m, d in ((1772, 1, 1), (1899, 12, 31), (1900, 1, 1),
                          (1981, 12, 31), (1982, 1, 1), (2024, 2, 29))]
    assert wod.year_of_seconds(np.array(ts)).tolist() == \
        [1772, 1899, 1900, 1981, 1982, 2024]
    assert wod.WODAdapter.dir_of(1772) == "1800"
    assert wod.WODAdapter.dir_of(1900) == "1900"


def test_listings_are_refused_when_empty_or_reshaped():
    with pytest.raises(wod.FormatError, match="empty listing"):
        wod.parse_root("<html>nothing</html>")
    with pytest.raises(wod.FormatError, match="empty listing"):
        wod.parse_year_listing("<html>nothing</html>", "2005")
    with pytest.raises(wod.FormatError, match="size column"):
        wod.parse_year_listing('<a href="wod_osd_2005.nc">x</a>', "2005")
    row = ('<tr><td valign="top"><img src="/icons/unknown.gif" alt="[   ]">'
           '</td><td><a href="wod_osd_2005.nc">wod_osd_2005.nc</a></td><td '
           'align="right">2025-09-29 10:22  </td><td align="right">146M</td>'
           '<td>&nbsp;</td></tr>')
    assert wod.parse_year_listing(row, "2005") == \
        [("wod_osd_2005.nc", "OSD", 146_000_000)]
    with pytest.raises(wod.FormatError, match="listed under"):
        wod.parse_year_listing(row, "2006")


def test_a_truncated_file_is_an_absence(tmp_path):
    ad = wod.WODAdapter()
    src = str(tmp_path / "src")
    lo, hi = (b10.parse_date(x) for x in ad.smoke_window)
    ad.smoke_sources(src, lo, hi)
    p = os.path.join(src, "wod", "1900", "wod_ctd_1900.nc")
    raw = open(p, "rb").read()
    open(p, "wb").write(raw[:len(raw) // 2])
    ctx = _ctx(tmp_path, ad, src)
    list(ad.fetch_year(ctx, 1900))
    assert len(ctx.absent) == 1 and "wod_ctd_1900.nc" in ctx.absent[0]["why"]
    assert ctx.absent[0]["unit"] == "1900"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
