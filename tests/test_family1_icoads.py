#!/usr/bin/env python3
"""The `icoads` adapter's smoke (family 1.gf, E-082) — no network.

    python3 -m pytest -q tests/test_family1_icoads.py

The synthetic archive (`icoads.make_smoke_sources`) is written in the real
on-disk layout: the two NCEI Apache listings (final-untrim/ with exact byte
sizes, nrt/monthly/ with three versions of the same month plus the `total`
and netCDF decoys) and IMMA1 records built column by column from Tables C0/C1
of R3.0-imma1_short.pdf, each followed by a second attachment the parser must
ignore. Expected counts are exact.
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
import family10_store as f10                                    # noqa: E402
from family1 import adapters as fam                             # noqa: E402
from family1.adapters import icoads                             # noqa: E402

# window 2014-12-30 .. 2015-01-02: four days x three hours x three platforms
TRUTH_ROWS = 4 * 3 * 3
PROBE_ROWS = 2 * 3 * 3                     # 2015-01-01 and -02
DROPPERS = 6                               # per written day


def test_registered():
    assert fam.REGISTRY["icoads"] is icoads.ICOADSAdapter
    ad = icoads.ICOADSAdapter()
    assert ad.C == 8 and ad.family == "1gf" and ad.time_dtype == "int64"
    assert ad.first_year == 1662 and ad.per_year
    assert ad.channel_names == ["sst", "airt", "slp", "wind_u", "wind_v",
                                "dewpt", "wave_h", "cloud"]
    assert ad.log2_fp == -4.0
    assert abs(ad.log2_dt - np.log2(1 / 24 / 5)) < 1e-12


def test_smoke_all_stages_and_the_probe(tmp_path):
    res = b1.run_smoke("icoads", root=str(tmp_path / "s"), keep=True)
    truth = res["truth"]
    assert len(truth) == TRUTH_ROWS
    p = res["probe"]
    assert p["rows"] == PROBE_ROWS
    assert p["rows_per_day"][:3] == [9, 9, 0]
    # two named platforms + one deck/sid pooled platform
    assert p["distinct_platforms"] == 3
    c = p["counts"]
    # the month file holds Jan 1 and 2: 2 days x (9 + 6 droppers)
    assert c["lines"] == 2 * (9 + DROPPERS)
    assert c["files"] == 1 and c["file_nrt"] == 1
    assert c["reports_no_hour"] == 2
    assert c["reports_worse_duplicate"] == 2
    assert c["reports_landlocked"] == 2
    assert c["reports_position_erroneous"] == 2
    assert c["reports_no_icoads_attm"] == 2
    assert c["reports_no_value"] == 2
    assert "reports_outside_window" not in c
    assert c["reports_blank_id"] == 6                # 2 kept days x 3 hours
    assert c["ncdc_erroneous"] == {"sst": 1}
    assert c["out_of_bounds"] == {"wave_h": 1}
    assert p["out_of_bounds"]["wave_h"] == 1
    assert sum(p["out_of_bounds_stored"].values()) == 0
    assert c["wind_incomplete"] == 1
    assert c["cloud_obscured"] == 1
    # qc classes on Jan 1-2: suspect 1, unassessed 1, correctable 1
    assert p["qc"] == {"0": PROBE_ROWS - 3, "1": 1, "2": 1, "3": 1}
    assert p["bytes_fetched"] > 0 and p["schema_version"] == 3
    assert p["time_dtype"] == "int64"
    assert c["file"].startswith("icoads-nrt_r3.0.3_final_d201501")


def _ctx(tmp_path, ad, src, stage="all"):
    import argparse
    a = argparse.Namespace(
        store="icoads", work=str(tmp_path / "w"), source_dir=src,
        start=ad.smoke_window[0], end=ad.smoke_window[1], stage=stage,
        force=False, attempts=1, qc_keep=2,
        check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
        parts_from_hub=False, push_parts=False, allow_missing_years=False)
    return b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))


def test_the_stages_and_the_store(tmp_path):
    ad = icoads.ICOADSAdapter()
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
    assert plan["listing"]["months"] == 2
    assert plan["listing"]["final_untrim_months"] == 1
    assert plan["listing"]["nrt_final_months"] == 1
    assert plan["listing"]["nrt_files_superseded"] == 2
    assert plan["releases"] == ["3.0.3", "3.1.0"]
    assert plan["first_record"]["rows_parsed"] > 0
    m = json.load(open(os.path.join(ctx.store, "store.json")))
    assert m["per_year"] == {"2014": 18, "2015": 18}
    # Dec 29: nine reports plus the no-value dropper (the window is applied
    # before the no-value test)
    assert m["counts"]["reports_outside_window"] == 10
    st = f10.Store(ctx.store)
    assert st.N == TRUTH_ROWS
    q = np.asarray(st["qc"])
    assert sorted(np.unique(q).tolist()) == [0, 1, 2, 3]


def test_the_fixed_width_record_parses_to_the_spec():
    line = icoads.imma1_line(1880, 1, 1, 1250, -3525, 35050, ii=10,
                             ident="Jeannette", d=90, w=100, slp=10276,
                             at=-389, dpt=None, sst=12, n=9, wh=3,
                             dck=710, sid=166)
    assert len(line) == 173 + 14
    c = {}
    t, lat, lon, v, ids, dck, sid, qc, _ = icoads.parse_block(
        [line.encode()], c)
    want_t = b10.seconds_since_epoch(b10.dt.datetime(1880, 1, 1, 12, 30))
    assert t.tolist() == [want_t] and want_t < 0
    assert lat[0] == -35.25 and abs(lon[0] - (-9.5)) < 1e-9
    assert abs(v[0, 3] - (-10.0)) < 1e-9 and abs(v[0, 4]) < 1e-9
    assert v[0, 2] == pytest.approx(1027.6) and v[0, 1] == pytest.approx(-38.9)
    assert np.isnan(v[0, 5]) and v[0, 0] == pytest.approx(1.2)
    assert v[0, 6] == 1.5 and np.isnan(v[0, 7])
    assert c["cloud_obscured"] == 1
    assert ids[0] == b"Jeannette" and dck[0] == 710 and sid[0] == 166
    # the blank-ID fallback pools by deck and source
    blank = icoads.imma1_line(1880, 1, 2, 0, 0, 0, ii=None, ident="",
                              sst=100, dck=710, sid=166)
    _, _, _, _, ids2, dck2, sid2, _, _ = icoads.parse_block(
        [blank.encode()], {})
    c2 = {}
    h = icoads.platforms_of(np.concatenate([ids, ids2]),
                            np.concatenate([dck, dck2]),
                            np.concatenate([sid, sid2]), c2)
    assert h[0] == b10.platform_hash("Jeannette")
    assert h[1] == b10.platform_hash("icoads-deck-710-sid-166")
    assert c2["reports_blank_id"] == 1


def test_a_listing_that_is_empty_or_reshaped_is_refused():
    with pytest.raises(icoads.FormatError, match="empty listing"):
        icoads.parse_untrim("<html><table></table></html>")
    with pytest.raises(icoads.FormatError, match="size column"):
        icoads.parse_untrim('<a href="IMMA1_R3.1.0_2010-07.gz">x</a>')
    with pytest.raises(icoads.FormatError, match="empty listing"):
        icoads.parse_nrt('<tr>\n<td><a href="icoads-nrt_r3.0.2_total_d201501'
                         '_c20220228.dat.gz">x</a></td>\n<td align="right">d'
                         '</td>\n<td align="right">5</td>\n')


def test_a_truncated_month_is_an_absence(tmp_path):
    ad = icoads.ICOADSAdapter()
    src = str(tmp_path / "src")
    lo, hi = (b10.parse_date(x) for x in ad.smoke_window)
    ad.smoke_sources(src, lo, hi)
    p = os.path.join(src, "icoads", "final-untrim", "IMMA1_R3.1.0_2014-12.gz")
    raw = open(p, "rb").read()
    open(p, "wb").write(raw[:len(raw) // 2])
    ctx = _ctx(tmp_path, ad, src)
    list(ad.fetch_year(ctx, 2014))
    assert len(ctx.absent) == 1 and "truncated" in ctx.absent[0]["why"]
    assert ctx.absent[0]["unit"] == "2014-12"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
