#!/usr/bin/env python3
"""The `ndbc` adapter's smoke (family 1.0.tf, E-082) — no network.

    python3 -m pytest -q tests/test_family1_ndbc.py

The synthetic archive (`ndbc.make_smoke_sources`) is written in the real
on-disk layout: an Apache listing, station_table.txt, station_owners.txt and
`<sid>h<year>.txt.gz` files in the header eras measured on the real archive
(two-digit YY, YYYY without minutes, the modern `#YY ... mm` with its units
line, a 10-minute station with an upper-case id). Expected counts are exact.
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
from family1.adapters import ndbc                               # noqa: E402

# window 1998-12-30 .. 1999-01-02: four days. Per station, reports inside it:
# 41001 hourly 96 (one all-missing dropped -> 95), 45T01 half-hourly 192 (a
# repeated second dropped), 46042 hourly 96, 51xx1 ten-minute 576.
TRUTH_ROWS = 95 + 192 + 96 + 576
PROBE_ROWS = 47 + 96 + 48 + 288            # 1999-01-01 and -02 only


def test_registered():
    assert fam.REGISTRY["ndbc"] is ndbc.NDBCAdapter
    ad = ndbc.NDBCAdapter()
    assert ad.C == 10 and ad.family == "1tf" and ad.time_dtype == "int32"
    assert ad.channel_names == ["WDIR", "WSPD", "GST", "WVHT", "DPD", "APD",
                                "MWD", "PRES", "ATMP", "WTMP"]
    assert ad.log2_fp == -4.0
    assert abs(ad.log2_dt - np.log2(1 / 24 / 5)) < 1e-12


def test_smoke_all_stages_and_the_probe(tmp_path):
    res = b1.run_smoke("ndbc", root=str(tmp_path / "s"), keep=True)
    truth = res["truth"]
    assert len(truth) == TRUTH_ROWS
    p = res["probe"]
    assert p["rows"] == PROBE_ROWS
    assert p["distinct_platforms"] == 4
    # Jan 1 loses 41001's all-missing 07:00 report
    assert p["rows_per_day"][:2] == [23 + 48 + 24 + 144, 24 + 48 + 24 + 144]
    assert p["days_with_no_rows"] == 29
    c = p["counts"]
    assert c["stations_listed"] == 6
    assert c["stations_excluded_gtmba"] == 1
    assert c["stations_unknown"] == 1 and c["rows_station_unknown"] == 1
    assert c["files"] == 4
    assert c["lines"] == 48 + 97 + 48 + 288 + 2
    assert c["lines_bad_time"] == 1 and c["rows_other_year"] == 1
    assert c["rows_duplicate_time"] == 1 and c["rows_all_blanked"] == 1
    assert c["out_of_bounds"] == {"ATMP": 1}
    assert p["out_of_bounds"]["ATMP"] == 1
    assert sum(p["out_of_bounds_stored"].values()) == 0
    assert c["qc"] == {"0": PROBE_ROWS}
    # every sentinel was turned into NaN and counted; a real 99-degree WDIR
    # was not (it is in the truth as 99.0 and the store matched it)
    assert sum(c["value_missing"].values()) > 0
    assert p["platforms"]["probe_platforms_without_entry"] == 0
    assert p["bytes_fetched"] > 0 and p["schema_version"] == 2
    assert any(r["v"][0] == 99.0 for r in truth)


def test_the_five_stages_and_the_store(tmp_path):
    ad = ndbc.NDBCAdapter()
    src = str(tmp_path / "src")
    lo, hi = (b10.parse_date(x) for x in ad.smoke_window)
    truth = ad.smoke_sources(src, lo, hi)
    import argparse
    a = argparse.Namespace(
        store="ndbc", work=str(tmp_path / "w"), source_dir=src,
        start=ad.smoke_window[0], end=ad.smoke_window[1], stage="all",
        force=False, attempts=1, qc_keep=2,
        check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
        parts_from_hub=False, push_parts=False, allow_missing_years=False)
    ctx = b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))
    b10.run_stages(ctx, ["index", "fetch", "assemble"], stage_fn=b1.STAGE_FN,
                   deps=b1.DEPS)
    b10.check_smoke(ctx, truth)
    out = b1.stage_check(ctx)
    assert out["N"] == TRUTH_ROWS and out["schema_version"] == 2
    plan = json.load(open(os.path.join(ctx.root, "plan.json")))
    assert plan["listing"] == {"files": 12, "stations": 6, "first": 1998,
                               "last": 1999}
    assert plan["listed_stations_gtmba"] == ["13008"]
    assert plan["listed_stations_without_position"] == ["42360"]
    assert plan["gtmba_entries"] == ["13008"]
    assert plan["first_record"]["lines"] > 0
    m = json.load(open(os.path.join(ctx.store, "store.json")))
    assert m["platforms"]["in_store"] == 4
    assert m["platforms"]["in_store_without_entry"] == 0
    pj = json.load(open(os.path.join(ctx.store, "platforms.json")))
    assert "13008" not in {v["id"] for v in pj.values()}
    assert {v["id"] for v in pj.values()} == {"41001", "45T01", "46042",
                                              "51xx1"}
    st = f10.Store(ctx.store)
    assert st.N == TRUTH_ROWS
    assert m["counts"]["rows_outside_window"] == 3 + 6 + 3 + 18
    assert m["per_year"] == {"1998": 48 + 96 + 48 + 288,
                             "1999": 47 + 96 + 48 + 288}


def test_header_eras_parse_by_name():
    c = {}
    body = (b"YY MM DD hh WD   WSPD GST  WVHT  DPD   APD  MWD  BAR    ATMP  "
            b"WTMP  DEWP  VIS\n"
            b"76 06 01 18 226 07.9 99.0 99.00 99.00 99.00 999 1017.9  22.6  "
            b"22.2 999.0 99.0\n")
    t, v = ndbc.parse_stdmet(body, 1976, -10**12, 10**12, c)
    assert t.tolist() == [b10.seconds_since_epoch(
        b10.dt.datetime(1976, 6, 1, 18))]
    assert v[0, 0] == 226 and v[0, 1] == 7.9 and v[0, 7] == 1017.9
    assert np.isnan(v[0, [2, 3, 4, 5, 6]]).all()
    assert c["value_missing"] == {"GST": 1, "WVHT": 1, "DPD": 1, "APD": 1,
                                  "MWD": 1}
    body = (b"#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  "
            b"ATMP  WTMP  DEWP  VIS  TIDE\n#yr  mo dy hr mn degT m/s  m/s     "
            b"m   sec   sec degT   hPa  degC  degC  degC   mi    ft\n"
            b"2023 01 01 00 50  99 17.2 22.3 99.00 99.00 99.00 999 9999.0  "
            b"11.9 999.0 999.0 99.0 99.00 \n")
    c = {}
    t, v = ndbc.parse_stdmet(body, 2023, -10**12, 10**12, c)
    assert v[0, 0] == 99.0                       # a real direction
    assert np.isnan(v[0, 7]) and np.isnan(v[0, 9]) and v[0, 8] == 11.9
    assert c["lines_header"] == 2
    assert t[0] == b10.seconds_since_epoch(b10.dt.datetime(2023, 1, 1, 0, 50))
    with pytest.raises(ndbc.FormatError):
        ndbc.parse_stdmet(b"XX MM DD\n1 2 3\n", 2023, 0, 1, {})
    with pytest.raises(ndbc.FormatError):
        ndbc.parse_stdmet(b"YYYY MM DD hh WDIR\n2023 1 1 1 5\n", 2023, 0, 1,
                          {})


def test_the_gtmba_rule_reads_the_tables():
    owners = ndbc.parse_owners(
        "# x\nPR |Prediction and Research Moored Array in the Atlantic|FR\n"
        "RM |Research Moored Array for African-Asian-Australian Monsoon "
        "Analysis and Prediction|US\nN  |NDBC  |US\n")
    tab, bad = ndbc.parse_station_table(
        "# h\n"
        "13001|PR|Atlas Buoy|PM-595|NE Extension||12.000 N 23.000 W (x)|| |\n"
        "99001|RM|Buoy|||| 1.000 S 80.000 E (x)|| |\n"
        "51542|N|Drifting Buoy||TAO Buoy Adrift||0.018 N 179.903 W (x)|?| |\n"
        "41001|N|6-meter NOMAD buoy||EAST HATTERAS||34.724 N 72.317 W (x)|E| |\n"
        "bad|N|x||y||no position|| |\n", owners)
    assert bad == 1
    assert {k for k, e in tab.items() if e["gtmba"]} == {"13001", "99001",
                                                         "51542"}
    assert tab["41001"]["lat"] == 34.724 and tab["41001"]["lon"] == -72.317
    assert tab["99001"]["lat"] == -1.0 and tab["99001"]["lon"] == 80.0


def test_an_empty_or_reshaped_listing_is_refused(tmp_path):
    with pytest.raises(ndbc.FormatError, match="empty listing"):
        ndbc.parse_listing("<html>Index of /stdmet/</html>")
    with pytest.raises(ndbc.FormatError, match="size column"):
        ndbc.parse_listing('<a href="41001h1999.txt.gz">x</a>')
    got = ndbc.parse_listing(ndbc._listing_html(
        {"41001h1999.txt.gz": "8.8K", "45T01h1999.txt.gz": "1.2M"}))
    assert got == {1999: {"41001": ("41001", 8800),
                          "45t01": ("45T01", 1200000)}}


def test_a_truncated_file_is_an_absence_not_a_short_year(tmp_path):
    ad = ndbc.NDBCAdapter()
    src = str(tmp_path / "src")
    lo, hi = (b10.parse_date(x) for x in ad.smoke_window)
    ad.smoke_sources(src, lo, hi)
    p = os.path.join(src, "ndbc", "stdmet", "46042h1999.txt.gz")
    raw = open(p, "rb").read()
    open(p, "wb").write(raw[:len(raw) // 2])
    import argparse
    a = argparse.Namespace(
        store="ndbc", work=str(tmp_path / "w"), source_dir=src,
        start=ad.smoke_window[0], end=ad.smoke_window[1], force=False,
        attempts=1, qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS)
    ctx = b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))
    list(ad.fetch_year(ctx, 1999))
    assert len(ctx.absent) == 1 and "truncated" in ctx.absent[0]["why"]
    assert ctx.absent[0]["unit"] == "1999"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
