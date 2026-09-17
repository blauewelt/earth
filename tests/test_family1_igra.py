#!/usr/bin/env python3
"""The `igra` adapter's smoke (family 1.0.tf, E-082) — no network.

    python3 -m pytest -q tests/test_family1_igra.py

The synthetic archive (`igra.make_smoke_sources`) is IGRA v2.2's real layout:
per-station `<ID>-data.txt.zip` under data-por/ with an Apache listing, and
igra2-station-list.txt including the real list's blank-ID line. Expected
counts are exact and derived in the comments.
"""
import argparse
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
from family1.adapters import igra                               # noqa: E402

# 4 streamed stations (3 listed + the unlisted ICM00004018) x 2 soundings a
# day. The window is 5 days; a 00 UTC flight launches at 23:05 the day
# before, so each station loses its first-day 00 UTC flight to the window
# (4), and USM00072201 loses two more (a no-time and a pilot-balloon
# sounding): 40 - 4 - 1 = 35. January (3 days): 24 - 4 = 20.
TRUTH_ROWS = 35
PROBE_ROWS = 20


def ns(tmp, src, **over):
    ad = igra.IGRAAdapter()
    d = dict(store="igra", work=str(tmp / "w"), source_dir=src,
             start=ad.smoke_window[0], end=ad.smoke_window[1], stage="all",
             force=False, attempts=1, qc_keep=2,
             check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
             parts_from_hub=False, push_parts=False,
             allow_missing_years=False)
    d.update(over)
    return argparse.Namespace(**d)


def test_registered():
    ad = igra.IGRAAdapter()
    assert fam.REGISTRY["igra"] is igra.IGRAAdapter
    assert ad.C == 80 and ad.time_dtype == "int64" and not ad.per_year
    assert ad.channel_names[0] == "T_1000" and ad.channel_names[15] == "T_10"
    assert ad.channel_names[16] == "DPD_1000"
    assert ad.channel_names[-1] == "Z_10"
    assert [c[0] for c in ad.channels[32:34]] == ["U_1000", "U_925"]


def test_smoke_all_stages_and_the_probe(tmp_path):
    res = b1.run_smoke("igra", root=str(tmp_path / "s"), keep=True)
    assert len(res["truth"]) == TRUTH_ROWS
    p = res["probe"]
    assert p["rows"] == PROBE_ROWS and p["distinct_platforms"] == 4
    # Jan 1: its 12 UTC flights + Jan 2's 00 UTC flights (23:05 on Jan 1);
    # GMM's Jan 3 00 UTC flight has no release time and stays on Jan 3
    assert p["rows_per_day"][:3] == [8, 7, 5]
    assert p["schema_version"] == 3 and p["stored_bytes_per_row"] == 191
    c = p["counts"]
    assert c["stations_streamed"] == 4
    assert c["stations_skipped_by_years"] == 1
    assert c["stations_not_in_station_list"] == 1
    assert c["soundings_kept"] == PROBE_ROWS
    assert c["qa_removed"] == {"T": 1} and c["qc"] == {"0": 19, "1": 1}
    assert c["time_from_nominal_hour"] == 1
    assert p["platforms"]["probe_platforms_without_entry"] == 0

    m = json.load(open(os.path.join(res["work"], "igra", "igra",
                                    "store.json")))
    k = m["counts"]
    assert m["per_year"] == {"1905": 15, "1906": 20}
    assert m["schema_version"] == 3 and m["time_dtype"] == "int64"
    assert k["soundings_read"] == 4 * 7 * 2
    assert k["release_day_shifted"] == 4 * 7 - 2
    assert k["soundings_outside_window"] == 16 + 3
    assert k["soundings_no_time"] == 1
    assert k["soundings_no_mandatory_level"] == 1
    assert k["levels_duplicated"] == 1
    assert k["out_of_bounds"] == {"T_1000": 1}
    assert k["value_missing"] == {"DPD": 1}
    assert k["time_release_hour_only"] == 1
    assert k["qc"] == {"0": 34, "1": 1}
    assert m["platforms"]["in_store"] == 4
    assert m["platforms"]["source_entries"] == 5
    pj = json.load(open(os.path.join(res["work"], "igra", "igra",
                                     "platforms.json")))
    ship = [v for v in pj.values() if v["id"] == "ZZV00SHIP01"][0]
    assert ship["mobile"] and ship["lat"] is None
    unl = [v for v in pj.values() if v["id"] == "ICM00004018"][0]
    assert "absent" in unl["note"]
    plan = json.load(open(os.path.join(res["work"], "igra", "plan.json")))
    assert len(plan["station_list_bad_lines"]) == 1
    assert plan["window_stations"] == 4
    st = f10.Store(os.path.join(res["work"], "igra", "igra"))
    # the ship's store rows follow the ship
    assert len(np.unique(st["lat"])) > 3


def test_the_launch_time_rules():
    hdr = igra._header
    lvl = igra._level_line(1, 0, 50000, 5500, -150, 30, 270, 100)
    blk = "\n".join([
        hdr("USM00072201", 2026, 9, 15, 0, 2304, 1, 24.5531, -81.7886), lvl,
        hdr("USM00072201", 2026, 9, 15, 12, 1107, 1, 24.5531, -81.7886), lvl,
        hdr("USM00072201", 2026, 9, 16, 12, 1199, 1, 24.5531, -81.7886), lvl,
        hdr("USM00072201", 2026, 9, 17, 12, 9999, 1, 24.5531, -81.7886), lvl,
        hdr("USM00072201", 2026, 9, 18, 99, 9999, 1, 24.5531, -81.7886), lvl,
        hdr("USM00072201", 2026, 9, 18, 23, 30, 1, 24.5531, -81.7886), lvl,
    ]).encode() + b"\n"
    c = {}
    t, la, lo, v, qc, yr = igra.parse_block(blk, -2**62, 2**62, c)
    S = b10.seconds_since_epoch
    D = b10.dt.datetime
    assert t.tolist() == [S(D(2026, 9, 14, 23, 4)), S(D(2026, 9, 15, 11, 7)),
                          S(D(2026, 9, 16, 11)), S(D(2026, 9, 17, 12)),
                          S(D(2026, 9, 19, 0, 30))]
    assert c["release_day_shifted"] == 2 and c["soundings_no_time"] == 1
    assert c["time_from_nominal_hour"] == 1
    assert c["time_release_hour_only"] == 1
    k = igra.LEVEL_OF[50000]
    assert np.allclose(v[0, k], -15.0) and np.allclose(v[0, 16 + k], 3.0)
    assert np.allclose(v[0, 32 + k], 10.0) and abs(v[0, 48 + k]) < 1e-9
    assert v[0, 64 + k] == 5500
    assert np.isnan(v[0, :k]).all()
    assert np.allclose(la, 24.5531) and np.allclose(lo, -81.7886)


def test_format_errors_are_refused():
    h = igra._header("USM00072201", 2026, 9, 15, 0, 2304, 2, 24.5, -81.7)
    lvl = igra._level_line(1, 0, 50000, 5500, -150, 30, 270, 100)
    with pytest.raises(igra.FormatError, match="NUMLEV"):
        igra.parse_block(f"{h}\n{lvl}\n".encode(), 0, 1, {})
    with pytest.raises(igra.FormatError, match="data line"):
        igra.parse_block(f"{h}\n{lvl}\n{lvl[:-3]}\n".encode(), 0, 1, {})
    with pytest.raises(igra.FormatError, match="header line"):
        igra.parse_block(f"{h[:-1]}\n{lvl}\n{lvl}\n".encode(), 0, 1, {})
    with pytest.raises(igra.FormatError, match="empty listing"):
        igra.parse_listing("<html></html>")
    tab, bad = igra.parse_station_list(
        "ACM00078861  17.1170  -61.7830   10.0    COOLIDGE FIELD (UA)       "
        "     1947 1993  13896\n" + " " * 72 + "1946 2025  70410\n")
    assert list(tab) == ["ACM00078861"] and len(bad) == 1
    assert tab["ACM00078861"]["nobs"] == 13896


def test_a_station_stream_stops_after_the_window(tmp_path):
    """Tiny blocks: a station is abandoned at its first block past the
    window, and a block wholly before it is skipped without parsing."""
    src = str(tmp_path / "src")
    ad = igra.IGRAAdapter()
    lo, hi = (b10.parse_date(x) for x in ad.smoke_window)
    ad.smoke_sources(src, lo, hi)
    ad.BLOCK_BYTES = 800
    ctx = b10.Ctx(ns(tmp_path, src, start="1905-12-31", end="1906-01-01"),
                  adapter=ad, layout=b1.layout_for(ad))
    sid, by_year, c, err = ad._station(ctx, "GMM00010868",
                                       ctx.t_lo, ctx.t_hi)
    assert err is None
    assert c["stations_stopped_early"] == 1
    assert c["soundings_before_window_unparsed"] >= 1
    assert sum(len(r["bin"]) for r in by_year.values()) == 4
    # 12-28 and 12-29 skipped unparsed; 12-30 .. 01-02 parsed; 01-03 never
    assert c["soundings_before_window_unparsed"] == 4
    assert c["soundings_read"] == 8


def test_the_preflight_refuses_a_window_too_large_to_hold(tmp_path):
    src = str(tmp_path / "src")
    ad = igra.IGRAAdapter()
    lo, hi = (b10.parse_date(x) for x in ad.smoke_window)
    ad.smoke_sources(src, lo, hi)
    ctx = b10.Ctx(ns(tmp_path, src), adapter=ad, layout=b1.layout_for(ad))
    # 3 stations x 100 soundings x 2 of their 3 listed years
    assert ad.projected(ctx) == 200
    ad.MAX_WINDOW_SOUNDINGS = 10
    with pytest.raises(SystemExit, match="lanes"):
        ad.fetch_preflight(ctx)


def test_a_truncated_zip_is_an_absence(tmp_path):
    src = str(tmp_path / "src")
    ad = igra.IGRAAdapter()
    lo, hi = (b10.parse_date(x) for x in ad.smoke_window)
    ad.smoke_sources(src, lo, hi)
    p = os.path.join(src, "igra", "data-por", "USM00072201-data.txt.zip")
    raw = open(p, "rb").read()
    open(p, "wb").write(raw[:len(raw) // 2])
    ctx = b10.Ctx(ns(tmp_path, src), adapter=ad, layout=b1.layout_for(ad))
    out = list(ad.fetch_stream(ctx))
    assert out[-1][0] is None and out[-1][2]["stations_failed"] == 1
    assert len(ctx.absent) == 1
    assert "USM00072201" in ctx.absent[0]["why"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
