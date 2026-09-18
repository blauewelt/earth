#!/usr/bin/env python3
"""The `chirps05` adapter's smoke (family 1.0.tf, exception E4) — no network.

    python3 -m pytest -q tests/test_family1_chirps05.py

The synthetic archive (`chirps05.make_smoke_sources`) is the real layout: a
`tifs/index.html` in the CHC server's own table format (`<td class="size">`
with human-readable "18.0 MiB") and real LZW float32 GeoTIFFs on a
600 x 200 grid (`CHIRPS05_SMOKE_GRID`), fill -9999 outside a land band. The
record is 1981-01 pentad 1 .. 1981-02 pentad 6, and 1981-01 pentad 3 carries
one 9,000 mm pixel. There is no absent pentad, because `index` refuses a hole
between the listing's ends and the real listing has none — a listed file that
will not parse is an ABSENCE that stops the year's marker, which the last two
tests exercise. Expected counts are exact.

The pentad calendar itself is tested against the producer's own numbers: six
pentads a month of 5, 5, 5, 5, 5 and 3-6 days, and the measured fact that
made F = 2 necessary — fourteen five-day bins of the 1981-2026 record hold
TWO pentad midpoints.
"""
import argparse
import calendar
import collections
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
from family1.adapters import chirps05 as ch                     # noqa: E402

pytest.importorskip("rasterio")

ENV = ("CHIRPS05_SMOKE_GRID",)


def clear_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("chirps05_smoke"))
    keep = {k: os.environ.get(k) for k in ENV}
    try:
        return b1.run_smoke("chirps05", root=root, keep=True)
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def small_archive(tmp, **kw):
    src = os.path.join(tmp, "src")
    d_lo, d_hi = (b10.parse_date(x) for x in ch.Chirps05Adapter.smoke_window)
    truth = ch.make_smoke_sources(src, d_lo, d_hi, **kw)
    return src, truth


def ctx_for(tmp, src, lo=None, hi=None, **over):
    w = ch.Chirps05Adapter.smoke_window
    d = dict(store="chirps05", work=os.path.join(tmp, "work"), source_dir=src,
             start=lo or w[0], end=hi or w[1], stage="all", force=False,
             attempts=1, qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS,
             assemble="auto", parts_from_hub=False, push_parts=False,
             allow_missing_years=False, allow_unconfirmed_licence=False,
             probe_month="", smoke=False)
    d.update(over)
    ad = ch.Chirps05Adapter()
    ctx = b10.Ctx(argparse.Namespace(**d), adapter=ad,
                  layout=b1.layout_for(ad))
    return b1.prepare_grid_ctx(ctx)


def run(ctx, stages):
    b10.run_stages(ctx, stages, stage_fn=b1.GRID_STAGE_FN, deps=b1.DEPS)


# ================================================================ facts ====
def test_registered_with_the_real_grid(monkeypatch):
    clear_env(monkeypatch)
    assert fam.REGISTRY["chirps05"] is ch.Chirps05Adapter
    ad = ch.Chirps05Adapter()
    assert b1.is_grid(ad) and ad.layout == "sharded"
    assert ad.family == "1tf" and ad.distribution == "public"
    assert ad.licence["name"] == "CC BY 4.0"
    assert ad.licence["redistribution"] == "attribution"
    assert ad.licence["redistribution_confirmed"] is True
    assert ad.channel_names == ["precip"] and ad.C == 1
    assert ad.dtype == "float16"
    assert (ad.frames_per_bin, ad.frame_seconds) == (2, 216000)
    assert ad.frames_per_bin * ad.frame_seconds == sh.BIN_SECONDS
    assert ad.first_year == 1981
    # 0.05 degrees is 5.566 km at the equator, and a pentad is 5 days
    assert abs(ad.log2_fp - (-2.3219)) < 1e-3
    assert ad.log2_dt == 0.0
    sp = ad.specs()["chirps05"]
    assert (sp["H"], sp["W"], sp["C"]) == (2400, 7200, 1)
    # 7200 / 256 = 28.125 -> 29 across, 2400 / 256 = 9.375 -> 10 down
    assert (sp["n_tiles_y"], sp["n_tiles_x"]) == (10, 29)
    assert sp["pad_rows"] == 160 and sp["pad_cols"] == 224
    assert sp["dtype"] == "float16" and sp["missing"] == "NaN"
    g = sp["grid"]
    assert g["crs"] == "EPSG:4326"
    assert (g["x0"], g["y0"]) == (-180.0, 60.0)
    assert g["dx"] == 0.05 and g["dy"] == -0.05
    assert g["extent"] == [-180.0, -60.0, 180.0, 60.0]
    assert "row 0 is the NORTHERNMOST" in g["row_order"]
    assert "NETCDF" in g["row_order"]          # the opposite order is named


def test_the_pentad_calendar_is_the_producers(monkeypatch):
    clear_env(monkeypatch)
    # six a month; 5, 5, 5, 5, 5 and 3-6 days
    for y in (1981, 2000, 2020, 2024):
        for m in range(1, 13):
            last = calendar.monthrange(y, m)[1]
            days = []
            for p in range(1, 7):
                d0, d1 = ch.pentad_days(y, m, p)
                assert d0.year == y and d0.month == m
                days += [d0 + dt.timedelta(days=k)
                         for k in range((d1 - d0).days + 1)]
                assert (d1 - d0).days + 1 == (5 if p < 6 else last - 25)
            # the six pentads tile the month exactly
            assert days == [dt.date(y, m, k) for k in range(1, last + 1)]
    # the netCDF time axis stamps each pentad at its FIRST day: 2019's read
    # 01-01, 01-06, 01-11, 01-16, 01-21, 01-26, 02-01 ...
    want = ["2019-01-01", "2019-01-06", "2019-01-11", "2019-01-16",
            "2019-01-21", "2019-01-26", "2019-02-01"]
    got = [str(ch.pentad_days(2019, m, p)[0])
           for (m, p) in [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
                          (2, 1)]]
    assert got == want
    for bad in (0, 7, -1):
        with pytest.raises(ch.FormatError):
            ch.pentad_days(2019, 1, bad)


def test_f1_collides_and_f2_does_not_over_the_whole_record():
    """The measurement that decided F = 2 (the adapter's docstring)."""
    c1 = collections.Counter(ch.pentad_slot(y, m, p, sh.BIN_SECONDS)
                             for (y, m, p) in ch.all_pentads(1981, 2026))
    collisions = {k: n for k, n in c1.items() if n > 1}
    assert len(collisions) == 14
    # every one of them is in February's short sixth pentad or its neighbour
    for (b, _f) in collisions:
        assert sh.bin_start_date(b).month in (2, 3)
    for F in (2, 5):
        fs = sh.BIN_SECONDS // F
        c = collections.Counter(ch.pentad_slot(y, m, p, fs)
                                for (y, m, p) in ch.all_pentads(1981, 2026))
        assert max(c.values()) == 1
    # and the slot map refuses a collision rather than losing a pentad
    with pytest.raises(ch.FormatError):
        ch.slot_map(sh.BIN_SECONDS, 1981, 2026)


def test_the_frame_table_says_what_each_slot_holds(monkeypatch):
    clear_env(monkeypatch)
    sp = ch.Chirps05Adapter().specs()["chirps05"]
    rows = sp["frame_table"]
    assert sp["frame_table_columns"] == ["bin", "frame", "year", "month",
                                         "pentad", "first_day", "last_day",
                                         "days"]
    assert len(rows) == (ch.TABLE_YEARS[1] - ch.TABLE_YEARS[0] + 1) * 72
    assert len({(r[0], r[1]) for r in rows}) == len(rows)
    for r in rows[:200] + rows[-200:]:
        b, f, y, m, p, d0, d1, n = r
        assert ch.pentad_slot(y, m, p, 216000) == (b, f)
        assert [str(x) for x in ch.pentad_days(y, m, p)] == [d0, d1]
        assert 3 <= n <= 6
        # the pentad's midpoint really is inside its half-bin
        s = b10.seconds_since_epoch(b10.parse_date(d0))
        e = b10.seconds_since_epoch(b10.parse_date(d1)) + 86400
        mid = (s + e) // 2
        t0 = sh.frame_start_seconds(b, f, 216000)
        assert t0 <= mid < t0 + 216000
    # 1981-01 pentad 1 lands in the first bin of 1981 — a NEGATIVE bin, the
    # epoch being 1982-01-01
    assert rows[0][:2] == [-73, 1]
    assert sh.bin_year(-73) == 1981


def test_the_listing_parser_reads_the_servers_own_table():
    html = ch.listing_html([("chirps-v3.0.1981.01.1.tif", "18.0 MiB"),
                            ("chirps-v3.0.1981.01.2.tif", "17.9 MiB"),
                            ("notes.txt", "1.2 KiB")])
    files, counts = ch.parse_listing(html)
    assert sorted(files) == [(1981, 1, 1), (1981, 1, 2)]
    assert files[(1981, 1, 1)][1] == int(18.0 * 1024 ** 2)
    assert counts["files_other"] == 1
    with pytest.raises(ch.FormatError):
        ch.parse_listing(html + html)          # the same file twice
    with pytest.raises(ch.FormatError):
        ch.parse_listing(ch.listing_html([("chirps-v3.0.1981.13.1.tif",
                                           "1 MiB")]))
    with pytest.raises(ch.FormatError):
        ch.parse_listing(ch.listing_html([("chirps-v3.0.1981.01.7.tif",
                                           "1 MiB")]))


# ============================================================== the smoke ==
def test_smoke_reads_every_frame_back_exactly(smoke):
    res = smoke["check"]
    # 1981-01 and 1981-02 hold twelve pentads, all of them published
    assert res["frames_equal"] == 12
    assert "absent_upstream" not in res["by_reason"]
    assert res["by_reason"]["no_pentad_in_slot"] > 0
    sm = json.load(open(os.path.join(smoke["ctx"].store, "store.json")))
    assert sm["tier"] == "G" and sm["layout"] == "sharded"
    assert "absent_upstream" not in sm["frames_missing_by_reason"]
    assert list(sm["groups"]) == ["chirps05"]
    assert sm["C"] == 1 and sm["dtype"] == "float16"
    assert sm["frames_per_bin"] == 2 and sm["frame_seconds"] == 216000
    assert sm["counts"]["out_of_bounds"] == {"precip": 1}
    # every pentad's own length was counted, 5-day and short ones alike
    assert set(sm["counts"]["pentad_days"]) >= {"5"}
    assert "SMOKE GRID" in sm["notes"]
    g = sm["groups"]["chirps05"]
    assert g["frames_present"] == 12
    assert g["bin_first"] == -73


def test_an_empty_half_bin_is_not_an_absence(smoke):
    """About half of all slots hold no pentad at all; those are counted
    `no_pentad_in_slot` and are not in `absent_upstream`."""
    sm = json.load(open(os.path.join(smoke["ctx"].store, "store.json")))
    by = sm["frames_missing_by_reason"]
    assert by["no_pentad_in_slot"] >= 10
    assert "absent_upstream" not in by
    grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, "chirps05"))
    slots = ch.slot_map(216000, 1981, 1982)
    empty = [(int(r["bin"]), f) for r in grp.shard_index for f in range(2)
             if (int(r["bin"]), f) not in slots]
    assert empty
    for (b, f) in empty[:4]:
        assert grp.read_frame(b, f) is None


def test_a_real_pentad_reads_back_with_its_fill_as_nan(smoke):
    truth = smoke["truth"]
    grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, "chirps05"))
    n = 0
    for (g, b, f), (want, why) in sorted(truth.items()):
        if want is None:
            continue
        got = grp.read_frame(b, f, raw=True)
        assert got is not None and got.shape == want.shape
        assert np.array_equal(got, want, equal_nan=True)
        v = got[np.isfinite(got)]
        assert v.size and v.min() >= 0.0 and v.max() <= 5000.0
        assert np.isnan(got).any()          # the ocean fill
        n += 1
    assert n == 12


def test_the_probe_measured_the_same_frames_and_some_bytes(smoke):
    p = smoke["probe"]
    assert p["frames_fetched"] == 6       # January 1981's six pentads
    assert p["out_of_bounds_stored"] == 0
    assert p["out_of_bounds"] == {"precip": 1}
    assert p["bytes_fetched"] > 0
    gp = p["groups"]["chirps05"]
    assert gp["C"] == 1 and gp["dtype"] == "float16"
    assert "absent_upstream" not in gp["frames_missing"]
    assert gp["frames_missing"]["no_pentad_in_slot"] > 0
    assert 0.0 < gp["valid_fraction"]["precip"] < 1.0
    assert gp["record_frames"] == 12
    assert gp["estimate_store_bytes"] > 0


def test_index_refuses_a_hole_inside_the_record(tmp_path):
    src, _ = small_archive(str(tmp_path), skip=((1981, 2, 3),))
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index"])
    assert "not in the listing" in str(e.value)


def test_index_refuses_an_empty_listing(tmp_path):
    src, _ = small_archive(str(tmp_path))
    p = os.path.join(src, "chirps05", "tifs", "index.html")
    open(p, "w").write(ch.listing_html([]))
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index"])
    assert "lists no pentad GeoTIFF" in str(e.value)


def test_a_regridded_geotiff_is_refused_not_stored(tmp_path):
    src, _ = small_archive(str(tmp_path))
    bad = os.path.join(src, "chirps05", "tifs", "chirps-v3.0.1981.01.1.tif")
    a = ch.smoke_field(1981, 1, 1, ch.SMOKE_W // 2, ch.SMOKE_H)
    ch.write_tif(bad, a, ch.SMOKE_W // 2, ch.SMOKE_H)
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index"])
    assert "expected" in str(e.value)


def test_a_listed_file_that_will_not_parse_is_an_absence(tmp_path):
    """A file the listing has and the adapter cannot READ is `note_absent`
    and stops the year's marker, never a silently empty frame."""
    src, _ = small_archive(str(tmp_path))
    p = os.path.join(src, "chirps05", "tifs", "chirps-v3.0.1981.01.2.tif")
    with open(p, "r+b") as fh:
        fh.truncate(64)
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index", "fetch"])
    assert "could not be read" in str(e.value)
    assert any("1981-01 pentad 2" in a["unit"] for a in ctx.absent)
    from build_family7 import marked
    assert not marked(ctx.root, "fetch")


def test_check_store_verifies_every_stored_tile(smoke):
    st = sh.check_store(smoke["ctx"].store)
    g = st["groups"]["chirps05"]
    assert g["tiles_checked"] == g["tiles_stored"] > 0
    assert g["frames_present"] == 12
