#!/usr/bin/env python3
"""The `sif` adapter's smoke (family 1.0.tf, E-082 wave 6) — no network.

    python3 -m pytest -q tests/test_family1_sif.py

The synthetic archive (`sif.make_smoke_sources`) is written in the real
on-disk layout: the S5P-PAL STAC item page the adapter reads, plus one
netCDF4 granule a day in the product's own group layout
(`/PRODUCT`, `/PRODUCT/SUPPORT_DATA/{GEOLOCATIONS,INPUT_DATA}`,
`/METADATA/ALGORITHM_SETTINGS`). Expected counts are exact and are worked out
in the comments beside them, because the point of a smoke is a second
statement of the rules.
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
import family10_store as f10                                    # noqa: E402
from family1 import adapters as fam                             # noqa: E402
from family1.adapters import sif                                # noqa: E402

# Three days x 24 soundings = 72. Two are dropped every day: the sounding
# whose SIF_743 carries the undeclared netCDF4 fill, and the one whose
# latitude does. 72 - 6 = 66.
DAYS = 3
PER_DAY = sif.SMOKE_N
KEPT_PER_DAY = PER_DAY - 2
TRUTH_ROWS = DAYS * KEPT_PER_DAY                              # 66
# The cloud fraction runs 0 .. 0.79 in 24 steps, with four soundings forced
# to 0.05 / 0.3 / 0.7 / 0.9 and one forced past the solar-zenith threshold.
# Against the file's own threshold T = 0.8, quartered: qc 0 is cf <= 0.2,
# qc 1 is <= 0.4, qc 2 is <= 0.8, qc 3 is past T or past an angle limit.
QC_PER_DAY = {"0": 2, "1": 5, "2": 13, "3": 2}


def test_registered_and_declared():
    assert fam.REGISTRY["sif"] is sif.SIFAdapter
    ad = sif.SIFAdapter()
    assert ad.C == 6 and ad.family == "1tf" and ad.distribution == "public"
    assert ad.time_dtype == "int32" and ad.first_year == 2014
    assert ad.credentials == () and ad.platform_meta is True
    assert ad.channel_names == ["sif_740", "sif_740_daily", "sif_740_error",
                                "cloud_fraction", "solar_zenith",
                                "viewing_zenith"]
    # 7 x 3.5 km -> geometric mean 4.9497 km over the family's 27.83 km cell
    assert abs(ad.log2_fp - np.log2(np.sqrt(7.0 * 3.5) / 27.83)) < 1e-12
    assert abs(ad.log2_fp - (-2.4912)) < 1e-4
    assert ad.log2_dt == -4.0
    assert "TROPOMI only" in ad.notes


def test_the_platform_table_names_all_three_instruments():
    ad = sif.SIFAdapter()
    table = ad.platforms(None)
    assert len(table) == 3
    by_id = {v["id"]: v for v in table.values()}
    assert sorted(by_id) == ["OCO-2", "OCO-3", "TROPOMI"]
    t = by_id["TROPOMI"]
    assert t["footprint_km"] == [7.0, 3.5] and t["wavelength_nm"] == 740.0
    assert abs(t["log2_fp"] - ad.log2_fp) < 1e-3 and t["enabled"] is True
    # OCO is wired and GATED: the default build does not fetch it
    assert by_id["OCO-2"]["enabled"] is False
    assert by_id["OCO-2"]["footprint_km"] == [2.25, 1.3]
    assert abs(by_id["OCO-2"]["log2_fp"] - (-4.0)) < 0.1
    assert int(b10.platform_hash("TROPOMI")) in table


def test_the_oco_gate_refuses_before_its_first_byte(monkeypatch, tmp_path):
    """ml/CLAUDE.md §0.3 — the precondition is checked where the inputs are
    all it has cost, not at hour three."""
    monkeypatch.delenv("SIF_OCO", raising=False)
    ctx = argparse.Namespace(a=argparse.Namespace(stage="all"), source_dir="")
    assert sif.SIFAdapter().fetch_preflight(ctx) is None      # TROPOMI alone
    monkeypatch.setenv("SIF_OCO", "1")
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    monkeypatch.setenv("NETRC", str(tmp_path / "absent"))
    with pytest.raises(SystemExit, match="GES DISC is unapproved"):
        sif.SIFAdapter().fetch_preflight(ctx)
    nr = tmp_path / "netrc"
    nr.write_text("machine urs.earthdata.nasa.gov login u password p\n")
    monkeypatch.setenv("NETRC", str(nr))
    ad = sif.SIFAdapter()
    assert ad.fetch_preflight(ctx) is None
    assert "SIF_OCO=1" in ad.notes
    assert ad.platforms(None)[int(b10.platform_hash("OCO-3"))]["enabled"]


def test_the_quality_thresholds_come_from_the_file(tmp_path):
    """The product ships no per-sounding flag, so `qc` is graded against the
    numbers the granule itself declares — and a granule that declares none of
    them is a refusal, never a guess."""
    import netCDF4
    p = str(tmp_path / "ok.nc")
    sif.write_granule(p, dt.date(2023, 7, 1))
    ds = netCDF4.Dataset(p)
    try:
        assert sif.thresholds_of(ds) == {"cloud": 0.8, "sza": 70.0,
                                         "vza": 60.0}
    finally:
        ds.close()
    bad = str(tmp_path / "bad.nc")
    sif.write_granule(bad, dt.date(2023, 7, 1),
                      thresholds={"SZA threshold": "70.0"})
    ds = netCDF4.Dataset(bad)
    try:
        with pytest.raises(sif.FormatError, match="refuses to invent one"):
            sif.thresholds_of(ds)
    finally:
        ds.close()


def test_the_stac_item_parser():
    def item(day, size=402064007, uid="u"):
        return {"id": f"S5P_PAL__L2B_SIF____{day}",
                "properties": {"start_datetime": f"{day}T00:24:08+00:00",
                               "end_datetime": f"{day}T23:59:00+00:00"},
                "assets": {"product": {"href": f"https://x/{uid}",
                                       "file:local_path": f"{day}.nc",
                                       "file:size": size}}}
    by_day, counts = sif.parse_items([item("2023-07-01"), item("2023-07-02")])
    assert sorted(by_day) == [dt.date(2023, 7, 1), dt.date(2023, 7, 2)]
    assert counts["items"] == 2
    assert by_day[dt.date(2023, 7, 1)]["bytes"] == 402064007
    # the product is ONE file a day: two items on one day with no
    # processing stamp to order them would mean the adapter silently read
    # one of them
    with pytest.raises(sif.FormatError, match="two L2B items"):
        sif.parse_items([item("2023-07-01"), item("2023-07-01", uid="v")])

    # …but a RE-RUN day (family1-build #825/#826, 2026-09-24) lists two
    # items whose ids end in their processing time: the later one wins,
    # the other is counted, and two processed at the same second refuse
    def pal(day, start, end, proc, uid):
        it = item(day, uid=uid)
        it["id"] = f"S5P_PAL__L2B_SIF____{start}_{end}_{proc}"
        return it
    a = pal("2022-05-27", "20220527T002116", "20220527T222045",
            "20230913T082425", "a")
    b = pal("2022-05-27", "20220527T020246", "20220527T222045",
            "20230912T094900", "b")
    by_day, counts = sif.parse_items([b, a])
    assert by_day[dt.date(2022, 5, 27)]["url"] == "https://x/a"
    assert counts["items_superseded"] == 1 and counts["superseded_ids"] == \
        [b["id"]]
    by_day, counts = sif.parse_items([a, b])           # order-independent
    assert by_day[dt.date(2022, 5, 27)]["url"] == "https://x/a"
    c = pal("2023-10-29", "20231029T010258", "20231030T004350",
            "20260924T073817", "c")
    d = pal("2023-10-29", "20231029T010258", "20231030T004350",
            "20260924T073817", "d")
    with pytest.raises(sif.FormatError, match="two L2B items"):
        sif.parse_items([c, d])
    assert sif.processing_stamp(c["id"]) == "20260924T073817"
    assert sif.processing_stamp("S5P_PAL__L2B_SIF____2023-07-01") is None
    noasset = item("2023-07-03")
    noasset["assets"] = {}
    with pytest.raises(sif.FormatError, match="no `product` asset"):
        sif.parse_items([noasset])


def test_the_footprint_comes_from_the_corners_and_wraps_the_dateline():
    """A footprint whose corners straddle 180 deg is one pixel, not a
    360-degree one — the difference between 5 km and 376 km, which is what
    the real granule showed before the wrap was added."""
    lat = np.array([0.0, 0.0])
    lon = np.array([0.0, 179.99])
    dlat = np.array([[-0.0225, -0.0225, 0.0225, 0.0225]] * 2)
    dlon = np.array([[-0.0225, 0.0225, 0.0225, -0.0225]] * 2)
    lat_b = lat[:, None] + dlat
    lon_b = (lon[:, None] + dlon + 180.0) % 360.0 - 180.0
    km = sif.footprint_km(lat, lon, lat_b, lon_b)
    assert abs(km[0] - km[1]) < 1e-3, (km[0], km[1])
    assert 4.0 < km[0] < 6.0
    assert abs(np.log2(km[0] / 27.83) - (-2.48)) < 0.05


def test_smoke_all_stages_and_the_probe(tmp_path):
    res = b1.run_smoke("sif", root=str(tmp_path / "s"), keep=True)
    assert len(res["truth"]) == TRUTH_ROWS
    p = res["probe"]
    assert p["rows"] == TRUTH_ROWS
    assert p["rows_per_day"][:4] == [KEPT_PER_DAY, KEPT_PER_DAY,
                                     KEPT_PER_DAY, 0]
    assert p["distinct_platforms"] == 1
    assert p["counts_scope"] == "month"
    assert p["schema_version"] == 2 and p["time_dtype"] == "int32"
    assert p["stored_bytes_per_row"] == 27 + 2 * 6          # 39
    assert p["bytes_fetched"] > 0
    c = p["counts"]
    assert c["days_listed"] == DAYS and c["days_wanted"] == DAYS
    assert c["soundings_in_file"] == DAYS * PER_DAY
    # the undeclared netCDF4 fill: one SIF value a day, and the latitude of
    # another sounding, which is not a channel and so is not counted here
    assert c["fill_valued"] == DAYS
    assert c["soundings_no_sif"] == DAYS * 2
    assert c["rows_kept"] == TRUTH_ROWS and c["rows_in_window"] == TRUTH_ROWS
    assert c["qc_grade"] == {k: v * DAYS for k, v in QC_PER_DAY.items()}
    assert p["qc"] == c["qc_grade"]
    assert c["beyond_producer_threshold"] == {"cloud": DAYS,
                                              "solar_zenith": DAYS}
    # one fluorescence a day is 40 mW/m2/sr/nm, past the channel's bounds:
    # NaN for that channel alone, counted, and never stored out of bounds
    assert c["out_of_bounds"] == {"sif_740": DAYS}
    assert p["out_of_bounds"]["sif_740"] == DAYS
    assert sum(p["out_of_bounds_stored"].values()) == 0
    assert p["nan_fraction"]["sif_740"] == round(DAYS / TRUTH_ROWS, 6)
    for nm in ("sif_740_daily", "cloud_fraction", "solar_zenith"):
        assert p["nan_fraction"][nm] == 0.0
    # the OCO platforms are wired and skipped, and the skip is COUNTED
    assert c["platforms_skipped_no_ges_disc"] == 2
    # the corner geometry the layout cannot store is MEASURED anyway
    first = c["per_day"][0]
    assert first["thresholds"] == {"cloud": 0.8, "sza": 70.0, "vza": 60.0}
    assert 4.0 < first["footprint_km"]["median"] < 6.0
    assert -3.0 < first["log2_fp_measured"] < -2.0
    assert set(first["cloud_fraction_cut"]) == {"le_0.1", "le_0.2", "le_0.3",
                                                "le_0.5", "le_0.8"}


def test_the_stages_and_the_store(tmp_path):
    ad = sif.SIFAdapter()
    src = str(tmp_path / "src")
    lo, hi = (b10.parse_date(x) for x in ad.smoke_window)
    truth = ad.smoke_sources(src, lo, hi)
    a = argparse.Namespace(
        store="sif", work=str(tmp_path / "w"), source_dir=src,
        start=str(lo), end=str(hi), stage="all", force=False, attempts=1,
        qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
        parts_from_hub=False, push_parts=False, allow_missing_years=False)
    ctx = b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))
    b10.run_stages(ctx, ["index", "fetch", "assemble"], stage_fn=b1.STAGE_FN,
                   deps=b1.DEPS)
    b10.check_smoke(ctx, truth)
    out = b1.stage_check(ctx)
    assert out["N"] == TRUTH_ROWS and out["schema_version"] == 2
    plan = json.load(open(os.path.join(ctx.root, "plan.json")))
    assert plan["collection"] == "L2B_SIF___"
    assert plan["days_listed"] == DAYS
    assert plan["oco_enabled"] is False
    assert plan["platforms_enabled"] == ["TROPOMI"]
    assert plan["first_file"]["n_elem"] == PER_DAY
    assert plan["first_file"]["has_corners"] == 4
    assert "Caltech" in plan["alternative_source_not_read"] or \
        "caltech" in plan["alternative_source_not_read"]
    m = json.load(open(os.path.join(ctx.store, "store.json")))
    assert m["per_year"] == {"2023": TRUTH_ROWS}
    assert m["family"] == "family1_tf" and m["distribution"] == "public"
    assert m["platforms"]["in_store"] == 1
    assert m["platforms"]["in_store_without_entry"] == 0
    st = f10.Store(ctx.store)
    assert st.N == TRUTH_ROWS and st.C == 6
    assert sorted(np.unique(np.asarray(st["qc"])).tolist()) == [0, 1, 2, 3]
    # every stored footprint is the instrument's nominal one — the layout
    # carries a single pair per store and check_store asserts it
    fp = np.asarray(st["fp"], np.float32)
    assert np.allclose(fp[:, 0], np.float16(ad.log2_fp))
    assert np.allclose(fp[:, 1], np.float16(-4.0))
    pl = json.load(open(os.path.join(ctx.store, "platforms.json")))
    assert list(pl) == [str(int(b10.platform_hash("TROPOMI")))]


def test_a_truncated_granule_is_an_absence(tmp_path):
    ad = sif.SIFAdapter()
    src = str(tmp_path / "src")
    lo, hi = (b10.parse_date(x) for x in ad.smoke_window)
    ad.smoke_sources(src, lo, hi)
    base = os.path.join(src, "sif")
    nc = sorted(n for n in os.listdir(base) if n.endswith(".nc"))[1]
    raw = open(os.path.join(base, nc), "rb").read()
    open(os.path.join(base, nc), "wb").write(raw[:len(raw) // 3])
    a = argparse.Namespace(
        store="sif", work=str(tmp_path / "w"), source_dir=src,
        start=str(lo), end=str(hi), stage="fetch", force=False, attempts=1,
        qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
        parts_from_hub=False, push_parts=False, allow_missing_years=False)
    ctx = b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))
    list(ad.fetch_year(ctx, 2023))
    assert len(ctx.absent) == 1, ctx.absent
    assert "netCDF" in ctx.absent[0]["why"]
    assert ctx.absent[0]["unit"] == "2023"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_a_long_stac_window_is_walked_in_pieces_and_seams_kept_once(monkeypatch):
    """family1-build #875 (2026-09-24): the whole-record assembly asked the
    STAC for 2018-05 → 2026-09 in one page — 3,068 matched, 500 returned,
    refused. The window is walked in `STAC_WINDOW_DAYS` pieces; an item that
    the day-widened seam lists twice is kept once; a short piece still
    refuses."""
    asked = []

    def fake(url, attempts=4, **k):
        asked.append(url)
        q = dict(p.split("=", 1) for p in url.split("?", 1)[1].split("&"))
        lo, hi = [s[:10] for s in
                  __import__("urllib.parse").parse.unquote(q["datetime"]).split("/")]
        d0, d1 = dt.date.fromisoformat(lo), dt.date.fromisoformat(hi)
        feats = []
        d = d0
        while d <= d1:
            feats.append({"id": f"S5P_PAL__L2B_SIF____{d:%Y%m%d}T000000_"
                                f"{d:%Y%m%d}T235959_20240101T000000",
                          "properties": {"start_datetime": f"{d}T00:00:00Z"},
                          "assets": {"product": {"href": f"https://x/{d}",
                                                 "file:size": 1}}})
            d += dt.timedelta(days=1)
        js = {"features": feats, "context": {"matched": len(feats)}}
        return json.dumps(js).encode(), "ok"
    monkeypatch.setattr(sif.cm, "get_bytes", fake)
    monkeypatch.setattr(sif, "STAC_WINDOW_DAYS", 10)
    feats = sif.stac_items(dt.date(2024, 1, 1), dt.date(2024, 1, 25))
    # 25 days plus the one-day widening at each end of the whole window,
    # each item once although every seam day was listed by two pieces
    ids = [f["id"] for f in feats]
    assert len(ids) == len(set(ids)) == 27 and len(asked) == 3
    by_day, _ = sif.parse_items(feats)
    assert min(by_day) == dt.date(2023, 12, 31) and max(by_day) == \
        dt.date(2024, 1, 26)

    def short(url, attempts=4, **k):
        js = {"features": [], "context": {"matched": 7}}
        return json.dumps(js).encode(), "ok"
    monkeypatch.setattr(sif.cm, "get_bytes", short)
    with pytest.raises(sif.FormatError, match="matched 7"):
        sif.stac_items(dt.date(2024, 1, 1), dt.date(2024, 1, 2))
