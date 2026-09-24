#!/usr/bin/env python3
"""The `swot` adapter's smoke (family 1.gf, E-082 wave 4) — no network.

    python3 -m pytest -q tests/test_family1_swot.py

`swot` is built as TIER P (main session, 2026-09-23): one row per valid
KaRIn pixel, 2024 first, one hosted lane a calendar month. A build is opt-in
(`SWOT_ALLOW_BUILD=1`), reads EVERY granule of its window (the probe's cycle
and pass cap refuse in a build), and keeps exactly the rows whose own second
lies in the window, so a pass straddling two lanes is split, never doubled.
These tests pin the guards, the granule-name parser (with the product's
re-processing rule), the CMR pager, the reader (the product's own fill, valid
range, scale and units), the channel bounds, the lane split, and the probe's
`storage_options` arithmetic against a synthetic archive in the product's own
layout.
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
from family1.adapters import swot                               # noqa: E402

# Four passes x 12 lines x 8 pixels. Per pass, dropped: the two nadir-gap
# pixels of every line and the whole line graded `bad` -> 11 lines x 6 = 66
# valid pixels a pass, 264 in all. Pass 004 straddles the window's (and the
# month's) last midnight: its lines 0-4 are inside, 5-11 on the next day, so
# the store and the probe keep 3 x 66 + 4 x 6 = 222 rows and 7 x 6 = 42 fall
# outside the window.
VALID_PIXELS = 264
TRUTH_ROWS = 222
OUTSIDE = 42


def test_registered_and_declared():
    assert fam.REGISTRY["swot"] is swot.SWOTAdapter
    ad = swot.SWOTAdapter()
    assert ad.C == 3 and ad.family == "1gf" and ad.distribution == "public"
    assert ad.channel_names == ["ssha_karin", "sig0_karin",
                               "ssh_karin_uncert"]
    # THE BOUNDS AND UNITS (module docstring): sig0_karin is LINEAR, unit
    # "1", as the product's own `units` says; the old "dB" [-10, 40] refused
    # every pixel above 16 dB. 65504 is the largest finite float16.
    assert [c[1] for c in ad.channels] == ["m", "1", "m"]
    lo, hi = ad.bounds()
    assert list(lo) == [-3.0, -1000.0, 0.0]
    assert list(hi) == [3.0, 65504.0, 6.0]
    with np.errstate(over="ignore"):
        assert np.isfinite(np.float16(hi[1])) and \
            not np.isfinite(np.float16(65520.0))
    # every bound is exactly a float16, so a value inside it cannot round
    # past it on its way into the store
    for b in list(lo) + list(hi):
        assert float(np.float16(b)) == b
    assert ad.credentials == ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    # 2 km pixels: the ledger's (-3.80, -4)
    assert abs(ad.log2_fp - (-3.798)) < 1e-3 and ad.log2_dt == -4.0
    assert "TIER P" in ad.notes and "every granule" in ad.notes


def test_a_build_refuses_before_its_first_byte(monkeypatch, tmp_path):
    """ml/CLAUDE.md §0.3 — both guards are at dispatch, where the inputs are
    all they have cost: the storage decision, and the Earthdata netrc."""
    for k in ("SWOT_ALLOW_BUILD", "SWOT_CYCLE", "SWOT_MAX_PASSES",
              "SWOT_ALLOW_CAPPED_BUILD"):
        monkeypatch.delenv(k, raising=False)
    ad = swot.SWOTAdapter()
    a = argparse.Namespace(stage="all", source_dir="")
    ctx = argparse.Namespace(a=a, source_dir="")
    with pytest.raises(SystemExit, match="without SWOT_ALLOW_BUILD=1"):
        ad.fetch_preflight(ctx)
    # a probe is allowed past the storage guard; the netrc guard is separate
    # and only applies when a real archive is being read
    assert ad.fetch_preflight(
        argparse.Namespace(a=argparse.Namespace(stage="probe"),
                           source_dir=str(tmp_path))) is None
    # with the decision made and NO netrc, the SECOND guard fires: the
    # granules are Earthdata-protected and a fetch without one spends the
    # whole listing to discover a page of 401s (run #152)
    monkeypatch.setenv("SWOT_ALLOW_BUILD", "1")
    monkeypatch.setenv("NETRC", str(tmp_path / "absent"))
    with pytest.raises(SystemExit, match="no netrc naming"):
        swot.SWOTAdapter().fetch_preflight(ctx)
    nr = tmp_path / "netrc"
    nr.write_text("machine urs.earthdata.nasa.gov login u password p\n")
    monkeypatch.setenv("NETRC", str(nr))
    assert swot.SWOTAdapter().fetch_preflight(ctx) is None


def test_the_probe_knobs_refuse_in_a_build(monkeypatch, tmp_path):
    """The laser stores' lesson: a probe's cap reaching a build marks a lane
    done with a fraction of it. SWOT_CYCLE and SWOT_MAX_PASSES are the
    PROBE's; a build that names either refuses before its first byte."""
    nr = tmp_path / "netrc"
    nr.write_text("machine urs.earthdata.nasa.gov login u password p\n")
    monkeypatch.setenv("NETRC", str(nr))
    monkeypatch.setenv("SWOT_ALLOW_BUILD", "1")
    monkeypatch.delenv("SWOT_ALLOW_CAPPED_BUILD", raising=False)
    build = argparse.Namespace(a=argparse.Namespace(stage="index,fetch"),
                               source_dir="")
    probe = argparse.Namespace(a=argparse.Namespace(stage="probe"),
                               source_dir="")
    for k, v in (("SWOT_CYCLE", "010"), ("SWOT_MAX_PASSES", "12")):
        monkeypatch.delenv("SWOT_CYCLE", raising=False)
        monkeypatch.delenv("SWOT_MAX_PASSES", raising=False)
        monkeypatch.setenv(k, v)
        with pytest.raises(SystemExit, match=f"{k} set: those are the PROBE"):
            swot.SWOTAdapter().fetch_preflight(build)
        with pytest.raises(SystemExit, match="PROBE's knobs"):
            list(swot.SWOTAdapter().fetch_year(build, 2024))
        # ... the probe is what they are for
        assert swot.SWOTAdapter().fetch_preflight(probe) is None
        monkeypatch.setenv("SWOT_ALLOW_CAPPED_BUILD", "1")
        ad = swot.SWOTAdapter()
        assert ad.fetch_preflight(build) is None
        assert "CAPPED" in ad.notes
        monkeypatch.delenv("SWOT_ALLOW_CAPPED_BUILD")
    # unset, a build carries no cap at all
    monkeypatch.delenv("SWOT_CYCLE", raising=False)
    monkeypatch.delenv("SWOT_MAX_PASSES", raising=False)
    ad = swot.SWOTAdapter()
    assert ad.knobs_set == [] and ad.fetch_preflight(build) is None


def test_a_box_assembling_from_hub_parts_needs_no_source(monkeypatch,
                                                       tmp_path):
    """`stage_fetch` calls `fetch_preflight` BEFORE it branches to the Hub
    pull, and a box has neither Earthdata credentials (ml/CLAUDE.md §6) nor
    SWOT_ALLOW_BUILD — so the box assembly must pass here, or
    `stage=all --parts-from-hub` refuses on a store whose parts are parked."""
    for k in ("SWOT_ALLOW_BUILD", "SWOT_CYCLE", "SWOT_MAX_PASSES"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    monkeypatch.setenv("NETRC", str(tmp_path / "absent"))
    monkeypatch.setenv("HOME", str(tmp_path))
    box = argparse.Namespace(a=argparse.Namespace(stage="all",
                                                  parts_from_hub=True),
                             source_dir="")
    assert swot.SWOTAdapter().fetch_preflight(box) is None
    assert b1.needs_source(argparse.Namespace(source_dir="",
                                              parts_from_hub=True),
                           ["index", "fetch", "assemble"]) is False


def test_the_granule_name_parser():
    def e(cycle, pss, extra=""):
        n = (f"SWOT_L2_LR_SSH_Expert_{cycle}_{pss}_20240125T001931_"
             f"20240125T011059_PGD0_02_swot{extra}")
        return {"title": n, "granule_size": "32.4",
                "time_start": "2024-01-25T00:19:17.011Z",
                "time_end": "2024-01-25T01:11:13.000Z",
                "links": [{"href": "https://x/podaac-swot-ops-cumulus-public/"
                                   f"{n}.nc"},
                          {"href": "https://x/podaac-swot-ops-cumulus-"
                                   f"protected/{n}.nc"}]}
    gran, counts = swot.parse_granules([e("010", "001"), e("010", "002"),
                                        {"title": "README", "links": []}])
    assert sorted(gran) == [("010", "001"), ("010", "002")]
    assert counts["granules_other_name"] == 1
    assert "protected" in gran[("010", "001")]["url"]
    assert gran[("010", "001")]["bytes"] == 32400000
    with pytest.raises(swot.FormatError, match="two granules"):
        swot.parse_granules([e("010", "001"), e("010", "001")])
    # THE PRODUCT'S RE-PROCESSING RULE (2025-06 lists 76 passes twice): the
    # same baseline with a higher counter supersedes, whichever comes first
    def v(cycle, pss, crid, nn):
        g = e(cycle, pss)
        t = g["title"].replace("PGD0_02", f"{crid}_{nn}")
        g["title"] = t
        g["links"] = [{"href": f"https://x/podaac-swot-ops-cumulus-protected/"
                               f"{t}.nc"}]
        return g
    for order in ((1, 2), (2, 1)):
        gran, counts = swot.parse_granules(
            [v("033", "100", "PID0", f"0{n}") for n in order])
        assert gran[("033", "100")]["counter"] == 2
        assert gran[("033", "100")]["name"].endswith("PID0_02_swot")
        assert counts["granules_superseded"] == 1
    # ... but two BASELINES for one pass is a decision, not a tie-break
    with pytest.raises(swot.FormatError, match="two granules"):
        swot.parse_granules([v("033", "100", "PID0", "01"),
                             v("033", "100", "PGD0", "02")])
    # the misfiled Basic / Unsmoothed granules of 2024-05 are counted
    basic = e("015", "092")
    basic["title"] = basic["title"].replace("_Expert_", "_Basic_")
    gran, counts = swot.parse_granules([basic, e("015", "093")])
    assert list(gran) == [("015", "093")]
    assert counts["granules_other_name"] == 1
    # a granule with no protected .nc link is a refusal, not a skip
    bad = e("010", "003")
    bad["links"] = [{"href": "https://x/podaac-swot-ops-cumulus-public/a.nc"}]
    with pytest.raises(swot.FormatError, match="no protected"):
        swot.parse_granules([bad])


def test_the_quality_vocabulary_comes_from_the_file():
    class V:
        flag_meanings = ("suspect_large_ssh_delta degraded_media_delay "
                         "bad_outside_of_range good_measurement")
    assert swot.qc_grade(V()) == ["good", "suspect", "degraded", "bad"]

    class Bad:
        flag_meanings = "some_other_condition"
    with pytest.raises(swot.FormatError, match="refusing to guess"):
        swot.qc_grade(Bad())


def test_the_pager_follows_the_cursor_and_dedupes_a_repeat(monkeypatch):
    """The listing pages on `CMR-Search-After` through `cm.cmr_entries`
    (checked against CMR-Hits — a cut listing is a refusal, see
    tests/test_family1_cmr_walk.py). A granule CMR lists twice is kept once,
    so a seam cannot raise "two granules for cycle … pass …"."""
    def g(i):
        n = (f"SWOT_L2_LR_SSH_Expert_010_{i:03d}_20240125T000000_"
             f"20240125T005000_PGD0_02_swot")
        return {"id": f"G{i}", "title": n,
                "time_start": f"2024-01-25T{i:02d}:00:00Z",
                "time_end": f"2024-01-25T{i:02d}:50:00Z"}
    # seven rows, one of them a repeat of G3 — CMR-Hits counts rows
    pages = [([g(1), g(2), g(3)], 7, "A"), ([g(3), g(4), g(5)], 7, "B"),
             ([g(6)], 7, None)]
    asked = []

    def fake(url, headers=None, attempts=4, sleep=3.0, count=None):
        asked.append((url, (headers or {}).get("CMR-Search-After")))
        return pages[len(asked) - 1]
    monkeypatch.setattr(swot.cm, "cmr_page", fake)
    got = swot.cmr_granules(temporal="2024-01-01T00:00:00Z,2024-01-31T23:59:59Z",
                            page=3)
    assert [x["id"] for x in got] == [f"G{i}" for i in range(1, 7)]
    assert [a[1] for a in asked] == [None, "A", "B"]
    assert "page_size=3" in asked[0][0] and "temporal=" in asked[0][0]


def test_a_build_lists_its_window_and_a_probe_its_cycle(monkeypatch):
    """A calendar month is the tail of one cycle, a whole one and the head of
    a third (2024-01: cycles 008, 009, 010, 865 granules): the build lists
    CMR by TIME, all cycles; only the probe lists by cycle."""
    calls = []

    def fake(cycle=None, temporal=None, **k):
        calls.append((cycle, temporal))
        n = ("SWOT_L2_LR_SSH_Expert_009_001_20240110T000000_"
             "20240110T005000_PGD0_02_swot")
        return [{"id": "G", "title": n, "granule_size": "32.0",
                 "time_start": "2024-01-10T00:00:00Z",
                 "time_end": "2024-01-10T00:50:00Z",
                 "links": [{"href": f"https://x/protected/{n}.nc"}]}]
    monkeypatch.setattr(swot, "cmr_granules", fake)
    monkeypatch.delenv("SWOT_CYCLE", raising=False)
    lo = b10.seconds_since_epoch(dt.date(2024, 1, 1))
    hi = b10.seconds_since_epoch(dt.date(2024, 1, 31)) + 86399
    for stage, want in (("index,fetch", (None, "2024-01-01T00:00:00Z,"
                                                "2024-01-31T23:59:59Z")),
                        ("probe", ("010", None))):
        calls.clear()
        ctx = argparse.Namespace(a=argparse.Namespace(stage=stage, attempts=1),
                                 source_dir="", t_lo=lo, t_hi=hi,
                                 count_bytes=lambda n: None)
        plan = swot.SWOTAdapter().index(ctx)
        assert calls == [want]
        assert plan["granules"] == 1
        assert plan["granules_per_processing"] == {"PGD0_02": 1}
        assert (plan["max_passes"] == 0) is (stage != "probe")


def _lane(src, work, start, end):
    a = argparse.Namespace(
        store="swot", work=str(work), source_dir=src, start=start, end=end,
        stage="index,fetch", force=False, attempts=1, qc_keep=2,
        check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
        parts_from_hub=False, push_parts=False, allow_missing_years=False)
    ad = swot.SWOTAdapter()
    ctx = b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))
    rows, counts = [], {}
    for _label, r, c in ad.fetch_year(ctx, ctx.d_lo.year):
        if r is not None:
            rows.append(r)
        if c:
            counts = c
    t = np.concatenate([r["time_s"] for r in rows]) if rows else \
        np.zeros(0, np.int32)
    return t, counts, ctx


def test_two_lanes_split_a_straddling_pass_exactly_once(tmp_path,
                                                        monkeypatch):
    """The tier-P form of the year-boundary rule: a row belongs to the lane
    its own second falls in. Pass 004 starts 5 s before 2024-02-01; the
    January lane keeps its first five lines, the February lane the other
    seven, and together they hold every valid pixel once."""
    for k in ("SWOT_CYCLE", "SWOT_MAX_PASSES"):
        monkeypatch.delenv(k, raising=False)
    src = str(tmp_path / "src")
    ad = swot.SWOTAdapter()
    lo, hi = (b10.parse_date(x) for x in ad.smoke_window)
    ad.smoke_sources(src, lo, hi)
    tj, cj, ctx_j = _lane(src, tmp_path / "wj", "2024-01-01", "2024-01-31")
    tf, cf, ctx_f = _lane(src, tmp_path / "wf", "2024-02-01", "2024-02-29")
    assert len(tj) == TRUTH_ROWS and len(tf) == OUTSIDE
    assert cj["rows_outside_window"] == OUTSIDE
    assert cf["rows_outside_window"] == VALID_PIXELS - OUTSIDE
    assert tj.max() <= ctx_j.t_hi < ctx_f.t_lo <= tf.min()
    assert len(tj) + len(tf) == VALID_PIXELS == cj["pixels_valid"]
    # a build reads every pass of the window, uncapped, all cycles
    assert cj["passes_read"] == len(swot.SMOKE_PASSES)
    assert cj["passes_cap"] == 0 and cj["cycle"] == "all"
    assert "storage_options" not in cj and len(cj["per_pass"]) == 2


def test_the_reader_applies_the_products_own_packing(tmp_path):
    """Fill -> NaN; a raw value over valid_max -> NaN and COUNTED; scale
    applied; the declared unit must be the file's."""
    import netCDF4
    src = tmp_path / "p.nc"
    day = dt.date(2024, 1, 31)
    swot.write_pass(str(src), "010", "003", day, day, 1)
    c = {}
    (t, lat, lon, vals, qc), meta = swot.read_pass(str(src), c)
    assert c["outside_product_valid_range"] == {"ssh_karin_uncert": 1}
    assert meta["product_valid_range"]["ssh_karin_uncert"] == \
        pytest.approx([0.0, 6.0])
    assert meta["units"] == {"ssha_karin": "m", "sig0_karin": "1",
                             "ssh_karin_uncert": "m"}
    u = vals[:, 2]
    assert np.nanmax(u) < 0.2 and np.isnan(u).sum() == 2
    # a sig0 declared in dB by the file is a refusal, not a silent relabel
    ds = netCDF4.Dataset(str(src), "a")
    ds.variables["sig0_karin"].units = "dB"
    ds.close()
    with pytest.raises(swot.FormatError, match="units 'dB'"):
        swot.read_pass(str(src), {})


def test_smoke_all_stages_and_the_probe(tmp_path, monkeypatch):
    for k in ("SWOT_CYCLE", "SWOT_MAX_PASSES"):
        monkeypatch.delenv(k, raising=False)
    res = b1.run_smoke("swot", root=str(tmp_path / "s"), keep=True)
    assert len(res["truth"]) == TRUTH_ROWS
    p = res["probe"]
    assert p["rows"] == TRUTH_ROWS
    assert p["distinct_platforms"] == 1
    assert p["counts_scope"] == "month"
    c = p["counts"]
    assert c["passes_read"] == len(swot.SMOKE_PASSES)
    assert c["pixels_in_pass"] == \
        len(swot.SMOKE_PASSES) * swot.SMOKE_LINES * swot.SMOKE_PIXELS
    assert c["pixels_valid"] == VALID_PIXELS
    assert c["rows_kept"] == TRUTH_ROWS
    assert c["rows_outside_window"] == OUTSIDE
    assert c["outside_product_valid_range"] == {"ssh_karin_uncert": 1}
    # THE DISTRIBUTION the bounds are judged against, before the bounds:
    # every kept row's ssha is in exactly one bin, the 9.0 m one included
    h = c["hist_ssha_karin"]
    assert sum(h.values()) == TRUTH_ROWS and h["[5,10)"] == 1
    assert sum(c["hist_sig0_karin"].values()) == TRUTH_ROWS
    assert sum(c["hist_ssh_karin_uncert"].values()) == TRUTH_ROWS - 2
    assert c["pass_geometry"] == \
        {f"{swot.SMOKE_LINES}x{swot.SMOKE_PIXELS}": len(swot.SMOKE_PASSES)}
    # the whole line graded `bad` is dropped, the suspect line is kept
    assert c["qual_grade"]["bad"] == len(swot.SMOKE_PASSES) * \
        swot.SMOKE_PIXELS
    assert c["qual_grade"]["suspect"] == len(swot.SMOKE_PASSES) * \
        swot.SMOKE_PIXELS
    assert p["qc"]["0"] > 0 and p["qc"]["1"] == \
        len(swot.SMOKE_PASSES) * (swot.SMOKE_PIXELS - 2)
    assert c["out_of_bounds"] == {"ssha_karin": 1}
    # THE DECISION MATERIAL: both options, computed from the measurement
    so = c["storage_options"]
    assert isinstance(so, list) and len(so) == 1
    so = so[0]
    assert so["measured"]["passes_read"] == len(swot.SMOKE_PASSES)
    assert so["tier_p"]["bytes_per_row"] == 33
    assert so["tier_g_per_pass_group"]["groups"] == 584
    assert abs(so["ratio_p_over_g"] - 33 / 7.2) < 1e-3
    assert so["tier_p"]["bytes_per_year"] > \
        so["tier_g_per_pass_group"]["bytes_per_year"]


def test_storage_options_is_arithmetic_not_a_guess():
    counts = {"passes_read": 4, "pixels_in_pass": 4_000_000,
              "pixels_valid": 1_000_000}
    so = swot.storage_options(counts, 3)
    assert so["measured"]["pixels_per_pass"] == 1_000_000.0
    assert so["measured"]["valid_pixels_per_pass"] == 250_000.0
    assert so["measured"]["valid_fraction"] == 0.25
    # the REPORTED passes_per_year is rounded to one decimal; the byte
    # arithmetic uses the exact value, so the test recomputes the exact one
    py = 584 * 365.25 / 20.86
    assert abs(so["extrapolation_basis"]["passes_per_year"] - py) < 0.1
    assert so["tier_p"]["bytes_per_year"] == round(250_000 * py * 33)
    assert so["tier_g_per_pass_group"]["bytes_per_year"] == \
        round(250_000 * py * 7.2)
    assert swot.storage_options({"passes_read": 0}, 3) is None


def test_the_index_plan(tmp_path):
    ad = swot.SWOTAdapter()
    src = str(tmp_path / "src")
    lo = b10.parse_date(ad.smoke_window[0])
    hi = b10.parse_date(ad.smoke_window[1])
    ad.smoke_sources(src, lo, hi)
    a = argparse.Namespace(
        store="swot", work=str(tmp_path / "w"), source_dir=src,
        start=str(lo), end=str(hi), stage="index", force=False, attempts=1,
        qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
        parts_from_hub=False, push_parts=False, allow_missing_years=False)
    ctx = b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))
    b10.run_stages(ctx, ["index"], stage_fn=b1.STAGE_FN, deps=b1.DEPS)
    plan = json.load(open(os.path.join(ctx.root, "plan.json")))
    assert plan["granules"] == len(swot.SMOKE_PASSES)
    assert plan["collection"] == swot.COLLECTION
    assert plan["build_allowed"] is False
    assert plan["listing"].startswith("build:")
    assert plan["cycle_requested"] is None and plan["max_passes"] == 0


def _ns(work, start, end, src, **over):
    d = dict(store="swot", work=str(work), source_dir=src, start=start,
             end=end, stage="index,fetch", force=False, attempts=1, qc_keep=2,
             check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="streaming",
             parts_from_hub=False, push_parts=False, lanes="",
             allow_missing_years=False, allow_unconfirmed_licence=False,
             probe_month="", smoke=False)
    d.update(over)
    return argparse.Namespace(**d)


def test_monthly_lanes_park_parts_and_the_box_assembles_them(tmp_path,
                                                            monkeypatch):
    """The 2024 form end to end, against a directory standing in for the
    Hub: two hosted monthly lanes (`index,fetch --push-parts`, lanes m01 and
    m02) each park only their own folder; the box (`all --parts-from-hub
    --lanes m01,m02 --assemble streaming`, no SWOT_ALLOW_BUILD, no Earthdata
    credential) pulls both and assembles every valid pixel exactly once."""
    import importlib.util
    import family10_parts_hub as ph
    spec = importlib.util.spec_from_file_location(
        "lanes_t", os.path.join(HERE, "test_family1_lanes.py"))
    lt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lt)
    tmp = str(tmp_path)
    fake = lt.FakePartsHub(os.path.join(tmp, "hub"))
    monkeypatch.setattr(ph, "_list_files", fake.list_files)
    monkeypatch.setattr(ph, "_upload", fake.upload)
    monkeypatch.setattr(ph, "_download", fake.download)
    monkeypatch.setattr(ph, "_hub", lambda: pytest.fail("family 10's hub"))
    for k in ("SWOT_ALLOW_BUILD", "SWOT_CYCLE", "SWOT_MAX_PASSES",
              "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    src = os.path.join(tmp, "src")
    ad = swot.SWOTAdapter()
    lo, hi = (b10.parse_date(x) for x in ad.smoke_window)
    ad.smoke_sources(src, lo, hi)

    for lane, (a, z) in {"m01": ("2024-01-01", "2024-01-31"),
                         "m02": ("2024-02-01", "2024-02-29")}.items():
        ctx = b1.make_ctx(_ns(os.path.join(tmp, f"lane_{lane}"), a, z, src,
                              push_parts=True), swot.SWOTAdapter)
        monkeypatch.setattr(ctx.layout, "hub", fake.hub)
        assert b1.apply_lane(ctx) == lane
        b10.run_stages(ctx, ["index", "fetch"], stage_fn=b1.STAGE_FN,
                       deps=b1.DEPS)
    pre = os.path.join(fake.root, "partials/family1_gf/swot/2024")
    assert sorted(os.listdir(pre)) == ["m01", "m02"]
    rows = {n: json.load(open(os.path.join(pre, n, "counts.json")))["rows"]
            for n in ("m01", "m02")}
    assert rows == {"m01": TRUTH_ROWS, "m02": OUTSIDE}

    box = b1.make_ctx(_ns(os.path.join(tmp, "box"), "2024-01-01",
                          "2024-12-31", src, stage="all", parts_from_hub=True,
                          lanes="m01,m02"), swot.SWOTAdapter)
    monkeypatch.setattr(box.layout, "hub", fake.hub)
    assert b1.apply_lane(box) == ""
    b10.run_stages(box, ["index", "fetch", "assemble", "check"],
                   stage_fn=b1.STAGE_FN, deps=b1.DEPS)
    meta = json.load(open(os.path.join(box.store, "store.json")))
    assert meta["N"] == VALID_PIXELS
    assert meta["lanes_by_year"]["2024"] == {
        "lanes": ["m01", "m02"], "declared": True,
        "expected": ["m01", "m02"]}
    t = np.asarray(b10.f10.Store(box.store).time_s())
    assert len(np.unique(t)) == len(t) // (swot.SMOKE_PIXELS - 2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
