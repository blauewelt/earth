#!/usr/bin/env python3
"""The `cygnss_mss` adapter's smoke (family 1.3, E-092 phase C) — no network.

    python3 -m pytest -q tests/test_family13_cygnss_mss.py

`cygnss_mss` is CYGNSS's sea-surface mean-square slope, built DAILY on the
producer's 0.2-degree grid by combining each L3 V3.2 file's 24 hourly bins
with inverse-variance weights. Everything here runs on a synthetic archive
(`ml/family1/adapters/_cygnss_mss_smoke.py`) written with the real variable
names, dimensions and fill values: the daily combination's algebra, the grid
convention (rotation, south-first rows, a flipped latitude axis), every stage
of the build, the probe, and falsifier F1 — passing on a producer that bins
the way the dictionary says and FAILING, with numbers, on three that do not.
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

import build_family1_stores as b1                               # noqa: E402
from family1 import sharded as sh                               # noqa: E402
from family1 import adapters as fam                             # noqa: E402
from family1.adapters import cygnss_mss as cy                   # noqa: E402
from family1.adapters import _cygnss_mss_smoke as smk           # noqa: E402

H, W = smk.SMOKE_H, smk.SMOKE_W
D0 = dt.date(2019, 1, 1)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in ("CYGNSS_MSS_SMOKE_GRID", "CYGNSS_MSS_F1", "CYGNSS_MSS_F1_DAYS"):
        monkeypatch.delenv(k, raising=False)


def test_registered_and_declared():
    assert fam.REGISTRY["cygnss_mss"] is cy.CygnssMSSAdapter
    ad = cy.CygnssMSSAdapter()
    assert ad.family == "13" and ad.tier == "G" and ad.C == 4
    assert ad.channel_names == ["mss", "mss_uncertainty", "n_mss",
                                "wind_speed"]
    assert ad.dtype == "float16" and ad.frames_per_bin == 5
    assert ad.frame_seconds == 86400
    assert ad.credentials == ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    assert ad.licence["redistribution_confirmed"] is True
    assert "10.5067/CYGNS-L3X32" in ad.licence["attribution"]
    g = ad.grid
    assert (g["H"], g["W"]) == (400, 1800)
    assert g["x0"] == -180.0 and g["y0"] == -40.0
    assert abs(g["dx"] - 0.2) < 1e-12 and abs(g["dy"] - 0.2) < 1e-12
    assert abs(ad.log2_fp - (-0.322)) < 0.01
    assert ad.f1_mode == "probe"
    spec = ad.specs()["cygnss_mss"]
    assert "falsifier" in spec["daily_combination"]


def test_the_daily_combination_is_the_iv_mean_of_all_the_samples():
    """If each hour is the IV mean of its samples with u_h = 1/sqrt(sum w),
    the IV mean of the hours IS the IV mean of the day — exactly."""
    rng = np.random.default_rng(3)
    nh, cells = 24, 50
    hours = {k: np.full((nh, 1, cells), np.nan) for k in
             ("mss", "mss_u", "ws", "ws_u")}
    hours["mss_n"] = np.zeros((nh, 1, cells))
    hours["ws_n"] = np.zeros((nh, 1, cells))
    allw = np.zeros(cells)
    allwv = np.zeros(cells)
    alln = np.zeros(cells)
    for h in range(nh):
        for c in range(cells):
            k = rng.integers(0, 4)
            if not k:
                continue
            v = rng.uniform(0.01, 0.04, k)
            u = v * rng.uniform(0.01, 0.03, k)
            w = 1 / u ** 2
            hours["mss"][h, 0, c] = (w * v).sum() / w.sum()
            hours["mss_u"][h, 0, c] = 1 / np.sqrt(w.sum())
            hours["mss_n"][h, 0, c] = k
            hours["ws"][h, 0, c] = 7.0
            hours["ws_u"][h, 0, c] = 1.0
            hours["ws_n"][h, 0, c] = k
            allw[c] += w.sum()
            allwv[c] += (w * v).sum()
            alln[c] += k
    counts = {}
    d = cy.combine_hours(hours, counts)
    has = alln > 0
    np.testing.assert_allclose(d[0, has, 0], allwv[has] / allw[has],
                               rtol=1e-12)
    np.testing.assert_allclose(d[0, has, 1], 1 / np.sqrt(allw[has]),
                               rtol=1e-12)
    assert np.array_equal(d[0, has, 2], alln[has])
    assert np.isnan(d[0, ~has]).all()
    assert np.allclose(d[0, has, 3], 7.0)
    # an hour with a value and no usable uncertainty is excluded AND counted
    hours["mss_u"][0, 0, :] = np.where(np.isfinite(hours["mss"][0, 0]),
                                       0.0, np.nan)
    c2 = {}
    cy.combine_hours(hours, c2)
    assert c2["hours_mss_unweightable"] == int(np.isfinite(
        hours["mss"][0, 0]).sum())


def test_grid_rotation_rows_and_axis_checks():
    lat = -40.0 + (np.arange(H) + 0.5) * 2.0
    lon = (np.arange(W) + 0.5) * 2.0
    flip, info = cy.check_axes(lat, lon, H, W)
    assert flip is False and info["lat_first"] < info["lat_last"]
    assert cy.check_axes(lat[::-1], lon, H, W)[0] is True
    with pytest.raises(cy.FormatError, match="differs from the declared"):
        cy.check_axes(lat + 1.0, lon, H, W)
    with pytest.raises(cy.FormatError, match="0 .. 360"):
        cy.check_axes(lat, lon - 180.0, H, W)      # a -180..180 axis
    with pytest.raises(cy.FormatError, match="declared grid is"):
        cy.check_axes(lat[:5], lon, H, W)
    # the rotation: producer column 0 (1 E) lands at the store column of 1 E
    a = np.zeros((H, W, 4))
    a[:, 0, 0] = 1.0
    r = cy.to_store_cols(a, W)
    g = cy.grid(H, W)
    col = int(np.nonzero(r[0, :, 0])[0][0])
    assert abs(g["x0"] + (col + 0.5) * g["dx"] - 1.0) < 1e-9


def _days(tmp_path, variant="consistent", n=3, grid=(H, W), **kw):
    pairs = []
    for k in range(n):
        d = D0 + dt.timedelta(days=k)
        _a, p2, p3 = smk.make_day(str(tmp_path), d, grid[0], grid[1],
                                  20261013, variant=variant, **kw)
        pairs.append((d, p2, p3))
    return pairs


def test_f1_passes_on_a_producer_that_bins_as_documented(tmp_path):
    rep = cy.f1_check(_days(tmp_path), H, W)
    assert rep["passed"], rep["failures"]
    for day in rep["days"]:
        for lvl in ("hourly", "daily"):
            L = day[lvl]
            assert L["count_equal_fraction"] == 1.0
            assert L["cells_compared"] > 300
            assert L["mean_within_rel_fraction"] == 1.0
            assert L["uncertainty_within_rel_fraction"] == 1.0
        assert day["l2_samples_fatal_flag"] > 0          # they were excluded
        assert day["l2_samples_nonfatal_flag_kept"] > 0  # and these kept
        assert day["l2_samples_outside_grid_or_day"] > 0
        assert day["wind_info"]["count_equal_fraction"] == 1.0
    json.dumps(rep)                                       # JSON-able


def test_f1_runs_on_the_real_400_by_1800_grid(tmp_path):
    rep = cy.f1_check(_days(tmp_path, n=1, grid=(400, 1800)), 400, 1800)
    assert rep["passed"], rep["failures"]
    assert rep["days"][0]["hourly"]["cells_compared"] > 1000


@pytest.mark.parametrize("variant,key,alt", [
    ("flags_ignored", "count_equal_fraction",
     "count_equal_fraction_if_no_flag_filter"),
    ("unweighted", "mean_within_rel_fraction", "mean_unweighted_vs_l3"),
    ("unc_sqrt2", "uncertainty_within_rel_fraction", None),
])
def test_f1_fails_with_numbers_on_a_different_definition(tmp_path, variant,
                                                         key, alt):
    rep = cy.f1_check(_days(tmp_path, variant=variant), H, W)
    assert not rep["passed"]
    assert rep["verdict"].startswith("FAILED")
    assert any(key in f for f in rep["failures"])
    hr = rep["days"][0]["hourly"]
    assert hr[key] < cy.F1_COUNT_FRAC
    if alt == "count_equal_fraction_if_no_flag_filter":
        # the report names the selection that DOES fit
        assert hr[alt] > 0.999
    elif alt:
        assert hr["alternatives"][alt]["within_rel_fraction"] == 1.0
    if variant == "unc_sqrt2":
        e = hr["uncertainty_rel_error"]
        assert abs(e["median"] - (np.sqrt(2) - 1) / np.sqrt(2)) < 1e-5
        # the counts and the means still agree: the mismatch is LOCATED
        assert hr["count_equal_fraction"] == 1.0
        assert hr["mean_within_rel_fraction"] == 1.0


def test_a_descending_latitude_axis_is_flipped_and_counted(tmp_path):
    a, _p2, p3 = smk.make_day(str(tmp_path / "up"), D0, H, W, 1)
    _b, _q2, q3 = smk.make_day(str(tmp_path / "dn"), D0, H, W, 1,
                               lat_descending=True)
    os.environ["CYGNSS_MSS_SMOKE_GRID"] = f"{H},{W}"
    ad = cy.CygnssMSSAdapter()
    c1, c2 = {}, {}
    f1 = ad.frame_from_file(p3, c1)
    f2 = ad.frame_from_file(q3, c2)
    assert np.array_equal(f1, f2, equal_nan=True)
    assert np.array_equal(f1, smk.day_truth(a, W), equal_nan=True)
    assert c2["lat_axis_flipped"] == 1 and "lat_axis_flipped" not in c1


def test_smoke_every_stage_and_the_probe_with_f1(tmp_path, monkeypatch):
    monkeypatch.setenv("CYGNSS_MSS_F1", "always")
    res = b1.run_smoke("cygnss_mss", root=str(tmp_path / "s"), keep=True)
    chk = res["check"]
    # 2019-01-01 .. 08 with the 3rd absent: seven frames, compared exactly
    assert chk["frames_equal"] == 7
    assert chk["by_reason"]["absent_upstream"] == 1
    assert sum(1 for v in res["truth"].values() if v[0] is not None) == 7
    p = res["probe"]
    g = p["groups"]["cygnss_mss"]
    assert g["C"] == 4 and g["dtype"] == "float16"
    assert g["frames_fetched"] == 7
    assert p["out_of_bounds_stored"] == 0
    # the 150 m/s wind cell is NaN and COUNTED, never clipped
    assert p["out_of_bounds"] == {"wind_speed": 1}
    assert p["bytes_fetched"] > 0
    f1 = p["counts"]["f1_producer_reproduction"]
    assert len(f1) == 1 and f1[0]["passed"], f1[0].get("failures")
    assert f1[0]["days_requested"] == ["2019-01-01", "2019-01-05",
                                       "2019-01-08"]
    assert not p["inputs_not_read"]
    # the store itself
    st = json.load(open(os.path.join(res["work"], "cygnss_mss", "cygnss_mss",
                                     "store.json")))
    assert st["family"] == "family1_3"
    grp = sh.ShardedGroup(os.path.join(res["work"], "cygnss_mss",
                                       "cygnss_mss", "cygnss_mss"))
    b, f = next((b, f) for (_g, b, f), (arr, _w) in res["truth"].items()
                if arr is not None)
    fr = grp.read_frame(b, f)
    assert fr.shape == (H, W, 4)
    n = fr[..., 2]
    assert np.all(n[np.isfinite(n)] == np.rint(n[np.isfinite(n)]))
    assert np.isfinite(fr[..., 0]).sum() > 100


def test_f1_failure_makes_the_probe_refuse_after_reporting(tmp_path,
                                                           monkeypatch):
    src = tmp_path / "src"
    for k in range(5):
        smk.make_day(str(src), D0 + dt.timedelta(days=k), H, W, 7,
                     variant="unweighted")
    monkeypatch.setenv("CYGNSS_MSS_SMOKE_GRID", f"{H},{W}")
    ad = cy.CygnssMSSAdapter()
    absent = []
    ctx = argparse.Namespace(
        a=argparse.Namespace(stage="probe", attempts=1,
                             work=str(tmp_path / "work")),
        source_dir=str(src), d_lo=D0, d_hi=D0 + dt.timedelta(days=4),
        count_bytes=lambda n: None,
        note_absent=lambda u, w: absent.append((u, w)))
    assert ad.fetch_preflight(ctx) is None
    assert absent and absent[0][0] == "F1" and "FAILED" in absent[0][1]
    assert ad._f1["passed"] is False
    files = os.listdir(tmp_path / "work" / "f1")
    assert len(files) == 1
    rep = json.load(open(tmp_path / "work" / "f1" / files[0]))
    alt = rep["days"][0]["hourly"]["alternatives"]["mean_unweighted_vs_l3"]
    assert alt["within_rel_fraction"] == 1.0
    # the report rides on the first frame's counts, once
    assert ad._f1_counts()["f1_producer_reproduction"][0] is ad._f1
    assert ad._f1_counts() == {}


def test_f1_needs_three_days_and_off_means_off(tmp_path, monkeypatch):
    src = tmp_path / "src"
    for k in range(2):
        smk.make_day(str(src), D0 + dt.timedelta(days=k), H, W, 7)
    monkeypatch.setenv("CYGNSS_MSS_SMOKE_GRID", f"{H},{W}")
    ctx = argparse.Namespace(
        a=argparse.Namespace(stage="probe", attempts=1,
                             work=str(tmp_path / "work")),
        source_dir=str(src), d_lo=D0, d_hi=D0 + dt.timedelta(days=1),
        count_bytes=lambda n: None, note_absent=lambda u, w: None)
    rep = cy.CygnssMSSAdapter().run_f1(ctx)
    assert not rep["passed"] and any("needs 3" in f for f in rep["failures"])
    monkeypatch.setenv("CYGNSS_MSS_F1", "off")
    ad = cy.CygnssMSSAdapter()
    ad.fetch_preflight(ctx)
    assert ad._f1 is None
    monkeypatch.setenv("CYGNSS_MSS_F1", "sometimes")
    with pytest.raises(SystemExit, match="probe\\|always\\|off"):
        cy.CygnssMSSAdapter()


def test_no_credentials_refuses_before_a_byte(monkeypatch, tmp_path):
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    monkeypatch.setenv("NETRC", str(tmp_path / "absent"))
    ctx = argparse.Namespace(a=argparse.Namespace(stage="all"),
                             source_dir="")
    with pytest.raises(SystemExit, match="Earthdata"):
        cy.CygnssMSSAdapter().fetch_preflight(ctx)


def test_listing_names_and_the_one_day_rule():
    d = dt.date(2019, 1, 15)
    assert cy._day_of(smk.l3_name(d), cy.L3_NAME) == d
    assert cy._day_of(smk.l2_name(d), cy.L2_NAME) == d
    assert cy._day_of(smk.l2_name(d), cy.L3_NAME) is None
    with pytest.raises(cy.FormatError, match="one UTC day"):
        cy._day_of("cyg.ddmi.s20190115-003000-e20190116-233000.l3.grid-wind"
                   ".a32.d33", cy.L3_NAME)
