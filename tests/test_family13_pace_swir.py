#!/usr/bin/env python3
"""The `pace_swir` adapter's smoke (family 1.3, E-092 phase C) — no network.

    python3 -m pytest -q tests/test_family13_pace_swir.py

The synthetic archive (`pace_swir.make_smoke_sources`) is the real shape:
`cmr_sfrefl_<year>.json` / `cmr_sfrefl_nrt_<year>.json` as NASA's CMR returns
them (the final entries with the `<SHORT>_<file>_3.1` titles CMR really serves
for 2024), native 122-band `rhos` files with the measured wavelength axis, and
pace4k's BGC chlorophyll beside them as the clear-water mask. The adapter
reads them through the SAME subset reader it uses on OPeNDAP's answers (the
local path cuts the identical strided band slices). One day has only an NRT
file (absent upstream), the final edge falls inside the window with NRT days
past it (after the record), one day has no BGC file, one day's clear water is
bright enough to fail the soft dark-water test, and one rhos_1249 is out of
bounds. Expected counts are exact.
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
from family1.adapters import pace4k as p4                       # noqa: E402
from family1.adapters import pace_swir as ps                    # noqa: E402

pytest.importorskip("netCDF4")

ENV = ("PACE_SWIR_SMOKE_GRID", "PACE_SWIR_LANES", "PACE_SWIR_WORKERS",
       "PACE_SWIR_TIMEOUT")


def clear_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("pace_swir_smoke"))
    keep = {k: os.environ.get(k) for k in ENV}
    try:
        return b1.run_smoke("pace_swir", root=root, keep=True)
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def small_archive(tmp, **kw):
    src = os.path.join(tmp, "src")
    d_lo, d_hi = (b10.parse_date(x) for x in ps.PaceSwirAdapter.smoke_window)
    truth = ps.make_smoke_sources(src, d_lo, d_hi, **kw)
    return src, truth


def ctx_for(tmp, src, lo=None, hi=None, **over):
    w = ps.PaceSwirAdapter.smoke_window
    d = dict(store="pace_swir", work=os.path.join(tmp, "work"),
             source_dir=src, start=lo or w[0], end=hi or w[1], stage="all",
             force=False, attempts=1, qc_keep=2,
             check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
             parts_from_hub=False, push_parts=False,
             allow_missing_years=False, allow_unconfirmed_licence=False,
             probe_month="", smoke=False)
    d.update(over)
    ad = ps.PaceSwirAdapter()
    ctx = b10.Ctx(argparse.Namespace(**d), adapter=ad,
                  layout=b1.layout_for(ad))
    return b1.prepare_grid_ctx(ctx)


def run(ctx, stages):
    b10.run_stages(ctx, stages, stage_fn=b1.GRID_STAGE_FN, deps=b1.DEPS)


def bf(d):
    s = b10.seconds_since_epoch(d)
    return s // sh.BIN_SECONDS, (s % sh.BIN_SECONDS) // 86400


# ================================================================ facts ====
def test_registered_in_family_13_on_pace4k_s_grid(monkeypatch):
    clear_env(monkeypatch)
    assert fam.REGISTRY["pace_swir"] is ps.PaceSwirAdapter
    ad = ps.PaceSwirAdapter()
    assert b1.is_grid(ad) and ad.layout == "sharded"
    assert ad.family == "13" and ad.distribution == "public"
    assert b1.layout_for(ad).hf_root == "tensors/family1_3"
    assert ad.credentials == ()                 # OPeNDAP answered anonymously
    assert ad.C == 8 and ad.dtype == "float16"
    assert ad.channel_names == ["rhos_645", "rhos_740", "rhos_865",
                                "rhos_1038", "rhos_1249", "rhos_1618",
                                "rhos_2131", "rhos_2258"]
    for _n, unit, lo, hi in ad.channels:
        assert unit.startswith("1 ") and (lo, hi) == (-0.05, 1.5)
    assert (ad.frames_per_bin, ad.frame_seconds) == (5, 86400)
    assert ad.first_year == 2024 and str(ps.RECORD_START) == "2024-03-05"
    a, b = ps.grid_of(), p4.grid_of()
    for k in ("H", "W", "x0", "dx", "y0", "dy", "crs", "pixel_deg",
              "row_order"):
        assert a[k] == b[k], k
    assert "no land mask" in a["extent_note"]
    assert ad.log2_fp == p4.Pace4kAdapter.log2_fp
    assert ad.licence["redistribution_confirmed"] is True
    assert "10.5067/PACE/OCI/L3M/SFREFL/3.1" in ad.licence["attribution"]


def test_the_bands_are_chosen_from_the_axis_and_740_is_739():
    """MEASURED axis of 2025-07-01: 645 is exact (index 56); 740 has no band,
    739 (index 84) is 1 nm away and 742 (index 85) 2 nm."""
    got = ps.choose_bands(ps.MEASURED_AXIS)
    assert {n: (i, nm) for n, i, nm in got} == ps.MEASURED_BANDS
    assert ps.MEASURED_AXIS[84] == 739 and ps.MEASURED_AXIS[85] == 742
    assert ps.MEASURED_AXIS[56] == 645
    assert len(ps.MEASURED_AXIS) == 122
    # a reordered axis moves the index, never the wavelength
    ax = list(ps.MEASURED_AXIS)
    ax.insert(0, 300)
    moved = {n: (i, nm) for n, i, nm in ps.choose_bands(ax)}
    assert moved["rhos_740"] == (85, 739)
    # nothing within 2.5 nm is a refusal; so is a tie
    with pytest.raises(ps.FormatError):
        ps.choose_bands([v for v in ps.MEASURED_AXIS if v not in (739, 742)])
    with pytest.raises(ps.FormatError):
        ps.choose_bands([v if v != 739 else 738 for v in ps.MEASURED_AXIS
                         if v != 742] + [742])


def test_four_requests_a_day_one_per_contiguous_run():
    """Contiguous, never strided: a strided slab was measured ~3x slower on
    the server (289 s for two bands against 85 s for one)."""
    idx = [i for i, _nm in ps.MEASURED_BANDS.values()]
    runs = ps.band_requests(idx)
    assert runs == [(56, 1, 56), (84, 1, 84), (110, 1, 110), (117, 1, 121)]
    assert all(r[1] == 1 for r in runs)
    assert sorted(sum((ps.run_indices(r) for r in runs), [])) == sorted(idx)
    assert ps.band_requests([5]) == [(5, 1, 5)]
    assert ps.band_requests([1, 2, 3, 7, 9]) == [(1, 1, 3), (7, 1, 7),
                                                 (9, 1, 9)]
    u = ps.subset_url(dt.date(2025, 7, 1), (56, 28, 84), latlon=True)
    assert u.startswith(ps.OPENDAP + "L3SMI/2025/0701/PACE_OCI.20250701."
                                     "L3m.DAY.SFREFL.V3_1.rhos.4km.nc"
                                     ".dap.nc4?dap4.ce=")
    assert "/rhos%5B0:1:4319%5D%5B0:1:8639%5D%5B56:28:84%5D" in u
    assert ";/wavelength%5B56:28:84%5D;/lat;/lon" in u
    assert "%5B1611%5D%5B3040%5D%5B56:1:121%5D" in ps.pixel_url(
        dt.date(2025, 7, 1), 1611, 3040, 56, 121)


def test_names_titles_and_the_nrt_suffix():
    d = dt.date(2025, 7, 1)
    assert ps.object_name(d) == \
        "PACE_OCI.20250701.L3m.DAY.SFREFL.V3_1.rhos.4km.nc"
    assert ps.object_name(d, nrt=True).endswith(".rhos.4km.NRT.nc")
    # CMR's 2024-25 titles are prefixed and suffixed; 2026's are bare
    e = {"title": "PACE_OCI_L3M_SFREFL_PACE_OCI.20250701.L3m.DAY.SFREFL."
                  "V3_1.rhos.4km.nc_3.1"}
    assert ps.granule_name(e) == ps.object_name(d)
    e = {"title": "PACE_OCI_L3M_SFREFL_NRT_PACE_OCI.20250701.L3m.DAY."
                  "SFREFL.V3_1.rhos.4km.NRT.nc_3.1"}
    assert ps.granule_name(e, nrt=True) == ps.object_name(d, nrt=True)
    assert ps.granule_name({"producer_granule_id": ps.object_name(d),
                            "title": "anything"}) == ps.object_name(d)
    assert ps.parse_day(ps.object_name(d)) == d
    assert ps.parse_day(ps.object_name(d, True), nrt=True) == d
    with pytest.raises(ps.FormatError):            # NRT where final expected
        ps.parse_day(ps.object_name(d, nrt=True))
    with pytest.raises(ps.FormatError):
        ps.parse_day("PACE_OCI.20250701.L3m.DAY.SFREFL.V3_1.rhos.2km.nc")


def test_the_dark_water_rule():
    assert ps.judge_dark(9999, 0.005, 0.02) == "too_few"
    assert ps.judge_dark(10000, 0.005, 0.02) == "pass"
    assert ps.judge_dark(10000, 0.0101, 0.02) == "soft"
    assert ps.judge_dark(10000, 0.005, 0.031) == "soft"
    assert ps.judge_dark(10000, 0.03, 0.05) == "gross"
    assert ps.judge_dark(10000, 0.005, 0.10) == "gross"
    # the measurement it was set from passes
    assert ps.judge_dark(2928631, 0.005238, 0.021666) == "pass"


def test_lanes_are_declared_only_when_asked(monkeypatch):
    clear_env(monkeypatch)
    ad = ps.PaceSwirAdapter()
    ad._edge = dt.date(2026, 8, 31)
    ad._window_hi = dt.date(2026, 8, 31)
    assert ad.declared_lanes([2024, 2025, 2026]) == {}
    monkeypatch.setenv("PACE_SWIR_LANES", "months")
    got = ad.declared_lanes([2024, 2025, 2026])
    assert got["2024"] == [f"m{m:02d}" for m in range(3, 13)]
    assert got["2025"] == [f"m{m:02d}" for m in range(1, 13)]
    assert got["2026"] == [f"m{m:02d}" for m in range(1, 9)]
    ad._window_hi = dt.date(2026, 6, 30)
    assert ad.declared_lanes([2026])["2026"][-1] == "m06"


# ============================================================== the smoke ==
def test_smoke_reads_every_frame_back_exactly(smoke):
    res = smoke["check"]
    truth = smoke["truth"]
    n_real = sum(1 for a, _w in truth.values() if a is not None)
    assert n_real == 12                  # 03-05..03-17 less the NRT-only day
    assert res["frames_equal"] == n_real
    assert res["by_reason"]["absent_upstream"] == 1
    assert res["by_reason"]["after_record"] == 3     # 03-18, 03-19, 03-20
    sm = json.load(open(os.path.join(smoke["ctx"].store, "store.json")))
    assert sm["tier"] == "G" and sm["layout"] == "sharded"
    assert sorted(sm["groups"]) == ["pace_swir"]
    assert sm["C"] == 8 and sm["dtype"] == "float16"
    c = sm["counts"]
    assert c["out_of_bounds"] == {"rhos_1249": 1}
    assert c["nrt_only_inside_record"] == 1
    assert c["nrt_after_final"] == 2                # 03-20 has no NRT file
    # both falsifiers ran on every real frame
    assert c["pixel_reread_f16_equal"] == n_real
    assert c["dark_swir_no_mask"] == 1
    assert c["dark_swir_judged"] == n_real - 1
    assert c["dark_swir_soft_fail"] == 1
    assert c["dark_swir_soft_fail_days"] == [str(ps.SMOKE_SOFT)]
    assert len(c["dark_swir_days"]) == n_real - 1
    assert c["max_dark_swir_p50"] > ps.DARK_P50 > c["dark_swir_days"][0][2]
    assert c["license"] == {ps.LICENCE_URL: n_real}
    assert c["processing_version"] == {"3.1": n_real}
    assert "SMOKE GRID" in sm["notes"]


def test_the_plan_records_the_nrt_boundary_and_the_bands(smoke):
    plan = json.load(open(os.path.join(smoke["ctx"].root, "plan.json")))
    nb = plan["nrt_boundary"]
    assert nb["final_last_day"] == str(ps.SMOKE_FINAL_LAST)
    assert nb["nrt_first_day_listed"] == str(ps.SMOKE_ABSENT)
    assert nb["nrt_last_day"] == "2024-03-19"
    assert nb["nrt_days_after_final"] == 2
    assert nb["nrt_after_final_first"] == "2024-03-18"
    assert nb["nrt_only_inside_record"] == [str(ps.SMOKE_ABSENT)]
    assert plan["record"] == ["2024-03-05", str(ps.SMOKE_FINAL_LAST)]
    assert plan["days_absent_upstream_in_window"] == [str(ps.SMOKE_ABSENT)]
    assert plan["bands_match_measurement"] is True
    assert [b["wavelength_nm"] for b in plan["bands"]] == \
        [645, 739, 865, 1038, 1249, 1618, 2131, 2258]
    assert plan["requests_per_day"] == [[56, 1, 56], [84, 1, 84],
                                        [110, 1, 110], [117, 1, 121]]
    ic = plan["index_check"]
    assert ic["pixel_reread_f16_equal"] == 1 and ic["day"] == "2024-03-05"
    assert ic["rows"] == [ps.SMOKE_H // 2 - 32, ps.SMOKE_H // 2 + 31]
    assert 0 < ic["bytes"] < 2_000_000            # a band, not the day
    assert ic["license"] == ps.LICENCE_URL
    # only the window's year and the edge search are listed, not the record
    assert nb["listed_years"] == [2024, 2025, 2026]
    spec = json.load(open(os.path.join(smoke["ctx"].store, "pace_swir",
                                       "tile_grid.json")))
    wl = {c["name"]: c["wavelength_nm"] for c in spec["channels"]}
    assert wl["rhos_740"] == 739 and wl["rhos_2258"] == 2258


def test_all_surfaces_are_kept_and_the_frame_is_the_source(smoke):
    """No land mask: the bright 'land' disc is stored, and every frame is
    the native file's eight bands, fill -> NaN, out of bounds -> NaN."""
    grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, "pace_swir"))
    d = dt.date(2024, 3, 6)
    a = grp.read_frame(*bf(d))
    rh, chl = ps.smoke_fields(d, ps.SMOKE_H, ps.SMOKE_W)
    land = (chl == np.float32(-32767.0)) & (rh[:, :, 121] != ps.FILL)
    assert land.sum() > 1000
    assert np.isfinite(a[:, :, 7][land]).all()
    assert float(np.nanmin(a[:, :, 7][land])) > 0.04      # bright, stored
    for k, (i, _nm) in enumerate(ps.MEASURED_BANDS.values()):
        want = rh[:, :, i].astype(np.float32)
        want[want == ps.FILL] = np.nan
        assert np.array_equal(a[:, :, k], want.astype(np.float16)
                              .astype(np.float32), equal_nan=True), k
    o = grp.read_frame(*bf(ps.SMOKE_OOB))
    assert np.isnan(o[ps.SMOKE_H // 2, ps.SMOKE_W // 3, 4])
    assert np.isfinite(o[ps.SMOKE_H // 2, ps.SMOKE_W // 3, 3])


def test_absent_and_after_record_frames_are_zero_length(smoke):
    grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, "pace_swir"))
    assert grp.read_frame(*bf(ps.SMOKE_ABSENT)) is None
    assert grp.entry(*bf(ps.SMOKE_ABSENT), 0, 0) == (sh.FRAME_ABSENT, 0)
    assert grp.read_frame(*bf(dt.date(2024, 3, 18))) is None


def test_the_probe_measured_the_same_frames_and_some_bytes(smoke):
    p = smoke["probe"]
    assert p["out_of_bounds_stored"] == 0
    assert p["bytes_fetched"] > 0 and p["month"] == "2024-03"
    g = p["groups"]["pace_swir"]
    assert g["C"] == 8 and g["frames_fetched"] == 12
    assert p["counts"]["pixel_reread_f16_equal"] == 12


# ============================================================== refusals ===
def test_a_pixel_reread_that_disagrees_is_refused(tmp_path, monkeypatch):
    src, _ = small_archive(str(tmp_path))
    ctx = ctx_for(str(tmp_path), src)
    real = ps.PaceSwirAdapter.native_pixel

    def off(self, ctx, d, r, c, lo, hi):
        v, wl = real(self, ctx, d, r, c, lo, hi)
        if d == dt.date(2024, 3, 7):
            v = v.copy()
            v[-1] = v[-1] + np.float32(0.01)    # rhos_2258, far past f16 ulp
        return v, wl

    monkeypatch.setattr(ps.PaceSwirAdapter, "native_pixel", off)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index", "fetch"])
    assert "not equal to float16 precision" in str(e.value)
    assert "2024-03-07" in str(e.value)


def test_a_gross_dark_water_failure_is_refused(tmp_path):
    src, _ = small_archive(str(tmp_path), gross_day=dt.date(2024, 3, 9))
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index", "fetch"])
    assert "GROSS" in str(e.value) and "2024-03-09" in str(e.value)


def test_a_wrong_wavelength_in_a_response_is_refused(tmp_path):
    """Every response's wavelength slice is compared with the chosen bands."""
    src, _ = small_archive(str(tmp_path))
    d = dt.date(2024, 3, 5)
    nat = os.path.join(src, "pace_swir", "opendap", "L3SMI", "2024", "0305",
                       ps.object_name(d))
    out = os.path.join(str(tmp_path), "run.nc4")
    ps.local_subset(nat, out, (56, 28, 84), latlon=True)
    a, attrs = ps.read_subset(out, [645, 739], ps.SMOKE_H, ps.SMOKE_W,
                              need_grid=True)
    assert a.shape == (ps.SMOKE_H, ps.SMOKE_W, 2)
    assert attrs["license"] == ps.LICENCE_URL
    # the row-block decode the real grid uses is the whole array, exactly
    blocks = list(ps.iter_subset(out, [645, 739], ps.SMOKE_H, ps.SMOKE_W,
                                 block=50))
    assert [r0 for r0, _a, _t in blocks] == [0, 50, 100, 150, 200]
    assert np.array_equal(np.concatenate([b for _r, b, _t in blocks]), a,
                          equal_nan=True)
    with pytest.raises(ps.FormatError):
        ps.read_subset(out, [645, 742], ps.SMOKE_H, ps.SMOKE_W)
    with pytest.raises(ps.FormatError):               # the wrong grid
        ps.read_subset(out, [645, 739], ps.SMOKE_H + 2, ps.SMOKE_W,
                       need_grid=True)


def test_index_refuses_an_empty_listing(tmp_path):
    src, _ = small_archive(str(tmp_path))
    json.dump({"feed": {"entry": []}},
              open(os.path.join(src, "pace_swir", "cmr_sfrefl_2024.json"),
                   "w"))
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index"])
    assert "no final SFREFL granule" in str(e.value)


def test_a_listed_file_that_is_absent_stops_the_year(tmp_path):
    src, _ = small_archive(str(tmp_path))
    d = dt.date(2024, 3, 7)
    os.remove(os.path.join(src, "pace_swir", "opendap", "L3SMI", "2024",
                           "0307", ps.object_name(d)))
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index", "fetch"])
    assert "could not be read" in str(e.value)
    assert ctx.absent


# ========================================================= the network ====
class _Resp:
    def __init__(self, body, length=None):
        import io
        self._b = io.BytesIO(body)
        self.headers = {"Content-Length": str(len(body) if length is None
                                              else length)}

    def read(self, n=-1):
        return self._b.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _script(monkeypatch, outcomes):
    """urlopen replaced by a script of outcomes; returns (calls, waits)."""
    import urllib.error
    calls, waits = [], []
    it = iter(outcomes)

    def fake(req, timeout=None):
        calls.append(timeout)
        o = next(it)
        if isinstance(o, BaseException):
            raise o
        if isinstance(o, int):
            raise urllib.error.HTTPError(req.full_url, o, "x", {}, None)
        return _Resp(*o) if isinstance(o, tuple) else _Resp(o)

    monkeypatch.setattr(ps.urllib.request, "urlopen", fake)
    monkeypatch.setattr(ps, "SLEEP", waits.append)
    return calls, waits


GOOD = ps.HDF5_MAGIC + b"payload"
PAGE = b"<!doctype html><title>ERROR @ Ocean Biology DAAC</title>"


def test_the_knobs_and_their_defaults(monkeypatch):
    clear_env(monkeypatch)
    ad = ps.PaceSwirAdapter()
    assert (ad.WORKERS, ad.TIMEOUT) == (2, 600.0)
    monkeypatch.setenv("PACE_SWIR_WORKERS", "1")
    monkeypatch.setenv("PACE_SWIR_TIMEOUT", "900")
    ad = ps.PaceSwirAdapter()
    assert (ad.WORKERS, ad.TIMEOUT) == (1, 900.0)
    monkeypatch.setenv("PACE_SWIR_WORKERS", "0")
    with pytest.raises(SystemExit):
        ps.PaceSwirAdapter()
    assert [ps.backoff(i) for i in range(6)] == [30, 60, 120, 240, 480, 480]


def test_timeouts_and_error_pages_back_off_and_are_counted(tmp_path,
                                                           monkeypatch):
    calls, waits = _script(monkeypatch, [TimeoutError("The read operation "
                                                      "timed out"),
                                         PAGE, 503, GOOD])
    c = {}
    p = str(tmp_path / "x.nc4")
    assert ps.http_get("https://h/x", p, 6, 600.0, c) == p
    assert open(p, "rb").read() == GOOD
    assert calls == [600.0] * 4                # the timeout is EXPLICIT
    assert waits == [30, 60, 120]
    assert c["net_retry"] == {"timeout": 1, "error_page": 1, "http_5xx": 1}
    assert c["net_backoff_seconds"] == 210
    assert c["net_recovered"] == 1 and c["max_net_attempts"] == 4
    assert not os.path.exists(p + ".part")


def test_a_truncated_body_is_retried_and_exhaustion_raises(tmp_path,
                                                           monkeypatch):
    _c, waits = _script(monkeypatch, [(GOOD, 999), PAGE, PAGE])
    c = {}
    with pytest.raises(IOError) as e:
        ps.http_get("https://h/x", str(tmp_path / "x"), 3, 5.0, c)
    assert "3 attempt(s) failed" in str(e.value)
    assert waits == [30, 60]                   # no wait after the last
    assert c["net_retry"] == {"truncated": 1, "error_page": 2}
    assert c["net_gave_up"] == 1


def test_404_is_none_and_400_is_definite(tmp_path, monkeypatch):
    _c, waits = _script(monkeypatch, [404])
    assert ps.http_get("https://h/x", str(tmp_path / "x"), 6, 5.0, {}) is None
    _c, waits = _script(monkeypatch, [400])
    with pytest.raises(IOError) as e:
        ps.http_get("https://h/x", str(tmp_path / "x"), 6, 5.0, {})
    assert "not retried" in str(e.value) and waits == []


def test_the_ledger_reaches_the_frame(tmp_path, monkeypatch):
    """A retry inside a day's download is counted in that frame's counts."""
    src, _ = small_archive(str(tmp_path))
    ctx = ctx_for(str(tmp_path), src)
    real = ps.PaceSwirAdapter.fetch_day

    def noisy(self, ctx, d):
        paths, nb, tmp = real(self, ctx, d)
        paths["net"] = {"net_retry": {"timeout": 2},
                        "net_backoff_seconds": 90.0, "max_net_attempts": 3}
        return paths, nb, tmp

    monkeypatch.setattr(ps.PaceSwirAdapter, "fetch_day", noisy)
    run(ctx, ["index", "fetch"])
    led = json.load(open(os.path.join(ctx.year_dir(2024), "counts.json")))
    c = led.get("counts", led)
    assert c["net_retry"]["timeout"] == 2 * 12
    assert c["max_net_attempts"] == 3
