#!/usr/bin/env python3
"""E-087 — the daily form of family 7.2 (`ml/family7_daily.py`), no network.

Family 7.2 is the global input tensor of five-day means; E-087 plans the same
channels at DAILY resolution. What is asserted here is the one promise a daily
store makes about itself: **averaging its five frames by each channel's own
rule reproduces the family-7 builder's pentad value.** The reference is the
REAL builder (`ml/build_family7.py`'s `glorys`, `sst`, `ncep` and `occci`
stages) run on its own synthetic smoke sources, and compared BEFORE its
z-score — g025 against its float16 fill (physical units), g100 and oc025
against their float32 fill — so the tolerance is the fill's own rounding and
nothing else.

    python3 -m pytest -q tests/test_family7_daily.py

  the rules       hypot of the MEAN currents (not the mean speed), log10 of the
                  MEAN mixed-layer depth, the >= 3-day rule (2 days -> NaN),
                  log1p(mean(expm1)) for the two log channels, frame 2 for the
                  centred sigma, >= 1 day and sum/5 for ocean colour.
  the builder     every bin of the smoke window, every channel of g025, g100
                  and oc025: the reconstruction equals the builder's fill, and
                  is NaN exactly where the fill is.
  6-hourly        NCEP's day value is the mean of the day's four samples, and
                  frame 2's sigma is the population sigma of the bin's twenty.
  the writer      frames through the family-1 sharded writer read back
                  bit-exact in float16; an absent frame reads back as None.
"""
import argparse
import datetime as dt
import os
import shutil
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(os.path.dirname(HERE), "ml")
sys.path.insert(0, ML)

import build_family7 as f7                                       # noqa: E402
import family7_daily as fd                                       # noqa: E402
from family1 import sharded as sh                                # noqa: E402

START, END = f7.SMOKE_START, f7.SMOKE_END
OC_START = f7.SMOKE_OC_START


# ================================================================ rules ====
def _frames(vals, C):
    """[5] list of scalars per channel -> [5, 1, 1, C]."""
    a = np.full((5, 1, 1, C), np.nan)
    for c, v in enumerate(vals):
        a[:, 0, 0, c] = v
    return a


def test_glorys_rules_hypot_of_means_and_log_of_mean():
    u = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    v = np.zeros(5)
    mld = np.array([10.0, 100.0, 10.0, 100.0, 10.0])
    st = np.zeros((5, 1, 1, 5))
    st[..., 0, 0, 0] = np.hypot(u, v)
    st[..., 0, 0, 1] = np.log10(mld)
    st[..., 0, 0, 3] = u
    st[..., 0, 0, 4] = v
    out = fd.pentad_from_daily("glorys025d", st)[0, 0]
    assert out[3] == pytest.approx(0.2)
    assert out[0] == pytest.approx(0.2)            # NOT mean(|u|) = 1
    assert out[1] == pytest.approx(np.log10(46.0))  # NOT mean(log10) = 1.4


def test_three_day_rule():
    st = _frames([np.nan] * 2, 2)
    st[:2, 0, 0, 0] = 20.0                          # two days only
    st[:3, 0, 0, 1] = 0.5                           # three days
    out = fd.pentad_from_daily("oisst025d", st)[0, 0]
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(0.5)


def test_ncep_log_channels_and_frame_two_sigma():
    st = np.zeros((5, 1, 1, 15))
    rate = np.array([0.0, 2.0, 10.0, 0.5, 30.0])    # mm/day
    st[:, 0, 0, 8] = np.log1p(rate)
    st[:, 0, 0, 2] = [9, 9, 0.7, 9, 9]              # only frame 2 counts
    out = fd.pentad_from_daily("ncep100d", st)[0, 0]
    assert out[8] == pytest.approx(np.log1p(rate.mean()))
    assert out[2] == pytest.approx(0.7)


def test_colour_one_day_and_coverage_sum():
    st = _frames([np.nan, np.nan], 2)
    st[3, 0, 0] = [-0.5, 0.4]                       # one clear day
    out = fd.pentad_from_daily("occci025d", st)[0, 0]
    assert out[0] == pytest.approx(-0.5)
    assert out[1] == pytest.approx(0.4 / 5)


# ======================================= against the real family-7 builder ==
@pytest.fixture(scope="module")
def smoke():
    tmp = tempfile.mkdtemp(prefix="f7daily_")
    src = os.path.join(tmp, "src")
    work = os.path.join(tmp, "work")
    d_lo = dt.date(*(int(x) for x in START.split("-")))
    d_hi = dt.date(*(int(x) for x in END.split("-")))
    days = [d_lo + dt.timedelta(days=k) for k in range((d_hi - d_lo).days + 1)]
    f7.make_smoke_sources(src, d_lo, d_hi)
    f7.make_smoke_oc_sources(src, days, dt.date(*(int(x) for x in
                                                   OC_START.split("-"))))
    a = argparse.Namespace(work=work, source_dir=src, start=START, end=END,
                           force=False, stage="glorys,sst,ncep,occci",
                           smoke=True, seed_from="", oc_source="occci",
                           oc_start=OC_START, oc_preflight=False)
    ctx = f7.Ctx(a)
    for st in (f7.stage_glorys, f7.stage_sst, f7.stage_ncep, f7.stage_occci):
        st(ctx)
    yield ctx, src, days
    shutil.rmtree(tmp, ignore_errors=True)


def _daily(store, src, days):
    if store == "glorys025d":
        months = sorted({(d.year, d.month) for d in days})
        return fd.glorys_daily([os.path.join(
            src, "daily025_global", f"glorys025_global_{y}{m:02d}.nc")
            for y, m in months], days)
    if store == "oisst025d":
        y = days[0].year
        return fd.oisst_daily(os.path.join(src, "oisst", f"sst.day.mean.{y}.nc"),
                              os.path.join(src, "oisst",
                                           f"icec.day.mean.{y}.nc"), days)
    if store == "ncep100d":
        import netCDF4 as ncdf
        paths = {v: [os.path.join(src, "ncep",
                                  f"{f7.NCEP_FILES[v]}.{days[0].year}.nc")]
                 for v in fd.NCEP_ORDER}
        d = ncdf.Dataset(os.path.join(src, "ncep", f"{f7.NCEP_LAND}.nc"))
        land = f7.squeeze_level(np.ma.filled(
            np.asarray(f7.pick_var(d, "land")[:]), 0.0)) >= 0.5
        d.close()
        return fd.ncep_daily(paths, land, days)
    out = {}
    for d in days:
        p = os.path.join(src, "occci", f7.oc_canonical_name(d))
        if os.path.exists(p):
            out[d] = fd.occci_daily(p)[0]
    return out


@pytest.mark.parametrize("store", ["glorys025d", "oisst025d", "ncep100d",
                                   "occci025d"])
def test_mean_of_daily_reproduces_the_builders_pentad(smoke, store):
    ctx, src, days = smoke
    cfg = fd.STORES[store]
    g = cfg["group"]
    shape = ctx.shapes()[g]
    X = np.asarray(f7.open_fill(ctx.work, g, shape), np.float64)
    daily = _daily(store, src, days)
    assert daily, f"{store}: no daily frames from the smoke sources"
    H, W, C = shape[1], shape[2], len(cfg["channels"])
    checked = 0
    for b in ctx.bins:
        row = ctx.oc_row_of(b) if g == "oc025" else ctx.row_of(b)
        if row is None:
            continue
        fr = np.stack([daily.get(d, np.full((H, W, C), np.nan, np.float32))
                       for d in fd.bin_days(b)])
        R = fd.pentad_from_daily(store, fr)
        mag = np.nanmax(np.abs(np.where(np.isfinite(fr), fr, 0.0)), axis=0)
        for c, (name, *_r) in enumerate(cfg["channels"]):
            P = X[row, ..., cfg["pentad_index"][c]]
            fp, fr_ = np.isfinite(P), np.isfinite(R[..., c])
            assert np.array_equal(fp, fr_), (
                f"{store} {name} bin {b}: NaN pattern differs "
                f"({int((fp & ~fr_).sum())} pentad-only, "
                f"{int((fr_ & ~fp).sum())} daily-only)")
            if not fp.any():
                continue
            d = np.abs(R[..., c] - P)[fp]
            # float32 on both paths (interp2_nan's output, K/Pa transforms, a
            # float32 daily log10): 8 half-ulps of the largest summand ...
            off = 273.15 if name in ("t2m", "tsoil", "skt") else 0.0
            tol = 2.0 ** -21 * (np.maximum(np.abs(P[fp]),
                                           mag[..., c][fp]) + off)
            if g == "g025":                      # ... plus a float16 fill's
                tol = tol + fd.f16_half_step(    # half step
                    np.maximum(np.abs(P[fp]), np.abs(R[..., c][fp])))
            tol = tol * (1 + 1e-6) + 1e-12
            assert (d <= tol).all(), (
                f"{store} {name} bin {b}: max |mean of daily - pentad| "
                f"{d.max():.3g} exceeds the fill's rounding")
            checked += int(fp.sum())
    assert checked > 0


# ============================================================= 6-hourly ===
def test_ncep_day_is_the_mean_of_its_four_samples(tmp_path):
    lat = np.linspace(80.0, -80.0, 9)
    lon = np.arange(0.0, 360.0, 30.0)
    d0 = dt.date(2015, 1, 1)
    days = [d0 + dt.timedelta(days=k) for k in range(15)]
    hours = np.array([(d - dt.date(1800, 1, 1)).days * 24.0 + h
                      for d in days for h in (0, 6, 12, 18)])
    rng = np.random.default_rng(7)
    paths = {}
    data = {}
    for v in fd.NCEP_ORDER:
        a = (rng.standard_normal((len(hours), len(lat), len(lon))) * 0.1
             + (300.0 if v in ("air", "tmp", "skt") else
                1e5 if v == "pres" else 1e-5 if v == "prate" else 0.2)
             ).astype(np.float32)
        data[v] = a
        p = str(tmp_path / f"{v}.nc")
        f7._nc_write(p, {"time": len(hours), "lat": len(lat), "lon": len(lon)},
                     {"lat": (("lat",), lat, None),
                      "lon": (("lon",), lon, None),
                      "time": (("time",), hours,
                               {"units": "hours since 1800-01-01 00:00:0.0"}),
                      v: (("time", "lat", "lon"), a, None)})
        paths[v] = [p]
    land = np.ones((len(lat), len(lon)), bool)
    b = f7.bin_index(dt.date(2015, 1, 3), 5)        # 2015-01-03 .. 07
    want = fd.bin_days(b)
    out = fd.ncep_daily(paths, land, want)
    assert sorted(out) == want
    # the day value: mean of the four 6-hourly samples, through the same to1
    f3l = fd.f3
    lat1, lon1 = f7.grid100()
    W = (f3l.lin_weights(lat, lat1),
         f3l.lin_weights(lon, np.where(lon1 < 0, lon1 + 360.0, lon1),
                         wrap_period=360.0))
    k = days.index(want[1])
    m = data["skt"][4 * k:4 * k + 4].astype(np.float64).mean(0)
    np.testing.assert_allclose(out[want[1]][..., 14],
                               f3l.interp2_nan(m, *W) - 273.15, atol=2e-5)
    # frame 2's sigma is the pentad sigma over all twenty samples of the bin
    k0 = days.index(want[0])
    tau = -data["uflx"][4 * k0:4 * k0 + 20].astype(np.float64)
    np.testing.assert_allclose(out[want[2]][..., 2],
                               f3l.interp2_nan(tau.std(0), *W), atol=1e-6)


# =============================================================== writer ===
def test_frames_round_trip_through_the_sharded_writer(tmp_path):
    rng = np.random.default_rng(3)
    cfg = fd.STORES["ncep100d"]
    H, W, C = 181, 360, 15
    frames = []
    for f in range(5):
        a = rng.standard_normal((H, W, C)).astype(np.float32)
        a[:40, :, 10:12] = np.nan                    # soil over the sea
        frames.append(a)
    frames[3] = None                                 # absent upstream
    b = 2411
    entries, spec, gd = fd.write_store("ncep100d", {b: frames}, str(tmp_path))
    assert entries[0]["frames_present"] == 4
    assert spec["tile"] == cfg["tile"] and spec["frames_per_bin"] == 5
    g = sh.ShardedGroup(gd)
    for f in range(5):
        got = g.read_frame(b, f, raw=True)
        if frames[f] is None:
            assert got is None
            continue
        np.testing.assert_array_equal(got, frames[f].astype(np.float16))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ======================================== the lane's consistency refusal ===
class _Ctx:
    source_dir = ""
    scratch = "/nonexistent"

    def __init__(self):
        self.absent = []

    def note_absent(self, unit, why):
        self.absent.append((unit, why))


def _lane(perturb):
    """ncep100d's fetch_frames over one bin of synthetic frames, against a
    'published' pentad computed from those same frames (optionally nudged)."""
    from family1.adapters import REGISTRY
    ad = REGISTRY["ncep100d"]()
    rng = np.random.default_rng(11)
    b = 2411
    days = fd.bin_days(b)
    frames = {d: (rng.standard_normal((181, 360, 15)) * 0.5 + 2.0)
              .astype(np.float32) for d in days}
    for a in frames.values():
        a[..., 10:12] = np.abs(a[..., 10:12]) * 0.1        # soil in bounds
        a[..., 7] = 1000.0 + a[..., 7]                      # sp in hPa
        a[..., 2:4] = np.abs(a[..., 2:4])                   # sigma >= 0
        a[..., 8:10] = np.abs(a[..., 8:10])                 # log1p >= 0
    ad.days_frames = lambda ctx, ds: ((d, frames[d], None) for d in ds)
    st = [frames[d].astype(np.float16).astype(np.float32) for d in days]
    P = fd.pentad_from_daily("ncep100d", np.stack(st))
    if perturb:
        P[90, 180, 4] += 0.05                               # one t2m cell
    norm = np.tile([0.0, 1.0], (15, 1))

    class _Ref:
        def bin(self, bb):
            return P, P.copy(), norm
    ad._ref = _Ref()
    os.environ["F7D_PENTAD_CHECK"] = "on"
    try:
        ctx = _Ctx()
        out = list(ad.fetch_frames(ctx, [("ncep100d", b, f)
                                         for f in range(5)]))
    finally:
        del os.environ["F7D_PENTAD_CHECK"]
    return ctx, out


def test_a_bin_that_reproduces_the_pentad_is_yielded():
    ctx, out = _lane(False)
    assert not ctx.absent and len(out) == 5
    c = out[0][4]
    assert c["pentad_bins_checked"] == 1 and c["max_pentad_excess_t2m"] <= 0


def test_a_bin_that_does_not_is_refused_and_its_year_left_unmarked():
    ctx, out = _lane(True)
    assert out == []
    assert len(ctx.absent) == 1
    unit, why = ctx.absent[0]
    assert unit.startswith("2015 bin 2411") and "PENTAD CONSISTENCY" in why
    assert "t2m" in why


def test_a_nonpositive_depth_is_counted_like_family7_counts_it():
    """GLORYS sometimes reports a FINITE depth <= 0: log_mld is NaN that day,
    but family 7's pentad mean counted it as a zero. check_bin rebuilds
    log_mld with those days counted when told how many there were."""
    H, W = 721, 1440
    st = np.full((5, H, W, 5), np.nan, np.float32)
    st[:, 500, 700, :] = [0.2, 1.0, 0.1, 0.1, 0.1]       # depth 10 m
    st[4, 500, 700, 1] = np.nan                           # day 5: depth 0
    P = np.full((H, W, 5), np.nan)
    P[500, 700] = [np.hypot(0.1, 0.1), np.log10(40.0 / 5), 0.1, 0.1, 0.1]
    z = np.where(np.isfinite(P), 0.0, np.nan)
    norm = np.tile([0.0, 1.0], (5, 1))
    bad = fd.check_bin("glorys025d", list(st.astype(np.float16)
                                          .astype(np.float32)), P, z, norm)
    assert not bad["ok"]                                  # log10(10) != log10(8)
    nz = np.zeros((H, W), np.int64)
    nz[500, 700] = 1
    good = fd.check_bin("glorys025d", list(st.astype(np.float16)
                                           .astype(np.float32)), P, z, norm,
                        mld_nonpos=nz)
    assert good["ok"], good["channels"]["log_mld"]


def test_family7s_float32_z_score_is_in_the_tolerance():
    """Family 7 z-scores in FLOAT32 ((X32 - mu32) / sd32, then float16), so a
    value whose exact z sits just past a float16 midpoint is stored as the FAR
    neighbour. The real case: OC-CCI 2013 bin 2324, one cell, one clear day —
    refused by 7.0e-9 before the tolerance carried the two float32 roundings."""
    v = np.float32(0.011219031)                     # the day's block mean
    mu, sd = np.float32(-0.77646887), np.float32(0.45660481)
    z16 = np.float16((v - mu) / sd)                 # family 7's arithmetic
    exact = np.float16((np.float64(v) - np.float64(mu)) / np.float64(sd))
    assert z16 != exact                             # the double rounding is real
    H, W = 721, 1440
    st = np.full((5, H, W, 2), np.nan, np.float32)
    st[1, 130, 511] = [v, 0.5]
    z = np.full((H, W, 2), np.nan)
    z[130, 511] = [float(z16), 0.0]
    norm = np.array([[mu, sd], [0.1, 1.0]], np.float64)
    P = z * norm[:, 1] + norm[:, 0]
    P[130, 511, 1] = 0.1
    r = fd.check_bin("occci025d", list(st.astype(np.float16)
                                       .astype(np.float32)), P, z, norm)
    assert r["ok"], r["channels"]["log_chl"]
    assert r["channels"]["log_chl"]["max_excess"] < 0
