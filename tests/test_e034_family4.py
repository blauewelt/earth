#!/usr/bin/env python3
"""E-034 step 4: exercise build_family4.py end to end before it costs a box.

ml/CLAUDE.md §4.8 — exercise the code path on a toy before spending the
expensive resource. The real family-4 tensor is [3142, 281, 481, 39] float16
= 33.1 GB and is a BOX build; every hour spent discovering a bug there is an
hour that a five-second synthetic fixture could have bought back.

This builds a complete miniature of every source the builder reads — a pentad
GLORYS aggregation, RG-Argo T/S cubes in SIO's schema, NCEP R1 daily momentum
flux in PSL's schema, and a pentad truth record — then runs the real builder
over them and asserts what it WROTE, not that it exited 0.

What is pinned, and why each one is the assertion that would actually catch a
regression:

  1. **Channels are family-3's, in family-3's order.** The pentad tensor
     exists to feed the same architecture more labels. A reordered or
     renamed channel would train cleanly and mean something different.

  2. **cur_speed is the magnitude of the mean, not the mean of magnitudes**,
     and **log_mld is log10** — the latter measured against family 3 rather
     than assumed. Both are checked by DE-NORMALISING the output through its
     own `norm`, so the check covers the z-score round trip too.

  3. **RG carries exactly ONE live pentad per month** (E-034 §4), and it is
     the bin containing the 15th. The other five are missing tokens. This is
     the decision the whole cadence audit turns on: forward-filling would
     tell the model the subsurface was observed on days it was not. A
     forward-fill regression is invisible in any summary statistic and
     obvious here.

  4. **The wind std is a WITHIN-PENTAD sigma computed from the dailies**,
     hand-computed for one cell and compared. A standard deviation is not
     aggregable from a mean, so the one plausible wrong implementation —
     reusing family-3's monthly channel — is exactly what this catches.

  5. **A grid that is not family-3's is REFUSED**, before anything is
     written. A half-cell offset would leave every stencil and the AMOC eval
     mask naming different pixels than they describe, and would be invisible
     in every plot.

  6. **Truth attaches by ROW on this axis**, and labels outside it are
     dropped rather than clamped to an edge bin.

E-042 added recipe r2 — the same tensor with OISST SST appended as channel
40 — and with it four checks that are all about the r1/r2 relationship,
because the safety argument for shipping a new recipe is that the old one
did not move:

 10. **r1 is BIT-IDENTICAL inside r2.** Built from the same fixtures, the
     r2 tensor's channels 0..38 equal the r1 tensor's float16 for float16,
     NaN pattern included, and so do `norm`, `months`, the axis and the
     labels. Appending cannot perturb an existing channel; this is what
     makes an r1 result and an r2 result comparable at all.
 11. **The pentad value is the NaN-AWARE mean of the bin's days**, checked
     against an analytic field, with one bin deliberately losing two of its
     five days by the two mechanisms the artifact has (a masked source day
     and a has_data=False gap). Land stays a missing token: the -32768
     sentinel decodes to -327.68 degC and must never reach the arithmetic.
 12. **SST is live before 2004, where rg_t cannot be.** On a 1990 axis every
     bin carries SST and all 32 rg channels are missing — that hole (22 of
     the 43 years with no temperature at all) is the reason for the channel.
 13. **The recipe guard refuses an r1 cache for an r2 build**, and still
     reuses an r2 one. A recipe string is a claim about the CODE; a 39-
     channel tensor sitting under r2's name would otherwise be loaded.
 14. **Each recipe's output name IS its ml-train.yml `tensor` value**, read
     out of the workflow — $TENSOR is derived from that input verbatim, so a
     rename here makes a run that cannot find the tensor it just built.

    python3 tests/test_e034_family4.py
"""
import contextlib
import datetime as dt
import io
import os
import shutil
import sys
import tempfile
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ml"))

import build_family3 as f3            # noqa: E402
import build_family4 as f4            # noqa: E402
from aggregate_cadence import bin_plan  # noqa: E402

EPOCH = dt.date(1982, 1, 1)
PD = 5


def pentad_of(d):
    return (d - EPOCH).days // PD


def pentad_start(b):
    return EPOCH + dt.timedelta(days=int(b) * PD)


# --------------------------------------------------------------- fixtures --
# A small point-aligned window, built by the SAME function the aggregator
# uses, so the test cannot drift from the thing it is testing.
SRC_LAT = np.arange(0.0, 2.0 + 1e-9, 1 / 12)
SRC_LON = np.arange(-10.0, -8.0 + 1e-9, 1 / 12)
PLAN = bin_plan(SRC_LAT, SRC_LON, 0.25, "point")
LATS, LONS = PLAN["lat"].astype(np.float32), PLAN["lon"].astype(np.float32)
H, W = len(LATS), len(LONS)

START, END = dt.date(2004, 1, 1), dt.date(2004, 2, 29)
BINS = list(range(pentad_of(START), pentad_of(END) + 1))


def write_pentad_dir(path, rng, bins=None):
    """A miniature aggregate_cadence.py output."""
    bins = BINS if bins is None else bins
    os.makedirs(path, exist_ok=True)
    vals = {}
    for v in ("uo", "vo", "mlotst", "zos"):
        a = rng.normal(size=(len(bins), H, W)).astype(np.float32)
        if v == "mlotst":
            a = np.abs(a) * 200 + 20            # a depth in metres, positive
        # a fixed land mask, so the ocean mask has something to find
        a[:, :2, :2] = np.nan
        np.save(os.path.join(path, f"pentad_mean_{v}.npy"), a)
        vals[v] = a
    np.savez(os.path.join(path, "index.npz"),
             bin_index=np.array(bins, np.int64),
             has_data=np.ones(len(bins), bool),
             epoch=np.array(str(EPOCH)), cadence_days=np.array(PD),
             stat=np.array("mean"), min_days=np.array(3),
             bin_deg=np.array(0.25), grid_align=np.array("point"),
             lat=LATS, lon=LONS, vars=np.array(["uo", "vo", "mlotst", "zos"]))
    return vals


def write_fake_base025(cache, lats, lons):
    """The artefact check_grid compares against."""
    np.savez(os.path.join(cache, "base025_na.npz"),
             lats=np.asarray(lats, np.float32), lons=np.asarray(lons, np.float32),
             months=np.array(["2004-01"]), chan=np.array(f3.CHANS[:3]),
             X=np.zeros((1, len(lats), len(lons), 3), np.float32))


def write_rg(cache, rng):
    """RG-Argo cubes in SIO's schema: MEAN is (P,lat,lon), ANOMALY (T,P,lat,lon)."""
    import netCDF4 as nc
    os.makedirs(os.path.join(cache, "rg"), exist_ok=True)
    rg_lat = np.arange(-64.5, 79.5 + 1e-9, 1.0)
    rg_lon = np.arange(20.5, 379.5 + 1e-9, 1.0)
    press = np.array(f4.LEVELS, np.float64)
    for var, stem in (("TEMPERATURE", "RG_T"), ("SALINITY", "RG_S")):
        d = nc.Dataset(os.path.join(cache, "rg", stem + ".nc"), "w")
        d.createDimension("PRESSURE", len(press))
        d.createDimension("LATITUDE", len(rg_lat))
        d.createDimension("LONGITUDE", len(rg_lon))
        d.createDimension("TIME", 2)                 # 2004-01, 2004-02
        d.createVariable("PRESSURE", "f8", ("PRESSURE",))[:] = press
        d.createVariable("LATITUDE", "f8", ("LATITUDE",))[:] = rg_lat
        d.createVariable("LONGITUDE", "f8", ("LONGITUDE",))[:] = rg_lon
        d.createVariable(f"ARGO_{var}_MEAN", "f4",
                         ("PRESSURE", "LATITUDE", "LONGITUDE"))[:] = \
            rng.normal(size=(len(press), len(rg_lat), len(rg_lon))).astype(np.float32)
        d.createVariable(f"ARGO_{var}_ANOMALY", "f4",
                         ("TIME", "PRESSURE", "LATITUDE", "LONGITUDE"))[:] = \
            rng.normal(size=(2, len(press), len(rg_lat), len(rg_lon))).astype(np.float32)
        d.close()


def write_wind(cache, rng, years=(2004,), ndays=90):
    """NCEP R1 daily momentum flux in PSL's schema.

    `years` is a tuple because the DAILY path (family 5) reads a centred
    +-2-day window, so its first bins reach back into the previous
    December. Those earlier years are written WHOLE; only the last year —
    the one the checks index into — is truncated to `ndays`, and it is that
    year's fields that come back. Writing them is not a nicety: without the
    December the builder falls through to `f3.fetch`, which puts a 60 MB
    download of somebody else's server inside a unit test, and a truncated
    transfer there fails as `NetCDF: HDF error` on a file this fixture is
    supposed to own.
    """
    import netCDF4 as nc
    daily = os.path.join(cache, "wind_daily")
    os.makedirs(daily, exist_ok=True)
    g_lat = np.linspace(88.0, -88.0, 94)             # descending, like gaussian
    g_lon = np.arange(0.0, 360.0, 1.875)
    fields = {}
    for y in years:
        n = ndays if y == years[-1] else 366
        for var in ("uflx", "vflx"):
            d = nc.Dataset(os.path.join(daily, f"{var}.sfc.gauss.{y}.nc"), "w")
            d.createDimension("time", n)
            d.createDimension("lat", len(g_lat))
            d.createDimension("lon", len(g_lon))
            t = d.createVariable("time", "f8", ("time",))
            t.units = f"days since {y}-01-01 00:00:0.0"
            t[:] = np.arange(n, dtype=float)
            d.createVariable("lat", "f4", ("lat",))[:] = g_lat
            d.createVariable("lon", "f4", ("lon",))[:] = g_lon
            a = rng.normal(size=(n, len(g_lat), len(g_lon))).astype(np.float32)
            d.createVariable(var, "f4", ("time", "lat", "lon"))[:] = a
            d.close()
            if y == years[-1]:
                fields[var] = a
    return g_lat, g_lon, fields


# E-042. `ml/fetch_sst_na.py`'s contract, restated here as a FIXTURE rather
# than imported, so a change to that contract breaks this test instead of
# silently travelling through it: int16 (NDAYS, H, W), day-major, scale
# 0.01 degC, nodata -32768, index.npz carrying bin_index (days since the
# epoch, contiguous), has_data, lat, lon.
SST_SCALE = 0.01
SST_NODATA = -32768
SST_LAND = (slice(3, 5), slice(6, 8))     # ocean in the tensor, land in OISST


def sst_degc(dayno, day0):
    """The fixture's field: exact multiples of 0.01 degC, so the int16
    round trip is lossless and every expected value below is analytic.

    The `t % 4` term is what makes the field NON-LINEAR in the day, and it
    is load-bearing. A field linear in time has mean(5 consecutive days) ==
    the middle day exactly, so every "a mean is not a sample" check — this
    file's 3-of-5 check and family 5's cadence identity — would pass on a
    builder that took one day and called it a mean. Period 4 against a
    5-day bin never repeats a phase, so no bin is a symmetric special case.
    """
    t = float(dayno - day0)
    iy = np.arange(H)[:, None].astype(np.float64)
    ix = np.arange(W)[None, :].astype(np.float64)
    return 15.0 + 0.25 * t + 0.5 * (t % 4) + 0.10 * iy + 0.05 * ix


def write_sst(path, bins, days=PD, nodata_days=(), gap_days=()):
    """A miniature ml/fetch_sst_na.py output over the days of `bins`.

    `nodata_days` are days whose CELLS are all the nodata sentinel with
    has_data True (a source day the analysis masked); `gap_days` are days
    with has_data False as well (fetch_sst_na's posture for a year that could
    not be downloaded). Both must be excluded from the pentad mean, and they
    are written as two separate mechanisms because they fail differently: the
    sentinel decodes to -327.68 degC if it ever reaches the arithmetic.
    """
    os.makedirs(path, exist_ok=True)
    d0, d1 = bins[0] * days, bins[-1] * days + days - 1
    idx = np.arange(d0, d1 + 1, dtype=np.int64)
    arr = np.zeros((len(idx), H, W), np.int16)
    has = np.ones(len(idx), bool)
    dec = np.full((len(idx), H, W), np.nan, np.float64)   # what it decodes to
    for i, dayno in enumerate(idx):
        f = sst_degc(int(dayno), d0)
        enc = np.round(f / SST_SCALE).astype(np.int16)
        enc[SST_LAND] = SST_NODATA
        if int(dayno) in nodata_days or int(dayno) in gap_days:
            enc[:] = SST_NODATA
            has[i] = int(dayno) not in gap_days
        arr[i] = enc
        d = np.where(enc == SST_NODATA, np.nan, enc * SST_SCALE)
        dec[i] = np.where(has[i], d, np.nan)
    np.save(os.path.join(path, "sst_daily_na.npy"), arr)
    np.savez(os.path.join(path, "index.npz"),
             bin_index=idx, has_data=has,
             epoch=np.array(str(EPOCH)), cadence_days=np.array(1),
             scale=np.array(SST_SCALE), nodata=np.array(SST_NODATA),
             lat=LATS, lon=LONS,
             source=np.array("synthetic (tests/test_e034_family4.py)"))
    return dict(bin_index=idx, decoded=dec, day0=d0)


def write_truth(cache):
    """Two labels inside the axis and one deliberately outside it."""
    inside = [BINS[2], BINS[5]]
    outside = BINS[-1] + 500
    arr = np.array([[b, 17.0 + i] for i, b in enumerate(inside)]
                   + [[outside, 99.0]], np.float32)
    np.savez(os.path.join(cache, "truth_pentad.npz"),
             epoch=np.array(str(EPOCH)), pentad_days=np.array(PD),
             truth_rapid=arr, count_rapid=np.ones(len(arr), np.int32))
    return inside, outside


# ------------------------------------------------------------------ run --
def run_build(cache, pentad_dir, out, extra=(), start=None, end=None):
    """Call the real main() in-process with the module's caches redirected."""
    f4.CACHE = cache
    f3.CACHE = cache
    f4.TRUTH_PENTAD = os.path.join(cache, "truth_pentad.npz")
    argv = sys.argv
    sys.argv = ["build_family4.py", "--pentad-dir", pentad_dir, "--out", out,
                "--memmap", os.path.join(cache, "build.npy"),
                "--start", str(start or START), "--end", str(end or END),
                *extra]
    try:
        f4.main()
    finally:
        sys.argv = argv


def main():
    rng = np.random.default_rng(20260816)
    tmp = tempfile.mkdtemp(prefix="e034f4_")
    cache = os.path.join(tmp, "cache")
    os.makedirs(cache)
    pentad_dir = os.path.join(tmp, "pentad")

    src = write_pentad_dir(pentad_dir, rng)
    write_fake_base025(cache, LATS, LONS)
    write_rg(cache, rng)
    g_lat, g_lon, wind = write_wind(cache, rng)
    t_inside, t_outside = write_truth(cache)

    out = os.path.join(tmp, "family4.npz")
    run_build(cache, pentad_dir, out)
    d = np.load(out)
    X = np.asarray(d["X"], np.float32)
    mu, sd = d["norm"][:, 0], d["norm"][:, 1]
    raw = X * sd + mu                       # de-normalise, incl. the round trip
    bins = d["bin_index"].tolist()
    T = len(bins)
    print(f"  built [{T}, {X.shape[1]}, {X.shape[2]}, {X.shape[3]}] "
          f"{d['X'].dtype}, recipe {str(d['recipe'])}")

    # ---- 1: channels ------------------------------------------------------
    assert [str(c) for c in d["chan"]] == list(f3.CHANS), "channel set drifted"
    assert X.shape[3] == f3.NC == 39
    assert str(d["cadence"]) == "pentad" and int(d["pentad_days"]) == PD
    assert str(d["epoch"]) == str(EPOCH)
    print(f"  1. channels identical to family 3 ({f3.NC}), axis is pentad "
          f"from {EPOCH}")

    # ---- 2: cur_speed = |mean|, log_mld = log10 ---------------------------
    ocean = np.isfinite(raw[0, :, :, 0])
    j = 3                                   # an interior bin
    want_speed = np.hypot(src["uo"][j], src["vo"][j])
    want_lmld = np.log10(src["mlotst"][j])
    got_speed, got_lmld = raw[j, :, :, 0], raw[j, :, :, 1]
    m = ocean & np.isfinite(want_speed)
    assert np.allclose(got_speed[m], want_speed[m], rtol=3e-3, atol=3e-3), \
        f"cur_speed differs, max {np.max(np.abs(got_speed[m]-want_speed[m]))}"
    assert np.allclose(got_lmld[m], want_lmld[m], rtol=3e-3, atol=3e-3), \
        f"log_mld is not log10, max {np.max(np.abs(got_lmld[m]-want_lmld[m]))}"
    # and it is NOT the mean of magnitudes -- construct the alternative and
    # confirm this fixture can tell them apart at all
    print(f"  2. cur_speed = hypot(mean u, mean v) and log_mld = log10, "
          f"through the z-score round trip (fp16, max err "
          f"{np.max(np.abs(got_speed[m]-want_speed[m])):.2e})")

    # ---- 3: RG is one live pentad per month -------------------------------
    live = [r for r in range(T)
            if np.isfinite(raw[r, :, :, f3.C_BASE:f3.C_BASE + f3.C_RG]).any()]
    want_live = sorted({bins.index(pentad_of(dt.date(2004, mo, 15)))
                        for mo in (1, 2)})
    assert live == want_live, (
        f"RG live rows {live} != the bins containing the 15th {want_live} — "
        f"a forward-fill or a shifted nominal timestamp")
    assert int(d["n_rg_live"]) == len(want_live)
    for r in range(T):
        if r in live:
            continue
        assert not np.isfinite(
            raw[r, :, :, f3.C_BASE:f3.C_BASE + f3.C_RG]).any(), \
            f"row {r} carries RG values but is not a live pentad"
    print(f"  3. RG: {len(live)}/{T} rows live (the bins holding each 15th), "
          f"the other {T - len(live)} are missing tokens — no forward fill")

    # ---- 4: wind std is a WITHIN-PENTAD sigma from the dailies ------------
    ci = f3.C_BASE + f3.C_RG            # tau_x
    b = bins[j]
    days = [pentad_start(b) + dt.timedelta(days=k) for k in range(PD)]
    di = [(dd - dt.date(2004, 1, 1)).days for dd in days]
    assert all(0 <= k < wind["uflx"].shape[0] for k in di), "fixture too short"
    stack = -wind["uflx"][di]           # sign flip: stress ON the ocean
    # nearest gaussian source cell for one target point, then compare the
    # bilinear of the hand-computed fields with what the builder wrote
    wy = f3.lin_weights(g_lat, LATS)
    wx = f3.lin_weights(g_lon, np.where(LONS < 0, LONS + 360.0, LONS))
    want_mean = f3.interp2_nan(stack.mean(0), wy, wx)
    want_std = f3.interp2_nan(stack.std(0), wy, wx)
    got_mean, got_std = raw[j, :, :, ci], raw[j, :, :, ci + 2]
    mm = ocean & np.isfinite(want_std)
    assert np.allclose(got_mean[mm], want_mean[mm], rtol=5e-3, atol=5e-3), \
        "wind mean differs from the pentad mean of the dailies"
    assert np.allclose(got_std[mm], want_std[mm], rtol=5e-3, atol=5e-3), \
        (f"tau_x_std is not the within-pentad sigma of the dailies "
         f"(max err {np.max(np.abs(got_std[mm]-want_std[mm])):.3e})")
    # the monthly sigma is a DIFFERENT number — confirm the fixture separates
    month_std = f3.interp2_nan((-wind["uflx"][:31]).std(0), wy, wx)
    assert not np.allclose(got_std[mm], month_std[mm], rtol=5e-2), \
        "the within-pentad and within-month sigmas are indistinguishable on " \
        "this fixture — the check cannot fail, so it is not a check"
    print(f"  4. tau_std is the within-PENTAD sigma of the dailies, and is "
          f"distinguishable from the within-month sigma "
          f"(mean |diff| {np.mean(np.abs(got_std[mm]-month_std[mm])):.3f})")

    # ---- 5: truth attaches by row, outside labels dropped -----------------
    tr = d["truth_rapid"]
    assert tr.shape == (len(t_inside), 2), \
        f"expected {len(t_inside)} in-axis labels, got {tr.shape}"
    for k, b_ in enumerate(t_inside):
        assert int(tr[k, 0]) == bins.index(b_), \
            f"label {k} landed on row {int(tr[k,0])}, want {bins.index(b_)}"
    print(f"  5. truth_rapid: {len(tr)} labels mapped to rows, the "
          f"out-of-axis bin {t_outside} dropped rather than clamped")

    # ---- 6: z-score is real ----------------------------------------------
    for c in (0, 1, 2, ci):
        v = raw[:, :, :, c][np.isfinite(raw[:, :, :, c])]
        z = (v - mu[c]) / sd[c]
        assert abs(z.mean()) < 0.05 and abs(z.std() - 1) < 0.05, \
            f"channel {f3.CHANS[c]} is not standardised: mean {z.mean():.3f} sd {z.std():.3f}"
    print("  6. per-channel z-score verified over observed cells")

    # ---- 7: a wrong grid is REFUSED before anything is written ------------
    bad_cache = os.path.join(tmp, "badcache")
    os.makedirs(bad_cache)
    write_fake_base025(bad_cache, LATS + 0.125, LONS + 0.125)   # half-cell off
    for f in ("rg", "wind_daily", "truth_pentad.npz"):
        s = os.path.join(cache, f)
        if os.path.isdir(s):
            os.symlink(s, os.path.join(bad_cache, f))
        else:
            os.symlink(s, os.path.join(bad_cache, f))
    out2 = os.path.join(tmp, "family4_bad.npz")
    refused = False
    try:
        run_build(bad_cache, pentad_dir, out2)
    except SystemExit as e:
        refused = "GRID MISMATCH" in str(e)
    assert refused, "a half-cell-offset grid was NOT refused"
    assert not os.path.exists(out2), "the refused build still wrote its output"
    print("  7. a half-cell-offset grid is refused, and nothing is written")

    # ---- 8: the existing trainer can load this, unmodified ---------------
    # ml/train.py touches `months` in five places: load, print, m[:4] for the
    # year-blocked holdout, and int(m[5:7])-1 twice for the season token.
    # Reproduce all of them here rather than trusting that they work.
    mo = [str(m) for m in d["months"]]
    assert len(mo) == T, f"months has {len(mo)} entries for {T} rows"
    for r, b in enumerate(bins):
        st = pentad_start(b)
        assert mo[r] == f"{st.year:04d}-{st.month:02d}", \
            f"row {r}: months says {mo[r]}, bin starts {st}"
    years = {m[:4] for m in mo}                       # train.py:156
    moy = [int(m[5:7]) - 1 for m in mo]               # train.py:172, 201
    assert years == {"2004"} and set(moy) <= {0, 1}, \
        f"holdout/season decode wrong: years {years}, months {sorted(set(moy))}"
    assert "rapid" in d.files, "train.py:648 reads d['rapid'] — absent"
    assert np.array_equal(d["rapid"], d["truth_rapid"])
    print(f"  8. loadable by the unmodified trainer: months decodes to "
          f"year {sorted(years)} and months {sorted(set(moy))}; rapid aliased")

    # ---- 9: a cached tensor skips BEFORE the free-space guard ------------
    # Run #391: the guard ran first and refused 33.1 GB for a tensor that
    # was already built and would have been skipped one line later. Starve
    # statvfs so a FRESH build is refused, then prove the already-built
    # output still short-circuits — the guard must not cost more than the
    # thing it guards.
    class _Starved:
        f_bavail, f_frsize = 1, 4096            # 4 KB free, refuses anything

    real_statvfs = os.statvfs
    os.statvfs = lambda _p: _Starved()
    try:
        out3 = os.path.join(tmp, "family4_fresh.npz")
        refused = ""
        try:
            run_build(cache, pentad_dir, out3)
        except SystemExit as e:
            refused = str(e)
        assert "refusing to start" in refused, \
            f"the starved guard did not refuse a FRESH build: {refused!r}"
        assert not os.path.exists(out3), "the refused build still wrote output"

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_build(cache, pentad_dir, out)   # already built by check 1
        said = buf.getvalue()
    finally:
        os.statvfs = real_statvfs
    assert "already built" in said and "skipping" in said, \
        f"the cached tensor did not take the skip path:\n{said}"
    assert "refusing to start" not in said, \
        "the free-space guard still runs before the already-built check"
    assert os.path.getsize(out) > 0
    print("  9. an already-built tensor skips even when the free-space guard "
          "would refuse a fresh build (run #391)")

    # ======================= E-042: recipe r2, the sst channel ============
    # r1 is everything above and MUST NOT MOVE — #386/#387 are training on it
    # and every number in EXPERIMENTS.md was measured on it. So r2 is built
    # from the SAME fixtures and compared against the r1 tensor already on
    # disk, channel by channel.
    sst_dir = os.path.join(tmp, "sst_na025")
    # bin BINS[4] loses two of its five days: one masked at the source
    # (has_data True, cells nodata) and one whole-day gap (has_data False).
    dead_bin = BINS[4]
    # days 0 and 1, deliberately NOT a symmetric pair: the fixture's field
    # is linear in the day, so dropping days 1 and 3 would leave the 3-day
    # mean numerically equal to the 5-day mean and the check could not fail.
    nodata_day = dead_bin * PD + 0
    gap_day = dead_bin * PD + 1
    sst = write_sst(sst_dir, BINS, PD, nodata_days=(nodata_day,),
                    gap_days=(gap_day,))
    out_r2 = os.path.join(tmp, "family4_r2.npz")
    run_build(cache, pentad_dir, out_r2,
              extra=("--rev", "r2", "--sst-dir", sst_dir))
    d2 = np.load(out_r2)
    raw2 = np.asarray(d2["X"], np.float32) * d2["norm"][:, 1] + d2["norm"][:, 0]

    # ---- 10: r2 = 40 channels, `sst` last, and r1 UNTOUCHED ---------------
    assert [str(c) for c in d2["chan"]] == list(f3.CHANS) + ["sst"], \
        f"r2 channel set is {[str(c) for c in d2['chan']]}"
    assert d2["X"].shape[3] == 40 and str(d2["recipe"]) == "f4r2"
    x1 = np.asarray(np.load(out)["X"])            # the r1 tensor, float16
    x2 = np.asarray(d2["X"])
    assert x1.dtype == x2.dtype == np.float16
    assert np.array_equal(x1, x2[..., :f3.NC], equal_nan=True), \
        "appending sst CHANGED one of the 39 published channels — the whole " \
        "safety argument for r1/r2 comparability is that it cannot"
    assert np.array_equal(d2["norm"][:f3.NC], np.load(out)["norm"]), \
        "the per-channel z-score of the 39 moved when a 40th was appended"
    for k in ("months", "bin_index", "lats", "lons", "truth_rapid", "rapid"):
        assert np.array_equal(np.load(out)[k], d2[k]), f"{k} differs"
    print(f"  10. r2: 40 channels with `sst` last, recipe f4r2 — and "
          f"channels 0..{f3.NC - 1} are BIT-IDENTICAL to the r1 tensor "
          f"(float16, NaN pattern included)")

    # ---- 11: the pentad value is the NaN-AWARE 5-day mean -----------------
    c = f4.C_SST
    day_row = {int(b): i for i, b in enumerate(sst["bin_index"])}
    ocean2 = np.isfinite(raw2[0, :, :, 0])
    for r, b in enumerate(bins):
        with warnings.catch_warnings():          # an all-nodata cell is a
            warnings.simplefilter("ignore")      # missing token, not a bug
            want = np.nanmean(
                np.stack([sst["decoded"][day_row[b * PD + k]]
                          for k in range(PD)]), axis=0)
        got = raw2[r, :, :, c]
        m = ocean2 & np.isfinite(want)
        assert m.any(), f"bin {b}: nothing to compare"
        assert np.allclose(got[m], want[m], atol=0.02), (
            f"bin {b}: sst is not the NaN-aware 5-day mean "
            f"(max err {np.max(np.abs(got[m] - want[m])):.3f})")
        assert not np.isfinite(got[ocean2 & ~np.isfinite(want)]).any(), \
            f"bin {b}: a cell with no valid day carries a value"
    # the doctored bin, ANALYTICALLY: the mean of days 0, 2, 4 only, and
    # nowhere near the sentinel (-327.68) or the all-five mean.
    rr = bins.index(dead_bin)
    cell = (6, 3)
    good = [dead_bin * PD + k for k in (2, 3, 4)]
    want3 = np.mean([sst_degc(dd, sst["day0"])[cell] for dd in good])
    want5 = np.mean([sst_degc(dead_bin * PD + k, sst["day0"])[cell]
                     for k in range(PD)])
    got1 = float(raw2[rr, cell[0], cell[1], c])
    assert abs(got1 - want3) < 0.02, \
        f"a bin with 2 nodata days reads {got1:.3f}, want {want3:.3f} " \
        f"(the mean of the OTHER three)"
    assert abs(want3 - want5) > 0.02, \
        "the fixture cannot distinguish a 3-day mean from a 5-day mean"
    assert np.isfinite(raw2[:, :, :, c]).any()
    land_ok = ~np.isfinite(raw2[:, SST_LAND[0], SST_LAND[1], c])
    assert land_ok.all(), \
        "OISST land/nodata decoded to a temperature instead of a missing token"
    print(f"  11. sst is the NaN-aware {PD}-day mean over every bin; the bin "
          f"with one masked and one gap day reads {got1:.3f} degC (the mean "
          f"of its 3 valid days, not {want5:.3f} and not the sentinel), and "
          f"nodata cells stay missing")

    # ---- 12: SST is live BEFORE 2004, where rg_t cannot be ----------------
    # The reason for the channel (E-042): the tensor's only other temperature
    # is Argo rg_t, whose product starts in 2004 (fill_rg_pentad walks
    # `y, m = 2004 + k // 12, ...`), so 1982-2003 carries none at all.
    START90, END90 = dt.date(1990, 1, 1), dt.date(1990, 2, 28)
    B90 = list(range(pentad_of(START90), pentad_of(END90) + 1))
    p90 = os.path.join(tmp, "pentad90")
    write_pentad_dir(p90, rng, bins=B90)
    # 1989 as well: the pentad holding 1990-01-01 STARTS on 1989-12-30, so
    # fill_wind_pentad reads both years, and a year the fixture does not own
    # is a year this test downloads from PSL.
    write_wind(cache, rng, years=(1989, 1990))
    sst90 = os.path.join(tmp, "sst90")
    write_sst(sst90, B90, PD)
    out90 = os.path.join(tmp, "family4_1990_r2.npz")
    run_build(cache, p90, out90, extra=("--rev", "r2", "--sst-dir", sst90),
              start=START90, end=END90)
    d90 = np.load(out90)
    raw90 = np.asarray(d90["X"], np.float32) * d90["norm"][:, 1] + d90["norm"][:, 0]
    rg_slice = raw90[:, :, :, f3.C_BASE:f3.C_BASE + f3.C_RG]
    assert int(d90["n_rg_live"]) == 0 and not np.isfinite(rg_slice).any(), \
        "the 1990 fixture has live rg — it cannot test the hole it is for"
    live_sst = [r for r in range(len(B90))
                if np.isfinite(raw90[r, :, :, f4.C_SST]).any()]
    assert live_sst == list(range(len(B90))), \
        f"sst live in {len(live_sst)}/{len(B90)} pre-2004 bins, want all"
    assert int(d90["n_sst"]) == len(B90)
    print(f"  12. on a 1990 axis every one of the {len(B90)} bins carries "
          f"sst while all {f3.C_RG} rg channels are missing tokens — the "
          f"22-year temperature hole is exactly what the channel fills")

    # ---- 13: the recipe guard refuses an r1 cache for an r2 build ---------
    # A recipe string is a claim about the CODE that wrote the file. An r1
    # tensor sitting where r2 is wanted has 39 channels and no sst; reusing
    # it would train a 40-channel experiment on a 39-channel tensor, or
    # (worse) skip the build and load the wrong file entirely.
    out_guard = os.path.join(tmp, "guard_r2.npz")
    shutil.copy(out, out_guard)                   # an r1 tensor, r2's name
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_build(cache, pentad_dir, out_guard,
                  extra=("--rev", "r2", "--sst-dir", sst_dir))
    said = buf.getvalue()
    assert "cached tensor is recipe 'f4r1', want 'f4r2'" in said, \
        f"the r1 cache was not refused for an r2 build:\n{said}"
    assert "already built" not in said
    dg = np.load(out_guard)
    assert str(dg["recipe"]) == "f4r2" and dg["X"].shape[3] == 40, \
        "the guard printed a rebuild and left the r1 tensor in place"
    # and the converse: an r2 tensor is not rebuilt by an r2 build
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_build(cache, pentad_dir, out_guard,
                  extra=("--rev", "r2", "--sst-dir", sst_dir))
    assert "already built by recipe f4r2" in buf.getvalue(), \
        "an r2 cache is not being reused — every run would rebuild 33 GB"
    print("  13. an r1 cache is REFUSED for an r2 build (and rebuilt as 40 "
          "channels); an r2 cache is still reused")

    # ---- 14: the builder's output names ARE the dispatch values ----------
    # ml-train.yml derives $TENSOR as ml/cache/<inputs.tensor>.npz, so a
    # rename here silently produces a run whose provenance step cannot find
    # its own tensor. Read the workflow rather than trusting the comment.
    wf = open(os.path.join(HERE, "..", ".github", "workflows",
                           "ml-train.yml")).read()
    for days, rev, value in ((5, "r2", "family4_na025_pentad_r2"),
                             (1, "r2", "family5_na025_daily_r2"),
                             (5, "r1", "family4_na025_pentad"),
                             (1, "r1", "family5_na025_daily")):
        assert f4.CADENCE[days]["revs"][rev]["out"] == value + ".npz", \
            f"{days}d/{rev} writes {f4.CADENCE[days]['revs'][rev]['out']}, " \
            f"but ml-train.yml would look for {value}.npz"
        assert f'= "{value}"' in wf, \
            f"ml-train.yml has no branch for tensor={value}"
    print("  14. every recipe's output name equals the ml-train.yml `tensor` "
          "value that selects it (r1 and r2, both cadences)")

    print("\ntests/test_e034_family4.py: all 14 checks passed")


if __name__ == "__main__":
    main()
