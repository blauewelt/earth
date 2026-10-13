"""The `cygnss_mss` smoke: a synthetic CYGNSS archive — L2 V3.2 day files and
the L3 V3.2 hourly-grid day files made FROM them — with the real variable
names, dimension names, dtypes and fill values (CMR UMM-Var, measured
2026-10-13; data dictionaries 148-0348-13 and 148-0351-5).

Underscore-prefixed, so the adapter registry skips it.

THE SYNTHETIC PRODUCER. `l3_from_l2` is written independently of the adapter
(np.add.at over explicit cell indices, not `np.bincount`), and it is the
DOCUMENTED L3 definition: per (hour, 0.2-degree cell), the inverse-variance
mean of the L2 samples whose `mss_sample_flags` bit 1 is clear, its
uncertainty 1/sqrt(sum w), and the sample count; the wind the same way with
`fds_sample_flags` bit 1. `variant` makes a DIFFERENT producer, so a test can
show F1 failing with numbers:

  consistent      the definition above
  flags_ignored   counts and means include the fatally flagged samples
  unweighted      the mean is a plain mean (uncertainty unchanged)
  unc_sqrt2       the uncertainty is sqrt(2) x 1/sqrt(sum w)

The STORE truth (`make_smoke_sources`) is computed from the L3 arrays as
written, by a third, loop-over-hours implementation, so the smoke's exact
frame comparison tests the adapter's reading, combination, rotation and
bounds rather than replaying the adapter against itself.
"""
import datetime as dt
import os

import numpy as np

import build_family10_stores as f10b
from family1 import sharded as sh

SMOKE_H, SMOKE_W = 40, 180              # 2-degree cells: the same rules
SMOKE_ABSENT_DAY = dt.date(2019, 1, 3)  # the archive holds no files for it
SMOKE_OOB_DAY = dt.date(2019, 1, 2)     # one cell's wind is 150 m/s
TIME_FRAC = 0.999261459                 # the L2 epoch's odd fraction (UMM-Var)
FILL_F, FILL_I = -9999.0, -99


def l3_name(d):
    s = d.strftime("%Y%m%d")
    return f"cyg.ddmi.s{s}-003000-e{s}-233000.l3.grid-wind.a32.d33"


def l2_name(d):
    s = d.strftime("%Y%m%d")
    return f"cyg.ddmi.s{s}-000000-e{s}-235959.l2.wind-mss.a32.d33"


# ============================================================ L2 samples ==
def synth_l2(d, seed, n_tracks=400, per_track=25):
    """Deterministic L2 samples for one day: specular TRACKS (1 s apart, so
    several samples share an hour-cell), some outside +-40, flagged ones,
    non-fatal ones and fill."""
    rng = np.random.default_rng(seed + d.toordinal())
    n = n_tracks * per_track
    t0 = rng.uniform(0, 86400 - per_track, n_tracks)
    lat0 = rng.uniform(-41.0, 41.0, n_tracks)
    lon0 = rng.uniform(0.0, 360.0, n_tracks)
    k = np.tile(np.arange(per_track), n_tracks)
    sod = np.repeat(t0, per_track) + k * 1.0
    lat = np.repeat(lat0, per_track) + k * 0.05
    lon = (np.repeat(lon0, per_track) + k * 0.05) % 360.0
    mss = 0.012 + 0.02 * np.cos(np.radians(lat)) ** 2 + \
        0.005 * rng.random(n)
    unc = mss * rng.uniform(0.012, 0.025, n)
    flags = np.zeros(n, np.int32)
    r = rng.random(n)
    fatal = r < 0.06
    flags[fatal] |= 1
    lo = fatal & (r < 0.02)
    flags[lo] |= 2                                 # fatal_low_mss
    mss[lo] = -0.004
    hi = fatal & (r >= 0.02) & (r < 0.03)
    flags[hi] |= 4                                 # fatal_high_mss
    mss[hi] = 1.3
    inc = fatal & (r >= 0.03)
    flags[inc] |= 8                                # fatal_mss_inc_angle
    nonfatal = (r >= 0.06) & (r < 0.11)
    flags[nonfatal] |= 32                          # preliminary GPS antenna
    fill = (r >= 0.11) & (r < 0.12)
    mss[fill] = FILL_F
    unc[fill] = FILL_F
    flags[fill] = -9999
    ws = 3.0 + 10.0 * rng.random(n)
    wsu = 1.0 + rng.random(n)
    fds = np.zeros(n, np.int32)
    fds[rng.random(n) < 0.04] |= 1
    return {"sod": sod, "lat": lat.astype(np.float32),
            "lon": lon.astype(np.float32),
            "mean_square_slope": mss.astype(np.float32),
            "mean_square_slope_uncertainty": unc.astype(np.float32),
            "mss_sample_flags": flags,
            "wind_speed": ws.astype(np.float32),
            "wind_speed_uncertainty": wsu.astype(np.float32),
            "fds_sample_flags": fds}


def write_l2(path, d, s):
    import netCDF4
    n = len(s["sod"])
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    start = f"{d.isoformat()} 00:00:00.{int(round(TIME_FRAC * 1e9)):09d}"
    ds.time_coverage_start = f"{d.isoformat()}T00:00:00.999261459Z"
    ds.l2_algorithm_version = "3.2 (synthetic smoke)"
    ds.createDimension("sample", n)
    ds.createDimension("ddm", 5)
    v = ds.createVariable("sample", "i4", ("sample",))
    v[:] = np.arange(n)
    st = ds.createVariable("sample_time", "f8", ("sample",))
    st.units = f"seconds since {start}"
    st[:] = s["sod"] - TIME_FRAC
    for name in ("lat", "lon", "mean_square_slope",
                 "mean_square_slope_uncertainty", "wind_speed",
                 "wind_speed_uncertainty"):
        x = ds.createVariable(name, "f4", ("sample",),
                              fill_value=np.float32(FILL_F))
        x.set_auto_mask(False)
        x[:] = s[name]
    for name in ("mss_sample_flags", "fds_sample_flags"):
        x = ds.createVariable(name, "i4", ("sample",),
                              fill_value=np.int32(-9999))
        x.set_auto_mask(False)
        x[:] = s[name]
    x = ds.createVariable("spacecraft_num", "i1", ("sample",),
                          fill_value=np.int8(FILL_I))
    x[:] = (np.arange(n) % 8 + 1).astype(np.int8)
    ds.close()


# ========================================================= the producer ===
def l3_from_l2(s, H, W, variant="consistent"):
    """The documented L3 binning, written independently of the adapter.

    -> {mss, mss_u, mss_n, ws, ws_u, ws_n}: float32 / int32 [24, H, W],
    fill where a cell-hour has no sample."""
    d = 80.0 / H
    hr = np.floor(s["sod"] / 3600.0).astype(np.int64)
    lat = s["lat"].astype(np.float64)
    lon = s["lon"].astype(np.float64) % 360.0
    i = np.floor((lat + 40.0) / d).astype(np.int64)
    j = np.floor(lon / (360.0 / W)).astype(np.int64)
    inside = (hr >= 0) & (hr < 24) & (i >= 0) & (i < H) & (j >= 0) & (j < W)
    out = {}
    for tag, v, u, fl, keep_fatal in (
            ("mss", s["mean_square_slope"], s["mean_square_slope_uncertainty"],
             s["mss_sample_flags"], variant == "flags_ignored"),
            ("ws", s["wind_speed"], s["wind_speed_uncertainty"],
             s["fds_sample_flags"], False)):
        v = v.astype(np.float64)
        u = u.astype(np.float64)
        ok = inside & (v != FILL_F) & (u != FILL_F) & (u > 0) & (fl != -9999)
        if not keep_fatal:
            ok &= (fl & 1) == 0
        sw = np.zeros((24, H, W))
        swv = np.zeros((24, H, W))
        sv = np.zeros((24, H, W))
        cnt = np.zeros((24, H, W), np.int64)
        idx = (hr[ok], i[ok], j[ok])
        w = 1.0 / u[ok] ** 2
        np.add.at(sw, idx, w)
        np.add.at(swv, idx, w * v[ok])
        np.add.at(sv, idx, v[ok])
        np.add.at(cnt, idx, 1)
        has = cnt > 0
        mean = np.full((24, H, W), FILL_F)
        unc = np.full((24, H, W), FILL_F)
        if tag == "mss" and variant == "unweighted":
            mean[has] = sv[has] / cnt[has]
        else:
            mean[has] = swv[has] / sw[has]
        unc[has] = 1.0 / np.sqrt(sw[has])
        if tag == "mss" and variant == "unc_sqrt2":
            unc[has] *= np.sqrt(2.0)
        n = np.where(has, cnt, FILL_I).astype(np.int32)
        out[tag] = mean.astype(np.float32)
        out[f"{tag}_u"] = unc.astype(np.float32)
        out[f"{tag}_n"] = n
    return out


def write_l3(path, d, a, H, W, lat_descending=False):
    import netCDF4
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("time", 24)
    ds.createDimension("lat", H)
    ds.createDimension("lon", W)
    dd = 80.0 / H
    lat = -40.0 + (np.arange(H) + 0.5) * dd
    sl = slice(None, None, -1) if lat_descending else slice(None)
    t = ds.createVariable("time", "f8", ("time",))
    t.units = f"hours since {d.isoformat()} 00:30:00"
    t[:] = np.arange(24)
    e = ds.createVariable("epoch_time", "f8", ("time",))
    e.units = "hours since 2018-08-01 00:30:00.000000000"
    e[:] = (d - dt.date(2018, 8, 1)).days * 24 + np.arange(24)
    vl = ds.createVariable("lat", "f4", ("lat",))
    vl[:] = np.ascontiguousarray(lat[sl])
    vo = ds.createVariable("lon", "f4", ("lon",))
    vo[:] = (np.arange(W) + 0.5) * (360.0 / W)
    names = {"mss": "mean_square_slope",
             "mss_u": "mean_square_slope_uncertainty",
             "mss_n": "num_mss_samples", "ws": "wind_speed",
             "ws_u": "wind_speed_uncertainty",
             "ws_n": "num_wind_speed_samples"}
    for k, name in names.items():
        isint = k.endswith("_n")
        x = ds.createVariable(name, "i4" if isint else "f4",
                              ("time", "lat", "lon"), zlib=True,
                              fill_value=(np.int32(FILL_I) if isint
                                          else np.float32(FILL_F)))
        x.set_auto_mask(False)
        x[:] = np.ascontiguousarray(a[k][:, sl, :])
    for name in ("fds_rcg", "mean_square_slope_rcg"):
        x = ds.createVariable(name, "f4", ("time", "lat", "lon"), zlib=True,
                              fill_value=np.float32(FILL_F))
        x.units = "1e-27 dBi meter-4"
        x.set_auto_mask(False)
        x[:] = np.ascontiguousarray(np.where(
            a["mss_n"] > 0, 12.5, FILL_F).astype(np.float32)[:, sl, :])
    ds.close()


def make_day(src_root, d, H, W, seed, variant="consistent", oob=False,
             lat_descending=False):
    """Write one day's L2 and L3; return the L3 arrays as written."""
    s = synth_l2(d, seed)
    l2d = os.path.join(src_root, "cygnss_mss", "L2")
    l3d = os.path.join(src_root, "cygnss_mss", "L3")
    os.makedirs(l2d, exist_ok=True)
    os.makedirs(l3d, exist_ok=True)
    p2 = os.path.join(l2d, l2_name(d) + ".nc")
    p3 = os.path.join(l3d, l3_name(d) + ".nc")
    write_l2(p2, d, s)
    a = l3_from_l2(s, H, W, variant)
    if oob:
        hh, ii, jj = (int(x[0]) for x in np.nonzero(a["ws_n"] > 0))
        cell = a["ws_n"][:, ii, jj] > 0
        a["ws"][cell, ii, jj] = 150.0                # past the 99 m/s bound
    write_l3(p3, d, a, H, W, lat_descending)
    return a, p2, p3


# ============================================================ the truth ===
def day_truth(a, W):
    """The stored day from the L3 arrays as written — a loop over hours."""
    H = a["mss"].shape[1]
    acc = {}
    for tag in ("mss", "ws"):
        S = np.zeros((H, W))
        SV = np.zeros((H, W))
        N = np.zeros((H, W))
        for h in range(24):
            v = a[tag][h].astype(np.float64)
            u = a[f"{tag}_u"][h].astype(np.float64)
            n = a[f"{tag}_n"][h].astype(np.float64)
            ok = (v != FILL_F) & (u != FILL_F) & (u > 0) & (n > 0)
            w = np.zeros((H, W))
            w[ok] = 1.0 / (u[ok] * u[ok])
            S += w
            SV += np.where(ok, w * v, 0.0)
            N += np.where(ok, n, 0.0)
        acc[tag] = (S, SV, N)
    S, SV, N = acc["mss"]
    Sw, SVw, _Nw = acc["ws"]
    out = np.full((H, W, 4), np.nan)
    m = S > 0
    out[m, 0] = SV[m] / S[m]
    out[m, 1] = 1.0 / np.sqrt(S[m])
    out[m, 2] = N[m]
    mw = Sw > 0
    out[mw, 3] = SVw[mw] / Sw[mw]
    out = np.roll(out, -(W // 2), axis=1)
    bounds = ((0.0, 1.0), (0.0, 1.0), (1.0, 2048.0), (0.0, 99.0))
    for c, (lo, hi) in enumerate(bounds):
        x = out[..., c]
        with np.errstate(invalid="ignore"):
            x[np.isfinite(x) & ((x < lo) | (x > hi))] = np.nan
    return out.astype(np.float16)


def make_smoke_sources(root, d_lo, d_hi, seed=20261013):
    """`<root>/cygnss_mss/{L2,L3}/<name>.nc` for every day but the absent
    one, and the truth {(group, bin, frame): (float16 [H, W, 4] | None,
    reason)} for every frame of every bin the record touches."""
    os.environ["CYGNSS_MSS_SMOKE_GRID"] = f"{SMOKE_H},{SMOKE_W}"
    H, W = SMOKE_H, SMOKE_W
    fields = {}
    d = d_lo
    while d <= d_hi:
        if d != SMOKE_ABSENT_DAY:
            a, _p2, _p3 = make_day(root, d, H, W, seed,
                                   oob=(d == SMOKE_OOB_DAY))
            fields[d] = day_truth(a, W)
        d += dt.timedelta(days=1)
    lo, hi = min(fields), max(fields)
    truth = {}
    t_lo = f10b.seconds_since_epoch(lo)
    t_hi = f10b.seconds_since_epoch(hi) + 86399
    for b in sh.bins_overlapping(t_lo, t_hi):
        for f in range(5):
            day = sh.frame_day(b, f, 86400)
            if day < lo:
                truth[("cygnss_mss", b, f)] = (None, "before_record")
            elif day > hi:
                truth[("cygnss_mss", b, f)] = (None, "after_record")
            elif day not in fields:
                truth[("cygnss_mss", b, f)] = (None, "absent_upstream")
            else:
                truth[("cygnss_mss", b, f)] = (fields[day], None)
    return truth
