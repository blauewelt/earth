#!/usr/bin/env python3
"""E-087 — family 7.2 at DAILY resolution: the daily derivation, the rule that
turns five daily frames back into the published pentad value, and the
one-month probe that measures both.

PLAIN ENGLISH. Family 7.2 (`family7_global025_pentad_l2`) is the global input
tensor: 56 channels as FIVE-DAY means. Every one of its channels is computed
from DAILY inputs, so a daily version is the same arithmetic without the last
averaging step. This module holds that arithmetic for the four daily sources —
GLORYS12 ocean reanalysis, NOAA OISST, NCEP/NCAR Reanalysis 1 and ESA OC-CCI
ocean colour — writes the daily frames through the family-1 sharded writer
(`ml/family1/sharded.py`: compressed 0.25° or 1° tiles, one shard per
five-day bin, five frames a bin), and checks the one promise a daily store
can make about itself: averaging its five frames the way family 7 averages
reproduces the published pentad value.

Every derivation below CALLS the family-7 builder's own helpers
(`build_family3.lin_weights` / `interp2_nan`, `build_family7.ice_divisor`,
`drop_sentinels`, `log1p_channel`, `oc_open`, `oc_block_factors`,
`oc_block_stats`, `oc_block_cells`) rather than re-deriving them, and keeps
the family-7 order of operations (bilinear AFTER the native-grid daily mean
for NCEP; the log of the binned MEAN mixed-layer depth; `hypot` of the binned
MEAN currents) — so the falsifier tests the store, not a second opinion.

THE STORES (one store per SOURCE, so a missing source never blocks the
others; plan §3):

  glorys025d   0.25°, C 5: cur_speed log_mld ssh cur_u cur_v   1993-01-01 →
  oisst025d    0.25°, C 2: sst sea_ice                          1982-01-01 →
  ncep100d     1°,    C 15: the g100 channels                   1982-01-01 →
  occci025d    0.25°, C 2: log_chl chl_cov                      1997-09-04 →

`rg100` (the Argo product) is monthly by nature and has no daily form.

USE
  python3 ml/family7_daily.py probe --src <dir> --out <dir> \
          --bins 2411-2416 [--stores glorys025d,oisst025d,ncep100d,occci025d] \
          [--report <json>]

`--src` holds the inputs in the family-7 builder's `--source-dir` layout
(`daily025_global/`, `oisst/`, `ncep/`, `occci/`). The published pentad slabs
are read by HTTP range from the Hub (`data/family7_index.json` gives the
offsets and the norm). Nothing is uploaded.
"""
import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import build_family3 as f3                                     # noqa: E402
import build_family7 as f7                                     # noqa: E402
from family1 import sharded as sh                              # noqa: E402

EPOCH = dt.date(1982, 1, 1)
F = 5
FRAME_SECONDS = 86400
MIN_DAYS = f7.MIN_DAYS                    # 3, family 4's rule
REPO_ROOT = os.path.dirname(HERE)
INDEX_JSON = os.path.join(REPO_ROOT, "data", "family7_index.json")

# --------------------------------------------------------------- grids ----
G025 = {"H": 721, "W": 1440, "x0": -180.0, "dx": 0.25, "y0": -90.0,
        "dy": 0.25, "row_order": "south_first", "crs": "EPSG:4326",
        "align": "point — row r is latitude -90 + 0.25 r, column c is "
                 "longitude -180 + 0.25 c (family 7's g025 grid)"}
G100 = {"H": 181, "W": 360, "x0": -180.0, "dx": 1.0, "y0": -90.0,
        "dy": 1.0, "row_order": "south_first", "crs": "EPSG:4326",
        "align": "point — family 7's g100 grid"}


def _ll(grid):
    def latlon(X, Y):
        return np.asarray(Y, np.float64), np.asarray(X, np.float64)
    return latlon


# Bounds are SANITY envelopes (contract rule 3: outside -> NaN, counted).
STORES = {
    "glorys025d": {
        "group": "g025", "grid": G025, "tile": 256, "first_day": "1993-01-01",
        "channels": [("cur_speed", "m/s", 0.0, 5.0),
                     ("log_mld", "log10(m)", -2.0, 4.0),
                     ("ssh", "m", -4.0, 4.0),
                     ("cur_u", "m/s", -5.0, 5.0),
                     ("cur_v", "m/s", -5.0, 5.0)],
        "pentad_index": [0, 1, 2, 3, 4]},
    "oisst025d": {
        "group": "g025", "grid": G025, "tile": 256, "first_day": "1982-01-01",
        "channels": [("sst", "degC", -3.0, 40.0),
                     ("sea_ice", "fraction (NaN below 0.15)", 0.0, 1.0)],
        "pentad_index": [5, 6]},
    "ncep100d": {
        "group": "g100", "grid": G100, "tile": 64, "first_day": "1982-01-01",
        "channels": [("tau_x", "N/m2", -10.0, 10.0),
                     ("tau_y", "N/m2", -10.0, 10.0),
                     ("tau_x_std", "N/m2 (centred 5-day sigma)", 0.0, 10.0),
                     ("tau_y_std", "N/m2 (centred 5-day sigma)", 0.0, 10.0),
                     ("t2m", "degC", -150.0, 70.0),
                     ("u10", "m/s", -100.0, 100.0),
                     ("v10", "m/s", -100.0, 100.0),
                     ("sp", "hPa", 400.0, 1100.0),
                     ("log_prate", "log1p(mm/day)", 0.0, 10.0),
                     ("log_swe", "log1p(mm w.e.)", 0.0, 15.0),
                     ("soilw", "fraction", 0.0, 1.0),
                     ("tsoil", "degC", -150.0, 80.0),
                     ("lhtfl", "W/m2", -2000.0, 3000.0),
                     ("shtfl", "W/m2", -2000.0, 3000.0),
                     ("skt", "degC", -150.0, 90.0)],
        "pentad_index": list(range(15))},
    "occci025d": {
        "group": "oc025", "grid": G025, "tile": 256, "first_day": "1997-09-04",
        "channels": [("log_chl", "log10(mg/m3)", -10.0, 5.0),
                     ("chl_cov", "fraction of 4 km cells seen", 0.0, 1.0)],
        "pentad_index": [0, 1]},
}


def bin_days(b):
    d0 = EPOCH + dt.timedelta(days=5 * int(b))
    return [d0 + dt.timedelta(days=k) for k in range(F)]


def mask_bounds(arr, channels):
    """NaN outside each channel's [lo, hi], IN PLACE; {name: n} counted."""
    out = {}
    for c, (n, _u, lo, hi) in enumerate(channels):
        v = arr[..., c]
        with np.errstate(invalid="ignore"):
            bad = np.isfinite(v) & ((v < lo) | (v > hi))
        k = int(bad.sum())
        if k:
            v[bad] = np.nan
        out[n] = k
    return out


# =========================================================== derivation ===
def glorys_chunk(path, days, nonpos=None):
    """One GLORYS monthly chunk -> {day: frame} for the wanted days in it."""
    return glorys_daily([path], days, nonpos=nonpos)


def glorys_daily(paths, days, nonpos=None):
    """GLORYS12 0.25° daily chunks -> {day: [721, 1440, 5] float32}.

    The chunk IS the daily field (one value a day per variable on the
    point grid, rows 40..720); nothing is interpolated. `log_mld` is log10 of
    that day's mixed-layer depth where it is > 0 (family 7: of the binned
    mean), `cur_speed` is hypot of that day's u and v.
    """
    import netCDF4 as ncdf
    want = set(days)
    out = {}
    sl = slice(f7.GLORYS_ROW0, f7.GLORYS_ROW0 + f7.GLORYS_ROWS)
    for p in paths:
        d = ncdf.Dataset(p)
        lat = np.asarray(d.variables["latitude"][:], np.float64)
        if len(lat) != f7.GLORYS_ROWS or float(lat[0]) != f7.GLORYS_LAT0:
            sys.exit(f"{p}: latitude is not -80..90 at 0.25")
        vs = [f7.pick_var(d, v) for v in ("uo", "vo", "mlotst", "zos")]
        for k, day in enumerate(f7.nc_dates(d)):
            if day not in want:
                continue
            uo, vo, ml, zs = [np.squeeze(np.ma.filled(np.asarray(v[k]), np.nan)
                                         ).astype(np.float64) for v in vs]
            a = np.full((721, 1440, 5), np.nan, np.float32)
            a[sl, :, 0] = np.hypot(uo, vo)
            with np.errstate(invalid="ignore", divide="ignore"):
                a[sl, :, 1] = np.where(ml > 0, np.log10(np.maximum(ml, 1e-6)),
                                       np.nan)
            a[sl, :, 2] = zs
            a[sl, :, 3] = uo
            a[sl, :, 4] = vo
            out[day] = a
            if nonpos is not None:
                # a FINITE non-positive depth: NaN in log_mld (family 7's
                # rule for a value) but COUNTED in family 7's pentad mean
                # (its count is of finite values) — the check needs it
                full = np.zeros((721, 1440), bool)
                with np.errstate(invalid="ignore"):
                    full[sl] = np.isfinite(ml) & (ml <= 0)
                nonpos[day] = full
        d.close()
    return out


def oisst_daily(sst_path, ice_path, days):
    """OISST v2.1 -> {day: [721, 1440, 2] float32}: bilinear of each day,
    exactly the per-day field `stage_sst` accumulates."""
    import netCDF4 as ncdf
    want = set(days)
    dS, dI = ncdf.Dataset(sst_path), ncdf.Dataset(ice_path)
    lats, lons = f7.grid025()
    src_lat = np.asarray(dS.variables["lat"][:], np.float64)
    src_lon = np.asarray(dS.variables["lon"][:], np.float64)
    wy = f3.lin_weights(src_lat, lats)
    wx = f3.lin_weights(src_lon, np.where(lons < 0, lons + 360.0, lons),
                        wrap_period=360.0)
    v_sst, v_ice = f7.pick_var(dS, "sst"), f7.pick_var(dI, "icec")
    div = f7.ice_divisor(v_ice)
    di = {day: k for k, day in enumerate(f7.nc_dates(dI))}
    out = {}
    for k, day in enumerate(f7.nc_dates(dS)):
        if day not in want:
            continue
        a = np.empty((721, 1440, 2), np.float32)
        s = f7.drop_sentinels(np.ma.filled(np.asarray(v_sst[k]), np.nan))
        a[..., 0] = f3.interp2_nan(np.squeeze(s).astype(np.float64), wy, wx)
        ice = f7.drop_sentinels(np.ma.filled(np.asarray(v_ice[di[day]]),
                                             np.nan)) / div
        g = f3.interp2_nan(np.squeeze(ice).astype(np.float64), wy, wx)
        a[..., 1] = np.clip(g, 0.0, 1.0)
        out[day] = a
    dS.close()
    dI.close()
    return out


NCEP_ORDER = ("uflx", "vflx", "air", "uwnd", "vwnd", "pres", "prate",
              "weasd", "soilw", "tmp", "lhtfl", "shtfl", "skt")


# the g100 channels each NCEP variable feeds
NCEP_CHANNELS = {"uflx": ("tau_x", "tau_x_std"), "vflx": ("tau_y", "tau_y_std"),
                 "air": ("t2m",), "uwnd": ("u10",), "vwnd": ("v10",),
                 "pres": ("sp",), "prate": ("log_prate",),
                 "weasd": ("log_swe",), "soilw": ("soilw",),
                 "tmp": ("tsoil",), "lhtfl": ("lhtfl",), "shtfl": ("shtfl",),
                 "skt": ("skt",)}


def ncep_daily(paths, land, days, sigma_half=2, negmin=None):
    """NCEP R1 4x-daily gaussian files -> {day: [181, 360, 15] float32}.

    `paths` maps variable -> list of files (the year, plus the neighbours a
    centred window needs). `negmin`, if a dict, receives {day: {var: x}}
    for `prate` and `weasd`: the most negative native value of that day's
    samples. Some years' files store a dry cell as -2.3e-10 (2**-32, the
    packing quantum; `np.asarray` drops netCDF4's valid_range mask exactly
    as `stage_ncep` does, so the value is read, not masked), and
    `log1p_channel` clamps at zero PER DAY here and PER PENTAD in family
    7.2, so the two can differ by up to |x| times the unit scale; the lanes'
    check allows exactly that (check_bin `allow`).

    The day's value is the NaN-aware mean of that
    day's 6-hourly samples ON THE NATIVE GRID, then the same bilinear and the
    same transform `stage_ncep.flush` applies to a pentad mean. tau_x_std /
    tau_y_std are the population sigma of the 6-hourly samples of the five
    days CENTRED on the day (family 5's meaning, at family 7's estimator), so
    frame 2 of every bin is that bin's pentad sigma by construction; written
    where >= 3 of those five days are present (family 7's distinct-day rule).
    """
    import netCDF4 as ncdf
    lat1, lon1 = f7.grid100()
    W = {}
    samples = {}             # var -> {date: [native fields in file order]}
    for v in NCEP_ORDER:
        samples[v] = {}
        for p in paths[v]:
            d = ncdf.Dataset(p)
            g_lat = np.asarray(d.variables["lat"][:], np.float64)
            g_lon = np.asarray(d.variables["lon"][:], np.float64)
            if "y1" not in W:
                W["y1"] = f3.lin_weights(g_lat, lat1)
                W["x1"] = f3.lin_weights(g_lon, np.where(lon1 < 0, lon1 + 360.0,
                                                         lon1),
                                         wrap_period=360.0)
            var = f7.pick_var(d, v)
            lo = min(days) - dt.timedelta(days=sigma_half)
            hi = max(days) + dt.timedelta(days=sigma_half)
            for k, day in enumerate(f7.nc_dates(d)):
                if day < lo or day > hi:
                    continue
                f = f7.squeeze_level(np.ma.filled(np.asarray(var[k]), np.nan))
                # float32 is the source's own dtype, so holding it in float32
                # loses nothing (the sign flip is exact) and halves a year's
                # memory; every sum below is float64, as in stage_ncep
                f = np.asarray(f, np.float32)
                if v in f7.NCEP_FLIP:
                    f = -f
                samples[v].setdefault(day, []).append(f)
                if negmin is not None and v in ("prate", "weasd"):
                    lo_v = float(np.nanmin(f)) if np.isfinite(f).any() \
                        else 0.0
                    if lo_v < 0:
                        mv = negmin.setdefault(day, {})
                        mv[v] = min(mv.get(v, 0.0), lo_v)
            d.close()

    def to1(x):
        return f3.interp2_nan(x, W["y1"], W["x1"])

    def mean_of(fields):
        acc = np.zeros(fields[0].shape, np.float64)
        cnt = np.zeros(fields[0].shape, np.int32)
        for f in fields:
            f = np.asarray(f, np.float64)
            ok = np.isfinite(f)
            acc[ok] += f[ok]
            cnt += ok
        with np.errstate(invalid="ignore"):
            return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)

    def sigma_of(v, day):
        win = [day + dt.timedelta(days=k)
               for k in range(-sigma_half, sigma_half + 1)]
        have = [w for w in win if w in samples[v]]
        if len(have) < MIN_DAYS:
            return None
        acc = acc2 = None
        cnt = None
        for w in have:
            for f in samples[v][w]:
                f = np.asarray(f, np.float64)
                if acc is None:
                    acc = np.zeros(f.shape, np.float64)
                    acc2 = np.zeros(f.shape, np.float64)
                    cnt = np.zeros(f.shape, np.int32)
                ok = np.isfinite(f)
                acc[ok] += f[ok]
                acc2[ok] += f[ok] ** 2
                cnt += ok
        with np.errstate(invalid="ignore"):
            mu = acc / np.maximum(cnt, 1)
            var = acc2 / np.maximum(cnt, 1) - mu ** 2
            return np.where(cnt > 0, np.sqrt(np.maximum(var, 0)), np.nan)

    out = {}
    for day in days:
        if any(day not in samples[v] for v in NCEP_ORDER):
            continue                       # absent upstream (counted by caller)
        m = {v: mean_of(samples[v][day]) for v in NCEP_ORDER}
        a = np.full((181, 360, 15), np.nan, np.float32)
        a[..., 0] = to1(m["uflx"])
        a[..., 1] = to1(m["vflx"])
        for ci, v in ((2, "uflx"), (3, "vflx")):
            s = sigma_of(v, day)
            if s is not None:
                a[..., ci] = to1(s)
        a[..., 4] = to1(m["air"]) - 273.15
        a[..., 5] = to1(m["uwnd"])
        a[..., 6] = to1(m["vwnd"])
        a[..., 7] = to1(m["pres"]) / 100.0
        a[..., 8] = f7.log1p_channel(to1(m["prate"]), 86400.0)
        a[..., 9] = f7.log1p_channel(to1(m["weasd"]))
        so, ts = m["soilw"], m["tmp"]
        a[..., 10] = to1(np.where(land, so, np.nan))
        a[..., 11] = to1(np.where(land, ts, np.nan)) - 273.15
        a[..., 12] = to1(m["lhtfl"])
        a[..., 13] = to1(m["shtfl"])
        a[..., 14] = to1(m["skt"]) - 273.15
        out[day] = a
    return out


def occci_daily(path):
    """One OC-CCI 4 km day -> [721, 1440, 2] float32: the day's block mean of
    log10(chl) (exactly what `oc_year_reduce` adds to its accumulator) and the
    fraction of the block's 4 km cells that saw the sea. NaN in both where no
    cell did."""
    chl, s_lat, s_lon, fill = f7.oc_open(path)
    by, bx = f7.oc_block_factors(s_lat, s_lon)
    S, C = f7.oc_block_stats(chl, by, bx, fill=fill)
    cells = f7.oc_block_cells(by, bx, len(s_lat))
    a = np.empty((721, 1440, 2), np.float32)
    got = C > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        a[..., 0] = np.where(got, (S / np.maximum(C, 1)).astype(np.float32),
                             np.nan)
        a[..., 1] = np.where(got, C / np.maximum(cells[:, None], 1), np.nan)
    return a, {"blk": [by, bx], "cells_full": int(by * bx)}


# ======================================================= reconstruction ===
def _nanmean_min(st, need):
    n = np.isfinite(st).sum(0)
    with np.errstate(invalid="ignore"):
        s = np.where(np.isfinite(st), st, 0.0).sum(0, dtype=np.float64)
        return np.where(n >= need, s / np.maximum(n, 1), np.nan), n


def pentad_from_daily(store, frames):
    """Five daily frames [5, H, W, C] (physical, NaN = missing; a frame not in
    the source is all-NaN) -> the pentad value family 7 publishes, by the
    channel's own aggregation rule (plan §4):

      linear channels       mean over the finite days, >= 3 of them
      cur_speed             hypot(mean cur_u, mean cur_v)
      log_mld               log10(mean(10 ** log_mld))
      log_prate, log_swe    log1p(mean(expm1(x)))
      tau_x_std, tau_y_std  frame 2 (the centred window IS the bin)
      log_chl               mean over the finite days, >= 1
      chl_cov               sum over the finite days / 5
    """
    st = np.asarray(frames, np.float64)
    C = st.shape[-1]
    out = np.full(st.shape[1:], np.nan, np.float64)
    if store == "glorys025d":
        for c in (2, 3, 4):
            out[..., c], _ = _nanmean_min(st[..., c], MIN_DAYS)
        out[..., 0] = np.hypot(out[..., 3], out[..., 4])
        mld, _ = _nanmean_min(10.0 ** st[..., 1], MIN_DAYS)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[..., 1] = np.where(mld > 0, np.log10(mld), np.nan)
    elif store == "oisst025d":
        out[..., 0], _ = _nanmean_min(st[..., 0], MIN_DAYS)
        m, _ = _nanmean_min(st[..., 1], MIN_DAYS)
        out[..., 1] = np.clip(m, 0.0, 1.0)
    elif store == "ncep100d":
        for c in range(C):
            if c in (2, 3):
                out[..., c] = st[2, ..., c]
            elif c in (8, 9):
                m, _ = _nanmean_min(np.expm1(st[..., c]), MIN_DAYS)
                out[..., c] = np.log1p(np.maximum(m, 0.0))
            else:
                out[..., c], _ = _nanmean_min(st[..., c], MIN_DAYS)
    elif store == "occci025d":
        out[..., 0], n = _nanmean_min(st[..., 0], 1)
        cov = np.where(np.isfinite(st[..., 1]), st[..., 1], 0.0).sum(0) / F
        out[..., 1] = np.where(n > 0, cov, np.nan)
    else:
        raise KeyError(store)
    return out


# ========================================================= pentad reads ===
def load_index(path=INDEX_JSON):
    with open(path) as fh:
        return json.load(fh)


def pentad_slabs(index, group, b0, n):
    """`n` consecutive published pentad bins of one group, by ONE range read,
    un-z-scored to physical units (float64) -> [n, H, W, C]."""
    g = index["groups"][group]
    H, W, C = g["shape"][1:]
    slab = int(g["slab_bytes"])
    row0 = int(b0) - int(g.get("bin_first", 0))
    if row0 < 0 or row0 + n > int(g["shape"][0]):
        raise IndexError(f"{group}: bins {b0}..{b0 + n - 1} outside the file")
    blob = f7.http_range(g["url"], int(g["header_len"]) + row0 * slab,
                         n * slab)
    z = np.frombuffer(blob, "<f2").reshape(n, H, W, C).astype(np.float64)
    norm = np.asarray(g["norm"], np.float64)
    return z * norm[:, 1] + norm[:, 0], z, norm


def f16_half_step(x):
    """Half the float16 spacing at |x| (the largest rounding error)."""
    x = np.abs(np.asarray(x, np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.floor(np.log2(np.maximum(x, 2.0 ** -14)))
    return 0.5 * 2.0 ** (e - 10)


def check_bin(store, frames, P, z, norm, allow=None, mld_nonpos=None):
    """THE FALSIFIER FOR ONE BIN, as a lane runs it before a year is marked.

    `frames` are the five frames AS STORED (float16, decoded to float32; an
    absent frame is all-NaN), `P` the published pentad value un-z-scored to
    physical units, `z` the stored z-score, `norm` the group's (mean, sd) —
    all three already restricted to this store's channels (`pentad_index`).
    The tolerance is `compare()`'s "stored" one (plan §4), plus
    `allow[channel]` (absolute, physical units) where the caller knows of a
    mechanism by which the daily and pentad paths may differ — today only
    NCEP's per-day clamp of tiny negative rates (ncep_daily `negmin`) —
    reported per channel as `allowed`. `mld_nonpos` (glorys025d) is the
    per-cell count of days in the bin whose mixed-layer depth was FINITE
    and <= 0: stored NaN in `log_mld` (no logarithm), yet counted as a
    zero in family 7.2's pentad mean — so log_mld is rebuilt with those
    days counted (exactly family 7's arithmetic), and the count is
    reported. Returns a dict with
    `ok` and, per channel, the cells compared, NaN-pattern mismatches, the
    largest |difference| and the largest excess over the tolerance.
    """
    cfg = STORES[store]
    st = np.stack([np.asarray(f, np.float32) for f in frames])
    R = pentad_from_daily(store, st)
    if mld_nonpos is not None and store == "glorys025d":
        x = st[..., 1].astype(np.float64)
        ok = np.isfinite(x)
        ssum = np.where(ok, 10.0 ** np.where(ok, x, 0.0), 0.0).sum(0)
        cnt = ok.sum(0) + np.asarray(mld_nonpos)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.where(cnt >= MIN_DAYS, ssum / np.maximum(cnt, 1), np.nan)
            R[..., 1] = np.where(mu > 0, np.log10(np.maximum(mu, 1e-300)),
                                 np.nan)
    fin = np.isfinite(st)
    hs = f16_half_step(st)
    hs[~fin] = 0.0
    hs = hs.max(0)
    mag = np.abs(np.where(fin, st, 0.0)).max(0)
    double = cfg["group"] == "g025"
    out = {"ok": True, "channels": {}}
    for c, (name, _u, _lo, _hi) in enumerate(cfg["channels"]):
        Pc, Rc, zc = P[..., c], R[..., c], z[..., c]
        fp, fr = np.isfinite(Pc), np.isfinite(Rc)
        both = fp & fr
        mism = int((fp != fr).sum())
        d = np.abs(Rc - Pc)[both]
        tol = f16_half_step(zc[both]) * norm[c, 1]
        # family 7 z-scores in FLOAT32 (build_family7 norm stage: mu, sd cast
        # to float32, (X32 - mu) / sd, then float16), so the stored z is
        # rounded twice: two float32 half-ulps of |z| before the float16
        # cast, which can move a value sitting on a float16 midpoint to the
        # far neighbour (OC-CCI 2013 bin 2324, one cell: z 1.7250977 stored
        # as 1.7246094, 7.0e-9 past the old tolerance)
        tol = tol + 2.0 ** -23 * np.abs(zc[both]) * norm[c, 1]
        off = 273.15 if name in ("t2m", "tsoil", "skt") else 0.0
        tol = tol + 2.0 ** -21 * (np.maximum(np.abs(Pc[both]),
                                             mag[..., c][both]) + off)
        if double:
            tol = tol + f16_half_step(np.maximum(np.abs(Pc[both]),
                                                 np.abs(Rc[both])))
        tol = tol + (1.5 if name == "cur_speed" else 1.0) * hs[..., c][both]
        extra = float((allow or {}).get(name, 0.0))
        tol = tol + extra
        tol = tol * (1 + 1e-6) + 1e-12
        exc = float((d - tol).max()) if d.size else 0.0
        r = {"cells": int(both.sum()), "nan_mismatch": mism,
             "max_abs_diff": float(d.max()) if d.size else 0.0,
             "max_excess": exc, "allowed": extra}
        if mism or exc > 0:
            out["ok"] = False
        out["channels"][name] = r
    return out


# ================================================================ probe ===
def write_store(store, frames_by_bin, out_dir):
    """Frames -> a local sharded group via the REAL writer. Returns entries,
    bytes and the group directory."""
    cfg = STORES[store]
    spec = sh.make_spec(store, cfg["grid"], cfg["channels"], F, FRAME_SECONDS,
                        "float16", tile=cfg["tile"], level=sh.DEFAULT_LEVEL,
                        latlon=_ll(cfg["grid"]))
    gd = os.path.join(out_dir, store, store)
    os.makedirs(gd, exist_ok=True)
    w = sh.ShardWriter(spec)
    entries = []
    for b in sorted(frames_by_bin):
        entries.append(w.write_bin(
            b, frames_by_bin[b], os.path.join(gd, sh.shard_relpath(b)),
            os.path.join(gd, sh.index_relpath(b))))
    f7.atomic_json(os.path.join(gd, "tile_grid.json"), spec)
    sh.save_shard_index(os.path.join(gd, "shard_index.npy"), entries,
                        len(cfg["channels"]))
    return entries, spec, gd


def compare(store, recon, pentad, z, norm, stored_recon=None, stored_hs=None,
            mag=None):
    """Per channel: the falsifier numbers (plan §4).

    The TOLERANCE is the rounding the published value has already been
    through, and no more: half a float16 step of the stored z-score times the
    channel's sd — plus, for `g025` only, half a float16 step of the PHYSICAL
    value, because `build_family7` fills g025 as a float16 memmap in physical
    units and z-scores it IN PLACE (`RAW_F32` is g100, rg100, oc025), so a
    g025 value is rounded twice — and 8 float32 half-ulps of the largest
    summand, for the float32 arithmetic both paths share. The `stored`
    comparison adds the daily
    store's own rounding: the largest half-step among the five frames (x1.5
    for `cur_speed`, the hypot of two rounded means).
    """
    cfg = STORES[store]
    double = cfg["group"] == "g025"
    rep = {}
    for c, (name, unit, _lo, _hi) in enumerate(cfg["channels"]):
        pc = cfg["pentad_index"][c]
        P = pentad[..., pc]
        sd = norm[pc, 1]
        tol_p = f16_half_step(z[..., pc]) * sd          # the pentad's own f16
        tol_p = tol_p + 2.0 ** -23 * np.abs(z[..., pc]) * sd  # its f32 z-score
        r = {"unit": unit}
        for label, R in (("derivation", recon[..., c]),
                         ("stored", None if stored_recon is None
                          else stored_recon[..., c])):
            if R is None:
                continue
            fp, fr = np.isfinite(P), np.isfinite(R)
            both = fp & fr
            d = np.abs(R - P)[both]
            tol = tol_p[both]
            # ... and a few float32 roundings of the value in the SOURCE's own
            # units (K, Pa): `interp2_nan` returns float32 and the transform
            # (-273.15, /100) and the z-score are float32 on both paths, so
            # the two orders of summation may land on neighbouring float32
            # values before the float16 cast: 8 half-ulps of the largest
            # summand (a mean near zero of +-50 W/m2 days carries the
            # summands' rounding, not its own).
            # (applies to every group: g025's log_mld is log10 of a float32)
            off = 273.15 if name in ("t2m", "tsoil", "skt") else 0.0
            big = np.abs(P[both]) if mag is None else np.maximum(
                np.abs(P[both]), mag[..., c][both])  # the summands' magnitude
            tol = tol + 2.0 ** -21 * (big + off)
            if double:                                   # g025's first rounding,
                tol = tol + f16_half_step(               # at whichever side of a
                    np.maximum(np.abs(P[both]), np.abs(R[both])))  # power of 2
            if label == "stored":
                k = 1.5 if name == "cur_speed" else 1.0
                tol = tol + k * stored_hs[..., c][both]  # the daily's own f16
            within = d <= tol * (1 + 1e-6) + 1e-12
            r[label] = {
                "cells_compared": int(both.sum()),
                "finite_pentad_only": int((fp & ~fr).sum()),
                "finite_daily_only": int((fr & ~fp).sum()),
                "max_abs_diff": float(d.max()) if d.size else None,
                "p999_abs_diff": float(np.quantile(d, 0.999)) if d.size
                else None,
                "max_abs_diff_in_sd": float(d.max() / sd) if d.size else None,
                "within_float16_tolerance": float(within.mean()) if d.size
                else None,
                "max_excess_over_tolerance": float((d - tol).max())
                if d.size else None,
            }
        rep[name] = r
    return rep


def precision_table(store, daily_stack, norm):
    """float16 rounding error of a day's value stored PHYSICAL vs stored
    z-scored with the pentad norm: max abs, and relative to the channel's
    day-to-day standard deviation in this month."""
    cfg = STORES[store]
    out = {}
    for c, (name, _u, _lo, _hi) in enumerate(cfg["channels"]):
        x = daily_stack[..., c]
        x = x[np.isfinite(x)].astype(np.float64)
        if not x.size:
            continue
        pc = cfg["pentad_index"][c]
        mu, sd = norm[pc]
        e_phys = np.abs(x.astype(np.float16).astype(np.float64) - x)
        zz = ((x - mu) / sd).astype(np.float16).astype(np.float64)
        e_z = np.abs(zz * sd + mu - x)
        dsd = float(np.nanstd(np.diff(daily_stack[..., c], axis=0)))
        out[name] = {"max_err_physical": float(e_phys.max()),
                     "max_err_zscored": float(e_z.max()),
                     "mean_err_physical": float(e_phys.mean()),
                     "mean_err_zscored": float(e_z.mean()),
                     "day_to_day_sd": dsd,
                     "max_err_physical_over_day_sd":
                         float(e_phys.max() / dsd) if dsd > 0 else None}
    return out


def _paths(src, sub, names):
    return [os.path.join(src, sub, n) for n in names]


def probe(args):
    bins = list(range(int(args.bins.split("-")[0]),
                      int(args.bins.split("-")[-1]) + 1))
    days = [d for b in bins for d in bin_days(b)]
    years = sorted({d.year for d in days})
    index = load_index(args.index)
    stores = args.stores.split(",")
    report = {"bins": bins, "days": [str(days[0]), str(days[-1])],
              "zstandard": sh.zstd().__version__, "numpy": np.__version__,
              "builder_git_sha": f7.git_sha(), "stores": {}}
    for store in stores:
        cfg = STORES[store]
        t0 = time.time()
        extra = {}
        if store == "glorys025d":
            months = sorted({(d.year, d.month) for d in days})
            daily = glorys_daily(
                _paths(args.src, "daily025_global",
                       [f"glorys025_global_{y}{m:02d}.nc" for y, m in months]),
                days)
        elif store == "oisst025d":
            if len(years) != 1:
                sys.exit("probe: one calendar year of OISST at a time")
            y = years[0]
            daily = oisst_daily(
                os.path.join(args.src, "oisst", f"sst.day.mean.{y}.nc"),
                os.path.join(args.src, "oisst", f"icec.day.mean.{y}.nc"), days)
        elif store == "ncep100d":
            import netCDF4 as ncdf
            ys = sorted({(d + dt.timedelta(days=k)).year for d in days
                         for k in (-2, 2)})
            paths = {v: [p for p in (os.path.join(
                args.src, "ncep", f"{f7.NCEP_FILES[v]}.{yy}.nc") for yy in ys)
                if os.path.exists(p)] for v in NCEP_ORDER}
            dl = ncdf.Dataset(os.path.join(args.src, "ncep",
                                           f"{f7.NCEP_LAND}.nc"))
            land = f7.squeeze_level(np.ma.filled(
                np.asarray(f7.pick_var(dl, "land")[:]), 0.0)) >= 0.5
            dl.close()
            daily = ncep_daily(paths, land, days)
        elif store == "occci025d":
            daily = {}
            for d in days:
                p = os.path.join(args.src, "occci",
                                 f7.oc_canonical_name(d))
                if os.path.exists(p):
                    daily[d], extra["geom"] = occci_daily(p)
        t_derive = time.time() - t0
        oob = {}
        for d, a in daily.items():
            for k, v in mask_bounds(a, cfg["channels"]).items():
                oob[k] = oob.get(k, 0) + v
        frames_by_bin = {b: [daily.get(d) for d in bin_days(b)] for b in bins}
        t1 = time.time()
        entries, spec, gd = write_store(store, frames_by_bin, args.out)
        t_write = time.time() - t1
        g = sh.ShardedGroup(gd)
        nbytes = sum(e["nbytes"] for e in entries)
        idx_bytes = sum(os.path.getsize(os.path.join(gd, e["index"]))
                        for e in entries)
        n_frames = sum(e["frames_present"] for e in entries)
        H, W, C = spec["H"], spec["W"], spec["C"]
        valid = np.sum([e["valid_pixels"] for e in entries], axis=0)
        # --- falsifier: published pentad vs mean of the daily frames ---
        pent, z, norm = pentad_slabs(index, cfg["group"], bins[0], len(bins))
        fals = {}
        t2 = time.time()
        rec_all, srec_all, hs_all, mag_all = [], [], [], []
        for i, b in enumerate(bins):
            fr = [a if a is not None else np.full((H, W, C), np.nan,
                                                  np.float32)
                  for a in frames_by_bin[b]]
            rec_all.append(pentad_from_daily(store, fr))
            fs = np.stack(fr)
            mag_all.append(np.abs(np.where(np.isfinite(fs), fs, 0.0)).max(0))
            stored = [g.read_frame(b, f) for f in range(F)]
            stored = [s if s is not None else np.full((H, W, C), np.nan,
                                                      np.float32)
                      for s in stored]
            srec_all.append(pentad_from_daily(store, stored))
            hs = f16_half_step(np.stack(stored))
            hs[~np.isfinite(np.stack(stored))] = 0.0
            hs_all.append(hs.max(0))
            exact = all(np.array_equal(
                np.asarray(frames_by_bin[b][f], np.float16), stored[f]
                .astype(np.float16), equal_nan=True)
                for f in range(F) if frames_by_bin[b][f] is not None)
            fals.setdefault("readback_bit_exact", True)
            fals["readback_bit_exact"] &= bool(exact)
        fals["channels"] = compare(store, np.stack(rec_all), pent, z, norm,
                                   np.stack(srec_all), np.stack(hs_all),
                                   np.stack(mag_all))
        t_fals = time.time() - t2
        stack = np.stack([daily[d] for d in sorted(daily)])
        report["stores"][store] = {
            "frames_written": n_frames,
            "frames_missing": len(bins) * F - n_frames,
            "out_of_bounds": oob,
            "H": H, "W": W, "C": C, "tile": spec["tile"],
            "tiles_per_frame": spec["n_tiles_y"] * spec["n_tiles_x"],
            "stored_bytes": nbytes, "index_bytes": idx_bytes,
            "bytes_per_frame": nbytes / max(n_frames, 1),
            "raw_float16_bytes_per_frame": H * W * C * 2,
            "compression_vs_raw_f16": H * W * C * 2 * n_frames / max(nbytes, 1),
            "valid_fraction": {cn[0]: float(valid[c] / max(n_frames * H * W, 1))
                               for c, cn in enumerate(cfg["channels"])},
            "wall_derive_s": round(t_derive, 1),
            "wall_write_s": round(t_write, 1),
            "wall_falsifier_s": round(t_fals, 1),
            "precision": precision_table(store, stack, norm),
            "falsifier": fals,
            **extra,
        }
        print(json.dumps({store: {k: report["stores"][store][k] for k in
                                  ("frames_written", "bytes_per_frame",
                                   "valid_fraction", "wall_derive_s",
                                   "wall_write_s")}}), flush=True)
    import resource
    report["peak_rss_mb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    if args.report:
        f7.atomic_json(args.report, report)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bins", default="2411-2416")
    p.add_argument("--stores", default=",".join(STORES))
    p.add_argument("--index", default=INDEX_JSON)
    p.add_argument("--report")
    a = ap.parse_args(argv)
    if a.cmd == "probe":
        probe(a)


if __name__ == "__main__":
    main()
