"""CYGNSS sea-surface mean-square slope, daily on the producer's 0.2-degree
grid (family 1.3, E-092 phase C, D13; sharded tier G).

PLAIN ENGLISH. Eight small NASA satellites (CYGNSS) listen to GPS signals
reflected off the sea. The sharper the reflection, the smoother the sea: the
MEAN-SQUARE SLOPE (MSS) is the variance of the slopes of the waves longer than
about 20 cm, recovered from how strongly the reflection comes back. It is the
raw roughness observable underneath CYGNSS's "microplastic index" — which the
family rejected (E-092 D7) because that index is a roughness ANOMALY fitted to
a plastic model at 1 degree and 30 days. This store keeps the observable, not
the model.

SOURCE. `CYGNSS_L3_V3.2` (C2832196567-POCLOUD, DOI 10.5067/CYGNS-L3X32), one
netCDF-4 file per UTC day at PO.DAAC, Earthdata-protected — so the BYTES are a
hosted runner's job; the CMR listing is anonymous. Each file holds 24 HOURLY
bins on a 0.2 x 0.2 degree grid, lat 400 (-39.9 .. 39.9, bin centres) x lon
1800 (0.1 .. 359.9). Per bin the producer's data dictionary (UM 148-0351-5)
says: `mean_square_slope` is the INVERSE-VARIANCE weighted mean of all L2 MSS
samples in the bin, `mean_square_slope_uncertainty` the "standard deviation of
the error in the mean", `num_mss_samples` the number of L2 samples used; the
same three exist for `wind_speed`. Fill values (CMR UMM-Var, measured
2026-10-13): -9999.0 for the floats, -99 for the counts.

WHAT IS STORED: one group `cygnss_mss`, 400 x 1800, F = 5 DAILY frames a bin,
float16, C = 4:

  mss               the day's inverse-variance mean of the hourly bins,
                    w_h = 1 / mean_square_slope_uncertainty_h^2,
                    mss = sum(w_h mss_h) / sum(w_h)          (unitless)
  mss_uncertainty   1 / sqrt(sum(w_h))                       (unitless)
  n_mss             sum(num_mss_samples_h)                   (samples)
  wind_speed        CYGNSS's OWN wind, combined THE PRODUCER'S WAY: the
                    inverse-variance mean of the hourly `wind_speed` with
                    w_h = 1 / wind_speed_uncertainty_h^2 (the L3 ATBD's
                    minimum-variance estimator, one level up). It is NOT
                    independent of MSS — the L2 wind is retrieved from the
                    same normalised radar cross section that MSS is (L2 MSS
                    ATBD, UM 148-0139) — so it is kept to STUDY the wind
                    confound, never as the independent wind a physics check
                    or a plastic model may regress on. It is combined over
                    ITS OWN valid hours (fds flags, num_wind_speed_samples),
                    so a cell can carry a wind and no MSS.

WHY THE DAILY COMBINATION IS EXACT — AND WHAT FALSIFIES IT (E-092 §6, F1). If
each hourly bin is the IV mean of its L2 samples and its uncertainty is
1/sqrt(sum of the samples' weights), then the IV mean of the hours with
w_h = 1/u_h^2 is algebraically the IV mean of all of the day's samples and
1/sqrt(sum w_h) its uncertainty. That rests on the producer's definition of
the uncertainty, which the dictionary words loosely ("standard deviation of
the error in the mean"). `f1_check` tests it: for three days it bins the good
L2 V3.2 samples (`mss_sample_flags` bit 1, value 1 = fatal_composite_mss,
clear) into the L3 hour x 0.2-degree cells with weights 1/uncertainty^2, and
compares counts (exact, >= 99.9 % of non-empty cells), means (<= 1e-5
relative, >= 99.9 % of compared cells) and the uncertainty identity — hourly
AND at the daily level this store writes. A definition that differs is
REPORTED with numbers (and the best-fitting alternative definition), never
passed. It runs inside `--stage probe` (CYGNSS_MSS_F1, below) and its report
lands in the probe JSON under counts.f1_producer_reproduction.

THE GRID, AS STORED. The producer's pixels exactly — no resampling — with the
columns ROTATED by 900 so column 0 is 180 W (the family's lon in [-180, 180)
convention; a permutation, lossless). Row 0 is the SOUTHERNMOST row (-39.9),
the order the L3 file's ascending `lat` has; a file whose `lat` descends is
flipped and COUNTED (`lat_axis_flipped`), one whose axes do not match the
declared affine grid within 0.001 degree is a refusal.

FLOAT16. mss (~0.005 .. 0.05) keeps ~3 significant digits (relative half-step
2.4e-4, well under the ATBD's 1.2-2.5 % retrieval error). n_mss is an integer
and float16 is exact to 2048; the channel's upper bound is 2048, so a larger
day-cell count becomes NaN and is COUNTED rather than rounded.
mss_uncertainty values below float16's smallest normal (6.1e-5) lose relative
precision; they are counted per frame (`mss_uncertainty_subnormal_f16`) so the
probe measures how often.

LICENCE. NASA Earth Science data policy: open, no restriction — but the
collection record carries no use-constraint sentence of its own
(`ml/family1/probes/cygnss_mss_measure.json`). The policy itself is the
licence, as for pace4k and every other NASA store in the family, so
`redistribution_confirmed` is True (plan D13). Cite DOI 10.5067/CYGNS-L3X32.

KNOBS (adapter_env):
  CYGNSS_MSS_F1        probe (default: run F1 in --stage probe only) |
                       always (also in fetch; the smoke uses it) | off
  CYGNSS_MSS_F1_DAYS   comma list of YYYY-MM-DD; default = first, middle and
                       last L3 day listed in the window
  CYGNSS_MSS_SMOKE_GRID  "H,W" — a shrunken grid for the synthetic smoke only
"""
import datetime as dt
import json
import os
import re
import sys
import time

import numpy as np

import build_family10_stores as f10b
from family1 import sharded as sh
from family1.adapters import _common as cm

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
L3_CID, L3_SHORT, L3_DOI = ("C2832196567-POCLOUD", "CYGNSS_L3_V3.2",
                            "10.5067/CYGNS-L3X32")
L2_CID, L2_SHORT, L2_DOI = ("C2832196001-POCLOUD", "CYGNSS_L2_V3.2",
                            "10.5067/CYGNS-L2X32")
# `cyg.ddmi.s20190115-003000-e20190115-233000.l3.grid-wind.a32.d33`
L3_NAME = re.compile(r"^cyg\.ddmi\.s(\d{8})-\d{6}-e(\d{8})-\d{6}"
                     r"\.l3\.grid-wind\.a32\.d33")
# `cyg.ddmi.s20190115-000000-e20190115-235959.l2.wind-mss.a32.d33`
L2_NAME = re.compile(r"^cyg\.ddmi\.s(\d{8})-\d{6}-e(\d{8})-\d{6}"
                     r"\.l2\.wind-mss\.a32\.d33")
FULL_H, FULL_W = 400, 1800
LAT_S, LAT_N = -40.0, 40.0
HOURS = 24
FILL_F, FILL_I = -9999.0, -99
FIRST_YEAR = 2018
WORKERS = 3
EPOCH = dt.datetime(1982, 1, 1)
assert EPOCH.date() == f10b.START, (EPOCH, f10b.START)
# mss_sample_flags (L2 dictionary 148-0348-13): bit 1 (value 1) is
# fatal_composite_mss = low | high | inc>65 | bin ratio; bit 6 (value 32) is
# the NON-fatal preliminary GPS antenna knowledge flag (kept).
MSS_FATAL = 1
FDS_FATAL = 1                      # fds_sample_flags bit 1, fatal composite
F16_MIN_NORMAL = float(np.finfo(np.float16).tiny)       # 6.1035e-05
N_MAX = 2048.0                     # float16 integers are exact to here

CHANNELS = (("mss", "1", 0.0, 1.0),
            ("mss_uncertainty", "1", 0.0, 1.0),
            ("n_mss", "L2 samples", 1.0, N_MAX),
            ("wind_speed", "m s-1", 0.0, 99.0))
L3_VARS = {"mss": "mean_square_slope",
           "mss_u": "mean_square_slope_uncertainty",
           "mss_n": "num_mss_samples",
           "ws": "wind_speed",
           "ws_u": "wind_speed_uncertainty",
           "ws_n": "num_wind_speed_samples"}
L2_VARS = ("sample_time", "lat", "lon", "mean_square_slope",
           "mean_square_slope_uncertainty", "mss_sample_flags")
L2_WIND_VARS = ("wind_speed", "wind_speed_uncertainty", "fds_sample_flags")

# F1 criteria (E-092 §6 / the measurement's falsifiable_checks)
F1_COUNT_FRAC = 0.999
F1_REL = 1e-5
F1_VALUE_FRAC = 0.999


class FormatError(ValueError):
    """A listing or a CYGNSS file that does not match the product."""


# ================================================================== grid ===
def grid(full_h=FULL_H, full_w=FULL_W):
    dy = (LAT_N - LAT_S) / int(full_h)
    dx = 360.0 / int(full_w)
    if abs(dx - dy) > 1e-9:
        raise FormatError(f"grid {full_h} x {full_w}: {dy} x {dx} degrees is "
                          f"not square")
    return {
        "H": int(full_h), "W": int(full_w),
        "pixel_deg": dx, "pixel_km_equator": round(dx * 111.31949, 4),
        "crs": "EPSG:4326",
        "proj4": "+proj=longlat +datum=WGS84 +no_defs",
        "x0": -180.0, "dx": dx, "y0": LAT_S, "dy": dy,
        "coordinates": ("lon = x0 + (col + 0.5) * dx, lat = y0 + (row + 0.5) "
                        "* dy; x0/y0 are the grid's WEST and SOUTH edges"),
        "row_order": ("row 0 is the SOUTHERNMOST row (bin centre -39.9) — "
                      "the order of the L3 file's ascending `lat`"),
        "col_order": ("col 0 is 180 W (centre -179.9): the producer's "
                      "0.1 .. 359.9 E columns ROTATED by W/2, a lossless "
                      "permutation; the pixels are the producer's own"),
        "extent": [-180.0, LAT_S, 180.0, LAT_N],
        "extent_note": "[west, south, east, north] in degrees; +-40 latitude",
        "no_tile_corners": ("a plain geographic grid: the affine rule above "
                            "IS the lat/lon of every pixel and tile corner"),
    }


def check_axes(lat, lon, full_h, full_w):
    """-> (flip_lat, info). Refuses an axis off the declared grid."""
    lat = np.asarray(lat, np.float64)
    lon = np.asarray(lon, np.float64)
    if lat.shape != (int(full_h),) or lon.shape != (int(full_w),):
        raise FormatError(f"lat {lat.shape} and lon {lon.shape}; the declared "
                          f"grid is {full_h} x {full_w}")
    d = (LAT_N - LAT_S) / full_h
    want_lat = LAT_S + (np.arange(full_h) + 0.5) * d
    want_lon = (np.arange(full_w) + 0.5) * (360.0 / full_w)
    lerr = float(np.max(np.abs(lon - want_lon)))
    if lerr > 1e-3:
        raise FormatError(f"the file's `lon` is not the declared 0 .. 360 E "
                          f"ascending axis (off by up to {lerr:.4f} degrees, "
                          f"first {lon[0]:.4f}); this adapter rotates that "
                          f"axis and refuses to guess another")
    flip = bool(lat.size > 1 and lat[1] < lat[0])
    got_lat = lat[::-1] if flip else lat
    err = float(np.max(np.abs(got_lat - want_lat)))
    if err > 1e-3:
        raise FormatError(f"the file's `lat` axis differs from the declared "
                          f"grid by up to {err:.4f} degrees (first "
                          f"{got_lat[0]:.4f} vs {want_lat[0]:.4f})")
    return flip, {"lat_first": float(lat[0]), "lat_last": float(lat[-1]),
                  "lon_first": float(lon[0]), "lon_last": float(lon[-1])}


def to_store_cols(a, full_w):
    """[H, W, C] in producer column order (0.1 .. 359.9 E) -> store order
    (column 0 = 180 W): a rotation by W/2, lossless."""
    return np.roll(a, -(int(full_w) // 2), axis=1)


# ================================================================ listing ==
def _day_of(title, rx):
    m = rx.match(title)
    if not m:
        return None
    s, e = m.group(1), m.group(2)
    if s != e:
        raise FormatError(f"{title}: starts {s} and ends {e} — a CYGNSS L2/L3 "
                          f"file is one UTC day")
    try:
        return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except ValueError:
        raise FormatError(f"{title}: {s} is not a date") from None


def _entry_url(g, title):
    for ln in g.get("links", []):
        h = str(ln.get("href", ""))
        if h.startswith("https") and "protected" in h and h.endswith(".nc"):
            return h
    raise FormatError(f"{title}: no protected https .nc link in its CMR entry")


def cmr_days(cid, rx, t_lo, t_hi, attempts=4, count=None):
    """{date: {name, url, bytes}} for the window (anonymous CMR listing)."""
    q = {"collection_concept_id": cid, "sort_key": "start_date",
         "temporal": f"{t_lo},{t_hi}"}
    out = {}
    for g in cm.cmr_entries(CMR, q, attempts=attempts, count=count,
                            what=f"{cid} {t_lo}..{t_hi}"):
        title = str(g.get("title", ""))
        d = _day_of(title, rx)
        if d is None:
            continue
        url = _entry_url(g, title)
        # CMR's granule_size is NOT evidence about the file: measured
        # 2026-10-13, 2019-01-15's L3 is declared 9.6e-5 MB (100 bytes) and
        # its neighbour 15.5 MB. Recorded, never acted on (sst_acspo02's
        # lesson); the FILE decides.
        nb = int(float(g.get("granule_size") or 0) * 1e6)
        if d in out and out[d]["name"] != title:
            raise FormatError(f"two granules for {d}: {out[d]['name']} and "
                              f"{title}")
        out[d] = {"name": title, "url": url, "bytes": nb}
    return out


def cmr_record(cid, rx, attempts=4, count=None):
    """(first day, last day) of the WHOLE collection, from CMR itself."""
    out = []
    for key in ("start_date", "-start_date"):
        url = (f"{CMR}?collection_concept_id={cid}&sort_key={key}"
               f"&page_size=1")
        ents, _h, _a = cm.cmr_page(url, attempts=attempts, count=count)
        if not ents:
            raise FormatError(f"{cid}: CMR lists no granule at all")
        d = _day_of(str(ents[0].get("title", "")), rx)
        if d is None:
            raise FormatError(f"{cid}: first/last granule "
                              f"{ents[0].get('title')!r} is not named as "
                              f"expected")
        out.append(d)
    return out[0], out[1]


def local_days(root, rx):
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for n in sorted(files):
            if not n.endswith(".nc"):
                continue
            d = _day_of(n[:-3], rx)
            if d is None:
                continue
            p = os.path.join(dirpath, n)
            out[d] = {"name": n[:-3], "url": p, "bytes": os.path.getsize(p)}
    return out


# ================================================================ reading ==
def _as_float(v, sentinel):
    """A netCDF variable -> float64 with NaN for masked, fill and non-finite."""
    a = v[:]
    m = np.ma.getmaskarray(np.ma.asarray(a))
    out = np.asarray(np.ma.getdata(a), np.float64).copy()
    out[m] = np.nan
    out[~np.isfinite(out)] = np.nan
    out[out == sentinel] = np.nan
    return out


def read_l3(path, full_h=FULL_H, full_w=FULL_W, counts=None):
    """One L3 day -> {key: float64 [24, H, W]} in the PRODUCER's column order
    (0.1 .. 359.9 E) and lat ASCENDING; counts NaN -> 0."""
    import netCDF4
    counts = {} if counts is None else counts
    ds = netCDF4.Dataset(path)
    try:
        missing = [v for v in list(L3_VARS.values()) + ["lat", "lon"]
                   if v not in ds.variables]
        if missing:
            raise FormatError(f"the L3 file lacks {missing}; it holds "
                              f"{sorted(ds.variables)}")
        flip, info = check_axes(ds.variables["lat"][:],
                                ds.variables["lon"][:], full_h, full_w)
        out = {}
        for k, name in L3_VARS.items():
            v = ds.variables[name]
            if tuple(v.shape) != (HOURS, int(full_h), int(full_w)):
                raise FormatError(f"{name} is {tuple(v.shape)}; expected "
                                  f"(24, {full_h}, {full_w})")
            dims = tuple(v.dimensions)
            if len(dims) == 3 and not (dims[1].startswith("lat")
                                       and dims[2].startswith("lon")):
                raise FormatError(f"{name} has dimensions {dims}; expected "
                                  f"(time, lat, lon)")
            a = _as_float(v, FILL_I if k.endswith("_n") else FILL_F)
            if flip:
                a = a[:, ::-1, :]
            if k.endswith("_n"):
                a = np.where(np.isfinite(a), a, 0.0)
            out[k] = a
    finally:
        ds.close()
    if flip:
        counts["lat_axis_flipped"] = counts.get("lat_axis_flipped", 0) + 1
    gm = counts.setdefault("grid_meta", {})
    gm[json.dumps(info, sort_keys=True)[:200]] = 1
    return out


def combine_hours(h, counts=None):
    """24 hourly L3 bins -> the day's float64 [H, W, 4] (producer columns).

    Inverse-variance over the VALID hours of each quantity; a quantity's hour
    is valid when its value is finite, its uncertainty finite and > 0 and its
    sample count > 0. An hour with a value but no usable uncertainty or count
    (or the reverse) cannot be weighted and is EXCLUDED and COUNTED.
    """
    counts = {} if counts is None else counts
    out = []
    for vk, uk, nk, tag in (("mss", "mss_u", "mss_n", "mss"),
                            ("ws", "ws_u", "ws_n", "wind")):
        v, u, n = h[vk], h[uk], h[nk]
        ok = np.isfinite(v) & np.isfinite(u) & (u > 0) & (n > 0)
        any_ = np.isfinite(v) | (n > 0)
        bad = int((any_ & ~ok).sum())
        if bad:
            counts[f"hours_{tag}_unweightable"] = \
                counts.get(f"hours_{tag}_unweightable", 0) + bad
        counts[f"hours_{tag}_used"] = counts.get(f"hours_{tag}_used", 0) + \
            int(ok.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(ok, 1.0 / np.where(ok, u, 1.0) ** 2, 0.0)
            S = w.sum(axis=0)
            mean = (w * np.where(ok, v, 0.0)).sum(axis=0) / S
            unc = 1.0 / np.sqrt(S)
        has = S > 0
        mean[~has] = np.nan
        unc[~has] = np.nan
        nsum = np.where(ok, n, 0.0).sum(axis=0)
        nsum = np.where(has, nsum, np.nan)
        if tag == "mss":
            out += [mean, unc, nsum]
        else:
            out.append(mean)
    return np.stack(out, axis=-1)


# =================================================================== F1 ====
def _sample_seconds_of_day(units, st, day):
    """`seconds since YYYY-MM-DD HH:MM:SS[.fffffffff]` + values -> seconds
    since `day` 00:00 UTC (float64)."""
    m = re.match(r"\s*seconds since (\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})"
                 r":(\d{2})(\.\d+)?", str(units))
    if not m:
        raise FormatError(f"sample_time units {units!r} are not 'seconds "
                          f"since <date time>'")
    base = dt.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                       int(m.group(4)), int(m.group(5)), int(m.group(6)))
    off = (base - dt.datetime(day.year, day.month, day.day)).total_seconds()
    off += float(m.group(7) or 0.0)
    return np.asarray(st, np.float64) + off


def read_l2(path, day, with_wind=True):
    """One L2 day -> dict of 1-D float64 arrays (NaN for fill)."""
    import netCDF4
    ds = netCDF4.Dataset(path)
    try:
        want = L2_VARS + (L2_WIND_VARS if with_wind else ())
        missing = [v for v in want if v not in ds.variables]
        if missing:
            raise FormatError(f"the L2 file lacks {missing}")
        st = ds.variables["sample_time"]
        st.set_auto_mask(False)
        out = {"sod": _sample_seconds_of_day(getattr(st, "units", ""),
                                             st[:], day)}
        for v in want[1:]:
            # every L2 variable read here has fill -9999 (CMR UMM-Var:
            # -9999.0 for the floats, -9999 for the int32 flag words)
            out[v] = _as_float(ds.variables[v], FILL_F)
    finally:
        ds.close()
    return out


def _cells(s, full_h, full_w):
    """L2 sample positions -> (cell id in [24, H, W] producer order, ok)."""
    d = (LAT_N - LAT_S) / full_h
    with np.errstate(invalid="ignore"):
        hr = np.floor(s["sod"] / 3600.0)
        i = np.floor((s["lat"] - LAT_S) / d)
        j = np.floor((s["lon"] % 360.0) / (360.0 / full_w))
    ok = (np.isfinite(hr) & (hr >= 0) & (hr < HOURS) & np.isfinite(i)
          & (i >= 0) & (i < full_h) & np.isfinite(j) & (j >= 0)
          & (j < full_w))
    cid = np.zeros(hr.shape, np.int64)
    cid[ok] = ((hr[ok].astype(np.int64) * full_h + i[ok].astype(np.int64))
               * full_w + j[ok].astype(np.int64))
    return cid, ok


def _bin(cid, sel, w, v, ncell):
    n = np.bincount(cid[sel], minlength=ncell).astype(np.float64)
    W = np.bincount(cid[sel], weights=w[sel], minlength=ncell)
    WV = np.bincount(cid[sel], weights=(w * v)[sel], minlength=ncell)
    V = np.bincount(cid[sel], weights=v[sel], minlength=ncell)
    S2 = np.bincount(cid[sel], weights=(1.0 / w)[sel], minlength=ncell)
    return n, W, WV, V, S2


def _rel(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(a - b) / np.maximum(np.abs(b), 1e-30)


def _q(x):
    if x.size == 0:
        return None
    return {"median": float(np.median(x)), "p99": float(np.quantile(x, .99)),
            "p999": float(np.quantile(x, .999)), "max": float(x.max())}


def _compare(n2, mean2, unc2, n3, mean3, unc3, alts=None):
    """One level (hourly or daily) of the F1 comparison; numbers only."""
    union = (n2 > 0) | (n3 > 0)
    nu = int(union.sum())
    eq = union & (n2 == n3)
    both = eq & (n2 > 0)
    nb = int(both.sum())
    rm = _rel(mean2[both], mean3[both])
    ru = _rel(unc2[both], unc3[both])
    out = {
        "cells_nonempty": nu,
        "cells_l2_only": int(((n2 > 0) & ~(n3 > 0)).sum()),
        "cells_l3_only": int(((n3 > 0) & ~(n2 > 0)).sum()),
        "count_equal_fraction": (float(eq.sum()) / nu) if nu else None,
        "cells_compared": nb,
        "mean_within_rel_fraction": (float((rm <= F1_REL).mean())
                                     if nb else None),
        "mean_rel_error": _q(rm),
        "uncertainty_within_rel_fraction": (float((ru <= F1_REL).mean())
                                            if nb else None),
        "uncertainty_rel_error": _q(ru),
    }
    if nu:
        d = np.abs(n2[union] - n3[union])
        out["count_abs_diff"] = {"mean": float(d.mean()),
                                 "max": float(d.max())}
    for nm, (arr, ref) in (alts or {}).items():
        r = _rel(arr[both], ref[both])
        out.setdefault("alternatives", {})[nm] = {
            "within_rel_fraction": float((r <= F1_REL).mean()) if nb
            else None, "rel_error": _q(r)}
    return out


def _passes(level):
    return bool(level["cells_compared"] > 0
                and (level["count_equal_fraction"] or 0) >= F1_COUNT_FRAC
                and (level["mean_within_rel_fraction"] or 0) >= F1_VALUE_FRAC
                and (level["uncertainty_within_rel_fraction"] or 0)
                >= F1_VALUE_FRAC)


def f1_day(l2_path, l3_path, day, full_h=FULL_H, full_w=FULL_W):
    """F1 for one day -> a JSON-able report (see the module docstring)."""
    s = read_l2(l2_path, day)
    h = read_l3(l3_path, full_h, full_w)
    ncell = HOURS * full_h * full_w
    cid, inside = _cells(s, full_h, full_w)
    m, u, fl = (s["mean_square_slope"], s["mean_square_slope_uncertainty"],
                s["mss_sample_flags"])
    flags = np.where(np.isfinite(fl), fl, 0).astype(np.int64)
    usable = inside & np.isfinite(m) & np.isfinite(u) & (u > 0)
    good = usable & ((flags & MSS_FATAL) == 0) & np.isfinite(fl)
    w = np.where(usable, 1.0 / np.where(usable, u, 1.0) ** 2, 1.0)
    mz = np.where(usable, m, 0.0)
    n2, W2, WV2, V2, S22 = _bin(cid, good, w, mz, ncell)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean2 = WV2 / W2
        unc2 = 1.0 / np.sqrt(W2)
        mean_unw = V2 / n2
        unc_rms = np.sqrt(S22 / n2) / np.sqrt(n2)
    shape = (HOURS, full_h, full_w)
    n3 = h["mss_n"].reshape(-1)
    mean3, unc3 = h["mss"].reshape(-1), h["mss_u"].reshape(-1)
    # counts under other flag rules, to name a different L3 selection
    n_all = np.bincount(cid[usable], minlength=ncell).astype(np.float64)
    n_noflag = np.bincount(cid[usable & (flags == 0)],
                           minlength=ncell).astype(np.float64)
    hourly = _compare(n2, mean2, unc2, n3, mean3, unc3,
                      alts={"mean_unweighted_vs_l3": (mean_unw, mean3),
                            "uncertainty_rms_over_sqrt_n_vs_l3":
                                (unc_rms, unc3)})
    un = (n3 > 0) | (n_all > 0)
    un2 = (n3 > 0) | (n_noflag > 0)
    hourly["count_equal_fraction_if_no_flag_filter"] = \
        float((n_all[un] == n3[un]).mean()) if un.any() else None
    hourly["count_equal_fraction_if_any_flag_excluded"] = \
        float((n_noflag[un2] == n3[un2]).mean()) if un2.any() else None
    # the DAILY level — what this store writes
    nd2 = n2.reshape(shape).sum(axis=0).ravel()
    Wd = W2.reshape(shape).sum(axis=0).ravel()
    WVd = WV2.reshape(shape).sum(axis=0).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        md2, ud2 = WVd / Wd, 1.0 / np.sqrt(Wd)
    day3 = combine_hours(h)
    daily = _compare(nd2, md2, ud2,
                     np.where(np.isfinite(day3[..., 2]), day3[..., 2],
                              0.0).ravel(),
                     day3[..., 0].ravel(), day3[..., 1].ravel())
    # CYGNSS's wind, the same way — information, not gating
    wv, wu, wf = (s["wind_speed"], s["wind_speed_uncertainty"],
                  s["fds_sample_flags"])
    wflags = np.where(np.isfinite(wf), wf, 0).astype(np.int64)
    wus = inside & np.isfinite(wv) & np.isfinite(wu) & (wu > 0)
    wgood = wus & ((wflags & FDS_FATAL) == 0) & np.isfinite(wf)
    ww = np.where(wus, 1.0 / np.where(wus, wu, 1.0) ** 2, 1.0)
    nw, Ww, WVw, _V, _S = _bin(cid, wgood, ww, np.where(wus, wv, 0.0), ncell)
    with np.errstate(divide="ignore", invalid="ignore"):
        wind = _compare(nw, WVw / Ww, 1.0 / np.sqrt(Ww),
                        h["ws_n"].reshape(-1), h["ws"].reshape(-1),
                        h["ws_u"].reshape(-1))
    edge = np.abs(np.mod(s["sod"], 3600.0))
    near_edge = int((good & ((edge < 1.0) | (edge > 3599.0))).sum())
    return {
        "day": str(day),
        "l2_samples": int(m.size),
        "l2_samples_outside_grid_or_day": int((~inside).sum()),
        "l2_samples_unusable": int((inside & ~usable).sum()),
        "l2_samples_fatal_flag": int((usable & ((flags & MSS_FATAL) != 0))
                                     .sum()),
        "l2_samples_nonfatal_flag_kept": int((good & (flags != 0)).sum()),
        "l2_samples_good": int(good.sum()),
        "l2_good_within_1s_of_an_hour_edge": near_edge,
        "hourly": hourly, "daily": daily, "wind_info": wind,
        "passed": bool(_passes(hourly) and _passes(daily)),
    }


def f1_check(pairs, full_h=FULL_H, full_w=FULL_W):
    """pairs = [(day, l2_path, l3_path)] -> the F1 report over all of them."""
    days = [f1_day(l2, l3, d, full_h, full_w) for d, l2, l3 in pairs]
    ok = bool(days) and all(x["passed"] for x in days)
    fails = []
    for x in days:
        for lvl in ("hourly", "daily"):
            L = x[lvl]
            for key, thr in (("count_equal_fraction", F1_COUNT_FRAC),
                             ("mean_within_rel_fraction", F1_VALUE_FRAC),
                             ("uncertainty_within_rel_fraction",
                              F1_VALUE_FRAC)):
                if (L[key] or 0) < thr:
                    fails.append(f"{x['day']} {lvl} {key} = {L[key]} "
                                 f"(needs >= {thr})")
            if not L["cells_compared"]:
                fails.append(f"{x['day']} {lvl}: ZERO cells compared")
    return {"check": "F1 producer reproduction (E-092 §6, cygnss_mss)",
            "criteria": {"count_equal_fraction": F1_COUNT_FRAC,
                         "rel_tolerance": F1_REL,
                         "value_within_fraction": F1_VALUE_FRAC,
                         "good_sample": (f"mss_sample_flags & {MSS_FATAL} == "
                                         f"0 (fatal_composite_mss clear), "
                                         f"finite mss and uncertainty > 0"),
                         "weights": "1 / mean_square_slope_uncertainty^2",
                         "cell": ("hour = floor(seconds of UTC day / 3600), "
                                  "0.2-degree cell by floor from -40 N and "
                                  "0 E")},
            "days": days, "passed": ok, "failures": fails,
            "verdict": ("PASSED — the hourly-to-daily inverse-variance "
                        "combination reproduces the L2 day exactly" if ok
                        else "FAILED — the L3 definition differs from the "
                        "one this store's daily combination assumes; see "
                        "failures and alternatives (E-092 §6: build the "
                        "daily grid from L2 instead)")}


# ================================================================ adapter ==
class CygnssMSSAdapter(sh.GridAdapter):
    store = "cygnss_mss"
    title = ("CYGNSS sea-surface mean-square slope (GPS-reflection roughness) "
             "and its own wind, daily on the producer's 0.2-degree grid, "
             "+-40 latitude, 2018-08 ->")
    family = "13"
    distribution = "public"
    licence = {"name": "NASA Earth Science data policy (open data)",
               "redistribution": "yes",
               # NASA's Earth Science Data and Information Policy: data are
               # "made available ... with no restrictions"; the same footing
               # as pace4k and the other NASA stores (plan D13).
               "redistribution_confirmed": True,
               "derived_works": "free",
               "attribution": (
                   "CYGNSS (2024). CYGNSS Level 3 Science Data Record "
                   "Version 3.2. PO.DAAC, CA, USA. "
                   "doi:10.5067/CYGNS-L3X32 (Univ. of Michigan / NASA)")}
    time_dtype = "int32"
    credentials = ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    channels = CHANNELS
    log2_fp = float(np.log2(0.2 * 111.31949 / 27.83))           # -0.322
    log2_dt = float(np.log2(1.0 / 5.0))                         # -2.322
    per_year = True
    first_year = FIRST_YEAR
    frames_per_bin = 5
    frame_seconds = 86400
    dtype = "float16"
    tile = sh.TILE
    zstd_level = sh.DEFAULT_LEVEL
    note_estimate = {"bytes": 7e9,
                     "what": ("cygnss_mss_measure.json: 400 x 1800 x 4 "
                              "float16 = 5.76 MB raw a day, 0.6-1.1 GB/yr "
                              "zstd (compression not measured), 5-9 GB for "
                              "2018-08 -> 2026")}
    qc_policy = (
        "The PRODUCER's QC is inherited, not redone: an L3 hourly bin is the "
        "inverse-variance mean of the L2 samples the producer kept "
        "(fatal_composite_mss clear). A day-cell combines the hours whose "
        "value is finite, whose uncertainty is finite and > 0 and whose "
        "sample count is > 0 — per quantity, so the wind and the MSS keep "
        "their own valid hours; an hour that has a value but cannot be "
        "weighted is excluded and counted (`hours_*_unweightable`). Values "
        "outside the channel bounds (mss and its uncertainty in [0, 1], "
        "n_mss in [1, 2048], wind in [0, 99] m/s) become NaN and are "
        "counted, never clipped. A file that will not open or whose axes do "
        "not match the declared grid is `ctx.note_absent`, never a smaller "
        "frame. No independent-wind filter and no rain or slick mask is "
        "applied: those are the confounders the consumer must model "
        "(cygnss_mss_measure.json, `confounders`).")
    sources = (f"{CMR}?collection_concept_id={L3_CID} ({L3_SHORT}, anonymous "
               f"listing; data Earthdata-protected at PO.DAAC)",
               f"{CMR}?collection_concept_id={L2_CID} ({L2_SHORT}, F1 only)")
    verified = (
        "2026-10-13 from the sandbox (anonymous CMR only): the L3 V3.2 "
        "collection's 12 UMM-Var records — mean_square_slope, "
        "mean_square_slope_uncertainty, wind_speed, wind_speed_uncertainty "
        "float (time 24, lat 400, lon 1800) fill -9999.0; num_mss_samples, "
        "num_wind_speed_samples int fill -99; lat 400, lon 1800 — the L2 "
        "V3.2 sample variables and the mss_sample_flags bits from data "
        "dictionary 148-0348-13 (bit 1 = fatal_composite_mss), the granule "
        "titles and protected https links of 2019-01-15/16 for both levels, "
        "and CMR granule_size values of ~1e-4 MB for real day files (so "
        "per-year CMR totals are LOWER bounds). NOT measured, and what the "
        "hosted probe settles: the files' own layout (inventoried into "
        "plan.json at index), the lat axis order, the zstd ratio, and F1.")
    notes = ""
    smoke_window = ("2019-01-01", "2019-01-08")
    smoke_probe_month = "2019-01"

    def __init__(self):
        self._smoke = (os.environ.get("CYGNSS_MSS_SMOKE_GRID") or "").strip()
        if self._smoke:
            h, w = (int(x) for x in self._smoke.split(","))
            self.full_h, self.full_w = h, w
        else:
            self.full_h, self.full_w = FULL_H, FULL_W
        self.f1_mode = (os.environ.get("CYGNSS_MSS_F1") or "probe").strip()
        if self.f1_mode not in ("probe", "always", "off"):
            sys.exit(f"CYGNSS_MSS_F1={self.f1_mode!r} is not probe|always|off")
        self.f1_days = [x.strip() for x in
                        (os.environ.get("CYGNSS_MSS_F1_DAYS") or "")
                        .split(",") if x.strip()]
        self._days = None
        self._record = None
        self._session = None
        self._f1 = None
        self._f1_attached = False
        self.notes = (
            "Daily frames are the inverse-variance combination of the L3 "
            "file's 24 hourly bins; F1 (E-092 §6) tests that this equals the "
            "L2 day exactly and runs inside --stage probe. wind_speed is "
            "CYGNSS's own and NOT independent of MSS.")
        if self._smoke:
            self.notes += (f"\nSMOKE GRID: CYGNSS_MSS_SMOKE_GRID shrank the "
                           f"grid to {self.full_h} x {self.full_w}. This is a "
                           f"synthetic store.")

    @property
    def grid(self):
        g = grid(self.full_h, self.full_w)
        if self._smoke:
            g["smoke"] = True
        return g

    def specs(self):
        out = super().specs()
        spec = out[self.store]
        spec["daily_combination"] = {
            "mss": "sum(w_h * mss_h) / sum(w_h), w_h = 1/u_h^2 over valid hours",
            "mss_uncertainty": "1 / sqrt(sum(w_h))",
            "n_mss": "sum(num_mss_samples_h)",
            "wind_speed": ("sum(v_h * ws_h) / sum(v_h), v_h = "
                           "1/wind_speed_uncertainty_h^2 over the wind's own "
                           "valid hours (the producer's minimum-variance "
                           "estimator); NOT independent of mss"),
            "falsifier": "F1, E-092 §6; report in the probe JSON"}
        spec["source"] = {"concept_id": L3_CID, "short_name": L3_SHORT,
                          "doi": L3_DOI}
        return out

    # -------------------------------------------------------------- listing --
    def record(self, ctx):
        if self._record is None:
            if ctx.source_dir:
                days = local_days(os.path.join(ctx.source_dir, self.store,
                                               "L3"), L3_NAME)
                self._record = (min(days), max(days)) if days else None
            else:
                try:
                    self._record = cmr_record(L3_CID, L3_NAME,
                                              attempts=ctx.a.attempts,
                                              count=ctx.count_bytes)
                except (FormatError, IOError) as e:
                    sys.exit(f"REFUSING cygnss_mss: the record's ends could "
                             f"not be listed: {e}")
        return self._record

    def days(self, ctx):
        if self._days is not None:
            return self._days
        if ctx.source_dir:
            root = os.path.join(ctx.source_dir, self.store, "L3")
            out = local_days(root, L3_NAME)
            if not out:
                sys.exit(f"REFUSING cygnss_mss: no L3 file under {root} — "
                         f"the smoke's synthetic archive is missing")
        else:
            lo = (EPOCH + dt.timedelta(seconds=int(ctx.t_lo))).strftime(
                "%Y-%m-%dT%H:%M:%SZ")
            hi = (EPOCH + dt.timedelta(seconds=int(ctx.t_hi))).strftime(
                "%Y-%m-%dT%H:%M:%SZ")
            try:
                out = cmr_days(L3_CID, L3_NAME, lo, hi,
                               attempts=ctx.a.attempts, count=ctx.count_bytes)
            except (FormatError, cm.CMRTruncated) as e:
                sys.exit(f"REFUSING cygnss_mss: {e}")
            rec = self.record(ctx)
            if not out and rec and not (rec[1] < ctx.d_lo or
                                        rec[0] > ctx.d_hi):
                sys.exit(f"REFUSING cygnss_mss: CMR lists no {L3_SHORT} "
                         f"granule between {lo} and {hi}, inside the record "
                         f"{rec} — an empty listing is a refusal")
        self._days = out
        return out

    def record_frames(self, ctx, group):
        rec = self.record(ctx)
        return None if not rec else (rec[1] - rec[0]).days + 1

    def _session_get(self, ctx, entry, sub):
        if ctx.source_dir:
            p = entry["url"]
            if not os.path.exists(p):
                raise f10b._NotFound(p)
            ctx.count_bytes(os.path.getsize(p))
            return p, False
        dest = os.path.join(ctx.scratch, self.store, sub,
                            entry["name"] + ".nc")
        if self._session is None:
            self._session = cm.earthdata_session()
        n, why = cm.earthdata_download(self._session, entry["url"], dest,
                                       attempts=max(1, ctx.a.attempts))
        if n is None:
            raise f10b._NotFound(f"{entry['url']}: {why}")
        return dest, True

    # ---------------------------------------------------------------- F1 ----
    def f1_wanted(self, ctx):
        if self.f1_mode == "off":
            return False
        if self.f1_mode == "always":
            return True
        return getattr(ctx.a, "stage", "") == "probe"

    def run_f1(self, ctx):
        """Download 3 days of L2 + L3, run `f1_check`, record the report."""
        l3 = self.days(ctx)
        if self.f1_days:
            want = [dt.date.fromisoformat(x) for x in self.f1_days]
        else:
            ds = sorted(d for d in l3 if ctx.d_lo <= d <= ctx.d_hi)
            want = sorted({ds[0], ds[len(ds) // 2], ds[-1]}) if ds else []
        if ctx.source_dir:
            l2 = local_days(os.path.join(ctx.source_dir, self.store, "L2"),
                            L2_NAME)
        else:
            lo, hi = min(want), max(want)
            l2 = cmr_days(L2_CID, L2_NAME, f"{lo}T00:00:00Z",
                          f"{hi}T23:59:59Z", attempts=ctx.a.attempts,
                          count=ctx.count_bytes) if want else {}
        pairs, tmp, missing = [], [], []
        try:
            for d in want:
                if d not in l3 or d not in l2:
                    missing.append(f"{d}: L3 {'listed' if d in l3 else 'NOT listed'}"
                                   f", L2 {'listed' if d in l2 else 'NOT listed'}")
                    continue
                p2, t2 = self._session_get(ctx, l2[d], "L2")
                if t2:
                    tmp.append(p2)
                p3, t3 = self._session_get(ctx, l3[d], "L3")
                if t3:
                    tmp.append(p3)
                pairs.append((d, p2, p3))
            rep = f1_check(pairs, self.full_h, self.full_w)
        finally:
            for p in tmp:
                if os.path.exists(p):
                    os.remove(p)
        rep["days_requested"] = [str(d) for d in want]
        rep["days_not_available"] = missing
        if missing:
            rep["passed"] = False
            rep["failures"].append(f"days not available: {missing}")
        if len(pairs) < 3:
            rep["passed"] = False
            rep["failures"].append(f"only {len(pairs)} day(s) compared; F1 "
                                   f"needs 3")
        if not rep["passed"]:
            rep["verdict"] = rep["verdict"].replace("PASSED", "FAILED") \
                if rep["verdict"].startswith("PASSED") else rep["verdict"]
        return rep

    # ------------------------------------------------------------- contract --
    def fetch_preflight(self, ctx):
        if not ctx.source_dir and not cm.earthdata_ready():
            sys.exit(
                "REFUSING cygnss_mss: the CYGNSS files are Earthdata-protected "
                "and this process can authenticate to Earthdata Login in "
                "NEITHER way (no EARTHDATA_USERNAME / EARTHDATA_PASSWORD, no "
                "netrc naming urs.earthdata.nasa.gov). family1-build.yml "
                "gives both on a HOSTED runner only. Nothing has been "
                "fetched.")
        if self.f1_wanted(ctx) and self._f1 is None:
            t0 = time.time()
            try:
                rep = self.run_f1(ctx)
            except (FormatError, IOError, OSError, f10b._NotFound,
                    ValueError) as e:
                rep = {"check": "F1 producer reproduction", "passed": False,
                       "failures": [f"F1 could not run: "
                                    f"{type(e).__name__}: {e}"],
                       "verdict": "NOT RUN — an input could not be read"}
            rep["seconds"] = round(time.time() - t0, 1)
            self._f1 = rep
            p = os.path.join(os.path.abspath(ctx.a.work), "f1",
                             f"f1_{ctx.d_lo}_{ctx.d_hi}.json")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            sh.save_json(p, rep)
            print(f"F1 {'PASSED' if rep['passed'] else 'FAILED'} -> {p}")
            print(json.dumps({k: rep.get(k) for k in ("passed", "failures",
                                                      "verdict")}, indent=1))
            if not rep["passed"]:
                # The probe has no check hook yet; an absence is the one way
                # to make it exit non-zero AFTER writing its JSON. The label
                # says what this is: a failed falsifier, not a lost file.
                ctx.note_absent("F1", "F1 producer reproduction FAILED (not "
                                "a read failure): " + "; ".join(
                                    rep["failures"][:4]))
        return None

    def index(self, ctx):
        days = self.days(ctx)
        rec = self.record(ctx)
        per_year, nb, gaps = {}, 0, []
        for d, e in sorted(days.items()):
            per_year[str(d.year)] = per_year.get(str(d.year), 0) + 1
            nb += e["bytes"]
        if days:
            d = min(days)
            while d <= max(days):
                if d not in days:
                    gaps.append(str(d))
                d += dt.timedelta(days=1)
        out = {"dataset": f"{L3_SHORT} (CYGNSS L3 V3.2 hourly 0.2 degree)",
               "url": f"{CMR}?collection_concept_id={L3_CID}",
               "doi": L3_DOI, "collection": L3_CID,
               "record": [str(rec[0]), str(rec[1])] if rec else None,
               "files": len(days), "files_per_year": per_year,
               "bytes_declared": int(nb),
               "bytes_declared_note": ("CMR granule_size; LOWER bound — "
                                       "real day files are declared ~100 B"),
               "days_absent_in_window": gaps[:60],
               "days_absent_n": len(gaps), "grid": self.grid}
        if not days:
            return out
        key = min(days)
        if getattr(ctx.a, "parts_from_hub", False):
            out["first_file"] = {"name": days[key]["name"], "day": str(key),
                                 "not_read": "parts from the Hub"}
            return out
        try:
            path, tmp = self._session_get(ctx, days[key], "L3")
        except (IOError, f10b._NotFound) as e:
            sys.exit(f"REFUSING cygnss_mss: the first file "
                     f"{days[key]['name']} could not be read: {e}")
        try:
            import netCDF4
            from family1.adapters.sst_acspo02 import inventory
            ds = netCDF4.Dataset(path)
            try:
                inv = inventory(ds)
                dims = {n: int(len(x)) for n, x in ds.dimensions.items()}
                attrs = {k: str(ds.getncattr(k))[:200] for k in ds.ncattrs()}
                try:
                    flip, gmeta = check_axes(ds.variables["lat"][:],
                                             ds.variables["lon"][:],
                                             self.full_h, self.full_w)
                except (FormatError, KeyError) as e:
                    sys.exit(f"REFUSING cygnss_mss: {days[key]['name']}: {e}")
            finally:
                ds.close()
            missing = [v for v in L3_VARS.values() if v not in inv]
            if missing:
                sys.exit(f"REFUSING cygnss_mss: {days[key]['name']} lacks "
                         f"{missing}; it holds {sorted(inv)}")
        finally:
            if tmp and os.path.exists(path):
                os.remove(path)
        out["first_file"] = {"name": days[key]["name"], "date": str(key),
                             "dimensions": dims, "inventory": inv,
                             "global_attributes": attrs,
                             "lat_descending": flip, "grid_checked": gmeta}
        return out

    def fetch_frames(self, ctx, wanted):
        days = self.days(ctx)
        rec = self.record(ctx)
        jobs = []
        for (g, b, f) in wanted:
            d = sh.frame_day(b, f, self.frame_seconds)
            why = None
            if rec is None or d < rec[0]:
                why = "before_record"
            elif d > rec[1]:
                why = "after_record"
            elif d not in days:
                why = "absent_upstream"
            jobs.append((g, b, f, d, why))
        todo = [j for j in jobs if j[4] is None]

        def work(job):
            try:
                return job, self._session_get(ctx, days[job[3]], "L3"), None
            except f10b._NotFound:
                return job, None, "listed in CMR, and 404"
            except IOError as e:
                return job, None, f"{type(e).__name__}: {e}"

        workers = 1 if ctx.source_dir else WORKERS
        it = cm.ordered_map(work, todo, workers, lookahead=workers + 1)
        t0 = time.time()
        for job in jobs:
            g, b, f, d, why = job
            extra = self._f1_counts()
            if why is not None:
                yield g, b, f, None, {"frame_missing": why, **extra}
                continue
            j2, got, err = next(it)
            if j2[:3] != (g, b, f):
                sys.exit(f"cygnss_mss: the download pool answered {j2[:3]} "
                         f"where {(g, b, f)} was asked for")
            if got is None:
                ctx.note_absent(str(d), f"{days[d]['name']}: {err}")
                continue
            path, tmp = got
            counts = dict(extra)
            try:
                arr = self.frame_from_file(path, counts)
            except FormatError as e:
                ctx.note_absent(str(d), f"{days[d]['name']}: {e}")
                continue
            except (OSError, IOError) as e:
                ctx.note_absent(str(d), f"{days[d]['name']}: not a readable "
                                        f"netCDF ({type(e).__name__}: {e})")
                continue
            finally:
                if tmp and os.path.exists(path):
                    os.remove(path)
            counts["files_read"] = 1
            counts["fetch_seconds"] = round(time.time() - t0, 2)
            t0 = time.time()
            yield g, b, f, arr, counts

    def _f1_counts(self):
        """The F1 report rides on the FIRST yielded item's counts (a list, so
        `_merge_counts` concatenates rather than sums it)."""
        if self._f1 is None or self._f1_attached:
            return {}
        self._f1_attached = True
        return {"f1_producer_reproduction": [self._f1]}

    def frame_from_file(self, path, counts):
        h = read_l3(path, self.full_h, self.full_w, counts)
        day = combine_hours(h, counts)
        day = to_store_cols(day, self.full_w)
        # `mask_bounds` reads an (n, C) table: a VIEW, so the mask lands
        oob = self.mask_bounds(day.reshape(-1, len(CHANNELS)))
        if oob:
            counts["out_of_bounds"] = oob
        u = day[..., 1]
        counts["mss_uncertainty_subnormal_f16"] = counts.get(
            "mss_uncertainty_subnormal_f16", 0) + int(
            (np.isfinite(u) & (u < F16_MIN_NORMAL)).sum())
        counts["pixels_mss"] = counts.get("pixels_mss", 0) + \
            int(np.isfinite(day[..., 0]).sum())
        counts["pixels_wind"] = counts.get("pixels_wind", 0) + \
            int(np.isfinite(day[..., 3]).sum())
        return day.astype(np.float16)

    # ---------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20261013):
        from family1.adapters import _cygnss_mss_smoke as smk
        truth = smk.make_smoke_sources(root, d_lo, d_hi, seed)
        self.__init__()
        return truth


ADAPTER = CygnssMSSAdapter
