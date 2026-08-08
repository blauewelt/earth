#!/usr/bin/env python3
"""Family-3 tensor: the 0.25-degree North Atlantic — ml/cache/family3_na025.npz.

The third channel family (HANDBOOK.md §5). Never cross-compare with
families 1/2 — new grid, new channels, new baselines.

Recipe (fixed here, not by flags — the recipe IS the family):
  1. Base: base025_na.npz (cur_speed, log_mld, ssh; CMEMS 0.25-degree
     GLORYS member, NA window, 1993-01..2024-12). Auto-downloaded from
     the data-cache-v1 release when absent — no CMEMS credentials needed.
  2. Time axis extended to 1982-01 (build_dataset.py --start-year
     pattern): pre-1993 base months are missing tokens; wind fills them,
     and the Florida cable's 1982-92 decade becomes usable truth.
  3. RG-Argo T/S at 16 pressure levels, bilinearly upsampled from RG's
     native 1-degree grid. NOTE: RG is intrinsically 1-degree-smooth —
     the upsample adds no sub-degree information, it only aligns the
     grid; sub-degree structure in family 3 comes from the base channels.
     Coverage-preserving bilinear: corner weights renormalise over the
     finite parents, so 0.25-degree coastal cells inside a partly-NaN
     1-degree neighbourhood keep a value instead of inheriting NaN.
  4. NCEP R1 wind stress: monthly mean (tau_x, tau_y; sign flipped so
     positive = stress ON the ocean, as fetch_wind_channels.py) AND
     within-month std (tau_x_std, tau_y_std) — BOTH computed from the
     daily gaussian-grid files (one source; the release seeds 1993+,
     1982-92 fetched from PSL once and cached), bilinear onto 0.25.
  5. Truth series: rapid (month-index pairs) + truth_fc/osnap/move/samba
     via fetch_truth.py's fetchers, attached to the extended axis.

Memory: the dense tensor is [516, 281, 481, 39] float32 ~ 10.9 GB — far
over the 7 GB sandbox, so X lives in a .npy memmap and every whole-tensor
pass (ocean mask, stats, z-score) walks per-month slabs (~21 MB each).
Boxes (64 GB) can afford np.load, but the memmap costs them nothing.

Chinchilla bookkeeping (standing directive): the build counts observed
values per channel group and prints total/20 — record it in SCALING.md
and HANDBOOK.md after every data change.

Run:  python3 ml/build_family3.py            # ~15 min, needs the caches
      python3 ml/build_family3.py --prune-sources   # sandbox: delete each
             source cache after its channels land (NEVER on a box — the
             cache dirs there are shared with family-2 builds)
"""
import argparse
import glob
import gzip
import json
import os
import shutil
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(HERE, "cache")
BASE_NPZ = os.path.join(CACHE, "base025_na.npz")
OUT_NPZ = os.path.join(CACHE, "family3_na025.npz")
MEMMAP = os.path.join(CACHE, "family3_na025_build.npy")
RELEASE = "https://github.com/blauewelt/earth/releases/download/data-cache-v1"
START_YEAR = 1982

# The 16 AMOC-relevant pressures (dbar) — data-ladder rung (c), fixed for
# family 3. Same list as fetch_rg_channels.py at EARTH_RG_LEVELS=16.
LEVELS = [10.0, 30.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0,
          500.0, 700.0, 900.0, 1100.0, 1300.0, 1500.0, 1700.0, 1900.0]

CHANS = (["cur_speed", "log_mld", "ssh"]
         + [f"rg_t{int(p)}" for p in LEVELS]
         + [f"rg_s{int(p)}" for p in LEVELS]
         + ["tau_x", "tau_y", "tau_x_std", "tau_y_std"])
C_BASE, C_RG, C_WIND = 3, 2 * len(LEVELS), 4
NC = C_BASE + C_RG + C_WIND


# ---------------------------------------------------------------- interp --
def lin_weights(src, dst, wrap_period=None):
    """1-D linear interpolation indices/weights from axis `src` (monotonic,
    either direction, possibly non-uniform — NCEP's gaussian latitudes) to
    points `dst`. Returns (i0, i1, w1) with f(dst) = (1-w1)*f[i0]+w1*f[i1].
    Ends clamp; wrap_period (e.g. 360 for longitudes) interpolates across
    the seam instead."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    flip = src[0] > src[-1]
    s = src[::-1] if flip else src
    if wrap_period is not None:
        dst = s[0] + (dst - s[0]) % wrap_period
        s = np.concatenate([s, [s[0] + wrap_period]])
    j = np.searchsorted(s, dst, side="right") - 1
    j = np.clip(j, 0, len(s) - 2)
    w1 = (dst - s[j]) / (s[j + 1] - s[j])
    w1 = np.clip(w1, 0.0, 1.0)
    i0, i1 = j, j + 1
    if wrap_period is not None:
        n = len(src)
        i0, i1 = i0 % n, i1 % n
    if flip:
        n = len(src)
        i0, i1 = n - 1 - i0, n - 1 - i1
    return i0.astype(np.int64), i1.astype(np.int64), w1.astype(np.float32)


def interp2_nan(f, wy, wx):
    """NaN-aware bilinear of 2-D field f via precomputed per-axis weights
    (from lin_weights). Corner weights renormalise over finite parents;
    all-NaN neighbourhoods stay NaN."""
    (iy0, iy1, vy), (ix0, ix1, vx) = wy, wx
    num = np.zeros((len(iy0), len(ix0)), dtype=np.float64)
    den = np.zeros_like(num)
    for iy, ay in ((iy0, 1.0 - vy), (iy1, vy)):
        for ix, ax in ((ix0, 1.0 - vx), (ix1, vx)):
            corner = f[iy][:, ix]
            w = np.outer(ay, ax)
            fin = np.isfinite(corner)
            num += np.where(fin, corner, 0.0) * w * fin
            den += w * fin
    with np.errstate(invalid="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out.astype(np.float32)


def fetch(url, path, attempts=4, mirrors=()):
    """Download url -> path. `mirrors` are alternate URLs for the same file;
    attempts cycle through [url, *mirrors] so a dead host is skipped rather
    than hammered — downloads.psl.noaa.gov 504'd every retry of run #47 on
    2026-08-08 while the thredds mirror on psl.noaa.gov served fine."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ua = {"User-Agent": "earth-science-pipeline/1.0 (research; github blauewelt/earth)"}
    urls = [url, *mirrors]
    for i in range(attempts):
        u = urls[i % len(urls)]
        try:
            print(f"  downloading {u}" + (f" (attempt {i + 1})" if i else ""), flush=True)
            req = urllib.request.Request(u, headers=ua)
            with urllib.request.urlopen(req, timeout=600) as r, \
                    open(path + ".part", "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            os.rename(path + ".part", path)
            return path
        except Exception as e:
            if i == attempts - 1:
                raise
            wait = 30 * (2 ** i)
            print(f"  fetch failed ({e}); retrying in {wait}s", flush=True)
            import time as _t
            _t.sleep(wait)


# ------------------------------------------------------------------ base --
def load_base():
    if not os.path.exists(BASE_NPZ):
        print("base025_na.npz missing — fetching from the data-cache-v1 release")
        fetch(f"{RELEASE}/base025_na.npz", BASE_NPZ)
    d = np.load(BASE_NPZ)
    return d["X"], [str(m) for m in d["months"]], d["lats"], d["lons"]


# -------------------------------------------------------------------- rg --
def rg_file(stem):
    """Path to an RG NetCDF in the cache: prefer extracted .nc, gunzip a
    cached .gz, else None."""
    nc = os.path.join(CACHE, "rg", stem + ".nc")
    gz = nc + ".gz"
    if os.path.exists(nc):
        return nc
    if os.path.exists(gz):
        with gzip.open(gz, "rb") as fin, open(nc, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        return nc
    return None


def fill_rg(X, months, lats, lons):
    """RG T/S at LEVELS, bilinear 1-degree -> 0.25, into channels 3..34."""
    import netCDF4 as ncdf
    tf, sf = rg_file("RG_T"), rg_file("RG_S")
    if not tf or not sf:
        print("  RG base cubes not cached — fetching from SIO (flaky origin!)")
        base = "https://sio-argo.ucsd.edu/pub/www-argo/RG"
        fetch(f"{base}/RG_ArgoClim_Temperature_2019.nc.gz",
              os.path.join(CACHE, "rg", "RG_T.nc.gz"))
        fetch(f"{base}/RG_ArgoClim_Salinity_2019.nc.gz",
              os.path.join(CACHE, "rg", "RG_S.nc.gz"))
        tf, sf = rg_file("RG_T"), rg_file("RG_S")
    dT, dS = ncdf.Dataset(tf), ncdf.Dataset(sf)
    press = np.array(dT.variables["PRESSURE"][:])
    rg_lat = np.array(dT.variables["LATITUDE"][:])
    rg_lon = np.array(dT.variables["LONGITUDE"][:])       # 20.5 .. 379.5
    lidx = [int(np.argmin(np.abs(press - p))) for p in LEVELS]
    L = len(LEVELS)

    wy = lin_weights(rg_lat, lats)
    lon360 = np.where(lons < 20.0, lons + 360.0, lons)
    wx = lin_weights(rg_lon, lon360, wrap_period=360.0)

    mean_t = np.ma.filled(dT.variables["ARGO_TEMPERATURE_MEAN"][lidx], np.nan)
    mean_s = np.ma.filled(dS.variables["ARGO_SALINITY_MEAN"][lidx], np.nan)
    midx = {m: i for i, m in enumerate(months)}

    def write_month(t, at, as_):
        """at/as_: (L, 145, 360) anomaly slabs -> X[t, :, :, 3:35]."""
        for k in range(L):
            X[t, :, :, C_BASE + k] = interp2_nan(mean_t[k] + at[k], wy, wx)
            X[t, :, :, C_BASE + L + k] = interp2_nan(mean_s[k] + as_[k], wy, wx)

    print("  RG base cube (2004-01..2018-12) …", flush=True)
    base_months = [f"{2004 + k // 12}-{k % 12 + 1:02d}" for k in range(180)]
    anom_t, anom_s = dT.variables["ARGO_TEMPERATURE_ANOMALY"], \
        dS.variables["ARGO_SALINITY_ANOMALY"]
    n = 0
    for k, m in enumerate(base_months):
        if m in midx:
            write_month(midx[m], np.ma.filled(anom_t[k][lidx], np.nan),
                        np.ma.filled(anom_s[k][lidx], np.nan))
            n += 1
    dT.close(), dS.close()

    # Extension months from the LOCAL cache only — enumerating them by
    # scraping the SIO index took down five runs in one day (HANDBOOK §3);
    # the data-cache release defines the extension set now. A month the
    # release lacks is simply absent, loudly.
    exts = sorted(set(
        os.path.basename(p).split(".")[0].split("_")[1]
        for p in glob.glob(os.path.join(CACHE, "rg", "RG_2*.nc*"))))
    print(f"  RG extensions in cache: {len(exts)}"
          + (f" ({exts[0]}..{exts[-1]})" if exts else " — NONE (2019+ RG missing!)"),
          flush=True)
    for ym in exts:
        m = f"{ym[:4]}-{ym[4:]}"
        if m not in midx:
            continue
        f = rg_file(f"RG_{ym}")
        dE = ncdf.Dataset(f)
        write_month(midx[m],
                    np.ma.filled(dE.variables["ARGO_TEMPERATURE_ANOMALY"][0][lidx], np.nan),
                    np.ma.filled(dE.variables["ARGO_SALINITY_ANOMALY"][0][lidx], np.nan))
        dE.close()
        n += 1
    print(f"  RG: {n} months written")


# ------------------------------------------------------------------ wind --
def fill_wind(X, months, lats, lons):
    """NCEP R1 daily momentum flux -> monthly mean (negated: stress ON the
    ocean) + within-month std, bilinear gaussian -> 0.25, channels 35..38."""
    import netCDF4 as ncdf
    daily = os.path.join(CACHE, "wind_daily")
    psl = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/surface_gauss"
    midx = {m: i for i, m in enumerate(months)}
    years = sorted({int(m[:4]) for m in months})
    wy = wx = None
    n = 0
    for y in years:
        fields = {}
        for var in ("uflx", "vflx"):
            path = os.path.join(daily, f"{var}.sfc.gauss.{y}.nc")
            if not os.path.exists(path):
                thredds = ("https://psl.noaa.gov/thredds/fileServer/Datasets"
                           "/ncep.reanalysis/surface_gauss")
                fetch(f"{psl}/{var}.sfc.gauss.{y}.nc", path,
                      mirrors=(f"{thredds}/{var}.sfc.gauss.{y}.nc",))
            nc = ncdf.Dataset(path)
            if wy is None:
                g_lat = np.array(nc.variables["lat"][:])   # descending
                g_lon = np.array(nc.variables["lon"][:])   # 0..360 cyclic
                wy = lin_weights(g_lat, lats)
                lon360 = np.where(lons < 0, lons + 360.0, lons)
                wx = lin_weights(g_lon, lon360, wrap_period=360.0)
            tv = nc.variables["time"]
            dates = ncdf.num2date(tv[:], tv.units, only_use_cftime_datetimes=False)
            momo = np.array([d.month for d in dates])
            fields[var] = (momo, np.ma.filled(nc.variables[var][:], np.nan))
            nc.close()
        for mo in range(1, 13):
            key = f"{y}-{mo:02d}"
            if key not in midx:
                continue
            t = midx[key]
            ok = True
            for ci, var in enumerate(("uflx", "vflx")):
                momo, vals = fields[var]
                sel = momo == mo
                if sel.sum() < 20:            # incomplete month: no channel
                    ok = False
                    break
                X[t, :, :, C_BASE + C_RG + ci] = interp2_nan(
                    -np.nanmean(vals[sel], axis=0), wy, wx)
                X[t, :, :, C_BASE + C_RG + 2 + ci] = interp2_nan(
                    np.nanstd(vals[sel], axis=0), wy, wx)
            n += ok
        print(f"  wind {y}: done ({n} months so far)", flush=True)
    print(f"  wind: {n} months written")


# ----------------------------------------------------------------- truth --
def truth_series(months):
    """rapid (month-index pairs, build_dataset contract) + truth_* arrays
    (yyyymm pairs, fetch_truth contract) on the extended axis."""
    out = {}
    idx = {m: i for i, m in enumerate(months)}
    r = json.load(open(os.path.join(ROOT, "data", "rapid_moc.json")))
    acc = {}
    for tstr, v in zip(r["t"], r["moc"]):
        k = str(tstr)[:7]
        if k in idx and v is not None:
            acc.setdefault(k, []).append(float(v))
    out["rapid"] = np.array([(idx[k], float(np.mean(vs)))
                             for k, vs in sorted(acc.items())],
                            dtype=np.float32).reshape(-1, 2)
    print(f"  rapid: {len(out['rapid'])} monthly means")
    sys.path.insert(0, HERE)
    import fetch_truth
    os.makedirs(fetch_truth.TRUTH, exist_ok=True)
    covered = {int(m[:4]) * 100 + int(m[5:7]) for m in months}
    for name, fn in (("fc", fetch_truth.fc), ("osnap", fetch_truth.osnap),
                     ("move", fetch_truth.move), ("samba", fetch_truth.samba)):
        try:
            arr = fn()
            out[f"truth_{name}"] = arr
            inside = int(sum(1 for ym in arr[:, 0] if int(ym) in covered))
            print(f"  truth_{name}: {len(arr)} months, {inside} inside the axis")
        except Exception as e:                 # a dead archive must not kill the build
            print(f"  truth_{name}: UNAVAILABLE ({str(e)[:100]})")
    return out


# ------------------------------------------------------------------ main --
RECIPE_REV = "f3r1"     # bump on ANY recipe change — it is the skip guard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prune-sources", action="store_true",
                    help="delete rg/ and wind_daily/ caches after use "
                         "(sandbox disk; NEVER on a box — shared cache)")
    ap.add_argument("--out", default=OUT_NPZ)
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if a same-recipe tensor exists")
    a = ap.parse_args()

    # The boxes keep ml/cache persistent, and this build costs ~15 min —
    # skip it when the cached tensor was built by THIS recipe revision.
    # (build_dataset.py rebuilds every run because its build is cheap.)
    if not a.force and os.path.exists(a.out):
        try:
            prev = str(np.load(a.out)["recipe"])
        except Exception:
            prev = "unreadable"
        if prev == RECIPE_REV:
            print(f"{a.out} already built by recipe {RECIPE_REV} — skipping "
                  f"(--force to rebuild)")
            return
        print(f"cached tensor is recipe {prev!r}, want {RECIPE_REV!r} — rebuilding")

    print("loading base025_na.npz …", flush=True)
    baseX, base_months, lats, lons = load_base()
    H, W = len(lats), len(lons)
    pre = [f"{y:04d}-{m:02d}" for y in range(START_YEAR, int(base_months[0][:4]))
           for m in range(1, 13)]
    months = pre + base_months
    T = len(months)
    print(f"axis {months[0]}..{months[-1]} T={T} ({len(pre)} pre-GLORYS) · "
          f"grid {H}x{W} · C={NC}")

    X = np.lib.format.open_memmap(MEMMAP, mode="w+", dtype=np.float32,
                                  shape=(T, H, W, NC))
    print("initialising to NaN …", flush=True)
    for t in range(T):
        X[t] = np.nan

    print("base channels …", flush=True)
    t0 = len(pre)
    for i in range(baseX.shape[0]):
        X[t0 + i, :, :, :C_BASE] = baseX[i]
    ocean = np.zeros((H, W), dtype=bool)
    for i in range(baseX.shape[0]):            # slab-wise: no 620 MB reduce
        ocean |= np.isfinite(baseX[i, :, :, 0])
    del baseX
    print(f"  ocean cells: {int(ocean.sum())}/{H * W}")

    print("RG channels (16 levels, bilinear from 1 degree) …", flush=True)
    fill_rg(X, months, lats, lons)
    if a.prune_sources:
        shutil.rmtree(os.path.join(CACHE, "rg"), ignore_errors=True)
        print("  pruned rg/")

    print("wind channels (NCEP daily -> monthly mean + std) …", flush=True)
    fill_wind(X, months, lats, lons)
    if a.prune_sources:
        shutil.rmtree(os.path.join(CACHE, "wind_daily"), ignore_errors=True)
        print("  pruned wind_daily/")

    print("truth series …", flush=True)
    truths = truth_series(months)

    # ---- ocean mask + per-channel stats, one slab pass -------------------
    print("mask + stats pass …", flush=True)
    cnt = np.zeros(NC, dtype=np.int64)
    s1 = np.zeros(NC, dtype=np.float64)
    s2 = np.zeros(NC, dtype=np.float64)
    for t in range(T):
        slab = np.array(X[t])                  # (H, W, C) ~ 21 MB
        slab[~ocean] = np.nan
        X[t] = slab
        fin = np.isfinite(slab)
        cnt += fin.sum((0, 1))
        v = np.where(fin, slab, 0.0)
        s1 += v.sum((0, 1), dtype=np.float64)
        s2 += (v.astype(np.float64) ** 2).sum((0, 1))
    mu = np.where(cnt > 0, s1 / np.maximum(cnt, 1), 0.0)
    sd = np.sqrt(np.maximum(s2 / np.maximum(cnt, 1) - mu ** 2, 0.0)) + 1e-6
    norm = np.stack([mu, sd], 1).astype(np.float32)

    print("z-score pass …", flush=True)
    muf, sdf = mu.astype(np.float32), sd.astype(np.float32)
    for t in range(T):
        X[t] = (X[t] - muf) / sdf
    X.flush()

    # ---- Chinchilla bookkeeping (standing directive) ---------------------
    groups = (("base", 0, C_BASE), ("rg", C_BASE, C_BASE + C_RG),
              ("wind", C_BASE + C_RG, NC))
    total = int(cnt.sum())
    print("\nobserved values (the Chinchilla inventory):")
    for name, lo, hi in groups:
        print(f"  {name:<5} {int(cnt[lo:hi].sum()):>13,}")
    print(f"  TOTAL {total:>13,}  ->  params anchor ~ {total / 20 / 1e6:.1f} M "
          f"(values/20; record in SCALING.md + HANDBOOK.md)")
    for c in range(NC):
        print(f"    {CHANS[c]:<10} coverage {cnt[c] / (T * int(ocean.sum())):6.1%}"
              f"  mu {mu[c]:9.3f}  sd {sd[c]:8.3f}")

    print(f"\nwriting {a.out} (streams from the memmap) …", flush=True)
    np.savez_compressed(
        a.out, X=X, months=np.array(months), lats=lats.astype(np.float32),
        lons=lons.astype(np.float32), chan=np.array(CHANS), norm=norm,
        window=np.array("na025"), recipe=np.array(RECIPE_REV), **truths)
    del X
    os.remove(MEMMAP)
    print(f"wrote {a.out}  [T={T} H={H} W={W} C={NC}]  "
          f"{os.path.getsize(a.out) / 1e9:.2f} GB")


if __name__ == "__main__":
    sys.exit(main())
