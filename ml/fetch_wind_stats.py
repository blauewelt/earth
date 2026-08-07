#!/usr/bin/env python3
"""Within-month wind variability — sub-monthly information as channels.

The 2009-10 case study's residual lesson: the November onset of the AMOC
collapse was triggered INSIDE a month, invisible to monthly means. This
fetcher adds the cheapest sub-monthly signal that exists in a
credential-free archive: the per-month STANDARD DEVIATION of daily NCEP
R1 momentum flux — storminess. Two channels, `tau_x_std` and
`tau_y_std` (z-scored, land-masked), appended re-runnably.

Daily source files (one per year per component, ~7 MB each):
  https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/Datasets/
      ncep.reanalysis/surface_gauss/uflx.sfc.gauss.YYYY.nc
Downloads cache in ml/cache/wind_daily/ (~0.5 GB for 1993->present);
the computed monthly-std cube caches as wind_std_monthly.npz so the
expensive pass runs once.

Run AFTER fetch_wind_channels:  python3 ml/fetch_wind_stats.py
Optional: --download-only (prefetch the daily files, compute nothing).
"""
import argparse
import datetime
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
DAILY = os.path.join(CACHE, "wind_daily")
NPZ = os.path.join(CACHE, "na_pixels.npz")
STATS = os.path.join(CACHE, "wind_std_monthly.npz")
BASE = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/surface_gauss"
UA = {"User-Agent": "earth-science-pipeline/1.0 (research; github blauewelt/earth)"}


def fetch(name, attempts=4):
    path = os.path.join(DAILY, name)
    if os.path.exists(path):
        return path
    for i in range(attempts):
        try:
            print(f"  downloading {BASE}/{name}" + (f" (attempt {i + 1})" if i else ""))
            req = urllib.request.Request(f"{BASE}/{name}", headers=UA)
            with urllib.request.urlopen(req, timeout=600) as r, \
                    open(path + ".part", "wb") as f:
                f.write(r.read())
            os.rename(path + ".part", path)
            return path
        except Exception as e:
            if i == attempts - 1:
                raise
            wait = 30 * (2 ** i)
            print(f"  fetch failed ({e}); retrying in {wait}s")
            time.sleep(wait)


def years_needed(months):
    return sorted({int(m[:4]) for m in months})


def compute_stats(months):
    """Per-month std of daily tau on the NCEP gaussian grid, cached."""
    import netCDF4
    if os.path.exists(STATS):
        d = np.load(STATS)
        have = set(str(m) for m in d["months"])
        if have >= set(months):
            return d
    print("computing monthly std of daily momentum flux …")
    out = {}                                    # month -> [2, ny, nx]
    g_lat = g_lon = None
    for y in years_needed(months):
        arrs = {}
        for comp, var in (("u", "uflx"), ("v", "vflx")):
            path = fetch(f"{var}.sfc.gauss.{y}.nc")
            nc = netCDF4.Dataset(path)
            if g_lat is None:
                g_lat = np.array(nc.variables["lat"][:])
                g_lon = np.array(nc.variables["lon"][:])
            t = nc.variables["time"]
            base = datetime.datetime(1800, 1, 1)
            days = [base + datetime.timedelta(hours=float(h)) for h in t[:]]
            vals = np.ma.filled(nc.variables[var][:], np.nan)
            arrs[comp] = (days, vals)
            nc.close()
        for m in range(1, 13):
            key = f"{y}-{m:02d}"
            if key not in months:
                continue
            stds = []
            for comp in ("u", "v"):
                days, vals = arrs[comp]
                sel = [i for i, d0 in enumerate(days)
                       if d0.year == y and d0.month == m]
                if len(sel) < 20:
                    stds = None
                    break
                stds.append(np.nanstd(vals[sel], axis=0))
            if stds is not None:
                out[key] = np.stack(stds, 0)
        print(f"  {y}: {sum(1 for k in out if k.startswith(str(y)))} months")
    keys = sorted(out)
    np.savez_compressed(STATS, months=np.array(keys),
                        std=np.stack([out[k] for k in keys], 0),
                        lat=g_lat, lon=g_lon)
    return np.load(STATS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(DAILY, exist_ok=True)
    d = dict(np.load(NPZ))
    months = [str(m) for m in d["months"]]
    if a.download_only:
        for y in years_needed(months):
            fetch(f"uflx.sfc.gauss.{y}.nc")
            fetch(f"vflx.sfc.gauss.{y}.nc")
        print("daily files cached")
        return
    st = compute_stats(months)
    X = d["X"]
    chans = [str(c) for c in d["chan"]]
    new = ["tau_x_std", "tau_y_std"]
    if new[0] in chans:                          # re-run: rebuild
        keep = [i for i, c in enumerate(chans) if c not in new]
        X = X[..., keep]
        chans = [chans[i] for i in keep]
        d["norm"] = d["norm"][keep]
    T, H, W, C = X.shape
    lats, lons = d["lats"], d["lons"]
    g_lat, g_lon = st["lat"], st["lon"]          # gaussian, lat descending
    iy = np.array([int(np.argmin(np.abs(g_lat - la))) for la in lats])
    lon360 = np.where(lons < 0, lons + 360.0, lons)
    ix = np.array([int(np.argmin(np.abs(g_lon - lo))) for lo in lon360])
    midx = {str(m): i for i, m in enumerate(st["months"])}
    add = np.full((T, H, W, 2), np.nan, dtype=np.float32)
    for t, m in enumerate(months):
        if m in midx:
            cube = st["std"][midx[m]]            # [2, ny, nx]
            for c in range(2):
                add[t, :, :, c] = cube[c][np.ix_(iy, ix)]
    ocean = np.isfinite(X[..., 0]).any(axis=0)
    add[:, ~ocean, :] = np.nan
    add_norm = np.zeros((2, 2), dtype=np.float32)
    for c in range(2):
        v = add[..., c][np.isfinite(add[..., c])]
        mu, sd = float(v.mean()), float(v.std() + 1e-6)
        add_norm[c] = (mu, sd)
        add[..., c] = (add[..., c] - mu) / sd
        cov = np.isfinite(add[..., c]).mean()
        print(f"  {new[c]:<10} coverage {cov:5.1%}  mu {mu:8.4f}  sd {sd:7.4f}  N/m2")
    d["X"] = np.concatenate([X, add], axis=-1).astype(np.float32)
    d["chan"] = np.array(chans + new)
    d["norm"] = np.concatenate([d["norm"], add_norm], 0)
    np.savez_compressed(NPZ, **d)
    print(f"rewrote {NPZ}: C={d['X'].shape[-1]} channels "
          f"({os.path.getsize(NPZ) / 1e6:.0f} MB)")


if __name__ == "__main__":
    sys.exit(main())
