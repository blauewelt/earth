#!/usr/bin/env python3
"""Add the AMOC's depth structure to the pilot tensor: RG-Argo T/S channels.

Runs where the bandwidth is (a Colab runtime, or any machine) — the downloads
are ~2 GB, which is exactly why these channels are not baked into the repo.
Requires: numpy, netCDF4  (pip install netCDF4).

What it adds and why: the overturning is geostrophy read off the density
field, and density is T/S AT DEPTH — the one thing no surface raster carries.
The Roemmich–Gilson Argo climatology (Scripps, open, no account) provides
monthly T and S on a 1° grid, 2004→now: the 2004–2018 base files carry the
whole 180-month anomaly cube inline, and each month since arrives as one
extension file. This script samples T and S at four AMOC-critical pressures —
10 dbar (surface), 200 (thermocline), 700 (upper-mid), 1500 (deep limb) —
onto the pilot's grid/months and appends 8 channels to na_pixels.npz.

  python3 ml/build_dataset.py            # first: the base tensor (repo data)
  python3 ml/fetch_rg_channels.py        # then: + rg_t10 … rg_s1500
  python3 ml/train.py                    # trains on whatever channels exist

Re-runnable: already-present RG channels are replaced, never duplicated.
"""
import gzip
import os
import re
import shutil
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
NPZ = os.path.join(CACHE, "na_pixels.npz")
RG_BASE = "https://sio-argo.ucsd.edu/pub/www-argo/RG"
UA = {"User-Agent": "earth-globe-ml/1.0 (github.com/blauewelt/earth)"}
LEVELS = [10.0, 200.0, 700.0, 1500.0]          # dbar


def fetch(url, path):
    if os.path.exists(path):
        return path
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r, open(path + ".part", "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.rename(path + ".part", path)
    return path


def gunzip(path):
    out = path[:-3]
    if not os.path.exists(out):
        with gzip.open(path, "rb") as fin, open(out, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    return out


def main():
    import netCDF4 as ncdf

    d = dict(np.load(NPZ, allow_pickle=False))
    months = [str(m) for m in d["months"]]
    lats, lons = d["lats"], d["lons"]
    T = len(months)
    print(f"pilot tensor: {T} months, {len(lats)}x{len(lons)} cells")

    dl = os.path.join(CACHE, "rg")
    os.makedirs(dl, exist_ok=True)

    tf = gunzip(fetch(f"{RG_BASE}/RG_ArgoClim_Temperature_2019.nc.gz",
                      os.path.join(dl, "RG_T.nc.gz")))
    sf = gunzip(fetch(f"{RG_BASE}/RG_ArgoClim_Salinity_2019.nc.gz",
                      os.path.join(dl, "RG_S.nc.gz")))
    dT, dS = ncdf.Dataset(tf), ncdf.Dataset(sf)
    press = np.array(dT.variables["PRESSURE"][:])
    rg_lat = np.array(dT.variables["LATITUDE"][:])
    rg_lon = np.array(dT.variables["LONGITUDE"][:])          # 20.5 .. 379.5
    lidx = [int(np.argmin(np.abs(press - p))) for p in LEVELS]

    # nearest-cell index maps from the pilot grid onto RG's grid
    iy = np.array([int(np.argmin(np.abs(rg_lat - la))) for la in lats])
    lon360 = np.where(lons < 20.0, lons + 360.0, lons)       # RG longitudes start at 20
    ix = np.array([int(np.argmin(np.abs(rg_lon - lo))) for lo in lon360])

    # the RG time axis: 180 base months 2004-01..2018-12, then extensions
    base_months = [f"{2004 + k // 12}-{k % 12 + 1:02d}" for k in range(180)]
    print("scraping the extension index …")
    req = urllib.request.Request("https://sio-argo.ucsd.edu/RG_Climatology.html", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        html = r.read().decode("utf-8", "replace")
    ext_months = sorted(set(re.findall(r"RG_ArgoClim_(\d{6})_2019\.nc\.gz", html)))
    print(f"  base 2004-01..2018-12 + {len(ext_months)} extension months "
          f"(.. {ext_months[-1][:4]}-{ext_months[-1][4:]})")

    def sample(cube_lvl_latlon):
        """(L, 145, 360) masked → (L, H, W) float with NaN, on the pilot grid."""
        a = np.ma.filled(cube_lvl_latlon, np.nan)
        return a[:, iy][:, :, ix]

    mean_t = np.ma.filled(dT.variables["ARGO_TEMPERATURE_MEAN"][lidx], np.nan)[:, iy][:, :, ix]
    mean_s = np.ma.filled(dS.variables["ARGO_SALINITY_MEAN"][lidx], np.nan)[:, iy][:, :, ix]

    chans = [f"rg_t{int(p)}" for p in LEVELS] + [f"rg_s{int(p)}" for p in LEVELS]
    L = len(LEVELS)
    add = np.full((T, len(lats), len(lons), 2 * L), np.nan, dtype=np.float32)
    midx = {m: i for i, m in enumerate(months)}

    # base cube: anomalies are (time, pressure, lat, lon)
    print("sampling the 2004-2018 base cube …")
    anom_t = dT.variables["ARGO_TEMPERATURE_ANOMALY"]
    anom_s = dS.variables["ARGO_SALINITY_ANOMALY"]
    for k, m in enumerate(base_months):
        if m not in midx:
            continue
        t = midx[m]
        add[t, :, :, :L] = np.moveaxis(mean_t + sample(anom_t[k][lidx]), 0, -1)
        add[t, :, :, L:] = np.moveaxis(mean_s + sample(anom_s[k][lidx]), 0, -1)

    print("fetching + sampling extension months …")
    for ym in ext_months:
        m = f"{ym[:4]}-{ym[4:]}"
        if m not in midx:
            continue
        ef = gunzip(fetch(f"{RG_BASE}/RG_ArgoClim_{ym}_2019.nc.gz",
                          os.path.join(dl, f"RG_{ym}.nc.gz")))
        dE = ncdf.Dataset(ef)
        t = midx[m]
        add[t, :, :, :L] = np.moveaxis(
            mean_t + sample(dE.variables["ARGO_TEMPERATURE_ANOMALY"][0][lidx]), 0, -1)
        add[t, :, :, L:] = np.moveaxis(
            mean_s + sample(dE.variables["ARGO_SALINITY_ANOMALY"][0][lidx]), 0, -1)
        dE.close()

    # z-score the new channels; drop any prior copy of them (re-runnable)
    old_chan = [str(c) for c in d["chan"]]
    keep = [i for i, c in enumerate(old_chan) if c not in chans]
    X = d["X"][..., keep]
    norm = d["norm"][keep]
    add_norm = np.zeros((2 * L, 2), dtype=np.float32)
    for c in range(2 * L):
        v = add[..., c][np.isfinite(add[..., c])]
        if not len(v):
            print(f"  WARNING: channel {chans[c]} came back empty")
            continue
        mu, sd = float(v.mean()), float(v.std() + 1e-6)
        add_norm[c] = (mu, sd)
        add[..., c] = (add[..., c] - mu) / sd
        cov = np.isfinite(add[..., c]).mean()
        print(f"  {chans[c]:<10} coverage {cov:5.1%}  mu {mu:8.3f}  sd {sd:6.3f}")

    np.savez_compressed(
        NPZ, X=np.concatenate([X, add], axis=-1).astype(np.float32),
        months=d["months"], lats=lats, lons=lons,
        chan=np.array([old_chan[i] for i in keep] + chans),
        norm=np.concatenate([norm, add_norm], axis=0), rapid=d["rapid"],
        window=d["window"] if "window" in d else np.array("na"))
    print(f"rewrote {NPZ}: C={len(keep) + 2 * L} channels "
          f"({os.path.getsize(NPZ) / 1e6:.0f} MB)")


if __name__ == "__main__":
    sys.exit(main())
