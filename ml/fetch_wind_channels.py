#!/usr/bin/env python3
"""Add wind stress to the pilot tensor — the 2009-10 dip's missing physics.

The famous winter 2009-10 AMOC collapse was substantially WIND-driven
(extreme negative-NAO Ekman forcing), and it is exactly the event our
probe half-misses: sign right from January, onset and amplitude wrong
(measured 2026-08-06, out-of-fold dz64 predictions: observed -6.9 Sv dip
mean, predicted -1.1). Ekman transport is τ/(ρf) — the model cannot see
what the tensor does not carry.

Source: NCEP/NCAR Reanalysis 1 monthly momentum flux (uflx/vflx, gaussian
~1.9° grid, 1948→now, updated monthly, no account — same PSL host as
GPCP/OISST). Coarse and old-generation, but it covers every tensor month
and the right physics; the ERA5 upgrade (1940→, 0.25°) stays on the
SCALING.md list for a credentialed Colab session. Sign note: NCEP serves
momentum flux INTO the surface; we negate so positive tau_x = eastward
stress ON the ocean (the embedding is sign-agnostic, humans reading maps
are not).

Appends channels tau_x, tau_y (N/m², z-scored) to ml/cache/na_pixels.npz,
re-runnably, following fetch_rg_channels.py's contract.
"""
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
NPZ = os.path.join(CACHE, "na_pixels.npz")
BASE = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/Monthlies/surface_gauss"
UA = {"User-Agent": "earth-globe-ml/1.0 (github.com/blauewelt/earth)"}


def fetch(name):
    path = os.path.join(CACHE, "wind", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return path
    print(f"  downloading {BASE}/{name}")
    req = urllib.request.Request(f"{BASE}/{name}", headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r, open(path + ".part", "wb") as f:
        f.write(r.read())
    os.rename(path + ".part", path)
    return path


def main():
    import netCDF4 as ncdf
    d = dict(np.load(NPZ, allow_pickle=False))
    months = [str(m) for m in d["months"]]
    lats, lons = d["lats"], d["lons"]
    T = len(months)
    print(f"pilot tensor: {T} months, {len(lats)}x{len(lons)} cells")

    chans = ["tau_x", "tau_y"]
    add = np.full((T, len(lats), len(lons), 2), np.nan, dtype=np.float32)

    for ci, (var, fn) in enumerate((("uflx", "uflx.sfc.mon.mean.nc"),
                                    ("vflx", "vflx.sfc.mon.mean.nc"))):
        nc = ncdf.Dataset(fetch(fn))
        t = nc.variables["time"]
        dates = ncdf.num2date(t[:], t.units, only_use_cftime_datetimes=False)
        stamps = [f"{x.year:04d}-{x.month:02d}" for x in dates]
        g_lat = np.array(nc.variables["lat"][:])            # descending
        g_lon = np.array(nc.variables["lon"][:])            # 0..360
        iy = np.array([int(np.argmin(np.abs(g_lat - la))) for la in lats])
        lon360 = np.where(lons < 0, lons + 360.0, lons)
        ix = np.array([int(np.argmin(np.abs(g_lon - lo))) for lo in lon360])
        v = nc.variables[var]
        midx = {m: i for i, m in enumerate(months)}
        for k, s in enumerate(stamps):
            if s not in midx:
                continue
            field = np.ma.filled(v[k], np.nan)
            add[midx[s], :, :, ci] = -field[iy][:, ix]      # sign: stress ON ocean
        nc.close()

    # the tensor's ocean mask: wind exists everywhere, but only ocean cells
    # participate anywhere else — mask land so coverage stats stay honest
    ocean = np.isfinite(d["X"][..., 0]).any(axis=0)
    add[:, ~ocean, :] = np.nan

    old_chan = [str(c) for c in d["chan"]]
    keep = [i for i, c in enumerate(old_chan) if c not in chans]
    X = d["X"][..., keep]
    norm = d["norm"][keep]
    add_norm = np.zeros((2, 2), dtype=np.float32)
    for c in range(2):
        v = add[..., c][np.isfinite(add[..., c])]
        mu, sd = float(v.mean()), float(v.std() + 1e-6)
        add_norm[c] = (mu, sd)
        add[..., c] = (add[..., c] - mu) / sd
        cov = np.isfinite(add[..., c]).mean()
        print(f"  {chans[c]:<6} coverage {cov:5.1%}  mu {mu:8.4f}  sd {sd:7.4f}  N/m²")

    np.savez_compressed(
        NPZ, X=np.concatenate([X, add], axis=-1).astype(np.float32),
        months=d["months"], lats=lats, lons=lons,
        chan=np.array([old_chan[i] for i in keep] + chans),
        norm=np.concatenate([norm, add_norm], axis=0), rapid=d["rapid"],
        window=d["window"] if "window" in d else np.array("na"))
    print(f"rewrote {NPZ}: C={len(keep) + 2} channels "
          f"({os.path.getsize(NPZ) / 1e6:.0f} MB)")


if __name__ == "__main__":
    sys.exit(main())
