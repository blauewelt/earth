#!/usr/bin/env python3
"""Lever 1 — more truth: additional never-input overturning transport series.

The probe's binding constraint is truth months (SCALING.md), and no model
choice fixes it. This fetcher adds every basin-scale transport array with a
credential-free archive, monthly-means each onto YYYYMM stamps, and stores
them in the tensor npz as `truth_<name>` [n, 2] float32 arrays of
(yyyymm, Sv). RAPID stays in the legacy `rapid` key (month-index pairs).

  fc     Florida Current cable, 27°N Florida Straits — DAILY since 1982
         (several times RAPID's span; 1982-1992 becomes usable when the
         tensor's time axis extends before 1993). AOML WBTS.
  osnap  OSNAP MOC (subpolar, ~53-60°N), monthly 2014-08..2022-07.
         Georgia Tech repository, doi:10.35090/gatech/70342.
  move   MOVE 16°N NADW transport, ~daily 2000-2022. OceanSITES via NDBC
         THREDDS. Sign: southward NADW flow — the DEEP branch (anti-
         correlated with upper-cell strength by construction).
  samba  SAMBA 34.5°S total MOC ANOMALY (vs 14.7 Sv record mean), daily
         2009-2017. AOML SAMOC. Anomalies are fine: every probe centres
         its target anyway.

Downloads cache in ml/cache/truth/. Re-runnable; preserves every other
key in the npz (window, channels, …). Citations: Meinen et al. 2010
(cable), Fu et al. 2023 (OSNAP), Send et al. 2011 (MOVE), Meinen et al.
2018 / Kersalé et al. 2020 (SAMBA).

Run:  python3 ml/fetch_truth.py
"""
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
TRUTH = os.path.join(CACHE, "truth")
NPZ = os.path.join(CACHE, "na_pixels.npz")
UA = {"User-Agent": "earth-science-pipeline/1.0 (research; github blauewelt/earth)"}

FC_BASE = "https://www.aoml.noaa.gov/ftp/phod/WBTS/cable"
OSNAP_URL = ("https://repository.gatech.edu/server/api/core/bitstreams/"
             "597db471-e2ea-4109-b1a1-b94451f1b884/content")  # MOC_MHT_MFT 201408_202207
MOVE_URL = ("https://dods.ndbc.noaa.gov/thredds/fileServer/oceansites/DATA_GRIDDED/"
            "MOVE/OS_MOVE_20000206-20221014_DPR_VOLUMETRANSPORT.nc")
SAMBA_URL = ("https://www.aoml.noaa.gov/phod/SAMOC_international/documents/"
             "MOC_TotalAnomaly_and_constituents.asc")


def fetch(url, name, attempts=4):
    path = os.path.join(TRUTH, name)
    if os.path.exists(path):
        return path
    for i in range(attempts):
        try:
            print(f"  downloading {url}" + (f" (attempt {i + 1})" if i else ""))
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r, \
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


def nc_series_ym(t_days_since_1950, vals):
    """(time array, values) -> (yyyymm list, values), skipping bad times."""
    ok = np.isfinite(t_days_since_1950)
    base = np.datetime64("1950-01-01")
    ymd = base + t_days_since_1950[ok].astype("timedelta64[D]")
    yms = [int(str(x)[:4]) * 100 + int(str(x)[5:7]) for x in ymd]
    return yms, vals[ok]


def monthly(dates_ym, vals):
    """(list of yyyymm ints, values) -> [n,2] monthly means, sorted."""
    acc = {}
    for ym, v in zip(dates_ym, vals):
        if np.isfinite(v):
            acc.setdefault(int(ym), []).append(float(v))
    out = [(ym, float(np.mean(vs))) for ym, vs in sorted(acc.items())]
    return np.array(out, dtype=np.float32).reshape(-1, 2)


def fc():
    """Florida Current cable: per-year dat files, columns y m d transport [flag]."""
    import datetime
    yms, vals = [], []
    this_year = datetime.date.today().year
    for y in range(1982, this_year + 1):
        for suffix in (f"_{y}_v3.dat", f"_{y}_v2.dat", f"_{y}.dat"):
            try:
                path = fetch(f"{FC_BASE}/FC_cable_transport{suffix}",
                             f"fc{suffix}", attempts=1)
                break
            except Exception:
                path = None
        if not path:
            print(f"  fc {y}: no file (gap year)")
            continue
        for line in open(path):
            parts = line.split()
            if len(parts) >= 4 and not line.lstrip().startswith("%"):
                try:
                    yy, mm, t = int(parts[0]), int(parts[1]), float(parts[3])
                except ValueError:
                    continue
                yms.append(yy * 100 + mm)
                vals.append(t)
    return monthly(yms, vals)


def osnap():
    import netCDF4
    d = netCDF4.Dataset(fetch(OSNAP_URL, "osnap_moc.nc"))
    t = np.ma.filled(d.variables["TIME"][:], np.nan).astype(float)  # days since 1950
    v = np.ma.filled(d.variables["MOC_ALL"][:], np.nan)
    return monthly(*nc_series_ym(t, v))


def move():
    import netCDF4
    d = netCDF4.Dataset(fetch(MOVE_URL, "move.nc"))
    t = np.ma.filled(d.variables["TIME"][:], np.nan).astype(float)
    v = np.ma.filled(d.variables["TRANSPORT_TOTAL"][:], np.nan)
    return monthly(*nc_series_ym(t, v))


def samba():
    path = fetch(SAMBA_URL, "samba.asc")
    yms, vals = [], []
    for line in open(path):
        if line.lstrip().startswith("%"):
            continue
        parts = line.split()
        if len(parts) >= 5:
            try:
                y, m, v = int(parts[0]), int(parts[1]), float(parts[4])
            except ValueError:
                continue
            yms.append(y * 100 + m)
            vals.append(v)
    return monthly(yms, vals)


def main():
    os.makedirs(TRUTH, exist_ok=True)
    d = dict(np.load(NPZ))
    print(f"tensor: {d['X'].shape}, window {d.get('window', 'na')}")
    series = {"fc": fc, "osnap": osnap, "move": move, "samba": samba}
    months = [str(m) for m in d["months"]]
    covered = {int(m[:4]) * 100 + int(m[5:7]) for m in months}
    for name, fn in series.items():
        arr = fn()
        d[f"truth_{name}"] = arr
        inside = int(sum(1 for ym in arr[:, 0] if int(ym) in covered))
        print(f"  truth_{name}: {len(arr)} months "
              f"{int(arr[0, 0])}..{int(arr[-1, 0])} · {inside} inside the tensor axis")
    np.savez_compressed(NPZ, **d)
    print(f"rewrote {NPZ} with {len(series)} truth series")


if __name__ == "__main__":
    sys.exit(main())
