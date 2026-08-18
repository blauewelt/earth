#!/usr/bin/env python3
"""OISST v2.1 DAILY, 0.25 degree, transposed to PIXEL-MAJOR int16 — one file
per year, read by the browser two bytes at a time.

Why this shape (E-040, measured 2026-08-18): the app needs one PIXEL's value,
not one DAY's map — the map is already GIBS tiles. Storing [pixel][day] rather
than [day][pixel] turns a point query into a single contiguous read:

    one value          2 bytes
    one pixel-year   730 bytes   (365 daily values, ONE request)
    1991-2026        ~26 KB      (36 parallel reads)

out of a 757 MB year file. Verified end to end from a browser on a foreign
origin: HTTP 206, CORS open, 24.22 degC returned, bit-identical to the source
NetCDF. That is the whole argument for hosting this off-repo instead of
shipping monthly means: the transfer is bounded by the QUESTION, not by the
archive.

int16 at 0.01 degC, -32768 = no data (land, ice). Longitude rolled 0..360 ->
-180..180 so the grid matches every other artifact in the repo.

Usage:  python3 scripts/bake_sst_daily.py 2015 [--keep]
The source is ~476 MB/year and is deleted after baking unless --keep.
"""
import json
import os
import sys
import urllib.request

import numpy as np
import netCDF4

YR = int(sys.argv[1])
KEEP = "--keep" in sys.argv
SRC = f"/tmp/oisst_day_{YR}.nc"
OUT = os.environ.get("SST_OUT", "/tmp")

if not os.path.exists(SRC):
    url = ("https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/"
           f"sst.day.mean.{YR}.nc")
    print("downloading", url, flush=True)
    urllib.request.urlretrieve(url, SRC)
print(f"source {os.path.getsize(SRC) / 1e6:.0f} MB", flush=True)

d = netCDF4.Dataset(SRC)
sst = d.variables["sst"]                          # (time, lat 720 asc, lon 1440)
T, NY, NX = sst.shape
print("shape", sst.shape, flush=True)

# Build [pixel][day]. Held in memory a year at a time: 1440*720*365*2 = 757 MB,
# which is the reason this is per-year rather than one file for the record.
out = np.full((NY * NX, T), -32768, dtype=np.int16)
for t in range(T):
    a = np.ma.filled(sst[t], np.nan).astype(np.float32)
    a = np.roll(a, NX // 2, axis=1)               # lon 0..360 -> -180..180
    ok = np.isfinite(a)
    out[:, t] = np.where(ok, np.clip(a * 100, -32767, 32767), -32768).astype(np.int16).ravel()
    if t % 60 == 0:
        print("  day", t, flush=True)

path = os.path.join(OUT, f"oisst_daily_{YR}.i16")
out.tofile(path)
json.dump({"year": YR, "days": int(T), "nx": NX, "ny": NY,
           "dlon": 0.25, "dlat": 0.25, "west": -180, "south": -90,
           "scale": 0.01, "nodata": -32768,
           "layout": "pixel-major: value(px, day) at byte (px*days + day)*2",
           "source": "NOAA OISST v2.1 daily (PSL)",
           "citation": "Huang et al. 2021, doi:10.1175/JCLI-D-20-0166.1"},
          open(os.path.join(OUT, f"oisst_daily_{YR}.json"), "w"))
if not KEEP:
    os.remove(SRC)                                # 476 MB each
print(f"wrote {path} {os.path.getsize(path) / 1e6:.0f} MB", flush=True)
