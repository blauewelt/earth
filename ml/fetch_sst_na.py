#!/usr/bin/env python3
"""OISST v2.1 DAILY 0.25 degree SST, cropped and INTERPOLATED onto the ML
window's own grid — ml/cache/sst_na025/{sst_daily_na.npy, index.npz}.

WHY THIS FILE EXISTS, AND WHY IT IS NOT `scripts/bake_sst_daily.py`
(E-040 §5, resolved with Chris 2026-08-18). One upstream download, two
consumers, two artifacts, shared at the DOWNLOAD level:

    sst.day.mean.YYYY.nc   (PSL, ~476 MB/yr, streamed then deleted)
        |-- pixel-major int16, global, lon rolled  -> scripts/bake_sst_daily.py
        |                                             -> HF -> the browser's
        |                                                2-byte point reads
        `-- day-major int16, NA window, ML grid    -> THIS FILE -> the E-034
                                                      tensor's SST channel

The two layouts are transposes of each other. Building the tensor from the
pixel-major file would mean reading 757 MB/yr back and transposing it;
building the app file from a pentad mean would throw the daily axis away.
Each consumer takes its natural cut of one download.

This is a QUALITY fix, not only plumbing: the tensor's SST currently arrives
via the 1-degree monthly bake upsampled to 0.25, carrying no sub-degree
structure. OISST is an AVHRR analysis (its effective resolution is coarser
than its 0.25 posting) but it is real sub-degree information against a
1-degree bin, on the tensor's own grid.

THE CLIMATOLOGY IS NOT SHARED, AND MUST NEVER BE (E-040 §5). This file emits
the FIELD. The pipeline's anomaly baseline is computed inside temporal.py
from TRAIN YEARS ONLY; the app's baseline is the WMO 1991-2020 normal, which
includes the holdout years. They look interchangeable and are not —
substituting the app's normal here would leak test-period information into
training through the baseline. Keep this paragraph next to any future
refactor that notices "we compute the climatology twice".

THE GRIDS DO NOT COINCIDE — THIS IS THE WHOLE DIFFICULTY.
OISST's `lat`/`lon` variables are CELL CENTRES: 0.125 + k*0.25 (lat
-89.875..89.875, lon 0.125..359.875). The ML window samples ON multiples of
0.25 (lats 0..70 / 281, lons -100..20 / 481, measured from base025_na.npz).
Every target point therefore falls exactly HALFWAY between two source
centres in each axis, and nearest-indexing would displace the whole field by
half a cell in both directions — invisible in any plot, fatal to every
stencil and to the AMOC eval mask. So: interpolate, with the SAME machinery
the wind channel uses (`build_family4.fill_wind_pentad`):

    wy = f3.lin_weights(src_lat, lats)
    wx = f3.lin_weights(src_lon, np.where(lons < 0, lons + 360.0, lons),
                        wrap_period=360.0)
    f3.interp2_nan(field, wy, wx)

`interp2_nan` renormalises the corner weights over the FINITE parents, so a
coastal target keeps a value instead of inheriting NaN from a land corner,
and an all-land neighbourhood stays NaN. Land and ice are masked in the
source (`np.ma.filled(..., np.nan)`) and NaN travels all the way to the
artifact's nodata.

OUTPUT (a contract `ml/build_family4.py` is coded against — do not deviate):
  sst_daily_na.npy  int16 (NDAYS, 281, 481), scale 0.01 degC, nodata -32768
  index.npz         bin_index (int64, days since 1982-01-01, contiguous),
                    has_data (bool, per row), epoch, cadence_days=1,
                    scale=0.01, nodata=-32768, lat, lon (float32),
                    source="OISST v2.1 sst.day.mean, PSL"
1982-01-01..2024-12-31 = 15,706 rows = 4.25 GB. DAILY, not pentad: E-034's
cadence table asks for a pentad MEAN of this, and the standing rule
(`aggregate_cadence.py`) is fetch daily once and derive every coarser cadence
by aggregation — one downloaded byte-stream, one aggregation rule.

ROUNDING, one deliberate difference from `scripts/bake_sst_daily.py`. That
script writes `(a * 100).astype(np.int16)`, which TRUNCATES: the source's own
int16-at-0.01 encoding comes back as a float32 that can sit a hair below the
exact multiple (24.22 -> 2421.9998 -> 2421 -> 24.21 degC, one count low). We
round, which makes the round-trip lossless w.r.t. the source's own encoding
and is what `tests/test_sst_na.py` check 4 pins at 0.005 degC.

DISK AND TIME, measured on one real year (1993) 2026-08-18: 477 MB source,
365 rows folded in **14.5 s**, **peak RSS 0.26 GB** — so the 43-year run is
bounded by the download, not by the arithmetic. Years are streamed and each
source is DELETED after it is folded, so peak disk is ~1 GB (one 477 MB
source + its part-file) plus the growing output. A year that cannot be
downloaded or read leaves its rows at nodata with `has_data` False and a
::warning:: — never a silent hole.

Spot-checked against the source on that year (1993-07-02): the artifact at
(60N, 30W) is 8.10 degC and the four straddling OISST centres average 8.1000;
(25N, 80W) 28.46 vs 28.4650; (0N, 20W) 25.13 vs 25.1275; (45N, 10E) is land
and comes back nodata. Whole-frame range -1.80..30.87 degC, 65.3% wet.

Run:
  python3 ml/fetch_sst_na.py --out ml/cache/sst_na025 --report-mem
  python3 ml/fetch_sst_na.py --year 1993 --out /tmp/sst_smoke   # smoke
"""
import argparse
import datetime as dt
import os
import resource
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_family3 as f3                                    # noqa: E402

CACHE = os.path.join(HERE, "cache")
EPOCH = dt.date(1982, 1, 1)
SCALE = 0.01
NODATA = -32768
SOURCE = "OISST v2.1 sst.day.mean, PSL"
PSL = "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres"
THREDDS = ("https://psl.noaa.gov/thredds/fileServer/Datasets/"
           "noaa.oisst.v2.highres")
# The ML window, if base025_na.npz is not there to be authoritative.
NLAT, NLON = 281, 481
LAT0, LON0, DEG = 0.0, -100.0, 0.25


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def source_url(year):
    return f"{PSL}/sst.day.mean.{year}.nc"


def remote_size(url):
    """Content-Length, or None if the server will not say."""
    import urllib.request
    try:
        req = urllib.request.Request(
            url, method="HEAD",
            headers={"User-Agent": "earth-science-pipeline/1.0 "
                                   "(research; github blauewelt/earth)"})
        with urllib.request.urlopen(req, timeout=120) as r:
            n = r.headers.get("Content-Length")
        return int(n) if n else None
    except Exception:                                         # noqa: BLE001
        return None


def download_year(year, path, attempts=3):
    """Fetch one year of OISST to `path` and return it, SIZE-VERIFIED.

    A seam, on purpose: `tests/test_sst_na.py` replaces this with a copy of a
    synthetic OISST-shaped file so the interpolation can be checked against an
    analytic answer without 476 MB of network. The mirror is the same pair
    `build_family4.fill_wind_pentad` uses — downloads.psl.noaa.gov 504'd every
    retry of run #47 while the thredds host served fine.

    The size check is not decoration. MEASURED 2026-08-18 in the sandbox: a
    477,350,790-byte year came back short with no exception raised, and the
    only symptom was `NetCDF: HDF error` at open time — i.e. one silently
    truncated transfer costs a whole year of the axis. Comparing against
    Content-Length turns that into a retry instead.
    """
    url = source_url(year)
    mirrors = (f"{THREDDS}/sst.day.mean.{year}.nc",)
    want = remote_size(url)
    for i in range(attempts):
        f3.fetch(url, path, mirrors=mirrors)
        got = os.path.getsize(path)
        if want is None or got == want:
            return path
        print(f"  ::warning:: {year}: {got:,} bytes of {want:,} — truncated "
              f"transfer, refetching ({i + 1}/{attempts})", flush=True)
        os.remove(path)
    raise IOError(f"{url}: could not be downloaded whole in {attempts} "
                  f"attempts (expected {want:,} bytes)")


def target_grid():
    """The ML window's axes — from base025_na.npz when it is there.

    Mirrors `build_family4.check_grid`'s posture: the reference file is the
    authority, its absence is a LOUD warning and not a silent guess, because
    a half-cell offset makes every stencil and the AMOC eval mask name
    different pixels than they describe.
    """
    ref = os.path.join(CACHE, "base025_na.npz")
    if os.path.exists(ref):
        d = np.load(ref)
        lats = np.asarray(d["lats"], dtype=np.float32)
        lons = np.asarray(d["lons"], dtype=np.float32)
        print(f"grid     {len(lats)}x{len(lons)} from base025_na.npz "
              f"(lat {lats[0]}..{lats[-1]}, lon {lons[0]}..{lons[-1]})")
        return lats, lons
    lats = (LAT0 + np.arange(NLAT) * DEG).astype(np.float32)
    lons = (LON0 + np.arange(NLON) * DEG).astype(np.float32)
    print("::warning:: base025_na.npz absent — the target grid is assumed "
          f"({NLAT}x{NLON}, lat {lats[0]}..{lats[-1]}, lon {lons[0]}.."
          f"{lons[-1]}) and comparability with family 3 is UNCHECKED. Fetch "
          "it from data-cache-v1 to close this.")
    return lats, lons


def weights_for(src_lat, src_lon, lats, lons):
    """Bilinear weights source -> target, exactly as fill_wind_pentad calls it.

    Longitude goes through 0..360 (the source's convention) with
    `wrap_period=360.0`, so a target at -100 becomes 260 and the seam between
    359.875 and 0.125 interpolates instead of clamping. Latitude is plain:
    the source axis is ascending and covers the window with room to spare.
    """
    wy = f3.lin_weights(np.asarray(src_lat, dtype=np.float64), lats)
    wx = f3.lin_weights(np.asarray(src_lon, dtype=np.float64),
                        np.where(lons < 0, lons + 360.0, lons),
                        wrap_period=360.0)
    return wy, wx


def encode(field):
    """float32 degC (NaN = land/ice) -> int16 at 0.01 degC, NODATA elsewhere.

    ROUNDS (see the module docstring): the source is itself int16-at-0.01, so
    rounding makes this re-encoding exact rather than one count low.
    """
    ok = np.isfinite(field)
    v = np.clip(np.round(np.where(ok, field, 0.0) * (1.0 / SCALE)),
                -32767, 32767)
    return np.where(ok, v, NODATA).astype(np.int16)


def fold_year(path, out, lats, lons, row_of, wcache):
    """Read one year's dailies, interpolate each day, write its row.

    One day at a time: a whole year on the source grid is 720*1440*365*4 =
    1.5 GB in float32, and nothing here needs two days at once.
    """
    import netCDF4 as ncdf

    d = ncdf.Dataset(path)
    try:
        sst = d.variables["sst"]
        src_lat = np.array(d.variables["lat"][:], dtype=np.float64)
        src_lon = np.array(d.variables["lon"][:], dtype=np.float64)
        key = (src_lat.tobytes(), src_lon.tobytes())
        if wcache.get("key") != key:
            if wcache.get("key") is not None:
                print("::warning:: the source grid CHANGED mid-archive — "
                      "recomputing the interpolation weights")
            wcache["key"] = key
            wcache["w"] = weights_for(src_lat, src_lon, lats, lons)
            print(f"  interp {len(src_lat)}x{len(src_lon)} centres "
                  f"({src_lat[0]:+.3f}.., {src_lon[0]:.3f}..) -> "
                  f"{len(lats)}x{len(lons)} points "
                  f"({lats[0]:+.2f}.., {lons[0]:.2f}..)")
        wy, wx = wcache["w"]

        tv = d.variables["time"]
        dates = ncdf.num2date(tv[:], tv.units,
                              only_use_cftime_datetimes=False)
        wrote = []
        for i, dd in enumerate(dates):
            day = dt.date(dd.year, dd.month, dd.day)
            row = row_of.get((day - EPOCH).days)
            if row is None:                       # outside the planned axis
                continue
            a = np.ma.filled(sst[i], np.nan).astype(np.float32)
            out[row] = encode(f3.interp2_nan(a, wy, wx))
            wrote.append(row)
        return wrote
    finally:
        d.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1982)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--year", type=int, default=None,
                    help="a single year, for a smoke run (sets --start/--end)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default=None,
                    help="where the source NetCDFs land before being deleted "
                         "(default <out>/_src)")
    ap.add_argument("--report-mem", action="store_true",
                    help="print peak RSS at the end — this is how the "
                         "streaming claim stays a measurement")
    a = ap.parse_args(argv)
    if a.year is not None:
        a.start = a.end = a.year
    if a.start > a.end:
        sys.exit(f"--start {a.start} is after --end {a.end}")
    if a.start < 1982:
        # The record starts 1981-09; the axis starts 1982-01-01, so a year
        # before that would write rows at negative indices.
        sys.exit("--start must be >= 1982 (the axis epoch is 1982-01-01)")

    os.makedirs(a.out, exist_ok=True)
    scratch = a.scratch or os.path.join(a.out, "_src")
    os.makedirs(scratch, exist_ok=True)

    lats, lons = target_grid()
    d0 = dt.date(a.start, 1, 1)
    d1 = dt.date(a.end, 12, 31)
    b0, b1 = (d0 - EPOCH).days, (d1 - EPOCH).days
    ndays = b1 - b0 + 1
    idx = np.arange(b0, b1 + 1, dtype=np.int64)
    row_of = {int(b): i for i, b in enumerate(idx)}
    print(f"axis     {ndays:,} daily rows, {d0} .. {d1} "
          f"(bin_index = days since {EPOCH})")

    npy = os.path.join(a.out, "sst_daily_na.npy")
    out = np.lib.format.open_memmap(
        npy, mode="w+", dtype=np.int16, shape=(ndays, len(lats), len(lons)))
    # A fresh memmap is ZEROS, and zero decodes to 0.00 degC — a plausible
    # temperature. Every row starts as nodata instead, in slabs so the fill
    # never needs the whole 4.25 GB resident.
    for s in range(0, ndays, 512):
        out[s:s + 512] = NODATA
    print(f"output   {npy} int16 {out.shape} "
          f"({out.nbytes / 1e9:.2f} GB, prefilled nodata)")

    has = np.zeros(ndays, bool)
    wcache = {}
    failed = []
    for y in range(a.start, a.end + 1):
        src = os.path.join(scratch, f"sst.day.mean.{y}.nc")
        try:
            path = download_year(y, src)
            mb = os.path.getsize(path) / 1e6
            wrote = fold_year(path, out, lats, lons, row_of, wcache)
        except Exception as e:                                # noqa: BLE001
            # A year that fails leaves its rows nodata and has_data False.
            # Loud, never silent: a hole that looks like ocean is worse than
            # no artifact at all.
            print(f"::warning:: {y} unavailable ({type(e).__name__}: "
                  f"{str(e)[:120]}) — its rows stay nodata / has_data False")
            failed.append(y)
            continue
        finally:
            for p in (src, src + ".part"):
                if os.path.exists(p):
                    os.remove(p)                  # ~476 MB each, per year
        has[wrote] = True
        print(f"  {y}: {len(wrote)} day(s) written from {mb:,.0f} MB source "
              f"· {int(has.sum()):,}/{ndays:,} rows have data", flush=True)

    np.savez(os.path.join(a.out, "index.npz"),
             bin_index=idx, has_data=has,
             epoch=np.array(str(EPOCH)), cadence_days=np.array(1),
             scale=np.array(SCALE), nodata=np.array(NODATA),
             lat=lats.astype(np.float32), lon=lons.astype(np.float32),
             source=np.array(SOURCE))
    out.flush()

    # ---- say what came back, in degrees, so a reader can see it is ocean --
    pct = 100.0 * has.sum() / ndays
    print(f"\nwrote {a.out}/sst_daily_na.npy + index.npz "
          f"({os.path.getsize(npy) / 1e9:.2f} GB)")
    print(f"rows with data {int(has.sum()):,}/{ndays:,} ({pct:.1f}%)"
          + (f" · {len(failed)} year(s) FAILED: "
             f"{', '.join(map(str, failed))}" if failed else ""))
    if has.any():
        r = int(np.flatnonzero(has)[len(np.flatnonzero(has)) // 2])
        raw = np.asarray(out[r])
        ok = raw != NODATA
        v = raw[ok].astype(np.float32) * SCALE
        day = EPOCH + dt.timedelta(days=int(idx[r]))
        print(f"sample   row {r} ({day}): {ok.sum():,}/{ok.size:,} cells wet "
              f"({100.0 * ok.sum() / ok.size:.1f}%), "
              f"min {v.min():.2f} / mean {v.mean():.2f} / max {v.max():.2f} "
              f"degC")
    else:
        print("::warning:: NOT ONE ROW HAS DATA — the artifact is all nodata")
    if a.report_mem:
        print(f"peak RSS {peak_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
