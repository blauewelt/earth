#!/usr/bin/env python3
"""Read a published family-7.2d store back from the Hub (E-087).

Two checks per instant, neither through the adapter's code path:

  SOURCE     the whole frame read through `sharded.ShardedGroup` on the Hub's
             `resolve/main` URL, against an INDEPENDENT read of the source
             file straight from the producer (OISST and NCEP from NOAA PSL,
             not the Hub mirror the lanes used; GLORYS from its parked chunk,
             which is the source): GLORYS the chunk's own value; OISST a
             bilinear written here as "the NaN-aware mean of the four cell
             centres around each point" (every 0.25-degree point sits exactly
             between four OISST centres); NCEP the mean of the day's four
             6-hourly samples, regridded with scipy's RegularGridInterpolator
             on the gaussian axes (soil channels compared only where all four
             neighbours are land). Every value must be within half a float16
             step of the reference (plus 1e-5 relative for the regrid's own
             float arithmetic, and 8 float32 half-ulps in the source's own
             units for family 7's float32 interpolation and K -> degC).
  PENTAD     the bin's five frames from the Hub, averaged by each channel's
             rule, against the published family-7.2 pentad
             (`family7_daily.check_bin`, the lanes' falsifier, on the
             PUBLISHED bytes).

  python3 ml/family1/f7d_hub_check.py --store oisst025d \\
      --days 1985-07-15,2023-01-15 --out ml/family1/probes/oisst025d_hubcheck.json
"""
import argparse
import datetime as dt
import json
import os
import sys
import tempfile
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.dirname(HERE)
sys.path.insert(0, ML)

from family1 import sharded as sh                               # noqa: E402
import family7_daily as fd                                      # noqa: E402

HUB = "https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main"
PSL = "https://downloads.psl.noaa.gov/Datasets"
EPOCH = dt.date(1982, 1, 1)


def get(url, path, attempts=4):
    """Download once, size-checked (urlretrieve raises on a short body)."""
    for i in range(attempts):
        if os.path.exists(path):
            return path
        try:
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)
        except Exception:                                    # noqa: BLE001
            if i + 1 == attempts:
                raise
            time.sleep(10 * (i + 1))
    return path


def nc_day_index(ds, day):
    import netCDF4 as ncdf
    tv = ds.variables["time"]
    ts = ncdf.num2date(tv[:], tv.units, only_use_cftime_datetimes=False)
    return [i for i, t in enumerate(np.atleast_1d(ts))
            if dt.date(t.year, t.month, t.day) == day]


def ref_glorys(day, tmp):
    import netCDF4 as ncdf
    n = f"glorys025_global_{day.year}{day.month:02d}.nc"
    p = get(f"{HUB}/daily025_global/{n}", os.path.join(tmp, n))
    d = ncdf.Dataset(p)
    k = nc_day_index(d, day)[0]
    u, v, ml, zs = [np.ma.filled(d.variables[x][k], np.nan).astype(np.float64)
                    for x in ("uo", "vo", "mlotst", "zos")]
    d.close()
    out = np.full((721, 1440, 5), np.nan)
    out[40:, :, 0] = np.sqrt(u * u + v * v)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[40:, :, 1] = np.where(ml > 0, np.log10(ml), np.nan)
    out[40:, :, 2], out[40:, :, 3], out[40:, :, 4] = zs, u, v
    return out


def four_centre_mean(f):
    """OISST [720 lat -89.875.., 1440 lon 0.125..] -> the 0.25-degree POINT
    grid [721 lat -90..90, 1440 lon -180..179.75]: each point is the NaN-aware
    mean of the four centres around it (bilinear at a midpoint), with the
    pole rows taking the two centres beside them in longitude."""
    f = np.asarray(f, np.float64)
    # longitude: point lon L (0..359.75) sits between centres L-0.125, L+0.125
    a = np.roll(f, 1, axis=1)          # centre at L - 0.125 for point index
    b = f                              # centre at L + 0.125
    lon_pairs = np.stack([a, b])       # [2, 720, 1440] -> points at L = 0.25j
    # latitude: point row r (lat -90 + 0.25 r) sits between centre rows r-1, r
    pad = np.full((2, 1, 1440), np.nan)
    lo = np.concatenate([pad, lon_pairs], axis=1)       # row r-1
    hi = np.concatenate([lon_pairs, pad], axis=1)       # row r
    st = np.concatenate([lo, hi])                       # [4, 721, 1440]
    with np.errstate(invalid="ignore"):
        m = np.nanmean(st, axis=0)
    # reorder longitudes 0..359.75 -> -180..179.75
    return np.roll(m, 720, axis=1)


def ref_oisst(day, tmp):
    import netCDF4 as ncdf
    out = np.full((721, 1440, 2), np.nan)
    for c, k in enumerate(("sst", "icec")):
        n = f"{k}.day.mean.{day.year}.nc"
        p = get(f"{PSL}/noaa.oisst.v2.highres/{n}", os.path.join(tmp, n))
        d = ncdf.Dataset(p)
        i = nc_day_index(d, day)[0]
        f = np.ma.filled(d.variables[k][i], np.nan).astype(np.float64)
        f[np.abs(f) > 1e30] = np.nan
        d.close()
        out[..., c] = four_centre_mean(f)
    return out


def ref_ncep(day, tmp):
    import netCDF4 as ncdf
    from scipy.interpolate import RegularGridInterpolator
    import build_family7 as f7
    lat1 = np.arange(-90.0, 91.0)
    lon1 = np.arange(-180.0, 180.0)
    land_p = get(f"{PSL}/ncep.reanalysis/surface_gauss/land.sfc.gauss.nc",
                 os.path.join(tmp, "land.sfc.gauss.nc"))
    dl = ncdf.Dataset(land_p)
    land = np.squeeze(np.ma.filled(dl.variables["land"][:], 0.0)) >= 0.5
    dl.close()
    cols = {}
    lat = lon = None
    for v in ("uflx", "vflx", "air", "uwnd", "vwnd", "pres", "prate",
              "weasd", "soilw", "tmp", "lhtfl", "shtfl", "skt"):
        n = f"{f7.NCEP_FILES[v]}.{day.year}.nc"
        p = get(f"{PSL}/ncep.reanalysis/surface_gauss/{n}",
                os.path.join(tmp, n))
        d = ncdf.Dataset(p)
        lat = np.asarray(d.variables["lat"][:], np.float64)
        lon = np.asarray(d.variables["lon"][:], np.float64)
        ii = nc_day_index(d, day)
        a = np.squeeze(np.ma.filled(d.variables[v][ii], np.nan)
                       .astype(np.float64))
        d.close()
        cols[v] = a.mean(axis=0)
    # ascending latitude, longitude wrapped
    la = lat[::-1]
    lo_ = np.concatenate([lon, [lon[0] + 360.0]])

    def regrid(f):
        g = f[::-1]
        g = np.concatenate([g, g[:, :1]], axis=1)
        ip = RegularGridInterpolator((la, lo_), g, bounds_error=False,
                                     fill_value=None)
        Y, X = np.meshgrid(np.clip(lat1, la[0], la[-1]),
                           np.where(lon1 < 0, lon1 + 360.0, lon1),
                           indexing="ij")
        return ip(np.stack([Y, X], -1))
    out = np.full((181, 360, 15), np.nan)
    out[..., 0] = regrid(-cols["uflx"])
    out[..., 1] = regrid(-cols["vflx"])
    out[..., 4] = regrid(cols["air"]) - 273.15
    out[..., 5] = regrid(cols["uwnd"])
    out[..., 6] = regrid(cols["vwnd"])
    out[..., 7] = regrid(cols["pres"]) / 100.0
    out[..., 8] = np.log1p(np.maximum(regrid(cols["prate"]) * 86400.0, 0))
    out[..., 9] = np.log1p(np.maximum(regrid(cols["weasd"]), 0))
    landf = regrid(land.astype(np.float64))
    allland = landf >= 1.0 - 1e-9
    out[..., 10] = np.where(allland, regrid(cols["soilw"]), np.nan)
    out[..., 11] = np.where(allland, regrid(cols["tmp"]) - 273.15, np.nan)
    out[..., 12] = regrid(cols["lhtfl"])
    out[..., 13] = regrid(cols["shtfl"])
    out[..., 14] = regrid(cols["skt"]) - 273.15
    return out           # channels 2, 3 (the centred sigma) not referenced


REF = {"glorys025d": ref_glorys, "oisst025d": ref_oisst,
       "ncep100d": ref_ncep}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, choices=sorted(REF))
    ap.add_argument("--days", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--base", default="",
                    help="a local group directory instead of the Hub")
    a = ap.parse_args(argv)
    base = a.base or f"{HUB}/tensors/family7_2d/{a.store}/{a.store}"
    g = sh.ShardedGroup(base)
    cfg = fd.STORES[a.store]
    index = fd.load_index()
    report = {"store": a.store, "hub": base, "instants": []}
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for ds in a.days.split(","):
            day = dt.date(*(int(x) for x in ds.split("-")))
            k = (day - EPOCH).days
            b, f = k // 5, k % 5
            t0 = time.time()
            frame = g.read_frame(b, f)
            ranges = g.src.ranges
            ref = REF[a.store](day, tmp)
            r = {"day": ds, "bin": b, "frame": f, "hub_range_reads": ranges,
                 "channels": {}}
            for c, (name, *_x) in enumerate(cfg["channels"]):
                if a.store == "ncep100d" and c in (2, 3):
                    continue
                H, R = frame[..., c].astype(np.float64), ref[..., c]
                both = np.isfinite(H) & np.isfinite(R)
                # NaN only where the reference is NaN, except where the
                # independent read omits it (NCEP soil at partial coasts)
                hub_only = int((np.isfinite(H) & ~np.isfinite(R)).sum())
                ref_only = int((~np.isfinite(H) & np.isfinite(R)).sum())
                d = np.abs(H - R)[both]
                # + family 7's own float32 arithmetic in the source's units
                # (interp2_nan returns float32; K -> degC in float32), the
                # same term the lanes' falsifier carries
                off = 273.15 if name in ("t2m", "tsoil", "skt") else 0.0
                # family 7's bilinear weights are float32
                # (build_family3.lin_weights returns w1 as float32), so its
                # rounding scales with the CORNERS' magnitude, not the
                # result's: near a zero crossing (a heat flux, a coast) the
                # neighbourhood decides
                from scipy.ndimage import maximum_filter
                big = maximum_filter(np.where(np.isfinite(R), np.abs(R), 0.0),
                                     size=3, mode="wrap")
                # half a float16 step at the LARGER of the two magnitudes:
                # the store's pre-rounding value and this reference can sit
                # on either side of a power of two, where the step doubles
                tol = fd.f16_half_step(np.maximum(np.abs(R[both]),
                                                  np.abs(H[both]))) \
                    + 1e-5 * np.abs(R[both]) \
                    + 2.0 ** -21 * (big[both] + off) + 1e-7
                bad = int((d > tol).sum())
                r["channels"][name] = {
                    "values": int(both.sum()), "max_abs_diff": float(d.max()),
                    "beyond_half_f16_step": bad, "finite_on_hub_only":
                        hub_only, "finite_in_reference_only": ref_only}
                if bad or ref_only or (hub_only and not (
                        a.store == "ncep100d" and c in (10, 11))):
                    ok = False
            # the pentad: the whole bin from the Hub against f7l2
            frames = []
            for ff in range(5):
                x = g.read_frame(b, ff)
                frames.append(np.full((cfg["grid"]["H"], cfg["grid"]["W"],
                                       len(cfg["channels"])), np.nan,
                                      np.float32) if x is None else x)
            pent, z, norm = fd.pentad_slabs(index, cfg["group"], b, 1)
            idx = cfg["pentad_index"]
            res = fd.check_bin(a.store, frames, pent[0][..., idx],
                               z[0][..., idx], norm[idx])
            r["pentad"] = res
            ok = ok and res["ok"]
            r["seconds"] = round(time.time() - t0, 1)
            report["instants"].append(r)
            print(json.dumps({"day": ds, "pentad_ok": res["ok"],
                              "channels": {n: (v["max_abs_diff"],
                                               v["beyond_half_f16_step"])
                                           for n, v in r["channels"].items()}
                              }), flush=True)
    report["ok"] = ok
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(report, fh, indent=1)
    print("HUB CHECK", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
