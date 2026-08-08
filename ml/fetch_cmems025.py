#!/usr/bin/env python3
"""Data-ladder rung (b): the 0.25-degree base channels, straight from CMEMS.

The 1-degree pipeline reaches the ML tensor through the app's baked JSON
grids; at 0.25 degrees that detour makes no sense (the app has no use for a
94 GB globe), so this fetcher goes straight from the Copernicus subset API
to an npz. Product: the GLORYS member of the 1/4-degree ensemble reanalysis
(cmems_mod_glo_phy-all_my_0.25deg_P1M-m, uo_glor/vo_glor/mlotst_glor, and
zos_glor where served) -- NATIVE 0.25 degrees, so subsetting is the whole
job. One year per request, resume-friendly, NetCDFs kept in
ml/cache/cmems025/ (a NA-window year is ~26 MB; the request, not the bytes,
is the scarce resource).

Credentials: COPERNICUSMARINE_SERVICE_USERNAME/_PASSWORD env vars, per
claude/copernicus-marine-access.md -- never a file, never in the repo.

Validation scope first (CLAUDE.md family discipline): the NA window. The
global 0.25-degree tensor is ~16x this and needs the 64 GB boxes plus a
memmapped builder; that lands as its own step.

Usage:
  python3 ml/fetch_cmems025.py --window na            # fetch + assemble
  python3 ml/fetch_cmems025.py --window na --assemble-only
Writes ml/cache/base025_<window>.npz: X [T,H,W,3] (cur_speed, log_mld,
ssh), months, lats, lons.
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache", "cmems025")
DS = "cmems_mod_glo_phy-all_my_0.25deg_P1M-m"
WINDOWS = {"na": (-100.0, 20.0, 0.0, 70.0),
           "global": (-180.0, 180.0, -85.0, 85.0)}


def fetch(window, start_year, end_year):
    import copernicusmarine as cm
    w, e, s_, n = WINDOWS[window]
    os.makedirs(CACHE, exist_ok=True)
    for y in range(start_year, end_year + 1):
        out = os.path.join(CACHE, f"{window}_{y}.nc")
        if os.path.exists(out):
            print(f"  {y}: cached")
            continue
        # zos_glor first; some product versions serve it, some refuse — a
        # refusal falls back to the three core variables rather than dying.
        for variables in (["uo_glor", "vo_glor", "mlotst_glor", "zos_glor"],
                          ["uo_glor", "vo_glor", "mlotst_glor"]):
            try:
                print(f"  {y}: requesting {variables} …", flush=True)
                cm.subset(dataset_id=DS, variables=variables,
                          minimum_longitude=w, maximum_longitude=e,
                          minimum_latitude=s_, maximum_latitude=n,
                          start_datetime=f"{y}-01-01",
                          end_datetime=f"{y}-12-31",
                          minimum_depth=0, maximum_depth=1,
                          output_filename=out)
                break
            except Exception as ex:
                print(f"    refused ({str(ex)[:120]})")
                if os.path.exists(out):
                    os.remove(out)
        else:
            print(f"  {y}: FAILED all variable sets")


def assemble(window):
    import netCDF4 as ncdf
    files = sorted(glob.glob(os.path.join(CACHE, f"{window}_*.nc")))
    if not files:
        sys.exit("nothing fetched")
    months, slices = [], []
    lats = lons = None
    have_zos = None
    for f in files:
        d = ncdf.Dataset(f)
        if lats is None:
            lats = np.array(d.variables["latitude"][:], dtype=np.float32)
            lons = np.array(d.variables["longitude"][:], dtype=np.float32)
        tv = d.variables["time"]
        u = np.ma.filled(d.variables["uo_glor"][:], np.nan).squeeze()
        v = np.ma.filled(d.variables["vo_glor"][:], np.nan).squeeze()
        m = np.ma.filled(d.variables["mlotst_glor"][:], np.nan).squeeze()
        z = (np.ma.filled(d.variables["zos_glor"][:], np.nan).squeeze()
             if "zos_glor" in d.variables else None)
        if have_zos is None:
            have_zos = z is not None
        for i in range(u.shape[0]):
            months.append(str(ncdf.num2date(tv[i], tv.units))[:7])
            spd = np.sqrt(u[i] ** 2 + v[i] ** 2)
            mld = np.log10(np.clip(m[i], 1.0, None))
            zz = z[i] if z is not None else np.full_like(spd, np.nan)
            slices.append(np.stack([spd, mld, zz], -1).astype(np.float32))
        d.close()
    X = np.stack(slices)                          # [T,H,W,3]
    out = os.path.join(HERE, "cache", f"base025_{window}.npz")
    np.savez_compressed(out, X=X, months=np.array(months), lats=lats,
                        lons=lons,
                        chan=np.array(["cur_speed", "log_mld", "ssh"]))
    gb = X.nbytes / 1e9
    print(f"wrote {out}: T={len(months)} H={len(lats)} W={len(lons)} "
          f"({months[0]}..{months[-1]}, {gb:.2f} GB dense, zos={have_zos})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=sorted(WINDOWS), default="na")
    ap.add_argument("--start-year", type=int, default=1993)
    ap.add_argument("--end-year", type=int, default=2026)
    ap.add_argument("--assemble-only", action="store_true")
    a = ap.parse_args()
    if not a.assemble_only:
        fetch(a.window, a.start_year, a.end_year)
    assemble(a.window)


if __name__ == "__main__":
    main()
