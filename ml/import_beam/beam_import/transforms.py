"""Putting the fetched bytes on our grid.

THE ONE RULE OF THIS FILE: the 0.25 degree binning is `ml/aggregate_cadence.py`'s
`bin_plan` / `bin_slice`, IMPORTED from the earth checkout. It is never
re-implemented, not even "just for this one case". A second implementation of a
bin rule is the defect class this repository has already paid for (DESIGN §6),
and a half-cell offset is invisible in every plot ever made from the result.

The earth checkout is found through the EARTH_REPO environment variable, and
otherwise at ../earth next to this handover directory.

Three transforms:

    passthrough        the file is already what we want to store (NCEP, RG,
                       the labels, the statics)
    bin025             CMEMS day files -> one monthly NetCDF on the 0.25 deg
                       point grid (GLORYS, DUACS)
    oisst_year_fold    365 NCEI day files -> one NetCDF for the year, with
                       sst and ice, on OISST's OWN grid (no regridding here)
"""
from __future__ import annotations

import datetime as dt
import os
import shutil
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# The grid the whole project is on. 0.25 degrees, samples ON multiples of
# 0.25 ("point" alignment) — the family-3 grid, measured, not chosen.
BIN_DEG = 0.25
BIN_ALIGN = "point"


def earth_repo() -> str:
    """Where the blauewelt/earth checkout is."""
    env = os.environ.get("EARTH_REPO")
    if env:
        return os.path.abspath(env)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.abspath(os.path.join(here, "..", "earth"))


def import_bin_rule() -> Tuple[Callable, Callable]:
    """Return (bin_plan, bin_slice) from ml/aggregate_cadence.py.

    Imported at call time rather than at module import, so that a machine with
    no earth checkout can still run `registry --check` and `manifest --print`.
    """
    repo = earth_repo()
    ml = os.path.join(repo, "ml")
    if not os.path.isdir(ml):
        raise RuntimeError(
            f"no earth checkout at {repo!r}. Clone "
            "https://github.com/blauewelt/earth.git and set EARTH_REPO to it. "
            "The binning rule is imported from ml/aggregate_cadence.py and is "
            "never re-implemented here.")
    if ml not in sys.path:
        sys.path.insert(0, ml)
    from aggregate_cadence import bin_plan, bin_slice     # noqa: PLC0415
    return bin_plan, bin_slice


# --------------------------------------------------------------------------
# passthrough
# --------------------------------------------------------------------------
def passthrough(item: Dict[str, Any], paths: List[str], outdir: str,
                note=None) -> List[str]:
    """The fetched file IS the stored file. Only the name is normalised."""
    if len(paths) != 1:
        raise ValueError(f"{item['item_id']}: passthrough wants exactly one "
                         f"file, got {len(paths)}")
    dest = os.path.join(outdir, os.path.basename(item["hub_path"]))
    os.makedirs(outdir, exist_ok=True)
    if os.path.abspath(paths[0]) != os.path.abspath(dest):
        shutil.move(paths[0], dest)
    return [dest]


# --------------------------------------------------------------------------
# bin025 — CMEMS day files onto the 0.25 degree point grid
# --------------------------------------------------------------------------
def _open(path: str):
    import netCDF4
    return netCDF4.Dataset(path, "r")


def _axis(ds, *names):
    for n in names:
        if n in ds.variables:
            return np.asarray(ds.variables[n][:], dtype=np.float64)
    raise KeyError(f"none of {names} in {ds.filepath()}")


def bin025(item: Dict[str, Any], paths: List[str], outdir: str,
           note=None) -> List[str]:
    """One month of CMEMS day files -> one NetCDF on the 0.25 degree grid."""
    import netCDF4
    bin_plan, bin_slice = import_bin_rule()
    if not paths:
        raise ValueError(f"{item['item_id']}: nothing to bin")

    with _open(paths[0]) as ds0:
        lat = _axis(ds0, "latitude", "lat")
        lon = _axis(ds0, "longitude", "lon")
        variables = [v for v in (item.get("variables") or [])
                     if v in ds0.variables]
    if not variables:
        raise ValueError(f"{item['item_id']}: none of "
                         f"{item.get('variables')} is in the served file")
    plan = bin_plan(lat, lon, BIN_DEG, BIN_ALIGN)

    os.makedirs(outdir, exist_ok=True)
    dest = os.path.join(outdir, os.path.basename(item["hub_path"]))
    tmp = dest + ".part"
    nt = len(paths)

    with netCDF4.Dataset(tmp, "w", format="NETCDF4") as out:
        out.createDimension("time", nt)
        out.createDimension("lat", plan["nlat"])
        out.createDimension("lon", plan["nlon"])
        vlat = out.createVariable("lat", "f8", ("lat",))
        vlon = out.createVariable("lon", "f8", ("lon",))
        vlat[:] = plan["lat"]
        vlon[:] = plan["lon"]
        vtime = out.createVariable("time", "i4", ("time",))
        vtime.units = "days since 1970-01-01"
        ovars = {v: out.createVariable(v, "f4", ("time", "lat", "lon"),
                                       zlib=True, complevel=4,
                                       fill_value=np.float32(np.nan))
                 for v in variables}
        out.binning = ("ml/aggregate_cadence.py bin_plan/bin_slice — mean of "
                       f"the finite source cells in each target cell, "
                       f"{BIN_DEG} deg, align={BIN_ALIGN}")
        out.source_dataset_id = str(item.get("dataset_id", ""))
        out.item_id = item["item_id"]

        for i, p in enumerate(sorted(paths)):
            with _open(p) as ds:
                tv = ds.variables.get("time")
                if tv is not None and len(tv) > 0:
                    import netCDF4 as nc
                    when = nc.num2date(np.asarray(tv[:])[0], tv.units)
                    vtime[i] = (dt.date(when.year, when.month, when.day)
                                - dt.date(1970, 1, 1)).days
                for v in variables:
                    arr = np.ma.filled(
                        np.asarray(ds.variables[v][:], dtype=np.float64),
                        np.nan)
                    arr = np.squeeze(arr)          # drop time / depth of size 1
                    if arr.ndim != 2:
                        raise ValueError(
                            f"{item['item_id']}: {v} in {os.path.basename(p)} "
                            f"is {arr.shape} after squeeze, expected 2-D. The "
                            "registry's chunk is wrong — do not widen it here.")
                    ovars[v][i, :, :] = bin_slice(arr, plan)
    os.replace(tmp, dest)
    return [dest]


# --------------------------------------------------------------------------
# oisst_year_fold — 365 day files -> one year
# --------------------------------------------------------------------------
# OISST is stored NATIVE. Its latitudes and longitudes are CELL CENTRES
# (0.125, 0.375, ...) rather than our point grid, and regridding it here would
# bake a half-cell decision into the mirror where the tensor build should be
# making it. DESIGN §5: the import mirrors bytes, it does not build tensors.
OISST_VARS = ("sst", "ice")


def oisst_year_fold(item: Dict[str, Any], paths: List[str], outdir: str,
                    note=None) -> List[str]:
    """Fold a year of NCEI day files into one NetCDF holding sst and ice."""
    import netCDF4
    if not paths:
        raise ValueError(f"{item['item_id']}: no day files")
    paths = sorted(paths)

    with _open(paths[0]) as ds0:
        lat = _axis(ds0, "lat", "latitude")
        lon = _axis(ds0, "lon", "longitude")
        present = [v for v in OISST_VARS if v in ds0.variables]
        units = {v: getattr(ds0.variables[v], "units", "") for v in present}
        vrange = {v: list(getattr(ds0.variables[v], "valid_range", []))
                  for v in present}
    if "sst" not in present:
        raise ValueError(f"{item['item_id']}: no `sst` variable in the day file")

    os.makedirs(outdir, exist_ok=True)
    dest = os.path.join(outdir, os.path.basename(item["hub_path"]))
    tmp = dest + ".part"

    with netCDF4.Dataset(tmp, "w", format="NETCDF4") as out:
        out.createDimension("time", len(paths))
        out.createDimension("lat", len(lat))
        out.createDimension("lon", len(lon))
        out.createVariable("lat", "f8", ("lat",))[:] = lat
        out.createVariable("lon", "f8", ("lon",))[:] = lon
        vtime = out.createVariable("time", "i4", ("time",))
        vtime.units = "days since 1970-01-01"
        ovars = {}
        for v in present:
            ov = out.createVariable(v, "f4", ("time", "lat", "lon"),
                                    zlib=True, complevel=4,
                                    fill_value=np.float32(np.nan))
            ov.units = units[v]
            if vrange[v]:
                ov.valid_range = np.asarray(vrange[v], dtype=np.float32)
            ovars[v] = ov
        out.grid = ("OISST native grid — lat/lon are CELL CENTRES, not our "
                    "point grid. Not regridded here, deliberately.")
        out.ice_units_trap = ("the `ice` variable declares units 'percent' but "
                              "its valid_range is 0..1; trust the range")
        out.item_id = item["item_id"]

        for i, p in enumerate(paths):
            stamp = os.path.basename(p).split(".")[1]          # oisst.YYYYMMDD.nc
            day = dt.date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))
            vtime[i] = (day - dt.date(1970, 1, 1)).days
            with _open(p) as ds:
                for v in present:
                    arr = np.ma.filled(
                        np.asarray(ds.variables[v][:], dtype=np.float64),
                        np.nan)
                    ovars[v][i, :, :] = np.squeeze(arr).astype(np.float32)
    os.replace(tmp, dest)
    return [dest]


TRANSFORMS: Dict[str, Callable[..., List[str]]] = {
    "passthrough": passthrough,
    "bin025": bin025,
    "oisst_year_fold": oisst_year_fold,
}


def apply_transform(item: Dict[str, Any], paths: List[str], outdir: str,
                    note=None) -> List[str]:
    return TRANSFORMS[item["transform"]](item, paths, outdir, note=note)
