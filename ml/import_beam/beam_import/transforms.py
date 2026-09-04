"""From downloaded bytes to `tf.train.Example` records — one per source-day.

THE ONE RULE OF THIS FILE: the 0.25 degree binning is `ml/aggregate_cadence.py`'s
`bin_plan` / `bin_slice`, and the pentad epoch is that module's `EPOCH` /
`bin_index`, all IMPORTED from the earth checkout. They are never
re-implemented, not even "just for this one case". A second implementation of
a bin rule is the defect class this repository has already paid for
(DESIGN §7), and a half-cell offset is invisible in every plot ever made.

The earth checkout is found through EARTH_REPO, else at ../earth.

Stage A does NOT decide anything a model depends on. It converts, it does not
average across days, it does not z-score, and it keeps every day it received.
Units are the source's own. Everything downstream of here is Stage B.

Transform kinds (the registry names one per source):

    bin025_days   CMEMS day files      -> one record per day, binned to 0.25°
    nc025_days    an already-binned    -> one record per day, no regridding
                  monthly NetCDF          (the `glorys_from_mirror` path)
    oisst_days    NCEI day files       -> one record per day, OISST's own grid
    ncep_var_year one 4x-daily file    -> one record per day: the daily mean,
                                          plus the daily mean of SQUARES for
                                          the two stress variables, which is
                                          what lets Stage B recover the exact
                                          within-pentad sigma
    rg_months     Roemmich-Gilson      -> one record per month (the 15th)
    series        a label / index file -> one record per date, grid `series`
    opaque        a mask, a zip, a     -> one record carrying the file bytes
                  GeoJSON                 verbatim (see `opaque_record`)
"""
from __future__ import annotations

import datetime as dt
import gzip
import os
import re
import shutil
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .example import make_example
from .sinks import sha256_bytes

BIN_DEG = 0.25
BIN_ALIGN = "point"

# One record is (date string, the Example's feature dict).
Record = Tuple[str, Dict[str, Any]]


def earth_repo() -> str:
    env = os.environ.get("EARTH_REPO")
    if env:
        return os.path.abspath(env)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.abspath(os.path.join(here, "..", "earth"))


def _ml_on_path() -> str:
    repo = earth_repo()
    ml = os.path.join(repo, "ml")
    if not os.path.isdir(ml):
        raise RuntimeError(
            f"no earth checkout at {repo!r}. Clone "
            "https://github.com/blauewelt/earth.git and set EARTH_REPO to it. "
            "The binning rule and the pentad epoch are imported from ml/, "
            "never re-implemented here.")
    if ml not in sys.path:
        sys.path.insert(0, ml)
    return ml


def import_bin_rule() -> Tuple[Callable, Callable]:
    """(bin_plan, bin_slice) from ml/aggregate_cadence.py."""
    _ml_on_path()
    from aggregate_cadence import bin_plan, bin_slice     # noqa: PLC0415
    return bin_plan, bin_slice


def import_epoch() -> Tuple[dt.date, Callable]:
    """(EPOCH, bin_index) from ml/aggregate_cadence.py — the pentad clock."""
    _ml_on_path()
    from aggregate_cadence import EPOCH, bin_index        # noqa: PLC0415
    return EPOCH, bin_index


def day_index(date: str) -> int:
    """Days since the tensor epoch. `bin = day_index // 5`."""
    epoch, _ = import_epoch()
    y, m, d = (int(x) for x in date.split("-"))
    return (dt.date(y, m, d) - epoch).days


# --------------------------------------------------------------------------
# building a record
# --------------------------------------------------------------------------
def _step(axis: np.ndarray) -> float:
    if len(axis) < 2:
        return 0.0
    return float(np.mean(np.diff(np.asarray(axis, dtype=np.float64))))


def gridded_record(item: Dict[str, Any], date: str, grid: str,
                   lat: np.ndarray, lon: np.ndarray,
                   var_names: Sequence[str], var_units: Sequence[str],
                   cube: np.ndarray, prov: Dict[str, Any],
                   transform: str) -> Record:
    """One Example: float32 [C, H, W] plus a bit-packed finite mask.

    `lat_values` / `lon_values` carry the axes explicitly ALONGSIDE the
    (lat0, lat_step, nlat) triple DESIGN §4 asks for. The triple cannot
    describe NCEP's gaussian latitudes or Roemmich-Gilson's grid, and a
    reader that has to guess is a reader that will guess wrong.
    """
    cube = np.ascontiguousarray(np.asarray(cube, dtype=np.float32))
    if cube.ndim != 3:
        raise ValueError(f"{item['item_id']}: values must be [C, H, W], "
                         f"got {cube.shape}")
    values = cube.tobytes()
    mask = np.packbits(np.isfinite(cube).ravel()).tobytes()
    feat = {
        "source": item["source"], "item_id": item["item_id"],
        "date": date, "day_index": day_index(date),
        "grid": grid,
        "lat0": float(lat[0]), "lat_step": _step(lat), "nlat": int(len(lat)),
        "lon0": float(lon[0]), "lon_step": _step(lon), "nlon": int(len(lon)),
        "lat_values": [float(x) for x in lat],
        "lon_values": [float(x) for x in lon],
        "var_names": list(var_names), "var_units": list(var_units),
        "values": values, "mask": mask, "shape": list(cube.shape),
        "source_url": prov.get("url", ""),
        "source_bytes": int(prov.get("bytes", 0)),
        "source_sha256": prov.get("sha256", ""),
        "fetched_at": prov.get("fetched_at", ""),
        "transform": transform,
    }
    return date, feat


def series_record(item: Dict[str, Any], date: str, var_names: Sequence[str],
                  values: Sequence[float], prov: Dict[str, Any]) -> Record:
    """A non-gridded series (RAPID, the cable, the ENSO indices): the same
    schema with `grid = series` and nlat = nlon = 1 (DESIGN §4)."""
    cube = np.asarray(values, dtype=np.float32).reshape(len(var_names), 1, 1)
    return gridded_record(item, date, "series", np.array([0.0]),
                          np.array([0.0]), var_names,
                          ["" for _ in var_names], cube, prov, "none")


def opaque_record(item: Dict[str, Any], date: str, path: str,
                  prov: Dict[str, Any]) -> Record:
    """A file that is not a grid and not a series: a GeoJSON coastline mask,
    an EN4 zip, ETOPO's 450 MB relief.

    DESIGN §4's schema has no place for these, so this is a documented
    EXTENSION: `grid = opaque`, empty `values`, and the file verbatim in a
    `raw` feature with its own sha256. Nothing downstream reads `raw` yet —
    it exists so the import can promise it lost nothing, and so a later stage
    can decide what to do with a coastline without re-downloading it.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    empty = np.zeros((0, 0, 0), dtype=np.float32)
    date_, feat = gridded_record(item, date, "opaque", np.array([0.0]),
                                 np.array([0.0]), [], [], empty, prov,
                                 "none")
    feat["raw"] = raw
    feat["raw_sha256"] = sha256_bytes(raw)
    feat["raw_name"] = os.path.basename(path)
    return date_, feat


# --------------------------------------------------------------------------
# reading NetCDF
# --------------------------------------------------------------------------
def _open(path: str):
    import netCDF4
    if path.endswith(".gz"):
        plain = path[:-3]
        if not os.path.exists(plain):
            with gzip.open(path, "rb") as src, open(plain, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 22)
        path = plain
    return netCDF4.Dataset(path, "r")


def _axis(ds, *names) -> np.ndarray:
    for n in names:
        if n in ds.variables:
            return np.asarray(ds.variables[n][:], dtype=np.float64)
    raise KeyError(f"none of {names} among {list(ds.variables)}")


def _dates_of(ds) -> List[str]:
    """The file's time axis as ISO days."""
    import netCDF4
    tv = ds.variables["time"]
    vals = np.atleast_1d(np.asarray(tv[:]))
    out = []
    for v in vals:
        when = netCDF4.num2date(v, tv.units,
                                only_use_cftime_datetimes=False,
                                only_use_python_datetimes=True)
        out.append(dt.date(when.year, when.month, when.day).isoformat())
    return out


def _pick_var(ds, prefer: Optional[str] = None):
    """The data variable in a one-variable file."""
    if prefer and prefer in ds.variables:
        return prefer, ds.variables[prefer]
    coords = {"time", "lat", "lon", "latitude", "longitude", "zlev", "depth",
              "level", "pressure", "time_bnds", "nbnds"}
    for name, var in ds.variables.items():
        if name.lower() in coords or var.ndim < 2:
            continue
        return name, var
    raise KeyError(f"no data variable among {list(ds.variables)}")


def _squeeze2(arr: np.ndarray, what: str) -> np.ndarray:
    out = np.squeeze(np.ma.filled(np.asarray(arr, dtype=np.float64), np.nan))
    if out.ndim != 2:
        raise ValueError(f"{what}: expected a 2-D slice, got {out.shape} — "
                         "the registry's chunk is wrong; do not widen it here")
    return out


# --------------------------------------------------------------------------
# the transforms
# --------------------------------------------------------------------------
def bin025_days(item, paths, prov, note=None) -> List[Record]:
    """CMEMS day files -> one record per day on the 0.25 degree point grid."""
    bin_plan, bin_slice = import_bin_rule()
    out: List[Record] = []
    plan = None
    variables = list(item.get("variables") or [])
    for path in sorted(paths):
        with _open(path) as ds:
            lat = _axis(ds, "latitude", "lat")
            lon = _axis(ds, "longitude", "lon")
            if plan is None:
                plan = bin_plan(lat, lon, BIN_DEG, BIN_ALIGN)
            present = [v for v in variables if v in ds.variables] or [
                _pick_var(ds)[0]]
            dates = _dates_of(ds)
            for t, date in enumerate(dates):
                cube = np.stack([
                    bin_slice(_squeeze2(ds.variables[v][t], f"{path}:{v}"),
                              plan) for v in present])
                units = [getattr(ds.variables[v], "units", "")
                         for v in present]
                out.append(gridded_record(
                    item, date, "point025", plan["lat"], plan["lon"],
                    present, units, cube, prov.get(path, {}),
                    "bin025:nearest-scatter:aggregate_cadence.bin_slice"))
    return out


def nc025_days(item, paths, prov, note=None) -> List[Record]:
    """A NetCDF that is ALREADY on the 0.25 degree grid — one record per day.

    This is the `glorys_from_mirror` path: the monthly chunks that already
    exist were binned by the same imported rule when they were made, so
    binning them again would be a second decision about the same bytes.
    """
    out: List[Record] = []
    variables = list(item.get("variables") or [])
    for path in sorted(paths):
        with _open(path) as ds:
            lat = _axis(ds, "lat", "latitude")
            lon = _axis(ds, "lon", "longitude")
            present = [v for v in variables if v in ds.variables] or [
                _pick_var(ds)[0]]
            for t, date in enumerate(_dates_of(ds)):
                cube = np.stack([
                    _squeeze2(ds.variables[v][t], f"{path}:{v}")
                    for v in present])
                units = [getattr(ds.variables[v], "units", "")
                         for v in present]
                out.append(gridded_record(
                    item, date, item.get("grid") or "point025", lat, lon,
                    present, units, cube, prov.get(path, {}), "none"))
    return out


OISST_VARS = ("sst", "ice")


def oisst_days(item, paths, prov, note=None) -> List[Record]:
    """NCEI day files -> one record per day, on OISST's OWN grid.

    NOT regridded. OISST's latitudes and longitudes are cell CENTRES, and
    choosing how to put them on our point grid is a modelling decision that
    belongs in Stage B beside every other one (it is `f3.interp2_nan` there).
    The `ice` trap is preserved rather than corrected: the variable declares
    units "percent" and its valid_range is 0..1, so the units string is
    copied through and Stage B trusts the range.
    """
    out: List[Record] = []
    for path in sorted(paths):
        with _open(path) as ds:
            lat = _axis(ds, "lat", "latitude")
            lon = _axis(ds, "lon", "longitude")
            present = [v for v in OISST_VARS if v in ds.variables]
            if not present:
                # The PSL yearly fallback files hold one variable each and
                # call sea ice `icec`, not `ice`.
                present = [_pick_var(ds)[0]]
            units = [getattr(ds.variables[v], "units", "") for v in present]
            for t, date in enumerate(_dates_of(ds)):
                cube = np.stack([_squeeze2(ds.variables[v][t],
                                           f"{path}:{v}") for v in present])
                out.append(gridded_record(
                    item, date, "oisst_center025", lat, lon, present, units,
                    cube, prov.get(path, {}), "oisst_day"))
    return out


def ncep_var_year(item, paths, prov, note=None) -> List[Record]:
    """One 4x-daily NCEP gaussian file -> one record per DAY.

    Two things are decided here and nowhere else, and both are forced by the
    Stage A / Stage B split:

    * The file holds four samples a day (1464 steps in a leap year), and a
      day record is their mean. The sign flip on `uflx`/`vflx` — stress ON
      the surface — is NOT applied here; `build_family7` applies it per
      sample before accumulating, and since it is a linear operation Stage B
      applies it once to the pentad mean instead, which is the same number.
    * For the two stress variables the record ALSO carries the daily mean of
      SQUARES, as a `<var>_sq` channel. `tau_*_std` is the within-pentad
      POPULATION sigma over the 6-hourly samples, and a sigma is not
      aggregable from a mean — but it is exactly recoverable as
      sqrt(E[x^2] - E[x]^2), and every day has the same four samples, so the
      pentad mean of the daily square-means is the pentad E[x^2] exactly.
    """
    out: List[Record] = []
    stem = str(item.get("var", "")).split(".")[0]
    want_sq = stem in _ncep_sigma_vars()
    for path in sorted(paths):
        with _open(path) as ds:
            lat = _axis(ds, "lat", "latitude")
            lon = _axis(ds, "lon", "longitude")
            name, var = _pick_var(ds, stem or None)
            units = getattr(var, "units", "")
            dates = _dates_of(ds)
            by_day: Dict[str, List[int]] = {}
            for k, date in enumerate(dates):
                by_day.setdefault(date, []).append(k)
            for date in sorted(by_day):
                steps = by_day[date]
                stack = np.stack([_squeeze2(var[k], f"{path}:{name}")
                                  for k in steps])
                with np.errstate(invalid="ignore"):
                    mean = np.nanmean(stack, axis=0)
                    chans, names, us = [mean], [name], [units]
                    if want_sq:
                        chans.append(np.nanmean(stack ** 2, axis=0))
                        names.append(f"{name}_sq")
                        us.append(f"({units})^2")
                out.append(gridded_record(
                    item, date, "ncep_t62", lat, lon, names, us,
                    np.stack(chans), prov.get(path, {}), "daily_mean"))
    return out


def _ncep_sigma_vars() -> Sequence[str]:
    """The variables that need a second moment — imported, not restated."""
    try:
        _ml_on_path()
        import build_family7 as f7                       # noqa: PLC0415
        return tuple(f7.NCEP_SIGMA)
    except Exception:                                    # noqa: BLE001
        return ("uflx", "vflx")


def rg_months(item, paths, prov, note=None) -> List[Record]:
    """Roemmich-Gilson -> one record per month, dated the 15th.

    The stored field is the ABSOLUTE temperature/salinity at the 16 AMOC
    pressure levels: the file's climatological MEAN plus that month's
    ANOMALY, which is what `build_family7`'s rg100 channels are. The level
    list is imported from `build_family3.LEVELS` and never retyped.
    """
    _ml_on_path()
    import build_family3 as f3                           # noqa: PLC0415
    levels = list(f3.LEVELS)
    out: List[Record] = []
    for path in sorted(paths):
        with _open(path) as ds:
            lat = _axis(ds, "LATITUDE", "latitude", "lat")
            lon = _axis(ds, "LONGITUDE", "longitude", "lon")
            press = _axis(ds, "PRESSURE", "pressure")
            lidx = [int(np.argmin(np.abs(press - p))) for p in levels]
            kinds = [("TEMPERATURE", "rg_t", "degC"),
                     ("SALINITY", "rg_s", "psu")]
            fields, names, units = [], [], []
            months = None
            for kind, prefix, unit in kinds:
                anom_name = f"ARGO_{kind}_ANOMALY"
                mean_name = f"ARGO_{kind}_MEAN"
                if anom_name not in ds.variables:
                    continue
                anom = ds.variables[anom_name]
                mean = np.ma.filled(
                    np.asarray(ds.variables[mean_name][lidx],
                               dtype=np.float64), np.nan)
                months = months or _rg_months_of(ds, anom.shape[0])
                fields.append((anom, mean, lidx))
                names += [f"{prefix}{int(p)}" for p in levels]
                units += [unit] * len(levels)
            if not fields:
                raise ValueError(f"{path}: no ARGO_*_ANOMALY variable")
            for t, month in enumerate(months or []):
                planes = []
                for anom, mean, li in fields:
                    a = np.ma.filled(np.asarray(anom[t, li],
                                                dtype=np.float64), np.nan)
                    planes.append(a + mean)
                cube = np.concatenate(planes, axis=0)
                out.append(gridded_record(
                    item, f"{month}-15", "rg_1deg_center", lat, lon,
                    names, units, cube, prov.get(path, {}), "rg_absolute"))
    return out


def _rg_months_of(ds, n: int) -> List[str]:
    """Roemmich-Gilson's time axis: months from 2004-01, the file's own base."""
    try:
        return [d[:7] for d in _dates_of(ds)]
    except Exception:                                          # noqa: BLE001
        out = []
        for k in range(n):
            out.append(f"{2004 + k // 12}-{k % 12 + 1:02d}")
        return out


_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def series(item, paths, prov, note=None) -> List[Record]:
    """A label or index file -> one record per date.

    Two readers, tried in order: NetCDF with a `time` axis (RAPID, MOVE), and
    a whitespace-separated numeric table whose first two columns are year and
    month or year and day-of-year (the Florida cable, SAMBA, the indices).
    A file neither reader understands is stored as `opaque` rather than
    dropped — Stage A's job is to lose nothing, not to understand everything.
    """
    out: List[Record] = []
    for path in sorted(paths):
        try:
            with _open(path) as ds:
                name, var = _pick_var(ds)
                dates = _dates_of(ds)
                vals = np.ma.filled(np.asarray(var[:], dtype=np.float64),
                                    np.nan).reshape(len(dates), -1)
                names = ([name] if vals.shape[1] == 1
                         else [f"{name}_{i}" for i in range(vals.shape[1])])
                for t, date in enumerate(dates):
                    out.append(series_record(item, date, names, vals[t],
                                             prov.get(path, {})))
            continue
        except Exception:                                      # noqa: BLE001
            pass
        rows = _numeric_rows(path)
        if not rows:
            out.append(opaque_record(item, "1982-01-01", path,
                                     prov.get(path, {})))
            continue
        for date, vals in rows:
            names = [f"v{i}" for i in range(len(vals))]
            out.append(series_record(item, date, names, vals,
                                     prov.get(path, {})))
    return out


def _numeric_rows(path: str) -> List[Tuple[str, List[float]]]:
    """(date, values) for a plain numeric table. Year + day-of-year, year +
    month, or year + month + day in the first columns; anything else fails
    the parse and the file is kept opaque instead."""
    out: List[Tuple[str, List[float]]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            nums = _NUM.findall(line)
            if len(nums) < 3:
                continue
            try:
                year = int(float(nums[0]))
            except ValueError:
                continue
            if not (1800 <= year <= 2100):
                continue
            a = float(nums[1])
            if a.is_integer() and 1 <= a <= 366 and len(nums) >= 3:
                if 1 <= a <= 12 and len(nums) >= 4 and \
                        float(nums[2]).is_integer() and 1 <= float(nums[2]) <= 31:
                    date = dt.date(year, int(a), int(float(nums[2])))
                    vals = [float(x) for x in nums[3:]]
                else:
                    date = dt.date(year, 1, 1) + dt.timedelta(days=int(a) - 1)
                    vals = [float(x) for x in nums[2:]]
            else:
                continue
            out.append((date.isoformat(), vals))
    return out


def opaque(item, paths, prov, note=None) -> List[Record]:
    out = []
    for path in sorted(paths):
        out.append(opaque_record(item, item.get("static_date", "1982-01-01"),
                                 path, prov.get(path, {})))
    return out


TRANSFORMS: Dict[str, Callable[..., List[Record]]] = {
    "bin025_days": bin025_days,
    "nc025_days": nc025_days,
    "oisst_days": oisst_days,
    "ncep_var_year": ncep_var_year,
    "rg_months": rg_months,
    "series": series,
    "opaque": opaque,
}


def to_examples(item: Dict[str, Any], paths: List[str],
                prov: Dict[str, Dict[str, Any]],
                note=None) -> Tuple[List[bytes], List[str], List[str]]:
    """Run the item's transform and serialise.

    Returns (payloads, sha256 of each record's `values`, dates covered), in
    date order — DESIGN §4 says records inside a shard are in date order, and
    a deterministic order is also what makes the shard's own sha256 mean
    something.
    """
    fn = TRANSFORMS[item["transform"]]
    records = fn(item, paths, prov, note=note)
    records.sort(key=lambda r: r[0])
    payloads = [make_example(feat) for _date, feat in records]
    shas = [sha256_bytes(feat["values"]) for _date, feat in records]
    return payloads, shas, [d for d, _ in records]
