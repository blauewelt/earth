"""NOAA ACSPO L3S-LEO daily 0.02-degree sea-surface temperature — the finest
satellite SST field there is (family 1.gf, E-082 wave 4; sharded tier G).

PLAIN ENGLISH. An infrared radiometer in a polar orbit measures the
temperature of the sea surface wherever the sky is clear. NOAA's ACSPO system
processes every one of them — VIIRS, AVHRR, MODIS — and SUPER-COLLATES them
onto one 2 km grid a day: each pixel takes the best-quality retrieval any
sensor got there that day. That is a factor of three finer than the 5 km CCI
record and twelve times finer than the 0.25-degree analysis the model has seen
so far, which matters because an SST FRONT is 2-10 km wide and an analysis
fitted to these measurements cannot show where the fit fails
(`family1gf.tex` §2).

THIS STORE IS PHASE C IN THE NOTE AND A MEASUREMENT HERE. The ledger puts
`sst_acspo02` at ~2.3 TB for the whole record and in phase C ("decisions, not
builds, until A and B have verdicts"). The brief accordingly asks for a probe
of 2020-01 and ONE YEAR (2020) built in lanes and PARKED, with the projection
reported — not the record. So `fetch_preflight` refuses any window wider than
`SST_ACSPO02_MAX_YEARS` (default 1) unless that is raised deliberately: the
guard is at dispatch, where the inputs are all it has cost (ml/CLAUDE.md
§0.3), rather than at hour five of a 2 TB fetch.

SOURCE, VERIFIED 2026-09-18 FROM THIS SANDBOX (CMR is anonymous; the granules
are Earthdata-protected at PO.DAAC, so the BYTES are a hosted runner's job):

  collection                    concept id            record        measured
  L3S_LEO_DY-STAR-v2.81         C2805339147-POCLOUD   2000-02-24 -> 423 MB/day
  L3S_LEO_PM-STAR-v2.81         C2805331435-POCLOUD   2002-07-04 -> 286/289 MB
  L3S_LEO_AM-STAR-v2.80         C2050135480-POCLOUD   2006-12-01 -> (v2.80)

  granule  20200101120000-STAR-L3S_GHRSST-SSTsubskin-LEO_Daily-ACSPO_V2.81-
           v02.0-fv01.0.nc, ONE A DAY for the DY collection; the PM and AM
           collections publish a DAY and a NIGHT granule each (`_PM_D`,
           `_PM_N`).

`SST_ACSPO02_COLLECTION` (default `DY`) picks which. **DY is the default and
the one the F = 5 daily layout fits**: one granule a day is one frame a day,
and the note's `F = 5` for a daily field means exactly that. PM and AM would
be TEN frames a bin (day and night separately) and are a different store
shape, so they are listed, reachable by the knob, and not the default — the
ledger's own words are "day and night as 2 frames/day" for the 5 km CCI
store, and that is the shape to give this one when phase C decides to.

CMR'S DECLARED GRANULE SIZE IS NOT EVIDENCE ABOUT THE FILE, and this
collection is the case that proves it. CMR reports 423 MB for 2020-01-01 and
2020-01-02 and **108 bytes for most of the rest of January 2020** (measured,
probe #172), while the real files download and open perfectly. So the size
field is a metadata artefact here, not a placeholder granule: `index` REPORTS
every granule declared under `MIN_GRANULE_BYTES` as
`granules_suspiciously_small`, so the list is visible before the fetch spends
anything, and the FETCH downloads them anyway and lets the FILE decide.
`cm.earthdata_download` already refuses a body short of its Content-Length or
empty, and `read_frame` refuses a file that will not open or whose lat/lon
axes do not check out — both `ctx.note_absent`, which leaves the year
unmarked, never a smaller frame (ml/CLAUDE.md, the 2026-09-14 rule). The first
version of this adapter refused on the DECLARED size and turned a whole month
into absences; verifying the artefact rather than its metadata is the rule
that was missing (ml/CLAUDE.md §0.1).

WHAT IS STORED. One group, `sst_acspo02`, on the GHRSST 0.02-degree grid
(18,000 x 9,000), F = 5 daily frames a bin, dtype **FLOAT16**, C = 2:

  sst              degrees Celsius (the source is subskin SST in kelvin)
  quality_level    the product's own 0..5 grade, as a NUMBER

The brief asks for "SST °C float16, quality uint8". THE SHARDED LAYOUT HAS
ONE dtype PER GROUP (`family1/sharded.py`: `dtype` is a group field), so two
channels cannot be float16 and uint8 in one group. float16 represents every
integer to 2,048 exactly, so 0..5 is carried without loss and the quality
level is stored as a float16 number; splitting it into a second uint8 group
would double the groups, the index files and the range reads for a channel
that costs 2 bytes. `tile_grid.json` says this in `channel_encoding`.

QUALITY IS A CHANNEL, NOT A FILTER. ACSPO grades every pixel 0 (no data) to 5
(best). A pixel graded 0 or 1 is not a measurement and becomes missing (NaN),
counted by grade; 2..5 are stored WITH their grade, so the consumer picks the
threshold rather than inheriting one — the rule `family1tf.tex` §6 states
("fields carry their quality layers as channels rather than as a mask, so the
consumer decides the threshold").

THE FILE'S LAYOUT IS RESOLVED, NOT TRANSCRIBED, exactly as `xco2`'s and
`irtb`'s are: there is no anonymous route to a PO.DAAC granule's variable
list, so `resolve()` matches each quantity against a candidate list, `index`
writes ONE real granule's WHOLE inventory into plan.json, and a required
quantity that matches nothing refuses with what the file does hold. Every
file's own `lat`/`lon` arrays are checked against the declared affine grid and
a mismatch is a refusal, never a silent reprojection.

MEMORY, BECAUSE THIS IS THE BIGGEST FRAME IN THE FAMILY. 18,000 x 9,000 x 2 is
324 M values: 648 MB as float16 and 1.30 GB as float32. `fetch_grid_year`
holds all F = 5 frames of a bin before it writes the shard, so the adapter
yields FLOAT16 arrays (the writer's own dtype, so `encode_frame` is a cast
with no copy of a wider array) and the buffer peaks at 3.24 GB rather than
6.5 GB. Bounds are masked per channel in float32, never by upcasting the whole
frame to float64. `stage_probe_grid` DOES convert one frame to float64 to
check the channel bounds — 2.59 GB plus its boolean temporaries — so the probe
peaks near 4 GB and fits a hosted runner's 16 GB but not this sandbox's 7 GB;
the smoke therefore runs on a shrunken grid (`SST_ACSPO02_SMOKE_GRID`).

THE FOOTPRINT. 0.02 degrees is 2.226 km at the equator, so
log2_fp = log2(2.226 / 27.83) = -3.644, the ledger's -3.64. log2_dt is
log2(1 / 5) = -2.32: a daily field.
"""
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.parse

import numpy as np

import build_family10_stores as f10b
from family1 import sharded as sh
from family1.adapters import _common as cm

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
COLLECTIONS = {
    "DY": ("C2805339147-POCLOUD", "L3S_LEO_DY-STAR-v2.81", "2.81",
           "one granule a day (day and night super-collated together)"),
    "PM": ("C2805331435-POCLOUD", "L3S_LEO_PM-STAR-v2.81", "2.81",
           "TWO granules a day, _PM_D and _PM_N — a ten-frame bin, not this "
           "store's five"),
    "AM": ("C2050135480-POCLOUD", "L3S_LEO_AM-STAR-v2.80", "2.80",
           "TWO granules a day, _AM_D and _AM_N, and processing v2.80"),
}
# `20200101120000-STAR-L3S_GHRSST-...-ACSPO_V2.81-v02.0-fv01.0.nc`
NAME = re.compile(r"^(\d{14})-STAR-L3S_GHRSST-")
FULL_W, FULL_H = 18000, 9000
DX, DY = 360.0 / FULL_W, 180.0 / FULL_H      # 0.02 degrees
LON0, LAT0 = -180.0, 90.0                    # the grid's WEST and NORTH edges
KELVIN = 273.15
FIRST_YEAR = 2000
# 423 MB is a real day; CMR reports 114 bytes for at least one granule. A
# file under this is reported at index time and refused at read time.
MIN_GRANULE_BYTES = 1_000_000
# grades 0 and 1 are "no data" and "bad": not measurements.
MIN_QUALITY = 2
WORKERS = 3
# The epoch as a DATETIME, as swot and irtb use it. `f10b.START` is a `date`,
# and `date + timedelta` drops the seconds, so the catalogue window's end came
# out as 00:00 of its last day. Harmless here — a granule is one DAY and CMR's
# temporal match is inclusive, so the last day's granule still overlaps — but
# it cost irtb the last seven three-hourly frames of every lane.
EPOCH = dt.datetime(1982, 1, 1)
assert EPOCH.date() == f10b.START, (EPOCH, f10b.START)

WANT = {
    "sst": (("sea_surface_temperature",), True),
    "quality": (("quality_level",), True),
    "lat": (("lat", "latitude"), True),
    "lon": (("lon", "longitude"), True),
    "time": (("time",), False),
}
CHANNELS = (("sst", "degC", -2.5, 40.0),
            ("quality_level", "ACSPO grade 0-5", float(MIN_QUALITY), 5.0))


class FormatError(ValueError):
    """A granule listing or an L3S file that does not match the product."""


# ================================================================ listing ==
def cmr_days(cid, t_lo, t_hi, attempts=4, count=None):
    """{date: entry} for the window. ANONYMOUS; the BYTES need a login.

    Paged by `cm.cmr_entries`, which checks the walk against `CMR-Hits`
    and raises `cm.CMRTruncated` on a cut listing (the 2026-09-23 irtb
    failure: a partial page during a slow hour of CMR read as the end).
    """
    q = {"collection_concept_id": cid, "sort_key": "start_date",
         "temporal": f"{t_lo},{t_hi}"}
    out = {}
    for g in cm.cmr_entries(CMR, q, attempts=attempts, count=count,
                            what=f"{cid} {t_lo}..{t_hi}"):
        title = str(g.get("title", ""))
        m = NAME.match(title)
        if not m:
            continue
        s = m.group(1)
        try:
            d = dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
        except ValueError:
            raise FormatError(f"{title}: {s[:8]} is not a date") from None
        url = None
        for ln in g.get("links", []):
            h = str(ln.get("href", ""))
            if h.startswith("https") and "protected" in h \
                    and h.endswith(".nc"):
                url = h
                break
        if url is None:
            raise FormatError(f"{title}: no protected https .nc link in "
                              f"its CMR entry")
        nb = int(float(g.get("granule_size") or 0) * 1e6)
        if d in out and out[d]["name"] == title:
            continue                    # the same granule listed twice
        if d in out:
            raise FormatError(
                f"two granules for {d}: {out[d]['name']} and {title} — "
                f"this collection is one a day, so a second one means "
                f"the day/night split (a PM or AM collection) rather "
                f"than the daily super-collation")
        out[d] = {"name": title, "url": url, "bytes": nb}
    return out


# ================================================================== grid ===
def grid(full_h=FULL_H, full_w=FULL_W):
    dx, dy = 360.0 / int(full_w), 180.0 / int(full_h)
    return {
        "H": int(full_h), "W": int(full_w),
        "pixel_deg": dx, "pixel_km_equator": round(dx * 111.31949, 4),
        "crs": "EPSG:4326",
        "proj4": "+proj=longlat +datum=WGS84 +no_defs",
        "x0": LON0, "dx": dx, "y0": LAT0, "dy": -dy,
        "coordinates": ("lon = x0 + (col + 0.5) * dx, lat = y0 + (row + 0.5) "
                        "* dy with dy NEGATIVE; x0/y0 are the grid's WEST and "
                        "NORTH edges"),
        "row_order": ("row 0 is the NORTHERNMOST row — the GHRSST "
                      "convention, whose `lat` array DESCENDS, and what "
                      "`grid_check` refuses a file for not having"),
        "extent": [LON0, LAT0 - 180.0, LON0 + 360.0, LAT0],
        "extent_note": "[west, south, east, north] in degrees; global",
        "no_tile_corners": ("a plain geographic grid: the affine rule above "
                            "IS the lat/lon of every pixel and tile corner"),
    }


def grid_check(lat, lon, full_h=FULL_H, full_w=FULL_W):
    dx, dy = 360.0 / int(full_w), 180.0 / int(full_h)
    if lat.shape != (int(full_h),) or lon.shape != (int(full_w),):
        raise FormatError(f"lat {lat.shape} and lon {lon.shape}; the declared "
                          f"grid is {full_h} x {full_w}")
    if lat[1] >= lat[0]:
        raise FormatError("the file's `lat` axis ASCENDS; this store declares "
                          "row 0 northernmost, the GHRSST convention")
    want_lat = LAT0 - (np.arange(int(full_h)) + 0.5) * dy
    want_lon = LON0 + (np.arange(int(full_w)) + 0.5) * dx
    for nm, got, wnt in (("lat", lat, want_lat), ("lon", lon, want_lon)):
        d = float(np.max(np.abs(got - wnt)))
        if d > 0.01:
            raise FormatError(f"the file's `{nm}` axis differs from the "
                              f"declared grid by up to {d:.4f} degrees "
                              f"(first {got[0]:.4f} vs {wnt[0]:.4f})")
    return {"lat_first": float(lat[0]), "lat_last": float(lat[-1]),
            "lon_first": float(lon[0]), "lon_last": float(lon[-1])}


# ============================================================== one frame ==
def inventory(ds, prefix=""):
    out = {}
    for name, v in ds.variables.items():
        out[f"{prefix}{name}"] = {
            "dtype": str(v.dtype), "shape": [int(x) for x in v.shape],
            "dims": list(v.dimensions),
            "units": str(getattr(v, "units", "")),
            "scale_factor": (float(getattr(v, "scale_factor"))
                             if hasattr(v, "scale_factor") else None),
            "add_offset": (float(getattr(v, "add_offset"))
                           if hasattr(v, "add_offset") else None),
            "fill": (float(getattr(v, "_FillValue"))
                     if hasattr(v, "_FillValue") else None),
            "long_name": str(getattr(v, "long_name", ""))[:120]}
    for name, g in ds.groups.items():
        out.update(inventory(g, f"{prefix}{name}/"))
    return out


def resolve(inv):
    got, missing = {}, []
    for q, (cands, required) in WANT.items():
        for c in cands:
            if c in inv:
                got[q] = c
                break
        else:
            if required:
                missing.append((q, cands))
    if missing:
        raise FormatError(
            "the L3S granule holds none of the candidate names for "
            + "; ".join(f"{q} ({', '.join(c)})" for q, c in missing)
            + ". The file's own variables are: " + ", ".join(sorted(inv)))
    return got


def _plane(ds, path, full_h, full_w):
    """One (time, lat, lon) or (lat, lon) variable -> float32 [H, W], NaN."""
    v = ds.variables[path]
    if len(v.shape) == 3:
        if int(v.shape[0]) != 1:
            raise FormatError(f"{path} has {v.shape[0]} time steps; the daily "
                              f"L3S granule has one")
        a = v[0, :, :]
    elif len(v.shape) == 2:
        a = v[:, :]
    else:
        raise FormatError(f"{path} has shape {v.shape}")
    # CAST TO FLOAT FIRST. `quality_level` is int8, and `np.ma.filled(..., nan)`
    # on an integer masked array raises rather than widening — so the array is
    # made float32 and only then does the mask become NaN.
    m = np.ma.getmaskarray(np.ma.asarray(a))
    out = np.asarray(np.ma.getdata(a), np.float32)
    out[m] = np.nan
    out[~np.isfinite(out)] = np.nan
    if out.shape != (int(full_h), int(full_w)):
        raise FormatError(f"{path} is {out.shape}, the declared grid is "
                          f"{(int(full_h), int(full_w))}")
    return out


def mask_bounds_f32(a, channels):
    """`SourceAdapter.mask_bounds`, per channel and IN FLOAT32.

    The framework's version compares against a float64 bounds array, which
    upcasts a 324-million-value frame to 2.59 GB. Per channel with float32
    scalars it is the same rule at a fifth of the memory, and the rule is the
    contract's: out of bounds becomes NaN and is COUNTED, never clipped.
    """
    out = {}
    for j, (nm, _u, lo, hi) in enumerate(channels):
        ch = a[:, :, j]
        with np.errstate(invalid="ignore"):
            bad = np.isfinite(ch) & ((ch < np.float32(lo))
                                     | (ch > np.float32(hi)))
        n = int(bad.sum())
        if n:
            ch[bad] = np.nan
            out[nm] = n
    return out


def read_frame(path, counts, full_h=FULL_H, full_w=FULL_W):
    """One L3S granule -> float16 [H, W, 2] of (SST degC, quality)."""
    import netCDF4
    ds = netCDF4.Dataset(path)
    try:
        inv = inventory(ds)
        names = resolve(inv)
        lat = np.asarray(ds.variables[names["lat"]][:], np.float64)
        lon = np.asarray(ds.variables[names["lon"]][:], np.float64)
        gmeta = grid_check(lat, lon, full_h, full_w)
        sst = _plane(ds, names["sst"], full_h, full_w)
        q = _plane(ds, names["quality"], full_h, full_w)
        units = str(getattr(ds.variables[names["sst"]], "units", ""))
    finally:
        ds.close()
    if units and units.strip().lower() not in ("kelvin", "k"):
        raise FormatError(f"`{names['sst']}` is in {units!r}; this adapter "
                          f"converts from kelvin and refuses to guess an "
                          f"offset")
    sst -= np.float32(KELVIN)
    # GRADE 0 AND 1 ARE NOT MEASUREMENTS. Counted by grade, then dropped.
    hist = counts.setdefault("quality_grade", {})
    qq = np.where(np.isfinite(q), q, -1.0)
    for g in (-1, 0, 1, 2, 3, 4, 5):
        n = int((np.rint(qq) == g).sum())
        if n:
            k = "absent" if g < 0 else str(g)
            hist[k] = hist.get(k, 0) + n
    drop = ~np.isfinite(q) | (q < np.float32(MIN_QUALITY))
    sst[drop] = np.nan
    q[drop] = np.nan
    # SST present but quality absent, or the other way round, is not a pixel
    both = np.isfinite(sst) & np.isfinite(q)
    sst[~both] = np.nan
    q[~both] = np.nan
    counts["pixels_in_frame"] = counts.get("pixels_in_frame", 0) + sst.size
    counts["pixels_valid"] = counts.get("pixels_valid", 0) + int(both.sum())
    res = counts.setdefault("resolution", {})
    for k, v in names.items():
        res[f"{k}={v}"] = res.get(f"{k}={v}", 0) + 1
    gm = counts.setdefault("grid_meta", {})
    gm[json.dumps(gmeta, sort_keys=True)[:200]] = 1
    a = np.stack([sst, q], axis=-1)
    oob = mask_bounds_f32(a, CHANNELS)
    if oob:
        counts["out_of_bounds"] = oob
    # FLOAT16 IS WHAT THE WRITER STORES, and yielding it is what keeps
    # `fetch_grid_year`'s five-frame buffer at 3.24 GB instead of 6.5.
    return a.astype(np.float16)


# ================================================================ adapter ==
class SSTACSPO02Adapter(sh.GridAdapter):
    store = "sst_acspo02"
    title = ("NOAA ACSPO L3S-LEO daily sea-surface temperature, 0.02 degrees "
             "— the finest satellite SST field; phase C, probed and parked")
    family = "1gf"
    distribution = "public"
    licence = {"name": "NOAA open data",
               "redistribution": "yes",
               "redistribution_confirmed": True,
               "derived_works": "free",
               "attribution": ("NOAA/STAR (2022), Advanced Clear-Sky "
                               "Processor for Ocean (ACSPO) L3S-LEO "
                               "super-collated SST, version 2.81, "
                               "distributed by NASA PO.DAAC")}
    time_dtype = "int32"
    credentials = ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    channels = CHANNELS
    log2_fp = float(np.log2(DX * 111.31949 / 27.83))             # -3.644
    log2_dt = float(np.log2(1.0 / 5.0))                          # -2.322
    per_year = True
    first_year = FIRST_YEAR
    frames_per_bin = 5
    frame_seconds = 86400
    dtype = "float16"
    tile = sh.TILE
    zstd_level = sh.DEFAULT_LEVEL
    note_estimate = {"bytes": 2.3e12,
                     "what": ("family1gf.tex: 162 M px x 9,700 days, "
                              "v ~ 0.3 -> ~2.3 TB for the whole record "
                              "(phase C)")}
    qc_policy = (
        "QUALITY IS A CHANNEL, NOT A FILTER, above the floor. ACSPO grades "
        "every pixel 0 (no data) to 5 (best); a pixel graded 0 or 1 is not a "
        "measurement and becomes missing, counted by grade "
        "(`quality_grade`), and 2..5 are stored WITH their grade so the "
        "consumer picks the threshold rather than inheriting one "
        "(family1tf.tex §6). A pixel with an SST and no grade, or a grade and "
        "no SST, is missing in both channels. SST is converted from the "
        "source's kelvin to degrees Celsius and a file whose `units` is not "
        "kelvin is a REFUSAL rather than a guessed offset. A value outside "
        "its channel's bounds becomes NaN for that channel and is counted, "
        "never clipped. A granule that cannot be opened or whose grid does "
        "not check out is `ctx.note_absent` and leaves the year unmarked — "
        "CMR reports at least one granule (2020-01-03) as 114 bytes, and a "
        "114-byte file is not a day of SST.")
    sources = tuple(
        f"{CMR}?collection_concept_id={cid} ({sn}, anonymous listing)"
        for (cid, sn, _v, _n) in COLLECTIONS.values())
    verified = (
        "2026-09-18 from this sandbox (CMR only — the granules are "
        "Earthdata-protected at PO.DAAC, so the BYTES are a hosted runner's "
        "job): the three collections and their records — "
        "L3S_LEO_DY-STAR-v2.81 (C2805339147-POCLOUD, 2000-02-24 ->), "
        "L3S_LEO_PM-STAR-v2.81 (C2805331435-POCLOUD, 2002-07-04 ->) and "
        "L3S_LEO_AM-STAR-v2.80 (C2050135480-POCLOUD, 2006-12-01 ->) — and "
        "2020-01's first granules: DY 422.93 MB on 2020-01-01 and 423.51 MB "
        "on 2020-01-02 (the note's 408 MB/day), PM publishing a _PM_D and a "
        "_PM_N granule of 285.86 and 288.71 MB on 2020-01-01, and "
        "**2020-01-03's DY granule reported as 1.09e-4 MB (114 bytes)**, "
        "which is why a granule under 1 MB is listed as suspicious at index "
        "time and refused at read time. NOT MEASURED HERE, and what the "
        "runner probe settles: the granule's internal variable layout "
        "(resolved against a candidate list, whole inventory written into "
        "plan.json), the lat/lon axes against the declared grid, the valid "
        "fraction after the quality floor, and the compressed bytes a tile "
        "and a frame — which is what replaces the note's 2.3 TB.")
    notes = ""
    smoke_window = ("2020-01-01", "2020-01-05")
    smoke_probe_month = "2020-01"

    def __init__(self):
        which = (os.environ.get("SST_ACSPO02_COLLECTION") or "DY").strip()
        if which not in COLLECTIONS:
            sys.exit(f"SST_ACSPO02_COLLECTION={which!r} is not one of "
                     f"{sorted(COLLECTIONS)}")
        self.which = which
        self.cid, self.short_name, self.proc_version, self.shape_note = \
            COLLECTIONS[which]
        try:
            self.max_years = int(os.environ.get("SST_ACSPO02_MAX_YEARS")
                                 or 1)
        except ValueError:
            sys.exit("SST_ACSPO02_MAX_YEARS must be an integer")
        self._smoke = (os.environ.get("SST_ACSPO02_SMOKE_GRID") or "")
        if self._smoke:
            w, h = (int(x) for x in self._smoke.split(","))
            self.full_w, self.full_h = w, h
        else:
            self.full_w, self.full_h = FULL_W, FULL_H
        self._days = None
        self._session = None
        self.notes = (
            f"PHASE C, PROBED AND PARKED. family1gf.tex puts this store at "
            f"~2.3 TB and in phase C; the brief asks for a probe and ONE "
            f"YEAR, so `fetch_preflight` refuses a window wider than "
            f"SST_ACSPO02_MAX_YEARS (= {self.max_years}) rather than "
            f"discovering the size at hour five. COLLECTION "
            f"{self.which} ({self.short_name}): {self.shape_note}. The PM and "
            f"AM collections publish a day and a night granule each, which is "
            f"a TEN-frame bin rather than this store's five, so they are "
            f"reachable by SST_ACSPO02_COLLECTION and are not the default.")
        if self._smoke:
            self.notes = (f"{self.notes}\nSMOKE GRID: "
                          f"SST_ACSPO02_SMOKE_GRID shrank the grid to "
                          f"{self.full_w} x {self.full_h}. This is a "
                          f"synthetic store.")

    # ------------------------------------------------------------- the grid --
    @property
    def grid(self):
        g = grid(self.full_h, self.full_w)
        if self._smoke:
            g["smoke"] = True
        return g

    def specs(self):
        out = super().specs()
        spec = out[self.store]
        spec["channel_encoding"] = (
            "BOTH channels are float16, because the sharded layout has one "
            "dtype per GROUP: `sst` in degrees Celsius and `quality_level` as "
            "the product's own 0..5 grade, which float16 carries exactly "
            "(every integer below 2,048 is exact). The brief's 'quality "
            "uint8' would need a second group, doubling the index files and "
            "the range reads for a 2-byte channel.")
        spec["quality_floor"] = {
            "min_quality_level": MIN_QUALITY,
            "note": ("grades 0 (no data) and 1 (bad) are not measurements "
                     "and are stored as missing, counted by grade; 2..5 are "
                     "stored with their grade so the consumer picks the "
                     "threshold")}
        spec["collection"] = {"key": self.which, "concept_id": self.cid,
                              "short_name": self.short_name,
                              "processing_version": self.proc_version,
                              "shape": self.shape_note}
        return out

    def record_frames(self, ctx, group):
        """The frames the WHOLE record holds, for the probe's extrapolation:
        the collection's first granule to its last, one a day."""
        days = self.days(ctx)
        if not days:
            return None
        return (max(days) - min(days)).days + 1

    # -------------------------------------------------------------- listing --
    def days(self, ctx):
        if self._days is not None:
            return self._days
        if ctx.source_dir:
            root = os.path.join(ctx.source_dir, self.store)
            out = {}
            for dirpath, _dirs, files in os.walk(root):
                for n in sorted(files):
                    m = NAME.match(n)
                    if not m or not n.endswith(".nc"):
                        continue
                    s = m.group(1)
                    d = dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
                    p = os.path.join(dirpath, n)
                    out[d] = {"name": n[:-3], "url": p,
                              "bytes": os.path.getsize(p)}
            if not out:
                sys.exit(f"REFUSING sst_acspo02: no L3S granule under {root} "
                         f"— the smoke's synthetic archive is missing")
        else:
            lo = (EPOCH + dt.timedelta(
                seconds=int(ctx.t_lo))).strftime("%Y-%m-%dT%H:%M:%SZ")
            hi = (EPOCH + dt.timedelta(
                seconds=int(ctx.t_hi))).strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                out = cmr_days(self.cid, lo, hi, attempts=ctx.a.attempts,
                               count=ctx.count_bytes)
            except (FormatError, cm.CMRTruncated) as e:
                sys.exit(f"REFUSING sst_acspo02: {e}")
            if not out:
                sys.exit(f"REFUSING sst_acspo02: CMR lists no "
                         f"{self.short_name} granule between {lo} and {hi} — "
                         f"an empty listing is a refusal (ml/CLAUDE.md, the "
                         f"2026-09-14 rule)")
        self._days = out
        return out

    def _get(self, ctx, entry):
        if ctx.source_dir:
            p = entry["url"]
            if not os.path.exists(p):
                raise f10b._NotFound(p)
            ctx.count_bytes(os.path.getsize(p))
            return p, False
        dest = os.path.join(ctx.scratch, self.store, entry["name"] + ".nc")
        if self._session is None:
            self._session = cm.earthdata_session()
        n, why = cm.earthdata_download(self._session, entry["url"], dest,
                                       attempts=max(1, ctx.a.attempts))
        if n is None:
            raise f10b._NotFound(f"{entry['url']}: {why}")
        return dest, True

    # ------------------------------------------------------------- contract --
    def fetch_preflight(self, ctx):
        if not ctx.source_dir and not cm.earthdata_ready():
            sys.exit(
                "REFUSING sst_acspo02: "
                "the L3S granules are Earthdata-protected and this "
                "process can authenticate to Earthdata Login in NEITHER way: "
                "no EARTHDATA_USERNAME / EARTHDATA_PASSWORD in the "
                "environment and no netrc naming urs.earthdata.nasa.gov. "
                "family1-build.yml gives both on a HOSTED runner from the "
                "repository secrets (ml/CLAUDE.md §6); a box gets neither. "
                "Nothing has been fetched.")
        # THE WINDOW'S OWN CALENDAR YEARS, not `ctx.years`: a tier-G "year"
        # is the bins whose FIRST day falls in it, so a window of 2020 alone
        # touches 2019 as well (bin 2775 opens 2019-12-28) and counting those
        # would refuse the very build the brief asks for.
        years = int(ctx.d_hi.year) - int(ctx.d_lo.year) + 1
        if self.max_years and years > self.max_years and \
                getattr(ctx.a, "stage", "") != "probe":
            sys.exit(
                f"REFUSING sst_acspo02: the window covers {years} year(s) and "
                f"SST_ACSPO02_MAX_YEARS is {self.max_years}. This store is "
                f"family1gf.tex's PHASE C — ~2.3 TB for the record — and the "
                f"brief scopes the first build to ONE year (2020) with the "
                f"parts parked and the projection reported. Raise "
                f"SST_ACSPO02_MAX_YEARS deliberately (adapter_env) once that "
                f"projection has been read. Nothing has been fetched.")
        return None

    def index(self, ctx):
        days = self.days(ctx)
        lo, hi = min(days), max(days)
        per_year, nb = {}, 0
        small, gaps = [], []
        for d, e in sorted(days.items()):
            per_year[str(d.year)] = per_year.get(str(d.year), 0) + 1
            nb += e["bytes"]
            if e["bytes"] and e["bytes"] < MIN_GRANULE_BYTES:
                small.append({"date": str(d), "name": e["name"],
                              "bytes": e["bytes"]})
        d = lo
        while d <= hi:
            if d not in days:
                gaps.append(str(d))
            d += dt.timedelta(days=1)
        out = {"dataset": (f"NOAA ACSPO L3S-LEO {self.which} daily 0.02 "
                           f"degree SST ({self.short_name})"),
               "url": f"{CMR}?collection_concept_id={self.cid}",
               "version": self.proc_version, "collection": self.cid,
               "collection_key": self.which,
               "collection_shape": self.shape_note,
               "files": len(days),
               "files_per_year": per_year,
               "bytes": int(nb),
               "bytes_per_day_mean": (round(nb / len(days), 1) if days
                                      else None),
               "record": [str(lo), str(hi)],
               "days_absent_in_window": gaps[:60],
               "days_absent_n": len(gaps),
               "granules_suspiciously_small": small[:40],
               "granules_suspiciously_small_n": len(small),
               "min_granule_bytes": MIN_GRANULE_BYTES,
               "grid": self.grid}
        # ONE REAL GRANULE, downloaded, INVENTORIED and grid-checked. The
        # first day whose granule is not one of the suspicious ones: reading
        # a 114-byte file would only prove it is 114 bytes.
        good = [d for d in sorted(days)
                if not days[d]["bytes"] or
                days[d]["bytes"] >= MIN_GRANULE_BYTES]
        # A window whose granules are ALL declared small is the ordinary case
        # for this collection (probe #172), so the first day is read anyway —
        # what decides is whether the FILE opens and its grid checks out.
        key = (good or sorted(days))[0]
        if getattr(ctx.a, "parts_from_hub", False):
            # a box assembling parked lanes has no Earthdata credentials
            # (ml/CLAUDE.md §6); the lane ledgers inventoried the granules
            # (irtb's whole-record assembly #721 died on this read, 2026-09-24)
            out["first_file"] = {"name": days[key]["name"], "day": str(key),
                                 "not_read": "parts from the Hub — the lane "
                                             "ledgers inventoried the files"}
            return out
        try:
            path, tmp = self._get(ctx, days[key])
        except (IOError, f10b._NotFound) as e:
            sys.exit(f"REFUSING sst_acspo02: the first granule "
                     f"{days[key]['name']} could not be read: {e}")
        try:
            import netCDF4
            ds = netCDF4.Dataset(path)
            try:
                inv = inventory(ds)
                dims = {n: int(len(dd)) for n, dd in ds.dimensions.items()}
                fmt = ds.data_model
                try:
                    names = resolve(inv)
                    lat = np.asarray(ds.variables[names["lat"]][:],
                                     np.float64)
                    lon = np.asarray(ds.variables[names["lon"]][:],
                                     np.float64)
                    gmeta = grid_check(lat, lon, self.full_h, self.full_w)
                except FormatError as e:
                    sys.exit(f"REFUSING sst_acspo02: {days[key]['name']}: "
                             f"{e}")
            finally:
                ds.close()
        finally:
            if tmp and os.path.exists(path):
                os.remove(path)
        out["first_file"] = {"name": days[key]["name"], "date": str(key),
                             "format": fmt, "dimensions": dims,
                             "variables": len(inv), "inventory": inv,
                             "resolution": names, "grid_checked": gmeta,
                             "bytes": days[key]["bytes"]}
        return out

    def fetch_frames(self, ctx, wanted):
        days = self.days(ctx)
        lo, hi = (min(days), max(days)) if days else (None, None)
        jobs = []
        for (g, b, f) in wanted:
            d = sh.frame_day(b, f, self.frame_seconds)
            why = None
            if lo is None or d < lo:
                why = "before_record"
            elif d > hi:
                why = "after_record"
            elif d not in days:
                why = "absent_upstream"
            jobs.append((g, b, f, d, why))
        todo = [j for j in jobs if j[4] is None]

        def work(job):
            g, b, f, d, _why = job
            try:
                return job, self._get(ctx, days[d]), None
            except f10b._NotFound:
                return job, None, "listed in CMR, and 404"
            except IOError as e:
                return job, None, f"{type(e).__name__}: {e}"

        workers = 1 if ctx.source_dir else WORKERS
        it = cm.ordered_map(work, todo, workers, lookahead=workers + 1)
        t0 = time.time()
        for job in jobs:
            g, b, f, d, why = job
            if why is not None:
                yield g, b, f, None, {"frame_missing": why}
                continue
            j2, got, err = next(it)
            if j2[:3] != (g, b, f):
                sys.exit(f"sst_acspo02: the download pool answered {j2[:3]} "
                         f"where {(g, b, f)} was asked for")
            if got is None:
                ctx.note_absent(str(d), f"{days[d]['name']}: {err}")
                continue
            path, tmp = got
            counts = {}
            try:
                # CMR'S DECLARED SIZE IS NOT EVIDENCE ABOUT THE FILE. Probe
                # #172 found 2020-01-01 and -02 declared at 423 MB and most
                # of the rest of the month at 108 BYTES, with the real files
                # downloading and opening fine — so the size field is a
                # metadata artefact of this collection, not a placeholder
                # granule. Refusing on it turned a whole month into
                # absences. The FILE decides: `cm.earthdata_download` already
                # refuses a body short of its Content-Length or empty, and
                # `read_frame` refuses one that will not open or whose grid
                # does not check out.
                if days[d]["bytes"] and \
                        days[d]["bytes"] < MIN_GRANULE_BYTES:
                    counts["granules_small_declared"] = \
                        counts.get("granules_small_declared", 0) + 1
                    counts["granule_bytes_declared_vs_read"] = [
                        {"date": str(d), "declared": days[d]["bytes"],
                         "read": os.path.getsize(path)}]
                arr = read_frame(path, counts, self.full_h, self.full_w)
            except FormatError as e:
                ctx.note_absent(str(d), f"{days[d]['name']}: {e}")
                continue
            except (OSError, IOError) as e:
                ctx.note_absent(str(d), f"{days[d]['name']}: not a readable "
                                        f"netCDF ({type(e).__name__}: {e})")
                continue
            finally:
                if tmp and os.path.exists(path):
                    os.remove(path)
            counts["files_read"] = 1
            counts["bytes_files"] = days[d]["bytes"]
            counts["fetch_seconds"] = round(time.time() - t0, 2)
            t0 = time.time()
            yield g, b, f, arr, counts

    # ---------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260918):
        truth = make_smoke_sources(root, d_lo, d_hi, seed)
        self.__init__()
        return truth


# ================================================================== smoke ==
SMOKE_W, SMOKE_H = 720, 360
SMOKE_ABSENT_DAY = dt.date(2020, 1, 3)       # a day the archive does not hold
SMOKE_OOB_DAY = dt.date(2020, 1, 2)


def smoke_fields(d, w, h, seed):
    """Deterministic (kelvin, grade) planes: a warm belt and a cloudy band."""
    rng = np.random.default_rng(seed + d.toordinal())
    yy = (np.arange(h, dtype=np.float64)[:, None] / h) * 180.0 - 90.0
    xx = (np.arange(w, dtype=np.float64)[None, :] / w) * 360.0 - 180.0
    k = 273.15 + 28.0 * np.cos(np.radians(yy)) ** 2 - 2.0 \
        + 1.5 * np.sin(np.radians(xx * 3 + d.day * 9))
    q = np.full((h, w), 5.0)
    polar = np.broadcast_to(np.abs(yy) > 70, (h, w))
    q[polar] = 0.0                              # ice / no retrieval
    cloud = rng.random((h, w)) < 0.25
    q[cloud] = 1.0                              # cloudy: grade 1, dropped
    band = np.broadcast_to(np.abs(yy) < 5, (h, w)) & ~cloud
    q[band] = 3.0                               # a band of middling grade
    return k, q


def write_day(path, d, w, h, oob=False):
    import netCDF4
    k, q = smoke_fields(d, w, h, 20260918)
    if oob:
        k[h // 2, w // 2] = 400.0               # 126.85 degC -> NaN, counted
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("time", 1)
    ds.createDimension("lat", h)
    ds.createDimension("lon", w)
    vt = ds.createVariable("time", "i4", ("time",))
    vt[:] = [int((d - dt.date(1981, 1, 1)).days * 86400)]
    vt.units = "seconds since 1981-01-01 00:00:00"
    vlat = ds.createVariable("lat", "f4", ("lat",))
    vlat[:] = LAT0 - (np.arange(h) + 0.5) * (180.0 / h)     # DESCENDING
    vlon = ds.createVariable("lon", "f4", ("lon",))
    vlon[:] = LON0 + (np.arange(w) + 0.5) * (360.0 / w)
    v = ds.createVariable("sea_surface_temperature", "f4",
                          ("time", "lat", "lon"),
                          fill_value=np.float32(-32768.0))
    v[0, :, :] = k
    v.units = "kelvin"
    vq = ds.createVariable("quality_level", "i1", ("time", "lat", "lon"))
    vq[0, :, :] = np.rint(q).astype(np.int8)
    vq.long_name = "SST measurement quality"
    ds.close()
    return np.asarray(k, np.float32).astype(np.float64), q


def make_smoke_sources(root, d_lo, d_hi, seed=20260918):
    """`<root>/sst_acspo02/<name>.nc`, one granule a day, and the truth:
    {(group, bin, frame): (stored float16 [H, W, 2] or None, reason)}."""
    os.environ["SST_ACSPO02_SMOKE_GRID"] = f"{SMOKE_W},{SMOKE_H}"
    os.environ.setdefault("SST_ACSPO02_COLLECTION", "DY")
    ad = SSTACSPO02Adapter()
    w, h = SMOKE_W, SMOKE_H
    base = os.path.join(root, "sst_acspo02")
    os.makedirs(base, exist_ok=True)
    fields = {}
    d = d_lo
    while d <= d_hi:
        if d != SMOKE_ABSENT_DAY:
            name = (f"{d.strftime('%Y%m%d')}120000-STAR-L3S_GHRSST-"
                    f"SSTsubskin-LEO_Daily-ACSPO_V2.81-v02.0-fv01.0")
            fields[d] = write_day(os.path.join(base, name + ".nc"), d, w, h,
                                  oob=(d == SMOKE_OOB_DAY))
        d += dt.timedelta(days=1)
    lo, hi = min(fields), max(fields)
    truth = {}
    t_lo = f10b.seconds_since_epoch(lo)
    t_hi = f10b.seconds_since_epoch(hi) + 86399
    for b in sh.bins_overlapping(t_lo, t_hi):
        for f in range(ad.frames_per_bin):
            day = sh.frame_day(b, f, ad.frame_seconds)
            if day < lo:
                truth[(ad.store, b, f)] = (None, "before_record")
            elif day > hi:
                truth[(ad.store, b, f)] = (None, "after_record")
            elif day not in fields:
                truth[(ad.store, b, f)] = (None, "absent_upstream")
            else:
                k, q = fields[day]
                sst = (np.asarray(k, np.float32) - np.float32(KELVIN))
                qf = np.asarray(q, np.float32)
                drop = ~np.isfinite(qf) | (qf < np.float32(MIN_QUALITY))
                sst = sst.copy()
                qf = qf.copy()
                sst[drop] = np.nan
                qf[drop] = np.nan
                a = np.stack([sst, qf], axis=-1)
                mask_bounds_f32(a, CHANNELS)
                truth[(ad.store, b, f)] = (a.astype(np.float16), None)
    return truth


ADAPTER = SSTACSPO02Adapter
