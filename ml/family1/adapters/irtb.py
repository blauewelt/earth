"""NCEP/CPC globally merged infrared brightness temperature — how cold the
cloud tops are, everywhere in the tropics, every three hours (family 1.gf,
E-082 wave 4; sharded tier G).

PLAIN ENGLISH. A geostationary weather satellite's infrared channel measures
the temperature of whatever it can see from above: the ground where the sky is
clear, and the TOP of the cloud where it is not. A cold reading means a tall
cloud, and a tall cloud means deep convection — the thunderstorms that drive
the westerly wind bursts El Nino grows out of. NOAA's Climate Prediction
Center merges every geostationary satellite into one 4 km field from 60 S to
60 N every half hour, and NASA publishes it as `GPM_MERGIR`. It is the largest
raw source in this family's ledger (`family1gf.tex`: ~6.96 TB native), so
phase A takes the part the science needs first.

THE SCOPE IS A DECISION THE NOTE ALREADY MADE, and it is settings, not a
fork. `family1gf.tex` §5: "`irtb` is the largest raw source in the ledger and
its scientific value for El Nino is in the tropics, so phase B takes
30 S-30 N three-hourly (0.7 TB) and the full half-hourly 60 S-60 N record is a
phase-C decision; the frame count, not the pixel count, is what the design has
to hold." So the adapter has two knobs, read at CONSTRUCTION time (so the
fresh adapter `stage_probe_grid` builds for itself sees them) and recorded in
`tile_grid.json` AND in `notes`, which reaches store.json:

  IRTB_LAT_BAND   the half-width of the latitude band in degrees. 30 (the
                  default) is the tropics; 60 is the whole published field.
  IRTB_EVERY      how many of the archive's half-hourly frames to skip. 6
                  (the default) is three-hourly; 1 is every frame.

F = 432,000 / (1,800 * IRTB_EVERY), so the default is **F = 40** and
frame_seconds 10,800. AND THE FULL HALF-HOURLY FORM CANNOT BE BUILT TODAY:
F would be 240 and `family1/sharded.py`'s shard-index `frame_mask` is a
uint64, so `make_spec` refuses anything above 64 with "widen it before a
half-hourly group is built". IRTB_EVERY 4 (F = 60, two-hourly) is the finest
the current layout holds. That is a real limit of the storage design, found by
arithmetic rather than at hour three, and it is what a phase-C decision has to
pay for.

SOURCE, VERIFIED 2026-09-18 FROM THIS SANDBOX (CMR is anonymous; the files are
Earthdata-protected, so the BYTES are a hosted runner's job):
  collection  GPM_MERGIR v1, C1432254058-GES_DISC, 1998-01-01 -> now - 24 h
  granules    `merg_<YYYYMMDDHH>_4km-pixel.nc4`, ONE FILE AN HOUR holding TWO
              half-hourly frames, ~27.6-28.3 MB each (2015-01-01 measured:
              24 files, 698.5 MB, i.e. 29.1 MB an hour)
  bytes       https://data.gesdisc.earthdata.nasa.gov/data/MERGED_IR/
                  GPM_MERGIR.1/<YYYY>/<DDD>/merg_<YYYYMMDDHH>_4km-pixel.nc4
              (the URL is read out of the granule's own CMR entry, never
              constructed)

WHICH FRAME COMES FROM WHICH FILE. The archive's frame k of a day (k = 0..47,
half-hourly) lives in file hour k // 2 at its own frame index k % 2. A store
frame f of bin b starts at `b * 432000 + f * frame_seconds` seconds, so the
adapter turns that instant into (day, hour, sub-frame) and reads exactly one
file for it. With IRTB_EVERY = 6 only the frames on the hour at 00, 03, 06 ...
are wanted, so eight of the day's twenty-four files are read and each is read
ONCE — a three-hourly store costs an eighth of the archive's bytes, not all of
them.

WHAT IS STORED. One group, `irtb`, one channel, dtype **UINT8**: `tb`, the
brightness temperature in kelvin MINUS 160, which is the note's own encoding
("1: `tb` (uint8, K-160)"). The source is int16 kelvin, the physical range of
a merged IR field is about 160-330 K, and 0..254 covers 160-414 K exactly with
1 K steps; 255 is the layout's missing value. A kelvin outside [160, 414]
becomes NaN before the writer sees it and is COUNTED
(`out_of_bounds`), never clipped, so a corrupt file shows up instead of being
admitted as a plausible cloud top. THE CHANNEL'S DECLARED BOUNDS ARE IN THE
STORED UNIT (0..254 = K-160), because `sharded.check_sharded` checks a stored
tile against the channel table and a table in kelvin would refuse every tile.
`tile_grid.json` carries `channel_encoding` saying so in one sentence.

THE FILE'S LAYOUT IS RESOLVED, NOT TRANSCRIBED. There is no anonymous route
to a GES DISC file's variable list (an unauthenticated GET of the .nc4
answers 302 to Earthdata Login and then 401), so `resolve()` matches each
quantity against a candidate list, `index` downloads ONE real file and writes
its WHOLE inventory into plan.json, and a required quantity that matches
nothing is a refusal that prints what the file does hold. `read_frame`
additionally CHECKS the file's own `lat` and `lon` arrays against the declared
affine grid and refuses on a mismatch — the grid is a hypothesis every file
falsifies, the way `chirps05.tif_check` treats the CHIRPS transform.

THE FOOTPRINT. 4 km pixels: log2_fp = log2(4 / 27.83) = -2.80, the ledger's
number. log2_dt = log2((1800 / 86400) / 5) = -7.9 for a half-hourly SAMPLE,
and that is what the ledger states — the SLOT is three hours wide but the
measurement is a half-hourly snapshot, and the footprint field is the
measurement's support, not the slot's width (the rule `chirps05` states for
its own pentads). `tile_grid.json` says both.

SIZE, FROM THE NOTE UNTIL THE PROBE REPLACES IT. 16.3 M pixels x 8 frames a
day x ~10,500 days, uint8, "v = 1 but Tb compresses ~2x" -> ~0.7 TB for the
whole 1998 -> record. The brief scopes the first build to 2015-2020: 6 years,
73 bins a year, ~16.3 MB a raw frame and 40 frames a bin. The probe measures
the compressed number.
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
COLLECTION = "C1432254058-GES_DISC"
SHORT_NAME = "GPM_MERGIR"
NAME = re.compile(r"merg_(\d{10})_4km-pixel\.nc4$")
# The published grid, from the product's own documentation and CHECKED against
# every file's `lat`/`lon` arrays by `grid_check`.
FULL_W, FULL_H = 9896, 3298
LON0, LAT0 = -180.0, -60.0          # the field's WEST and SOUTH edges
DX = 360.0 / FULL_W                 # 0.036379
DY = 120.0 / FULL_H                 # 0.036386
ARCHIVE_FRAME_SECONDS = 1800        # the archive's own half-hourly cadence
FIRST_YEAR = 1998
KELVIN_OFFSET = 160.0               # the note's encoding: uint8 = K - 160
U8_MAX = 254                        # 255 is `sharded.U8_MISSING`
WORKERS = 4
# The epoch as a DATETIME. `f10b.START` is a `date`, and `date + timedelta`
# drops the seconds: the catalogue window's end came out as 00:00 of its last
# day, so the last owned bin of every lane lost its last seven three-hourly
# frames to `after_record` (bins 2428/2446/2465/2483 of the 2015 store, 28
# frames the archive holds). swot has always used this form.
EPOCH = dt.datetime(1982, 1, 1)
assert EPOCH.date() == f10b.START, (EPOCH, f10b.START)

WANT = {
    "tb": (("Tb", "tb", "IRbrightness_temperature"), True),
    "lat": (("lat", "latitude"), True),
    "lon": (("lon", "longitude"), True),
    "time": (("time",), False),
}


class FormatError(ValueError):
    """A granule listing or a MERGIR file that does not match the product."""


# ================================================================ listing ==
def cmr_hours(t_lo, t_hi, attempts=4, count=None):
    """{datetime hour: entry} for the window, from CMR. ANONYMOUS.

    Paged by `cm.cmr_entries`, which checks the walk against CMR's own
    `CMR-Hits` count and raises `cm.CMRTruncated` on a cut listing — the
    2026-09-23 failure, where a short page during a slow hour of CMR was
    taken as the end of the window and 15,388 hours were booked absent.
    """
    q = {"collection_concept_id": COLLECTION, "sort_key": "start_date",
         "temporal": f"{t_lo},{t_hi}"}
    out = {}
    for g in cm.cmr_entries(CMR, q, attempts=attempts, count=count,
                            what=f"{SHORT_NAME} {t_lo}..{t_hi}"):
        title = str(g.get("title", ""))
        m = NAME.search(title)
        if not m:
            continue
        s = m.group(1)
        try:
            when = dt.datetime(int(s[:4]), int(s[4:6]), int(s[6:8]),
                               int(s[8:10]))
        except ValueError:
            raise FormatError(f"{title}: {s} is not YYYYMMDDHH") from None
        url = None
        for ln in g.get("links", []):
            h = str(ln.get("href", ""))
            if h.startswith("https") and h.endswith(".nc4") \
                    and "opendap" not in h:
                url = h
                break
        if url is None:
            raise FormatError(f"{title}: no https .nc4 link in its CMR "
                              f"entry")
        out[when] = {"name": os.path.basename(title), "url": url,
                     "bytes": int(float(g.get("granule_size") or 0) * 1e6)}
    return out


# ================================================================== grid ===
def band_rows(band, full_h=FULL_H):
    """(row0, H) of the latitude band +/- `band` degrees on the full grid.

    The band is centred on the equator and snapped to whole rows; the exact
    edges the snap produced go into `tile_grid.json` rather than the nominal
    ones, because a reader must be able to place a pixel to the metre.
    `full_h` is the SOURCE field's row count — 3298 for the real archive, the
    smoke's own row count for a synthetic one, so the band arithmetic is the
    same code in both.
    """
    band = float(band)
    if not 0 < band <= 60:
        raise FormatError(f"IRTB_LAT_BAND {band} — the published field is "
                          f"60 S .. 60 N, so the band is in (0, 60]")
    dy = 120.0 / int(full_h)
    h = min(int(round(2.0 * band / dy)), int(full_h))
    r0 = (int(full_h) - h) // 2
    return r0, h


def grid(band, full_h=FULL_H, full_w=FULL_W):
    r0, h = band_rows(band, full_h)
    dx, dy = 360.0 / int(full_w), 120.0 / int(full_h)
    south = LAT0 + r0 * dy
    return {
        "H": int(h), "W": int(full_w),
        "pixel_deg_lon": dx, "pixel_deg_lat": dy,
        "pixel_km_equator": round(dx * 111.31949, 4),
        "crs": "EPSG:4326",
        "proj4": "+proj=longlat +datum=WGS84 +no_defs",
        "x0": LON0, "dx": dx, "y0": south, "dy": dy,
        "coordinates": ("lon = x0 + (col + 0.5) * dx, lat = y0 + (row + 0.5) "
                        "* dy with dy POSITIVE; x0/y0 are the group's WEST "
                        "and SOUTH edges"),
        "row_order": ("row 0 is the SOUTHERNMOST row — the source netCDF's "
                      "own order, whose `lat` array ASCENDS. `grid_check` "
                      "compares every file's lat/lon arrays against this "
                      "affine rule and refuses on a mismatch"),
        "extent": [LON0, south, LON0 + 360.0, south + h * dy],
        "extent_note": "[west, south, east, north] in degrees",
        "full_grid": {"H": int(full_h), "W": int(full_w), "south": LAT0,
                      "note": "the SOURCE field this band is cut from "
                              "(3298 x 9896 for the real archive)"},
        "band_rows": [int(r0), int(r0 + h)],
        "band_note": ("rows [band_rows[0], band_rows[1]) of the source "
                      "field above; IRTB_LAT_BAND chose them"),
        "no_tile_corners": ("a plain geographic grid: the affine rule above "
                            "IS the lat/lon of every pixel and tile corner"),
    }


def grid_check(lat, lon, band, full_h=FULL_H, full_w=FULL_W):
    """The file's own axes against the declared affine grid. Raises."""
    r0, h = band_rows(band, full_h)
    dx, dy = 360.0 / int(full_w), 120.0 / int(full_h)
    if lat.shape != (int(full_h),) or lon.shape != (int(full_w),):
        raise FormatError(f"lat {lat.shape} and lon {lon.shape}; the "
                          f"declared source field is {full_h} x {full_w}")
    if lat[1] <= lat[0]:
        raise FormatError("the file's `lat` axis DESCENDS; this store "
                          "declares row 0 southernmost, as the product's own "
                          "ascending axis does")
    want_lat = LAT0 + (np.arange(int(full_h)) + 0.5) * dy
    want_lon = LON0 + (np.arange(int(full_w)) + 0.5) * dx
    for nm, got, wnt in (("lat", lat, want_lat), ("lon", lon, want_lon)):
        d = float(np.max(np.abs(got - wnt)))
        if d > 0.02:
            raise FormatError(f"the file's `{nm}` axis differs from the "
                              f"declared grid by up to {d:.4f} degrees "
                              f"(first {got[0]:.4f} vs {wnt[0]:.4f}, last "
                              f"{got[-1]:.4f} vs {wnt[-1]:.4f})")
    return {"lat_first": float(lat[0]), "lat_last": float(lat[-1]),
            "lon_first": float(lon[0]), "lon_last": float(lon[-1]),
            "band_rows": [int(r0), int(r0 + h)],
            "band_lat": [float(lat[r0]), float(lat[r0 + h - 1])]}


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
            "the MERGIR file holds none of the candidate names for "
            + "; ".join(f"{q} ({', '.join(c)})" for q, c in missing)
            + ". The file's own variables are: " + ", ".join(sorted(inv)))
    return got


def read_frame(path, sub, band, counts, full_h=FULL_H,
               full_w=FULL_W):
    """One MERGIR file's sub-frame -> a float32 [H, W] of K-160, with NaN.

    `sub` is 0 or 1 — the archive puts two half-hourly frames in one hourly
    file. netCDF4 applies `scale_factor`/`add_offset` itself, so the array
    arrives in kelvin; the store's unit is K - 160 and the conversion happens
    here, once.
    """
    import netCDF4
    ds = netCDF4.Dataset(path)
    try:
        inv = inventory(ds)
        names = resolve(inv)
        tb = ds.variables[names["tb"]]
        if len(tb.shape) != 3:
            raise FormatError(f"{names['tb']} has shape {tb.shape}; the "
                              f"MERGIR field is (time, lat, lon)")
        nt = int(tb.shape[0])
        if not 0 <= int(sub) < nt:
            raise FormatError(f"sub-frame {sub} of a file with {nt} time "
                              f"step(s)")
        lat = np.asarray(ds.variables[names["lat"]][:], np.float64)
        lon = np.asarray(ds.variables[names["lon"]][:], np.float64)
        gmeta = grid_check(lat, lon, band, full_h, full_w)
        r0, h = band_rows(band, full_h)
        # MERGIR stores Tb as a masked int16 (a fill value, no scale factor);
        # widen to float64 BEFORE filling, because an int16 cannot hold NaN
        # (probe #427, the first read after the GES DISC approval, 2026-09-22).
        a = np.ma.filled(np.ma.asarray(tb[int(sub), r0:r0 + h, :])
                         .astype(np.float64), np.nan)
        res = counts.setdefault("resolution", {})
        for q, pth in names.items():
            res[f"{q}={pth}"] = res.get(f"{q}={pth}", 0) + 1
        counts["time_steps_per_file"] = \
            counts.setdefault("time_steps_per_file", {})
        counts["time_steps_per_file"][str(nt)] = \
            counts["time_steps_per_file"].get(str(nt), 0) + 1
        counts["grid_checked"] = counts.get("grid_checked", 0) + 1
        gm = counts.setdefault("grid_meta", {})
        gm[json.dumps(gmeta, sort_keys=True)[:200]] = 1
    finally:
        ds.close()
    if a.shape != (h, int(full_w)):
        raise FormatError(f"the band slice is {a.shape}, declared "
                          f"{(h, int(full_w))}")
    a[~np.isfinite(a)] = np.nan
    counts["pixels_valid"] = counts.get("pixels_valid", 0) + \
        int(np.isfinite(a).sum())
    counts["pixels_in_frame"] = counts.get("pixels_in_frame", 0) + a.size
    # kelvin -> the stored unit, K - 160, and NOTHING is clipped: the caller's
    # `mask_bounds` turns an out-of-range value into NaN and counts it.
    return (a - KELVIN_OFFSET).astype(np.float32)


# ================================================================ adapter ==
class IRTBAdapter(sh.GridAdapter):
    store = "irtb"
    title = ("NCEP/CPC merged infrared brightness temperature (GPM_MERGIR "
             "v1), 4 km — cloud-top temperature; phase A is the tropics "
             "three-hourly")
    family = "1gf"
    distribution = "public"
    licence = {"name": "NASA open data (CC0-equivalent)",
               "redistribution": "yes",
               "redistribution_confirmed": True,
               "derived_works": "free",
               "attribution": ("Janowiak, J., B. Joyce, P. Xie (2017), "
                               "NCEP/CPC L3 half hourly 4 km global "
                               "(60S-60N) merged IR V1, Goddard Earth "
                               "Sciences Data and Information Services "
                               "Center (GES DISC), "
                               "doi:10.5067/P4HZB9N27EKU")}
    time_dtype = "int32"
    credentials = ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    # The channel's bounds are in the STORED unit, K - 160, because
    # `sharded.check_sharded` compares a stored tile against this table.
    channels = (("tb", "K - 160 (uint8; add 160 for kelvin)", 0.0,
                 float(U8_MAX)),)
    log2_fp = float(np.log2(DX * 111.31949 / 27.83))             # -2.80
    log2_dt = float(np.log2((ARCHIVE_FRAME_SECONDS / 86400.0) / 5.0))  # -7.9
    per_year = True
    first_year = FIRST_YEAR
    dtype = "uint8"
    tile = sh.TILE
    # Tb is a smooth field with no NaN over most of the domain, so the high
    # zstd level the layout defaults to buys little for a lot of time; level 9
    # is the knee `lossyear` measured for uint8 imagery and is used until this
    # store's own probe says otherwise.
    zstd_level = 9
    note_estimate = {"bytes": 0.7e12,
                     "what": ("family1gf.tex: 16.3 M px x 8 frames/day x "
                              "~10,500 days, 'v = 1 but Tb compresses ~2x' "
                              "-> ~0.7 TB for the tropics three-hourly, whole "
                              "1998 -> record")}
    qc_policy = (
        "The merged IR field publishes no per-pixel quality flag; a pixel no "
        "satellite could see is the netCDF fill and becomes the layout's "
        "missing value 255. A kelvin outside [160, 414] — i.e. a stored value "
        "outside 0..254 — becomes missing and is COUNTED (`out_of_bounds`), "
        "never clipped: 160-330 K is the physical range of a merged IR field "
        "and anything past it is a broken file rather than a colder cloud. "
        "The store's unit is kelvin MINUS 160, the note's own encoding, and "
        "`tile_grid.json` says so in `channel_encoding`.")
    sources = (f"{CMR}?collection_concept_id={COLLECTION} (anonymous)",
               "https://data.gesdisc.earthdata.nasa.gov/data/MERGED_IR/"
               "GPM_MERGIR.1/<YYYY>/<DDD>/merg_<YYYYMMDDHH>_4km-pixel.nc4")
    verified = (
        "2026-09-18 from this sandbox (CMR only — a GES DISC .nc4 answers an "
        "anonymous GET with 302 to urs.earthdata.nasa.gov and then 401, so "
        "the BYTES are a hosted runner's job): the collection GPM_MERGIR v1 "
        "(C1432254058-GES_DISC, 1998-01-01 ->) and 2015-01-01's granules — "
        "24 files, one an hour, `merg_2015010100_4km-pixel.nc4` .. "
        "`...23...`, 698.5 MB in total, 27.6-28.3 MB each — with each "
        "granule's own https .nc4 link read from its CMR entry. NOT MEASURED "
        "HERE, and what the runner probe settles: the file's internal "
        "variable layout (resolved against a candidate list, with the whole "
        "inventory written into plan.json), whether an hourly file really "
        "holds TWO time steps, the lat/lon axes against the declared affine "
        "grid, and the compressed bytes a tile and a frame.")
    smoke_window = ("2015-01-01", "2015-01-05")
    smoke_probe_month = "2015-01"

    def __init__(self):
        self.band = float(os.environ.get("IRTB_LAT_BAND") or 30.0)
        try:
            self.every = int(os.environ.get("IRTB_EVERY") or 6)
        except ValueError:
            sys.exit("IRTB_EVERY must be an integer number of half-hours")
        if self.every < 1:
            sys.exit("IRTB_EVERY must be at least 1 (every archive frame)")
        fs = ARCHIVE_FRAME_SECONDS * self.every
        if sh.BIN_SECONDS % fs:
            sys.exit(f"IRTB_EVERY {self.every} gives {fs} s a frame, which "
                     f"does not divide the {sh.BIN_SECONDS}-second bin")
        self.frame_seconds = fs
        self.frames_per_bin = sh.BIN_SECONDS // fs
        if self.frames_per_bin > sh.MAX_FRAMES_BITMASK:
            sys.exit(
                f"IRTB_EVERY {self.every} gives F = {self.frames_per_bin} "
                f"frames a bin, and family1/sharded.py's shard-index "
                f"`frame_mask` is a uint64 that holds at most "
                f"{sh.MAX_FRAMES_BITMASK}. The note's full half-hourly form "
                f"(F = 240) therefore CANNOT be stored by the current layout; "
                f"IRTB_EVERY 4 (F = 60, two-hourly) is the finest it holds. "
                f"Widening `frame_mask` is the phase-C change.")
        self._smoke = (os.environ.get("IRTB_SMOKE_GRID") or "")
        if self._smoke:
            w, h = (int(x) for x in self._smoke.split(","))
            self.full_w, self.full_h = w, h
        else:
            self.full_w, self.full_h = FULL_W, FULL_H
        self._hours = None
        self._session = None
        scope = (f"IRTB_LAT_BAND {self.band:g} degrees and IRTB_EVERY "
                 f"{self.every} (every {self.every * 30} minutes, F = "
                 f"{self.frames_per_bin})")
        self.notes = (
            f"SCOPE: {scope}. The published field is 60 S .. 60 N half-hourly; "
            f"this store holds the band and cadence named above, which "
            f"family1gf.tex §5 chose for phase A ('the tropics three-hourly "
            f"first'). A consumer reading this store is NOT reading the whole "
            f"product, and store.json says so here. The full half-hourly form "
            f"needs F = 240 and the layout's frame_mask holds 64, so it is a "
            f"phase-C change to family1/sharded.py, not a re-run.")
        if self._smoke:
            self.notes = (f"{self.notes}\nSMOKE GRID: IRTB_SMOKE_GRID shrank "
                          f"the grid. This is a synthetic store.")

    # ------------------------------------------------------------- the grid --
    @property
    def grid(self):
        g = grid(self.band, self.full_h, self.full_w)
        if self._smoke:
            g["smoke"] = True
        return g

    def specs(self):
        out = super().specs()
        spec = out[self.store]
        spec["channel_encoding"] = (
            "`tb` is stored as uint8 = round(kelvin) - 160, so kelvin = "
            "value + 160 and the channel's declared bounds 0..254 are in the "
            "STORED unit. 255 is missing (no satellite saw the pixel, or the "
            "kelvin was outside [160, 414] and was counted out of bounds).")
        spec["scope"] = {"lat_band_deg": self.band,
                         "every_half_hours": self.every,
                         "archive_frame_seconds": ARCHIVE_FRAME_SECONDS,
                         "note": ("the archive is half-hourly over "
                                  "60 S .. 60 N; this group is a BAND and a "
                                  "SUBSAMPLE of it, chosen by IRTB_LAT_BAND "
                                  "and IRTB_EVERY")}
        spec["frame_rule"] = (
            f"frame f of bin b is the archive's half-hourly SNAPSHOT at "
            f"b * 432000 + f * {self.frame_seconds} seconds since the epoch. "
            f"The slot is {self.frame_seconds} s wide and the measurement is "
            f"{ARCHIVE_FRAME_SECONDS} s of support, which is what log2_dt "
            f"carries — the footprint is the measurement's, not the slot's.")
        return out

    def record_frames(self, ctx, group):
        """How many frames the WHOLE record holds, for the probe's
        extrapolation: the scoped cadence over 1998-01-01 -> the archive's
        last listed hour."""
        hours = self.hours(ctx)
        if not hours:
            return None
        lo, hi = min(hours), max(hours)
        span = (hi - lo).total_seconds() + 3600
        return int(span // self.frame_seconds)

    # -------------------------------------------------------------- listing --
    def hours(self, ctx):
        if self._hours is not None:
            return self._hours
        if ctx.source_dir:
            root = os.path.join(ctx.source_dir, self.store)
            out = {}
            for dirpath, _dirs, files in os.walk(root):
                for n in sorted(files):
                    m = NAME.search(n)
                    if not m:
                        continue
                    s = m.group(1)
                    when = dt.datetime(int(s[:4]), int(s[4:6]), int(s[6:8]),
                                       int(s[8:10]))
                    p = os.path.join(dirpath, n)
                    out[when] = {"name": n, "url": p,
                                 "bytes": os.path.getsize(p)}
            if not out:
                sys.exit(f"REFUSING irtb: no merg_*.nc4 under {root} — the "
                         f"smoke's synthetic archive is missing")
        else:
            lo = (EPOCH + dt.timedelta(
                seconds=int(ctx.t_lo))).strftime("%Y-%m-%dT%H:%M:%SZ")
            hi = (EPOCH + dt.timedelta(
                seconds=int(ctx.t_hi))).strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                out = cmr_hours(lo, hi, attempts=ctx.a.attempts,
                                count=ctx.count_bytes)
            except (FormatError, cm.CMRTruncated) as e:
                sys.exit(f"REFUSING irtb: {e}")
            if not out:
                sys.exit(f"REFUSING irtb: CMR lists no {SHORT_NAME} granule "
                         f"between {lo} and {hi} — an empty listing is a "
                         f"refusal (ml/CLAUDE.md, the 2026-09-14 rule)")
        self._hours = out
        return out

    def _slot_source(self, b, f):
        """(the hour's datetime, the sub-frame index) for store frame (b, f)."""
        t = sh.frame_start_seconds(b, f, self.frame_seconds)
        when = dt.datetime(f10b.START.year, f10b.START.month,
                           f10b.START.day) + dt.timedelta(seconds=int(t))
        sub = 1 if when.minute >= 30 else 0
        return when.replace(minute=0, second=0, microsecond=0), sub

    def _get(self, ctx, entry):
        if ctx.source_dir:
            p = entry["url"]
            if not os.path.exists(p):
                raise f10b._NotFound(p)
            ctx.count_bytes(os.path.getsize(p))
            return p, False
        dest = os.path.join(ctx.scratch, self.store, entry["name"])
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
                "REFUSING irtb: "
                "the GPM_MERGIR files are Earthdata-protected and this "
                "process can authenticate to Earthdata Login in NEITHER way: "
                "no EARTHDATA_USERNAME / EARTHDATA_PASSWORD in the "
                "environment and no netrc naming urs.earthdata.nasa.gov. "
                "family1-build.yml gives both on a HOSTED runner from the "
                "repository secrets (ml/CLAUDE.md §6); a box gets neither. "
                "Nothing has been fetched.")
        return None

    def index(self, ctx):
        hours = self.hours(ctx)
        lo, hi = min(hours), max(hours)
        per_year, per_day = {}, {}
        nb = 0
        for when, e in hours.items():
            per_year[str(when.year)] = per_year.get(str(when.year), 0) + 1
            per_day[str(when.date())] = per_day.get(str(when.date()), 0) + 1
            nb += e["bytes"]
        missing = sorted(str(k) for k in _missing_hours(hours))
        out = {"dataset": (f"{SHORT_NAME} v1 — NCEP/CPC globally merged "
                           f"infrared brightness temperature, 4 km, "
                           f"60 S .. 60 N, half-hourly"),
               "url": f"{CMR}?collection_concept_id={COLLECTION}",
               "version": "1", "collection": COLLECTION,
               "files": len(hours),
               "files_per_year": per_year,
               "hours_per_day_histogram": _hist(per_day.values()),
               "bytes": int(nb),
               "record": [str(lo), str(hi)],
               "hours_absent_in_window": missing[:50],
               "hours_absent_n": len(missing),
               "scope": {"lat_band_deg": self.band,
                         "every_half_hours": self.every,
                         "frames_per_bin": self.frames_per_bin,
                         "frame_seconds": self.frame_seconds},
               "grid": self.grid}
        # ONE REAL FILE, downloaded, INVENTORIED and grid-checked — unless
        # the parts come from the Hub: a box assembling parked lanes has no
        # Earthdata credentials (ml/CLAUDE.md §6), every lane's ledger already
        # inventoried its own files, and the whole-record irtb assembly
        # (family1-build #721, 2026-09-24) died here on a 401 for exactly that
        # reason. `needs_source` says the same thing one level up.
        if getattr(ctx.a, "parts_from_hub", False):
            out["first_file"] = {"name": hours[lo]["name"], "hour": str(lo),
                                 "not_read": "parts from the Hub — the lane "
                                             "ledgers inventoried the files"}
            return out
        key = lo
        try:
            path, tmp = self._get(ctx, hours[key])
        except (IOError, f10b._NotFound) as e:
            sys.exit(f"REFUSING irtb: the first file {hours[key]['name']} "
                     f"could not be read: {e}")
        try:
            import netCDF4
            ds = netCDF4.Dataset(path)
            try:
                inv = inventory(ds)
                dims = {n: int(len(d)) for n, d in ds.dimensions.items()}
                fmt = ds.data_model
                try:
                    names = resolve(inv)
                    lat = np.asarray(ds.variables[names["lat"]][:],
                                     np.float64)
                    lon = np.asarray(ds.variables[names["lon"]][:],
                                     np.float64)
                    gmeta = grid_check(lat, lon, self.band, self.full_h,
                                       self.full_w)
                except FormatError as e:
                    sys.exit(f"REFUSING irtb: {hours[key]['name']}: {e}")
            finally:
                ds.close()
        finally:
            if tmp and os.path.exists(path):
                os.remove(path)
        out["first_file"] = {"name": hours[key]["name"], "hour": str(key),
                             "format": fmt, "dimensions": dims,
                             "variables": len(inv), "inventory": inv,
                             "resolution": names, "grid_checked": gmeta,
                             "bytes": hours[key]["bytes"]}
        return out

    def fetch_frames(self, ctx, wanted):
        hours = self.hours(ctx)
        jobs = []
        for (g, b, f) in wanted:
            when, sub = self._slot_source(b, f)
            why = None
            if hours:
                lo, hi = min(hours), max(hours)
                if when < lo:
                    why = "before_record"
                elif when > hi:
                    why = "after_record"
                elif when not in hours:
                    why = "absent_upstream"
            jobs.append((g, b, f, when, sub, why))
        todo = [j for j in jobs if j[5] is None]

        def work(job):
            g, b, f, when, sub, _why = job
            try:
                return job, self._get(ctx, hours[when]), None
            except f10b._NotFound:
                return job, None, "listed in CMR, and 404"
            except IOError as e:
                return job, None, f"{type(e).__name__}: {e}"

        workers = 1 if ctx.source_dir else WORKERS
        it = cm.ordered_map(work, todo, workers, lookahead=2 * workers)
        t0 = time.time()
        for job in jobs:
            g, b, f, when, sub, why = job
            if why is not None:
                yield g, b, f, None, {"frame_missing": why}
                continue
            got, err = None, None
            j2, got, err = next(it)
            if j2[:3] != (g, b, f):
                sys.exit(f"irtb: the download pool answered {j2[:3]} where "
                         f"{(g, b, f)} was asked for")
            if got is None:
                ctx.note_absent(f"{when:%Y-%m-%dT%H}Z frame {sub}",
                                f"{hours[when]['name']}: {err}")
                continue
            path, tmp = got
            counts = {}
            try:
                arr = read_frame(path, sub, self.band, counts,
                                 self.full_h, self.full_w)
            except FormatError as e:
                sys.exit(f"REFUSING irtb: {hours[when]['name']}: {e}")
            except (OSError, IOError) as e:
                ctx.note_absent(f"{when:%Y-%m-%dT%H}Z frame {sub}",
                                f"{hours[when]['name']}: not a readable "
                                f"netCDF ({type(e).__name__}: {e})")
                continue
            finally:
                if tmp and os.path.exists(path):
                    os.remove(path)
            v = arr.reshape(-1, 1).astype(np.float64)
            oob = self.mask_bounds(v)
            arr = v.reshape(arr.shape).astype(np.float32)
            if oob:
                counts["out_of_bounds"] = oob
            counts["files_read"] = 1
            counts["bytes_files"] = hours[when]["bytes"]
            counts["fetch_seconds"] = round(time.time() - t0, 2)
            t0 = time.time()
            yield g, b, f, arr, counts

    # ---------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260918):
        truth = make_smoke_sources(root, d_lo, d_hi, seed)
        self.__init__()
        return truth


def _missing_hours(hours):
    if not hours:
        return []
    lo, hi = min(hours), max(hours)
    out, t = [], lo
    while t <= hi:
        if t not in hours:
            out.append(t)
        t += dt.timedelta(hours=1)
    return out


def _hist(values):
    out = {}
    for v in values:
        out[str(int(v))] = out.get(str(int(v)), 0) + 1
    return out


# ================================================================== smoke ==
SMOKE_W, SMOKE_H = 512, 128
# One LISTED hour whose file is unreadable is exercised by the test; one hour
# the archive simply does not hold is `absent_upstream`.
SMOKE_ABSENT_HOUR = dt.datetime(2015, 1, 2, 12)
SMOKE_OOB_HOUR = dt.datetime(2015, 1, 1, 3)


def smoke_field(when, sub, w, h):
    """A deterministic kelvin field: warm ground, a cold convective blob."""
    yy = (np.arange(h, dtype=np.float64)[:, None] - h / 2) / h
    xx = (np.arange(w, dtype=np.float64)[None, :] - w / 2) / w
    k = (when.hour * 2 + sub) % 11
    a = 290.0 - 60.0 * np.exp(-((xx * 6 - np.sin(k)) ** 2
                                + (yy * 6 - np.cos(k)) ** 2))
    a[:2, :] = np.nan                     # a strip no satellite saw
    return a


def write_hour(path, when, w, h, oob=False):
    import netCDF4
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("time", 2)
    ds.createDimension("lat", h)
    ds.createDimension("lon", w)
    vt = ds.createVariable("time", "f8", ("time",))
    vt[:] = [0.0, 0.5]
    vt.units = "hours since 1970-01-01"
    vlat = ds.createVariable("lat", "f4", ("lat",))
    vlat[:] = LAT0 + (np.arange(h) + 0.5) * (120.0 / h)
    vlon = ds.createVariable("lon", "f4", ("lon",))
    vlon[:] = LON0 + (np.arange(w) + 0.5) * (360.0 / w)
    v = ds.createVariable("Tb", "f4", ("time", "lat", "lon"),
                          fill_value=np.float32(-9999.0))
    frames = []
    for sub in (0, 1):
        a = smoke_field(when, sub, w, h)
        if oob and sub == 0:
            a[h // 2, w // 2] = 600.0      # out of bounds -> missing, counted
        # THE TRUTH IS WHAT THE FILE HOLDS, not what was computed: the
        # variable is float32, so the float64 field loses bits on the way in
        # and a truth built from the float64 copy disagrees with the store by
        # one unit wherever the float32 value rounds the other way.
        a = np.asarray(a, np.float32).astype(np.float64)
        frames.append(a)
        v[sub, :, :] = np.where(np.isfinite(a), a, -9999.0)
    v.units = "K"
    v.long_name = "brightness temperature"
    ds.close()
    return frames


def make_smoke_sources(root, d_lo, d_hi, seed=20260918):
    """`<root>/irtb/<YYYY>/<DDD>/merg_<YYYYMMDDHH>_4km-pixel.nc4` and the
    truth: {(group, bin, frame): (stored array or None, reason or None)}."""
    os.environ["IRTB_SMOKE_GRID"] = f"{SMOKE_W},{SMOKE_H}"
    os.environ.setdefault("IRTB_LAT_BAND", "60")
    os.environ.setdefault("IRTB_EVERY", "6")
    ad = IRTBAdapter()
    w, h = SMOKE_W, SMOKE_H
    base = os.path.join(root, "irtb")
    fields = {}
    when = dt.datetime(d_lo.year, d_lo.month, d_lo.day)
    end = dt.datetime(d_hi.year, d_hi.month, d_hi.day, 23)
    while when <= end:
        if when != SMOKE_ABSENT_HOUR:
            dirn = os.path.join(base, f"{when.year:04d}",
                                f"{when.timetuple().tm_yday:03d}")
            os.makedirs(dirn, exist_ok=True)
            name = f"merg_{when:%Y%m%d%H}_4km-pixel.nc4"
            fields[when] = write_hour(os.path.join(dirn, name), when, w, h,
                                      oob=(when == SMOKE_OOB_HOUR))
        when += dt.timedelta(hours=1)
    lo, hi = min(fields), max(fields)
    t_lo = f10b.seconds_since_epoch(lo.date()) + lo.hour * 3600
    t_hi = f10b.seconds_since_epoch(hi.date()) + hi.hour * 3600
    truth = {}
    for b in sh.bins_overlapping(t_lo, t_hi):
        for f in range(ad.frames_per_bin):
            src, sub = ad._slot_source(b, f)
            if src < lo:
                truth[(ad.store, b, f)] = (None, "before_record")
            elif src > hi:
                truth[(ad.store, b, f)] = (None, "after_record")
            elif src not in fields:
                truth[(ad.store, b, f)] = (None, "absent_upstream")
            else:
                a = fields[src][sub].copy() - KELVIN_OFFSET
                a[(a < 0) | (a > U8_MAX)] = np.nan
                st = np.where(np.isnan(a), sh.U8_MISSING,
                              np.rint(a)).astype(np.uint8)
                truth[(ad.store, b, f)] = (st[:, :, None], None)
    return truth


ADAPTER = IRTBAdapter
