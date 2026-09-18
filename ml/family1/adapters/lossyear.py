"""Hansen Global Forest Change v1.13 — the year each 30 m pixel lost its
forest, and the tree cover it had in 2000 (family 1.0.tf, E-082 wave 2).

PLAIN ENGLISH. Landsat has photographed every land pixel on Earth at 30 m
since 1999, and the University of Maryland turns that stack into two maps: how
much of each pixel was tree canopy in 2000, and — for every pixel that lost
its canopy since — the year it happened. That is the ANNUAL TARGET this
family is asked to predict, so the store keeps it at its native 30 m, tiled
and compressed, over the mapped land surface only.

SOURCE, VERIFIED 2026-09-18 FROM THIS SANDBOX (no account; public bucket):
  https://storage.googleapis.com/earthenginepartners-hansen/GFC-2025-v1.13/
      Hansen_GFC-2025-v1.13_lossyear_<TILE>.tif        uint8, 0 | 1..25
      Hansen_GFC-2025-v1.13_treecover2000_<TILE>.tif   uint8, 0..100 percent
      Hansen_GFC-2025-v1.13_datamask_<TILE>.tif        uint8, 0 | 1 | 2
  <TILE> is `<lat><N|S>_<lon><E|W>` of the tile's NORTH-WEST corner, e.g.
  `00N_060W`, on the product's documented 10 degree x 10 degree grid: 14
  latitude rows (80N, 70N, ... 50S) x 36 longitude columns (180W, 170W, ...
  170E) = 504 tiles covering 80 N .. 60 S.

THE LISTING, MEASURED 2026-09-18 (2,599 objects under the prefix):
  datamask 504 tiles (8.4 GB), treecover2000 504 (58.9 GB), gain 504 (7.5),
  first 504 (786 GB), last 280 (807 GB), LOSSYEAR 280 tiles (9.2 GB).
  ONLY 280 OF THE 504 TILES HAVE A `lossyear` GRANULE, and those 280 are the
  tiles with mapped land. The 224 without concentrate in the ocean bands (33
  of 36 missing at 50S, 31 at 40S) and the spot checks land where they should:
  `00N_180W` (mid-Pacific) and `40S_010E` (South Atlantic) are absent and
  answer HTTP 404 for lossyear while their datamask answers 200, while
  `20N_000E` (the Sahara), `30S_130E` (the Australian interior) and
  `50N_090E` (Siberia) are all PRESENT — so it is LAND, not forest, that
  decides. (`80N_060W`, the northern Greenland ice sheet, is one of the 224.)
  The store's groups are those 280 tiles, named in `TILES_WITH_LOSSYEAR`
  below, and `index` REFUSES unless the live listing's lossyear set is
  exactly that tuple and its datamask set is exactly the 504-tile grid — the
  tile-name pattern and the tile list are hypotheses the index falsifies
  (ADAPTER_CONTRACT rule 4), not facts, so a v1.14 that moves a tile stops
  the build instead of quietly changing the store's shape.

WHAT IS STORED. Each Hansen tile is cut into SIXTEEN groups of
2.5 degrees x 2.5 degrees = 10,000 x 10,000 pixels, named
`<TILE>_r<row>c<col>` with row 0 northernmost and col 0 westernmost — 280 x
16 = 4,480 groups on 0.00025-degree EPSG:4326 grids (27.8 m at the equator),
ROW 0 NORTHERNMOST (the GeoTIFFs' own order: `transform` is
(+dx, 0, west edge; 0, -dx, north edge), read off the real
`lossyear_00N_060W.tif`). dtype UINT8, C = 2:

  lossyear        0 = mapped land that did not lose its canopy,
                  k = 2000 + k for k in 1..25 (v1.13 covers 2001..2025)
  treecover2000   canopy closure in the year 2000, percent 0..100

WHY 2.5 DEGREES AND NOT THE WHOLE 10-DEGREE TILE. A group's frame is handed
to the writer whole, and `--stage probe` additionally converts it to float64
to check the channel bounds: a 40,000 x 40,000 x 2 frame is 3.2 GB as float16
and 25.6 GB as float64, so the PROBE — the measurement this store is not
allowed to be built without — could not run at any tile size above about
12,000 pixels a side. At 10,000 the frame is 200 MB, the float64 copy 1.6 GB,
and the whole probe fits the sandbox's 7 GB; at 20,000 it would need 12 GB and
not fit a hosted runner either. The fetch is still ROW-BAND AT A TIME over
the whole 40,000-wide GeoTIFF (one full-width read per 10,000 rows per
variable, four sub-tiles sliced out of it), because these are LZW files
striped one row at a time: a 10,000-wide window still decompresses the whole
40,000-pixel row, so reading sixteen square windows would decompress every
tile four times over.

A PIXEL IS VALID WHERE `datamask == 1` ("mapped land surface"), and 255
(`U8_MISSING`) everywhere else. That one rule is what makes the store sparse:
ocean, permanent water and the no-data collar cost nothing, a group or tile
that touches no mapped land is a zero-length tile, and `datamask == 1` is
recoverable from the store exactly (a pixel is valid iff it was mapped land).
The two classes datamask DISTINGUISHES and the store does not — 2 (permanent
water) from 0 (no data) — are the one thing the mask loses; a reader that
needs them fetches the 17 MB datamask granule itself. A value outside its
channel's bounds becomes 255 and is COUNTED (`out_of_bounds`), never clipped,
so a v1.14 lossyear of 26 would show up instead of being silently admitted.

WHY C = 2 AND NOT TWO GROUPS PER SUB-TILE, MEASURED 2026-09-18 on the
1,024-row strip through the middle of `00N_060W` (624 stored 256 x 256 tiles,
zstd 6): two C = 1 groups cost 11,266 bytes a tile and the interleaved C = 2
tile costs 13,346 — INTERLEAVING TWO UNCORRELATED BYTE STREAMS COSTS 18.5 %.
It is paid on purpose: C = 2 halves the number of groups, index files and
range reads, gives a reader both channels of a place in ONE tile fetch, and
lets the framework carry one honest channel table with per-channel bounds (a
store-level `channels` cannot describe two groups with different C = 1
channels). The note's own arithmetic charges 2C bytes a valid pixel anyway.

ZSTD LEVEL 6, NOT THE LAYOUT'S DEFAULT 15, and that is a measurement, not a
preference. On the same strip, per stored tile and megabytes a second of
input:
    level  1   lossyear 1,786 B @ 1,650 MB/s   treecover 10,263 B @ 428 MB/s
    level  6   lossyear 1,624 B @   404 MB/s   treecover  9,642 B @  92 MB/s
    level  9   lossyear 1,562 B @   258 MB/s   treecover  9,228 B @  45 MB/s
    level 15   lossyear 1,347 B @    34 MB/s   treecover  8,253 B @   6 MB/s
Level 15 would cost about ten hours of compression for the whole store and
level 6 about one, for 4 % more bytes than level 9 and 17 % more than 15.
THE WHOLE STORE IS ONE BIN, so there is no year axis to spread the work over
lanes (ADAPTER_CONTRACT: a year is the bins whose first day falls in it) and
one job's six hours is the binding constraint. Level 6 is the knee.

THE ONE BIN. `family1tf.tex` §4: "an annual target map carries
log2_ft = log2(365/5) = +6.2 and is filed under the bin of its period's end,
so that a target is never read as an input from inside its own year." v1.13's
period ends 2025-12-31, and the FIRST BIN THAT BEGINS AFTER IT is bin 3215,
2026-01-05 .. 2026-01-09 (bin 3214 begins ON 2025-12-31 and would let the
target be read from inside its own year). F = 1, frame_seconds = 432,000:
one frame per group, in bin 3215. Every earlier bin is `before_record` and
every later one `after_record`, so any `--start`/`--end` window is safe and
only bin 3215 is ever written. The rule is written into every
`tile_grid.json` as `annual_target`. `treecover2000` is a year-2000
snapshot travelling in the same frame; its own period is named there too.

TWO ENVIRONMENT KNOBS, both read at CONSTRUCTION time so that the fresh
adapter `stage_probe_grid` builds for itself sees them. `LOSSYEAR_TILES` (a
comma list of Hansen tile names) restricts the store's groups — the probe
cannot fetch 280 tiles, and the five the brief asks for are one flag. When it
is set the adapter says so in `notes`, which reaches store.json, so a
restricted store can never look like a whole one. `LOSSYEAR_SMOKE_PX` shrinks
the synthetic grid for `--smoke`.

PHASE B: the JRC Tropical Moist Forest v2025 annual change 1990--2025 (the
note's second source for this store) and Hansen's `gain`, `first` and `last`
composites (1.6 TB of the last two alone).
"""
import datetime as dt
import json
import os
import re
import sys
import tempfile
import threading
import time

import numpy as np

import build_family10_stores as f10b
from family1 import sharded as sh
from family1.adapters import _common as cm

VERSION = "GFC-2025-v1.13"
BUCKET = "earthenginepartners-hansen"
BASE = f"https://storage.googleapis.com/{BUCKET}/{VERSION}/"
LIST_API = (f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
            f"?prefix={VERSION}/&maxResults=1000")
PX = 40000                     # pixels per side of a 10 degree tile
DEG = 10.0
DX = DEG / PX                  # 0.00025 degrees
SUB = 4                        # sub-tiles per side: 4 x 4 of 2.5 degrees
LAT_TOPS = [80 - 10 * i for i in range(14)]        # 80 N .. -50 (50 S)
LON_WESTS = [-180 + 10 * i for i in range(36)]     # 180 W .. 170 E
VARIABLES = ("lossyear", "treecover2000", "datamask")
# v1.13 encodes loss years 2001..2025 as 1..25; 0 is "mapped land, no loss".
LOSS_YEAR_ZERO = 2000
LAST_LOSS_YEAR = 25
CHANNEL_MAX = (LAST_LOSS_YEAR, 100)
RECORD_END = dt.date(2025, 12, 31)
# The bin the annual target is filed under: the FIRST bin that BEGINS after
# the product's period ends. Computed, not copied — `index` prints it.
RECORD_BIN = next(b for b in range(3200, 3300)
                  if sh.bin_start_date(b) > RECORD_END)

# THE 280 TILES THE ARCHIVE LISTS A `lossyear` GRANULE FOR, measured from the
# bucket listing on 2026-09-18 (the other 224 of the 504 have no mapped land;
# see the module docstring). This is a HYPOTHESIS `index` falsifies: it lists
# the bucket and REFUSES unless the live set is exactly this one, so a v1.14
# that adds or drops a tile stops the build instead of silently changing the
# store's groups.
TILES_WITH_LOSSYEAR = (
    "00N_000E", "00N_010E", "00N_020E", "00N_030E", "00N_040E", "00N_040W",
    "00N_050W", "00N_060W", "00N_070W", "00N_080W", "00N_090E", "00N_090W",
    "00N_100E", "00N_100W", "00N_110E", "00N_120E", "00N_130E", "00N_140E",
    "00N_150E", "00N_160E", "10N_000E", "10N_010E", "10N_010W", "10N_020E",
    "10N_020W", "10N_030E", "10N_040E", "10N_050E", "10N_050W", "10N_060W",
    "10N_070E", "10N_070W", "10N_080E", "10N_080W", "10N_090E", "10N_090W",
    "10N_100E", "10N_100W", "10N_110E", "10N_120E", "10N_130E", "10S_010E",
    "10S_020E", "10S_030E", "10S_040E", "10S_040W", "10S_050E", "10S_050W",
    "10S_060W", "10S_070W", "10S_080W", "10S_110E", "10S_120E", "10S_130E",
    "10S_140E", "10S_150E", "10S_160E", "10S_170E", "20N_000E", "20N_010E",
    "20N_010W", "20N_020E", "20N_020W", "20N_030E", "20N_030W", "20N_040E",
    "20N_050E", "20N_060W", "20N_070E", "20N_070W", "20N_080E", "20N_080W",
    "20N_090E", "20N_090W", "20N_100E", "20N_100W", "20N_110E", "20N_110W",
    "20N_120E", "20N_160W", "20S_010E", "20S_020E", "20S_030E", "20S_040E",
    "20S_050E", "20S_050W", "20S_060W", "20S_070W", "20S_080W", "20S_110E",
    "20S_120E", "20S_130E", "20S_140E", "20S_150E", "20S_160E", "30N_000E",
    "30N_010E", "30N_010W", "30N_020E", "30N_020W", "30N_030E", "30N_040E",
    "30N_050E", "30N_060E", "30N_070E", "30N_080E", "30N_080W", "30N_090E",
    "30N_090W", "30N_100E", "30N_100W", "30N_110E", "30N_110W", "30N_120E",
    "30N_120W", "30N_130E", "30N_160W", "30N_170W", "30S_010E", "30S_020E",
    "30S_030E", "30S_060W", "30S_070W", "30S_080W", "30S_110E", "30S_120E",
    "30S_130E", "30S_140E", "30S_150E", "30S_170E", "40N_000E", "40N_010E",
    "40N_010W", "40N_020E", "40N_020W", "40N_030E", "40N_040E", "40N_050E",
    "40N_060E", "40N_070E", "40N_070W", "40N_080E", "40N_080W", "40N_090E",
    "40N_090W", "40N_100E", "40N_100W", "40N_110E", "40N_110W", "40N_120E",
    "40N_120W", "40N_130E", "40N_130W", "40N_140E", "40S_070W", "40S_080W",
    "40S_140E", "40S_160E", "40S_170E", "50N_000E", "50N_010E", "50N_010W",
    "50N_020E", "50N_030E", "50N_040E", "50N_050E", "50N_060E", "50N_060W",
    "50N_070E", "50N_070W", "50N_080E", "50N_080W", "50N_090E", "50N_090W",
    "50N_100E", "50N_100W", "50N_110E", "50N_110W", "50N_120E", "50N_120W",
    "50N_130E", "50N_130W", "50N_140E", "50N_150E", "50S_060W", "50S_070W",
    "50S_080W", "60N_000E", "60N_010E", "60N_010W", "60N_020E", "60N_020W",
    "60N_030E", "60N_040E", "60N_050E", "60N_060E", "60N_060W", "60N_070E",
    "60N_070W", "60N_080E", "60N_080W", "60N_090E", "60N_090W", "60N_100E",
    "60N_100W", "60N_110E", "60N_110W", "60N_120E", "60N_120W", "60N_130E",
    "60N_130W", "60N_140E", "60N_140W", "60N_150E", "60N_150W", "60N_160E",
    "60N_160W", "60N_170E", "60N_170W", "60N_180W", "70N_000E", "70N_010E",
    "70N_010W", "70N_020E", "70N_020W", "70N_030E", "70N_030W", "70N_040E",
    "70N_050E", "70N_060E", "70N_070E", "70N_070W", "70N_080E", "70N_080W",
    "70N_090E", "70N_090W", "70N_100E", "70N_100W", "70N_110E", "70N_110W",
    "70N_120E", "70N_120W", "70N_130E", "70N_130W", "70N_140E", "70N_140W",
    "70N_150E", "70N_150W", "70N_160E", "70N_160W", "70N_170E", "70N_170W",
    "70N_180W", "80N_010E", "80N_020E", "80N_030E", "80N_050E", "80N_060E",
    "80N_070E", "80N_070W", "80N_080E", "80N_080W", "80N_090E", "80N_090W",
    "80N_100E", "80N_100W", "80N_110E", "80N_110W", "80N_120E", "80N_120W",
    "80N_130E", "80N_130W", "80N_140E", "80N_140W", "80N_150E", "80N_150W",
    "80N_160E", "80N_160W", "80N_170E", "80N_170W",
)


OBJ = re.compile(r"^" + re.escape(VERSION) +
                 r"/Hansen_" + re.escape(VERSION) +
                 r"_([a-z0-9_]+)_(\d{2}[NS])_(\d{3}[EW])\.tif$")
GROUP = re.compile(r"^(\d{2}[NS]_\d{3}[EW])_r(\d)c(\d)$")

# GDAL is not thread-safe across datasets in every build, and the strip reads
# below are the expensive part of the fetch; every rasterio open/read/close
# holds this lock, exactly as seaice_asi holds NC_LOCK for HDF5.
TIF_LOCK = threading.Lock()


class FormatError(ValueError):
    """A listing or a GeoTIFF that does not look like the archive measured."""


def tile_name(lat_top, lon_west):
    return (f"{abs(int(lat_top)):02d}{'N' if lat_top >= 0 else 'S'}_"
            f"{abs(int(lon_west)):03d}{'E' if lon_west >= 0 else 'W'}")


def tile_corner(name):
    """`00N_060W` -> (north edge latitude, west edge longitude), degrees."""
    m = re.fullmatch(r"(\d{2})([NS])_(\d{3})([EW])", str(name))
    if not m:
        raise FormatError(f"{name!r} is not a Hansen tile name")
    la = int(m.group(1)) * (1 if m.group(2) == "N" else -1)
    lo = int(m.group(3)) * (1 if m.group(4) == "E" else -1)
    return float(la), float(lo)


def all_tiles():
    """The product's documented 10 degree grid: 14 x 36 = 504 names."""
    return [tile_name(la, lo) for la in LAT_TOPS for lo in LON_WESTS]


def group_name(tile, r, c):
    return f"{tile}_r{int(r)}c{int(c)}"


def group_parts(g):
    """`00N_060W_r1c3` -> ('00N_060W', 1, 3)."""
    m = GROUP.match(str(g))
    if not m:
        raise FormatError(f"{g!r} is not a lossyear group name")
    return m.group(1), int(m.group(2)), int(m.group(3))


def object_name(var, tile):
    return f"Hansen_{VERSION}_{var}_{tile}.tif"


def grid_of(group, px=PX, sub=SUB):
    """One sub-tile's grid. `px` is the WHOLE tile's side; the sub-tile's is
    px // sub."""
    tile, r, c = group_parts(group)
    la, lo = tile_corner(tile)
    dx = DEG / int(px)
    span = DEG / int(sub)
    spx = int(px) // int(sub)
    y0 = la - r * span
    x0 = lo + c * span
    return {
        "H": spx, "W": spx, "pixel_m": round(DEG * 111319.49 / int(px), 3),
        "crs": "EPSG:4326",
        "proj4": "+proj=longlat +datum=WGS84 +no_defs",
        "projection": {"grid_mapping_name": "latitude_longitude",
                       "semi_major_axis": 6378137.0,
                       "inverse_flattening": 298.257223563},
        "x0": x0, "dx": dx, "y0": y0, "dy": -dx,
        "coordinates": ("lon = x0 + (col + 0.5) * dx, lat = y0 + (row + 0.5) "
                        "* dy with dy NEGATIVE; x0/y0 are this sub-tile's "
                        "WEST and NORTH edges, so the grid is exactly the "
                        "source GeoTIFF's affine transform offset by the "
                        "sub-tile's own origin"),
        "row_order": ("row 0 is the NORTHERNMOST row (the source GeoTIFF's "
                      "own order: transform = (dx, 0, west edge; 0, -dx, "
                      "north edge), read off the real "
                      "Hansen_GFC-2025-v1.13_lossyear_00N_060W.tif); col 0 "
                      "is the westernmost"),
        "hansen_tile": tile, "sub_row": int(r), "sub_col": int(c),
        "sub_tiles_per_side": int(sub),
        "pixel_window_in_tile": [r * spx, (r + 1) * spx, c * spx,
                                 (c + 1) * spx],
        "extent": [x0, y0 - span, x0 + span, y0],
        "extent_note": "[west, south, east, north] in degrees",
        "source_variables": ("lossyear (uint8) and treecover2000 (uint8) of "
                             f"{object_name('lossyear', tile)} and "
                             f"{object_name('treecover2000', tile)}, masked "
                             f"to {object_name('datamask', tile)} == 1"),
        "no_tile_corners": ("a plain geographic grid: the affine rule above "
                            "IS the lat/lon of every pixel and every tile "
                            "corner, so no corner table is stored"),
    }


# ============================================================== listing ====
def parse_listing(objects):
    """GCS objects -> ({var: {tile: bytes}}, counts). REFUSES a bad name."""
    out = {v: {} for v in VARIABLES}
    counts = {"objects": len(objects), "other_variables": {}, "other": 0}
    for o in objects:
        name, size = o["name"], int(o.get("size") or 0)
        m = OBJ.match(name)
        if not m:
            counts["other"] += 1
            continue
        var, la, lo = m.group(1), m.group(2), m.group(3)
        t = f"{la}_{lo}"
        if var not in out:
            counts["other_variables"][var] = \
                counts["other_variables"].get(var, 0) + 1
            continue
        if t in out[var]:
            raise FormatError(f"{name}: two objects for {var} {t}")
        out[var][t] = size
    return out, counts


class LossyearAdapter(sh.GridAdapter):
    store = "lossyear"
    title = ("Forest loss year 2001-2025 and tree cover 2000, Hansen Global "
             "Forest Change v1.13, 30 m, 280 ten-degree tiles as 4,480 "
             "2.5-degree groups (annual target)")
    family = "1tf"
    distribution = "public"
    licence = {
        "name": "CC BY 4.0",
        "redistribution": "attribution",
        "redistribution_confirmed": True,
        "derived_works": "free",
        "attribution": ("Hansen, M. C., P. V. Potapov, R. Moore, M. Hancher, "
                        "S. A. Turubanova, A. Tyukavina, D. Thau, S. V. "
                        "Stehman, S. J. Goetz, T. R. Loveland, A. "
                        "Kommareddy, A. Egorov, L. Chini, C. O. Justice and "
                        "J. R. G. Townshend (2013), High-Resolution Global "
                        "Maps of 21st-Century Forest Cover Change, Science "
                        "342, 850-853. Data available on-line from "
                        "https://glad.earthengine.app/view/"
                        "global-forest-change (GFC-2025-v1.13)."),
        "terms": ("the product's own download page states the data are "
                  "licensed CC BY 4.0 and asks for the citation above"),
    }
    time_dtype = "int32"
    credentials = ()
    channels = (("lossyear", "loss year index (0 = mapped land with no loss, "
                             "k = 2000 + k)", 0.0, float(LAST_LOSS_YEAR)),
                ("treecover2000", "%", 0.0, 100.0))
    # 10 degrees / 40,000 = 0.00025 deg = 27.83 m at the equator, exactly a
    # thousandth of the family's 27.83 km reference: log2(0.001) = -9.9658.
    # (The note's "a 30 m pixel is -9.9" is the same to one decimal.)
    log2_fp = float(np.log2(DEG * 111.31949 / PX / 27.83))       # -9.966
    log2_dt = float(np.log2(365.0 / 5.0))                        # +6.190
    per_year = True
    first_year = int(sh.bin_year(RECORD_BIN))
    frames_per_bin = 1
    frame_seconds = sh.BIN_SECONDS
    dtype = "uint8"
    tile = sh.TILE
    zstd_level = 6
    note_estimate = {"bytes": 30e9,
                     "what": "family1tf.tex ledger: 'loss is sparse: "
                             "~30 GB' for Hansen v1.13 tree cover 2000 and "
                             "loss year at 30 m"}
    qc_policy = (
        "a pixel is valid where its Hansen tile's datamask == 1 (mapped land "
        "surface) and 255 (missing) everywhere else -- ocean, permanent "
        "water (datamask 2) and the no-data collar (datamask 0). A lossyear "
        "outside 0..25 or a treecover2000 outside 0..100 becomes 255 and is "
        "counted (`out_of_bounds`), never clipped. No other per-pixel flag "
        "exists in the source")
    sources = (BASE + "Hansen_" + VERSION + "_lossyear_<TILE>.tif",
               BASE + "Hansen_" + VERSION + "_treecover2000_<TILE>.tif",
               BASE + "Hansen_" + VERSION + "_datamask_<TILE>.tif")
    verified = (
        "2026-09-18 from the sandbox: the Google Cloud Storage JSON listing "
        "of the whole GFC-2025-v1.13 prefix (2,599 objects; datamask, "
        "treecover2000, gain and first on all 504 tiles, lossyear and last "
        "on 280); lossyear_00N_060W.tif, treecover2000_00N_060W.tif and "
        "datamask_00N_060W.tif opened with rasterio (40000 x 40000 uint8, "
        "EPSG:4326, transform (0.00025, 0, -60; 0, -0.00025, 0), LZW, one "
        "row a strip); the whole lossyear tile's histogram (14.44 % of its "
        "1.6 G pixels carry a loss year, values 1..25 only); HTTP 404 for "
        "lossyear_00N_180W and lossyear_40S_010E against 200 for their "
        "datamask")
    notes = (
        "280 Hansen tiles x 16 sub-tiles of 2.5 degrees = 4,480 groups of "
        "10,000 x 10,000 pixels, C = 2 (lossyear, treecover2000) uint8, "
        "valid where datamask == 1. The 224 of the product's 504 tiles with "
        "no lossyear granule hold no mapped land and are not groups; `index` "
        "names them and refuses if the live listing's set differs from the "
        "measured one. 2.5 degrees is the largest sub-tile whose frame the "
        "probe can convert to float64 (the check it does) inside a runner's "
        "memory. zstd level 6, measured: level 15 would cost about ten hours "
        "of compression for 17 % fewer bytes, and the whole store is ONE bin "
        "so it cannot be spread over year lanes. Interleaving the two "
        "channels in one tile costs 18.5 % against two C = 1 groups "
        "(measured) and buys one range read a place. The frame is filed in "
        f"bin {RECORD_BIN} ({sh.bin_start_date(RECORD_BIN)} .. "
        f"{sh.bin_start_date(RECORD_BIN + 1) - dt.timedelta(days=1)}), the "
        "first bin that BEGINS after the product's 2025-12-31 period end, so "
        "the target is never read as an input from inside its own year.")
    smoke_window = ("2026-01-05", "2026-01-09")
    smoke_probe_month = "2026-01"

    WORKERS = 3                  # download workers; the decode is serial
    BAND_ROWS = None             # set from px // SUB

    def __init__(self):
        self.px = int(os.environ.get("LOSSYEAR_SMOKE_PX") or PX)
        self.sub = SUB
        self.spx = self.px // self.sub
        want = [t.strip() for t in
                (os.environ.get("LOSSYEAR_TILES") or "").split(",")
                if t.strip()]
        for t in want:
            tile_corner(t)                       # refuses a bad name
        self.tiles = want or list(TILES_WITH_LOSSYEAR)
        self._listing = None
        if want:
            self.notes = (
                f"{self.notes}\nRESTRICTED BUILD: LOSSYEAR_TILES limited this "
                f"store to {len(want)} of the product's 280 land tiles "
                f"({', '.join(want[:8])}{' ...' if len(want) > 8 else ''}). "
                f"It is NOT the whole product.")
        if self.px != PX:
            self.notes = (f"{self.notes}\nSMOKE GRID: LOSSYEAR_SMOKE_PX set "
                          f"each Hansen tile to {self.px} x {self.px} instead "
                          f"of {PX} x {PX}. This is a synthetic store.")

    # ------------------------------------------------------------ the grid --
    def group_grids(self):
        return {group_name(t, r, c): grid_of(group_name(t, r, c), self.px,
                                             self.sub)
                for t in self.tiles
                for r in range(self.sub) for c in range(self.sub)}

    def specs(self):
        out = super().specs()
        end = sh.bin_start_date(RECORD_BIN + 1) - dt.timedelta(days=1)
        for spec in out.values():
            spec["annual_target"] = {
                "rule": ("family1tf.tex 4: an annual target map carries "
                         "log2_ft = log2(365/5) = +6.2 and is filed under the "
                         "bin of its period's end. This store uses the FIRST "
                         "BIN THAT BEGINS AFTER the period's last day, so the "
                         "target can never be read as an input from inside "
                         "its own year"),
                "period": ["2001-01-01", str(RECORD_END)],
                "bin": int(RECORD_BIN),
                "bin_days": [str(sh.bin_start_date(RECORD_BIN)), str(end)],
                "frames_per_bin": 1,
                "treecover2000_period": ["2000-01-01", "2000-12-31"],
                "treecover2000_note": ("a year-2000 snapshot carried in the "
                                       "same frame as the 2001-2025 loss "
                                       "year; it is the state the loss is "
                                       "measured against"),
                "lossyear_values": {"0": "mapped land, no loss 2001-2025",
                                    "1..25": "loss in year 2000 + k",
                                    "255": "not mapped land (datamask != 1)"},
            }
        return out

    def record_frames(self, ctx, group):
        return 1

    # ------------------------------------------------------------- listing --
    def listing(self, ctx):
        """({var: {tile: bytes}}, counts) from the bucket (or the smoke)."""
        if self._listing is not None:
            return self._listing
        if ctx.source_dir:
            p = os.path.join(ctx.source_dir, self.store, "listing.json")
            if not os.path.exists(p):
                sys.exit(f"REFUSING lossyear: no {p} — the smoke's synthetic "
                         f"bucket listing is missing")
            with open(p, "rb") as fh:
                raw = fh.read()
            ctx.count_bytes(len(raw))
            objects = json.loads(raw)["items"]
        else:
            objects, token, pages = [], None, 0
            while True:
                url = LIST_API + (f"&pageToken={token}" if token else "")
                raw, why = cm.get_bytes(url, attempts=ctx.a.attempts)
                if raw is None:
                    sys.exit(f"REFUSING lossyear: the bucket listing "
                             f"answered {why} at page {pages} — a listing "
                             f"that comes back empty is a refusal "
                             f"(ml/CLAUDE.md, the 2026-09-14 rule)")
                page = json.loads(raw)
                objects += page.get("items") or []
                token = page.get("nextPageToken")
                pages += 1
                if not token:
                    break
                if pages > 200:
                    sys.exit("REFUSING lossyear: the bucket listing did not "
                             "end after 200 pages")
        try:
            by_var, counts = parse_listing(objects)
        except FormatError as e:
            sys.exit(f"REFUSING lossyear: the bucket listing: {e}")
        if not by_var["lossyear"]:
            sys.exit(f"REFUSING lossyear: the listing of {BASE} holds no "
                     f"lossyear granule at all ({counts}) — an empty listing "
                     f"is a refusal")
        self._listing = (by_var, counts)
        return self._listing

    def index(self, ctx):
        by_var, counts = self.listing(ctx)
        full = set(all_tiles())
        seen = set(by_var["datamask"])
        if seen != full:
            extra = sorted(seen - full)[:6]
            miss = sorted(full - seen)[:6]
            sys.exit(f"REFUSING lossyear: the listing's datamask granules are "
                     f"not the product's 504-tile grid (only in the listing: "
                     f"{extra}; only in the grid: {miss}). The tile-name "
                     f"pattern is a hypothesis this check exists to falsify.")
        live = set(by_var["lossyear"])
        want = set(TILES_WITH_LOSSYEAR)
        if live != want:
            sys.exit(f"REFUSING lossyear: the listing lists a lossyear "
                     f"granule for {len(live)} tile(s) and this adapter's "
                     f"TILES_WITH_LOSSYEAR names {len(want)} (only live: "
                     f"{sorted(live - want)[:8]}; only named: "
                     f"{sorted(want - live)[:8]}). The store's GROUPS are "
                     f"that tuple, so a changed tile set must be re-measured "
                     f"and committed rather than silently altering the store.")
        for v in ("treecover2000", "datamask"):
            gap = sorted(live - set(by_var[v]))
            if gap:
                sys.exit(f"REFUSING lossyear: {len(gap)} tile(s) have a "
                         f"lossyear granule and no {v} ({gap[:6]}) — the "
                         f"mask and the canopy are not optional")
        have = sorted(set(self.tiles) & live)
        absent = sorted(set(self.tiles) - live)
        if not have:
            sys.exit(f"REFUSING lossyear: none of the {len(self.tiles)} "
                     f"declared tile(s) has a lossyear granule")
        probe_tile = min(have, key=lambda t: by_var["lossyear"][t])
        out = {"dataset": f"Hansen Global Forest Change {VERSION} "
                          f"(lossyear, treecover2000, datamask)",
               "url": BASE, "version": VERSION,
               "listing": counts,
               "tiles_in_product_grid": len(full),
               "tiles_with_lossyear": len(live),
               "tiles_without_lossyear": sorted(full - live),
               "tiles_declared": len(self.tiles),
               "groups": len(self.tiles) * self.sub * self.sub,
               "sub_tiles_per_side": self.sub,
               "group_pixels": self.spx,
               "tiles_absent_upstream": absent,
               "bytes_by_variable": {v: int(sum(by_var[v].get(t, 0)
                                                for t in self.tiles))
                                     for v in VARIABLES},
               "bytes_total": int(sum(by_var[v].get(t, 0)
                                      for v in VARIABLES
                                      for t in self.tiles)),
               "bin": int(RECORD_BIN),
               "bin_days": [str(sh.bin_start_date(RECORD_BIN)),
                            str(sh.bin_start_date(RECORD_BIN + 1)
                                - dt.timedelta(days=1))],
               "record_end": str(RECORD_END)}
        # ONE REAL TILE, opened and checked against the declared grid
        out["first_file"] = {"tile": probe_tile,
                            **self.read_tile_header(ctx, probe_tile)}
        out["grid"] = grid_of(group_name(probe_tile, 0, 0), self.px, self.sub)
        return out

    # ------------------------------------------------------------ one file --
    def _url(self, var, tile):
        return BASE + object_name(var, tile)

    def _local(self, ctx, var, tile):
        return os.path.join(ctx.source_dir, self.store, VERSION,
                            object_name(var, tile))

    def fetch_tifs(self, ctx, tile, tmpdir):
        """The tile's three GeoTIFFs on local disk -> ({paths, bytes}, None)
        or (None, why). Nothing is decoded here."""
        paths, nbytes = {}, 0
        for var in VARIABLES:
            if ctx.source_dir:
                p = self._local(ctx, var, tile)
                if not os.path.exists(p):
                    return None, f"{object_name(var, tile)}: listed, absent"
                nbytes += os.path.getsize(p)
                ctx.count_bytes(os.path.getsize(p))
            else:
                p = os.path.join(tmpdir, object_name(var, tile))
                if f10b.fetch_first([self._url(var, tile)], p,
                                    attempts=ctx.a.attempts) is None:
                    return None, (f"{self._url(var, tile)}: listed in the "
                                  f"bucket, and 404")
                nbytes += os.path.getsize(p)
            paths[var] = p
        return {"paths": paths, "bytes": nbytes}, None

    def read_tile_header(self, ctx, tile):
        """Open the tile's lossyear GeoTIFF and CHECK the declared grid."""
        tmpdir = None if ctx.source_dir else \
            tempfile.mkdtemp(prefix="gfc_", dir=ctx.scratch)
        try:
            if ctx.source_dir:
                path = self._local(ctx, "lossyear", tile)
                if not os.path.exists(path):
                    sys.exit(f"REFUSING lossyear: {path} is missing")
                ctx.count_bytes(os.path.getsize(path))
            else:
                path = os.path.join(tmpdir, object_name("lossyear", tile))
                if f10b.fetch_first([self._url("lossyear", tile)], path,
                                    attempts=ctx.a.attempts) is None:
                    sys.exit(f"REFUSING lossyear: "
                             f"{self._url('lossyear', tile)} is in the "
                             f"listing and answers 404")
            with TIF_LOCK:
                meta = tif_meta(path, tile, self.px)
            meta["bytes"] = os.path.getsize(path)
            return meta
        finally:
            if tmpdir:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)

    def read_band(self, ctx, tile, paths, r):
        """Sub-tile ROW r of `tile` as (bands, land, counts, oob).

        One full-width windowed read per variable per row band: these GeoTIFFs
        are LZW striped ONE ROW at a time, so a narrow window decompresses the
        whole 40,000-pixel row anyway and sixteen square windows would
        decompress the tile four times over. What is HELD is the two uint8
        bands (400 MB each at 40,000 x 10,000) and the land mask, not the
        float16 frames — `sub_frame` builds one of those at a time, because
        four at once plus the probe's float64 copy does not fit a runner.
        """
        rasterio = _rasterio()
        px, spx = self.px, self.spx
        r0, r1 = r * spx, (r + 1) * spx
        oob = {}
        dmc = {}
        with TIF_LOCK:
            src = {v: rasterio.open(paths[v]) for v in VARIABLES}
            try:
                for v in VARIABLES:
                    tif_check(src[v], paths[v], tile, px)
                win = ((r0, r1), (0, px))
                dm = src["datamask"].read(1, window=win)
                land = dm == 1
                bc = np.bincount(dm.ravel(), minlength=3)
                for k in np.flatnonzero(bc):
                    dmc[int(k)] = dmc.get(int(k), 0) + int(bc[k])
                del dm, bc
                bands = []
                for ci, v in enumerate(("lossyear", "treecover2000")):
                    a = src[v].read(1, window=win)
                    bad = land & (a > CHANNEL_MAX[ci])
                    nb = int(bad.sum())
                    if nb:
                        # OUT OF BOUNDS: counted by channel name and turned
                        # into the missing code, never clipped (contract 3).
                        # 255 is not a value either channel can take, so the
                        # band now says "missing" in one place.
                        oob[self.channel_names[ci]] = nb
                        a[bad] = sh.U8_MISSING
                    del bad
                    bands.append(a)
            finally:
                for d in src.values():
                    d.close()
        counts = {"datamask_pixels": dmc, "land_pixels": int(dmc.get(1, 0))}
        return bands, land, counts, oob

    def sub_frame(self, bands, land, c):
        """One sub-tile's float16 [spx, spx, C] frame with NaN for missing.

        float16 (not uint8) because the GridAdapter contract says raw units
        with NaN, and because `--stage probe` checks the channel bounds on a
        float64 copy of whatever the adapter yields: a raw uint8 255 would
        read there as a VALUE of 255 and fail rule 3 on every missing pixel.
        """
        spx = self.spx
        c0, c1 = c * spx, (c + 1) * spx
        lm = land[:, c0:c1]
        out = np.empty((spx, spx, len(bands)), np.float16)
        for ci, b in enumerate(bands):
            v = b[:, c0:c1]
            f = v.astype(np.float16)
            f[~lm | (v == sh.U8_MISSING)] = np.nan
            out[:, :, ci] = f
        return out

    # ------------------------------------------------------- the contract --
    def classify(self, ctx, b):
        """-> None for the record's bin, else the reason it is not."""
        if int(b) == int(RECORD_BIN):
            return None
        return "before_record" if int(b) < int(RECORD_BIN) else "after_record"

    def fetch_frames(self, ctx, wanted):
        by_var, _ = self.listing(ctx)
        live = set(by_var["lossyear"])
        # Every wanted frame, split into the absences (no I/O at all) and the
        # real ones, grouped by HANSEN TILE so one download and one pass over
        # the GeoTIFFs answers all sixteen of its sub-tiles.
        absent, real = [], {}
        for (g, b, f) in wanted:
            tile, r, c = group_parts(g)
            why = self.classify(ctx, b)
            if why is None and tile not in live:
                why = "absent_upstream"
            if why is not None:
                absent.append((g, b, f, why))
            else:
                real.setdefault(tile, []).append((g, b, f, r, c))
        tiles = sorted(real)

        def work(tile):
            tmpdir = None if ctx.source_dir else \
                tempfile.mkdtemp(prefix="gfc_", dir=ctx.scratch)
            got, err = self.fetch_tifs(ctx, tile, tmpdir)
            return tile, got, err, tmpdir

        for (g, b, f, why) in absent:
            yield g, b, f, None, {"frame_missing": why}

        workers = 1 if ctx.source_dir else self.WORKERS
        t0 = time.time()
        for tile, got, err, tmpdir in cm.ordered_map(
                work, tiles, workers, lookahead=workers + 1):
            try:
                if err:
                    ctx.note_absent(tile, err)
                    continue
                jobs = sorted(real[tile], key=lambda j: (j[3], j[4], j[1]))
                by_row = {}
                for j in jobs:
                    by_row.setdefault(j[3], []).append(j)
                for r in sorted(by_row):
                    bands = None
                    try:
                        bands, land, counts, oob = self.read_band(
                            ctx, tile, got["paths"], r)
                    except FormatError as e:
                        sys.exit(f"REFUSING lossyear: {tile}: {e}")
                    except (IOError, OSError) as e:
                        ctx.note_absent(tile, f"{type(e).__name__}: {e}")
                    if bands is None:
                        break
                    for (g, b, f, rr, c) in by_row[r]:
                        cc = dict(counts) if c == 0 else {}
                        if oob and c == 0:
                            cc["out_of_bounds"] = dict(oob)
                        if c == 0:
                            cc["files_read"] = len(VARIABLES)
                            cc["bytes_files"] = got["bytes"] if r == 0 else 0
                            cc["bands_read"] = 1
                        cc["fetch_seconds"] = round(time.time() - t0, 2)
                        t0 = time.time()
                        yield g, b, f, self.sub_frame(bands, land, c), cc
                    del bands, land
            finally:
                if tmpdir:
                    import shutil
                    shutil.rmtree(tmpdir, ignore_errors=True)

    # --------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260918):
        """`make_smoke_sources` sets LOSSYEAR_SMOKE_PX / LOSSYEAR_TILES, and
        THIS instance was built before they existed (`run_smoke_grid` makes
        the adapter first), so it re-reads them. Every later instance —
        including the one `stage_probe_grid` builds for itself — reads them
        from the environment at construction time."""
        truth = make_smoke_sources(root, d_lo, d_hi, seed)
        self.__init__()
        return truth


# ============================================================== GeoTIFF ====
def _rasterio():
    try:
        import rasterio
    except ImportError:                                     # pragma: no cover
        sys.exit("lossyear needs the `rasterio` package (pip install "
                 "rasterio) — it is in the family1-build workflow's install "
                 "step")
    return rasterio


def tif_check(d, path, tile, px):
    """An open rasterio dataset against the declared grid. Raises."""
    la, lo = tile_corner(tile)
    dx = DEG / int(px)
    if (d.height, d.width) != (int(px), int(px)):
        raise FormatError(f"{os.path.basename(path)}: {d.height} x {d.width}, "
                          f"expected {px} x {px}")
    if d.count != 1 or d.dtypes[0] != "uint8":
        raise FormatError(f"{os.path.basename(path)}: {d.count} band(s) of "
                          f"{d.dtypes}, expected one uint8 band")
    t = d.transform
    want = (dx, 0.0, lo, 0.0, -dx, la)
    got = (t.a, t.b, t.c, t.d, t.e, t.f)
    if not all(abs(g - w) <= 1e-9 * max(1.0, abs(w))
               for g, w in zip(got, want)):
        raise FormatError(f"{os.path.basename(path)}: transform {got}, the "
                          f"tile name says {want}")
    if d.crs is None or d.crs.to_epsg() != 4326:
        raise FormatError(f"{os.path.basename(path)}: CRS {d.crs}, expected "
                          f"EPSG:4326")


def tif_meta(path, tile, px):
    """Open one GeoTIFF, CHECK the grid, and report what it is."""
    rasterio = _rasterio()
    try:
        d = rasterio.open(path)
    except Exception as ex:                                 # noqa: BLE001
        raise IOError(f"{os.path.basename(path)}: not a readable GeoTIFF "
                      f"({ex}) — a truncated download") from None
    try:
        tif_check(d, path, tile, px)
        prof = d.profile
        row = d.read(1, window=((0, min(int(px), 1024)), (0, int(px))))
        u, c = np.unique(row, return_counts=True)
        return {"height": d.height, "width": d.width,
                "dtype": d.dtypes[0], "crs": str(d.crs),
                "transform": [d.transform.a, d.transform.b, d.transform.c,
                              d.transform.d, d.transform.e, d.transform.f],
                "bounds": [d.bounds.left, d.bounds.bottom, d.bounds.right,
                           d.bounds.top],
                "compress": str(prof.get("compress")),
                "block_shapes": [list(b) for b in d.block_shapes],
                "values_first_rows": {int(k): int(v) for k, v in
                                      zip(u[:32], c[:32])}}
    finally:
        d.close()


def write_tif(path, tile, a, px):
    rasterio = _rasterio()
    from rasterio.transform import from_origin
    la, lo = tile_corner(tile)
    dx = DEG / int(px)
    with rasterio.open(path, "w", driver="GTiff", height=int(px),
                       width=int(px), count=1, dtype="uint8",
                       crs="EPSG:4326",
                       transform=from_origin(lo, la, dx, dx),
                       compress="lzw") as d:
        d.write(a, 1)


# ================================================================== smoke ==
# The smoke's bucket: three real land tiles, one declared-but-absent (named in
# LOSSYEAR_TILES and with no lossyear granule, as 224 of the real 504 are),
# one lossyear pixel of 40 (out of bounds for v1.13's 1..25) and one
# treecover of 120.
SMOKE_TILES = ("00N_060W", "00N_020E", "50N_090E")
SMOKE_ABSENT = "40S_010E"
SMOKE_OOB_TILE = "00N_020E"
SMOKE_PX = 512               # 4 x 4 sub-tiles of 128 -> 1 store tile each


def smoke_fields(tile, px):
    """Deterministic (lossyear, treecover2000, datamask) on a small grid."""
    la, lo = tile_corner(tile)
    k = int(abs(la) + abs(lo)) % 17
    yy = (np.arange(px, dtype=np.float64)[:, None] - px / 2) / px
    xx = (np.arange(px, dtype=np.float64)[None, :] - px / 2) / px
    r = np.hypot(yy, xx)
    dm = np.full((px, px), 1, np.uint8)
    # the disc is tight enough that the four CORNER sub-tiles of every
    # smoke tile are entirely ocean, so the store skips them: the
    # sparsity the whole design rests on is exercised, not assumed
    dm[r > 0.34] = 2                                   # "ocean"
    dm[(np.abs(yy - 0.2) < 0.03) & (xx > 0)] = 0       # a no-data stripe
    band = ((r * 60 + k) % 26).astype(np.uint8)
    keep = (band > 0) & (band <= 25) & ((yy + xx) > -0.3)
    lossy = np.where(keep, band, np.uint8(0)).astype(np.uint8)
    tc = np.clip(100 - 180 * r + 20 * np.sin(8 * xx + k), 0, 100) \
        .astype(np.uint8)
    if tile == SMOKE_OOB_TILE:
        lossy[px // 2, px // 2] = 40                   # out of bounds
        tc[px // 2, px // 2 + 1] = 120                 # out of bounds
        dm[px // 2, px // 2] = 1
        dm[px // 2, px // 2 + 1] = 1
    return lossy, tc, dm


def make_smoke_sources(root, d_lo, d_hi, seed=20260918):
    """The bucket in its real shape: a `listing.json` holding the GCS JSON
    API's `items` (three tiles with all three variables, one tile with
    treecover2000 and datamask but NO lossyear, the whole 504-tile datamask
    grid so `index`'s checks see the product, and the full
    TILES_WITH_LOSSYEAR set so the lossyear-set check passes) and real LZW
    GeoTIFFs on a `LOSSYEAR_SMOKE_PX` grid.

    Returns the truth: {(group, bin, frame): (uint8 [spx, spx, 2] or None,
    reason or None)} for every frame of every bin overlapping the record.
    """
    os.environ["LOSSYEAR_SMOKE_PX"] = str(SMOKE_PX)
    os.environ["LOSSYEAR_TILES"] = ",".join(SMOKE_TILES + (SMOKE_ABSENT,))
    px = SMOKE_PX
    spx = px // SUB
    base = os.path.join(root, "lossyear", VERSION)
    os.makedirs(base, exist_ok=True)
    items, truth, fields = [], {}, {}
    for t in SMOKE_TILES + (SMOKE_ABSENT,):
        lossy, tc, dm = smoke_fields(t, px)
        fields[t] = (lossy, tc, dm)
        for var, a in (("lossyear", lossy), ("treecover2000", tc),
                       ("datamask", dm)):
            if t == SMOKE_ABSENT and var == "lossyear":
                continue
            p = os.path.join(base, object_name(var, t))
            write_tif(p, t, a, px)
            items.append({"name": f"{VERSION}/{object_name(var, t)}",
                          "size": str(os.path.getsize(p))})
    # the rest of the product's grid, so `index`'s two set checks see the
    # whole product. No bytes are written for them and the adapter never asks:
    # LOSSYEAR_TILES restricts the groups.
    written = set(SMOKE_TILES) | {SMOKE_ABSENT}
    for t in all_tiles():
        if t not in written:
            items.append({"name": f"{VERSION}/{object_name('datamask', t)}",
                          "size": "1"})
    for t in TILES_WITH_LOSSYEAR:
        if t not in written:
            for var in ("lossyear", "treecover2000"):
                items.append({"name": f"{VERSION}/{object_name(var, t)}",
                              "size": "1"})
    items.append({"name": f"{VERSION}/download.html", "size": "71893"})
    with open(os.path.join(root, "lossyear", "listing.json"), "w") as fh:
        json.dump({"kind": "storage#objects", "items": items}, fh)

    t_lo = f10b.seconds_since_epoch(d_lo)
    t_hi = f10b.seconds_since_epoch(d_hi) + 86399
    for b in sh.bins_overlapping(t_lo, t_hi):
        for t in SMOKE_TILES + (SMOKE_ABSENT,):
            for r in range(SUB):
                for c in range(SUB):
                    g = group_name(t, r, c)
                    if int(b) != int(RECORD_BIN):
                        truth[(g, b, 0)] = (
                            None, "before_record" if int(b) < int(RECORD_BIN)
                            else "after_record")
                        continue
                    if t == SMOKE_ABSENT:
                        truth[(g, b, 0)] = (None, "absent_upstream")
                        continue
                    lossy, tc, dm = fields[t]
                    sl = (slice(r * spx, (r + 1) * spx),
                          slice(c * spx, (c + 1) * spx))
                    land = dm[sl] == 1
                    out = np.full((spx, spx, 2), sh.U8_MISSING, np.uint8)
                    ly_ = lossy[sl]
                    tc_ = tc[sl]
                    out[:, :, 0] = np.where(land & (ly_ <= LAST_LOSS_YEAR),
                                            ly_, np.uint8(sh.U8_MISSING))
                    out[:, :, 1] = np.where(land & (tc_ <= 100), tc_,
                                            np.uint8(sh.U8_MISSING))
                    truth[(g, b, 0)] = (out, None)
    return truth


ADAPTER = LossyearAdapter
