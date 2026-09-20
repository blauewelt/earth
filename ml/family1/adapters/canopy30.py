"""GLAD's 2020 forest height at 30 m — how tall the canopy is, everywhere
(family 1.0.tf, E-082 wave 6, the biosphere wave).

PLAIN ENGLISH. GEDI (the Global Ecosystem Dynamics Investigation, the laser on
the International Space Station) measures the height of a forest directly, but
only in 25 m footprints along eight ground tracks between 51.6 degrees south
and north — a few per cent of the land surface. GLAD (the Global Land Analysis
and Discovery laboratory at the University of Maryland) fitted those laser
heights to Landsat's reflectance metrics and painted the answer over every
30 m pixel of the planet for the year 2020 (Potapov et al. 2022, the method in
Potapov et al. 2021). That MAP is what this store holds: one number per pixel,
the height in whole metres of the woody vegetation standing on it. It is the
canopy READ-OUT TARGET of the biosphere wave — what a user asks the model for
— and it is admitted as a target exactly as `lossyear` is: a fit to a
measurement the family also holds.

SOURCE, VERIFIED 2026-09-20 FROM THIS SANDBOX (no account; anonymous HTTPS):
  https://gladxfer.umd.edu/users/Potapov/GLCLUC2020/Forest_height_2020/
      2020_<TILE>.tif
  <TILE> is `<lat><N|S>_<lon><E|W>` of the tile's NORTH-WEST corner, the same
  10 degree x 10 degree grid the Hansen forest-loss tiles use, so this store's
  groups are `lossyear`'s groups and a reader fetches canopy height and loss
  year for the same place with the same two tile addresses.
  (`https://glad.umd.edu/users/Potapov/GLCLUC2020/...` answers HTTP 301 to the
  `gladxfer` transfer host; the redirect was followed and recorded rather than
  guessed.)

THE LISTING, MEASURED 2026-09-20: **261 files**, every one of them a
`2020_<TILE>.tif`, with no other object in the directory at all. Their sizes
run 13 MB .. 616 MB (median 77 MB) and sum to **35.95 GB** on the listing's
own human-readable size column. THE NOTE'S LEDGER ROW SAYS "0.33--0.61 GB a
tile ... ~120 GB native": the 0.33--0.61 GB band is the top of the
distribution, not the middle, and the native total is 36 GB, not 120. Both
numbers are the listing's, re-measured here, and the probe replaces them with
exact bytes. The 261 tiles are the land tiles, and they are a strict SUBSET of
`lossyear`'s 280: every canopy tile has a Hansen loss-year tile, and nineteen
Hansen tiles have no canopy tile (`00N_100W`, `10N_100W`, `20N_030W`,
`20S_050E`, `40N_070W`, `50S_060W`, `70N_010W`, `70N_020W`, `70N_030W`,
`70N_180W`, `80N_050E`, `80N_060E`, `80N_070W`, `80N_080W`, `80N_090W`,
`80N_100W`, `80N_110W`, `80N_120W`, `80N_170E` — measured, mostly high-Arctic
and small-island tiles). A group of this store therefore always has a
`lossyear` twin, and the reverse does not hold.

THE PRODUCT HAS NO WATER CODE AND NO NO-DATA CODE, and that is a MEASUREMENT,
not a reading of the documentation. The producer's own page (glad.umd.edu,
read 2026-09-20) says of this layer only "Forest height, 2020 (pixel value:
forest height in meters)". Three whole tiles were downloaded and histogrammed
here — `2020_50N_000E` (France and the Bay of Biscay, 303.8 MB),
`2020_20S_010E` (the Namib and the South Atlantic, 13.4 MB) and
`2020_80N_010E` (Svalbard and the Arctic Ocean, 13.6 MB) — and every one of
the 4.8 x 10^9 pixels read 0..35. There is no 101, no 102, no 255. So:

  * **0 is a MEASUREMENT** — "no woody vegetation 3 m or taller here" — over
    ocean, desert, ice and mown grass alike, and it is stored as a value, not
    as missing. The store is therefore DENSE, which is why the ledger's own
    estimate says "60--100 GB (dense metres)" where `lossyear`'s says "loss is
    sparse".
  * Zeros cost almost nothing: the ocean-and-desert tile is 13.4 MB for
    1.6 x 10^9 pixels, i.e. 0.008 bytes a pixel, against 0.19 for the forested
    French tile.
  * The model's own floor is 3 m (the page: "mapped globally for woody
    vegetation of >= 3 m"), so values of 1 and 2 are rare and are kept as they
    come.
  * A value ABOVE 100 m is not a height — the tallest tree ever measured is
    116 m and a Landsat model saturates far below that — so the channel bound
    is 0..100 and anything past it becomes missing and is COUNTED
    (`out_of_bounds`), never clipped, with the offending raw values listed by
    name in `height_codes`. If a future release adopts the 2019 map's class
    codes (101 water, 102 no data) they will show up there instead of being
    silently admitted as 101-metre trees.

WHAT IS STORED. Each 10 degree tile is cut into SIXTEEN groups of
2.5 degrees x 2.5 degrees = 10,000 x 10,000 pixels, named
`<TILE>_r<row>c<col>` with row 0 northernmost and col 0 westernmost — 261 x 16
= **4,176 groups** on 0.00025-degree EPSG:4326 grids (27.83 m at the equator),
ROW 0 NORTHERNMOST (the GeoTIFF's own order: the real
`2020_50N_000E.tif` opened here reads 40000 x 40000 uint8, EPSG:4326,
transform (0.00025, 0, 0; 0, -0.00025, 50), LZW, one row a strip). dtype
UINT8, C = 1:

  canopy_height   the height of the woody vegetation, whole metres, 0..100

The 2.5-degree cut is `lossyear`'s and is kept for its reason as well as for
its addresses: `--stage probe` converts every frame it is handed to float64 to
check the channel bounds, and a 40,000 x 40,000 float64 copy is 12.8 GB, which
fits neither this sandbox nor a hosted runner. At 10,000 the frame is 100 MB
as uint8 and the float64 copy 800 MB. The fetch is still ROW-BAND AT A TIME
over the whole 40,000-wide GeoTIFF (one full-width read per 10,000 rows, four
sub-tiles sliced out of it), because these are LZW files striped one row at a
time: a 10,000-wide window decompresses the whole 40,000-pixel row anyway, so
sixteen square windows would decompress the tile four times over.

ZSTD LEVEL 6, MEASURED HERE ON THE REAL TILE, not transcribed. A 1,024-row
strip through the middle of `2020_50N_000E` (624 tiles of 256 x 256):

    level  1   17,959 B a tile   3.65x   412 MB/s of input
    level  6   17,349 B a tile   3.78x    78 MB/s
    level  9   17,087 B a tile   3.84x    45 MB/s
    level 15   16,624 B a tile   3.94x     7 MB/s

The whole store is 261 x 1.6 x 10^9 = 4.18 x 10^11 pixels, so level 6 costs
about 1.5 hours of compression and level 15 about 16 hours for 4 % fewer
bytes. THE WHOLE STORE IS ONE BIN, so there is no year axis to spread that
work over lanes (ADAPTER_CONTRACT: a year is the bins whose first day falls in
it) and one job's six hours is the binding constraint. Level 6 is the knee,
and it is the same level `lossyear` measured on the same tile grid.

THE ONE BIN. The map's period is the calendar year 2020, and the brief files
it under **the bin holding 2020-07-01** — the middle of the year it describes:
bin **2812**, 2020-06-30 .. 2020-07-04. F = 1 and frame_seconds = 432,000, so
one frame per group lives in that one bin; every earlier bin is
`before_record` and every later one `after_record`, so any `--start`/`--end`
window is safe and only bin 2812 is ever written. The rule is written into
every `tile_grid.json` as `annual_map`. (This differs deliberately from
`lossyear`, which is a CHANGE map covering 2001--2025 and is filed under the
first bin that BEGINS after its period ends so a target can never be read as
an input from inside its own year. A single-year STATE map has no such hazard:
the value at a place in July 2020 is what the canopy was in 2020, and the cone
reads it as a state at lag >= one year, `E082_biosphere_wave.md` §1.)

TWO ENVIRONMENT KNOBS, both read at CONSTRUCTION time so that the fresh
adapter `stage_probe_grid` builds for itself sees them. `CANOPY30_TILES` (a
comma list of tile names) restricts the store's groups — a probe cannot fetch
261 tiles, and the one the brief asks for is one flag. When it is set the
adapter says so in `notes`, which reaches store.json, so a restricted store
can never look like a whole one. `CANOPY30_SMOKE_PX` shrinks the synthetic
grid for `--smoke`.

PHASE B: GLAD's `Forest_height_2000` (the same grid for the year 2000, which
makes a twenty-year pair), and the ETH 10 m and Meta/WRI 1 m maps, which are
catalogued rather than copied (`canopy_ref`).
"""
import datetime as dt
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

YEAR = 2020
HOST = "https://gladxfer.umd.edu"
BASE = f"{HOST}/users/Potapov/GLCLUC2020/Forest_height_{YEAR}/"
# The umd.edu name answers 301 to the transfer host above; it is recorded here
# because a reader will look for it and must not have to guess.
REDIRECT_FROM = ("https://glad.umd.edu/users/Potapov/GLCLUC2020/"
                 "Forest_height_2020/")
PX = 40000                     # pixels per side of a 10 degree tile
DEG = 10.0
DX = DEG / PX                  # 0.00025 degrees
SUB = 4                        # sub-tiles per side: 4 x 4 of 2.5 degrees
LAT_TOPS = [80 - 10 * i for i in range(14)]        # 80 N .. -50 (50 S)
LON_WESTS = [-180 + 10 * i for i in range(36)]     # 180 W .. 170 E
MAX_HEIGHT = 100.0             # the channel bound: a tripwire, not a filter

# The map's period and the bin it is filed under: the bin holding 1 July of
# the mapped year. Computed, not copied — `index` prints it.
RECORD_YEAR = YEAR
RECORD_MID = dt.date(YEAR, 7, 1)
RECORD_BIN = f10b.seconds_since_epoch(RECORD_MID) // sh.BIN_SECONDS

# THE 261 TILES THE PRODUCER'S DIRECTORY LISTS, measured 2026-09-20 (see the
# module docstring). This is a HYPOTHESIS `index` falsifies: it lists the
# directory and REFUSES unless the live set is exactly this one, so a re-issue
# that adds or drops a tile stops the build instead of silently changing the
# store's groups.
TILES_2020 = (
    "00N_000E", "00N_010E", "00N_020E", "00N_030E", "00N_040E",
    "00N_040W", "00N_050W", "00N_060W", "00N_070W", "00N_080W",
    "00N_090E", "00N_090W", "00N_100E", "00N_110E", "00N_120E",
    "00N_130E", "00N_140E", "00N_150E", "00N_160E", "10N_000E",
    "10N_010E", "10N_010W", "10N_020E", "10N_020W", "10N_030E",
    "10N_040E", "10N_050E", "10N_050W", "10N_060W", "10N_070E",
    "10N_070W", "10N_080E", "10N_080W", "10N_090E", "10N_090W",
    "10N_100E", "10N_110E", "10N_120E", "10N_130E", "10S_010E",
    "10S_020E", "10S_030E", "10S_040E", "10S_040W", "10S_050E",
    "10S_050W", "10S_060W", "10S_070W", "10S_080W", "10S_110E",
    "10S_120E", "10S_130E", "10S_140E", "10S_150E", "10S_160E",
    "10S_170E", "20N_000E", "20N_010E", "20N_010W", "20N_020E",
    "20N_020W", "20N_030E", "20N_040E", "20N_050E", "20N_060W",
    "20N_070E", "20N_070W", "20N_080E", "20N_080W", "20N_090E",
    "20N_090W", "20N_100E", "20N_100W", "20N_110E", "20N_110W",
    "20N_120E", "20N_160W", "20S_010E", "20S_020E", "20S_030E",
    "20S_040E", "20S_050W", "20S_060W", "20S_070W", "20S_080W",
    "20S_110E", "20S_120E", "20S_130E", "20S_140E", "20S_150E",
    "20S_160E", "30N_000E", "30N_010E", "30N_010W", "30N_020E",
    "30N_020W", "30N_030E", "30N_040E", "30N_050E", "30N_060E",
    "30N_070E", "30N_080E", "30N_080W", "30N_090E", "30N_090W",
    "30N_100E", "30N_100W", "30N_110E", "30N_110W", "30N_120E",
    "30N_120W", "30N_130E", "30N_160W", "30N_170W", "30S_010E",
    "30S_020E", "30S_030E", "30S_060W", "30S_070W", "30S_080W",
    "30S_110E", "30S_120E", "30S_130E", "30S_140E", "30S_150E",
    "30S_170E", "40N_000E", "40N_010E", "40N_010W", "40N_020E",
    "40N_020W", "40N_030E", "40N_040E", "40N_050E", "40N_060E",
    "40N_070E", "40N_080E", "40N_080W", "40N_090E", "40N_090W",
    "40N_100E", "40N_100W", "40N_110E", "40N_110W", "40N_120E",
    "40N_120W", "40N_130E", "40N_130W", "40N_140E", "40S_070W",
    "40S_080W", "40S_140E", "40S_160E", "40S_170E", "50N_000E",
    "50N_010E", "50N_010W", "50N_020E", "50N_030E", "50N_040E",
    "50N_050E", "50N_060E", "50N_060W", "50N_070E", "50N_070W",
    "50N_080E", "50N_080W", "50N_090E", "50N_090W", "50N_100E",
    "50N_100W", "50N_110E", "50N_110W", "50N_120E", "50N_120W",
    "50N_130E", "50N_130W", "50N_140E", "50N_150E", "50S_070W",
    "50S_080W", "60N_000E", "60N_010E", "60N_010W", "60N_020E",
    "60N_020W", "60N_030E", "60N_040E", "60N_050E", "60N_060E",
    "60N_060W", "60N_070E", "60N_070W", "60N_080E", "60N_080W",
    "60N_090E", "60N_090W", "60N_100E", "60N_100W", "60N_110E",
    "60N_110W", "60N_120E", "60N_120W", "60N_130E", "60N_130W",
    "60N_140E", "60N_140W", "60N_150E", "60N_150W", "60N_160E",
    "60N_160W", "60N_170E", "60N_170W", "60N_180W", "70N_000E",
    "70N_010E", "70N_020E", "70N_030E", "70N_040E", "70N_050E",
    "70N_060E", "70N_070E", "70N_070W", "70N_080E", "70N_080W",
    "70N_090E", "70N_090W", "70N_100E", "70N_100W", "70N_110E",
    "70N_110W", "70N_120E", "70N_120W", "70N_130E", "70N_130W",
    "70N_140E", "70N_140W", "70N_150E", "70N_150W", "70N_160E",
    "70N_160W", "70N_170E", "70N_170W", "80N_010E", "80N_020E",
    "80N_030E", "80N_070E", "80N_080E", "80N_090E", "80N_100E",
    "80N_110E", "80N_120E", "80N_130E", "80N_130W", "80N_140E",
    "80N_140W", "80N_150E", "80N_150W", "80N_160E", "80N_160W",
    "80N_170W",
)

FILE = re.compile(r"^" + str(YEAR) + r"_(\d{2}[NS])_(\d{3}[EW])\.tif$")
ROW = re.compile(r'<a href="(' + str(YEAR) +
                 r'_\d{2}[NS]_\d{3}[EW]\.tif)">[^<]*</a>\s*</td>'
                 r'\s*<td align="right">[^<]*</td>'
                 r'\s*<td align="right">\s*([0-9.]+[KMGT]?)\s*</td>')
ANY_HREF = re.compile(r'href="([^"?/][^"]*)"')
GROUP = re.compile(r"^(\d{2}[NS]_\d{3}[EW])_r(\d)c(\d)$")

# GDAL is not thread-safe across datasets in every build, and the strip reads
# below are the expensive part of the fetch; every rasterio open/read/close
# holds this lock, exactly as `lossyear` holds TIF_LOCK and `seaice_asi` holds
# NC_LOCK for HDF5.
TIF_LOCK = threading.Lock()


class FormatError(ValueError):
    """A listing or a GeoTIFF that does not look like the archive measured."""


def approx_bytes(s):
    """The Apache listing's human-readable size column -> bytes (approximate).

    The column is rounded to three characters ("303M"), so the sum over the
    directory is an estimate and the probe measures the exact number.
    """
    s = str(s).strip()
    m = re.fullmatch(r"([0-9.]+)\s*([KMGT]?)", s)
    if not m:
        return 0
    mult = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3,
            "T": 1024 ** 4}[m.group(2)]
    return int(float(m.group(1)) * mult)


def tile_name(lat_top, lon_west):
    return (f"{abs(int(lat_top)):02d}{'N' if lat_top >= 0 else 'S'}_"
            f"{abs(int(lon_west)):03d}{'E' if lon_west >= 0 else 'W'}")


def tile_corner(name):
    """`00N_060W` -> (north edge latitude, west edge longitude), degrees."""
    m = re.fullmatch(r"(\d{2})([NS])_(\d{3})([EW])", str(name))
    if not m:
        raise FormatError(f"{name!r} is not a 10-degree tile name")
    la = int(m.group(1)) * (1 if m.group(2) == "N" else -1)
    lo = int(m.group(3)) * (1 if m.group(4) == "E" else -1)
    return float(la), float(lo)


def all_tiles():
    """The documented 10-degree grid: 14 latitude rows x 36 columns = 504."""
    return [tile_name(la, lo) for la in LAT_TOPS for lo in LON_WESTS]


def group_name(tile, r, c):
    return f"{tile}_r{int(r)}c{int(c)}"


def group_parts(g):
    """`00N_060W_r1c3` -> ('00N_060W', 1, 3)."""
    m = GROUP.match(str(g))
    if not m:
        raise FormatError(f"{g!r} is not a canopy30 group name")
    return m.group(1), int(m.group(2)), int(m.group(3))


def object_name(tile):
    return f"{YEAR}_{tile}.tif"


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
                      "north edge), read off the real 2020_50N_000E.tif); "
                      "col 0 is the westernmost"),
        "tile_10deg": tile, "sub_row": int(r), "sub_col": int(c),
        "sub_tiles_per_side": int(sub),
        "pixel_window_in_tile": [r * spx, (r + 1) * spx, c * spx,
                                 (c + 1) * spx],
        "extent": [x0, y0 - span, x0 + span, y0],
        "extent_note": "[west, south, east, north] in degrees",
        "source_variable": (f"the single uint8 band of {object_name(tile)}: "
                            f"the height in whole metres of woody vegetation "
                            f"3 m or taller, 0 where there is none"),
        "no_tile_corners": ("a plain geographic grid: the affine rule above "
                            "IS the lat/lon of every pixel and every tile "
                            "corner, so no corner table is stored"),
    }


# ============================================================== listing ====
def parse_listing(html):
    """The Apache index -> ({tile: approximate bytes}, counts). REFUSES a
    duplicate or a name that is not a tile."""
    out, counts = {}, {}
    for name, size in ROW.findall(html):
        m = FILE.match(name)
        if not m:
            raise FormatError(f"{name}: not a Forest_height_{YEAR} file name")
        t = f"{m.group(1)}_{m.group(2)}"
        tile_corner(t)                         # refuses an impossible corner
        if t in out:
            raise FormatError(f"{name}: two files for tile {t}")
        out[t] = approx_bytes(size)
    listed = {n for n in ANY_HREF.findall(html)
              if not n.startswith(("http", "#", "?", ".")) and "/" not in n}
    other = sorted(listed - {r[0] for r in ROW.findall(html)})
    if other:
        counts["files_other"] = len(other)
        counts["files_other_names"] = other[:20]
    return out, counts


class Canopy30Adapter(sh.GridAdapter):
    store = "canopy30"
    title = ("Forest canopy height 2020, GLAD Global Land Cover and Land Use "
             "Change (Potapov et al. 2022), 30 m, 261 ten-degree tiles as "
             "4,176 2.5-degree groups (annual map)")
    family = "1tf"
    distribution = "public"
    licence = {
        "name": "CC BY",
        "redistribution": "attribution",
        "redistribution_confirmed": True,
        "derived_works": "free",
        "attribution": ("Potapov, P., M. C. Hansen, A. Pickens, A. "
                        "Hernandez-Serna, A. Tyukavina, S. Turubanova, V. "
                        "Zalles, X. Li, A. Khan, F. Stolle, N. Harris, X.-P. "
                        "Song, A. Baggett, I. Kommareddy and A. Kommareddy "
                        "(2022), The global 2000-2020 land cover and land use "
                        "change dataset derived from the Landsat archive: "
                        "first results, Front. Remote Sens. 3, 856903. "
                        "Forest height: Potapov, P. et al. (2021), Mapping "
                        "global forest canopy height through integration of "
                        "GEDI and Landsat data, Remote Sens. Environ. 253, "
                        "112165. Data: University of Maryland GLAD, "
                        "https://glad.umd.edu/dataset/GLCLUC2020"),
        "terms": ("the producer's own dataset page (glad.umd.edu, read "
                  "2026-09-20) names the Creative Commons Attribution licence "
                  "and no version number; the store carries the citation "
                  "above and the attribution track"),
    }
    time_dtype = "int32"
    credentials = ()
    channels = (("canopy_height", "m (whole metres of woody vegetation 3 m "
                                  "or taller; 0 = none)", 0.0, MAX_HEIGHT),)
    # 10 degrees / 40,000 = 0.00025 deg = 27.83 m at the equator, exactly a
    # thousandth of the family's 27.83 km reference: log2(0.001) = -9.9658.
    log2_fp = float(np.log2(DEG * 111.31949 / PX / 27.83))       # -9.966
    log2_dt = float(np.log2(365.0 / 5.0))                        # +6.190
    per_year = True
    first_year = int(sh.bin_year(RECORD_BIN))
    frames_per_bin = 1
    frame_seconds = sh.BIN_SECONDS
    dtype = "uint8"
    tile = sh.TILE
    zstd_level = 6
    note_estimate = {"bytes": 80e9,
                     "what": "family1tf.tex ledger: '60-100 GB (dense "
                             "metres)' for GLAD forest height 2020 at 30 m "
                             "over 261 tiles"}
    qc_policy = (
        "the product publishes ONE number a pixel and no flag of any kind: no "
        "water code, no no-data code and no fill value, measured here on "
        "three "
        "whole tiles (France, the Namib, Svalbard: 4.8e9 pixels, all of them "
        "0..35). A 0 is therefore a MEASUREMENT -- 'no woody vegetation "
        "3 m or "
        "taller' -- over ocean, ice and bare ground alike, and the store is "
        "dense. A value above 100 m is not a height (the tallest tree ever "
        "measured is 116 m and a Landsat model saturates far below it), so it "
        "becomes missing and is counted (`out_of_bounds`), never clipped, "
        "with "
        "the raw values listed in `height_codes` -- which is where the 2019 "
        "map's class codes (101 water, 102 no data) would appear if a future "
        "release adopted them")
    sources = (BASE + f"{YEAR}_<TILE>.tif",)
    verified = (
        "2026-09-20 from the sandbox, anonymously: the whole directory "
        "listing of " + BASE + " (261 files, every one a 2020_<TILE>.tif, no "
        "other object, sizes 13 MB .. 616 MB summing to 35.95 GB on the "
        "listing's own size column); the 301 redirect from " + REDIRECT_FROM +
        "; 2020_50N_000E.tif (303,815,710 bytes) opened with rasterio "
        "(40000 x 40000 uint8, EPSG:4326, transform (0.00025, 0, 0; 0, "
        "-0.00025, 50), LZW, one row a strip) and histogrammed whole (values "
        "0..35 only, 71.5 % zero); 2020_20S_010E.tif and 2020_80N_010E.tif "
        "histogrammed whole as well (0..14 and 0..27, no code above 100 in "
        "any of the three); zstd levels 1/6/9/15 timed on a 1,024-row strip "
        "of the French tile; and the licence sentence on "
        "glad.umd.edu/dataset/GLCLUC2020")
    notes = (
        "261 ten-degree tiles x 16 sub-tiles of 2.5 degrees = 4,176 groups of "
        "10,000 x 10,000 pixels, C = 1 (canopy_height) uint8 on the SAME tile "
        "grid as `lossyear`, so canopy height and loss year share a place's "
        "address. The product carries no water or no-data code -- measured on "
        "three whole tiles -- so a 0 is a measurement and the store is dense; "
        "zeros nevertheless cost 0.008 bytes a pixel compressed against 0.19 "
        "for a forested tile. 2.5 degrees is the largest sub-tile whose frame "
        "the probe can convert to float64 (the check it does) inside a "
        "runner's memory. zstd level 6, measured on the real tile: level 15 "
        "would cost about sixteen hours of compression for 4 % fewer bytes, "
        "and the whole store is ONE bin so it cannot be spread over year "
        f"lanes. The frame is filed in bin {RECORD_BIN} "
        f"({sh.bin_start_date(RECORD_BIN)} .. "
        f"{sh.bin_start_date(RECORD_BIN + 1) - dt.timedelta(days=1)}), the "
        f"bin holding 1 July {YEAR} -- the middle of the year the map "
        "describes. The ledger's '0.33-0.61 GB a tile, ~120 GB native' is the "
        "top of the size distribution and not its middle: re-measured here "
        "the tiles run 13-616 MB (median 77) and sum to 35.95 GB.")
    smoke_window = (str(sh.bin_start_date(RECORD_BIN)),
                    str(sh.bin_start_date(RECORD_BIN + 1)
                        - dt.timedelta(days=1)))
    smoke_probe_month = f"{sh.bin_start_date(RECORD_BIN):%Y-%m}"

    WORKERS = 3                  # download workers; the decode is serial

    def __init__(self):
        self.px = int(os.environ.get("CANOPY30_SMOKE_PX") or PX)
        self.sub = SUB
        self.spx = self.px // self.sub
        want = [t.strip() for t in
                (os.environ.get("CANOPY30_TILES") or "").split(",")
                if t.strip()]
        for t in want:
            tile_corner(t)                       # refuses a bad name
        self.tiles = want or list(TILES_2020)
        self._listing = None
        if want:
            self.notes = (
                f"{self.notes}\nRESTRICTED BUILD: CANOPY30_TILES limited this "
                f"store to {len(want)} of the product's {len(TILES_2020)} "
                f"tiles ({', '.join(want[:8])}"
                f"{' ...' if len(want) > 8 else ''}). It is NOT the whole "
                f"product.")
        if self.px != PX:
            self.notes = (f"{self.notes}\nSMOKE GRID: CANOPY30_SMOKE_PX set "
                          f"each tile to {self.px} x {self.px} instead of "
                          f"{PX} x {PX}. This is a synthetic store.")

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
            spec["annual_map"] = {
                "rule": ("a single-year STATE map is filed under the bin "
                         "holding 1 July of the year it describes -- the "
                         "middle of its period. (A CHANGE map such as "
                         "`lossyear` is filed under the first bin that BEGINS "
                         "after its period ends, so a target can never be "
                         "read as an input from inside its own year; a state "
                         "map has no such hazard, and the cone reads it at "
                         "lag >= one year anyway.)"),
                "period": [f"{YEAR}-01-01", f"{YEAR}-12-31"],
                "bin": int(RECORD_BIN),
                "bin_days": [str(sh.bin_start_date(RECORD_BIN)), str(end)],
                "frames_per_bin": 1,
                "values": {"0": "no woody vegetation 3 m or taller (ocean, "
                                "bare ground, ice, short vegetation) -- a "
                                "measurement, not a gap",
                           "1..100": "the canopy height in whole metres",
                           "255": "missing (only a value the source put "
                                  "outside 0..100, counted in out_of_bounds; "
                                  "the product itself has no such code, "
                                  "measured)"},
            }
        return out

    def record_frames(self, ctx, group):
        return 1

    # ------------------------------------------------------------- listing --
    def listing(self, ctx):
        """({tile: approximate bytes}, counts) from the directory or the
        smoke's copy of it."""
        if self._listing is not None:
            return self._listing
        if ctx.source_dir:
            p = os.path.join(ctx.source_dir, self.store, "index.html")
            if not os.path.exists(p):
                sys.exit(f"REFUSING canopy30: no {p} — the smoke's synthetic "
                         f"directory listing is missing")
            with open(p, "rb") as fh:
                raw = fh.read()
            ctx.count_bytes(len(raw))
        else:
            raw, why = cm.get_bytes(BASE, attempts=ctx.a.attempts)
            if raw is None:
                sys.exit(f"REFUSING canopy30: the listing of {BASE} answered "
                         f"{why} — a listing that comes back empty is a "
                         f"refusal (ml/CLAUDE.md, the 2026-09-14 rule)")
        try:
            files, counts = parse_listing(raw.decode("utf-8", "replace"))
        except FormatError as e:
            sys.exit(f"REFUSING canopy30: the listing of {BASE}: {e}")
        if not files:
            sys.exit(f"REFUSING canopy30: {BASE} lists no {YEAR}_<TILE>.tif "
                     f"at all ({counts}) — an empty listing is a refusal")
        self._listing = (files, counts)
        return self._listing

    def index(self, ctx):
        files, counts = self.listing(ctx)
        live = set(files)
        want = set(TILES_2020)
        if live != want:
            sys.exit(f"REFUSING canopy30: the listing of {BASE} holds "
                     f"{len(live)} tile(s) and this adapter's TILES_2020 "
                     f"names {len(want)} (only live: "
                     f"{sorted(live - want)[:8]}; only named: "
                     f"{sorted(want - live)[:8]}). The store's GROUPS are "
                     f"that tuple, so a changed tile set must be re-measured "
                     f"and committed rather than silently altering the store.")
        grid = set(all_tiles())
        if not live <= grid:
            sys.exit(f"REFUSING canopy30: the listing names tile(s) outside "
                     f"the documented 10-degree grid: "
                     f"{sorted(live - grid)[:8]}")
        have = sorted(set(self.tiles) & live)
        absent = sorted(set(self.tiles) - live)
        if not have:
            sys.exit(f"REFUSING canopy30: none of the {len(self.tiles)} "
                     f"declared tile(s) is in the listing")
        probe_tile = min(have, key=lambda t: files[t])
        out = {"dataset": f"GLAD Global Land Cover and Land Use Change "
                          f"2000-2020: forest height {YEAR}, 30 m",
               "url": BASE, "redirects_from": REDIRECT_FROM,
               "version": f"Forest_height_{YEAR}",
               "listing": counts,
               "tiles_in_product_grid": len(grid),
               "tiles_listed": len(live),
               "tiles_declared": len(self.tiles),
               "tiles_absent_upstream": absent,
               "groups": len(self.tiles) * self.sub * self.sub,
               "sub_tiles_per_side": self.sub,
               "group_pixels": self.spx,
               "bytes_approx": int(sum(files.get(t, 0)
                                       for t in self.tiles)),
               "bytes_approx_whole_product": int(sum(files.values())),
               "bytes_note": ("the listing's size column is rounded to three "
                              "characters (\"303M\"); the probe measures "
                              "exact bytes"),
               "bin": int(RECORD_BIN),
               "bin_days": [str(sh.bin_start_date(RECORD_BIN)),
                            str(sh.bin_start_date(RECORD_BIN + 1)
                                - dt.timedelta(days=1))],
               "record_period": [f"{YEAR}-01-01", f"{YEAR}-12-31"]}
        # ONE REAL TILE, opened and checked against the declared grid
        out["first_file"] = {"tile": probe_tile,
                             **self.read_tile_header(ctx, probe_tile)}
        out["grid"] = grid_of(group_name(probe_tile, 0, 0), self.px, self.sub)
        return out

    # ------------------------------------------------------------ one file --
    def _url(self, tile):
        return BASE + object_name(tile)

    def _local(self, ctx, tile):
        return os.path.join(ctx.source_dir, self.store, object_name(tile))

    def fetch_tif(self, ctx, tile, tmpdir):
        """The tile's GeoTIFF on local disk -> ({path, bytes}, None) or
        (None, why). Nothing is decoded here."""
        if ctx.source_dir:
            p = self._local(ctx, tile)
            if not os.path.exists(p):
                return None, f"{object_name(tile)}: listed, absent"
            ctx.count_bytes(os.path.getsize(p))
        else:
            p = os.path.join(tmpdir, object_name(tile))
            if f10b.fetch_first([self._url(tile)], p,
                                attempts=ctx.a.attempts) is None:
                return None, (f"{self._url(tile)}: listed in the directory, "
                              f"and 404")
        return {"path": p, "bytes": os.path.getsize(p)}, None

    def read_tile_header(self, ctx, tile):
        """Open the tile's GeoTIFF and CHECK the declared grid."""
        tmpdir = None if ctx.source_dir else \
            tempfile.mkdtemp(prefix="canopy_", dir=ctx.scratch)
        try:
            got, why = self.fetch_tif(ctx, tile, tmpdir)
            if got is None:
                sys.exit(f"REFUSING canopy30: {why}")
            with TIF_LOCK:
                meta = tif_meta(got["path"], tile, self.px)
            meta["bytes"] = got["bytes"]
            return meta
        finally:
            if tmpdir:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)

    def read_band(self, ctx, tile, path, r):
        """Sub-tile ROW r of `tile` as (band, counts, out_of_bounds).

        One full-width windowed read per row band: these GeoTIFFs are LZW
        striped ONE ROW at a time, so a narrow window decompresses the whole
        40,000-pixel row anyway and sixteen square windows would decompress
        the tile four times over. What is HELD is the uint8 band (400 MB at
        40,000 x 10,000), not the float16 frames — `sub_frame` builds one of
        those at a time, because four at once plus the probe's float64 copy
        does not fit a runner.
        """
        rasterio = _rasterio()
        px, spx = self.px, self.spx
        r0, r1 = r * spx, (r + 1) * spx
        with TIF_LOCK:
            d = rasterio.open(path)
            try:
                tif_check(d, path, tile, px)
                a = d.read(1, window=((r0, r1), (0, px)))
            finally:
                d.close()
        oob, codes = {}, {}
        bad = a > int(MAX_HEIGHT)
        nb = int(bad.sum())
        if nb:
            # OUT OF BOUNDS: counted by channel name AND by the raw value that
            # caused it, then turned into the missing code — never clipped
            # (contract rule 3). The product has no such value today; if a
            # re-issue adopts the 2019 map's 101 (water) / 102 (no data) they
            # will appear here by name.
            oob[self.channel_names[0]] = nb
            u, n = np.unique(a[bad], return_counts=True)
            codes = {int(k): int(v) for k, v in zip(u[:16], n[:16])}
            a[bad] = sh.U8_MISSING
        del bad
        counts = {"pixels_read": int(a.size)}
        if codes:
            counts["height_codes"] = codes
        return a, counts, oob

    def sub_frame(self, band, c):
        """One sub-tile's float16 [spx, spx, 1] frame with NaN for missing.

        float16 (not uint8) because the GridAdapter contract says raw units
        with NaN, and because `--stage probe` checks the channel bounds on a
        float64 copy of whatever the adapter yields: a raw uint8 255 would
        read there as a VALUE of 255 and fail rule 3 on every missing pixel.
        """
        spx = self.spx
        c0, c1 = c * spx, (c + 1) * spx
        v = band[:, c0:c1]
        f = v.astype(np.float16)
        f[v == sh.U8_MISSING] = np.nan
        return f[:, :, None]

    # ------------------------------------------------------- the contract --
    def classify(self, ctx, b):
        """-> None for the map's bin, else the reason it is not."""
        if int(b) == int(RECORD_BIN):
            return None
        return "before_record" if int(b) < int(RECORD_BIN) else "after_record"

    def fetch_frames(self, ctx, wanted):
        files, _ = self.listing(ctx)
        live = set(files)
        # Every wanted frame, split into the absences (no I/O at all) and the
        # real ones, grouped by TILE so one download and one pass over the
        # GeoTIFF answers all sixteen of its sub-tiles.
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
                tempfile.mkdtemp(prefix="canopy_", dir=ctx.scratch)
            got, err = self.fetch_tif(ctx, tile, tmpdir)
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
                    band = None
                    try:
                        band, counts, oob = self.read_band(
                            ctx, tile, got["path"], r)
                    except FormatError as e:
                        sys.exit(f"REFUSING canopy30: {tile}: {e}")
                    except (IOError, OSError) as e:
                        ctx.note_absent(tile, f"{type(e).__name__}: {e}")
                    if band is None:
                        break
                    for (g, b, f, rr, c) in by_row[r]:
                        cc = dict(counts) if c == 0 else {}
                        if oob and c == 0:
                            cc["out_of_bounds"] = dict(oob)
                        if c == 0:
                            cc["files_read"] = 1 if r == 0 else 0
                            cc["bytes_files"] = got["bytes"] if r == 0 else 0
                            cc["bands_read"] = 1
                        cc["fetch_seconds"] = round(time.time() - t0, 2)
                        t0 = time.time()
                        yield g, b, f, self.sub_frame(band, c), cc
                    del band
            finally:
                if tmpdir:
                    import shutil
                    shutil.rmtree(tmpdir, ignore_errors=True)

    # --------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260920):
        """`make_smoke_sources` sets CANOPY30_SMOKE_PX / CANOPY30_TILES, and
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
        sys.exit("canopy30 needs the `rasterio` package (pip install "
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
        rows = min(int(px), 1024)
        row = d.read(1, window=((0, rows), (0, int(px))))
        u, c = np.unique(row, return_counts=True)
        return {"height": d.height, "width": d.width,
                "dtype": d.dtypes[0], "crs": str(d.crs),
                "nodata_declared": d.nodatavals[0],
                "transform": [d.transform.a, d.transform.b, d.transform.c,
                              d.transform.d, d.transform.e, d.transform.f],
                "bounds": [d.bounds.left, d.bounds.bottom, d.bounds.right,
                           d.bounds.top],
                "compress": str(prof.get("compress")),
                "block_shapes": [list(b) for b in d.block_shapes],
                "rows_histogrammed": rows,
                "values_first_rows": {int(k): int(v) for k, v in
                                      zip(u[:40], c[:40])},
                "max_first_rows": int(u.max()) if u.size else None}
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
# The smoke's directory: three real tiles, one tile that is on the product's
# 10-degree GRID and NOT in its listing (`00N_180W`, the mid-Pacific — the
# shape 243 of the grid's 504 tiles really have), named in CANOPY30_TILES so
# the store is asked for it and answers `absent_upstream`; and one pixel of
# 200 m, out of bounds for a canopy height and the value that exercises
# `height_codes`.
SMOKE_TILES = ("50N_000E", "00N_020E", "20S_010E")
SMOKE_ABSENT = "00N_180W"
SMOKE_OOB_TILE = "00N_020E"
SMOKE_OOB_VALUE = 200
SMOKE_PX = 512               # 4 x 4 sub-tiles of 128 -> 1 store tile each


def smoke_field(tile, px):
    """A deterministic canopy-height field on a small grid.

    A disc of forest with a bare ring around it, so some sub-tiles hold only
    zeros (a dense store's cheapest case, which the test measures) and some
    hold tall trees.
    """
    la, lo = tile_corner(tile)
    k = int(abs(la) + abs(lo)) % 17
    yy = (np.arange(px, dtype=np.float64)[:, None] - px / 2) / px
    xx = (np.arange(px, dtype=np.float64)[None, :] - px / 2) / px
    r = np.hypot(yy, xx)
    h = np.clip(45.0 - 120.0 * r + 6.0 * np.sin(9 * xx + k)
                * np.cos(7 * yy - k), 0.0, 60.0)
    h[r > 0.30] = 0.0                              # "ocean" and bare ground
    a = np.rint(h).astype(np.uint8)
    if tile == SMOKE_OOB_TILE:
        a[px // 2, px // 2] = SMOKE_OOB_VALUE      # out of bounds
    return a


def listing_html(entries):
    rows = ['<tr><td valign="top"><img src="/icons/back.gif" '
            'alt="[PARENTDIR]"></td><td><a href="/users/Potapov/GLCLUC2020/">'
            'Parent Directory</a></td><td>&nbsp;</td>'
            '<td align="right">  - </td><td>&nbsp;</td></tr>']
    for name, size in entries:
        rows.append(f'<tr><td valign="top"><img src="/icons/image2.gif" '
                    f'alt="[IMG]"></td><td><a href="{name}">{name}</a>'
                    f'      </td><td align="right">2022-04-15 22:49  </td>'
                    f'<td align="right">{size}</td><td>&nbsp;</td></tr>')
    return ("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 3.2 Final//EN\">\n"
            f"<html><head><title>Index of /users/Potapov/GLCLUC2020/"
            f"Forest_height_{YEAR}</title></head><body>\n"
            f"<h1>Index of /users/Potapov/GLCLUC2020/Forest_height_{YEAR}"
            f"</h1><table>\n" + "\n".join(rows) + "\n</table></body></html>\n")


def _human(n):
    return f"{n / 1024 ** 2:.0f}M" if n >= 1024 ** 2 else f"{n // 1024}K"


def make_smoke_sources(root, d_lo, d_hi, seed=20260920):
    """The directory in its real shape: an Apache `index.html` naming the
    whole 261-tile product (so `index`'s set check sees what it is written to
    falsify) and real LZW GeoTIFFs on a `CANOPY30_SMOKE_PX` grid for the
    tiles `CANOPY30_TILES` names.

    Returns the truth: {(group, bin, frame): (uint8 [spx, spx, 1] or None,
    reason or None)} for every frame of every bin overlapping the record.
    """
    os.environ["CANOPY30_SMOKE_PX"] = str(SMOKE_PX)
    os.environ["CANOPY30_TILES"] = ",".join(SMOKE_TILES + (SMOKE_ABSENT,))
    px = SMOKE_PX
    spx = px // SUB
    base = os.path.join(root, "canopy30")
    os.makedirs(base, exist_ok=True)
    entries, fields = [], {}
    # The listing names the WHOLE product — all 261 tiles — because `index`'s
    # set check is written to falsify exactly that tuple; only the three the
    # smoke actually reads have bytes on disk. SMOKE_ABSENT is not in the
    # product at all, so it never appears here.
    for t in TILES_2020:
        if t in SMOKE_TILES:
            a = smoke_field(t, px)
            fields[t] = a
            p = os.path.join(base, object_name(t))
            write_tif(p, t, a, px)
            entries.append((object_name(t), _human(os.path.getsize(p))))
        else:
            entries.append((object_name(t), "77M"))
    with open(os.path.join(base, "index.html"), "w") as fh:
        fh.write(listing_html(sorted(entries)))

    truth = {}
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
                    a = fields[t]
                    sl = (slice(r * spx, (r + 1) * spx),
                          slice(c * spx, (c + 1) * spx))
                    v = a[sl]
                    out = np.where(v > int(MAX_HEIGHT),
                                   np.uint8(sh.U8_MISSING), v).astype(np.uint8)
                    truth[(g, b, 0)] = (out[:, :, None], None)
    return truth


ADAPTER = Canopy30Adapter
