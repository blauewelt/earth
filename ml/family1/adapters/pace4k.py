"""PACE OCI daily 4 km — hyperspectral ocean colour and the phytoplankton
community (family 1.gf, E-082 wave 6, the biosphere wave).

PLAIN ENGLISH. PACE (Plankton, Aerosol, Cloud, ocean Ecosystem) is NASA's 2024
ocean-colour satellite, and its Ocean Color Instrument resolves the WHOLE
visible spectrum instead of the handful of bands every earlier sensor had.
That buys two things nothing else in these families holds: a single number for
the colour of the water (the apparent visible wavelength, the wavelength the
whole spectrum averages to), and — by reading the shape of the spectrum rather
than its brightness — the abundances of the three groups of small
phytoplankton that dominate the open ocean. This store keeps the daily 4 km
mapped fields whole, tiled and compressed, the way `oc4k` keeps ESA's merged
record, so a model can read the bloom near the place it predicts at the
resolution it was measured.

SOURCE, VERIFIED 2026-09-20 FROM THIS SANDBOX. Three products, all version
3.2, all beginning **2024-03-05** (measured; the note says the same):

  PACE_OCI_L3M_BGC    C4184125847-OB_CLOUD
      PACE_OCI.<YYYYMMDD>.L3m.DAY.BGC.V3_2.4km.nc
  PACE_OCI_L3M_AOP    C4184125822-OB_CLOUD
      PACE_OCI.<YYYYMMDD>.L3m.DAY.AOP.V3_2.4km.nc
  PACE_OCI_L4M_MOANA  C4124891014-OB_CLOUD
      PACE_OCI.<YYYYMMDD>.L4m.DAY.MOANA.V3_2.4km.nc

THE ACCESS PATH IS OPeNDAP, AND IT IS ANONYMOUS — MEASURED, NOT ASSUMED. The
ordinary download hosts want an Earthdata Login: both
`oceandata.sci.gsfc.nasa.gov/getfile/<name>` and the cumulus bucket
`obdaac-tea.earthdatacloud.nasa.gov/ob-cumulus-prod-public/<name>` answered
HTTP 302 to `urs.earthdata.nasa.gov/oauth/authorize` for every one of the
three files. **OB.DAAC's own Hyrax server does not**: a DAP4 request to
`oceandata.sci.gsfc.nasa.gov/opendap/PACE_OCI/L3SMI/2025/0701/
PACE_OCI.20250701.L3m.DAY.BGC.V3_2.4km.nc.dap.nc4?dap4.ce=/chlor_a` returned
HTTP 200 and a 12.1 MB netCDF-4 file with no credential at all, in 2.4
seconds. So this store declares NO credential and its probe runs anywhere —
and an Earthdata netrc, where one exists, is still used, because the session
is built with `trust_env`.

AND THE SUBSETTING IS WHAT MAKES THE STORE AFFORDABLE. The AOP bundle carries
a `Rrs` cube of **172 wavelengths** x 4320 x 8640 int16 and is **745.6 MB a
day** (CMR's declared size for 2025-07-01); this store wants ONE variable out
of it, `avw`. Measured over one day through OPeNDAP with a DAP4 constraint
expression:

    BGC   chlor_a + poc + carbon_phyto          28.7 MB   5.5 s
    AOP   avw                                   11.0 MB   2.4 s
    MOANA the three cell abundances              6.5 MB   2.0 s
                                                -------
                                                46.2 MB a day, all 7 channels

against 788 MB a day for the three whole bundles — seventeen times less. Over
the 289 published days of 2024 that is 13.4 GB instead of 217 GB (CMR's
declared sizes for 2024: BGC 9.97 GB, AOP 205.4 GB, MOANA 1.64 GB).

THE GRID, MEASURED ON 2025-07-01 AND CHECKED IN EVERY FILE: 4,320 rows x
8,640 columns at 1/24 degree (4.64 km at the equator), plain lat/lon. `lat`
DESCENDS, 89.97917 .. -89.97917, and `lon` ascends from -179.97917 — both are
PIXEL CENTRES, so the grid's north and west EDGES are exactly +90 and -180 and
ROW 0 IS THE NORTHERNMOST. It is `oc4k`'s grid exactly, which is the point:
the two ocean-colour stores are read with one tile scheme.

THE MOANA FILE IS REGIONAL, AND THE LEDGER SAYS IT IS GLOBAL. Measured on the
real files: `PACE_OCI.20250701.L4m.DAY.MOANA.V3_2.4km.nc` is **3,360 x 2,640**
covering 70 S .. 70 N and 85 W .. 25 E — the Atlantic — while CMR's own
collection metadata declares the bounding rectangle -180..180, -90..90. CMR's
title says "Level-4 **Regional** Mapped" and the file agrees with the title,
not with the rectangle. The window is an EXACT sub-block of the global grid
(its first pixel centre sits at global row 480.00006, column 2280.00006, i.e.
row 480 and column 2280 to float32's last digit), so the three MOANA channels
are placed into the global array at that offset and are NaN everywhere else —
and `read_moana` recomputes the offset from the file's own axes every time and
REFUSES a window that is not aligned, so a widened product would be noticed
rather than silently mis-registered.

WHAT IS STORED: C = 7, dtype FLOAT16, one group on the global grid.

  log_chl        log10 of chlorophyll-a in mg m-3. STORED AS THE LOGARITHM,
                 for `oc4k`'s reason and with `oc4k`'s channel name and unit
                 string: chlorophyll spans three orders of magnitude (measured
                 on 2025-07-01: 0.0052 .. 99.19 mg m-3) and float16 would
                 space the raw value 0.0625 apart at 100.
  poc            particulate organic carbon, mg m-3 (source int16 with
                 scale_factor 0.2 and add_offset 6400, valid raw
                 -32000..-22000, i.e. 0..2000; measured 15.0 .. 1999.2).
  carbon_phyto   phytoplankton carbon, mg m-3 (0..1000; measured 0.6..999.6).
  avw_400        the apparent visible wavelength MINUS 400 nm. The offset is
                 not decoration: float16 spaces values 0.5 nm apart at 700 and
                 0.125 nm apart at 150, and AVW's whole useful range is
                 400..700 nm, so storing `avw - 400` keeps the quantisation at
                 0.03-0.25 nm across the ocean's real range (440..600 nm)
                 instead of a flat 0.5. ADD 400 TO GET NANOMETRES.
  pro_moana      Prochlorococcus,  in THOUSANDS of cells per millilitre
  syn_moana      Synechococcus,    in THOUSANDS of cells per millilitre
  pico_moana     picoeukaryotes,   in THOUSANDS of cells per millilitre
                 (the source is int32 cells per mL and reaches 3.6 x 10^7,
                 which float16 cannot hold at all — its largest number is
                 65,504 — so the three are divided by 1,000. Dividing rather
                 than taking a logarithm keeps a measured ZERO a zero, and the
                 relative precision of float16 is 0.05 % everywhere, so
                 nothing is lost by the scaling.)

THE THREE MOANA CHANNELS ARE A LEVEL-4 DERIVED PRODUCT and are flagged as such
in `qc_policy`, in every `tile_grid.json` (`derived_channels`) and in
store.json's notes. They are admitted because no other source in these
families carries the community composition at all; a reader that wants only
measurements takes the first four channels.

THE FILL RULES, MEASURED IN THE REAL FILES rather than read off a page:
  chlor_a, carbon_phyto, avw   float32, `_FillValue` -32767.0
  poc                          int16,   `_FillValue` -32767
  the three MOANA fields       int32,   `_FillValue` -32767 — AND a second,
                               UNDOCUMENTED sentinel: 2025-07-01's
                               `prococcus_moana` carries five pixels at
                               -2,147,483,648 (int32's minimum) and two other
                               negative values, the smallest -691,495,040. A
                               cell count is never negative, so ANY negative
                               value becomes NaN and is counted
                               `negative_cells`.
  Every field's `valid_min`/`valid_max` is recorded, and for the MOANA fields
  the DATA RUN PAST IT: 0.022 % of valid Prochlorococcus pixels, 0.014 % of
  Synechococcus and 0.58 % of picoeukaryotes exceed the producer's declared
  maxima (600,000 / 300,000 / 40,000 cells per mL). Those pixels are KEPT and
  COUNTED (`above_producer_valid_max`) rather than discarded: half a per cent
  of a bloom signal is not noise, and the fact is made visible instead of
  being decided silently.

A FRAME THAT IS NOT IN THE SOURCE is a zero-length frame with a reason:
`before_record` (before 2024-03-05), `after_record`, `absent_upstream` (inside
the record, no BGC or no AOP file). **MOANA is OPTIONAL**: it was published on
277 of 2024's 289 days (measured), so a day with BGC and AOP but no MOANA is a
real frame whose last three channels are NaN, counted `moana_not_published` —
not an absent frame, because four of the seven channels are there.

PHASE B: the six-band `Rrs` subset the AOP bundle also holds, `nflh` (the
chlorophyll fluorescence line height), `pic`, and the near-real-time
collections; the 1.2 km level-2 swath is a phase-C row (`family1gf.tex` §5).
"""
import calendar
import datetime as dt
import json
import os
import sys
import tempfile
import threading
import time
import urllib.parse

import numpy as np

import build_family10_stores as f10b
from family1 import sharded as sh
from family1.adapters import _common as cm
from family1.adapters import _modis_sin as ms          # cmr_page, FormatError

H, W = 4320, 8640
DX = 1.0 / 24.0
X0, Y0 = -180.0, 90.0          # the grid's WEST and NORTH edges
OPENDAP = "https://oceandata.sci.gsfc.nasa.gov/opendap/PACE_OCI/"
GETFILE = "https://oceandata.sci.gsfc.nasa.gov/getfile/"
VERSION = "3.2"
VTAG = "V3_2"
RECORD_START = dt.date(2024, 3, 5)
CHLA_FLOOR = 1e-4              # below this, log10 is not a measurement
AVW_OFFSET = 400.0             # nm; the stored channel is avw - AVW_OFFSET
MOANA_SCALE = 1000.0           # cells per mL -> thousands of cells per mL

# product key -> (CMR short_name, OPeNDAP directory, the file-name middle,
#                 the variables this store reads, whether it is required)
PRODUCTS = {
    "bgc": ("PACE_OCI_L3M_BGC", "L3SMI", "L3m.DAY.BGC",
            ("chlor_a", "poc", "carbon_phyto"), True),
    "aop": ("PACE_OCI_L3M_AOP", "L3SMI", "L3m.DAY.AOP", ("avw",), True),
    "moana": ("PACE_OCI_L4M_MOANA", "L4SMI", "L4m.DAY.MOANA",
              ("prococcus_moana", "syncoccus_moana", "picoeuk_moana"), False),
}
DERIVED = ("pro_moana", "syn_moana", "pico_moana")

# variable -> (channel, unit, bound lo, bound hi)
CHANNELS = (
    ("log_chl", "log10(mg m-3) — the base-10 LOGARITHM of chlorophyll-a, "
                "not the concentration", -4.0, 2.5),
    ("poc", "mg m-3", 0.0, 5000.0),
    ("carbon_phyto", "mg m-3", 0.0, 5000.0),
    ("avw_400", "nm above 400 — the apparent visible wavelength MINUS 400 nm; "
                "add 400 to get nanometres", -50.0, 400.0),
    ("pro_moana", "thousands of cells per mL (level-4 derived)", 0.0, 60000.0),
    ("syn_moana", "thousands of cells per mL (level-4 derived)", 0.0, 60000.0),
    ("pico_moana", "thousands of cells per mL (level-4 derived)", 0.0,
     60000.0),
)
# the producer's own declared maxima, measured in the 2025-07-01 files
MOANA_VALID_MAX = {"prococcus_moana": 600000, "syncoccus_moana": 300000,
                   "picoeuk_moana": 40000}

# HDF5 under netCDF4 is NOT thread-safe (seaice_asi measured a SIGBUS with
# four workers each opening a file). Downloads stay parallel; every open,
# read and close holds this lock.
NC_LOCK = threading.Lock()

FormatError = ms.FormatError


def object_name(prod, d):
    return f"PACE_OCI.{d:%Y%m%d}.{PRODUCTS[prod][2]}.{VTAG}.4km.nc"


def opendap_url(prod, d, variables):
    """The DAP4 subset request for one product and one day.

    `dap4.ce` is Hyrax's constraint expression: naming the variables returns a
    netCDF-4 file holding only those (and their coordinate axes). The `;`
    separators and the leading `/` are part of the syntax and must survive
    quoting, which is why `safe` keeps them.
    """
    ce = ";".join(f"/{v}" for v in list(variables) + ["lat", "lon"])
    return (f"{OPENDAP}{PRODUCTS[prod][1]}/{d:%Y}/{d:%m%d}/"
            f"{object_name(prod, d)}.dap.nc4?dap4.ce="
            + urllib.parse.quote(ce, safe=";/"))


def grid_of(h=H, w=W):
    dy = 180.0 / int(h)
    dx = 360.0 / int(w)
    return {
        "H": int(h), "W": int(w), "pixel_deg": dx,
        "pixel_km_equator": round(dx * 111.31949, 4),
        "crs": "EPSG:4326",
        "proj4": "+proj=longlat +datum=WGS84 +no_defs",
        "projection": {"grid_mapping_name": "latitude_longitude",
                       "semi_major_axis": 6378137.0,
                       "inverse_flattening": 298.257223563},
        "x0": X0, "dx": dx, "y0": Y0, "dy": -dy,
        "coordinates": ("lon = x0 + (col + 0.5) * dx, lat = y0 + (row + 0.5) "
                        "* dy with dy NEGATIVE; x0/y0 are the grid's WEST and "
                        "NORTH edges, and the source's `lat`/`lon` are the "
                        "pixel CENTRES those edges imply"),
        "row_order": ("row 0 is the NORTHERNMOST row (the source netCDF's "
                      "DESCENDING `lat` axis, 89.97917 .. -89.97917, checked "
                      "on the real 2025-07-01 files); col 0 is the "
                      "westernmost, lon -179.97917"),
        "extent": [-180.0, -90.0, 180.0, 90.0],
        "extent_note": "[west, south, east, north] in degrees; ocean only",
        "same_grid_as": ("oc4k — the two ocean-colour stores share this grid "
                         "exactly, so one tile scheme reads both"),
        "source_variables": (
            "chlor_a (float32 mg m-3, stored as log10), poc (int16 with "
            "scale_factor 0.2 and add_offset 6400, mg m-3), carbon_phyto "
            "(float32 mg m-3) and avw (float32 nm, stored minus 400) on this "
            "grid; the three MOANA cell abundances (int32 cells per mL, "
            "stored in thousands) on a REGIONAL sub-block of it, 70 S..70 N "
            "and 85 W..25 E, placed at its exact offset and NaN elsewhere"),
        "no_tile_corners": ("a plain geographic grid: the affine rule above "
                            "IS the lat/lon of every pixel and every tile "
                            "corner, so no corner table is stored"),
    }


def sub_block(lat, lon, h=H, w=W):
    """(row0, col0) of a regional file inside the global grid. REFUSES a
    window that is not an exact sub-block of it."""
    dy, dx = 180.0 / int(h), 360.0 / int(w)
    r0 = (Y0 - float(lat[0])) / dy - 0.5
    c0 = (float(lon[0]) - X0) / dx - 0.5
    if abs(r0 - round(r0)) > 1e-3 or abs(c0 - round(c0)) > 1e-3:
        raise FormatError(
            f"a regional file whose first pixel centre is ({lat[0]}, "
            f"{lon[0]}) does not land on the global 1/24-degree grid (row "
            f"{r0}, col {c0}) — a window that is not an exact sub-block "
            f"cannot be placed without resampling, and this store never "
            f"resamples")
    r0, c0 = int(round(r0)), int(round(c0))
    if r0 < 0 or c0 < 0 or r0 + len(lat) > int(h) or c0 + len(lon) > int(w):
        raise FormatError(f"a regional file of {len(lat)} x {len(lon)} at "
                          f"row {r0}, col {c0} does not fit the global "
                          f"{h} x {w} grid")
    return r0, c0


# source variable -> the channel it becomes
CHANNEL_OF = {"chlor_a": 0, "poc": 1, "carbon_phyto": 2, "avw": 3,
              "prococcus_moana": 4, "syncoccus_moana": 5, "picoeuk_moana": 6}


def new_frame(h=H, w=W):
    """An all-missing [h, w, 7] frame. 1.05 GB at the real grid, which is why
    the reader fills it ONE VARIABLE AT A TIME and frees each source array as
    it goes: the probe additionally makes a float64 copy of whatever the
    adapter yields (2.1 GB), so a reader that held all seven source arrays as
    well would need six gigabytes and was measured being killed for it."""
    return np.full((int(h), int(w), len(CHANNELS)), np.nan, np.float32)


def put_channel(out, name, a, counts=None, r0=0, c0=0):
    """Place ONE source variable into its channel of the frame, in its stored
    units. `a` is float32 with the source's fill ALREADY NaN; `r0`/`c0` offset
    a regional array (the MOANA window) inside the global grid.

    ONE numeric path, called by `read_day` and by the smoke's truth, so the
    two cannot disagree by a rounding: everything stays float32, because
    log10 of a float32 rounded to float16 is not always log10 of the float64
    rounded to float16.
    """
    i = CHANNEL_OF[name]
    a = np.asarray(a, np.float32)
    if name == "chlor_a":
        a = a.copy()
        low = np.isfinite(a) & (a <= np.float32(CHLA_FLOOR))
        n_low = int(low.sum())
        if n_low and counts is not None:
            counts["chl_below_floor"] = counts.get("chl_below_floor", 0) \
                + n_low
        a[low] = np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            a = np.log10(a)
        a[~np.isfinite(a)] = np.nan
    elif name == "avw":
        a = a - np.float32(AVW_OFFSET)
    elif name in MOANA_VALID_MAX:
        a = a / np.float32(MOANA_SCALE)
    out[r0:r0 + a.shape[0], c0:c0 + a.shape[1], i] = a
    return out


def mask_bounds_channelwise(out, bounds):
    """`bounds` applied one channel at a time, in place -> {channel: n}.

    The framework's own `mask_bounds` compares the whole (N, C) array at once,
    which is a 1.05 GB boolean at this grid; this holds one channel's mask at
    a time (36 MB) and returns the identical dict.
    """
    lo_b, hi_b = bounds
    oob = {}
    for i, c in enumerate(CHANNELS):
        v = out[:, :, i]
        with np.errstate(invalid="ignore"):
            bad = np.isfinite(v) & ((v < float(lo_b[i]))
                                    | (v > float(hi_b[i])))
        n = int(bad.sum())
        if n:
            v[bad] = np.nan
            oob[c[0]] = n
        del bad
    return oob


def to_frame(fields, bounds=None, h=H, w=W):
    """The raw variables of one day -> the stored frame [h, w, 7] and counts.

    The smoke's truth path: small grids, every variable in memory at once.
    `read_day` uses the same `put_channel` and `mask_bounds_channelwise` one
    variable at a time.
    """
    out = new_frame(h, w)
    counts = {}
    for name in CHANNEL_OF:
        a = fields.get(name)
        if a is not None:
            put_channel(out, name, a, counts)
    oob = {} if bounds is None else mask_bounds_channelwise(out, bounds)
    return out, counts, oob


class Pace4kAdapter(sh.GridAdapter):
    store = "pace4k"
    title = ("Ocean colour and the phytoplankton community, PACE OCI daily "
             "4 km mapped v3.2: log10 chlorophyll-a, particulate organic "
             "carbon, phytoplankton carbon, apparent visible wavelength and "
             "the three MOANA cell abundances")
    family = "1gf"
    distribution = "public"
    licence = {
        "name": "NASA open data (no restrictions)",
        "redistribution": "attribution",
        "redistribution_confirmed": True,
        "derived_works": "free",
        "attribution": ("NASA Ocean Biology Processing Group (2024-2026). "
                        "PACE OCI Level-3 Global Mapped Ocean Biogeochemical "
                        "Properties (doi:10.5067/PACE/OCI/L3M/BGC/3.2), "
                        "Apparent Optical Properties "
                        "(doi:10.5067/PACE/OCI/L3M/AOP/3.2) and Level-4 "
                        "Mapped MOANA phytoplankton community "
                        "(doi:10.5067/PACE/OCI/L4M/MOANA/3.2). NASA Goddard "
                        "Space Flight Center, Ocean Ecology Laboratory, "
                        "Ocean Biology Distributed Active Archive Center. "
                        "MOANA algorithm: Lange, P. K. et al. (2020), Optics "
                        "Express 28, 25682-25705, doi:10.1364/OE.398127."),
        "terms": ("the files' own `license` attribute, read 2026-09-20: "
                  "'https://science.nasa.gov/earth-science/earth-science-data/"
                  "data-information-policy/' — NASA's Earth science data are "
                  "open, full and without restriction, free of charge, with "
                  "no period of exclusive access"),
    }
    time_dtype = "int32"
    # NO CREDENTIAL, and that is a measurement: OB.DAAC's Hyrax server answered
    # a DAP4 subset request for all three products with HTTP 200 and no login
    # (2026-09-20), while its own `getfile` and cumulus download hosts answered
    # 302 to Earthdata Login. An Earthdata netrc is still USED where one
    # exists, because the session trusts the environment.
    credentials = ()
    channels = CHANNELS
    log2_fp = float(np.log2(360.0 / W * 111.31949 / 27.83))       # -2.583
    log2_dt = float(np.log2(1.0 / 5.0))                           # -2.322
    per_year = True
    first_year = RECORD_START.year
    frames_per_bin = 5
    frame_seconds = 86400
    dtype = "float16"
    tile = sh.TILE
    zstd_level = sh.DEFAULT_LEVEL
    grid = grid_of()
    note_estimate = {"bytes": 50e9,
                     "what": ("family1gf.tex ledger: '~50 GB a year, ~0.13 TB "
                              "so far' for PACE OCI daily 4 km, 37.3 M px at "
                              "v ~ 0.10-0.15")}
    qc_policy = (
        "THE LAST THREE CHANNELS ARE A LEVEL-4 DERIVED PRODUCT. `pro_moana`, "
        "`syn_moana` and `pico_moana` are not radiances or a retrieval of "
        "one: they are cell abundances inferred from the SHAPE of the "
        "reflectance spectrum with sea-surface temperature as an ancillary "
        "input (the MOANA algorithm, Lange et al. 2020), and they are "
        "admitted -- against this family's preference for measurements -- "
        "because nothing else carries the phytoplankton community at all "
        "(E-082 wave 6 2.2). They are flagged `derived: true` in every "
        "tile_grid.json, and a reader that wants measurements only takes the "
        "first four channels. They are also REGIONAL: the source file covers "
        "70 S..70 N and 85 W..25 E, an exact sub-block of the global grid, "
        "and the three channels are NaN outside it. "
        "EVERYTHING ELSE follows the producer. The only per-pixel flag in "
        "these files is the fill value: chlor_a, carbon_phyto and avw carry "
        "float -32767 and poc and the MOANA fields int -32767, and all become "
        "NaN. A chlorophyll at or below 1e-4 mg m-3 is NaN too, because log10 "
        "of it is not a measurement, and it is counted (`chl_below_floor`). A "
        "NEGATIVE cell abundance is NaN and counted (`negative_cells`): the "
        "measured files carry a second, undocumented sentinel at int32's "
        "minimum and a few other negative values, and a cell count is never "
        "negative. A value ABOVE the producer's own valid_max for a MOANA "
        "field is KEPT and counted (`above_producer_valid_max`) -- 0.014 % to "
        "0.58 % of valid pixels, measured -- because half a per cent of a "
        "bloom signal is not noise and the fact belongs in the open. A value "
        "outside a channel's bounds becomes NaN and is counted "
        "(`out_of_bounds`), never clipped. The AOP bundle's 172-wavelength "
        "Rrs cube, nflh, aot_865 and angstrom, and the BGC bundle's pic, are "
        "in the same files and are NOT stored: they are phase B")
    sources = tuple(OPENDAP + f"{PRODUCTS[p][1]}/<YYYY>/<MMDD>/"
                    + f"PACE_OCI.<YYYYMMDD>.{PRODUCTS[p][2]}.{VTAG}.4km.nc"
                    for p in ("bgc", "aop", "moana")) + (ms.CMR,)
    verified = (
        "2026-09-20 from the sandbox, anonymously: the CMR collection search "
        "for PACE_OCI_L3M_BGC, PACE_OCI_L3M_AOP and PACE_OCI_L4M_MOANA "
        "(C4184125847 / C4184125822 / C4124891014-OB_CLOUD, all version 3.2, "
        "all time_start 2024-03-05); the granule listing of each for 2024 by "
        "name pattern (BGC 289 days 9.97 GB, AOP 289 days 205.4 GB, MOANA "
        "**277** days 1.64 GB by CMR's declared sizes, all "
        "2024-03-05..2024-12-31); a 302 to urs.earthdata.nasa.gov from both "
        "`getfile` and the cumulus bucket for all three files of 2025-07-01, "
        "against HTTP 200 and real data from OB.DAAC's OPeNDAP for the same "
        "files with no credential; the DMRs of all three (BGC lat 4320 lon "
        "8640 with poc/pic int16 and chlor_a/carbon_phyto float32; AOP "
        "wavelength 172 x lat x lon with Rrs/nflh int16, avw float32; MOANA "
        "lat **3360** lon **2640** with three int32 fields); the three DAP4 "
        "subsets of 2025-07-01 downloaded and opened (28.7 + 11.0 + 6.5 MB, "
        "9.9 s, valid fractions 8.34 % / 8.34 % / 9.06 %, chlor_a "
        "0.0052..99.19, poc 15.0..1999.2, carbon_phyto 0.60..999.57, avw "
        "400.39..699.65, and the MOANA fields' negative sentinels and "
        "above-valid_max tails); the MOANA window shown to sit at global row "
        "480, column 2280 exactly; and THREE REAL DAYS (2024-03-05, "
        "2024-03-06, 2024-07-15) fetched and written through this adapter's "
        "own code path — 46.6-53.5 MB a day through OPeNDAP, 20 s to read and "
        "15 s to encode a frame, a 25.4-27.0 MB shard of 387-410 stored tiles "
        "of 578 (19.4-20.6x over the raw float16 frame), valid fractions "
        "0.084-0.091 for the four global channels and 0.0229-0.0235 for the "
        "three regional MOANA ones, and the MOANA window measured at row 480, "
        "column 2280, 3360 x 2640 in every one of them")
    notes = (
        "C = 7 on `oc4k`'s own global 4 km grid. Read through OB.DAAC's "
        "OPeNDAP with a DAP4 constraint expression, which needs NO Earthdata "
        "Login (measured) and takes the day's fetch from 788 MB to 46.2 MB by "
        "leaving the AOP bundle's 172-wavelength Rrs cube where it is. "
        "`log_chl` is the base-10 LOGARITHM of chlorophyll-a, `oc4k`'s "
        "channel exactly; `avw_400` is the apparent visible wavelength MINUS "
        "400 nm, because float16 spaces values 0.5 nm apart at 700 and 0.125 "
        "at 150; the three MOANA channels are in THOUSANDS of cells per "
        "millilitre, because the source reaches 3.6e7 cells per mL and "
        "float16's largest number is 65,504. THE MOANA FILE IS REGIONAL -- "
        "70 S..70 N, 85 W..25 E, an exact sub-block of the global grid -- "
        "although CMR's collection metadata declares a global bounding "
        "rectangle; the three channels are placed at the measured offset and "
        "are NaN outside it, and the offset is recomputed from every file's "
        "own axes. MOANA is also OPTIONAL: it was published on 277 of 2024's "
        "289 days, and a day without it is a real frame whose last three "
        "channels are NaN, counted `moana_not_published`. The three MOANA "
        "channels are a LEVEL-4 DERIVED product and are flagged as such. "
        "MEASURED ON THREE REAL DAYS through this adapter's own code path: a "
        "frame is 26.4 MB stored (mean of 25.4, 26.9 and 27.0), so a year of "
        "about 350 published days is ~9.3 GB and the record from 2024-03-05 "
        "to 2026-09 about 23 GB — five times less than the ledger's ~50 GB a "
        "year, which charged every valid pixel two uncompressed bytes. "
        "`--stage probe` needs about 6 GB of memory at C = 7, because the "
        "framework converts each frame to float64 to check the channel bounds "
        "(2.1 GB) while holding the frame (1.05 GB) and its float16 encoding: "
        "that is a hosted runner's job, not a 6 GB sandbox's, and the "
        "adapter's own peak without it is 1.9 GB.")
    smoke_window = ("2024-03-05", "2024-03-19")
    smoke_probe_month = "2024-03"

    WORKERS = 3

    def __init__(self):
        g = (os.environ.get("PACE4K_SMOKE_GRID") or "").split(",")
        self.w, self.h = (int(g[0]), int(g[1])) if len(g) == 2 else (W, H)
        self._listing = {}
        self._session = None
        if (self.w, self.h) != (W, H):
            self.notes = (f"{self.notes}\nSMOKE GRID: PACE4K_SMOKE_GRID set "
                          f"the grid to {self.w} x {self.h} instead of "
                          f"{W} x {H}. This is a synthetic store.")

    def group_grids(self):
        return {self.store: grid_of(self.h, self.w)}

    def specs(self):
        out = super().specs()
        for spec in out.values():
            for c in spec["channels"]:
                c["derived"] = c["name"] in DERIVED
            spec["derived_channels"] = {
                "names": list(DERIVED),
                "level": "L4",
                "what": ("cell abundances of Prochlorococcus, Synechococcus "
                         "and picoeukaryotes, inferred from the shape of the "
                         "reflectance spectrum with sea-surface temperature "
                         "as an ancillary input (MOANA, Lange et al. 2020) — "
                         "not a retrieval of a radiance"),
                "regional": ("the MOANA source file covers 70 S..70 N and "
                             "85 W..25 E only; these three channels are NaN "
                             "outside that window"),
                "why_admitted": ("nothing else in these families carries the "
                                 "phytoplankton community composition "
                                 "(E-082 wave 6 §2.2)")}
            spec["record_start"] = str(RECORD_START)
        return out

    # ------------------------------------------------------------- listing --
    def listing(self, ctx, prod, year):
        """{date: {name, url, bytes_cmr}} for one product and one year."""
        key = (prod, int(year))
        if key in self._listing:
            return self._listing[key]
        sn = PRODUCTS[prod][0]
        pat = (f"PACE_OCI.{int(year):04d}*.{PRODUCTS[prod][2]}.{VTAG}"
               f".4km.nc")
        if ctx.source_dir:
            p = os.path.join(ctx.source_dir, self.store,
                             f"cmr_{prod}_{int(year):04d}.json")
            if not os.path.exists(p):
                self._listing[key] = ({}, {"year_not_listed": 1})
                return self._listing[key]
            with open(p, "rb") as fh:
                raw = fh.read()
            ctx.count_bytes(len(raw))
            entries = (json.loads(raw).get("feed") or {}).get("entry") or []
            counts = {"cmr_pages": 0}
        else:
            entries, counts = [], {"cmr_pages": 0, "cmr_bytes": 0}
            after = None
            while True:
                got, after, hits, nb = ms.cmr_page(
                    {"short_name": sn, "version": VERSION, "page_size": 2000,
                     "sort_key": "start_date", "readable_granule_name": pat,
                     "options[readable_granule_name][pattern]": "true"},
                    after, attempts=ctx.a.attempts)
                entries += got
                counts["cmr_pages"] += 1
                counts["cmr_bytes"] += nb
                counts["cmr_hits"] = hits
                if not after or not got:
                    break
                if counts["cmr_pages"] > 100:
                    sys.exit(f"REFUSING pace4k: CMR paged past 100 pages for "
                             f"{sn} {year}")
        files = {}
        for e in entries:
            title = str(e.get("title") or e.get("producer_granule_id") or "")
            d = parse_day(title, prod)
            if d.year != int(year):
                raise FormatError(f"{title}: not a {year} granule")
            if d in files:
                raise FormatError(f"{title}: two granules for {d}")
            try:
                size = float(e.get("granule_size") or 0.0) * 1024 * 1024
            except (TypeError, ValueError):
                size = 0.0
            files[d] = {"title": title, "bytes_cmr": int(size),
                        "url": opendap_url(prod, d, PRODUCTS[prod][3])}
        counts["granules_listed"] = len(files)
        self._listing[key] = (files, counts)
        return self._listing[key]

    def years(self, ctx):
        """The calendar years whose listings the window's FRAMES can need.

        A tier-G "year" is the bins whose FIRST day falls in it, so the last
        bin of a year runs up to four days into the next one: bin 3214 opens
        2025-12-31 and holds 2026-01-01 .. 04. Listing only `ctx.years` made
        the 2025 lane (#353) record those four days as `after_record` — the
        record looked like it ended where the lane's window did. The
        framework's `ctx.grid_frame_days` is the first and last day of every
        frame this context owns; without it (an older caller) the years are
        `ctx.years`, as before.
        """
        ys = sorted(ctx.years)
        if not ys:
            return []
        last = max(ys)
        span = getattr(ctx, "grid_frame_days", None)
        if span:
            last = max(last, span[1].year)
        return list(range(max(min(ys), RECORD_START.year), last + 1))

    def record(self, ctx):
        """(first, last) day BOTH REQUIRED products publish in the window."""
        lo = hi = None
        for prod in ("bgc", "aop"):
            days = set()
            for y in self.years(ctx):
                days |= set(self.listing(ctx, prod, y)[0])
            if not days:
                return None, None
            lo = min(days) if lo is None else max(lo, min(days))
            hi = max(days) if hi is None else min(hi, max(days))
        return lo, hi

    def day_files(self, ctx, d):
        """{product: rec} for one day; `bgc` and `aop` are required."""
        out = {}
        for prod in PRODUCTS:
            rec = self.listing(ctx, prod, d.year)[0].get(d)
            if rec is not None:
                out[prod] = rec
        return out

    def record_frames(self, ctx, group):
        n = 0
        lo, hi = self.record(ctx)
        if lo is None:
            return None
        for y in self.years(ctx):
            bgc = self.listing(ctx, "bgc", y)[0]
            aop = self.listing(ctx, "aop", y)[0]
            n += sum(1 for d in bgc if d in aop and lo <= d <= hi)
        return n

    def empty_months(self, ctx, y):
        lo, hi = self.record(ctx)
        have = set()
        bgc = self.listing(ctx, "bgc", y)[0]
        aop = self.listing(ctx, "aop", y)[0]
        for d in bgc:
            if d in aop:
                have.add(f"{d.year:04d}-{d.month:02d}")
        out = []
        for m in range(1, 13):
            m0 = dt.date(y, m, 1)
            m1 = dt.date(y, m, calendar.monthrange(y, m)[1])
            if m1 < lo or m0 > hi:
                continue
            if f"{y:04d}-{m:02d}" not in have:
                out.append(f"{y:04d}-{m:02d}")
        return out

    # --------------------------------------------------------------- index --
    def index(self, ctx):
        out = {"dataset": f"PACE OCI daily 4 km mapped v{VERSION} (BGC, AOP "
                          f"and MOANA)",
               "url": OPENDAP, "version": VERSION,
               "access": ("OB.DAAC OPeNDAP (Hyrax) DAP4 variable subsetting, "
                          "anonymous — measured 2026-09-20; the `getfile` and "
                          "cumulus download hosts answer 302 to Earthdata "
                          "Login for the same files"),
               "products": {}}
        for prod in PRODUCTS:
            per, days = {}, set()
            nb = 0
            for y in self.years(ctx):
                files, counts = self.listing(ctx, prod, y)
                per[str(y)] = len(files)
                days |= set(files)
                nb += sum(r["bytes_cmr"] for r in files.values())
            out["products"][prod] = {
                "short_name": PRODUCTS[prod][0], "version": VERSION,
                "variables": list(PRODUCTS[prod][3]),
                "required": PRODUCTS[prod][4],
                "days_per_year": per, "days": len(days),
                "record": ([str(min(days)), str(max(days))] if days else None),
                "bytes_cmr": int(nb),
                "bytes_note": ("CMR's own `granule_size` in MB, summed over "
                               "the WHOLE bundle; this store fetches only the "
                               "variables above, through OPeNDAP, and the "
                               "probe measures what that really costs")}
        lo, hi = self.record(ctx)
        if lo is None:
            sys.exit(f"REFUSING pace4k: CMR lists no BGC or no AOP granule "
                     f"for any of the years {self.years(ctx)} — an empty "
                     f"listing is a refusal (ml/CLAUDE.md, the 2026-09-14 "
                     f"rule)")
        absent, no_moana = [], []
        d = lo
        while d <= hi:
            have = self.day_files(ctx, d)
            if "bgc" not in have or "aop" not in have:
                absent.append(str(d))
            elif "moana" not in have:
                no_moana.append(str(d))
            d += dt.timedelta(days=1)
        empty = {str(y): self.empty_months(ctx, y) for y in self.years(ctx)
                 if self.empty_months(ctx, y)}
        if empty:
            sys.exit(f"REFUSING pace4k: a calendar month inside the record "
                     f"has no day with both required products: {empty}")
        out["record"] = [str(lo), str(hi)]
        out["record_start_declared"] = str(RECORD_START)
        out["days_in_record"] = (hi - lo).days + 1
        out["days_absent_upstream"] = absent
        out["days_without_moana"] = len(no_moana)
        out["days_without_moana_sample"] = no_moana[:40]
        out["frames_per_bin"] = self.frames_per_bin
        out["frame_seconds"] = self.frame_seconds
        out["grid"] = grid_of(self.h, self.w)
        out["derived_channels"] = list(DERIVED)
        # ONE REAL DAY, every file read and checked against the declared grid
        out["first_file"] = {"day": str(lo),
                             **self.read_first_day(ctx, lo)}
        return out

    # ------------------------------------------------------------ one day ---
    def fetch_one(self, ctx, prod, d, rec, tmpdir):
        """One product's subset on local disk -> (path, bytes), or
        (None, why) when the file the listing named is not there."""
        if ctx.source_dir:
            p = os.path.join(ctx.source_dir, self.store, "opendap",
                             PRODUCTS[prod][1], f"{d:%Y}", f"{d:%m%d}",
                             object_name(prod, d))
            if not os.path.exists(p):
                return None, f"{object_name(prod, d)}: listed and absent"
            ctx.count_bytes(os.path.getsize(p))
            return p, os.path.getsize(p)
        p = os.path.join(tmpdir, object_name(prod, d))
        if f10b.fetch_first([rec["url"]], p,
                            attempts=ctx.a.attempts) is None:
            return None, f"{rec['url']}: listed, and 404"
        return p, os.path.getsize(p)

    def fetch_day(self, ctx, d, have):
        """The day's files on local disk -> ({product: path}, bytes, tmpdir).

        DOWNLOAD ONLY. The decode is deliberately NOT here, because it is
        what costs memory: the pool below runs three of these at once and a
        decoded frame is 1.05 GB at the real grid.
        """
        tmpdir = None if ctx.source_dir else \
            tempfile.mkdtemp(prefix="pace_", dir=ctx.scratch)
        paths, nbytes = {}, 0
        try:
            for prod in ("bgc", "aop", "moana"):
                rec = have.get(prod)
                if rec is None:
                    continue
                p, n = self.fetch_one(ctx, prod, d, rec, tmpdir)
                if p is None:
                    raise IOError(n)
                paths[prod] = p
                nbytes += n
        except BaseException:
            if tmpdir:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
            raise
        return paths, nbytes, tmpdir

    def read_paths(self, paths, nbytes):
        """The day's files -> (float32 [h, w, 7] with NaN, counts, oob).

        SERIAL, and one variable at a time: the frame is 1.05 GB at the real
        grid and `--stage probe` additionally makes a float64 copy of it
        (2.1 GB), so a second frame in flight is what the kernel kills.
        """
        arr = new_frame(self.h, self.w)
        counts = {"files_read": len(paths), "bytes_files": int(nbytes)}
        if "moana" not in paths:
            counts["moana_not_published"] = 1
        for prod in ("bgc", "aop", "moana"):
            p = paths.get(prod)
            if p is None:
                continue
            with NC_LOCK:
                for name, a, r0, c0, c in self.read_nc(p, prod):
                    put_channel(arr, name, a, counts, r0, c0)
                    del a
                    f10b._merge_counts(counts, c)
        oob = mask_bounds_channelwise(arr, self.bounds())
        counts["valid_pixels"] = int(np.isfinite(arr[:, :, 0]).sum())
        return arr, counts, oob

    def read_day(self, ctx, d, have):
        """One day, fetched and read — `index`'s path and the smoke's."""
        paths, nbytes, tmpdir = self.fetch_day(ctx, d, have)
        try:
            return self.read_paths(paths, nbytes)
        finally:
            if tmpdir:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)

    def read_nc(self, path, prod):
        """One product's file -> an iterator of (variable, array, r0, c0,
        counts), so the caller can place and free each one."""
        return read_nc(path, PRODUCTS[prod][3], self.h, self.w,
                       regional=(prod == "moana"))

    def read_first_day(self, ctx, d):
        """`index`'s one real read: every file opened, the grid CHECKED."""
        have = self.day_files(ctx, d)
        try:
            arr, counts, oob = self.read_day(ctx, d, have)
        except FormatError as e:
            sys.exit(f"REFUSING pace4k: the first day {d} does not match the "
                     f"declared grid: {e}")
        except (IOError, OSError) as e:
            sys.exit(f"REFUSING pace4k: the first day {d} could not be read: "
                     f"{e}")
        out = dict(counts)
        out["products"] = sorted(have)
        out["valid_fraction"] = {
            n: round(float(np.isfinite(arr[:, :, i]).mean()), 6)
            for i, n in enumerate(self.channel_names)}
        out["range"] = {}
        for i, n in enumerate(self.channel_names):
            a = arr[:, :, i]
            g = a[np.isfinite(a)]
            out["range"][n] = [None, None] if not g.size else \
                [round(float(g.min()), 6), round(float(g.max()), 6)]
        if oob:
            out["out_of_bounds"] = oob
        return out

    # ------------------------------------------------------- the contract --
    def classify(self, ctx, d):
        """-> ({product: rec} or None, reason or None) for one day."""
        lo, hi = self.record(ctx)
        if lo is None or d < lo:
            return None, "before_record"
        if d > hi:
            return None, "after_record"
        if d.year not in self.years(ctx):
            return None, "year_not_listed"
        have = self.day_files(ctx, d)
        if "bgc" not in have or "aop" not in have:
            return None, "absent_upstream"
        return have, None

    def fetch_frames(self, ctx, wanted):
        refused = set()
        jobs = []
        for (g, b, f) in wanted:
            d = sh.frame_day(b, f, self.frame_seconds)
            have, why = self.classify(ctx, d)
            if why == "year_not_listed":
                if d.year not in refused:
                    ctx.note_absent(str(d.year),
                                    f"{d.year} is inside the record and not "
                                    f"in the listing")
                    refused.add(d.year)
                continue
            if have is None and why == "absent_upstream":
                em = self.empty_months(ctx, d.year)
                mk = f"{d.year:04d}-{d.month:02d}"
                if mk in em:
                    if mk not in refused:
                        ctx.note_absent(mk, f"the listing of {d.year} has no "
                                            f"day with both required "
                                            f"products in {mk} — an empty "
                                            f"month is refused")
                        refused.add(mk)
                    continue
            jobs.append((g, b, f, d, have, why))

        def work(job):
            """DOWNLOAD only — see `fetch_day`."""
            g, b, f, d, have, why = job
            if have is None:
                return job, None, None
            try:
                return job, self.fetch_day(ctx, d, have), None
            except (IOError, OSError) as e:
                return job, None, (None, f"{type(e).__name__}: {e}")

        workers = 1 if ctx.source_dir else self.WORKERS
        t0 = time.time()
        for job, got, err in cm.ordered_map(work, jobs, workers,
                                            lookahead=workers + 1):
            g, b, f, d, have, why = job
            if have is None:
                yield g, b, f, None, {"frame_missing": why}
                continue
            if err is not None:
                ctx.note_absent(str(d), err[1])
                continue
            paths, nbytes, tmpdir = got
            try:
                arr, counts, oob = self.read_paths(paths, nbytes)
            except FormatError as e:
                sys.exit(f"REFUSING pace4k: {d}: {e}")
            except (IOError, OSError) as e:
                ctx.note_absent(str(d), f"{type(e).__name__}: {e}")
                continue
            finally:
                if tmpdir:
                    import shutil
                    shutil.rmtree(tmpdir, ignore_errors=True)
            if oob:
                counts["out_of_bounds"] = oob
            counts["fetch_seconds"] = round(time.time() - t0, 2)
            t0 = time.time()
            yield g, b, f, arr, counts

    # --------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260920):
        truth = make_smoke_sources(root, d_lo, d_hi, seed)
        self.__init__()
        return truth


# ================================================================ netCDF ===
def parse_day(title, prod):
    """`PACE_OCI.20250701.L3m.DAY.BGC.V3_2.4km.nc` -> the date. REFUSES
    anything that is not this product's daily 4 km granule."""
    t = str(title or "")
    want = f".{PRODUCTS[prod][2]}.{VTAG}.4km.nc"
    if not t.startswith("PACE_OCI.") or not t.endswith(want):
        raise FormatError(f"{title!r}: not a PACE OCI daily 4 km {prod} "
                          f"granule (expected PACE_OCI.<YYYYMMDD>{want})")
    ymd = t.split(".")[1]
    if len(ymd) != 8 or not ymd.isdigit():
        raise FormatError(f"{title!r}: {ymd!r} is not a YYYYMMDD date")
    return dt.date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:]))


def read_nc(path, want, h, w, regional=False):
    """The named variables of one PACE file, ONE AT A TIME.

    Yields `(variable, float32 array, row offset, column offset, counts)` so
    the caller can place each array into the frame and free it: at the real
    grid a global variable is 149 MB and the frame it goes into is 1.05 GB,
    and holding all seven at once was measured being killed by the kernel.
    The source's fill becomes NaN and the producer's `scale_factor` and
    `add_offset` are applied. A REGIONAL file (the MOANA one) comes back at
    the offset its own axes imply, and a window that is not an exact
    sub-block of the global grid is a refusal.
    """
    try:
        import netCDF4
    except ImportError:                                     # pragma: no cover
        sys.exit("pace4k needs the `netCDF4` package (pip install netCDF4) — "
                 "it is in the family1-build workflow's install step")
    try:
        ds = netCDF4.Dataset(path)
    except OSError as ex:
        raise IOError(f"{os.path.basename(path)}: not a readable netCDF file "
                      f"({ex}) — a truncated download") from None
    counts = {"product_version": {
        str(getattr(ds, "processing_version", None)): 1}}
    try:
        lat = np.asarray(ds.variables["lat"][:], np.float64)
        lon = np.asarray(ds.variables["lon"][:], np.float64)
        dy, dx = 180.0 / int(h), 360.0 / int(w)
        if not regional:
            want_lat = Y0 - (np.arange(int(h)) + 0.5) * dy
            want_lon = X0 + (np.arange(int(w)) + 0.5) * dx
            if (lat.size != int(h) or lon.size != int(w)
                    or not (np.allclose(lat, want_lat, atol=1e-4)
                            and np.allclose(lon, want_lon, atol=1e-4))):
                raise FormatError(
                    f"{os.path.basename(path)}: lat/lon differ from the "
                    f"declared grid (lat {lat[0]}..{lat[-1]} n={lat.size}, "
                    f"lon {lon[0]}..{lon[-1]} n={lon.size}); row 0 is "
                    f"declared NORTHERNMOST")
            r0 = c0 = 0
        else:
            r0, c0 = sub_block(lat, lon, h, w)
            counts["moana_window"] = {
                f"row {r0} col {c0} size {lat.size}x{lon.size}": 1}
        for name in want:
            if name not in ds.variables:
                raise FormatError(f"{os.path.basename(path)}: no variable "
                                  f"{name!r} ({sorted(ds.variables)[:10]})")
            v = ds.variables[name]
            if v.dimensions[-2:] != ("lat", "lon"):
                raise FormatError(f"{os.path.basename(path)}: {name} "
                                  f"dimensions {v.dimensions}")
            v.set_auto_maskandscale(False)
            raw = np.array(v[0] if v.dimensions[0] == "time" else v[:])
            at = {k: v.getncattr(k) for k in v.ncattrs()}
            fill = at.get("_FillValue")
            bad = np.zeros(raw.shape, bool) if fill is None else \
                (raw == np.asarray(fill, raw.dtype))
            if name in MOANA_VALID_MAX:
                neg = (raw < 0) & ~bad
                n_neg = int(neg.sum())
                if n_neg:
                    # a cell count is never negative; the real files carry a
                    # second, undocumented sentinel at int32's minimum
                    counts.setdefault("negative_cells", {})[name] = n_neg
                bad |= neg
                vmax = MOANA_VALID_MAX[name]
                over = int(((raw > vmax) & ~bad).sum())
                if over:
                    counts.setdefault("above_producer_valid_max",
                                      {})[name] = over
            a = raw.astype(np.float32)
            sf = at.get("scale_factor")
            off = at.get("add_offset")
            if sf is not None:
                a = a * np.float32(sf)
            if off is not None:
                a = a + np.float32(off)
            a[bad] = np.nan
            a[~np.isfinite(a)] = np.nan
            del bad, raw
            if not regional and a.shape != (int(h), int(w)):
                raise FormatError(f"{os.path.basename(path)}: {name} is "
                                  f"{a.shape}, the declared grid is "
                                  f"{(int(h), int(w))}")
            # the file-level counts ride out with the FIRST variable and the
            # rest carry only their own, so nothing is counted twice
            c, counts = counts, {}
            yield name, a, r0, c0, c
            del a
    finally:
        ds.close()


def write_nc(path, h, w, fields, r0=0, c0=0, attrs=None):
    """One synthetic PACE file: the real dimensions, axes, fills and scalings.

    `r0`/`c0` place a REGIONAL file inside the global grid, which is what the
    MOANA product does.
    """
    import netCDF4
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    nh, nw = next(iter(fields.values())).shape
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    try:
        ds.setncattr("title", "OCI Level-3 Standard Mapped Image (SMOKE)")
        ds.setncattr("processing_version", VERSION)
        ds.setncattr("license", "https://science.nasa.gov/earth-science/"
                                "earth-science-data/data-information-policy/")
        ds.createDimension("lat", int(nh))
        ds.createDimension("lon", int(nw))
        dy, dx = 180.0 / int(h), 360.0 / int(w)
        la = ds.createVariable("lat", "f4", ("lat",))
        la.units = "degrees_north"
        la[:] = Y0 - (np.arange(int(nh)) + int(r0) + 0.5) * dy
        lo = ds.createVariable("lon", "f4", ("lon",))
        lo.units = "degrees_east"
        lo[:] = X0 + (np.arange(int(nw)) + int(c0) + 0.5) * dx
        for name, a in fields.items():
            kind = {np.dtype("float32"): "f4", np.dtype("int16"): "i2",
                    np.dtype("int32"): "i4"}[a.dtype]
            fill = np.float32(-32767.0) if kind == "f4" else \
                a.dtype.type(-32767)
            v = ds.createVariable(name, kind, ("lat", "lon"), zlib=True,
                                  complevel=1, fill_value=fill)
            v.set_auto_maskandscale(False)
            for k, val in ((attrs or {}).get(name) or {}).items():
                setattr(v, k, val)
            v[:, :] = a
    finally:
        ds.close()


# ================================================================== smoke ==
SMOKE_W, SMOKE_H = 480, 240
# the MOANA window as a fraction of the smoke grid, in whole cells, so the
# sub-block arithmetic is exercised rather than trivial
SMOKE_MOANA = (40, 120, 120, 240)          # row0, rows, col0, cols
SMOKE_ABSENT = dt.date(2024, 3, 10)        # listed by AOP, not by BGC
SMOKE_NO_MOANA = dt.date(2024, 3, 12)      # BGC and AOP, no MOANA file
SMOKE_LOW = dt.date(2024, 3, 8)            # one chlor_a of 1e-6 mg m-3
SMOKE_NEG = dt.date(2024, 3, 9)            # one cell count at int32's minimum
SMOKE_OVER = dt.date(2024, 3, 9)           # one picoeuk above its valid_max
POC_SCALE, POC_OFFSET = 0.2, 6400.0


def smoke_fields(d, h, w):
    """A deterministic day: a filled 'ocean' swath and a NaN 'continent'."""
    k = (d - dt.date(2024, 1, 1)).days
    yy = (np.arange(h, dtype=np.float64)[:, None] - h / 2) / h
    xx = (np.arange(w, dtype=np.float64)[None, :] - w / 2) / w
    r = np.hypot(yy, xx)
    chl = 10.0 ** (-1.0 + 1.6 * np.sin(6 * xx + k * 0.3) * np.cos(5 * yy))
    poc_phys = np.clip(20.0 + 900.0 * chl ** 0.5, 0.0, 2000.0)
    carbon = np.clip(5.0 + 60.0 * chl ** 0.7, 0.0, 1000.0)
    avw = 440.0 + 120.0 * (0.5 + 0.5 * np.sin(4 * xx - 3 * yy + k * 0.1))
    off = (r > 0.36) | (np.abs(yy + 0.12) < 0.04)      # land / cloud / night
    chl = chl.astype(np.float32)
    chl[off] = np.float32(-32767.0)
    if d == SMOKE_LOW:
        chl[h // 2, w // 2 + 1] = np.float32(1e-6)
    poc_raw = np.rint((poc_phys - POC_OFFSET) / POC_SCALE).astype(np.int16)
    poc_raw[off] = np.int16(-32767)
    carbon = carbon.astype(np.float32)
    carbon[off] = np.float32(-32767.0)
    avw = avw.astype(np.float32)
    avw[off] = np.float32(-32767.0)
    return ({"chlor_a": chl, "poc": poc_raw, "carbon_phyto": carbon},
            {"avw": avw}, off)


def smoke_moana(d, off, h, w):
    """The three regional cell-abundance fields, on the smoke's sub-block."""
    r0, nr, c0, nc = SMOKE_MOANA
    k = (d - dt.date(2024, 1, 1)).days
    yy = (np.arange(nr, dtype=np.float64)[:, None] - nr / 2) / nr
    xx = (np.arange(nc, dtype=np.float64)[None, :] - nc / 2) / nc
    base = 120000.0 * (0.5 + 0.5 * np.sin(5 * xx + 3 * yy + k * 0.2))
    out = {}
    for name, mult in (("prococcus_moana", 1.0), ("syncoccus_moana", 0.4),
                       ("picoeuk_moana", 0.1)):
        a = np.rint(base * mult).astype(np.int32)
        a[off[r0:r0 + nr, c0:c0 + nc]] = np.int32(-32767)
        out[name] = a
    if d == SMOKE_NEG:
        out["prococcus_moana"][nr // 2, nc // 2] = np.iinfo(np.int32).min
    if d == SMOKE_OVER:
        out["picoeuk_moana"][nr // 2, nc // 2 + 1] = 90000
    return out


def cmr_json(prod, days):
    entries = []
    for i, d in enumerate(sorted(days)):
        title = object_name(prod, d)
        entries.append({
            "producer_granule_id": title, "title": title,
            "id": f"G{3000 + i}-TEST", "granule_size": "36.0",
            "time_start": f"{d}T00:00:00.000Z",
            "time_end": f"{d}T23:59:59.000Z",
            "links": [{"rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                       "href": GETFILE + title}]})
    return {"feed": {"updated": "2026-09-20T00:00:00.000Z", "id": ms.CMR,
                     "title": "ECHO granule metadata", "entry": entries}}


def make_smoke_sources(root, d_lo, d_hi, seed=20260920, skip=()):
    """The archive in its real layout — `cmr_<product>_<year>.json` in CMR's
    own shape and netCDF-4 files under `opendap/<L3SMI|L4SMI>/<YYYY>/<MMDD>/`
    — and the truth for every frame.

    SMOKE_ABSENT is listed by AOP and NOT by BGC, so the store's record has a
    one-day hole inside it and the frame is `absent_upstream`; SMOKE_NO_MOANA
    has both required products and no MOANA file, which is a REAL frame whose
    last three channels are NaN. Returns {(group, bin, frame): (float16
    [h, w, 7] or None, reason or None)}.
    """
    os.environ["PACE4K_SMOKE_GRID"] = f"{SMOKE_W},{SMOKE_H}"
    h, w = SMOKE_H, SMOKE_W
    r0, nr, c0, nc = SMOKE_MOANA
    days = [d_lo + dt.timedelta(days=k)
            for k in range((d_hi - d_lo).days + 1)]
    days = [d for d in days if d not in skip]
    base = os.path.join(root, "pace4k")
    listed = {p: [] for p in PRODUCTS}
    fields = {}
    for d in days:
        bgc, aop, off = smoke_fields(d, h, w)
        moana = smoke_moana(d, off, h, w)
        fields[d] = (bgc, aop, moana)
        for prod, f in (("bgc", bgc), ("aop", aop), ("moana", moana)):
            if prod == "bgc" and d == SMOKE_ABSENT:
                continue
            if prod == "moana" and d in (SMOKE_NO_MOANA, SMOKE_ABSENT):
                continue
            listed[prod].append(d)
            p = os.path.join(base, "opendap", PRODUCTS[prod][1], f"{d:%Y}",
                             f"{d:%m%d}", object_name(prod, d))
            attrs = ({"poc": {"scale_factor": np.float32(POC_SCALE),
                              "add_offset": np.float32(POC_OFFSET)}}
                     if prod == "bgc" else None)
            if prod == "moana":
                write_nc(p, h, w, f, r0=r0, c0=c0)
            else:
                write_nc(p, h, w, f, attrs=attrs)
    for prod in PRODUCTS:
        for y in sorted({d.year for d in listed[prod]}):
            with open(os.path.join(base, f"cmr_{prod}_{y:04d}.json"),
                      "w") as fh:
                json.dump(cmr_json(prod, [d for d in listed[prod]
                                          if d.year == y]), fh)

    ad = Pace4kAdapter()
    ad.h, ad.w = h, w
    lo = min(d for d in listed["bgc"])
    hi = max(d for d in listed["bgc"])
    t_lo = f10b.seconds_since_epoch(lo)
    t_hi = f10b.seconds_since_epoch(hi) + 86399
    truth = {}
    for b in sh.bins_overlapping(t_lo, t_hi):
        for f in range(ad.frames_per_bin):
            d = sh.frame_day(b, f, ad.frame_seconds)
            if d < lo:
                truth[(ad.store, b, f)] = (None, "before_record")
            elif d > hi:
                truth[(ad.store, b, f)] = (None, "after_record")
            elif d not in fields or d == SMOKE_ABSENT:
                truth[(ad.store, b, f)] = (None, "absent_upstream")
            else:
                bgc, aop, moana = fields[d]
                got = {}
                for name, a in bgc.items():
                    got[name] = _unfill(a, name)
                for name, a in aop.items():
                    got[name] = _unfill(a, name)
                if d != SMOKE_NO_MOANA:
                    for name, a in moana.items():
                        full = np.full((h, w), np.nan, np.float32)
                        full[r0:r0 + nr, c0:c0 + nc] = _unfill(a, name)
                        got[name] = full
                arr, _c, _o = to_frame(got, bounds=ad.bounds(), h=h, w=w)
                truth[(ad.store, b, f)] = (arr.astype(np.float16), None)
    return truth


def _unfill(a, name):
    """A synthetic raw array -> float32 with the fill (and, for a cell count,
    any negative) turned into NaN and the producer's scaling applied."""
    raw = np.asarray(a)
    bad = raw == raw.dtype.type(-32767)
    if name in MOANA_VALID_MAX:
        bad = bad | (raw < 0)
    v = raw.astype(np.float32)
    if name == "poc":
        v = v * np.float32(POC_SCALE) + np.float32(POC_OFFSET)
    v[bad] = np.nan
    return v


ADAPTER = Pace4kAdapter
