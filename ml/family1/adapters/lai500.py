"""MOD15A2H v061 and its continuations — leaf area and the light the canopy
absorbs, every eight days at 500 m (family 1.0.tf, exception E7, E-082 wave 6,
the biosphere wave).

PLAIN ENGLISH. Leaf-area index is how many square metres of leaf stand over a
square metre of ground; FPAR is the fraction of the sunlight useful for
photosynthesis that the canopy absorbs rather than letting through or
reflecting. Together they say how much machinery the vegetation has running
and how hard it is running it, and MODIS has measured both for every 500-metre
patch of land since February 2000. This is the seasonal field the biosphere
wave's forecast at weeks has to get right: `canopy30` says how tall the forest
is once, `pheno500` says when its year turns, and this says what it is doing
now.

EXCEPTION E7 — EIGHT DAYS, NOT FIVE. The family's rule is five days or finer.
This product composites EIGHT, and `family1tf.tex` §2 admits it by name as
exception E7 because no other global record of leaf area exists at this
resolution and cadence. The exception is carried the way `chirps05` carries
its pentads: a `frame_table` in every `tile_grid.json` giving each frame its
period's own first day, last day and length, so nothing downstream can infer
an eight-day composite's span from the five-day slot it sits in.

THE COMPOSITE CALENDAR, MEASURED ON CMR 2026-09-20, not assumed. Composites
begin on days of year 1, 9, 17, ... 361 — **46 a year, every year from 2000 to
2035** — and each covers EIGHT days. The last one of a year therefore RUNS
INTO THE NEXT: `MOD15A2H.A2020361` is stamped 2020-12-26 -> 2021-01-02 and
`MOD15A2H.A2019361` 2019-12-27 -> 2020-01-03, both read off CMR's own
time_start/time_end, so the year's last composite OVERLAPS the next year's
first by two or three days. That is why the listing is asked for by the
product's own A-date (`MOD15A2H.A2020*`) and never by a calendar window: a
`temporal` filter on 2020 returns **13,466** granules, 274 of which belong to
2019's last composite, against the **13,192** the A-date pattern returns.

WHERE A COMPOSITE IS FILED, AND WHY F = 1 IS ENOUGH. Each composite goes in
the five-day bin holding its MIDPOINT — the note's rule. `chirps05` needed
F = 2 half-bins because fourteen five-day bins of its record hold two pentad
midpoints each; this product does not, and that is measured rather than
argued: over 2000..2035 there are **1,656 composites and 1,656 distinct bins**,
with a minimum gap between consecutive midpoints of exactly **5.0 days** (at
the non-leap year boundary, where the last composite's midpoint is 31 December
and the next year's first is 5 January). Two midpoints five days apart cannot
share a five-day bin, so F = 1 and `frame_seconds` = 432,000 carry the whole
calendar with nothing dropped. `slot_map` recomputes that and REFUSES a
collision anyway — the check `chirps05` wrote, kept because the margin is
exactly one bin and a changed calendar must stop the build rather than lose a
composite.

A bin with no composite midpoint in it is counted `no_frame_in_bin` and writes
NO SHARD: 73 bins a year hold 46 composites, so 27 bins a year are in that
state BY CONSTRUCTION. It is deliberately not `absent_upstream`, which is a
tile the product should have published for a composite and did not.

SOURCE, VERIFIED 2026-09-20 FROM THIS SANDBOX (CMR is keyless; the granules
are behind Earthdata Login and are read on a hosted runner):
  MOD15A2H v061  C2218777082-LPCLOUD  Terra, 2000-02-18 ->   (HDF4, .hdf)
  MYD15A2H v061  C2565794061-LPCLOUD  Aqua,  2002-07-04 ->   (HDF4, .hdf)
  VNP15A2H v002  C2545314545-LPCLOUD  VIIRS, 2012-01-01 ->   (HDF5, .h5)
The three are CONTINUATION GROUPS in one store, the way `lst05` carries Terra
and Aqua: separate groups on the identical grid, so a reader asks for the
instrument it wants. `LAI500_GROUPS=terra,aqua` builds both; the default is
Terra alone, because each is a store's worth of bytes. **VIIRS' granules are
HDF5, not HDF4** (measured: its CMR data link ends `.h5`), and its datasets
drop MODIS' `_500m` suffix (`Lai`, `Fpar`, `LaiStdDev`) — both differences are
in the per-sensor tables below rather than in a comment.

THE LISTING, MEASURED 2026-09-20 for MOD15A2H v061 in 2020: 46 composites,
**13,192 granules**, 274 to 290 tiles a composite and **290 in the union** —
the polar tiles drop out of the winter composites, which is why the count
moves with the season. The union is the same 290 in every composite of 2020
through 2024 that was checked (DOY 1, 185 and 361 of each year), and it is the
tuple `TILES` below; `index` REFUSES a listing that names a tile outside it,
because the store's GROUPS are that tuple. CMR's declared sizes for 2020 run
0.15 to 16.3 MB a granule (median 1.60) and sum to **46.0 GB** for the year.

WHAT IS STORED: C = 4, dtype FLOAT16, one group per sensor per tile.

  lai            Lai_500m         raw uint8 x 0.1  -> 0..10 m2 of leaf per m2
  fpar           Fpar_500m        raw uint8 x 0.01 -> 0..1, a FRACTION
  lai_stddev     LaiStdDev_500m   raw uint8 x 0.1  -> 0..10, same units as lai
  fpar_lai_qc    FparLai_QC       the producer's own uint8 bit field, 0..254

`fpar` IS A FRACTION AND THE PRODUCER'S OWN METADATA CONTRADICTS ITSELF ABOUT
IT. `Fpar_500m` carries `units = "Percent"` and `scale_factor = 0.01`, and the
file specification's conversion is `value = scale_factor * file data`, so the
scaled number runs 0..1 and is a fraction, not a percentage. This store keeps
the producer's scaling and names the unit what the number actually is.

WHY THE QUALITY BITS ARE A CHANNEL AND NOT A MASK (`family1tf.tex` §5): a
consumer, not the store, decides the threshold, so `FparLai_QC` is carried as
it comes — whole numbers 0..254, which float16 represents EXACTLY (integers
are exact in float16 up to 2,048). Its bits, from the producer's file
specification: 0 the main-algorithm flag, 1 the sensor (Terra or Aqua), 2 the
dead-detector flag, 3-4 the cloud state, 5-7 a five-level confidence score
whose top value means "pixel not produced at all".

THE FILL RULES, TAKEN FROM THE PRODUCER'S OWN LAYER TABLE (LP DAAC, read
2026-09-20) AND ITS FILE SPECIFICATION (LAADS, MODIS/61/MOD15A2H, read the
same day). Every one of the four datasets is uint8 with `valid_range`
**0..100** for the three physical fields and **0..254** for the quality field.
Above the physical range the product does not put a number, it puts a CLASS:
249 is its `_FillValue` on LP DAAC's table, and the file specification names
the whole ladder — 255 fill, 254 water, 253 barren or sparse, 252 permanent
snow and ice, 251 wetland, 250 urban, 249 unclassified. None of those is a
leaf-area index, so a raw value outside the producer's valid range becomes NaN
and is COUNTED BY ITS RAW VALUE under `outside_valid_range_codes`, which is
what makes "this pixel is a lake" visible in the probe rather than silently
missing. The channel BOUNDS are then tripwires: a leaf-area index above 15 or
an FPAR above 1.5 cannot come out of a uint8 times its scale factor and would
mean the scaling changed.

SIZE, AND THE GATE. The ledger's estimate is about 60 GB per sensor-year, and
phase A is Terra 2020--2024 — five years. `LAI500_MAX_YEARS` (default **1**)
refuses a wider window at DISPATCH, where the inputs are all it has cost
(ml/CLAUDE.md §0.3), rather than at hour five of a fetch; the phase-A build
raises it deliberately, or runs one year a lane, which is the shape a lane has
anyway. `LAI500_TILES` restricts the groups for a probe and also makes the CMR
listing ask per tile, which takes one probe's listing from 70 MB to a few
hundred kilobytes.

PHASE B: the rest of the Terra record (2000-02-18 -> 2019 and 2025 ->), the
Aqua and VIIRS groups, and `FparStdDev_500m` and `FparExtra_QC`, the two
datasets this store leaves behind.
"""
import datetime as dt
import os
import sys

import numpy as np

import build_family10_stores as f10b
from family1 import sharded as sh
from family1.adapters import _modis_sin as ms

LP_PREFIX_MODIS = ("https://data.lpdaac.earthdatacloud.nasa.gov/"
                   "lp-prod-protected/")
DOY_STEP = 8
COMPOSITE_DAYS = 8
FIRST_DOY = 1
VALID_PHYS = (0, 100)
VALID_QC = (0, 254)
LAI_SCALE = 0.1
FPAR_SCALE = 0.01
PHASE_A_YEARS = (2020, 2024)

# sensor -> (short_name, version, file format, first A-date)
COLLECTIONS = {
    "terra": ("MOD15A2H", "061", "hdf4", dt.date(2000, 2, 18)),
    "aqua": ("MYD15A2H", "061", "hdf4", dt.date(2002, 7, 4)),
    "viirs": ("VNP15A2H", "002", "hdf5", dt.date(2012, 1, 1)),
}
# sensor -> the dataset names in ITS files, in channel order. MODIS keeps the
# `_500m` suffix and VIIRS drops it (LP DAAC's own layer tables, read
# 2026-09-20).
DATASETS = {
    "terra": ("Lai_500m", "Fpar_500m", "LaiStdDev_500m", "FparLai_QC"),
    "aqua": ("Lai_500m", "Fpar_500m", "LaiStdDev_500m", "FparLai_QC"),
    "viirs": ("Lai", "Fpar", "LaiStdDev", "FparLai_QC"),
}
# channel -> (unit, bound lo, bound hi, source scale, source valid range)
CHANNELS = (
    ("lai", "m2 of leaf per m2 of ground", 0.0, 15.0, LAI_SCALE, VALID_PHYS),
    ("fpar", "fraction (0-1) of photosynthetically active radiation absorbed",
     0.0, 1.5, FPAR_SCALE, VALID_PHYS),
    ("lai_stddev", "m2 of leaf per m2 of ground", 0.0, 15.0, LAI_SCALE,
     VALID_PHYS),
    ("fpar_lai_qc", "bitfield (the producer's own FparLai_QC, 0..254)",
     0.0, 254.0, 1.0, VALID_QC),
)

TILES = (
    "h00v08", "h00v09", "h00v10", "h01v07", "h01v08", "h01v09",
    "h01v10", "h01v11", "h02v06", "h02v08", "h02v09", "h02v10",
    "h02v11", "h03v06", "h03v07", "h03v09", "h03v10", "h03v11",
    "h04v09", "h04v10", "h04v11", "h05v10", "h05v11", "h05v13",
    "h06v03", "h06v11", "h07v03", "h07v05", "h07v06", "h07v07",
    "h08v03", "h08v04", "h08v05", "h08v06", "h08v07", "h08v08",
    "h08v09", "h08v11", "h09v02", "h09v03", "h09v04", "h09v05",
    "h09v06", "h09v07", "h09v08", "h09v09", "h10v02", "h10v03",
    "h10v04", "h10v05", "h10v06", "h10v07", "h10v08", "h10v09",
    "h10v10", "h10v11", "h11v02", "h11v03", "h11v04", "h11v05",
    "h11v06", "h11v07", "h11v08", "h11v09", "h11v10", "h11v11",
    "h11v12", "h12v01", "h12v02", "h12v03", "h12v04", "h12v05",
    "h12v07", "h12v08", "h12v09", "h12v10", "h12v11", "h12v12",
    "h12v13", "h13v01", "h13v02", "h13v03", "h13v04", "h13v08",
    "h13v09", "h13v10", "h13v11", "h13v12", "h13v13", "h13v14",
    "h14v01", "h14v02", "h14v03", "h14v04", "h14v09", "h14v10",
    "h14v11", "h14v14", "h15v01", "h15v02", "h15v03", "h15v05",
    "h15v07", "h15v11", "h15v14", "h16v00", "h16v01", "h16v02",
    "h16v05", "h16v06", "h16v07", "h16v08", "h16v09", "h16v12",
    "h16v14", "h17v00", "h17v01", "h17v02", "h17v03", "h17v04",
    "h17v05", "h17v06", "h17v07", "h17v08", "h17v10", "h17v12",
    "h17v13", "h18v00", "h18v01", "h18v02", "h18v03", "h18v04",
    "h18v05", "h18v06", "h18v07", "h18v08", "h18v09", "h18v14",
    "h19v00", "h19v01", "h19v02", "h19v03", "h19v04", "h19v05",
    "h19v06", "h19v07", "h19v08", "h19v09", "h19v10", "h19v11",
    "h19v12", "h20v01", "h20v02", "h20v03", "h20v04", "h20v05",
    "h20v06", "h20v07", "h20v08", "h20v09", "h20v10", "h20v11",
    "h20v12", "h20v13", "h21v01", "h21v02", "h21v03", "h21v04",
    "h21v05", "h21v06", "h21v07", "h21v08", "h21v09", "h21v10",
    "h21v11", "h21v13", "h22v01", "h22v02", "h22v03", "h22v04",
    "h22v05", "h22v06", "h22v07", "h22v08", "h22v09", "h22v10",
    "h22v11", "h22v13", "h22v14", "h23v01", "h23v02", "h23v03",
    "h23v04", "h23v05", "h23v06", "h23v07", "h23v08", "h23v09",
    "h23v10", "h23v11", "h24v02", "h24v03", "h24v04", "h24v05",
    "h24v06", "h24v07", "h24v12", "h25v02", "h25v03", "h25v04",
    "h25v05", "h25v06", "h25v07", "h25v08", "h25v09", "h26v02",
    "h26v03", "h26v04", "h26v05", "h26v06", "h26v07", "h26v08",
    "h27v03", "h27v04", "h27v05", "h27v06", "h27v07", "h27v08",
    "h27v09", "h27v10", "h27v11", "h27v12", "h27v14", "h28v03",
    "h28v04", "h28v05", "h28v06", "h28v07", "h28v08", "h28v09",
    "h28v10", "h28v11", "h28v12", "h28v13", "h28v14", "h29v03",
    "h29v05", "h29v06", "h29v07", "h29v08", "h29v09", "h29v10",
    "h29v11", "h29v12", "h29v13", "h30v05", "h30v06", "h30v07",
    "h30v08", "h30v09", "h30v10", "h30v11", "h30v12", "h30v13",
    "h31v06", "h31v07", "h31v08", "h31v09", "h31v10", "h31v11",
    "h31v12", "h31v13", "h32v07", "h32v08", "h32v09", "h32v10",
    "h32v11", "h32v12", "h33v07", "h33v08", "h33v09", "h33v10",
    "h33v11", "h34v07", "h34v08", "h34v09", "h34v10", "h35v08",
    "h35v09", "h35v10",
)


# ========================================================= the calendar ====
def composite_doys(year):
    """The days of year an eight-day composite BEGINS on: 1, 9, ... 361."""
    out = []
    doy = FIRST_DOY
    while True:
        try:
            d = dt.date(int(year), 1, 1) + dt.timedelta(days=doy - 1)
        except (ValueError, OverflowError):
            break
        if d.year != int(year):
            break
        out.append(doy)
        doy += DOY_STEP
    return out


def composite_days(key):
    """(first day, last day) of composite `key` = (year, day of year).

    EIGHT days always, including the last of the year, which is why
    `A2020361` ends on 2021-01-02 and overlaps the next year's first
    composite. Measured on CMR's own time_start/time_end.
    """
    y, doy = int(key[0]), int(key[1])
    d0 = dt.date(y, 1, 1) + dt.timedelta(days=doy - 1)
    return d0, d0 + dt.timedelta(days=COMPOSITE_DAYS - 1)


def composite_slot(key, frame_seconds):
    """(bin, frame) of composite `key`, by its MIDPOINT — the note's rule."""
    d0, d1 = composite_days(key)
    s = f10b.seconds_since_epoch(d0)
    e = f10b.seconds_since_epoch(d1) + 86400
    mid = (s + e) // 2
    b = mid // sh.BIN_SECONDS
    return int(b), int((mid - b * sh.BIN_SECONDS) // int(frame_seconds))


def key_of(d):
    """The composite key whose A-date is `d`. REFUSES a date off the grid."""
    doy = (d - dt.date(d.year, 1, 1)).days + 1
    if (doy - FIRST_DOY) % DOY_STEP:
        raise ms.FormatError(
            f"{d}: day of year {doy} is not the first day of an eight-day "
            f"composite (they begin on {FIRST_DOY}, {FIRST_DOY + DOY_STEP}, "
            f"...) — a granule with an A-date off the product's own calendar "
            f"is a refusal, never a frame filed at a guess")
    return (d.year, doy)


def frame_table(frame_seconds, y_lo=2000, y_hi=2035):
    """Every (bin, frame) -> its composite's own dates, compactly.

    A list of [bin, frame, year, day of year, first day, last day, days],
    sorted, so a reader never has to infer an eight-day span from the
    five-day slot it sits in — the mechanism `chirps05` wrote for its pentads.
    """
    rows = []
    for y in range(int(y_lo), int(y_hi) + 1):
        for doy in composite_doys(y):
            b, f = composite_slot((y, doy), frame_seconds)
            d0, d1 = composite_days((y, doy))
            rows.append([int(b), int(f), int(y), int(doy), str(d0), str(d1),
                         (d1 - d0).days + 1])
    return sorted(rows)


def slot_collisions(frame_seconds, y_lo=2000, y_hi=2035):
    """{(bin, frame): [keys]} for every slot holding more than one composite.

    Empty is the measurement F = 1 rests on, and `slot_map` re-runs it on the
    window the build actually covers.
    """
    seen = {}
    for y in range(int(y_lo), int(y_hi) + 1):
        for doy in composite_doys(y):
            seen.setdefault(composite_slot((y, doy), frame_seconds),
                            []).append((y, doy))
    return {k: v for k, v in seen.items() if len(v) > 1}


class Lai500Adapter(ms.SinTileAdapter):
    store = "lai500"
    title = ("Leaf-area index and FPAR every eight days, MODIS MOD15A2H v061 "
             "(Terra) with MYD15A2H (Aqua) and VNP15A2H v002 (VIIRS) as "
             "continuation groups, 500 m sinusoidal tiles (exception E7)")
    licence = {
        "name": "NASA open data (no restrictions)",
        "redistribution": "attribution",
        "redistribution_confirmed": True,
        "derived_works": "free",
        "attribution": ("Myneni, R., Y. Knyazikhin, T. Park (2021). MOD15A2H "
                        "MODIS/Terra Leaf Area Index/FPAR 8-Day L4 Global "
                        "500m SIN Grid V061. NASA EOSDIS Land Processes "
                        "Distributed Active Archive Center. "
                        "doi:10.5067/MODIS/MOD15A2H.061 (Aqua: "
                        "doi:10.5067/MODIS/MYD15A2H.061; VIIRS: "
                        "doi:10.5067/VIIRS/VNP15A2H.002)"),
        "terms": ("earthdata.nasa.gov/engage/open-data-services-and-software/"
                  "data-information-policy, read 2026-09-18: NASA's Earth "
                  "science data are open, full and without restriction, free "
                  "of charge, with no period of exclusive access; users are "
                  "asked to cite the data set"),
    }
    collections = {k: v[:3] for k, v in COLLECTIONS.items()}
    groups_default = ("terra",)
    groups_env = "LAI500_GROUPS"
    tiles_env = "LAI500_TILES"
    px_env = "LAI500_SMOKE_PX"
    tiles = TILES
    datasets = DATASETS["terra"]
    channels = tuple((n, u, lo, hi) for (n, u, lo, hi, _s, _v) in CHANNELS)
    dtype = "float16"
    first_year = 2000
    frames_per_bin = 1
    frame_seconds = sh.BIN_SECONDS
    log2_fp = ms.log2_fp_for(ms.PX_500)                  # -5.908
    # THE FOOTPRINT IN TIME IS THE COMPOSITE, NOT THE SLOT. The frames are one
    # bin apart in the store, but each is an EIGHT-day composite, and
    # `log2_dt` is the measurement's own support: log2(8 / 5) = +0.678. The
    # `frame_table` carries the real first and last day of every one.
    log2_dt = float(np.log2(COMPOSITE_DAYS / 5.0))       # +0.678
    note_estimate = {
        "bytes": 60e9,
        "what": ("family1tf.tex ledger: '~60 GB per sensor-year' for "
                 "MOD15A2H v061 at 500 m, 46 composites of 290 tiles")}
    qc_policy = (
        "the producer's own FparLai_QC uint8 bit field is CARRIED AS A "
        "CHANNEL, unchanged, so the consumer picks the threshold "
        "(family1tf.tex 5). Its bits are: 0 MODLAND_QC (0 main algorithm, 1 "
        "back-up or fill), 1 the sensor (0 Terra, 1 Aqua), 2 dead detectors, "
        "3-4 cloud state (0 clear, 1 cloudy, 2 mixed, 3 undefined), 5-7 a "
        "five-level confidence score whose value 4 means the pixel was not "
        "produced at all. The three physical fields carry the producer's "
        "valid_range 0..100 and, above it, a CLASS rather than a number: 255 "
        "fill, 254 water, 253 barren or sparse, 252 permanent snow and ice, "
        "251 wetland, 250 urban, 249 unclassified (the product's own file "
        "specification). None of those is a leaf-area index, so a raw value "
        "outside 0..100 becomes NaN and is counted BY ITS RAW VALUE under "
        "`outside_valid_range_codes`, which is what makes 'this pixel is a "
        "lake' visible in the probe instead of merely missing. The channel "
        "bounds are then tripwires: a leaf-area index above 15 or an FPAR "
        "above 1.5 cannot come out of a uint8 times its scale factor and "
        "would mean the scaling changed; a value outside them becomes NaN and "
        "is counted (`out_of_bounds`), never clipped")
    sources = (LP_PREFIX_MODIS + "MOD15A2H.061/<granule>/<granule>.hdf",
               ms.CMR + "?short_name=MOD15A2H&version=061")
    verified = (
        "2026-09-20 from the sandbox, keyless: the CMR collection searches "
        "for MOD15A2H v061 (C2218777082-LPCLOUD, time_start 2000-02-18), "
        "MYD15A2H v061 (C2565794061-LPCLOUD, 2002-07-04) and VNP15A2H v002 "
        "(C2545314545-LPCLOUD, 2012-01-01, data link ending .h5 and not "
        ".hdf); the full MOD15A2H granule listing of 2020 composite by "
        "composite (46 composites, 13,192 granules, 274-290 tiles each, union "
        "290, CMR sizes 0.15-16.3 MB summing to 46.0 GB) and the DOY 1, 185 "
        "and 361 composites of 2021-2024 (the same 290-tile union); the "
        "time_start/time_end of A2019361 (2019-12-27 -> 2020-01-03) and "
        "A2020361 (2020-12-26 -> 2021-01-02), which is how the eight-day "
        "composite was shown to cross the year boundary; the MODLAND "
        "sinusoidal grid parameters on modis-land.gsfc.nasa.gov/GCTP.html "
        "checked against CMR's own corner polygon for h18v04; and the dataset "
        "names, uint8 types, fill values, valid ranges and scale factors on "
        "lpdaac.usgs.gov/products/mod15a2hv061 and vnp15a2hv002 plus the "
        "LAADS file specification MODIS/61/MOD15A2H (which also gives the QC "
        "bit legend and the 249-255 non-terrestrial class ladder). The "
        "granules themselves are behind Earthdata Login and are opened on a "
        "GitHub-hosted runner by the probe, which re-checks the tile's own "
        "StructMetadata corners in every file it reads")
    notes = (
        "Exception E7: an EIGHT-day composite on a five-day axis. 46 "
        "composites a year beginning on days of year 1, 9, ... 361, each of "
        "them eight days long, so the last of a year runs into the next and "
        "the listing is asked for by the product's own A-date and never by a "
        "calendar window (a temporal filter on 2020 returns 13,466 granules "
        "against the A-date pattern's 13,192). Each composite is filed in the "
        "bin holding its MIDPOINT; F = 1 suffices because consecutive "
        "midpoints are at least 5.0 days apart (measured over 1,656 "
        "composites, 2000-2035, zero collisions), and `slot_map` refuses a "
        "collision anyway. tile_grid.json carries a frame_table giving every "
        "(bin, frame) its composite's first day, last day and length. A bin "
        "with no composite midpoint writes NO shard and is counted "
        "`no_frame_in_bin` -- 27 bins in every 73 by construction. FPAR is "
        "stored as a FRACTION 0..1: the producer's units attribute says "
        "'Percent' and its scale_factor is 0.01, which is a contradiction in "
        "the producer's own metadata, and the store keeps the scaling and "
        "names the number what it is. Aqua and VIIRS are separate GROUPS on "
        "the identical grid; LAI500_GROUPS selects them and the default is "
        "Terra alone. VIIRS' files are HDF5 and its datasets drop the `_500m` "
        "suffix.")
    smoke_window = ("2020-12-20", "2021-01-15")
    smoke_probe_month = "2020-12"
    WORKERS = 3

    def __init__(self):
        try:
            self.max_years = int(os.environ.get("LAI500_MAX_YEARS") or 1)
        except ValueError:
            sys.exit("LAI500_MAX_YEARS must be an integer")
        super().__init__()

    # ------------------------------------------------------------ the grid --
    def datasets_of(self, sensor):
        return DATASETS[sensor]

    def datasets_want_of(self, sensor):
        # Nothing is DECLARED as a hard requirement here: `check_attrs` in
        # `frame_from` refuses an attribute the file carries with a different
        # value and merely counts one it does not carry at all, which is the
        # §5.17 rule (only a DEFINITE answer may be fatal).
        return {}

    def specs(self):
        out = super().specs()
        for spec in out.values():
            spec["frame_rule"] = (
                "frame 0 of bin b is the whole five-day bin, and it holds the "
                "eight-day composite whose MIDPOINT falls inside it. A "
                "composite is EIGHT days long, NOT five: read its real span "
                "from `frame_table`")
            spec["composite_rule"] = (
                "composites begin on days of year 1, 9, 17, ... 361 — 46 a "
                "year — and each covers eight days, so the last of a year "
                "ends two or three days into the next (measured on the "
                "producer's own time_start/time_end)")
            spec["exception"] = {
                "name": "E7",
                "what": ("family1tf.tex §2 admits an eight-day composite on a "
                         "five-day axis because no other global record of "
                         "leaf area exists at this resolution and cadence"),
                "log2_dt": self.log2_dt,
                "log2_dt_note": ("log2(8 / 5): the footprint in time is the "
                                 "composite's own support, not the five-day "
                                 "slot it sits in")}
            spec["frame_table_columns"] = ["bin", "frame", "year",
                                           "day_of_year", "first_day",
                                           "last_day", "days"]
            spec["frame_table_span_years"] = [2000, 2035]
            spec["frame_table_note"] = (
                "every (bin, frame) the composite calendar fills between 2000 "
                "and 2035, whether or not this store holds it; store.json's "
                "date_range says which part is filled. A (bin, frame) absent "
                "from this table holds no composite at all, writes no shard "
                f"and is counted `{self.empty_slot_reason}`")
            spec["frame_table"] = frame_table(self.frame_seconds)
            spec["source_scalings"] = {
                n: {"scale_factor": s, "valid_range": list(v)}
                for (n, _u, _lo, _hi, s, v) in CHANNELS}
        return out

    def record_frames(self, ctx, group):
        """Composites from the sensor's first A-date to the last one listed.

        The probe multiplies its measured bytes a frame by this, so it is the
        record SO FAR rather than a forecast of the record's end.
        """
        sensor, _tile = self.group_parts(group)
        first = COLLECTIONS[sensor][3]
        _lo, hi = self.record(ctx, sensor)
        if hi is None:
            return None
        n = 0
        for y in range(first.year, hi.year + 1):
            for doy in composite_doys(y):
                d0 = dt.date(y, 1, 1) + dt.timedelta(days=doy - 1)
                if first <= d0 <= hi:
                    n += 1
        return n

    # ------------------------------------------------- the product calendar --
    def periods(self, ctx, sensor, year):
        first = COLLECTIONS[sensor][3]
        out = {}
        for doy in composite_doys(year):
            key = (int(year), int(doy))
            d0, _d1 = composite_days(key)
            if d0 < first:
                continue
            out[composite_slot(key, self.frame_seconds)] = key
        return out

    def granule_date(self, key):
        return composite_days(key)[0]

    def key_of_date(self, d):
        return key_of(d)

    def period_days(self, key):
        return composite_days(key)

    # -------------------------------------------------------------- preflight
    def fetch_preflight(self, ctx):
        super().fetch_preflight(ctx)
        # THE WINDOW'S OWN CALENDAR YEARS, not `ctx.years`: a tier-G "year" is
        # the bins whose FIRST day falls in it, so a window of 2020 alone can
        # touch 2019 as well and counting those would refuse the very build
        # the brief asks for.
        years = int(ctx.d_hi.year) - int(ctx.d_lo.year) + 1
        if self.max_years and years > self.max_years and \
                getattr(ctx.a, "stage", "") != "probe":
            sys.exit(
                f"REFUSING lai500: the window covers {years} year(s) and "
                f"LAI500_MAX_YEARS is {self.max_years}. This store is about "
                f"60 GB per sensor-year (family1tf.tex), phase A is Terra "
                f"{PHASE_A_YEARS[0]}-{PHASE_A_YEARS[1]}, and a lane is one "
                f"year. Raise LAI500_MAX_YEARS deliberately (adapter_env) "
                f"once the probe's projection has been read, or dispatch one "
                f"year a lane. Nothing has been fetched.")
        return None

    # --------------------------------------------------------------- index --
    def index(self, ctx):
        years = self.listing_years(ctx)
        out = {"dataset": "MODIS/VIIRS eight-day leaf-area index and FPAR, "
                          "500 m sinusoidal tiles",
               "url": ms.CMR, "groups": {}, "granules": 0,
               "frames_per_bin": self.frames_per_bin,
               "frame_seconds": self.frame_seconds,
               "exception": "E7 (eight-day composites on a five-day axis)",
               "years_listed": years,
               "tiles_declared": len(self.tiles),
               "tiles_built": len(self.tile_names),
               "grid": ms.sin_grid(*ms.tile_hv(self.tile_names[0]),
                                   px=self.px)}
        for sensor in self.groups:
            sn, ver, fmt = self.collection(sensor)
            per_year, counts_all, live = {}, {}, set()
            files_all, composites = {}, {}
            for y in years:
                files, counts = self.listing(ctx, sensor, y)
                if not files:
                    continue
                per_year[str(y)] = len(files)
                counts_all[str(y)] = counts
                live |= {ms.tile_name(*hv) for (_d, hv) in files}
                files_all.update(files)
                for (d, hv) in files:
                    composites.setdefault(str(key_of(d)), set()).add(hv)
            if not files_all:
                sys.exit(f"REFUSING lai500: CMR lists no granule of {sn} "
                         f"v{ver} for any of the years {years} — an empty "
                         f"listing is a refusal (ml/CLAUDE.md, the "
                         f"2026-09-14 rule)")
            want = set(self.tiles)
            extra = sorted(live - want)
            if extra:
                sys.exit(f"REFUSING lai500: the {sensor} listing names "
                         f"{len(extra)} tile(s) this adapter's TILES does "
                         f"not: {extra[:8]}. The store's GROUPS are that "
                         f"tuple, so a changed tile set must be re-measured "
                         f"and committed rather than silently altering the "
                         f"store.")
            lo, hi = self.record(ctx, sensor)
            d_lo, d_hi = self.record_span(ctx, sensor)
            sizes = [len(v) for v in composites.values()]
            g = {"short_name": sn, "version": ver, "file_format": fmt,
                 "granules": len(files_all),
                 "granules_per_year": per_year,
                 "composites_listed": len(composites),
                 "tiles_per_composite": {"min": min(sizes), "max": max(sizes)},
                 "tiles_in_listing": len(live),
                 "tiles_declared_and_absent": sorted(want - live)[:40],
                 "record_a_dates": [str(lo), str(hi)],
                 "record_days": [str(d_lo), str(d_hi)],
                 "collection_first_a_date": str(COLLECTIONS[sensor][3]),
                 "bytes_cmr": int(sum(r["bytes_cmr"]
                                      for r in files_all.values())),
                 "bytes_note": ("CMR's own `granule_size` in MB, summed; the "
                                "probe measures exact bytes"),
                 "listing": counts_all}
            mine = {k: v for k, v in files_all.items()
                    if ms.tile_name(*k[1]) in set(self.tile_names)}
            g["first_file"] = self.read_first_file(ctx, sensor,
                                                   mine or files_all)
            out["groups"][sensor] = g
            out["granules"] += len(files_all)
        # the collision check the F = 1 choice rests on, re-run on the window
        out["slot_collisions"] = {str(k): [str(x) for x in v] for k, v in
                                  slot_collisions(self.frame_seconds,
                                                  min(years),
                                                  max(years) + 1).items()}
        if out["slot_collisions"]:
            sys.exit(f"REFUSING lai500: two composites share a slot: "
                     f"{out['slot_collisions']} — frames_per_bin = "
                     f"{self.frames_per_bin} cannot carry this calendar")
        return out

    # ---------------------------------------------------------- the frame --
    def frame_from(self, arrs, attrs, rec=None):
        """{dataset: raw uint8} -> (float32 [px, px, 4] with NaN, counts)."""
        px = self.px
        out = np.empty((px, px, len(CHANNELS)), np.float32)
        counts = {}
        absent = ms.check_attrs(attrs, {}, self.store)
        if absent:
            counts["attributes_absent"] = absent
        names = self._names_in(arrs)
        for i, (ch, _u, _lo, _hi, scale, valid) in enumerate(CHANNELS):
            raw = arrs[names[i]]
            if raw.dtype != np.uint8:
                raise ms.FormatError(f"{names[i]}: {raw.dtype}, expected "
                                     f"uint8")
            bad = (raw < valid[0]) | (raw > valid[1])
            n_bad = int(bad.sum())
            v = raw.astype(np.float32) * np.float32(scale)
            v[bad] = np.nan
            out[:, :, i] = v
            if n_bad:
                counts.setdefault("outside_valid_range", {})[ch] = n_bad
                # FLAT keys "<channel>:<raw code>", because the framework's
                # counter merges one level of dict and a nested one would add
                # an int to a dict.
                u, n = np.unique(raw[bad], return_counts=True)
                codes = counts.setdefault("outside_valid_range_codes", {})
                for k, x in zip(u[:12], n[:12]):
                    codes[f"{ch}:{int(k)}"] = int(x)
        if rec is not None:
            counts["composite"] = {str(key_of(ms.granule_key(
                rec["title"])[0])): 1}
        return out, counts

    def _names_in(self, arrs):
        """Which sensor's dataset names this granule used, in channel order."""
        for sensor, names in DATASETS.items():
            if all(n in arrs for n in names):
                return names
        raise ms.FormatError(
            f"lai500: the granule carries {sorted(arrs)}, which is none of "
            f"the sensors' dataset name sets "
            f"({ {s: list(n) for s, n in DATASETS.items()} })")

    # --------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260920):
        truth = make_smoke_sources(root, d_lo, d_hi, seed)
        self.smoke_reinit()
        return truth


# ================================================================== smoke ==
SMOKE_PX = 240                        # a tenth of the real 2,400
SMOKE_TILES = ("h18v04", "h12v09")
SMOKE_SENSOR = "terra"
# THE YEAR BOUNDARY IS THE POINT OF THIS WINDOW. A2020361 covers 2020-12-26 ..
# 2021-01-02 and A2021001 covers 2021-01-01 .. 2021-01-08, so the two
# OVERLAP by two days and their midpoints are six days apart — the tightest
# the calendar gets in a leap year, and the case the frame_table has to place
# correctly.
SMOKE_COMPOSITES = ((2020, 353), (2020, 361), (2021, 1), (2021, 9))
SMOKE_ABSENT = ((2021, 1), "h12v09")   # listed for one tile and not the other
SMOKE_CODES = ((2020, 361), "h18v04")  # a patch of 254 (water) and 255 (fill)


def smoke_fields(tile, key, px):
    """A deterministic composite: a leaf-area field, a water class, a fill
    patch and a quality field."""
    h, v = ms.tile_hv(tile)
    y, doy = key
    k = (h * 5 + v * 11 + doy) % 19
    yy = (np.arange(px, dtype=np.float64)[:, None] - px / 2) / px
    xx = (np.arange(px, dtype=np.float64)[None, :] - px / 2) / px
    r = np.hypot(yy, xx)
    lai = np.clip(60.0 - 130.0 * r + 12.0 * np.sin(9 * xx + k)
                  * np.cos(7 * yy - k), 0.0, 100.0)
    lai_raw = np.rint(lai).astype(np.uint8)
    fpar_raw = np.rint(np.clip(lai * 0.9, 0.0, 100.0)).astype(np.uint8)
    sd_raw = np.rint(np.clip(lai * 0.1, 0.0, 100.0)).astype(np.uint8)
    qc = np.where(lai_raw > 0, np.uint8(0), np.uint8(32)).astype(np.uint8)
    # the non-terrestrial classes the product really uses
    water = r > 0.40
    for a in (lai_raw, fpar_raw, sd_raw):
        a[water] = 254                     # "perennial salt or fresh water"
    qc[water] = 255                        # the QC field's own fill
    if (key, tile) == SMOKE_CODES:
        lai_raw[px // 2, px // 2] = 255    # _FillValue
        lai_raw[px // 2, px // 2 + 1] = 250  # urban
    return {"Lai_500m": lai_raw, "Fpar_500m": fpar_raw,
            "LaiStdDev_500m": sd_raw, "FparLai_QC": qc}


def make_smoke_sources(root, d_lo, d_hi, seed=20260920, skip=(), absent=()):
    """The archive in its real layout — `cmr_terra_<year>.json` in CMR's own
    shape and HDF-EOS2 granules with a real `StructMetadata.0` — and the truth
    for every frame.

    Returns {(group, bin, frame): (float16 [px, px, 4] or None, reason or
    None)} for every frame of every bin overlapping the record.
    """
    os.environ["LAI500_SMOKE_PX"] = str(SMOKE_PX)
    os.environ["LAI500_TILES"] = ",".join(SMOKE_TILES)
    os.environ["LAI500_GROUPS"] = SMOKE_SENSOR
    # The smoke's window STRADDLES NEW YEAR on purpose — that is the case the
    # frame_table has to place correctly — so it raises the size gate the way
    # a real two-year lane would have to. `test_family1_lai500.py` asserts the
    # default is 1 and that a two-year window refuses without it.
    os.environ["LAI500_MAX_YEARS"] = "2"
    px = SMOKE_PX
    ad = Lai500Adapter()
    sn, ver, _fmt, _first = COLLECTIONS[SMOKE_SENSOR]
    prefix = LP_PREFIX_MODIS + f"{sn}.{ver}/"
    fields, titles = {}, {}
    by_year = {}
    for key in SMOKE_COMPOSITES:
        by_year.setdefault(key[0], []).append(key)
    for y, keys in sorted(by_year.items()):
        recs, arrays = [], {}
        for key in keys:
            for t in SMOKE_TILES:
                if (key, t) in skip or (key, t) == SMOKE_ABSENT:
                    continue
                h, v = ms.tile_hv(t)
                title = (f"{sn}.A{key[0]:04d}{key[1]:03d}.{t}.{ver}."
                         f"{2026000000000 + key[1] * 100 + h}")
                recs.append((title, (h, v)))
                arrays[title] = smoke_fields(t, key, px)
                fields[(key, t)] = arrays[title]
                titles[(key, t)] = title
        ms.write_smoke_archive(root, "lai500", SMOKE_SENSOR, y, recs, arrays,
                               px, prefix, attrs=None, absent=absent)

    truth = {}
    lo_key, hi_key = min(SMOKE_COMPOSITES), max(SMOKE_COMPOSITES)
    day_lo = composite_days(lo_key)[0]
    day_hi = composite_days(hi_key)[1]
    t_lo = f10b.seconds_since_epoch(d_lo)
    t_hi = f10b.seconds_since_epoch(d_hi) + 86399
    slots = {composite_slot(k, ad.frame_seconds): k for k in SMOKE_COMPOSITES}
    for b in sh.bins_overlapping(t_lo, t_hi):
        key = slots.get((b, 0))
        t0 = sh.frame_start_seconds(b, 0, ad.frame_seconds)
        for t in SMOKE_TILES:
            g = f"{SMOKE_SENSOR}_{t}"
            if key is None:
                why = ("before_record"
                       if t0 < f10b.seconds_since_epoch(day_lo)
                       else "after_record"
                       if t0 > f10b.seconds_since_epoch(day_hi) + 86399
                       else sh.FRAME_NO_FRAME_IN_BIN)
                truth[(g, b, 0)] = (None, why)
                continue
            if (key, t) not in fields:
                truth[(g, b, 0)] = (None, "absent_upstream")
                continue
            a, _c = ad.frame_from(fields[(key, t)], {},
                                  {"title": titles[(key, t)]})
            ms.mc.mask_bounds_low_memory(ad, a)
            truth[(g, b, 0)] = (a.astype(np.float16), None)
    return truth


ADAPTER = Lai500Adapter
