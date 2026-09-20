"""MCD12Q2 v061 — the leaf calendar, one map a year at 500 m
(family 1.0.tf, E-082 wave 6, the biosphere wave).

PLAIN ENGLISH. A forest's year has dates in it: the day the leaves come out,
the day the canopy is at its greenest, the day it starts to die back, the day
it is bare. MODIS has watched every 500-metre patch of land since 2001 and
NASA's land-cover-dynamics product turns that watching into those dates — one
map a year, one number per date per pixel. That CALENDAR is what this store
holds, and it is the second of the biosphere wave's three annual targets
(`canopy30` is how tall the canopy is, `lossyear` where it was cleared, and
this is when it leafs out and when it dies back).

HOW THE DATES ARE FOUND, in the producer's own words (the MCD12Q2 v061 user
guide, lpdaac.usgs.gov/documents/1417, read 2026-09-20): a two-band vegetation
index (NBAR-EVI2) is smoothed with a penalised cubic spline through the year,
the spline's peaks are found, and each peak's green-up segment is walked to
find the dates on which the index first crosses 15 %, 50 % and 90 % of the
segment's amplitude and the dates on which it last crosses 90 %, 50 % and
15 % on the way down. Those six crossings plus the peak itself are the seven
`phenometrics`; the product publishes up to TWO vegetation cycles a year, to
describe double-cropping and rain-driven green-ups, and this store keeps the
FIRST (`cycle = 0`, the one with the earlier peak).

SOURCE, VERIFIED 2026-09-20 FROM THIS SANDBOX (CMR is keyless; the granules
are behind Earthdata Login and are read on a hosted runner):
  CMR collection C2484079943-LPCLOUD, short_name MCD12Q2, version 061,
  "MODIS/Terra+Aqua Land Cover Dynamics Yearly L3 Global 500m SIN Grid V061",
  time_start 2001-01-01. The data link is
  `https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MCD12Q2.061/
  <granule>/<granule>.hdf`, and the granule id carries the PRODUCTION
  timestamp, so the URL cannot be derived from the year and CMR is not a
  convenience but the only honest way to find the file (contract rule 4).
  `ml/family1/adapters/_modis_sin.py` holds the grid, the listing, the login,
  the download and the HDF4 reader; this module holds what the numbers MEAN.

THE LISTING, MEASURED 2026-09-20 (one CMR query a year, keyless):
  2001  315 granules, 315 tiles          2020  315 granules, 315 tiles
  2010  315 granules, 315 tiles          2024  315 granules, 315 tiles
  2025  **626** granules, **315** tiles
The tile set is IDENTICAL in all five years and is the 315 named in `TILES`
below; `index` REFUSES unless the live set is exactly that, the way
`lossyear` refuses a changed tile list, because the store's GROUPS are that
tuple. 2025's 626 granules are 311 tiles PRODUCED TWICE — a September-2026
reprocessing that did not retire its predecessor — so the listing keys on
(year, tile), keeps the higher production timestamp and counts the rest as
`granules_superseded`. CMR reports the same declared size, 274.756 MB, for
every single granule of every year, which is a metadata artefact rather than
a measurement (the same thing `sst_acspo02` documents); 315 x 274.756 MB is
84.5 GB a year declared, and the probe measures the real bytes.

WHAT IS STORED: C = 10, dtype FLOAT16, one group per 500 m sinusoidal tile,
ONE FRAME A YEAR. The names are the product's own scientific datasets, taken
from the LP DAAC layer table and the user guide's Table 1:

  greenup        Greenup        the day EVI2 first crossed 15 % rising
  mid_greenup    MidGreenup     ... 50 % rising
  peak           Peak           the day EVI2 reached the segment maximum
  senescence     Senescence     the day EVI2 last crossed 90 % falling
  mid_greendown  MidGreendown   ... 50 % falling
  dormancy       Dormancy       ... 15 % falling
  evi_minimum    EVI_Minimum    the segment's minimum EVI2      (x 0.0001)
  evi_amplitude  EVI_Amplitude  maximum minus minimum EVI2      (x 0.0001)
  evi_area       EVI_Area       the integral of EVI2 over the cycle (x 0.1)
  qa_overall     QA_Overall     0 best, 1 good, 2 fair, 3 poor

THE DATES ARE STORED AS A DAY NUMBER IN THE PRODUCT YEAR, AND THE CONVERSION
IS THE ONE THING A READER MUST NOT GET WRONG. The source stores every date as
**int16 days since 1970-01-01** (the user guide: "All dates are converted to
UNIX-epoch time: days since Jan 1, 1970"), valid 11138..32766, fill 32767.
This store subtracts the product year's own 1 January and adds one:

    day_number = (days_since_1970) - (days from 1970-01-01 to Y-01-01) + 1

so 1 is 1 January of the product year and 365 (or 366) is 31 December. A
NEGATIVE number is a date in the PREVIOUS calendar year and a number above
365/366 one in the NEXT: that is not an error but the southern hemisphere,
whose growing season straddles New Year, and it is why the channel is called a
day NUMBER and not a day of year. float16 holds every integer up to 2,048
exactly, so no date is rounded.

THE OTHER SCALINGS, and what float16 costs them, measured rather than assumed.
`EVI_Minimum` and `EVI_Amplitude` are raw 0..10,000 at 0.0001, i.e. EVI2 in
[0, 1]; float16 spaces values 0.00098 apart at 1.0 and 0.00006 at 0.1, against
the source's flat 0.0001 — so the store is coarser than the source in the top
decade and finer below it, and NBAR-EVI2's own uncertainty is about 0.01-0.02,
ten to twenty times the worst of that. `EVI_Area` is raw 0..3,700 at 0.1, i.e.
0..370 EVI2-days; float16 spaces it 0.25 apart at 370 and 0.0078 at 10, so the
top of the range loses a factor of two against the source's 0.1 step. Both
losses are recorded here rather than discovered later.

THE PRODUCER'S OWN VALID RANGE IS APPLIED AND COUNTED, exactly as `lst05`
applies MOD11C1's: a raw value outside the range the producer publishes is
not a measurement, it becomes NaN, and it is counted by channel name under
`outside_valid_range`. The fill value 32767 is counted separately. The channel
BOUNDS are then tripwires and not filters: a date more than about a year away
from the product year, an EVI2 outside [-0.5, 2] or an area outside
[-10, 1000] is a corrupt file, not a season.

THE FRAME AND ITS BIN. One frame a year, `frames_per_bin = 1` and
`frame_seconds = 432,000`, filed in **the bin holding 1 July of the product
year** — the middle of the year it describes, the same rule `canopy30` uses
for its single-year map. 2001 is bin 1424 (2001-06-30 .. 07-04), 2020 is bin
2812 (2020-06-30 .. 07-04), 2025 is bin 3177 (2025-06-29 .. 07-03). Two
product years are 73 bins apart, so no two frames can collide, and
`slot_map` refuses one anyway. A bin with no product year in it is counted
`no_frame_in_bin` and writes NO SHARD AT ALL — 72 bins out of every 73 are in
that state BY CONSTRUCTION, and it is deliberately not `absent_upstream`,
which is a frame the product should have and does not. `tile_grid.json`
carries a
`frame_table` giving every (bin, frame) its year's own first and last day, so
nothing downstream infers a twelve-month period from the five-day slot it
sits in — the mechanism `chirps05` wrote for its pentads.

TWO ENVIRONMENT KNOBS, both read at CONSTRUCTION time so the fresh adapter
`stage_probe_grid` builds for itself sees them. `PHENO500_TILES` (a comma list
of tile names) restricts the store's groups — a probe cannot fetch 315 tiles
of 275 MB, and a handful is one flag; a restricted listing also asks CMR per
tile, which makes a probe cheap. `PHENO500_MAX_YEARS` (default 25, the whole
record) is the size gate `sst_acspo02` wears, at dispatch rather than at hour
five.

PHASE B: the SECOND vegetation cycle (`cycle = 1`), which doubles C and
matters for double-cropped land; `Maturity`, the seventh phenometric; and
`QA_Detailed`, whose bit pairs carry a quality score per phenometric.
"""
import datetime as dt
import os
import sys

import numpy as np

import build_family10_stores as f10b
from family1 import sharded as sh
from family1.adapters import _modis_sin as ms

SHORT_NAME = "MCD12Q2"
VERSION = "061"
LP_PREFIX = ("https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/"
             f"{SHORT_NAME}.{VERSION}/")
EPOCH_1970 = dt.date(1970, 1, 1)
FILL = 32767
DATE_VALID = (11138, 32766)         # days since 1970-01-01: 2000-06-24 ->
EVI_VALID = (0, 10000)
AREA_VALID = (0, 3700)
QA_VALID = (0, 3)
EVI_SCALE = 0.0001
AREA_SCALE = 0.1
FIRST_YEAR = 2001
LAST_YEAR_MEASURED = 2025

# dataset name -> (channel name, unit, lo bound, hi bound, kind)
DATE_BOUNDS = (-400.0, 800.0)
LAYERS = (
    ("Greenup", "greenup", "day number in the product year", "date"),
    ("MidGreenup", "mid_greenup", "day number in the product year", "date"),
    ("Peak", "peak", "day number in the product year", "date"),
    ("Senescence", "senescence", "day number in the product year", "date"),
    ("MidGreendown", "mid_greendown", "day number in the product year",
     "date"),
    ("Dormancy", "dormancy", "day number in the product year", "date"),
    ("EVI_Minimum", "evi_minimum", "NBAR-EVI2 (dimensionless)", "evi"),
    ("EVI_Amplitude", "evi_amplitude", "NBAR-EVI2 (dimensionless)", "evi"),
    ("EVI_Area", "evi_area", "NBAR-EVI2 x days", "area"),
    ("QA_Overall", "qa_overall", "0 best, 1 good, 2 fair, 3 poor", "qa"),
)
KIND_BOUNDS = {"date": DATE_BOUNDS, "evi": (-0.5, 2.0),
               "area": (-10.0, 1000.0),
               "qa": (0.0, 3.0)}
KIND_VALID = {"date": DATE_VALID, "evi": EVI_VALID, "area": AREA_VALID,
              "qa": QA_VALID}

TILES = (
    "h00v08", "h00v09", "h00v10", "h01v08", "h01v09", "h01v10",
    "h01v11", "h02v06", "h02v08", "h02v09", "h02v10", "h02v11",
    "h03v06", "h03v07", "h03v09", "h03v10", "h03v11", "h04v09",
    "h04v10", "h04v11", "h05v10", "h05v11", "h05v13", "h06v03",
    "h06v11", "h07v03", "h07v05", "h07v06", "h07v07", "h08v03",
    "h08v04", "h08v05", "h08v06", "h08v07", "h08v08", "h08v09",
    "h09v02", "h09v03", "h09v04", "h09v05", "h09v06", "h09v07",
    "h09v08", "h09v09", "h10v02", "h10v03", "h10v04", "h10v05",
    "h10v06", "h10v07", "h10v08", "h10v09", "h10v10", "h10v11",
    "h11v02", "h11v03", "h11v04", "h11v05", "h11v06", "h11v07",
    "h11v08", "h11v09", "h11v10", "h11v11", "h11v12", "h12v01",
    "h12v02", "h12v03", "h12v04", "h12v05", "h12v07", "h12v08",
    "h12v09", "h12v10", "h12v11", "h12v12", "h12v13", "h13v01",
    "h13v02", "h13v03", "h13v04", "h13v08", "h13v09", "h13v10",
    "h13v11", "h13v12", "h13v13", "h13v14", "h14v01", "h14v02",
    "h14v03", "h14v04", "h14v09", "h14v10", "h14v11", "h14v14",
    "h14v16", "h14v17", "h15v01", "h15v02", "h15v03", "h15v05",
    "h15v07", "h15v11", "h15v14", "h15v15", "h15v16", "h15v17",
    "h16v00", "h16v01", "h16v02", "h16v05", "h16v06", "h16v07",
    "h16v08", "h16v09", "h16v12", "h16v14", "h16v16", "h16v17",
    "h17v00", "h17v01", "h17v02", "h17v03", "h17v04", "h17v05",
    "h17v06", "h17v07", "h17v08", "h17v10", "h17v12", "h17v13",
    "h17v15", "h17v16", "h17v17", "h18v00", "h18v01", "h18v02",
    "h18v03", "h18v04", "h18v05", "h18v06", "h18v07", "h18v08",
    "h18v09", "h18v14", "h18v15", "h18v16", "h18v17", "h19v00",
    "h19v01", "h19v02", "h19v03", "h19v04", "h19v05", "h19v06",
    "h19v07", "h19v08", "h19v09", "h19v10", "h19v11", "h19v12",
    "h19v15", "h19v16", "h19v17", "h20v01", "h20v02", "h20v03",
    "h20v04", "h20v05", "h20v06", "h20v07", "h20v08", "h20v09",
    "h20v10", "h20v11", "h20v12", "h20v13", "h20v15", "h20v16",
    "h20v17", "h21v01", "h21v02", "h21v03", "h21v04", "h21v05",
    "h21v06", "h21v07", "h21v08", "h21v09", "h21v10", "h21v11",
    "h21v13", "h21v15", "h21v16", "h21v17", "h22v01", "h22v02",
    "h22v03", "h22v04", "h22v05", "h22v06", "h22v07", "h22v08",
    "h22v09", "h22v10", "h22v11", "h22v13", "h22v14", "h22v15",
    "h22v16", "h23v01", "h23v02", "h23v03", "h23v04", "h23v05",
    "h23v06", "h23v07", "h23v08", "h23v09", "h23v10", "h23v11",
    "h23v15", "h23v16", "h24v02", "h24v03", "h24v04", "h24v05",
    "h24v06", "h24v07", "h24v12", "h24v15", "h25v02", "h25v03",
    "h25v04", "h25v05", "h25v06", "h25v07", "h25v08", "h25v09",
    "h26v02", "h26v03", "h26v04", "h26v05", "h26v06", "h26v07",
    "h26v08", "h27v03", "h27v04", "h27v05", "h27v06", "h27v07",
    "h27v08", "h27v09", "h27v10", "h27v11", "h27v12", "h27v14",
    "h28v03", "h28v04", "h28v05", "h28v06", "h28v07", "h28v08",
    "h28v09", "h28v10", "h28v11", "h28v12", "h28v13", "h28v14",
    "h29v03", "h29v05", "h29v06", "h29v07", "h29v08", "h29v09",
    "h29v10", "h29v11", "h29v12", "h29v13", "h30v05", "h30v06",
    "h30v07", "h30v08", "h30v09", "h30v10", "h30v11", "h30v12",
    "h30v13", "h31v06", "h31v07", "h31v08", "h31v09", "h31v10",
    "h31v11", "h31v12", "h31v13", "h32v07", "h32v08", "h32v09",
    "h32v10", "h32v11", "h32v12", "h33v07", "h33v08", "h33v09",
    "h33v10", "h33v11", "h34v07", "h34v08", "h34v09", "h34v10",
    "h35v08", "h35v09", "h35v10",
)


def days_1970(d):
    return (d - EPOCH_1970).days


def year_bin(year):
    """The bin holding 1 July of `year` — where that year's frame is filed."""
    return f10b.seconds_since_epoch(dt.date(int(year), 7, 1)) \
        // sh.BIN_SECONDS


def year_days(year):
    return dt.date(int(year), 1, 1), dt.date(int(year), 12, 31)


class Pheno500Adapter(ms.SinTileAdapter):
    store = "pheno500"
    title = ("Land-surface phenology, MODIS MCD12Q2 v061, yearly, 500 m "
             "sinusoidal tiles: six phenometric dates, three EVI2 statistics "
             "and quality, first vegetation cycle")
    licence = {
        "name": "NASA open data (no restrictions)",
        "redistribution": "attribution",
        "redistribution_confirmed": True,
        "derived_works": "free",
        "attribution": ("Friedl, M., J. Gray, D. Sulla-Menashe (2022). "
                        "MCD12Q2 MODIS/Terra+Aqua Land Cover Dynamics Yearly "
                        "L3 Global 500m SIN Grid V061. NASA EOSDIS Land "
                        "Processes Distributed Active Archive Center. "
                        "doi:10.5067/MODIS/MCD12Q2.061"),
        "terms": ("earthdata.nasa.gov/engage/open-data-services-and-software/"
                  "data-information-policy, read 2026-09-18: NASA's Earth "
                  "science data are open, full and without restriction, free "
                  "of charge, with no period of exclusive access; users are "
                  "asked to cite the data set"),
    }
    collections = {"combined": (SHORT_NAME, VERSION, "hdf4")}
    groups_default = ("combined",)
    groups_env = ""
    tiles_env = "PHENO500_TILES"
    px_env = "PHENO500_SMOKE_PX"
    tiles = TILES
    datasets = tuple(name for (name, _c, _u, _k) in LAYERS)
    datasets_want = {name: {"_FillValue": FILL}
                     for (name, _c, _u, _k) in LAYERS}
    cycle = 0
    channels = tuple((c, u) + KIND_BOUNDS[k] for (_n, c, u, k) in LAYERS)
    dtype = "float16"
    first_year = FIRST_YEAR
    frames_per_bin = 1
    frame_seconds = sh.BIN_SECONDS
    log2_fp = ms.log2_fp_for(ms.PX_500)                  # -5.908
    log2_dt = float(np.log2(365.0 / 5.0))                # +6.190
    note_estimate = {
        "bytes": 10e9,
        "what": ("family1tf.tex ledger: '~10 GB a year' for MCD12Q2 v061 at "
                 "500 m over 315 tiles")}
    qc_policy = (
        "the product's own QA_Overall is CARRIED AS A CHANNEL, unchanged, so "
        "the consumer picks the threshold (family1tf.tex 5): 0 best, 1 good, "
        "2 fair, 3 poor, a weighted combination of how much of the year's "
        "vegetation index was present rather than filled and how well the "
        "spline fitted. Everything else follows the producer: a raw 32767 is "
        "the fill value and is NaN; a raw value outside the producer's own "
        "valid range -- 11138..32766 for a date, 0..10000 for the two EVI2 "
        "levels, 0..3700 for the area, 0..3 for the quality code -- is not a "
        "measurement either, becomes NaN and is counted "
        "(`outside_valid_range`). The channel bounds are then TRIPWIRES and "
        "not filters: a date more than about a year from the product year, an "
        "EVI2 outside [-0.5, 2] or an area outside [-10, 1000] is a corrupt "
        "file and not a season, and a value outside them becomes NaN and is "
        "counted (`out_of_bounds`), never clipped. The second vegetation "
        "cycle and the per-phenometric QA_Detailed bits are phase B")
    sources = (LP_PREFIX + "<granule>/<granule>.hdf",
               ms.CMR + f"?short_name={SHORT_NAME}&version={VERSION}")
    verified = (
        "2026-09-20 from the sandbox, keyless: the CMR collection search "
        "(C2484079943-LPCLOUD, MCD12Q2 v061, cloud_hosted, time_start "
        "2001-01-01, no end) and the granule listing for 2001, 2010, 2020, "
        "2024 and 2025 (315 tiles in every one; 2025 carries 626 granules "
        "because 311 tiles were reprocessed in September 2026 and the "
        "predecessor was not retired; CMR declares the same 274.756 MB for "
        "every granule of every year, which is a metadata artefact); the "
        "https 'lp-prod-protected' .hdf data link of "
        "MCD12Q2.A2020001.h18v04.061.2022117225528; the MODLAND sinusoidal "
        "grid parameters on modis-land.gsfc.nasa.gov/GCTP.html (sphere "
        "6371007.181 m, 36 x 18 tiles) checked against the corner polygon CMR "
        "publishes for h18v04; and the layer table on "
        "lpdaac.usgs.gov/products/mcd12q2v061 plus the v061 user guide "
        "(lpdaac.usgs.gov/documents/1417) for the thirteen dataset names, "
        "int16, fill 32767, the valid ranges and the 'days since Jan 1, 1970' "
        "date convention. The granules themselves are behind Earthdata Login "
        "and are opened on a GitHub-hosted runner by the probe, which "
        "re-checks the tile's own StructMetadata corners in every file it "
        "reads")
    notes = (
        "MCD12Q2 v061, one HDF-EOS2 granule per 500 m sinusoidal tile per "
        "year, 315 tiles. C = 10, the FIRST vegetation cycle. The six date "
        "channels are stored as a DAY NUMBER IN THE PRODUCT YEAR (1 = 1 "
        "January), converted from the source's 'days since 1970-01-01'; a "
        "negative number is a date in the previous calendar year and one "
        "above 365/366 a date in the next, which is the southern hemisphere's "
        "growing season and not an error. float16 holds every integer to "
        "2,048 exactly, so no date is rounded; EVI2 levels lose their last "
        "digit at the top of their range (0.00098 against the source's "
        "0.0001) and the EVI2 area a factor of two (0.25 against 0.1), both "
        "far inside the product's own uncertainty. One frame a year in the "
        "bin holding 1 July of that year, with a frame_table giving every "
        "(bin, frame) its year's first and last day. Row 0 of a tile is its "
        "northernmost, the HDF-EOS grid's own orientation.")
    smoke_window = ("2020-06-25", "2021-07-10")
    smoke_probe_month = "2020-06"
    WORKERS = 3

    def __init__(self):
        try:
            self.max_years = int(os.environ.get("PHENO500_MAX_YEARS")
                                 or (LAST_YEAR_MEASURED - FIRST_YEAR + 1))
        except ValueError:
            sys.exit("PHENO500_MAX_YEARS must be an integer")
        super().__init__()

    # ------------------------------------------------------------ the grid --
    def specs(self):
        out = super().specs()
        for spec in out.values():
            spec["frame_rule"] = (
                "frame 0 of bin b is the whole five-day bin, and it holds the "
                "MCD12Q2 product YEAR whose 1 July falls inside it. A product "
                "year is twelve months long, NOT five days: read its real "
                "span from `frame_table`")
            spec["annual_map"] = {
                "rule": ("a single-year map is filed under the bin holding 1 "
                         "July of the year it describes — the middle of its "
                         "period, the same rule `canopy30` uses"),
                "cycle": self.cycle,
                "cycle_note": ("the product publishes up to two vegetation "
                               "cycles a year; this store keeps the first"),
                "date_convention": (
                    "the six date channels are a DAY NUMBER in the product "
                    "year: 1 is 1 January, 365 or 366 is 31 December, a "
                    "negative number is a date in the previous calendar year "
                    "and one above 365/366 a date in the next. The source "
                    "stores days since 1970-01-01 (int16, valid "
                    f"{DATE_VALID[0]}..{DATE_VALID[1]}, fill {FILL}) and this "
                    "store subtracts the product year's own 1 January and "
                    "adds one"),
                "source_scalings": {
                    "EVI_Minimum": EVI_SCALE, "EVI_Amplitude": EVI_SCALE,
                    "EVI_Area": AREA_SCALE},
            }
            spec["frame_table_columns"] = ["bin", "frame", "year",
                                           "first_day", "last_day", "days"]
            spec["frame_table_note"] = (
                "every (bin, frame) a product year fills between "
                f"{FIRST_YEAR} and {LAST_YEAR_MEASURED + 10}, whether or not "
                "this store holds it; store.json's date_range says which part "
                "is filled. A (bin, frame) absent from this table holds no "
                "product year at all and is a zero-length frame counted "
                f"`{self.empty_slot_reason}`")
            spec["frame_table"] = frame_table()
        return out

    def record_frames(self, ctx, group):
        """How many frames the whole record holds, for the probe's
        extrapolation: one a year over the measured record."""
        return LAST_YEAR_MEASURED - FIRST_YEAR + 1

    # ------------------------------------------------- the product calendar --
    def periods(self, ctx, sensor, year):
        y = int(year)
        if y < FIRST_YEAR:
            return {}
        return {(year_bin(y), 0): y}

    def granule_date(self, key):
        return dt.date(int(key), 1, 1)

    def key_of_date(self, d):
        return int(d.year)

    def period_days(self, key):
        return year_days(key)

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
                f"REFUSING pheno500: the window covers {years} year(s) and "
                f"PHENO500_MAX_YEARS is {self.max_years}. Raise it "
                f"deliberately (adapter_env) once the probe's projection has "
                f"been read. Nothing has been fetched.")
        return None

    # --------------------------------------------------------------- index --
    def index(self, ctx):
        sensor = self.groups[0]
        sn, ver, fmt = self.collection(sensor)
        years = self.listing_years(ctx)
        per_year, counts_all, live = {}, {}, set()
        files_all = {}
        for y in years:
            files, counts = self.listing(ctx, sensor, y)
            if not files:
                continue
            per_year[str(y)] = len(files)
            counts_all[str(y)] = counts
            live |= {ms.tile_name(*hv) for (_d, hv) in files}
            files_all.update(files)
        if not files_all:
            sys.exit(f"REFUSING pheno500: CMR lists no granule of {sn} "
                     f"v{ver} for any of the years {years} — an empty listing "
                     f"is a refusal (ml/CLAUDE.md, the 2026-09-14 rule)")
        want = set(self.tiles)
        extra = sorted(live - want)
        if extra:
            sys.exit(f"REFUSING pheno500: the listing names {len(extra)} "
                     f"tile(s) this adapter's TILES does not: {extra[:8]}. "
                     f"The store's GROUPS are that tuple, so a changed tile "
                     f"set must be re-measured and committed rather than "
                     f"silently altering the store.")
        lo, hi = self.record(ctx, sensor)
        out = {"dataset": f"{sn} v{ver} yearly 500 m sinusoidal land-surface "
                          f"phenology (HDF-EOS2)",
               "url": ms.CMR + f"?short_name={sn}&version={ver}",
               "short_name": sn, "version": ver,
               "years_listed": sorted(per_year),
               "granules_per_year": per_year,
               "granules": sum(per_year.values()),
               "tiles_declared": len(self.tiles),
               "tiles_in_listing": len(live),
               "tiles_declared_and_absent": sorted(want - live)[:40],
               "tiles_built": len(self.tile_names),
               "record": [str(lo), str(hi)],
               "record_years": [FIRST_YEAR, LAST_YEAR_MEASURED],
               "bytes_cmr": int(sum(r["bytes_cmr"]
                                    for r in files_all.values())),
               "bytes_note": ("CMR declares the same 274.756 MB for every "
                              "granule of this collection, which is a "
                              "metadata artefact and not a measurement; the "
                              "probe measures exact bytes"),
               "listing": counts_all,
               "frames_per_bin": self.frames_per_bin,
               "frame_seconds": self.frame_seconds,
               "cycle": self.cycle,
               "bins": {str(y): int(year_bin(y)) for y in sorted(per_year)},
               "grid": ms.sin_grid(*ms.tile_hv(self.tile_names[0]),
                                   px=self.px)}
        # ONE REAL GRANULE, downloaded and read against the declared grid, the
        # declared attributes and the tile's own StructMetadata corners
        mine = {k: v for k, v in files_all.items()
                if ms.tile_name(*k[1]) in set(self.tile_names)}
        out["first_file"] = self.read_first_file(ctx, sensor,
                                                 mine or files_all)
        return out

    # ---------------------------------------------------------- the frame --
    def frame_from(self, arrs, attrs, rec=None):
        """{dataset: raw int16} -> (float32 [px, px, 10] with NaN, counts).

        `rec` is the granule's CMR record, and the PRODUCT YEAR is read out of
        its title: a date channel means "days since 1970-01-01" in the file
        and "day number in the product year" in the store, so the conversion
        needs the year and it is never taken from shared state.
        """
        px = self.px
        out = np.empty((px, px, len(LAYERS)), np.float32)
        counts = {}
        absent = ms.check_attrs(attrs, self.datasets_want, self.store)
        if absent:
            counts["attributes_absent"] = absent
        year = int(self.product_year(rec))
        base = days_1970(dt.date(year, 1, 1))
        for i, (name, ch, _u, kind) in enumerate(LAYERS):
            raw = arrs[name]
            if raw.dtype != np.int16:
                raise ms.FormatError(f"{name}: {raw.dtype}, expected int16")
            lo, hi = KIND_VALID[kind]
            fill = raw == np.int16(FILL)
            bad = (raw < lo) | (raw > hi)
            n_fill = int(fill.sum())
            n_bad = int(bad.sum()) - int((bad & fill).sum())
            v = raw.astype(np.float32)
            if kind == "date":
                v = v - np.float32(base - 1)
            elif kind == "evi":
                v = v * np.float32(EVI_SCALE)
            elif kind == "area":
                v = v * np.float32(AREA_SCALE)
            v[bad] = np.nan
            out[:, :, i] = v
            counts[f"fill_pixels_{ch}"] = n_fill
            if n_bad:
                counts.setdefault("outside_valid_range", {})[ch] = n_bad
        counts["product_year"] = {str(year): 1}
        return out, counts

    def product_year(self, rec):
        """The granule's own A-date year — the year its dates are relative
        to. `rec` is the CMR record; the smoke passes a bare {'title': ...}."""
        if rec is None:
            raise ms.FormatError("pheno500: a frame was read with no granule "
                                 "record, so its product year is unknown — "
                                 "the date conversion cannot be done blind")
        return ms.granule_key(rec["title"])[0].year

    # --------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260920):
        truth = make_smoke_sources(root, d_lo, d_hi, seed)
        self.smoke_reinit()
        return truth


def frame_table(y_lo=FIRST_YEAR, y_hi=LAST_YEAR_MEASURED + 10):
    """Every (bin, frame) a product year fills -> its year's own dates.

    A list of [bin, frame, year, first day, last day, days], sorted, so a
    reader never has to infer a twelve-month period from the five-day slot it
    sits in — the mechanism `chirps05` wrote for its pentads.
    """
    rows = []
    for y in range(int(y_lo), int(y_hi) + 1):
        d0, d1 = year_days(y)
        rows.append([int(year_bin(y)), 0, int(y), str(d0), str(d1),
                     (d1 - d0).days + 1])
    return sorted(rows)


# ================================================================== smoke ==
SMOKE_PX = 240                       # a tenth of the real 2,400
SMOKE_TILES = ("h18v04", "h12v09", "h29v11")
SMOKE_ABSENT_TILE = "h12v09"         # listed in 2021 and not in 2020
SMOKE_YEARS = (2020, 2021)
SMOKE_OOB = ("h29v11", 2020)         # one EVI_Area of 9,000 -> 900, in bounds
SMOKE_FILL_FRACTION = 0.25


def smoke_fields(tile, year, px):
    """A deterministic granule: a green-up date that walks with latitude, a
    fill patch, one value outside the producer's valid range and one past the
    channel bound."""
    h, v = ms.tile_hv(tile)
    k = (h * 7 + v * 13 + year) % 23
    yy = (np.arange(px, dtype=np.float64)[:, None] - px / 2) / px
    xx = (np.arange(px, dtype=np.float64)[None, :] - px / 2) / px
    base = days_1970(dt.date(year, 1, 1))
    greenup = base + 90 + np.rint(40 * yy + 8 * np.sin(9 * xx + k))
    out = {}
    offsets = {"Greenup": 0, "MidGreenup": 12, "Peak": 45, "Senescence": 120,
               "MidGreendown": 150, "Dormancy": 180}
    nodata = (np.hypot(yy, xx) > 0.42)           # the "ocean" of this tile
    for name, off in offsets.items():
        a = (greenup + off).astype(np.int16)
        a[nodata] = np.int16(FILL)
        out[name] = a
    evi_min = np.rint(500 + 2000 * (0.5 + 0.5 * np.sin(7 * xx - k))
                      + 0 * yy).astype(np.int16)
    evi_amp = np.rint(1500 + 5000 * (0.5 + 0.5 * np.cos(5 * yy + k))
                      + 0 * xx).astype(np.int16)
    evi_area = np.rint(300 + 2500 * (0.5 + 0.5 * np.sin(3 * xx + 4 * yy))
                       ).astype(np.int16)
    qa = np.rint(np.abs(np.sin(11 * xx + 3 * yy)) * 3).astype(np.int16)
    for name, a in (("EVI_Minimum", evi_min), ("EVI_Amplitude", evi_amp),
                    ("EVI_Area", evi_area), ("QA_Overall", qa)):
        a = a.copy()
        a[nodata] = np.int16(FILL)
        out[name] = a
    if (tile, year) == SMOKE_OOB:
        # a raw 12,000 is outside the producer's own 0..10000 for an EVI2
        # level (counted `outside_valid_range`), and a raw date of 1,000 is
        # outside its 11138..32766 (counted the same way)
        out["EVI_Minimum"][px // 2, px // 2] = 12000
        out["Greenup"][px // 2, px // 2 + 1] = 1000
    return out


def make_smoke_sources(root, d_lo, d_hi, seed=20260920, skip=(), absent=()):
    """The archive in its real layout — `cmr_combined_<year>.json` in CMR's
    own shape and HDF-EOS2 granules with a real `StructMetadata.0` — and the
    truth for every frame.

    `SMOKE_ABSENT_TILE` is listed in 2021 and NOT in 2020, so one frame of one
    group is `absent_upstream`; a title in `absent` is listed and its file is
    not written, which is `note_absent` and is exercised by its own test.

    Returns {(group, bin, frame): (float16 [px, px, 10] or None, reason or
    None)} for every frame of every bin overlapping the record.
    """
    os.environ["PHENO500_SMOKE_PX"] = str(SMOKE_PX)
    os.environ["PHENO500_TILES"] = ",".join(SMOKE_TILES)
    px = SMOKE_PX
    ad = Pheno500Adapter()
    fields, titles = {}, {}
    for y in SMOKE_YEARS:
        recs, arrays = [], {}
        for t in SMOKE_TILES:
            if (t, y) in skip or (t == SMOKE_ABSENT_TILE and y == 2020):
                continue
            h, v = ms.tile_hv(t)
            title = (f"{SHORT_NAME}.A{y:04d}001.{t}.{VERSION}."
                     f"{2026000000000 + h * 1000 + v}")
            recs.append((title, (h, v)))
            arrays[title] = smoke_fields(t, y, px)
            fields[(t, y)] = arrays[title]
            titles[(t, y)] = title
        ms.write_smoke_archive(
            root, "pheno500", "combined", y, recs, arrays, px, LP_PREFIX,
            attrs={n: {"_FillValue": FILL} for n in ad.datasets},
            absent=absent)

    truth = {}
    t_lo = f10b.seconds_since_epoch(d_lo)
    t_hi = f10b.seconds_since_epoch(d_hi) + 86399
    bins = {year_bin(y): y for y in SMOKE_YEARS}
    lo_y, hi_y = min(SMOKE_YEARS), max(SMOKE_YEARS)
    for b in sh.bins_overlapping(t_lo, t_hi):
        y = bins.get(b)
        for t in SMOKE_TILES:
            g = t
            if y is None:
                t0 = sh.frame_start_seconds(b, 0, ad.frame_seconds)
                why = ("before_record"
                       if t0 < f10b.seconds_since_epoch(dt.date(lo_y, 1, 1))
                       else "after_record"
                       if t0 > f10b.seconds_since_epoch(
                           dt.date(hi_y, 12, 31)) + 86399
                       else sh.FRAME_NO_FRAME_IN_BIN)
                truth[(g, b, 0)] = (None, why)
                continue
            if (t, y) not in fields:
                truth[(g, b, 0)] = (None, "absent_upstream")
                continue
            a, _c = ad.frame_from(fields[(t, y)], {},
                                  {"title": titles[(t, y)]})
            ms.mc.mask_bounds_low_memory(ad, a)
            truth[(g, b, 0)] = (a.astype(np.float16), None)
    return truth


ADAPTER = Pheno500Adapter
