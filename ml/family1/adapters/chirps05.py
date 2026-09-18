"""CHIRPS v3.0 pentad precipitation — five-day rainfall from satellite
infrared and rain gauges, 0.05 degrees over land (family 1.0.tf, exception
E4, E-082 wave 2).

PLAIN ENGLISH. Cold cloud tops seen by geostationary infrared say how long it
rained; tens of thousands of rain gauges say how much. The Climate Hazards
Center at UC Santa Barbara combines the two into a 5.6 km rainfall field for
all the land between 60 S and 60 N, every five days since 1981 — the longest
high-resolution rainfall record that exists, and the one derived field this
family admits as an INPUT (`family1tf.tex`: exception E4, "derived,
gauge-bearing, CC BY 4.0").

SOURCE, VERIFIED 2026-09-18 FROM THIS SANDBOX (no account; HTTPS):
  https://data.chc.ucsb.edu/products/CHIRPS/v3.0/pentads/global/tifs/
      chirps-v3.0.<YYYY>.<MM>.<P>.tif      P = 1..6, 18-20 MB each
  One GeoTIFF PER PENTAD, which is exactly one store frame, so a frame costs
  one file and no file is read twice. The same pentads also sit as one netCDF
  PER YEAR (`../netcdf/chirps-v3.0.<YYYY>.pentads.nc`, 890 MB), as BILs and
  as COGs; the yearly netCDF was downloaded and opened to CHECK the pentad
  calendar (below) and is not what the adapter reads — a whole year would
  have to be fetched to get one frame.

THE LISTING, MEASURED 2026-09-18: 3,288 GeoTIFFs,
`chirps-v3.0.1981.01.1.tif` .. `chirps-v3.0.2026.08.6.tif` — 46 years x 72
pentads less the four months of 2026 not yet published, with no gap anywhere
in between. Sizes are printed human-readable ("19.1 MiB"), so the index's
byte total is approximate and the probe measures the exact number.

THE PENTAD CALENDAR, CHECKED AGAINST THE PRODUCER'S OWN TIME AXIS. A CHIRPS
pentad P of month M covers days [5(P-1)+1, 5P] for P = 1..5 and
[26, last day of the month] for P = 6 — so SIX per month, of 5, 5, 5, 5, 5
and 3 to 6 days. Two independent confirmations, neither of them a guess:
the 2019 yearly netCDF's `time` axis (float32 days since 1980-01-01,
gregorian) reads 2019-01-01, -06, -11, -16, -21, -26, 2019-02-01, ... — each
pentad stamped at its FIRST day, 72 of them; and the README says the
preliminary product appears "two days after the end of a pentad (on the 2nd,
7th, 12th, 17th, 22nd and 27th day of each month)", i.e. pentads end on the
5th, 10th, 15th, 20th, 25th and the last day.

THE FILING RULE, AND WHY F = 2 AND NOT 1. `family1tf.tex`'s ledger row for
this store says the pentads are "carried as a frame_table (a list of each
frame's dates), each frame filed under the family bin holding its midpoint".
F = 1 CANNOT DO THAT, and the reason is measured rather than argued: over
1981-01 .. 2026-12 there are 3,312 pentads and only 3,298 distinct bins hold
their midpoints — FOURTEEN five-day bins hold TWO pentad midpoints each, all
of them in February, where the short sixth pentad (3 or 4 days) puts its
midpoint within four days of its neighbour's (1989-02-21/25 with 1989-02-26/28
in bin 522; 1992-02-26/29 with 1992-03-01/05 in bin 742; and twelve more).
With F = 1 one of each pair would have to be dropped or overwritten. So
F = 2, frame_seconds = 216,000 (two half-bins of 2.5 days), and the pentad
goes in the half-bin holding its midpoint: **zero collisions over the whole
record**, measured, and the same at F = 5. `tile_grid.json` carries the rule
AND a `frame_table` giving every (bin, frame) its pentad's own first day,
last day and length in days, because a pentad is not 2.5 days long and
nothing downstream may infer its span from the slot it sits in.

A half-bin with no pentad in it is a ZERO-LENGTH frame counted
`no_pentad_in_slot` — 5 days of calendar hold two half-bins and only one
pentad, so about half of all slots are empty BY CONSTRUCTION and that is not
an absence. Slots before the first published pentad are `before_record`,
after the last `after_record`, and a pentad the listing has and the server
will not serve is `ctx.note_absent`, never a silent gap.

THE GRID, MEASURED ON chirps-v3.0.2019.06.1.tif AND CHECKED IN EVERY FILE:
7,200 x 2,400 at 0.05 degrees, EPSG:4326, one float32 band, LZW, striped one
row at a time; transform (0.05, 0, -180; 0, -0.05, 60), so ROW 0 IS THE
NORTHERNMOST (60 N) and col 0 the westernmost (180 W), covering 60 S .. 60 N.
The transform's numbers are stored as float32, so 7,200 x dx reads 360.000005
degrees rather than 360 and the check uses a relative tolerance. (The YEARLY
NETCDF's `latitude` axis runs the other way, -59.975 .. 59.975 — ascending,
row 0 southernmost. The store keeps the GeoTIFF's order and says so.)

WHAT IS STORED. One group, one channel, dtype FLOAT16: `precip`, millimetres
of rain in the pentad (the source's own `mm/pentad`). The source's
`_FillValue` and `missing_value` are both -9999, which is what every pixel
outside the land mask carries; those become NaN. A finite value outside
[0, 5000] mm becomes NaN and is COUNTED (`out_of_bounds`), never clipped —
the wettest pentad measured on the checked day is 753.9 mm and the wettest
five days ever recorded anywhere is about 4,900 mm (Commerson, Reunion,
1980), so 5,000 is a bound a real measurement should not reach and a corrupt
file will. float16 spaces values 0.0625 mm apart at 100 mm and 0.5 mm apart
at 750 mm, which is far inside CHIRPS's own uncertainty.

THE FOOTPRINT. 0.05 degrees is 5.566 km at the equator, so
log2_fp = log2(5.566 / 27.83) = -2.322. log2_dt is log2(5 / 5) = 0: the
frames are 2.5 days APART but each one is a five-day ACCUMULATION, and the
footprint field is the measurement's support, not the slot's width. The
`frame_table` carries the 3-to-6-day real lengths.

`CHIRPS05_SMOKE_GRID` ("W,H") shrinks the grid for `--smoke` and is read at
construction time, so the fresh adapter `stage_probe_grid` builds for itself
sees it too.
"""
import calendar
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

VERSION = "v3.0"
BASE = f"https://data.chc.ucsb.edu/products/CHIRPS/{VERSION}/pentads/global/"
TIFS = BASE + "tifs/"
NETCDF = BASE + "netcdf/"
W, H = 7200, 2400
DX = 0.05
X0, Y0 = -180.0, 60.0          # the tile's WEST and NORTH edges
FILL = -9999.0
PENTADS_PER_MONTH = 6
FIRST_YEAR = 1981
# The frame_table covers a superset of the record: `specs()` runs before the
# listing does (the framework builds the grid specs from the adapter alone),
# so the table is generated from the pentad RULE over a span that cannot run
# out, and the store's own `date_range` says which part of it is filled.
TABLE_YEARS = (1981, 2035)

FILE = re.compile(r"chirps-" + re.escape(VERSION) +
                  r"\.(\d{4})\.(\d{2})\.(\d)\.tif")
ROW = re.compile(r'<a href="(chirps-' + re.escape(VERSION) +
                 r'\.\d{4}\.\d{2}\.\d\.tif)"[^>]*>[^<]*</a></td>'
                 r'\s*<td class="size">\s*([0-9.]+\s*[KMGT]?i?B|-)\s*</td>')
ANY_HREF = re.compile(r'href="([^"?/][^"]*)"')

# GDAL is not thread-safe across datasets in every build; every rasterio
# open/read/close holds this lock (seaice_asi holds NC_LOCK for the same
# reason).
TIF_LOCK = threading.Lock()


class FormatError(ValueError):
    """A listing or a GeoTIFF that does not look like the archive measured."""


def approx_bytes(s):
    s = s.strip()
    m = re.fullmatch(r"([0-9.]+)\s*([KMGT]?)i?B", s)
    if not m:
        return 0
    mult = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3,
            "T": 1024 ** 4}[m.group(2)]
    return int(float(m.group(1)) * mult)


# =============================================================== pentads ===
def pentad_days(year, month, p):
    """(first day, last day) of pentad `p` (1..6) of that month."""
    if not 1 <= int(p) <= PENTADS_PER_MONTH:
        raise FormatError(f"pentad {p} of {year}-{month:02d}: 1..6 only")
    last = calendar.monthrange(int(year), int(month))[1]
    d0 = 5 * (int(p) - 1) + 1
    d1 = 5 * int(p) if int(p) < PENTADS_PER_MONTH else last
    return dt.date(int(year), int(month), d0), dt.date(int(year), int(month),
                                                       d1)


def pentad_slot(year, month, p, frame_seconds):
    """(bin, frame) of the pentad, by its MIDPOINT — the note's rule."""
    d0, d1 = pentad_days(year, month, p)
    s = f10b.seconds_since_epoch(d0)
    e = f10b.seconds_since_epoch(d1) + 86400
    mid = (s + e) // 2
    b = mid // sh.BIN_SECONDS
    return int(b), int((mid - b * sh.BIN_SECONDS) // int(frame_seconds))


def all_pentads(y_lo=TABLE_YEARS[0], y_hi=TABLE_YEARS[1]):
    for y in range(int(y_lo), int(y_hi) + 1):
        for m in range(1, 13):
            for p in range(1, PENTADS_PER_MONTH + 1):
                yield y, m, p


def slot_map(frame_seconds, y_lo=TABLE_YEARS[0], y_hi=TABLE_YEARS[1]):
    """{(bin, frame): (year, month, pentad)} — REFUSES a collision."""
    out = {}
    for (y, m, p) in all_pentads(y_lo, y_hi):
        k = pentad_slot(y, m, p, frame_seconds)
        if k in out:
            raise FormatError(
                f"two pentads in slot {k}: {out[k]} and {(y, m, p)} — "
                f"frames_per_bin {sh.BIN_SECONDS // int(frame_seconds)} "
                f"cannot carry the pentad calendar")
        out[k] = (y, m, p)
    return out


def frame_table(frame_seconds, y_lo=TABLE_YEARS[0], y_hi=TABLE_YEARS[1]):
    """Every (bin, frame) -> its pentad's own dates, compactly.

    A list of [bin, frame, year, month, pentad, first day, last day, days],
    sorted, so a reader never has to infer a pentad's span from the 2.5-day
    slot it sits in.
    """
    rows = []
    for (b, f), (y, m, p) in sorted(slot_map(frame_seconds, y_lo,
                                             y_hi).items()):
        d0, d1 = pentad_days(y, m, p)
        rows.append([int(b), int(f), int(y), int(m), int(p), str(d0), str(d1),
                     (d1 - d0).days + 1])
    return rows


def grid(w=W, h=H):
    dx = 360.0 / int(w)
    dy = 120.0 / int(h)
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
                        "NORTH edges, so this is exactly the source "
                        "GeoTIFF's affine transform"),
        "row_order": ("row 0 is the NORTHERNMOST row, 60 N (the source "
                      "GeoTIFF's own order, transform "
                      "(0.05, 0, -180; 0, -0.05, 60), read off the real "
                      "chirps-v3.0.2019.06.1.tif). The YEARLY NETCDF copies "
                      "of the same pentads run the other way (latitude "
                      "ascending, -59.975 .. 59.975); this store keeps the "
                      "GeoTIFF order"),
        "extent": [X0, Y0 - 120.0, X0 + 360.0, Y0],
        "extent_note": "[west, south, east, north] in degrees; land only",
        "source_variable": (f"the GeoTIFF's single float32 band, mm/pentad, "
                            f"{FILL:.0f} for not-land"),
        "no_tile_corners": ("a plain geographic grid: the affine rule above "
                            "IS the lat/lon of every pixel and every tile "
                            "corner, so no corner table is stored"),
    }


def parse_listing(html):
    """-> ({(y, m, p): (name, approx bytes)}, counts). REFUSES a duplicate."""
    out, counts = {}, {}
    for name, size in ROW.findall(html):
        m = FILE.fullmatch(name)
        if not m:
            raise FormatError(f"{name}: not a CHIRPS pentad file name")
        y, mo, p = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not 1 <= mo <= 12:
            raise FormatError(f"{name}: month {mo}")
        try:
            pentad_days(y, mo, p)
        except FormatError as e:
            raise FormatError(f"{name}: {e}") from None
        if (y, mo, p) in out:
            raise FormatError(f"{name}: two files for {y}-{mo:02d} pentad {p}")
        out[(y, mo, p)] = (name, approx_bytes(size))
    listed = {n for n in ANY_HREF.findall(html)
              if not n.startswith(("http", "#", "?", ".")) and "/" not in n}
    other = sorted(listed - {r[0] for r in ROW.findall(html)})
    if other:
        counts["files_other"] = len(other)
        counts["files_other_names"] = other[:20]
    return out, counts


class Chirps05Adapter(sh.GridAdapter):
    store = "chirps05"
    title = ("Pentad precipitation, CHIRPS v3.0, 0.05 degrees, land "
             "60 S - 60 N (exception E4: six pentads a month on a "
             "frame_table)")
    family = "1tf"
    distribution = "public"
    licence = {
        "name": "CC BY 4.0",
        "redistribution": "attribution",
        "redistribution_confirmed": True,
        "derived_works": "free",
        "attribution": ("Funk, C., P. Peterson, L. Harrison et al. (2026), "
                        "The Climate Hazards Center Infrared Precipitation "
                        "with Stations, Version 3, Sci. Data 13, 718, "
                        "doi:10.1038/s41597-026-07096-4. Data: Climate "
                        "Hazards Center Infrared Precipitation with Stations "
                        "version 3 (CHIRPS3) Data Repository, "
                        "doi:10.15780/G2JQ0P (2025)."),
        "terms": ("chc.ucsb.edu/data/chirps3, read 2026-09-18: 'CHIRPS3 is "
                  "in the public domain, as registered with Creative "
                  "Commons, and is licensed under a Creative Commons "
                  "Attribution 4.0 International License. To the extent "
                  "possible under the law, the Climate Hazards Center has "
                  "waived all copyright and related or neighboring rights to "
                  "CHIRPS3.'"),
    }
    time_dtype = "int32"
    credentials = ()
    channels = (("precip", "mm/pentad", 0.0, 5000.0),)
    log2_fp = float(np.log2(360.0 / W * 111.31949 / 27.83))      # -2.322
    log2_dt = 0.0        # a five-day accumulation: log2(5 / 5)
    per_year = True
    first_year = FIRST_YEAR
    frames_per_bin = 2
    frame_seconds = sh.BIN_SECONDS // 2                          # 216,000
    dtype = "float16"
    tile = sh.TILE
    zstd_level = sh.DEFAULT_LEVEL
    note_estimate = {"bytes": 40e9,
                     "what": "family1tf.tex ledger: 'v ~ 0.3: ~40 GB' for "
                             "~3,300 CHIRPS v3.0 pentads at 0.05 degrees"}
    qc_policy = (
        "the source's only flag is its fill value: _FillValue and "
        "missing_value are both -9999 and mark every pixel outside the land "
        "mask, which becomes NaN. A finite value outside [0, 5000] mm "
        "becomes NaN and is counted (`out_of_bounds`), never clipped -- the "
        "wettest five days ever recorded anywhere is about 4,900 mm, so the "
        "bound is one a measurement should not reach and a corrupt file "
        "will. No per-pixel gauge count or uncertainty is published with the "
        "pentad fields")
    sources = (TIFS + "chirps-" + VERSION + ".<YYYY>.<MM>.<P>.tif",)
    verified = (
        "2026-09-18 from the sandbox: the whole tifs/ listing (3,288 files, "
        "chirps-v3.0.1981.01.1 .. 2026.08.6, no gap); "
        "chirps-v3.0.2019.06.1.tif opened with rasterio (7200 x 2400 "
        "float32, EPSG:4326, transform (0.05, 0, -180; 0, -0.05, 60), LZW, "
        "one row a strip, values -9999 .. 753.93); the 2019 yearly netCDF "
        "downloaded and opened to check the pentad calendar (72 pentads, "
        "`time` = days since 1980-01-01 stamped at each pentad's FIRST day, "
        "precip float32 mm/pentad with fill -9999, latitude ASCENDING); and "
        "the licence sentence on chc.ucsb.edu/data/chirps3")
    notes = (
        "Exception E4: six pentads a month, of 5, 5, 5, 5, 5 and 3-6 days. "
        "F = 2 half-bins of 2.5 days and the pentad goes in the half-bin "
        "holding its MIDPOINT -- F = 1 is impossible, because 14 five-day "
        "bins of the record hold two pentad midpoints each (measured, all in "
        "February). tile_grid.json carries a frame_table giving every (bin, "
        "frame) its pentad's own first day, last day and length, so nothing "
        "downstream infers a pentad's five-day span from its 2.5-day slot. A "
        "half-bin with no pentad is a zero-length frame counted "
        "`no_pentad_in_slot`, which about half of all slots are by "
        "construction. Row 0 is the northernmost, as in the source GeoTIFFs "
        "(the yearly netCDF copies run the other way).")
    smoke_window = ("1981-01-01", "1981-02-28")
    smoke_probe_month = "1981-01"

    WORKERS = 4

    def __init__(self):
        g = (os.environ.get("CHIRPS05_SMOKE_GRID") or "").split(",")
        self.w, self.h = (int(g[0]), int(g[1])) if len(g) == 2 else (W, H)
        self._listing = None
        self._slots = None
        if (self.w, self.h) != (W, H):
            self.notes = (f"{self.notes}\nSMOKE GRID: CHIRPS05_SMOKE_GRID set "
                          f"the grid to {self.w} x {self.h} instead of "
                          f"{W} x {H}. This is a synthetic store.")

    # ------------------------------------------------------------ the grid --
    @property
    def grid(self):
        return grid(self.w, self.h)

    def slots(self):
        if self._slots is None:
            self._slots = slot_map(self.frame_seconds)
        return self._slots

    def specs(self):
        out = super().specs()
        spec = out[self.store]
        spec["frame_rule"] = (
            "frame f of bin b is the half-bin [b * 432000 + f * 216000, "
            "+ 216000) seconds since the epoch, and it holds the CHIRPS "
            "pentad whose MIDPOINT falls inside it. A pentad is 3 to 6 days "
            "long, NOT 2.5: read its real span from `frame_table`")
        spec["pentad_rule"] = (
            "pentad P of month M covers days [5(P-1)+1, 5P] for P = 1..5 and "
            "[26, last day of the month] for P = 6; six a month, checked "
            "against the producer's own netCDF time axis")
        spec["frame_table_columns"] = ["bin", "frame", "year", "month",
                                       "pentad", "first_day", "last_day",
                                       "days"]
        spec["frame_table_span_years"] = list(TABLE_YEARS)
        spec["frame_table_note"] = (
            "every (bin, frame) the pentad calendar fills between "
            f"{TABLE_YEARS[0]} and {TABLE_YEARS[1]}, whether or not this "
            "store holds it; store.json's date_range says which part is "
            "filled. A (bin, frame) absent from this table holds no pentad "
            "at all and is a zero-length frame counted `no_pentad_in_slot`")
        spec["frame_table"] = frame_table(self.frame_seconds)
        return out

    def record_frames(self, ctx, group):
        files, _ = self.listing(ctx)
        return len(files)

    # ------------------------------------------------------------- listing --
    def listing(self, ctx):
        if self._listing is not None:
            return self._listing
        if ctx.source_dir:
            p = os.path.join(ctx.source_dir, self.store, "tifs", "index.html")
            if not os.path.exists(p):
                sys.exit(f"REFUSING chirps05: no {p} — the smoke's synthetic "
                         f"listing is missing")
            with open(p, "rb") as fh:
                raw = fh.read()
            ctx.count_bytes(len(raw))
        else:
            raw, why = cm.get_bytes(TIFS, attempts=ctx.a.attempts)
            if raw is None:
                sys.exit(f"REFUSING chirps05: the listing of {TIFS} answered "
                         f"{why} — a listing that comes back empty is a "
                         f"refusal (ml/CLAUDE.md, the 2026-09-14 rule)")
        try:
            files, counts = parse_listing(raw.decode("utf-8", "replace"))
        except FormatError as e:
            sys.exit(f"REFUSING chirps05: the listing of {TIFS}: {e}")
        if not files:
            sys.exit(f"REFUSING chirps05: {TIFS} lists no pentad GeoTIFF at "
                     f"all ({counts}) — an empty listing is a refusal")
        self._listing = (files, counts)
        return self._listing

    def record(self, ctx):
        """(first pentad, last pentad) as (year, month, pentad) keys."""
        files, _ = self.listing(ctx)
        return min(files), max(files)

    def index(self, ctx):
        files, counts = self.listing(ctx)
        lo, hi = min(files), max(files)
        per_year, gaps = {}, []
        for (y, m, p) in all_pentads(lo[0], hi[0]):
            if not (lo <= (y, m, p) <= hi):
                continue
            if (y, m, p) in files:
                per_year[str(y)] = per_year.get(str(y), 0) + 1
            else:
                gaps.append(f"{y:04d}-{m:02d} pentad {p}")
        # the slot map is generated here too, so a collision is a refusal at
        # index time rather than a lost pentad at fetch time
        try:
            slots = self.slots()
        except FormatError as e:
            sys.exit(f"REFUSING chirps05: {e}")
        d0, _ = pentad_days(*lo)
        _, d1 = pentad_days(*hi)
        out = {"dataset": f"CHIRPS {VERSION} pentad precipitation, global "
                          f"0.05 degrees (GeoTIFF)",
               "url": TIFS, "version": VERSION,
               "files": len(files),
               "files_per_year": per_year,
               "record": [f"{lo[0]:04d}-{lo[1]:02d} pentad {lo[2]}",
                          f"{hi[0]:04d}-{hi[1]:02d} pentad {hi[2]}"],
               "record_days": [str(d0), str(d1)],
               "pentads_missing_in_record": gaps,
               "bytes_approx": int(sum(b for _, b in files.values())),
               "bytes_note": ("the listing's size column is human-readable "
                              "(\"19.1 MiB\"); the probe measures exact "
                              "bytes"),
               "other_files": counts,
               "frames_per_bin": self.frames_per_bin,
               "frame_seconds": self.frame_seconds,
               "slots_in_table": len(slots),
               "bin_first": pentad_slot(*lo, self.frame_seconds)[0],
               "bin_last": pentad_slot(*hi, self.frame_seconds)[0],
               "grid": self.grid}
        if gaps:
            sys.exit(f"REFUSING chirps05: {len(gaps)} pentad(s) inside the "
                     f"record {out['record']} are not in the listing "
                     f"({gaps[:6]}) — a hole in the middle of a published "
                     f"record is a listing problem until proven otherwise")
        # ONE REAL FILE, read and checked against the declared grid
        name = files[lo][0] if lo[0] >= FIRST_YEAR else files[hi][0]
        want = lo if lo[0] >= FIRST_YEAR else hi
        meta = self.read_header(ctx, files[want][0])
        out["first_file"] = {"name": name, **meta}
        return out

    # ------------------------------------------------------------ one file --
    def _url(self, name):
        return TIFS + name

    def _local(self, ctx, name):
        return os.path.join(ctx.source_dir, self.store, "tifs", name)

    def _get(self, ctx, name, tmpdir):
        """The GeoTIFF on local disk -> (path, bytes) or (None, why)."""
        if ctx.source_dir:
            p = self._local(ctx, name)
            if not os.path.exists(p):
                return None, f"{name}: listed and absent"
            ctx.count_bytes(os.path.getsize(p))
            return p, os.path.getsize(p)
        p = os.path.join(tmpdir, name)
        if f10b.fetch_first([self._url(name)], p,
                            attempts=ctx.a.attempts) is None:
            return None, f"{self._url(name)}: listed, and 404"
        return p, os.path.getsize(p)

    def read_header(self, ctx, name):
        tmpdir = None if ctx.source_dir else \
            tempfile.mkdtemp(prefix="chirps_", dir=ctx.scratch)
        try:
            p, n = self._get(ctx, name, tmpdir)
            if p is None:
                sys.exit(f"REFUSING chirps05: the first file {name} could "
                         f"not be read: {n}")
            try:
                with TIF_LOCK:
                    meta = tif_meta(p, self.w, self.h)
            except (FormatError, IOError) as e:
                sys.exit(f"REFUSING chirps05: the first file {name} does not "
                         f"match the declared grid: {e}")
            meta["bytes"] = n
            return meta
        finally:
            if tmpdir:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)

    def read_frame(self, ctx, name, path):
        """One pentad -> (float32 [h, w] with NaN, counts, out_of_bounds)."""
        with TIF_LOCK:
            a = read_tif(path, self.w, self.h)
        n_fill = int((a == np.float32(FILL)).sum())
        a[a == np.float32(FILL)] = np.nan
        a[~np.isfinite(a)] = np.nan
        v = a.reshape(-1, 1).astype(np.float64)
        oob = self.mask_bounds(v)
        a = v.reshape(a.shape).astype(np.float32)
        counts = {"files_read": 1, "fill_pixels": n_fill,
                  "valid_pixels": int(np.isfinite(a).sum())}
        return a, counts, oob

    # ------------------------------------------------------- the contract --
    def classify(self, ctx, b, f):
        """-> (pentad key or None, reason or None) for one slot."""
        files, _ = self.listing(ctx)
        lo, hi = min(files), max(files)
        key = self.slots().get((int(b), int(f)))
        if key is None:
            # no pentad's midpoint lands in this half-bin: about half of all
            # slots, and not an absence. Still classify the ends honestly.
            t = sh.frame_start_seconds(b, f, self.frame_seconds)
            d0, _ = pentad_days(*lo)
            _, d1 = pentad_days(*hi)
            if t < f10b.seconds_since_epoch(d0):
                return None, "before_record"
            if t > f10b.seconds_since_epoch(d1) + 86399:
                return None, "after_record"
            return None, "no_pentad_in_slot"
        if key < lo:
            return None, "before_record"
        if key > hi:
            return None, "after_record"
        if key not in files:
            return None, "absent_upstream"
        return key, None

    def fetch_frames(self, ctx, wanted):
        files, _ = self.listing(ctx)
        jobs = []
        for (g, b, f) in wanted:
            key, why = self.classify(ctx, b, f)
            jobs.append((g, b, f, key, why))
        todo = [j for j in jobs if j[4] is None]

        def work(job):
            g, b, f, key, why = job
            tmpdir = None if ctx.source_dir else \
                tempfile.mkdtemp(prefix="chirps_", dir=ctx.scratch)
            p, n = self._get(ctx, files[key][0], tmpdir)
            return job, p, n, tmpdir

        workers = 1 if ctx.source_dir else self.WORKERS
        it = cm.ordered_map(work, todo, workers, lookahead=2 * workers)
        t0 = time.time()
        for (g, b, f, key, why) in jobs:
            if why is not None:
                yield g, b, f, None, {"frame_missing": why}
                continue
            job, p, n, tmpdir = next(it)
            if job[:3] != (g, b, f):
                sys.exit(f"chirps05: the download pool answered {job[:3]} "
                         f"where {(g, b, f)} was asked for")
            try:
                if p is None:
                    ctx.note_absent(f"{key[0]:04d}-{key[1]:02d} pentad "
                                    f"{key[2]}", n)
                    continue
                try:
                    arr, counts, oob = self.read_frame(ctx, files[key][0], p)
                except FormatError as e:
                    sys.exit(f"REFUSING chirps05: {files[key][0]}: {e}")
                except (IOError, OSError) as e:
                    ctx.note_absent(f"{key[0]:04d}-{key[1]:02d} pentad "
                                    f"{key[2]}", f"{type(e).__name__}: {e}")
                    continue
            finally:
                if tmpdir:
                    import shutil
                    shutil.rmtree(tmpdir, ignore_errors=True)
            d0, d1 = pentad_days(*key)
            counts["bytes_files"] = n
            counts["pentad_days"] = {str((d1 - d0).days + 1): 1}
            if oob:
                counts["out_of_bounds"] = oob
            counts["fetch_seconds"] = round(time.time() - t0, 2)
            t0 = time.time()
            yield g, b, f, arr, counts

    # --------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260918):
        truth = make_smoke_sources(root, d_lo, d_hi, seed)
        self.__init__()
        return truth


# ============================================================== GeoTIFF ====
def _rasterio():
    try:
        import rasterio
    except ImportError:                                     # pragma: no cover
        sys.exit("chirps05 needs the `rasterio` package (pip install "
                 "rasterio) — it is in the family1-build workflow's install "
                 "step")
    return rasterio


def tif_check(d, w, h):
    """An open rasterio dataset against the declared grid. Raises.

    The GeoTIFFs store their transform as float32, so 7,200 x dx reads
    360.000005 degrees and the comparison is relative, not exact.
    """
    dx, dy = 360.0 / int(w), 120.0 / int(h)
    if (d.height, d.width) != (int(h), int(w)):
        raise FormatError(f"{d.height} x {d.width}, expected {h} x {w}")
    if d.count != 1 or d.dtypes[0] != "float32":
        raise FormatError(f"{d.count} band(s) of {d.dtypes}, expected one "
                          f"float32 band")
    t = d.transform
    want = (dx, 0.0, X0, 0.0, -dy, Y0)
    got = (t.a, t.b, t.c, t.d, t.e, t.f)
    if not all(abs(g - v) <= 1e-6 * max(1.0, abs(v))
               for g, v in zip(got, want)):
        raise FormatError(f"transform {got}, the declared grid is {want}")
    if d.crs is None or d.crs.to_epsg() != 4326:
        raise FormatError(f"CRS {d.crs}, expected EPSG:4326")


def read_tif(path, w, h):
    """One pentad GeoTIFF -> float32 [h, w], grid VERIFIED. Fill kept."""
    rasterio = _rasterio()
    try:
        d = rasterio.open(path)
    except Exception as ex:                                 # noqa: BLE001
        raise IOError(f"{os.path.basename(path)}: not a readable GeoTIFF "
                      f"({ex}) — a truncated download") from None
    try:
        tif_check(d, w, h)
        return np.array(d.read(1), dtype=np.float32)
    finally:
        d.close()


def tif_meta(path, w, h):
    rasterio = _rasterio()
    d = rasterio.open(path)
    try:
        tif_check(d, w, h)
        a = d.read(1)
        fill = a == np.float32(FILL)
        good = a[~fill]
        return {"height": d.height, "width": d.width, "dtype": d.dtypes[0],
                "crs": str(d.crs), "nodata_declared": d.nodatavals[0],
                "transform": [d.transform.a, d.transform.b, d.transform.c,
                              d.transform.d, d.transform.e, d.transform.f],
                "bounds": [d.bounds.left, d.bounds.bottom, d.bounds.right,
                           d.bounds.top],
                "compress": str(d.profile.get("compress")),
                "block_shapes": [list(b) for b in d.block_shapes],
                "fill_pixels": int(fill.sum()),
                "valid_fraction": round(float((~fill).mean()), 6),
                "min": (float(good.min()) if good.size else None),
                "max": (float(good.max()) if good.size else None)}
    finally:
        d.close()


def write_tif(path, a, w, h):
    rasterio = _rasterio()
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", height=int(h), width=int(w),
                       count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(X0, Y0, 360.0 / int(w),
                                             120.0 / int(h)),
                       compress="lzw") as d:
        d.write(a.astype(np.float32), 1)


# ================================================================== smoke ==
SMOKE_W, SMOKE_H = 600, 200
# NO ABSENT PENTAD in the main smoke, and that is a property of the source:
# `index` REFUSES a hole between the listing's first and last pentad, and the
# real listing has none (3,288 files, no gap), so `absent_upstream` is
# unreachable for this store by construction. A LISTED file that will not
# download or does not parse is `ctx.note_absent` and stops the year's
# marker -- `skip=` and the truncated-file test exercise both.
SMOKE_ABSENT = None
SMOKE_OOB = (1981, 1, 3)           # one pixel of 9,000 mm


def smoke_field(y, m, p, w, h):
    """A deterministic pentad: a dry ocean fill and a wet land band."""
    k = (m * 6 + p) % 13
    yy = (np.arange(h, dtype=np.float64)[:, None] - h / 2) / h
    xx = (np.arange(w, dtype=np.float64)[None, :] - w / 2) / w
    a = np.full((h, w), FILL, np.float32)
    land = (np.abs(yy) < 0.30) & (np.abs(xx) < 0.40)
    rain = np.clip(40.0 + 60.0 * np.sin(9 * xx + k) * np.cos(7 * yy - k),
                   0.0, None)
    a[land] = rain[land].astype(np.float32)
    if (y, m, p) == SMOKE_OOB:
        a[h // 2, w // 2] = 9000.0
    return a


def listing_html(entries):
    rows = ['<tr><td class="link"><a href="../">Parent directory/</a></td>'
            '<td class="size">-</td><td class="date">-</td></tr>']
    for name, size in entries:
        rows.append(f'<tr><td class="link"><a href="{name}" title="{name}">'
                    f'{name}</a></td><td class="size">{size}</td>'
                    f'<td class="date">2026-Sep-11 20:57</td></tr>')
    return ("<!DOCTYPE html><html><head><title>Index of "
            "/products/CHIRPS/v3.0/pentads/global/tifs/</title></head><body>"
            "<h1>Index of /products/CHIRPS/v3.0/pentads/global/tifs/</h1>"
            '<table id="list"><tbody>\n' + "\n".join(rows) +
            "\n</tbody></table></body></html>\n")


def make_smoke_sources(root, d_lo, d_hi, seed=20260918, skip=()):
    """The archive in its real layout: `tifs/index.html` in the server's own
    table format and the pentad GeoTIFFs themselves on a
    `CHIRPS05_SMOKE_GRID` grid. `d_lo` .. `d_hi` bounds the pentads written;
    SMOKE_ABSENT is LISTED and not written (a download that 404s), and any
    key in `skip` is neither listed nor written.

    Returns the truth: {(group, bin, frame): (float16 [h, w, 1] or None,
    reason or None)} for every frame of every bin overlapping the record.
    """
    os.environ["CHIRPS05_SMOKE_GRID"] = f"{SMOKE_W},{SMOKE_H}"
    w, h = SMOKE_W, SMOKE_H
    base = os.path.join(root, "chirps05", "tifs")
    os.makedirs(base, exist_ok=True)
    ad = Chirps05Adapter()
    ad.w, ad.h = w, h
    keys = [(y, m, p) for (y, m, p) in all_pentads(d_lo.year, d_hi.year)
            if d_lo <= pentad_days(y, m, p)[0] and
            pentad_days(y, m, p)[1] <= d_hi and (y, m, p) not in skip]
    entries, fields = [], {}
    for (y, m, p) in keys:
        name = f"chirps-{VERSION}.{y:04d}.{m:02d}.{p}.tif"
        a = smoke_field(y, m, p, w, h)
        fields[(y, m, p)] = a
        if (y, m, p) != SMOKE_ABSENT:
            write_tif(os.path.join(base, name), a, w, h)
            size = f"{os.path.getsize(os.path.join(base, name)) / 1024:.1f} KiB"
        else:
            size = "18.0 MiB"          # listed, and the file is not there
        entries.append((name, size))
    with open(os.path.join(base, "index.html"), "w") as fh:
        fh.write(listing_html(entries))

    lo, hi = min(keys), max(keys)
    fs = ad.frame_seconds
    slots = slot_map(fs, d_lo.year, d_hi.year + 1)
    t_lo = f10b.seconds_since_epoch(pentad_days(*lo)[0])
    t_hi = f10b.seconds_since_epoch(pentad_days(*hi)[1]) + 86399
    truth = {}
    for b in sh.bins_overlapping(t_lo, t_hi):
        for f in range(ad.frames_per_bin):
            key = slots.get((b, f))
            t = sh.frame_start_seconds(b, f, fs)
            if key is None:
                why = ("before_record" if t < t_lo else
                       "after_record" if t > t_hi else "no_pentad_in_slot")
                truth[(ad.store, b, f)] = (None, why)
                continue
            if key < lo:
                truth[(ad.store, b, f)] = (None, "before_record")
            elif key > hi:
                truth[(ad.store, b, f)] = (None, "after_record")
            elif key == SMOKE_ABSENT or key in skip:
                truth[(ad.store, b, f)] = (None, "absent_upstream")
            else:
                a = fields[key].astype(np.float64)
                a[a == FILL] = np.nan
                a[(a < 0) | (a > 5000)] = np.nan
                truth[(ad.store, b, f)] = (
                    a.astype(np.float16)[:, :, None], None)
    return truth


ADAPTER = Chirps05Adapter
