"""ESA Ocean Colour CCI v6.0 daily 4 km — chlorophyll, water clarity and how
many satellites saw each pixel (family 1.gf, E-082 wave 2).

PLAIN ENGLISH. The colour of the sea says how much phytoplankton is in it, and
phytoplankton is where the ocean's carbon uptake starts. ESA's Ocean Colour
CCI merges every civilian ocean-colour sensor since SeaWiFS — SeaWiFS, MERIS,
MODIS-Aqua, VIIRS, OLCI-A and OLCI-B — into one daily 4 km field on a plain
geographic grid, band-shifted and inter-calibrated so the record is a
CLIMATE record rather than six instrument records side by side. This store
keeps the daily fields whole, tiled and compressed, so a model can read the
bloom near the place it predicts at the resolution it was measured.

SOURCE, VERIFIED 2026-09-18 FROM THIS SANDBOX (no account; HTTPS):
  https://dap.ceda.ac.uk/neodc/esacci/ocean_colour/data/v6.0-release/
      geographic/netcdf/chlor_a/daily/v6.0/<YYYY>/
        ESACCI-OC-L3S-CHLOR_A-MERGED-1D_DAILY_4km_GEO_PML_OCx-<YYYYMMDD>-fv6.0.nc
      geographic/netcdf/kd/daily/v6.0/<YYYY>/
        ESACCI-OC-L3S-K_490-MERGED-1D_DAILY_4km_GEO_PML_KD490_Lee-<YYYYMMDD>-fv6.0.nc
  83 MB each, measured on 2015-01-15.

WHAT THE RECORD REALLY IS, AND WHERE THE LEDGER IS WRONG. `family1gf.tex`'s
row says "1997-09-04 -> 2026-06-30, quarterly updates". The v6.0-release tree
on CEDA holds years 1997 .. 2022 and its 2022 directory ends
`...-20221231-fv6.0.nc`: the daily record available here STOPS AT
2022-12-31, and `v6.0-release` is the newest release directory under
`ocean_colour/data/` (the others are v1 .. v5.0). The store's record is
therefore the listing's, measured at index time, and `index` prints both ends
so the disagreement is visible rather than inherited.

WHAT IS STORED, AND WHY NOT THE BRIEF'S NINE CHANNELS. One group, dtype
FLOAT16, C = 3 — the note's own phase A ("3: log_chl, kd_490, nobs; +6 Rrs
bands in phase B"):

  log_chl    log10 of chlorophyll-a in mg m-3. STORED AS THE LOGARITHM on
             purpose: chlorophyll spans 0.001 to 100 mg m-3 (measured on
             2015-01-15: 0.00105 .. 99.76), and float16 has ten mantissa
             bits, so the raw value would be spaced 0.0625 mg m-3 apart at
             100 — a 6 % error on the top of the range and useless below
             0.01. log10 puts the whole range in [-3, 2], where float16
             spaces values 0.001 apart. tile_grid.json says so in the
             channel's unit, and a reader raises 10 to the power.
  kd_490     the diffuse attenuation coefficient at 490 nm, m-1: how fast
             blue-green light dies with depth. 0.011 .. 9.68 measured.
  total_nobs the number of sensor observations merged into the pixel that
             day, 0.11 .. 197.9 measured (it is a float in the source, not
             an integer count). Exact in float16 to 2,048.

The BRIEF asked for chlor_a, six Rrs bands, kd_490 and a water-class/flags
uint8 channel. Three measurements say no, for now, and every one of them is a
byte count rather than an opinion. The six Rrs bands and the fourteen
`water_classN` membership fields live in ONE file,
`...-RRS-MERGED-1D_DAILY_4km_GEO_PML_RRS-<date>-fv6.0.nc`, which is 521 MB a
day against chlor_a's 83 and kd's 83: adding them takes the fetch from 166 MB
a day to 687 MB, i.e. from 1.54 TB to 6.35 TB over the 1997-2022 record. The
all-in-one `all_products` file is 1.3-1.7 GB a day (17 TB). And a uint8
water class cannot share a float16 group at all — the 14 memberships would
have to become an argmax, which is a DERIVED field this family does not make
up. So Rrs and the water class are phase B, exactly as the note has them,
and the note's C = 3 is what is built.

THE GRID, MEASURED ON 2015-01-15 AND CHECKED IN EVERY FILE: 4,320 rows x
8,640 columns at 1/24 degree (4.64 km at the equator), plain lat/lon.
`lat` DESCENDS, 89.97917 .. -89.97917, and `lon` ascends from -179.97917 —
both are PIXEL CENTRES, so the grid's north and west EDGES are exactly +90
and -180 and ROW 0 IS THE NORTHERNMOST. `chlor_a` and `kd_490` are float32
with `_FillValue` 9.96921e36 (netCDF's default float fill) over land, cloud,
night and ice; on 2015-01-15, 10.657 % of the grid carries a value, which is
the `v ~ 0.10-0.15` the note assumes. The per-sensor `<SENSOR>_nobs`
variables carry fill 0.0 and are not stored.

A FRAME THAT IS NOT IN THE SOURCE is a zero-length frame with a reason:
`before_record` (before the first listed day), `after_record` (after the
last), `absent_upstream` (inside the record, no file). A calendar month
inside the record with no file at all is REFUSED (`note_absent`), because a
month-long hole is a listing problem until proven otherwise; so is a year
inside the record missing from the top listing. A listed file that will not
download or does not parse is an absence, never a None frame.

CEDA IS SLOW FROM HERE AND THE PROBE SAYS SO. The kd file came down at
2.0 MB/s from this sandbox, so a day costs about 80 s of network and the
whole record about 1.5 TB; the probe measures the real rate on the runner
that will do the fetching, and the lanes are sized from THAT number, one
year each.
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

RELEASE = "v6.0-release"
VERSION = "v6.0"
BASE = ("https://dap.ceda.ac.uk/neodc/esacci/ocean_colour/data/"
        f"{RELEASE}/geographic/netcdf/")
H, W = 4320, 8640
DX = 1.0 / 24.0
X0, Y0 = -180.0, 90.0          # the grid's WEST and NORTH edges
# The netCDF default float fill, which is what both products use over land,
# cloud, night and ice.
NC_FILL = 9.969209968386869e36

# product -> (directory, file-name middle, the variable this store keeps)
PRODUCTS = {
    "chlor_a": ("chlor_a", "CHLOR_A-MERGED-1D_DAILY_4km_GEO_PML_OCx",
                ("chlor_a", "total_nobs")),
    "kd": ("kd", "K_490-MERGED-1D_DAILY_4km_GEO_PML_KD490_Lee",
           ("kd_490",)),
}
CHLA_FLOOR = 1e-4              # below this, log10 is not a measurement


def to_frame(chl, kd, nobs, bounds=None):
    """(chlor_a, kd_490, total_nobs) as the source gives them, with the fill
    already NaN -> the stored frame [h, w, 3] and its counts.

    ONE numeric path, called by `read_day` and by the smoke's truth, so the
    two cannot disagree by a rounding: everything stays float32 (log10 of a
    float32 rounded to float16 is not always log10 of the float64 rounded to
    float16) and only the bounds test is widened.
    """
    chl = np.asarray(chl, np.float32).copy()
    kd = np.asarray(kd, np.float32).copy()
    nobs = np.asarray(nobs, np.float32).copy()
    low = np.isfinite(chl) & (chl <= np.float32(CHLA_FLOOR))
    n_low = int(low.sum())
    chl[low] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        log_chl = np.log10(chl)
    log_chl[~np.isfinite(log_chl)] = np.nan
    arr = np.stack([log_chl, kd, nobs], axis=-1)
    oob = {}
    if bounds is not None:
        lo_b, hi_b = bounds
        with np.errstate(invalid="ignore"):
            bad = np.isfinite(arr) & ((arr < lo_b) | (arr > hi_b))
        n = bad.reshape(-1, arr.shape[-1]).sum(axis=0)
        arr[bad] = np.nan
        oob = {nm: int(k) for nm, k in
               zip([c[0] for c in OC4kAdapter.channels], n) if k}
    return arr, n_low, oob

YEAR_DIR = re.compile(r'href="(\d{4})/"')
ANY_HREF = re.compile(r'href="([^"?/][^"]*)"')

# HDF5 under netCDF4 is NOT thread-safe (seaice_asi measured a SIGBUS with
# four workers each opening a file). Downloads stay parallel; every open,
# read and close holds this lock.
NC_LOCK = threading.Lock()


class FormatError(ValueError):
    """A listing or a file that does not look like the archive measured."""


def file_re(prod):
    mid = PRODUCTS[prod][1]
    return re.compile(r'href="(ESACCI-OC-L3S-' + re.escape(mid) +
                      r'-(\d{8})-fv' + re.escape(VERSION.lstrip("v")) +
                      r'\.nc)"')


def object_name(prod, d):
    return (f"ESACCI-OC-L3S-{PRODUCTS[prod][1]}-{d:%Y%m%d}-"
            f"fv{VERSION.lstrip('v')}.nc")


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
                      "on the real 2015-01-15 files); col 0 is the "
                      "westernmost, lon -179.97917"),
        "extent": [-180.0, -90.0, 180.0, 90.0],
        "extent_note": "[west, south, east, north] in degrees; ocean only",
        "source_variables": ("chlor_a (float32 mg m-3, stored as log10), "
                             "kd_490 (float32 m-1) and total_nobs (float32), "
                             "with the netCDF default float fill over land, "
                             "cloud, night and ice"),
        "no_tile_corners": ("a plain geographic grid: the affine rule above "
                            "IS the lat/lon of every pixel and every tile "
                            "corner, so no corner table is stored"),
    }


def parse_years(html):
    return sorted({int(y) for y in YEAR_DIR.findall(html)})


def parse_year_listing(html, prod, year):
    """-> ({date: name}, counts). REFUSES a date outside the year."""
    out, counts = {}, {}
    rx = file_re(prod)
    for name, ymd in rx.findall(html):
        d = dt.date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:]))
        if d.year != year:
            raise FormatError(f"{name} in the {year} listing")
        if d in out:
            raise FormatError(f"{name}: two files for {d}")
        out[d] = name
    listed = {n for n in ANY_HREF.findall(html)
              if not n.startswith(("http", ".")) and "/" not in n}
    other = sorted(listed - set(out.values()))
    if other:
        counts["files_other"] = len(other)
        counts["files_other_names"] = other[:20]
    return out, counts


def month_of(d):
    return f"{d.year:04d}-{d.month:02d}"


class OC4kAdapter(sh.GridAdapter):
    store = "oc4k"
    title = ("Ocean colour, ESA OC-CCI v6.0 merged L3S, daily 4 km "
             "geographic: log10 chlorophyll-a, Kd(490) and the merged "
             "observation count")
    family = "1gf"
    distribution = "public"
    licence = {
        "name": "ESA CCI Data Policy: free and open access, cite",
        "redistribution": "attribution",
        "redistribution_confirmed": True,
        "derived_works": "free",
        "attribution": ("Ocean Colour Climate Change Initiative dataset, "
                        "Version 6.0, European Space Agency, available "
                        "online at http://www.oceancolour.org/. Produced by "
                        "Plymouth Marine Laboratory; distributed by CEDA "
                        "(catalogue uuid 0875b4675f1e46ebadb526e0b95505c5)."),
        "terms": ("the files' own `license` attribute, read 2026-09-18: 'ESA "
                  "CCI Data Policy: free and open access. When referencing, "
                  "please use: Ocean Colour Climate Change Initiative "
                  "dataset, Version <Version Number>, European Space "
                  "Agency, available online at ...'. CEDA's "
                  "00README_catalogue_and_licence.txt adds "
                  "esacci_oc_terms_and_conditions.pdf and says access is "
                  "public to registered and non-registered users alike"),
    }
    time_dtype = "int32"
    credentials = ()
    channels = (
        ("log_chl", "log10(mg m-3) — the base-10 LOGARITHM of chlorophyll-a, "
                    "not the concentration", -4.0, 2.5),
        ("kd_490", "m-1", 0.0, 20.0),
        ("total_nobs", "count (a float in the source)", 0.0, 2048.0),
    )
    log2_fp = float(np.log2(360.0 / W * 111.31949 / 27.83))       # -2.583
    log2_dt = float(np.log2(1.0 / 5.0))                           # -2.322
    per_year = True
    first_year = 1997
    frames_per_bin = 5
    frame_seconds = 86400
    dtype = "float16"
    tile = sh.TILE
    zstd_level = sh.DEFAULT_LEVEL
    grid = grid_of()
    note_estimate = {"bytes": 3.5e11,
                     "what": "family1gf.tex storage table: '0.3-0.4 TB "
                             "(C = 3); ~1.1 TB with Rrs' for 37.3 M px x "
                             "10,500 d at v ~ 0.10-0.15"}
    qc_policy = (
        "the only per-pixel flag in these two products is the fill value: "
        "chlor_a, kd_490 and total_nobs all carry netCDF's default float "
        "fill (9.96921e36) over land, cloud, night and ice, and that becomes "
        "NaN. A chlorophyll at or below 1e-4 mg m-3 is NaN too, because "
        "log10 of it is not a measurement, and it is counted "
        "(`chl_below_floor`). A value outside its channel's bounds becomes "
        "NaN and is counted (`out_of_bounds`), never clipped. The source's "
        "own per-pixel uncertainties (chlor_a_log10_bias / _rmsd, "
        "kd_490_bias / _rmsd) and the six per-sensor nobs fields are in the "
        "same files and are NOT stored: they are phase B, with the Rrs "
        "bands")
    sources = tuple(
        BASE + f"{PRODUCTS[p][0]}/daily/{VERSION}/<YYYY>/" + f"ESACCI-OC-L3S-"
        f"{PRODUCTS[p][1]}-<YYYYMMDD>-fv{VERSION.lstrip('v')}.nc"
        for p in ("chlor_a", "kd"))
    verified = (
        "2026-09-18 from the sandbox: the release tree "
        "(v6.0-release/geographic/netcdf/{all_products,chlor_a,iop,kd,rrs}), "
        "the chlor_a and kd daily year listings (1997..2022, 365 files in "
        "2015, 366 entries listed including the parent link, and 2022 ending "
        "20221231), and the chlor_a, kd and rrs files of 2015-01-15 opened "
        "with netCDF4: 4320 x 8640, lat DESCENDING 89.97917..-89.97917, lon "
        "-179.97917.., chlor_a float32 mg m-3 fill 9.96921e36 with 10.657 % "
        "valid and values 0.00105..99.758, total_nobs 0.111..197.89, kd_490 "
        "0.01145..9.6808, and the rrs file's 6 Rrs bands + 14 water_classN "
        "fields at 521 MB a day against chlor_a's 83")
    notes = (
        "C = 3, the note's phase A. `log_chl` is the base-10 LOGARITHM of "
        "chlorophyll-a: the concentration spans 0.001-100 mg m-3 and float16 "
        "would space it 0.0625 apart at the top of that range, while log10 "
        "puts it all in [-3, 2] at 0.001 spacing. The six Rrs bands and the "
        "14 water-class memberships are phase B because they live in a "
        "521 MB-a-day file against chlor_a's 83, which would take the fetch "
        "from 1.54 TB to 6.35 TB. THE DAILY RECORD ON CEDA ENDS 2022-12-31, "
        "not the ledger's 2026-06-30; `index` measures and prints both ends.")
    smoke_window = ("1997-09-04", "1997-09-18")
    smoke_probe_month = "1997-09"

    WORKERS = 4

    def __init__(self):
        g = (os.environ.get("OC4K_SMOKE_GRID") or "").split(",")
        self.w, self.h = (int(g[0]), int(g[1])) if len(g) == 2 else (W, H)
        self._years = {}
        self._ylist = {}
        if (self.w, self.h) != (W, H):
            self.notes = (f"{self.notes}\nSMOKE GRID: OC4K_SMOKE_GRID set the "
                          f"grid to {self.w} x {self.h} instead of {W} x {H}. "
                          f"This is a synthetic store.")

    def group_grids(self):
        return {self.store: grid_of(self.h, self.w)}

    # ------------------------------------------------------------ listings --
    def _url(self, prod, *parts):
        return BASE + "/".join((PRODUCTS[prod][0], "daily", VERSION) + parts)

    def _local(self, ctx, prod, *parts):
        return os.path.join(ctx.source_dir, self.store, PRODUCTS[prod][0],
                            "daily", VERSION, *parts)

    def _page(self, ctx, prod, *parts):
        if ctx.source_dir:
            p = self._local(ctx, prod, *parts, "index.html")
            if not os.path.exists(p):
                return None
            with open(p, "rb") as fh:
                raw = fh.read()
            ctx.count_bytes(len(raw))
            return raw.decode("utf-8", "replace")
        raw, why = cm.get_bytes(self._url(prod, *parts) + "/",
                                attempts=ctx.a.attempts)
        if raw is None:
            return None
        return raw.decode("utf-8", "replace")

    def years(self, ctx, prod):
        if prod not in self._years:
            ys = parse_years(self._page(ctx, prod) or "")
            if not ys:
                sys.exit(f"REFUSING oc4k: an empty listing at "
                         f"{self._url(prod)}/ — no year directories (the "
                         f"2026-09-14 rule: an empty listing is a refusal)")
            self._years[prod] = ys
        return self._years[prod]

    def year_listing(self, ctx, prod, y):
        key = (prod, y)
        if key not in self._ylist:
            html = self._page(ctx, prod, f"{y:04d}")
            if html is None:
                self._ylist[key] = None
            else:
                try:
                    self._ylist[key] = parse_year_listing(html, prod, y)
                except FormatError as e:
                    sys.exit(f"REFUSING oc4k: {prod} {y} listing: {e}")
        return self._ylist[key]

    def record(self, ctx, prod):
        ys = self.years(ctx, prod)
        lo = self.year_listing(ctx, prod, ys[0])
        hi = self.year_listing(ctx, prod, ys[-1])
        if not lo or not lo[0] or not hi or not hi[0]:
            sys.exit(f"REFUSING oc4k: the {prod} listing of {ys[0]} or "
                     f"{ys[-1]} is empty — cannot place the record's ends")
        return min(lo[0]), max(hi[0])

    def common_record(self, ctx):
        """The days BOTH products publish — the store's record."""
        a, b = self.record(ctx, "chlor_a"), self.record(ctx, "kd")
        return max(a[0], b[0]), min(a[1], b[1])

    def days_of(self, ctx, y):
        """{date: {prod: name}} for the days BOTH products have in year y."""
        out = {}
        per = {}
        for prod in PRODUCTS:
            yl = self.year_listing(ctx, prod, y)
            per[prod] = (yl[0] if yl else {})
        for d, name in per["chlor_a"].items():
            if d in per["kd"]:
                out[d] = {"chlor_a": name, "kd": per["kd"][d]}
        return out, per

    def record_frames(self, ctx, group):
        lo, hi = self.common_record(ctx)
        n = 0
        for y in range(lo.year, hi.year + 1):
            days, _ = self.days_of(ctx, y)
            n += sum(1 for d in days if lo <= d <= hi)
        return n

    def empty_months(self, ctx, y):
        lo, hi = self.common_record(ctx)
        days, _ = self.days_of(ctx, y)
        have = {month_of(d) for d in days}
        out = []
        for m in range(1, 13):
            m0 = dt.date(y, m, 1)
            m1 = dt.date(y, m, calendar.monthrange(y, m)[1])
            if m1 < lo or m0 > hi:
                continue
            if f"{y:04d}-{m:02d}" not in have:
                out.append(f"{y:04d}-{m:02d}")
        return out

    def index(self, ctx):
        out = {"dataset": f"ESA OC-CCI {VERSION} merged L3S daily 4 km "
                          f"geographic ({RELEASE})",
               "url": BASE, "version": VERSION, "release": RELEASE,
               "products": {}}
        for prod in PRODUCTS:
            ys = self.years(ctx, prod)
            first, last = self.record(ctx, prod)
            per, other = {}, {}
            for y in ys:
                yl = self.year_listing(ctx, prod, y)
                files, counts = yl if yl else ({}, {})
                per[str(y)] = len(files)
                if counts:
                    other[str(y)] = counts
            out["products"][prod] = {
                "listing": self._url(prod) + "/",
                "years": ys, "record": [str(first), str(last)],
                "files_per_year": per, "files": sum(per.values()),
                "other_files": other}
        lo, hi = self.common_record(ctx)
        absent = []
        for y in range(lo.year, hi.year + 1):
            days, _ = self.days_of(ctx, y)
            d = max(lo, dt.date(y, 1, 1))
            e = min(hi, dt.date(y, 12, 31))
            while d <= e:
                if d not in days:
                    absent.append(str(d))
                d += dt.timedelta(days=1)
        empty = {str(y): self.empty_months(ctx, y) for y in ctx.years
                 if self.empty_months(ctx, y)}
        if empty:
            sys.exit(f"REFUSING oc4k: a calendar month inside the record has "
                     f"no file in the listing: {empty}")
        for prod in PRODUCTS:
            ys = self.years(ctx, prod)
            gone = [y for y in ctx.years if ys[0] <= y <= ys[-1]
                    and y not in ys]
            if gone:
                sys.exit(f"REFUSING oc4k: {prod} year(s) {gone} inside the "
                         f"record are not in the listing")
        out["record"] = [str(lo), str(hi)]
        out["days_in_both_products"] = self.record_frames(ctx, self.store)
        out["days_absent_in_record"] = absent
        out["ledger_says"] = ("family1gf.tex: '1997-09-04 -> 2026-06-30, "
                              "quarterly updates'. MEASURED HERE: the daily "
                              f"record of {RELEASE} ends {hi}")
        out["grid"] = grid_of(self.h, self.w)
        # ONE REAL DAY, both files read and checked against the declared grid
        want = [y for y in ctx.years if lo.year <= y <= hi.year] or [hi.year]
        days, _ = self.days_of(ctx, want[0])
        d = min(d for d in days if lo <= d <= hi)
        out["first_file"] = {"day": str(d), "names": days[d],
                             **self.read_first_day(ctx, d, days[d])}
        return out

    # ------------------------------------------------------------ one day ---
    def fetch_day(self, ctx, d, names, tmpdir):
        """Both products' files on local disk -> ({prod: path}, bytes) or
        (None, why)."""
        paths, nbytes = {}, 0
        for prod, name in names.items():
            if ctx.source_dir:
                p = self._local(ctx, prod, f"{d.year:04d}", name)
                if not os.path.exists(p):
                    return None, f"{name}: listed and absent"
                ctx.count_bytes(os.path.getsize(p))
            else:
                p = os.path.join(tmpdir, name)
                if f10b.fetch_first([self._url(prod, f"{d.year:04d}", name)],
                                    p, attempts=ctx.a.attempts) is None:
                    return None, f"{name}: listed, and 404"
            nbytes += os.path.getsize(p)
            paths[prod] = p
        return {"paths": paths, "bytes": nbytes}, None

    def read_day(self, ctx, d, names):
        """One day -> (float32 [h, w, 3] with NaN, counts, out_of_bounds).

        Channel 0 is log10(chlor_a) — NOT the concentration; the floor below
        which the logarithm is not a measurement is `CHLA_FLOOR` and every
        pixel under it is NaN and COUNTED.
        """
        tmpdir = None if ctx.source_dir else \
            tempfile.mkdtemp(prefix="occci_", dir=ctx.scratch)
        try:
            got, why = self.fetch_day(ctx, d, names, tmpdir)
            if got is None:
                raise IOError(why)
            with NC_LOCK:
                (chl, nobs), m1 = read_nc(got["paths"]["chlor_a"],
                                          ("chlor_a", "total_nobs"),
                                          self.h, self.w)
                (kd,), m2 = read_nc(got["paths"]["kd"], ("kd_490",),
                                    self.h, self.w)
            nbytes = got["bytes"]
        finally:
            if tmpdir:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
        arr, n_low, oob = to_frame(chl, kd, nobs, bounds=self.bounds())
        counts = {"files_read": len(names), "bytes_files": nbytes,
                  "valid_pixels": int(np.isfinite(arr[:, :, 0]).sum()),
                  "product_version": {str(m1.get("product_version")): 1}}
        if n_low:
            counts["chl_below_floor"] = n_low
        return arr, counts, oob

    def read_first_day(self, ctx, d, names):
        """`index`'s one real read: both files opened, the grid CHECKED."""
        try:
            arr, counts, oob = self.read_day(ctx, d, names)
        except FormatError as e:
            sys.exit(f"REFUSING oc4k: the first day {d} does not match the "
                     f"declared grid: {e}")
        except (IOError, OSError) as e:
            sys.exit(f"REFUSING oc4k: the first day {d} could not be read: "
                     f"{e}")
        out = dict(counts)
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
        """-> ({prod: name} or None, reason or None) for one day."""
        lo, hi = self.common_record(ctx)
        if d < lo:
            return None, "before_record"
        if d > hi:
            return None, "after_record"
        for prod in PRODUCTS:
            ys = self.years(ctx, prod)
            if d.year not in ys:
                return None, "year_not_listed"
        days, _ = self.days_of(ctx, d.year)
        hit = days.get(d)
        if hit is None:
            return None, "absent_upstream"
        return hit, None

    def fetch_frames(self, ctx, wanted):
        refused = set()
        jobs = []
        for (g, b, f) in wanted:
            d = sh.frame_day(b, f, self.frame_seconds)
            names, why = self.classify(ctx, d)
            if why == "year_not_listed":
                if d.year not in refused:
                    ctx.note_absent(str(d.year),
                                    f"{d.year} is inside the record and not "
                                    f"in both products' listings")
                    refused.add(d.year)
                continue
            if names is None and why == "absent_upstream":
                em = self.empty_months(ctx, d.year)
                if month_of(d) in em:
                    if month_of(d) not in refused:
                        ctx.note_absent(month_of(d),
                                        f"the listing of {d.year} has no "
                                        f"file in {month_of(d)} — an empty "
                                        f"month is refused")
                        refused.add(month_of(d))
                    continue
            jobs.append((g, b, f, d, names, why))

        def work(job):
            g, b, f, d, names, why = job
            if names is None:
                return job, None, None
            try:
                return job, self.read_day(ctx, d, names), None
            except FormatError as e:
                return job, None, ("REFUSE", str(e))
            except (IOError, OSError) as e:
                return job, None, (None, f"{type(e).__name__}: {e}")

        workers = 1 if ctx.source_dir else self.WORKERS
        t0 = time.time()
        for job, res, err in cm.ordered_map(work, jobs, workers,
                                            lookahead=workers + 1):
            g, b, f, d, names, why = job
            if names is None:
                yield g, b, f, None, {"frame_missing": why}
                continue
            if err is not None:
                if err[0] == "REFUSE":
                    sys.exit(f"REFUSING oc4k: {d}: {err[1]}")
                ctx.note_absent(str(d), err[1])
                continue
            arr, counts, oob = res
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


def read_nc(path, want, h, w):
    """The named variables of one OC-CCI file -> ([arrays], meta).

    Every array is float32 [h, w] with the source's fill turned into NaN, and
    the grid is VERIFIED against the declaration before anything is read.
    """
    try:
        import netCDF4
    except ImportError:                                     # pragma: no cover
        sys.exit("oc4k needs the `netCDF4` package (pip install netCDF4) — "
                 "it is in the family1-build workflow's install step")
    try:
        ds = netCDF4.Dataset(path)
    except OSError as ex:
        raise IOError(f"{os.path.basename(path)}: not a readable netCDF file "
                      f"({ex}) — a truncated download") from None
    try:
        dims = {k: len(v) for k, v in ds.dimensions.items()}
        if dims.get("lat") != int(h) or dims.get("lon") != int(w):
            raise FormatError(f"{os.path.basename(path)}: dimensions {dims}, "
                              f"expected lat={h} lon={w}")
        lat = np.asarray(ds.variables["lat"][:], np.float64)
        lon = np.asarray(ds.variables["lon"][:], np.float64)
        dy, dx = 180.0 / int(h), 360.0 / int(w)
        want_lat = Y0 - (np.arange(int(h)) + 0.5) * dy
        want_lon = X0 + (np.arange(int(w)) + 0.5) * dx
        if not (np.allclose(lat, want_lat, atol=1e-6)
                and np.allclose(lon, want_lon, atol=1e-6)):
            raise FormatError(
                f"{os.path.basename(path)}: lat/lon differ from the declared "
                f"grid (lat {lat[0]}..{lat[-1]}, lon {lon[0]}..{lon[-1]}); "
                f"row 0 is declared NORTHERNMOST")
        out = []
        for name in want:
            if name not in ds.variables:
                raise FormatError(f"{os.path.basename(path)}: no variable "
                                  f"{name!r} ({sorted(ds.variables)[:8]}...)")
            v = ds.variables[name]
            if v.dimensions not in (("time", "lat", "lon"), ("lat", "lon")):
                raise FormatError(f"{os.path.basename(path)}: {name} "
                                  f"dimensions {v.dimensions}")
            v.set_auto_maskandscale(False)
            a = np.array(v[0] if v.dimensions[0] == "time" else v[:],
                         dtype=np.float32)
            fill = np.float32(getattr(v, "_FillValue", NC_FILL))
            a[a == fill] = np.nan
            a[~np.isfinite(a)] = np.nan
            out.append(a)
        meta = {"time_coverage_start": getattr(ds, "time_coverage_start",
                                               None),
                "product_version": getattr(ds, "product_version", None),
                "dims": dims}
        return out, meta
    finally:
        ds.close()


def write_nc(path, h, w, fields):
    """One synthetic OC-CCI file: the real dimensions, axes and fills."""
    import netCDF4
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    try:
        ds.setncattr("title", "ESA CCI Ocean Colour Product (SMOKE)")
        ds.setncattr("product_version", VERSION.lstrip("v"))
        ds.setncattr("license", "ESA CCI Data Policy: free and open access.")
        ds.createDimension("time", 1)
        ds.createDimension("lat", int(h))
        ds.createDimension("lon", int(w))
        dy, dx = 180.0 / int(h), 360.0 / int(w)
        la = ds.createVariable("lat", "f8", ("lat",))
        la.units = "degrees_north"
        la[:] = Y0 - (np.arange(int(h)) + 0.5) * dy
        lo = ds.createVariable("lon", "f8", ("lon",))
        lo.units = "degrees_east"
        lo[:] = X0 + (np.arange(int(w)) + 0.5) * dx
        tv = ds.createVariable("time", "i4", ("time",))
        tv.units = "days since 1970-01-01 00:00:00"
        tv[:] = [0]
        for name, a in fields.items():
            v = ds.createVariable(name, "f4", ("time", "lat", "lon"),
                                  zlib=True, complevel=1,
                                  fill_value=np.float32(NC_FILL))
            # the synthetic field already CARRIES the fill over its land, so
            # the writer must not try to mask it again
            v.set_auto_maskandscale(False)
            v[0, :, :] = a
    finally:
        ds.close()


# ================================================================== smoke ==
SMOKE_W, SMOKE_H = 480, 240
SMOKE_ABSENT = dt.date(1997, 9, 10)     # listed by kd, not by chlor_a
SMOKE_OOB = dt.date(1997, 9, 8)         # one kd_490 of 99 m-1
SMOKE_LOW = dt.date(1997, 9, 9)         # one chlor_a of 1e-6 mg m-3


def smoke_fields(d, h, w):
    """A deterministic day: a filled 'ocean' swath and a NaN 'continent'."""
    k = (d - dt.date(1997, 1, 1)).days
    yy = (np.arange(h, dtype=np.float64)[:, None] - h / 2) / h
    xx = (np.arange(w, dtype=np.float64)[None, :] - w / 2) / w
    r = np.hypot(yy, xx)
    chl = 10.0 ** (-1.0 + 1.6 * np.sin(6 * xx + k * 0.3) * np.cos(5 * yy))
    kd = 0.02 + 0.9 * chl ** 0.6
    nobs = np.rint(1 + 40 * (0.5 + 0.5 * np.sin(3 * xx - k * 0.2))
                   + 0 * yy)
    off = (r > 0.36) | (np.abs(yy + 0.12) < 0.04)      # land / cloud / night
    for a in (chl, kd, nobs):
        a[off] = NC_FILL
    if d == SMOKE_OOB:
        kd[h // 2, w // 2] = 99.0
    if d == SMOKE_LOW:
        chl[h // 2, w // 2 + 1] = 1e-6
    return (chl.astype(np.float32), kd.astype(np.float32),
            nobs.astype(np.float32))


def _listing_html(title, entries, dirs=False):
    rows = []
    for name in entries:
        href = f"{name}/" if dirs else name
        rows.append(f'<tr><td><a href="{href}">{href}</a></td>'
                    f'<td align="right">2023-01-02 05:28</td>'
                    f'<td align="right">83M</td></tr>')
    return ("<!DOCTYPE HTML><html><head><title>Index of " + title +
            "</title></head><body><h1>Index of " + title + "</h1><table>\n"
            '<tr><td><a href="../">Parent Directory</a></td></tr>\n'
            + "\n".join(rows) + "\n</table></body></html>\n")


def make_smoke_sources(root, d_lo, d_hi, seed=20260918, skip=()):
    """Both products' archives in their real layout — a year-directory
    listing, a per-year file listing, and netCDF-4 days on the declared grid
    — under `<root>/oc4k/<product>/daily/v6.0/`.

    SMOKE_ABSENT is listed by `kd` and NOT by `chlor_a`, so the store's
    record (the days BOTH products publish) has a one-day hole inside it and
    the frame is `absent_upstream`. Returns
    {(group, bin, frame): (float16 [h, w, 3] or None, reason or None)}.
    """
    os.environ["OC4K_SMOKE_GRID"] = f"{SMOKE_W},{SMOKE_H}"
    h, w = SMOKE_H, SMOKE_W
    days = [d_lo + dt.timedelta(days=k)
            for k in range((d_hi - d_lo).days + 1)]
    days = [d for d in days if d not in skip]
    fields = {d: smoke_fields(d, h, w) for d in days}
    years = sorted({d.year for d in days})
    for prod in PRODUCTS:
        base = os.path.join(root, "oc4k", PRODUCTS[prod][0], "daily", VERSION)
        os.makedirs(base, exist_ok=True)
        with open(os.path.join(base, "index.html"), "w") as fh:
            fh.write(_listing_html(f"/{PRODUCTS[prod][0]}/daily/{VERSION}",
                                   [str(y) for y in years], dirs=True))
        for y in years:
            yd = os.path.join(base, f"{y:04d}")
            os.makedirs(yd, exist_ok=True)
            names = []
            for d in [x for x in days if x.year == y]:
                if prod == "chlor_a" and d == SMOKE_ABSENT:
                    continue
                chl, kd, nobs = fields[d]
                f = {"chlor_a": {"chlor_a": chl, "total_nobs": nobs},
                     "kd": {"kd_490": kd, "total_nobs": nobs}}[prod]
                name = object_name(prod, d)
                write_nc(os.path.join(yd, name), h, w, f)
                names.append(name)
            names.append("00README_catalogue_and_licence.txt")
            with open(os.path.join(yd, "index.html"), "w") as fh:
                fh.write(_listing_html(
                    f"/{PRODUCTS[prod][0]}/daily/{VERSION}/{y}",
                    sorted(names)))

    ad = OC4kAdapter()
    ad.h, ad.w = h, w
    lo, hi = min(days), max(days)
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
            elif d == SMOKE_ABSENT:
                truth[(ad.store, b, f)] = (None, "absent_upstream")
            elif d not in fields:
                truth[(ad.store, b, f)] = (None, "absent_upstream")
            else:
                chl, kd, nobs = (x.copy() for x in fields[d])
                for a in (chl, kd, nobs):
                    a[a == np.float32(NC_FILL)] = np.nan
                arr, _n, _o = to_frame(chl, kd, nobs, bounds=ad.bounds())
                truth[(ad.store, b, f)] = (arr.astype(np.float16), None)
    return truth


ADAPTER = OC4kAdapter
