"""University of Bremen ASI v5.4 AMSR2 sea-ice concentration — the first
SHARDED tier-G store (family 1.gf, E-082 wave 2).

PLAIN ENGLISH. Microwave radiometers see through cloud and polar night, and the
ASI algorithm (ARTIST sea ice, Spreen et al. 2008) turns AMSR2's 89 GHz
channels into the percentage of each 6.25 km cell covered by sea ice, every
day, for both poles. This store keeps those daily maps whole — two polar
grids, one tiled, compressed shard per five-day bin — so a model can read the
ice edge near the place it predicts at the resolution it was measured.

SOURCE, VERIFIED 2026-09-17 FROM THIS SANDBOX (no account; HTTPS):
  https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/n6250/netcdf/<YYYY>/
      asi-AMSR2-n6250-<YYYYMMDD>-v5.4.nc     (~1.0-1.3 MB)
  https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/s6250/netcdf/<YYYY>/
      asi-AMSR2-s6250-<YYYYMMDD>-v5.4.nc     (~0.4-0.6 MB)
The same days also sit as HDF4 under `<dir>/<YYYY>/<mon>/Arctic|Antarctic/`
(plus GeoTIFF and PNG). This adapter reads the NETCDF copies: netCDF-4 (HDF5)
opens with the `netCDF4` package, which the runners have, while HDF4 needs
pyhdf, which they do not — and the netCDF directory is one flat listing per
year instead of twelve. Their `history` attribute says they are the HDF file
through `grdmath ... FLIPUD`, i.e. the same numbers, rows flipped.

THE LISTING, MEASURED 2026-09-17 (both hemispheres identical in shape):
  years 2012 .. 2026; 2012 starts 2012-07-02; 2026 ends (so far) 2026-09-16;
  every file is v5.4; 5,187 days listed per hemisphere (5,190 in the span); THREE DAYS ABSENT
  UPSTREAM in both, 2013-05-11 .. 2013-05-13 — each becomes a zero-length
  frame counted `absent_upstream`; one stray file (n6250/netcdf/2018/
  2018AugSepGeoNetCDF.zip) is ignored and counted. The size column is
  human-readable ("1.1M"), so the index's byte total is approximate.

THE GRID, MEASURED ON 2020-03-01 (n and s) AND CHECKED IN EVERY FILE:
  n  1792 rows x 1216 cols; x centres -3,846,875 .. 3,746,875 m, y centres
     -5,346,875 .. 5,846,875 m, 6,250 m spacing; grid_mapping
     polar_stereographic with latitude_of_projection_origin 90,
     standard_parallel 70, straight_vertical_longitude_from_pole -45, Hughes
     1980 ellipsoid (a 6,378,273 m, 1/f 298.279411123064) = EPSG:3411 (NSIDC
     north).
  s  1328 rows x 1264 cols; x centres -3,946,875 .. 3,946,875 m, y centres
     -3,946,875 .. 4,346,875 m; origin -90, standard parallel -70, central
     meridian 0 = EPSG:3412 (NSIDC south).
  Variable `z`, float32, (y, x), `_FillValue` NaN, `actual_range` [0, 100];
  values are CONTINUOUS percentages (258,269 distinct values on the northern
  day), not integers. ROW 0 IS THE SMALLEST PROJECTED y (the file's `y` axis
  is ascending) — checked against pyproj on the real day: Hudson Bay (60 N,
  85 W) reads 100, the Barents Sea at 72 N 30 E reads 0 and the North Pole
  cell is NaN (the pole hole); the flipped reading gives 0, 92 and 96. The
  store keeps the file's order and says so in tile_grid.json.
  Missing (NaN): land, the pole hole, no retrieval. On 2020-03-01, 48.9 % of
  the northern grid and 78.8 % of the southern grid is valid (open water
  included — a 0 is a measurement).

WHAT IS STORED. One channel, `sic` (%), dtype UINT8: the float is rounded
half-to-even to a whole percent (0..100) and NaN is stored as 255 (the
layout's `U8_MISSING`). That is the brief's choice and the ASI product's own
accuracy (several percent) makes the rounding immaterial; float16 would cost
2.2x the bytes (measured on the northern 2020-03-01 day at zstd 15: 445 kB
against 198 kB). A finite value outside [0, 100] is NaN and counted
(`out_of_bounds`), never clipped.

TWO GROUPS IN ONE STORE, `n` and `s` (the note's "two sub-groups N/S"):
`tensors/family1_gf/seaice_asi/n/...` and `.../s/...`, each with its own
tile_grid.json and shard_index.npy. One adapter, one dispatch, one licence
row; the two grids never share a shard. F = 5 daily frames per five-day bin,
frame_seconds 86,400; frame f of bin b is the UTC day b*5 + f since
1982-01-01.

A FRAME THAT IS NOT IN THE SOURCE is a zero-length frame with a reason:
`before_record` (before the first listed day, 2012-07-02), `after_record`
(after the last listed day), `absent_upstream` (inside the record, no file).
A calendar month inside the record with NO file is refused (`note_absent`),
because a month-long hole is a listing problem until proven otherwise; a
year inside the record missing from the top listing likewise. A listed file
that will not download or does not parse is an absence, never a None frame.

LICENCE — PENDING. Bremen's page says "free for scientific use", citation
required; the redistribution terms were asked for on 2026-09-17 and have not
been answered. `licence["redistribution_confirmed"]` is False, and
`ml/build_family1_stores.py` refuses a PUBLIC publish (and a public parts
push) of this store until it is True or `--allow-unconfirmed-licence` is
passed. If Bremen refuses redistribution, the store moves to the private
track and OSI SAF OSI-408-a is the public companion (note, "Checked and set
aside").

PHASE B: the AMSR-E record 2002-06-01 .. 2011-10-04 (`amsre/asi_daygrid_
swath/`) on the same grids, and the 3.125 km grids (n3125/s3125).

THE PROBE, 2020-03, MEASURED 2026-09-17 FROM THIS SANDBOX
(ml/family1/probes/seaice_asi_2020-03.json): 62 frames (31 per grid, none
missing), 54,951,401 bytes of netCDF (886 kB a frame) in 22.6 s with four
download workers (0.36 s a frame including the zstd-15 encode). Valid
fraction 0.485 north, 0.788 south. Compressed: north 197 kB a frame (174-216
kB; 961 stored tiles of 1,085, mean 6.4 kB, median 3.4 kB, max 29.7 kB; 11x
over the raw uint8 frame; 0.19 bytes per valid pixel), south 76 kB a frame
(899 of 930 tiles stored, mean 2.6 kB, median 97 B; 22x; 0.06 B/valid px).
Extrapolated over the listed record (5,187 days per grid) plus the per-bin
indices: 1.02 GB north + 0.40 GB south = 1.42 GB, against the note's 8 GB for
the AMSR2 record (whose "pixels x frames x v x 2C x 1.2" arithmetic gives 30
GB, because it charges every valid pixel two uncompressed bytes).
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

BASE = "https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/"
VERSION = "5.4"
HUGHES_A = 6378273.0
HUGHES_RF = 298.279411123064
# From 2018-11-02 Bremen's files declare the SAME x/y grid on the WGS 84
# ellipsoid (EPSG:3413 north / 3976 south) instead of Hughes 1980 (3411 /
# 3412); the axes are identical to the metre (measured 2026-09-17 on
# 2018-11-01/02/03, both hemispheres). Reading one label as the other moves a
# pixel centre by at most 149 m north / 125 m south (median 80 / 70 m) — 2.4 %
# of the 6.25 km pixel. Both labels are accepted; the store declares Hughes
# 1980 and counts frames per declared ellipsoid (`frames_by_declared_
# ellipsoid`) so the relabel is visible, never silent.
WGS84_A = 6378137.0
WGS84_RF = 298.257223563
ELLIPSOIDS = {"hughes1980": (HUGHES_A, HUGHES_RF),
              "wgs84": (WGS84_A, WGS84_RF)}
DX = 6250.0

HEMI = {
    "n": {"dir": "n6250", "crs": "EPSG:3411", "lat0": 90.0, "lat_ts": 70.0,
          "lon0": -45.0, "H": 1792, "W": 1216,
          "x0": -3850000.0, "y0": -5350000.0},
    "s": {"dir": "s6250", "crs": "EPSG:3412", "lat0": -90.0, "lat_ts": -70.0,
          "lon0": 0.0, "H": 1328, "W": 1264,
          "x0": -3950000.0, "y0": -3950000.0},
}

YEAR_DIR = re.compile(r'href="(\d{4})/"')
FILE_ROW = re.compile(
    r'href="(asi-AMSR2-([ns])6250-(\d{8})-v([0-9.]+)\.nc)">[^<]*</a>\s*</td>'
    r'\s*<td[^>]*>[^<]*</td>\s*<td[^>]*>\s*([0-9.]+[KMG]?)\s*</td>')
ANY_HREF = re.compile(r'href="([^"?/][^"]*)"')


# HDF5 (under netCDF4) is NOT thread-safe: measured 2026-09-17, four
# download workers each opening their file crashed the probe with SIGBUS.
# Downloads stay parallel; every netCDF open/read/close holds this lock.
NC_LOCK = threading.Lock()


class FormatError(ValueError):
    """A listing or a file that does not look like the archive measured."""


def approx_bytes(s):
    s = s.strip()
    mult = {"K": 1e3, "M": 1e6, "G": 1e9}.get(s[-1:], 1.0)
    return int(float(s.rstrip("KMG")) * mult)


def grid_of(h):
    e = HEMI[h]
    proj4 = (f"+proj=stere +lat_0={e['lat0']:g} +lat_ts={e['lat_ts']:g} "
             f"+lon_0={e['lon0']:g} +k=1 +x_0=0 +y_0=0 +a={HUGHES_A:.0f} "
             f"+rf={HUGHES_RF} +units=m +no_defs")
    return {
        "H": e["H"], "W": e["W"], "pixel_km": DX / 1000.0,
        "crs": e["crs"], "proj4": proj4,
        "projection": {"grid_mapping_name": "polar_stereographic",
                       "latitude_of_projection_origin": e["lat0"],
                       "standard_parallel": e["lat_ts"],
                       "straight_vertical_longitude_from_pole": e["lon0"],
                       "semi_major_axis": HUGHES_A,
                       "inverse_flattening": HUGHES_RF,
                       "false_easting": 0.0, "false_northing": 0.0},
        "x0": e["x0"], "dx": DX, "y0": e["y0"], "dy": DX,
        "coordinates": ("x = x0 + (col + 0.5) * dx, y = y0 + (row + 0.5) * "
                        "dy, metres on the projection; x0/y0 are the grid's "
                        "left and bottom EDGES"),
        "row_order": ("row 0 is the SMALLEST projected y (the source "
                      "netCDF's ascending `y` axis, verified against the "
                      "real 2020-03-01 file); col 0 the smallest x"),
        "source_variable": "z (float32, NaN fill), rounded to uint8",
        "hemisphere": {"n": "north", "s": "south"}[h],
    }


# ======================================================== the projection ===
def _e():
    f = 1.0 / HUGHES_RF
    return np.sqrt(2 * f - f * f)


def _t(phi, e):
    s = np.sin(phi)
    return np.tan(np.pi / 4 - phi / 2) / ((1 - e * s) / (1 + e * s)) ** (e / 2)


def polar_stereo_inverse(x, y, lat0, lat_ts, lon0, a=HUGHES_A):
    """Projected (x, y) m -> (lat, lon) degrees; Snyder (1987) eq. 21-38.

    The south case uses the north formulas with x, y, the latitudes and the
    longitudes negated (Snyder p. 161).
    """
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    south = lat0 < 0
    sgn = -1.0 if south else 1.0
    x, y = sgn * x, sgn * y
    phic = np.radians(sgn * lat_ts)
    lam0 = np.radians(sgn * lon0)
    e = _e()
    mc = np.cos(phic) / np.sqrt(1 - e * e * np.sin(phic) ** 2)
    tc = _t(phic, e)
    rho = np.hypot(x, y)
    t = rho * tc / (a * mc)
    chi = np.pi / 2 - 2 * np.arctan(t)
    e2, e4, e6, e8 = e ** 2, e ** 4, e ** 6, e ** 8
    phi = (chi
           + (e2 / 2 + 5 * e4 / 24 + e6 / 12 + 13 * e8 / 360) * np.sin(2 * chi)
           + (7 * e4 / 48 + 29 * e6 / 240 + 811 * e8 / 11520) * np.sin(4 * chi)
           + (7 * e6 / 120 + 81 * e8 / 1120) * np.sin(6 * chi)
           + (4279 * e8 / 161280) * np.sin(8 * chi))
    lam = lam0 + np.arctan2(x, -y)
    lat = sgn * np.degrees(phi)
    lon = sgn * np.degrees(lam)
    lon = (lon + 180.0) % 360.0 - 180.0
    return lat, lon


def polar_stereo_forward(lat, lon, lat0, lat_ts, lon0, a=HUGHES_A):
    """(lat, lon) degrees -> projected (x, y) m; Snyder eq. 21-33/34."""
    south = lat0 < 0
    sgn = -1.0 if south else 1.0
    phi = np.radians(sgn * np.asarray(lat, np.float64))
    lam = np.radians(sgn * np.asarray(lon, np.float64))
    phic = np.radians(sgn * lat_ts)
    lam0 = np.radians(sgn * lon0)
    e = _e()
    mc = np.cos(phic) / np.sqrt(1 - e * e * np.sin(phic) ** 2)
    rho = a * mc * _t(phi, e) / _t(phic, e)
    x = rho * np.sin(lam - lam0)
    y = -rho * np.cos(lam - lam0)
    return sgn * x, sgn * y


# ============================================================== parsing ====
def parse_years(html):
    return sorted({int(y) for y in YEAR_DIR.findall(html)})


def parse_year_listing(html, h, year):
    """-> ({date: (name, approx_bytes)}, counts). Only v5.4 files kept."""
    out, counts = {}, {}
    for name, hh, ymd, ver, size in FILE_ROW.findall(html):
        if hh != h:
            raise FormatError(f"{name} in the {h} listing")
        d = dt.date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:]))
        if d.year != year:
            raise FormatError(f"{name} in the {year} listing")
        if ver != VERSION:
            counts["files_other_version"] = \
                counts.get("files_other_version", 0) + 1
            continue
        if d in out:
            raise FormatError(f"{name}: two v{VERSION} files for {d}")
        out[d] = (name, approx_bytes(size))
    listed = {n for n in ANY_HREF.findall(html)
              if not n.startswith(("http", "#"))}
    matched = {r[0] for r in FILE_ROW.findall(html)}
    other = sorted(listed - matched)
    if other:
        counts["files_other"] = len(other)
        counts["files_other_names"] = other[:20]
    return out, counts


def month_of(d):
    return f"{d.year:04d}-{d.month:02d}"


class SeaIceASIAdapter(sh.GridAdapter):
    store = "seaice_asi"
    title = ("Sea-ice concentration, University of Bremen ASI v5.4 from "
             "AMSR2, daily, 6.25 km polar stereographic (north and south)")
    family = "1gf"
    distribution = "public"
    licence = {
        "name": "free for scientific use, cite Spreen et al. 2008",
        "redistribution": "attribution",
        "redistribution_confirmed": False,
        "derived_works": "non-commercial",
        "attribution": ("Spreen, G., L. Kaleschke and G. Heygster (2008), Sea "
                        "ice remote sensing using AMSR-E 89 GHz channels, J. "
                        "Geophys. Res. 113, C02S03, doi:10.1029/2005JC003384. "
                        "Data: University of Bremen, "
                        "https://data.seaice.uni-bremen.de"),
        "pending": ("redistribution terms asked of the Bremen sea-ice group "
                    "2026-09-17; a public publish is refused until "
                    "redistribution_confirmed is True or "
                    "--allow-unconfirmed-licence is passed"),
    }
    time_dtype = "int32"
    credentials = ()
    channels = (("sic", "%", 0.0, 100.0),)
    log2_fp = float(np.log2(6.25 / 27.83))       # -2.154
    log2_dt = float(np.log2(1.0 / 5.0))           # -2.322
    per_year = True
    first_year = 2012
    frames_per_bin = 5
    frame_seconds = 86400
    dtype = "uint8"
    grids = {"n": grid_of("n"), "s": grid_of("s")}
    note_estimate = {"bytes": 8e9,
                     "what": "family1gf.tex ledger: 'AMSR2 record 8 GB' "
                             "(6.25 km, both hemispheres); the storage "
                             "table says ~30 GB with the AMSR-E years"}
    qc_policy = (
        "no per-pixel flag in the source. NaN (land, pole hole, no "
        "retrieval) -> 255; a finite value outside [0, 100] -> 255 and "
        "counted (`out_of_bounds`); values rounded half-to-even to whole "
        "percent")
    sources = (BASE + "n6250/netcdf/<YYYY>/asi-AMSR2-n6250-<YYYYMMDD>-v5.4.nc",
               BASE + "s6250/netcdf/<YYYY>/asi-AMSR2-s6250-<YYYYMMDD>-v5.4.nc")
    verified = (
        "2026-09-17 from the sandbox: both netcdf listings, every year "
        "2012..2026 (5,187 v5.4 days per hemisphere, 2012-07-02 .. "
        "2026-09-16, absent 2013-05-11..13); n and s files of 2020-03-01 "
        "opened (dims, x/y axes, grid_mapping, z dtype and fill); row order "
        "checked against pyproj EPSG:3411/3412 at five known places")
    notes = (
        "Reads the netCDF copies (HDF4 originals need pyhdf). Two groups, n "
        "and s. uint8 whole percent, 255 = missing. AMSR-E 2002-06 .. "
        "2011-10 and the 3.125 km grids are phase B. From 2018-11-02 the "
        "source files declare the identical x/y grid on WGS 84 (EPSG:3413 / "
        "3976) instead of Hughes 1980 (3411 / 3412); reading one label as "
        "the other moves a pixel centre by <= 149 m (2.4 % of a pixel). The "
        "store declares Hughes 1980 and counts frames per declared ellipsoid "
        "(counts.frames_by_declared_ellipsoid).")
    smoke_window = ("2012-12-28", "2013-01-08")
    smoke_probe_month = "2013-01"

    WORKERS = 4

    def __init__(self):
        self._years = {}
        self._ylist = {}
        self._first_file = {}

    def grid_latlon(self, group, x, y):
        e = HEMI[group]
        return polar_stereo_inverse(x, y, e["lat0"], e["lat_ts"], e["lon0"])

    # --------------------------------------------------------------- paths --
    def _dir(self, h):
        return HEMI[h]["dir"]

    def _url(self, h, *parts):
        return BASE + "/".join((self._dir(h), "netcdf") + parts)

    def _local(self, ctx, h, *parts):
        return os.path.join(ctx.source_dir, "seaice_asi", self._dir(h),
                            "netcdf", *parts)

    def _page(self, ctx, h, *parts):
        if ctx.source_dir:
            p = self._local(ctx, h, *parts, "index.html")
            if not os.path.exists(p):
                return None
            with open(p, "rb") as fh:
                raw = fh.read()
            ctx.count_bytes(len(raw))
            return raw.decode("utf-8", "replace")
        url = self._url(h, *parts) + "/"
        raw, why = cm.get_bytes(url, attempts=ctx.a.attempts)
        if raw is None:
            return None
        return raw.decode("utf-8", "replace")

    # ------------------------------------------------------------ listings --
    def years(self, ctx, h):
        if h not in self._years:
            html = self._page(ctx, h)
            ys = parse_years(html or "")
            if not ys:
                sys.exit(f"REFUSING seaice_asi: an empty listing at "
                         f"{self._url(h)}/ — no year directories (the "
                         f"2026-09-14 rule: an empty listing is a refusal)")
            self._years[h] = ys
        return self._years[h]

    def year_listing(self, ctx, h, y):
        key = (h, y)
        if key not in self._ylist:
            html = self._page(ctx, h, f"{y:04d}")
            if html is None:
                self._ylist[key] = None
            else:
                try:
                    files, counts = parse_year_listing(html, h, y)
                except FormatError as e:
                    sys.exit(f"REFUSING seaice_asi: {h} {y} listing: {e}")
                self._ylist[key] = (files, counts)
        return self._ylist[key]

    def record(self, ctx, h):
        """(first listed day, last listed day) of the whole record."""
        ys = self.years(ctx, h)
        lo = self.year_listing(ctx, h, ys[0])
        hi = self.year_listing(ctx, h, ys[-1])
        if not lo or not lo[0] or not hi or not hi[0]:
            sys.exit(f"REFUSING seaice_asi: the {h} listing of {ys[0]} or "
                     f"{ys[-1]} is empty — cannot place the record's ends")
        return min(lo[0]), max(hi[0])

    def record_frames(self, ctx, group):
        n = 0
        for y in self.years(ctx, group):
            yl = self.year_listing(ctx, group, y)
            n += len(yl[0]) if yl else 0
        return n

    def empty_months(self, ctx, h, y):
        """Calendar months of year y inside the record with no file."""
        first, last = self.record(ctx, h)
        yl = self.year_listing(ctx, h, y)
        have = {month_of(d) for d in (yl[0] if yl else {})}
        out = []
        for m in range(1, 13):
            m0 = dt.date(y, m, 1)
            m1 = dt.date(y, m, calendar.monthrange(y, m)[1])
            if m1 < first or m0 > last:
                continue
            if f"{y:04d}-{m:02d}" not in have:
                out.append(f"{y:04d}-{m:02d}")
        return out

    # ----------------------------------------------------------- one file --
    def _read(self, ctx, h, d, name):
        """-> (z float32 [H, W], counts, error)."""
        y = d.year
        tmpdir = None
        try:
            if ctx.source_dir:
                path = self._local(ctx, h, f"{y:04d}", name)
                if not os.path.exists(path):
                    return None, {}, f"{name}: listed and absent"
                nbytes = os.path.getsize(path)
                ctx.count_bytes(nbytes)
            else:
                tmpdir = tempfile.mkdtemp(prefix="asi_", dir=ctx.scratch)
                path = os.path.join(tmpdir, name)
                url = self._url(h, f"{y:04d}", name)
                got = f10b.fetch_first([url], path, attempts=ctx.a.attempts)
                if got is None:
                    return None, {}, f"{url}: listed, and 404"
                nbytes = os.path.getsize(path)
            info = {}
            with NC_LOCK:
                z = read_nc(path, h, info)
            return z, {"files_read": 1, "bytes_files": nbytes,
                       "frames_by_declared_ellipsoid":
                           {info.get("ellipsoid", "?"): 1}}, None
        except FormatError as e:
            sys.exit(f"REFUSING seaice_asi: {name}: {e}")
        except (IOError, OSError) as e:
            return None, {}, f"{name}: {type(e).__name__}: {e}"
        finally:
            if tmpdir:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)

    # -------------------------------------------------------- the contract --
    def index(self, ctx):
        out = {"dataset": f"Bremen ASI v{VERSION} AMSR2 6.25 km daily "
                          f"(netCDF copies)",
               "url": BASE, "version": VERSION, "hemispheres": {}}
        for h in HEMI:
            ys = self.years(ctx, h)
            first, last = self.record(ctx, h)
            per, other, absent, bytes_ = {}, {}, [], 0
            for y in ys:
                yl = self.year_listing(ctx, h, y)
                files, counts = yl if yl else ({}, {})
                per[str(y)] = len(files)
                bytes_ += sum(b for _, b in files.values())
                if counts:
                    other[str(y)] = counts
                d0 = max(first, dt.date(y, 1, 1))
                d1 = min(last, dt.date(y, 12, 31))
                d = d0
                while d <= d1:
                    if d not in files:
                        absent.append(str(d))
                    d += dt.timedelta(days=1)
            empty = {str(y): self.empty_months(ctx, h, y) for y in ctx.years
                     if y in ys and self.empty_months(ctx, h, y)}
            if empty:
                sys.exit(f"REFUSING seaice_asi: {h}: a calendar month inside "
                         f"the record has no file in the listing: {empty}")
            missing_years = [y for y in ctx.years
                             if ys[0] <= y <= ys[-1] and y not in ys]
            if missing_years:
                sys.exit(f"REFUSING seaice_asi: {h}: year(s) {missing_years} "
                         f"inside the record are not in the listing")
            # ONE REAL FILE, read and checked against the declared grid
            want = [y for y in ctx.years if y in ys] or [ys[-1]]
            yl = self.year_listing(ctx, h, want[0])
            d = min(yl[0])
            name = yl[0][d][0]
            z, c, err = self._read(ctx, h, d, name)
            if err:
                sys.exit(f"REFUSING seaice_asi: the first file {name} could "
                         f"not be read: {err}")
            out["hemispheres"][h] = {
                "listing": self._url(h) + "/",
                "years": ys, "record": [str(first), str(last)],
                "files_per_year": per, "files": sum(per.values()),
                "days_absent_in_record": absent,
                "other_files": other,
                "bytes_approx": bytes_,
                "first_file": {"name": name, "bytes": c.get("bytes_files"),
                               "valid_fraction": round(
                                   float(np.isfinite(z).mean()), 6)},
                "grid": grid_of(h)}
        out["bytes_note"] = ("the listing's size column is human-readable; "
                             "the probe measures exact bytes")
        return out

    def classify(self, ctx, h, d):
        """-> (name or None, reason or None) for one day."""
        ys = self.years(ctx, h)
        if d.year < ys[0]:
            return None, "before_record"
        if d.year > ys[-1]:
            return None, "after_record"
        first, last = self.record(ctx, h)
        if d < first:
            return None, "before_record"
        if d > last:
            return None, "after_record"
        yl = self.year_listing(ctx, h, d.year)
        if yl is None:
            return None, "year_not_listed"
        hit = yl[0].get(d)
        if hit is None:
            return None, "absent_upstream"
        return hit[0], None

    def fetch_frames(self, ctx, wanted):
        """Every (group, bin, frame) in `wanted`, in order."""
        refused_units = set()
        jobs = []
        for (g, b, f) in wanted:
            d = sh.frame_day(b, f, self.frame_seconds)
            name, why = self.classify(ctx, g, d)
            if why == "year_not_listed":
                unit = f"{d.year} {g}"
                if unit not in refused_units:
                    ctx.note_absent(unit, f"{d.year} is inside the {g} record "
                                          f"and not in its listing")
                    refused_units.add(unit)
                continue
            if name is None and why == "absent_upstream":
                em = self.empty_months(ctx, g, d.year)
                if month_of(d) in em:
                    unit = f"{month_of(d)} {g}"
                    if unit not in refused_units:
                        ctx.note_absent(unit, f"the {g} listing of "
                                              f"{d.year} has no file in "
                                              f"{month_of(d)} — an empty "
                                              f"month is refused")
                        refused_units.add(unit)
                    continue
            jobs.append((g, b, f, d, name, why))

        def work(job):
            g, b, f, d, name, why = job
            if name is None:
                return job, None, {}, None
            z, c, err = self._read(ctx, g, d, name)
            return job, z, c, err

        workers = 1 if ctx.source_dir else self.WORKERS
        t0 = time.time()
        for job, z, c, err in cm.ordered_map(work, jobs, workers):
            g, b, f, d, name, why = job
            if name is None:
                yield g, b, f, None, {"frame_missing": why}
                continue
            if err:
                ctx.note_absent(f"{d.year} {g} {d}", err)
                continue
            v = z.reshape(-1, 1).astype(np.float64)
            oob = self.mask_bounds(v)
            counts = dict(c)
            if oob:
                counts["out_of_bounds"] = oob
            z = v.reshape(z.shape)
            counts["fetch_seconds"] = round(time.time() - t0, 2)
            t0 = time.time()
            yield g, b, f, z, counts

    # --------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260917):
        return make_smoke_sources(root, d_lo, d_hi, seed)


def read_nc(path, h, info=None):
    """One netCDF day -> float32 [H, W] with NaN; the grid is VERIFIED.

    `info`, if given, receives {"ellipsoid": "hughes1980" | "wgs84"}."""
    try:
        import netCDF4
    except ImportError:                                     # pragma: no cover
        sys.exit("seaice_asi needs the `netCDF4` package (pip install "
                 "netCDF4) — it is in the family1-build install step")
    e = HEMI[h]
    try:
        ds = netCDF4.Dataset(path)
    except OSError as ex:
        raise IOError(f"not a readable netCDF file ({ex}) — a truncated "
                      f"download") from None
    try:
        dims = {k: len(v) for k, v in ds.dimensions.items()}
        if dims.get("y") != e["H"] or dims.get("x") != e["W"]:
            raise FormatError(f"dimensions {dims}, expected y={e['H']} "
                              f"x={e['W']}")
        for k in ("x", "y", "z", "polar_stereographic"):
            if k not in ds.variables:
                raise FormatError(f"no variable {k!r} "
                                  f"({sorted(ds.variables)})")
        gm = ds.variables["polar_stereographic"]
        want = {"latitude_of_projection_origin": e["lat0"],
                "standard_parallel": e["lat_ts"],
                "straight_vertical_longitude_from_pole": e["lon0"]}

        def attr(k):
            return float(gm.getncattr(k)) if k in gm.ncattrs() else None

        def same(got, v):
            return got is not None and abs(got - v) <= 1e-6 * max(1.0, abs(v))
        for k, v in want.items():
            if not same(attr(k), v):
                raise FormatError(f"grid_mapping {k} = {attr(k)}, expected {v}")
        a_, rf_ = attr("semi_major_axis"), attr("inverse_flattening")
        ell = [n for n, (a0, rf0) in ELLIPSOIDS.items()
               if same(a_, a0) and same(rf_, rf0)]
        if not ell:
            raise FormatError(f"grid_mapping semi_major_axis = {a_}, "
                              f"inverse_flattening = {rf_}: neither Hughes "
                              f"1980 nor WGS 84")
        if info is not None:
            info["ellipsoid"] = ell[0]
        x = np.asarray(ds.variables["x"][:], np.float64)
        y = np.asarray(ds.variables["y"][:], np.float64)
        xc = e["x0"] + (np.arange(e["W"]) + 0.5) * DX
        yc = e["y0"] + (np.arange(e["H"]) + 0.5) * DX
        if not (np.allclose(x, xc, atol=0.5) and np.allclose(y, yc, atol=0.5)):
            raise FormatError(f"x/y axes differ from the declared grid "
                              f"(x {x[0]}..{x[-1]}, y {y[0]}..{y[-1]})")
        zv = ds.variables["z"]
        if zv.dimensions != ("y", "x"):
            raise FormatError(f"z dimensions {zv.dimensions}")
        zv.set_auto_maskandscale(False)
        z = np.array(zv[:], dtype=np.float32)
        fill = getattr(zv, "_FillValue", np.nan)
        if not np.isnan(fill):
            z[z == fill] = np.nan
        z[~np.isfinite(z)] = np.nan
        return z
    finally:
        ds.close()


# ================================================================== smoke ==
SMOKE_ABSENT = dt.date(2013, 1, 3)
SMOKE_OOB = dt.date(2013, 1, 2)            # one pixel of 103.7 %, both grids
SMOKE_OTHER_VERSION = dt.date(2012, 12, 30)


def smoke_field(h, d):
    """A deterministic day on the real grid: a NaN 'continent' and pole
    hole, open water (0), fractional concentrations, a NaN strip."""
    e = HEMI[h]
    H, W = e["H"], e["W"]
    k = (d - dt.date(2012, 1, 1)).days
    yy = (np.arange(H, dtype=np.float64)[:, None] - H / 2) / H
    xx = (np.arange(W, dtype=np.float64)[None, :] - W / 2) / W
    r = np.hypot(yy, xx)
    z = np.clip(130.0 - 260.0 * r + 3.0 * np.sin(k + 20 * xx), 0.0, 100.0)
    z = z + 0.37 * np.cos(7 * yy + k)            # fractional, in-range
    z = np.clip(z, 0.0, 100.0)
    z[r > 0.62] = np.nan                          # outside the swath
    z[(np.abs(yy - 0.1) < 0.05) & (np.abs(xx + 0.2) < 0.08)] = np.nan
    z[r < 0.004] = np.nan                         # the pole hole
    z[:, 5 + (k % 7)] = np.nan                    # a strip
    if d == SMOKE_OOB:
        z[H // 2 + 3, W // 2 + 3] = 103.7
    return z.astype(np.float32)


def _listing_html(path_title, entries, dirs=False):
    rows = []
    for name, size in entries:
        if dirs:
            rows.append(f'<tr><td valign="top"><img src="/icons/folder.gif" '
                        f'alt="[DIR]"></td><td><a href="{name}/">{name}/</a>'
                        f'</td><td align="right">2026-01-02 05:28  </td>'
                        f'<td align="right">  - </td><td>&nbsp;</td></tr>')
        else:
            rows.append(f'<tr><td valign="top"><img src="/icons/unknown.gif" '
                        f'alt="[   ]"></td><td><a href="{name}">{name}</a>'
                        f'</td><td align="right">2020-03-02 04:51  </td>'
                        f'<td align="right">{size}</td><td>&nbsp;</td></tr>')
    return ("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 3.2 Final//EN\">\n"
            f"<html><head><title>Index of {path_title}</title></head><body>\n"
            f"<h1>Index of {path_title}</h1><table>\n"
            '<tr><th><a href="?C=N;O=D">Name</a></th></tr>\n'
            f'<tr><td><a href="/amsr2/">Parent Directory</a></td></tr>\n'
            + "\n".join(rows) + "\n</table></body></html>\n")


def _human(n):
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{max(1, n // 1000)}K"


def write_nc(path, h, z):
    import netCDF4
    e = HEMI[h]
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    try:
        ds.setncattr("info", "AMSR2 sea ice concentration based on the ASI "
                             "algorithm (Spreen et al.,2008). (SMOKE)")
        ds.createDimension("x", e["W"])
        ds.createDimension("y", e["H"])
        gm = ds.createVariable("polar_stereographic", "S1", ())
        gm.grid_mapping_name = "polar_stereographic"
        gm.straight_vertical_longitude_from_pole = e["lon0"]
        gm.false_easting = 0.0
        gm.false_northing = 0.0
        gm.latitude_of_projection_origin = e["lat0"]
        gm.standard_parallel = e["lat_ts"]
        gm.semi_major_axis = HUGHES_A
        gm.inverse_flattening = HUGHES_RF
        xv = ds.createVariable("x", "f8", ("x",))
        xv.units = "m"
        xv[:] = e["x0"] + (np.arange(e["W"]) + 0.5) * DX
        yv = ds.createVariable("y", "f8", ("y",))
        yv.units = "m"
        yv[:] = e["y0"] + (np.arange(e["H"]) + 0.5) * DX
        zv = ds.createVariable("z", "f4", ("y", "x"), zlib=True, complevel=1,
                               fill_value=np.float32(np.nan))
        zv.long_name = "z"
        zv.grid_mapping = "polar_stereographic"
        zv[:] = z
    finally:
        ds.close()


def make_smoke_sources(root, d_lo, d_hi, seed=20260917):
    """The archive in its real layout: netcdf/index.html listing year
    directories, netcdf/<YYYY>/index.html listing the day files (with a
    non-v5.4 file and a stray zip), and the netCDF-4 days themselves on the
    real grids. `d_lo` .. `d_hi` is the record; SMOKE_ABSENT is not listed.

    Returns the truth: {(group, bin, frame): (uint8 [H, W, 1] or None,
    reason or None)} for every frame of every bin overlapping the record.
    """
    days = [d_lo + dt.timedelta(days=k) for k in range((d_hi - d_lo).days + 1)]
    truth = {}
    ad = SeaIceASIAdapter()
    for h in HEMI:
        base = os.path.join(root, "seaice_asi", HEMI[h]["dir"], "netcdf")
        years = sorted({d.year for d in days})
        os.makedirs(base, exist_ok=True)
        with open(os.path.join(base, "index.html"), "w") as fh:
            fh.write(_listing_html(f"/{HEMI[h]['dir']}/netcdf",
                                   [(str(y), None) for y in years],
                                   dirs=True))
        for y in years:
            yd = os.path.join(base, str(y))
            os.makedirs(yd, exist_ok=True)
            entries = []
            for d in [x for x in days if x.year == y]:
                if d == SMOKE_ABSENT:
                    continue
                z = smoke_field(h, d)
                name = f"asi-AMSR2-{HEMI[h]['dir']}-{d:%Y%m%d}-v{VERSION}.nc"
                write_nc(os.path.join(yd, name), h, z)
                entries.append((name, _human(os.path.getsize(
                    os.path.join(yd, name)))))
                if d == SMOKE_OTHER_VERSION:
                    entries.append((f"asi-AMSR2-{HEMI[h]['dir']}-"
                                    f"{d:%Y%m%d}-v5.nc", "1.0M"))
            if y == years[0]:
                entries.append((f"{y}SmokeGeoNetCDF.zip", "12M"))
            with open(os.path.join(yd, "index.html"), "w") as fh:
                fh.write(_listing_html(f"/{HEMI[h]['dir']}/netcdf/{y}",
                                       sorted(entries)))
        t_lo = f10b.seconds_since_epoch(d_lo)
        t_hi = f10b.seconds_since_epoch(d_hi) + 86399
        for b in sh.bins_overlapping(t_lo, t_hi):
            for f in range(ad.frames_per_bin):
                d = sh.frame_day(b, f, ad.frame_seconds)
                if d < d_lo:
                    truth[(h, b, f)] = (None, "before_record")
                elif d > d_hi:
                    truth[(h, b, f)] = (None, "after_record")
                elif d == SMOKE_ABSENT:
                    truth[(h, b, f)] = (None, "absent_upstream")
                else:
                    z = smoke_field(h, d).astype(np.float64)
                    z[(z < 0) | (z > 100)] = np.nan
                    u = np.where(np.isnan(z), sh.U8_MISSING,
                                 np.rint(z)).astype(np.uint8)
                    truth[(h, b, f)] = (u[:, :, None], None)
    return truth


ADAPTER = SeaIceASIAdapter
