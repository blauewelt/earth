"""OceanSITES moored time series — one row per (site, timestamp) (family 1.gf).

PLAIN ENGLISH. OceanSITES is the network of long-term deep-ocean moorings
and fixed stations — the Northwest Tropical Atlantic Station, Station Papa,
the Kuroshio Extension Observatory, the Hawaii and Bermuda time series and
about a hundred more. This adapter turns the network's data files into a
tier-P store: one row per site per report time, carrying the surface weather
(air temperature, pressure, wind, sunlight, rain) and the sea temperature and
salinity at ten fixed depths, so a model can read what the moorings measured.
The tropical moored arrays (TAO/TRITON, PIRATA, RAMA) are left out: they are
family 10.1's `gtmba`.

SOURCE, VERIFIED 2026-09-17 FROM THIS SANDBOX (OceanSITES GDAC at NOAA NDBC,
no account):
  https://dods.ndbc.noaa.gov/thredds/fileServer/oceansites/oceansites_index.txt
  https://dods.ndbc.noaa.gov/thredds/fileServer/oceansites/DATA/<SITE>/<file>.nc
  https://dods.ndbc.noaa.gov/thredds/ncml/oceansites/DATA/<SITE>/<file>.nc
  * The other GDAC (Ifremer: ftp.ifremer.fr/ifremer/oceansites, and its
    THREDDS) refused from here (proxy 502 / connection reset / 404); the
    data-argo.ifremer.fr host has no oceansites tree. NDBC answers.
  * `oceansites_index.txt` (20,727,619 B, "Index File v2.0", updated
    2026-09-17T05:50Z): 8 `#` lines, then one line per file with 16
    comma-separated fields FILE, DATE_UPDATE, START_DATE, END_DATE, S, N, W,
    E latitude/longitude, MINIMUM_DEPTH, MAXIMUM_DEPTH, UPDATE_INTERVAL,
    SIZE (bytes, exact), GDAC_CREATION_DATE, GDAC_UPDATE_DATE, DATA_MODE,
    PARAMETERS (space-separated CF standard names). 60,027 files: 59,647
    under DATA/ (112 site directories — the same 112 the THREDDS catalog
    lists) and 380 under DATA_GRIDDED/ (gridded products, NOT read). 5 lines
    are the tail of a parameter list broken by a newline (skipped and
    counted); 3 files have no dates ("unknown", read for any year); END_DATE
    can be "...T24:00:00Z" (parsed as the next midnight); a few START_DATE
    years are garbage ("0018", "8158") and are treated as unknown.
    DATA/ALOHA holds 26,013 files of the ALOHA Cabled Observatory at
    4,555-4,726 m (161 GB of the 216 GB listed).
  * The files are NetCDF-3 classic (most) or NetCDF-4 (e.g. KEO), in several
    layouts, all with a TIME dimension (days since 1950-01-01): (TIME,),
    (TIME, DEPTH) with a DEPTH coordinate, (TIME, DEPTH_TEMP, LATITUDE,
    LONGITUDE) with length-1 position dimensions, (TIME, HEIGHT...) for
    meteorology, or (TIME,) with a scalar DEPTH variable. Fill values
    measured: NaN, -99999, 1e35, or none declared. QC is a `<VAR>_QC` int8
    with OceanSITES reference table 2 (0 unknown/no QC, 1 good, 2 probably
    good, 3 bad-correctable, 4 bad, 5 changed, 7 nominal, 8 interpolated,
    9 missing; fill -128). I-ORS and S-ORS write TIME units as a bare
    "1950-01-01T00:00:00Z" with comment "Julian days": a bare ISO origin is
    read as days since it only when the variable's comment or long_name says
    days (`files_time_units_bare_origin`); any other layout the reader cannot
    trust is refused and the file named under `files_layout_refused` (its
    bytes were read, so it is not an absence; a file that cannot be opened
    is one). Variables are found by their CF standard_name,
    never by their name (NTAS's air temperature is `AIRT` in one file and
    `TA_H` in another).
  * The global attributes `project` / `array` / `network` / `title` / `id`
    name the array: TAO sites say `network: TAO` and `id: TAO_T0N110W_...`,
    P12N23W says `array: PIRATA, network: PIRATA`; NTAS says `project: NTAS`,
    KEO `network: OCS`, PAPA `array: STATION-P`. NcML (THREDDS's header
    service, ~20 KB, both NetCDF-3 and -4) returns them without the data.

WHICH FILES ARE READ (all counted):
  1. under DATA/ only (`files_not_data_tree`);
  2. whose PARAMETERS name at least one standard name a channel uses
     (`files_no_channel_params`);
  3. not entirely below the deepest standard depth: a file with no surface
     or meteorological parameter and MINIMUM_DEPTH > 525 m is skipped
     (`files_below_500m`) — this is what removes the ALOHA Cabled
     Observatory, by the index's own depth column;
  4. whose [START_DATE, END_DATE] overlaps the year;
  5. whose SITE is not a tropical-array site. RULE: the site's first index
     file with channel parameters has its NcML header read; if any of its
     `project`, `array`, `network`, `title` or `id` attributes names TAO,
     TRITON, PIRATA or RAMA (as a word, underscores counting as a
     separator), every file of the site is excluded (`sites_excluded_gtmba`).
     Each file actually read re-checks its own attributes the same way
     (`files_excluded_gtmba_attr`).

WHAT A ROW IS.
  time_s    TIME, rounded to the second, int32 seconds since 1982-01-01.
  lat, lon  the position of the source that supplied the row (per-time
            LATITUDE/LONGITUDE when the file carries them, else its scalar
            position).
  platform  platform_hash(site directory name, e.g. "NTAS"); platforms.json
            carries the site's position and file span from the index.
  values    C = 28 = 8 surface/meteorological channels
              airt (air_temperature, degC), sst (sea_surface_temperature, or
              sea_water_temperature at a nominal depth <= 1.5 m), sss
              (sea_surface_salinity, or sea_water_[practical_]salinity at
              <= 1.5 m), wind_u / wind_v (eastward_wind / northward_wind, or
              from wind_speed with wind_to_direction / wind_from_direction),
              pres (air_pressure / _at_sea_level / _at_mean_sea_level, hPa),
              sw (surface_downwelling_shortwave_flux_in_air, W/m2), precip
              (rainfall_rate / precipitation_flux, mm/h)
            + T and S at 10, 20, 30, 50, 75, 100, 150, 200, 300, 500 m: the
              instrument whose nominal depth is within max(1 m, 5 %) of the
              standard depth (nearest wins); NaN where the mooring has none.
              A mooring's instruments are not interpolated between.
  qc        bit 1: a kept value had no QC (flag 0, fill, or no _QC
            variable); bit 2: a kept value was flagged 5, 7 or 8 (changed,
            nominal, interpolated). Flags 3, 4, 9 -> NaN, counted
            (`values_qc_removed`).
MERGING. A site has many files (per-instrument files, whole-mooring files,
hourly averages and flux products of the same sensors). For each channel
group (u and v travel together; every other channel alone) the sources are
ranked finest cadence first, then data mode D, M, P, R, then file name; a
source's values inside the time span of a better-ranked source of that
channel are dropped (`values_superseded`), so an hourly average never
duplicates the 10-minute record it was computed from, but fills where that
record is absent. Two sources at the same second: the better one wins. A row
with no value is dropped (`rows_no_value`).

MORE LAYOUTS, MEASURED IN THE 2018-09 PROBE and handled: E1M3A puts every
variable on (TIME, DEPTH) with no DEPTH coordinate and a 2-D DEPH(TIME,
DEPTH) — the per-column median of DEPH is the instrument depth, and a
meteorological variable on that axis uses its one populated level; the
merged MLTS files give TEMP(TIME) with `coordinates` naming DEPTH_SST
(Deployment) — used when the deployments' depths agree within the depth
tolerance; SOTS ASIMET gives TEMP(TIME) with a per-time TEMP_H ("sensor
height below water level", positive down) and names a NOMINAL_DEPTH that is
not in the file — the median of <VAR>_H is the depth; WHOTS writes its fill
as `FillValue`; unit spellings "degreeofcelcius", ".001", "hectopascal",
"millimetershour-1", "wattsmeter-2", "metersecond-1", "knots". Sources whose
whole-file median spacing exceeds one day (the monthly "-1M" flux and merged
products) are not used (`files_cadence_over_1day`): a monthly mean is not a
report. ADCP transducer temperatures carry no depth and are skipped
(`vars_no_depth`, named per file). The PMEL OCS 10-minute rain rates (KEO,
PAPA) go slightly negative (to -2.4 mm/h): NaN and counted as out of bounds.

THE PROBE, 2018-09, MEASURED 2026-09-17 (ml/family1/probes/
oceansites_2018-09.json): 163 files selected at 26 sites (54 tropical-array
sites active that month excluded, 816 of their files never opened), all 163
read — 3,000,888,040 bytes (whole deployment files: 8.7 kB a kept row) in
957 s, peak RSS 689 MB (WHOTS: 45 files merged into 151,620 rows);
344,798 rows from 17 sites (WHOTS 151,620, NTAS 43,200, SOTS 43,200, PAPA
43,083, KEO 43,064, EC1 5,760, S-ORS 4,312, I-ORS 2,175, MOVE1/3 2,160 each,
DYFAMED/LION 1,380 each, MBARI 720, E1M3A 240, PYLOS 194, FRAM 149, SATS 1);
218,835 values superseded by a finer source, 10,382 removed by QC 3/4/9,
2,926 out of bounds (negative rain), 3,075 rows left empty; 7 files read
through the bare-origin time rule, 5 refused by cadence; qc 0 166,842,
1 173,636, 2 4,320. NaN fractions: sw 0.37, airt/pres 0.58, wind 0.58,
precip 0.61, sst 0.72, sss 0.74; T_10 0.85, T_50 0.87, T_100 0.87, T_150
0.86, T_500 0.97; S 0.88-1.0. The index stage took 20-432 s (102 NcML
headers; the sandbox's egress relay dropped NDBC tunnels repeatedly — every
download retries six times, and one run still recorded one truncated file as
an absence, which is the intended refusal).
WHOLE ARCHIVE, from the index and the site decisions: 41 sites kept, 16,573
files, 22.3 GB listed (the note's 26 GB); a multi-year file is downloaded
once per year it spans, so a full build reads ≈ 61.6 GB (1953-2026). The
kept sites span 537 site-years; at the probe's ≈ 20 k rows per active
site-month that is ≤ 1.3e8 rows (many site-years are sparse ship casts, so
the true figure is lower — the note's ≈ 1e8), ≤ 11 GB stored at 83 B a row
(the note's 4 GB was for C = 5). 2018 alone: ≈ 4.1 M rows.

MEMORY. One site-year at a time: its files are downloaded (4 at a time,
at most 8 on disk), read variable by variable (only the window's TIME slice
and only the needed depth columns), deleted, then merged. What is held is
the site's sources (per used variable: time, value, qc, position for the
window) plus the merged (rows x 28) float64 block, ≈ 230 B a row. Measured
peak for a month: 689 MB (WHOTS, 45 files); a site-YEAR with 1-minute
meteorology and ~20 instruments is ≈ 0.5 M rows and a few million source
samples, ≈ 1-2 GB (ESTIMATED from the month, not measured) — provision
for that.
"""
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET

import numpy as np

import build_family10_stores as f10b
from family1.adapters import _common as cm

HOST = "https://dods.ndbc.noaa.gov/thredds/"
INDEX = HOST + "fileServer/oceansites/oceansites_index.txt"
FILES = HOST + "fileServer/oceansites/"
NCML = HOST + "ncml/oceansites/"

DEPTHS = (10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0)
SURFACE_MAX = 1.5
DEPTH_SKIP = 525.0
METEO = (("airt", "degC", -80.0, 60.0),
         ("sst", "degC", -3.0, 40.0),
         ("sss", "PSU", 0.0, 42.0),
         ("wind_u", "m/s", -80.0, 80.0),
         ("wind_v", "m/s", -80.0, 80.0),
         ("pres", "hPa", 850.0, 1090.0),
         ("sw", "W/m2", -20.0, 1500.0),
         ("precip", "mm/h", 0.0, 500.0))
CHANNELS = METEO + tuple((f"T_{int(d)}", "degC", -3.0, 40.0)
                         for d in DEPTHS) + \
    tuple((f"S_{int(d)}", "PSU", 0.0, 42.0) for d in DEPTHS)
NT = len(DEPTHS)
T0, S0 = len(METEO), len(METEO) + NT

SN_AIRT = {"air_temperature"}
SN_SST = {"sea_surface_temperature"}
SN_SSS = {"sea_surface_salinity"}
SN_TEMP = {"sea_water_temperature"}
SN_SAL = {"sea_water_practical_salinity", "sea_water_salinity"}
SN_U, SN_V = {"eastward_wind"}, {"northward_wind"}
SN_SPD = {"wind_speed"}
SN_DIR_TO, SN_DIR_FROM = {"wind_to_direction"}, {"wind_from_direction"}
SN_PRES = {"air_pressure", "air_pressure_at_sea_level",
           "air_pressure_at_mean_sea_level"}
SN_SW = {"surface_downwelling_shortwave_flux_in_air"}
SN_RAIN = {"rainfall_rate", "precipitation_flux"}
CHANNEL_SN = (SN_AIRT | SN_SST | SN_SSS | SN_TEMP | SN_SAL | SN_U | SN_V
              | SN_SPD | SN_PRES | SN_SW | SN_RAIN)
SURFACE_SN = CHANNEL_SN - SN_TEMP - SN_SAL
READ_SN = CHANNEL_SN | SN_DIR_TO | SN_DIR_FROM

GTMBA = re.compile(r"(?<![A-Za-z])(TAO|TRITON|PIRATA|RAMA)(?![A-Za-z])")
GTMBA_ATTRS = ("project", "array", "network", "title", "id")
MODE_RANK = {"D": 0, "M": 1, "P": 2, "R": 3}
BAD_QC = (3, 4, 9)
FLAG_QC = (5, 7, 8)
ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z?$")
TIME_UNITS = re.compile(r"^\s*(days|hours|minutes|seconds)\s+since\s+"
                        r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[T ](\d{1,2}):(\d{2})"
                        r"(?::(\d{2})(?:\.\d*)?)?)?")
BARE_ORIGIN = re.compile(r"^\s*\d{4}-\d{1,2}-\d{1,2}([T ][0-9:.]+Z?)?\s*$")
UNIT_S = {"days": 86400.0, "hours": 3600.0, "minutes": 60.0, "seconds": 1.0}
WORKERS = 4
MAX_CADENCE = 86400.0          # seconds; coarser sources are not used


class FormatError(ValueError):
    """An index or a file that is not the layout verified above."""


# ================================================================== index ==
def iso_seconds(s):
    """ISO time -> seconds since 1982 (None if unparseable/implausible)."""
    m = ISO.match(s.strip())
    if not m:
        return None
    y, mo, d, h, mi, se = (int(x) for x in m.groups())
    if not (1950 <= y <= 2100) or not bool(cm.valid_date(y, mo, d)) or \
            h > 24 or mi > 59 or se > 60:
        return None
    return int(cm.days_from_civil(y, mo, d)) * 86400 + h * 3600 + mi * 60 + se


def _float(s):
    try:
        return float(s)
    except ValueError:
        return None


def parse_index(text):
    """-> (files [dict], counts). Refuses an empty or reshaped index."""
    lines = text.splitlines()
    if not lines or "OceanSITES" not in lines[0]:
        raise FormatError(f"not an OceanSITES index: {lines[:1]!r}")
    counts = {"index_lines": 0}
    files = []
    for ln in lines:
        if ln.startswith("#") or not ln.strip():
            continue
        counts["index_lines"] += 1
        p = ln.split(",")
        if len(p) != 16:
            counts["index_lines_malformed"] = \
                counts.get("index_lines_malformed", 0) + 1
            continue
        path = p[0].strip()
        parts = path.split("/")
        if len(parts) != 3 or not parts[2].endswith(".nc"):
            counts["index_lines_malformed"] = \
                counts.get("index_lines_malformed", 0) + 1
            continue
        lat = [_float(p[4]), _float(p[5])]
        lon = [_float(p[6]), _float(p[7])]
        files.append({
            "path": path, "tree": parts[0], "site": parts[1],
            "name": parts[2], "t0": iso_seconds(p[2]), "t1": iso_seconds(p[3]),
            "lat": (None if None in lat else sum(lat) / 2),
            "lon": (None if None in lon else sum(lon) / 2),
            "min_depth": _float(p[8]), "size": int(p[11]) if
            p[11].strip().isdigit() else 0,
            "mode": p[14].strip(), "params": set(p[15].split())})
    if not files:
        raise FormatError("the OceanSITES index lists no file — an empty "
                          "listing is a broken listing")
    return files, counts


def usable(f, counts):
    """Rules 1-3 of the docstring; counts the rejections."""
    if f["tree"] != "DATA":
        key = "files_not_data_tree"
    elif not f["params"] & CHANNEL_SN:
        key = "files_no_channel_params"
    elif (not f["params"] & SURFACE_SN and f["min_depth"] is not None
          and f["min_depth"] > DEPTH_SKIP):
        key = "files_below_500m"
    else:
        return True
    counts[key] = counts.get(key, 0) + 1
    return False


def is_gtmba(attrs):
    for k in GTMBA_ATTRS:
        v = attrs.get(k)
        if v is not None and GTMBA.search(str(v)):
            return f"{k}={v}"
    return None


def ncml_attrs(xml_bytes):
    root = ET.fromstring(xml_bytes)
    out = {}
    for el in root:
        if el.tag.endswith("}attribute") or el.tag == "attribute":
            out[el.get("name")] = el.get("value")
    return out


# ================================================================= a file ==
def _attr(v, k, default=None):
    return v.getncattr(k) if k in v.ncattrs() else default


def _unit(v):
    return str(_attr(v, "units", "")).strip().lower().replace(" ", "")


# unit spellings, lower-cased with spaces removed; the uncommon ones were
# measured in the 2018-09 probe ("degreeofcelcius", ".001", "hectopascal",
# "millimetershour-1", "wattsmeter-2", "metersecond-1", "knots")
TEMP_UNITS = {"degree_celsius", "degree_c", "degrees_c", "degc", "deg_c",
              "celsius", "degrees_celsius", "°c", "degreec", "c",
              "degreeofcelcius", "degreesofcelsius", "degreeofcelsius",
              "degree_celcius", "celcius"}
SAL_UNITS = {"psu", "1", "0.001", "1e-3", "pss-78", "pss78", "",
             "psu(pss-78)", ".001", "practicalsalinityunit",
             "practicalsalinityunits", "practical_salinity_units"}
PRES_UNITS = {"hpa": 1.0, "mbar": 1.0, "millibars": 1.0, "millibar": 1.0,
              "mb": 1.0, "pa": 0.01, "hectopascal": 1.0, "hectopascals": 1.0}
RAIN_UNITS = {"mm/hr": 1.0, "mm/hour": 1.0, "mmh-1": 1.0, "mm/h": 1.0,
              "mm.h-1": 1.0, "millimeters/hour": 1.0,
              "millimetershour-1": 1.0, "millimeter/hour": 1.0,
              "millimetres/hour": 1.0, "mmhr-1": 1.0,
              "kgm-2s-1": 3600.0, "kg/m2/s": 3600.0, "mm/s": 3600.0,
              "m/s": 3.6e6}
SPEED_UNITS = {"m/s": 1.0, "meters/second": 1.0, "ms-1": 1.0, "m.s-1": 1.0,
               "meter/second": 1.0, "metersecond-1": 1.0,
               "meterssecond-1": 1.0, "cm/s": 0.01, "knots": 0.514444,
               "knot": 0.514444, "kt": 0.514444}
SW_UNITS = {"wm-2", "w/m2", "w/m^2", "w/m**2", "watt/m2", "watts/m2",
            "w.m-2", "w/meter2", "watts/meter2", "wattsmeter-2",
            "wattmeter-2", "watts/meter^2", "w/m-2"}
DIR_UNITS = {"degree", "degrees", "degree_true", "degrees_true", "deg",
             "degreet", "degrees_north"}


class OSFile:
    """One OceanSITES netCDF: its attributes, time axis and variables."""

    def __init__(self, path):
        import netCDF4
        self.time_units_fallback = None
        self.ds = netCDF4.Dataset(path)
        self.ds.set_auto_mask(False)
        self.ds.set_auto_scale(True)
        self.attrs = {k: self.ds.getncattr(k) for k in self.ds.ncattrs()}
        V = self.ds.variables
        tv = None
        for name, v in V.items():
            if _attr(v, "standard_name") == "time" and v.ndim == 1:
                if tv is None or name == "TIME":
                    tv = (name, v)
        if tv is None:
            raise FormatError("no 1-D time variable")
        self.tname, tvar = tv
        self.tdim = tvar.dimensions[0]
        units = str(_attr(tvar, "units", ""))
        m = TIME_UNITS.match(units)
        bare = BARE_ORIGIN.match(units)
        hint = (str(_attr(tvar, "comment", "")) + " " +
                str(_attr(tvar, "long_name", ""))).lower()
        if not m and bare and ("julian day" in hint or "days" in hint):
            # I-ORS/S-ORS write units "1950-01-01T00:00:00Z" with comment
            # "Julian days" (measured 2026-09-17): days since that origin
            m = TIME_UNITS.match("days since " + units)
            self.time_units_fallback = units
        if not m:
            raise FormatError(f"time units {units!r}")
        unit, y, mo, d, h, mi, se = m.groups()
        base = int(cm.days_from_civil(int(y), int(mo), int(d))) * 86400 + \
            int(h or 0) * 3600 + int(mi or 0) * 60 + int(se or 0)
        raw = np.asarray(tvar[:], np.float64)
        fill = _attr(tvar, "_FillValue")
        ok = np.isfinite(raw) & (np.abs(raw) < 1e20)
        if fill is not None:
            ok &= raw != fill
        self.t = np.where(ok, np.rint(raw * UNIT_S[unit] + base), 0
                          ).astype(np.int64)
        self.t_ok = ok

    def close(self):
        self.ds.close()

    def cadence(self):
        """Median spacing of the WHOLE file's valid times (a window can
        hold one sample of a monthly file); a single-time file ranks last."""
        t = np.unique(self.t[self.t_ok])
        return float(np.median(np.diff(t))) if t.size > 1 else float("inf")

    def position(self, sel, fallback):
        """(lat, lon) arrays for the selected times."""
        V = self.ds.variables
        out = []
        for sn, pref in (("latitude", "LATITUDE"), ("longitude", "LONGITUDE")):
            cands = [(n, v) for n, v in V.items()
                     if _attr(v, "standard_name") == sn or n == pref]
            cands.sort(key=lambda nv: (nv[0] != pref,
                                       nv[1].dimensions != (self.tdim,)))
            val = None
            for n, v in cands:
                a = np.asarray(v[:], np.float64)
                f = _attr(v, "_FillValue")
                bad = ~np.isfinite(a) | (np.abs(a) > 1e3)
                if f is not None:
                    bad |= a == f
                a = np.where(bad, np.nan, a)
                if v.dimensions == (self.tdim,):
                    val = a[sel]
                    med = np.nanmedian(a) if np.isfinite(a).any() else np.nan
                    val = np.where(np.isfinite(val), val, med)
                elif a.size and np.isfinite(a).any():
                    val = np.full(int(sel.sum()), float(
                        a.reshape(-1)[np.isfinite(a.reshape(-1))][0]))
                if val is not None and np.isfinite(val).any():
                    break
                val = None
            if val is None:
                fb = fallback[0 if sn == "latitude" else 1]
                val = np.full(int(sel.sum()), np.nan if fb is None else fb)
            out.append(val)
        return out

    def _vertical(self, v):
        """-> (column depths or None, index tuple builder) for a variable
        whose first dimension is TIME. Depth positive down; None if the
        variable has no vertical information (a surface variable)."""
        V = self.ds.variables
        dims = v.dimensions
        if not dims or dims[0] != self.tdim:
            return "bad", None
        vert = None
        take = []
        for d in dims[1:]:
            n = len(self.ds.dimensions[d])
            cv = V.get(d)
            sn = _attr(cv, "standard_name") if cv is not None else None
            is_z = sn in ("depth", "height") or d.upper().startswith(
                ("DEPTH", "DEP", "HEIGHT", "Z"))
            if cv is None and n > 1:
                # E1M3A: no coordinate for the vertical dimension; the depth
                # is a 2-D variable DEPH(TIME, DEPTH) — its per-column
                # median over time is the instrument's nominal depth
                for zn, zv in V.items():
                    if _attr(zv, "standard_name") == "depth" and \
                            zv.dimensions == (self.tdim, d):
                        za = np.asarray(zv[:], np.float64)
                        za[~np.isfinite(za) | (np.abs(za) >= 1e30)] = np.nan
                        with np.errstate(all="ignore"):
                            import warnings
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                z = np.nanmedian(za, axis=0)
                        if str(_attr(zv, "positive", "down")).lower() == "up":
                            z = -z
                        if vert is not None:
                            return "bad", None
                        vert = z
                        take.append(slice(None))
                        break
                else:
                    return "bad", None
                continue
            if is_z and cv is not None and cv.ndim == 1:
                z = np.asarray(cv[:], np.float64)
                pos = str(_attr(cv, "positive", "down")).lower()
                if sn == "height" or d.upper().startswith("HEIGHT") or \
                        pos == "up":
                    z = -z
                if vert is not None:
                    return "bad", None
                vert = z
                take.append(slice(None))
            elif n == 1:
                take.append(0)
            else:
                return "bad", None
        if vert is None:
            sd = _attr(v, "sensor_depth")
            if sd is not None:
                vert = np.array([float(np.asarray(sd).reshape(-1)[0])])
            else:
                for cname in str(_attr(v, "coordinates", "")).split() + \
                        ["DEPTH"]:
                    cv = V.get(cname)
                    if cv is None or self.tdim in cv.dimensions:
                        continue
                    sn = _attr(cv, "standard_name")
                    if not (sn == "depth" or (sn is None and
                                              cname.upper().startswith(
                                                  "DEPTH"))):
                        continue
                    za = np.asarray(cv[:], np.float64).reshape(-1)
                    za = za[np.isfinite(za) & (np.abs(za) < 1e30)]
                    if not za.size:
                        continue
                    # MLTS files give one depth per deployment
                    # (DEPTH_SST(Deployment)): usable when they agree
                    med = float(np.median(za))
                    if za.max() - za.min() > max(1.0, 0.05 * abs(med)):
                        continue
                    z = med
                    if str(_attr(cv, "positive", "down")).lower() == "up":
                        z = -z
                    vert = np.array([z])
                    break
            hv = V.get(v.name + "_H")
            if vert is None and hv is not None and \
                    _attr(hv, "standard_name") in ("height", "depth") and \
                    hv.dimensions == (self.tdim,):
                # SOTS ASIMET: TEMP_H(TIME), "Sensor height below water
                # level", positive down — the median is the sensor depth
                za = np.asarray(hv[:], np.float64)
                za = za[np.isfinite(za) & (np.abs(za) < 1e30)]
                if za.size:
                    z = float(np.median(za))
                    pos = str(_attr(hv, "positive", "down")).lower()
                    vert = np.array([-z if pos == "up" else z])
        return vert, tuple(take)

    def read(self, name, sel, counts):
        """(values (n, k), qcbits (n, k)) for the selected times; the
        declared fill, NaN and |x| >= 1e30 are NaN; QC 3/4/9 are NaN."""
        V = self.ds.variables
        v = V[name]
        vert, take = self._vertical(v)
        idx = np.flatnonzero(sel)
        i0, i1 = int(idx[0]), int(idx[-1]) + 1
        a = np.asarray(v[(slice(i0, i1),) + take], np.float64)
        a = a.reshape(i1 - i0, -1)[idx - i0]
        bad = ~np.isfinite(a) | (np.abs(a) >= 1e30)
        # WHOTS writes its fill as "FillValue" (no underscore): measured
        for k in ("_FillValue", "missing_value", "FillValue"):
            f = _attr(v, k)
            if f is not None:
                bad |= a == np.asarray(f, np.float64).reshape(-1)[0]
        a[bad] = np.nan
        qbits = np.ones(a.shape, np.uint8)          # no QC variable: bit 1
        qn = name + "_QC"
        if qn in V and V[qn].dimensions == v.dimensions:
            q = np.asarray(V[qn][(slice(i0, i1),) + take], np.int64)
            q = q.reshape(i1 - i0, -1)[idx - i0]
            rem = np.isfinite(a) & np.isin(q, BAD_QC)
            nrem = int(rem.sum())
            if nrem:
                counts["values_qc_removed"] = \
                    counts.get("values_qc_removed", 0) + nrem
            a[rem] = np.nan
            qbits = np.where((q == 0) | (q < 0) | (q > 9), 1,
                             np.where(np.isin(q, FLAG_QC), 2, 0)
                             ).astype(np.uint8)
        return a, qbits, vert


def _match_depths(z):
    """column depths -> ({standard index: column}, surface column or None)."""
    out = {}
    for j, d in enumerate(DEPTHS):
        tol = max(1.0, 0.05 * d)
        dist = np.abs(z - d)
        k = int(np.argmin(dist)) if z.size else -1
        if k >= 0 and dist[k] <= tol:
            out[j] = k
    surf = None
    s = np.flatnonzero((z >= -0.5) & (z <= SURFACE_MAX))
    if s.size:
        surf = int(s[np.argmin(z[s])])
    return out, surf


def read_os_file(path, meta, t_lo, t_hi, counts):
    """One file -> {"gtmba": reason} or {"groups": [(chan tuple, t, v, q,
    lat, lon)], "rank": ...}."""
    f = OSFile(path)
    try:
        why = is_gtmba(f.attrs)
        if why:
            return {"gtmba": why}
        if f.time_units_fallback:
            counts["files_time_units_bare_origin"] = \
                counts.get("files_time_units_bare_origin", 0) + 1
        sel = f.t_ok & (f.t >= t_lo) & (f.t <= t_hi)
        if not sel.any():
            counts["files_no_time_in_window"] = \
                counts.get("files_no_time_in_window", 0) + 1
            return {"groups": []}
        t = f.t[sel]
        lat, lon = f.position(sel, (meta.get("lat"), meta.get("lon")))
        mode = str(f.attrs.get("data_mode", meta.get("mode", ""))).strip()
        cad = f.cadence()
        if MAX_CADENCE < cad < float("inf"):
            # monthly means (the *-1M flux / merged products): not reports
            counts["files_cadence_over_1day"] = \
                counts.get("files_cadence_over_1day", 0) + 1
            return {"groups": []}
        rank = (cad, MODE_RANK.get(mode, 9), meta["name"])
        V = f.ds.variables
        by_sn = {}
        for name, v in V.items():
            sn = _attr(v, "standard_name")
            if sn in READ_SN and name != f.tname and v.ndim >= 1 and \
                    v.dimensions[0] == f.tdim:
                by_sn.setdefault(sn, []).append(name)
        groups = []
        bad_units = counts.setdefault("units_unrecognized", {})
        unsupported = counts.setdefault("vars_unsupported_shape", {})

        def note_unit(name):
            k = f"{meta['name']}:{name} {_unit(V[name])!r}"
            bad_units[k] = bad_units.get(k, 0) + 1

        def get(name, conv=None):
            vert, _take = f._vertical(V[name])
            if isinstance(vert, str):
                k = f"{meta['name']}:{name}"
                unsupported[k] = unsupported.get(k, 0) + 1
                return None
            a, q, vert = f.read(name, sel, counts)
            if conv is not None:
                a = conv(a)
            return a, q, vert

        def surface_col(a, q, vert):
            """a (n, k): the column to use for a surface/meteo variable."""
            if a.shape[1] == 1:
                return a[:, 0], q[:, 0]
            live = np.flatnonzero(np.isfinite(a).any(axis=0))
            if live.size == 1:              # E1M3A: one level holds the air
                k = int(live[0])
                return a[:, k], q[:, k]
            return None

        def first_surface(sns, units_ok, conv_of, chan):
            for sn in sns:
                for name in by_sn.get(sn, []):
                    u = _unit(V[name])
                    if units_ok is not None and u not in units_ok:
                        note_unit(name)
                        continue
                    r = get(name, conv_of(u) if conv_of else None)
                    if r is None:
                        continue
                    col = surface_col(r[0], r[1], r[2])
                    if col is None:
                        k = f"{meta['name']}:{name}"
                        unsupported[k] = unsupported.get(k, 0) + 1
                        continue
                    groups.append(((chan,), t, col[0][:, None],
                                   col[1][:, None], lat, lon))
                    return True
            return False

        first_surface(sorted(SN_AIRT), TEMP_UNITS, None, 0)
        first_surface(sorted(SN_PRES), set(PRES_UNITS),
                      lambda u: (lambda a: a * PRES_UNITS[u]), 5)
        first_surface(sorted(SN_SW), SW_UNITS, None, 6)
        first_surface(sorted(SN_RAIN), set(RAIN_UNITS),
                      lambda u: (lambda a: a * RAIN_UNITS[u]), 7)
        have_sst = first_surface(sorted(SN_SST), TEMP_UNITS, None, 1)
        have_sss = first_surface(sorted(SN_SSS), SAL_UNITS, None, 2)
        # wind: components, else speed and direction
        u_n = [n for n in by_sn.get("eastward_wind", [])]
        v_n = [n for n in by_sn.get("northward_wind", [])]
        done = False
        for un, vn in zip(u_n, v_n):
            if _unit(V[un]) not in SPEED_UNITS or \
                    _unit(V[vn]) not in SPEED_UNITS:
                note_unit(un)
                continue
            ru, rv = get(un), get(vn)
            cu = surface_col(*ru) if ru is not None else None
            cv_ = surface_col(*rv) if rv is not None else None
            if cu is None or cv_ is None:
                continue
            su, sv = SPEED_UNITS[_unit(V[un])], SPEED_UNITS[_unit(V[vn])]
            groups.append(((3, 4), t,
                           np.stack([cu[0] * su, cv_[0] * sv], 1),
                           np.stack([cu[1], cv_[1]], 1), lat, lon))
            done = True
            break
        if not done:
            for sname in by_sn.get("wind_speed", []):
                dname, sign = None, None
                for dsn, sg in (("wind_to_direction", 1.0),
                                ("wind_from_direction", -1.0)):
                    for dn in by_sn.get(dsn, []):
                        if V[dn].dimensions == V[sname].dimensions:
                            dname, sign = dn, sg
                            break
                    if dname:
                        break
                if dname is None:
                    continue
                if _unit(V[sname]) not in SPEED_UNITS or \
                        _unit(V[dname]) not in DIR_UNITS:
                    note_unit(sname)
                    continue
                rs, rd = get(sname), get(dname)
                cs = surface_col(*rs) if rs is not None else None
                cd = surface_col(*rd) if rd is not None else None
                if cs is None or cd is None:
                    continue
                spd = cs[0] * SPEED_UNITS[_unit(V[sname])]
                ang = np.deg2rad(cd[0])
                uu = np.round(sign * spd * np.sin(ang), 6)
                vv = np.round(sign * spd * np.cos(ang), 6)
                qq = np.maximum(cs[1], cd[1])
                groups.append(((3, 4), t, np.stack([uu, vv], 1),
                               np.stack([qq, qq], 1), lat, lon))
                counts["wind_from_speed_direction"] = \
                    counts.get("wind_from_speed_direction", 0) + 1
                break
        # the water column
        for sns, units, base, surf_chan, have in (
                (SN_TEMP, TEMP_UNITS, T0, 1, have_sst),
                (SN_SAL, SAL_UNITS, S0, 2, have_sss)):
            taken = set()
            for sn in sorted(sns):
                for name in by_sn.get(sn, []):
                    if _unit(V[name]) not in units:
                        note_unit(name)
                        continue
                    r = get(name)
                    if r is None:
                        continue
                    a, q, vert = r
                    if vert is None or isinstance(vert, str) or \
                            len(vert) != a.shape[1]:
                        nd = counts.setdefault("vars_no_depth", {})
                        k = f"{meta['name']}:{name}"
                        nd[k] = nd.get(k, 0) + 1
                        continue
                    m, surf = _match_depths(np.asarray(vert, np.float64))
                    for j, k in m.items():
                        if base + j in taken:
                            continue
                        taken.add(base + j)
                        groups.append(((base + j,), t, a[:, k:k + 1],
                                       q[:, k:k + 1], lat, lon))
                    if surf is not None and not have and \
                            surf_chan not in taken:
                        taken.add(surf_chan)
                        groups.append(((surf_chan,), t, a[:, surf:surf + 1],
                                       q[:, surf:surf + 1], lat, lon))
        return {"groups": groups, "rank": rank}
    finally:
        f.close()


def merge_site(sources, C, counts):
    """sources: [(rank, groups)] -> (t, lat, lon, values (n, C), qc)."""
    allt = [g[1] for _, gs in sources for g in gs]
    if not allt:
        return None
    t = np.unique(np.concatenate(allt))
    n = t.size
    vals = np.full((n, C), np.nan)
    qc = np.zeros(n, np.uint8)
    la = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    by_group = {}
    for rank, gs in sources:
        for g in gs:
            by_group.setdefault(g[0], []).append((rank, g))
    sup = 0
    for chans, lst in sorted(by_group.items()):
        lst.sort(key=lambda x: x[0])
        spans = []
        for rank, (_c, gt, gv, gq, glat, glon) in lst:
            fin = np.isfinite(gv).all(axis=1) if len(chans) > 1 else \
                np.isfinite(gv[:, 0])
            keep = fin.copy()
            for a, b in spans:
                inside = keep & (gt >= a) & (gt <= b)
                sup += int(inside.sum())
                keep &= ~inside
            if fin.any():
                spans.append((int(gt[fin].min()), int(gt[fin].max())))
            if not keep.any():
                continue
            ri = np.searchsorted(t, gt[keep])
            cols = list(chans)
            free = np.isnan(vals[ri][:, cols]).all(axis=1)
            ri2 = ri[free]
            src = np.flatnonzero(keep)[free]
            vals[np.ix_(ri2, cols)] = gv[src]
            qc[ri2] |= gq[src].max(axis=1)
            nopos = np.isnan(la[ri2])
            la[ri2[nopos]] = glat[src[nopos]]
            lo[ri2[nopos]] = glon[src[nopos]]
            dup = int((~free).sum())
            if dup:
                counts["values_same_second"] = \
                    counts.get("values_same_second", 0) + dup
    if sup:
        counts["values_superseded"] = counts.get("values_superseded", 0) + sup
    return t, la, lo, vals, qc


# ================================================================ adapter ==
class OceanSITESAdapter(f10b.SourceAdapter):
    store = "oceansites"
    title = ("OceanSITES moored time series (GDAC at NOAA NDBC), tropical "
             "arrays excluded: surface meteorology and T/S at ten depths, one "
             "row per site and time")
    family = "1gf"
    distribution = "public"
    licence = {"name": "OceanSITES data policy (free and unrestricted, "
                       "with acknowledgement)",
               "redistribution": "attribution", "derived_works": "free",
               "attribution": "OceanSITES (www.oceansites.org), the site "
                              "PIs and data assembly centres; GDAC at NOAA "
                              "NDBC"}
    time_dtype = "int32"
    platform_meta = True
    credentials = ()
    channels = CHANNELS
    log2_fp = -4.0
    log2_dt = float(np.log2((1.0 / 24.0) / 5.0))
    per_year = True
    first_year = 1950
    qc_policy = (
        "OceanSITES <VAR>_QC flags 3/4/9 -> NaN (counted); qc bit 1 = a "
        "kept value had no QC (0/fill/absent), bit 2 = a kept value flagged "
        "5/7/8. Per site and channel, sources are ranked finest cadence, "
        "then mode D/M/P/R, then name; values inside a better source's span "
        "are dropped. Out-of-bounds values -> NaN, counted.")
    sources = (FILES + "DATA/<SITE>/<file>.nc", INDEX,
               NCML + "DATA/<SITE>/<file>.nc")
    verified = (
        "2026-09-17 from the sandbox: oceansites_index.txt (20,727,619 B, "
        "60,027 files, 112 DATA sites); the THREDDS catalog (same 112 "
        "sites); NcML headers; files of NTAS (M, FLTS-1H), CCE1 MICROCAT, "
        "PAPA TEMP-10min, KEO ADCP (NetCDF-4), P12N23W TVSM (PIRATA), "
        "T0N110W AIRT (TAO): layouts, units, fills, QC tables, array "
        "attributes. Ifremer's GDAC hosts refused from the sandbox.")
    notes = (
        "Tropical-array sites (TAO/TRITON/PIRATA/RAMA) excluded by their "
        "own global attributes — family 10.1's gtmba. DATA_GRIDDED not "
        "read. Files entirely below 525 m with no surface parameter "
        "(the ALOHA Cabled Observatory) skipped by the index's depth.")
    smoke_window = ("2021-12-30", "2022-01-02")
    smoke_probe_month = "2022-01"
    fetch_month_scope = "month"

    def __init__(self):
        self._files = None
        self._gtmba = {}
        self._disk = None

    # ------------------------------------------------------------ listing --
    def _local(self, ctx, *p):
        return os.path.join(ctx.source_dir, "oceansites", *p)

    def _get(self, ctx, url, local):
        if ctx.source_dir:
            p = self._local(ctx, *local)
            if not os.path.exists(p):
                return None
            raw = open(p, "rb").read()
            ctx.count_bytes(len(raw))
            return raw
        # NDBC resets connections now and then (measured): retry longer
        raw, _why = cm.get_bytes(url, attempts=max(6, ctx.a.attempts))
        return raw

    def files(self, ctx):
        if self._files is None:
            raw = self._get(ctx, INDEX, ("oceansites_index.txt",))
            if raw is None:
                sys.exit(f"{INDEX} answered 404 — the GDAC index moved")
            try:
                fs, c = parse_index(raw.decode("latin-1"))
            except FormatError as e:
                sys.exit(f"REFUSING oceansites: {e}")
            keep = [f for f in fs if usable(f, c)]
            self._files = {"all": fs, "usable": keep, "counts": c}
        return self._files

    def _cache_path(self, ctx):
        return os.path.join(ctx.root, "oceansites_site_headers.json")

    def site_gtmba(self, ctx, site):
        """The site's decision, from the NcML header of its first usable
        file (index order). Cached beside plan.json, keyed by that file's
        path, so a later stage (a new process) does not ask again."""
        if self._disk is None:
            self._disk = {}
            try:
                with open(self._cache_path(ctx)) as fh:
                    self._disk = json.load(fh)
            except (OSError, ValueError):
                pass
        if site not in self._gtmba:
            first = next(f for f in self.files(ctx)["usable"]
                         if f["site"] == site)
            hit = self._disk.get(site)
            if hit and hit.get("path") == first["path"]:
                self._gtmba[site] = hit
                return hit
            raw = self._get(ctx, NCML + first["path"], ("ncml", first["path"]))
            if raw is None:
                sys.exit(f"REFUSING oceansites: no NcML header for "
                         f"{first['path']} — the site's array cannot be "
                         f"decided")
            try:
                attrs = ncml_attrs(raw)
            except ET.ParseError as e:
                sys.exit(f"REFUSING oceansites: NcML of {first['path']} is "
                         f"not XML ({e})")
            self._gtmba[site] = {"file": first["name"],
                                 "path": first["path"],
                                 "gtmba": is_gtmba(attrs),
                                 "attrs": {k: attrs.get(k)
                                           for k in GTMBA_ATTRS}}
            f10b.atomic_json(self._cache_path(ctx), self._gtmba)
        return self._gtmba[site]

    def selected(self, ctx, lo, hi):
        """Usable, non-GTMBA files overlapping [lo, hi] -> {site: [file]}."""
        out, c = {}, {}
        for f in self.files(ctx)["usable"]:
            t0, t1 = f["t0"], f["t1"]
            if t0 is None or t1 is None:
                c["files_undated"] = c.get("files_undated", 0) + 1
            elif t1 < lo or t0 > hi:
                continue
            out.setdefault(f["site"], []).append(f)
        sites = {}
        excl = []
        for s in sorted(out):
            if self.site_gtmba(ctx, s)["gtmba"]:
                excl.append(s)
                c["files_excluded_gtmba_site"] = \
                    c.get("files_excluded_gtmba_site", 0) + len(out[s])
                continue
            sites[s] = sorted(out[s], key=lambda f: f["name"])
        c["sites_excluded_gtmba"] = excl
        return sites, c

    # ------------------------------------------------------------ reading --
    def _fetch(self, ctx, f):
        if ctx.source_dir:
            p = self._local(ctx, f["path"])
            if not os.path.exists(p):
                raise f10b._NotFound(p)
            ctx.count_bytes(os.path.getsize(p))
            return p, False
        dest = os.path.join(ctx.scratch, "oceansites",
                            f["path"].replace("/", "__"))
        # the index holds names with spaces ("OS_I-ORS_2019_D_atmos_AIRT
        # RELH.nc"); the request path must be percent-encoded
        url = FILES + urllib.parse.quote(f["path"], safe="/")
        err = None
        # NDBC resets connections and cuts bodies short now and then
        # (measured in three probe runs: one file each time); six tries
        for i in range(max(6, ctx.a.attempts)):
            try:
                f10b.http_to_file(url, dest)
                return dest, True
            except f10b._NotFound:
                raise
            except (IOError, *cm.RETRY_ERRORS) as e:
                err = e
                time.sleep(min(60.0, 3.0 * (2 ** i)))
        raise IOError(f"{url}: {err}")

    def _rows(self, ctx, lo, hi, year, label):
        y0 = int(cm.days_from_civil(year, 1, 1)) * 86400
        y1 = int(cm.days_from_civil(year + 1, 1, 1)) * 86400 - 1
        lo, hi = max(lo, y0), min(hi, y1)
        sites, counts = self.selected(ctx, lo, hi)
        counts["sites_selected"] = len(sites)
        counts["files_selected"] = sum(len(v) for v in sites.values())
        counts["bytes_listed"] = sum(f["size"] for v in sites.values()
                                     for f in v)
        for site in sorted(sites):
            fl = sites[site]

            def get(f):
                try:
                    return f, self._fetch(ctx, f), None
                except f10b._NotFound:
                    return f, None, "404"
                except IOError as e:
                    return f, None, str(e)

            sources = []
            workers = 1 if ctx.source_dir else WORKERS
            used = counts.setdefault("files_supplying", {})
            for f, got, err in cm.ordered_map(get, fl, workers):
                if got is None:
                    ctx.note_absent(label, f"{f['path']}: listed, {err}")
                    continue
                path, tmp = got
                try:
                    r = read_os_file(path, f, lo, hi, counts)
                except OSError as e:
                    # the bytes could not be opened: a broken download
                    ctx.note_absent(label, f"{f['path']}: unreadable "
                                           f"({type(e).__name__}: {e})")
                    continue
                except (FormatError, KeyError, ValueError, IndexError) as e:
                    # the bytes were read and the layout is refused: named
                    # and counted, not an absence (a retry reads the same)
                    # a COUNT per path (counters merge by addition across
                    # sites and parts) and the reason in a list ledger
                    rl = counts.setdefault("files_layout_refused", {})
                    rl[f["path"]] = rl.get(f["path"], 0) + 1
                    counts.setdefault("files_layout_refused_why", []).append(
                        f"{f['path']}: {type(e).__name__}: {e}"[:240])
                    continue
                finally:
                    if tmp and os.path.exists(path):
                        os.remove(path)
                counts["files_read"] = counts.get("files_read", 0) + 1
                if "gtmba" in r:
                    counts["files_excluded_gtmba_attr"] = \
                        counts.get("files_excluded_gtmba_attr", 0) + 1
                    continue
                if r["groups"]:
                    sources.append((r["rank"], r["groups"]))
                    for g in r["groups"]:
                        for ch in g[0]:
                            nm = CHANNELS[ch][0]
                            used[nm] = used.get(nm, 0) + 1
            m = merge_site(sources, self.C, counts)
            if m is None:
                continue
            t, la, lo_, v, qc = m
            oob = self.mask_bounds(v)
            if oob:
                f10b._merge_counts(counts, {"out_of_bounds": oob})
            half = np.isnan(v[:, 3]) ^ np.isnan(v[:, 4])
            if half.any():
                v[half, 3:5] = np.nan
                counts["wind_half_blanked"] = \
                    counts.get("wind_half_blanked", 0) + int(half.sum())
            alive = np.isfinite(v).any(axis=1) & np.isfinite(la) & \
                np.isfinite(lo_)
            counts["rows_no_value"] = counts.get("rows_no_value", 0) + \
                int((~np.isfinite(v).any(axis=1)).sum())
            if not alive.any():
                continue
            n = int(alive.sum())
            cs = counts.setdefault("rows_by_site", {})
            cs[site] = cs.get(site, 0) + n
            qd = counts.setdefault("qc", {})
            for u, k in zip(*np.unique(qc[alive], return_counts=True)):
                qd[str(int(u))] = qd.get(str(int(u)), 0) + int(k)
            yield label, self.pack(t[alive], la[alive], lo_[alive], v[alive],
                                   np.full(n, f10b.platform_hash(site),
                                           np.int64), qc[alive]), None
        yield label, None, counts

    # --------------------------------------------------------- the contract --
    def index(self, ctx):
        fs = self.files(ctx)
        allf, use = fs["all"], fs["usable"]
        sites = sorted({f["site"] for f in use})
        dec = {s: self.site_gtmba(ctx, s) for s in sites}
        per = {}
        for y in ctx.years:
            y0 = int(cm.days_from_civil(y, 1, 1)) * 86400
            y1 = int(cm.days_from_civil(y + 1, 1, 1)) * 86400 - 1
            sel, c = self.selected(ctx, max(y0, ctx.t_lo), min(y1, ctx.t_hi))
            per[str(y)] = {"sites": len(sel),
                           "files": sum(len(v) for v in sel.values()),
                           "bytes": sum(f["size"] for v in sel.values()
                                        for f in v)}
        first = None
        cand = []
        if ctx.years:
            y = ctx.years[0]
            y0 = int(cm.days_from_civil(y, 1, 1)) * 86400
            y1 = int(cm.days_from_civil(y + 1, 1, 1)) * 86400 - 1
            sel, _c = self.selected(ctx, max(y0, ctx.t_lo),
                                    min(y1, ctx.t_hi))
            cand = [f for v in sel.values() for f in v
                    if f["t0"] is not None]
        if cand:
            f = min(cand, key=lambda x: (x["size"] or 1 << 62, x["name"]))
            path, tmp = self._fetch(ctx, f)
            try:
                o = OSFile(path)
                first = {"file": f["path"], "times": int(o.t_ok.sum()),
                         "attrs": {k: str(o.attrs.get(k)) for k in
                                   GTMBA_ATTRS + ("data_mode",)}}
                o.close()
            finally:
                if tmp:
                    os.remove(path)
        return {"dataset": "OceanSITES GDAC (NDBC)", "url": INDEX,
                "index": {"files": len(allf), "usable": len(use),
                          **fs["counts"],
                          "bytes_all": sum(f["size"] for f in allf),
                          "bytes_usable": sum(f["size"] for f in use)},
                "sites": len(sites),
                "sites_excluded_gtmba": sorted(s for s in sites
                                               if dec[s]["gtmba"]),
                "site_decisions": {s: dec[s]["gtmba"] for s in sites},
                "per_year": per, "first_record": first}

    def fetch_year(self, ctx, year):
        t0 = time.time()
        agg = {}
        for label, rows, c in self._rows(ctx, ctx.t_lo, ctx.t_hi, year,
                                         str(year)):
            if c:
                f10b._merge_counts(agg, c)
            if rows is not None:
                yield label, rows, None
        agg["fetch_seconds"] = round(time.time() - t0, 1)
        yield str(year), None, agg

    def fetch_month(self, ctx, year, month):
        lo, hi = f10b.month_bounds_s(year, month)
        lo, hi = max(lo, ctx.t_lo), min(hi, ctx.t_hi)
        yield from self._rows(ctx, lo, hi, year, f"{year}-{month:02d}")

    def platforms(self, ctx):
        use = self.files(ctx)["usable"]
        out = {}
        for s in sorted({f["site"] for f in use}):
            if self.site_gtmba(ctx, s)["gtmba"]:
                continue
            fl = [f for f in use if f["site"] == s]
            la = [f["lat"] for f in fl if f["lat"] is not None]
            lo = [f["lon"] for f in fl if f["lon"] is not None]
            t0 = [f["t0"] for f in fl if f["t0"] is not None]
            t1 = [f["t1"] for f in fl if f["t1"] is not None]
            out[f10b.platform_hash(s)] = {
                "id": s, "lat": float(np.median(la)) if la else None,
                "lon": float(f10b.f10.wrap_lon(np.median(lo))) if lo
                else None,
                "files": len(fl),
                "first": _iso(min(t0)) if t0 else None,
                "last": _iso(max(t1)) if t1 else None,
                "array_attrs": self.site_gtmba(ctx, s)["attrs"]}
        return out

    # ---------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260917):
        return make_smoke_sources(root, d_lo, d_hi, seed)


def _iso(s):
    return (dt.datetime(1982, 1, 1) + dt.timedelta(seconds=int(s))
            ).strftime("%Y-%m-%dT%H:%M:%SZ")


# ================================================================== smoke ==
def _nc(path, fmt, times_s, dims, variables, gattrs):
    """A minimal OceanSITES-layout file. dims: {name: size}; variables:
    [(name, dtype, dims, data, attrs)]. TIME is days since 1950."""
    import netCDF4
    ds = netCDF4.Dataset(path, "w", format=fmt)
    ds.createDimension("TIME", len(times_s))
    for d, n in dims.items():
        ds.createDimension(d, n)
    for k, v in gattrs.items():
        ds.setncattr(k, v)
    tv = ds.createVariable("TIME", "f8", ("TIME",))
    tv.standard_name = "time"
    tv.units = "days since 1950-01-01T00:00:00Z"
    e = int(cm.days_from_civil(1950, 1, 1)) * 86400
    tv[:] = (np.asarray(times_s, np.float64) - e) / 86400.0
    for name, dtype, vd, data, attrs in variables:
        fill = attrs.pop("_FillValue", None)
        v = ds.createVariable(name, dtype, vd, fill_value=fill)
        for k, a in attrs.items():
            v.setncattr(k, a)
        v[:] = data
    ds.close()


def _ncml(path, gattrs):
    body = "".join(f'  <attribute name="{k}" value="{v}" />\n'
                   for k, v in gattrs.items())
    with open(path, "w") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n<netcdf xmlns="'
                 'http://www.unidata.ucar.edu/namespaces/netcdf/ncml-2.2" '
                 'location="Not provided because of security concerns.">\n'
                 + body + '</netcdf>\n')


def make_smoke_sources(root, d_lo, d_hi, seed=20260917):
    """The NDBC GDAC layout: oceansites_index.txt (real header, a broken
    continuation line), DATA/<SITE>/<file>.nc, and the NcML headers.

    NTAS (kept): a 10-minute NetCDF-3 met file with scalar DEPTH 1 m (SST,
    SSS from TEMP/PSAL), no QC variables, a declared fill, an air
    temperature out of bounds; a 30-minute (TIME, DEPTH) MicroCAT file with
    QC flags 4 and 8, depths 10.4, 20, 47, 100, 160, 500, 800 m; an hourly
    flux product whose air temperature is superseded where the 10-minute
    file has data and fills the day before it starts. KEO (kept): NetCDF-4,
    (TIME, HEIGHT) wind speed and to-direction, air pressure in Pa, TEMP on
    (TIME, DEPTH_TEMP, LATITUDE, LONGITUDE) at 200 m with a QC 3 flag,
    longitude 215 E. T0N110W (TAO by `network`), an ALOHA Cabled
    Observatory file (4,726 m), a currents-only file, a DATA_GRIDDED file
    and a 2019 file are listed and never opened (their bytes are garbage).
    """
    rng = np.random.default_rng(seed)
    base = os.path.join(root, "oceansites")
    w_lo = f10b.seconds_since_epoch(d_lo)
    w_hi = f10b.seconds_since_epoch(d_hi) + 86399
    day0 = f10b.seconds_since_epoch(d_lo)          # Dec 30 00:00
    H, M10, M30 = 3600, 600, 1800
    index = []
    truth = {}

    def row(site, t):
        return truth.setdefault((site, t), {"v": [np.nan] * len(CHANNELS),
                                            "qc": 0, "site": site})

    def add_index(path, t0, t1, la, lo, mind, maxd, size, mode, params):
        index.append(f"{path},2026-01-01T00:00:00Z,{_iso(t0)},{_iso(t1)},"
                     f"{la:.2f},{la:.2f},{lo:.2f},{lo:.2f},{mind},{maxd},"
                     f"void,{size},2026-01-01T00:00:00Z,"
                     f"2026-01-01T00:00:00Z,{mode},{params}")

    def write(site, name, fmt, times, dims, variables, gattrs, la, lo,
              mind, maxd, mode, params):
        d = os.path.join(base, "DATA", site)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, name)
        _nc(p, fmt, times, dims, variables, dict(gattrs))
        nd = os.path.join(base, "ncml", "DATA", site)
        os.makedirs(nd, exist_ok=True)
        _ncml(os.path.join(nd, name), gattrs)
        add_index(f"DATA/{site}/{name}", int(times[0]), int(times[-1]), la,
                  lo, mind, maxd, os.path.getsize(p), mode, params)

    def garbage(tree, site, name, t0, t1, la, lo, mind, maxd, params,
                gattrs=None):
        d = os.path.join(base, tree, site)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, name), "wb") as fh:
            fh.write(b"never read")
        if gattrs is not None:
            nd = os.path.join(base, "ncml", tree, site)
            os.makedirs(nd, exist_ok=True)
            _ncml(os.path.join(nd, name), gattrs)
        add_index(f"{tree}/{site}/{name}", t0, t1, la, lo, mind, maxd, 10,
                  "D", params)

    # ------------------------------------------------------------- NTAS ----
    la, lo = 14.8, -51.0
    g_ntas = {"project": "NTAS", "data_type": "OceanSites time-series data",
              "site_code": "NTAS", "platform_code": "NTAS", "data_mode": "D",
              "title": "ASIMet logger data from surface mooring NTAS"}
    # A: 10-minute met, Dec 31 00:00 .. Jan 2 23:50
    ta = np.arange(day0 + 86400, w_hi + 1, M10)
    n = ta.size
    airt = np.round(25 + np.sin(np.arange(n) / 30.0), 3)
    airt[5] = 99.0                                     # out of bounds
    sst = np.round(26 + 0.1 * np.cos(np.arange(n) / 50.0), 3)
    sss = np.round(35.5 + 0.01 * np.arange(n) / n, 4)
    uw = np.round(rng.normal(-5, 1, n), 3)
    vw = np.round(rng.normal(2, 1, n), 3)
    atms = np.round(1013 + rng.normal(0, 1, n), 2)
    sw = np.round(np.clip(800 * np.sin(np.arange(n) / 20.0), 0, None), 2)
    sw[7] = -99999.0                                   # declared fill
    rain = np.round(np.abs(rng.normal(0, 0.5, n)), 3)
    one = {"_FillValue": -99999.0}
    write("NTAS", "OS_NTAS_2021_D_M.nc", "NETCDF3_CLASSIC", ta,
          {"LATITUDE": 1, "LONGITUDE": 1},
          [("LATITUDE", "f4", ("LATITUDE",), [la],
            {"standard_name": "latitude", "units": "degrees_north"}),
           ("LONGITUDE", "f4", ("LONGITUDE",), [lo],
            {"standard_name": "longitude", "units": "degrees_east"}),
           ("DEPTH", "f4", (), 1.0, {"standard_name": "depth",
                                     "units": "meters", "positive": "down"}),
           ("AIRT", "f8", ("TIME",), airt,
            {"standard_name": "air_temperature", "units": "degree_C"}),
           ("TEMP", "f8", ("TIME",), sst,
            {"standard_name": "sea_water_temperature", "units": "degree_C"}),
           ("PSAL", "f8", ("TIME",), sss,
            {"standard_name": "sea_water_practical_salinity", "units": "1"}),
           ("UWND", "f8", ("TIME",), uw,
            {"standard_name": "eastward_wind", "units": "meters/second"}),
           ("VWND", "f8", ("TIME",), vw,
            {"standard_name": "northward_wind", "units": "meters/second"}),
           ("ATMS", "f8", ("TIME",), atms,
            {"standard_name": "air_pressure", "units": "millibars"}),
           ("SW", "f8", ("TIME",), sw,
            dict(one, standard_name="surface_downwelling_shortwave_flux_in_"
                 "air", units="W m-2")),
           ("RAIN", "f8", ("TIME",), rain,
            {"standard_name": "rainfall_rate", "units": "mm/hour"})],
          g_ntas, la, lo, "-0.0", "1.0", "D",
          "time latitude longitude depth air_temperature "
          "sea_water_temperature sea_water_practical_salinity eastward_wind "
          "northward_wind air_pressure surface_downwelling_shortwave_flux_in_"
          "air rainfall_rate")
    for i, t in enumerate(ta):
        if not (w_lo <= t <= w_hi):
            continue
        r = row("NTAS", int(t))
        vv = [airt[i], sst[i], sss[i], uw[i], vw[i], atms[i], sw[i], rain[i]]
        vv[0] = np.nan if vv[0] > 60 else vv[0]
        vv[6] = np.nan if vv[6] < -1000 else vv[6]
        r["v"][:8] = [float(x) for x in vv]
        r["qc"] |= 1
    # B: 30-minute MicroCAT column, Dec 29 00:00 .. Jan 2 23:30
    tb = np.arange(day0 - 86400, w_hi + 1, M30)
    nb = tb.size
    zb = np.array([10.4, 20.0, 47.0, 100.0, 160.0, 500.0, 800.0])
    temp = np.round(26 - zb[None, :] / 40 + rng.normal(0, 0.01,
                                                       (nb, zb.size)), 3)
    psal = np.round(35.8 + zb[None, :] / 2000 + np.zeros((nb, zb.size)), 3)
    tq = np.ones((nb, zb.size), np.int8)
    tq[100, 0] = 4                                   # bad: removed
    tq[101, 1] = 8                                   # interpolated: kept
    sq = np.ones((nb, zb.size), np.int8)
    write("NTAS", "OS_NTAS_2021_D_MICROCAT.nc", "NETCDF3_CLASSIC", tb,
          {"DEPTH": zb.size, "LATITUDE": 1, "LONGITUDE": 1},
          [("LATITUDE", "f4", ("LATITUDE",), [la],
            {"standard_name": "latitude", "units": "degrees_north"}),
           ("LONGITUDE", "f4", ("LONGITUDE",), [lo],
            {"standard_name": "longitude", "units": "degrees_east"}),
           ("DEPTH", "f4", ("DEPTH",), zb,
            {"standard_name": "depth", "units": "meters", "positive": "down"}),
           ("TEMP", "f4", ("TIME", "DEPTH"), temp,
            {"standard_name": "sea_water_temperature",
             "units": "degree_Celsius", "_FillValue": np.float32(-99999.0)}),
           ("TEMP_QC", "i1", ("TIME", "DEPTH"), tq,
            {"_FillValue": np.int8(-128)}),
           ("PSAL", "f4", ("TIME", "DEPTH"), psal,
            {"standard_name": "sea_water_practical_salinity", "units": "psu",
             "_FillValue": np.float32(-99999.0)}),
           ("PSAL_QC", "i1", ("TIME", "DEPTH"), sq,
            {"_FillValue": np.int8(-128)})],
          g_ntas, la, lo, "10.4", "800.0", "D",
          "time latitude longitude depth sea_water_temperature "
          "sea_water_practical_salinity")
    colmap = {0: 0, 1: 1, 5: 3, 9: 5}               # standard j -> column k
    for i, t in enumerate(tb):
        if not (w_lo <= t <= w_hi):
            continue
        r = row("NTAS", int(t))
        for j, k in colmap.items():
            tv = float(np.float32(temp[i, k])) if tq[i, k] != 4 else np.nan
            r["v"][T0 + j] = tv
            r["v"][S0 + j] = float(np.float32(psal[i, k]))
            if tq[i, k] == 8:
                r["qc"] |= 2
    # C: hourly flux product, Dec 28 .. Jan 2 23:00 (air temperature only)
    tc = np.arange(day0 - 2 * 86400, w_hi, H)
    fl = np.round(20 + 0.01 * np.arange(tc.size), 3)
    write("NTAS", "OS_NTAS_2021_D_FLTS-1H.nc", "NETCDF3_CLASSIC", tc,
          {"LATITUDE": 1, "LONGITUDE": 1},
          [("LATITUDE", "f4", ("LATITUDE",), [la],
            {"standard_name": "latitude", "units": "degrees_north"}),
           ("LONGITUDE", "f4", ("LONGITUDE",), [lo],
            {"standard_name": "longitude", "units": "degrees_east"}),
           ("TA_H", "f8", ("TIME",), fl,
            {"standard_name": "air_temperature", "units": "degree_C"}),
           ("TSKIN", "f8", ("TIME",), fl,
            {"standard_name": "sea_surface_skin_temperature",
             "units": "degree"})],
          g_ntas, la, lo, "-10.0", "-0.0", "D",
          "time latitude longitude air_temperature "
          "sea_surface_skin_temperature")
    a_lo, a_hi = int(ta[0]), int(ta[-1])
    for i, t in enumerate(tc):
        if not (w_lo <= t <= w_hi) or a_lo <= t <= a_hi:
            continue
        r = row("NTAS", int(t))
        r["v"][0] = float(fl[i])
        r["qc"] |= 1
    # -------------------------------------------------------------- KEO ----
    kla, klo = 32.3, 215.0
    tk = np.arange(day0 + 2 * 86400, w_hi + 1, H)
    nk = tk.size
    spd = np.round(np.abs(rng.normal(8, 2, nk)), 3)
    dirn = np.round(rng.uniform(0, 360, nk), 2)
    pa = np.round(101300 + rng.normal(0, 50, nk), 1)
    t200 = np.round(12 + rng.normal(0, 0.1, nk), 3)
    kq = np.ones(nk, np.int8)
    kq[3] = 3                                         # bad: removed
    g_keo = {"project": "Ocean Climate Stations (OCS)", "network": "OCS",
             "site_code": "KEO", "platform_code": "KEO", "data_mode": "D",
             "title": "OceanSITES Station KEO", "id": "OS_KEO_2021_D_MET"}
    write("KEO", "OS_KEO_2021_D_MET.nc", "NETCDF4", tk,
          {"HEIGHT": 1, "DEPTH_TEMP": 1, "LATITUDE": 1, "LONGITUDE": 1},
          [("LATITUDE", "f4", ("LATITUDE",), [kla],
            {"standard_name": "latitude", "units": "degrees_north"}),
           ("LONGITUDE", "f4", ("LONGITUDE",), [klo],
            {"standard_name": "longitude", "units": "degrees_east"}),
           ("HEIGHT", "f4", ("HEIGHT",), [4.0],
            {"standard_name": "height", "units": "m", "positive": "up"}),
           ("DEPTH_TEMP", "f4", ("DEPTH_TEMP",), [200.0],
            {"standard_name": "depth", "units": "m", "positive": "down"}),
           ("WSPD", "f4", ("TIME", "HEIGHT"), spd[:, None],
            {"standard_name": "wind_speed", "units": "m/s",
             "_FillValue": np.float32(1e35)}),
           ("WDIR", "f4", ("TIME", "HEIGHT"), dirn[:, None],
            {"standard_name": "wind_to_direction", "units": "degree",
             "_FillValue": np.float32(1e35)}),
           ("ATMS", "f8", ("TIME", "HEIGHT"), pa[:, None],
            {"standard_name": "air_pressure_at_sea_level", "units": "Pa"}),
           ("TEMP", "f4", ("TIME", "DEPTH_TEMP", "LATITUDE", "LONGITUDE"),
            t200[:, None, None, None],
            {"standard_name": "sea_water_temperature",
             "units": "degree_Celsius", "_FillValue": np.float32(np.nan)}),
           ("TEMP_QC", "i1", ("TIME", "DEPTH_TEMP", "LATITUDE", "LONGITUDE"),
            kq[:, None, None, None], {"_FillValue": np.int8(-128)})],
          g_keo, kla, klo, "-4.0", "200.0", "D",
          "time height depth latitude longitude wind_speed "
          "wind_to_direction air_pressure_at_sea_level "
          "sea_water_temperature")
    for i, t in enumerate(tk):
        r = row("KEO", int(t))
        s = float(np.float32(spd[i]))
        a = np.deg2rad(float(np.float32(dirn[i])))
        r["v"][3] = round(s * np.sin(a), 6)
        r["v"][4] = round(s * np.cos(a), 6)
        r["v"][5] = float(pa[i]) * 0.01
        r["v"][T0 + 7] = float(np.float32(t200[i])) if kq[i] != 3 else np.nan
        r["qc"] |= 1                     # wind and pressure carry no QC
    # ------------------------------------------------------ never opened ----
    tao = {"network": "TAO", "site_code": "T0N110W", "data_mode": "D",
           "title": "TAO Delayed/Mixed Mode Data",
           "id": "TAO_T0N110W_DM072A-20140621_D_AIRT_10min"}
    garbage("DATA", "T0N110W", "OS_T0N110W_DM134A-20211001_D_AIRT_10min.nc",
            day0 - 90 * 86400, w_hi, 0.0, -110.0, "-3.0", "-3.0",
            "time height latitude longitude air_temperature", tao)
    garbage("DATA", "T0N110W", "OS_T0N110W_DM134A-20211001_D_SST_10min.nc",
            day0 - 90 * 86400, w_hi, 0.0, -110.0, "1.0", "1.0",
            "time depth latitude longitude sea_surface_temperature", tao)
    garbage("DATA", "ALOHA", "OS_ACO_20211231-00-08_P_CTD3-4726m.nc",
            day0 + 2 * 86400, day0 + 2 * 86400 + 8 * H, 22.74, -158.01,
            "4726.50", "4726.50",
            "time latitude longitude depth sea_water_temperature "
            "sea_water_practical_salinity")
    garbage("DATA", "NTAS", "OS_NTAS_2021_D_ADCP.nc", day0, w_hi, la, lo,
            "20.0", "200.0", "time latitude longitude depth "
            "eastward_sea_water_velocity northward_sea_water_velocity")
    garbage("DATA_GRIDDED", "PIRATA", "OS_0n0e_199802_M_TSM_dy.nc", day0,
            w_hi, 0.0, 0.0, "1.0", "500.0",
            "time latitude longitude depth sea_water_temperature")
    add_index("DATA/NTAS/OS_NTAS_2019_D_M.nc",
              f10b.seconds_since_epoch(dt.date(2019, 1, 1)),
              f10b.seconds_since_epoch(dt.date(2019, 12, 31)), la, lo,
              "-0.0", "1.0", 10, "D", "time air_temperature")
    head = ("#OceanSITES Global Data Assembly Center (GDAC) Index File v2.0\n"
            "#Two GDACs FTP servers are on-line at ftp://data.ndbc.noaa.gov/"
            "data/oceansites and ftp://ftp.ifremer.fr/ifremer/oceansites\n"
            "#Also the datasets are available at the THREDDS server http://"
            "dods.ndbc.noaa.gov/thredds/catalog/data/oceansites/catalog.html"
            "\n#For more information, please contact: http://www.oceansites."
            "org\n#\n#This OceanSITES index file was last updated on : "
            "2026-09-17T05:50:01Z. Columns are defined as follows:\n#FILE "
            "(relative to current file directory),DATE_UPDATE,START_DATE,"
            "END_DATE,SOUTHERN_MOST_LATITUDE,NORTHERN_MOST_LATITUDE,"
            "WESTERN_MOST_LONGITUDE,EASTERN_MOST_LONGITUDE,MINIMUM_DEPTH,"
            "MAXIMUM_DEPTH,UPDATE_INTERVAL,SIZE (in bytes),GDAC_CREATION_DATE,"
            "GDAC_UPDATE_DATE,DATA_MODE (R: real-time D: delayed mode M: "
            "mixed P: provisional),PARAMETERS (space delimited CF standard "
            "names)\n#\n")
    index.insert(3, " values surface_partial_pressure_of_carbon_dioxide_in_"
                    "sea_water")
    with open(os.path.join(base, "oceansites_index.txt"), "w") as fh:
        fh.write(head + "\n".join(index) + "\n")
    site_rank = {"KEO": 0, "NTAS": 1}
    out = []
    for (site, t), r in sorted(truth.items(),
                               key=lambda kv: (kv[0][1], site_rank[kv[0][0]])):
        if not any(np.isfinite(x) for x in r["v"]):
            continue
        out.append({"t": t, "lat": kla if site == "KEO" else la,
                    "lon": float(f10b.f10.wrap_lon(klo)) if site == "KEO"
                    else lo, "platform": f10b.platform_hash(site),
                    "v": r["v"], "qc": r["qc"]})
    return out


ADAPTER = OceanSITESAdapter
