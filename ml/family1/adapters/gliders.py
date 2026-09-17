"""Ocean gliders (IOOS Glider DAC) — one row per profile, 64 channels.

PLAIN ENGLISH. Underwater gliders are small autonomous vehicles that dive and
climb through the upper ocean for weeks, measuring temperature, salinity and
often oxygen and chlorophyll on every dive. The US Integrated Ocean Observing
System collects them in its Glider Data Assembly Center (DAC). This adapter
turns each dive or climb into one row: the four quantities at the same 16
Roemmich-Gilson pressures the Argo family uses, so a glider profile and an
Argo profile can be read side by side.

SOURCE, VERIFIED 2026-09-17 FROM THIS SANDBOX (no account; the DAC's files
say "This data may be redistributed and used without restriction"):
  https://gliders.ioos.us/erddap/tabledap/allDatasets.csv      (the list)
  https://gliders.ioos.us/erddap/info/<datasetID>/index.csv    (variables)
  https://gliders.ioos.us/thredds/catalog/deployments/catalog.xml
  https://gliders.ioos.us/thredds/fileServer/deployments/<user>/<id>/<id>.nc3.nc
The ERDDAP list holds 2,541 TrajectoryProfile datasets (the note's 2,540), 868
of them `<deployment>-delayed` twins of a real-time dataset, first profile
2003-10-28. Their data through ERDDAP tabledap is SLOW: one delayed
deployment's month (bass-20230610T0000-delayed, nine variables) delivered
142 MB of CSV at 244 kB/s and had not finished after 9 min 40 s. The same
DAC's THREDDS serves each deployment as ONE aggregated NetCDF-3 file
(`<id>.nc3.nc`, dimensions trajectory x profile x obs, obs padded to the
longest profile; either trajectory = 1 — angus-20230526T0051 — or profile = 1
with one "trajectory" per dive — sg180-20230417T2100, 319 x 1 x 2401, a
Seaglider) at 90 MB/s with `Accept-Ranges: bytes` — angus-
20200319T0000.nc3.nc is 104,213,412 bytes, 12,440 profiles x 71 obs — so
the adapter reads each file's header by Range request and then ONLY the
variables it needs (`_nc3.py`). The THREDDS tree is 57 user directories
holding 3,064 deployment directories (523 not published in ERDDAP — test and
unpublished deployments — which are not read); a delayed deployment's
directory can exist with no file yet (bass-20230610T0000-delayed, measured).
Both hosts RESET connections intermittently under parallel load; every
request retries with backoff and the pool is `WORKERS = 2`.

THE OceanGliders GDAC WAS NOT IMPLEMENTED. Found: Ifremer's ERDDAP lists
`OceanGlidersGDACTrajectories` (erddap.ifremer.fr; Trajectory, time
1970 .. 2026-09-11; PRES/TEMP/PSAL/MOLAR_DOXY/CHLA with *_QC, PHASE_NUMBER;
licence "Follows CLIVAR standards") — one trajectory table of samples, not
profiles, which would need its own segmentation into dives. The file hosts
(data-glider.ifremer.fr, data-gliders.ifremer.fr: proxy 502;
ftp.ifremer.fr/ifremer/glider/v2: connection reset / timeout) did not answer
from this sandbox. Many OceanGliders deployments are also in the IOOS DAC.

VARIABLES, CHOSEN FROM EACH DATASET'S OWN METADATA (ERDDAP info), never by
position: `time`, `latitude`, `longitude` (per profile), `pressure`,
`temperature`, `salinity` (per sample; the DAC's required NGDAC names), and
  O2   the variable whose standard_name is moles_of_oxygen_per_unit_mass_in_
       sea_water (901 datasets; umol kg-1 as is) or else mole_concentration_
       of_dissolved_molecular_oxygen_in_sea_water (369; umol L-1 -> umol kg-1
       with the sample's own `density`: x 1000 / rho)
  CHL  mass_concentration_of_chlorophyll_a_in_sea_water (931) or
       mass_concentration_of_chlorophyll_in_sea_water (78) or
       concentration_of_chlorophyll_in_sea_water (208), in mg m-3 = ug L-1
preferring the plain names (dissolved_oxygen, oxygen, chlorophyll_a,
chlorophyll) over instrument-prefixed ones (sci_, m_, ...). A unit outside
the accepted spellings (measured in a sample of 14 datasets: "micrograms
m^-3", and an oxygen variable with no units at all) is NOT converted: the
channel is left NaN for that deployment and the unit is counted
(`units_unrecognized`). The THREDDS file itself carries no units for
temperature or salinity (measured), which is why the units come from ERDDAP.
Missing samples are the declared _FillValue (-9999.9 or NaN, measured) OR
NetCDF's default float fill 9.96921e36 where none is declared (measured in
angus-20230526T0051's temperature): any |x| > 1e30 is missing, family 8's
rule (`samples_default_fill`). THREDDS lists each file's size rounded to
10 kB ("37.24 Mbytes"), so sizes are reported as approximate and Range reads
are never clamped to them.

WHAT A ROW IS.
  time_s    the profile's `time`, int32 seconds since 1982-01-01.
  lat, lon  the profile's `latitude`, `longitude`.
  platform  platform_hash(the DEPLOYMENT id — the dataset id without
            "-delayed"): a deployment is one platform whichever mode was read.
  values    C = 64 = T (degC), S (PSU), O2 (umol/kg), CHL (mg/m3), each at the
            16 pressures of `build_family8_argo.LEVELS` (10, 30, 50, 100, 150,
            200, 300, 400, 500, 700, 900, 1100, 1300, 1500, 1700, 1900 dbar —
            imported, never restated, so a glider level is an Argo level),
            VARIABLE-MAJOR, by
            `build_family8_argo.interp_levels` (linear between two samples no
            further apart than 25 / 50 / 100 dbar, the 10 dbar level also from
            a first sample shallower than 20 dbar). Gliders rarely pass
            1000 dbar, so the deep levels are mostly NaN.
  qc        1 when the file carries QARTOD primary flags for temperature or
            salinity and samples flagged 3 (suspect), 4 (fail) or 9 (missing)
            were removed before interpolation (counted in `qartod_removed`);
            0 when the file has no such flags (not assessed).
When a deployment has both a real-time and a delayed-mode dataset, the
DELAYED one (recovered full resolution, the DAC's definitive version) is read
if THREDDS has its file, else the real-time one (`delayed_used`,
`realtime_used_delayed_missing`). A profile with none of the 64 values is
dropped (`profiles_no_level`); two profiles of a deployment at the same second
keep the later (`profiles_duplicate_time`).

THE PROBE, 2023-06, MEASURED 2026-09-17: 56 deployments overlap June (2,540
datasets listed); 6 read in delayed mode, 47 in real time (5 of those because
the delayed directory has no file yet), 3 with no aggregate at all
(ng644-20230614T0957, ru36-20230611T2254, ru40-20230629T1430), 2 skipped for
pressure in units of 'bar'; 46 in the trajectory = 1 layout and 5 in the
per-dive layout. 371,817,355 bytes of Range reads (the 53 files list
≈ 10.05 GB whole) in 270 s; 13,687 June profiles kept from 51 deployments
(127 without time or position, 220 without any level, 3 repeated times);
QARTOD removed 1,496,365 salinity, 172,083 temperature and 124,200 pressure
samples; qc 1 on 11,013. Oxygen was found in 19 deployments (one with no
units: `units_unrecognized`) and chlorophyll in 24; sp030-20230221T1804's
oxygen is a constant ≈ -3 umol/kg (a dead sensor) — 1,347 level values out of
bounds, NaN and counted. NaN fractions at 10 / 100 / 500 / 900 dbar: T 0.004 /
0.33 / 0.63 / 0.73, S 0.15 / 0.46 / 0.69 / 0.79, O2 0.74 / 0.77 / 0.89 /
0.97, CHL 0.66 / 0.76 / 0.88 / 0.98; 1100 dbar and deeper is almost all NaN
(1900 dbar: 1.0). At ≈ 27 kB of Range reads and ≈ 20 ms per kept profile,
and 155 bytes a stored row, a year like 2023 (≈ 165 k profiles) is ≈ 4.5 GB
of reads, ≈ 1 h and ≈ 26 MB stored.
THE WHOLE ARCHIVE, INDEXED 2026-09-17 (`--stage index --start 2003-01-01`,
2,086 s for ≈ 2,500 THREDDS catalogs): 2,222 deployments, 2,058 with an
aggregate (652 delayed, 1,406 real-time — 98 of them because the delayed
file is missing), 164 with none; the 2,058 files list ≈ 1.09 TB WHOLE (the
obs dimension is padded to each deployment's longest profile). Reading only
the needed variables takes ≈ 13 % of a file's bytes for all of its profiles
(the probe: 371.8 MB for 29 % of the 47,504 profiles in ≈ 10.05 GB of
files), so the whole store is ≈ 140
GB of Range reads for ≈ 5 M profiles (≈ 27 kB each; the note's 5e6) and
≈ 0.8 GB stored (155 B/row; the note's 0.8 GB). Through this sandbox's proxy
that is ≈ 28 h at the probe's 1.4 MB/s (resets on gliders.ioos.us, backoff,
many small requests); THREDDS served a whole file at 90 MB/s, so a hosted
runner should need a few hours.

MEMORY. One deployment at a time per worker, read `CHUNK_PROFILES` profiles at
a time: 2,000 x obs x (4 bytes x up to 11 variables); a delayed-mode obs
dimension of 10,000 samples would be ≈ 880 MB per chunk, so the chunk is
shrunk to keep each read under `CHUNK_BYTES` (64 MB of raw variables).
"""
import csv
import datetime as dt
import io
import os
import re
import sys
import time

import numpy as np

import build_family10_stores as f10b
import build_family8_argo as f8
from family1.adapters import _common as cm
from family1.adapters import _nc3

ERDDAP = "https://gliders.ioos.us/erddap"
ALL_FIELDS = ("datasetID", "institution", "minTime", "maxTime",
              "minLatitude", "maxLatitude", "minLongitude", "maxLongitude",
              "cdm_data_type")
ALL_URL = (f"{ERDDAP}/tabledap/allDatasets.csv?" + "%2C".join(ALL_FIELDS))
INFO_URL = ERDDAP + "/info/{}/index.csv"
THREDDS = "https://gliders.ioos.us/thredds"
CATALOG = THREDDS + "/catalog/deployments/{}catalog.xml"
FILESERVER = THREDDS + "/fileServer/{}"

LEVELS = f8.LEVELS
NLEV = len(LEVELS)
QUANTITIES = (("T", "degC", -3.0, 40.0),
              ("S", "PSU", 0.0, 42.0),
              ("O2", "umol/kg", 0.0, 600.0),
              ("CHL", "mg/m3", -0.5, 100.0))
CHANNELS = tuple((f"{q}_{int(p)}", u, lo, hi) for q, u, lo, hi in QUANTITIES
                 for p in LEVELS)
O2_MASS = "moles_of_oxygen_per_unit_mass_in_sea_water"
O2_VOL = "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water"
CHL_SN = ("mass_concentration_of_chlorophyll_a_in_sea_water",
          "mass_concentration_of_chlorophyll_in_sea_water",
          "concentration_of_chlorophyll_in_sea_water")
UNITS = {
    "pressure": {"dbar", "decibar", "decibars", "db"},
    "temperature": {"celsius", "degree_celsius", "degrees_c", "degc",
                    "deg_c", "degree_c", "degrees_celsius", "°c"},
    "salinity": {"1", "1e-3", "psu", "pss-78", "0.001"},
    "o2_mass": {"umol kg-1", "umol/kg", "µmol/kg", "micromoles/kg",
                "micromol/kg", "umol.kg-1", "µmol kg-1", "micromoles kg-1",
                "umol kg^-1", "micromol kg-1", "micromole kg-1",
                "micromole/kg", "µmol.kg-1"},
    "o2_vol": {"micromoles l-1", "umol l-1", "umol/l", "µmol/l", "micromolar",
               "um", "µm", "mmol m-3", "mmol.m-3", "mmol/m3", "umol.l-1",
               "µmol l-1", "micromoles/l", "micromol l-1", "micromole l-1",
               "micromol/l"},
    "chl": {"ug l-1", "ug/l", "µg/l", "µg l-1", "mg m-3", "mg.m-3", "mg/m3",
            "mg m^-3", "mg/m^3", "microgram/l", "micrograms l-1", "ug.l-1",
            "micrograms/l"},
    "density": {"kg m-3", "kg.m-3", "kg/m3", "kg m^-3"},
}
PREFER = {"o2": ("dissolved_oxygen", "oxygen", "oxygen_concentration"),
          "chl": ("chlorophyll_a", "chlorophyll", "chlor_a")}
RAW_PREFIX = re.compile(r"^(sci|m|c|u|f|x)_|^instrument", re.I)
QARTOD_BAD = (3, 4, 9)
CATREF = re.compile(r'<catalogRef xlink:href="([^"]+)/catalog\.xml"')
DATASET_FILE = re.compile(
    r'<dataset name="([^"]+\.nc3\.nc)"[^>]*urlPath="([^"]+)"[^>]*>\s*'
    r'(?:<dataSize units="([A-Za-z]+)">([0-9.]+)</dataSize>)?', re.S)
SIZE_UNIT = {"bytes": 1, "Kbytes": 1e3, "Mbytes": 1e6, "Gbytes": 1e9}


class FormatError(ValueError):
    """A glider answer that is not the layout this reader was written for."""


def parse_all(text):
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or tuple(rows[0]) != ALL_FIELDS:
        raise FormatError(f"allDatasets header {rows[:1]}")
    out = {}
    for r in rows[2:]:
        d = dict(zip(ALL_FIELDS, r))
        if d["datasetID"] == "allDatasets" or not d["minTime"]:
            continue
        if d["cdm_data_type"] != "TrajectoryProfile":
            continue
        out[d["datasetID"]] = d
    if not out:
        raise FormatError("allDatasets lists ZERO glider datasets — an empty "
                          "listing is a broken listing")
    return out


def _iso_s(s):
    d = dt.datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
    return f10b.seconds_since_epoch(d)


def parse_info(text):
    """ERDDAP info CSV -> {variable: {attr: value}}."""
    out = {}
    for r in csv.reader(io.StringIO(text)):
        if len(r) < 5:
            continue
        if r[0] == "variable":
            out.setdefault(r[1], {})
        elif r[0] == "attribute" and r[1] != "NC_GLOBAL":
            out.setdefault(r[1], {})[r[2]] = r[4]
    return out


def _unit(a):
    return (a.get("units") or "").strip().lower()


def choose(info, header_vars, counts):
    """The variables of one dataset -> {role: (name, factor-kind)}."""
    ch = {}
    bad_units = counts.setdefault("units_unrecognized", {})
    for role in ("pressure", "temperature", "salinity"):
        a = info.get(role)
        if a is None or role not in header_vars:
            return None, f"no `{role}` variable"
        if _unit(a) not in UNITS[role]:
            return None, f"`{role}` units {a.get('units')!r}"
        ch[role] = role

    def pick(sns, prefer):
        cands = [v for v, a in info.items()
                 if a.get("standard_name") in sns and v in header_vars
                 and not v.startswith("qartod")]
        if not cands:
            return None
        rank = {n: i for i, n in enumerate(prefer)}
        cands.sort(key=lambda v: (rank.get(v, len(rank)),
                                  bool(RAW_PREFIX.search(v)),
                                  sns.index(info[v]["standard_name"]), v))
        return cands[0]

    o2 = pick((O2_MASS, O2_VOL), PREFER["o2"])
    if o2:
        sn, u = info[o2]["standard_name"], _unit(info[o2])
        kind = "o2_mass" if sn == O2_MASS else "o2_vol"
        if u in UNITS[kind] and (kind == "o2_mass" or (
                "density" in header_vars and
                _unit(info.get("density", {})) in UNITS["density"])):
            ch["o2"] = o2
            ch["o2_kind"] = kind
        else:
            key = f"O2 {info[o2].get('units')!r}"
            bad_units[key] = bad_units.get(key, 0) + 1
    chl = pick(CHL_SN, PREFER["chl"])
    if chl:
        if _unit(info[chl]) in UNITS["chl"]:
            ch["chl"] = chl
        else:
            key = f"CHL {info[chl].get('units')!r}"
            bad_units[key] = bad_units.get(key, 0) + 1
    return ch, None


def profile_levels(p, vals, good):
    """One profile: pressures p (obs,), a value (obs,) and its good mask ->
    the 16 levels, by the family-8 interpolation."""
    m = good & np.isfinite(p) & np.isfinite(vals)
    if not m.any():
        return None
    ps, vs = f8._sorted_unique(p[m], vals[m])
    return f8.interp_levels(ps, vs)


# ================================================================ adapter ==
class GlidersAdapter(f10b.SourceAdapter):
    store = "gliders"
    title = ("Ocean glider profiles (IOOS Glider DAC): T, S, O2, chlorophyll "
             "at the 16 Roemmich-Gilson pressures, one row per profile")
    family = "1tf"
    distribution = "public"
    licence = {"name": "IOOS Glider DAC: 'This data may be redistributed and "
                       "used without restriction'",
               "redistribution": "yes", "derived_works": "free",
               "attribution": "U.S. IOOS National Glider Data Assembly "
                              "Center, https://gliders.ioos.us; the "
                              "deployment's institution (platforms.json)"}
    time_dtype = "int32"
    platform_meta = True
    credentials = ()
    channels = CHANNELS
    log2_fp = -4.0
    log2_dt = float(np.log2((1.0 / 24.0) / 5.0))
    per_year = False
    first_year = 2003
    qc_policy = (
        "qc 1: the file carries QARTOD primary flags for temperature or "
        "salinity, and samples flagged 3/4/9 were removed before "
        "interpolation (`qartod_removed`); qc 0: no such flags (not "
        "assessed). A level value outside its channel's bounds is NaN and "
        "counted; a profile with no level value is dropped.")
    sources = (FILESERVER.format("deployments/<user>/<id>/<id>.nc3.nc"),
               ALL_URL, INFO_URL.format("<id>"), CATALOG.format(""))
    verified = (
        "2026-09-17 from the sandbox: allDatasets (2,541 TrajectoryProfile "
        "datasets, 868 -delayed); ERDDAP info for 15 datasets; the THREDDS "
        "tree (57 users, 3,064 deployment dirs); angus-20200319T0000.nc3.nc "
        "whole (104,213,412 B, 90 MB/s, Accept-Ranges) read bit-identically "
        "by _nc3 and netCDF4; tabledap throughput 244 kB/s (not used)")
    notes = (
        "IOOS DAC only: the OceanGliders GDAC (Ifremer ERDDAP "
        "OceanGlidersGDACTrajectories) is a sample-level trajectory table "
        "and its file hosts did not answer from the sandbox. Delayed mode "
        "preferred where THREDDS has its file. Levels are family 8's.")
    smoke_window = ("2023-06-01", "2023-06-04")
    smoke_probe_month = "2023-06"
    fetch_month_scope = "month"
    WORKERS = 2
    CHUNK_PROFILES = 2000
    CHUNK_BYTES = 64 << 20

    def __init__(self):
        self._all = None
        self._users = None
        self._files = {}

    # ------------------------------------------------------------ sources --
    def _local(self, ctx, *p):
        return os.path.join(ctx.source_dir, "gliders", *p)

    def _text(self, ctx, url, local, what):
        if ctx.source_dir:
            p = self._local(ctx, *local)
            if not os.path.exists(p):
                return None
            with open(p, "rb") as fh:
                raw = fh.read()
            ctx.count_bytes(len(raw))
            return raw.decode("utf-8", "replace")
        raw, why = cm.get_bytes(url, attempts=max(5, ctx.a.attempts))
        return None if raw is None else raw.decode("utf-8", "replace")

    def datasets(self, ctx):
        if self._all is None:
            t = self._text(ctx, ALL_URL, ("allDatasets.csv",), "the list")
            if t is None:
                sys.exit(f"REFUSING: {ALL_URL} answered 404")
            try:
                self._all = parse_all(t)
            except FormatError as e:
                sys.exit(f"REFUSING: {e}")
        return self._all

    def users(self, ctx):
        """{deployment dir name: [user, ...]} from the THREDDS tree."""
        if self._users is None:
            top = self._text(ctx, CATALOG.format(""),
                             ("thredds", "catalog.xml"), "THREDDS")
            users = CATREF.findall(top or "")
            if not users:
                sys.exit("REFUSING: the THREDDS deployments catalog lists no "
                         "user directory")
            m = {}
            for u in users:
                t = self._text(ctx, CATALOG.format(u + "/"),
                               ("thredds", u, "catalog.xml"), "a user")
                if t is None:
                    sys.exit(f"REFUSING: the THREDDS catalog of {u} answered "
                             f"404")
                for d in CATREF.findall(t):
                    m.setdefault(d, []).append(u)
            self._users = m
        return self._users

    def file_of(self, ctx, dsid):
        """(urlPath, bytes) of a dataset's aggregate, or (None, why)."""
        if dsid not in self._files:
            self._files[dsid] = self._file_of(ctx, dsid)
        return self._files[dsid]

    def _file_of(self, ctx, dsid):
        for u in sorted(self.users(ctx).get(dsid, [])):
            t = self._text(ctx, CATALOG.format(f"{u}/{dsid}/"),
                           ("thredds", u, dsid, "catalog.xml"), "a deployment")
            if not t:
                continue
            for name, path, unit, size in DATASET_FILE.findall(t):
                if name == f"{dsid}.nc3.nc":
                    n = int(float(size) * SIZE_UNIT.get(unit, 1)) \
                        if size else None
                    return path, n
        return None, ("no THREDDS directory" if dsid not in self.users(ctx)
                      else "THREDDS directory without an aggregate file")

    def plan(self, ctx, t_lo, t_hi):
        """The deployments overlapping [t_lo, t_hi] and the dataset to read
        for each; the choice is counted."""
        ds = self.datasets(ctx)
        counts = {"datasets_listed": len(ds)}
        base = {}
        for k, d in ds.items():
            if _iso_s(d["minTime"]) > t_hi or _iso_s(d["maxTime"]) < t_lo:
                continue
            b = k[:-len("-delayed")] if k.endswith("-delayed") else k
            base.setdefault(b, []).append(k)
        counts["deployments_in_window"] = len(base)
        out = []
        for b in sorted(base):
            ks = base[b]
            delayed = [k for k in ks if k.endswith("-delayed")]
            choice = None
            if delayed:
                path, n = self.file_of(ctx, delayed[0])
                if path:
                    choice = (delayed[0], path, n)
                    counts["delayed_used"] = counts.get("delayed_used", 0) + 1
                elif b in ks:
                    counts["realtime_used_delayed_missing"] = counts.get(
                        "realtime_used_delayed_missing", 0) + 1
            if choice is None and b in ks:
                path, n = self.file_of(ctx, b)
                if path:
                    choice = (b, path, n)
                    counts["realtime_used"] = counts.get("realtime_used", 0) \
                        + 1
            if choice is None:
                counts["deployments_without_file"] = counts.get(
                    "deployments_without_file", 0) + 1
                missing = counts.setdefault("deployments_without_file_ids",
                                            [])
                missing.append(b)
                continue
            out.append((b,) + choice)
        return out, counts

    def _reader(self, ctx, path, size):
        if ctx.source_dir:
            return _nc3.file_reader(self._local(ctx, "thredds", *path.split(
                "/")[1:]), counter=ctx.count_bytes)
        return cm.range_reader(FILESERVER.format(path),
                               attempts=max(5, ctx.a.attempts))

    # ---------------------------------------------------------- deployment --
    def _deployment(self, ctx, item, t_lo, t_hi):
        base, dsid, path, size = item
        counts = {"deployments_read": 1}
        try:
            info_t = self._text(ctx, INFO_URL.format(dsid),
                                ("info", f"{dsid}.csv"), "info")
            if info_t is None:
                return base, None, counts, f"{dsid}: ERDDAP info answered 404"
            info = parse_info(info_t)
            nc = _nc3.NC3(self._reader(ctx, path, size))
            ch, why = choose(info, set(nc.vars), counts)
            if ch is None:
                counts["deployments_skipped_variables"] = 1
                sk = counts.setdefault("skipped_why", {})
                sk[why] = sk.get(why, 0) + 1
                return base, {}, counts, None
            for k in ("o2", "chl"):
                if k in ch:
                    key = f"{k}:{ch[k]}"
                    counts.setdefault("variables", {})[key] = 1
            if nc.dimnames("pressure") != ("trajectory", "profile", "obs"):
                raise FormatError(f"pressure dims {nc.dimnames('pressure')}")
            # TWO AGGREGATE LAYOUTS, both measured: (trajectory=1, profile=P,
            # obs) — angus-20230526T0051 — and (trajectory=P, profile=1,
            # obs), one "trajectory" per dive — sg180-20230417T2100, 319 x 1
            # x 2401. The profile axis is whichever leading one is not 1.
            n_tr, n_pr = nc.dim_size("trajectory"), nc.dim_size("profile")
            if n_tr == 1:
                pax = 1
            elif n_pr == 1:
                pax = 0
            else:
                # A THIRD LAYOUT (measured 2026-09-17, run #33:
                # bass-20150827T1909 is 2 x 1053 x obs). Which axis is the
                # profile is not decidable from the dimensions alone, so the
                # deployment is SKIPPED BY NAME and counted, not guessed —
                # and one file no longer ends a lane.
                counts["deployments_skipped_layout"] = 1
                counts.setdefault("skipped_layout_ids", []).append(
                    f"{dsid}: trajectory {n_tr} x profile {n_pr}")
                sk = counts.setdefault("skipped_why", {})
                sk["layout_neither_axis_is_1"] = \
                    sk.get("layout_neither_axis_is_1", 0) + 1
                return base, {}, counts, None
            counts.setdefault("layouts", {})[
                "trajectory_1" if pax == 1 else "profile_1"] = 1
            tm = nc.read("time").reshape(-1).astype(np.float64)
            la = nc.read("latitude").reshape(-1).astype(np.float64)
            lo = nc.read("longitude").reshape(-1).astype(np.float64)
            for nm, arr in (("time", tm), ("latitude", la),
                            ("longitude", lo)):
                f = nc.vars[nm]["attrs"].get("_FillValue")
                if f is not None and np.isfinite(f):
                    arr[arr == f] = np.nan
                arr[np.abs(arr) > 1e30] = np.nan
            units = nc.vars["time"]["attrs"].get("units", "")
            if not units.startswith("seconds since 1970-01-01"):
                raise FormatError(f"time units {units!r}")
            P = tm.size
            counts["profiles_in_file"] = P
            ok = np.isfinite(tm) & np.isfinite(la) & np.isfinite(lo) & \
                (np.abs(la) <= 90) & (np.abs(lo) <= 360)
            counts["profiles_no_time_or_position"] = int((~ok).sum())
            t = np.where(ok, np.rint(tm), 0).astype(np.int64) - \
                cm.EPOCH_S_1970
            sel = ok & (t >= t_lo) & (t <= t_hi)
            counts["profiles_outside_window"] = int((ok & ~sel).sum())
            idx = np.flatnonzero(sel)
            if not idx.size:
                return base, {}, counts, None
            flags = {r: f"qartod_{r}_primary_flag" for r in
                     ("pressure", "temperature", "salinity")}
            flags = {r: v for r, v in flags.items() if v in nc.vars}
            qc_val = int("temperature" in flags or "salinity" in flags)
            names = {"p": "pressure", "T": "temperature", "S": "salinity"}
            if "o2" in ch:
                names["O2"] = ch["o2"]
                if ch["o2_kind"] == "o2_vol":
                    names["rho"] = "density"
            if "chl" in ch:
                names["CHL"] = ch["chl"]
            fills = {k: nc.vars[v]["attrs"].get("_FillValue")
                     for k, v in names.items()}
            obs = nc.dim_size("obs")
            per = obs * 4 * (len(names) + len(flags))
            step = max(1, min(self.CHUNK_PROFILES, self.CHUNK_BYTES // per))
            out_v = np.full((idx.size, 4 * NLEV), np.nan, np.float64)
            removed = counts.setdefault("qartod_removed", {})
            j = 0
            lo_i, hi_i = int(idx[0]), int(idx[-1]) + 1
            for a in range(lo_i, hi_i, step):
                b = min(hi_i, a + step)
                want = idx[(idx >= a) & (idx < b)]
                if not want.size:
                    continue
                cols = {}
                for k, v in names.items():
                    x = nc.read_part(v, pax, a, b).reshape(
                        b - a, obs).astype(np.float64)
                    f = fills[k]
                    if f is not None and np.isfinite(f):
                        x[x == f] = np.nan
                    # NetCDF's DEFAULT float fill (9.96921e36) where no
                    # _FillValue is declared — measured in angus-
                    # 20230526T0051's temperature; family 8's rule
                    big = np.abs(x) > 1e30
                    if big.any():
                        x[big] = np.nan
                        counts["samples_default_fill"] = counts.get(
                            "samples_default_fill", 0) + int(big.sum())
                    cols[k] = x[want - a]
                fl = {r: nc.read_part(v, pax, a, b).reshape(b - a, obs)[
                    want - a].astype(np.int64) for r, v in flags.items()}
                bad_p = np.isin(fl["pressure"], QARTOD_BAD) \
                    if "pressure" in fl else np.zeros(cols["p"].shape, bool)
                for r, fv in fl.items():
                    n = int((np.isin(fv, QARTOD_BAD)
                             & np.isfinite(cols[{"pressure": "p",
                                                 "temperature": "T",
                                                 "salinity": "S"}[r]])).sum())
                    if n:
                        removed[r] = removed.get(r, 0) + n
                p = np.where(bad_p, np.nan, cols["p"])
                good = {"T": ~np.isin(fl["temperature"], QARTOD_BAD)
                        if "temperature" in fl else True,
                        "S": ~np.isin(fl["salinity"], QARTOD_BAD)
                        if "salinity" in fl else True}
                if "O2" in cols and "rho" in cols:
                    cols["O2"] = cols["O2"] * 1000.0 / cols["rho"]
                for i in range(want.size):
                    for q, key in enumerate(("T", "S", "O2", "CHL")):
                        if key not in cols:
                            continue
                        g = good.get(key, True)
                        g = g[i] if isinstance(g, np.ndarray) else \
                            np.ones(obs, bool)
                        lv = profile_levels(p[i], cols[key][i], g)
                        if lv is not None:
                            out_v[j + i, q * NLEV:(q + 1) * NLEV] = lv
                j += want.size
            tt, la_, lo_ = t[idx], la[idx], lo[idx]
            oob = self.mask_bounds(out_v)
            if oob:
                counts["out_of_bounds"] = oob
            alive = np.isfinite(out_v).any(axis=1)
            counts["profiles_no_level"] = int((~alive).sum())
            tt, la_, lo_, out_v = tt[alive], la_[alive], lo_[alive], \
                out_v[alive]
            order = np.argsort(tt, kind="stable")
            tt, la_, lo_, out_v = tt[order], la_[order], lo_[order], \
                out_v[order]
            last = np.ones(tt.size, bool)
            if tt.size:
                last[:-1] = tt[1:] != tt[:-1]
            counts["profiles_duplicate_time"] = int((~last).sum())
            tt, la_, lo_, out_v = tt[last], la_[last], lo_[last], out_v[last]
        except (FormatError, _nc3.NC3Error) as e:
            raise FormatError(f"{dsid}: {e}") from None
        except IOError as e:
            return base, None, counts, f"{dsid}: {e}"
        n = int(tt.size)
        counts["profiles_kept"] = n
        counts["qc"] = {str(qc_val): n} if n else {}
        if not n:
            return base, {}, counts, None
        ty = (np.datetime64("1982-01-01", "s")
              + tt.astype("timedelta64[s]")).astype("datetime64[Y]") \
            .astype(np.int64) + 1970
        plat = f10b.platform_hash(base)
        by_year = {}
        for y in np.unique(ty).tolist():
            m = ty == y
            k = int(m.sum())
            by_year[int(y)] = self.pack(
                tt[m], la_[m], lo_[m], out_v[m], np.full(k, plat, np.int64),
                np.full(k, qc_val, np.uint8))
        return base, by_year, counts, None

    def _stream(self, ctx, t_lo, t_hi, unit):
        items, counts = self.plan(ctx, t_lo, t_hi)
        counts = dict(counts)
        workers = 1 if ctx.source_dir else self.WORKERS
        t0 = time.time()
        fn = (lambda it: self._deployment(ctx, it, t_lo, t_hi))
        for i, (base, by_year, c, err) in enumerate(
                cm.ordered_map(fn, items, workers), 1):
            f10b._merge_counts(counts, c)
            if err:
                ctx.note_absent(unit, err)
                counts["deployments_failed"] = counts.get(
                    "deployments_failed", 0) + 1
                counts.setdefault("deployments_failed_ids", []).append(base)
                continue
            for y in sorted(by_year):
                yield y, by_year[y], None
            if i % 50 == 0:
                ctx.prog.item(f"gliders deployments {i}/{len(items)}", None,
                              {"profiles_kept":
                               counts.get("profiles_kept", 0),
                               "elapsed_s": round(time.time() - t0, 1)})
        counts["bytes_listed_approx"] = int(sum(it[3] or 0 for it in items))
        counts["stream_seconds"] = round(time.time() - t0, 1)
        yield None, None, counts

    # --------------------------------------------------------- the contract --
    def index(self, ctx):
        ds = self.datasets(ctx)
        users = self.users(ctx)
        items, counts = self.plan(ctx, ctx.t_lo, ctx.t_hi)
        first = None
        if items:
            base, dsid, path, size = items[0]
            nc = _nc3.NC3(self._reader(ctx, path, size))
            first = {"dataset": dsid, "file": path, "bytes": size,
                     "dims": dict(nc.dims),
                     "header_bytes": nc.header_bytes}
        return {
            "dataset": f"{ERDDAP} allDatasets + {THREDDS} deployments",
            "datasets": len(ds),
            "delayed_datasets": sum(k.endswith("-delayed") for k in ds),
            "thredds_deployment_dirs": len(users),
            "thredds_dirs_not_in_erddap": len(set(users) - set(ds)),
            "window_plan": counts,
            "window_deployments": len(items),
            "window_bytes_listed_approx": int(sum(it[3] or 0 for it in items)),
            "levels_dbar": list(LEVELS),
            "first_record": first,
        }

    def fetch_stream(self, ctx):
        yield from self._stream(ctx, ctx.t_lo, ctx.t_hi, str(ctx.d_lo.year))

    def fetch_month(self, ctx, year, month):
        lo, hi = f10b.month_bounds_s(year, month)
        lo, hi = max(lo, ctx.t_lo), min(hi, ctx.t_hi)
        parts, counts = [], {}
        for y, rows, c in self._stream(ctx, lo, hi, str(year)):
            if y is None:
                counts = c
            else:
                parts.append(rows)
        rows = cm.concat_rows(parts, self.C, self.time_dtype)
        yield from cm.batch_rows(self, rows, counts, f"{year}-{month:02d}")

    def platforms(self, ctx):
        ds = self.datasets(ctx)
        out = {}
        for k in sorted(ds):
            d = ds[k]
            b = k[:-len("-delayed")] if k.endswith("-delayed") else k
            h = f10b.platform_hash(b)
            try:
                la = (float(d["minLatitude"]) + float(d["maxLatitude"])) / 2
                lo = (float(d["minLongitude"]) + float(d["maxLongitude"])) / 2
            except ValueError:
                la = lo = None
            e = out.setdefault(h, {"id": b, "lat": la, "lon": lo,
                                   "institution": d["institution"],
                                   "glider": b.rsplit("-", 1)[0],
                                   "datasets": [], "min_time": d["minTime"],
                                   "max_time": d["maxTime"],
                                   "bbox": [d["minLatitude"],
                                            d["maxLatitude"],
                                            d["minLongitude"],
                                            d["maxLongitude"]]})
            e["datasets"].append(k)
        return out

    def smoke_sources(self, root, d_lo, d_hi, seed=20260917):
        return make_smoke_sources(root, d_lo, d_hi, seed)


# ================================================================== smoke ==
def _catalog(entries=(), files=()):
    refs = "".join(f'<catalogRef xlink:href="{e}/catalog.xml" '
                   f'xlink:title="{e}" ID="x/{e}" name="{e}"/>\n'
                   for e in entries)
    ds = "".join(f'<dataset name="{n}" ID="deployments/{p}" urlPath="{p}">\n'
                 f'  <dataSize units="Mbytes">{sz:.4f}</dataSize>\n'
                 f'</dataset>\n' for n, p, sz in files)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<catalog xmlns="http://'
            'www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0" xmlns:'
            'xlink="http://www.w3.org/1999/xlink" version="1.2">\n<dataset '
            'name="x">\n' + refs + ds + '</dataset>\n</catalog>\n')


def _info(variables):
    """variables = [(name, {attr: value})] -> an ERDDAP info CSV."""
    out = ["Row Type,Variable Name,Attribute Name,Data Type,Value",
           "attribute,NC_GLOBAL,license,String,This data may be "
           "redistributed and used without restriction."]
    for v, at in variables:
        out.append(f"variable,{v},,float,")
        for k, x in at.items():
            out.append(f'attribute,{v},{k},String,"{x}"')
    return "\n".join(out) + "\n"


SMOKE_DEPLOYMENTS = (
    # dataset id, user, kind
    ("alpha-20230530T0000", "rutgers", "realtime_twin"),     # delayed wins
    ("alpha-20230530T0000-delayed", "rutgers", "delayed"),
    ("bravo-20230601T0000", "secoora", "o2_volume"),         # delayed absent
    ("bravo-20230601T0000-delayed", "secoora", "no_file"),
    ("charlie-20230515T0000", "usf", "bad_units"),           # CHL unit skipped
    ("delta-20220101T0000", "usf", "old"),                   # outside window
)


def make_smoke_sources(root, d_lo, d_hi, seed=20260917):
    """An IOOS DAC in miniature: allDatasets.csv, ERDDAP info per dataset,
    the THREDDS catalog tree and `<id>.nc3.nc` aggregates written by
    `_nc3.write` in the DAC's (trajectory=1, profile, obs) layout.

    Hostile: a delayed twin that must win over its real-time dataset, a
    delayed directory without a file (the real-time one is read), oxygen in
    umol L-1 converted with the sample's density, a chlorophyll unit outside
    the accepted spellings (NaN, counted), QARTOD flags removing samples, a
    fill value, a profile with no position, a profile with no usable sample,
    a duplicated profile time, profiles outside the window, a test deployment
    only in THREDDS, a NetCDF default fill (9.96921e36) with no _FillValue
    declared, a deployment in the one-trajectory-per-dive layout, and one
    deployment outside the window altogether.
    Returns the truth rows.
    """
    rng = np.random.default_rng(seed)
    base = os.path.join(root, "gliders")
    os.makedirs(os.path.join(base, "info"), exist_ok=True)
    t_lo = f10b.seconds_since_epoch(d_lo)
    t_hi = f10b.seconds_since_epoch(d_hi) + 86399
    rows = [",".join(ALL_FIELDS), ",,UTC,UTC,degrees_north,degrees_north,"
            "degrees_east,degrees_east,",
            "allDatasets,Set-me Institution,,,NaN,NaN,NaN,NaN,Other"]
    users = {}
    truth = []
    obs = 40
    for dsid, user, kind in SMOKE_DEPLOYMENTS:
        users.setdefault(user, []).append(dsid)
        start = t_lo - 86400 if kind != "old" else t_lo - 400 * 86400
        stop = t_hi + 1 if kind != "old" else t_lo - 390 * 86400
        pt = np.arange(start, stop, 3 * 3600, dtype=np.int64)
        P = pt.size
        la = np.round(38.0 + rng.uniform(-1, 1, P), 4)
        lo = np.round(-74.0 + rng.uniform(-1, 1, P), 4)
        rows.append(f"{dsid},SMOKE {user.upper()},"
                    f"{dt.datetime(1982, 1, 1) + dt.timedelta(seconds=int(pt[0])):%Y-%m-%dT%H:%M:%SZ},"
                    f"{dt.datetime(1982, 1, 1) + dt.timedelta(seconds=int(pt[-1])):%Y-%m-%dT%H:%M:%SZ},"
                    f"{la.min()},{la.max()},{lo.min()},{lo.max()},"
                    f"TrajectoryProfile")
        variables = [("time", {"units": "seconds since 1970-01-01T00:00:00Z",
                               "standard_name": "time"}),
                     ("latitude", {"units": "degrees_north"}),
                     ("longitude", {"units": "degrees_east"}),
                     ("pressure", {"units": "dbar",
                                   "standard_name": "sea_water_pressure"}),
                     ("temperature", {"units": "Celsius",
                                      "standard_name":
                                      "sea_water_temperature"}),
                     ("salinity", {"units": "1", "standard_name":
                                   "sea_water_practical_salinity"}),
                     ("density", {"units": "kg m-3",
                                  "standard_name": "sea_water_density"})]
        if kind == "o2_volume":
            variables.append(("oxygen_concentration",
                              {"units": "micromoles L-1",
                               "standard_name": O2_VOL}))
        else:
            variables.append(("dissolved_oxygen", {"units": "umol kg-1",
                                                   "standard_name": O2_MASS}))
            variables.append(("sci_oxy4_oxygen", {"units": "umol kg-1",
                                                  "standard_name": O2_MASS}))
        variables.append(("chlorophyll_a", {
            "units": "micrograms m^-3" if kind == "bad_units" else "ug l-1",
            "standard_name": CHL_SN[0]}))
        with open(os.path.join(base, "info", f"{dsid}.csv"), "w") as fh:
            fh.write(_info(variables))
        if kind == "no_file":
            continue
        # the samples: a dive to 0..obs*25 dbar, profile by profile
        pres = np.full((P, obs), -9999.9, np.float32)
        tem = np.full((P, obs), np.nan, np.float32)
        sal = np.full((P, obs), np.nan, np.float32)
        rho = np.full((P, obs), -9999.9, np.float32)
        oxy = np.full((P, obs), -9999.9, np.float32)
        oxr = np.full((P, obs), -9999.9, np.float32)
        chl = np.full((P, obs), -9999.9, np.float32)
        qt = np.full((P, obs), 1, np.int8)
        for i in range(P):
            n = int(rng.integers(20, obs + 1))
            p = np.sort(rng.uniform(1, 25 * n, n)).astype(np.float32)
            pres[i, :n] = p
            tem[i, :n] = (25 - p / 50).astype(np.float32)
            sal[i, :n] = (35 + p / 1000).astype(np.float32)
            rho[i, :n] = (1025 + p / 200).astype(np.float32)
            oxy[i, :n] = (250 - p / 10).astype(np.float32)
            oxr[i, :n] = (230 - p / 10).astype(np.float32)
            chl[i, :n] = np.maximum(0, 1 - p / 200).astype(np.float32)
        tm = (pt + cm.EPOCH_S_1970).astype(np.float64)
        la_f, lo_f = la.astype(np.float64), lo.astype(np.float64)
        ins = np.flatnonzero((pt >= t_lo) & (pt <= t_hi))
        specials = {}
        if kind == "delayed":
            i0 = int(ins[2])
            la_f[i0] = np.nan                      # no position: dropped
            specials[i0] = "drop"
            i1 = int(ins[3])
            qt[i1, :] = 4                          # every T sample fails
            i2 = int(ins[4])
            tm[i2 + 1] = tm[i2]                    # duplicated time
            pt[i2 + 1] = pt[i2]
            specials[i2] = "dup_first"
            i3 = int(ins[7])
            pres[i3, :] = -9999.9                  # no usable sample
            specials[i3] = "drop"
        if kind == "bad_units":
            tem[int(ins[0]), 0] = np.float32(9.96921e36)   # default fill
        per_traj = kind == "bad_units"          # the Seaglider layout
        TD, PD = (("trajectory", P), ("profile", 1)) if per_traj else \
            (("trajectory", 1), ("profile", P))
        tid = np.frombuffer(b"".join(
            dsid.encode()[:32].ljust(32, b"\0")
            for _ in range(TD[1])), "S1").reshape(TD[1], 32)
        vars_ = [("trajectory", ("trajectory", "traj_strlen"), tid, {}),
                 ("time", ("trajectory", "profile"), tm.reshape(1, P),
                  {"units": "seconds since 1970-01-01T00:00:00Z"}),
                 ("latitude", ("trajectory", "profile"), la_f.reshape(1, P),
                  {"units": "degrees_north"}),
                 ("longitude", ("trajectory", "profile"), lo_f.reshape(1, P),
                  {"units": "degrees_east"}),
                 ("profile_id", ("trajectory", "profile"),
                  np.arange(P, dtype=np.int32).reshape(1, P), {}),
                 ("pressure", ("trajectory", "profile", "obs"),
                  pres.reshape(1, P, obs),
                  {"_FillValue": np.float32(-9999.9), "units": "dbar"}),
                 ("temperature", ("trajectory", "profile", "obs"),
                  tem.reshape(1, P, obs), {"_FillValue": np.float32(np.nan)}),
                 ("salinity", ("trajectory", "profile", "obs"),
                  sal.reshape(1, P, obs), {"_FillValue": np.float32(np.nan)}),
                 ("density", ("trajectory", "profile", "obs"),
                  rho.reshape(1, P, obs),
                  {"_FillValue": np.float32(-9999.9)}),
                 ("chlorophyll_a", ("trajectory", "profile", "obs"),
                  chl.reshape(1, P, obs),
                  {"_FillValue": np.float32(-9999.9)})]
        if kind == "o2_volume":
            vars_.append(("oxygen_concentration",
                          ("trajectory", "profile", "obs"),
                          oxr.reshape(1, P, obs),
                          {"_FillValue": np.float32(-9999.9)}))
        else:
            vars_ += [("dissolved_oxygen", ("trajectory", "profile", "obs"),
                       oxy.reshape(1, P, obs),
                       {"_FillValue": np.float32(-9999.9)}),
                      ("sci_oxy4_oxygen", ("trajectory", "profile", "obs"),
                       (oxy * 0 + 1).reshape(1, P, obs),
                       {"_FillValue": np.float32(-9999.9)})]
        if kind == "delayed":
            vars_.append(("qartod_temperature_primary_flag",
                          ("trajectory", "profile", "obs"),
                          qt.reshape(1, P, obs), {}))
        d = os.path.join(base, "thredds", user, dsid)
        os.makedirs(d, exist_ok=True)
        f = os.path.join(d, f"{dsid}.nc3.nc")
        shp2 = (TD[1], PD[1])
        vars_ = [(n, d, (a.reshape(shp2 + a.shape[2:])
                         if d[:2] == ("trajectory", "profile") else a), at)
                 for n, d, a, at in vars_]
        _nc3.write(f, [TD, PD, ("obs", obs), ("traj_strlen", 32)], vars_,
                   {"license": "This data may be redistributed and used "
                               "without restriction."})
        with open(os.path.join(d, "catalog.xml"), "w") as fh:
            fh.write(_catalog(files=[(f"{dsid}.nc3.nc",
                                      f"deployments/{user}/{dsid}/"
                                      f"{dsid}.nc3.nc",
                                      os.path.getsize(f) / 1e6)]))
        # ---- the truth, for the dataset the adapter will read
        if kind in ("realtime_twin", "old"):
            continue
        bid = dsid[:-len("-delayed")] if dsid.endswith("-delayed") else dsid
        rows_t = []
        for i in ins.tolist():
            if specials.get(i) == "drop":
                continue
            if specials.get(i) == "dup_first":
                continue
            v = [np.nan] * (4 * NLEV)
            n = int((pres[i] > -9000).sum())
            p = pres[i, :n].astype(np.float64)
            good_t = qt[i, :n] == 1 if kind == "delayed" else np.ones(n, bool)
            t_i = np.where(np.abs(tem[i, :n]) > 1e30, np.nan, tem[i, :n])
            series = {
                0: (t_i, good_t),
                1: (sal[i, :n], np.ones(n, bool)),
                2: ((oxr[i, :n].astype(np.float64) * 1000.0
                     / rho[i, :n].astype(np.float64))
                    if kind == "o2_volume" else oxy[i, :n],
                    np.ones(n, bool)),
                3: (chl[i, :n], np.ones(n, bool)),
            }
            for q, (vals, g) in series.items():
                if q == 3 and kind == "bad_units":
                    continue
                lv = profile_levels(p, np.asarray(vals, np.float64), g)
                if lv is not None:
                    v[q * NLEV:(q + 1) * NLEV] = [
                        np.nan if not np.isfinite(x) else float(x)
                        for x in lv]
            if not np.isfinite(v).any():
                continue
            rows_t.append({"t": int(pt[i]), "lat": float(la[i]),
                           "lon": float(lo[i]),
                           "platform": f10b.platform_hash(bid), "v": v,
                           "qc": 1 if kind == "delayed" else 0})
        truth.extend(rows_t)
    users["noaa-test"] = ["zulu-20230601T0000"]              # THREDDS only
    os.makedirs(os.path.join(base, "thredds", "noaa-test"), exist_ok=True)
    with open(os.path.join(base, "allDatasets.csv"), "w") as fh:
        fh.write("\n".join(rows) + "\n")
    with open(os.path.join(base, "thredds", "catalog.xml"), "w") as fh:
        fh.write(_catalog(entries=sorted(users)))
    for u, ds in users.items():
        os.makedirs(os.path.join(base, "thredds", u), exist_ok=True)
        with open(os.path.join(base, "thredds", u, "catalog.xml"), "w") as fh:
            fh.write(_catalog(entries=ds))
        for dsid in ds:
            os.makedirs(os.path.join(base, "thredds", u, dsid), exist_ok=True)
            p = os.path.join(base, "thredds", u, dsid, "catalog.xml")
            if not os.path.exists(p):
                with open(p, "w") as fh:
                    fh.write(_catalog())
    return truth


ADAPTER = GlidersAdapter
