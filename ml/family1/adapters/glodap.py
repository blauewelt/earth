"""GLODAPv3 bottle data — one row per bottle sample (family 1.gf).

PLAIN ENGLISH. GLODAP (the Global Ocean Data Analysis Project) is the
cross-calibrated collection of research-cruise bottle samples: dissolved
inorganic carbon, alkalinity, pH, oxygen, nutrients and the CFC/SF6 tracers
that date water masses, from surface to seafloor, 1972 onward. This adapter
turns its merged master file into a tier-P store, one row per bottle, so a
model can read the ocean's carbon and ventilation state where ships sampled
it. It is a BOTTLE store, not a profile store: each row is one sample at its
own depth, and the depth is channel 0.

SOURCE, VERIFIED 2026-09-17 FROM THIS SANDBOX (NOAA NCEI OCADS, no account):
  https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0315582/
      GLODAPv3_Merged_Master_File.csv        (1,029,335,273 bytes, 2026-07-06)
  * glodap.info announces GLODAPv3 (1972-2023, all of GLODAPv2.2023 plus 57
    new cruises) as the newest release; the NCEI accession 0315582 is where
    it is served. The directory (Apache listing, human-readable sizes) also
    holds per-basin CSVs, a .mat (130 M, needs HDF5 tooling) and a .nc
    (902 M, HDF5); the merged CSV is the one streamable file with everything.
    The file name is taken from the listing (`GLODAPv<ver>_Merged_Master_
    File.csv`, highest version wins), never assumed.
  * One header line of 119 names, then 1,498,120 data lines, every one with
    119 fields (measured over the whole file). 1,181 distinct expocodes.
    Missing values are -9999. Each measured variable X has `Xf` (WOCE flag:
    2 acceptable, 0 interpolated or calculated, 9 no data — measured
    distributions: only 2, 0 and 9 occur) and, for most, `Xqc` (GLODAP's
    secondary QC: 1 adjusted/passed, 0 not). `phtsinsitutp` (pH, total
    scale, in-situ T and p) has a flag and NO qc column in v3. `hour` and
    `minute` are -9999 on 177,630 lines; `depth` (and `pressure`) on 29,872.
  * Units, read from the .nc twin's attributes (a 2 MB Range read of its
    HDF5 header): temperature degree_C, salinity (PSS-78), oxygen, nitrate,
    phosphate, silicate, TCO2 and TAlk micromol/kg, CFC-11/12 pmol/kg; the
    SF6 CONCENTRATION's unit was not in the header block read (only its
    partial pressure's, ppt) and is taken as fmol/kg from the GLODAP product
    documentation — the measured values (median 0.50, max 6.39) agree.

WHAT A ROW IS.
  time_s    year month day hour minute, UTC, int32 seconds since 1982-01-01.
            A line without hour/minute (12 % of the file) is placed at 12:00
            UTC and flagged (qc bit 2, `time_of_day_missing`) — a sub-day
            uncertainty inside a five-day bin — rather than dropped.
  lat, lon  latitude, longitude.
  platform  platform_hash(expocode) — one cruise.
  values    C = 13: depth (m) FIRST, then temperature, salinity, oxygen,
            nitrate, phosphate, silicate, DIC (tco2), TA (talk), pH
            (phtsinsitutp), CFC-11, CFC-12, SF6 at the sample's own depth. A
            value is kept only when its WOCE flag is 2 (measured and
            acceptable): flag 0 values are interpolated or calculated and are
            dropped and counted (`value_flag0`), -9999 is NaN.
  qc        bit 1: at least one kept value has GLODAP secondary QC 0 (not
            adjusted/assessed); bit 2: the time of day was missing.
            qc 0 = every kept value passed secondary QC and the time is known.
A line with no depth is dropped (`bottles_no_depth`); a line with no kept
value is dropped (`bottles_no_value`). Bounds allow the small negative
readings of oxygen, nutrients and tracers at detection limit and refuse the
rest. Measured over the whole file with flag 2: 912 oxygen values in
[-1.54, 0) (anoxic zones) and one of 712.9; 176 phosphate values near 2,300
(a unit error in a few cruises) and two of -0.22; silicate down to -0.81 and
one -2.04; nitrate -0.9, CFC-11 -0.04, SF6 -0.11.

THE PROBE, 2010-07, MEASURED 2026-09-17 (ml/family1/probes/
glodap_2010-07.json): the whole file streamed, 1,029,335,273 bytes in 61 s
(peak RSS 66 MB); 2,999 July-2010 bottles from 6 cruises; NaN fractions
depth/temperature/salinity 0, nitrate/phosphate 0.086, silicate 0.092,
oxygen 0.41, DIC 0.53, TA 0.56, pH 0.67, CFC-11 0.84, CFC-12 0.83, SF6 1.0;
qc 0 2,874, 1 104, 3 21; 0 out of bounds.
WHOLE ARCHIVE, MEASURED (one full pass over the same file, 63 s, window
1972-2026): 1,460,215 rows (the note's 1.50 M bottles less 29,872 without
depth and 8,033 with no flag-2 value), years 1972-2023, 1,181 cruises; qc 0
1,008,326, 1 276,603, 2 103,180, 3 72,106; flag-0 values dropped: nitrate
54,591, silicate 52,408, phosphate 51,969, oxygen 32,092, salinity 3,721;
at 53 B a stored row ≈ 77 MB (the note's 80 MB).

MEMORY. The CSV streams (never written to disk) in 4 MB reads; kept rows are
buffered per year as python lists (≈ 13 floats + 5 scalars a row) and packed
per year at the end of the pass or every FLUSH_ROWS; the whole store is 1.5 M
rows, so the peak is ≈ 0.5 GB at most.
"""
import os
import re
import sys
import time

import numpy as np

import build_family10_stores as f10b
from family1.adapters import _common as cm

ACCESSION = "https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0315582/"
MASTER = re.compile(r'<a href="(GLODAPv([0-9][0-9.]*)_Merged_Master_File'
                    r'\.csv)">')
CHUNK = 1 << 22
FLUSH_ROWS = 1_000_000
MISSING = -9999.0

# (channel, unit, lo, hi, value column, flag column, qc column)
SPEC = (
    ("depth", "m", 0.0, 11000.0, "depth", None, None),
    ("temperature", "degC", -2.5, 40.0, "temperature", "temperaturef",
     "temperatureqc"),
    ("salinity", "PSS-78", 0.0, 42.0, "salinity", "salinityf",
     "salinityqc"),
    ("oxygen", "umol/kg", -2.0, 600.0, "oxygen", "oxygenf", "oxygenqc"),
    ("nitrate", "umol/kg", -1.0, 60.0, "nitrate", "nitratef", "nitrateqc"),
    ("phosphate", "umol/kg", -0.25, 5.0, "phosphate", "phosphatef",
     "phosphateqc"),
    ("silicate", "umol/kg", -1.0, 300.0, "silicate", "silicatef",
     "silicateqc"),
    ("dic", "umol/kg", 500.0, 3000.0, "tco2", "tco2f", "tco2qc"),
    ("ta", "umol/kg", 500.0, 3000.0, "talk", "talkf", "talkqc"),
    ("ph", "1", 6.5, 9.5, "phtsinsitutp", "phtsinsitutpf", None),
    ("cfc11", "pmol/kg", -0.1, 15.0, "cfc11", "cfc11f", "cfc11qc"),
    ("cfc12", "pmol/kg", -0.1, 10.0, "cfc12", "cfc12f", "cfc12qc"),
    ("sf6", "fmol/kg", -0.2, 20.0, "sf6", "sf6f", "sf6qc"),
)
CHANNELS = tuple((c, u, lo, hi) for c, u, lo, hi, *_ in SPEC)
TIME_COLS = ("year", "month", "day", "hour", "minute")
POS_COLS = ("latitude", "longitude")


class FormatError(ValueError):
    """A listing or a header that is not the layout verified above."""


def parse_listing(html):
    """(name, version) of the newest merged master CSV."""
    found = MASTER.findall(html)
    if not found:
        raise FormatError("the accession lists no GLODAPv*_Merged_Master_"
                          "File.csv — an empty listing is a broken listing")

    def key(v):
        return tuple(int(x) for x in v.split(".") if x.isdigit())
    name, ver = max(found, key=lambda x: key(x[1]))
    return name, ver


def text_lines(chunks):
    tail = b""
    for c in chunks:
        tail += c
        parts = tail.split(b"\n")
        tail = parts.pop()
        for p in parts:
            yield p.decode("utf-8", "replace")
    if tail:
        yield tail.decode("utf-8", "replace")


def _f(s):
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return np.nan if v == MISSING else v


class Parser:
    """Header -> column indices; line -> (t, lat, lon, values, expo, qc)."""

    def __init__(self, header):
        names = header.rstrip("\r\n").split(",")
        self.ncol = len(names)
        col = {n: i for i, n in enumerate(names)}
        want = set(TIME_COLS) | set(POS_COLS) | {"expocode"}
        for _c, _u, _lo, _hi, v, f, q in SPEC:
            want |= {x for x in (v, f, q) if x}
        miss = sorted(want - set(col))
        if miss:
            raise FormatError(f"the GLODAP header lacks {miss}")
        self.col = col
        self.names = names

    def parse(self, line, counts):
        p = line.rstrip("\r").split(",")
        if len(p) != self.ncol:
            counts["lines_bad_length"] = counts.get("lines_bad_length", 0) + 1
            return None
        c = self.col
        try:
            y = int(float(p[c["year"]]))
            mo = int(float(p[c["month"]]))
            d = int(float(p[c["day"]]))
        except ValueError:
            counts["bottles_bad_date"] = counts.get("bottles_bad_date", 0) + 1
            return None
        if not bool(cm.valid_date(y, mo, d)):
            counts["bottles_bad_date"] = counts.get("bottles_bad_date", 0) + 1
            return None
        hh, mi = _f(p[c["hour"]]), _f(p[c["minute"]])
        qc = 0
        if not (np.isfinite(hh) and np.isfinite(mi)) or not (
                0 <= hh <= 23 and 0 <= mi <= 59):
            counts["time_of_day_missing"] = \
                counts.get("time_of_day_missing", 0) + 1
            hh, mi = 12, 0
            qc |= 2
        t = int(cm.days_from_civil(y, mo, d)) * 86400 + int(hh) * 3600 + \
            int(mi) * 60
        la, lo = _f(p[c["latitude"]]), _f(p[c["longitude"]])
        if not (np.isfinite(la) and np.isfinite(lo) and abs(la) <= 90
                and abs(lo) <= 360):
            counts["bottles_no_position"] = \
                counts.get("bottles_no_position", 0) + 1
            return None
        v = np.full(len(SPEC), np.nan)
        dep = _f(p[c["depth"]])
        if not np.isfinite(dep):
            counts["bottles_no_depth"] = counts.get("bottles_no_depth", 0) + 1
            return None
        v[0] = dep
        f0 = counts.setdefault("value_flag0", {})
        fo = counts.setdefault("value_flag_other", {})
        unqc = False
        for j in range(1, len(SPEC)):
            name, _u, _lo, _hi, vc, fc, qcol = SPEC[j]
            x = _f(p[c[vc]])
            if not np.isfinite(x):
                continue
            fl = _f(p[c[fc]])
            if fl != 2:
                d_ = f0 if fl == 0 else fo
                d_[name] = d_.get(name, 0) + 1
                continue
            v[j] = x
            if qcol is not None and _f(p[c[qcol]]) != 1:
                unqc = True
        if unqc:
            qc |= 1
        return t, y, la, lo, v, p[c["expocode"]].strip(), qc


# ================================================================ adapter ==
class GLODAPAdapter(f10b.SourceAdapter):
    store = "glodap"
    title = ("GLODAPv3 merged bottle data (NOAA NCEI OCADS 0315582), one row "
             "per bottle sample, depth as channel 0")
    family = "1gf"
    distribution = "public"
    licence = {"name": "CC BY 4.0 (GLODAP)",
               "redistribution": "attribution", "derived_works": "free",
               "attribution": "GLODAPv3 (Lauvset, Key, Olsen et al.), NOAA "
                              "NCEI OCADS accession 0315582, glodap.info"}
    time_dtype = "int32"
    platform_meta = False
    credentials = ()
    channels = CHANNELS
    log2_fp = -4.0
    log2_dt = float(np.log2((1.0 / 24.0) / 5.0))
    per_year = False
    first_year = 1972
    qc_policy = (
        "A value is kept only with WOCE flag 2 (flag 0 = interpolated or "
        "calculated, dropped and counted). qc bit 1: a kept value has "
        "GLODAP secondary QC 0; bit 2: time of day missing (placed at 12:00 "
        "UTC). Lines without depth or without any kept value are dropped. "
        "Out-of-bounds values -> NaN, counted.")
    sources = (ACCESSION + "GLODAPv<ver>_Merged_Master_File.csv",
               ACCESSION, "https://glodap.info")
    verified = (
        "2026-09-17 from the sandbox: the accession listing; the whole "
        "GLODAPv3 merged CSV (1,029,335,273 B, 1,498,120 lines x 119 "
        "fields, 1,181 expocodes, flag/qc distributions, missing hour 177,630, "
        "missing depth 29,872); units from the .nc twin's HDF5 header")
    notes = (
        "A bottle store: depth is channel 0, fp stays the log2 footprint. "
        "One streamed pass (per_year = False); the probe reads the whole "
        "file and keeps its month.")
    smoke_window = ("1999-12-30", "2000-01-02")
    smoke_probe_month = "2000-01"
    fetch_month_scope = "archive"

    def __init__(self):
        self._name = None

    def _local(self, ctx, *p):
        return os.path.join(ctx.source_dir, "glodap", *p)

    def master(self, ctx):
        if self._name is None:
            if ctx.source_dir:
                raw = open(self._local(ctx, "index.html"), "rb").read()
                ctx.count_bytes(len(raw))
            else:
                raw, why = cm.get_bytes(ACCESSION, attempts=ctx.a.attempts)
                if raw is None:
                    sys.exit(f"{ACCESSION} answered 404 ({why})")
            try:
                self._name = parse_listing(raw.decode("latin-1"))
            except FormatError as e:
                sys.exit(f"REFUSING glodap: {e}")
        return self._name

    def _chunks(self, ctx, limit=None):
        name, _ver = self.master(ctx)
        if ctx.source_dir:
            p = self._local(ctx, name)
            with open(p, "rb") as fh:
                while True:
                    b = fh.read(CHUNK if limit is None else min(CHUNK, limit))
                    if not b:
                        return
                    ctx.count_bytes(len(b))
                    yield b
                    if limit is not None:
                        return
        else:
            s = f10b.CountingStream(ACCESSION + name)
            try:
                if limit is not None:
                    yield s.read(limit)
                    return
                while True:
                    b = s.read(CHUNK)
                    if not b:
                        return
                    yield b
            finally:
                s.close()

    def _stream(self, ctx, t_lo, t_hi):
        counts = {"lines": 0}
        lines = text_lines(self._chunks(ctx))
        header = next(lines, None)
        if header is None:
            raise FormatError("the GLODAP file is empty")
        parser = Parser(header)
        buf = {}
        expos = set()
        t0 = time.time()
        for line in lines:
            if not line.strip():
                continue
            counts["lines"] += 1
            r = parser.parse(line, counts)
            if r is None:
                continue
            t, y, la, lo, v, expo, qc = r
            if not (t_lo <= t <= t_hi):
                counts["bottles_outside_window"] = \
                    counts.get("bottles_outside_window", 0) + 1
                continue
            b = buf.setdefault(y, {"t": [], "la": [], "lo": [], "v": [],
                                   "p": [], "q": [], "e": []})
            b["t"].append(t)
            b["la"].append(la)
            b["lo"].append(lo)
            b["v"].append(v)
            b["e"].append(expo)
            b["q"].append(qc)
            if len(b["t"]) >= FLUSH_ROWS:
                rows = self._pack(b, counts)
                buf[y] = {k: [] for k in b}
                if rows is not None:
                    yield y, rows, None
            if counts["lines"] % 500_000 == 0:
                ctx.prog.item(f"glodap {counts['lines'] // 1000:,}k lines",
                              extra={"elapsed_s": round(time.time() - t0, 1)})
            expos.add(expo)
        for y in sorted(buf):
            rows = self._pack(buf[y], counts)
            if rows is not None:
                yield y, rows, None
        counts["expocodes_in_window"] = len(expos)
        counts["file"] = self.master(ctx)[0]
        counts["stream_seconds"] = round(time.time() - t0, 1)
        yield None, None, counts

    def _pack(self, b, counts):
        if not b["t"]:
            return None
        v = np.asarray(b["v"], np.float64)
        oob = self.mask_bounds(v)
        if oob:
            f10b._merge_counts(counts, {"out_of_bounds": oob})
        alive = np.isfinite(v[:, 1:]).any(axis=1) & np.isfinite(v[:, 0])
        counts["bottles_no_value"] = counts.get("bottles_no_value", 0) + \
            int((~alive).sum())
        if not alive.any():
            return None
        idx = np.flatnonzero(alive)
        cache = {}
        plat = np.array([cache.setdefault(b["e"][i],
                                          f10b.platform_hash(b["e"][i]))
                         for i in idx], np.int64)
        qc = np.asarray(b["q"], np.uint8)[idx]
        qd = counts.setdefault("qc", {})
        for u, k in zip(*np.unique(qc, return_counts=True)):
            qd[str(int(u))] = qd.get(str(int(u)), 0) + int(k)
        counts["rows_kept"] = counts.get("rows_kept", 0) + int(idx.size)
        return self.pack(np.asarray(b["t"], np.int64)[idx],
                         np.asarray(b["la"])[idx], np.asarray(b["lo"])[idx],
                         v[idx], plat, qc)

    # --------------------------------------------------------- the contract --
    def index(self, ctx):
        name, ver = self.master(ctx)
        head = b"".join(self._chunks(ctx, limit=1 << 16))
        lines = head.decode("utf-8", "replace").split("\n")
        parser = Parser(lines[0])
        c = {}
        first = None
        for ln in lines[1:-1]:
            r = parser.parse(ln, c)
            if r is not None:
                first = {"expocode": r[5], "t": r[0],
                         "values": [None if np.isnan(x) else x
                                    for x in r[4].tolist()]}
                break
        if first is None:
            sys.exit(f"REFUSING glodap: no parsable line in the first "
                     f"{len(head):,} bytes of {name}")
        return {"dataset": f"GLODAPv{ver} merged master file",
                "url": ACCESSION + name, "version": ver,
                "columns": parser.ncol, "first_record": first,
                "head_counts": c}

    def fetch_stream(self, ctx):
        yield from self._stream(ctx, ctx.t_lo, ctx.t_hi)

    def fetch_month(self, ctx, year, month):
        """The whole file is read (it is ordered by cruise, not by time);
        only the month is kept. The counts describe that pass."""
        lo, hi = f10b.month_bounds_s(year, month)
        lo, hi = max(lo, ctx.t_lo), min(hi, ctx.t_hi)
        label = f"{year}-{month:02d}"
        for y, rows, c in self._stream(ctx, lo, hi):
            yield label, rows, c

    # ---------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260917):
        return make_smoke_sources(root, d_lo, d_hi, seed)


# ================================================================== smoke ==
def _header():
    """The real v3 header (119 names, measured 2026-09-17)."""
    return ("expocode,cruise,v22023_cruise,station,cast,year,month,day,hour,"
            "minute,latitude,longitude,bottomdepth,maxsampdepth,bottle,"
            "pressure,depth,temperature,temperaturef,temperatureqc,theta,"
            "constemperature,salinity,salinityf,salinityqc,abssalinity,"
            "sigma0,sigma1,sigma2,sigma3,sigma4,gamma,oxygen,oxygenf,"
            "oxygenqc,aou,aouf,nitrate,nitratef,nitrateqc,nitrite,nitritef,"
            "silicate,silicatef,silicateqc,phosphate,phosphatef,phosphateqc,"
            "tco2,tco2f,tco2qc,tco2calc,talk,talkf,talkqc,talkcalc,phts25p0,"
            "phts25p0f,phtsinsitutp,phtsinsitutpf,ph_tmp,ph_scale,"
            "phts25p0_calc,cfc11,pcfc11,cfc11f,cfc11qc,cfc12,pcfc12,cfc12f,"
            "cfc12qc,cfc113,pcfc113,cfc113f,cfc113qc,ccl4,pccl4,ccl4f,ccl4qc,"
            "sf6,psf6,sf6f,sf6qc,c13,c13f,c13qc,c14,c14f,c14err,h3,h3f,h3err,"
            "he3,he3f,he3err,he,hef,heerr,neon,neonf,neonerr,o18,o18f,toc,"
            "tocf,doc,docf,don,donf,tdn,tdnf,chla,chlaf,fco2,fco2f,fco2temp,"
            "fco2calc,doi,region")


def make_smoke_sources(root, d_lo, d_hi, seed=20260917):
    """GLODAP in its real layout: the accession listing (with the .mat, .nc
    and an older version's CSV as decoys) and a merged master CSV with the
    real 119-column header, ordered by cruise (so the window's rows are
    scattered through the file).

    Hostile on purpose: flag-0 and flag-9 values, a secondary-QC 0 value,
    missing hour/minute, a missing depth, a line with no kept value, an
    out-of-bounds phosphate, detection-limit negative nitrate (kept), a
    short line, and lines outside the window.
    """
    import datetime as dt
    rng = np.random.default_rng(seed)
    base = os.path.join(root, "glodap")
    os.makedirs(base, exist_ok=True)
    names = _header().split(",")
    col = {n: i for i, n in enumerate(names)}
    lines = []
    truth = []
    w_lo = f10b.seconds_since_epoch(d_lo)
    w_hi = f10b.seconds_since_epoch(d_hi) + 86399
    cruises = ("33RR20000101", "06AQ19991229", "49NZ20000102")
    k = 0
    for ci, expo in enumerate(cruises):
        day = d_lo - dt.timedelta(days=1)
        while day <= d_hi:
            for bottle, dep in enumerate((5.0, 100.0, 1000.0)):
                row = ["-9999"] * len(names)
                row[col["expocode"]] = expo
                row[col["doi"]] = "https://doi.org/10.0/x"
                row[col["year"]] = f"{day.year}.0"
                row[col["month"]] = f"{day.month}.0"
                row[col["day"]] = f"{day.day}.0"
                hh, mi = 6 + ci, 30
                row[col["hour"]] = f"{hh}.0"
                row[col["minute"]] = f"{mi}.0"
                la, lo = -30.0 + ci * 20 + day.day * 0.01, 150.0 + ci * 10
                row[col["latitude"]] = f"{la:.4f}"
                row[col["longitude"]] = f"{lo:.4f}"
                row[col["depth"]] = f"{dep}"
                vals = {"temperature": 20 - dep / 60, "salinity": 35.0,
                        "oxygen": 220.0, "nitrate": 1 + dep / 100,
                        "phosphate": 0.2 + dep / 1000, "silicate": 2.0,
                        "tco2": 2000.0 + dep / 10, "talk": 2300.0,
                        "phtsinsitutp": 8.05, "cfc11": 2.0, "cfc12": 1.0,
                        "sf6": 1.5}
                flags = {n: 2 for n in vals}
                qcs = {n: 1 for n in vals}
                v = [dep] + [np.nan] * (len(SPEC) - 1)
                qc = 0
                keep = True
                tag = (ci, day.day, bottle)
                if tag == (0, 1, 0):
                    flags["oxygen"] = 0               # calculated -> dropped
                if tag == (0, 1, 1):
                    qcs["tco2"] = 0                   # not secondary-QC'd
                    qc |= 1
                if tag == (1, 1, 0):
                    row[col["hour"]] = row[col["minute"]] = "-9999"
                    hh, mi = 12, 0
                    qc |= 2
                if tag == (1, 1, 1):
                    row[col["depth"]] = "-9999"
                    keep = False
                if tag == (1, 2, 2):
                    for n in flags:
                        flags[n] = 9
                    keep = False
                if tag == (2, 2, 0):
                    vals["phosphate"] = 2364.6          # out of bounds
                if tag == (2, 2, 1):
                    vals["nitrate"] = -0.04             # detection limit
                for j in range(1, len(SPEC)):
                    name, _u, lo_b, hi_b, vc, fc, qcol = SPEC[j]
                    x = vals[vc] + float(rng.normal(0, 0.001))
                    if vc == "nitrate" and tag == (2, 2, 1):
                        x = -0.04
                    row[col[vc]] = f"{x:.4f}"
                    row[col[fc]] = f"{flags[vc]}.0"
                    if qcol:
                        row[col[qcol]] = f"{qcs[vc]}.0"
                    if flags[vc] == 2 and lo_b <= round(x, 4) <= hi_b:
                        v[j] = round(x, 4)
                t = int(cm.days_from_civil(day.year, day.month, day.day)) * \
                    86400 + hh * 3600 + mi * 60
                if keep and w_lo <= t <= w_hi:
                    truth.append({"t": t, "lat": la, "lon": lo,
                                  "platform": f10b.platform_hash(expo),
                                  "v": v, "qc": qc,
                                  "_o": (day.year, k)})
                lines.append(",".join(row))
                k += 1
            day += dt.timedelta(days=1)
    lines.insert(5, "33RR20000101,1.0,short line")
    name = "GLODAPv3_Merged_Master_File.csv"
    with open(os.path.join(base, name), "w") as fh:
        fh.write(_header() + "\n" + "\n".join(lines) + "\n")
    decoys = ("GLODAPv2.2023_Merged_Master_File.csv",
              "GLODAPv3_Merged_Master_File.mat",
              "GLODAPv3_Merged_Master_File.nc", "GLODAPv3_Atlantic_Ocean.csv")
    for d in decoys:
        with open(os.path.join(base, d), "w") as fh:
            fh.write("not this one\n")
    rows = "".join(
        f'<tr><td valign="top"><img src="/icons/text.gif" alt="[TXT]"></td>'
        f'<td><a href="{n}">{n}</a></td><td align="right">2026-07-06 13:50  '
        f'</td><td align="right">1.0G</td><td>&nbsp;</td></tr>\n'
        for n in sorted(decoys + (name,)))
    with open(os.path.join(base, "index.html"), "w") as fh:
        fh.write("<html><head><title>Index of /data/oceans/ncei/ocads/data/"
                 "0315582</title></head><body><table>\n" + rows +
                 "</table></body></html>\n")
    truth.sort(key=lambda r: r.pop("_o"))
    return truth


ADAPTER = GLODAPAdapter
