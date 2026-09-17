"""NDBC standard meteorological data — one row per station report (family 1.0.tf).

PLAIN ENGLISH. NOAA's National Data Buoy Center archives the hourly (and,
since the 2000s, often 10-minute) reports of its moored buoys and coastal
automated (C-MAN) stations, plus partner stations it relays: wind, gusts,
waves, air pressure, air and sea temperature. This adapter turns that archive
into a tier-P store — one row per station per report time with ten channels —
so a model can ask "what did the nearest buoys measure in the last few days".

SOURCE, VERIFIED 2026-09-17 FROM THIS SANDBOX (no account, US government):
  https://www.ndbc.noaa.gov/data/historical/stdmet/                (listing)
  https://www.ndbc.noaa.gov/data/historical/stdmet/<sid>h<YYYY>.txt.gz
  https://www.ndbc.noaa.gov/data/stations/station_table.txt
  https://www.ndbc.noaa.gov/data/stations/station_owners.txt
The listing is an Apache index (3,717,742 bytes) of 17,049 files named
`<station>h<year>.txt.gz`: 1,350 stations (1,345 lower-case ids plus five
upper-case ones — 4cONF, 43WSL, 45T01, 53ANF, 53MKF — so ids are compared
case-insensitively and the URL keeps the listing's spelling), years
1970 .. 2025 (829 station files in 2025, the note's "829 with 2025 data").
The size column is HUMAN-READABLE ("8.8K", "1.2M"), so the index reports an
APPROXIMATE byte total (≈ 6.19 GB for the whole archive, 399.5 MB for 2023);
the exact bytes are the probe's. The current year is NOT here (NDBC publishes
it month by month elsewhere) — the record ends with the last full year.

THE HEADER CHANGED FOUR TIMES, SO COLUMNS ARE FOUND BY NAME. Measured on real
files: ≤1998 `YY MM DD hh WD WSPD GST WVHT DPD APD MWD BAR ATMP WTMP DEWP VIS`
(two-digit year: 76 = 1976); 1999 `YYYY MM DD hh ...`; 2000-2004 adds `TIDE`;
2005-2006 `YYYY MM DD hh mm WD ...`; 2007 onward `#YY MM DD hh mm WDIR WSPD
GST WVHT DPD APD MWD PRES ATMP WTMP DEWP VIS TIDE` followed by a `#yr mo dy hr
mn degT m/s ...` units line. `WD` is `WDIR` and `BAR` is `PRES` under the old
names. Missing values are per-column sentinels, measured over 40 random files
(1,642,943 lines): WDIR and MWD 999 (99 is a REAL direction — 3,286 WDIR and
622 MWD values of exactly 99 occur), WSPD/GST/WVHT/DPD/APD 99, PRES 9999 (999
and 999.9 are real pressures), ATMP/WTMP 999. Each is NaN and counted per
channel (`value_missing`). One line in the sample carried a year other than
its file's; such lines are dropped and counted (`rows_other_year`), since the
neighbouring year's file may hold the same report.

STATION POSITIONS come from station_table.txt (1,938 entries; `|`-separated
STATION_ID | OWNER | TTYPE | HULL | NAME | PAYLOAD | LOCATION | TIMEZONE |
FORECAST | NOTE, LOCATION as "44.794 N 87.313 W (...)"). The historical files
carry no position, and a station is given ONE position — a moved buoy is not
tracked. 45 of the 1,350 listed stations are absent from the table (retired
oil-platform and test ids such as 42360-42395, 4h361, 32st1); their reports
have no position and are dropped and counted (`rows_station_unknown`,
`stations_unknown`).

TAO / TRITON / PIRATA / RAMA ARE EXCLUDED — they are family 10.1's `gtmba`.
The rule reads the station table and the owner table, never a list of ids: a
station is a GTMBA site if its TTYPE is "Atlas Buoy" (the ATLAS / T-Flex
mooring those four arrays use), or its owner's name in station_owners.txt
names one of the arrays ("Prediction and Research Moored Array in the
Atlantic" = PIRATA, "Research Moored Array for African-Asian-Australian
Monsoon Analysis and Prediction" = RAMA, TAO, TRITON), or its name or type
mentions TAO or TRITON. Measured 2026-09-17: 49 entries of the station table
match (48 Atlas Buoys and the drifting "TAO Buoy Adrift"), and NONE of them
has a file in stdmet/ — NDBC serves the TAO array elsewhere — so the rule
removes 0 files today and is kept so that a future listing cannot leak a
mooring into two families (`stations_excluded_gtmba`).

WHAT A ROW IS.
  time_s    the report time (YYYY MM DD hh [mm]) in UTC, int32 seconds since
            1982-01-01 (negative before 1982; the record starts in 1970).
  lat, lon  the station table's position.
  platform  platform_hash(lower-case station id); platforms.json maps it back.
  values    WDIR (degT), WSPD (m/s), GST (m/s), WVHT (m), DPD (s), APD (s),
            MWD (degT), PRES (hPa), ATMP (degC), WTMP (degC). DEWP, VIS and
            TIDE are not kept.
  qc        0 for every row: the historical files are NDBC's quality-controlled
            archive and carry no per-value flag.
A report whose ten values are all NaN is dropped (`rows_all_blanked`); two
lines with the same station and second keep the LAST (`rows_duplicate_time`).

THE PROBE, 2023-06, MEASURED 2026-09-17 (`--stage probe --probe-month
2023-06`): 785 station files of 787 listed for 2023 read whole (the 2
without a table position counted: 1,190 June rows, 13,317 lines) — 409,960,838
bytes in 162 s — 40,408,079 lines for the year, of which 3,392,066 June rows
kept from 731 platforms (1,947 all-missing, 1,106 repeated seconds, 11 lines
of another year, 0 out of bounds); NaN fractions WDIR 0.23, WSPD 0.22, GST
0.24, WVHT 0.88, DPD 0.89, APD 0.93, MWD 0.90, PRES 0.27, ATMP 0.16, WTMP
0.38. So a recent year is ≈ 40 M rows (≈ 1.9 GB stored at 47 B/row) and
≈ 410 MB of gzip; scaling the listing's ≈ 6.19 GB by the measured 10.1 bytes
a line gives ≈ 6.3e8 rows and ≈ 30 GB stored for 1970-2025 (the note's 3-5e8
and 15-25 GB were low: the 10-minute stations of the 2010s dominate), and ≈
45 min for the whole archive at the probe's 2.5 MB/s (download and parse,
four workers).

MEMORY. One station-year at a time per download worker (the largest 2023 file
is ≈ 0.6 MB gzip, 47k lines), four workers, and the year's packed rows
(≈ 47 bytes each) until `fetch_year` yields them in batches of 1 M.
"""
import gzip
import os
import re
import sys
import time
import zlib

import numpy as np

import build_family10_stores as f10b
from family1.adapters import _common as cm

BASE = "https://www.ndbc.noaa.gov/data/historical/stdmet/"
STATIONS_URL = "https://www.ndbc.noaa.gov/data/stations/station_table.txt"
OWNERS_URL = "https://www.ndbc.noaa.gov/data/stations/station_owners.txt"

LISTING_ROW = re.compile(
    r'href="([0-9A-Za-z]+)h(\d{4})\.txt\.gz">[^<]*</a>\s*</td>\s*'
    r'<td[^>]*>[^<]*</td>\s*<td[^>]*>\s*([0-9.]+[KMG]?)\s*</td>')
LISTING_NAME = re.compile(r'href="([0-9A-Za-z]+)h(\d{4})\.txt\.gz"')
LOCATION = re.compile(r"^\s*([0-9.]+)\s*([NS])\s+([0-9.]+)\s*([EW])")

CHANNELS = (("WDIR", "degT", 0.0, 360.0),
            ("WSPD", "m/s", 0.0, 80.0),
            ("GST", "m/s", 0.0, 100.0),
            ("WVHT", "m", 0.0, 30.0),
            ("DPD", "s", 0.0, 40.0),
            ("APD", "s", 0.0, 40.0),
            ("MWD", "degT", 0.0, 360.0),
            ("PRES", "hPa", 850.0, 1090.0),
            ("ATMP", "degC", -60.0, 60.0),
            ("WTMP", "degC", -3.0, 40.0))
ALIASES = {"WD": "WDIR", "BAR": "PRES"}
SENTINEL = {"WDIR": 999.0, "MWD": 999.0, "WSPD": 99.0, "GST": 99.0,
            "WVHT": 99.0, "DPD": 99.0, "APD": 99.0, "PRES": 9999.0,
            "ATMP": 999.0, "WTMP": 999.0}
GTMBA_OWNER = re.compile(
    r"Prediction and Research Moored Array|Research Moored Array for "
    r"African|\bTAO\b|TRITON|PIRATA|\bRAMA\b", re.I)
GTMBA_TEXT = re.compile(r"\bTAO\b|TRITON", re.I)
GTMBA_TYPE = "atlas buoy"


class FormatError(ValueError):
    """A stdmet file that does not have a layout this parser knows."""


def approx_bytes(s):
    s = s.strip()
    mult = {"K": 1e3, "M": 1e6, "G": 1e9}
    if s and s[-1] in mult:
        return int(float(s[:-1]) * mult[s[-1]])
    return int(float(s))


def parse_listing(html):
    """{year: {sid_lower: (sid_as_listed, approx_bytes)}}."""
    names = LISTING_NAME.findall(html)
    rows = LISTING_ROW.findall(html)
    if not names:
        raise FormatError("the stdmet listing names no <sid>h<year>.txt.gz "
                          "file — an empty listing is a broken listing")
    if len(rows) != len(names):
        raise FormatError(f"the stdmet listing names {len(names)} files but "
                          f"the size column parsed for {len(rows)} — the page "
                          f"layout has changed; fix LISTING_ROW")
    out = {}
    for sid, y, sz in rows:
        d = out.setdefault(int(y), {})
        k = sid.lower()
        if k in d:
            raise FormatError(f"{sid}h{y} and {d[k][0]}h{y} are the same "
                              f"station up to case")
        d[k] = (sid, approx_bytes(sz))
    return dict(sorted(out.items()))


def parse_owners(text):
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or "|" not in line:
            continue
        p = [x.strip() for x in line.split("|")]
        out[p[0]] = {"name": p[1], "country": p[2] if len(p) > 2 else ""}
    if not out:
        raise FormatError("station_owners.txt parsed to ZERO owners")
    return out


def parse_station_table(text, owners):
    """{sid_lower: {...}} with lat/lon, plus the GTMBA flag per entry."""
    out = {}
    bad = 0
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = [x.strip() for x in line.split("|")]
        if len(p) < 7:
            bad += 1
            continue
        m = LOCATION.match(p[6])
        if not m:
            bad += 1
            continue
        lat = float(m.group(1)) * (1 if m.group(2) == "N" else -1)
        lon = float(m.group(3)) * (1 if m.group(4) == "E" else -1)
        if not (-90 <= lat <= 90 and -180 <= lon <= 360):
            bad += 1
            continue
        own = owners.get(p[1], {})
        gtmba = (p[2].lower() == GTMBA_TYPE
                 or bool(GTMBA_OWNER.search(own.get("name", "")))
                 or bool(GTMBA_TEXT.search(p[4]))
                 or bool(GTMBA_TEXT.search(p[2])))
        out[p[0].lower()] = {
            "id": p[0], "owner": p[1], "owner_name": own.get("name") or None,
            "type": p[2] or None, "hull": p[3] or None, "name": p[4] or None,
            "payload": p[5] or None, "lat": lat,
            "lon": float(f10b.f10.wrap_lon(lon)), "gtmba": gtmba}
    if not out:
        raise FormatError("station_table.txt parsed to ZERO stations")
    return out, bad


def parse_stdmet(raw, year, t_lo, t_hi, counts):
    """One decompressed station-year -> (t int64, values (n, 10)).

    Parses by header name. Every drop is counted in `counts`.
    """
    text = raw.decode("latin-1")
    lines = text.splitlines()
    if not lines:
        raise FormatError("an empty stdmet file")
    head = lines[0].lstrip("#").split()
    if not head or head[0] not in ("YY", "YYYY"):
        raise FormatError(f"unknown stdmet header {lines[0][:60]!r}")
    names = [ALIASES.get(h, h) for h in head]
    need = ["MM", "DD", "hh"] + [c[0] for c in CHANNELS]
    miss = [n for n in need if n not in names]
    if miss:
        raise FormatError(f"stdmet header {head} lacks {miss}")
    body = [ln for ln in lines[1:] if ln.strip() and not ln.startswith("#")]
    counts["lines_header"] = counts.get("lines_header", 0) + \
        (len(lines) - len(body))
    ncol = len(names)
    toks = " ".join(body).split()
    if len(toks) == ncol * len(body):
        arr = np.array(toks, dtype=np.float64).reshape(len(body), ncol)
    else:
        good = []
        for ln in body:
            p = ln.split()
            if len(p) == ncol:
                good.append(p)
        counts["lines_bad_length"] = counts.get("lines_bad_length", 0) + \
            len(body) - len(good)
        if len(body) and len(good) < 0.99 * len(body):
            raise FormatError(f"{len(body) - len(good)} of {len(body)} lines "
                              f"do not have the header's {ncol} columns")
        arr = np.array(good, dtype=np.float64).reshape(len(good), ncol)
    col = {n: i for i, n in enumerate(names)}
    n = arr.shape[0]
    counts["lines"] = counts.get("lines", 0) + n
    if n == 0:
        return np.zeros(0, np.int64), np.zeros((0, len(CHANNELS)))
    yr = arr[:, 0].astype(np.int64)
    if names[0] == "YY":
        yr = np.where(yr < 100, yr + 1900, yr)
    mo = arr[:, col["MM"]].astype(np.int64)
    dd = arr[:, col["DD"]].astype(np.int64)
    hh = arr[:, col["hh"]].astype(np.int64)
    mi = arr[:, col["mm"]].astype(np.int64) if "mm" in col else \
        np.zeros(n, np.int64)
    ok = cm.valid_date(yr, mo, dd) & (hh >= 0) & (hh <= 23) & \
        (mi >= 0) & (mi <= 59)
    counts["lines_bad_time"] = counts.get("lines_bad_time", 0) + \
        int((~ok).sum())
    other = ok & (yr != year)
    counts["rows_other_year"] = counts.get("rows_other_year", 0) + \
        int(other.sum())
    ok &= ~other
    t = (cm.days_from_civil(np.where(ok, yr, 2000), np.where(ok, mo, 1),
                            np.where(ok, dd, 1)) * 86400
         + hh * 3600 + mi * 60)
    inside = ok & (t >= t_lo) & (t <= t_hi)
    counts["rows_outside_window"] = counts.get("rows_outside_window", 0) + \
        int((ok & ~inside).sum())
    vals = np.stack([arr[:, col[c[0]]] for c in CHANNELS], axis=1)
    t, vals = t[inside], vals[inside]
    miss = counts.setdefault("value_missing", {})
    for j, c in enumerate(CHANNELS):
        s = vals[:, j] == SENTINEL[c[0]]
        k = int(s.sum())
        if k:
            miss[c[0]] = miss.get(c[0], 0) + k
        vals[s, j] = np.nan
    # the LAST line of a repeated second wins
    if t.size:
        order = np.argsort(t, kind="stable")
        t, vals = t[order], vals[order]
        last = np.ones(t.size, bool)
        last[:-1] = t[1:] != t[:-1]
        counts["rows_duplicate_time"] = counts.get("rows_duplicate_time", 0) \
            + int((~last).sum())
        t, vals = t[last], vals[last]
    return t, vals


# ================================================================ adapter ==
class NDBCAdapter(f10b.SourceAdapter):
    store = "ndbc"
    title = ("NDBC buoys and C-MAN stations, standard meteorological "
             "reports (NOAA), one row per station report")
    family = "1tf"
    distribution = "public"
    licence = {"name": "US government work (NOAA NDBC), no restriction",
               "redistribution": "yes", "derived_works": "free",
               "attribution": "NOAA National Data Buoy Center, "
                              "https://www.ndbc.noaa.gov"}
    time_dtype = "int32"
    platform_meta = True
    credentials = ()
    channels = CHANNELS
    log2_fp = -4.0                                   # a point instrument
    log2_dt = float(np.log2((1.0 / 24.0) / 5.0))     # one hour of a 5-day bin
    per_year = True
    first_year = 1970
    qc_policy = (
        "qc 0 for every row: stdmet/ is NDBC's quality-controlled historical "
        "archive and carries no per-value flag. Per-column missing sentinels "
        "(WDIR/MWD 999, WSPD/GST/WVHT/DPD/APD 99, PRES 9999, ATMP/WTMP 999) "
        "are NaN and counted (`value_missing`); a value outside its channel's "
        "bounds is NaN and counted (`out_of_bounds`); a report whose ten "
        "values are all NaN is dropped (`rows_all_blanked`); a repeated "
        "station-second keeps the last line (`rows_duplicate_time`).")
    sources = (BASE + "<station>h<YYYY>.txt.gz", STATIONS_URL, OWNERS_URL,
               BASE)
    verified = (
        "2026-09-17 from the sandbox: the stdmet listing (17,049 files, "
        "1,350 stations, 1970..2025, ≈6.19 GB by its human-readable size "
        "column); station_table.txt (364,536 B, 1,938 entries); "
        "station_owners.txt (7,731 B); files in all four header eras "
        "(41001h1976, h1995, h1999, h2000, h2005, h2006, h2007, 46042h2023, "
        "0y2w3h2012, 42360h2020, 45T01h2018) and 40 random files for the "
        "sentinels")
    notes = (
        "Columns are read by header name (WD = WDIR, BAR = PRES). One "
        "position per station, from station_table.txt; stations absent from "
        "it are dropped and counted. GTMBA moorings (TAO/TRITON/PIRATA/RAMA) "
        "are excluded by the station and owner tables — they are family "
        "10.1's gtmba. The current year is not in stdmet/.")
    smoke_window = ("1998-12-30", "1999-01-02")
    smoke_probe_month = "1999-01"
    fetch_month_scope = "month"

    WORKERS = 4
    YIELD_ROWS = 1_000_000

    def __init__(self):
        self._listing = None
        self._stations = None

    # -------------------------------------------------------------- paths --
    def _local(self, ctx, *parts):
        return os.path.join(ctx.source_dir, "ndbc", *parts)

    def _read_local(self, ctx, *parts):
        with open(self._local(ctx, *parts), "rb") as fh:
            raw = fh.read()
        ctx.count_bytes(len(raw))
        return raw

    def _get(self, ctx, url, what):
        raw, why = cm.get_bytes(url, attempts=ctx.a.attempts)
        if raw is None:
            sys.exit(f"{url} answered 404 ({why}) — {what} has moved")
        return raw

    # ------------------------------------------------------------ listing --
    def listing(self, ctx):
        if self._listing is None:
            raw = (self._read_local(ctx, "stdmet", "index.html")
                   if ctx.source_dir else self._get(ctx, BASE, "the listing"))
            try:
                self._listing = parse_listing(raw.decode("latin-1"))
            except FormatError as e:
                sys.exit(f"REFUSING: {e} ({len(raw)} bytes read)")
        return self._listing

    def stations(self, ctx):
        if self._stations is None:
            if ctx.source_dir:
                st = self._read_local(ctx, "station_table.txt")
                ow = self._read_local(ctx, "station_owners.txt")
            else:
                st = self._get(ctx, STATIONS_URL, "the station table")
                ow = self._get(ctx, OWNERS_URL, "the owner table")
            owners = parse_owners(ow.decode("latin-1"))
            tab, bad = parse_station_table(st.decode("latin-1"), owners)
            for k, e in tab.items():
                e["platform"] = f10b.platform_hash(k)
            self._stations = {"table": tab, "bad_lines": bad,
                              "owners": len(owners)}
        return self._stations

    def _file(self, ctx, sid, year):
        """(raw gzip bytes, url) for one station-year."""
        name = f"{sid}h{year}.txt.gz"
        if ctx.source_dir:
            return self._read_local(ctx, "stdmet", name), name
        url = BASE + name
        raw, why = cm.get_bytes(url, attempts=ctx.a.attempts)
        return raw, url

    # ------------------------------------------------------------ parsing --
    def _station_year(self, ctx, key, sid, year, t_lo, t_hi):
        """One file -> (key, t, vals, counts, error)."""
        counts = {}
        try:
            raw, url = self._file(ctx, sid, year)
            if raw is None:
                return key, None, None, counts, f"{url}: listed, and 404"
            try:
                text = gzip.decompress(raw)
            except (OSError, EOFError, zlib.error) as e:
                return key, None, None, counts, (
                    f"{url}: not a whole gzip ({type(e).__name__}: {e}) — a "
                    f"truncated download")
            t, v = parse_stdmet(text, year, t_lo, t_hi, counts)
            counts["files"] = 1
            return key, t, v, counts, None
        except FormatError as e:
            raise FormatError(f"{sid}h{year}: {e}") from None
        except IOError as e:
            return key, None, None, counts, f"{sid}h{year}: {e}"

    def _year(self, ctx, year, t_lo, t_hi, label):
        lst = self.listing(ctx).get(year, {})
        st = self.stations(ctx)["table"]
        counts = {"stations_listed": len(lst)}
        todo, unknown, gt = [], [], []
        for key in sorted(lst):
            e = st.get(key)
            if e is None:
                unknown.append(key)
            elif e["gtmba"]:
                gt.append(key)
            else:
                todo.append((key, lst[key][0]))
        counts["stations_excluded_gtmba"] = len(gt)
        counts["stations_unknown"] = len(unknown)
        # a station absent from the table has no position: its file is still
        # READ, so the rows it would have given are counted, not guessed
        for key in unknown:
            k, t, v, c, err = self._station_year(ctx, key, lst[key][0], year,
                                                 t_lo, t_hi)
            if err:
                ctx.note_absent(year, err)
                continue
            counts["rows_station_unknown"] = \
                counts.get("rows_station_unknown", 0) + int(len(t))
            counts["lines_station_unknown"] = \
                counts.get("lines_station_unknown", 0) + c.get("lines", 0)
        parts, n = [], 0
        work = [(k, sid) for k, sid in todo]
        fn = (lambda ks: self._station_year(ctx, ks[0], ks[1], year,
                                            t_lo, t_hi))
        workers = 1 if ctx.source_dir else self.WORKERS
        for key, t, v, c, err in cm.ordered_map(fn, work, workers):
            if err:
                ctx.note_absent(year, err)
                counts["files_failed"] = counts.get("files_failed", 0) + 1
                counts.setdefault("files_failed_ids", []).append(key)
                continue
            f10b._merge_counts(counts, {k: vv for k, vv in c.items()})
            if t is None or not len(t):
                continue
            e = st[key]
            oob = self.mask_bounds(v)
            if oob:
                f10b._merge_counts(counts, {"out_of_bounds": oob})
            alive = np.isfinite(v).any(axis=1)
            counts["rows_all_blanked"] = counts.get("rows_all_blanked", 0) + \
                int((~alive).sum())
            t, v = t[alive], v[alive]
            if not len(t):
                continue
            m = len(t)
            parts.append(self.pack(t, np.full(m, e["lat"]),
                                   np.full(m, e["lon"]), v,
                                   np.full(m, e["platform"], np.int64),
                                   np.zeros(m, np.uint8)))
            n += m
            counts["rows_kept"] = counts.get("rows_kept", 0) + m
            if n >= self.YIELD_ROWS:
                yield label, cm.concat_rows(parts, self.C, self.time_dtype), \
                    None
                parts, n = [], 0
        counts["qc"] = {"0": counts.get("rows_kept", 0)}
        rows = cm.concat_rows(parts, self.C, self.time_dtype)
        yield label, rows, counts

    # --------------------------------------------------------- the contract --
    def index(self, ctx):
        lst = self.listing(ctx)
        st = self.stations(ctx)
        want = [y for y in ctx.years if y in lst]
        first = None
        if want:
            y = want[0]
            key = min(lst[y], key=lambda k: lst[y][k][1])
            sid = lst[y][key][0]
            raw, url = self._file(ctx, sid, y)
            if raw is None:
                sys.exit(f"{url} is listed and answered 404")
            text = gzip.decompress(raw)
            c = {}
            parse_stdmet(text, y, -2**62, 2**62, c)        # raises if malformed
            first = {"file": f"{sid}h{y}.txt.gz",
                     "header": text.decode("latin-1").splitlines()[0],
                     "lines": c.get("lines", 0)}
        tab = st["table"]
        all_keys = set().union(*[set(d) for d in lst.values()]) if lst else set()
        return {
            "dataset": "NDBC historical stdmet",
            "url": BASE,
            "listing": {"files": sum(len(d) for d in lst.values()),
                        "stations": len(all_keys),
                        "first": min(lst), "last": max(lst)},
            "year_files": {str(y): len(lst[y]) for y in want},
            "bytes_approx": int(sum(b for y in want
                                    for _, b in lst[y].values())),
            "bytes_note": "the listing's size column is human-readable "
                          "(K/M); the probe measures exact bytes",
            "years_not_listed": [y for y in ctx.years if y not in lst],
            "station_table_entries": len(tab),
            "station_table_bad_lines": st["bad_lines"],
            "owners": st["owners"],
            "gtmba_entries": sorted(e["id"] for e in tab.values()
                                    if e["gtmba"]),
            "listed_stations_gtmba": sorted(k for k in all_keys
                                            if k in tab and tab[k]["gtmba"]),
            "listed_stations_without_position": sorted(
                k for k in all_keys if k not in tab),
            "first_record": first,
        }

    def fetch_year(self, ctx, year):
        lst = self.listing(ctx)
        if year not in lst:
            ctx.note_absent(year, f"no <sid>h{year}.txt.gz in the stdmet "
                                  f"listing ({min(lst)}..{max(lst)})")
            return
        t0 = time.time()
        for label, rows, counts in self._year(ctx, year, ctx.t_lo, ctx.t_hi,
                                              str(year)):
            if counts is not None:
                counts["fetch_seconds"] = round(time.time() - t0, 1)
            yield from cm.batch_rows(self, rows, counts, label,
                                     self.YIELD_ROWS)

    def fetch_month(self, ctx, year, month):
        """The probe: every station file of the year, rows of `month` kept."""
        lst = self.listing(ctx)
        if year not in lst:
            sys.exit(f"no stdmet files for {year} in the listing")
        lo, hi = f10b.month_bounds_s(year, month)
        lo, hi = max(lo, ctx.t_lo), min(hi, ctx.t_hi)
        for label, rows, counts in self._year(ctx, year, lo, hi,
                                              f"{year}-{month:02d}"):
            yield from cm.batch_rows(self, rows, counts, label,
                                     self.YIELD_ROWS)

    def platforms(self, ctx):
        tab = self.stations(ctx)["table"]
        out = {}
        for k, e in tab.items():
            if e["gtmba"]:
                continue
            out[int(e["platform"])] = {
                "id": e["id"], "lat": e["lat"], "lon": e["lon"],
                "name": e["name"], "owner": e["owner"],
                "owner_name": e["owner_name"], "type": e["type"],
                "hull": e["hull"], "payload": e["payload"]}
        return out

    # ---------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260917):
        return make_smoke_sources(root, d_lo, d_hi, seed)


# ================================================================== smoke ==
# SORTED BY LOWER-CASE ID, because the adapter emits a year in that order and
# several stations report at the same second (ties keep yield order).
SMOKE_STATIONS = (
    # id, owner, ttype, name, lat, lon, era
    ("41001", "N", "6-meter NOMAD buoy", "EAST HATTERAS", 34.7, -72.7, "yy"),
    ("45T01", "GL", "Buoy", "SMOKE UPPER CASE", 41.9, -87.3, "modern"),
    ("46042", "N", "3-meter discus buoy", "MONTEREY", 36.8, -122.4, "yyyy"),
    ("51xx1", "N", "Moored Buoy", "SMOKE TENMIN", -14.3, 170.5, "modern10"),
)
assert [s[0].lower() for s in SMOKE_STATIONS] == \
    sorted(s[0].lower() for s in SMOKE_STATIONS)
SMOKE_GTMBA = ("13008", "PR", "Atlas Buoy", "Reggae", 15.0, -38.0)
SMOKE_UNKNOWN = "42360"

HEADERS = {
    "yy": ("YY MM DD hh WD   WSPD GST  WVHT  DPD   APD  MWD  BAR    ATMP  "
           "WTMP  DEWP  VIS", None),
    "yyyy": ("YYYY MM DD hh WD   WSPD GST  WVHT  DPD   APD  MWD  BAR    ATMP  "
             "WTMP  DEWP  VIS  TIDE", None),
    "modern": ("#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  "
               "ATMP  WTMP  DEWP  VIS  TIDE",
               "#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  "
               "degC  degC  degC   mi    ft"),
}


def _line(era, y, mo, d, h, mi, v):
    """v = the ten channel values in CHANNELS order, raw (sentinels kept)."""
    wd, ws, gs, wh, dp, ap, mw, pr, at, wt = v
    body = (f"{int(wd):3d} {ws:4.1f} {gs:4.1f} {wh:5.2f} {dp:5.2f} {ap:5.2f} "
            f"{int(mw):3d} {pr:6.1f} {at:5.1f} {wt:5.1f}")
    if era == "yy":
        return f"{y % 100:02d} {mo:02d} {d:02d} {h:02d} {body} 999.0 99.0"
    if era == "yyyy":
        return f"{y:4d} {mo:02d} {d:02d} {h:02d} {body} 999.0 99.0 99.00"
    return (f"{y:4d} {mo:02d} {d:02d} {h:02d} {mi:02d} {body} 999.0 99.0 "
            f"99.00")


def _listing_html(files):
    rows = "".join(
        f'<tr><td valign="top"><img src="/icons/compressed.gif" alt="[   ]">'
        f'</td><td><a href="{n}">{n}</a></td><td align="right">2026-02-12 '
        f'19:41  </td><td align="right">{sz}</td><td>&nbsp;</td></tr>\n'
        for n, sz in sorted(files.items()))
    return ("<!DOCTYPE html>\n<html><head><title>NDBC - Index of "
            "/data/historical/stdmet/</title></head><body><main>\n<h1>Index "
            "of /data/historical/stdmet/</h1>\n<table>\n<tr><th><a href="
            "\"?C=N;O=D\">Name</a></th></tr>\n<tr><td></td><td><a href=\""
            "/data/historical/\">Parent Directory</a></td><td>&nbsp;</td>"
            "<td align=\"right\">  - </td></tr>\n" + rows +
            "</table></main></body></html>\n")


def _human(n):
    return f"{n / 1e3:.1f}K" if n < 1e6 else f"{n / 1e6:.1f}M"


def make_smoke_sources(root, d_lo, d_hi, seed=20260917):
    """A two-year NDBC archive in the real on-disk layout, four header eras.

    Hostile on purpose: a two-digit-year file, a no-minute YYYY file, the
    modern `#YY ... mm` file with its units line, a 10-minute station with an
    upper-case id, every per-column sentinel (and a REAL direction of 99), a
    value out of bounds, an all-missing report, a repeated second, a line
    from another year, a malformed date, a day before the window, a GTMBA
    Atlas buoy (excluded) and a station absent from the table.
    """
    import datetime as dt
    rng = np.random.default_rng(seed)
    base = os.path.join(root, "ndbc")
    os.makedirs(os.path.join(base, "stdmet"), exist_ok=True)
    with open(os.path.join(base, "station_owners.txt"), "w") as fh:
        fh.write("## Station Owners file format is:\n# OWNERCODE | OWNERNAME "
                 "| COUNTRYCODE\nN  |NDBC  |US    \nGL |Great Lakes|US\n"
                 "PR |Prediction and Research Moored Array in the Atlantic|FR"
                 "    \n")
    with open(os.path.join(base, "station_table.txt"), "w") as fh:
        fh.write("# STATION_ID | OWNER | TTYPE | HULL | NAME | PAYLOAD | "
                 "LOCATION | TIMEZONE | FORECAST | NOTE\n#\n")
        for sid, own, typ, nm, la, lo, _ in SMOKE_STATIONS + (
                SMOKE_GTMBA + ("modern",),):
            fh.write(f"{sid}|{own}|{typ}||{nm}||{abs(la):.3f} "
                     f"{'N' if la >= 0 else 'S'} {abs(lo):.3f} "
                     f"{'E' if lo >= 0 else 'W'} (x)|| |\n")
    lines = {}                        # (sid, year) -> [str]
    truth = []

    def emit(sid, era, when, v, keep, la, lo):
        lines.setdefault((sid, when.year), []).append(
            _line("modern" if era.startswith("modern") else era, when.year,
                  when.month, when.day, when.hour, when.minute, v))
        if keep:
            vv = [np.nan if x == SENTINEL[c[0]] else float(x)
                  for x, c in zip(v, CHANNELS)]
            truth.append({"t": f10b.seconds_since_epoch(when), "lat": la,
                          "lon": lo, "platform": f10b.platform_hash(
                              sid.lower()), "v": vv, "qc": 0})

    start = dt.datetime.combine(d_lo, dt.time()) - dt.timedelta(hours=3)
    stop = dt.datetime.combine(d_hi, dt.time(23, 59))
    w_lo = dt.datetime.combine(d_lo, dt.time())
    for sid, own, typ, nm, la, lo, era in SMOKE_STATIONS:
        step = dt.timedelta(minutes={"yy": 60, "yyyy": 60, "modern": 30,
                                     "modern10": 10}[era])
        when = start
        i = 0
        while when <= stop:
            if era == "yy" and when.year > 1998:
                era_now = "yyyy"
            else:
                era_now = era
            v = [float(rng.integers(0, 360)),
                 round(float(rng.uniform(0, 20)), 1),
                 round(float(rng.uniform(0, 25)), 1),
                 round(float(rng.uniform(0, 5)), 2),
                 round(float(rng.uniform(2, 15)), 2),
                 round(float(rng.uniform(2, 10)), 2),
                 float(rng.integers(0, 360)),
                 round(float(rng.uniform(990, 1030)), 1),
                 round(float(rng.uniform(-5, 25)), 1),
                 round(float(rng.uniform(0, 25)), 1)]
            # sentinels, on a rotating channel
            if i % 3 == 0:
                j = i // 3 % 10
                v[j] = SENTINEL[CHANNELS[j][0]]
            if i % 7 == 1:
                v[0] = 99.0                    # a REAL direction of 99 degT
            keep = when >= w_lo
            if sid == "46042" and when == dt.datetime(1999, 1, 1, 5):
                v[8] = 70.0                     # ATMP out of bounds -> NaN
                emit(sid, era_now, when, v, False, la, lo)
                vv = list(v)
                truth.append({"t": f10b.seconds_since_epoch(when),
                              "lat": la, "lon": lo,
                              "platform": f10b.platform_hash(sid.lower()),
                              "v": [np.nan if (k == 8 or x == SENTINEL[
                                  CHANNELS[k][0]]) else float(x)
                                  for k, x in enumerate(vv)], "qc": 0})
            elif sid == "41001" and when == dt.datetime(1999, 1, 1, 7):
                emit(sid, era_now, when,
                     [SENTINEL[c[0]] for c in CHANNELS], False, la, lo)
            elif sid == "45T01" and when == dt.datetime(1999, 1, 2, 0):
                # the same second twice: the first line is dropped
                emit(sid, era_now, when, [1.0] * 7 + [1000.0, 1.0, 1.0],
                     False, la, lo)
                emit(sid, era_now, when, v, True, la, lo)
            else:
                emit(sid, era_now, when, v, keep, la, lo)
            when += step
            i += 1
    # a GTMBA mooring and a station without a position: both files exist
    for sid in (SMOKE_GTMBA[0], SMOKE_UNKNOWN):
        for day in (d_lo, d_hi):
            when = dt.datetime.combine(day, dt.time(12))
            lines.setdefault((sid, when.year), []).append(
                _line("modern", when.year, when.month, when.day, 12, 0,
                      [10, 5.0, 6.0, 1.0, 8.0, 6.0, 20, 1010.0, 10.0, 12.0]))
    # a line from another year, and a malformed date, in the 1999 46042 file
    lines[("46042", 1999)].insert(0, _line("yyyy", 1998, 12, 31, 23, 0,
                                           [10, 5.0, 6.0, 1.0, 8.0, 6.0, 20,
                                            1010.0, 10.0, 12.0]))
    lines[("46042", 1999)].append(_line("yyyy", 1999, 2, 30, 1, 0,
                                        [10, 5.0, 6.0, 1.0, 8.0, 6.0, 20,
                                         1010.0, 10.0, 12.0]))
    sizes = {}
    eras = {s[0]: s[6] for s in SMOKE_STATIONS}
    for (sid, y), ls in sorted(lines.items()):
        era = eras.get(sid, "modern")
        if era == "yy" and y > 1998:
            era = "yyyy"
        h, u = HEADERS["modern" if era.startswith("modern") else era]
        body = h + "\n" + (u + "\n" if u else "") + "\n".join(ls) + "\n"
        name = f"{sid}h{y}.txt.gz"
        p = os.path.join(base, "stdmet", name)
        with open(p, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
                gz.write(body.encode("latin-1"))
        sizes[name] = _human(os.path.getsize(p))
    with open(os.path.join(base, "stdmet", "index.html"), "w") as fh:
        fh.write(_listing_html(sizes))
    return truth


ADAPTER = NDBCAdapter
