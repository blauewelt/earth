"""GHCN-Daily — one row per (station, day), five channels (family 1.0.tf).

PLAIN ENGLISH. The Global Historical Climatology Network daily archive is
NOAA NCEI's compilation of daily weather-station reports: over 130,000
stations, the oldest from 1763. This adapter turns it into a tier-P store —
one row per station per day with the day's maximum and minimum temperature,
precipitation, snowfall and snow depth — so a model can ask "what did the
nearest stations measure in the last few days".

SOURCE, VERIFIED 2026-09-17 FROM THIS SANDBOX (no account, CC0):
  https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/          (listing)
  https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/YYYY.csv.gz
  https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt
The listing is an Apache HTML table of 264 `YYYY.csv.gz` files (1763 .. 2026)
plus two text files, each with its size; the year list is read from it, never
assumed. A year file holds one line per ELEMENT per station-day:
`ID,YYYYMMDD,ELEMENT,VALUE,MFLAG,QFLAG,SFLAG,OBSTIME`, and the first three
commas sit at FIXED offsets 11, 20 and 25 (the ID is 11 characters, the date
8, the element 4) — which is what lets the parser below be vectorised.

THE FILE IS NOT SORTED, AND THE GROUPING IS DESIGNED FOR THAT. Measured on the
whole 2024 file (168,280,901 bytes gzip, 37,106,916 lines, 12,470,777
station-days over 44,668 stations): not by station, and not strictly by date
— the lines run 01-01 .. 02-29 in date order, then 96,937 short date runs
seesawing between 02-28 and 02-29, then 02-28 .. 12-31 in date order; and
132,171 station-days are SPLIT across non-adjacent lines. A streaming
"flush when the key changes" grouping would therefore write duplicate rows.
So each year is parsed, 16 MiB of decompressed text at a time (`BLOCK_BYTES`),
into compact columns — station id S11, day of year int16, element int8, value
int32, value width int8, QFLAG bool, month int8: 21 bytes per KEPT element
line — and grouped by ONE stable argsort on (station, day, element). MEMORY IS
BOUNDED BY THE YEAR, not the archive: measured on the whole 2024 file (28.4 M
kept element lines of 37.1 M), the parse peaks at 1.5 GB RSS and the grouping
at 2.7 GB — about 95 bytes per kept line including the interpreter — in 27 s
and 15 s. A hosted runner's 16 GB therefore holds a year five times 2024's.

WHAT A ROW IS.
  time_s    the DAY at 00:00 UTC, int64 seconds since 1982-01-01 (schema 3:
            the record starts in 1763). OBSTIME — the station's local
            observation hour from NCEI's station history — is IGNORED: a
            daily value is a statement about a day, the hour is populated for
            a minority of stations, and it is local time with no zone.
  lat, lon  the station's position from ghcnd-stations.txt. GHCN-Daily keeps
            one position per station; a station that moved is not tracked.
  platform  platform_hash(ID); `platforms.json` maps each hash back to the ID,
            position, elevation and name.
  values    TMAX, TMIN (degC; the archive stores tenths), PRCP (mm; tenths),
            SNOW (mm), SNWD (mm). NaN where the station reported no such
            element that day.
  qc        0 when no used element carried a QFLAG; 1 when at least one did —
            that element's value is set to NaN and counted per element in
            `qflag_blanked`. (A different meaning from family 10's 0 = "not
            assessed": here every value has been through NCEI's own QC.)
MFLAG and SFLAG are ignored. A value outside the channel's bounds becomes NaN
and is counted in `out_of_bounds` (never clipped); a -9999 value is counted in
`value_missing`. A station-day whose five values all ended up NaN is dropped
and counted (`rows_all_blanked`); a line for a station missing from
ghcnd-stations.txt has no position and is dropped and counted
(`lines_station_unknown`); a line whose date is not a real day of the file's
year is dropped and counted (`lines_bad_date`).
"""
import calendar
import datetime as dt
import gzip
import os
import re
import sys
import time
import zlib

import numpy as np

import build_family10_stores as f10b

BASE = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/"
BY_YEAR = BASE + "by_year/"
STATIONS_URL = BASE + "ghcnd-stations.txt"
ELEMENTS = ("TMAX", "TMIN", "PRCP", "SNOW", "SNWD")
SCALE = np.array([0.1, 0.1, 0.1, 1.0, 1.0], np.float64)
MISSING = -9999
LISTING_ROW = re.compile(
    r'href="(\d{4})\.csv\.gz"[^<]*</a>\s*</td>\s*<td[^>]*>[^<]*</td>\s*'
    r'<td[^>]*>\s*(\d+)\s*</td>', re.S)
LISTING_NAME = re.compile(r'href="(\d{4})\.csv\.gz"')


def _elem_code(s):
    b = s.encode("ascii")
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]


ELEM_CODES = np.array([_elem_code(e) for e in ELEMENTS], np.int64)


class FormatError(ValueError):
    """A line that does not have the layout the parser was written for."""


# ------------------------------------------------------------ the stations --
def parse_stations(text):
    """ghcnd-stations.txt (fixed width) -> (sorted ids S11, dict of arrays).

    Columns (NCEI readme, section IV): ID 1-11, LATITUDE 13-20, LONGITUDE
    22-30, ELEVATION 32-37 (-999.9 = missing), STATE 39-40, NAME 42-71, GSN
    73-75, HCN/CRN 77-79, WMO ID 81-85.
    """
    ids, lat, lon, elev, meta = [], [], [], [], []
    bad = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            sid = line[0:11]
            la = float(line[12:20])
            lo = float(line[21:30])
            el = float(line[31:37])
        except ValueError:
            bad += 1
            continue
        if len(sid.strip()) != 11 or not (-90 <= la <= 90) \
                or not (-180 <= lo <= 180):
            bad += 1
            continue
        ids.append(sid)
        lat.append(la)
        lon.append(lo)
        elev.append(el)
        meta.append((line[38:40].strip(), line[41:71].strip(),
                     line[72:75].strip(), line[76:79].strip(),
                     line[80:85].strip()))
    if not ids:
        raise ValueError("ghcnd-stations.txt parsed to ZERO stations")
    arr = np.array(ids, dtype="S11")
    order = np.argsort(arr, kind="stable")
    arr = arr[order]
    if np.unique(arr).size != arr.size:
        raise ValueError("ghcnd-stations.txt lists a station id twice")
    tab = {
        "id": arr,
        "lat": np.asarray(lat, np.float64)[order],
        "lon": np.asarray(lon, np.float64)[order],
        "elev": np.asarray(elev, np.float64)[order],
        "meta": [meta[i] for i in order],
        "bad_lines": bad,
    }
    tab["platform"] = np.array([f10b.platform_hash(s.decode()) for s in arr],
                               np.int64)
    return tab


# ------------------------------------------------------------- the parser --
def parse_block(buf, year):
    """Complete lines (ending in b"\\n") -> the kept element lines, as columns.

    Returns (cols, counts, date_max) where cols has `sid` (S11), `doy`,
    `el`, `val`, `vlen`, `qf` for the lines whose element is one of the five,
    and `date_max` is the largest YYYYMMDD in the block over ALL lines (the
    probe's stop rule reads it).
    """
    a = np.frombuffer(buf, np.uint8)
    nl = np.flatnonzero(a == 10)
    counts = {}
    if nl.size == 0:
        return None, counts, 0
    if nl[-1] != a.size - 1:
        raise FormatError("parse_block needs whole lines")
    starts = np.empty(nl.size, np.int64)
    starts[0] = 0
    starts[1:] = nl[:-1] + 1
    ends = nl.copy()
    cr = a[np.maximum(ends - 1, 0)] == 13
    ends[cr] -= 1
    blank = ends <= starts
    if blank.any():
        starts, ends = starts[~blank], ends[~blank]
    n = starts.size
    counts["lines"] = int(n)
    commas = np.flatnonzero(a == 44)
    if commas.size != 7 * n:
        raise FormatError(f"{commas.size} commas over {n} lines — expected "
                          f"exactly 7 per line")
    cm = commas.reshape(n, 7)
    ok = ((cm[:, 0] == starts + 11) & (cm[:, 1] == starts + 20)
          & (cm[:, 2] == starts + 25) & (cm[:, 6] < ends)
          & (cm[:, 0] > np.concatenate(([-1], ends[:-1]))))
    if not ok.all():
        i = int(np.flatnonzero(~ok)[0])
        raise FormatError(f"line {i} does not have the layout "
                          f"ID(11),YYYYMMDD,ELEM(4),VALUE,M,Q,S,OBSTIME: "
                          f"{bytes(a[starts[i]:ends[i]])!r}")
    # the date of every line (the stop rule wants the block's maximum)
    dig = a[starts[:, None] + 12 + np.arange(8)].astype(np.int64) - 48
    if (dig < 0).any() or (dig > 9).any():
        raise FormatError("a date field is not eight digits")
    ymd = dig @ (10 ** np.arange(7, -1, -1))
    date_max = int(ymd.max())
    # the element, as one integer
    eb = a[starts[:, None] + 21 + np.arange(4)].astype(np.int64)
    code = (eb[:, 0] << 24) | (eb[:, 1] << 16) | (eb[:, 2] << 8) | eb[:, 3]
    which = np.full(n, -1, np.int64)
    for k, c in enumerate(ELEM_CODES):
        which[code == c] = k
    keep = which >= 0
    counts["lines_other_element"] = int((~keep).sum())
    starts, cm, ymd, which = starts[keep], cm[keep], ymd[keep], which[keep]
    m = starts.size
    if m == 0:
        return None, counts, date_max
    y = ymd // 10000
    mo = (ymd // 100) % 100
    d = ymd % 100
    # day of year, with the month's real length (leap years included)
    mlen = np.array([calendar.monthrange(year, k)[1] if 1 <= k <= 12 else 0
                     for k in range(13)], np.int64)
    cum = np.concatenate(([0, 0], np.cumsum(mlen[1:])))[:13]
    good = (y == year) & (mo >= 1) & (mo <= 12)
    moc = np.clip(mo, 0, 12)
    good &= (d >= 1) & (d <= mlen[moc])
    counts["lines_bad_date"] = int((~good).sum())
    doy = cum[moc] + d - 1
    # the value: variable width, optional minus, right after comma 3
    v0 = starts + 26
    L = cm[:, 3] - v0
    K = 6
    if (L > K).any():
        raise FormatError("a VALUE field is longer than six characters")
    ar = np.arange(K)
    valid = ar[None, :] < L[:, None]
    ch = np.where(valid, a[np.minimum(v0[:, None] + ar, a.size - 1)],
                  48).astype(np.int64)
    neg = ch[:, 0] == 45
    ch[neg, 0] = 48
    dv = ch - 48
    if ((dv < 0) | (dv > 9))[valid].any():
        raise FormatError("a VALUE field is not an integer")
    pw = np.where(valid, L[:, None] - 1 - ar, 0)
    val = (dv * (10 ** pw) * valid).sum(axis=1)
    val = np.where(neg, -val, val)
    qf = (cm[:, 5] - cm[:, 4]) > 1
    sid = np.ascontiguousarray(
        a[starts[:, None] + np.arange(11)]).view("S11").ravel()
    cols = {"sid": sid[good], "doy": doy[good].astype(np.int16),
            "el": which[good].astype(np.int8),
            "val": val[good].astype(np.int32),
            "vlen": L[good].astype(np.int8), "qf": qf[good],
            "month": mo[good].astype(np.int8)}
    return cols, counts, date_max


def iter_blocks(read, block_bytes):
    """A compressed byte source -> decompressed blocks of WHOLE lines.

    `read(n)` returns compressed bytes (b"" at the end). The gzip stream is
    inflated incrementally and never written to disk; a stream that ends
    before its end-of-stream marker RAISES (a short download is a refusal).
    """
    z = zlib.decompressobj(31)
    pend = b""
    read_size = max(1, min(1 << 20, int(block_bytes)))
    while True:
        chunk = read(read_size)
        if not chunk:
            break
        pend += z.decompress(chunk)
        while z.eof and z.unused_data:
            # a multi-member gzip: start the next member
            rest = z.unused_data
            z = zlib.decompressobj(31)
            pend += z.decompress(rest)
        while len(pend) >= block_bytes:
            cut = pend.rfind(b"\n", 0, block_bytes) + 1
            if not cut:                       # one line longer than a block
                cut = pend.find(b"\n") + 1
                if not cut:
                    break
            yield pend[:cut]
            pend = pend[cut:]
    pend += z.flush()
    if not z.eof:
        raise IOError("the gzip stream ended before its end-of-stream marker "
                      "— a truncated download")
    if pend:
        if not pend.endswith(b"\n"):
            pend += b"\n"
        yield pend


# ================================================================ adapter ==
class GHCNDAdapter(f10b.SourceAdapter):
    store = "ghcnd"
    title = "GHCN-Daily station observations (NOAA NCEI), one row per station-day"
    family = "1tf"
    distribution = "public"
    licence = {"name": "CC0 1.0 (NOAA NCEI)", "redistribution": "yes",
               "derived_works": "free",
               "attribution": "Menne et al. (2012), GHCN-Daily v3, NOAA NCEI, "
                              "doi:10.7289/V5D21VHZ"}
    time_dtype = "int64"
    platform_meta = True
    credentials = ()
    channels = (("TMAX", "degC", -90.0, 65.0),
                ("TMIN", "degC", -90.0, 65.0),
                ("PRCP", "mm", 0.0, 2000.0),
                ("SNOW", "mm", 0.0, 2000.0),
                ("SNWD", "mm", 0.0, 15000.0))
    log2_fp = -4.0                                   # a point instrument
    log2_dt = float(np.log2(1.0 / 5.0))              # one day of a five-day bin
    per_year = True
    first_year = 1763
    qc_policy = (
        "qc 0: no used element (TMAX, TMIN, PRCP, SNOW, SNWD) of the "
        "station-day carried an NCEI quality flag (QFLAG); qc 1: at least one "
        "did, and that element's value is NaN (counted per element in "
        "`qflag_blanked`). A value outside its channel's bounds is NaN and "
        "counted (`out_of_bounds`); -9999 is NaN and counted "
        "(`value_missing`); MFLAG and SFLAG are ignored; a station-day whose "
        "five values are all NaN is dropped and counted (`rows_all_blanked`). "
        "Unlike family 10, qc 0 means 'passed the producer's QC', not 'not "
        "assessed'.")
    sources = (BY_YEAR + "YYYY.csv.gz", STATIONS_URL, BY_YEAR)
    verified = (
        "2026-09-17 from the sandbox: by_year listing (264 files, "
        "1763..2026); ghcnd-stations.txt (11,395,086 bytes); the whole "
        "2024.csv.gz (168,280,901 bytes) — 37,106,916 lines, 12,470,777 "
        "station-days, 44,668 stations, not sorted by station or strictly by "
        "date, 132,171 station-days split across non-adjacent lines")
    notes = (
        "time_s is the day at 00:00 UTC; OBSTIME is ignored. Positions are "
        "the station list's (one per station). The store is schema 3 "
        "(int64 time_s) because the record begins in 1763.")
    smoke_window = ("1899-12-25", "1900-01-06")
    smoke_probe_month = "1900-01"
    fetch_month_scope = "month"

    BLOCK_BYTES = 16 << 20          # decompressed bytes parsed at a time
    YIELD_ROWS = 1_000_000

    def __init__(self):
        self._stations = None
        self._listing = None

    # -------------------------------------------------------------- paths --
    def _local_root(self, ctx):
        return os.path.join(ctx.source_dir, "ghcnd")

    def _stations_path(self, ctx):
        return os.path.join(ctx.root, "src", "ghcnd-stations.txt")

    # ------------------------------------------------------------ listing --
    def listing(self, ctx):
        """{year: bytes} from the by_year directory listing. Never assumed."""
        if self._listing is not None:
            return self._listing
        if ctx.source_dir:
            p = os.path.join(self._local_root(ctx), "by_year", "index.html")
            with open(p, "rb") as fh:
                raw = fh.read()
            ctx.count_bytes(len(raw))
        else:
            raw = f10b.fetch_first([BY_YEAR], attempts=ctx.a.attempts)
            if raw is None:
                sys.exit(f"{BY_YEAR} answered 404 — the archive has moved")
        html = raw.decode("utf-8", "replace")
        sizes = {int(y): int(s) for y, s in LISTING_ROW.findall(html)}
        names = {int(y) for y in LISTING_NAME.findall(html)}
        if not names:
            sys.exit(f"REFUSING: the by_year listing names no YYYY.csv.gz "
                     f"file — an empty listing is a broken listing, not an "
                     f"empty archive ({len(raw)} bytes read)")
        if names != set(sizes):
            sys.exit(f"the by_year listing names {len(names)} year files but "
                     f"the size column parsed for {len(sizes)} — the page "
                     f"layout has changed; fix LISTING_ROW before building")
        self._listing = dict(sorted(sizes.items()))
        return self._listing

    def stations(self, ctx):
        if self._stations is not None:
            return self._stations
        p = self._stations_path(ctx)
        if ctx.source_dir:
            src = os.path.join(self._local_root(ctx), "ghcnd-stations.txt")
            with open(src, "rb") as fh:
                raw = fh.read()
            ctx.count_bytes(len(raw))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as fh:
                fh.write(raw)
        elif not os.path.exists(p):
            got = f10b.fetch_first([STATIONS_URL], path=p,
                                   attempts=ctx.a.attempts)
            if got is None:
                sys.exit(f"{STATIONS_URL} answered 404")
        with open(p, "rb") as fh:
            text = fh.read().decode("latin-1")
        self._stations = parse_stations(text)
        return self._stations

    # ------------------------------------------------------------- stream --
    def _open(self, ctx, year):
        """(read(n), close(), size-or-None) for one year's gzip."""
        if ctx.source_dir:
            p = os.path.join(self._local_root(ctx), "by_year",
                             f"{year}.csv.gz")
            fh = open(p, "rb")

            def read(n):
                b = fh.read(n)
                ctx.count_bytes(len(b))
                return b
            return read, fh.close, os.path.getsize(p)
        s = f10b.CountingStream(f"{BY_YEAR}{year}.csv.gz")
        return s.read, s.close, s.want

    def _parse_year(self, ctx, year, month=None):
        """Parse one year (or its first `month`) into kept columns + counts."""
        read, close, _size = self._open(ctx, year)
        cols, counts = [], {}
        stopped = False
        try:
            for blk in iter_blocks(read, self.BLOCK_BYTES):
                c, k, dmax = parse_block(blk, year)
                f10b._merge_counts(counts, k)
                if c is not None:
                    if month is not None:
                        sel = c["month"] == month
                        c = {kk: vv[sel] for kk, vv in c.items()}
                    cols.append(c)
                if month is not None and dmax // 100 > year * 100 + month \
                        and self._block_all_after(blk, year, month):
                    stopped = True
                    break
        finally:
            close()
        if month is not None:
            counts["probe_stopped_early"] = int(stopped)
        if not cols:
            return None, counts
        cat = {k: np.concatenate([c[k] for c in cols]) for k in cols[0]}
        return cat, counts

    @staticmethod
    def _block_all_after(blk, year, month):
        """Every line of the block is dated after `month` (the stop rule).

        By the measured layout of a year file (date-ordered apart from the
        Feb 28/29 seam), no line of an earlier month follows a block that
        lies wholly after it. The probe records whether it stopped.
        """
        a = np.frombuffer(blk, np.uint8)
        nl = np.flatnonzero(a == 10)
        starts = np.concatenate(([0], nl[:-1] + 1))
        starts = starts[nl > starts]
        dig = a[starts[:, None] + 12 + np.arange(6)].astype(np.int64) - 48
        ym = dig @ (10 ** np.arange(5, -1, -1))
        return bool((ym > year * 100 + month).all())

    def _group(self, ctx, year, cat, counts):
        """Kept element lines -> packed station-day rows (see the docstring).

        ONE stable argsort on ck = (station * 366 + day) * 5 + element, in
        file order: sorting ck sorts the station-day key too, a duplicated
        element is two equal neighbours, and the LAST of them in the file
        is the one kept (counted in `element_lines_duplicated`). `cat` is
        consumed — its columns are dropped as soon as they are used, so the
        peak is the columns plus two int64 vectors.
        """
        st = self.stations(ctx)
        sid = cat.pop("sid")
        pos = np.searchsorted(st["id"], sid)
        np.minimum(pos, st["id"].size - 1, out=pos)
        known = st["id"][pos] == sid
        n_unknown = int((~known).sum())
        counts["lines_station_unknown"] = n_unknown
        if n_unknown:
            counts["stations_unknown"] = int(np.unique(sid[~known]).size)
        del sid
        ck = pos * 366
        del pos
        ck += cat.pop("doy")
        ck *= 5
        ck += cat["el"]
        if n_unknown:
            ck = ck[known]
            for k in ("el", "val", "vlen", "qf"):
                cat[k] = cat[k][known]
        del known
        # SPLIT STATION-DAYS, measured in file order: a station-day that
        # appears in more than one run of consecutive kept lines — the reason
        # the year is grouped by sort rather than by run (module docstring).
        key_f = ck // 5
        runs = 1 + int(np.count_nonzero(np.diff(key_f))) if ck.size else 0
        del key_f
        order = np.argsort(ck, kind="stable")
        ck = ck[order]
        el = cat.pop("el")[order].astype(np.int64)
        val = cat.pop("val")[order]
        vlen = cat.pop("vlen")[order]
        qf = cat.pop("qf")[order]
        del order
        cat.clear()
        m = int(ck.size)
        # the LAST line of every (station, day, element) run survives
        last = np.ones(m, bool)
        if m:
            last[:-1] = ck[1:] != ck[:-1]
        counts["element_lines_duplicated"] = int(m - last.sum())
        if not last.all():
            ck, el, val, vlen, qf = (ck[last], el[last], val[last],
                                     vlen[last], qf[last])
        del last
        key = ck // 5
        del ck
        newkey = np.ones(key.size, bool)
        if key.size:
            newkey[1:] = key[1:] != key[:-1]
        uk = key[newkey]
        row = np.cumsum(newkey) - 1
        del key, newkey
        K = int(uk.size)
        counts["station_days_split"] = runs - K
        vals = np.full((K, 5), np.nan, np.float64)
        scaled = val * SCALE[el]
        missing = (val == MISSING) | (vlen == 0)
        del val, vlen
        scaled[missing] = np.nan
        blanked = qf & ~missing
        scaled[qf] = np.nan
        # (row, el) pairs are unique here — duplicates were resolved above
        vals[row, el] = scaled
        qcol = np.zeros(K, np.uint8)
        qcol[row[blanked]] = 1
        counts["value_missing"] = {ELEMENTS[k]: int(((el == k) & missing).sum())
                                   for k in range(5)
                                   if ((el == k) & missing).any()}
        counts["qflag_blanked"] = {ELEMENTS[k]: int(((el == k) & blanked).sum())
                                   for k in range(5)
                                   if ((el == k) & blanked).any()}
        counts["out_of_bounds"] = self.mask_bounds(vals)
        alive = np.isfinite(vals).any(axis=1)
        counts["rows_all_blanked"] = int((~alive).sum())
        s_of = (uk // 366)[alive]
        d_of = (uk % 366)[alive]
        vals, qcol = vals[alive], qcol[alive]
        day0 = f10b.seconds_since_epoch(dt.date(year, 1, 1))
        t = day0 + d_of * f10b.SECONDS_PER_DAY
        inside = (t >= ctx.t_lo) & (t <= ctx.t_hi)
        counts["rows_outside_window"] = int((~inside).sum())
        t, s_of, vals, qcol = t[inside], s_of[inside], vals[inside], qcol[inside]
        counts["station_days"] = int(t.size)
        counts["qc"] = {str(q): int((qcol == q).sum()) for q in (0, 1)
                        if (qcol == q).any()}
        return self.pack(t, st["lat"][s_of], st["lon"][s_of], vals,
                         st["platform"][s_of], qcol)

    def _batches(self, ctx, rows, counts, label):
        n = len(rows["bin"])
        if n == 0:
            yield label, None, counts
            return
        for i, lo in enumerate(range(0, n, self.YIELD_ROWS)):
            part = {k: v[lo:lo + self.YIELD_ROWS] for k, v in rows.items()}
            yield f"{label} rows {lo:,}+", part, (counts if i == 0 else None)

    # --------------------------------------------------------- the contract --
    def index(self, ctx):
        lst = self.listing(ctx)
        st = self.stations(ctx)
        want = [y for y in ctx.years if y in lst]
        missing = [y for y in ctx.years if y not in lst]
        first = None
        if want:
            read, close, _ = self._open(ctx, want[0])
            try:
                for blk in iter_blocks(read, 1 << 16):
                    line = blk.split(b"\n", 1)[0]
                    parse_block(line + b"\n", want[0])   # raises if malformed
                    first = line.decode("ascii", "replace")
                    break
            finally:
                close()
            if first is None:
                sys.exit(f"{want[0]}.csv.gz yielded no first line")
        return {
            "dataset": "GHCN-Daily by_year",
            "url": BY_YEAR,
            "listing": {"years": len(lst), "first": min(lst),
                        "last": max(lst)},
            "year_bytes": {str(y): lst[y] for y in want},
            "bytes": int(sum(lst[y] for y in want)),
            "years_not_listed": missing,
            "stations": int(st["id"].size),
            "stations_bad_lines": int(st["bad_lines"]),
            "first_record": first,
            "block_bytes": self.BLOCK_BYTES,
        }

    def fetch_year(self, ctx, year):
        lst = self.listing(ctx)
        if year not in lst:
            ctx.note_absent(year, f"{year}.csv.gz is not in the by_year "
                                  f"listing ({min(lst)}..{max(lst)})")
            return
        attempts = max(1, int(getattr(ctx.a, "attempts", 3) or 3))
        err = None
        for i in range(attempts):
            try:
                t0 = time.time()
                cat, counts = self._parse_year(ctx, year)
                break
            except (IOError, EOFError, zlib.error) as e:
                err = f"{type(e).__name__}: {e}"
                print(f"  ::warning::ghcnd {year}: attempt {i + 1}/{attempts} "
                      f"failed — {err}", flush=True)
                if i < attempts - 1:
                    time.sleep(5.0 * (2 ** i))
        else:
            ctx.note_absent(year, f"{year}.csv.gz could not be read: {err}")
            return
        if cat is None:
            ctx.note_absent(year, f"{year}.csv.gz is listed and holds no "
                                  f"TMAX/TMIN/PRCP/SNOW/SNWD line")
            return
        counts["parse_seconds"] = round(time.time() - t0, 1)
        rows = self._group(ctx, year, cat, counts)
        yield from self._batches(ctx, rows, counts, str(year))

    def fetch_month(self, ctx, year, month):
        """The probe: stream the year file only until a whole block lies
        after `month` (see `_block_all_after`), keep that month's lines."""
        lst = self.listing(ctx)
        if year not in lst:
            sys.exit(f"{year}.csv.gz is not in the by_year listing")
        cat, counts = self._parse_year(ctx, year, month=month)
        if cat is None:
            yield f"{year}-{month:02d}", None, counts
            return
        rows = self._group(ctx, year, cat, counts)
        yield from self._batches(ctx, rows, counts, f"{year}-{month:02d}")

    def platforms(self, ctx):
        st = self.stations(ctx)
        out = {}
        for i in range(st["id"].size):
            state, name, gsn, hcn, wmo = st["meta"][i]
            el = float(st["elev"][i])
            out[int(st["platform"][i])] = {
                "id": st["id"][i].decode(),
                "lat": float(st["lat"][i]), "lon": float(st["lon"][i]),
                "elev_m": None if el <= -999.0 else el,
                "name": name, "state": state or None, "gsn": gsn or None,
                "hcn_crn": hcn or None, "wmo": wmo or None}
        return out

    # ---------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260917):
        return make_smoke_sources(root, d_lo, d_hi, seed)


# ================================================================== smoke ==
# SORTED BY ID, and that is load-bearing: every station reports at the same
# second (00:00 UTC), so the store's order inside a day is its tie-break —
# the adapter emits a year in (station index, day) order and both assemblers
# keep that order among equal (bin, time_s). Generating the truth in the same
# station order lets `check_smoke` compare row for row.
SMOKE_STATIONS = (
    # id, lat, lon, elev, name
    ("ASN00000002", -33.9000, 151.2000, 5.0, "SMOKE SYDNEY"),
    ("GMM00000003", 52.5000, 13.4000, -999.9, "SMOKE BERLIN"),
    ("RSM00000004", 64.7500, 177.5000, 12.0, "SMOKE ANADYR"),
    ("USW00000001", 40.5000, -105.2500, 1500.0, "SMOKE BOULDER"),
)
assert list(SMOKE_STATIONS) == sorted(SMOKE_STATIONS)


def _station_line(sid, lat, lon, elev, name):
    return (f"{sid:<11} {lat:8.4f} {lon:9.4f} {elev:6.1f}    "
            f"{name:<30}                 ")


def _listing_html(sizes):
    rows = "".join(
        f'<tr>\n<td><a href="{y}.csv.gz">{y}.csv.gz</a></td>\n'
        f'<td align="right">2026-09-16 19:28</td>\n'
        f'<td align="right">{s}</td>\n<td> </td>\n</tr>\n'
        for y, s in sorted(sizes.items()))
    return ("<!DOCTYPE html>\n<html><body><h1>Index of /pub/data/ghcn/daily/"
            "by_year/</h1>\n<table><tbody>\n<tr><td><a href=\"..\">Parent "
            "Directory</a></td></tr>\n" + rows +
            '<tr>\n<td><a href="readme-by_year.txt">readme-by_year.txt</a>'
            '</td>\n<td align="right">2026-09-16 19:28</td>\n'
            '<td align="right">1086</td>\n</tr>\n</tbody></table></body>'
            '</html>\n')


def make_smoke_sources(root, d_lo, d_hi, seed=20260917):
    """A two-year GHCN-Daily archive in the real on-disk format.

    Deliberately hostile: lines SHUFFLED within each year (station-days split
    across the file, as 132,171 are in the real 2024 file), an element the
    store does not keep (TAVG), a QFLAG, a value outside bounds, a -9999, a
    station-day whose only element is flagged (dropped), a station missing
    from the station list, an invalid date (1900-02-29), and a day before the
    window. Returns the truth rows `check_smoke` compares against.
    """
    rng = np.random.default_rng(seed)
    base = os.path.join(root, "ghcnd")
    os.makedirs(os.path.join(base, "by_year"), exist_ok=True)
    with open(os.path.join(base, "ghcnd-stations.txt"), "w") as fh:
        for s in SMOKE_STATIONS:
            fh.write(_station_line(*s) + "\n")
    pos = {s[0]: (s[1], s[2]) for s in SMOKE_STATIONS}
    lines = {}
    truth = []
    day = d_lo - dt.timedelta(days=1)             # one day BEFORE the window
    while day <= d_hi:
        for sid, _la, _lo, _el, _nm in SMOKE_STATIONS:
            vals = [np.nan] * 5
            qfl = False
            text = []
            for k, e in enumerate(ELEMENTS):
                if rng.random() < 0.3:
                    continue
                if k < 2:
                    raw = int(rng.integers(-300, 350))
                elif k == 2:
                    raw = int(rng.integers(0, 500))
                elif k == 3:
                    raw = int(rng.integers(0, 300))
                else:
                    raw = int(rng.integers(0, 15001))
                flag = ""
                text.append((e, raw, flag))
                vals[k] = raw * SCALE[k]
            if rng.random() < 0.5:
                text.append(("TAVG", int(rng.integers(-100, 200)), ""))
            if not text:
                continue
            # the specials, on fixed station-days
            key = (sid, day)
            if key == ("USW00000001", dt.date(1900, 1, 2)):
                text = [(e, r, f) for e, r, f in text if e != "TMAX"]
                text.append(("TMAX", 123, "I"))          # flagged -> NaN, qc 1
                vals[0] = np.nan
                qfl = True
            if key == ("ASN00000002", dt.date(1900, 1, 3)):
                text = [(e, r, f) for e, r, f in text if e != "TMIN"]
                text.append(("TMIN", 700, ""))           # 70.0 degC: OOB -> NaN
                vals[1] = np.nan
            if key == ("GMM00000003", dt.date(1900, 1, 4)):
                text = [(e, r, f) for e, r, f in text if e != "PRCP"]
                text.append(("PRCP", -9999, ""))         # missing -> NaN
                vals[2] = np.nan
            if key == ("RSM00000004", dt.date(1900, 1, 5)):
                text = [("TMAX", 55, "X")]               # only element flagged
                vals = [np.nan] * 5
            lines.setdefault(day.year, []).extend(
                f"{sid},{day:%Y%m%d},{e},{r},,{f},S,0700" for e, r, f in text)
            t = f10b.seconds_since_epoch(day)
            if d_lo <= day <= d_hi and np.isfinite(vals).any():
                truth.append({"t": t, "lat": pos[sid][0], "lon": pos[sid][1],
                              "platform": f10b.platform_hash(sid),
                              "v": vals, "qc": int(qfl)})
        day += dt.timedelta(days=1)
    lines.setdefault(d_hi.year, []).extend([
        f"XXX00000009,{d_hi:%Y%m%d},TMAX,100,,,S,",      # unknown station
        f"USW00000001,{d_hi.year}0229,TMAX,100,,,S," if not
        calendar.isleap(d_hi.year) else
        f"USW00000001,{d_hi.year}0230,TMAX,100,,,S,",    # not a real date
    ])
    sizes = {}
    for y, ls in lines.items():
        order = rng.permutation(len(ls))
        body = "".join(ls[i] + "\n" for i in order).encode()
        p = os.path.join(base, "by_year", f"{y}.csv.gz")
        with open(p, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
                gz.write(body)
        sizes[y] = os.path.getsize(p)
    with open(os.path.join(base, "by_year", "index.html"), "w") as fh:
        fh.write(_listing_html(sizes))
    return truth


ADAPTER = GHCNDAdapter
