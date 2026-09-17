"""IGRA v2.2 radiosondes — one row per sounding, 80 channels (family 1.0.tf).

PLAIN ENGLISH. The Integrated Global Radiosonde Archive is NOAA NCEI's
collection of weather-balloon soundings: about 2,900 launch sites, some from
1905, most launching twice a day. Each sounding measures temperature,
humidity and wind from the ground to the stratosphere. This adapter keeps the
16 "mandatory" pressure levels every station reports (1000 hPa near the
ground up to 10 hPa at about 31 km), five quantities at each — so one row is
one balloon flight, and a model can read the vertical structure of the
atmosphere above the nearest launch sites.

SOURCE, VERIFIED 2026-09-17 FROM THIS SANDBOX (no account, US government):
  https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/
  .../access/data-por/<ID>-data.txt.zip        (one zip per station)
  .../doc/igra2-station-list.txt  .../doc/igra2-data-format.txt
The data-por listing is an Apache table of 2,931 zips with EXACT byte sizes
(27,870,184,241 bytes in all), each holding one member `<ID>-data.txt`; the
largest measured, USM00072201, is a 101,678,874-byte zip of a 313,027,164-byte
text. data-por is refreshed daily (last modified 2026-09-15) and so already
holds the current year: data-y2d/ (807 `<ID>-data-beg2026.txt.zip`) is a
subset and is not read. Throughput from NCEI: 58 MB/s on one stream.

THE FILE LAYOUT (igra2-data-format.txt, and measured on 12 random full
stations: every header line exactly 71 characters, every data line exactly 52,
header dates never decreasing). A sounding is one header line
  #ID YEAR MM DD HH RELTIME NUMLEV P_SRC NP_SRC LAT LON
followed by NUMLEV fixed-width level lines
  LVLTYP1 LVLTYP2 ETIME PRESS(Pa) PFLAG GPH(m) ZFLAG TEMP(0.1C) TFLAG RH DPDP(0.1C) WDIR WSPD(0.1 m/s)
with -9999 = missing and -8888 = removed by IGRA's quality assurance. The
parser is VECTORISED over 16 MiB blocks of decompressed text cut at sounding
boundaries, and a block whose last sounding is before the window is skipped
without being parsed.

WHAT A ROW IS.
  time_s    the LAUNCH time: RELTIME (HHMM, UTC) on the header's date, and when
            RELTIME has only an hour (HH99) that hour at :00. RELTIME is
            written on the date of the NOMINAL observation, so a 00 UTC
            sounding released at 23:04 carries the NEXT day's date (measured:
            "USM00072201 2026 09 15 00 2304"); when the release time is more
            than 12 h after the nominal hour the date is moved back a day,
            more than 12 h before, forward a day (`release_day_shifted`). When
            RELTIME is missing (9999) the NOMINAL hour HH is used
            (`time_from_nominal_hour` — COMMON: 26,666 of USM00072201's 69,360
            soundings, and 3,119,661 of the 5,921,445 soundings the 2020-01
            probe parsed); a nominal hour of 99 with RELTIME HH99 takes the
            release hour (`time_release_hour_only`, 23 at USM00072201); with
            neither the sounding has no time and is dropped
            (`soundings_no_time`, 0 in both measurements). int64 seconds since
            1982-01-01 (schema 3: the record starts in 1905).
  lat, lon  the header's LAT/LON (1e-4 degrees): the station's fixed position,
            or a ship's position at that sounding (102 mobile stations).
  platform  platform_hash(station ID); platforms.json maps it back with the
            station list's position, elevation, name and years.
  values    C = 80 = five quantities x 16 mandatory levels, VARIABLE-MAJOR:
            T_<p> (degC), DPD_<p> (dew-point depression, degC), U_<p>, V_<p>
            (m/s, from WDIR/WSPD: u = -s sin(d), v = -s cos(d); NaN if either
            is missing) and Z_<p> (geopotential height, m), for p = 1000, 925,
            850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10
            hPa. A level is taken where PRESS equals p exactly; if a sounding
            carries p twice (a surface level at 1000 hPa and the standard
            level, say) the LVLTYP1 = 1 standard level wins, else the later
            line (`levels_duplicated`). NaN where the sounding lacks the level.
            Measured 2026-09-17 on three whole stations (GMM00010868,
            ASM00094120, USM00072520; 2,484,760 lines at an exact mandatory
            pressure): EVERY such line is LVLTYP1 = 1 — IGRA marks the
            surface (LVLTYP2 = 1) and the tropopause (LVLTYP2 = 2) in the
            second digit — so in practice the level is always a
            mandatory-level record and the fallback above never fires.
  qc        0 when no kept value was removed by IGRA's QA; 1 when at least one
            was (-8888, counted per quantity in `qa_removed`). The A/B
            climatology flags are not used.
A sounding with none of the 80 values (a pilot-balloon or height-only
sounding) is dropped (`soundings_no_mandatory_level`).

STREAMING, NOT PER YEAR. The archive is per STATION and each zip is
chronological, so `per_year = False`: `fetch_stream` streams each station's
zip once, keeps the --start/--end window, and yields (year, rows) per station
— re-reading every station for every year would multiply 27.9 GB by the 122
years of the record. Stations whose station-list FSTYEAR..LSTYEAR does not
overlap the window are not requested (`stations_skipped_by_years`); every
station that IS streamed has its soundings' years checked against that range
(`stations_outside_listed_years`, which should be 0), and a station missing
from the list — measured: the list has one line with a blank ID and name
(1946-2025, 70,410 soundings), and ICM00004018 is in the listing but not the
list — is always streamed. A station stream stops as soon as a block begins
after the window (`stations_stopped_early`).

THE PROBE, 2020-01, MEASURED 2026-09-17: 979 stations requested (1,952
skipped by their listed years), 22,448,547,000 bytes of zip streamed in
1,249 s (18 MB/s over four workers; 460 s of CPU), 671 stations abandoned
after January, 28,094,092 earlier soundings skipped unparsed and 5,921,445
parsed; 41,921 January soundings kept from 865 stations (3,371 had no
mandatory level; qc 1 on 1,093; no value out of bounds; no station outside
its listed years; no failure). NaN fractions run from 0.03 (U_700) to 0.73
(DPD_10); V_1000 is 0.52 NaN because 1000 hPa is below high stations. A full
parse runs at 65 MB/s of text per core (USM00072201: 313 MB in 4.8 s). So:
≈ 0.5 M soundings a year now; the whole archive is 27.87 GB of zip ≈ 86 GB
of text, ≈ 10-30 min of download at 18-58 MB/s plus ≈ 22 core-minutes of
parsing, and ≈ 39-48 M rows (52.2 M listed soundings x the 74 % kept at
USM00072201 .. 93 % kept in the probe) ≈ 7.4-9.2 GB stored (the note: 10 GB).
The month probe cannot be cheap for a per-station archive: its bytes are the
stations' whole history up to the month.

MEMORY. Per station: one 16 MiB text block and its parsed columns, plus the
station's packed window rows (≈ 200 bytes a sounding). Per build: the
framework's `PartWriter` holds every year's rows until the year reaches 1 M,
and no IGRA year does — so a build holds its whole window in memory, ≈ 190
bytes x soundings. The whole archive is ≈ 52.2 M soundings (the station list's
NOBS) ≈ 9.9 GB, more than a hosted runner should hold, so `fetch_preflight`
REFUSES a window whose projected soundings (NOBS pro rata by years) exceed
`MAX_WINDOW_SOUNDINGS` and asks for --start/--end lanes.
"""
import datetime as dt
import os
import re
import sys
import time
import zlib

import numpy as np

import build_family10_stores as f10b
from family1.adapters import _common as cm

BASE = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/"
POR = BASE + "access/data-por/"
STATIONS_URL = BASE + "doc/igra2-station-list.txt"
LEVELS = (1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30,
          20, 10)
NLEV = len(LEVELS)
QUANTITIES = (("T", "degC", -100.0, 60.0),
              ("DPD", "degC", 0.0, 80.0),
              ("U", "m/s", -150.0, 150.0),
              ("V", "m/s", -150.0, 150.0),
              ("Z", "m", -1000.0, 40000.0))
CHANNELS = tuple((f"{q}_{p}", u, lo, hi) for q, u, lo, hi in QUANTITIES
                 for p in LEVELS)
MISSING, REMOVED = -9999, -8888
LISTING_ROW = re.compile(
    r'href="([A-Z0-9]{11})-data\.txt\.zip"[^<]*</a>\s*</td>\s*'
    r'<td[^>]*>[^<]*</td>\s*<td[^>]*>\s*(\d+)\s*</td>', re.S)
LISTING_NAME = re.compile(r'href="([A-Z0-9]{11})-data\.txt\.zip"')
HEAD_LEN, DATA_LEN = 71, 52
LEVEL_OF = {p * 100: i for i, p in enumerate(LEVELS)}
PA = np.array(sorted(LEVEL_OF), np.int64)
PA_LEV = np.array([LEVEL_OF[x] for x in PA], np.int64)


class FormatError(ValueError):
    """IGRA text that does not have the documented fixed-width layout."""


def parse_listing(html):
    names = LISTING_NAME.findall(html)
    rows = LISTING_ROW.findall(html)
    if not names:
        raise FormatError("the data-por listing names no <ID>-data.txt.zip — "
                          "an empty listing is a broken listing")
    if len(rows) != len(names):
        raise FormatError(f"the data-por listing names {len(names)} zips but "
                          f"the size column parsed for {len(rows)}")
    return {sid: int(sz) for sid, sz in rows}


def parse_station_list(text):
    """{ID: {...}}, plus the lines that do not parse (kept, and counted)."""
    out, bad = {}, []
    for line in text.splitlines():
        if not line.strip():
            continue
        sid = line[0:11].strip()
        try:
            lat = float(line[12:20])
            lon = float(line[21:30])
            elev = float(line[31:37])
            fy, ly, nobs = int(line[72:76]), int(line[77:81]), \
                int(line[82:88])
        except ValueError:
            bad.append(line.rstrip())
            continue
        if len(sid) != 11:
            bad.append(line.rstrip())
            continue
        mobile = lat < -90 or lon < -180
        out[sid] = {"id": sid, "lat": None if mobile else lat,
                    "lon": None if mobile else lon,
                    "elev_m": None if elev <= -999 else elev,
                    "state": line[38:40].strip() or None,
                    "name": line[41:71].strip(), "first_year": fy,
                    "last_year": ly, "nobs": nobs, "mobile": mobile}
    if not out:
        raise FormatError("igra2-station-list.txt parsed to ZERO stations")
    return out, bad


def _day_seconds(y, m, d):
    return cm.days_from_civil(y, m, d) * 86400


def parse_block(buf, t_lo, t_hi, counts):
    """Whole soundings of text -> (t, lat, lon, values (S, 80), qc).

    `buf` starts with a header line and ends with a newline.
    """
    a = np.frombuffer(buf, np.uint8)
    nl = np.flatnonzero(a == 10)
    if nl.size == 0:
        return None
    starts = np.empty(nl.size, np.int64)
    starts[0] = 0
    starts[1:] = nl[:-1] + 1
    ends = nl.copy()
    cr = a[np.maximum(ends - 1, 0)] == 13
    ends[cr] -= 1
    keep = ends > starts
    starts, ends = starts[keep], ends[keep]
    L = ends - starts
    head = a[starts] == 35
    if not head[0]:
        raise FormatError("a block does not begin with a header line")
    if (L[head] != HEAD_LEN).any():
        i = int(np.flatnonzero(head & (L != HEAD_LEN))[0])
        raise FormatError(f"a header line of {L[i]} characters (expected "
                          f"{HEAD_LEN}): {bytes(a[starts[i]:ends[i]])!r}")
    if (L[~head] != DATA_LEN).any():
        i = int(np.flatnonzero(~head & (L != DATA_LEN))[0])
        raise FormatError(f"a data line of {L[i]} characters (expected "
                          f"{DATA_LEN}): {bytes(a[starts[i]:ends[i]])!r}")
    hs = starts[head]
    S = hs.size
    snd = np.cumsum(head) - 1
    ds = starts[~head]
    dsnd = snd[~head]
    counts["soundings_read"] = counts.get("soundings_read", 0) + int(S)
    # ---- headers
    fi = cm.fixed_int
    yr = fi(a, hs, 13, 17, "YEAR")[0]
    mo = fi(a, hs, 18, 20, "MONTH")[0]
    dy = fi(a, hs, 21, 23, "DAY")[0]
    hr = fi(a, hs, 24, 26, "HOUR")[0]
    rt = fi(a, hs, 27, 31, "RELTIME")[0]
    numlev = fi(a, hs, 32, 36, "NUMLEV")[0]
    lat = fi(a, hs, 55, 62, "LAT")[0] / 1e4
    lon = fi(a, hs, 63, 71, "LON")[0] / 1e4
    got = np.bincount(dsnd, minlength=S)
    if not np.array_equal(got, numlev):
        i = int(np.flatnonzero(got != numlev)[0])
        raise FormatError(f"sounding {bytes(a[hs[i]:hs[i] + HEAD_LEN])!r} "
                          f"says NUMLEV {numlev[i]} and has {got[i]} lines")
    date_ok = cm.valid_date(yr, mo, dy)
    day0 = _day_seconds(np.where(date_ok, yr, 2000), np.where(date_ok, mo, 1),
                        np.where(date_ok, dy, 1))
    hr_ok = (hr >= 0) & (hr <= 23)
    rh, rm = rt // 100, rt % 100
    rel_ok = (rt != 9999) & (rt >= 0) & (rh <= 23) & ((rm <= 59) | (rm == 99))
    rel = day0 + rh * 3600 + np.where(rm == 99, 0, rm) * 60
    nom = day0 + hr * 3600
    diff = rel - nom
    back = rel_ok & hr_ok & (diff > 12 * 3600)
    fwd = rel_ok & hr_ok & (diff < -12 * 3600)
    rel = rel - back * 86400 + fwd * 86400
    t = np.where(rel_ok, rel, nom)
    has_t = date_ok & (rel_ok | hr_ok)
    counts["release_day_shifted"] = counts.get("release_day_shifted", 0) + \
        int((date_ok & (back | fwd)).sum())
    counts["time_from_nominal_hour"] = counts.get(
        "time_from_nominal_hour", 0) + int((date_ok & ~rel_ok & hr_ok).sum())
    counts["time_release_hour_only"] = counts.get(
        "time_release_hour_only", 0) + int((date_ok & rel_ok &
                                            (rm == 99)).sum())
    counts["soundings_no_time"] = counts.get("soundings_no_time", 0) + \
        int((~has_t).sum())
    inside = has_t & (t >= t_lo) & (t <= t_hi)
    counts["soundings_outside_window"] = counts.get(
        "soundings_outside_window", 0) + int((has_t & ~inside).sum())
    pos_ok = (np.abs(lat) <= 90) & (np.abs(lon) <= 360)
    counts["soundings_no_position"] = counts.get("soundings_no_position", 0) \
        + int((inside & ~pos_ok).sum())
    inside &= pos_ok
    # ---- the mandatory levels
    vals = np.full((S, 5 * NLEV), np.nan, np.float64)
    qc = np.zeros(S, np.uint8)
    if ds.size:
        press = fi(a, ds, 9, 15, "PRESS")[0]
        j = np.searchsorted(PA, press)
        j = np.minimum(j, PA.size - 1)
        hit = (PA[j] == press) & inside[dsnd]
        idx = np.flatnonzero(hit)
    else:
        idx = np.zeros(0, np.int64)
    if idx.size:
        lrow = ds[idx]
        ls = dsnd[idx]
        lev = PA_LEV[j[idx]]
        typ1 = a[lrow].astype(np.int64) - 48
        # one level per (sounding, level): the standard level (LVLTYP1 == 1)
        # wins, else the later line
        key = (ls * NLEV + lev) * 2 + (typ1 == 1)
        order = np.argsort(key, kind="stable")
        k2 = key[order] // 2
        last = np.ones(order.size, bool)
        last[:-1] = k2[1:] != k2[:-1]
        counts["levels_duplicated"] = counts.get("levels_duplicated", 0) + \
            int((~last).sum())
        sel = order[last]
        lrow, ls, lev = lrow[sel], ls[sel], lev[sel]
        raw = {"T": fi(a, lrow, 22, 27, "TEMP")[0],
               "DPD": fi(a, lrow, 34, 39, "DPDP")[0],
               "WDIR": fi(a, lrow, 40, 45, "WDIR")[0],
               "WSPD": fi(a, lrow, 46, 51, "WSPD")[0],
               "Z": fi(a, lrow, 16, 21, "GPH")[0]}
        miss = counts.setdefault("value_missing", {})
        qa = counts.setdefault("qa_removed", {})
        flagged = np.zeros(lrow.size, bool)
        for nm, v in raw.items():
            m, r = v == MISSING, v == REMOVED
            if m.any():
                miss[nm] = miss.get(nm, 0) + int(m.sum())
            if r.any():
                qa[nm] = qa.get(nm, 0) + int(r.sum())
            flagged |= r
        np.maximum.at(qc, ls, flagged.astype(np.uint8))

        def val(nm, scale):
            v = raw[nm].astype(np.float64) * scale
            v[(raw[nm] == MISSING) | (raw[nm] == REMOVED)] = np.nan
            return v
        T, D, Z = val("T", 0.1), val("DPD", 0.1), val("Z", 1.0)
        wd, ws = val("WDIR", 1.0), val("WSPD", 0.1)
        rad = np.deg2rad(wd)
        U, V = -ws * np.sin(rad), -ws * np.cos(rad)
        for q, arr in enumerate((T, D, U, V, Z)):
            vals[ls, q * NLEV + lev] = arr
    alive = inside & np.isfinite(vals).any(axis=1)
    counts["soundings_no_mandatory_level"] = counts.get(
        "soundings_no_mandatory_level", 0) + int((inside & ~alive).sum())
    return (t[alive], lat[alive], lon[alive], vals[alive], qc[alive],
            yr[alive])


def last_header_date(buf):
    """(y, m, d) of the last header line in a block, as an int YYYYMMDD."""
    i = buf.rfind(b"\n#")
    i = 0 if i < 0 else i + 1
    h = buf[i:i + HEAD_LEN]
    return int(h[13:17]) * 10000 + int(h[18:20]) * 100 + int(h[21:23])


def first_header_date(buf):
    h = buf[:HEAD_LEN]
    return int(h[13:17]) * 10000 + int(h[18:20]) * 100 + int(h[21:23])


def iter_sounding_blocks(chunks, block_bytes):
    """Decompressed chunks -> blocks of WHOLE soundings (cut before a '#')."""
    pend = b""
    for c in chunks:
        pend += c
        while len(pend) >= block_bytes:
            cut = pend.rfind(b"\n#", 0, block_bytes) + 1
            if cut <= 0:
                cut = pend.find(b"\n#", block_bytes) + 1
                if cut <= 0:
                    break
            yield pend[:cut]
            pend = pend[cut:]
    if pend:
        if not pend.endswith(b"\n"):
            pend += b"\n"
        yield pend


# ================================================================ adapter ==
class IGRAAdapter(f10b.SourceAdapter):
    store = "igra"
    title = ("IGRA v2.2 radiosonde soundings (NOAA NCEI), one row per "
             "sounding at the 16 mandatory levels")
    family = "1tf"
    distribution = "public"
    licence = {"name": "US government work (NOAA NCEI), no restriction",
               "redistribution": "yes", "derived_works": "free",
               "attribution": "Durre, I., X. Yin, R. S. Vose, S. Applequist, "
                              "J. Arnfield (2016), IGRA version 2, NOAA NCEI, "
                              "doi:10.7289/V5X63K0Q"}
    time_dtype = "int64"
    platform_meta = True
    credentials = ()
    channels = CHANNELS
    log2_fp = -4.0
    log2_dt = float(np.log2((1.0 / 24.0) / 5.0))   # a balloon flight ~ 1-2 h
    per_year = False
    first_year = 1905
    qc_policy = (
        "qc 0: no kept value of the sounding was removed by IGRA's quality "
        "assurance; qc 1: at least one was (-8888), and that value is NaN "
        "(counted per quantity in `qa_removed`). -9999 is NaN and counted "
        "(`value_missing`); a value outside its channel's bounds is NaN and "
        "counted (`out_of_bounds`); the A/B climatology flags are ignored; a "
        "sounding with no value at any mandatory level is dropped "
        "(`soundings_no_mandatory_level`).")
    sources = (POR + "<ID>-data.txt.zip", STATIONS_URL, POR)
    verified = (
        "2026-09-17 from the sandbox: data-por listing (2,931 zips, "
        "27,870,184,241 B); igra2-station-list.txt (2,932 lines, one with a "
        "blank ID); igra2-data-format.txt; USM00072201 whole (101,678,874 B "
        "zip, 69,360 soundings, headers 71 / data 52 characters, dates "
        "monotone) and 12 random whole stations (same layout, all "
        "monotone); 58 MB/s")
    notes = (
        "time_s is the launch time (RELTIME), moved across midnight when the "
        "release is more than 12 h from the nominal hour; the nominal hour "
        "when RELTIME is missing. The stream skips stations whose station-list "
        "years miss the window and stops a station at the first block past "
        "it. Schema 3 (int64 time_s): the record begins in 1905.")
    smoke_window = ("1905-12-30", "1906-01-03")
    smoke_probe_month = "1906-01"
    fetch_month_scope = "month"

    BLOCK_BYTES = 16 << 20
    WORKERS = 4
    # 30 M soundings x ~190 B ≈ 5.7 GB held by PartWriter: the hosted runner's
    # 16 GB with room for the interpreter, parse buffers and four workers
    MAX_WINDOW_SOUNDINGS = 30_000_000

    def __init__(self):
        self._listing = None
        self._stations = None

    # ------------------------------------------------------------ sources --
    def _local(self, ctx, *p):
        return os.path.join(ctx.source_dir, "igra", *p)

    def listing(self, ctx):
        if self._listing is None:
            if ctx.source_dir:
                with open(self._local(ctx, "data-por", "index.html"),
                          "rb") as fh:
                    raw = fh.read()
                ctx.count_bytes(len(raw))
            else:
                raw, why = cm.get_bytes(POR, attempts=ctx.a.attempts)
                if raw is None:
                    sys.exit(f"{POR} answered 404 — the archive has moved")
            try:
                self._listing = parse_listing(raw.decode("utf-8", "replace"))
            except FormatError as e:
                sys.exit(f"REFUSING: {e} ({len(raw)} bytes read)")
        return self._listing

    def stations(self, ctx):
        if self._stations is None:
            if ctx.source_dir:
                with open(self._local(ctx, "igra2-station-list.txt"),
                          "rb") as fh:
                    raw = fh.read()
                ctx.count_bytes(len(raw))
            else:
                raw, _ = cm.get_bytes(STATIONS_URL, attempts=ctx.a.attempts)
                if raw is None:
                    sys.exit(f"{STATIONS_URL} answered 404")
            tab, bad = parse_station_list(raw.decode("latin-1"))
            for sid, e in tab.items():
                e["platform"] = f10b.platform_hash(sid)
            self._stations = (tab, bad)
        return self._stations

    def _open(self, ctx, sid):
        """(chunk iterator, close) over a station's decompressed text."""
        if ctx.source_dir:
            fh = open(self._local(ctx, "data-por", f"{sid}-data.txt.zip"),
                      "rb")

            class _R:
                def read(self_, n):
                    b = fh.read(n)
                    ctx.count_bytes(len(b))
                    return b

                def close(self_):
                    fh.close()
            r = _R()
        else:
            r = f10b.CountingStream(f"{POR}{sid}-data.txt.zip")
        return f10b._zip_member_stream(r), r.close

    def select(self, ctx, y_lo, y_hi):
        """The stations to stream for years [y_lo, y_hi], and the counts."""
        lst = self.listing(ctx)
        tab, _ = self.stations(ctx)
        todo, skipped, unlisted = [], 0, 0
        for sid in sorted(lst):
            e = tab.get(sid)
            if e is None:
                unlisted += 1
                todo.append(sid)
            elif e["first_year"] <= y_hi and e["last_year"] >= y_lo:
                todo.append(sid)
            else:
                skipped += 1
        return todo, {"stations_listed": len(lst),
                      "stations_skipped_by_years": skipped,
                      "stations_not_in_station_list": unlisted,
                      "stations_in_list_without_zip": len(set(tab) -
                                                          set(lst))}

    # ------------------------------------------------------------ station --
    def _station(self, ctx, sid, t_lo, t_hi):
        """Stream one station -> (sid, {year: packed rows}, counts, error)."""
        tab, _ = self.stations(ctx)
        e = tab.get(sid)
        attempts = max(1, int(getattr(ctx.a, "attempts", 3) or 3))
        lo_day = (dt.date(1982, 1, 1) + dt.timedelta(
            days=int(t_lo // 86400))) if abs(t_lo) < 10**11 else None
        hi_day = (dt.date(1982, 1, 1) + dt.timedelta(
            days=int(t_hi // 86400))) if abs(t_hi) < 10**11 else None
        lo_key = int(lo_day.strftime("%Y%m%d")) - 1 if lo_day else -1
        hi_key = int(hi_day.strftime("%Y%m%d")) + 1 if hi_day else 10**9
        err = None
        for i in range(attempts):
            counts = {"stations_streamed": 1}
            parts = []
            try:
                chunks, close = self._open(ctx, sid)
                stopped = False
                try:
                    prev_last = None
                    for blk in iter_sounding_blocks(chunks, self.BLOCK_BYTES):
                        fd, ld = first_header_date(blk), last_header_date(blk)
                        if prev_last is not None and fd < prev_last:
                            counts["blocks_out_of_order"] = counts.get(
                                "blocks_out_of_order", 0) + 1
                        prev_last = ld
                        if ld < lo_key:
                            counts["soundings_before_window_unparsed"] = \
                                counts.get("soundings_before_window_unparsed",
                                           0) + blk.count(b"\n#") + \
                                (blk[:1] == b"#")
                            continue
                        if fd > hi_key and not counts.get(
                                "blocks_out_of_order"):
                            stopped = True
                            break
                        out = parse_block(blk, t_lo, t_hi, counts)
                        if out is not None and len(out[0]):
                            parts.append(out)
                finally:
                    close()
                counts["stations_stopped_early"] = int(stopped)
                break
            except FormatError as ex:
                raise FormatError(f"{sid}: {ex}") from None
            except (IOError, EOFError, zlib.error, ValueError) as ex:
                err = f"{sid}: {type(ex).__name__}: {ex}"
                if i < attempts - 1:
                    time.sleep(5.0 * (2 ** i))
        else:
            return sid, None, {"stations_failed": 1}, err
        by_year = {}
        if parts:
            t = np.concatenate([p[0] for p in parts])
            la = np.concatenate([p[1] for p in parts])
            lo = np.concatenate([p[2] for p in parts])
            v = np.concatenate([p[3] for p in parts])
            q = np.concatenate([p[4] for p in parts])
            hy = np.concatenate([p[5] for p in parts])
            oob = self.mask_bounds(v)
            if oob:
                f10b._merge_counts(counts, {"out_of_bounds": oob})
            alive = np.isfinite(v).any(axis=1)
            counts["soundings_no_mandatory_level"] = counts.get(
                "soundings_no_mandatory_level", 0) + int((~alive).sum())
            t, la, lo, v, q, hy = (t[alive], la[alive], lo[alive], v[alive],
                                   q[alive], hy[alive])
            if e is not None and hy.size and (
                    hy.min() < e["first_year"] or hy.max() > e["last_year"]):
                counts["stations_outside_listed_years"] = 1
            # the year is the TIME's year (a 23:04 release belongs to the day
            # before its header date)
            ty = (np.datetime64("1982-01-01", "s")
                  + t.astype("timedelta64[s]")).astype("datetime64[Y]") \
                .astype(np.int64) + 1970
            plat = f10b.platform_hash(sid)
            for y in np.unique(ty).tolist():
                m = ty == y
                n = int(m.sum())
                by_year[int(y)] = self.pack(
                    t[m], la[m], lo[m], v[m], np.full(n, plat, np.int64),
                    q[m])
                counts["soundings_kept"] = counts.get("soundings_kept", 0) + n
                qc = counts.setdefault("qc", {})
                for k in (0, 1):
                    c = int((q[m] == k).sum())
                    if c:
                        qc[str(k)] = qc.get(str(k), 0) + c
        return sid, by_year, counts, None

    def _stream(self, ctx, t_lo, t_hi, y_lo, y_hi):
        todo, head = self.select(ctx, y_lo, y_hi)
        counts = dict(head)
        workers = 1 if ctx.source_dir else self.WORKERS
        t0 = time.time()
        fn = (lambda sid: self._station(ctx, sid, t_lo, t_hi))
        for i, (sid, by_year, c, err) in enumerate(
                cm.ordered_map(fn, todo, workers), 1):
            f10b._merge_counts(counts, c)
            if err:
                ctx.note_absent(str(y_lo), f"{sid}-data.txt.zip could not be "
                                           f"read: {err}")
                counts.setdefault("stations_failed_ids", []).append(sid)
                continue
            for y in sorted(by_year):
                yield y, by_year[y], None
            if i % 100 == 0:
                ctx.prog.item(f"igra stations {i}/{len(todo)}", None,
                              {"soundings_kept":
                               counts.get("soundings_kept", 0),
                               "elapsed_s": round(time.time() - t0, 1)})
        counts["stream_seconds"] = round(time.time() - t0, 1)
        yield None, None, counts

    # --------------------------------------------------------- the contract --
    def index(self, ctx):
        lst = self.listing(ctx)
        tab, bad = self.stations(ctx)
        todo, sel = self.select(ctx, ctx.d_lo.year, ctx.d_hi.year)
        first = None
        if todo:
            sid = min(todo, key=lambda s: lst[s])
            chunks, close = self._open(ctx, sid)
            try:
                for blk in iter_sounding_blocks(chunks, 1 << 16):
                    one = blk.split(b"\n#", 1)[0] + b"\n"
                    c = {}
                    parse_block(one, -2**62, 2**62, c)   # raises if malformed
                    first = {"station": sid,
                             "header": one.split(b"\n", 1)[0].decode()}
                    break
            finally:
                close()
        return {
            "dataset": "IGRA v2.2 data-por",
            "url": POR,
            "listing": {"zips": len(lst), "bytes": int(sum(lst.values()))},
            "window_stations": len(todo),
            "window_bytes": int(sum(lst[s] for s in todo)),
            "selection": sel,
            "station_list_entries": len(tab),
            "station_list_bad_lines": bad,
            "projected_soundings": self.projected(ctx),
            "first_record": first,
            "levels_hpa": list(LEVELS),
            "quantities": [q[0] for q in QUANTITIES],
        }

    def projected(self, ctx):
        """Soundings in the window by the station list's NOBS, pro rata."""
        tab, _ = self.stations(ctx)
        lo, hi = ctx.d_lo.year, ctx.d_hi.year
        tot = 0.0
        for e in tab.values():
            span = e["last_year"] - e["first_year"] + 1
            ov = min(hi, e["last_year"]) - max(lo, e["first_year"]) + 1
            if span > 0 and ov > 0:
                tot += e["nobs"] * ov / span
        return int(tot)

    def fetch_preflight(self, ctx):
        if getattr(ctx.a, "parts_from_hub", False):
            # the guard is about holding a window's rows while STREAMING the
            # source; a box or hosted assembly only pulls parked year parts
            # (measured 2026-09-17, run #40 refused the whole-record pull)
            return
        n = self.projected(ctx)
        if n > self.MAX_WINDOW_SOUNDINGS:
            sys.exit(
                f"REFUSING igra {ctx.d_lo}..{ctx.d_hi}: the station list "
                f"projects {n:,} soundings in this window, and the framework "
                f"holds a year's rows in memory until the year reaches 1 M "
                f"rows (no IGRA year does) — ≈ {n * 190 / 1e9:.1f} GB. Run "
                f"the fetch in --start/--end lanes of at most "
                f"{self.MAX_WINDOW_SOUNDINGS:,} projected soundings "
                f"(≈ 30 years of the modern record) and --push-parts each.")

    def fetch_stream(self, ctx):
        yield from self._stream(ctx, ctx.t_lo, ctx.t_hi, ctx.d_lo.year,
                                ctx.d_hi.year)

    def fetch_month(self, ctx, year, month):
        """The probe: stations the station list calls active in `year`, each
        streamed only until its first block after the month."""
        lo, hi = f10b.month_bounds_s(year, month)
        lo, hi = max(lo, ctx.t_lo), min(hi, ctx.t_hi)
        parts, counts = [], {}
        for y, rows, c in self._stream(ctx, lo, hi, year, year):
            if y is None:
                counts = c
                continue
            parts.append(rows)
        rows = cm.concat_rows(parts, self.C, self.time_dtype)
        yield from cm.batch_rows(self, rows, counts, f"{year}-{month:02d}")

    def platforms(self, ctx):
        tab, _ = self.stations(ctx)
        out = {}
        for sid, e in tab.items():
            out[int(e["platform"])] = {
                "id": sid, "lat": e["lat"], "lon": e["lon"],
                "elev_m": e["elev_m"], "name": e["name"],
                "state": e["state"], "mobile": e["mobile"],
                "first_year": e["first_year"], "last_year": e["last_year"],
                "nobs": e["nobs"]}
        # a station in the listing and not in the list (measured:
        # ICM00004018) still gets an entry, with the id and nothing else
        for sid in self.listing(ctx):
            h = f10b.platform_hash(sid)
            if h not in out:
                out[h] = {"id": sid, "lat": None, "lon": None,
                          "note": "listed in data-por, absent from "
                                  "igra2-station-list.txt"}
        return out

    # ---------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260917):
        return make_smoke_sources(root, d_lo, d_hi, seed)


# ================================================================== smoke ==
def _header(sid, y, m, d, hh, rt, n, lat, lon, psrc="ncdc-gts"):
    s = (f"#{sid:<11} {y:4d} {m:02d} {d:02d} {hh:02d} {rt:04d} {n:4d} "
         f"{psrc:<8} {'':<8} {int(round(lat * 1e4)):7d} "
         f"{int(round(lon * 1e4)):8d}")
    assert len(s) == HEAD_LEN, (len(s), s)
    return s


def _level_line(t1, t2, press, gph, temp, dpd, wdir, wspd, et=-9999,
                rh=-9999, pf=" ", zf=" ", tf=" "):
    s = (f"{t1}{t2} {et:5d} {press:6d}{pf}{gph:5d}{zf}{temp:5d}{tf}{rh:5d} "
         f"{dpd:5d} {wdir:5d} {wspd:5d} ")
    assert len(s) == DATA_LEN, (len(s), s)
    return s


def make_smoke_sources(root, d_lo, d_hi, seed=20260917):
    """A three-station IGRA archive in the real layout (zips, listing, list).

    Hostile on purpose: a release at 23:xx dated the next day, a release with
    only an hour (HH99), a sounding with no release time (nominal hour used),
    one with neither (dropped), a pilot-balloon sounding with no pressure
    levels (dropped), a -8888 QA removal (qc 1), -9999s, a duplicated 1000 hPa
    level (standard level wins), an out-of-bounds temperature, a mobile ship
    station, a station whose station-list years miss the window (not
    requested), a station absent from the list (streamed anyway), and
    soundings before the window. Returns the truth rows.
    """
    import zipfile
    rng = np.random.default_rng(seed)
    base = os.path.join(root, "igra")
    os.makedirs(os.path.join(base, "data-por"), exist_ok=True)
    y0 = d_lo.year
    stations = [
        # id, lat, lon, elev, name, fy, ly, mobile
        ("GMM00010868", 48.24, 11.55, 489.0, "SMOKE MUNICH", y0 - 1, y0 + 1,
         False),
        ("USM00072201", 24.5531, -81.7886, 1.0, "SMOKE KEY WEST", y0 - 1,
         y0 + 1, False),
        ("ZZV00SHIP01", -98.8888, -998.8888, -999.9, "SMOKE SHIP", y0 - 1,
         y0 + 1, True),
        ("ZZM00000009", 10.0, 10.0, 5.0, "SMOKE OLD", y0 - 40, y0 - 30,
         False),
    ]
    unlisted = "ICM00004018"
    with open(os.path.join(base, "igra2-station-list.txt"), "w") as fh:
        for sid, la, lo, el, nm, fy, ly, mob in stations:
            fh.write(f"{sid:<11} {la:8.4f} {lo:9.4f} {el:6.1f}    "
                     f"{nm:<30} {fy:4d} {ly:4d} {100:6d}\n")
        fh.write(" " * 72 + f"{y0 - 50} {y0 + 1}  70410\n")   # the real quirk
    truth = []
    texts = {}
    t_lo = f10b.seconds_since_epoch(d_lo)
    t_hi = f10b.seconds_since_epoch(d_hi) + 86399

    def sounding(sid, lat, lon, when_hdr, hh, rt, launch, levels, keep,
                 qc=0):
        """levels = [(t1, t2, pa, gph, temp, dpd, wdir, wspd)]"""
        lines = [_header(sid, when_hdr.year, when_hdr.month, when_hdr.day,
                         hh, rt, len(levels), lat, lon)]
        v = [np.nan] * (5 * NLEV)
        for (t1, t2, pa, gph, tp, dp, wd, ws) in levels:
            lines.append(_level_line(t1, t2, pa, gph, tp, dp, wd, ws))
            if pa in LEVEL_OF and (t1 == 1 or not any(
                    x[2] == pa and x[0] == 1 for x in levels)):
                k = LEVEL_OF[pa]
                def f(x, s):
                    return np.nan if x in (MISSING, REMOVED) else x * s
                tt = f(tp, 0.1)
                v[k] = np.nan if (tt == tt and not (-100 <= tt <= 60)) \
                    else tt
                v[NLEV + k] = f(dp, 0.1)
                if wd in (MISSING, REMOVED) or ws in (MISSING, REMOVED):
                    v[2 * NLEV + k] = v[3 * NLEV + k] = np.nan
                else:
                    r = np.deg2rad(wd)
                    v[2 * NLEV + k] = -ws * 0.1 * np.sin(r)
                    v[3 * NLEV + k] = -ws * 0.1 * np.cos(r)
                v[4 * NLEV + k] = f(gph, 1.0)
        texts.setdefault(sid, []).extend(lines)
        if keep and launch is not None:
            t = f10b.seconds_since_epoch(launch)
            if t_lo <= t <= t_hi and np.isfinite(v).any():
                truth.append({"t": t, "lat": lat,
                              "lon": float(f10b.f10.wrap_lon(lon)),
                              "platform": f10b.platform_hash(sid), "v": v,
                              "qc": qc})

    def std_levels(n_levels=None):
        out = []
        z = 100
        for p in LEVELS[:n_levels]:
            out.append((1, 0, p * 100, z, int(rng.integers(-600, 300)),
                        int(rng.integers(0, 300)), int(rng.integers(0, 360)),
                        int(rng.integers(0, 400))))
            z += int(rng.integers(300, 2000))      # 10 hPa < 40 km
        return out

    day = d_lo - dt.timedelta(days=2)
    while day <= d_hi:
        for sid, la, lo, el, nm, fy, ly, mob in stations[:3] + [
                (unlisted, 64.0, -22.6, 50.0, "", 0, 0, False)]:
            if mob:
                la, lo = round(-30.0 + day.day * 0.5, 4), \
                    round(150.0 + day.day, 4)
            for hh in (0, 12):
                nominal = dt.datetime.combine(day, dt.time(hh))
                # an ordinary launch 55 minutes before the nominal hour: for
                # 00 UTC that is 23:05 of the PREVIOUS day, dated this day
                launch = nominal - dt.timedelta(minutes=55)
                rt = launch.hour * 100 + launch.minute
                levs = std_levels()
                kind = (sid, day, hh)
                qc = 0
                if kind == (stations[0][0], d_lo, 12):
                    rt = 1199                       # hour only -> 11:00
                    launch = nominal.replace(hour=11)
                elif kind == (stations[0][0], d_hi, 0):
                    rt = 9999                       # nominal hour used
                    launch = nominal
                elif kind == (stations[1][0], d_lo, 0):
                    hh_hdr = 99                     # neither: dropped
                    sounding(sid, la, lo, day, hh_hdr, 9999, None, levs,
                             False)
                    continue
                elif kind == (stations[1][0], d_lo, 12):
                    # a pilot balloon: height levels only -> dropped
                    sounding(sid, la, lo, day, hh, rt, launch,
                             [(3, 0, -9999, 500, -9999, -9999, 90, 50),
                              (3, 0, -9999, 1000, -9999, -9999, 95, 70)],
                             True)
                    continue
                elif kind == (stations[1][0], d_hi, 12):
                    levs[3] = levs[3][:4] + (REMOVED,) + levs[3][5:]
                    levs[5] = levs[5][:5] + (MISSING,) + levs[5][6:]
                    qc = 1
                elif kind == (stations[2][0], d_lo, 12):
                    levs[0] = levs[0][:4] + (700,) + levs[0][5:]  # 70.0 C
                    # a surface level at 1000 hPa BEFORE the standard one
                    levs.insert(0, (2, 1, 100000, 10, 250, 20, 180, 30))
                    # and a significant level between two standard ones
                    levs.insert(3, (2, 0, 88000, 1200, 150, 30, 200, 60))
                sounding(sid, la, lo, day, hh, rt, launch, levs, True, qc)
        day += dt.timedelta(days=1)
    # the old station: data years outside the window (never requested)
    sounding(stations[3][0], 10.0, 10.0, dt.date(y0 - 35, 6, 1), 0, 2300,
             None, std_levels(3), False)
    sizes = {}
    for sid, lines in sorted(texts.items()):
        p = os.path.join(base, "data-por", f"{sid}-data.txt.zip")
        with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"{sid}-data.txt", "\n".join(lines) + "\n")
        sizes[sid] = os.path.getsize(p)
    rows = "".join(
        f'<tr>\n<td><a href="{s}-data.txt.zip">{s}-data.txt.zip</a></td>\n'
        f'<td align="right">2026-09-15 21:37</td>\n<td align="right">{n}'
        f'</td>\n<td> </td>\n</tr>\n' for s, n in sorted(sizes.items()))
    with open(os.path.join(base, "data-por", "index.html"), "w") as fh:
        fh.write("<html><body><h1>Index of /data/integrated-global-radiosonde"
                 "-archive/access/data-por/</h1>\n<table><tbody>\n<tr><td><a "
                 "href=\"..\">Parent Directory</a></td></tr>\n" + rows +
                 "</tbody></table></body></html>\n")
    # the truth in the store's tie order: station order (sorted ids), then
    # the station's own order
    order = {s: i for i, s in enumerate(sorted(texts))}
    truth.sort(key=lambda r: order[next(s for s in texts if
                                        f10b.platform_hash(s) ==
                                        r["platform"])])
    return truth


ADAPTER = IGRAAdapter
