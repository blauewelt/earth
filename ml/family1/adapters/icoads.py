"""ICOADS marine reports, IMMA1 — one row per ship/buoy report (family 1.gf).

PLAIN ENGLISH. The International Comprehensive Ocean-Atmosphere Data Set is
the archive of marine weather reports: ships' logbooks back to 1662, buoys,
platforms and research vessels, each report a position, a time and whatever
the observer measured — sea temperature, air temperature, pressure, wind,
dew point, cloud, waves. This adapter turns NCEI's IMMA1 copy of it into a
tier-P store, one row per report with eight channels, so a model can ask
"what did the ships near here report in the last few days" for 360 years.

SOURCE, VERIFIED 2026-09-17 FROM THIS SANDBOX (NOAA NCEI, no account):
  https://www.ncei.noaa.gov/data/international-comprehensive-ocean-atmosphere/
      v3/archive/final-untrim/IMMA1_R3.1.0_<YYYY>-<MM>.gz          (<= 2014)
      v3/archive/nrt/monthly/icoads-nrt_r<rel>_final_d<YYYYMM>_c<YYYYMMDD>.dat.gz
      v3/doc/R3.0-imma1_short.pdf                    (the format, read here)
  * `https://www.ncei.noaa.gov/data/global-marine/` is NOT ICOADS IMMA1: it
    is NCEI's "Global Marine Data" CSV product (per-month directories of
    10-degree-box CSVs in US units — inches of mercury, deg F). Not used.
  * final-untrim/ is an Apache listing (532,801 bytes) of 3,232 monthly
    gzip files, 1662-12 .. 2014-12, 44.97 GB by its EXACT byte column (the
    note's 45 GB). The files are named `IMMA1_R3.1.0_...` although the README
    beside them says R3.0.0 (the "R3.0.1"/3.1 correction of 2019); the name is
    read from the listing, never assumed. One file per month; many early
    months are absent (1662-12, then 1663-01..05, then 1677-04 ...) — the
    listing decides which months exist.
  * nrt/monthly/ (791,849 bytes) mixes products. Only
    `icoads-nrt_r<rel>_final_d<YYYYMM>_c<date>.dat.gz` is read ("final" =
    duplicate-eliminated; "total" keeps the worse duplicates, "dupinventory",
    "lastuid", "outputsummary" are bookkeeping, the `.nc` files are the same
    reports in netCDF). r3.0.2 `final` is listed for 2015-01 .. 2025-07 plus
    2026-07 and 2026-08 (131 files, 44.93 GB), r3.0.3 for 2025-08 .. 2026-08
    (13 files, 4.74 GB): 2026-07 and 2026-08 are in BOTH releases, and two
    r3.0.2 months (202403, 202409) are listed twice with different creation
    dates (the Readme.Date_Notes says c20241023 was replaced by c20241027).
    RULE: per month the highest release wins, then the newest creation date
    (`nrt_files_superseded` = 4 today). 140 NRT months result: 127 r3.0.2,
    13 r3.0.3, 48.13 GB. A month in both final-untrim and nrt is read from
    final-untrim (none today).
  * The files stream: `Accept-Ranges: bytes`, exact Content-Length; a body
    that ends short RAISES (`f10b.CountingStream`).

THE RECORD (IMMA1, Tables C0 and C1 of R3.0-imma1_short.pdf, read 2026-09-17;
fields are right-aligned, blank = missing, NOT zero-filled). Core, 108 chars:
YR 0-4, MO 4-6, DY 6-8, HR 8-12 (0.01 h), LAT 12-17 (0.01 degN), LON 17-23
(0.01 degE, 0..359.99), IM 23-25, ATTC 25, TI 26, LI 27, DS 28, VS 29,
NID 30-32, II 32-34, ID 34-43 (9 chars), C1 43-45 | DI 45, D 46-49 (deg,
361 calm, 362 variable), WI 49, W 50-53 (0.1 m/s), VI 53, VV 54-56, WW 56-58,
W1 58, SLP 59-64 (0.1 hPa), A 64, PPP 65-68, IT 68, AT 69-73 (0.1 degC),
WBTI 73, WBT 74-78, DPTI 78, DPT 79-83 (0.1 degC), SI 83-85, SST 85-89
(0.1 degC), N 89 (oktas, 9 = obscured), NH 90, CL 91, HI 92, H 93, CM 94,
CH 95, WD 96-98, WP 98-100, WH 100-102 (wind-wave height in HALF METRES),
SD, SP, SH 102-108. Then the attachments, each starting ATTI(2) ATTL(2).
The Icoads attm (ATTI 1, ATTL 65) was the FIRST attachment of every one of
300,002 records checked in 2010-07 (and of every record read from 1662-12,
1880-01, 2015-01 and 2026-08): at 108: " 165", BSI 112, B10 113-116, B1
116-118, DCK 118-121, SID 121-124, PT 124-126, DUPS 126-128, DUPC 128,
TC 129, PB 130, WX 131, SX 132, C2 133-135, 12 adaptive-QC flags 135-147,
ND 147, 6 trimming flags 148-154 (SF AF UF VF PF RF), 14 NCDC-QC flags
154-168 (ZNC WNC BNC XNC YNC PNC ANC GNC DNC SNC CNC ENC FNC TNC; base36,
1 correct, 2-3 correctable, 4-6 suspect, 7-9 erroneous, A = 10 missing),
QCE 168-170, LZ 170, QCZ 171-173. A record whose Icoads attm is not at 108
is refused and counted (`reports_no_icoads_attm`). Only the first 173 bytes
of a line are read. Measured on 2010-07 (500,000 records): II 3 (WMO buoy
number) 77 %, 5 (C-MAN) 13 %, 1 (callsign) 8 %; DUPS only 0 or 1 (final
products drop DUPS > 2 — kept as a refusal here, `reports_worse_duplicate`);
LZ always blank; 7 blank IDs; ZNC 7 on 0.17 %; ENC 7 on 3.8 %; CNC 8 on
9.4 %. In 1880-01, 4,325 of 14,233 IDs are blank.

WHAT A ROW IS.
  time_s    YR MO DY + HR (0.01 h = 36 s, exact), int64 seconds since
            1982-01-01 (schema 3: the record starts in 1662). A report with
            no day or no hour is dropped (`reports_no_day`, `reports_no_hour`),
            never placed at an invented hour.
  lat, lon  LAT/100, LON/100 wrapped to [-180, 180); missing -> dropped
            (`reports_no_position`); ZNC (position flag) 7-9 -> dropped
            (`reports_position_erroneous`).
  platform  platform_hash(ID stripped) — the callsign / buoy number exactly as
            the record spells it. FALLBACK when ID is blank:
            platform_hash("icoads-deck-<DCK>-sid-<SID>"), one pooled platform
            per (deck, source) — the anonymous logbooks of one collection
            (`reports_blank_id`). Generic IDs (II = 2: "SHIP", "BUOY", masked
            callsigns) are hashed as spelled, so they pool too
            (`reports_generic_id` counts them).
  values    C = 8, the note's order: sst (SST), airt (AT), slp (SLP),
            wind_u, wind_v (from D and W: u = -W sin D, v = -W cos D; D 361 =
            calm -> 0, 0; D 362 = variable -> NaN, `wind_variable_direction`;
            one of D, W missing -> NaN, `wind_incomplete`), dewpt — the core
            carries the DEW-POINT temperature (DPT), not specific humidity —,
            wave_h (WH x 0.5 m, the core's wind-wave height; swell not kept),
            cloud (N, oktas; 9 "sky obscured" -> NaN, `cloud_obscured`).
  qc        per value, its NCDC-QC flag (SNC ANC PNC WNC WNC DNC ENC CNC):
            7-9 (erroneous) -> NaN, counted per channel (`ncdc_erroneous`).
            The row's qc is the worst class among its KEPT values: 0 all
            flagged 1 (correct), 1 some 2-3 (correctable), 2 some not
            assessed (blank or A = missing flag), 3 some 4-6 (suspect).
            The trimming flags (SF..RF) are NOT used: their decoding is in
            R3.0-stat_trim.pdf, which answered HTTP 301 from here and was not
            read (§4: nothing from memory).
A report with none of the eight values is dropped (`reports_no_value`);
LZ = 1 (landlocked) is dropped (`reports_landlocked`).

THE PROBE, 2010-07, MEASURED 2026-09-17 (`--stage probe --probe-month
2010-07`, ml/family1/probes/icoads_2010-07.json): one file,
IMMA1_R3.1.0_2010-07.gz, 138,827,309 bytes streamed and parsed in 15 s
(peak RSS 362 MB for the whole process); 2,108,040 lines -> 2,019,801 rows
from 7,540 platforms (84,700 reports with no value, 3,496 with ZNC 7-9, 43
with no hour, 41 blank IDs, 6,069 generic IDs); 68.7 bytes of gzip a row.
NaN fractions sst 0.18, airt 0.61, slp 0.41, wind 0.62, dewpt 0.84, wave_h
0.88, cloud 0.96. NCDC-QC erroneous values removed: wave_h 76,416 (ENC 7 is
common on buoy wave reports), dewpt 2,502, sst 1,765, airt 1,420, slp 438,
wind 215; out of bounds sst 12, wave_h 3, wind 1 (the other component of
that vector blanked with it, `wind_half_blanked`). qc: 0 1,984,676, 1 30,560,
3 4,565.
WHOLE ARCHIVE, extrapolated from the listings' exact bytes and gzip bytes per
line measured on 16 whole months (1860-06 .. 2025-06, 47-77 B a line): about
716 M lines to 2014 and 710 M in the NRT final product 2015-2026 (≈ 62 M a
year since 2016), i.e. ≈ 1.37 B rows at the 2010-07 keep rate (0.958) and ≈
64 GB stored at 47 B a row. The note's "455 M reports to 2014" is LOW by
about half and its "≈ 1e8 a year" since 2015 HIGH by ~1.6x; its 1.5 B rows /
75 GB total holds. At the probe's 8.7 MB/s the 93 GB of gzip is ≈ 3 h of
streaming on one core.

MEMORY. One month is streamed and never written to disk: gzip is decoded as
it arrives and parsed in blocks of 250,000 lines (a (250k x 173) byte matrix,
43 MB, plus ~25 int64 field arrays, ~50 MB); each block is packed and yielded
at once (≈ 57 B a row), so the peak is ≈ 150 MB whatever the month's size.
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

HOST = "https://www.ncei.noaa.gov/data/international-comprehensive-ocean-atmosphere/v3/"
UNTRIM = HOST + "archive/final-untrim/"
NRT = HOST + "archive/nrt/monthly/"
DOC = HOST + "doc/R3.0-imma1_short.pdf"

ROW_RE = re.compile(r'<a href="([^"]+)">[^<]*</a></td>\s*<td[^>]*>([^<]*)</td>'
                    r'\s*<td[^>]*>\s*([0-9]+|-)\s*</td>')
UNTRIM_NAME = re.compile(r"^IMMA1_R(\d+)\.(\d+)\.(\d+)_(\d{4})-(\d{2})\.gz$")
NRT_NAME = re.compile(r"^icoads-nrt_r(\d+)\.(\d+)\.(\d+)_final_d(\d{4})(\d{2})"
                      r"_c(\d{8})\.dat\.gz$")

W = 173                       # core (108) + the Icoads attm (65)
BLOCK_LINES = 250_000

CHANNELS = (("sst", "degC", -3.0, 40.0),
            ("airt", "degC", -80.0, 60.0),
            ("slp", "hPa", 870.0, 1090.0),
            ("wind_u", "m/s", -80.0, 80.0),
            ("wind_v", "m/s", -80.0, 80.0),
            ("dewpt", "degC", -80.0, 40.0),
            ("wave_h", "m", 0.0, 30.0),
            ("cloud", "okta", 0.0, 8.0))

# (name, start, end) in the 173-byte prefix
F = {"YR": (0, 4), "MO": (4, 6), "DY": (6, 8), "HR": (8, 12),
     "LAT": (12, 17), "LON": (17, 23), "II": (32, 34),
     "D": (46, 49), "W": (50, 53), "SLP": (59, 64), "AT": (69, 73),
     "DPT": (79, 83), "SST": (85, 89), "N": (89, 90), "WH": (100, 102),
     "ATTI": (108, 110), "ATTL": (110, 112), "DCK": (118, 121),
     "SID": (121, 124), "DUPS": (126, 128), "LZ": (170, 171)}
ID_SPAN = (34, 43)
NCDC0 = 154
NCDC = ("ZNC", "WNC", "BNC", "XNC", "YNC", "PNC", "ANC", "GNC", "DNC", "SNC",
        "CNC", "ENC", "FNC", "TNC")
# the NCDC-QC flag that governs each channel
FLAG_OF = ("SNC", "ANC", "PNC", "WNC", "WNC", "DNC", "ENC", "CNC")


class FormatError(ValueError):
    """A listing or a record that is not the layout verified above."""


# ================================================================ listing ==
def parse_rows(html):
    """[(name, size_bytes)] from an NCEI Apache listing (exact byte column)."""
    names = re.findall(r'<a href="([^"?/][^"]*)">', html)
    rows = [(n, int(s)) for n, _, s in ROW_RE.findall(html) if s != "-"]
    files = [n for n in names if not n.endswith("/") and n != ".."]
    if len(rows) != len(files):
        raise FormatError(f"the listing names {len(files)} files but the "
                          f"size column parsed for {len(rows)} — the page "
                          f"layout has changed; fix ROW_RE")
    return rows


def parse_untrim(html):
    """{(y, m): {...}} from final-untrim/. Refuses an empty listing."""
    out = {}
    for n, sz in parse_rows(html):
        m = UNTRIM_NAME.match(n)
        if not m:
            continue
        k = (int(m.group(4)), int(m.group(5)))
        if k in out:
            raise FormatError(f"final-untrim lists {k} twice ({n}, "
                              f"{out[k]['name']})")
        out[k] = {"name": n, "url": UNTRIM + n, "bytes": sz,
                  "product": "final-untrim",
                  "release": ".".join(m.group(i) for i in (1, 2, 3))}
    if not out:
        raise FormatError("final-untrim lists no IMMA1_R*_YYYY-MM.gz file — "
                          "an empty listing is a broken listing")
    return out


def parse_nrt(html):
    """{(y, m): {...}} from nrt/monthly/: the `final` product, highest
    release then newest creation date per month. Also returns how many
    listed files lost to a newer one."""
    cand = {}
    for n, sz in parse_rows(html):
        m = NRT_NAME.match(n)
        if not m:
            continue
        rel = tuple(int(m.group(i)) for i in (1, 2, 3))
        k = (int(m.group(4)), int(m.group(5)))
        cand.setdefault(k, []).append((rel, m.group(6), n, sz))
    if not cand:
        raise FormatError("nrt/monthly lists no icoads-nrt_r*_final_d*.dat.gz "
                          "file — an empty listing is a broken listing")
    out, lost = {}, 0
    for k, c in cand.items():
        c.sort()
        rel, cdate, n, sz = c[-1]
        lost += len(c) - 1
        out[k] = {"name": n, "url": NRT + n, "bytes": sz, "product": "nrt",
                  "release": ".".join(map(str, rel)), "created": cdate,
                  "superseded": [x[2] for x in c[:-1]]}
    return out, lost


# ================================================================ parsing ==
def fw_int(A, lo, hi):
    """Right-aligned decimal fields of a (n, W) uint8 matrix -> (int64, missing,
    malformed). Blank = missing; a character other than digit/space/minus is
    malformed (and missing)."""
    ch = A[:, lo:hi].astype(np.int64)
    blank = ch == 32
    neg = ch == 45
    dig = (ch >= 48) & (ch <= 57)
    bad = ~(blank | neg | dig).all(axis=1) | (neg.sum(axis=1) > 1)
    val = np.where(dig, ch - 48, 0)
    right = np.cumsum(dig[:, ::-1], axis=1)[:, ::-1] - dig
    out = (val * (10 ** right)).sum(axis=1)
    out = np.where(neg.any(axis=1), -out, out)
    miss = ~dig.any(axis=1) | bad
    return out, miss, bad


def b36_flag(col):
    """One base36 flag column -> int (0 = blank)."""
    c = col.astype(np.int64)
    v = np.zeros(c.shape, np.int64)
    d = (c >= 48) & (c <= 57)
    u = (c >= 65) & (c <= 90)
    v[d] = c[d] - 48
    v[u] = c[u] - 55
    return v


def _add(counts, key, n):
    if n:
        counts[key] = counts.get(key, 0) + int(n)


def parse_block(lines, counts):
    """Raw IMMA1 lines (bytes) -> column arrays; every drop counted.

    Returns (t, lat, lon, vals (n, 8), plat_text (n,) S-array, dck, sid, qc).
    """
    n0 = len(lines)
    _add(counts, "lines", n0)
    buf = b"".join(ln[:W].rstrip(b"\r\n").ljust(W) for ln in lines)
    A = np.frombuffer(buf, np.uint8).reshape(n0, W)
    fld, miss = {}, {}
    for k, (a, b) in F.items():
        v, m, bad = fw_int(A, a, b)
        fld[k], miss[k] = v, m
        if bad.any():
            c = counts.setdefault("fields_malformed", {})
            c[k] = c.get(k, 0) + int(bad.sum())
    keep = np.ones(n0, bool)

    def drop(mask, key):
        nonlocal keep
        m = keep & mask
        _add(counts, key, m.sum())
        keep &= ~mask

    drop((fld["ATTI"] != 1) | miss["ATTI"] | (fld["ATTL"] != 65),
         "reports_no_icoads_attm")
    yr, mo, dy = fld["YR"], fld["MO"], fld["DY"]
    drop(miss["YR"] | miss["MO"], "reports_no_month")
    drop(miss["DY"], "reports_no_day")
    drop(~cm.valid_date(yr, mo, np.where(miss["DY"], 1, dy)), "reports_bad_date")
    drop(miss["HR"], "reports_no_hour")
    drop((fld["HR"] < 0) | (fld["HR"] > 2399), "reports_bad_hour")
    drop(miss["LAT"] | miss["LON"], "reports_no_position")
    drop((np.abs(fld["LAT"]) > 9000) | (fld["LON"] < -17999)
         | (fld["LON"] > 35999), "reports_bad_position")
    flags = {nm: b36_flag(A[:, NCDC0 + i]) for i, nm in enumerate(NCDC)}
    # ZNC 7-9 is erroneous; A (10) is "no flag", not an error
    drop((flags["ZNC"] >= 7) & (flags["ZNC"] <= 9),
         "reports_position_erroneous")
    drop(~miss["DUPS"] & (fld["DUPS"] > 2), "reports_worse_duplicate")
    drop(~miss["LZ"] & (fld["LZ"] == 1), "reports_landlocked")
    idx = np.flatnonzero(keep)
    n = idx.size
    vals = np.full((n, len(CHANNELS)), np.nan, np.float64)

    def pick(k, scale):
        v = fld[k][idx].astype(np.float64) * scale
        v[miss[k][idx]] = np.nan
        return v

    vals[:, 0] = pick("SST", 0.1)
    vals[:, 1] = pick("AT", 0.1)
    vals[:, 2] = pick("SLP", 0.1)
    d = fld["D"][idx]
    dm = miss["D"][idx]
    w = pick("W", 0.1)
    calm = ~dm & (d == 361)
    var = ~dm & (d == 362)
    ok = ~dm & (d >= 1) & (d <= 360) & np.isfinite(w)
    rad = np.deg2rad(d.astype(np.float64))
    u = np.where(ok, -w * np.sin(rad), np.nan)
    v = np.where(ok, -w * np.cos(rad), np.nan)
    u[calm] = 0.0
    v[calm] = 0.0
    vals[:, 3] = np.round(u, 6)
    vals[:, 4] = np.round(v, 6)
    _add(counts, "wind_calm", calm.sum())
    _add(counts, "wind_variable_direction", var.sum())
    inc = ~calm & ~var & (dm ^ ~np.isfinite(w))
    _add(counts, "wind_incomplete", inc.sum())
    bad_d = ~dm & ~calm & ~var & ((d < 1) | (d > 360))
    _add(counts, "wind_bad_direction", bad_d.sum())
    vals[:, 5] = pick("DPT", 0.1)
    vals[:, 6] = pick("WH", 0.5)
    cl = pick("N", 1.0)
    obsc = cl == 9
    _add(counts, "cloud_obscured", obsc.sum())
    cl[obsc] = np.nan
    vals[:, 7] = cl
    # the NCDC-QC flags: erroneous -> NaN; the row's qc = worst kept class
    err = counts.setdefault("ncdc_erroneous", {})
    cls = np.zeros((n, len(CHANNELS)), np.int64)
    for j, (c, fname) in enumerate(zip(CHANNELS, FLAG_OF)):
        fl = flags[fname][idx]
        e = np.isfinite(vals[:, j]) & (fl >= 7) & (fl <= 9)
        if e.any():
            err[c[0]] = err.get(c[0], 0) + int(e.sum())
            vals[e, j] = np.nan
        k = np.where(fl == 1, 0, np.where((fl == 2) | (fl == 3), 1,
                     np.where((fl >= 4) & (fl <= 6), 3, 2)))
        cls[:, j] = np.where(np.isfinite(vals[:, j]), k, -1)
    qc = np.maximum(cls.max(axis=1), 0).astype(np.uint8)
    lat = fld["LAT"][idx] / 100.0
    lon = fld["LON"][idx] / 100.0
    lon = np.where(lon >= 180.0, lon - 360.0, lon)
    days = cm.days_from_civil(yr[idx], mo[idx], dy[idx])
    t = days * 86400 + fld["HR"][idx] * 36
    ids = np.ascontiguousarray(A[idx, ID_SPAN[0]:ID_SPAN[1]]).view(
        "S9").reshape(-1)
    ii = fld["II"][idx]
    _add(counts, "reports_generic_id", (~miss["II"][idx] & (ii == 2)).sum())
    return (t, lat, lon, vals, ids, fld["DCK"][idx], fld["SID"][idx], qc,
            miss["DCK"][idx] | miss["SID"][idx])


def platforms_of(ids, dck, sid, counts):
    """platform_hash per row, the blank-ID fallback applied."""
    u, inv = np.unique(ids, return_inverse=True)
    h = np.zeros(len(u), np.int64)
    blank_u = np.zeros(len(u), bool)
    for i, s in enumerate(u):
        txt = s.decode("latin-1").strip()
        if txt:
            h[i] = f10b.platform_hash(txt)
        else:
            blank_u[i] = True
    out = h[inv]
    blank = blank_u[inv]
    if blank.any():
        _add(counts, "reports_blank_id", blank.sum())
        keys = dck[blank] * 1000 + sid[blank]
        ku, kinv = np.unique(keys, return_inverse=True)
        kh = np.array([f10b.platform_hash(
            f"icoads-deck-{int(k) // 1000}-sid-{int(k) % 1000}") for k in ku],
            np.int64)
        out[blank] = kh[kinv]
    return out


# ================================================================ adapter ==
class ICOADSAdapter(f10b.SourceAdapter):
    store = "icoads"
    title = ("ICOADS marine reports (IMMA1, NOAA NCEI): Release 3.0 "
             "final-untrim 1662-2014 and the near-real-time final product "
             "2015 onward, one row per report")
    family = "1gf"
    distribution = "public"
    licence = {"name": "NOAA NCEI open data (ICOADS R3.0; CC BY 4.0 at GDEX)",
               "redistribution": "attribution", "derived_works": "free",
               "attribution": "ICOADS Release 3.0 (Freeman et al. 2017), "
                              "NOAA NCEI"}
    time_dtype = "int64"
    platform_meta = False
    credentials = ()
    channels = CHANNELS
    log2_fp = -4.0
    log2_dt = float(np.log2((1.0 / 24.0) / 5.0))
    per_year = True
    first_year = 1662
    qc_policy = (
        "Per value the IMMA1 NCDC-QC flag (SNC, ANC, PNC, WNC, DNC, ENC, "
        "CNC): 7-9 (erroneous) -> NaN and counted (`ncdc_erroneous`). Row qc "
        "= worst class among kept values: 0 all flagged 1 (correct), 1 some "
        "2-3 (correctable), 2 some unassessed (flag blank or A), 3 some 4-6 "
        "(suspect). Reports dropped and counted: no Icoads attm, no day/hour, "
        "no or bad position, ZNC 7-9, DUPS > 2, LZ = 1, no value. Trimming "
        "flags not applied. Out-of-bounds values -> NaN, counted.")
    sources = (UNTRIM + "IMMA1_R<rel>_<YYYY>-<MM>.gz",
               NRT + "icoads-nrt_r<rel>_final_d<YYYYMM>_c<date>.dat.gz",
               DOC)
    verified = (
        "2026-09-17 from the sandbox: final-untrim/ listing (3,232 monthly "
        "files, 1662-12..2014-12, 44.97 GB exact); nrt/monthly/ listing "
        "(final product: r3.0.2 131 files incl. 2 re-issued months, r3.0.3 "
        "13 files, 13 months in both); R3.0-imma1_short.pdf Tables C0/C1; "
        "records of 1662-12, 1880-01, 2010-07 (2,108,040 lines), 2015-01 and "
        "2026-08 (head)")
    notes = (
        "global-marine/ is NCEI's CSV product, not IMMA1. Humidity channel is "
        "the dew point (the core carries DPT). Wave height is the core's "
        "wind-wave WH in half metres. Blank ID -> one pooled platform per "
        "(deck, source id).")
    smoke_window = ("2014-12-30", "2015-01-02")
    smoke_probe_month = "2015-01"
    fetch_month_scope = "month"

    def __init__(self):
        self._months = None

    # ------------------------------------------------------------ listing --
    def _local(self, ctx, *p):
        return os.path.join(ctx.source_dir, "icoads", *p)

    def _read_local(self, ctx, *p):
        with open(self._local(ctx, *p), "rb") as fh:
            raw = fh.read()
        ctx.count_bytes(len(raw))
        return raw

    def _page(self, ctx, url, local):
        if ctx.source_dir:
            return self._read_local(ctx, *local).decode("latin-1")
        raw, why = cm.get_bytes(url, attempts=ctx.a.attempts)
        if raw is None:
            sys.exit(f"{url} answered 404 ({why}) — the ICOADS archive moved")
        return raw.decode("latin-1")

    def months(self, ctx):
        if self._months is None:
            try:
                un = parse_untrim(self._page(
                    ctx, UNTRIM, ("final-untrim", "index.html")))
                nrt, lost = parse_nrt(self._page(
                    ctx, NRT, ("nrt", "monthly", "index.html")))
            except FormatError as e:
                sys.exit(f"REFUSING icoads: {e}")
            both = sorted(set(un) & set(nrt))
            allm = dict(nrt)
            allm.update(un)                     # final-untrim wins
            self._months = {"months": dict(sorted(allm.items())),
                            "untrim": len(un), "nrt": len(nrt),
                            "nrt_superseded": lost, "in_both": both}
        return self._months

    # ------------------------------------------------------------ reading --
    def _open(self, ctx, e):
        if ctx.source_dir:
            sub = ("final-untrim",) if e["product"] == "final-untrim" else \
                ("nrt", "monthly")
            p = self._local(ctx, *sub, e["name"])
            ctx.count_bytes(os.path.getsize(p))
            return open(p, "rb")
        err = None
        for i in range(max(1, ctx.a.attempts)):
            try:
                return f10b.CountingStream(e["url"])
            except f10b._NotFound:
                raise
            except (IOError, *cm.RETRY_ERRORS) as x:
                err = x
                time.sleep(3.0 * (2 ** i))
        raise IOError(f"{e['url']}: {err}")

    def _month_rows(self, ctx, y, m, t_lo, t_hi, label):
        """Yield (label, rows, counts) for one month file, block by block."""
        e = self.months(ctx)["months"].get((y, m))
        counts = {"files": 0}
        if e is None:
            yield label, None, counts
            return
        try:
            raw = self._open(ctx, e)
        except f10b._NotFound:
            ctx.note_absent(label, f"{e['url']}: listed, and 404")
            return
        except IOError as x:
            ctx.note_absent(label, str(x))
            return
        first = True
        try:
            with raw, gzip.GzipFile(fileobj=raw, mode="rb") as gz:
                block = []
                for ln in gz:
                    block.append(ln)
                    if len(block) >= BLOCK_LINES:
                        c = {}
                        rows = self._pack_block(block, y, t_lo, t_hi, c)
                        block = []
                        yield label, rows, c
                        first = False
                c = {"files": 1, "bytes_listed": e["bytes"],
                     "file_" + e["product"]: 1}
                rows = self._pack_block(block, y, t_lo, t_hi, c) \
                    if block else None
                yield label, rows, c
        except (OSError, EOFError, zlib.error) as x:
            ctx.note_absent(label, f"{e['name']}: {type(x).__name__}: {x} — "
                                   f"a truncated or broken download"
                                   + ("" if first else " (after some rows)"))

    def _pack_block(self, block, year, t_lo, t_hi, c):
        t, la, lo, v, ids, dck, sid, qc, dsmiss = parse_block(block, c)
        # a record of another year in this year's file (none seen): drop
        y0 = int(cm.days_from_civil(year, 1, 1)) * 86400
        y1 = int(cm.days_from_civil(year + 1, 1, 1)) * 86400
        other = (t < y0) | (t >= y1)
        _add(c, "reports_other_year", other.sum())
        inside = ~other & (t >= t_lo) & (t <= t_hi)
        _add(c, "reports_outside_window", (~other & ~inside).sum())
        t, la, lo, v, ids, dck, sid, qc = (x[inside] for x in
                                           (t, la, lo, v, ids, dck, sid, qc))
        oob = self.mask_bounds(v)
        if oob:
            f10b._merge_counts(c, {"out_of_bounds": oob})
        # a wind vector is kept whole or not at all
        half = np.isnan(v[:, 3]) ^ np.isnan(v[:, 4])
        _add(c, "wind_half_blanked", half.sum())
        v[half, 3:5] = np.nan
        alive = np.isfinite(v).any(axis=1)
        _add(c, "reports_no_value", (~alive).sum())
        t, la, lo, v, ids, dck, sid, qc = (x[alive] for x in
                                           (t, la, lo, v, ids, dck, sid, qc))
        if not t.size:
            return None
        plat = platforms_of(ids, dck, sid, c)
        _add(c, "rows_kept", t.size)
        qd = c.setdefault("qc", {})
        for q, k in zip(*np.unique(qc, return_counts=True)):
            qd[str(int(q))] = qd.get(str(int(q)), 0) + int(k)
        return self.pack(t, la, lo, v, plat, qc)

    # --------------------------------------------------------- the contract --
    def index(self, ctx):
        ms = self.months(ctx)
        allm = ms["months"]
        want = {k: e for k, e in allm.items() if k[0] in ctx.years}
        first = None
        if want:
            k = min(want)
            e = want[k]
            if ctx.source_dir:
                raw = self._open(ctx, e)
                head = raw.read(65536)
                raw.close()
            else:
                head, why = cm.get_bytes(e["url"], attempts=ctx.a.attempts,
                                         headers={"Range": "bytes=0-65535"})
                if head is None:
                    sys.exit(f"{e['url']} is listed and answered 404")
            text = zlib.decompressobj(31).decompress(head)
            lines = text.split(b"\n")[:-1][:50]
            c = {}
            t = parse_block(lines, c)[0]
            if not len(lines) or c.get("reports_no_icoads_attm", 0) == \
                    len(lines):
                sys.exit(f"REFUSING icoads: the first records of {e['name']} "
                         f"do not carry the Icoads attm at column 108")
            first = {"file": e["name"], "lines_read": len(lines),
                     "rows_parsed": int(len(t)),
                     "first_line": lines[0][:120].decode("latin-1")}
        per_year = {}
        for (y, m), e in want.items():
            d = per_year.setdefault(str(y), {"months": 0, "bytes": 0})
            d["months"] += 1
            d["bytes"] += e["bytes"]
        return {
            "dataset": "ICOADS IMMA1 (R3.0 final-untrim + NRT final)",
            "url": HOST,
            "listing": {"months": len(allm),
                        "first": "%04d-%02d" % min(allm),
                        "last": "%04d-%02d" % max(allm),
                        "final_untrim_months": ms["untrim"],
                        "nrt_final_months": ms["nrt"],
                        "nrt_files_superseded": ms["nrt_superseded"],
                        "months_in_both": ["%04d-%02d" % k
                                           for k in ms["in_both"]],
                        "bytes_all": int(sum(e["bytes"]
                                             for e in allm.values()))},
            "window_bytes": int(sum(e["bytes"] for e in want.values())),
            "per_year": per_year,
            "releases": sorted({e["release"] for e in want.values()}),
            "first_record": first,
        }

    def fetch_year(self, ctx, year):
        allm = self.months(ctx)["months"]
        months = [m for (y, m) in allm if y == year]
        if not months:
            yield str(year), None, {"months_listed": 0}
            return
        t0 = time.time()
        agg = {"months_listed": len(months)}
        for m in sorted(months):
            for label, rows, c in self._month_rows(
                    ctx, year, m, ctx.t_lo, ctx.t_hi, f"{year}-{m:02d}"):
                if c:
                    f10b._merge_counts(agg, c)
                yield label, rows, None
        agg["fetch_seconds"] = round(time.time() - t0, 1)
        yield str(year), None, agg

    def fetch_month(self, ctx, year, month):
        allm = self.months(ctx)["months"]
        if (year, month) not in allm:
            sys.exit(f"icoads: {year}-{month:02d} is not in either listing")
        lo, hi = f10b.month_bounds_s(year, month)
        lo, hi = max(lo, ctx.t_lo), min(hi, ctx.t_hi)
        agg = {}
        label = f"{year}-{month:02d}"
        for _l, rows, c in self._month_rows(ctx, year, month, lo, hi, label):
            if c:
                f10b._merge_counts(agg, c)
            yield label, rows, None
        agg["file"] = allm[(year, month)]["name"]
        yield label, None, agg

    # ---------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260917):
        return make_smoke_sources(root, d_lo, d_hi, seed)


# ================================================================== smoke ==
def imma1_line(y, mo, dy, hr, lat, lon, ii=1, ident="SMOKE01", d=None, w=None,
               slp=None, at=None, dpt=None, sst=None, n=None, wh=None,
               dck=926, sid=63, dups=0, lz=None, flags=None, attl=65,
               tail=True):
    """One IMMA1 record from raw coded integers (None = blank)."""
    def f(v, width):
        return " " * width if v is None else f"{int(v):>{width}d}"
    core = (f(y, 4) + f(mo, 2) + f(dy, 2) + f(hr, 4) + f(lat, 5) + f(lon, 6)
            + " 1" + "2" + "1" + "6" + " " + " " + "  " + f(ii, 2)
            + f"{ident:<9.9s}" + "  "
            + ("5" if d is not None else " ") + f(d, 3)
            + ("1" if w is not None else " ") + f(w, 3)
            + " " + "  " + "  " + " " + f(slp, 5) + " " + "   " + " "
            + f(at, 4) + " " + "    " + ("1" if dpt is not None else " ")
            + f(dpt, 4) + (" 1" if sst is not None else "  ") + f(sst, 4)
            + f(n, 1) + " " * 6 + "  " + "  " + f(wh, 2) + " " * 6)
    assert len(core) == 108, len(core)
    fl = flags or {}
    ncdc = "".join(fl.get(k, "1") for k in NCDC)
    att = (" 1" + f(attl, 2) + " " + "  1" + " 0" + f(dck, 3) + f(sid, 3)
           + " 7" + f(dups, 2) + "000  " + "  " + " " * 12 + "2"
           + "111111" + ncdc + "  " + f(lz, 1) + "  ")
    assert len(att) == 65, len(att)
    rec = core + att
    if tail:
        rec += " 5941234567890"          # a following attm the parser ignores
    return rec


def make_smoke_sources(root, d_lo, d_hi, seed=20260917):
    """ICOADS in its real layout: final-untrim/ with 2014-12, nrt/monthly/
    with 2015-01 in three versions (the r3.0.3 one wins) plus decoys.

    Hostile on purpose: a report with a blank ID (deck fallback), wind calm /
    variable / half-missing, cloud 9, an erroneous SST flag, a suspect and an
    unassessed flag, a wave height out of bounds, a missing hour, a worse
    duplicate, a landlocked report, a report with no value, a position flag 7,
    a longitude past 180, and records outside the window.
    """
    import datetime as dt
    rng = np.random.default_rng(seed)
    base = os.path.join(root, "icoads")
    un_d = os.path.join(base, "final-untrim")
    nrt_d = os.path.join(base, "nrt", "monthly")
    os.makedirs(un_d, exist_ok=True)
    os.makedirs(nrt_d, exist_ok=True)
    truth = []
    files = {}                                   # (y, m) -> [lines]
    SHIPS = (("WDC1234", 3500, 32000), ("41001", -1500, 1500),
             ("", 1000, 20000))
    w_lo = f10b.seconds_since_epoch(d_lo)
    w_hi = f10b.seconds_since_epoch(d_hi) + 86399
    day = d_lo - dt.timedelta(days=1)
    while day <= d_hi:
        for h in (0, 1250, 2399):
            for k, (ident, la, lo) in enumerate(SHIPS):
                sst = int(rng.integers(50, 300))
                at = int(rng.integers(-50, 300))
                slp = int(rng.integers(9800, 10300))
                dd = int(rng.integers(1, 361))
                ww = int(rng.integers(0, 300))
                dpt = int(rng.integers(-50, 200))
                n = int(rng.integers(0, 9))
                wh = int(rng.integers(0, 12))
                kw = dict(ii=1 if ident else None, ident=ident, d=dd, w=ww,
                          slp=slp, at=at, dpt=dpt, sst=sst, n=n, wh=wh,
                          dck=926 + k, sid=63)
                qc = 0
                v = [sst / 10, at / 10, slp / 10,
                     -ww / 10 * np.sin(np.deg2rad(dd)),
                     -ww / 10 * np.cos(np.deg2rad(dd)), dpt / 10, wh * 0.5,
                     float(n)]
                tag = (day.day, h, k)
                if tag == (31, 1250, 0):
                    kw["d"] = 361                       # calm
                    v[3] = v[4] = 0.0
                elif tag == (31, 1250, 1):
                    kw["d"] = 362                       # variable
                    v[3] = v[4] = np.nan
                elif tag == (1, 0, 0):
                    kw["w"] = None                      # direction only
                    v[3] = v[4] = np.nan
                elif tag == (1, 0, 1):
                    kw["n"] = 9                         # sky obscured
                    v[7] = np.nan
                elif tag == (1, 1250, 0):
                    kw["flags"] = {"SNC": "9"}          # erroneous SST
                    v[0] = np.nan
                elif tag == (1, 1250, 1):
                    kw["flags"] = {"ANC": "5"}          # suspect AT -> qc 3
                    qc = 3
                elif tag == (1, 1250, 2):
                    kw["flags"] = {"WNC": "A"}          # unassessed -> qc 2
                    qc = 2
                elif tag == (2, 0, 0):
                    kw["wh"] = 80                       # 40 m: out of bounds
                    v[6] = np.nan
                elif tag == (2, 0, 1):
                    kw["flags"] = {"PNC": "3"}          # correctable -> qc 1
                    qc = 1
                lat, lon = la + h, lo + 100 * day.day
                kw_lon = lon if lon < 36000 else lon - 36000
                line = imma1_line(day.year, day.month, day.day, h, lat,
                                  kw_lon, **kw)
                files.setdefault((day.year, day.month), []).append(line)
                t = f10b.seconds_since_epoch(dt.datetime.combine(
                    day, dt.time())) + h * 36
                if w_lo <= t <= w_hi:
                    lonw = kw_lon / 100.0
                    lonw = lonw - 360.0 if lonw >= 180.0 else lonw
                    plat = f10b.platform_hash(ident) if ident else \
                        f10b.platform_hash(f"icoads-deck-{926 + k}-sid-63")
                    truth.append({"t": t, "lat": lat / 100.0, "lon": lonw,
                                  "platform": plat,
                                  "v": [float(x) for x in v], "qc": qc})
        # the droppers, one set per day
        for kw in (dict(hr=None), dict(dups=4), dict(lz=1),
                   dict(flags={"ZNC": "7"}),
                   dict(sst=None, at=None, slp=None, d=None, w=None,
                        dpt=None, n=None, wh=None),
                   dict(attl=94)):
            args = dict(ident="DROPME", d=10, w=10, slp=10000, at=100,
                        dpt=50, sst=150, n=4, wh=2, hr=600)
            args.update(kw)
            hr = args.pop("hr")
            files.setdefault((day.year, day.month), []).append(
                imma1_line(day.year, day.month, day.day, hr, 100, 100,
                           **args))
        day += dt.timedelta(days=1)

    def gz(path, lines):
        with open(path, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as g:
                g.write(("\n".join(lines) + "\n").encode("latin-1"))
        return os.path.getsize(path)

    def listing(title, rows):
        body = "".join(
            f'<tr>\n<td><a href="{n}">{n}</a></td>\n<td align="right">'
            f'2019-09-18 19:40</td>\n<td align="right">{s}</td>\n<td> </td>'
            f'\n</tr>\n' for n, s in rows)
        return ("<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 3.2 Final//EN\">\n"
                f"<html>\n<head>\n<title>Index of {title}</title>\n</head>\n"
                f"<body>\n<h1>Index of {title}</h1>\n<table><tbody>\n<tr>\n"
                "<th>Name</th>\n<th>Last modified</th>\n<th>Size</th>\n"
                "<th>Description</th>\n</tr>\n<tr><th colspan=\"4\"><hr></th>"
                "</tr>\n<tr>\n<td><a href=\"..\">Parent Directory</a></td>\n"
                "<td> </td>\n<td align=\"right\">-</td>\n<td> </td>\n</tr>\n"
                + body + "<tr>\n<td><a href=\"doc/\">doc/</a></td>\n<td "
                "align=\"right\">2019-09-18 19:40</td>\n<td align=\"right\">-"
                "</td>\n<td> </td>\n</tr>\n</tbody></table>\n</body></html>\n")

    un_rows, nrt_rows = [], []
    for (y, m), ls in sorted(files.items()):
        if y <= 2014:
            n = f"IMMA1_R3.1.0_{y}-{m:02d}.gz"
            un_rows.append((n, gz(os.path.join(un_d, n), ls)))
        else:
            n = f"icoads-nrt_r3.0.3_final_d{y}{m:02d}_c20260101.dat.gz"
            nrt_rows.append((n, gz(os.path.join(nrt_d, n), ls)))
            # the losers and decoys: garbage bodies, never to be read
            for dn in (f"icoads-nrt_r3.0.2_final_d{y}{m:02d}_c20250101.dat.gz",
                       f"icoads-nrt_r3.0.2_final_d{y}{m:02d}_c20250301.dat.gz",
                       f"icoads-nrt_r3.0.3_total_d{y}{m:02d}_c20260101.dat.gz",
                       f"ICOADS_R3.0.0_d{y}{m:02d}_c20230502.nc"):
                with open(os.path.join(nrt_d, dn), "wb") as fh:
                    fh.write(b"not this one")
                nrt_rows.append((dn, 12))
    un_rows.append(("R3.0-imma1_short.pdf", 928456))
    with open(os.path.join(un_d, "index.html"), "w") as fh:
        fh.write(listing("/data/icoads/v3/archive/final-untrim/", un_rows))
    with open(os.path.join(nrt_d, "index.html"), "w") as fh:
        fh.write(listing("/data/icoads/v3/archive/nrt/monthly/",
                         sorted(nrt_rows)))
    return truth


ADAPTER = ICOADSAdapter
