#!/usr/bin/env python3
"""Family 10's TIER-P STORES — drifters, moorings, ship CO2 and altimeter tracks.

E-079 §2-§4, made runnable. One builder, four source adapters, three stages.
`ml/family10_store.py` is the reader; `ml/build_family10_registry.py` writes the
family registry. Nothing in `ml/build_family7.py`, `ml/build_family8_argo.py`,
`ml/family8_store.py` or `ml/cone_sampler.py` is touched: family 10 is family
7.1's dense groups and family 8's Argo store, unchanged and already on the Hub,
PLUS these four stores.

PLAIN ENGLISH. A gridded product asks "is there a value in this cell at this
five-day bin", and for a sparse observing system the answer is usually no.
Family 8 replaced that question for Argo with "what are the k nearest
measurements, and how far away are they". This builder asks the same question of
four more observing systems, in one schema, so a consumer that can read one can
read all of them:

  gdp       the Global Drifter Program's 6-hourly quality-controlled
            interpolated product — surface velocity at 15 m, sea-surface
            temperature, and whether the drogue was still attached. 1979-02 ->
  gtmba     the Global Tropical Moored Buoy Array (TAO/TRITON in the Pacific,
            PIRATA in the Atlantic, RAMA in the Indian Ocean) — one row per
            mooring per day: surface and air temperature, salinity, winds,
            currents, and subsurface temperature at eleven standard depths.
            1977-11 ->
  socat     the Surface Ocean CO2 Atlas synthesis — underway ship and mooring
            measurements of surface-water CO2 fugacity with sea-surface
            temperature, salinity and atmospheric pressure. 1957 ->
  slatrack  Copernicus Marine's reprocessed level-3 along-track sea-level
            anomaly, every altimeter mission at 1 Hz (about 7 km). 1993 ->

THREE STAGES, IN FIXED ORDER — `index | fetch | publish` (or `all`):

  index    LIST the archive before spending anything: ask the service what it
           holds, read one real record, and write the plan of files/years into
           `<work>/<store>/plan.json`. Nothing is fetched in bulk here.
  fetch    stream year by year, parse, filter, append per-year column parts
           under `<work>/<store>/parts/<year>/`, and mark each year DONE only
           after its parts are on disk (ml/CLAUDE.md §5.21 — a marker may only
           under-claim). Then assemble the store.
  publish  upload to `tensors/family10/<store>/` on the Hub and DOWNLOAD EVERY
           FILE BACK to compare sha256. A publish that cannot verify fails the
           job (§0.2 — an upload that returns 200 is not evidence the bytes are
           retrievable).

RESUMABILITY IS THE CONTRACT, the same one families 7 and 8 have: re-run with
the same `--work` and finished stages and finished years are skipped.

ONE EXCEPTION, AND IT IS A PROPERTY OF THE SOURCE, NOT A SHORTCUT. The SOCAT
synthesis is a single 1.4 GB file sorted by EXPOCODE, not by time — measured
2026-09-13, its first data rows are a 2018 sailing yacht and its second cruise
is 1986 — so there is no per-year request to make. Its fetch stage streams the
file ONCE, buckets rows into per-year parts as it goes, flushing every
`FLUSH_ROWS`, and marks every year in range only when the stream has run to the
end. Resume granularity for `socat` is therefore the whole pass; for the other
three it is the year. The store.json says so.

Run:
  python3 ml/build_family10_stores.py --store gdp --smoke        # synthetic, seconds
  python3 ml/build_family10_stores.py --store gdp --work /opt/earth-cache/f10 \
      --start 2015-01-01 --end 2015-01-31 --stage index
  python3 ml/build_family10_stores.py --store socat --work ... --stage all
  python3 ml/build_family10_stores.py --store gtmba --work ... --source-dir DIR
"""
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import family10_store as f10                                   # noqa: E402
from build_family7 import (START, END, Progress, atomic_json,   # noqa: E402
                           git_sha, hub_add_ops, hub_commit, hub_repo,
                           hub_upload_with_backoff, mark, marked, marker,
                           read_json, sha256, utcnow)

CACHE = os.path.join(HERE, "cache")
HF_DATASET = "earth-tensors"
HF_ROOT = "tensors/family10"
FAMILY = "family10"

PENTAD_DAYS = 5
assert f10.PENTAD_DAYS == PENTAD_DAYS
assert str(START) == f10.EPOCH, (START, f10.EPOCH)

UA = {"User-Agent": "earth-science-pipeline/1.0 "
                    "(research; github blauewelt/earth)"}
SOCKET_TIMEOUT = 120          # per socket operation — a stall detector
CHUNK = 1 << 22               # 4 MiB reads
FLUSH_ROWS = 1_000_000        # rows held in memory per year before a part flush

STAGES = ["index", "fetch", "publish"]
DEPS = {"fetch": ["index"], "publish": ["fetch"]}

# int16 holds -32768..32767; the axis needs bin -1825 (1957-01-01) to +3141
# (2024-12-31) today and roughly +6000 by 2100, so int16 is not a constraint —
# but the assertion is here because a silently wrapped bin is unfindable later.
BIN_MIN_INT16, BIN_MAX_INT16 = -32768, 32767


def days_since_epoch(d):
    """A date or datetime -> fractional days since 1982-01-01 00:00 UTC."""
    if isinstance(d, dt.datetime):
        base = dt.datetime(START.year, START.month, START.day)
        return (d - base).total_seconds() / 86400.0
    return float((d - START).days)


def parse_date(s):
    return dt.date(*(int(x) for x in str(s)[:10].split("-")))


def platform_hash(text):
    """A stable positive int64 for a string platform id.

    Deterministic across machines and python versions (`hash()` is not), and
    60 bits so it never collides with a real WMO number and never goes
    negative. The store.json records the rule, and every store that uses it
    also ships the id -> hash table where the vocabulary is small enough to be
    worth shipping (moorings, altimeter missions).
    """
    h = hashlib.sha1(str(text).strip().encode("utf-8")).hexdigest()
    return int(h[:15], 16)


# ================================================================== fetching ==
class _NotFound(Exception):
    """A definite 404 — the resource does not exist, a legitimate answer."""


class _Empty(Exception):
    """The service answered, and its answer is 'no rows'. Not an error."""


def http_bytes(url, timeout=SOCKET_TIMEOUT, headers=None):
    """GET url -> bytes, with ERDDAP's empty-result 404 told apart from a 404."""
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        body = b""
        try:
            body = e.read()[:4000]
        except Exception:                                       # noqa: BLE001
            pass
        if e.code == 404 and b"produced no matching results" in body:
            raise _Empty(url) from None
        if e.code == 404:
            raise _NotFound(url) from None
        raise IOError(f"{url}: HTTP {e.code} — "
                      f"{body.decode('utf-8', 'replace')[:400]}") from None


def http_to_file(url, path, timeout=SOCKET_TIMEOUT):
    """GET url -> path, SIZE-VERIFIED against Content-Length.

    Family 8's rule: a truncated transfer raises nothing and surfaces later as
    a parse error in a completely different place.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    part = f"{path}.part{os.getpid()}"
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            want = r.headers.get("Content-Length")
            with open(part, "wb") as fh:
                shutil.copyfileobj(r, fh, CHUNK)
    except urllib.error.HTTPError as e:
        if os.path.exists(part):
            os.remove(part)
        body = b""
        try:
            body = e.read()[:4000]
        except Exception:                                       # noqa: BLE001
            pass
        if e.code == 404 and b"produced no matching results" in body:
            raise _Empty(url) from None
        if e.code == 404:
            raise _NotFound(url) from None
        raise IOError(f"{url}: HTTP {e.code} — "
                      f"{body.decode('utf-8', 'replace')[:400]}") from None
    except BaseException:
        if os.path.exists(part):
            os.remove(part)
        raise
    got = os.path.getsize(part)
    if want is not None and got != int(want):
        os.remove(part)
        raise IOError(f"{url}: {got:,} of {int(want):,} bytes — truncated")
    os.replace(part, path)
    return path


def fetch_first(urls, path=None, attempts=3, sleep=5.0):
    """The first URL that serves. None if EVERY url is a 404 or an empty result.

    A definite "not there" is a legitimate gap and the caller continues; any
    other error retries across the whole list and then RAISES, never silently
    (ml/CLAUDE.md §4.6).
    """
    errs = []
    for i in range(attempts):
        n_gone = 0
        for u in urls:
            try:
                return http_to_file(u, path) if path else http_bytes(u)
            except (_NotFound, _Empty):
                n_gone += 1
            except Exception as e:                              # noqa: BLE001
                errs.append(f"{u}: {type(e).__name__}: {e}")
        if n_gone == len(urls):
            return None
        if i < attempts - 1:
            time.sleep(sleep * (2 ** i))
    raise IOError(f"{attempts} attempt(s) over {len(urls)} url(s) all failed — "
                  + " | ".join(errs[-3:]))


def erddap_url(base, dataset, fmt, variables, constraints):
    """An ERDDAP tabledap query, encoded the way the service actually wants it.

    MEASURED 2026-09-13: `curl --data-urlencode` percent-encodes the `&` and
    `>=` that ERDDAP uses as SYNTAX and the service answers HTTP 400. The
    variable list is comma-separated (the commas encoded), the constraints are
    `&`-joined with the operator's `>`/`<` encoded and the value's `:`
    encoded — which is exactly what the browser sends.
    """
    q = "%2C".join(variables)
    for c in constraints:
        q += "&" + c.replace(">", "%3E").replace("<", "%3C") \
                    .replace(":", "%3A")
    return f"{base}/tabledap/{dataset}.{fmt}?{q}"


# =============================================================== the columns ==
# One tier-P row. `values` is the only wide column; everything else is per row.
ROW_KEYS = ("bin", "time_days", "lat", "lon", "values", "platform", "qc")
ROW_DTYPE = {"bin": np.int16, "time_days": np.float32, "lat": np.float32,
             "lon": np.float32, "values": np.float16,
             "platform": np.int64, "qc": np.uint8}


def empty_rows(C):
    out = {k: np.zeros(0, ROW_DTYPE[k]) for k in ROW_KEYS if k != "values"}
    out["values"] = np.zeros((0, C), np.float16)
    return out


# ================================================================= adapters ==
class SourceAdapter:
    """What a source must answer. Four of them below; nothing else is special.

    `channels` is the ORDER of `values`' columns and its names and units are
    what `store.json` publishes, so a consumer never has to know which file a
    column came from. `bounds` is the physical range of each channel from
    E-079 §4's assertion list: a value outside it is set to NaN and COUNTED,
    never silently clipped — clipping would put a fabricated number where a
    broken instrument was.
    """

    store = ""
    title = ""
    channels = ()            # ((name, unit, lo, hi), ...)
    log2_fp = -4.0
    log2_dt = -4.0
    per_year = True          # False => one stream over the whole archive
    first_year = 1982
    qc_policy = ""
    sources = ()
    verified = ""            # what was checked against the live archive, when
    notes = ""

    # -- derived -----------------------------------------------------------
    @property
    def C(self):
        return len(self.channels)

    @property
    def channel_names(self):
        return [c[0] for c in self.channels]

    def bounds(self):
        lo = np.array([c[2] for c in self.channels], np.float64)
        hi = np.array([c[3] for c in self.channels], np.float64)
        return lo, hi

    def schema(self):
        return [{"name": n, "unit": u, "min": lo, "max": hi}
                for (n, u, lo, hi) in self.channels]

    # -- the two things an adapter does ------------------------------------
    def index(self, ctx):
        """List the archive and read ONE real record. Returns the plan dict."""
        raise NotImplementedError

    def fetch_year(self, ctx, year):
        """Yield row batches for `year`. Only for `per_year` adapters."""
        raise NotImplementedError

    def fetch_stream(self, ctx):
        """Yield `(year, rows)` over the whole archive. `per_year = False`."""
        raise NotImplementedError

    # -- helpers shared by the adapters ------------------------------------
    def blank(self, n):
        return np.full((n, self.C), np.nan, np.float64)


# ------------------------------------------------------------------- gdp ----
GDP_ERDDAP = ("https://erddap.aoml.noaa.gov/gdp/erddap",
              "https://osmc.noaa.gov/erddap")
GDP_DATASET = "drifter_6hour_qc"
GDP_VARS = ("ID", "WMO", "time", "latitude", "longitude", "ve", "vn",
            "err_lat", "err_lon", "sst", "err_sst", "drogue_lost_date")
GDP_FILL = -999999.0
GDP_POS_ERR_KM_MAX = 50.0     # E-079 §3.1
GDP_POS_ERR_KM_GOOD = 5.0     # the 'good' / 'probably good' split, below


class GDPAdapter(SourceAdapter):
    """The Global Drifter Program 6-hourly QC interpolated product.

    VERIFIED 2026-09-13 from this sandbox. `GET
    https://erddap.aoml.noaa.gov/gdp/erddap/tabledap/index.json` lists seven
    datasets, of which `drifter_6hour_qc` is "Global Drifter Program - 6 Hour
    Interpolated QC Drifter Data"; its `info/.../index.json` gives `time`
    covering 287,884,800 .. 1,750,248,000 s since 1970 (1979-02-02 ..
    2025-06-18 at the snapshot), `ve`/`vn` with an actual range inside
    +/- 4.1 m/s, `sst` in degree_C, `err_lat`/`err_lon` in degrees, and
    `drogue_lost_date` with "_FillValue = drogue still attached; 0 = drogue
    status uncertain from beginning". A one-month query for 2015-01 returned
    149,658 rows / 17.2 MB / 66 s over 1,275 drifters at 124 six-hourly times.

    ONE REQUEST PER MONTH, not per year: 17 MB and a minute for January 2015
    means a year is ~200 MB, which streams comfortably, while a whole year in
    one query is a request the service has to hold open for a quarter of an
    hour. Each month's CSV is parsed and DELETED before the next is asked for.

    THE DROGUE IS A CHANNEL, NOT A FILTER (E-079 §3.1): a drogued drifter
    follows the 15 m current and an undrogued one follows the surface, and
    those are different measurements the model must be able to tell apart.
    """

    store = "gdp"
    title = "Global Drifter Program, 6-hourly QC interpolated"
    channels = (("u", "m s-1", -5.0, 5.0),
                ("v", "m s-1", -5.0, 5.0),
                ("sst", "degC", -3.0, 45.0),
                ("drogue", "1 = drogue attached", 0.0, 1.0))
    log2_fp = -4.0                       # a point
    log2_dt = -4.3                       # log2(0.25 d / 5 d) = -4.32, a 6-hourly sample
    first_year = 1979
    qc_policy = (
        "The source IS the quality-controlled product — the GDP's kriging "
        "interpolation onto 00/06/12/18 UTC, with per-sample position and SST "
        "error estimates and no per-row flag column. So the flag is derived "
        "from the errors the archive does publish, and the derivation is here "
        "rather than in a consumer: position error = "
        "hypot(err_lat, err_lon * cos(lat)) * 111.32 km; a row with > 50 km is "
        "DROPPED and counted (E-079 §3.1), <= 5 km is qc 1 (good) and the rest "
        "qc 2 (probably good). A velocity or SST equal to the -999999 fill, "
        "or outside the physical bounds above, becomes NaN for that channel "
        "only and is counted; the row survives if any channel does. `drogue` "
        "is 1 where the observation's time precedes `drogue_lost_date`, 0 "
        "where it follows it, and NaN where that date is 1970-01-01, which the "
        "archive documents as 'drogue status uncertain from the beginning' — "
        "an unknown, not an undrogued drifter.")
    sources = (f"{GDP_ERDDAP[0]}/tabledap/{GDP_DATASET}.csvp?<vars>"
               f"&time>=<start>&time<=<end>",
               f"{GDP_ERDDAP[1]}/tabledap/{GDP_DATASET}.csvp (mirror)")
    verified = ("2026-09-13: dataset listing, variable metadata and a real "
                "2015-01 month fetch (149,658 rows) from this sandbox")

    def months(self, ctx):
        d = dt.date(ctx.d_lo.year, ctx.d_lo.month, 1)
        out = []
        while d <= ctx.d_hi:
            nxt = dt.date(d.year + (d.month == 12), d.month % 12 + 1, 1)
            out.append((d, min(nxt - dt.timedelta(days=1), ctx.d_hi)))
            d = nxt
        return [(a, b) for a, b in out if b >= ctx.d_lo]

    def urls(self, lo, hi):
        cons = [f"time>={lo:%Y-%m-%d}T00:00:00Z",
                f"time<={hi:%Y-%m-%d}T23:59:59Z"]
        return [erddap_url(b, GDP_DATASET, "csvp", GDP_VARS, cons)
                for b in GDP_ERDDAP]

    def local(self, ctx, lo):
        return os.path.join(ctx.source_dir, "gdp", f"{lo:%Y-%m}.csv")

    def index(self, ctx):
        """List the tabledap datasets, then read the dataset's own metadata."""
        listing = {}
        info = {}
        if not ctx.source_dir:
            raw = fetch_first([f"{b}/tabledap/index.json?page=1&itemsPerPage=200"
                               for b in GDP_ERDDAP], attempts=ctx.a.attempts)
            if raw is None:
                sys.exit("neither ERDDAP answered the dataset listing")
            t = json.loads(raw)["table"]
            i_id = t["columnNames"].index("Dataset ID")
            i_ti = t["columnNames"].index("Title")
            listing = {r[i_id]: r[i_ti] for r in t["rows"]}
            if GDP_DATASET not in listing:
                sys.exit(f"{GDP_DATASET} is not in the ERDDAP listing "
                         f"({sorted(listing)}) — the access path has moved and "
                         f"E-079 §3.1 must be re-verified before a build")
            raw = fetch_first([f"{b}/info/{GDP_DATASET}/index.json"
                               for b in GDP_ERDDAP], attempts=ctx.a.attempts)
            it = json.loads(raw)["table"]
            for r in it["rows"]:
                if r[0] == "variable":
                    info[r[1]] = {"dtype": r[3]}
                elif r[0] == "attribute" and r[1] in info and \
                        r[2] in ("units", "actual_range", "long_name"):
                    info[r[1]][r[2]] = r[4]
            missing = [v for v in GDP_VARS if v not in info]
            if missing:
                sys.exit(f"{GDP_DATASET} no longer publishes {missing}")
        # PREFLIGHT ONE REAL FILE (E-079 §4): the first month in range, parsed,
        # so a changed column order fails here and not in hour three.
        first = self.months(ctx)[0]
        rows, counts = self._month(ctx, *first, preflight=True)
        return {"dataset": GDP_DATASET, "datasets_listed": listing,
                "variables": info, "months": len(self.months(ctx)),
                "preflight": {"month": f"{first[0]:%Y-%m}",
                              "rows_parsed": int(len(rows["bin"])),
                              "counts": counts}}

    def _month(self, ctx, lo, hi, preflight=False):
        tmp = os.path.join(ctx.scratch, "gdp", f"{lo:%Y-%m}.csv")
        owned = False
        if ctx.source_dir:
            p = self.local(ctx, lo)
            if not os.path.exists(p):
                return empty_rows(self.C), {"missing_month": 1}
        else:
            got = fetch_first(self.urls(lo, hi), tmp, attempts=ctx.a.attempts)
            if got is None:
                return empty_rows(self.C), {"missing_month": 1}
            p, owned = got, True
        try:
            with open(p, newline="") as fh:
                rows, counts = self._parse(ctx, fh)
        finally:
            if owned and not preflight and os.path.exists(p):
                os.remove(p)
            elif owned and preflight and os.path.exists(p):
                os.remove(p)
        return rows, counts

    def _parse(self, ctx, fh):
        r = csv.reader(fh)
        hdr = next(r, None)
        if hdr is None:
            return empty_rows(self.C), {"empty_file": 1}
        col = {h.split(" (")[0].strip(): i for i, h in enumerate(hdr)}
        need = ("time", "latitude", "longitude", "ve", "vn", "sst",
                "err_lat", "err_lon", "drogue_lost_date")
        miss = [n for n in need if n not in col]
        if miss:
            raise ValueError(f"GDP csv is missing {miss}; header was {hdr}")
        c = {n: col[n] for n in need}
        c_id = col.get("ID", col.get("WMO"))
        t_l, lat_l, lon_l, plat_l, qc_l = [], [], [], [], []
        vals = []
        counts = {"rows_read": 0, "drop_pos_err": 0, "drop_no_time": 0,
                  "drop_no_position": 0, "drop_out_of_range": 0,
                  "drop_no_values": 0, "drogue_uncertain": 0}
        oob = np.zeros(self.C, np.int64)
        lo_b, hi_b = self.bounds()
        for row in r:
            counts["rows_read"] += 1
            ts = row[c["time"]].strip()
            if not ts or ts == "NaN":
                counts["drop_no_time"] += 1
                continue
            try:
                when = dt.datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                counts["drop_no_time"] += 1
                continue
            td = days_since_epoch(when)
            if not (ctx.t_lo <= td <= ctx.t_hi):
                counts["drop_out_of_range"] += 1
                continue
            la = _num(row[c["latitude"]])
            lo_ = _num(row[c["longitude"]])
            if la is None or lo_ is None or abs(la) > 90.0:
                counts["drop_no_position"] += 1
                continue
            el, eo = _num(row[c["err_lat"]]), _num(row[c["err_lon"]])
            if el is None or eo is None:
                err_km = np.nan
            else:
                cl = max(np.cos(np.radians(la)), f10.COS_FLOOR)
                err_km = float(np.hypot(el, eo * cl) * f10.KM_PER_DEG)
            if np.isfinite(err_km) and err_km > GDP_POS_ERR_KM_MAX:
                counts["drop_pos_err"] += 1
                continue
            v = np.full(self.C, np.nan)
            v[0] = _num(row[c["ve"]])
            v[1] = _num(row[c["vn"]])
            v[2] = _num(row[c["sst"]])
            dro_unc = False
            dl = row[c["drogue_lost_date"]].strip()
            if not dl or dl == "NaN":
                v[3] = 1.0                      # fill = drogue still attached
            else:
                try:
                    lost = dt.datetime.strptime(dl[:19], "%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    lost = None
                if lost is None:
                    v[3] = np.nan
                elif lost <= dt.datetime(1970, 1, 2):
                    v[3] = np.nan               # "uncertain from the beginning"
                    dro_unc = True
                else:
                    v[3] = 1.0 if when < lost else 0.0
            bad = np.isfinite(v) & ((v < lo_b) | (v > hi_b))
            oob += bad
            v[bad] = np.nan
            if not np.isfinite(v).any():
                # Nothing measured on this row at all — it is not stored, and
                # it must not be counted either. The 2026-09-14 verification of
                # the first published `gdp` store found `drogue_uncertain`
                # 1,417,396 against 1,416,828 NaNs in `values`, and the same
                # 568 missing from `rows_read − drop_pos_err − N`: these rows,
                # counted as drogue-uncertain and then silently dropped here.
                # Both halves close now — the counter is incremented only for a
                # row that is kept, and the drop has a reason of its own.
                counts["drop_no_values"] += 1
                continue
            if dro_unc:
                counts["drogue_uncertain"] += 1
            t_l.append(td)
            lat_l.append(la)
            lon_l.append(lo_)
            vals.append(v)
            plat_l.append(_platform_int(row[c_id] if c_id is not None else ""))
            qc_l.append(1 if (np.isfinite(err_km)
                              and err_km <= GDP_POS_ERR_KM_GOOD) else 2)
        counts["out_of_bounds"] = {n: int(x) for n, x
                                   in zip(self.channel_names, oob) if x}
        return _pack(t_l, lat_l, lon_l, vals, plat_l, qc_l, self.C), counts

    def fetch_year(self, ctx, year):
        for lo, hi in self.months(ctx):
            if lo.year != year:
                continue
            rows, counts = self._month(ctx, lo, hi)
            yield f"{lo:%Y-%m}", rows, counts


# ----------------------------------------------------------------- gtmba ----
# VERIFIED 2026-09-13 from this sandbox. `https://osmc.noaa.gov/erddap` lists
# nine `pmelTaoDy*` datasets ("TAO/TRITON, RAMA, and PIRATA Buoys, Daily,
# ...") and answers a tabledap query with a 302 to
# `https://coastwatch.pfeg.noaa.gov/erddap`, which serves it. The variable
# names below were read out of each dataset's own `info/<id>/index.json`, not
# transcribed from documentation.
GTMBA_ERDDAP = ("https://osmc.noaa.gov/erddap",
                "https://coastwatch.pfeg.noaa.gov/erddap")
GTMBA_KEYS = ("array", "station", "wmo_platform_code", "longitude", "latitude",
              "time", "depth")
# (dataset, value variable, qc variable, the channel it fills, a unit note)
GTMBA_SETS = (
    ("pmelTaoDySst", "T_25", "QT_5025", "sst", 1.0),
    ("pmelTaoDySss", "S_41", "QS_5041", "sss", 1.0),
    ("pmelTaoDyAirt", "AT_21", "QAT_5021", "airt", 1.0),
    ("pmelTaoDyW", "WU_422", "QWS_5401", "wind_u", 1.0),
    ("pmelTaoDyW", "WV_423", "QWS_5401", "wind_v", 1.0),
    ("pmelTaoDyAdcp", "u_1205", "QU_5205", "u_cur", None),
    ("pmelTaoDyAdcp", "v_1206", "QV_5206", "v_cur", None),
)
# E-079 §3.2's eleven standard depths. A site that does not carry a level
# leaves it NaN; the per-depth fill counts go into store.json so the coverage
# is visible rather than assumed.
GTMBA_DEPTHS = (1, 20, 40, 60, 80, 100, 120, 140, 180, 300, 500)
GTMBA_DEPTH_TOL = 0.6         # m — the archive publishes integral depths
GTMBA_FILL = 1.0e35


class GTMBAAdapter(SourceAdapter):
    """PMEL's Global Tropical Moored Buoy Array, DAILY, one row per site per day.

    Seven quantities from six ERDDAP datasets plus the depth-resolved daily
    temperature, joined on `(station, day)`. The join is what makes a row a
    STATE of a mooring rather than seven unrelated series, and it is done here
    rather than by a consumer because the identity of a site across datasets is
    a property of the archive.

    VERIFIED 2026-09-13: `pmelTaoDyT` covers 1977-11-03 .. 2026-09-02 with
    `T_20` (degree_C), `QT_5020`, `ST_6020` and a `depth` axis 1 .. 2000 m; a
    real query for 0N 140W, January 2015, returned 930 rows over the thirty
    depths 1, 3, 5, 10, 13, 15, 20, 25, 28, 30, 35, 40, 45, 48, 50, 60, 80, 83,
    100, 120, 123, 125, 140, 150, 160, 180, 200, 250, 300, 500 — every one of
    E-079 §3.2's eleven standard depths present.

    ADCP VELOCITIES ARE IN cm/s IN THE ARCHIVE and m/s in the store. The factor
    is not hardcoded: `.csvp` puts the unit in the column header
    (`u_1205 (cm s-1)`), and the parser reads it and converts, so a unit change
    upstream is caught rather than silently absorbed.
    """

    store = "gtmba"
    title = "Global Tropical Moored Buoy Array (TAO/TRITON, PIRATA, RAMA), daily"
    channels = (("sst", "degC", -3.0, 45.0),
                ("sss", "PSU", 0.0, 45.0),
                ("airt", "degC", -60.0, 60.0),
                ("wind_u", "m s-1", -120.0, 120.0),
                ("wind_v", "m s-1", -120.0, 120.0),
                ("u_cur", "m s-1", -5.0, 5.0),
                ("v_cur", "m s-1", -5.0, 5.0)) + tuple(
        (f"t_{d}m", "degC", -3.0, 45.0) for d in GTMBA_DEPTHS)
    log2_fp = -4.0                       # a mooring is a point
    log2_dt = -2.3                       # log2(1 d / 5 d) = -2.32
    first_year = 1977
    qc_policy = (
        "PMEL publishes a per-sample quality code beside every daily value: "
        "0 = datum missing, 1 = highest quality, 2 = default quality, "
        "3 = adjusted, 4 = lower quality, 5 = sensor failed. A SAMPLE is kept "
        "only when its code is in {1, 2} and its value is neither the 1e35 "
        "fill nor outside the physical bounds above; everything else is NaN "
        "for that channel alone. A ROW (one site, one day) survives if any "
        "channel does, and its stored `qc` is the WORST code among the samples "
        "that survived — so 1 means every kept number was highest quality and "
        "2 means at least one was default quality. Wind components share the "
        "wind-speed code the archive publishes (there is no separate code for "
        "the components). Depth levels are matched to E-079 §3.2's eleven "
        "standard depths within 0.6 m; a site that does not carry a level "
        "leaves it NaN rather than interpolating one in.")
    sources = (f"{GTMBA_ERDDAP[0]}/tabledap/pmelTaoDy<X>.csvp?"
               f"array,station,wmo_platform_code,longitude,latitude,time,"
               f"depth,<var>,<qc>&time>=<start>&time<=<end>",
               "302 -> https://coastwatch.pfeg.noaa.gov/erddap (measured)")
    verified = ("2026-09-13: the nine pmelTaoDy* datasets listed on "
                "osmc.noaa.gov/erddap, every variable name read from each "
                "dataset's info json, and a real 0N 140W January 2015 daily "
                "temperature fetch (930 rows, 30 depths)")

    def datasets(self):
        out = []
        for ds, var, qc, ch, scale in GTMBA_SETS:
            out.append((ds, var, qc, ch, scale))
        out.append(("pmelTaoDyT", "T_20", "QT_5020", "__depth__", 1.0))
        return out

    def urls(self, ds, variables, lo, hi):
        cons = [f"time>={lo:%Y-%m-%d}T00:00:00Z",
                f"time<={hi:%Y-%m-%d}T23:59:59Z"]
        return [erddap_url(b, ds, "csvp", variables, cons)
                for b in GTMBA_ERDDAP]

    def index(self, ctx):
        listing, info = {}, {}
        if not ctx.source_dir:
            raw = fetch_first([f"{b}/tabledap/index.json?page=1&itemsPerPage=1000"
                               for b in GTMBA_ERDDAP], attempts=ctx.a.attempts)
            if raw is None:
                sys.exit("no ERDDAP answered the dataset listing for GTMBA")
            t = json.loads(raw)["table"]
            i_id = t["columnNames"].index("Dataset ID")
            i_ti = t["columnNames"].index("Title")
            listing = {r[i_id]: r[i_ti] for r in t["rows"]
                       if str(r[i_id]).startswith("pmelTao")}
            wanted = sorted({d for d, *_ in self.datasets()})
            missing = [d for d in wanted if d not in listing]
            if missing:
                sys.exit(f"{missing} are not on the ERDDAP listing "
                         f"({sorted(listing)}) — E-079 §3.2 must be "
                         f"re-verified before a build")
            for ds in wanted:
                raw = fetch_first([f"{b}/info/{ds}/index.json"
                                   for b in GTMBA_ERDDAP],
                                  attempts=ctx.a.attempts)
                it = json.loads(raw)["table"]
                vs = {}
                for r in it["rows"]:
                    if r[0] == "variable":
                        vs[r[1]] = {"dtype": r[3]}
                    elif r[0] == "attribute" and r[1] in vs and \
                            r[2] in ("units", "long_name"):
                        vs[r[1]][r[2]] = r[4]
                info[ds] = vs
                need = {v for d, v, q, _c, _s in self.datasets() if d == ds
                        for v in (v, q)}
                gone = sorted(need - set(vs))
                if gone:
                    sys.exit(f"{ds} no longer publishes {gone}")
        y0 = max(ctx.d_lo.year, self.first_year)
        return {"datasets": {d: listing.get(d, "") for d, *_ in self.datasets()},
                "variables": info,
                "standard_depths_m": list(GTMBA_DEPTHS),
                "years": list(range(y0, ctx.d_hi.year + 1))}

    def fetch_year(self, ctx, year):
        """Every dataset for one year, joined on (station, day)."""
        lo = max(ctx.d_lo, dt.date(year, 1, 1))
        hi = min(ctx.d_hi, dt.date(year, 12, 31))
        # site key -> row. A dict of python objects is fine here: the whole
        # array is ~70 sites x 365 days = 25,550 rows a year.
        acc = {}
        sites = {}
        counts = {"rows_read": 0, "samples_kept": 0, "drop_qc": 0,
                  "drop_fill": 0, "drop_depth": 0, "datasets_empty": 0}
        depth_hits = {d: 0 for d in GTMBA_DEPTHS}
        ch_index = {n: i for i, n in enumerate(self.channel_names)}
        for ds, var, qcv, ch, scale in self.datasets():
            variables = list(GTMBA_KEYS) + [var, qcv]
            path = None
            if ctx.source_dir:
                p = os.path.join(ctx.source_dir, "gtmba", ds, f"{year}.csv")
                if not os.path.exists(p):
                    counts["datasets_empty"] += 1
                    continue
                path = p
                owned = False
            else:
                tmp = os.path.join(ctx.scratch, "gtmba", ds, f"{year}.csv")
                got = fetch_first(self.urls(ds, variables, lo, hi), tmp,
                                  attempts=ctx.a.attempts)
                if got is None:
                    counts["datasets_empty"] += 1
                    continue
                path, owned = got, True
            try:
                with open(path, newline="") as fh:
                    self._merge(ctx, fh, var, qcv, ch, scale, acc, sites,
                                counts, depth_hits, ch_index)
            finally:
                if owned and os.path.exists(path):
                    os.remove(path)
        counts["sites"] = len(sites)
        counts["depth_samples"] = {str(k): int(v) for k, v in depth_hits.items()}
        yield str(year), self._pack(acc, counts), counts

    def _merge(self, ctx, fh, var, qcv, ch, scale, acc, sites, counts,
               depth_hits, ch_index):
        r = csv.reader(fh)
        hdr = next(r, None)
        if hdr is None:
            return
        names = [h.split(" (")[0].strip() for h in hdr]
        units = [h.split(" (")[1].rstrip(")") if " (" in h else ""
                 for h in hdr]
        col = {n: i for i, n in enumerate(names)}
        for n in ("station", "time"):
            if n not in col:
                raise ValueError(f"GTMBA csv is missing {n}; header was {hdr}")
        if var not in col or qcv not in col:
            # Only reachable with a local fixture: an ERDDAP query names the
            # columns it wants and the service either serves them or errors, so
            # a missing value column over the network has already failed above.
            # Counted and announced rather than swallowed (ml/CLAUDE.md §4.6).
            counts["column_absent"] = counts.get("column_absent", 0) + 1
            print(f"::warning::{var}/{qcv} absent from this file "
                  f"({names}) — that channel stays NaN for these rows",
                  flush=True)
            return
        # THE UNIT IS READ, NOT ASSUMED. The ADCP velocities are cm/s in the
        # archive and m/s in the store; a silent factor of 100 is exactly the
        # kind of thing that looks like a plausible ocean.
        u = units[col[var]].strip()
        if scale is None:
            if u in ("cm s-1", "cm/s", "centimeters second-1"):
                scale = 0.01
            elif u in ("m s-1", "m/s", ""):
                scale = 1.0
            else:
                raise ValueError(f"{var} has unit {u!r}, which is neither cm/s "
                                 f"nor m/s — refusing to guess a conversion")
        has_depth = "depth" in col
        for row in r:
            counts["rows_read"] += 1
            ts = row[col["time"]].strip()
            if len(ts) < 10:
                continue
            day = ts[:10]
            try:
                when = dt.datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                continue
            # The archive centres a daily value at 12:00 UTC; the store keeps
            # the stamp the archive gives rather than flooring it to midnight.
            td = days_since_epoch(when)
            if not (ctx.t_lo <= td <= ctx.t_hi):
                continue
            st = row[col["station"]].strip()
            if not st:
                continue
            val = _num(row[col[var]])
            q = _num(row[col[qcv]])
            if val is None or val >= GTMBA_FILL or not np.isfinite(val):
                counts["drop_fill"] += 1
                continue
            if q is None or int(q) not in (1, 2):
                counts["drop_qc"] += 1
                continue
            idx = None
            if ch == "__depth__":
                if not has_depth:
                    continue
                dv = _num(row[col["depth"]])
                if dv is None:
                    continue
                hit = [d for d in GTMBA_DEPTHS if abs(dv - d) <= GTMBA_DEPTH_TOL]
                if not hit:
                    counts["drop_depth"] += 1
                    continue
                depth_hits[hit[0]] += 1
                idx = ch_index[f"t_{hit[0]}m"]
            else:
                idx = ch_index[ch]
            key = (st, day)
            rec = acc.get(key)
            if rec is None:
                la = _num(row[col["latitude"]]) if "latitude" in col else None
                lo_ = _num(row[col["longitude"]]) if "longitude" in col else None
                wmo = _num(row[col["wmo_platform_code"]]) \
                    if "wmo_platform_code" in col else None
                if la is None or lo_ is None or abs(la) > 90.0:
                    continue
                plat = int(wmo) if (wmo is not None and 0 < wmo < 2 ** 31) \
                    else platform_hash(st)
                rec = {"t": td, "lat": la, "lon": lo_, "plat": plat,
                       "v": np.full(self.C, np.nan), "qc": 0}
                acc[key] = rec
                sites[st] = {"platform": plat, "lat": la, "lon": lo_}
            lo_b, hi_b = self.bounds()
            v = val * scale
            if v < lo_b[idx] or v > hi_b[idx]:
                # COUNTED, per channel, not merely skipped: a policy that says
                # "outside the physical bounds becomes NaN" and reports nothing
                # is indistinguishable from one that never fires.
                oob = counts.setdefault("out_of_bounds", {})
                nm = self.channel_names[idx]
                oob[nm] = oob.get(nm, 0) + 1
                continue
            rec["v"][idx] = v
            rec["qc"] = max(rec["qc"], int(q))
            counts["samples_kept"] += 1

    def _pack(self, acc, counts):
        t_l, lat_l, lon_l, vals, plat_l, qc_l = [], [], [], [], [], []
        for (_st, _day), rec in acc.items():
            if not np.isfinite(rec["v"]).any():
                continue
            t_l.append(rec["t"])
            lat_l.append(rec["lat"])
            lon_l.append(rec["lon"])
            vals.append(rec["v"])
            plat_l.append(rec["plat"])
            qc_l.append(max(1, rec["qc"]))
        return _pack(t_l, lat_l, lon_l, vals, plat_l, qc_l, self.C)


# ----------------------------------------------------------------- socat ----
SOCAT_RELEASE_PAGE = "https://socat.info/latest"
SOCAT_VERSION_DEFAULT = "v2026"
SOCAT_URL_DEFAULT = "https://socat.info/socat_files/v2026/SOCATv2026.tsv.zip"
SOCAT_NCEI = ("https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0315110/"
              "SOCATv2026.tsv")
SOCAT_HEADER_FIRST = "Expocode"
SOCAT_WANT = ("Expocode", "QC_Flag", "yr", "mon", "day", "hh", "mm", "ss",
              "longitude", "latitude", "sal", "SST", "PPPP", "fCO2rec",
              "fCO2rec_flag")
SOCAT_KEEP_QC = ("A", "B", "C", "D")
SOCAT_REPORT_ROWS = 2_000_000     # source rows between progress artefacts


class SOCATAdapter(SourceAdapter):
    """The Surface Ocean CO2 Atlas synthesis — the only observation of the sink.

    VERIFIED 2026-09-13 from this sandbox. `https://socat.info/latest` redirects
    to the v2026 release page (released 2026-06-16, DOI 10.25921/8dba-fr90,
    NCEI accession 0315110), which links the global synthesis as
    `https://socat.info/socat_files/v2026/SOCATv2026.tsv.zip` — HEAD 200,
    `content-length: 1,411,801,422`, `accept-ranges: bytes`, a single zip64
    deflate member named `SOCATv2026.tsv`. A Range read of the first 50 MB
    inflates to a 8,453-line preamble (the report header, then one line per
    contributing cruise, then the column glossary) followed by the tab-separated
    data header of 32 columns: Expocode, version, Source_DOI, QC_Flag, yr, mon,
    day, hh, mm, ss, longitude [dec.deg.E], latitude [dec.deg.N],
    sample_depth [m], sal, SST [deg.C], Tequ [deg.C], PPPP [hPa], Pequ [hPa],
    WOA_SSS, NCEP_SLP [hPa], ETOPO2_depth [m], dist_to_land [km],
    GVCO2 [umol/mol], xCO2water_equ_dry, xCO2water_SST_dry, pCO2water_equ_wet,
    pCO2water_SST_wet, fCO2water_equ_wet, fCO2water_SST_wet, fCO2rec [uatm],
    fCO2rec_src, fCO2rec_flag.

    `patm` IS THE MEASURED PRESSURE, `PPPP`, AND NOT `NCEP_SLP`. The file
    carries both, and the second is a reanalysis value interpolated to the ship
    — a model output wearing a measurement's column. A store of observations
    that silently substituted it would be making up data wherever the barometer
    was missing; `patm` is NaN there instead, which is the truth.

    THE FILE IS SORTED BY EXPOCODE, NOT BY TIME, so there is no per-year
    request and the fetch stage is ONE pass that buckets into per-year parts —
    see the module docstring. Longitude is 0..360 in the file and is wrapped
    into [-180, 180) here.
    """

    store = "socat"
    title = "Surface Ocean CO2 Atlas (SOCAT) synthesis"
    channels = (("fco2", "uatm", 0.0, 2000.0),
                ("sst", "degC", -3.0, 45.0),
                ("sss", "PSU", 0.0, 45.0),
                ("patm", "hPa", 800.0, 1100.0))
    log2_fp = -4.0                       # an underway measurement, a point
    log2_dt = -4.0                       # minutes; log2(0.003 d / 5 d) clamped
    per_year = False
    first_year = 1957
    qc_policy = (
        "SOCAT ships TWO flags and both are applied. The dataset-level "
        "`QC_Flag` (A-D acceptable, E not) is the synthesis file's own "
        "inclusion criterion; rows outside {A, B, C, D} are DROPPED and "
        "counted. The per-value `fCO2rec_flag` is the WOCE flag — 2 good, "
        "3 questionable, 4 bad, 9 not generated — mapped to the store's scale "
        "as 2 -> 1 (good), 3 -> 3, 4 -> 4, 9 -> 0 (not assessed), and rows "
        "worse than 2 are dropped by default. The published synthesis file "
        "states it contains only flag-2 values, so the mapping is a guard "
        "rather than a filter, and store.json records how many rows each "
        "clause actually removed — a guard that never fires and a guard that "
        "silently removes a third of the archive look identical from outside. "
        "A row with no finite fCO2rec is dropped; `sal`, `SST` and `PPPP` are "
        "NaN where the file says NaN, and `patm` is the MEASURED `PPPP` and "
        "never the reanalysis `NCEP_SLP` beside it.")
    sources = (SOCAT_URL_DEFAULT, SOCAT_NCEI + " (NCEI OCADS, uncompressed)")
    verified = ("2026-09-13: release page followed to v2026, HEAD on the zip "
                "(1,411,801,422 B, accept-ranges), and the 32-column data "
                "header read out of the first 50 MB by Range request")

    def url(self, ctx):
        return getattr(ctx.a, "socat_url", "") or SOCAT_URL_DEFAULT

    def index(self, ctx):
        """Read the release page for the version, then HEAD the synthesis file."""
        out = {"version": SOCAT_VERSION_DEFAULT, "url": self.url(ctx)}
        if ctx.source_dir:
            p = self._local(ctx)
            out["source"] = f"file://{p}"
            out["bytes"] = os.path.getsize(p) if os.path.exists(p) else 0
        else:
            try:
                page = http_bytes(SOCAT_RELEASE_PAGE).decode("utf-8", "replace")
                import re
                urls = re.findall(
                    r"https://(?:www\.)?socat\.info/socat_files/"
                    r"(v\d{4})/(SOCAT\1\.tsv\.zip)", page)
                if urls:
                    ver, name = urls[0]
                    out["version"] = ver
                    out["url"] = f"https://socat.info/socat_files/{ver}/{name}"
                out["release_page"] = SOCAT_RELEASE_PAGE
            except Exception as e:                              # noqa: BLE001
                # NOT fatal, and NOT silent: the default url is the one that was
                # verified, and a release page that has moved is information.
                print(f"::warning::could not read {SOCAT_RELEASE_PAGE} "
                      f"({type(e).__name__}: {e}) — using the verified "
                      f"{SOCAT_VERSION_DEFAULT} url", flush=True)
            req = urllib.request.Request(out["url"], headers=UA, method="HEAD")
            with urllib.request.urlopen(req, timeout=SOCKET_TIMEOUT) as r:
                out["bytes"] = int(r.headers.get("Content-Length") or 0)
                out["accept_ranges"] = r.headers.get("Accept-Ranges")
                out["last_modified"] = r.headers.get("Last-Modified")
            # PREFLIGHT: the first 8 MB, inflated, must contain the data header.
            head = http_bytes(out["url"], headers={"Range": "bytes=0-8388607"})
            hdr = _socat_header_from(_zip_member_stream(io.BytesIO(head)))
            out["columns"] = hdr
            out["n_columns"] = len(hdr)
        return out

    def _local(self, ctx):
        d = os.path.join(ctx.source_dir, "socat")
        for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if n.endswith((".tsv", ".tsv.zip", ".tsv.gz")):
                return os.path.join(d, n)
        return os.path.join(d, "SOCAT.tsv")

    def _lines(self, ctx):
        """The synthesis file as text lines, streamed — never held in memory."""
        if ctx.source_dir:
            p = self._local(ctx)
            if not os.path.exists(p):
                return iter(())
            if p.endswith(".zip"):
                return _text_lines(_zip_member_stream(open(p, "rb")))
            if p.endswith(".gz"):
                return _text_lines(_raw_chunks(gzip.open(p, "rb")))
            return _text_lines(_raw_chunks(open(p, "rb")))
        url = self.url(ctx)
        req = urllib.request.Request(url, headers=UA)
        r = urllib.request.urlopen(req, timeout=SOCKET_TIMEOUT)
        if url.endswith(".zip"):
            return _text_lines(_zip_member_stream(r))
        return _text_lines(_raw_chunks(r))

    def fetch_stream(self, ctx):
        # THE STREAM REPORTS ON ITSELF EVERY `SOCAT_REPORT_ROWS` SOURCE ROWS,
        # not every kept row. A one-month window keeps a few thousand rows out
        # of forty million, so a progress counter keyed to the OUTPUT would be
        # silent for the whole half-hour the 1.4 GB takes — which is exactly
        # the "slow or stuck?" question ml/CLAUDE.md §5.25 exists to answer.
        lines = self._lines(ctx)
        t0 = time.time()
        col = None
        counts = {"rows_read": 0, "drop_dataset_qc": 0, "drop_woce": 0,
                  "drop_no_fco2": 0, "drop_no_position": 0,
                  "drop_out_of_range": 0, "preamble_lines": 0}
        oob = np.zeros(self.C, np.int64)
        lo_b, hi_b = self.bounds()
        buf = {}
        for line in lines:
            if col is None:
                counts["preamble_lines"] += 1
                if line.startswith(SOCAT_HEADER_FIRST) and "\tyr\t" in line:
                    names = [h.split(" [")[0].strip()
                             for h in line.rstrip("\n").split("\t")]
                    col = {n: i for i, n in enumerate(names)}
                    miss = [n for n in SOCAT_WANT if n not in col]
                    if miss:
                        raise ValueError(
                            f"the SOCAT data header is missing {miss}; it read "
                            f"{names}. E-079 §3.3 must be re-verified.")
                    ctx.socat_columns = names
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < len(col):
                continue
            counts["rows_read"] += 1
            if counts["rows_read"] % SOCAT_REPORT_ROWS == 0:
                el = max(time.time() - t0, 1e-6)
                ctx.prog.item(
                    f"socat stream {counts['rows_read'] // 1000:,}k rows",
                    extra={"rows_read": counts["rows_read"],
                           "rows_per_s": round(counts["rows_read"] / el),
                           "elapsed_s": round(el, 1),
                           "kept_so_far": sum(len(b["t"])
                                              for b in buf.values())})
            if p[col["QC_Flag"]].strip().upper()[:1] not in SOCAT_KEEP_QC:
                counts["drop_dataset_qc"] += 1
                continue
            wf = _num(p[col["fCO2rec_flag"]])
            qc = {2: 1, 3: 3, 4: 4, 9: 0}.get(int(wf) if wf is not None else 9, 0)
            if qc == 0 or qc > ctx.qc_keep:
                counts["drop_woce"] += 1
                continue
            fco2 = _num(p[col["fCO2rec"]])
            if fco2 is None:
                counts["drop_no_fco2"] += 1
                continue
            la = _num(p[col["latitude"]])
            lo_ = _num(p[col["longitude"]])
            if la is None or lo_ is None or abs(la) > 90.0:
                counts["drop_no_position"] += 1
                continue
            try:
                yr = int(p[col["yr"]])
                td = days_since_epoch(dt.datetime(
                    yr, int(p[col["mon"]]), int(p[col["day"]]),
                    int(p[col["hh"]]), int(p[col["mm"]]),
                    min(59, int(float(p[col["ss"]] or 0)))))
            except (ValueError, TypeError):
                counts["drop_out_of_range"] += 1
                continue
            if not (ctx.t_lo <= td <= ctx.t_hi):
                counts["drop_out_of_range"] += 1
                continue
            v = np.array([fco2, _num_nan(p[col["SST"]]),
                          _num_nan(p[col["sal"]]),
                          _num_nan(p[col["PPPP"]])], np.float64)
            bad = np.isfinite(v) & ((v < lo_b) | (v > hi_b))
            oob += bad
            v[bad] = np.nan
            if not np.isfinite(v[0]):
                counts["drop_no_fco2"] += 1
                continue
            b = buf.setdefault(yr, {"t": [], "lat": [], "lon": [], "v": [],
                                    "plat": [], "qc": []})
            b["t"].append(td)
            b["lat"].append(la)
            b["lon"].append(float(f10.wrap_lon(lo_)))
            b["v"].append(v)
            b["plat"].append(platform_hash(p[col["Expocode"]]))
            b["qc"].append(qc)
            if len(b["t"]) >= FLUSH_ROWS:
                yield yr, _pack(b["t"], b["lat"], b["lon"], b["v"], b["plat"],
                                b["qc"], self.C), None
                buf[yr] = {"t": [], "lat": [], "lon": [], "v": [], "plat": [],
                           "qc": []}
        if col is None:
            raise ValueError("no SOCAT data header found in the stream — the "
                             "file layout has changed (E-079 §3.3)")
        counts["out_of_bounds"] = {n: int(x) for n, x
                                   in zip(self.channel_names, oob) if x}
        for yr, b in sorted(buf.items()):
            if b["t"]:
                yield yr, _pack(b["t"], b["lat"], b["lon"], b["v"], b["plat"],
                                b["qc"], self.C), None
        yield None, None, counts


# -------------------------------------------------------------- slatrack ----
# VERIFIED WITHOUT CREDENTIALS 2026-09-13. The Copernicus Marine STAC metadata
# is public: `https://stac.marine.copernicus.eu/metadata/
# SEALEVEL_GLO_PHY_L3_MY_008_062/product.stac.json` lists 29 per-mission
# datasets, and each dataset's own `dataset.stac.json` gives its time span and
# its `cube:variables` — `sla_filtered`, `sla_unfiltered`, `mdt`, `dac`,
# `internal_tide`, `lwe`, `ocean_tide`, all in metres. So the dataset ids and
# the VARIABLE NAMES below are measured, not assumed. What is NOT verified is
# the download itself: this sandbox has no credentials (deliberately) and
# cannot install the toolbox (pypi answers 403 through the egress proxy).
CMEMS_PRODUCT = "SEALEVEL_GLO_PHY_L3_MY_008_062"
CMEMS_STAC = ("https://stac.marine.copernicus.eu/metadata/"
              f"{CMEMS_PRODUCT}/product.stac.json")
CMEMS_ENV = ("COPERNICUSMARINE_SERVICE_USERNAME",
             "COPERNICUSMARINE_SERVICE_PASSWORD")
CMEMS_VARS = ("sla_filtered", "sla_unfiltered", "mdt")


class SLATrackAdapter(SourceAdapter):
    """Copernicus Marine level-3 along-track sea-level anomaly, all missions.

    Channels are E-079 §3.4's three: `sla` (the FILTERED anomaly, whose archive
    variable is `sla_filtered` — the plan's short name, the archive's real
    name, both recorded in store.json), `sla_unfiltered`, and `mdt`, the mean
    dynamic topography at the point, so `adt = sla + mdt` is one addition
    rather than a second product.

    CREDENTIALS. The adapter reads `COPERNICUSMARINE_SERVICE_USERNAME` and
    `COPERNICUSMARINE_SERVICE_PASSWORD` FROM THE ENVIRONMENT and refuses with a
    clear message when either is unset. It never reads a credential file, never
    looks for one, and never puts a credential on a command line — the box's
    host has root and argv is world readable (ml/CLAUDE.md §6).

    PER YEAR, PUBLISHED PER YEAR. ~3e9 rows over 32 years is tens of GB; a year
    is a part and the assemble step is the only thing that ever sees all of
    them. This is the one store E-079 §3.4 sends to a box rather than a hosted
    runner.
    """

    store = "slatrack"
    title = "Copernicus Marine along-track sea level anomaly (L3, all missions)"
    channels = (("sla", "m", -3.0, 3.0),
                ("sla_unfiltered", "m", -3.0, 3.0),
                ("mdt", "m", -3.0, 3.0))
    log2_fp = -2.0                       # log2(7 km / 27.83 km) = -1.99
    log2_dt = -4.0                       # a 1 Hz sample
    first_year = 1993
    qc_policy = (
        "The DUACS level-3 product is already quality controlled and edited "
        "upstream; it publishes no per-sample flag, so every kept row carries "
        "qc 1 and the store says so rather than inventing a grade. A sample "
        "whose `sla_filtered` is the netCDF fill or outside +/- 3 m is dropped "
        "and counted; `sla_unfiltered` and `mdt` outside the same bounds "
        "become NaN for that channel alone. Rows are kept per mission and the "
        "mission is the platform, so a consumer can hold out a mission the way "
        "it holds out a year.")
    sources = (f"copernicusmarine subset --dataset-id "
               f"cmems_obs-sl_glo_phy-ssh_my_<mission>-l3-duacs_PT1S_<ver>",
               CMEMS_STAC + " (public metadata, no login)")
    verified = ("2026-09-13: the product's 29 per-mission dataset ids and the "
                "variable names sla_filtered / sla_unfiltered / mdt read from "
                "the PUBLIC STAC metadata with no credentials. The download "
                "path itself is UNVERIFIED from this sandbox — no credentials "
                "here by design, and the copernicusmarine toolbox cannot be "
                "installed (pypi 403 through the egress proxy).")
    notes = ("the only store of the four that needs credentials and a box "
             "rather than a hosted runner")

    def missions(self, ctx):
        """The per-mission dataset ids, from the PUBLIC STAC metadata."""
        if getattr(ctx, "_cmems_missions", None) is not None:
            return ctx._cmems_missions
        out = []
        if ctx.source_dir:
            d = os.path.join(ctx.source_dir, "slatrack")
            out = sorted(os.listdir(d)) if os.path.isdir(d) else []
        else:
            raw = http_bytes(CMEMS_STAC)
            js = json.loads(raw)
            for ln in js.get("links", []):
                if ln.get("rel") == "item" and ln.get("href", "").endswith(
                        "/dataset.stac.json"):
                    out.append({"id": ln["href"].split("/")[0],
                                "title": ln.get("title", "")})
        ctx._cmems_missions = out
        return out

    def index(self, ctx):
        ms = self.missions(ctx)
        creds = {k: bool(os.environ.get(k)) for k in CMEMS_ENV}
        toolbox = True
        try:
            import copernicusmarine                             # noqa: F401
        except ImportError:
            toolbox = False
        return {"product": CMEMS_PRODUCT, "stac": CMEMS_STAC,
                "missions": ms, "n_missions": len(ms),
                "variables": list(CMEMS_VARS),
                "credentials_present": creds,
                "toolbox_importable": toolbox,
                "years": list(range(max(ctx.d_lo.year, self.first_year),
                                    ctx.d_hi.year + 1))}

    def _require_credentials(self):
        missing = [k for k in CMEMS_ENV if not os.environ.get(k)]
        if missing:
            sys.exit(
                f"slatrack needs Copernicus Marine credentials and {missing} "
                f"{'is' if len(missing) == 1 else 'are'} not set in the "
                f"environment. Set {' and '.join(CMEMS_ENV)} as environment "
                f"variables for the life of this command (never on the command "
                f"line, never written to disk, never committed). The workflow "
                f"passes them from repository secrets of the same names. "
                f"Refusing before anything is fetched rather than failing "
                f"after the first year (ml/CLAUDE.md §0.3).")

    def fetch_year(self, ctx, year):
        lo = max(ctx.d_lo, dt.date(year, 1, 1))
        hi = min(ctx.d_hi, dt.date(year, 12, 31))
        if not ctx.source_dir:
            self._require_credentials()
        for m in self.missions(ctx):
            mid = m["id"] if isinstance(m, dict) else str(m)
            paths, owned = self._files(ctx, mid, lo, hi, year)
            for p in paths:
                try:
                    rows, counts = self._read_nc(ctx, p, mid)
                finally:
                    if owned and os.path.exists(p):
                        os.remove(p)
                yield f"{year} {mid}", rows, counts

    def _files(self, ctx, mid, lo, hi, year):
        if ctx.source_dir:
            # `<source-dir>/slatrack/<mission>/<year>/*.nc` — the same shape
            # the toolbox writes below, so a fixture and a real pull differ in
            # nothing the parser can see. A mission with no directory for this
            # year simply did not fly then.
            d = os.path.join(ctx.source_dir, "slatrack", mid, str(year))
            if not os.path.isdir(d):
                return [], False
            return [os.path.join(d, n) for n in sorted(os.listdir(d))
                    if n.endswith(".nc")], False
        import copernicusmarine
        out = os.path.join(ctx.scratch, "slatrack", mid, str(year))
        os.makedirs(out, exist_ok=True)
        # The toolbox reads the credentials from the environment; nothing here
        # passes them, prints them, or writes them anywhere.
        copernicusmarine.subset(
            dataset_id=mid, variables=list(CMEMS_VARS),
            start_datetime=f"{lo:%Y-%m-%d}T00:00:00",
            end_datetime=f"{hi:%Y-%m-%d}T23:59:59",
            output_directory=out, output_filename=f"{mid}_{year}.nc",
            overwrite=True, disable_progress_bar=True)
        return [os.path.join(out, n) for n in sorted(os.listdir(out))
                if n.endswith(".nc")], True

    def _read_nc(self, ctx, path, mid):
        import netCDF4 as ncdf
        ds = ncdf.Dataset(path)
        try:
            names = {k: k for k in ds.variables}
            tvar = "time" if "time" in names else None
            if tvar is None:
                raise ValueError(f"{path} has no `time` variable")
            t = np.asarray(ds.variables[tvar][:], np.float64)
            units = str(getattr(ds.variables[tvar], "units", ""))
            la = np.asarray(ds.variables["latitude"][:], np.float64)
            lo_ = np.asarray(ds.variables["longitude"][:], np.float64)
            cols = []
            for v in CMEMS_VARS:
                if v in ds.variables:
                    cols.append(np.asarray(ds.variables[v][:], np.float64))
                else:
                    cols.append(np.full(len(t), np.nan))
        finally:
            ds.close()
        td = _cf_time_to_days(t, units)
        v = np.stack(cols, axis=1)
        v[np.abs(v) > 1e17] = np.nan               # the netCDF fill, whatever it is
        lo_b, hi_b = self.bounds()
        bad = np.isfinite(v) & ((v < lo_b[None, :]) | (v > hi_b[None, :]))
        oob = bad.sum(axis=0)
        v[bad] = np.nan
        keep = (np.isfinite(td) & np.isfinite(la) & np.isfinite(lo_)
                & (np.abs(la) <= 90.0) & np.isfinite(v[:, 0])
                & (td >= ctx.t_lo) & (td <= ctx.t_hi))
        n = int(keep.sum())
        counts = {"rows_read": int(len(td)), "kept": n,
                  "out_of_bounds": {nm: int(x) for nm, x
                                    in zip(self.channel_names, oob) if x}}
        rows = _pack(td[keep], la[keep], lo_[keep], v[keep],
                     np.full(n, platform_hash(mid), np.int64),
                     np.ones(n, np.uint8), self.C)
        return rows, counts


ADAPTERS = {a.store: a for a in
            (GDPAdapter, GTMBAAdapter, SOCATAdapter, SLATrackAdapter)}


# ================================================================== parsing ==
def _num(s):
    """A CSV/TSV field -> float, or None for empty / NaN / the -999999 fill."""
    if s is None:
        return None
    s = s.strip()
    if not s or s in ("NaN", "nan", "NA", "null"):
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if not np.isfinite(v) or v <= GDP_FILL + 1.0 or abs(v) >= 1e30:
        return None
    return v


def _num_nan(s):
    v = _num(s)
    return np.nan if v is None else v


def _platform_int(s):
    """A drifter id -> int64. Numeric ids stay themselves; anything else hashes."""
    s = (s or "").strip()
    if s.isdigit():
        return int(s)
    return platform_hash(s) if s else 0


def _cf_time_to_days(t, units):
    """A CF `<unit> since <date>` axis -> days since 1982-01-01, fractional."""
    u = (units or "").lower().strip()
    if " since " not in u:
        raise ValueError(f"time units {units!r} carry no epoch — refusing to "
                         f"guess one")
    unit, base = u.split(" since ", 1)
    base = base.strip().replace("t", " ").replace("z", "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            b = dt.datetime.strptime(base[:len("2000-01-01 00:00:00")], fmt)
            break
        except ValueError:
            b = None
    if b is None:
        raise ValueError(f"cannot parse the epoch out of {units!r}")
    per_day = {"days": 1.0, "day": 1.0, "hours": 24.0, "hour": 24.0,
               "minutes": 1440.0, "minute": 1440.0,
               "seconds": 86400.0, "second": 86400.0,
               "milliseconds": 86400e3, "microseconds": 86400e6}.get(unit)
    if per_day is None:
        raise ValueError(f"unknown time unit {unit!r} in {units!r}")
    off = (b - dt.datetime(START.year, START.month, START.day)).total_seconds() \
        / 86400.0
    return np.asarray(t, np.float64) / per_day + off


def _pack(t, lat, lon, values, platform, qc, C):
    """Lists (or arrays) -> the seven store columns, bins computed, lon wrapped.

    THE BIN IS COMPUTED FROM THE float32 TIME THAT IS ACTUALLY STORED, not
    from the float64 the parser had. That is not fussiness: `time_days` runs to
    ~16,000 days and float32 resolves about 0.001 d there, so a timestamp a
    microsecond before a pentad boundary can round UP across it. If `bin` were
    computed at full precision and the stored time rounded the other way, the
    row would sit in bin b with a timestamp that reads as bin b+1 — and the
    reader, which asks for `dt_days = 5*(b+1) - t >= 0`, would silently refuse
    to return it for its own anchor. A row the index says is there and the
    search will never return is exactly the kind of fault that leaves no trace.
    Making the store self-consistent costs nothing and removes the class.
    """
    n = len(t)
    if n == 0:
        return empty_rows(C)
    td = np.asarray(np.asarray(t, np.float64).astype(np.float32), np.float64)
    b = f10.bin_of_days(td)
    if b.size and (b.min() < BIN_MIN_INT16 or b.max() > BIN_MAX_INT16):
        raise ValueError(f"bin {b.min()}..{b.max()} does not fit int16 — the "
                         f"axis has outgrown the column's dtype")
    v = np.asarray(values, np.float64).reshape(n, C)
    return {
        "bin": b.astype(np.int16),
        "time_days": td.astype(np.float32),
        "lat": np.asarray(lat, np.float64).astype(np.float32),
        "lon": f10.wrap_lon(lon).astype(np.float32),
        "values": v.astype(np.float16),
        "platform": np.asarray(platform, np.int64),
        "qc": np.asarray(qc, np.uint8),
    }


def _raw_chunks(fh, size=CHUNK):
    while True:
        b = fh.read(size)
        if not b:
            fh.close()
            return
        yield b


def _zip_member_stream(fh, size=CHUNK):
    """The single deflate member of a zip, inflated as a byte-chunk generator.

    `zipfile` cannot read from a non-seekable HTTP body, and the SOCAT
    synthesis is a 1.4 GB zip64 member — so the local file header is parsed by
    hand (signature, method, name and extra lengths) and the rest is fed
    through `zlib` in 4 MiB chunks. VERIFIED against the real file 2026-09-13:
    `PK\\x03\\x04`, method 8 (deflate), sizes 0xFFFFFFFF (zip64), member name
    `SOCATv2026.tsv`.
    """
    head = b""
    while len(head) < 30:
        b = fh.read(30 - len(head))
        if not b:
            raise ValueError("truncated zip: no local file header")
        head += b
    if head[:4] != b"PK\x03\x04":
        raise ValueError(f"not a zip (magic {head[:4]!r})")
    import struct
    method = struct.unpack("<H", head[8:10])[0]
    nlen, elen = struct.unpack("<HH", head[26:30])
    extra = b""
    while len(extra) < nlen + elen:
        b = fh.read(nlen + elen - len(extra))
        if not b:
            raise ValueError("truncated zip: no local file name")
        extra += b
    if method == 0:
        for c in _raw_chunks(fh, size):
            yield c
        return
    if method != 8:
        raise ValueError(f"zip compression method {method} is not deflate")
    d = zlib.decompressobj(-zlib.MAX_WBITS)
    while True:
        b = fh.read(size)
        if not b:
            tail = d.flush()
            if tail:
                yield tail
            try:
                fh.close()
            except Exception:                                   # noqa: BLE001
                pass
            return
        out = d.decompress(b)
        if out:
            yield out
        if d.eof:
            try:
                fh.close()
            except Exception:                                   # noqa: BLE001
                pass
            return


def _text_lines(chunks, encoding="utf-8"):
    """Byte chunks -> text lines, without ever holding the whole file."""
    tail = b""
    for c in chunks:
        tail += c
        parts = tail.split(b"\n")
        tail = parts.pop()
        for p in parts:
            yield p.decode(encoding, "replace")
    if tail:
        yield tail.decode(encoding, "replace")


def _socat_header_from(chunks):
    """The 32-column data header out of a stream, skipping the preamble."""
    for line in _text_lines(chunks):
        if line.startswith(SOCAT_HEADER_FIRST) and "\tyr\t" in line:
            return [h.strip() for h in line.rstrip("\n").split("\t")]
    raise ValueError("no SOCAT data header in the prefix that was read")


# ================================================================== context ==
class Ctx:
    """Everything every stage needs: the dates, the paths, the progress file."""

    def __init__(self, a):
        self.a = a
        self.adapter = ADAPTERS[a.store]()
        self.work = os.path.abspath(a.work)
        self.root = os.path.join(self.work, a.store)
        os.makedirs(self.root, exist_ok=True)
        self.source_dir = os.path.abspath(a.source_dir) if a.source_dir else None
        self.scratch = os.path.join(self.root, "src")
        os.makedirs(self.scratch, exist_ok=True)
        self.parts = os.path.join(self.root, "parts")
        os.makedirs(self.parts, exist_ok=True)
        self.store = os.path.join(self.root, a.store)
        self.d_lo = parse_date(a.start) if a.start else \
            dt.date(self.adapter.first_year, 1, 1)
        self.d_hi = parse_date(a.end) if a.end else END
        if self.d_hi < self.d_lo:
            sys.exit(f"--end {self.d_hi} precedes --start {self.d_lo}")
        self.t_lo = days_since_epoch(self.d_lo)
        self.t_hi = days_since_epoch(self.d_hi) + 1.0        # inclusive of the day
        self.b_lo = int(np.floor(self.t_lo / PENTAD_DAYS))
        self.b_hi = int(np.floor((self.t_hi - 1e-9) / PENTAD_DAYS))
        self.years = list(range(self.d_lo.year, self.d_hi.year + 1))
        self.qc_keep = int(getattr(a, "qc_keep", 2) or 2)
        self.prog = Progress(self.root)
        self._cmems_missions = None
        self.socat_columns = None

    def year_dir(self, year):
        return os.path.join(self.parts, str(year))


# ============================================================ part plumbing ==
class PartWriter:
    """Per-year column parts, flushed to disk and THEN marked (§5.21).

    One directory per year, one `.npz` per flush, so a year that does not fit
    in memory (SOCAT's 1980s and 1990s do not) is still one resumable unit.
    """

    def __init__(self, ctx, year):
        self.ctx = ctx
        self.year = year
        self.dir = ctx.year_dir(year)
        self.seq = 0
        self.n = 0
        self.buf = []
        self.counts = {}

    def add(self, rows, counts=None):
        if counts:
            _merge_counts(self.counts, counts)
        if rows is None or len(rows["bin"]) == 0:
            return
        self.buf.append(rows)
        self.n += len(rows["bin"])
        if sum(len(r["bin"]) for r in self.buf) >= FLUSH_ROWS:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        os.makedirs(self.dir, exist_ok=True)
        cat = {k: np.concatenate([r[k] for r in self.buf], axis=0)
               for k in ROW_KEYS}
        p = os.path.join(self.dir, f"{self.seq:05d}.npz")
        tmp = f"{p}.tmp{os.getpid()}.npz"
        np.savez(tmp, **cat)
        os.replace(tmp, p)
        self.seq += 1
        self.buf = []

    def close(self):
        """Write what is left, the year's counts, and only THEN the marker."""
        self.flush()
        os.makedirs(self.dir, exist_ok=True)
        atomic_json(os.path.join(self.dir, "counts.json"),
                    {"year": self.year, "rows": self.n, "parts": self.seq,
                     "counts": self.counts, "at": utcnow()})
        mark(self.ctx.root, f"parts/{self.year}")
        return self.n


def _merge_counts(into, new):
    for k, v in (new or {}).items():
        if isinstance(v, dict):
            sub = into.setdefault(k, {})
            for kk, vv in v.items():
                sub[kk] = sub.get(kk, 0) + vv
        elif isinstance(v, (int, float)):
            into[k] = into.get(k, 0) + v
        else:
            into[k] = v
    return into


def read_parts(ctx):
    """Every part of every year in range, in year order. Yields arrays."""
    for y in ctx.years:
        d = ctx.year_dir(y)
        if not os.path.isdir(d):
            continue
        for n in sorted(os.listdir(d)):
            if not n.endswith(".npz"):
                continue
            with np.load(os.path.join(d, n)) as z:
                yield y, {k: z[k] for k in ROW_KEYS}


# ================================================================== stages ===
def stage_index(ctx):
    ad = ctx.adapter
    ctx.prog.stage_start(f"index {ad.store}", 1)
    plan = ad.index(ctx)
    plan.update({
        "store": ad.store, "title": ad.title,
        "channels": ad.schema(),
        "footprint": {"log2_fp": ad.log2_fp, "log2_dt": ad.log2_dt},
        "qc_policy": ad.qc_policy,
        "sources": list(ad.sources),
        "verified": ad.verified,
        "per_year_resume": bool(ad.per_year),
        "date_range": [str(ctx.d_lo), str(ctx.d_hi)],
        "bins": [ctx.b_lo, ctx.b_hi],
        "years": ctx.years,
        "builder_git_sha": git_sha(), "built_at": utcnow(),
    })
    atomic_json(os.path.join(ctx.root, "plan.json"), plan)
    ctx.prog.item("plan.json", 1, {"years": len(ctx.years)})
    mark(ctx.root, "index")
    print(f"  index: {ad.store} — {len(ctx.years)} year(s), C={ad.C} "
          f"({', '.join(ad.channel_names[:8])}"
          f"{' …' if ad.C > 8 else ''}), bins {ctx.b_lo}..{ctx.b_hi}")
    return plan


def stage_fetch(ctx):
    ad = ctx.adapter
    if ad.per_year:
        ctx.prog.stage_start(f"fetch {ad.store}", len(ctx.years))
        for i, y in enumerate(ctx.years, 1):
            if marked(ctx.root, f"parts/{y}") and not ctx.a.force:
                print(f"  {y}: already fetched — skipping")
                continue
            t0 = time.time()
            pw = PartWriter(ctx, y)
            for label, rows, counts in ad.fetch_year(ctx, y):
                pw.add(rows, counts)
                ctx.prog.item(f"{ad.store} {label}", i,
                              {"year_rows": pw.n})
            n = pw.close()
            print(f"  {y}: {n:,} row(s) in {pw.seq} part(s) "
                  f"({time.time() - t0:.1f}s)")
    else:
        # ONE PASS, ALL YEARS. The marker is written for every year only after
        # the stream has run to its end, so an interrupted pass re-reads the
        # file rather than leaving a half-filled year marked done.
        done = all(marked(ctx.root, f"parts/{y}") for y in ctx.years)
        if done and not ctx.a.force:
            print(f"  {ad.store}: every year already fetched — skipping")
        else:
            for y in ctx.years:
                shutil.rmtree(ctx.year_dir(y), ignore_errors=True)
                p = marker(ctx.root, f"parts/{y}")
                if os.path.exists(p):
                    os.remove(p)
            ctx.prog.stage_start(f"fetch {ad.store} (one stream)", 0)
            writers, total = {}, 0
            t0 = time.time()
            final = {}
            for year, rows, counts in ad.fetch_stream(ctx):
                if year is None:
                    final = counts or {}
                    continue
                w = writers.get(year)
                if w is None:
                    w = writers[year] = PartWriter(ctx, year)
                w.add(rows)
                total += len(rows["bin"])
                if total % (5 * FLUSH_ROWS) < len(rows["bin"]):
                    ctx.prog.item(f"{ad.store} {year}", None,
                                  {"rows": total,
                                   "elapsed_s": round(time.time() - t0, 1)})
            # The stream's counters describe ONE pass over the whole file, so
            # they belong to exactly one part. Giving a copy to every year made
            # `assemble_store` sum them once per non-empty year: the `socat`
            # store published 2026-09-14 records `rows_read` 2,597,074,036,
            # which is 44,018,204 — the file's real row count — times the 59
            # years that held rows. The arrays were never affected; only the
            # ledger in `store.json` was, and only for an adapter that fetches
            # in one stream rather than per year.
            ledger = next((y for y in sorted(writers) if y in ctx.years), None)
            if ledger is None:
                ledger = ctx.years[0] if ctx.years else None
            for year in sorted(writers):
                if year in ctx.years:
                    writers[year].counts = (dict(final) if year == ledger
                                            else {})
                    writers[year].close()
                else:
                    writers[year].flush()
            for y in ctx.years:
                if y not in writers:
                    w = PartWriter(ctx, y)
                    if y == ledger:
                        w.counts = dict(final)
                    w.close()
            print(f"  {ad.store}: {total:,} row(s) over "
                  f"{len(writers)} year(s) in one pass "
                  f"({time.time() - t0:.1f}s)")
    ctx.prog.stage_start(f"{ad.store} store", 1)
    meta = assemble_store(ctx)
    mark(ctx.root, "fetch")
    ctx.prog.item("store", 1, {"N": meta["N"], "bin_first": meta["bin_first"]})
    return meta


def assemble_store(ctx):
    """Concatenate the parts, sort by (bin, time), write CSR offsets and hashes."""
    ad = ctx.adapter
    dest = ctx.store
    os.makedirs(dest, exist_ok=True)
    parts = {k: [] for k in ROW_KEYS}
    per_year, counts_all = {}, {}
    for y, d in read_parts(ctx):
        for k in ROW_KEYS:
            parts[k].append(d[k])
        per_year[y] = per_year.get(y, 0) + int(len(d["bin"]))
    for y in ctx.years:
        per_year.setdefault(y, 0)
        c = read_json(os.path.join(ctx.year_dir(y), "counts.json"), {})
        _merge_counts(counts_all, c.get("counts") or {})

    if parts["bin"]:
        cat = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    else:
        cat = empty_rows(ad.C)
    N = int(len(cat["bin"]))
    # THE DEFINING ORDER: (bin, time_days) ascending. `lexsort` takes its keys
    # last-major, so time is the secondary key.
    order = np.lexsort((cat["time_days"].astype(np.float64),
                        cat["bin"].astype(np.int64)))
    cat = {k: v[order] for k, v in cat.items()}

    b = cat["bin"].astype(np.int64)
    bin_first = int(b.min()) if N else ctx.b_lo
    bin_last = int(b.max()) if N else ctx.b_lo
    n_bins = bin_last - bin_first + 1
    off = f10.csr_offsets(b, bin_first, n_bins)
    assert off[0] == 0 and off[-1] == N, (off[0], off[-1], N)

    fp = np.empty((N, 2), np.float16)
    fp[:, 0] = np.float16(ad.log2_fp)
    fp[:, 1] = np.float16(ad.log2_dt)

    files = {}
    to_write = {"bin": cat["bin"], "time_days": cat["time_days"],
                "lat": cat["lat"], "lon": cat["lon"], "values": cat["values"],
                "platform": cat["platform"], "qc": cat["qc"], "fp": fp,
                "bin_offsets": off}
    for k, arr in to_write.items():
        p = os.path.join(dest, k + ".npy")
        np.save(p, arr)
        files[k + ".npy"] = p

    live = int((np.diff(off) > 0).sum())
    finite = np.isfinite(cat["values"].astype(np.float32))
    per_channel = {}
    for i, (nm, unit, lo_b, hi_b) in enumerate(ad.channels):
        col = cat["values"][:, i].astype(np.float32)
        f = np.isfinite(col)
        per_channel[nm] = {
            "unit": unit, "measured": int(f.sum()),
            "fraction": round(float(f.mean()), 6) if N else 0.0,
            "min": float(np.nanmin(col)) if f.any() else None,
            "max": float(np.nanmax(col)) if f.any() else None,
            "mean": float(np.nanmean(col)) if f.any() else None,
            "bounds": [lo_b, hi_b]}

    plan = read_json(os.path.join(ctx.root, "plan.json"), {})
    meta = {
        "family": FAMILY, "tier": "P", "store": ad.store, "title": ad.title,
        "schema": {
            "bin.npy": {"dtype": "int16", "shape": [N],
                        "meaning": "floor((date - 1982-01-01)/5 days); "
                                   "NEGATIVE before 1982 and kept"},
            "time_days.npy": {"dtype": "float32", "shape": [N],
                              "meaning": "days since 1982-01-01, fractional"},
            "lat.npy": {"dtype": "float32", "shape": [N],
                        "meaning": "degrees north"},
            "lon.npy": {"dtype": "float32", "shape": [N],
                        "meaning": "degrees east, in [-180, 180)"},
            "values.npy": {"dtype": "float16", "shape": [N, ad.C],
                           "meaning": "the channels below, RAW units, "
                                      "NaN = not measured"},
            "platform.npy": {"dtype": "int64", "shape": [N],
                             "meaning": "platform id; see platform_id"},
            "qc.npy": {"dtype": "uint8", "shape": [N],
                       "meaning": "0 not assessed, 1 good, 2 probably good, "
                                  "3+ the source's worse grades"},
            "fp.npy": {"dtype": "float16", "shape": [N, 2],
                       "meaning": "(log2_fp, log2_dt), E-078 §2, per row"},
            "bin_offsets.npy": {"dtype": "int64", "shape": [n_bins + 1],
                                "meaning": "CSR: rows of bin b are "
                                           "[off[b - bin_first], "
                                           "off[b - bin_first + 1])"},
        },
        "channels": ad.schema(),
        "C": ad.C,
        "N": N, "n_bins": n_bins, "n_live_bins": live,
        "bin_first": bin_first, "bin_last": bin_last,
        "epoch": str(START), "pentad_days": PENTAD_DAYS,
        "footprint": {"log2_fp": ad.log2_fp, "log2_dt": ad.log2_dt,
                      "note": "E-078 §2: log2(footprint_km / 27.83) and "
                              "log2(support_days / 5), clamped to +/-12; "
                              "constant for this source but stored per row so "
                              "every tier-P store reads identically"},
        "normalisation": "RAW — not z-scored and not anomalised; the sampler "
                         "standardises from training years only",
        "qc_policy": ad.qc_policy,
        "qc_keep_max": ctx.qc_keep,
        "platform_id": ("a numeric source id where the archive has one, else "
                        "int(sha1(id)[:15], 16) — deterministic across machines "
                        "and python versions, 60 bits, never negative"),
        "date_range": [str(ctx.d_lo), str(ctx.d_hi)],
        "bins_requested": [ctx.b_lo, ctx.b_hi],
        "per_year": {str(k): int(v) for k, v in sorted(per_year.items())},
        "counts": counts_all,
        "per_channel": per_channel,
        "values_measured_fraction": (round(float(finite.mean()), 6)
                                     if N else 0.0),
        "resume_granularity": "year" if ad.per_year else
                              "the whole stream (the source file is sorted by "
                              "expocode, not by time — see the module docstring)",
        "sources": list(ad.sources),
        "verified": ad.verified,
        "plan": {k: plan.get(k) for k in
                 ("dataset", "datasets", "version", "url", "bytes",
                  "n_columns", "columns", "missions", "n_missions",
                  "credentials_present", "toolbox_importable", "preflight")
                 if k in plan},
        "source_dir": (f"file://{ctx.source_dir}" if ctx.source_dir else None),
        "builder": "ml/build_family10_stores.py",
        "builder_git_sha": git_sha(),
        "built_at": utcnow(),
    }
    if ad.notes:
        meta["notes"] = ad.notes
    meta["sha256"] = {n: sha256(p) for n, p in sorted(files.items())}
    atomic_json(os.path.join(dest, "store.json"), meta)
    check_store(dest, ad)
    print(f"  store: {N:,} row(s), C={ad.C}, bins {bin_first}..{bin_last} "
          f"({live:,} live) -> {dest}")
    return meta


# ============================================================== assertions ===
def check_store(path, adapter=None, anchor=None):
    """E-079 §4's assertions, run on the store before anybody trusts it.

    Every one of these is a property a broken build can have while looking
    completely ordinary from the outside, which is the only reason they are
    worth the seconds they cost.
    """
    st = f10.Store(path)
    b = np.asarray(st["bin"], np.int64)
    t = np.asarray(st["time_days"], np.float64)
    if st.N:
        assert np.all(np.diff(b) >= 0), "rows are not sorted by bin"
        same = b[1:] == b[:-1]
        assert np.all(t[1:][same] >= t[:-1][same]), \
            "rows inside a bin are not sorted by time"
        # The bin column must BE the bin of the time column — a store whose
        # index and timestamps disagree answers every search with the wrong
        # pentad and nothing says so.
        assert np.array_equal(b, f10.bin_of_days(t)), \
            "bin.npy disagrees with floor(time_days / 5)"
        lon = np.asarray(st["lon"], np.float64)
        assert np.all((lon >= -180.0) & (lon < 180.0)), \
            f"lon runs {lon.min()}..{lon.max()}, not [-180, 180)"
        lat = np.asarray(st["lat"], np.float64)
        assert np.all(np.abs(lat) <= 90.0), "lat outside [-90, 90]"
    off = st.bin_offsets
    assert off[0] == 0 and off[-1] == st.N, "CSR offsets do not span the rows"
    assert np.all(np.diff(off) >= 0), "bin_offsets is not monotone"
    for bb in (np.unique(b) if st.N else []):
        lo, hi = st._slice(int(bb), int(bb))
        assert np.all(b[lo:hi] == bb), f"bin {bb}'s CSR slice holds other bins"
    v = np.asarray(st["values"], np.float32)
    assert not np.isinf(v).any(), "values.npy holds an infinity"
    fp = np.asarray(st["fp"], np.float32)
    if st.N:
        assert np.all(fp[:, 0] == fp[0, 0]) and np.all(fp[:, 1] == fp[0, 1]), \
            "the footprint columns are not constant for this source"
        assert (f10.LOG2_FP_RANGE[0] <= fp[0, 0] <= f10.LOG2_FP_RANGE[1]), \
            f"log2_fp {fp[0, 0]} outside E-078 §2's clamp"
    if adapter is not None:
        lo_b, hi_b = adapter.bounds()
        fin = np.isfinite(v)
        if fin.any():
            lo_m = np.where(fin, v, np.inf).min(axis=0)
            hi_m = np.where(fin, v, -np.inf).max(axis=0)
            for i, nm in enumerate(adapter.channel_names):
                if not fin[:, i].any():
                    continue
                assert lo_b[i] <= lo_m[i] and hi_m[i] <= hi_b[i], (
                    f"channel {nm} runs {lo_m[i]}..{hi_m[i]}, outside its "
                    f"physical bounds {lo_b[i]}..{hi_b[i]}")
        assert st.C == adapter.C and st.channels == list(adapter.channel_names)
    if anchor is not None and st.N:
        lat0, lon0, bin0 = anchor
        tok = st.knearest(lat0, lon0, bin0, k=5, R_max_km=1000.0,
                          T_max_days=30.0)
        assert np.all(tok["dt_days"][tok["valid"]] >= 0.0), \
            "the search returned an observation from the future"
        assert np.all(b[tok["row"][tok["valid"]]] <= bin0), \
            "the search read past the anchor's bin"
    return st


# ========================================================== stage: publish ===
def stage_publish(ctx):
    """Upload the store, DOWNLOAD EACH FILE BACK and compare sha256.

    Family 7's and family 8's rule, file for file: an upload that returns 200
    is not evidence the bytes are retrievable, so a publish that cannot verify
    FAILS the job.
    """
    from huggingface_hub import hf_hub_download
    ad = ctx.adapter
    dest = ctx.store
    prefix = f"{HF_ROOT}/{ad.store}"
    api, repo, tok = hub_repo()
    names = sorted(n for n in os.listdir(dest) if n.endswith(".npy"))
    names += ["store.json"]
    for n in names:
        if not os.path.exists(os.path.join(dest, n)):
            sys.exit(f"cannot publish: {os.path.join(dest, n)} is missing")
    check_store(dest, ad)
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=False)
    ctx.prog.stage_start(f"publish {ad.store}", len(names))
    entries = []
    scratch = os.path.join(ctx.scratch, "verify")
    # ONE COMMIT for the nine arrays + store.json. The Hub allows 256 commits
    # per repository per hour and `upload_file` is one commit each; a store
    # publish that spent ten of them per store is what put the PSL mirror over
    # the line on 2026-09-14. The restore check below is unchanged.
    digests = {n: sha256(os.path.join(dest, n)) for n in names}
    hub_commit(api, repo,
               hub_add_ops([(f"{prefix}/{n}", os.path.join(dest, n))
                            for n in names]),
               f"family 10 ({ad.store}): {len(names)} file(s)")
    for i, n in enumerate(names, 1):
        p = os.path.join(dest, n)
        src = digests[n]
        shutil.rmtree(scratch, ignore_errors=True)
        back = hf_hub_download(repo, f"{prefix}/{n}", repo_type="dataset",
                               token=tok, local_dir=scratch)
        got = sha256(back)
        shutil.rmtree(scratch, ignore_errors=True)
        if got != src:
            sys.exit(f"RESTORE MISMATCH {n}: uploaded {src}, downloaded {got} "
                     f"— the publish is not trustworthy")
        entries.append({"name": n, "bytes": os.path.getsize(p), "sha256": src})
        ctx.prog.item(n, i, {"sha256": src[:16]})

    sm = read_json(os.path.join(dest, "store.json"), {})
    man = {"family": FAMILY, "tier": "P", "store": ad.store,
           "repo": repo, "prefix": prefix,
           "N": sm.get("N"), "C": sm.get("C"),
           "channels": sm.get("channels"),
           "bin_first": sm.get("bin_first"), "bin_last": sm.get("bin_last"),
           "footprint": sm.get("footprint"),
           "date_range": sm.get("date_range"),
           "builder_git_sha": git_sha(), "built_at": utcnow(),
           "files": entries}
    mp = os.path.join(ctx.root, "manifest.json")
    atomic_json(mp, man)
    hub_upload_with_backoff(api, repo, mp, f"{prefix}/manifest.json",
                            f"family 10 ({ad.store}): manifest")
    mark(ctx.root, "publish")
    print(f"  publish: {len(entries)} file(s) verified by restore -> "
          f"https://huggingface.co/datasets/{repo}/tree/main/{prefix}")
    return man


# =================================================================== driver ==
STAGE_FN = {"index": stage_index, "fetch": stage_fetch, "publish": stage_publish}


def run_stages(ctx, stages):
    for s in stages:
        for dep in DEPS.get(s, []):
            if not marked(ctx.root, dep):
                sys.exit(f"stage {s!r} needs {dep!r} first (E-079 §4: stage "
                         f"order is fixed) — {marker(ctx.root, dep)} is missing")
        if marked(ctx.root, s) and not ctx.a.force:
            print(f"stage {s}: already done — skipping (--force to redo)")
            continue
        t0 = time.time()
        print(f"\n=== stage {s} ({ctx.a.store}) ===", flush=True)
        STAGE_FN[s](ctx)
        print(f"=== stage {s} done in {time.time() - t0:.1f}s ===", flush=True)


# ==================================================================== smoke ==
SMOKE_START = "1981-12-20"     # deliberately BEFORE the epoch: negative bins
SMOKE_END = "1982-01-12"


def make_smoke_sources(root, store, d_lo, d_hi, seed=20260913):
    """Tiny synthetic sources in each archive's REAL on-disk format.

    The point is not that the numbers are realistic; it is that the parser
    meets the same columns, the same fills, the same flags and the same units
    it meets on the box, and that a store built from them can be checked
    against a truth the generator kept. Returns (truth_rows, n_expected).
    """
    rng = np.random.default_rng(seed)
    ad = ADAPTERS[store]()
    days = [d_lo + dt.timedelta(days=k) for k in range((d_hi - d_lo).days + 1)]
    os.makedirs(root, exist_ok=True)
    truth = []

    if store == "gdp":
        base = os.path.join(root, "gdp")
        os.makedirs(base, exist_ok=True)
        months = sorted({(d.year, d.month) for d in days})
        for (y, m) in months:
            p = os.path.join(base, f"{y:04d}-{m:02d}.csv")
            with open(p, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["ID", "WMO", "time (UTC)",
                            "latitude (degrees_north)",
                            "longitude (degrees_east)", "ve", "vn",
                            "err_lat", "err_lon", "sst (degree_C)",
                            "err_sst (degree_C)", "drogue_lost_date (UTC)"])
                for d in [x for x in days if (x.year, x.month) == (y, m)]:
                    for h in (0, 6, 12, 18):
                        for j in range(3):
                            lat = float(rng.uniform(-60, 60))
                            lon = float(rng.uniform(-179, 179))
                            when = dt.datetime(d.year, d.month, d.day, h)
                            u = float(rng.uniform(-0.6, 0.6))
                            v = float(rng.uniform(-0.6, 0.6))
                            sst = float(rng.uniform(0, 30))
                            # j == 2 is a POSITION-ERROR row that must be
                            # dropped: 1 degree of error is 111 km > 50.
                            el = 1.0 if j == 2 else 0.01
                            lost = "" if j == 0 else (
                                "1970-01-01T00:00:00Z" if j == 1
                                else "1981-12-25T00:00:00Z")
                            w.writerow([100000 + j, 1300000 + j,
                                        when.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                        f"{lat:.3f}", f"{lon:.3f}",
                                        f"{u:.5f}", f"{v:.5f}",
                                        f"{el:.3f}", "0.01", f"{sst:.3f}",
                                        "0.04", lost])
                            if j == 2:
                                continue
                            drog = 1.0 if j == 0 else np.nan
                            truth.append({
                                "t": days_since_epoch(when),
                                "lat": lat, "lon": lon,
                                "v": [u, v, sst, drog],
                                "platform": 100000 + j, "qc": 1})
    elif store == "gtmba":
        # Three of the six real datasets, in their real column shape: the
        # surface temperature, the depth-resolved temperature (which is where
        # the eleven standard depths and the "a depth the site does not carry"
        # case live), and the ADCP — whose velocities are cm/s in the archive
        # and must come back as m/s. Each carries one flag-4 sample that must
        # not reach the store.
        base = os.path.join(root, "gtmba")
        names = ad.channel_names
        rec = {}

        def put(day, idx, value):
            key = f"{day:%Y-%m-%d}"
            r = rec.setdefault(key, {
                "t": days_since_epoch(dt.datetime(day.year, day.month,
                                                  day.day, 12)),
                "lat": 0.0, "lon": -140.0, "v": [np.nan] * ad.C,
                "platform": 51311, "qc": 1})
            r["v"][idx] = value

        specs = [
            ("pmelTaoDySst", [("T_25", "degree_C", "QT_5025")], [1.0]),
            ("pmelTaoDyT", [("T_20", "degree_C", "QT_5020")],
             [1.0, 20.0, 40.0, 7.0]),
            ("pmelTaoDyAdcp", [("u_1205", "cm s-1", "QU_5205"),
                               ("v_1206", "cm s-1", "QV_5206")], [10.0]),
        ]
        for ds, vars_, depths in specs:
            d = os.path.join(base, ds)
            os.makedirs(d, exist_ok=True)
            for y in sorted({x.year for x in days}):
                p = os.path.join(d, f"{y}.csv")
                head = ["array", "station", "wmo_platform_code",
                        "longitude (degrees_east)",
                        "latitude (degrees_north)", "time (UTC)", "depth (m)"]
                for v, u, q in vars_:
                    head += [f"{v} ({u})", q]
                with open(p, "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(head)
                    for day in [x for x in days if x.year == y]:
                        ts = f"{day:%Y-%m-%d}T12:00:00Z"
                        for dep in depths:
                            row = ["TAO/TRITON", "0n140w", 51311, 220.0, 0.0,
                                   ts, f"{dep:.1f}"]
                            # depth 40 m carries a flag-4 temperature sample
                            q = 4 if (ds == "pmelTaoDyT" and dep == 40.0) else 1
                            for vi, (v, u, _q) in enumerate(vars_):
                                val = (12.5 + vi) if u == "cm s-1" \
                                    else 25.0 - 0.02 * dep
                                row += [f"{val:.4f}", f"{q:.1f}"]
                            w.writerow(row)
                            if q != 1:
                                continue
                            for vi, (v, u, _q) in enumerate(vars_):
                                val = (12.5 + vi) if u == "cm s-1" \
                                    else 25.0 - 0.02 * dep
                                if ds == "pmelTaoDyT":
                                    if int(dep) not in GTMBA_DEPTHS:
                                        continue   # 7 m: not a standard depth
                                    put(day, names.index(f"t_{int(dep)}m"), val)
                                elif ds == "pmelTaoDyAdcp":
                                    put(day, names.index(
                                        "u_cur" if vi == 0 else "v_cur"),
                                        val / 100.0)
                                else:
                                    put(day, names.index("sst"), val)
        truth = list(rec.values())
    elif store == "socat":
        base = os.path.join(root, "socat")
        os.makedirs(base, exist_ok=True)
        p = os.path.join(base, "SOCAT_smoke.tsv")
        cols = ["Expocode", "version", "Source_DOI", "QC_Flag", "yr", "mon",
                "day", "hh", "mm", "ss", "longitude [dec.deg.E]",
                "latitude [dec.deg.N]", "sample_depth [m]", "sal",
                "SST [deg.C]", "Tequ [deg.C]", "PPPP [hPa]", "Pequ [hPa]",
                "WOA_SSS", "NCEP_SLP [hPa]", "ETOPO2_depth [m]",
                "dist_to_land [km]", "GVCO2 [umol/mol]",
                "xCO2water_equ_dry [umol/mol]", "xCO2water_SST_dry [umol/mol]",
                "pCO2water_equ_wet [uatm]", "pCO2water_SST_wet [uatm]",
                "fCO2water_equ_wet [uatm]", "fCO2water_SST_wet [uatm]",
                "fCO2rec [uatm]", "fCO2rec_src", "fCO2rec_flag"]
        with open(p, "w") as fh:
            fh.write("SOCAT data report created: synthetic\n")
            fh.write("Expocode\tversion\tDataset Name\tPlatform Name\n")
            for i in range(20):
                fh.write(f"EXPO{i:04d}\t2026.0N\tN/A\tsynthetic\n")
            fh.write("\nMissing values are indicated by 'NaN'\n\n")
            fh.write("\t".join(cols) + "\n")
            for d in days:
                for j in range(4):
                    lon360 = 200.0 + j          # -> -160..-157 after wrapping
                    lat = 10.0 + j
                    # j == 3 is an E-flagged cruise and must be dropped;
                    # j == 2 carries WOCE flag 4 and must be dropped too.
                    qcf = "E" if j == 3 else "B"
                    woce = "4" if j == 2 else "2"
                    row = ["EXPO%04d" % j, "2026.0N", "10.x/y", qcf,
                           f"{d.year:04d}", f"{d.month:02d}", f"{d.day:02d}",
                           "06", "30", "00.",
                           f"{lon360:.5f}", f"{lat:.5f}", "NaN",
                           f"{35.0 + 0.1 * j:.3f}", f"{20.0 + j:.3f}", "NaN",
                           f"{1013.0 + j:.3f}", "NaN", "34.9", "1012.0",
                           "4000.", "500.", "400.0", "NaN", "NaN", "NaN",
                           "NaN", "NaN", "NaN",
                           f"{380.0 + j:.3f}", "4", woce]
                    fh.write("\t".join(row) + "\n")
                    if j >= 2:
                        continue
                    truth.append({
                        "t": days_since_epoch(
                            dt.datetime(d.year, d.month, d.day, 6, 30)),
                        "lat": lat,
                        "lon": float(f10.wrap_lon(lon360)),
                        "v": [380.0 + j, 20.0 + j, 35.0 + 0.1 * j,
                              1013.0 + j],
                        "platform": platform_hash("EXPO%04d" % j), "qc": 1})
    elif store == "slatrack":
        import netCDF4 as ncdf
        mid = "cmems_obs-sl_glo_phy-ssh_my_j3-l3-duacs_PT1S_202411"
        # `<mission>/<year>/*.nc` — the shape the toolbox writes, so a fixture
        # and a real pull differ in nothing the parser can see.
        n_per = 5
        for y in sorted({x.year for x in days}):
            ydays = [x for x in days if x.year == y]
            d = os.path.join(root, "slatrack", mid, str(y))
            os.makedirs(d, exist_ok=True)
            n = len(ydays) * n_per
            t = np.array([days_since_epoch(
                dt.datetime(x.year, x.month, x.day, 3)) + 0.001 * k
                for x in ydays for k in range(n_per)])
            lat = np.linspace(-60, 60, n)
            lon = np.linspace(-179, 179, n)
            sla = 0.01 * np.sin(np.arange(n) + y)
            slau = sla + 0.002
            mdt = 0.3 + 0.001 * np.arange(n)
            p = os.path.join(d, f"{mid}_{y}.nc")
            ds = ncdf.Dataset(p, "w", format="NETCDF3_CLASSIC")
            ds.createDimension("time", n)
            tv = ds.createVariable("time", "f8", ("time",))
            tv.units = "days since 1982-01-01 00:00:00"
            tv[:] = t
            for nm, arr in (("latitude", lat), ("longitude", lon),
                            ("sla_filtered", sla), ("sla_unfiltered", slau),
                            ("mdt", mdt)):
                v = ds.createVariable(nm, "f8", ("time",))
                v[:] = arr
            ds.close()
            for k in range(n):
                truth.append({"t": float(t[k]), "lat": float(lat[k]),
                              "lon": float(lon[k]),
                              "v": [float(sla[k]), float(slau[k]),
                                    float(mdt[k])],
                              "platform": platform_hash(mid), "qc": 1})
    else:
        raise ValueError(store)
    return truth


def check_smoke(ctx, truth):
    """The store against the truth the generator kept — order, values, search."""
    ad = ctx.adapter
    st = check_store(ctx.store, ad,
                     anchor=(float(truth[0]["lat"]), float(truth[0]["lon"]),
                             int(np.floor(truth[0]["t"] / PENTAD_DAYS))))
    assert st.N == len(truth), (
        f"the store holds {st.N} row(s), the generator kept {len(truth)} "
        f"(a row that should have been dropped survived, or a good row "
        f"was lost)")
    want = sorted(truth, key=lambda r: (int(np.floor(r["t"] / PENTAD_DAYS)),
                                        np.float32(r["t"])))
    v = np.asarray(st["values"], np.float32)
    for i, w in enumerate(want):
        assert abs(float(st["time_days"][i]) - w["t"]) < 1e-2, (
            i, float(st["time_days"][i]), w["t"])
        assert abs(float(st["lat"][i]) - w["lat"]) < 1e-2
        assert abs(float(st["lon"][i]) - w["lon"]) < 1e-2
        assert int(st["platform"][i]) == int(w["platform"]), (
            i, int(st["platform"][i]), w["platform"])
        got, wv = v[i], np.asarray(w["v"], np.float32)
        assert np.array_equal(np.isfinite(got), np.isfinite(wv)), (
            f"row {i}: measured channels {np.isfinite(got)} != "
            f"{np.isfinite(wv)}")
        f = np.isfinite(wv)
        assert np.allclose(got[f], wv[f], atol=2e-2, rtol=2e-3), (
            f"row {i}: {got[f]} != {wv[f]}")
    # negative bins really are kept, when the window reaches before 1982
    if want and want[0]["t"] < 0:
        assert int(st["bin"][0]) < 0, "a pre-1982 row lost its negative bin"
    return (f"N={st.N} · C={st.C} · bins {st.bin_first}..{st.bin_last} · "
            f"{int((np.diff(st.bin_offsets) > 0).sum())} live bin(s)")


def run_smoke(store, root=None, keep=False, start=SMOKE_START, end=SMOKE_END):
    tmp = root or tempfile.mkdtemp(prefix=f"f10smoke_{store}_")
    src = os.path.join(tmp, "src")
    work = os.path.join(tmp, "work")
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    d_lo, d_hi = parse_date(start), parse_date(end)
    truth = make_smoke_sources(src, store, d_lo, d_hi)
    print(f"smoke     {store}: sources -> {src} "
          f"({len(truth)} truth row(s), {time.time() - t0:.1f}s)")
    ap = argparse.Namespace(store=store, work=work, source_dir=src,
                            start=start, end=end, stage="all", force=False,
                            attempts=1, qc_keep=2, socat_url="", smoke=True)
    ctx = Ctx(ap)
    print(f"axis      bins {ctx.b_lo}..{ctx.b_hi} "
          f"({'NEGATIVE bins in range' if ctx.b_lo < 0 else 'all >= 1982'})")
    run_stages(ctx, ["index", "fetch"])
    out = check_smoke(ctx, truth)
    print(f"smoke     {store} OK in {time.time() - t0:.1f}s — {out}")
    if not keep and root is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return work, truth


# ===================================================================== main ==
def main():
    ap = argparse.ArgumentParser(
        description="Build a family-10 tier-P observation store. See "
                    "ml/plans/E079_family10_point_stores.md §2-§4.")
    ap.add_argument("--store", required=True, choices=sorted(ADAPTERS),
                    help="which source: gdp (surface drifters), gtmba (the "
                         "tropical moored arrays), socat (surface CO2), "
                         "slatrack (along-track sea level)")
    ap.add_argument("--work", default=os.path.join(CACHE, FAMILY),
                    help="the build directory: plan.json, per-year parts, the "
                         "store, markers, progress.json. RE-RUN WITH THE SAME "
                         "VALUE TO RESUME.")
    ap.add_argument("--stage", default="all", choices=["all"] + STAGES,
                    help="one stage, or `all`. Order is fixed: fetch needs "
                         "index, publish needs fetch.")
    ap.add_argument("--start", default="",
                    help="first day (YYYY-MM-DD); default is the source's own "
                         "first year")
    ap.add_argument("--end", default=str(END), help="last day (YYYY-MM-DD)")
    ap.add_argument("--source-dir", default="",
                    help="read every source file from local disk — no network "
                         "at all. Used by the tests and by --smoke.")
    ap.add_argument("--smoke", action="store_true",
                    help="generate tiny synthetic sources in a temp dir and "
                         "run index + fetch end to end, in seconds, with no "
                         "network")
    ap.add_argument("--smoke-dir", default="",
                    help="with --smoke: keep the temp tree here")
    ap.add_argument("--qc-keep", type=int, default=2,
                    help="the worst QC grade kept (default 2 = probably good; "
                         "E-079 §2). Recorded in store.json.")
    ap.add_argument("--socat-url", default="",
                    help="override the SOCAT synthesis url (the index stage "
                         "otherwise reads the release page for it)")
    ap.add_argument("--attempts", type=int, default=3,
                    help="download attempts per file before the year fails. A "
                         "404 or an empty ERDDAP result is not an attempt "
                         "failure — it is a gap in the archive.")
    ap.add_argument("--force", action="store_true",
                    help="redo a stage or a year whose .done marker exists")
    a = ap.parse_args()

    if a.smoke:
        run_smoke(a.store, root=a.smoke_dir or None, keep=bool(a.smoke_dir))
        return 0

    ctx = Ctx(a)
    ad = ctx.adapter
    print(f"store     {ad.store} — {ad.title}")
    print(f"axis      {ctx.d_lo} .. {ctx.d_hi}  bins {ctx.b_lo}..{ctx.b_hi} "
          f"(epoch {START}, {PENTAD_DAYS}-day bins; negative bins are KEPT)")
    print(f"channels  C={ad.C}: {', '.join(ad.channel_names)}")
    print(f"footprint log2_fp {ad.log2_fp:g}, log2_dt {ad.log2_dt:g}")
    print(f"source    {ad.sources[0] if not ctx.source_dir else ctx.source_dir}")
    print(f"verified  {ad.verified}")
    stages = STAGES if a.stage == "all" else [a.stage]
    run_stages(ctx, stages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
