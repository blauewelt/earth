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

SLATRACK IS BUILT ON TWO MACHINES, and neither could do both halves. Its fetch
needs Copernicus credentials, which ml/CLAUDE.md §6 forbids on a rented box, so
it may only run on a GitHub-hosted runner; its store is 50-80 GB (measured
2026-09-14: ~70 M samples in 2015 alone, ~1.5-2.5 e9 rows over 1993-2024 at ~33
bytes a row), which a hosted runner's ~14 GB of disk cannot hold. So
`.github/workflows/family10-slatrack-fetch.yml` fetches ONE YEAR at a time on
hosted lanes and parks its column parts at `partials/family10/slatrack/<year>/`
on the Hub (`ml/family10_parts_hub.py`, done.json written LAST), and a box with
HF_TOKEN and no credentials assembles them:

  python3 ml/build_family10_stores.py --store slatrack --work W \
      --stage index,fetch,publish --parts-from-hub --start 1993-01-01

`--parts-from-hub` replaces the fetch's source with that pull; `--assemble`
chooses between the in-RAM assembler and `assemble_store_streaming`, which
writes the same store BYTE FOR BYTE in three memmap passes and never holds more
than one part (default `auto`: streaming above 50 M rows, and always for
slatrack). Everything AFTER the assembly is bounded the same way: the store's
statistics and E-079 §4's assertion pass both read the finished arrays in row
blocks off their memmaps (`--check-chunk-rows`, default 16 M rows), and the
publish refuses up front if the scratch disk cannot hold one copy of the
largest file its restore check downloads back.

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
import re
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

    TWO LIVE ROUTES, `files` BY DEFAULT (`--slatrack-fetch`). The subset route
    works but is slow: the probe of 2026-09-14 (run 34849670866, ONE WEEK of
    January 2015 for four missions) measured ~100 s per mission-week at
    ~3.3k samples/s — data-rate bound, ~220 runner-hours for 1993-2024. The
    default route therefore takes the ORIGINAL per-day DUACS netCDF files
    through `copernicusmarine.get`, in month-sized batches, parsing and
    deleting each batch before the next is asked for. `dataframe` keeps the
    `read_dataframe` route as a fallback; both end in the same
    `_rows_from_frame`, so the store cannot differ between them.

    WHY NOT THE TOOLBOX'S OWN SUBSET-TO-NETCDF. The two probes of 2026-09-14
    (runs 34847980911 and 34848513007) showed that the toolbox DOES serve
    these sparse datasets, and that asking it for `file_format` netcdf (its
    default) crashes inside its OWN writer:
    `_dataframe_to_netcdf_per_platform` -> `_add_attributes_to_dataset` raises
    `ValueError: index must be monotonic increasing or decreasing`
    (copernicusmarine/download_functions/download_sparse.py). The data
    arrived; only the toolbox's netCDF serialisation failed. So the subset
    route asks for the DataFrame the toolbox already has in hand —
    `copernicusmarine.read_dataframe` — and never lets it write a file.

    THE SUBSET FRAME IS LONG, NOT WIDE (measured, run 34849670866). Its
    columns are ['variable', 'platform_id', 'platform_type', 'time',
    'longitude', 'latitude', 'depth', 'pressure', 'is_depth_from_producer',
    'value', 'value_qc', 'institution', 'doi', 'product_doi'] — ONE ROW PER
    (variable, sample), with `variable` taking the values sla_filtered /
    sla_unfiltered / mdt. A week of one mission is ~950k-1.06M long rows,
    ~320-355k samples. `_rows_from_frame` therefore PIVOTS such a frame to one
    row per (time, latitude, longitude[, platform_id]) with the three channels
    as columns before the shared filter runs; a channel a sample does not
    carry becomes NaN. `value_qc` is counted into a histogram and recorded,
    but NOT filtered on: the product is pre-edited (see `qc_policy`), and the
    next probe should teach us what the flag actually carries first.

    MONTH BY MONTH. A mission-year at 1 Hz is ~30 M samples; at three float64
    channels plus time and position that is several GB in one DataFrame (and
    three times that long). A month keeps the peak inside a 16 GB runner, so
    both routes work one month at a time and yield one part per month.
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
                "the PUBLIC STAC metadata with no credentials. 2026-09-14 "
                "(runs 34847980911 and 34848513007, a hosted runner with real "
                "credentials): the toolbox logs in, resolves these sparse "
                "datasets and SERVES the subset — the run got as far as the "
                "toolbox's own netCDF writer, which then crashed with "
                "`ValueError: index must be monotonic increasing or "
                "decreasing` in _dataframe_to_netcdf_per_platform / "
                "_add_attributes_to_dataset (a bug in the toolbox's "
                "download_sparse.py, not in the data). The adapter therefore "
                "never asks the toolbox to write a subset file. 2026-09-14 "
                "(run 34849670866, one week of January 2015, four missions — "
                "h2a, al, c2, j2): read_dataframe RETURNS DATA, in LONG "
                "format — columns ['variable', 'platform_id', "
                "'platform_type', 'time', 'longitude', 'latitude', 'depth', "
                "'pressure', 'is_depth_from_producer', 'value', 'value_qc', "
                "'institution', 'doi', 'product_doi'], one row per (variable, "
                "sample), `variable` in {sla_filtered, sla_unfiltered, mdt}, "
                "at ~950k-1.06M long rows (~320-355k samples) and ~100 s per "
                "mission-week, i.e. ~3.3k samples/s, which is ~220 runner-"
                "hours for 1993-2024. That measurement is why the DEFAULT "
                "route is now copernicusmarine.get on the original per-day "
                "files, month by month, with read_dataframe kept as "
                "--slatrack-fetch dataframe. Neither route can be run from "
                "this sandbox — no credentials here by design, and the "
                "toolbox cannot be installed (pypi 403 through the egress "
                "proxy); the pivot, the file batching and the frame-to-rows "
                "step they depend on are exercised by the tests and are the "
                "same code the fixture netCDFs go through. NOT YET MEASURED: "
                "the remote path layout of the original files (the first "
                "probe of the `files` route prints it) and what values "
                "`value_qc` takes (the pivot now records its histogram).")
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
            base = CMEMS_STAC.rsplit("/", 1)[0]
            for ln in js.get("links", []):
                if ln.get("rel") == "item" and ln.get("href", "").endswith(
                        "/dataset.stac.json"):
                    mid = ln["href"].split("/")[0]
                    # Each mission's item states WHEN it flew. Read it once
                    # here (public, no login) so the fetch can skip a mission
                    # that was not in orbit for the year asked and clip the
                    # ones that were: asking the toolbox for a window outside
                    # a dataset's coordinates raises CoordinatesOutOfDataset-
                    # Bounds and would fail the whole year (measured on the
                    # first probe, 2026-09-14: Saral/AltiKa geodetic covers
                    # 2015-03-31 -> 2026-01-16, and January 2015 refused).
                    item = json.loads(http_bytes(f"{base}/{ln['href']}"))
                    pr = item.get("properties") or {}
                    out.append({"id": mid, "title": ln.get("title", ""),
                                "start": pr.get("start_datetime"),
                                "end": pr.get("end_datetime")})
        ctx._cmems_missions = out
        return out

    @staticmethod
    def _split_id(mid):
        """`..._PT1S_202411` -> (`..._PT1S`, `202411`). The STAC ids carry
        the version as a suffix; the toolbox wants it as its own argument
        and warns when it is left inside the id."""
        m = re.match(r"^(.*)_(\d{6})$", mid)
        return (m.group(1), m.group(2)) if m else (mid, None)

    @staticmethod
    def _mission_window(m, lo, hi):
        """The part of [lo, hi] this mission actually flew, or None."""
        if not isinstance(m, dict) or not m.get("start"):
            return lo, hi                      # a fixture, or no metadata
        ms = dt.date.fromisoformat(m["start"][:10])
        me = dt.date.fromisoformat(m["end"][:10]) if m.get("end") else hi
        a, b = max(lo, ms), min(hi, me)
        return (a, b) if a <= b else None

    def index(self, ctx):
        ms = self.missions(ctx)
        creds = {k: bool(os.environ.get(k)) for k in CMEMS_ENV}
        toolbox = True
        try:
            import copernicusmarine                             # noqa: F401
        except ImportError:
            toolbox = False
        # `missions()` already carries each mission's coverage window (its STAC
        # start_datetime / end_datetime), so the index records WHEN every
        # mission flew as well as its id — that is what `_mission_window`
        # clips against, and a reader of the index can see the same thing.
        out = {"product": CMEMS_PRODUCT, "stac": CMEMS_STAC,
               "missions": ms, "n_missions": len(ms),
               "variables": list(CMEMS_VARS),
               "fetch": ("copernicusmarine.get (original files), month "
                         "batches; fallback read_dataframe"),
               "fetch_route": self._route(ctx),
               "credentials_present": creds,
               "toolbox_importable": toolbox,
               "years": list(range(max(ctx.d_lo.year, self.first_year),
                                   ctx.d_hi.year + 1))}
        # A MEASUREMENT, not a download: the remote path layout of the
        # original files is unknown, so when the toolbox is here and the
        # credentials are set the index lists one mission's files for the
        # first year of the window and writes what it saw into plan.json. It
        # is wrapped whole — an index must never fail because a probe did.
        if toolbox and all(creds.values()) and not ctx.source_dir:
            try:
                out["slatrack_files_probe"] = self._probe_files(ctx, ms)
            except Exception as e:                              # noqa: BLE001
                print(f"  slatrack files probe failed: {e!r}", flush=True)
        return out

    @staticmethod
    def _route(ctx):
        """`files` (the default) or `dataframe` — see `--slatrack-fetch`."""
        return str(getattr(ctx.a, "slatrack_fetch", "") or "files")

    def _probe_files(self, ctx, ms):
        """One `get(dry_run=True)` for the first mission overlapping the
        window: how many original files there are for its first year, and a
        handful of their remote paths, so the next probe knows the layout."""
        import copernicusmarine
        lo = max(ctx.d_lo, dt.date(self.first_year, 1, 1))
        hi = ctx.d_hi
        for m in ms:
            win = self._mission_window(m, lo, hi)
            if win is None:
                continue
            mid = m["id"] if isinstance(m, dict) else str(m)
            did, ver = self._split_id(mid)
            resp = copernicusmarine.get(
                dataset_id=did, dataset_version=ver,
                filter=f"*{win[0].year}*", dry_run=True,
                disable_progress_bar=True)
            paths = self._response_paths(resp)
            return {"mission": mid, "year": win[0].year,
                    "files_listed": len(paths), "sample_paths": paths[:6]}
        return {"files_listed": 0, "sample_paths": [],
                "note": "no mission overlaps the requested window"}

    @staticmethod
    def _response_paths(resp):
        """A `ResponseGet` -> the remote paths it listed, in order.

        The documented shape is `resp.files`, a list of `FileGet` carrying
        `s3_url` / `https_url` / `filename`; a dict is accepted too so a
        toolbox version that hands back plain JSON does not break the probe.
        """
        files = getattr(resp, "files", None)
        if files is None and isinstance(resp, dict):
            files = resp.get("files")
        out = []
        for f in files or []:
            p = None
            for k in ("s3_url", "https_url", "filename"):
                p = getattr(f, k, None) or (f.get(k) if isinstance(f, dict)
                                            else None)
                if p:
                    break
            if p:
                out.append(str(p))
        return out

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
            win = self._mission_window(m, lo, hi)
            if win is None:
                continue                       # not in orbit this year
            if ctx.source_dir:
                for p in self._files(ctx, mid, year):
                    rows, counts = self._read_nc(ctx, p, mid)
                    yield f"{year} {mid}", rows, counts
                continue
            route = self._route(ctx)
            gen = (self._fetch_months(ctx, mid, win[0], win[1])
                   if route == "dataframe"
                   else self._fetch_files(ctx, mid, win[0], win[1]))
            for part in gen:
                yield part

    def _files(self, ctx, mid, year):
        """The FIXTURE path only: `<source-dir>/slatrack/<mission>/<year>/*.nc`.

        A mission with no directory for this year simply did not fly then. The
        live path writes no file at all any more (see `_fetch_months`), so this
        is the one place that still deals in paths.
        """
        d = os.path.join(ctx.source_dir, "slatrack", mid, str(year))
        if not os.path.isdir(d):
            return []
        return [os.path.join(d, n) for n in sorted(os.listdir(d))
                if n.endswith(".nc")]

    @staticmethod
    def _months(lo, hi):
        """[lo, hi] -> the calendar months it touches, each clipped to it."""
        out = []
        a = dt.date(lo.year, lo.month, 1)
        while a <= hi:
            nxt = dt.date(a.year + (a.month == 12),
                          1 if a.month == 12 else a.month + 1, 1)
            out.append((max(a, lo), min(nxt - dt.timedelta(days=1), hi)))
            a = nxt
        return out

    def _fetch_months(self, ctx, mid, lo, hi):
        """The LIVE path: one `read_dataframe` call per month, no file written.

        The toolbox reads the credentials from the environment; nothing here
        passes them, prints them, or writes them anywhere. `read_dataframe`
        returns the same rows `subset` would have serialised — and skips the
        netCDF writer that crashed on both probes of 2026-09-14 (see the class
        docstring). One month at a time because a mission-year at 1 Hz is
        ~30 M rows and the runner has 16 GB.
        """
        import copernicusmarine
        did, ver = self._split_id(mid)
        for m_lo, m_hi in self._months(lo, hi):
            frame = copernicusmarine.read_dataframe(
                dataset_id=did, dataset_version=ver,
                variables=list(CMEMS_VARS),
                start_datetime=f"{m_lo:%Y-%m-%d}T00:00:00",
                end_datetime=f"{m_hi:%Y-%m-%d}T23:59:59",
                disable_progress_bar=True)
            try:
                mb = float(frame.memory_usage(deep=True).sum()) / 1e6
            except Exception:                                   # noqa: BLE001
                mb = float("nan")
            print(f"  {m_lo:%Y-%m} {mid}: {len(frame)} row(s), {mb:.1f} MB",
                  flush=True)
            rows, counts = self._rows_from_frame(ctx, frame, mid)
            del frame                 # before the next month is asked for
            yield f"{m_lo:%Y-%m} {mid}", rows, counts

    def _fetch_files(self, ctx, mid, lo, hi):
        """The DEFAULT live path: the ORIGINAL per-day files, month batches.

        The subset route is data-rate bound at ~3.3k samples/s (run
        34849670866) — ~220 runner-hours for the whole archive. The original
        DUACS files carry the same samples in their native netCDF and come
        down at storage speed, so this is the route the builder takes unless
        `--slatrack-fetch dataframe` says otherwise.

        THE REMOTE LAYOUT IS NOT MEASURED YET, so the method does not guess
        it. It LISTS a year first (`get(dry_run=True)`, `filter=*<year>*` —
        the toolbox matches the pattern against the absolute remote path) and
        prints how many files matched and the first and last three paths; the
        next probe's log therefore contains the layout. Batching is then done
        on what the listing SHOWED rather than on a guessed pattern: a file
        whose basename carries an 8-digit date token is put in that month's
        batch, and the batch is downloaded with a `regex` built from exactly
        those basenames. If no listed basename carries such a token the whole
        year is one batch — correct, only coarser — and the log says so.

        Each batch is downloaded flat (`no_directories=True`) into the
        build's own scratch, parsed with `_read_nc`, yielded, and deleted
        before the next batch is asked for, so a mission-year never occupies
        more than a month of files on disk.
        """
        import copernicusmarine
        did, ver = self._split_id(mid)
        base = dict(dataset_id=did, dataset_version=ver,
                    disable_progress_bar=True)
        out = os.path.join(ctx.scratch, "slatrack", mid)
        for year in range(lo.year, hi.year + 1):
            resp = copernicusmarine.get(filter=f"*{year}*", dry_run=True,
                                        **base)
            paths = self._response_paths(resp)
            print(f"  {mid} {year}: {len(paths)} original file(s) listed"
                  + (f"; first {paths[:3]} last {paths[-3:]}" if paths else ""),
                  flush=True)
            if not paths:
                continue
            y_lo = max(lo, dt.date(year, 1, 1))
            y_hi = min(hi, dt.date(year, 12, 31))
            for label, names in self._file_batches(paths, y_lo, y_hi,
                                                   str(year)):
                shutil.rmtree(out, ignore_errors=True)
                os.makedirs(out, exist_ok=True)
                try:
                    copernicusmarine.get(
                        regex="(" + "|".join(re.escape(n) for n in names)
                              + ")$",
                        output_directory=out, no_directories=True,
                        overwrite=True, **base)
                    got = sorted(n for n in os.listdir(out)
                                 if n.endswith(".nc"))
                    print(f"  {label} {mid}: {len(names)} file(s) asked, "
                          f"{len(got)} downloaded", flush=True)
                    for n in got:
                        rows, counts = self._read_nc(ctx,
                                                     os.path.join(out, n), mid)
                        yield f"{label} {mid}", rows, counts
                finally:
                    shutil.rmtree(out, ignore_errors=True)

    @staticmethod
    def _file_batches(paths, lo, hi, fallback):
        """Listed remote paths -> [(label, [basename, ...]), ...], per month.

        The date token is read from the BASENAME (`(\\d{8})`, the shape every
        DUACS along-track file carries) rather than from the directory part,
        which differs between products. A basename with no token cannot be
        placed in a month, so the moment one shows up the whole listing is
        returned as a single batch: a coarser batch is still correct, a
        silently dropped file is not.
        """
        months, order = {}, []
        for p in paths:
            n = str(p).rsplit("/", 1)[-1]
            m = re.search(r"(\d{8})", n)
            if not m:
                return [(fallback, [str(x).rsplit("/", 1)[-1]
                                    for x in paths])]
            try:
                d = dt.date(int(m.group(1)[:4]), int(m.group(1)[4:6]),
                            int(m.group(1)[6:8]))
            except ValueError:
                return [(fallback, [str(x).rsplit("/", 1)[-1]
                                    for x in paths])]
            if not (lo <= d <= hi):
                continue                       # outside the mission window
            k = f"{d:%Y-%m}"
            if k not in months:
                months[k] = []
                order.append(k)
            months[k].append(n)
        return [(k, months[k]) for k in sorted(order)]

    def _rows_from_frame(self, ctx, frame, mid):
        """A DataFrame (live) or a mapping of arrays (fixture) -> store rows.

        ONE filter implementation for both paths. The fixture netCDFs and the
        toolbox's frame differ in how the columns arrive and in nothing else,
        so everything below the column lookup — the fill, the bounds, the keep
        rule, the counts — is written once here.

        LONG OR WIDE. The subset route's frame is LONG — one row per
        (variable, sample), measured on run 34849670866 — so a frame carrying
        `variable` and `value` and none of `CMEMS_VARS` as columns is PIVOTED
        first (`_pivot_long`); a fixture's mapping of arrays is already wide
        and goes straight through. Everything after the pivot is shared.

        The lookup is deliberately loud: it prints the columns it was given
        once per mission, and `sys.exit`s naming them if time/latitude/
        longitude cannot be identified. A probe that silently kept zero rows
        would teach us nothing; one that stops and prints the schema does.
        """
        cols = self._frame_columns(frame)
        low0 = {str(c).lower(): c for c in cols}
        qc_hist = None
        if ("variable" in low0 and "value" in low0
                and not any(v.lower() in low0 for v in CMEMS_VARS)):
            frame, qc_hist = self._pivot_long(frame, mid, low0)
            cols = self._frame_columns(frame)
        if mid not in getattr(self, "_logged_cols", ()):
            if not hasattr(self, "_logged_cols"):
                self._logged_cols = set()
            self._logged_cols.add(mid)
            print(f"  {mid} frame columns: {cols}", flush=True)
        low = {str(c).lower(): c for c in cols}

        def pick(*names):
            for n in names:
                if n in low:
                    return low[n]
            return None

        tcol = pick("time_days", "time", "datetime", "date", "juld",
                    "time_counter")
        lacol = pick("latitude", "lat")
        locol = pick("longitude", "lon", "long")
        # If the index carries the time (the toolbox may hand back a
        # DatetimeIndex rather than a column) reset it and look again.
        if tcol is None and hasattr(frame, "reset_index"):
            try:
                frame = frame.reset_index()
            except Exception:                                   # noqa: BLE001
                pass
            else:
                cols = self._frame_columns(frame)
                low = {str(c).lower(): c for c in cols}
                tcol = pick("time", "datetime", "date", "index",
                            "juld", "time_counter")
                lacol = lacol if lacol is not None else pick("latitude", "lat")
                locol = locol if locol is not None else pick("longitude",
                                                             "lon", "long")
        if tcol is None or lacol is None or locol is None:
            sys.exit(
                f"slatrack: cannot find time/latitude/longitude in the frame "
                f"{mid} returned. Columns present: {list(cols)}. Refusing to "
                f"keep zero rows quietly — name the real columns in "
                f"`_rows_from_frame` (ml/CLAUDE.md §0.3).")

        t = self._column(frame, tcol)
        # `time_days` is the one column already in the store's own units: the
        # netCDF path resolves the axis's CF epoch itself (that epoch is the
        # one thing a frame does not carry) and hands the days straight over,
        # so no value is converted twice.
        td = (np.asarray(t, np.float64) if str(tcol).lower() == "time_days"
              else self._time_days(t))
        la = np.asarray(self._column(frame, lacol), np.float64)
        lo_ = np.asarray(self._column(frame, locol), np.float64)
        cols_v = []
        for v in CMEMS_VARS:
            c = low.get(v.lower())
            if c is None:
                cols_v.append(np.full(len(td), np.nan))
            else:
                cols_v.append(np.asarray(self._column(frame, c), np.float64))
        v = np.stack(cols_v, axis=1)
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
        if qc_hist:
            counts["value_qc"] = qc_hist
        rows = _pack(td[keep], la[keep], lo_[keep], v[keep],
                     np.full(n, platform_hash(mid), np.int64),
                     np.ones(n, np.uint8), self.C)
        return rows, counts

    def _pivot_long(self, frame, mid, low):
        """A LONG frame -> a wide mapping of arrays, plus the qc histogram.

        MEASURED SHAPE (run 34849670866): one row per (variable, sample), the
        value in `value`, its flag in `value_qc`, `variable` in {sla_filtered,
        sla_unfiltered, mdt}. The key of a sample is (time, latitude,
        longitude) — and `platform_id` as well IF the frame carries more than
        one, which is checked here rather than assumed: the subset is asked
        per mission dataset, so one platform is expected, and a second one
        would silently collapse two samples into one if the key ignored it.

        No `pivot_table` and no groupby: the key is factorised once and the
        three channels are SCATTERED into place by that code, which is one
        pass over each column and holds nothing but the output. A duplicate
        (variable, key) row — not expected — keeps the last of them.
        """
        import pandas as pd
        keys = [low[k] for k in ("time", "latitude", "longitude")
                if k in low]
        if len(keys) != 3:
            return frame, None                 # let the loud lookup refuse
        note = ""
        pid = low.get("platform_id")
        if pid is not None:
            try:
                nuniq = int(frame[pid].nunique(dropna=False))
            except Exception:                                   # noqa: BLE001
                nuniq = -1
            if nuniq != 1:
                keys.append(pid)               # or we would merge platforms
                note = f", platform_id {nuniq} distinct -> in the key"
            else:
                note = ", platform_id constant"
        var = np.asarray(frame[low["variable"]].astype(str))
        val = np.asarray(frame[low["value"]], np.float64)
        qc_hist = {}
        qcol = low.get("value_qc")
        if qcol is not None:
            # The histogram is of the CHANNEL THE KEEP RULE USES, sla_filtered
            # — recorded so the next probe tells us what flags this product
            # carries. Nothing filters on it yet: the product is pre-edited
            # (see `qc_policy`) and inventing a threshold before seeing the
            # distribution would drop rows for no measured reason.
            sel = frame[qcol][var == CMEMS_VARS[0]]
            for k, c in sel.value_counts(dropna=False).items():
                try:
                    kk = int(k)
                except (TypeError, ValueError):
                    kk = str(k)
                qc_hist[kk] = qc_hist.get(kk, 0) + int(c)
        codes, uniq = pd.MultiIndex.from_frame(frame[keys]).factorize()
        codes = np.asarray(codes, np.int64)
        n = len(uniq)
        wide = {str(k): uniq.get_level_values(i).to_numpy()
                for i, k in enumerate(keys)}
        for v in CMEMS_VARS:
            col = np.full(n, np.nan)
            m = (var == v) & (codes >= 0)
            if m.any():
                col[codes[m]] = val[m]
            wide[v] = col
        print(f"  {mid} long frame: {len(var)} row(s) -> {n} sample(s)"
              f"{note}, value_qc {qc_hist or 'absent'}", flush=True)
        return wide, qc_hist

    @staticmethod
    def _frame_columns(frame):
        """The column names of a DataFrame or of a plain mapping of arrays."""
        c = getattr(frame, "columns", None)
        return list(c) if c is not None else list(frame.keys())

    @staticmethod
    def _column(frame, name):
        """One column, as something numpy can take, from either shape."""
        col = frame[name]
        return getattr(col, "to_numpy", lambda: col)()

    @staticmethod
    def _time_days(t):
        """A time column -> days since START (1982-01-01), fractional float64.

        Two shapes, because the toolbox's frame is datetime64 and a netCDF axis
        is a CF number. A datetime64 column is converted through the SAME
        `_cf_time_to_days` the netCDF path uses, with the epoch it already
        carries, so the two paths cannot drift apart: nanoseconds -> seconds
        since 1970 -> days since START. Timezone-aware input is moved to UTC
        and made naive first — the store's axis is naive UTC throughout.
        """
        arr = np.asarray(t)
        if arr.dtype.kind == "M":
            ns = arr.astype("datetime64[ns]").astype(np.int64)
            ns = ns.astype(np.float64)
            ns[np.asarray(np.isnat(arr.astype("datetime64[ns]")))] = np.nan
            return _cf_time_to_days(ns / 1e9,
                                    "seconds since 1970-01-01 00:00:00")
        if arr.dtype.kind == "O":
            # a tz-aware pandas column arrives as object under .to_numpy();
            # ask pandas to normalise it rather than guessing here.
            import pandas as pd
            s = pd.to_datetime(pd.Series(arr), utc=True)
            arr = s.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
            return SLATrackAdapter._time_days(arr)
        return _cf_time_to_days(np.asarray(arr, np.float64),
                                "seconds since 1970-01-01 00:00:00")

    def _read_nc(self, ctx, path, mid):
        """The FIXTURE path: a netCDF -> the same frame shape, same filter.

        The variables are read into a mapping of arrays and handed to
        `_rows_from_frame`, so the fixture and the live pull share one filter
        implementation and a change to the keep rule cannot reach one path
        without the other. The CF `units` of the axis are honoured HERE, since
        they are the one thing a DataFrame does not carry, and the axis goes in
        as `time_days` — days since START, the store's own convention, which
        `_rows_from_frame` takes as-is rather than converting a second time.

        HARDENED FOR THE REAL DUACS FILES, which the `files` route now hands
        it: the fill is taken from the MASK (`np.ma.filled(..., nan)`) rather
        than left as the raw fill value — these variables are scaled int16, so
        a raw fill would arrive as something like -2e5 and be counted as an
        out-of-bounds sample instead of a missing one; variable names are
        matched case-insensitively; the axis epoch may be any CF `<unit> since
        <date>` (DUACS writes "days since 1950-01-01 00:00:00", which
        `_cf_time_to_days` already handles); and a longitude axis published in
        [0, 360) is folded to [-180, 180). The variable names and the time
        units are printed once per mission — a measurement for the next probe,
        since no real file has been opened here yet.
        """
        import netCDF4 as ncdf
        ds = ncdf.Dataset(path)
        try:
            names = {str(k).lower(): k for k in ds.variables}

            def rd(*want):
                for w in want:
                    k = names.get(w)
                    if k is not None:
                        # the MASK is the fill; a scaled int fill must not
                        # survive as a number.
                        return np.ma.filled(
                            np.ma.asarray(ds.variables[k][:]).astype(
                                np.float64), np.nan).ravel()
                return None

            t = rd("time", "juld", "time_counter")
            if t is None:
                raise ValueError(f"{path} has no `time` variable "
                                 f"(has {sorted(ds.variables)})")
            tkey = next(names[w] for w in ("time", "juld", "time_counter")
                        if w in names)
            units = str(getattr(ds.variables[tkey], "units", ""))
            if mid not in getattr(self, "_logged_nc", ()):
                if not hasattr(self, "_logged_nc"):
                    self._logged_nc = set()
                self._logged_nc.add(mid)
                print(f"  {mid} netCDF variables: {sorted(ds.variables)}; "
                      f"time units {units!r}", flush=True)
            la = rd("latitude", "lat")
            lo_ = rd("longitude", "lon", "long")
            if la is None or lo_ is None:
                raise ValueError(f"{path} has no latitude/longitude "
                                 f"(has {sorted(ds.variables)})")
            if np.isfinite(lo_).any() and np.nanmax(lo_) > 180.0:
                lo_ = np.where(lo_ > 180.0, lo_ - 360.0, lo_)
            frame = {"latitude": la, "longitude": lo_}
            for v in CMEMS_VARS:
                got = rd(v.lower())
                frame[v] = np.full(len(t), np.nan) if got is None else got
        finally:
            ds.close()
        frame["time_days"] = _cf_time_to_days(t, units)
        return self._rows_from_frame(ctx, frame, mid)


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
        self.check_chunk = int(getattr(a, "check_chunk_rows", 0)
                               or CHECK_CHUNK_ROWS)
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
    if getattr(ctx.a, "parts_from_hub", False):
        # THE KEYLESS HALF of slatrack's build. The credentialed fetch happens
        # on GitHub-HOSTED lanes (ml/CLAUDE.md §6 forbids CMEMS credentials on
        # a rented box) and parks each finished year under
        # `partials/family10/<store>/<year>/` on the Hub; this branch brings
        # those parts back and hands them straight to the assembler. Nothing
        # here touches the source archive, so HF_TOKEN is the only secret the
        # machine running it ever sees.
        import family10_parts_hub as ph
        ctx.prog.stage_start(f"pull {ad.store} parts", len(ctx.years))
        ph.pull(ad.store, ctx.years, ctx.work,
                allow_missing=bool(getattr(ctx.a, "allow_missing_years",
                                           False)))
    elif ad.per_year:
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
    meta = assemble(ctx)
    mark(ctx.root, "fetch")
    ctx.prog.item("store", 1, {"N": meta["N"], "bin_first": meta["bin_first"]})
    return meta


# ---------------------------------------------------- assembly, two ways ----
# `assemble_store` holds every row in RAM at once; that is fine for gdp, gtmba
# and socat (10^6..10^8 rows) and impossible for slatrack, measured 2026-09-14
# at ~70 M samples in 2015 alone — call it 1.5-2.5 e9 rows over 1993-2024, i.e.
# 50-80 GB in the store and rather more than that while sorting. So there is a
# second assembler, and its ONLY claim is that it writes THE SAME BYTES.
#
# WHY THE TWO ORDERS ARE THE SAME PERMUTATION, exactly:
#   `np.lexsort((time_days, bin))` sorts by bin, then by time_days, and it is
#   STABLE — rows equal in both keys keep their INPUT order, which is the order
#   `read_parts` yields (years ascending, parts in sorted name order).
#   The streaming assembler reproduces that in three passes: it counts rows per
#   bin (pass 1) to get the CSR offsets, scatters every part's rows into its
#   bin's slice IN THAT SAME INPUT ORDER (pass 2), and then sorts each bin's
#   slice by time_days with a STABLE argsort (pass 3). Sorting by bin is what
#   the scatter does; the stable per-bin sort by time breaks ties in the order
#   the scatter laid rows down, i.e. input order. Same permutation, therefore
#   the same bytes — and `tests/test_build_family10_stores.py` proves it by
#   building one synthetic archive and hashing both assemblers' output, with a
#   year that carries duplicate (bin, time_days) rows ACROSS two parts so the
#   tie-break is actually exercised.
STAT_CHUNK = 1 << 22          # rows per statistics block; both assemblers use it
CHECK_CHUNK_ROWS = 16_000_000  # rows per block in `check_store`
STREAM_ROWS = 50_000_000      # `--assemble auto` switches above this
# bytes per stored row, excluding the tiny bin_offsets vector: bin 2 +
# time_days 4 + lat 4 + lon 4 + platform 8 + qc 1 + fp 4 + values 2*C.
ROW_BYTES_FIXED = 2 + 4 + 4 + 4 + 8 + 1 + 4
DISK_HEADROOM = 1.2


def _channel_stats(values, channels, N, chunk=STAT_CHUNK):
    """Per-channel measured counts, ranges and means, read in fixed chunks.

    Chunked deliberately: `values` is a memmap in the streaming assembler and
    materialising a 2e9-row float32 column would defeat the whole exercise.
    The memory assembler calls the SAME function with the SAME chunk size, so
    the two cannot disagree about a number in store.json.

    `chunk` is a PARAMETER only so a test can drive tiny blocks across the
    boundaries; every caller in the publish path leaves it at `STAT_CHUNK`,
    because the mean is a sum of per-chunk `np.nansum`s and numpy's pairwise
    summation makes that sum depend, in the last bits, on where the blocks
    fall. Changing the default would rewrite store.json for the three stores
    that already exist.
    """
    chunk = max(1, int(chunk))
    C = len(channels)
    measured = [0] * C
    vmin = [None] * C
    vmax = [None] * C
    vsum = [0.0] * C
    finite_total = 0
    for lo in range(0, int(N), chunk):
        blk = np.asarray(values[lo:lo + chunk], np.float32)
        if blk.size == 0:
            continue
        fin = np.isfinite(blk)
        finite_total += int(fin.sum())
        for i in range(C):
            col = blk[:, i]
            f = fin[:, i]
            k = int(f.sum())
            if not k:
                continue
            measured[i] += k
            lo_v = float(np.nanmin(col))
            hi_v = float(np.nanmax(col))
            vmin[i] = lo_v if vmin[i] is None else min(vmin[i], lo_v)
            vmax[i] = hi_v if vmax[i] is None else max(vmax[i], hi_v)
            vsum[i] += float(np.nansum(col.astype(np.float64)))
    per_channel = {}
    for i, (nm, unit, lo_b, hi_b) in enumerate(channels):
        k = measured[i]
        per_channel[nm] = {
            "unit": unit, "measured": k,
            "fraction": round(k / N, 6) if N else 0.0,
            "min": vmin[i], "max": vmax[i],
            "mean": (vsum[i] / k) if k else None,
            "bounds": [lo_b, hi_b]}
    frac = round(finite_total / (int(N) * C), 6) if (N and C) else 0.0
    return per_channel, frac


def _part_paths(ctx):
    """Every part file, in `read_parts` order: years ascending, names sorted.

    THE ORDER IS THE CONTRACT — it is the tie-break of the store's defining
    sort, so both assemblers must walk the parts through this one function.
    """
    out = []
    for y in ctx.years:
        d = ctx.year_dir(y)
        if not os.path.isdir(d):
            continue
        for n in sorted(os.listdir(d)):
            if n.endswith(".npz"):
                out.append((y, os.path.join(d, n)))
    return out


def _part_ledgers(ctx):
    """per_year row counts from counts.json (cheap), and the merged counters."""
    per_year, counts_all = {}, {}
    for y in ctx.years:
        c = read_json(os.path.join(ctx.year_dir(y), "counts.json"), {})
        per_year[y] = int(c.get("rows", 0))
        _merge_counts(counts_all, c.get("counts") or {})
    return per_year, counts_all


def _scan_bins(ctx):
    """PASS 1 — read ONLY the `bin` member of every part.

    `np.load` on an npz is lazy per member, so this touches ~2 bytes a row
    instead of ~33. Returns (N, per_year, {bin: rows}).
    """
    counts, per_year, N = {}, {}, 0
    for y, p in _part_paths(ctx):
        with np.load(p) as z:
            b = np.asarray(z["bin"], np.int64)
        u, c = np.unique(b, return_counts=True)
        for k, v in zip(u.tolist(), c.tolist()):
            counts[k] = counts.get(k, 0) + v
        per_year[y] = per_year.get(y, 0) + int(b.size)
        N += int(b.size)
    return N, per_year, counts


def _disk_preflight(dest, N, C, n_bins):
    """Refuse BEFORE pass 2 rather than at 90% of a six-hour write (§5.18).

    The size is computable from the dtypes, so it is computed; a check that
    can only guess belongs nowhere near an 80 GB allocation.
    """
    need = int(N) * (ROW_BYTES_FIXED + 2 * int(C)) + 8 * (int(n_bins) + 1)
    free = shutil.disk_usage(dest).free
    want = need * DISK_HEADROOM
    print(f"  disk: the store is {need / 1e9:.2f} GB ({N:,} rows x "
          f"{ROW_BYTES_FIXED + 2 * C} B); {free / 1e9:.2f} GB free under "
          f"{dest}; {DISK_HEADROOM:g}x margin wants {want / 1e9:.2f} GB")
    if free < want:
        sys.exit(f"REFUSING to assemble: {dest} has {free / 1e9:.2f} GB free "
                 f"and this store needs {need / 1e9:.2f} GB "
                 f"({DISK_HEADROOM:g}x = {want / 1e9:.2f} GB with margin). "
                 f"Free space or build on a bigger disk; the parts are safe on "
                 f"the Hub and nothing has been overwritten.")
    return need


def assemble_store_streaming(ctx):
    """The same store as `assemble_store`, byte for byte, without the RAM.

    Three passes over the parts (see the block comment above for why the
    permutation is identical):
      1. count rows per bin -> the CSR offsets;
      2. scatter each part's rows into `offsets[bin] + cursor[bin]`, walking
         the parts in `read_parts` order, straight into `open_memmap`ed
         output arrays;
      3. sort each bin's slice by `time_days` with a STABLE argsort.
    Peak RAM is one part (FLUSH_ROWS rows) plus one bin's slice.
    """
    ad = ctx.adapter
    dest = ctx.store
    os.makedirs(dest, exist_ok=True)
    C = ad.C

    t0 = time.time()
    N, per_year_scan, bin_counts = _scan_bins(ctx)
    per_year, counts_all = _part_ledgers(ctx)
    for y, n in per_year_scan.items():
        per_year[y] = n                       # the parts, not the ledger
    for y in ctx.years:
        per_year.setdefault(y, 0)
    print(f"  pass 1: {N:,} row(s) over {len(bin_counts)} bin(s) "
          f"({time.time() - t0:.1f}s)", flush=True)

    if bin_counts:
        bin_first, bin_last = min(bin_counts), max(bin_counts)
    else:
        bin_first = bin_last = ctx.b_lo
    n_bins = bin_last - bin_first + 1
    off = np.zeros(n_bins + 1, np.int64)
    for k, v in bin_counts.items():
        off[k - bin_first + 1] = v
    np.cumsum(off, out=off)
    assert off[0] == 0 and off[-1] == N, (off[0], off[-1], N)

    _disk_preflight(dest, N, C, n_bins)

    from numpy.lib.format import open_memmap
    shapes = {"bin": (np.int16, (N,)), "time_days": (np.float32, (N,)),
              "lat": (np.float32, (N,)), "lon": (np.float32, (N,)),
              "values": (np.float16, (N, C)),
              "platform": (np.int64, (N,)), "qc": (np.uint8, (N,)),
              "fp": (np.float16, (N, 2))}
    files = {}
    mm = {}
    for k, (dtype, shape) in shapes.items():
        p = os.path.join(dest, k + ".npy")
        mm[k] = open_memmap(p, mode="w+", dtype=dtype, shape=shape)
        files[k + ".npy"] = p
    mm["fp"][:, 0] = np.float16(ad.log2_fp)
    mm["fp"][:, 1] = np.float16(ad.log2_dt)

    # PASS 2 — scatter, in part order.
    t0 = time.time()
    cursor = np.zeros(n_bins, np.int64)
    for _, p in _part_paths(ctx):
        with np.load(p) as z:
            d = {k: z[k] for k in ROW_KEYS}
        b = np.asarray(d["bin"], np.int64) - bin_first
        if b.size == 0:
            continue
        # Rows of one part that share a bin must land in the order they appear
        # in the part; `argsort(kind="stable")` groups them without disturbing
        # that, and `pos` is each row's rank inside its own group.
        grp = np.argsort(b, kind="stable")
        bs = b[grp]
        uniq, first, cnt = np.unique(bs, return_index=True, return_counts=True)
        pos = np.arange(bs.size, dtype=np.int64) - np.repeat(first, cnt)
        dest_sorted = off[bs] + cursor[bs] + pos
        idx = np.empty(b.size, np.int64)
        idx[grp] = dest_sorted
        for k in ROW_KEYS:
            mm[k][idx] = d[k]
        cursor[uniq] += cnt
    assert np.array_equal(cursor, np.diff(off)), "the scatter did not fill"
    print(f"  pass 2: scattered {N:,} row(s) ({time.time() - t0:.1f}s)",
          flush=True)

    # PASS 3 — stable sort by time inside each bin.
    t0 = time.time()
    moved = 0
    for j in range(n_bins):
        s, e = int(off[j]), int(off[j + 1])
        if e - s < 2:
            continue
        order = np.argsort(np.asarray(mm["time_days"][s:e], np.float32),
                           kind="stable")
        if np.array_equal(order, np.arange(e - s)):
            continue
        moved += 1
        for k in ROW_KEYS:
            mm[k][s:e] = np.asarray(mm[k][s:e])[order]
    for k in list(mm):
        mm[k].flush()
        del mm[k]
    print(f"  pass 3: {moved:,} bin(s) needed a time sort "
          f"({time.time() - t0:.1f}s)", flush=True)

    p = os.path.join(dest, "bin_offsets.npy")
    np.save(p, off)
    files["bin_offsets.npy"] = p

    values = np.load(os.path.join(dest, "values.npy"), mmap_mode="r")
    try:
        return _finish_store(ctx, dest, files, N, off, bin_first, bin_last,
                             n_bins, per_year, counts_all, values)
    finally:
        del values


def _peak_rss_gb():
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / 1e6 if sys.platform != "darwin" else kb / 1e9
    except Exception:                                         # noqa: BLE001
        return None


def assemble(ctx):
    """Pick an assembler and say which one ran (`--assemble`)."""
    how = str(getattr(ctx.a, "assemble", "auto") or "auto")
    if how == "auto":
        rows = sum(_part_ledgers(ctx)[0].values())
        big = rows > STREAM_ROWS or ctx.a.store == "slatrack"
        how = "streaming" if big else "memory"
        print(f"  assemble: auto -> {how} ({rows:,} part row(s) by the "
              f"per-year ledgers, threshold {STREAM_ROWS:,}"
              f"{'; slatrack is always streamed' if ctx.a.store == 'slatrack' else ''})")
    else:
        print(f"  assemble: {how} (forced by --assemble)")
    meta = (assemble_store_streaming(ctx) if how == "streaming"
            else assemble_store(ctx))
    rss = _peak_rss_gb()
    if rss is not None:
        print(f"  assemble: {how} done, peak RSS {rss:.2f} GB")
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

    return _finish_store(ctx, dest, files, N, off, bin_first, bin_last,
                         n_bins, per_year, counts_all,
                         cat["values"])



def _finish_store(ctx, dest, files, N, off, bin_first, bin_last, n_bins,
                  per_year, counts_all, values):
    """The bookkeeping tail BOTH assemblers share: statistics, store.json, the
    sha256 of every file, and the E-079 §4 assertion pass.

    It lives here and not twice so that `assemble_store` and
    `assemble_store_streaming` cannot drift: the streaming assembler's whole
    claim is that it produces the SAME BYTES, and a second copy of this
    function is the cheapest way to make that claim false.

    `values` is the (N, C) value matrix, in RAM for the memory assembler and a
    read-only memmap for the streaming one — `_channel_stats` reads it in
    fixed-size chunks either way, so the numbers are identical.
    """
    ad = ctx.adapter
    per_channel, measured_fraction = _channel_stats(values, ad.channels, N)
    live = int((np.diff(off) > 0).sum())
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
        "values_measured_fraction": measured_fraction,
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
    check_store(dest, ad, chunk_rows=ctx.check_chunk)
    print(f"  store: {N:,} row(s), C={ad.C}, bins {bin_first}..{bin_last} "
          f"({live:,} live) -> {dest}")
    return meta

# ============================================================== assertions ===
def check_store(path, adapter=None, anchor=None, chunk_rows=CHECK_CHUNK_ROWS):
    """E-079 §4's assertions, run on the store before anybody trusts it.

    Every one of these is a property a broken build can have while looking
    completely ordinary from the outside, which is the only reason they are
    worth the seconds they cost.

    IN BOUNDED MEMORY, by row blocks of `chunk_rows` over the store's own
    memmaps (`f10.Store` opens every column with `mmap_mode="r"`). The old
    form read `np.asarray(st["bin"], np.int64)` and a float32 view of
    `values` — 16 GB and 12 GB respectively at slatrack's ~2e9 rows, so the
    assertion pass after a streaming assembly would have OOM'd on any box
    that could afford the assembly. There is ONE implementation: the small
    stores take the same path with one block, because a second whole-array
    branch is a second set of numbers that can disagree.

    Two checks need care across a block boundary and get it:
      * SORTEDNESS carries the previous block's LAST row into the next
        block's comparison, so the seam is checked like any other pair.
      * The CSR index is checked twice — each block asserts its rows lie
        inside the slice `bin_offsets` gives their bin (`searchsorted`'s own
        answer, via `Store._slice`), and a running `np.bincount` recount is
        compared with `np.diff(bin_offsets)` at the end. The first catches a
        row in the wrong slice, the second an offset vector that is
        internally tidy but describes a different store.
    """
    st = f10.Store(path)
    N = int(st.N)
    chunk = max(1, int(chunk_rows))
    off = np.asarray(st.bin_offsets, np.int64)
    assert off[0] == 0 and off[-1] == st.N, "CSR offsets do not span the rows"
    assert np.all(np.diff(off) >= 0), "bin_offsets is not monotone"

    b_col, t_col = st["bin"], st["time_days"]
    lat_col, lon_col = st["lat"], st["lon"]
    v_col, fp_col = st["values"], st["fp"]
    counts = np.zeros(max(st.n_bins, 0), np.int64)
    C = int(st.C)
    lo_m = np.full(C, np.inf, np.float64)
    hi_m = np.full(C, -np.inf, np.float64)
    seen = np.zeros(C, bool)
    fp0 = None
    prev_b = prev_t = None

    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        b = np.asarray(b_col[lo:hi], np.int64)
        t = np.asarray(t_col[lo:hi], np.float64)
        # -- sorted by (bin, time), the seam included
        if prev_b is None:
            bb_, tt_ = b, t
        else:
            bb_ = np.concatenate(([prev_b], b))
            tt_ = np.concatenate(([prev_t], t))
        assert np.all(np.diff(bb_) >= 0), "rows are not sorted by bin"
        same = bb_[1:] == bb_[:-1]
        assert np.all(tt_[1:][same] >= tt_[:-1][same]), \
            "rows inside a bin are not sorted by time"
        prev_b, prev_t = b[-1], t[-1]

        # The bin column must BE the bin of the time column — a store whose
        # index and timestamps disagree answers every search with the wrong
        # pentad and nothing says so.
        assert np.array_equal(b, f10.bin_of_days(t)), \
            "bin.npy disagrees with floor(time_days / 5)"
        lon = np.asarray(lon_col[lo:hi], np.float64)
        assert np.all((lon >= -180.0) & (lon < 180.0)), \
            f"lon runs {lon.min()}..{lon.max()}, not [-180, 180)"
        lat = np.asarray(lat_col[lo:hi], np.float64)
        assert np.all(np.abs(lat) <= 90.0), "lat outside [-90, 90]"

        # -- CSR, per block: every row of bin bb inside bb's own slice
        assert np.all((b >= st.bin_first) & (b <= st.bin_last)), \
            "a row's bin lies outside the range bin_offsets indexes"
        uniq, first = np.unique(b, return_index=True)
        counts += np.bincount(b - st.bin_first, minlength=st.n_bins)
        s = lo + first
        e = lo + np.append(first[1:], b.size)
        s_off = off[uniq - st.bin_first]
        e_off = off[uniq - st.bin_first + 1]
        bad = np.nonzero((s_off > s) | (e > e_off))[0]
        assert bad.size == 0, \
            f"bin {int(uniq[bad[0]])}'s CSR slice holds other bins"

        # -- values and the footprint
        v = np.asarray(v_col[lo:hi], np.float32)
        assert not np.isinf(v).any(), "values.npy holds an infinity"
        fin = np.isfinite(v)
        if fin.any():
            lo_m = np.minimum(lo_m, np.where(fin, v, np.inf).min(axis=0))
            hi_m = np.maximum(hi_m, np.where(fin, v, -np.inf).max(axis=0))
            seen |= fin.any(axis=0)
        fp = np.asarray(fp_col[lo:hi], np.float32)
        if fp0 is None:
            fp0 = (fp[0, 0], fp[0, 1])
            assert (f10.LOG2_FP_RANGE[0] <= fp0[0] <= f10.LOG2_FP_RANGE[1]), \
                f"log2_fp {fp0[0]} outside E-078 §2's clamp"
        assert np.all(fp[:, 0] == fp0[0]) and np.all(fp[:, 1] == fp0[1]), \
            "the footprint columns are not constant for this source"

    assert np.array_equal(counts, np.diff(off)), \
        "bin_offsets disagrees with a recount of bin.npy"
    if adapter is not None:
        lo_b, hi_b = adapter.bounds()
        for i, nm in enumerate(adapter.channel_names):
            if not seen[i]:
                continue
            assert lo_b[i] <= lo_m[i] and hi_m[i] <= hi_b[i], (
                f"channel {nm} runs {lo_m[i]}..{hi_m[i]}, outside its "
                f"physical bounds {lo_b[i]}..{hi_b[i]}")
        assert st.C == adapter.C and st.channels == list(adapter.channel_names)
    if anchor is not None and N:
        lat0, lon0, bin0 = anchor
        tok = st.knearest(lat0, lon0, bin0, k=5, R_max_km=1000.0,
                          T_max_days=30.0)
        assert np.all(tok["dt_days"][tok["valid"]] >= 0.0), \
            "the search returned an observation from the future"
        rows = np.asarray(tok["row"][tok["valid"]], np.int64)
        # a handful of rows, fancy-indexed straight off the memmap
        assert np.all(np.asarray(b_col[rows], np.int64) <= bin0), \
            "the search read past the anchor's bin"
    return st


# ========================================================== stage: publish ===
RESTORE_HEADROOM = 1.1


def _restore_disk_preflight(ctx, dest, names):
    """Refuse BEFORE the upload if the RESTORE could not land (§5.18).

    The publish downloads every file back to hash it, one at a time into
    `<scratch>/verify`, which is removed between files — so the requirement
    is free space for the LARGEST file, not for the store. At slatrack's
    50-80 GB that largest file is `values.npy` at ~1.5x2e9x C bytes, and a
    box that cannot hold one more copy of it would upload for hours and then
    fail the verification it cannot skip.
    """
    sizes = {n: os.path.getsize(os.path.join(dest, n)) for n in names}
    big, need = max(sizes.items(), key=lambda kv: kv[1])
    os.makedirs(ctx.scratch, exist_ok=True)
    free = shutil.disk_usage(ctx.scratch).free
    want = need * RESTORE_HEADROOM
    print(f"  restore: the largest file is {big} at {need / 1e9:.2f} GB; "
          f"{free / 1e9:.2f} GB free under {ctx.scratch}; "
          f"{RESTORE_HEADROOM:g}x margin wants {want / 1e9:.2f} GB")
    if free < want:
        sys.exit(f"REFUSING to publish: the restore check downloads every "
                 f"file back and {big} is {need / 1e9:.2f} GB, but "
                 f"{ctx.scratch} has only {free / 1e9:.2f} GB free "
                 f"({RESTORE_HEADROOM:g}x = {want / 1e9:.2f} GB with margin). "
                 f"Free space or point --work at a bigger disk; nothing has "
                 f"been uploaded.")
    return need


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
    _restore_disk_preflight(ctx, dest, names)
    check_store(dest, ad, chunk_rows=ctx.check_chunk)
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


def parse_stages(spec):
    """`all` -> every stage; `fetch` -> [fetch]; `index,fetch` -> both, in the
    fixed stage order whatever order they were typed in. Unknown names refuse
    before anything runs."""
    if spec.strip() == "all":
        return list(STAGES)
    want = [x.strip() for x in spec.split(",") if x.strip()]
    bad = [x for x in want if x not in STAGES]
    if bad or not want:
        sys.exit(f"--stage {spec!r}: unknown stage(s) {bad} — choose from "
                 f"{STAGES}, `all`, or a comma list of them")
    return [s for s in STAGES if s in want]


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
                             int(np.floor(truth[0]["t"] / PENTAD_DAYS))),
                     chunk_rows=ctx.check_chunk)
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
    ap.add_argument("--stage", default="all",
                    help="`all`, one stage, or a comma-separated prefix of the "
                         "stage order such as `index,fetch` (a probe that "
                         "fetches but does not publish). Order is fixed: "
                         "fetch needs index, publish needs fetch.")
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
    ap.add_argument("--slatrack-fetch", default="files",
                    choices=("files", "dataframe"),
                    help="how the slatrack live path gets its data: `files` "
                         "(default) downloads the ORIGINAL per-day netCDFs "
                         "through copernicusmarine.get in month batches; "
                         "`dataframe` asks copernicusmarine.read_dataframe "
                         "for a subset month by month — measured at ~3.3k "
                         "samples/s, ~220 runner-hours for the archive, so it "
                         "is the fallback, not the default.")
    ap.add_argument("--assemble", default="auto",
                    choices=("auto", "streaming", "memory"),
                    help="how the store is assembled from the parts. `memory` "
                         "concatenates every row and lexsorts it; `streaming` "
                         "writes the same bytes through memmaps in three "
                         "passes and never holds more than one part; `auto` "
                         "(default) streams when the parts hold more than "
                         f"{STREAM_ROWS:,} rows or the store is slatrack "
                         "(50-80 GB — no box has that in RAM).")
    ap.add_argument("--parts-from-hub", action="store_true",
                    help="stage `fetch` does NOT touch the source archive: it "
                         "pulls the requested years' column parts from "
                         "partials/family10/<store>/<year>/ on the Hub (put "
                         "there by ml/family10_parts_hub.py push) and "
                         "assembles them. This is how slatrack is built on a "
                         "box: HF_TOKEN only, no Copernicus credentials "
                         "(ml/CLAUDE.md §6). Refuses if any requested year has "
                         "no done.json on the Hub.")
    ap.add_argument("--allow-missing-years", action="store_true",
                    help="with --parts-from-hub: assemble even though some "
                         "years are not on the Hub. The store will be short "
                         "and its per_year block says which years are empty.")
    ap.add_argument("--check-chunk-rows", type=int, default=CHECK_CHUNK_ROWS,
                    help="rows per block in the assertion pass over the "
                         "finished store (default "
                         f"{CHECK_CHUNK_ROWS:,}). The checks read the store's "
                         "memmaps block by block and carry the seam, so the "
                         "answer does not depend on this number; it only "
                         "bounds the peak RAM of the check. Lower it on a "
                         "small box, raise it on a big one.")
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
    run_stages(ctx, parse_stages(a.stage))
    return 0


if __name__ == "__main__":
    sys.exit(main())
