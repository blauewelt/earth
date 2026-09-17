#!/usr/bin/env python3
"""Family 10's TIER-P STORES — drifters, moorings, ship CO2, tracks, the fleet.

E-079 §2-§4 and E-081, made runnable. One builder, five source adapters, four
stages.
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
  fishing   Global Fishing Watch's AIS-based apparent fishing effort: one row
            per (day, 0.1 degree cell, vessel) with the hours that vessel
            broadcast in the cell and the part of them a neural network
            classed as fishing, plus the vessel's gear class. 2012 -> 2024
            (E-081, family 10.2 — the only store of 10.2; the four above are
            inherited from 10.1 by reference and not one byte is rebuilt).

FOUR STAGES, IN FIXED ORDER — `index | fetch | grid | publish` (or `all`):

  index    LIST the archive before spending anything: ask the service what it
           holds, read one real record, and write the plan of files/years into
           `<work>/<store>/plan.json`. Nothing is fetched in bulk here.
  fetch    stream year by year, parse, filter, append per-year column parts
           under `<work>/<store>/parts/<year>/`, and mark each year DONE only
           after its parts are on disk (ml/CLAUDE.md §5.21 — a marker may only
           under-claim). Then assemble the store.
  grid     sum the finished store onto the family-7 0.25 degree grid, one
           frame per month, month-major so one month is one range read
           (E-081 §3; float32, because the busiest cell-month of 2024 is
           595,726 vessel-hours and float16 stops at 65,504 — measured). A NO-OP for the four stores that have no gridded
           product, so `all` means the same thing for every store.
  publish  upload to `tensors/family10_2/<store>/` on the Hub and DOWNLOAD
           EVERY FILE BACK to compare sha256. A publish that cannot verify
           fails the job (§0.2 — an upload that returns 200 is not evidence the
           bytes are retrievable). Where a grid exists it is uploaded and
           restore-checked too, and `ml/publish_fishing_index.py` writes
           `data/fishing_index.json` from the published bytes.

FAMILY 10.1 — THE TIME COLUMN IS INTEGER SECONDS (E-079 §10.1). Schema 2
replaces v1's `time_days` (float32 days since the epoch) with `time_s`, int32
seconds. float32 days resolve 21 s in 1993 and 84 s in 2024, and slatrack
samples at 1 Hz, so up to 316 consecutive along-track samples shared one v1
timestamp. Every adapter below therefore produces EXACT SECONDS — the calendar
sources parse their own stamp to the second, the netCDF path rounds a float64
CF axis to the nearest second — and `bin = floor(time_s / 432000)` is integer
arithmetic with no float in it anywhere. int32 reaches 2050-01-19T03:14:07Z and
`_pack` refuses a row past it. Nothing else about the store changes: same nine
arrays, same bins, same footprints, same channels, same QC.

RESUMABILITY IS THE CONTRACT, the same one families 7 and 8 have: re-run with
the same `--work` and finished stages and finished years are skipped.

SLATRACK IS BUILT ON TWO MACHINES, and neither could do both halves. Its fetch
needs Copernicus credentials, which ml/CLAUDE.md §6 forbids on a rented box, so
it may only run on a GitHub-hosted runner; its store is 50-80 GB (measured
2026-09-14: ~70 M samples in 2015 alone, ~1.5-2.5 e9 rows over 1993-2024 at ~33
bytes a row), which a hosted runner's ~14 GB of disk cannot hold. So
`.github/workflows/family10-slatrack-fetch.yml` fetches ONE YEAR at a time on
hosted lanes and parks its column parts at
`partials/family10_1/slatrack/<year>/` on the Hub (`ml/family10_parts_hub.py`,
done.json written LAST), and a box with HF_TOKEN and no credentials assembles
them:

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

FAMILY 10.2 ADDS `fishing` AND CHANGES NOTHING ELSE (E-081). `FAMILY_VERSION`
is "10.2", so this builder writes and publishes under `tensors/family10_2/`;
the four stores above stay exactly where 10.1 published them and the registry
lists them from there (`ml/build_family10_registry.py`'s `STORE_ROOTS`). The
schema is untouched — still schema 2, still `time_s` int32 seconds.

E-082: THE SAME MACHINERY BUILDS FAMILY 1 (1.gf, 1.0.tf, 0.9.tf), BY IMPORT.
`ml/build_family1_stores.py` hands `Ctx` its own adapter and a `Layout` — the
Hub prefix (`tensors/family1_tf/`), the parts prefix (`partials/family1_tf/`),
the cache directory, the repository (`chfrank/earth-tensors` or, for a
`distribution = "private"` adapter, `chfrank/earth-tensors-private`), the
family name written into store.json. The default `Layout()` is family 10's,
spelled from the same constants as before, so nothing a family-10 build writes
or reads has moved. An adapter may also declare `time_dtype = "int64"`: the
store is then SCHEMA 3 (`time_s` int64, `store.json` carries `schema_version:
3` and `time_dtype: "int64"`) and `_pack` no longer refuses a row before 1914
or after 2050 — the int16 `bin` column is then the limit (1533 .. 2430).
Family 10's adapters all keep the int32 default, and their stores are schema
2 exactly as before.

Run:
  python3 ml/build_family10_stores.py --store gdp --smoke        # synthetic, seconds
  python3 ml/build_family10_stores.py --store fishing --work W \
      --start 2012-01-01 --end 2024-12-31 --stage all            # ~18.5 GB: a box
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
import zipfile
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
# ONE CONSTANT, IN ONE PLACE. `ml/family10_store.py` derives every path from
# FAMILY_VERSION ("10.1") — `tensors/family10_1`, `partials/family10_1`,
# `ml/cache/family10_1` — and this builder, `ml/build_family10_registry.py` and
# `ml/family10_parts_hub.py` all import them from there rather than spelling a
# prefix out. A version that appears twice is a version that will disagree with
# itself on the day one of the two is changed.
HF_ROOT = f10.HF_ROOT                          # tensors/family10_1
FAMILY = f10.FAMILY                            # family10
FAMILY_VERSION = f10.FAMILY_VERSION            # 10.1
SCHEMA_VERSION = f10.SCHEMA_VERSION            # 2

PENTAD_DAYS = 5
SECONDS_PER_DAY = 86400
PENTAD_SECONDS = f10.PENTAD_SECONDS            # 432_000, exactly
assert f10.PENTAD_DAYS == PENTAD_DAYS
assert f10.SECONDS_PER_DAY == SECONDS_PER_DAY
assert PENTAD_SECONDS == PENTAD_DAYS * SECONDS_PER_DAY
assert str(START) == f10.EPOCH, (START, f10.EPOCH)

UA = {"User-Agent": "earth-science-pipeline/1.0 "
                    "(research; github blauewelt/earth)"}
SOCKET_TIMEOUT = 120          # per socket operation — a stall detector
CHUNK = 1 << 22               # 4 MiB reads
FLUSH_ROWS = 1_000_000        # rows held in memory per year before a part flush

# FOUR STAGES SINCE E-081. `grid` sums the finished store onto the family-7
# 0.25° monthly grid the globe reads (E-081 §3); it is a no-op for the four
# stores that have no gridded product, so `all` means the same thing for every
# store and the order stays fixed. `publish` depends on `fetch` and NOT on
# `grid`, so re-publishing a store on its own is still one dispatch — but when
# a grid exists, the publish uploads it and writes its index.
STAGES = ["index", "fetch", "grid", "publish"]
DEPS = {"fetch": ["index"], "grid": ["fetch"], "publish": ["fetch"]}

# int16 holds -32768..32767; the axis needs bin -1825 (1957-01-01) to +3141
# (2024-12-31) today and roughly +6000 by 2100, so int16 is not a constraint —
# but the assertion is here because a silently wrapped bin is unfindable later.
BIN_MIN_INT16, BIN_MAX_INT16 = -32768, 32767

# THE TIME COLUMN'S WIDTH, per adapter (E-082). int32 is schema 2 — every
# family-10 store; int64 is schema 3, for records that start before 1914
# (GHCN-Daily from 1763, ICOADS, WOD, tide gauges). Nothing else in a row
# changes between the two.
TIME_DTYPES = {"int32": np.int32, "int64": np.int64}
SCHEMA_OF_TIME_DTYPE = {"int32": 2, "int64": 3}
INT32_TIME_BEFORE = 1914          # an adapter whose first_year is earlier MUST be int64


def time_dtype_name(adapter):
    """The adapter's `time_dtype`, validated. Family 10's adapters say int32."""
    name = str(getattr(adapter, "time_dtype", "int32") or "int32")
    if name not in TIME_DTYPES:
        raise ValueError(f"{getattr(adapter, 'store', adapter)}: time_dtype "
                         f"{name!r} is neither 'int32' (schema 2) nor 'int64' "
                         f"(schema 3)")
    if name == "int32" and int(getattr(adapter, "first_year", 1982)) \
            < INT32_TIME_BEFORE:
        raise ValueError(
            f"{adapter.store}: first_year {adapter.first_year} is before "
            f"{INT32_TIME_BEFORE}, and int32 seconds since 1982 begin at "
            f"{f10.TIME_S_MIN_DATE}. Declare time_dtype = 'int64' (schema 3).")
    return name


class Layout:
    """WHERE a store is published and WHAT it calls itself (E-082).

    Every default is family 10's, spelled from the constants this module has
    always used, so `Ctx(a)` with no layout builds, names and publishes a
    family-10 store exactly as before. `ml/build_family1_stores.py` passes its
    own: `tensors/family1_tf`, `partials/family1_tf`, `ml/cache/family1_tf`,
    and an explicit `repo_id` — which is how a private adapter is routed to
    `chfrank/earth-tensors-private` and nowhere else.

    `repo_id=None` keeps family 10's rule: `<whoami>/earth-tensors` through
    `build_family7.hub_repo()`. An explicit `repo_id` takes its token from the
    first of `token_env` that is set, and never from argv.
    """

    def __init__(self, family=None, family_version=None, hf_root=None,
                 hf_partials=None, cache_dirname=None, repo_id=None,
                 private=False, token_env=("HF_TOKEN",),
                 builder="ml/build_family10_stores.py", label="family 10"):
        self.family = family or FAMILY
        self.family_version = family_version or FAMILY_VERSION
        self.hf_root = hf_root or HF_ROOT
        self.hf_partials = hf_partials or f10.HF_PARTIALS
        self.cache_dirname = cache_dirname or f10.CACHE_DIRNAME
        self.repo_id = repo_id
        self.private = bool(private)
        self.token_env = tuple(token_env)
        self.builder = builder
        self.label = label

    def prefix(self, store):
        return f"{self.hf_root}/{store}"

    def token(self):
        for k in self.token_env:
            v = os.environ.get(k, "")
            if v:
                return k, v
        return None, ""

    def hub(self):
        """(api, repo, token). Family 10: `hub_repo()`, unchanged."""
        if self.repo_id is None:
            return hub_repo()
        name, tok = self.token()
        if not tok:
            sys.exit(f"no Hugging Face token for {self.repo_id}: none of "
                     f"{', '.join(self.token_env)} is set in the environment "
                     f"(never in argv) — see the project doc "
                     f"claude/huggingface-access.md")
        from huggingface_hub import HfApi
        print(f"  hub: {self.repo_id} with the token from ${name}")
        return HfApi(token=tok), self.repo_id, tok

    def describe(self):
        return {"family": self.family, "family_version": self.family_version,
                "hf_root": self.hf_root, "hf_partials": self.hf_partials,
                "cache_dirname": self.cache_dirname,
                "repo_id": self.repo_id, "private": self.private,
                "token_env": list(self.token_env)}


# BYTES OFF THE NETWORK, counted where they are read (E-082's probe reports
# bytes per row, and a number nobody measured is an estimate). The helpers
# below add to it; `Ctx.bytes_fetched` reads it relative to the context's own
# baseline, so two contexts in one process do not have to share a counter
# object. An adapter that reads a LOCAL source (`--source-dir`) calls
# `ctx.count_bytes(n)` for the bytes the network would have carried.
NET_BYTES = {"n": 0}


def count_bytes(n):
    NET_BYTES["n"] += int(n)


def seconds_since_epoch(d):
    """A date or datetime -> EXACT integer seconds since 1982-01-01T00:00:00Z.

    Family 10.1's one time conversion for every archive that publishes a
    calendar timestamp (gdp, gtmba, socat). `timedelta.total_seconds()` is a
    float, but the values here are whole seconds under 2.2e9 — far inside
    float64's exact-integer range — so `int(round(...))` is exact rather than
    nearly exact. A date (no time of day) is midnight UTC.

    NEGATIVE before the epoch, and kept: drifters start in 1979 and SOCAT in
    1957, and their rows are the reason `bin` is signed.
    """
    if isinstance(d, dt.datetime):
        base = dt.datetime(START.year, START.month, START.day)
        return int(round((d - base).total_seconds()))
    return int((d - START).days) * SECONDS_PER_DAY


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
            data = r.read()
        count_bytes(len(data))
        return data
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
    count_bytes(got)
    if want is not None and got != int(want):
        os.remove(part)
        raise IOError(f"{url}: {got:,} of {int(want):,} bytes — truncated")
    os.replace(part, path)
    return path


class CountingStream:
    """A GET response as a file-like that COUNTS its bytes and REFUSES a short
    read (E-082).

    For sources that are parsed as they stream — a year of GHCN-Daily is a
    170 MB gzip that is never written to disk whole. Every byte read is added
    to `NET_BYTES`. When the stream reaches its end, the byte count is compared
    with `Content-Length` and a short body RAISES (the 2026-09-14 rule: a short
    download is a refusal, never a smaller year). A caller that stops reading
    early on purpose — the probe, which wants one month — just closes it;
    nothing is claimed about a body nobody finished.
    """

    def __init__(self, url, timeout=SOCKET_TIMEOUT, headers=None):
        self.url = url
        req = urllib.request.Request(url, headers={**UA, **(headers or {})})
        try:
            self.r = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise _NotFound(url) from None
            raise IOError(f"{url}: HTTP {e.code}") from None
        cl = self.r.headers.get("Content-Length")
        self.want = int(cl) if cl is not None else None
        self.got = 0
        self.complete = False

    def read(self, n=-1):
        b = self.r.read(n) if n is not None and n >= 0 else self.r.read()
        self.got += len(b)
        count_bytes(len(b))
        if not b or (n is None or n < 0):
            if self.want is not None and self.got != self.want:
                raise IOError(f"{self.url}: {self.got:,} of {self.want:,} "
                              f"bytes — truncated")
            self.complete = True
        return b

    def readable(self):
        return True

    def close(self):
        try:
            self.r.close()
        except Exception:                                       # noqa: BLE001
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def fetch_first(urls, path=None, attempts=3, sleep=5.0, reason=False):
    """The first URL that serves. None if EVERY url is a 404 or an empty result.

    A definite "not there" is a legitimate gap and the caller continues; any
    other error retries across the whole list and then RAISES, never silently
    (ml/CLAUDE.md §4.6).

    `reason=True` RETURNS (result, why) AND THE TWO KINDS OF "NOT THERE" ARE
    TOLD APART, because they are not the same fact and the callers were
    treating them as one. ERDDAP answers `"your query produced no matching
    results"` — `_Empty` — when a dataset genuinely holds no row in the window
    asked for, which is how the archive says "the ADCP was not deployed in
    1977"; a bare 404 — `_NotFound` — says the URL itself is gone, i.e. the
    dataset id moved under a build that the index stage had already checked.
    Collapsing them made a moved dataset look exactly like an empty year, and
    the store that came out of that had a hole nothing named. `why` is:

      "ok"        something served
      "empty"     every url answered "no matching results" — a gap in the
                  archive, legitimate, and the caller counts it
      "notfound"  every url 404'd without saying that — an ABSENCE, and the
                  caller refuses on it
    """
    errs = []
    for i in range(attempts):
        n_gone = n_empty = 0
        for u in urls:
            try:
                got = http_to_file(u, path) if path else http_bytes(u)
                return (got, "ok") if reason else got
            except _Empty:
                n_gone += 1
                n_empty += 1
            except _NotFound:
                n_gone += 1
            except Exception as e:                              # noqa: BLE001
                errs.append(f"{u}: {type(e).__name__}: {e}")
        if n_gone == len(urls):
            why = "empty" if n_empty == len(urls) else "notfound"
            return (None, why) if reason else None
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
ROW_KEYS = ("bin", "time_s", "lat", "lon", "values", "platform", "qc")
ROW_DTYPE = {"bin": np.int16, "time_s": np.int32, "lat": np.float32,
             "lon": np.float32, "values": np.float16,
             "platform": np.int64, "qc": np.uint8}


def empty_rows(C, time_dtype="int32"):
    out = {k: np.zeros(0, ROW_DTYPE[k]) for k in ROW_KEYS if k != "values"}
    out["time_s"] = np.zeros(0, TIME_DTYPES[time_dtype])
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
    # E-082 (family 1). Family 10's adapters inherit these defaults and are
    # unchanged by them. `family`, `distribution` and `licence` are NOT
    # defaulted here: a family-1 adapter must state them, and
    # `ml/family1/adapters/__init__.py` refuses one that does not.
    time_dtype = "int32"     # "int32" (schema 2) | "int64" (schema 3)
    credentials = ()         # env var NAMES the fetch needs; refused at preflight
    platform_meta = False    # True => `platforms(ctx)` -> platforms.json

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

    # -- three optional hooks, all no-ops by default ------------------------
    def fetch_preflight(self, ctx):
        """Checked BEFORE the first byte is fetched (ml/CLAUDE.md §0.3).

        A precondition that depends only on the inputs — the projected disk,
        say — is free here and expensive at hour three.
        """
        return None

    def extra_files(self, ctx, dest):
        """Files this store publishes BESIDE its nine arrays: {name: path}.

        They are written into `dest` and land in store.json's sha256 block, so
        the publish uploads them, the restore check verifies them and
        `family10_store.resolve` brings them down with the store. Only
        `fishing` has one today (`vessels.csv.gz`, E-081 §2).
        """
        return {}

    def extra_meta(self, ctx, dest, N, values):
        """Extra store.json keys. `values` is the (N, C) matrix, memmap or not."""
        return {}

    # -- E-082 hooks (family 1) ----------------------------------------------
    def fetch_month(self, ctx, year, month):
        """Yield `(label, rows, counts)` for ONE calendar month — the probe.

        The default runs `fetch_year` (or `fetch_stream`) and keeps only the
        rows whose `time_s` falls inside the month, so it is always correct and
        costs the whole year. An adapter whose source is ordered so it can STOP
        after the month (or can ask the archive for just the month) overrides
        it. The `counts` the default passes on are the WHOLE year's; the probe
        records that as `counts_scope: "year"`.
        """
        lo, hi = month_bounds_s(year, month)
        src = (self.fetch_year(ctx, year) if self.per_year else
               ((str(y), r, c) for y, r, c in self.fetch_stream(ctx)
                if y is not None and int(y) == int(year)))
        for label, rows, counts in src:
            if rows is not None and len(rows["bin"]):
                t = np.asarray(rows["time_s"], np.int64)
                keep = (t >= lo) & (t <= hi)
                rows = {k: v[keep] for k, v in rows.items()}
            yield label, rows, counts

    fetch_month_scope = "year"   # what the default's counts describe

    def platforms(self, ctx):
        """{platform_hash: {id, lat, lon, ...}} — only when `platform_meta`."""
        raise NotImplementedError(
            f"{self.store}: platform_meta is True and platforms(ctx) is not "
            f"implemented")

    def pack(self, t, lat, lon, values, platform, qc):
        """`_pack` with this adapter's C and time dtype filled in."""
        return _pack(t, lat, lon, values, platform, qc, self.C,
                     time_dtype=time_dtype_name(self))

    def mask_bounds(self, values):
        """Values outside `[lo, hi]` -> NaN, IN PLACE; returns the counts.

        The contract's rule 3 (never clipped). The dict it returns is what an
        adapter puts under `counts["out_of_bounds"]`, keyed by channel name —
        the probe and store.json read it from there.
        """
        lo, hi = self.bounds()
        v = values
        with np.errstate(invalid="ignore"):
            bad = np.isfinite(v) & ((v < lo) | (v > hi))
        n = bad.sum(axis=0)
        v[bad] = np.nan
        return {nm: int(k) for nm, k in zip(self.channel_names, n) if k}

    # -- helpers shared by the adapters ------------------------------------
    def blank(self, n):
        return np.full((n, self.C), np.nan, np.float64)


def month_bounds_s(year, month):
    """[first second, last second] of a calendar month, since 1982, inclusive."""
    d0 = dt.date(int(year), int(month), 1)
    d1 = dt.date(int(year) + (int(month) == 12), int(month) % 12 + 1, 1)
    return seconds_since_epoch(d0), seconds_since_epoch(d1) - 1


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
        """One month's CSV, parsed. A month that could not be READ is an ABSENCE.

        This used to answer a missing file and a dead service with the same
        `{"missing_month": 1}` as an ERDDAP "no matching results", hand back
        zero rows, and let `stage_fetch` mark the year COMPLETE — so a month
        the mirrors would not serve became a permanent, invisible hole in a
        green build (ml/CLAUDE.md §5.21; family 7's 1989, commit fd3b446). The
        three cases are now distinct: an empty ERDDAP answer is a GAP and is
        counted; a 404 or an absent local file is an ABSENCE and is reported to
        `ctx.note_absent`, which stops the year from being marked.
        """
        tmp = os.path.join(ctx.scratch, "gdp", f"{lo:%Y-%m}.csv")
        owned = False
        if ctx.source_dir:
            p = self.local(ctx, lo)
            if not os.path.exists(p):
                # `months()` NAMES this month off the axis, not off a directory
                # listing, so a local mirror that does not hold it is short —
                # the same distinction family 7's `glorys_chunk_names` had to
                # make before its absence branch could fire at all.
                ctx.note_absent(f"{lo:%Y-%m}", f"{p} does not exist "
                                               f"(--source-dir {ctx.source_dir})")
                return empty_rows(self.C), {"missing_month": 1}
        else:
            got, why = fetch_first(self.urls(lo, hi), tmp,
                                   attempts=ctx.a.attempts, reason=True)
            if got is None and why == "empty":
                # The service answered, and its answer is "no rows in this
                # month". A legitimate gap — 1979-01 is one, the archive's
                # first sample is 1979-02-15.
                return empty_rows(self.C), {"empty_month": 1}
            if got is None:
                ctx.note_absent(
                    f"{lo:%Y-%m}",
                    f"every ERDDAP 404'd for {GDP_DATASET} "
                    f"{lo:%Y-%m-%d}..{hi:%Y-%m-%d} without saying the query "
                    f"produced no matching results — the access path has "
                    f"moved, this is not an empty month")
                return empty_rows(self.C), {"missing_month": 1}
            p, owned = got, True
        try:
            with open(p, newline="") as fh:
                rows, counts = self._parse(ctx, fh, f"{lo:%Y-%m}", p)
        finally:
            if owned and not preflight and os.path.exists(p):
                os.remove(p)
            elif owned and preflight and os.path.exists(p):
                os.remove(p)
        return rows, counts

    def _parse(self, ctx, fh, label="", path=""):
        r = csv.reader(fh)
        hdr = next(r, None)
        if hdr is None:
            # A CSV WITH NO HEADER LINE AT ALL is not an empty month — ERDDAP
            # writes the header before it knows whether there are rows, so a
            # zero-byte body is a transfer that ended before it began. It used
            # to be counted and forgotten, with the year marked done.
            ctx.note_absent(label or "?", f"{path or 'the month CSV'} is empty "
                                          f"— not even a header line, so the "
                                          f"transfer did not deliver a month")
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
            # ERDDAP's `time` is an ISO-8601 stamp TO THE SECOND, and this
            # product is kriged onto the 00/06/12/18 UTC grid, so every value
            # is a whole second and the conversion is exact.
            td = seconds_since_epoch(when)
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
                d_ds = os.path.join(ctx.source_dir, "gtmba", ds)
                p = os.path.join(d_ds, f"{year}.csv")
                if not os.path.isdir(d_ds):
                    # THE MIRROR DOES NOT CARRY THIS DATASET AT ALL. That is a
                    # property of the local archive, not of the year — the test
                    # fixtures carry three of the eight — so it is counted, not
                    # refused. A dataset directory that EXISTS and lacks the
                    # year is the other case, below.
                    counts["datasets_absent_locally"] = \
                        counts.get("datasets_absent_locally", 0) + 1
                    continue
                if not os.path.exists(p):
                    ctx.note_absent(
                        f"{year} {ds}", f"{p} does not exist, but "
                                        f"{d_ds}/ does — this mirror carries "
                                        f"{ds} and is missing that year")
                    counts["datasets_empty"] += 1
                    continue
                path = p
                owned = False
            else:
                tmp = os.path.join(ctx.scratch, "gtmba", ds, f"{year}.csv")
                got, why = fetch_first(self.urls(ds, variables, lo, hi), tmp,
                                       attempts=ctx.a.attempts, reason=True)
                if got is None and why == "empty":
                    # ERDDAP said "your query produced no matching results":
                    # this array published nothing for this quantity in this
                    # year, which is most of what `datasets_empty` counts (37
                    # of 384 dataset-years in the store published 2026-09-14 —
                    # the ADCP before it was deployed, RAMA before 2000).
                    counts["datasets_empty"] += 1
                    continue
                if got is None:
                    # A 404 that did NOT say "no matching results". The index
                    # stage checked this dataset id against the live listing an
                    # hour ago, so this is the access path moving mid-build —
                    # and it used to be indistinguishable from the empty answer
                    # above: one counter, seven channels quietly all-NaN for
                    # the year, and the year marked COMPLETE.
                    ctx.note_absent(
                        f"{year} {ds}", f"every ERDDAP 404'd for {ds} "
                                        f"({var}/{qcv}) over "
                                        f"{lo}..{hi} without saying the query "
                                        f"produced no matching results")
                    counts["datasets_empty"] += 1
                    continue
                path, owned = got, True
            try:
                with open(path, newline="") as fh:
                    self._merge(ctx, fh, var, qcv, ch, scale, acc, sites,
                                counts, depth_hits, ch_index,
                                label=f"{year} {ds}", path=path)
            finally:
                if owned and os.path.exists(path):
                    os.remove(path)
        counts["sites"] = len(sites)
        counts["depth_samples"] = {str(k): int(v) for k, v in depth_hits.items()}
        yield str(year), self._pack(acc, counts), counts

    def _merge(self, ctx, fh, var, qcv, ch, scale, acc, sites, counts,
               depth_hits, ch_index, label="", path=""):
        r = csv.reader(fh)
        hdr = next(r, None)
        if hdr is None:
            # Not one line, header included. A served CSV always carries its
            # header, so this is a transfer that delivered nothing — an
            # absence, not a year in which the array reported nothing.
            ctx.note_absent(label or "?", f"{path or 'the dataset CSV'} is "
                                          f"empty — not even a header line")
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
            ctx.note_absent(
                label or "?",
                f"{var}/{qcv} absent from {path or 'the dataset CSV'} "
                f"(columns {names}) — that channel would stay NaN for every "
                f"row of this dataset-year while the year read as complete")
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
            # Whole seconds out of an ISO stamp, so exact.
            td = seconds_since_epoch(when)
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
            # `partial_ok`: this prefix is deliberately truncated — 8 MB of a
            # 1.4 GB member — so ending without the deflate end marker is the
            # expected outcome here and an error everywhere else.
            hdr = _socat_header_from(_zip_member_stream(io.BytesIO(head),
                                                        partial_ok=True))
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
                # SOCAT publishes the stamp as six integer fields — yr, mon,
                # day, hh, mm, ss — so the store's second IS the file's second
                # and nothing is interpolated. `ss` is written as "00." in the
                # synthesis, hence the float parse; a leap second (60) is
                # clamped to 59, the only rounding in this path.
                yr = int(p[col["yr"]])
                td = seconds_since_epoch(dt.datetime(
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
                # STILL NOT FATAL — an index must not fail because a probe did
                # — but the failure goes into plan.json rather than into a log
                # line, because this probe is the only measurement anyone has
                # of the remote layout the fetch depends on. A plan.json that
                # simply lacks the key reads as "the probe was not run"; this
                # says which of the two it was.
                out["slatrack_files_probe"] = {
                    "failed": f"{type(e).__name__}: {e}",
                    "note": ("the remote path layout is therefore still "
                             "unmeasured; the fetch refuses any mission-year "
                             "whose listing comes back empty rather than "
                             "treating it as a year with no data")}
                print(f"  ::warning::slatrack files probe failed: {e!r} — "
                      f"recorded in plan.json", flush=True)
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

        AN EMPTY MISSION-YEAR LISTING IS NOT A VERDICT, it is a question, and
        `_empty_year_verdict` answers it by MEASURING the mission's whole
        archive rather than by guessing between the two stories. See there.
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
            y_lo = max(lo, dt.date(year, 1, 1))
            y_hi = min(hi, dt.date(year, 12, 31))
            if not paths:
                part = self._empty_year_verdict(ctx, mid, year, lo, hi,
                                                y_lo, y_hi, base)
                if part is not None:
                    yield part
                continue
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
                    if len(got) < len(names):
                        # THE BATCH IS SHORT. `copernicusmarine.get` does not
                        # raise on a file it could not serve; it returns, and
                        # the days in the gap simply have no samples. One
                        # missing day of one mission is invisible in a store of
                        # 2e9 rows — nothing downstream can tell a day the
                        # altimeter did not fly from a day the transfer
                        # dropped — so the difference is NAMED here and the
                        # year is not marked.
                        lost = [n for n in names if n not in set(got)]
                        ctx.note_absent(
                            f"{label} {mid}",
                            f"copernicusmarine.get returned {len(got)} of the "
                            f"{len(names)} file(s) the listing named; missing "
                            f"{lost[:4]}" + (" …" if len(lost) > 4 else ""))
                    for n in got:
                        rows, counts = self._read_nc(ctx,
                                                     os.path.join(out, n), mid)
                        yield f"{label} {mid}", rows, counts
                finally:
                    shutil.rmtree(out, ignore_errors=True)

    def _full_listing(self, ctx, mid, base):
        """Every file the mission's dataset holds, listed ONCE per process.

        `copernicusmarine.get(dry_run=True)` with no filter and no regex is
        the archive's own answer to "what do you hold for this mission", and
        it is the measurement the empty-year verdict below is built on. One
        mission is asked at most once — the listing is the same for every
        year of that mission, and a dataset with a decade of daily files is
        not something to fetch thirty times. The cache hangs off `ctx`
        (created lazily) rather than off `self`, so it dies with the build
        rather than outliving it.
        """
        import copernicusmarine
        cache = getattr(ctx, "_cmems_full_listing", None)
        if cache is None:
            cache = {}
            ctx._cmems_full_listing = cache
        if mid not in cache:
            cache[mid] = self._response_paths(
                copernicusmarine.get(dry_run=True, **base))
            print(f"  {mid}: unfiltered listing holds "
                  f"{len(cache[mid])} original file(s)", flush=True)
        return cache[mid]

    @staticmethod
    def _basename_date(path):
        """The 8-digit date token of a BASENAME, or None. `_file_batches`'s rule.

        Read exactly as `_file_batches` reads it, so "the archive holds a file
        in this window" and "this file goes in this month's batch" can never
        disagree: the same regex over the same basename, and a token that is
        not a valid calendar date counts as no token at all.
        """
        m = re.search(r"(\d{8})", str(path).rsplit("/", 1)[-1])
        if not m:
            return None
        try:
            return dt.date(int(m.group(1)[:4]), int(m.group(1)[4:6]),
                           int(m.group(1)[6:8]))
        except ValueError:
            return None

    def _empty_year_verdict(self, ctx, mid, year, lo, hi, y_lo, y_hi, base):
        """An empty `*{year}*` listing -> a MEASURED answer, not a guess.

        A STAC window is a CLAIM about what a mission covers; the file listing
        is the MEASUREMENT of what the archive holds. They disagree, and the
        disagreement is not rare: ERS-1's 35-day product
        (`…_my_e1-l3-duacs_PT1S_202411`) declares 1992-10-23..1995-05-15 and
        holds NO 1994 file at all, because the 1994 geodetic phase is a
        SEPARATE dataset (`…_my_e1g-…`, 1994-04-10..1995-03-21) which lists
        and downloads perfectly well. Refusing the whole year there refuses a
        gap that is real — the same shape as the root CLAUDE.md Part 2 lore
        that a published time domain can run PAST the served archive.

        So the ambiguity the old refusal named is turned into a measurement:
        ask the archive for the mission's WHOLE listing (once, cached) and
        read the date token off every basename. Four definite answers, and
        only the fourth is a gap:

          * the unfiltered listing is EMPTY   -> absence (a dataset id or a
            layout problem; the mission is not there at all).
          * a basename carries no valid date  -> absence (the archive's files
            cannot be placed in time, so "none in this year" is unmeasurable).
          * a file's date falls INSIDE the mission-year window -> absence, and
            now a DEFINITE one: `*{year}*` missed files the archive holds, so
            the filter does not match the remote layout.
          * files exist, all dated, none in the window -> a MEASURED GAP.
            Not an absence: it is counted under its own name in counts.json
            (`mission_year_gap_measured`, with the measurement beside it in
            `gaps_measured`), printed, and the year is free to be marked on
            the strength of the other missions' rows.

        Returns the part to yield for that gap, or None when it noted an
        absence (§5.17: only a DEFINITE answer may be fatal).
        """
        full = self._full_listing(ctx, mid, base)
        if not full:
            ctx.note_absent(
                f"{year} {mid}",
                f"copernicusmarine.get(filter='*{year}*', dry_run=True) "
                f"listed NO original file, and neither did the UNFILTERED "
                f"listing of the whole mission, though the mission's STAC "
                f"window covers {lo}..{hi}. An archive that lists nothing at "
                f"all for a mission is a dataset id or a remote-layout "
                f"problem, not a gap in the record, so this year is not "
                f"marked")
            return None
        dates, undated = [], []
        for p in full:
            d = self._basename_date(p)
            if d is None:
                undated.append(str(p).rsplit("/", 1)[-1])
            else:
                dates.append(d)
        if undated:
            ctx.note_absent(
                f"{year} {mid}",
                f"copernicusmarine.get(filter='*{year}*', dry_run=True) "
                f"listed no file, and the mission's unfiltered listing holds "
                f"{len(full)} file(s) of which {len(undated)} carry no valid "
                f"8-digit date token in the basename ({undated[:3]}"
                + (" …" if len(undated) > 3 else "")
                + f"). Those files cannot be placed in time, so whether the "
                  f"archive holds {year} for this mission CANNOT BE MEASURED "
                  f"— and an unmeasured empty year is not a gap, so this year "
                  f"is not marked")
            return None
        inside = [d for d in dates if y_lo <= d <= y_hi]
        if inside:
            ctx.note_absent(
                f"{year} {mid}",
                f"copernicusmarine.get(filter='*{year}*', dry_run=True) "
                f"listed no file, but the mission's unfiltered listing holds "
                f"{len(inside)} file(s) dated inside {y_lo}..{y_hi} "
                f"({min(inside)}..{max(inside)}): the filter DOES NOT MATCH "
                f"THE REMOTE LAYOUT and missed {len(inside)} file(s) the "
                f"archive holds. That is a listing bug, not a gap, so this "
                f"year is not marked")
            return None
        first, last = min(dates), max(dates)
        print(f"  {mid} {year}: the archive's own listing holds {len(dates)} "
              f"file(s) spanning {first}..{last} and none in {year} — a gap "
              f"inside the STAC window ({lo}..{hi}), measured, not guessed",
              flush=True)
        counts = {"kept": 0, "mission_year_gap_measured": 1,
                  "gaps_measured": [{"mission": mid, "year": int(year),
                                     "files_total": len(dates),
                                     "archive_first": str(first),
                                     "archive_last": str(last),
                                     "stac_window": [str(lo), str(hi)]}]}
        return f"{year} {mid}", empty_rows(self.C), counts

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

        tcol = pick("time_s", "time", "datetime", "date", "juld",
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
        # `time_s` is the one column already in the store's own units: the
        # netCDF path resolves the axis's CF epoch itself (that epoch is the
        # one thing a frame does not carry) and hands the SECONDS straight
        # over, so no value is converted twice. It stays float64 here rather
        # than int so that a NaT can still be NaN and fall out of `keep` below;
        # `_pack` rounds it to int32 once, at the end.
        td = (np.asarray(t, np.float64) if str(tcol).lower() == "time_s"
              else self._time_seconds(t))
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
    def _time_seconds(t):
        """A time column -> SECONDS since START (1982-01-01), float64.

        Two shapes, because the toolbox's frame is datetime64 and a netCDF axis
        is a CF number. A datetime64 column is converted through the SAME
        `_cf_time_to_seconds` the netCDF path uses, with the epoch it already
        carries, so the two paths cannot drift apart: nanoseconds -> seconds
        since 1970 -> seconds since START. Timezone-aware input is moved to UTC
        and made naive first — the store's axis is naive UTC throughout.

        float64 rather than int64 ONLY so a NaT survives as NaN; `_pack` rounds
        to the nearest second and stores int32. A nanosecond stamp divided by
        1e9 is exact to well under a microsecond over this range, so the
        rounding is a formality here and a real (documented) one for the DUACS
        `days since 1950-01-01` axis.
        """
        arr = np.asarray(t)
        if arr.dtype.kind == "M":
            ns = arr.astype("datetime64[ns]").astype(np.int64)
            ns = ns.astype(np.float64)
            ns[np.asarray(np.isnat(arr.astype("datetime64[ns]")))] = np.nan
            return _cf_time_to_seconds(ns / 1e9,
                                       "seconds since 1970-01-01 00:00:00")
        if arr.dtype.kind == "O":
            # a tz-aware pandas column arrives as object under .to_numpy();
            # ask pandas to normalise it rather than guessing here.
            import pandas as pd
            s = pd.to_datetime(pd.Series(arr), utc=True)
            arr = s.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
            return SLATrackAdapter._time_seconds(arr)
        return _cf_time_to_seconds(np.asarray(arr, np.float64),
                                   "seconds since 1970-01-01 00:00:00")

    def _read_nc(self, ctx, path, mid):
        """The FIXTURE path: a netCDF -> the same frame shape, same filter.

        The variables are read into a mapping of arrays and handed to
        `_rows_from_frame`, so the fixture and the live pull share one filter
        implementation and a change to the keep rule cannot reach one path
        without the other. The CF `units` of the axis are honoured HERE, since
        they are the one thing a DataFrame does not carry, and the axis goes in
        as `time_s` — SECONDS since START, the store's own convention, which
        `_rows_from_frame` takes as-is rather than converting a second time.

        THE 1 Hz SAMPLES LAND ON DISTINCT SECONDS, and that is a property of
        the source that was checked rather than hoped for. DUACS publishes the
        axis as float64 `days since 1950-01-01 00:00:00`; float64 carries 53
        bits of mantissa, so at ~27,400 days (the end of the record) the gap
        between representable values is ~6e-12 days = **~0.5 microseconds**.
        One second is 2 million times that, so consecutive 1 Hz samples are
        distinct float64 days, and `round(seconds)` sends them to consecutive
        integers. This is exactly what float32 DAYS could not do: 84 s of
        resolution against a 1 s sampling rate (E-079 §10.1).

        HARDENED FOR THE REAL DUACS FILES, which the `files` route now hands
        it: the fill is taken from the MASK (`np.ma.filled(..., nan)`) rather
        than left as the raw fill value — these variables are scaled int16, so
        a raw fill would arrive as something like -2e5 and be counted as an
        out-of-bounds sample instead of a missing one; variable names are
        matched case-insensitively; the axis epoch may be any CF `<unit> since
        <date>` (DUACS writes "days since 1950-01-01 00:00:00", which
        `_cf_time_to_seconds` already handles); and a longitude axis published in
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
            # The store's contract is [-180, 180). The DUACS files carry a
            # few samples at EXACTLY +180.0 (measured 2026-09-14: 1994, 2011,
            # 2014, 2015, 2016 each failed the assembler's range assertion on
            # "lon runs -180.0..180.0"), and some archives use 0..360; both
            # fold with one exact subtraction, which touches no other value.
            if np.isfinite(lo_).any() and np.nanmax(lo_) >= 180.0:
                lo_ = np.where(lo_ >= 180.0, lo_ - 360.0, lo_)
            frame = {"latitude": la, "longitude": lo_}
            for v in CMEMS_VARS:
                got = rd(v.lower())
                frame[v] = np.full(len(t), np.nan) if got is None else got
        finally:
            ds.close()
        # ONE rounding, HERE, where the CF units are in scope: float64 seconds
        # since the epoch -> the nearest whole second. `_rows_from_frame` takes
        # `time_s` as-is and `_pack` casts it to int32.
        frame["time_s"] = np.rint(_cf_time_to_seconds(t, units))
        return self._rows_from_frame(ctx, frame, mid)


# ----------------------------------------------------------------- fishing --
# E-081 §1. Zenodo record 14982712, Global Fishing Watch's *Global AIS-based
# Apparent Fishing Effort Dataset* v3.0 (2025-03-11), CC BY-NC 4.0, served
# anonymously — no account, no token, no rate limit that a 13-file build meets.
ZENODO_RECORD = "14982712"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
ZENODO_FILE = ZENODO_API + "/files/{name}/content"
FISHING_DOI = "10.5281/zenodo.14982712"
FISHING_VERSION = "3.0.0"
FISHING_RELEASED = "2025-03-11"
FISHING_LICENCE = "CC BY-NC 4.0"
FISHING_LICENCE_URL = "https://creativecommons.org/licenses/by-nc/4.0/"
FISHING_CITATION = (
    "Global Fishing Watch (2025). Global AIS-based Apparent Fishing Effort "
    "Dataset, version 3.0 [Data set]. Zenodo. "
    "https://doi.org/10.5281/zenodo.14982712")
FISHING_ATTRIBUTION = "Powered by Global Fishing Watch"
FISHING_YEARS = tuple(range(2012, 2025))          # 2012 .. 2024, the 13 zips
FISHING_ZIP = "mmsi-daily-csvs-10-v3-{year}.zip"
FISHING_VESSELS = "fishing-vessels-v3.csv"
FISHING_VESSELS_GZ = "vessels.csv.gz"
FISHING_SCHEMA_FILE = "mmsi-daily-v3.schema.json"
FISHING_DOCS = ("README-mmsi-v3.txt", "README-known-issues-v3.txt",
                "README-fishing-vessels-v3.txt", FISHING_SCHEMA_FILE,
                "fishing-vessels-v3.schema.json")
# MEASURED off the 2012 zip, 2026-09-16 (see the adapter's docstring).
FISHING_COLUMNS = ("date", "cell_ll_lat", "cell_ll_lon", "mmsi", "hours",
                   "fishing_hours")
# The vessel table's 22 columns, read off the real file 2026-09-16. The parser
# needs four of them and asks only for those (a release that ADDS a column must
# not fail a build); the full list is here so the test fixture meets the same
# header the box does.
FISHING_VESSEL_COLUMNS = (
    "mmsi", "year", "flag_ais", "flag_registry", "flag_gfw",
    "vessel_class_inferred", "vessel_class_inferred_score",
    "vessel_class_registry", "vessel_class_gfw",
    "self_reported_fishing_vessel", "length_m_inferred", "length_m_registry",
    "length_m_gfw", "engine_power_kw_inferred", "engine_power_kw_registry",
    "engine_power_kw_gfw", "tonnage_gt_inferred", "tonnage_gt_registry",
    "tonnage_gt_gfw", "registries_listed", "active_hours", "fishing_hours")
FISHING_CELL_DEG = 0.1
FISHING_HALF_CELL = FISHING_CELL_DEG / 2.0        # ll corner -> cell CENTRE
FISHING_DAY_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})\.csv$")
# THE TOLERANCE on the two value assertions. The source prints four decimals,
# and `fishing_hours == hours` to the last digit is COMMON (2012's maxima are
# the same row), so the comparison has to admit equality with a margin rather
# than demand strict inequality.
FISHING_TOL = 1e-4
# A DAY IS 24 HOURS AND A CELL-DAY-VESSEL ROW IS NOT BOUNDED BY ONE. Measured
# over the whole 2012 file (6,257,384 rows): `hours` runs to 47.5633, and
# 9,773 of the first 60 days' 619,273 rows (1.6 %) exceed 24 — mostly in
# single cells of the southern North Sea carrying Belgian and Dutch MMSI. That
# is the source's own first known issue: an MMSI is not always one vessel
# ("it is not uncommon to see the same MMSI number being used by two or more
# vessels simultaneously", README-known-issues-v3.txt), so a row is the hours
# broadcast under one IDENTITY in one cell on one day, and two vessels sharing
# an identity broadcast up to 48. E-081 §2's `hours <= 24 + 1e-3` assertion is
# therefore FALSE AGAINST THE ARCHIVE and would refuse the real 2012 file on
# its second day. What is kept is the assertion's purpose — catch a shifted
# column or a misparsed field before it reaches the store — at a ceiling the
# data cannot reach innocently: a week of vessel-hours in one cell-day. A
# longitude misread as hours (|lon| <= 180) still trips it, and every row
# above 24 is COUNTED into store.json (`hours_over_24h`, `max_hours`) so the
# departure is measured rather than hidden. `--max-hours 24.001` restores the
# plan's literal rule for anyone who wants to see it fire.
FISHING_HOURS_CEILING = 168.0
FISHING_HOURS_DAY = 24.001
# rows per BYTE of zip, measured on 2012: 6,257,384 / 56,049,115. Used only to
# project the build's disk before it spends an hour fetching (§0.3).
FISHING_ROWS_PER_ZIP_BYTE = 6_257_384 / 56_049_115
# The gear classes v3.0's vessel table actually carries — all 16 of them,
# counted over its 773,165 rows on 2026-09-16, in ALPHABETICAL order so the
# code of a class is a property of this table and not of how often it happened
# to occur. 0 is "unknown": an MMSI the vessel table does not carry for that
# year. The table goes into store.json as `qc_codes`, because a consumer must
# never have to guess what a 7 in `qc.npy` means.
FISHING_GEAR = ("dredge_fishing", "drifting_longlines", "fishing",
                "fixed_gear", "other_purse_seines", "other_seines",
                "pole_and_line", "pots_and_traps", "purse_seines", "seiners",
                "set_gillnets", "set_longlines", "squid_jigger", "trawlers",
                "trollers", "tuna_purse_seines")
FISHING_GEAR_CODE = {g: i + 1 for i, g in enumerate(FISHING_GEAR)}


def md5_file(path, buf=1 << 22):
    """The md5 of a file — Zenodo's own checksum algorithm, so its `checksum`
    field (`md5:<hex>`) can be compared against the bytes that arrived."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(buf), b""):
            h.update(b)
    return h.hexdigest()


class FishingAdapter(SourceAdapter):
    """Global Fishing Watch apparent fishing effort — 0.1° cell, day, vessel.

    E-081 §1-§2. One row per (day, 0.1° cell, MMSI): `hours`, the hours that
    identity was broadcasting inside the cell that day, and `fishing_hours`,
    the part of them a neural network classed as fishing. The fleet is the
    only observing system in family 10 that is not an instrument: it is
    ~10,000 vessels in 2012 and ~95,000 in 2024 reporting where they are, and
    what they do there is a read-out of fronts, upwelling and productivity.

    THE LAYOUT, MEASURED 2026-09-16 by downloading
    `mmsi-daily-csvs-10-v3-2012.zip` (56,049,115 B, md5
    920f3434b53af9f3423f2087947b595a — Zenodo's own) and opening it:

      * the zip holds **one CSV per day**, `mmsi-daily-csvs-10-v3-<YYYY-MM-DD>
        .csv`, 366 members for 2012 and NOT in date order in the central
        directory (the first entry is 01-01, the second 08-25), so the members
        are sorted here and a day is looked up by the date in its basename.
      * every member carries a HEADER LINE, identical in all 366:
        `date,cell_ll_lat,cell_ll_lon,mmsi,hours,fishing_hours`.
      * `mmsi` is an UNQUOTED decimal integer (0 non-numeric in 6,257,384
        rows); `date` is `YYYY-MM-DD` and equals the member's own date;
        `cell_ll_lat` / `cell_ll_lon` are the cell's LOWER-LEFT corner to one
        decimal with trailing zeros stripped (`120`, not `120.0`), running
        -77.7..82.0 and -180.0..179.9 in 2012; `hours` and `fishing_hours`
        are plain decimals, `0` where nothing was measured.
      * 2012: 6,257,384 rows, 366 dates, 10,447 distinct MMSI — which is
        EXACTLY the 2012 row count of the vessel table, so the two files
        describe the same fleet.
      * `fishing_hours <= hours` held on every one of those rows; `hours` did
        NOT stay under 24 (see FISHING_HOURS_CEILING above); 68,932 rows
        (1.1 %) carry `hours == 0`, which is a real value and is kept.
      * a lean parse of the whole year measured **1.6 µs a row** (6.26 M rows
        in 10.1 s, binning included), i.e. ~17 min of CPU for the projected
        596 M rows of the whole archive.

    THE VESSEL TABLE, `fishing-vessels-v3.csv` (114,823,860 B, 773,165 rows,
    measured the same day): one row per (MMSI, YEAR) — never two, checked —
    with `flag_gfw`, `vessel_class_gfw` (16 classes), length, engine power and
    tonnage. It is downloaded ONCE in the index stage, md5-verified, kept in
    the work directory, and published beside the store as `vessels.csv.gz`,
    because a `platform` column of hashes is otherwise a dead end: the hash
    rule is `platform_hash(str(mmsi))`, so a consumer hashes the table's own
    MMSI to join back to a flag, a gear and a length.

    WHAT EACH COLUMN OF THE STORE IS.
      `time_s`  the day at 00:00:00 UTC. The source is daily; every row of a
                day carries the same second, and the pentad bin is
                `floor_divide(time_s, 432000)` like everywhere else.
      `lat/lon` the cell CENTRE = lower-left + 0.05°, wrapped into [-180, 180)
                after the float32 cast (`_pack`, the fb5d5ab rule).
      `values`  (fishing_hours, hours), float16, in HOURS.
      `platform` platform_hash(mmsi) — the same 60-bit sha1 prefix the other
                stores use for a string id. An MMSI is numeric and could have
                been kept as itself; it is hashed because E-081 §2 says so and
                because an MMSI is an IDENTITY, not a vessel (the source's own
                known issues: spoofing, reflagging, recycling), so a store
                that printed it invites a join nobody should make by eye.
      `qc`      the vessel's GEAR CLASS as a small integer — a categorical
                channel, not a quality grade (the store's `qc_codes` says so
                in words). 0 = unknown, i.e. the vessel table has no row for
                that (MMSI, year).
      `fp`      log2(11.132 km / 27.83 km) = -1.32 and log2(1 d / 5 d) = -2.32
                — a 0.1° cell and a day, in E-078 §2's units.

    NO CREDENTIALS, and the whole year is one request: each zip is <= 0.75 GB,
    downloaded to scratch, md5-verified against the Zenodo record, streamed
    member by member through `zipfile` + `csv` — NEVER extracted to disk — and
    deleted before the next year is asked for.
    """

    store = "fishing"
    title = ("Global Fishing Watch AIS apparent fishing effort, "
             "0.1° daily by vessel")
    channels = (("fishing_hours", "h", 0.0, FISHING_HOURS_CEILING),
                ("hours", "h", 0.0, FISHING_HOURS_CEILING))
    # 0.1° of latitude is 11.132 km against the 0.25° cell's 27.83 km.
    log2_fp = -1.32                      # log2(11.132 / 27.83) = -1.3219
    log2_dt = -2.32                      # log2(1 d / 5 d) = -2.3219, daily
    first_year = FISHING_YEARS[0]
    per_year = True
    qc_policy = (
        "THE `qc` COLUMN OF THIS STORE IS NOT A QUALITY GRADE — it is the "
        "vessel's GEAR CLASS, a categorical channel a consumer may use or "
        "ignore, and `qc_codes` in this file is the whole code table (0 = the "
        "vessel table carries no row for that MMSI in that year). The source "
        "has no per-row flag to grade: it IS the processed product, one row "
        "per (day, 0.1° cell, MMSI) from Global Fishing Watch's own AIS "
        "pipeline and fishing classifier, and the rows it considered bad are "
        "not in it. What the builder does assert, per row, is the physics of "
        "the two numbers: 0 <= fishing_hours <= hours, and hours <= 168 "
        "(--max-hours). A row failing either REFUSES the build rather than "
        "being dropped, because both would mean the columns were misread. "
        "hours > 24 is NOT a failure and is counted (`hours_over_24h`, "
        "`max_hours`): the source states that an MMSI is not always one "
        "vessel, so two vessels sharing an identity broadcast up to 48 hours "
        "in a cell-day — measured at 1.6 % of rows in 2012, to 47.56 h. "
        "`hours == 0` is a real value (1.1 % of 2012) and is kept: the "
        "vessel was in the cell and broadcast nothing measurable. Neither "
        "channel is ever NaN.")
    sources = (f"{ZENODO_API} (record {ZENODO_RECORD}, "
               f"doi {FISHING_DOI}, v{FISHING_VERSION}, {FISHING_LICENCE})",
               ZENODO_FILE.format(name=FISHING_ZIP.format(year="<year>")),
               ZENODO_FILE.format(name=FISHING_VESSELS))
    verified = ("2026-09-16: the Zenodo record listed (13 mmsi zips, 5.34 GB, "
                "md5 per file), the 2012 zip downloaded and md5-verified, its "
                "366 day members and 6,257,384 rows parsed, and the 773,165-"
                "row vessel table read for its 16 gear classes")
    notes = ("E-081 §1 records the source's own known issues verbatim: 2024 "
             "is provisional; AIS reception is uneven in space and time, so "
             "absence of effort is NOT absence of fishing; an MMSI is not "
             "always one vessel (spoofing, reflagging, recycling); and one "
             "gear class is assigned per MMSI over the whole record.")

    # -- the plan ----------------------------------------------------------
    def __init__(self):
        self._gear = None                # (mmsi * 10000 + year) -> code
        self._vessel_summary = None

    def years_in(self, ctx):
        """The archive years this window asks for. 2012..2024 and no others."""
        return [y for y in ctx.years if y in FISHING_YEARS]

    def _record(self, ctx):
        raw = fetch_first([ZENODO_API], attempts=ctx.a.attempts)
        if raw is None:
            sys.exit(f"{ZENODO_API} did not answer — Zenodo serves this "
                     f"record anonymously, so a 404 here means the record id "
                     f"moved and E-081 §1 must be re-verified before a build")
        rec = json.loads(raw)
        files = {}
        for f in rec.get("files") or []:
            ck = str(f.get("checksum") or "")
            files[f["key"]] = {"bytes": int(f.get("size") or 0),
                               "md5": ck.split(":", 1)[-1] if ck else None,
                               "url": ZENODO_FILE.format(name=f["key"])}
        return rec, files

    def index(self, ctx):
        years = self.years_in(ctx)
        if not years:
            sys.exit(f"the window {ctx.d_lo}..{ctx.d_hi} contains none of the "
                     f"archive's years {FISHING_YEARS[0]}..{FISHING_YEARS[-1]} "
                     f"— there is nothing to build")
        out = {"record": ZENODO_API, "doi": FISHING_DOI,
               "version": FISHING_VERSION, "released": FISHING_RELEASED,
               "licence": FISHING_LICENCE, "licence_url": FISHING_LICENCE_URL,
               "citation": FISHING_CITATION,
               "attribution": FISHING_ATTRIBUTION,
               "columns": list(FISHING_COLUMNS),
               "n_columns": len(FISHING_COLUMNS),
               "years": years}
        if ctx.source_dir:
            out["source"] = f"file://{os.path.join(ctx.source_dir, 'fishing')}"
            out["zips"] = {}
            for y in years:
                p = self._local_year(ctx, y)
                rec = {"name": FISHING_ZIP.format(year=y), "local": p}
                # A LOCAL ZIP IS HASHED HERE so `fetch_year` can check the
                # bytes it reads against the bytes the plan was written from.
                if os.path.isfile(p):
                    rec["md5"] = md5_file(p)
                    rec["bytes"] = os.path.getsize(p)
                out["zips"][str(y)] = rec
        else:
            rec, files = self._record(ctx)
            out["title"] = (rec.get("metadata") or {}).get("title")
            miss = [FISHING_ZIP.format(year=y) for y in years
                    if FISHING_ZIP.format(year=y) not in files]
            if FISHING_VESSELS not in files:
                miss.append(FISHING_VESSELS)
            if miss:
                sys.exit(f"the Zenodo record no longer carries {miss} — it "
                         f"lists {sorted(files)[:6]} … ({len(files)} files). "
                         f"E-081 §1 must be re-verified before a build.")
            # THE 13 ZIPS, BY NAME, SIZE AND CHECKSUM, INTO plan.json. This is
            # what `fetch_year` verifies each download against, so the number
            # a year is checked by was written down before the year was
            # fetched rather than read off the file that arrived.
            out["zips"] = {str(y): dict(files[FISHING_ZIP.format(year=y)],
                                        name=FISHING_ZIP.format(year=y))
                           for y in years}
            out["vessels"] = dict(files[FISHING_VESSELS],
                                  name=FISHING_VESSELS)
            out["docs"] = {n: files[n] for n in FISHING_DOCS if n in files}
            out["bytes"] = sum(v["bytes"] for v in out["zips"].values())
            # The archive's OWN declaration of the six columns, cheap and
            # anonymous — the preflight that a hosted build can afford before
            # it downloads a gigabyte (E-079 §4's "read one real record", in
            # the form this archive makes available).
            if FISHING_SCHEMA_FILE in files:
                sch = json.loads(http_bytes(
                    files[FISHING_SCHEMA_FILE]["url"]))
                got = tuple(c["name"] for c in sch)
                if got != FISHING_COLUMNS:
                    sys.exit(f"{FISHING_SCHEMA_FILE} declares {got}; this "
                             f"builder parses {FISHING_COLUMNS}. The table's "
                             f"layout has changed — E-081 §1 must be "
                             f"re-verified.")
                out["schema_declared"] = list(got)
        # THE VESSEL TABLE IS AN INDEX-STAGE JOB: it is one 0.11 GB file, it
        # is needed by every year, and its summary belongs in the plan.
        self.vessel_map(ctx, plan=out)
        out["vessels_summary"] = self._vessel_summary
        out["projected"] = self.projection(ctx, out)
        out["preflight"] = self._preflight_one_day(
            ctx, years[0], rec=(out.get("zips") or {}).get(str(years[0])))
        return out

    # -- disk, before an hour is spent on a download -----------------------
    def projection(self, ctx, plan):
        """How big this build will be, from the zip sizes and a measured rate.

        E-081 §5 budgeted "a few GB" for the assembled store. It is not: the
        whole archive is 5.34 GB of zip and 0.1116 rows per zip byte, so
        ~596 M rows at 31 B a row is **~18.5 GB of store** plus ~16 GB of
        per-year parts — more than three times a GitHub-hosted runner's whole
        disk. So the number is computed here, written into plan.json, and
        `fetch_preflight` refuses a build that cannot land, while the inputs
        are all it has cost (ml/CLAUDE.md §0.3).
        """
        zips = plan.get("zips") or {}
        nbytes = sum(int(v.get("bytes") or 0) for v in zips.values())
        rows = int(nbytes * FISHING_ROWS_PER_ZIP_BYTE)
        row_b = ROW_BYTES_FIXED + 2 * self.C
        return {
            "zip_bytes": nbytes,
            "rows_per_zip_byte": round(FISHING_ROWS_PER_ZIP_BYTE, 6),
            "rows_estimated": rows,
            "store_bytes": rows * row_b,
            # a part carries every column but `fp`, which the assembler
            # synthesises, and np.savez does not compress.
            "parts_bytes": rows * (row_b - 4),
            "need_bytes": rows * (2 * row_b - 4),
            "note": ("measured 2026-09-16: 6,257,384 rows in the 56,049,115-"
                     "byte 2012 zip. The parts and the store coexist on disk "
                     "until the build finishes, so `need_bytes` is both."),
        }

    def fetch_preflight(self, ctx):
        plan = read_json(os.path.join(ctx.root, "plan.json"), {})
        pr = plan.get("projected") or {}
        need = int(pr.get("need_bytes") or 0)
        if not need:
            return
        free = shutil.disk_usage(ctx.work).free
        print(f"  disk: this window projects {pr['rows_estimated']:,} row(s), "
              f"{pr['store_bytes'] / 1e9:.1f} GB of store and "
              f"{pr['parts_bytes'] / 1e9:.1f} GB of parts "
              f"({need / 1e9:.1f} GB together); {free / 1e9:.1f} GB free "
              f"under {ctx.work}")
        if free >= need * DISK_HEADROOM:
            return
        msg = (f"REFUSING to fetch {ctx.adapter.store}: the window "
               f"{ctx.d_lo}..{ctx.d_hi} projects {need / 1e9:.1f} GB of parts "
               f"+ store ({DISK_HEADROOM:g}x = "
               f"{need * DISK_HEADROOM / 1e9:.1f} GB with margin) and "
               f"{ctx.work} has {free / 1e9:.1f} GB free. A GitHub-hosted "
               f"runner has ~14 GB in total, which holds roughly four years "
               f"of this archive; the whole 2012-2024 record is ~18.5 GB of "
               f"store and needs a box (`runner: <box>` in "
               f"family10-build.yml, whose /opt/earth-cache also makes the "
               f"resume real). NOTE THAT SPLITTING IT ACROSS DISPATCHES DOES "
               f"NOT HELP: a year's parts persist and are skipped on a "
               f"resume, but the ASSEMBLY at the end of every fetch stage "
               f"reads every year in the window at once, so the machine that "
               f"finishes the archive needs room for all of it whatever order "
               f"the years arrived in. Two ways on: build a SHORTER WINDOW "
               f"here — the store records its own `date_range` and says what "
               f"it covers — or pass --allow-small-disk to try anyway.")
        if getattr(ctx.a, "allow_small_disk", False):
            print(f"  ::warning::{msg} — --allow-small-disk says try anyway")
            return
        sys.exit(msg)

    # -- the vessel table ---------------------------------------------------
    def _vessels_path(self, ctx):
        if ctx.source_dir:
            return os.path.join(ctx.source_dir, "fishing", FISHING_VESSELS)
        return os.path.join(ctx.scratch, FISHING_VESSELS)

    def vessel_map(self, ctx, plan=None):
        """MMSI x YEAR -> gear code, from the source's own vessel table.

        Downloaded once (md5-verified), kept in the work directory so a
        `--stage fetch` on its own does not refetch it, and held as ONE int
        key per row — `mmsi * 10000 + year` — because 773,165 tuple keys cost
        four times the memory for the same lookup.
        """
        if self._gear is not None:
            return self._gear
        p = self._vessels_path(ctx)
        want = ((plan or {}).get("vessels") or {}).get("md5")
        if not os.path.exists(p):
            if ctx.source_dir:
                sys.exit(f"{p} is missing — a --source-dir build needs the "
                         f"vessel table beside the year archives")
            rec_files = None
            if want is None:
                _rec, rec_files = self._record(ctx)
                want = (rec_files.get(FISHING_VESSELS) or {}).get("md5")
            url = ZENODO_FILE.format(name=FISHING_VESSELS)
            print(f"  vessels: {url}", flush=True)
            http_to_file(url, p)
        got = md5_file(p)
        if want and got != want:
            os.remove(p)
            sys.exit(f"{FISHING_VESSELS}: md5 {got} != the record's {want} — "
                     f"the download is not the file Zenodo describes. It has "
                     f"been removed; re-run the index stage.")
        gear = {}
        per_year, hist, flags = {}, {}, {}
        unseen = {}
        rows = 0
        with open(p, newline="") as fh:
            r = csv.DictReader(fh)
            need = ("mmsi", "year", "vessel_class_gfw", "flag_gfw")
            missing = [c for c in need if c not in (r.fieldnames or [])]
            if missing:
                sys.exit(f"{FISHING_VESSELS} is missing {missing}; its header "
                         f"reads {r.fieldnames}")
            for d in r:
                rows += 1
                m, y = (d["mmsi"] or "").strip(), (d["year"] or "").strip()
                if not m.isdigit() or not y.isdigit():
                    continue
                cls = (d["vessel_class_gfw"] or "").strip()
                code = FISHING_GEAR_CODE.get(cls, 0)
                if cls and code == 0:
                    unseen[cls] = unseen.get(cls, 0) + 1
                gear[int(m) * 10000 + int(y)] = code
                per_year[y] = per_year.get(y, 0) + 1
                hist[cls or "(blank)"] = hist.get(cls or "(blank)", 0) + 1
                fl = (d["flag_gfw"] or "").strip()
                flags[fl] = flags.get(fl, 0) + 1
        if unseen:
            # NOT fatal and NOT silent: a release that adds a gear class must
            # not fail a build, and it must not be absorbed into "unknown"
            # without anybody being told which class it was.
            print(f"::warning::{FISHING_VESSELS} carries gear class(es) this "
                  f"builder has no code for: {sorted(unseen)} — they are "
                  f"stored as qc 0 (unknown) and counted in store.json's "
                  f"`gear_classes_unseen`. Add them to FISHING_GEAR (at the "
                  f"END, so no existing code moves) and rebuild to separate "
                  f"them.", flush=True)
        self._gear = gear
        self._vessel_summary = {
            "file": FISHING_VESSELS, "md5": got,
            "bytes": os.path.getsize(p), "rows": rows,
            "vessels_per_year": {k: per_year[k] for k in sorted(per_year)},
            "gear_histogram": {k: hist[k] for k in sorted(hist)},
            "gear_classes_unseen": unseen,
            "n_flags": len(flags),
            "published_as": FISHING_VESSELS_GZ,
        }
        return self._gear

    # -- the year's archive -------------------------------------------------
    def _local_year(self, ctx, year):
        """A --source-dir year: the zip if there is one, else a day directory."""
        base = os.path.join(ctx.source_dir, "fishing")
        z = os.path.join(base, FISHING_ZIP.format(year=year))
        if os.path.exists(z):
            return z
        d = os.path.join(base, FISHING_ZIP.format(year=year)[:-4])
        return d if os.path.isdir(d) else z

    def _members(self, ctx, year, note=True, rec=None):
        """(source label, zipfile or None, {member -> opener}), NOT extracted.

        A zip member is opened with `ZipFile.open`, which inflates on the fly;
        nothing is ever written out of the archive, live or local. A
        `--source-dir` may hold the same day CSVs in a plain DIRECTORY instead
        (the other layout the tests use) — same names, same header, no zip.

        A ZIP IS md5'd BEFORE IT IS READ, whichever side it came from. Zenodo
        publishes the checksum of every file in the record and the index stage
        wrote it into plan.json; a `--source-dir` zip is hashed at index time
        for the same reason. A short or altered zip inflates perfectly up to
        the cut and then raises somewhere unrelated, or worse does not raise
        at all — so it is caught here, before a single row is parsed, and the
        year is an ABSENCE rather than a quieter ocean.
        """
        # `rec` is the plan's entry for this year — its name, url and md5.
        # The INDEX stage passes the one it has just written, because
        # plan.json on disk is still the previous run's and checking a fresh
        # archive against a stale checksum would refuse the very build that is
        # re-indexing it.
        if rec is None:
            plan = read_json(os.path.join(ctx.root, "plan.json"), {})
            rec = ((plan.get("zips") or {}).get(str(year)) or {})
        name = rec.get("name") or FISHING_ZIP.format(year=year)
        if ctx.source_dir:
            p = self._local_year(ctx, year)
            if os.path.isdir(p):
                names = sorted(n for n in os.listdir(p) if n.endswith(".csv"))
                return (f"file://{p}", None,
                        {n: (lambda q=os.path.join(p, n): open(q, "rb"))
                         for n in names})
            src, dest = f"file://{p}", p
            if not os.path.exists(dest):
                if note:
                    ctx.note_absent(str(year),
                                    f"{p} does not exist "
                                    f"(--source-dir {ctx.source_dir})")
                return None, None, {}
        else:
            src = rec.get("url") or ZENODO_FILE.format(name=name)
            dest = os.path.join(ctx.scratch, name)
            if not os.path.exists(dest):
                print(f"  {year}: {src}", flush=True)
                got = fetch_first([src], dest, attempts=ctx.a.attempts)
                if got is None:
                    if note:
                        ctx.note_absent(str(year),
                                        f"{src} 404'd — the record no longer "
                                        f"serves {name}")
                    return None, None, {}
        want = rec.get("md5")
        if want:
            got = md5_file(dest)
            if got != want:
                if not ctx.source_dir:
                    os.remove(dest)          # never our file to delete locally
                why = (f"{name}: md5 {got} != the {'Zenodo record' if not ctx.source_dir else 'index stage'}"
                       f"'s {want} — this is not the file that was indexed"
                       f"{', and the download has been removed' if not ctx.source_dir else ''}")
                if note:
                    ctx.note_absent(str(year), why)
                    return None, None, {}
                raise IOError(why)
            print(f"  {year}: md5 ok ({want[:12]}…, "
                  f"{os.path.getsize(dest) / 1e9:.2f} GB)", flush=True)
        else:
            print(f"::warning::{name} has no md5 in plan.json — this archive "
                  f"cannot be checked against what the index saw")
        zf = zipfile.ZipFile(dest)
        return (src, zf, {n: (lambda q=n: zf.open(q))
                          for n in sorted(zf.namelist()) if n.endswith(".csv")})

    def _days(self, ctx, year):
        """The days of `year` this window asks for, as date objects."""
        lo = max(ctx.d_lo, dt.date(year, 1, 1))
        hi = min(ctx.d_hi, dt.date(year, 12, 31))
        out, d = [], lo
        while d <= hi:
            out.append(d)
            d += dt.timedelta(days=1)
        return out

    def _preflight_one_day(self, ctx, year, rec=None):
        """Parse ONE real day before the index stage says the plan is good.

        Only where it is free: a `--source-dir` build has the file on disk, and
        a live build would have to download a whole year's zip to read one day
        — so there the archive's own schema.json is the preflight (above) and
        this reports that it was not run.
        """
        if not ctx.source_dir:
            return {"parsed": False,
                    "why": ("a live build would have to download the year's "
                            "whole zip to read one day; the record's own "
                            f"{FISHING_SCHEMA_FILE} is checked instead")}
        _src, zf, members = self._members(ctx, year, note=False, rec=rec)
        try:
            for name in sorted(members):
                with members[name]() as fh:
                    rows, counts = self._parse_day(ctx, fh, name, year)
                return {"parsed": True, "member": name,
                        "rows": int(len(rows["bin"])), "counts": counts}
        finally:
            if zf is not None:
                zf.close()
        return {"parsed": False, "why": "the year holds no CSV member"}

    def fetch_year(self, ctx, year):
        gear = self.vessel_map(ctx)                 # loaded once per process
        src, zf, members = self._members(ctx, year)
        if not members:
            if src is not None:
                ctx.note_absent(str(year), f"{src} holds no CSV member at all")
            return
        by_day = {}
        for n in members:
            m = FISHING_DAY_RE.search(os.path.basename(n))
            if m:
                by_day[dt.date(*(int(x) for x in m.groups()))] = n
        days = self._days(ctx, year)
        missing = [d for d in days if d not in by_day]
        if missing:
            # A SHORT ZIP IS AN ABSENCE, NOT AN EMPTY DAY. The day list comes
            # off the calendar, not off the archive's listing, so a zip that
            # lost members is visible here rather than as a store with a
            # quieter ocean in March.
            ctx.note_absent(
                f"{year}", f"{src} holds no member for "
                           f"{', '.join(str(d) for d in missing[:6])}"
                           f"{' …' if len(missing) > 6 else ''} "
                           f"({len(missing)} of {len(days)} day(s) in range) "
                           f"— this year's archive is short, and a day with "
                           f"no file is indistinguishable from an ocean with "
                           f"no vessels once it is in the store")
        total = {"rows_read": 0, "rows_packed": 0}
        try:
            for d in days:
                n = by_day.get(d)
                if n is None:
                    continue
                with members[n]() as fh:
                    rows, counts = self._parse_day(ctx, fh, n, year, gear)
                total["rows_read"] += int(counts.get("rows_read", 0))
                total["rows_packed"] += int(len(rows["bin"]))
                yield f"{d:%Y-%m-%d}", rows, counts
        finally:
            if zf is not None:
                zf.close()
            if not ctx.source_dir:
                p = os.path.join(ctx.scratch, FISHING_ZIP.format(year=year))
                if os.path.exists(p):
                    os.remove(p)                    # one zip on disk at a time
        # THE YEAR'S OWN RECONCILIATION (E-081 deliverable 1): every data row
        # the archive held for this year either became a store row or was
        # dropped for a reason with a name. A row that vanished between the
        # two is a parse fault, and it is an ABSENCE — the year is not marked
        # and the retry re-reads the whole zip.
        dropped = total["rows_read"] - total["rows_packed"]
        counts = {"rows_read_year": total["rows_read"],
                  "rows_packed_year": total["rows_packed"],
                  "days_expected": len(days), "days_found":
                      len([d for d in days if d in by_day])}
        if dropped < 0:
            ctx.note_absent(str(year),
                            f"the parser produced {total['rows_packed']:,} "
                            f"row(s) from {total['rows_read']:,} source "
                            f"row(s) — more out than in, which is a parse "
                            f"fault, not an archive gap")
        yield f"{year} reconciled", empty_rows(self.C), counts

    def _parse_day(self, ctx, fh, label, year, gear=None):
        gear = self._gear if gear is None else gear
        r = csv.reader(io.TextIOWrapper(fh, "utf-8", newline=""))
        hdr = next(r, None)
        if hdr is None:
            ctx.note_absent(f"{year} {label}",
                            "the day's CSV is empty — not even a header line, "
                            "so the member did not deliver a day")
            return empty_rows(self.C), {"empty_file": 1}
        cols = tuple(h.strip() for h in hdr)
        if cols != FISHING_COLUMNS:
            raise ValueError(
                f"{label}: the header reads {cols}; this builder parses "
                f"{FISHING_COLUMNS}. E-081 §1 must be re-verified — a column "
                f"order that has moved would put longitude in `hours`.")
        max_h = float(getattr(ctx.a, "max_hours", 0) or FISHING_HOURS_CEILING)
        t_l, lat_l, lon_l, plat_l, qc_l, f_l, h_l = [], [], [], [], [], [], []
        counts = {"rows_read": 0, "drop_out_of_range": 0, "drop_no_position": 0,
                  "drop_bad_number": 0, "hours_over_24h": 0, "rows_zero_hours": 0,
                  "qc_unknown": 0}
        max_seen = 0.0
        last_date, t_day = None, None
        for row in r:
            counts["rows_read"] += 1
            try:
                d, la_s, lo_s, m_s, h_s, f_s = row
            except ValueError:
                counts["drop_bad_number"] += 1
                continue
            if d != last_date:
                try:
                    t_day = seconds_since_epoch(parse_date(d))
                except (ValueError, TypeError):
                    counts["drop_out_of_range"] += 1
                    continue
                last_date = d
            if not (ctx.t_lo <= t_day <= ctx.t_hi):
                counts["drop_out_of_range"] += 1
                continue
            try:
                la = float(la_s) + FISHING_HALF_CELL
                lo_ = float(lo_s) + FISHING_HALF_CELL
                h = float(h_s)
                f = float(f_s)
            except ValueError:
                counts["drop_bad_number"] += 1
                continue
            if not (abs(la) <= 90.0):
                counts["drop_no_position"] += 1
                continue
            # THE TWO VALUE ASSERTIONS, ON EVERY ROW. Neither can be true of a
            # correctly parsed row and false of the archive, so a failure is a
            # column that moved, not a datum to drop.
            if not (f == f and h == h) or f < -FISHING_TOL or h < -FISHING_TOL:
                raise ValueError(
                    f"{label}: hours={h_s!r} fishing_hours={f_s!r} is not a "
                    f"pair of non-negative numbers (row {row!r})")
            if f > h + FISHING_TOL:
                raise ValueError(
                    f"{label}: fishing_hours {f} EXCEEDS hours {h} (row "
                    f"{row!r}). The fishing hours are a PART of the broadcast "
                    f"hours by the source's own definition, and 0 of "
                    f"6,257,384 rows of 2012 broke it — so this is the "
                    f"columns having moved, and refusing is the only honest "
                    f"answer. E-081 §5's falsifier (3).")
            if h > max_h:
                raise ValueError(
                    f"{label}: hours {h} exceeds the ceiling {max_h} (row "
                    f"{row!r}). A cell-day-identity can exceed 24 h — an MMSI "
                    f"is not always one vessel — but not a week of them; this "
                    f"is a misread column. --max-hours changes the ceiling.")
            if h > FISHING_HOURS_DAY:
                counts["hours_over_24h"] += 1
            if h == 0.0:
                counts["rows_zero_hours"] += 1
            if h > max_seen:
                max_seen = h
            code = gear.get(int(m_s) * 10000 + year, 0) if m_s.isdigit() else 0
            if code == 0:
                counts["qc_unknown"] += 1
            t_l.append(t_day)
            lat_l.append(la)
            lon_l.append(lo_)
            f_l.append(f)
            h_l.append(h)
            plat_l.append(platform_hash(m_s.strip()))
            qc_l.append(code)
        counts["max_hours"] = max_seen
        vals = np.empty((len(t_l), self.C), np.float64)
        if t_l:
            vals[:, 0] = f_l
            vals[:, 1] = h_l
        return _pack(t_l, lat_l, lon_l, vals, plat_l, qc_l, self.C), counts

    # -- what this store publishes beside its columns ----------------------
    def extra_files(self, ctx, dest):
        """`vessels.csv.gz` — the vessel table, gzipped, beside the store.

        It goes into store.json's sha256 block like every column, so the
        publish uploads it, the restore check verifies it, and
        `family10_store.resolve` brings it down with the store. E-081 §2: it
        is the only way a consumer turns a platform hash back into a flag, a
        gear class or a length.
        """
        src = self._vessels_path(ctx)
        if not os.path.exists(src):
            sys.exit(f"cannot finish the fishing store: {src} is missing, and "
                     f"E-081 §2 publishes the vessel table beside it. Re-run "
                     f"the index stage (it downloads and md5-verifies it).")
        out = os.path.join(dest, FISHING_VESSELS_GZ)
        with open(src, "rb") as fi, gzip.GzipFile(out, "wb", mtime=0) as fo:
            shutil.copyfileobj(fi, fo, CHUNK)
        print(f"  vessels: {os.path.getsize(src) / 1e6:.0f} MB -> "
              f"{FISHING_VESSELS_GZ} {os.path.getsize(out) / 1e6:.0f} MB")
        return {FISHING_VESSELS_GZ: out}

    def extra_meta(self, ctx, dest, N, values):
        """E-081 deliverable 2: the source block, the licence, the code table
        and the four measured totals.

        The two sums are float64 accumulations over the STORE's float16
        values, in blocks — so they are the numbers the monthly grid has to
        reproduce (E-081 §5's falsifier 4), not the numbers the parser saw
        before the cast.
        """
        tot = np.zeros(self.C, np.float64)
        for lo in range(0, int(N), STAT_CHUNK):
            blk = np.asarray(values[lo:lo + STAT_CHUNK], np.float64)
            tot += np.nansum(blk, axis=0)
        vs = self._vessel_summary or {}
        return {
            "source": {
                "name": "Global Fishing Watch — Global AIS-based Apparent "
                        "Fishing Effort Dataset",
                "doi": FISHING_DOI, "version": FISHING_VERSION,
                "released": FISHING_RELEASED, "record": ZENODO_API,
                "citation": FISHING_CITATION,
                "table": "mmsi-daily-csvs-10-v3-<year>.zip — one row per "
                         "(day, 0.1° cell, MMSI)",
                "known_issues": self.notes,
            },
            "licence": FISHING_LICENCE,
            "licence_url": FISHING_LICENCE_URL,
            "attribution": FISHING_ATTRIBUTION,
            "qc_codes": {
                "meaning": "the vessel's GEAR CLASS (vessel_class_gfw in the "
                           "source's vessel table), NOT a quality grade",
                "0": "unknown — the vessel table has no row for this MMSI in "
                     "this year",
                **{str(c): g for g, c in sorted(FISHING_GEAR_CODE.items(),
                                                key=lambda kv: kv[1])},
            },
            "fishing_hours_total": float(tot[0]),
            "hours_total": float(tot[1]),
            "totals_note": ("float64 sums over the store's own float16 "
                            "values, in blocks of "
                            f"{STAT_CHUNK:,} rows — the numbers the monthly "
                            "grid must reproduce"),
            "vessels_per_year": vs.get("vessels_per_year"),
            "gear_histogram": vs.get("gear_histogram"),
            "gear_classes_unseen": vs.get("gear_classes_unseen"),
            "vessel_table": {k: vs.get(k) for k in
                             ("file", "md5", "bytes", "rows", "published_as")},
            "grid": {
                "file": f"{FISHING_GRID_DIR}/{FISHING_GRID_FILE}",
                "note": ("the monthly 0.25° sum of these rows, written by "
                         "`--stage grid` and indexed by "
                         "ml/publish_fishing_index.py into "
                         "data/fishing_index.json"),
            },
        }


ADAPTERS = {a.store: a for a in
            (GDPAdapter, GTMBAAdapter, SOCATAdapter, SLATrackAdapter,
             FishingAdapter)}


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


def _cf_time_to_seconds(t, units):
    """A CF `<unit> since <date>` axis -> SECONDS since 1982-01-01, float64.

    The epoch offset is computed as an exact integer number of seconds and
    added last, so the only inexactness is whatever the source's own axis
    carries. The caller rounds to the nearest second (`_read_nc`).
    """
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
    per_unit_s = {"days": 86400.0, "day": 86400.0,
                  "hours": 3600.0, "hour": 3600.0,
                  "minutes": 60.0, "minute": 60.0,
                  "seconds": 1.0, "second": 1.0,
                  "milliseconds": 1e-3, "microseconds": 1e-6}.get(unit)
    if per_unit_s is None:
        raise ValueError(f"unknown time unit {unit!r} in {units!r}")
    off = int(round(
        (b - dt.datetime(START.year, START.month, START.day)).total_seconds()))
    return np.asarray(t, np.float64) * per_unit_s + off


def _pack(t, lat, lon, values, platform, qc, C, time_dtype="int32"):
    """Lists (or arrays) -> the seven store columns, bins computed, lon wrapped.

    `t` is SECONDS since 1982-01-01T00:00:00Z — an integer array from the
    calendar adapters, a float64 one from the netCDF axis, which is rounded to
    the nearest second HERE and only here.

    THE BIN IS EXACT INTEGER ARITHMETIC on the value that is actually stored:
    `floor_divide(time_s, 432000)`. Under v1 this function had to cast the
    parsed float64 down to float32 FIRST and derive the bin from the cast
    value, because float32 days resolve ~0.001 d at the end of the record and a
    timestamp a microsecond before a pentad boundary could round up across it —
    leaving a row in bin b whose stored timestamp read as bin b+1, which the
    reader (`dt_days = 5*(b+1) - t >= 0`) would then never return for its own
    anchor. int32 seconds have no such cast: the stored value IS the parsed
    value, so the store is self-consistent by construction rather than by
    precaution, and the whole class of fault is gone rather than guarded.

    THE 2050 LIMIT IS REFUSED HERE, not discovered later. int32 seconds reach
    2050-01-19T03:14:07Z; a row past it would wrap to 1913 and look like an
    ordinary pre-epoch observation.

    `time_dtype="int64"` (E-082, schema 3) lifts that limit — the column can
    then hold any second the int16 `bin` can index (1533 .. 2430), and the bin
    range check below is the one that applies. Family 10 never passes it.
    """
    if time_dtype not in TIME_DTYPES:
        raise ValueError(f"time_dtype {time_dtype!r} is not int32 or int64")
    n = len(t)
    if n == 0:
        return empty_rows(C, time_dtype)
    ts = np.asarray(t)
    if ts.dtype.kind == "f":
        if not np.isfinite(ts).all():
            raise ValueError(
                f"{int((~np.isfinite(ts)).sum())} of {n} timestamp(s) are NaN "
                f"or infinite by the time they reach _pack — an adapter's keep "
                f"rule let a row without a time through, and it would be "
                f"stored as an arbitrary second")
        s = np.rint(np.asarray(ts, np.float64)).astype(np.int64)
    else:
        s = ts.astype(np.int64)
    if time_dtype == "int32" and s.size and (
            s.min() < f10.TIME_S_MIN or s.max() > f10.TIME_S_MAX):
        raise ValueError(
            f"time_s runs {int(s.min())}..{int(s.max())} s, outside int32 — "
            f"family {FAMILY_VERSION}'s time column spans "
            f"{f10.TIME_S_MIN_DATE} .. {f10.TIME_S_MAX_DATE} and this build "
            f"reaches past it. A row beyond 2050 would WRAP into 1913 and read "
            f"as an ordinary pre-epoch observation, so it is refused. Widening "
            f"the column to int64 doubles the store and is the change to make "
            f"when the archives get there.")
    b = f10.bin_of_seconds(s)
    if b.size and (b.min() < BIN_MIN_INT16 or b.max() > BIN_MAX_INT16):
        raise ValueError(f"bin {b.min()}..{b.max()} does not fit int16 — the "
                         f"axis has outgrown the column's dtype")
    v = np.asarray(values, np.float64).reshape(n, C)
    # The [-180, 180) invariant has to hold in the COLUMN's dtype, not in the
    # float64 it was wrapped in: a source longitude of 179.99999 wraps to
    # itself in float64 and then ROUNDS to 180.0 when cast to float32, and
    # check_store (rightly) refuses the store. Measured on the 1994 altimeter
    # year, 2026-09-14, after the float64-only fold had already been added —
    # so the wrap is applied once more after the cast, in float32, where
    # 180.0f - 360 is exactly -180.0f.
    lo32 = f10.wrap_lon(lon).astype(np.float32)
    lo32 = np.where(lo32 >= np.float32(180.0), lo32 - np.float32(360.0), lo32)
    return {
        "bin": b.astype(np.int16),
        "time_s": s.astype(TIME_DTYPES[time_dtype]),
        "lat": np.asarray(lat, np.float64).astype(np.float32),
        "lon": lo32.astype(np.float32),
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


def _zip_member_stream(fh, size=CHUNK, partial_ok=False):
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
            # THE STREAM ENDED BEFORE THE DEFLATE MEMBER DID. `d.eof` is the
            # decompressor saying it saw the member's end marker; without it
            # the body was cut mid-transfer, and this generator used to end
            # exactly as it does on a clean finish. Nothing above could tell:
            # `fetch_stream` had long since found the data header, so `col` is
            # set, no exception is raised, every year in range is marked done
            # and a SHORT socat store is published — 1.4 GB of source with no
            # length check anywhere (the live path is a bare `urlopen`, not
            # `http_to_file`, so Content-Length is never compared). A
            # truncation that reaches the store is unfindable afterwards; one
            # that raises here costs a retry.
            if not d.eof and not partial_ok:
                raise IOError(
                    "the zip member ended without its deflate end marker — "
                    "the transfer was cut short. Everything read so far is a "
                    "PREFIX of the synthesis file, and a prefix parses "
                    "perfectly: it would have produced a store that is short "
                    "by however much did not arrive, with every year marked "
                    "done. Re-run the fetch stage (it re-reads the file from "
                    "the start; socat's resume granularity is the whole "
                    "pass).")
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

    def __init__(self, a, adapter=None, layout=None):
        self.a = a
        # E-082: family 1 hands in its own adapter and layout; family 10
        # passes neither and gets exactly what it always got.
        self.adapter = adapter if adapter is not None else ADAPTERS[a.store]()
        self.layout = layout if layout is not None else Layout()
        self.time_dtype = time_dtype_name(self.adapter)
        self.schema_version = SCHEMA_OF_TIME_DTYPE[self.time_dtype]
        self._net0 = NET_BYTES["n"]
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
        # THE WINDOW IS IN SECONDS, like the column. `t_hi` is the last
        # instant of the last requested day — 23:59:59 — rather than midnight
        # of the day after, so the inclusive bound needs no epsilon and the
        # bin it lands in is the bin of a real observation.
        self.t_lo = seconds_since_epoch(self.d_lo)
        self.t_hi = seconds_since_epoch(self.d_hi) + SECONDS_PER_DAY - 1
        if self.time_dtype == "int32" and (self.t_lo < f10.TIME_S_MIN
                                           or self.t_hi > f10.TIME_S_MAX):
            sys.exit(f"{a.store}: the window {self.d_lo} .. {self.d_hi} leaves "
                     f"int32 seconds ({f10.TIME_S_MIN_DATE} .. "
                     f"{f10.TIME_S_MAX_DATE}); this adapter is schema 2. A "
                     f"record that long needs time_dtype = 'int64'.")
        self.b_lo = int(f10.bin_of_seconds(self.t_lo))
        self.b_hi = int(f10.bin_of_seconds(self.t_hi))
        self.years = list(range(self.d_lo.year, self.d_hi.year + 1))
        self.qc_keep = int(getattr(a, "qc_keep", 2) or 2)
        self.check_chunk = int(getattr(a, "check_chunk_rows", 0)
                               or CHECK_CHUNK_ROWS)
        self.prog = Progress(self.root)
        self._cmems_missions = None
        self.socat_columns = None
        # EVERY INPUT AN ADAPTER COULD NOT READ, collected here and answered by
        # `stage_fetch` — the family-7 shape (commit fd3b446). An adapter never
        # decides what a missing input means: it NAMES the unit it could not
        # read and carries on filling the year, and the stage then refuses to
        # mark that year and refuses to mark itself, so a resume retries the
        # year instead of inheriting a hole. Nothing in here is a legitimate
        # gap — those are counted into counts.json by their own names
        # (`empty_month`, `datasets_empty`) and never reach this list.
        self.absent = []
        self.degraded_years = []

    def note_absent(self, unit, why):
        """An input this build needed and could not read. NOT a gap: an ABSENCE.

        `unit` is the resumable unit it belongs to (`"2015"`, `"2015-03"`,
        `"2015 <mission>"`), `why` says which file or url and what is missing.
        """
        self.absent.append({"unit": str(unit), "why": str(why)})
        print(f"  ::warning::{self.a.store}: {unit} — {why}", flush=True)

    def absent_years(self):
        """The YEARS the absences touch: the first 4-digit token of each unit."""
        out = set()
        for e in self.absent:
            m = re.match(r"(-?\d{4})", e["unit"])
            if m:
                out.add(int(m.group(1)))
        return out

    def year_dir(self, year):
        return os.path.join(self.parts, str(year))

    # -- E-082: bytes off the network, for the probe ------------------------
    @property
    def bytes_fetched(self):
        return NET_BYTES["n"] - self._net0

    def reset_bytes(self):
        self._net0 = NET_BYTES["n"]

    def count_bytes(self, n):
        """A LOCAL read standing in for a download (`--source-dir`)."""
        count_bytes(n)

    def hub(self):
        return self.layout.hub()


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
        want = TIME_DTYPES[self.ctx.time_dtype]
        if rows["time_s"].dtype != want:
            # int32 -> int64 is exact, so a schema-3 adapter that packed with
            # the default dtype is widened here; the other way is a store
            # that cannot hold its rows, and is refused rather than wrapped.
            if np.dtype(want).itemsize < rows["time_s"].dtype.itemsize:
                raise ValueError(
                    f"{self.ctx.adapter.store}: rows carry time_s as "
                    f"{rows['time_s'].dtype} and the store is "
                    f"{self.ctx.time_dtype} — pack with the adapter's own "
                    f"time_dtype (SourceAdapter.pack)")
            rows = dict(rows)
            rows["time_s"] = rows["time_s"].astype(want)
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
            # A `max_*` COUNTER IS A MAXIMUM, NOT A TALLY. Everything else
            # here is a count of rows or files and adds up; `max_hours` is the
            # largest value any row carried, and summing one per day made the
            # 2012 build report 15,281 hours where the real maximum is 47.56.
            # The prefix is the rule so a new measurement gets it for free.
            if k.startswith("max_"):
                into[k] = max(into[k], v) if k in into else v
            else:
                into[k] = into.get(k, 0) + v
        elif isinstance(v, list):
            # A LIST IS A LEDGER OF MEASUREMENTS, so it CONCATENATES. It used
            # to fall into the `else` below and OVERWRITE, which meant the
            # last part of a year was the only one whose measurement reached
            # counts.json — and `gaps_measured` (one entry per mission-year
            # the archive's own listing shows is empty) is exactly a list with
            # one entry per part. A counter that survives and a measurement
            # that does not is the worst of both.
            cur = into.get(k)
            into[k] = (list(cur) if isinstance(cur, list) else []) + list(v)
        else:
            into[k] = v
    return into


def read_parts(ctx):
    """Every part of every year in range, in year order. Yields arrays."""
    for y, p in _part_paths(ctx):
        with np.load(p) as z:
            yield y, {k: z[k] for k in ROW_KEYS}


def year_part_names(ctx, y):
    """The `.npz` files a year directory holds, sorted. Never a marker read."""
    d = ctx.year_dir(y)
    if not os.path.isdir(d):
        return []
    return sorted(n for n in os.listdir(d) if n.endswith(".npz"))


def parts_preflight(ctx):
    """REFUSE to assemble parts that no marker claims (E-079 §4, §5.21).

    Both assemblers walk the parts DIRECTORY — `_part_paths` — and until now
    neither looked at the year's marker or at its ledger. Three ways that
    turned a failure into "no data there":

      * a fetch killed mid-year (the six-hour cap, an OOM, a lost box) leaves
        flushed parts behind and no marker. A later `--stage fetch` sees the
        year unmarked and re-fetches it, which is right — but a later
        `--stage publish`, or an assemble after a `--force` on one other year,
        read those orphan parts as if they were a year.
      * a year marked done whose counts.json says `parts: 7` while six files
        are on disk: one npz lost to a full disk or a half-written flush. Both
        assemblers would have assembled six and said nothing.
      * the same with rows — checked in `_finish_store`, where the scan has
        counted them.

    Under `--allow-missing-years` an unmarked year is ADMITTED rather than
    refused, and named in store.json's `degraded` block, because that flag is
    the caller saying "a short store is what I want" — but it is never the
    default, and it never silently rewrites the ledger.

    AND A FOURTH, ADDED WITH FAMILY 10.1: a part written by the v1 builder
    carries `time_days` and cannot be upgraded, so it is refused by name here
    rather than assembled into a store that would claim a precision it does not
    have (`f10.check_part_schema`). `--allow-missing-years` does NOT admit it:
    the flag says "a short store is what I want", never "a wrong one".
    """
    allow = bool(getattr(ctx.a, "allow_missing_years", False))
    bad, degraded = [], []
    for y in ctx.years:
        npz = year_part_names(ctx, y)
        for n in npz:
            # Reads the npz's zip directory only — a seek, on a part that may
            # be a gigabyte.
            f10.check_part_schema(os.path.join(ctx.year_dir(y), n))
        # `read_json` answers a missing file and an unparseable one with {},
        # so presence is asked of the filesystem: a counts.json that exists and
        # does not parse must NOT read as "no ledger here" and be skipped.
        cp = os.path.join(ctx.year_dir(y), "counts.json")
        c = read_json(cp, {}) if os.path.exists(cp) else None
        m = marked(ctx.root, f"parts/{y}")
        if not npz and c is None and not m:
            continue                       # never fetched; the stage said so
        if not m:
            msg = (f"{y}: {len(npz)} part file(s) in {ctx.year_dir(y)} and no "
                   f"{marker(ctx.root, f'parts/{y}')} — the fetch of that year "
                   f"did not finish, so these parts are a PREFIX of the year")
            (degraded if allow else bad).append(msg)
            continue
        if c is None:
            bad.append(f"{y}: marked done with no counts.json — the marker was "
                       f"written without the ledger it is supposed to describe")
            continue
        want = int(c.get("parts", -1))
        if want != len(npz):
            bad.append(f"{y}: counts.json says {want} part(s), "
                       f"{ctx.year_dir(y)} holds {len(npz)} "
                       f"({', '.join(npz[:4])}{' …' if len(npz) > 4 else ''})")
    if bad:
        sys.exit(
            f"REFUSING to assemble {ctx.adapter.store}: the parts on disk do "
            f"not match what claims them.\n  " + "\n  ".join(bad) +
            f"\nAssembling anyway would write a store whose `per_year` block "
            f"is simply the rows that happened to be there — the one number "
            f"that would have shown the loss, computed FROM the loss. Re-run "
            f"the fetch stage with the same --work value (an unmarked year is "
            f"re-fetched whole and its stale parts are cleared first), or pass "
            f"--allow-missing-years to assemble a deliberately short store.")
    if degraded:
        for m in degraded:
            print(f"  ::warning::{m} — --allow-missing-years admits it")
        ctx.degraded_years = degraded
    return degraded


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


def stage_fetch(ctx, assemble_store_after=True):
    """Fetch per year (or one stream) into parts, then assemble the store.

    `assemble_store_after=False` (E-082, family 1's separate `assemble`
    stage) stops after the parts and the absence check and marks nothing —
    the caller marks its own `fetch`. Family 10 always assembles here.
    """
    ad = ctx.adapter
    ad.fetch_preflight(ctx)
    if getattr(ctx.a, "parts_from_hub", False):
        # THE KEYLESS HALF of slatrack's build. The credentialed fetch happens
        # on GitHub-HOSTED lanes (ml/CLAUDE.md §6 forbids CMEMS credentials on
        # a rented box) and parks each finished year under
        # `partials/family10_1/<store>/<year>/` on the Hub; this branch brings
        # those parts back and hands them straight to the assembler. Nothing
        # here touches the source archive, so HF_TOKEN is the only secret the
        # machine running it ever sees.
        import family10_parts_hub as ph
        ctx.prog.stage_start(f"pull {ad.store} parts", len(ctx.years))
        ph.pull(ad.store, ctx.years, ctx.work,
                allow_missing=bool(getattr(ctx.a, "allow_missing_years",
                                           False)),
                **parts_hub_kwargs(ctx))
    elif ad.per_year:
        ctx.prog.stage_start(f"fetch {ad.store}", len(ctx.years))
        for i, y in enumerate(ctx.years, 1):
            if marked(ctx.root, f"parts/{y}") and not ctx.a.force:
                print(f"  {y}: already fetched — skipping")
                continue
            # A RE-FETCH STARTS FROM AN EMPTY YEAR. `PartWriter` numbers its
            # flushes from 00000, so a --force re-run that produces FEWER parts
            # than the last one left the tail of the old run on disk — and both
            # assemblers read the directory, not the ledger, so those stale
            # parts would be assembled into the new store. The one-stream
            # branch below has always done this; the per-year branch now does
            # too.
            shutil.rmtree(ctx.year_dir(y), ignore_errors=True)
            mp = marker(ctx.root, f"parts/{y}")
            if os.path.exists(mp):
                os.remove(mp)
            t0 = time.time()
            before = len(ctx.absent)
            pw = PartWriter(ctx, y)
            for label, rows, counts in ad.fetch_year(ctx, y):
                pw.add(rows, counts)
                ctx.prog.item(f"{ad.store} {label}", i,
                              {"year_rows": pw.n})
            lost = ctx.absent[before:]
            if lost:
                # DO NOT MARK A YEAR WHOSE INPUTS WERE NOT ALL READ. The rows
                # that DID arrive stay on disk — they cost hours and the retry
                # overwrites them — but without the marker and without the
                # counts.json ledger they are not a year: `parts_preflight`
                # refuses to assemble them, `family10_parts_hub.push` refuses
                # to publish them, and the next run re-fetches the year whole.
                # This is the whole mechanism (ml/CLAUDE.md §5.21, family 7's
                # commit fd3b446): a marker may only UNDER-claim.
                pw.flush()
                print(f"  ::warning::{y}: NOT MARKED — {len(lost)} input(s) "
                      f"could not be read ({'; '.join(e['unit'] for e in lost[:4])}"
                      f"{' …' if len(lost) > 4 else ''}); the {pw.n:,} row(s) "
                      f"that did arrive stay on disk, and the retry clears "
                      f"them and re-fetches the year whole")
                continue
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
            # THE RESUMABLE UNIT IS THE WHOLE PASS, so an absence anywhere in it
            # withholds EVERY year's marker, not one year's. There is no
            # cheaper granularity to fall back to: the file is sorted by
            # expocode, so a year is spread across the whole 1.4 GB.
            if ctx.absent:
                for year in sorted(writers):
                    writers[year].flush()
                print(f"  ::warning::{ad.store}: NO year marked — "
                      f"{len(ctx.absent)} input(s) could not be read in the "
                      f"one pass that is this store's resumable unit; "
                      f"{total:,} row(s) kept on disk for the retry")
            else:
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
    fetch_absence_check(ctx)
    if not assemble_store_after:
        return None
    ctx.prog.stage_start(f"{ad.store} store", 1)
    meta = assemble(ctx)
    mark(ctx.root, "fetch")
    ctx.prog.item("store", 1, {"N": meta["N"], "bin_first": meta["bin_first"]})
    return meta


def parts_hub_kwargs(ctx):
    """How `family10_parts_hub` reaches THIS store's parts (E-082).

    Empty for family 10's default layout, so its calls — and the tests that
    replace the module's `_hub` seam — are exactly what they were.
    """
    lay = ctx.layout
    if lay.hf_partials == f10.HF_PARTIALS and lay.repo_id is None:
        return {}
    return {"partials": lay.hf_partials, "hub": lay.hub,
            "private": lay.private}


def fetch_absence_check(ctx):
    """REFUSE the stage when an input could not be read (family 7, fd3b446).

    Every adapter above collects rather than decides, so this is the one place
    that answers, and it answers the way §5.21 requires: the years that could
    not be read are NOT marked, the stage is NOT marked, and a resume retries
    exactly those years. A store that is deliberately short has to be asked for
    BY NAME — and then the degrade is written into `store.json`, so the store
    says what it did rather than what it meant to do.
    """
    if not ctx.absent:
        return
    ad = ctx.adapter
    units = [e["unit"] for e in ctx.absent]
    years = sorted(ctx.absent_years())
    if not getattr(ctx.a, "allow_missing_years", False):
        sys.exit(
            f"stage fetch ({ad.store}): {len(ctx.absent)} input(s) could not "
            f"be read, and every one of them is a hole nothing downstream can "
            f"see — a store of point observations has no empty cell to look "
            f"at, so a month, a dataset-year or a mission-year that did not "
            f"arrive is indistinguishable from an ocean nobody sampled. "
            f"REFUSING to mark the stage.\n  "
            + "\n  ".join(f"{e['unit']}: {e['why']}" for e in ctx.absent[:8])
            + (f"\n  … and {len(ctx.absent) - 8} more" if len(ctx.absent) > 8
               else "")
            + f"\nThe year(s) {years} carry no `parts/<year>.done` marker and "
              f"no counts.json, so nothing will assemble or publish them and "
              f"a re-run with the SAME --work value retries exactly those — "
              f"the years that DID land are marked and skipped. "
              f"{'`family10_parts_hub.py push` will refuse them too. ' if ad.store == 'slatrack' else ''}"
              f"Two ways on: (1) fix the source (a moved ERDDAP dataset id, a "
              f"short mirror, a filter that does not match the remote layout) "
              f"and re-run; or (2) pass --allow-missing-years if a store with "
              f"those inputs missing is genuinely what you want.")
    print(f"  ::warning::{ad.store}: {len(ctx.absent)} input(s) could not be "
          f"read ({', '.join(units[:6])}{' …' if len(units) > 6 else ''}) — "
          f"--allow-missing-years says that is deliberate; the degrade goes "
          f"into store.json")


# ---------------------------------------------------- assembly, two ways ----
# `assemble_store` holds every row in RAM at once; that is fine for gdp, gtmba
# and socat (10^6..10^8 rows) and impossible for slatrack, measured 2026-09-14
# at ~70 M samples in 2015 alone — call it 1.5-2.5 e9 rows over 1993-2024, i.e.
# 50-80 GB in the store and rather more than that while sorting. So there is a
# second assembler, and its ONLY claim is that it writes THE SAME BYTES.
#
# WHY THE TWO ORDERS ARE THE SAME PERMUTATION, exactly:
#   `np.lexsort((time_s, bin))` sorts by bin, then by time_s, and it is
#   STABLE — rows equal in both keys keep their INPUT order, which is the order
#   `read_parts` yields (years ascending, parts in sorted name order).
#   The streaming assembler reproduces that in three passes: it counts rows per
#   bin (pass 1) to get the CSR offsets, scatters every part's rows into its
#   bin's slice IN THAT SAME INPUT ORDER (pass 2), and then sorts each bin's
#   slice by time_s with a STABLE argsort (pass 3). Sorting by bin is what
#   the scatter does; the stable per-bin sort by time breaks ties in the order
#   the scatter laid rows down, i.e. input order. Same permutation, therefore
#   the same bytes — and `tests/test_build_family10_stores.py` proves it by
#   building one synthetic archive and hashing both assemblers' output, with a
#   year that carries duplicate (bin, time_s) rows ACROSS two parts so the
#   tie-break is actually exercised. Ties are RARER under schema 2 than they
#   were under v1 — a second is finer than 84 s — but they are not gone (two
#   drifters report on the same six-hourly slot), so the tie-break still has to
#   be the same one in both assemblers.
STAT_CHUNK = 1 << 22          # rows per statistics block; both assemblers use it
CHECK_CHUNK_ROWS = 16_000_000  # rows per block in `check_store`
STREAM_ROWS = 50_000_000      # `--assemble auto` switches above this
# bytes per stored row, excluding the tiny bin_offsets vector: bin 2 +
# time_s 4 + lat 4 + lon 4 + platform 8 + qc 1 + fp 4 + values 2*C. Schema 2
# costs exactly what schema 1 did — int32 seconds are the same four bytes
# float32 days were, so 10.1 buys its precision for nothing.
ROW_BYTES_FIXED = 2 + 4 + 4 + 4 + 8 + 1 + 4
DISK_HEADROOM = 1.2


def row_bytes(C, time_dtype="int32"):
    """Stored bytes per row: 27 + 2C (schema 2), 31 + 2C (schema 3)."""
    extra = np.dtype(TIME_DTYPES[time_dtype]).itemsize - 4
    return ROW_BYTES_FIXED + extra + 2 * int(C)


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
    instead of the ~27 + 2C a whole row costs. Returns (N, per_year,
    {bin: rows}).

    The `bin` member is the one column this pass needs and it is written by
    `_pack` from `time_s` alone, so pass 1 never has to look at a timestamp.
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


def _disk_preflight(dest, N, C, n_bins, time_dtype="int32"):
    """Refuse BEFORE pass 2 rather than at 90% of a six-hour write (§5.18).

    The size is computable from the dtypes, so it is computed; a check that
    can only guess belongs nowhere near an 80 GB allocation.
    """
    rb = row_bytes(C, time_dtype)
    need = int(N) * rb + 8 * (int(n_bins) + 1)
    free = shutil.disk_usage(dest).free
    want = need * DISK_HEADROOM
    print(f"  disk: the store is {need / 1e9:.2f} GB ({N:,} rows x "
          f"{rb} B); {free / 1e9:.2f} GB free under "
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
      3. sort each bin's slice by `time_s` with a STABLE argsort.
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

    _disk_preflight(dest, N, C, n_bins, ctx.time_dtype)

    from numpy.lib.format import open_memmap
    shapes = {"bin": (np.int16, (N,)),
              "time_s": (TIME_DTYPES[ctx.time_dtype], (N,)),
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
        order = np.argsort(np.asarray(mm["time_s"][s:e], np.int64),
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
    parts_preflight(ctx)
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
        cat = empty_rows(ad.C, ctx.time_dtype)
    cat["time_s"] = cat["time_s"].astype(TIME_DTYPES[ctx.time_dtype],
                                         copy=False)
    N = int(len(cat["bin"]))
    # THE DEFINING ORDER: (bin, time_s) ascending. `lexsort` takes its keys
    # last-major, so time is the secondary key. Both keys are INTEGERS now, so
    # the comparison the order rests on is exact.
    order = np.lexsort((cat["time_s"].astype(np.int64),
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
    to_write = {"bin": cat["bin"], "time_s": cat["time_s"],
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



def _check_year_ledgers(ctx, per_year):
    """The rows the parts HOLD against the rows the year's ledger CLAIMS.

    `counts.json` is written by `PartWriter.close()` from the rows it actually
    flushed, so the two can only disagree if a part file was lost, truncated or
    replaced between the fetch and the assembly — and the store.json that came
    out would have recorded the smaller number in `per_year` as though it were
    the measurement. The one number that shows the loss must not be computed
    from the loss, which is why this compares against a record written earlier.

    Both assemblers reach this through `_finish_store`, so neither can skip it.
    """
    bad = []
    for y in ctx.years:
        cp = os.path.join(ctx.year_dir(y), "counts.json")
        if not os.path.exists(cp):
            continue                       # `parts_preflight` has ruled on it
        want = int(read_json(cp, {}).get("rows", -1))
        got = int(per_year.get(y, 0))
        if want != got:
            bad.append(f"{y}: counts.json says {want:,} row(s), the parts hold "
                       f"{got:,}")
    if bad:
        sys.exit(
            f"REFUSING to write {ctx.adapter.store}/store.json: a year's parts "
            f"no longer hold what its ledger recorded.\n  " + "\n  ".join(bad) +
            f"\nThe store would have published the smaller number as its own "
            f"`per_year` measurement. Re-fetch those years with the same "
            f"--work value.")


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
    _check_year_ledgers(ctx, per_year)
    # WHAT THIS STORE PUBLISHES BESIDE ITS COLUMNS, written BEFORE the sha256
    # block is computed so it is hashed, uploaded and restore-checked exactly
    # like a column rather than being a file that happens to sit next to one.
    files = dict(files)
    files.update(ad.extra_files(ctx, dest) or {})
    platforms_meta = None
    if getattr(ad, "platform_meta", False):
        files["platforms.json"], platforms_meta = write_platforms(ctx, dest)
    per_channel, measured_fraction = _channel_stats(values, ad.channels, N)
    live = int((np.diff(off) > 0).sum())
    plan = read_json(os.path.join(ctx.root, "plan.json"), {})
    lay = ctx.layout
    meta = {
        "family": lay.family, "tier": "P", "store": ad.store,
        "title": ad.title,
        "family_version": lay.family_version,
        "schema_version": SCHEMA_VERSION,
        "schema_version_note": (
            "schema 2 (family 10.1): the time column is `time_s`, int32 "
            "SECONDS since 1982-01-01T00:00:00Z. Schema 1 (family 10, family "
            "8) carried `time_days`, float32 days, which resolves 21 s in 1993 "
            "and 84 s in 2024 — against slatrack's 1 Hz sampling, up to 316 "
            "consecutive samples shared a timestamp. `ml/family10_store.py` "
            "reads BOTH; only the time column changed, and the bins, "
            "footprints, channels and QC policy are identical."),
        "schema": {
            "bin.npy": {"dtype": "int16", "shape": [N],
                        "meaning": "floor(time_s / 432000 s) — exact integer "
                                   "arithmetic; NEGATIVE before 1982 and kept"},
            "time_s.npy": {"dtype": "int32", "shape": [N],
                           "meaning": "seconds since 1982-01-01T00:00:00Z, "
                                      "negative before it; int32 spans "
                                      f"{f10.TIME_S_MIN_DATE} .. "
                                      f"{f10.TIME_S_MAX_DATE} and the builder "
                                      f"refuses a row past it"},
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
        "builder": lay.builder,
        "builder_git_sha": git_sha(),
        "built_at": utcnow(),
    }
    if ctx.time_dtype != "int32":
        # SCHEMA 3 (E-082). Family 10 never takes this branch, so its
        # store.json keeps exactly the keys it always had.
        meta["schema_version"] = ctx.schema_version
        meta["time_dtype"] = ctx.time_dtype
        meta["schema_version_note"] = (
            "schema 3 (E-082): `time_s` is int64 SECONDS since "
            "1982-01-01T00:00:00Z, because this record begins before 1914 "
            "where int32 seconds end. Everything else is schema 2: bin = "
            "floor_divide(time_s, 432000) as int16 (negative before 1982, "
            "and the int16 range 1533 .. 2430 is now the limit). "
            "`ml/family10_store.py` reads schema 1, 2 and 3.")
        meta["schema"]["time_s.npy"] = {
            "dtype": "int64", "shape": [N],
            "meaning": "seconds since 1982-01-01T00:00:00Z, negative before "
                       "it (schema 3)"}
    for k in ("family", "distribution", "licence"):
        if hasattr(ad, k):
            meta[k if k != "family" else "family_code"] = getattr(ad, k)
    if platforms_meta is not None:
        meta["platforms"] = platforms_meta
    if ad.notes:
        meta["notes"] = ad.notes
    meta.update(ad.extra_meta(ctx, dest, N, values) or {})
    # THE STORE SAYS WHAT IT DID, NOT WHAT IT MEANT TO DO. A build that was
    # allowed past an unreadable input carries the list of them here, in the
    # file every consumer already reads — `store.json`'s `sources` and
    # `per_year` are where a hole has to be visible, because the arrays
    # themselves cannot show one (family 7, fd3b446).
    if ctx.absent or ctx.degraded_years:
        meta["degraded"] = {
            "allow_missing_years": bool(getattr(ctx.a, "allow_missing_years",
                                                False)),
            "inputs_not_read": ctx.absent,
            "years_admitted_unmarked": ctx.degraded_years,
            "note": ("this store was built past inputs that could not be read "
                     "— the rows below are what arrived, not what the archive "
                     "holds. Every unit named here is a gap that nothing "
                     "downstream can see in the arrays."),
        }
    meta["sha256"] = {n: sha256(p) for n, p in sorted(files.items())}
    atomic_json(os.path.join(dest, "store.json"), meta)
    check_store(dest, ad, chunk_rows=ctx.check_chunk)
    print(f"  store: {N:,} row(s), C={ad.C}, bins {bin_first}..{bin_last} "
          f"({live:,} live) -> {dest}")
    return meta

def write_platforms(ctx, dest):
    """`platforms.json` beside the arrays, from the adapter's own table.

    Written BEFORE the sha256 block so it is hashed, published and
    restore-checked like a column (E-082). Keys are the platform hashes as
    decimal strings, sorted, so the file is deterministic. Returns (path,
    summary for store.json) — the summary counts entries and how many of the
    store's own platforms have none, which is measured over the finished
    `platform.npy` in blocks.
    """
    ad = ctx.adapter
    table = ad.platforms(ctx) or {}
    if not table:
        raise ValueError(f"{ad.store}: platform_meta is True and platforms() "
                         f"returned nothing — refusing to publish an empty "
                         f"platforms.json")
    # ONLY THE PLATFORMS THIS STORE HOLDS: the station list covers the whole
    # archive (132k GHCN-Daily stations) and a short window uses a fraction.
    seen = set()
    pp = os.path.join(dest, "platform.npy")
    col = np.load(pp, mmap_mode="r")
    for lo in range(0, int(col.shape[0]), CHECK_CHUNK_ROWS):
        seen.update(np.unique(np.asarray(col[lo:lo + CHECK_CHUNK_ROWS]))
                    .tolist())
    del col
    out = {str(int(k)): table[k] for k in sorted(table, key=int)
           if int(k) in seen}
    p = os.path.join(dest, "platforms.json")
    tmp = p + f".tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(out, fh, sort_keys=True, separators=(",", ":"))
    os.replace(tmp, p)
    have = {int(k) for k in out}
    missing = sorted(seen - have)
    if missing:
        print(f"  ::warning::{ad.store}: {len(missing)} platform(s) in the "
              f"store have no platforms.json entry")
    return p, {"file": "platforms.json", "entries": len(out),
               "source_entries": len(table), "in_store": len(seen),
               "in_store_without_entry": len(missing)}


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

    SCHEMA 2's three time checks, all on `time_s` as INTEGERS:
      * `bin == floor_divide(time_s, 432000)` on every row — the index and the
        timestamps must be the same statement. Exact now, where the v1 form
        compared against a float floor.
      * `time_s` NON-DECREASING inside every bin, seam included. Under v1 this
        could only ever be checked to 84 s; a v1 slatrack store passes it
        because equal timestamps are non-decreasing, which is precisely the
        weakness 10.1 removes.
      * every `time_s` INSIDE THE SOURCE WINDOW `store.json` claims
        (`date_range`), so a store whose rows escape the window it says it
        covers is refused rather than published with a `per_year` block that
        quietly disagrees with its own header. Skipped when a store carries no
        `date_range` — the hand-written fixtures in the tests do not.
    """
    # EVERY CHECK BELOW IS AN `assert`, AND `python3 -O` DELETES THOSE. Run
    # under -O (or PYTHONOPTIMIZE set in the environment, which a runner image
    # or a helpful wrapper can do without anybody typing it) this whole
    # function becomes: open the store, return it. `_finish_store` calls it
    # before writing store.json and `stage_publish` calls it before uploading,
    # so an optimised interpreter would publish an UNCHECKED store and print
    # the same lines as a checked one. That is the quietest possible way to
    # silence an assertion, so it is refused here rather than rewritten into
    # sixteen `if ...: raise`s that would drift from the ones above them.
    if not __debug__:
        raise RuntimeError(
            "check_store cannot run under `python3 -O` / PYTHONOPTIMIZE: every "
            "E-079 §4 assertion in it is an `assert` statement and -O removes "
            "them all, so the check would pass by not existing. Re-run without "
            "-O (unset PYTHONOPTIMIZE); the check is seconds on the small "
            "stores and reads the big ones in blocks.")
    st = f10.Store(path)
    N = int(st.N)
    chunk = max(1, int(chunk_rows))
    off = np.asarray(st.bin_offsets, np.int64)
    assert off[0] == 0 and off[-1] == st.N, "CSR offsets do not span the rows"
    assert np.all(np.diff(off) >= 0), "bin_offsets is not monotone"

    b_col = st["bin"]
    lat_col, lon_col = st["lat"], st["lon"]
    # The declared window, in seconds, when the store states one.
    win = None
    dr = st.meta.get("date_range") or []
    if len(dr) == 2 and all(dr):
        win = (seconds_since_epoch(parse_date(dr[0])),
               seconds_since_epoch(parse_date(dr[1])) + SECONDS_PER_DAY - 1)
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
        # INTEGER SECONDS under schema 2; a schema-1 store (family 8, the
        # published family-10 four) is converted on the fly by the reader so
        # this one implementation checks both.
        t = st.time_s(lo, hi)
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
        assert np.array_equal(b, f10.bin_of_seconds(t)), \
            "bin.npy disagrees with floor(time_s / 432000)"
        if win is not None:
            assert int(t.min()) >= win[0] and int(t.max()) <= win[1], (
                f"time_s runs {int(t.min())}..{int(t.max())} s, outside the "
                f"window store.json declares ({dr[0]} .. {dr[1]} = "
                f"{win[0]}..{win[1]} s)")
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


# ============================================================= stage: grid ===
# E-081 §3. THE GLOBE CANNOT RANGE-READ A POINT STORE, so beside the store the
# builder writes one gridded product: the store's rows summed onto the
# family-7 0.25° grid, one frame per month, month-major in C order so that one
# month of both channels is ONE contiguous range read (721 x 1440 x 2 cells,
# 8.3 MB at float32 — see F16_MAX below for why it is not the 4.15 MB the plan
# budgeted at float16). Zero is a real value — no vessel broadcast in that
# cell that month — and is stored as 0, never NaN.
#
# THE GEOMETRY, stated here because `data/fishing_index.json` publishes it and
# `src/app.js` must contain none of this arithmetic (root CLAUDE.md §3):
#   latitude index 0 is -90, longitude index 0 is -180, the step is 0.25°, and
#   a row's CELL CENTRE is binned with
#       iy = clip(floor((lat + 90) / 0.25), 0, 720)
#       ix = clip(floor((lon + 180) / 0.25), 0, 1439)
#   The clips are the poles and the dateline: 721 latitude points cover
#   [-90, 90] inclusive, so +90 would index 720 either way, and longitude runs
#   [-180, 180) so 1439 is only reached by a value the store cannot hold.
FISHING_GRID_DIR = "fishing_grid"
FISHING_GRID_FILE = "fishing_grid_monthly_025.npy"
GRID_NY, GRID_NX, GRID_STEP = 721, 1440, 0.25
GRID_LAT0, GRID_LON0 = -90.0, -180.0
GRID_MONTH_SPAN = ("2012-01", "2024-12")          # E-081 §3's 156 months
GRID_ROWS_CHUNK = 8_000_000
# float16's largest finite value — and the reason THE GRID IS float32.
#
# E-081 §3 specifies float16. MEASURED 2026-09-16 by building the real 2012
# and 2024 years: the largest 0.25° cell-month sum is **29,460 vessel-hours in
# 2012 and 595,726 in 2024**, against float16's largest finite value of
# 65,504. So a float16 grid of the real archive would store +inf in the cells
# where the fleet actually is — the East China Sea, the southern North Sea —
# and an infinity in a raster looks exactly like a colour. float32 doubles the
# month slab from 4.15 MB to 8.3 MB, which is still smaller than family 7's
# 14.5 MB frame, and `data/fishing_index.json` publishes `dtype`, `itemsize`
# and `slab_bytes`, so a consumer that reads the index needs no change (and
# gets a NATIVE Float32Array instead of a hand-rolled float16 decode).
#
# float16 stays available (`--grid-dtype float16`) because it is right for a
# narrow window, and the check below fires per month BEFORE that month is
# written rather than leaving an infinity to be found on the globe.
F16_MAX = 65504.0
GRID_DTYPE_DEFAULT = "float32"


def month_list(lo, hi, span=GRID_MONTH_SPAN):
    """Every `YYYY-MM` from `lo` to `hi` inclusive, clipped to the plan's span.

    The FULL build (2012-01-01 .. 2024-12-31) gives E-081 §3's 156 months
    exactly; a narrower window gives its own months and the index says which,
    so a partial grid describes itself instead of pretending to be the whole
    record with zeros in the gaps.
    """
    a = max(f"{lo.year:04d}-{lo.month:02d}", span[0])
    b = min(f"{hi.year:04d}-{hi.month:02d}", span[1])
    out = []
    y, m = int(a[:4]), int(a[5:7])
    while f"{y:04d}-{m:02d}" <= b:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + (m == 12), m % 12 + 1)
    return out


def _month_bounds_s(stamp):
    """[start, end) of month `YYYY-MM`, in seconds since the epoch."""
    y, m = int(stamp[:4]), int(stamp[5:7])
    lo = seconds_since_epoch(dt.date(y, m, 1))
    hi = seconds_since_epoch(dt.date(y + (m == 12), m % 12 + 1, 1))
    return lo, hi


def grid_cell_index(lat, lon):
    """(iy, ix) for cell centres — THE convention, in one place."""
    iy = np.clip(np.floor((np.asarray(lat, np.float64) - GRID_LAT0) / GRID_STEP)
                 .astype(np.int64), 0, GRID_NY - 1)
    ix = np.clip(np.floor((np.asarray(lon, np.float64) - GRID_LON0) / GRID_STEP)
                 .astype(np.int64), 0, GRID_NX - 1)
    return iy, ix


def stage_grid(ctx):
    """The monthly 0.25° grid, and the assertion that it sums to the store.

    Runs in the SAME job as the fetch and the publish (E-081 deliverable 3):
    it reads the finished store's memmaps month by month — the rows of a month
    are one contiguous range, because the store is sorted by (bin, time_s) and
    both are monotone in time — accumulates each month in FLOAT64 with
    `np.bincount`, and writes that month's slab as float16. Nothing larger
    than one month (16.6 MB of float64) is ever in memory.

    THE ASSERTION (E-081 §5's falsifier 4) is per month AND in total: the
    float64 sum of the store's own values over the month's rows against the
    float64 sum of the float16 slab that was written. They differ only by the
    float16 rounding of each cell, which is 2^-11 relative per cell and
    therefore the same bound on their total; the test is 1e-3 relative, two
    orders of margin, and it fires on an off-by-one month or a dropped block
    rather than on rounding.
    """
    ad = ctx.adapter
    if ad.store != "fishing":
        print(f"  grid: {ad.store} has no gridded product — nothing to do "
              f"(E-081 §3 defines one for `fishing` only)")
        mark(ctx.root, "grid")
        return None
    dest = ctx.store
    meta = read_json(os.path.join(dest, "store.json"), {})
    if not meta:
        sys.exit(f"cannot build the grid: {dest}/store.json is missing — the "
                 f"fetch stage assembles the store the grid sums.")
    st = f10.Store(dest)
    months = month_list(ctx.d_lo, ctx.d_hi)
    dtype = np.dtype(getattr(ctx.a, "grid_dtype", GRID_DTYPE_DEFAULT)
                     or GRID_DTYPE_DEFAULT)
    out_dir = os.path.join(ctx.root, FISHING_GRID_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, FISHING_GRID_FILE)
    shape = (len(months), GRID_NY, GRID_NX, int(st.C))
    need = int(np.prod(shape)) * dtype.itemsize
    free = shutil.disk_usage(out_dir).free
    print(f"  grid: {len(months)} month(s) {months[0]}..{months[-1]}, shape "
          f"{list(shape)} {dtype.name}, {need / 1e9:.2f} GB; "
          f"{free / 1e9:.2f} GB free under {out_dir}")
    if free < need * DISK_HEADROOM:
        sys.exit(f"REFUSING to write the grid: it is {need / 1e9:.2f} GB and "
                 f"{out_dir} has {free / 1e9:.2f} GB free.")
    ctx.prog.stage_start(f"grid {ad.store}", len(months))

    from numpy.lib.format import open_memmap
    arr = open_memmap(out_path + ".part", mode="w+", dtype=dtype, shape=shape)
    t_col = st["time_s"]
    lat_col, lon_col = st["lat"], st["lon"]
    v_col = st["values"]
    per_month, rows_seen = [], 0
    tot_store = np.zeros(st.C, np.float64)
    tot_grid = np.zeros(st.C, np.float64)
    worst = 0.0
    cells = GRID_NY * GRID_NX
    for i, stamp in enumerate(months):
        s0, s1 = _month_bounds_s(stamp)
        lo = int(np.searchsorted(t_col, s0, side="left"))
        hi = int(np.searchsorted(t_col, s1, side="left"))
        acc = np.zeros((cells, st.C), np.float64)
        want = np.zeros(st.C, np.float64)
        for a in range(lo, hi, GRID_ROWS_CHUNK):
            b = min(a + GRID_ROWS_CHUNK, hi)
            iy, ix = grid_cell_index(lat_col[a:b], lon_col[a:b])
            flat = iy * GRID_NX + ix
            v = np.asarray(v_col[a:b], np.float64)
            for c in range(st.C):
                col = np.nan_to_num(v[:, c], nan=0.0)
                acc[:, c] += np.bincount(flat, weights=col, minlength=cells)
                want[c] += float(col.sum())
        big = float(acc.max()) if acc.size else 0.0
        if dtype == np.float16 and big > F16_MAX:
            arr.flush()
            del arr
            sys.exit(
                f"REFUSING to write {stamp}: its largest cell-month sum is "
                f"{big:,.0f} h and float16's largest finite value is "
                f"{F16_MAX:,.0f} — the cell would be stored as +inf. Re-run "
                f"`--stage grid --grid-dtype float32` (the index publishes "
                f"the dtype and the slab size, so a consumer that reads "
                f"data/fishing_index.json needs no change), or narrow the "
                f"window. Measured 2026-09-16: 2012's largest cell-month is "
                f"29,460 h, so the headroom is real but not large.")
        arr[i] = acc.reshape(GRID_NY, GRID_NX, st.C).astype(dtype)
        got = np.asarray(arr[i], np.float64).reshape(-1, st.C).sum(axis=0)
        rel = max(abs(g - w) / max(abs(w), 1.0) for g, w in zip(got, want))
        worst = max(worst, rel)
        assert rel < 1e-3, (
            f"{stamp}: the grid sums to {got.tolist()} and the store's rows "
            f"to {want.tolist()} — relative difference {rel:.3e}, far past "
            f"float16 rounding. The grid does not describe the store.")
        tot_store += want
        tot_grid += got
        rows_seen += hi - lo
        per_month.append({"month": stamp, "rows": hi - lo,
                          "store_sum": want.tolist(), "grid_sum": got.tolist(),
                          "max_cell": big, "rel": rel})
        ctx.prog.item(stamp, i + 1, {"rows": hi - lo, "rel": rel})
    arr.flush()
    del arr
    os.replace(out_path + ".part", out_path)

    if rows_seen != st.N:
        sys.exit(f"the grid saw {rows_seen:,} of the store's {st.N:,} rows — "
                 f"{st.N - rows_seen:,} row(s) fall outside the months "
                 f"{months[0]}..{months[-1]} the grid covers, so the picture "
                 f"would be missing measurements the store has. Widen the "
                 f"window or rebuild the store to match it.")
    man = {
        "store": ad.store, "file": FISHING_GRID_FILE,
        "prefix": f"{HF_ROOT}/{FISHING_GRID_DIR}",
        "shape": list(shape), "dtype": dtype.name,
        "months": months, "n_months": len(months),
        "month_span": list(GRID_MONTH_SPAN),
        "complete": months == month_list(parse_date(f"{GRID_MONTH_SPAN[0]}-01"),
                                         parse_date(f"{GRID_MONTH_SPAN[1]}-28")),
        "channels": [c["name"] for c in (meta.get("channels") or [])],
        "grid": {"ny": GRID_NY, "nx": GRID_NX, "step": GRID_STEP,
                 "lat0": GRID_LAT0, "lon0": GRID_LON0,
                 "south_first": True, "wrap": True,
                 "index_rule": ("iy = clip(floor((lat + 90) / 0.25), 0, 720), "
                                "ix = clip(floor((lon + 180) / 0.25), 0, "
                                "1439), on the row's CELL CENTRE")},
        "rows": rows_seen, "store_N": int(st.N),
        "store_sum": tot_store.tolist(), "grid_sum": tot_grid.tolist(),
        "worst_month_rel": worst,
        "per_month": per_month,
        "bytes": os.path.getsize(out_path), "sha256": sha256(out_path),
        "builder": "ml/build_family10_stores.py --stage grid",
        "builder_git_sha": git_sha(), "built_at": utcnow(),
    }
    atomic_json(os.path.join(out_dir, "grid.json"), man)
    mark(ctx.root, "grid")
    print(f"  grid: {len(months)} month(s), {rows_seen:,} row(s), "
          f"{os.path.getsize(out_path) / 1e9:.2f} GB, worst month agreement "
          f"{worst:.2e} -> {out_path}")
    return man


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
    ad = ctx.adapter
    dest = ctx.store
    lay = ctx.layout
    prefix = lay.prefix(ad.store)
    # THE FILE LIST COMES FROM store.json, NOT FROM THE DIRECTORY. A listing
    # publishes whatever is there, and `family10_store.Store` opens `qc.npy`
    # and `fp.npy` through `_optional` — so a store that reached the Hub
    # without one of them opens cleanly, answers every search, and silently
    # reports no quality flag and the default footprint for every row. The
    # sha256 block names exactly the files the assembler wrote; anything else
    # on disk is not this store, and anything missing from disk is a publish
    # that must not happen.
    sm = read_json(os.path.join(dest, "store.json"), {})
    want = sorted(sm.get("sha256") or {})
    if not want:
        sys.exit(f"cannot publish: {os.path.join(dest, 'store.json')} carries "
                 f"no sha256 block, so there is no record of which files this "
                 f"store is made of. Re-run the fetch stage.")
    names = want + ["store.json"]
    for n in names:
        if not os.path.exists(os.path.join(dest, n)):
            sys.exit(f"cannot publish: {os.path.join(dest, n)} is missing — "
                     f"store.json names it in its sha256 block. Publishing the "
                     f"rest would put a store on the Hub that opens, searches "
                     f"and answers with that column silently absent.")
    have = sorted(n for n in os.listdir(dest) if n.endswith(".npy"))
    extra = sorted(set(have) - set(want))
    if extra:
        sys.exit(f"cannot publish: {extra} sit in {dest} and store.json does "
                 f"not name them. They are not part of this store — a leftover "
                 f"from an earlier build with a different schema — and a "
                 f"publish that uploaded them would leave the Hub holding "
                 f"columns of two different stores under one prefix. Remove "
                 f"them or rebuild into a clean directory.")
    _restore_disk_preflight(ctx, dest, names)
    check_store(dest, ad, chunk_rows=ctx.check_chunk)
    # The Hub is reached only after every question that can be answered from
    # the store itself has been (ml/CLAUDE.md §0.3).
    api, repo, tok = ctx.hub()
    # THE TWO-TRACK RULE, ASSERTED ON THE REPOSITORY THIS CALL WILL WRITE
    # (E-082 §1.3) — not on the adapter's intention. It runs after the id is
    # resolved and before the first request that touches the repository.
    check_publish_target(ad, lay, repo)
    from huggingface_hub import hf_hub_download
    api.create_repo(repo, repo_type="dataset", exist_ok=True,
                    private=lay.private)
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
               f"{lay.label} ({ad.store}): {len(names)} file(s)")
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
    man = {"family": lay.family, "tier": "P", "store": ad.store,
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
                            f"{lay.label} ({ad.store}): manifest")
    grid = publish_grid(ctx, api, repo, tok)
    if grid:
        man["grid"] = grid
        atomic_json(mp, man)
    mark(ctx.root, "publish")
    print(f"  publish: {len(entries)} file(s) verified by restore -> "
          f"https://huggingface.co/datasets/{repo}/tree/main/{prefix}")
    return man


PRIVATE_SUFFIX = "-private"


def check_publish_target(adapter, layout, repo):
    """REFUSE a publish whose repository contradicts the adapter's track.

    A `distribution = "private"` adapter may publish only to a repository
    whose id ends in `-private`, and a layout marked private must agree. A
    public adapter may not land in the private repository either — that is a
    routing error, not a safe default, because the registry would then point
    at bytes nobody can read. Raised, not asserted: `python3 -O` must not be
    able to delete the one check that keeps licensed data off the public repo.
    """
    dist = getattr(adapter, "distribution", "public")
    private_repo = str(repo).endswith(PRIVATE_SUFFIX)
    if dist == "private" and not (private_repo and layout.private):
        raise SystemExit(
            f"REFUSING to publish {adapter.store}: its distribution is "
            f"'private' and the target repository is {repo!r} "
            f"(layout.private={layout.private}). A private store goes to a "
            f"repository whose id ends in '{PRIVATE_SUFFIX}' and nowhere else; "
            f"nothing has been uploaded.")
    if dist != "private" and (private_repo or layout.private):
        raise SystemExit(
            f"REFUSING to publish {adapter.store}: its distribution is "
            f"{dist!r} and the target is the private repository {repo!r}.")
    return repo


def publish_grid(ctx, api, repo, tok):
    """Upload the monthly grid, verify its restore, and write its index.

    IN THE SAME JOB AS THE STORE (E-081 deliverable 3): the index measures the
    `.npy` header length and the CORS headers with a real ranged GET against
    the URL that was just published, which is a measurement only this machine,
    at this moment, can make (root CLAUDE.md §3 admits huggingface.co on
    measured properties, never assumed ones).
    """
    out_dir = os.path.join(ctx.root, FISHING_GRID_DIR)
    src = os.path.join(out_dir, FISHING_GRID_FILE)
    man_p = os.path.join(out_dir, "grid.json")
    if not (os.path.exists(src) and os.path.exists(man_p)):
        if ctx.adapter.store == "fishing":
            print(f"::warning::no {src} — the `grid` stage has not run, so "
                  f"the globe layer's monthly raster is not published. Run "
                  f"`--stage grid,publish`.")
        return None
    gm = read_json(man_p, {})
    if gm.get("sha256") != sha256(src):
        sys.exit(f"{src} does not match the sha256 in {man_p} — the grid "
                 f"changed after it was checked. Re-run `--stage grid`.")
    prefix = f"{ctx.layout.hf_root}/{FISHING_GRID_DIR}"
    names = [FISHING_GRID_FILE, "grid.json"]
    hub_commit(api, repo,
               hub_add_ops([(f"{prefix}/{n}", os.path.join(out_dir, n))
                            for n in names]),
               f"family 10.2 (fishing): the monthly 0.25° grid")
    from huggingface_hub import hf_hub_download
    scratch = os.path.join(ctx.scratch, "verify_grid")
    for n in names:
        want = sha256(os.path.join(out_dir, n))
        shutil.rmtree(scratch, ignore_errors=True)
        back = hf_hub_download(repo, f"{prefix}/{n}", repo_type="dataset",
                               token=tok, local_dir=scratch)
        got = sha256(back)
        shutil.rmtree(scratch, ignore_errors=True)
        if got != want:
            sys.exit(f"RESTORE MISMATCH {n}: uploaded {want}, downloaded "
                     f"{got} — the grid publish is not trustworthy")
    print(f"  grid: {len(names)} file(s) verified by restore -> "
          f"https://huggingface.co/datasets/{repo}/tree/main/{prefix}")
    index = None
    try:
        sys.path.insert(0, HERE)
        import publish_fishing_index as pfi
        index = pfi.write_index(
            grid_manifest=man_p,
            store_json=os.path.join(ctx.store, "store.json"),
            repo=repo,
            out=getattr(ctx.a, "index_out", "") or pfi.INDEX)
    except SystemExit:
        raise
    except Exception as e:                                  # noqa: BLE001
        # NOT silent (§4.6): the bytes are published and verified, and the
        # small JSON the browser reads first is what did not get written.
        print(f"::warning::the grid is published but its index was not "
              f"written: {type(e).__name__}: {e}. Run `python3 "
              f"ml/publish_fishing_index.py` to write data/fishing_index.json "
              f"from the published file.", flush=True)
    return {"prefix": prefix, "files": names, "shape": gm.get("shape"),
            "dtype": gm.get("dtype"), "months": gm.get("n_months"),
            "index_written": bool(index)}


# =================================================================== driver ==
STAGE_FN = {"index": stage_index, "fetch": stage_fetch, "grid": stage_grid,
            "publish": stage_publish}


def parse_stages(spec, stages=None):
    """`all` -> every stage; `fetch` -> [fetch]; `index,fetch` -> both, in the
    fixed stage order whatever order they were typed in. Unknown names refuse
    before anything runs. `stages` is the order (family 1 passes its own)."""
    order = list(stages or STAGES)
    if spec.strip() == "all":
        return list(order)
    want = [x.strip() for x in spec.split(",") if x.strip()]
    bad = [x for x in want if x not in order]
    if bad or not want:
        sys.exit(f"--stage {spec!r}: unknown stage(s) {bad} — choose from "
                 f"{order}, `all`, or a comma list of them")
    return [s for s in order if s in want]


def run_stages(ctx, stages, stage_fn=None, deps=None):
    stage_fn = STAGE_FN if stage_fn is None else stage_fn
    deps = DEPS if deps is None else deps
    for s in stages:
        for dep in deps.get(s, []):
            if not marked(ctx.root, dep):
                sys.exit(f"stage {s!r} needs {dep!r} first (E-079 §4: stage "
                         f"order is fixed) — {marker(ctx.root, dep)} is missing")
        if marked(ctx.root, s) and not ctx.a.force:
            print(f"stage {s}: already done — skipping (--force to redo)")
            continue
        t0 = time.time()
        print(f"\n=== stage {s} ({ctx.a.store}) ===", flush=True)
        stage_fn[s](ctx)
        print(f"=== stage {s} done in {time.time() - t0:.1f}s ===", flush=True)


# ==================================================================== smoke ==
SMOKE_START = "1981-12-20"     # deliberately BEFORE the epoch: negative bins
SMOKE_END = "1982-01-12"
# ONE STORE CANNOT USE THAT WINDOW: the fishing archive begins in 2012 and the
# adapter refuses a window with none of its years in it, which is the right
# answer for a real dispatch and the wrong one for a smoke test. Its window
# straddles a month boundary and a year boundary is added by the tests.
SMOKE_WINDOW = {"fishing": ("2012-01-01", "2012-02-05")}


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
                                "t": seconds_since_epoch(when),
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
                "t": seconds_since_epoch(dt.datetime(day.year, day.month,
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
                        "t": seconds_since_epoch(
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
        # 1 Hz, AND WRITTEN THE WAY DUACS WRITES IT: float64 `days since
        # 1950-01-01`, five consecutive SECONDS per day. Under v1 those five
        # samples collapsed onto one or two float32 days; under schema 2 they
        # must come back as five consecutive integers, which is what the
        # fixture exists to prove.
        n_per = 5
        cf_epoch = dt.datetime(1950, 1, 1)
        for y in sorted({x.year for x in days}):
            ydays = [x for x in days if x.year == y]
            d = os.path.join(root, "slatrack", mid, str(y))
            os.makedirs(d, exist_ok=True)
            n = len(ydays) * n_per
            # seconds since the CF epoch -> days, as a float64 the file carries
            t = np.array([
                ((dt.datetime(x.year, x.month, x.day, 3)
                  - cf_epoch).total_seconds() + k) / 86400.0
                for x in ydays for k in range(n_per)], np.float64)
            lat = np.linspace(-60, 60, n)
            lon = np.linspace(-179, 179, n)
            sla = 0.01 * np.sin(np.arange(n) + y)
            slau = sla + 0.002
            mdt = 0.3 + 0.001 * np.arange(n)
            p = os.path.join(d, f"{mid}_{y}.nc")
            ds = ncdf.Dataset(p, "w", format="NETCDF3_CLASSIC")
            ds.createDimension("time", n)
            tv = ds.createVariable("time", "f8", ("time",))
            tv.units = "days since 1950-01-01 00:00:00"   # DUACS's own epoch
            tv[:] = t
            for nm, arr in (("latitude", lat), ("longitude", lon),
                            ("sla_filtered", sla), ("sla_unfiltered", slau),
                            ("mdt", mdt)):
                v = ds.createVariable(nm, "f8", ("time",))
                v[:] = arr
            ds.close()
            for i, x in enumerate(ydays):
                base = seconds_since_epoch(dt.datetime(x.year, x.month,
                                                       x.day, 3))
                for k in range(n_per):
                    j = i * n_per + k
                    truth.append({"t": base + k, "lat": float(lat[j]),
                                  "lon": float(lon[j]),
                                  "v": [float(sla[j]), float(slau[j]),
                                        float(mdt[j])],
                                  "platform": platform_hash(mid), "qc": 1})
    elif store == "fishing":
        # THE REAL LAYOUT, IN MINIATURE (measured off the 2012 zip, see
        # FishingAdapter's docstring): one CSV per day inside
        # `mmsi-daily-csvs-10-v3-<year>.zip`, the six-column header, the cell's
        # LOWER-LEFT corner to one decimal, an unquoted integer MMSI — plus
        # the vessel table beside them. Both layouts are written: the zip,
        # which is what a real build streams, and the same day files in a
        # directory, which is the other thing `--source-dir` accepts.
        base = os.path.join(root, "fishing")
        os.makedirs(base, exist_ok=True)
        years = sorted({d.year for d in days})
        fleet = [(416000001, "trawlers"), (416000002, "drifting_longlines"),
                 (224000003, "squid_jigger")]
        with open(os.path.join(base, FISHING_VESSELS), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(FISHING_VESSEL_COLUMNS)
            for y in years:
                for (m, gear) in fleet:
                    # the third vessel is in the table for the FIRST year only,
                    # so a later year exercises qc 0 = unknown.
                    if m == 224000003 and y != years[0]:
                        continue
                    row = {"mmsi": str(m), "year": str(y),
                           "flag_gfw": "CHN" if str(m)[:3] == "412" else "FRA",
                           "vessel_class_gfw": gear,
                           "vessel_class_inferred": gear,
                           "self_reported_fishing_vessel": "true",
                           "length_m_gfw": "24.5", "active_hours": "1000.0",
                           "fishing_hours": "400.0"}
                    w.writerow([row.get(c, "") for c in
                                FISHING_VESSEL_COLUMNS])
        for y in years:
            ydays = [x for x in days if x.year == y]
            d_dir = os.path.join(base, FISHING_ZIP.format(year=y)[:-4])
            os.makedirs(d_dir, exist_ok=True)
            names = []
            for i, x in enumerate(ydays):
                # (ll_lat, ll_lon, mmsi, hours, fishing_hours), chosen to
                # exercise the edges: the dateline either side, equality of
                # the two channels, a zero-hour row, and a row above 24 h
                # (which the archive really does carry — an MMSI is not
                # always one vessel).
                cells = [(-65.5, 119.7, fleet[0][0], 1.5, 0.0),
                         (51.3, 3.2, fleet[1][0], 28.5, 12.25),
                         (0.0, 179.9, fleet[2][0], 4.0, 4.0),
                         (-0.1, -180.0, fleet[0][0], 0.0, 0.0)]
                p = os.path.join(d_dir,
                                 f"mmsi-daily-csvs-10-v3-{x:%Y-%m-%d}.csv")
                with open(p, "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(FISHING_COLUMNS)
                    for (la, lo, m, h, f) in cells:
                        w.writerow([f"{x:%Y-%m-%d}", _g(la), _g(lo), m,
                                    _g(h), _g(f)])
                        code = FISHING_GEAR_CODE[
                            dict(fleet)[m]] if not (
                                m == 224000003 and y != years[0]) else 0
                        truth.append({
                            "t": seconds_since_epoch(x),
                            "lat": la + FISHING_HALF_CELL,
                            "lon": float(f10.wrap_lon(lo + FISHING_HALF_CELL)),
                            "v": [f, h],
                            "platform": platform_hash(str(m)), "qc": code})
                names.append(p)
            with zipfile.ZipFile(os.path.join(
                    base, FISHING_ZIP.format(year=y)), "w",
                    zipfile.ZIP_DEFLATED) as z:
                # NOT in date order, exactly as the real archive's central
                # directory is not — so the adapter's sort is exercised.
                for p in sorted(names, key=lambda q: q[::-1]):
                    z.write(p, os.path.basename(p))
    else:
        raise ValueError(store)
    return truth


def _g(x):
    """A number the way the archive prints it: no trailing zeros (`120`)."""
    s = f"{float(x):.4f}".rstrip("0").rstrip(".")
    return s or "0"


def check_smoke(ctx, truth):
    """The store against the truth the generator kept — order, values, search.

    The truth's `t` is INTEGER SECONDS, so the comparison is EXACT: schema 2
    stores the second the generator wrote, and `!=` is the right operator where
    v1 needed a tolerance of a hundredth of a day.
    """
    ad = ctx.adapter
    st = check_store(ctx.store, ad,
                     anchor=(float(truth[0]["lat"]), float(truth[0]["lon"]),
                             int(truth[0]["t"] // PENTAD_SECONDS)),
                     chunk_rows=ctx.check_chunk)
    assert st.N == len(truth), (
        f"the store holds {st.N} row(s), the generator kept {len(truth)} "
        f"(a row that should have been dropped survived, or a good row "
        f"was lost)")
    want = sorted(truth, key=lambda r: (int(r["t"]) // PENTAD_SECONDS,
                                        int(r["t"])))
    ts = st.time_s()
    v = np.asarray(st["values"], np.float32)
    for i, w in enumerate(want):
        assert int(ts[i]) == int(w["t"]), (i, int(ts[i]), int(w["t"]))
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
        assert int(ts[0]) < 0, "a pre-1982 row lost its negative time_s"
    return (f"N={st.N} · C={st.C} · bins {st.bin_first}..{st.bin_last} · "
            f"{int((np.diff(st.bin_offsets) > 0).sum())} live bin(s) · "
            f"schema {st.schema_version}")


def run_smoke(store, root=None, keep=False, start="", end="",
              adapter=None, layout=None, make_sources=None, extra_ns=None,
              after=None):
    """Synthetic source -> index + fetch -> `check_smoke`, in seconds.

    E-082: family 1 passes its own `adapter`, `layout` and `make_sources`
    (the adapter's `smoke_sources(root, d_lo, d_hi)`), and an `after(ctx,
    truth, src)` that runs the probe on the same synthetic archive. Family 10
    passes none of them.
    """
    win = SMOKE_WINDOW.get(store) or getattr(adapter, "smoke_window", None) \
        or (SMOKE_START, SMOKE_END)
    start = start or win[0]
    end = end or win[1]
    tmp = root or tempfile.mkdtemp(prefix=f"f10smoke_{store}_")
    src = os.path.join(tmp, "src")
    work = os.path.join(tmp, "work")
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    d_lo, d_hi = parse_date(start), parse_date(end)
    truth = (make_sources or make_smoke_sources)(src, store, d_lo, d_hi)
    print(f"smoke     {store}: sources -> {src} "
          f"({len(truth)} truth row(s), {time.time() - t0:.1f}s)")
    ns = dict(store=store, work=work, source_dir=src,
              start=start, end=end, stage="all", force=False,
              attempts=1, qc_keep=2, socat_url="", smoke=True,
              max_hours=FISHING_HOURS_CEILING,
              grid_dtype=GRID_DTYPE_DEFAULT)
    ns.update(extra_ns or {})
    ap = argparse.Namespace(**ns)
    ctx = Ctx(ap, adapter=adapter, layout=layout)
    print(f"axis      bins {ctx.b_lo}..{ctx.b_hi} "
          f"({'NEGATIVE bins in range' if ctx.b_lo < 0 else 'all >= 1982'})")
    run_stages(ctx, ["index", "fetch"])
    out = check_smoke(ctx, truth)
    print(f"smoke     {store} OK in {time.time() - t0:.1f}s — {out}")
    if after is not None:
        after(ctx, truth, src)
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
                         "slatrack (along-track sea level), fishing (the AIS "
                         "fishing fleet, E-081)")
    ap.add_argument("--work", default=os.path.join(CACHE, f10.CACHE_DIRNAME),
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
                         f"pulls the requested years' column parts from "
                         f"{f10.HF_PARTIALS}/<store>/<year>/ on the Hub (put "
                         "there by ml/family10_parts_hub.py push) and "
                         "assembles them. This is how slatrack is built on a "
                         "box: HF_TOKEN only, no Copernicus credentials "
                         "(ml/CLAUDE.md §6). Refuses if any requested year has "
                         "no done.json on the Hub.")
    ap.add_argument("--allow-missing-years", action="store_true",
                    help="BUILD PAST AN INPUT THAT COULD NOT BE READ. Without "
                         "it the fetch stage refuses — and leaves the "
                         "unreadable years unmarked, so a re-run retries "
                         "exactly those — whenever a month's CSV, a GTMBA "
                         "dataset-year, a slatrack mission-year listing or a "
                         "batch of original files did not arrive, and the "
                         "assembler refuses parts that no marker claims. With "
                         "it, all of those become warnings, the store is built "
                         "from what arrived, and store.json grows a "
                         "`degraded` block naming every unit that is missing. "
                         "Also (as before) with --parts-from-hub: assemble "
                         "even though some years have no done.json on the Hub. "
                         "An ERDDAP 'produced no matching results' is NOT one "
                         "of these — that is the archive saying the year is "
                         "empty, and it is counted, not refused.")
    ap.add_argument("--check-chunk-rows", type=int, default=CHECK_CHUNK_ROWS,
                    help="rows per block in the assertion pass over the "
                         "finished store (default "
                         f"{CHECK_CHUNK_ROWS:,}). The checks read the store's "
                         "memmaps block by block and carry the seam, so the "
                         "answer does not depend on this number; it only "
                         "bounds the peak RAM of the check. Lower it on a "
                         "small box, raise it on a big one.")
    ap.add_argument("--max-hours", type=float, default=FISHING_HOURS_CEILING,
                    help="fishing only: the largest `hours` a single (day, "
                         "0.1° cell, MMSI) row may carry before the build "
                         f"REFUSES (default {FISHING_HOURS_CEILING:g} = a "
                         "week). E-081 §2 wrote 24.001; the archive breaks "
                         "that on 1.6 % of 2012's rows, to 47.56 h, because "
                         "an MMSI is not always one vessel — so the ceiling "
                         "is set where a misread column is caught and the "
                         "data is not. Pass 24.001 to see the plan's rule "
                         "fire; rows over 24 h are counted either way.")
    ap.add_argument("--allow-small-disk", action="store_true",
                    help="fishing only: fetch even though the projected parts "
                         "+ store do not fit the disk. The projection is in "
                         "plan.json (`projected`), measured at 0.1116 rows "
                         "per byte of zip; the whole 2012-2024 archive is "
                         "~18.5 GB of store and does NOT fit a hosted runner.")
    ap.add_argument("--grid-dtype", default=GRID_DTYPE_DEFAULT,
                    choices=("float16", "float32"),
                    help="the monthly 0.25° grid's dtype. E-081 §3 specifies "
                         "float16 and the ARCHIVE DOES NOT FIT IT: measured "
                         "2026-09-16, the largest 0.25° cell-month sum is "
                         "29,460 vessel-hours in 2012 and 595,726 in 2024, "
                         "against float16's largest finite value of 65,504 — "
                         "so the busiest cells would be +inf. float32 is "
                         "therefore the default (slab 8.3 MB a month, still "
                         "under family 7's 14.5 MB frame); float16 is right "
                         "for a narrow window and refuses per month if a cell "
                         "overflows. data/fishing_index.json publishes the "
                         "dtype and the slab size either way.")
    ap.add_argument("--index-out", default="",
                    help="where the publish stage writes the grid's index "
                         "(default data/fishing_index.json in this checkout)")
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
