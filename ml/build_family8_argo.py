#!/usr/bin/env python3
"""Family 8's ARGO OBSERVATION STORE — raw profiles, never gridded — `f8argo_l0`.

E-076 §3, built once as one small table. The executable specification is
`ml/plans/E076_family8_nearest_observations.md`; this file is §3 made runnable
and every decision below is the plan's, not a new one. NOTHING in
`ml/cone_sampler.py`, `ml/build_family7.py` or the family-7 tensor is touched:
family 8 is family 7's dense groups, unchanged and already on the Hub, PLUS
this store.

PLAIN ENGLISH. Family 7 hands each cone dot a gridded Argo depth column and,
at 92 % of five-day bins, a "no value here" token — the Roemmich-Gilson
product is a monthly MAPPED field, so most of the tensor's largest group is
the absence of a measurement. Family 8 stops asking "is there a value at this
cell and this bin" and asks "what are the k nearest measurements, and how far
away are they" (E-076 §0). That question needs the raw profiles, which is what
this builder extracts: every Argo temperature and salinity profile, quality
controlled, interpolated onto the sixteen Roemmich-Gilson pressure levels,
tagged with its own position and its own instant, sorted by five-day bin.
`ml/family8_store.py` is the reader and does the k-nearest search.

THREE STAGES, IN FIXED ORDER — `index | profiles | publish` (or `all`):

  index     the GDAC global profile index (one line per profile: file, date,
            latitude, longitude, ...). Parsed into `index.npz` and MEASURED
            into `index_stats.json` — profiles per year and per pentad, the
            real mean nearest-neighbour distance in one pentad and in a 30-day
            window, and the 3rd / 5th / 7th neighbour distance at 2,000 random
            anchors. That measurement REPLACES the Poisson estimate of E-076
            §2.3 (`~210 km in a pentad, ~87 km in 30 days, the 5th neighbour
            at ~195 km`), which is the number the choice of k = 5 rests on.
  profiles  the daily geo files, streamed: download one (basin, day), extract,
            DELETE it. Per-YEAR `.done` markers, then the store.
  publish   upload the store to the Hub and DOWNLOAD EVERY FILE BACK to
            compare sha256, exactly as family 7's publish does.

WHY IT MUST STREAM. The GDAC daily geo files are ~23 MB/day across the three
basins (measured 2015-01-03: Atlantic 6.3 MB / 106 profiles / 1236 levels,
Pacific 12.6 MB, Indian 4.1 MB), i.e. 180-250 GB over 2004-2024, and the box
has ~30 GB free. So at most `MAX_INFLIGHT` files exist on disk at once: a
bounded download-ahead feeds a process pool that parses the netCDF, and each
file is deleted the moment its profiles are extracted. The extracted result is
four orders of magnitude smaller — 16 levels x T/S x float16 per profile, about
0.2 GB for the whole archive.

A MISSING DAY IS A LEGITIMATE EMPTY DAY. A 404 from every mirror is recorded
in progress and the build continues; any other HTTP error retries three times
and then FAILS THE YEAR (ml/CLAUDE.md §4.6 — a step that can fail silently
will). The per-year markers are written only after the year's extract is on
disk (§5.21: a marker may only under-claim), so a resumed job restarts at the
year it lost rather than at 2004.

RESUMABILITY IS THE CONTRACT, the same one family 7 has: re-run with the same
`--work` and finished stages and finished years are skipped.

Run:
  python3 ml/build_family8_argo.py --smoke                    # synthetic, seconds
  python3 ml/build_family8_argo.py --work /opt/earth-cache/f8 --stage all
  python3 ml/build_family8_argo.py --work ... --stage index
  python3 ml/build_family8_argo.py --work ... --source-dir DIR  # no network
"""
import argparse
import datetime as dt
import gzip
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_family3 as f3                                      # noqa: E402
import build_family7 as f7                                      # noqa: E402
import family8_store as fs                                      # noqa: E402
from aggregate_cadence import EPOCH, bin_index, bin_start        # noqa: E402
# ONE definition of each of these, imported rather than restated (E-076 §3,
# and the same rule build_family7 follows for build_family3.LEVELS).
from build_family7 import (START, END, Progress, atomic_json,    # noqa: E402
                           download_verified, git_sha, hub_repo, mark,
                           marked, marker, read_json, sha256, utcnow)

CACHE = os.path.join(HERE, "cache")
RECIPE = "f8argo_l0"
STEM = "family8_argo_l0"
HF_DATASET = "earth-tensors"
HF_PREFIX = f"tensors/{STEM}"

# ---- the axis, shared with families 4-7 BY CONSTRUCTION --------------------
# bin = floor((date - 1982-01-01) / 5 days); bins 0..3141 cover
# 1982-01-01..2024-12-31. `aggregate_cadence.bin_index` is that rule, and the
# assertion below is what stops a second copy of it drifting.
PENTAD_DAYS = 5
N_BINS = 3142                                   # bins 0..3141
assert EPOCH == START, (EPOCH, START)
assert bin_index(dt.date(2024, 12, 31), PENTAD_DAYS) == N_BINS - 1
assert fs.N_BINS == N_BINS and fs.PENTAD_DAYS == PENTAD_DAYS

# ---- the levels: the SAME sixteen Roemmich-Gilson pressures family 3 and
# family 7 use, imported so a level list cannot exist twice.
LEVELS = f3.LEVELS
LEVELS_ARR = np.asarray(LEVELS, np.float64)
NLEV = len(LEVELS)
assert NLEV == 16 and LEVELS[0] == 10.0 and LEVELS[-1] == 1900.0

# ---- the source -----------------------------------------------------------
# VERIFIED 2026-09-07 from this sandbox: HEAD 200 on
# geo/atlantic_ocean/2015/01/20150103_prof.nc, content-length 6,345,092,
# accept-ranges bytes, content-type application/x-netcdf.
GDAC = "https://data-argo.ifremer.fr"
GDAC_MIRROR = "https://argo-gdac-sandbox.s3.eu-west-3.amazonaws.com/pub"
INDEX_NAME = "ar_index_global_prof.txt.gz"
BASINS = ("atlantic_ocean", "indian_ocean", "pacific_ocean")
UA = {"User-Agent": "earth-science-pipeline/1.0 "
                    "(research; github blauewelt/earth)"}

# The float array reached 3,000 floats in 2007 and the interior is essentially
# unobserved before 2004 (E-076 §2.3): "none within R" is then the honest,
# measured answer rather than a gap to be papered over. The STORE's axis still
# covers bins 0..3141 — the early bins are simply empty — so a build that
# wants them only has to pass --start.
ARGO_START = dt.date(2004, 1, 1)

# THE JOB IS DOWNLOAD-BOUND, AND THE FIRST VERSION OF THESE THREE NUMBERS COST
# A RUN. family8-build #2 (a fresh Vast box, 545 Mbps down, 6.4 cores) sat 20
# minutes in the profiles stage at ~2 MB/s aggregate with cpu 1 % and the disk
# flat — i.e. every second of it was spent waiting on sockets — which projects
# to over 30 h against the 24 h timeout. It was cancelled.
#
# Ifremer THROTTLES PER CONNECTION (measured from this sandbox 2026-09-07:
# ~6.5 MB/s on a single connection from data-argo.ifremer.fr, ~4 MB/s from the
# S3 mirror), so throughput is bought with connections, not with a faster host
# — and S3 is the one that scales with them, which is why `day_urls` asks it
# FIRST and leaves Ifremer as the fallback.
#
# MAX_INFLIGHT bounds FILES ON DISK, not connections: at 4-13 MB per daily
# file, 24 is <= ~300 MB, which is nothing against the box's headroom and is
# the whole reason the streaming design exists. DL_WORKERS is the number of
# concurrent connections. Both are flags (--inflight, --dl-workers) because
# the right value is a property of the box's link, not of the archive.
MAX_INFLIGHT = 24
DL_WORKERS = 12
MAX_PROCS = 8
SOCKET_TIMEOUT = 60          # per socket operation, not per file
CHUNK = 1 << 22             # 4 MiB reads: a 6 MB file is two syscalls, not 6

# The size of the job these projections are about: 2004-01-01..2024-12-31 is
# 7,671 days, three basins each, 23,013 (basin, day) tasks.
FULL_DAYS = (dt.date(2024, 12, 31) - dt.date(2004, 1, 1)).days + 1
FULL_TASKS = FULL_DAYS * 3
PROJECTION_WARN_H = 18.0     # the 24 h workflow timeout, with room to resume

# ---- the netCDF fills, as the real files declare them (measured) -----------
FILL_F = 99999.0            # PRES/TEMP/PSAL and LATITUDE/LONGITUDE
FILL_JULD = 999999.0
JULD_EPOCH = dt.date(1950, 1, 1)
JULD_OFFSET = (START - JULD_EPOCH).days          # 11688 days, 1950 -> 1982

STAGES = ["index", "profiles", "publish"]
DEPS = {"profiles": ["index"], "publish": ["profiles"]}


# =============================================================== QC policy ==
# FIXED by E-076 §3's build spec. Do not soften any clause: every one of them
# decides whether a number reaches the model, and a loosened flag is invisible
# downstream — it just makes the store slightly wrong everywhere.
QC_GOOD = (b"1", b"2")
QC_POLICY = (
    "Profile kept only if POSITION_QC in {1,2}, JULD_QC in {1,2}, LATITUDE "
    "and LONGITUDE finite and not fill, JULD not fill, DATA_MODE in {R,A,D}. "
    "For DATA_MODE D or A the *_ADJUSTED variables and their *_ADJUSTED_QC "
    "are used; for R the raw variables and their *_QC. A sample counts only "
    "if its value is not fill, finite, and its QC in {1,2}; its PRESSURE must "
    "additionally have QC in {1,2}. T and S are treated INDEPENDENTLY. "
    "Samples are sorted by pressure and duplicate pressures dropped (first "
    "kept). Linear interpolation onto the 16 RG levels; a level is filled "
    "only if bracketed by two good samples whose pressure gap <= tol(level), "
    "tol = 25 dbar for level <= 200, 50 dbar for <= 500, 100 dbar above; the "
    "shallowest level (10 dbar) may additionally be filled by the shallowest "
    "good sample if that sample is <= 20 dbar (nearest, not extrapolated). "
    "Everything else NaN. A profile with < 2 filled T levels AND < 2 filled S "
    "levels is dropped. Longitude stored in [-180, 180). Duplicate "
    "(PLATFORM_NUMBER, CYCLE_NUMBER, DIRECTION) keeps the first seen in "
    "(basin, date) order."
)
SHALLOW_NEAREST_DBAR = 20.0

DROP_REASONS = ("position_qc", "juld_qc", "latlon_fill", "juld_fill",
                "data_mode", "out_of_axis", "missing_variables",
                "too_few_levels")


def level_tol(levels=None):
    """tol(level) in dbar — 25 / 50 / 100, by the plan's two breakpoints."""
    p = LEVELS_ARR if levels is None else np.asarray(levels, np.float64)
    return np.where(p <= 200.0, 25.0, np.where(p <= 500.0, 50.0, 100.0))


TOL = level_tol()


# ============================================================ the extractor ==
def interp_levels(pres, val, levels=None, tol=None,
                  shallow_nearest=SHALLOW_NEAREST_DBAR):
    """Linear interpolation of one profile onto `levels`, with the gap rule.

    `pres` must already be ASCENDING and free of duplicates, and `val` its
    good samples. A level is filled only when two good samples bracket it and
    their pressure gap is within `tol` — a 1,500 dbar hole does not get to
    invent the thermocline. The shallowest level is additionally allowed to
    take the shallowest good sample when that sample is within
    `shallow_nearest` dbar of the surface, because a float's first bin is
    often at 4-15 dbar and refusing the 10 dbar level for it would throw away
    the surface layer of most of the array; the value is taken AS IS, never
    extrapolated along a gradient.

    Vectorised over levels, because it runs once per profile per variable and
    there are ~2.5 M profiles.
    """
    L = LEVELS_ARR if levels is None else np.asarray(levels, np.float64)
    tol = TOL if tol is None else np.asarray(tol, np.float64)
    out = np.full(len(L), np.nan, np.float64)
    n = len(pres)
    if n == 0:
        return out
    pres = np.asarray(pres, np.float64)
    val = np.asarray(val, np.float64)
    # pres[j-1] <= L < pres[j]; an EXACT hit is its own bracket (gap 0), which
    # is why side="right" plus the equality test rather than side="left".
    j = np.searchsorted(pres, L, side="right")
    lo = j - 1
    lo_c = np.clip(lo, 0, n - 1)
    exact = (lo >= 0) & (pres[lo_c] == L)
    hi = np.where(exact, lo, j)
    have = (lo >= 0) & (hi <= n - 1)
    lo_c = np.clip(lo, 0, n - 1)
    hi_c = np.clip(hi, 0, n - 1)
    gap = pres[hi_c] - pres[lo_c]
    ok = have & (gap <= tol)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(gap > 0, (L - pres[lo_c]) / np.where(gap > 0, gap, 1.0), 0.0)
    out = np.where(ok, val[lo_c] + w * (val[hi_c] - val[lo_c]), np.nan)
    if len(L) and not ok[0] and pres[0] <= shallow_nearest:
        out[0] = val[0]
    return out


def _good(value, qc, pres_qc_ok):
    """The samples that count: finite, not fill, QC in {1,2}, pressure good."""
    v = np.asarray(value, np.float64)
    return (pres_qc_ok & np.isfinite(v) & (v != FILL_F) & (np.abs(v) < 1e30)
            & np.isin(qc, QC_GOOD))


def _sorted_unique(pres, val):
    """Ascending by pressure, duplicate pressures dropped (FIRST kept).

    A stable sort makes "first" mean the first in the FILE among equal
    pressures, which is the only reading of the rule that does not depend on
    numpy's sort implementation.
    """
    order = np.argsort(pres, kind="stable")
    p, v = pres[order], val[order]
    if len(p) == 0:
        return p, v
    keep = np.empty(len(p), bool)
    keep[0] = True
    np.not_equal(p[1:], p[:-1], out=keep[1:])
    return p[keep], v[keep]


def _chars(a):
    """A netCDF |S1 array as a clean bytes array of the same shape."""
    a = np.asarray(a)
    if a.dtype.kind == "S":
        return a
    return np.char.encode(a.astype(str), "ascii")


def _platform_numbers(pn):
    """(N_PROF, 8) of |S1 -> int32 WMO numbers; 0 where unparseable."""
    out = np.zeros(pn.shape[0], np.int64)
    for i in range(pn.shape[0]):
        s = b"".join(bytes(c) for c in pn[i]).strip()
        try:
            out[i] = int(s)
        except (ValueError, TypeError):
            out[i] = 0
    return out


def dedupe_key(wmo, cycle, direction):
    """(PLATFORM_NUMBER, CYCLE_NUMBER, DIRECTION) packed into one int64.

    The same profile can appear in two daily files across a basin boundary or
    a re-issue (E-076 §3's build spec), and a set of 2.5 M python tuples costs
    more memory than the store itself. Packing makes the dedupe an
    `np.unique`. Cycle numbers are < 2^20 in the whole archive; direction is
    A / D / other.
    """
    d = np.where(np.asarray(direction) == b"A", 0,
                 np.where(np.asarray(direction) == b"D", 1, 2))
    return (np.asarray(wmo, np.int64) << 22) + \
        (np.asarray(cycle, np.int64) << 2) + d.astype(np.int64)


def extract_day(path, basin, date_iso, b_lo=0, b_hi=N_BINS - 1):
    """One daily geo file -> the profiles it contributes, plus WHY the rest went.

    Runs in a worker PROCESS (module-level so it pickles). Returns plain numpy
    arrays and a counts dict; the caller deletes the file.

    `set_auto_mask(False)` is deliberate: the fill values are handled here,
    explicitly and by the same test as the QC flags, rather than by netCDF4's
    masking, so "why is this sample absent" has exactly one answer in the code.
    """
    import netCDF4 as ncdf
    ds = ncdf.Dataset(path)
    ds.set_auto_mask(False)
    ds.set_auto_scale(True)
    try:
        n_prof = int(ds.dimensions["N_PROF"].size)
        counts = {"n_prof": n_prof, "kept": 0}
        counts.update({f"drop_{r}": 0 for r in DROP_REASONS})
        hist_t = np.zeros(NLEV + 1, np.int64)
        hist_s = np.zeros(NLEV + 1, np.int64)
        if n_prof == 0:
            return _empty_extract(counts, hist_t, hist_s)

        juld = np.asarray(ds.variables["JULD"][:], np.float64)
        lat = np.asarray(ds.variables["LATITUDE"][:], np.float64)
        lon = np.asarray(ds.variables["LONGITUDE"][:], np.float64)
        pos_qc = _chars(ds.variables["POSITION_QC"][:]).reshape(-1)
        juld_qc = _chars(ds.variables["JULD_QC"][:]).reshape(-1)
        mode = _chars(ds.variables["DATA_MODE"][:]).reshape(-1)
        direction = _chars(ds.variables["DIRECTION"][:]).reshape(-1)
        cycle = np.asarray(ds.variables["CYCLE_NUMBER"][:], np.int64)
        wmo = _platform_numbers(_chars(ds.variables["PLATFORM_NUMBER"][:]))

        raw = {k: np.asarray(ds.variables[k][:], np.float64)
               for k in ("PRES", "TEMP", "PSAL")}
        raw_qc = {k: _chars(ds.variables[f"{k}_QC"][:]) for k in raw}
        adj, adj_qc = {}, {}
        for k in ("PRES", "TEMP", "PSAL"):
            vk, qk = f"{k}_ADJUSTED", f"{k}_ADJUSTED_QC"
            adj[k] = (np.asarray(ds.variables[vk][:], np.float64)
                      if vk in ds.variables else None)
            adj_qc[k] = (_chars(ds.variables[qk][:])
                         if qk in ds.variables else None)
    finally:
        ds.close()

    keep_i, out_t, out_s = [], [], []
    nlev_v, maxp_v = [], []
    for i in range(n_prof):
        if pos_qc[i] not in QC_GOOD:
            counts["drop_position_qc"] += 1
            continue
        if juld_qc[i] not in QC_GOOD:
            counts["drop_juld_qc"] += 1
            continue
        if not np.isfinite(juld[i]) or juld[i] == FILL_JULD or \
                abs(juld[i]) > 1e30:
            counts["drop_juld_fill"] += 1
            continue
        if (not np.isfinite(lat[i]) or not np.isfinite(lon[i])
                or lat[i] == FILL_F or lon[i] == FILL_F
                or abs(lat[i]) > 90.0 or abs(lon[i]) > 360.0):
            counts["drop_latlon_fill"] += 1
            continue
        if mode[i] not in (b"R", b"A", b"D"):
            counts["drop_data_mode"] += 1
            continue
        b = int(np.floor((juld[i] - JULD_OFFSET) / PENTAD_DAYS))
        if b < b_lo or b > b_hi:
            counts["drop_out_of_axis"] += 1
            continue

        # DATA_MODE decides which triple is read — the adjusted values are the
        # delayed-mode science product, the raw ones are all a real-time
        # profile has. Falling back to raw when a file simply lacks the
        # adjusted variable would silently mix two products under one flag, so
        # the absence is treated as "no good samples" instead.
        use_adj = mode[i] in (b"D", b"A")
        src = adj if use_adj else raw
        src_qc = adj_qc if use_adj else raw_qc
        if any(src[k] is None or src_qc[k] is None
               for k in ("PRES", "TEMP", "PSAL")):
            counts["drop_missing_variables"] += 1
            continue
        p = src["PRES"][i]
        p_ok = (np.isfinite(p) & (p != FILL_F) & (np.abs(p) < 1e30)
                & np.isin(src_qc["PRES"][i], QC_GOOD))
        t_ok = _good(src["TEMP"][i], src_qc["TEMP"][i], p_ok)
        s_ok = _good(src["PSAL"][i], src_qc["PSAL"][i], p_ok)

        pt, vt = _sorted_unique(p[t_ok], src["TEMP"][i][t_ok])
        ps, vs = _sorted_unique(p[s_ok], src["PSAL"][i][s_ok])
        ti = interp_levels(pt, vt)
        si = interp_levels(ps, vs)
        n_t = int(np.isfinite(ti).sum())
        n_s = int(np.isfinite(si).sum())
        hist_t[n_t] += 1
        hist_s[n_s] += 1
        if n_t < 2 and n_s < 2:
            counts["drop_too_few_levels"] += 1
            continue

        any_ok = t_ok | s_ok
        nlev_v.append(int(any_ok.sum()))
        maxp_v.append(float(p[any_ok].max()) if any_ok.any() else np.nan)
        out_t.append(ti)
        out_s.append(si)
        keep_i.append(i)
        counts["kept"] += 1

    if not keep_i:
        return _empty_extract(counts, hist_t, hist_s)
    idx = np.asarray(keep_i, np.int64)
    time_days = (juld[idx] - JULD_OFFSET).astype(np.float64)
    lonw = np.mod(lon[idx] + 180.0, 360.0) - 180.0
    cyc = np.clip(cycle[idx], 0, 32767)
    return {
        "bin": np.floor(time_days / PENTAD_DAYS).astype(np.int16),
        "time_days": time_days.astype(np.float32),
        "lat": lat[idx].astype(np.float32),
        "lon": lonw.astype(np.float32),
        "temp": np.asarray(out_t, np.float64).astype(np.float16),
        "psal": np.asarray(out_s, np.float64).astype(np.float16),
        "wmo": np.clip(wmo[idx], 0, np.iinfo(np.int32).max).astype(np.int32),
        "cycle": cyc.astype(np.int16),
        "nlev": np.clip(np.asarray(nlev_v, np.int64), 0, 32767).astype(np.int16),
        "maxpres": np.asarray(maxp_v, np.float64).astype(np.float16),
        "mode": mode[idx].astype("|S1"),
        "key": dedupe_key(wmo[idx], cyc, direction[idx]),
        "counts": counts, "hist_t": hist_t, "hist_s": hist_s,
        "basin": basin, "date": date_iso,
    }


def _empty_extract(counts, hist_t, hist_s):
    z = lambda d: np.zeros(0, d)                                  # noqa: E731
    return {"bin": z(np.int16), "time_days": z(np.float32),
            "lat": z(np.float32), "lon": z(np.float32),
            "temp": np.zeros((0, NLEV), np.float16),
            "psal": np.zeros((0, NLEV), np.float16),
            "wmo": z(np.int32), "cycle": z(np.int16), "nlev": z(np.int16),
            "maxpres": z(np.float16), "mode": np.zeros(0, "|S1"),
            "key": z(np.int64), "counts": counts,
            "hist_t": hist_t, "hist_s": hist_s,
            "basin": "", "date": ""}


# ================================================================= fetching ==
class _NotFound(Exception):
    """A definite 404 — the day does not exist, which is a legitimate answer."""


def _http_to_file(url, path, timeout=SOCKET_TIMEOUT):
    """GET url -> path, SIZE-VERIFIED against Content-Length.

    family 7's `download_verified` rule, inline here because a daily file must
    also tell a 404 apart from an error (a HEAD-then-GET cannot: `remote_size`
    returns None for both). Measured 2026-08-18 on a 477 MB year: a truncated
    transfer raises nothing and surfaces later as `NetCDF: HDF error`.

    `timeout` is per SOCKET OPERATION, so it is a stall detector rather than a
    deadline for the file: 60 s of silence on a 6 MB read is a dead connection,
    while the old 600 s let one wedged socket hold a download slot for ten
    minutes. Reads are 4 MiB so a daily file costs two or three syscalls.
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
        if e.code == 404:
            raise _NotFound(url) from None
        raise
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


def fetch_first(urls, path, attempts=3, sleep=5.0):
    """The first URL that serves the file; None if EVERY url says 404.

    A 404 everywhere is a missing day and the caller continues. Anything else
    retries `attempts` times across the whole url list and then RAISES, which
    fails the year — never silently (ml/CLAUDE.md §4.6).
    """
    errs = []
    for i in range(attempts):
        n404 = 0
        for u in urls:
            try:
                return _http_to_file(u, path)
            except _NotFound:
                n404 += 1
            except Exception as e:                              # noqa: BLE001
                errs.append(f"{u}: {type(e).__name__}: {e}")
        if n404 == len(urls):
            return None
        if i < attempts - 1:
            time.sleep(sleep * (2 ** i))
    raise IOError(f"{os.path.basename(path)}: {attempts} attempt(s) over "
                  f"{len(urls)} url(s) all failed — " + " | ".join(errs[-4:]))


def day_urls(basin, day):
    """The urls for one (basin, day), S3 MIRROR FIRST, Ifremer as the fallback.

    Not a preference between two equal hosts. Ifremer throttles per connection,
    so twelve parallel fetches from it share roughly what one gets; the S3
    mirror scales with connections, which is the only thing that makes a
    23,013-file pull fit in a day. Ifremer stays in the list because it is the
    AUTHORITATIVE copy and the mirror can lag or 404 — `fetch_first` walks the
    list, so a day the mirror lacks is still fetched, just more slowly.
    """
    rel = f"geo/{basin}/{day.year:04d}/{day.month:02d}/{day:%Y%m%d}_prof.nc"
    return [f"{GDAC_MIRROR}/{rel}", f"{GDAC}/{rel}"], rel


# ----------------------------------------------------- the bounded pipeline --
class _Inline:
    """A pool-shaped object that runs the work here. `--jobs 1`, and the tests."""

    class _F:
        """A Future that is already finished — the work ran at submit time.

        `done` and `cancel` are not decoration: `stream_days` asks the head of
        its download queue whether it has finished before handing it to the
        parser, and cancels whatever is outstanding on the way out. A stub
        missing either one turns `--jobs 1` into an AttributeError inside a
        `finally`, which is where a real failure would otherwise have been
        reported from.
        """

        def __init__(self, v):
            self._v = v

        def result(self):
            return self._v

        def done(self):
            return True

        def cancel(self):
            return False

    def submit(self, fn, *a, **kw):
        return self._F(fn(*a, **kw))

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False


def _pools(n_proc, dl_workers=DL_WORKERS):
    if n_proc <= 1:
        return _Inline(), _Inline()
    return (ThreadPoolExecutor(max_workers=dl_workers),
            ProcessPoolExecutor(max_workers=n_proc,
                                mp_context=multiprocessing.get_context("fork")))


def stream_days(tasks, fetch, parse, n_proc=2, max_inflight=MAX_INFLIGHT,
                dl_workers=DL_WORKERS):
    """Yield `(task, result_or_None, nbytes)` in TASK ORDER, downloads AHEAD.

    THE PROPERTY THAT MATTERS IS THAT DOWNLOADING AND PARSING OVERLAP, and the
    first version of this loop only half had it. It refused to hand a finished
    download to the parser while more than `n_proc` parses were outstanding,
    and it blocked on the HEAD download before topping the queue back up — so
    the number of live connections sagged toward `DL_WORKERS` at best and
    toward one whenever the head of the queue was slow. On a download-bound
    job that is the whole cost: family8-build #2 ran at ~2 MB/s with the CPU
    at 1 %.

    The shape now, in the order the loop tries them each pass:

      1. TOP UP FIRST, always. `len(dl) + len(pending)` is the number of files
         that exist on disk or are on their way there, and it is held at
         `max_inflight` — so the thread pool always has more work queued than
         it has workers, and no download slot idles while the main thread is
         busy.
      2. HAND OVER EVERY *FINISHED* HEAD DOWNLOAD, without blocking. Only the
         head is eligible, because the yield order is task order and the
         duplicate rule ("keep the first seen in (basin, date) order") makes
         that order part of the store's definition. A finished download behind
         a slow one simply waits on disk; it is already paid for.
      3. Only then BLOCK — on the oldest parse if one is in flight, otherwise
         on the head download. Every blocking point leaves the thread pool and
         the process pool running, which is what makes the wait productive.

    Parses are submitted to the process pool without a second bound: the pool
    queues them, and `pending` is already inside the `max_inflight` budget, so
    at most `max_inflight` results can be resident.
    """
    it = iter(list(tasks))
    dl, pending = deque(), deque()
    tp, pp = _pools(n_proc, dl_workers)
    try:
        while True:
            while len(dl) + len(pending) < max_inflight:
                t = next(it, None)
                if t is None:
                    break
                dl.append((t, tp.submit(fetch, t)))
            if not dl and not pending:
                return
            moved = False
            while dl and dl[0][1].done():
                task, fut = dl.popleft()
                got = fut.result()
                if got is None:
                    yield task, None, 0
                else:
                    path, owned, nbytes = got
                    pending.append((task, path, owned, nbytes,
                                    pp.submit(parse, task, path)))
                moved = True
            if moved:
                continue                      # top up before doing anything else
            if pending:
                task, path, owned, nbytes, fut = pending.popleft()
                try:
                    res = fut.result()        # blocks; downloads keep running
                finally:
                    if owned and os.path.exists(path):
                        os.remove(path)
                yield task, res, nbytes
                continue
            # Nothing parsed and nothing finished downloading: the head of the
            # download queue is the only thing left to wait for.
            task, fut = dl.popleft()
            got = fut.result()
            if got is None:
                yield task, None, 0
                continue
            path, owned, nbytes = got
            pending.append((task, path, owned, nbytes,
                            pp.submit(parse, task, path)))
    finally:
        for _, fut in dl:
            fut.cancel()
        for _, path, owned, _n, fut in pending:
            fut.cancel()
            if owned and os.path.exists(path):
                os.remove(path)
        for pool in (tp, pp):
            if hasattr(pool, "shutdown"):
                pool.shutdown(wait=False, cancel_futures=True)


class Throughput:
    """files/s, MB/s and the PROJECTED hours for the whole 2004-2024 pull.

    ml/CLAUDE.md §0.2 and §4.7: a job that reports "downloading" is not
    evidence it will finish. family8-build #2 spent twenty minutes proving
    that a rate nobody computes is a rate nobody notices — the numbers were
    all there, in the elapsed time and the file count, and the only thing
    missing was the division. So the division happens every 50 files, lands in
    progress.json as an ARTEFACT rather than a log line (§5.25), and raises a
    `::warning::` the moment the projection passes `PROJECTION_WARN_H`, while
    the run is still cheap to cancel and re-tune with --dl-workers/--inflight.
    """

    def __init__(self, every=50, full_tasks=FULL_TASKS,
                 warn_hours=PROJECTION_WARN_H):
        self.t0 = time.time()
        self.every = every
        self.full_tasks = full_tasks
        self.warn_hours = warn_hours
        self.files = 0
        self.tasks = 0
        self.bytes = 0
        self.warned = False

    def add(self, nbytes, downloaded):
        self.tasks += 1
        if downloaded:
            self.files += 1
            self.bytes += int(nbytes)
        return self.tasks % self.every == 0

    def report(self):
        el = max(time.time() - self.t0, 1e-6)
        mean = self.bytes / self.files if self.files else 0.0
        mbps = self.bytes / el / 1e6
        fps = self.files / el
        # The projection is BYTES, not tasks: a missing day costs a request and
        # no transfer, and the mix of empty days changes with the decade.
        hours = ((self.full_tasks * mean) / (self.bytes / el) / 3600.0
                 if self.bytes else float("inf"))
        return {"files": self.files, "tasks": self.tasks,
                "mb": round(self.bytes / 1e6, 1),
                "elapsed_s": round(el, 1),
                "files_per_s": round(fps, 2),
                "mb_per_s": round(mbps, 2),
                "mean_file_mb": round(mean / 1e6, 2),
                "projected_full_pull_h": (round(hours, 1)
                                          if np.isfinite(hours) else None),
                "projection_basis": f"{self.full_tasks} tasks "
                                    f"(= {FULL_DAYS} days x 3 basins)"}

    def check(self, rep):
        """One loud warning per stage when the projection blows the timeout."""
        h = rep.get("projected_full_pull_h")
        if self.warned or h is None or h <= self.warn_hours:
            return None
        self.warned = True
        msg = (f"::warning::throughput {rep['mb_per_s']} MB/s "
               f"({rep['files_per_s']} files/s) projects {h} h for the full "
               f"2004-2024 pull — past the {self.warn_hours:g} h budget and "
               f"the workflow's 24 h timeout. This job is DOWNLOAD-BOUND: "
               f"raise --dl-workers / --inflight, or resume in year slices "
               f"with --start/--end (finished years are skipped).")
        print(msg, flush=True)
        return msg


# ================================================================== context ==
class Ctx:
    """Everything every stage needs: the dates, the paths, the progress file."""

    def __init__(self, a):
        self.a = a
        self.work = os.path.abspath(a.work)
        os.makedirs(self.work, exist_ok=True)
        self.source_dir = os.path.abspath(a.source_dir) if a.source_dir else None
        self.scratch = os.path.join(self.work, "src")
        os.makedirs(self.scratch, exist_ok=True)
        self.store = os.path.join(self.work, STEM)
        self.d_lo = dt.date(*(int(x) for x in a.start.split("-")))
        self.d_hi = dt.date(*(int(x) for x in a.end.split("-")))
        self.b_lo = max(0, bin_index(self.d_lo, PENTAD_DAYS))
        self.b_hi = min(N_BINS - 1, bin_index(self.d_hi, PENTAD_DAYS))
        self.years = list(range(self.d_lo.year, self.d_hi.year + 1))
        self.n_proc = int(a.jobs) if a.jobs else min(MAX_PROCS,
                                                     os.cpu_count() or 1)
        self.dl_workers = int(getattr(a, "dl_workers", 0) or DL_WORKERS)
        self.max_inflight = int(getattr(a, "inflight", 0) or MAX_INFLIGHT)
        # ONE meter for the whole stage, not one per year: a projection reset
        # at every January would be recomputed from a cold start twenty-one
        # times and would never see the archive's own year-to-year growth.
        self.throughput = Throughput()
        self.prog = Progress(self.work)

    def days(self, year):
        d = max(self.d_lo, dt.date(year, 1, 1))
        end = min(self.d_hi, dt.date(year, 12, 31))
        out = []
        while d <= end:
            out.append(d)
            d += dt.timedelta(days=1)
        return out

    def tasks(self, year):
        """(basin, day) in (BASIN, DATE) order — the duplicate rule's order."""
        return [(b, d) for b in BASINS for d in self.days(year)]

    def local_day(self, basin, day):
        _, rel = day_urls(basin, day)
        return os.path.join(self.source_dir, rel) if self.source_dir else None

    def index_path(self):
        if self.source_dir:
            return os.path.join(self.source_dir, INDEX_NAME)
        return os.path.join(self.scratch, INDEX_NAME)


# ============================================================ stage: index ==
def civil_days(y, m, d):
    """Days since 1970-01-01, vectorised (Howard Hinnant's civil_from_days).

    3 M `datetime.date` objects cost seconds and a gigabyte; this is exact
    integer arithmetic on int64 arrays. `tests/test_build_family8_argo.py`
    pins it against `datetime` over a span of dates.
    """
    y = np.asarray(y, np.int64)
    m = np.asarray(m, np.int64)
    d = np.asarray(d, np.int64)
    yy = y - (m <= 2)
    era = np.where(yy >= 0, yy, yy - 399) // 400
    yoe = yy - era * 400
    doy = (153 * (m + np.where(m > 2, -3, 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


DAYS_1970_TO_EPOCH = (START - dt.date(1970, 1, 1)).days          # 4383


def yyyymmdd_to_bin(date_int):
    """int32 YYYYMMDD -> five-day bin index, the family-4..8 axis."""
    d = np.asarray(date_int, np.int64)
    days = civil_days(d // 10000, (d // 100) % 100, d % 100) - DAYS_1970_TO_EPOCH
    return np.floor_divide(days, PENTAD_DAYS)


def parse_index(path):
    """`ar_index_global_prof.txt.gz` -> (file, date, lat, lon, date_update).

    Format (verified 2026-09-07 against the live file): `#` comment lines, one
    header line `file,date,latitude,longitude,ocean,profiler_type,institution,
    date_update`, then one line per profile with `date` as YYYYMMDDHHMMSS.

    TWO SKIP COUNTERS, because they mean different things and one of them is
    not an error. MEASURED on the live index: 35,824 of 3,402,399 records
    (1.06 %) carry a date and an EMPTY latitude/longitude — a profile the GDAC
    holds but cannot place. Those are counted as `lines_no_position` and are a
    property of the archive; a single `bad_lines` number would have read as a
    parser that is losing 1 % of the data. `lines_malformed` is the other kind
    — a truncated or non-numeric record — and should be ~0.
    """
    files, dates, lats, lons = [], [], [], []
    no_pos = 0
    malformed = 0
    date_update = 0
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", errors="replace") as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            p = line.rstrip("\n").split(",")
            if p[0] == "file":
                continue
            if len(p) < 4:
                malformed += 1
                continue
            if not p[1].strip() or not p[2].strip() or not p[3].strip():
                no_pos += 1
                continue
            try:
                d = int(p[1][:8])
                la = float(p[2])
                lo = float(p[3])
            except ValueError:
                malformed += 1
                continue
            if d < 19000101 or not (-90.0 <= la <= 90.0):
                malformed += 1
                continue
            files.append(p[0])
            dates.append(d)
            lats.append(la)
            lons.append(lo)
            if len(p) >= 8:
                try:
                    date_update = max(date_update, int(p[7][:14]))
                except ValueError:
                    pass
    lon = np.asarray(lons, np.float64)
    return {
        "file": np.asarray(files, dtype="S"),
        "date": np.asarray(dates, np.int32),
        "lat": np.asarray(lats, np.float32),
        "lon": (np.mod(lon + 180.0, 360.0) - 180.0).astype(np.float32),
        "lines_no_position": no_pos,
        "lines_malformed": malformed,
        "date_update_max": date_update,
    }


def _nn_distances(alat, alon, clat, clon, self_row=None, block=256):
    """Sorted distances (km) from each anchor to every candidate, blocked.

    Returns an (n_anchor, n_cand) array is deliberately NOT what this does —
    at 2,000 x 12,000 that is 190 MB per sample and the caller only ever wants
    the first few order statistics. It yields the sorted head instead.
    """
    n_a = len(alat)
    keep = min(16, len(clat))
    out = np.full((n_a, keep), np.inf)
    for i in range(0, n_a, block):
        j = min(i + block, n_a)
        dy = (clat[None, :] - alat[i:j, None]) * fs.KM_PER_DEG
        dlon = np.mod(clon[None, :] - alon[i:j, None] + 180.0, 360.0) - 180.0
        coslat = np.maximum(np.cos(np.radians(
            0.5 * (clat[None, :] + alat[i:j, None]))), fs.COS_FLOOR)
        dx = dlon * fs.KM_PER_DEG * coslat
        d = np.sqrt(dx * dx + dy * dy)
        if self_row is not None:
            m = self_row[i:j, None] == np.arange(len(clat))[None, :]
            d[m] = np.inf
        d.sort(axis=1)
        out[i:j] = d[:, :keep]
    return out


ANCHOR_NOTE = (
    "Two anchor draws. `catalogue`: positions taken from the profile index "
    "itself — ocean by construction, but weighted by float density, so it "
    "answers 'how far is the k-th neighbour from a place Argo samples'. "
    "`sphere`: drawn uniformly BY AREA over the globe (lat = asin(U(-1,1))), "
    "which answers 'from an arbitrary point' but includes land, where the "
    "honest answer is that there is no measurement at all. Reporting one "
    "without the other would be a choice presented as a fact."
)


def index_stats(idx, prog=None, n_anchor=2000, n_pentads=12, seed=20260907,
                year_lo=2004, year_hi=2024):
    """The MEASUREMENT that replaces E-076 §2.3's Poisson estimate.

    §2.3 argued from ~2,000 profiles per pentad over 3.61e8 km^2 that the mean
    nearest neighbour is ~210 km in one pentad and ~87 km in a 30-day window,
    and that the fifth neighbour sits at ~195 km — one surface correlation
    length, which is the entire justification for k = 5. Here those three
    numbers are counted off the real index instead.

    TWO ANCHOR DRAWS, because "a random ocean anchor" is not one thing.
    `catalogue` anchors are drawn from the profile positions themselves: ocean
    by construction, but weighted by float density, so they answer "how far is
    the k-th neighbour from a place Argo actually samples". `sphere` anchors
    are drawn uniformly BY AREA over the globe (lat = asin(U(-1,1))), which
    answers "from an arbitrary point" but includes land, where the honest
    answer is that there is no measurement at all. Reporting one without the
    other would be a choice presented as a fact.
    """
    rng = np.random.default_rng(seed)
    date, lat, lon = idx["date"], idx["lat"], idx["lon"]
    year = (date // 10000).astype(np.int64)
    b = yyyymmdd_to_bin(date)
    out = {"n_profiles": int(len(date)),
            "date_min": int(date.min()) if len(date) else 0,
            "date_max": int(date.max()) if len(date) else 0,
            "lines_no_position": int(idx.get("lines_no_position", 0)),
            "lines_malformed": int(idx.get("lines_malformed", 0)),
            "date_update_max": int(idx.get("date_update_max", 0)),
            "window_years": [year_lo, year_hi],
            "anchor_note": ANCHOR_NOTE}
    yrs, cnts = np.unique(year, return_counts=True)
    out["profiles_per_year"] = {int(y): int(c) for y, c in zip(yrs, cnts)}

    inw = (year >= year_lo) & (year <= year_hi) & (b >= 0) & (b < N_BINS)
    bw = b[inw]
    if bw.size:
        per = np.bincount(bw.astype(np.int64), minlength=N_BINS)
        lo = int(bw.min())
        hi = int(bw.max())
        per = per[lo:hi + 1]
        out["profiles_per_pentad"] = {
            "bins": [lo, hi], "n_bins": int(len(per)),
            "min": int(per.min()), "median": float(np.median(per)),
            "mean": float(per.mean()), "max": int(per.max()),
            "empty_bins": int((per == 0).sum())}
    else:
        out["profiles_per_pentad"] = {"n_bins": 0}
        return out

    # -- nearest neighbour, in ONE pentad and in a 30-day (6-pentad) window --
    order = np.argsort(b, kind="stable")
    b_s, lat_s, lon_s = b[order], lat[order], lon[order]
    edges = np.searchsorted(b_s, np.arange(N_BINS + 1), side="left")
    cand = np.unique(bw)
    picks = cand[np.linspace(0, len(cand) - 1, min(n_pentads, len(cand))
                             ).astype(int)] if len(cand) else []
    nn1, nn30, sizes = [], [], []
    for pb in picks:
        i0, i1 = int(edges[pb]), int(edges[pb + 1])
        if i1 - i0 < 2:
            continue
        a_lat, a_lon = lat_s[i0:i1], lon_s[i0:i1]
        d1 = _nn_distances(a_lat, a_lon, a_lat, a_lon,
                           self_row=np.arange(i1 - i0))
        nn1.append(float(np.mean(d1[:, 0])))
        w0 = int(edges[max(0, pb - 5)])
        d30 = _nn_distances(a_lat, a_lon, lat_s[w0:i1], lon_s[w0:i1],
                            self_row=np.arange(i0 - w0, i1 - w0))
        nn30.append(float(np.mean(d30[:, 0])))
        sizes.append([int(pb), i1 - i0, i1 - w0])
        if prog:
            prog.item(f"pentad {int(pb)}", extra={"nn1_km": nn1[-1],
                                                  "nn30_km": nn30[-1]})
    out["nearest_neighbour_km"] = {
        "sampled_pentads": sizes,
        "one_pentad_mean": float(np.mean(nn1)) if nn1 else None,
        "thirty_day_mean": float(np.mean(nn30)) if nn30 else None,
        "poisson_estimate_e076_2_3": {"one_pentad": 210.0, "thirty_day": 87.0},
    }

    # -- the 3rd / 5th / 7th neighbour at random anchors ---------------------
    out["kth_neighbour_km"] = {}
    pool = np.nonzero(inw)[0]
    for draw in ("catalogue", "sphere"):
        if len(pool) < 8:
            continue
        take = rng.choice(pool, size=min(n_anchor, len(pool)), replace=False)
        if draw == "catalogue":
            a_lat, a_lon, a_bin = lat[take], lon[take], b[take]
        else:
            a_lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, len(take))))
            a_lon = rng.uniform(-180.0, 180.0, len(take))
            a_bin = b[take]
        res = {"3": [], "5": [], "7": [], "n_R": []}
        for i in range(len(take)):
            pb = int(a_bin[i])
            w0 = int(edges[max(0, pb - 5)])
            w1 = int(edges[pb + 1])
            if w1 - w0 < 2:
                continue
            self_row = None
            if draw == "catalogue":
                # exclude the anchor's OWN profile, which sits at 0 km
                pos = np.nonzero(order[w0:w1] == take[i])[0]
                self_row = np.array([pos[0] if len(pos) else -1])
            d = _nn_distances(a_lat[i:i + 1], a_lon[i:i + 1],
                              lat_s[w0:w1], lon_s[w0:w1], self_row=self_row)[0]
            fin = d[np.isfinite(d)]
            for kk in (3, 5, 7):
                if len(fin) >= kk:
                    res[str(kk)].append(float(fin[kk - 1]))
            res["n_R"].append(int(np.isfinite(d).sum()))
        out["kth_neighbour_km"][draw] = {
            "n_anchors": len(take),
            **{f"k{kk}_mean": (float(np.mean(res[str(kk)]))
                               if res[str(kk)] else None) for kk in (3, 5, 7)},
            **{f"k{kk}_median": (float(np.median(res[str(kk)]))
                                 if res[str(kk)] else None) for kk in (3, 5, 7)},
            "window_days": 30,
            "poisson_estimate_e076_2_3_k5": 195.0,
        }
    return out


def stage_index(ctx):
    """Download / read the global profile index, parse it, MEASURE it."""
    work = ctx.work
    ctx.prog.stage_start("index", 3)
    path = ctx.index_path()
    if ctx.source_dir:
        if not os.path.exists(path):
            sys.exit(f"--source-dir given but {path} is absent — the index "
                     f"stage has nothing to read")
    else:
        url = f"{GDAC}/{INDEX_NAME}"
        download_verified(url, path, mirrors=(f"{GDAC_MIRROR}/{INDEX_NAME}",))
    ctx.prog.item(os.path.basename(path), 1,
                  {"bytes": os.path.getsize(path)})

    idx = parse_index(path)
    npz = os.path.join(work, "index.npz")
    tmp = f"{npz}.tmp{os.getpid()}.npz"
    # `profile_file`, not `file`: np.savez's own first parameter is named
    # `file`, so a column of that name silently becomes the OUTPUT PATH and
    # the call fails with "multiple values for argument 'file'".
    np.savez_compressed(tmp, profile_file=idx["file"], date=idx["date"],
                        lat=idx["lat"], lon=idx["lon"],
                        lines_no_position=np.array(idx["lines_no_position"]),
                        lines_malformed=np.array(idx["lines_malformed"]),
                        date_update_max=np.array(idx["date_update_max"]))
    os.replace(tmp, npz)
    ctx.prog.item("index.npz", 2,
                  {"profiles": int(len(idx["date"])),
                   "lines_no_position": int(idx["lines_no_position"]),
                   "lines_malformed": int(idx["lines_malformed"])})

    stats = index_stats(idx, prog=ctx.prog, n_anchor=ctx.a.index_anchors,
                        n_pentads=ctx.a.index_pentads)
    stats["source"] = (f"{GDAC}/{INDEX_NAME}" if not ctx.source_dir
                       else f"file://{path}")
    stats["builder_git_sha"] = git_sha()
    stats["built_at"] = utcnow()
    atomic_json(os.path.join(work, "index_stats.json"), stats)
    ctx.prog.item("index_stats.json", 3)
    mark(work, "index")
    nn = stats.get("nearest_neighbour_km", {})
    print(f"  index: {len(idx['date']):,} placed profiles, "
          f"{idx['lines_no_position']:,} with no position, "
          f"{idx['lines_malformed']:,} malformed; nearest neighbour "
          f"{nn.get('one_pentad_mean')} km in one pentad, "
          f"{nn.get('thirty_day_mean')} km in 30 days "
          f"(E-076 §2.3 estimated 210 / 87)")
    return stats


# ========================================================= stage: profiles ==
def _year_npz(work, year):
    return os.path.join(work, "profiles", f"{year}.npz")


def _fetch_task(ctx, task):
    """(basin, day) -> (path, owned, nbytes) or None for a missing day.

    The scratch path mirrors the ARCHIVE's full relative path, basin included.
    The three basins publish one file per day each and all three are named
    `YYYYMMDD_prof.nc`, so a destination built from the basename alone has
    every basin writing and deleting one shared file — up to eight of them at
    once, since the pipeline runs the downloads concurrently.
    """
    basin, day = task
    if ctx.source_dir:
        p = ctx.local_day(basin, day)
        return (p, False, os.path.getsize(p)) if os.path.exists(p) else None
    urls, rel = day_urls(basin, day)
    dest = os.path.join(ctx.scratch, rel)
    got = fetch_first(urls, dest, attempts=ctx.a.attempts)
    return (got, True, os.path.getsize(got)) if got else None


def _parse_task(task, path):
    basin, day = task
    return extract_day(path, basin, day.isoformat())


def extract_year(ctx, year):
    """Every (basin, day) of one year -> `<work>/profiles/<year>.npz`.

    The npz is written and only THEN is `profiles/<year>.done` marked, so a
    killed job replays exactly the year it lost (ml/CLAUDE.md §5.21).
    """
    work = ctx.work
    os.makedirs(os.path.join(work, "profiles"), exist_ok=True)
    tasks = ctx.tasks(year)
    cols = {k: [] for k in ("bin", "time_days", "lat", "lon", "temp", "psal",
                            "wmo", "cycle", "nlev", "maxpres", "mode", "key")}
    counts = {"n_prof": 0, "kept": 0}
    counts.update({f"drop_{r}": 0 for r in DROP_REASONS})
    hist_t = np.zeros(NLEV + 1, np.int64)
    hist_s = np.zeros(NLEV + 1, np.int64)
    missing, n_files = [], 0
    t0 = time.time()
    done = 0
    tp = ctx.throughput
    last = tp.report()
    for task, res, nbytes in stream_days(tasks,
                                         lambda t: _fetch_task(ctx, t),
                                         _parse_task, n_proc=ctx.n_proc,
                                         max_inflight=ctx.max_inflight,
                                         dl_workers=ctx.dl_workers):
        done += 1
        basin, day = task
        if res is None:
            missing.append(f"{basin}/{day.isoformat()}")
        else:
            n_files += 1
            for k in cols:
                cols[k].append(res[k])
            for k, v in res["counts"].items():
                counts[k] = counts.get(k, 0) + int(v)
            hist_t += res["hist_t"]
            hist_s += res["hist_s"]
        # THE SELF-REPORT. Every 50 files the rate and the projection go into
        # progress.json (§5.25 — progress is an artefact, not a log line) and
        # into the log, so a download-bound job says so in its first minutes
        # instead of in hour twenty.
        due = tp.add(nbytes, res is not None)
        if due:
            last = tp.report()
            print(f"  throughput: {last['files_per_s']} files/s · "
                  f"{last['mb_per_s']} MB/s · mean file "
                  f"{last['mean_file_mb']} MB · projected full pull "
                  f"{last['projected_full_pull_h']} h", flush=True)
            tp.check(last)
        if due or done % 25 == 0 or done == len(tasks):
            ctx.prog.item(f"{year} {basin} {day.isoformat()}", done,
                          {"kept": counts["kept"], "files": n_files,
                           "missing": len(missing), "throughput": last})
    joined = {k: (np.concatenate(v) if v else None) for k, v in cols.items()}
    if joined["bin"] is None:
        joined = {k: np.zeros(0, d) for k, d in
                  (("bin", np.int16), ("time_days", np.float32),
                   ("lat", np.float32), ("lon", np.float32),
                   ("wmo", np.int32), ("cycle", np.int16),
                   ("nlev", np.int16), ("maxpres", np.float16),
                   ("key", np.int64))}
        joined["temp"] = np.zeros((0, NLEV), np.float16)
        joined["psal"] = np.zeros((0, NLEV), np.float16)
        joined["mode"] = np.zeros(0, "|S1")
    out = _year_npz(work, year)
    tmp = f"{out}.tmp{os.getpid()}.npz"
    np.savez(tmp, hist_t=hist_t, hist_s=hist_s,
             counts=np.array(json.dumps(counts, sort_keys=True)),
             missing=np.asarray(missing, dtype="S"),
             n_files=np.array(n_files), n_tasks=np.array(len(tasks)),
             **joined)
    os.replace(tmp, out)
    mark(work, f"profiles/{year}")                # AFTER the data it describes
    print(f"  {year}: {counts['kept']:,} profiles kept of {counts['n_prof']:,} "
          f"in {n_files} file(s); {len(missing)} day(s) absent "
          f"({time.time() - t0:.1f}s)")
    return out


def assemble_store(ctx):
    """Concatenate the years, dedupe, sort by (bin, time), write the store."""
    work = ctx.work
    dest = ctx.store
    os.makedirs(dest, exist_ok=True)
    keys = ("bin", "time_days", "lat", "lon", "temp", "psal", "wmo", "cycle",
            "nlev", "maxpres", "mode", "key")
    parts = {k: [] for k in keys}
    per_year, missing_all, hist_t, hist_s = {}, [], np.zeros(NLEV + 1, np.int64), \
        np.zeros(NLEV + 1, np.int64)
    counts_all = {}
    for y in ctx.years:
        p = _year_npz(work, y)
        if not os.path.exists(p):
            sys.exit(f"{p} is missing — run the profiles stage for {y} first")
        d = np.load(p, allow_pickle=False)
        for k in keys:
            parts[k].append(d[k])
        c = json.loads(str(d["counts"]))
        per_year[y] = {"kept": int(c.get("kept", 0)),
                       "n_prof": int(c.get("n_prof", 0)),
                       "files": int(d["n_files"]),
                       "days_missing": int(len(d["missing"]))}
        for k, v in c.items():
            counts_all[k] = counts_all.get(k, 0) + int(v)
        hist_t += d["hist_t"]
        hist_s += d["hist_s"]
        missing_all.extend([s.decode() for s in d["missing"]])

    cat = {k: np.concatenate(v) if v else np.zeros(0) for k, v in parts.items()}
    n_raw = len(cat["bin"])
    # DEDUPE first, in (basin, date) order — np.unique with return_index gives
    # the FIRST occurrence, which is exactly the rule.
    _, first = np.unique(cat["key"], return_index=True)
    first = np.sort(first)
    n_dupe = n_raw - len(first)
    cat = {k: v[first] for k, v in cat.items()}
    # THEN sort by (bin, time) ascending — the store's defining order.
    order = np.lexsort((cat["time_days"], cat["bin"]))
    cat = {k: v[order] for k, v in cat.items()}
    N = len(cat["bin"])

    off = np.searchsorted(cat["bin"].astype(np.int64),
                          np.arange(N_BINS + 1, dtype=np.int64), side="left")
    off = off.astype(np.int64)
    assert off[0] == 0 and off[-1] == N, (off[0], off[-1], N)

    files = {}
    for k in fs.COLUMNS:
        p = os.path.join(dest, k + ".npy")
        np.save(p, cat[k])
        files[k + ".npy"] = p
    p = os.path.join(dest, "bin_offsets.npy")
    np.save(p, off)
    files["bin_offsets.npy"] = p

    live = int((np.diff(off) > 0).sum())
    meta = {
        "schema": {k: {"dtype": v,
                       "shape": ([int(N), NLEV] if k in fs.VALUE_COLUMNS
                                 else [int(N)])}
                   for k, v in fs.COLUMNS.items()},
        "index": {"bin_offsets.npy": {"dtype": "int64",
                                      "shape": [N_BINS + 1],
                                      "meaning": "CSR: rows of bin b are "
                                                 "[off[b], off[b+1])"}},
        "recipe": RECIPE, "stem": STEM,
        "N": int(N), "n_bins": N_BINS, "n_live_bins": live,
        "levels": [float(x) for x in LEVELS],
        "level_tol_dbar": [float(x) for x in TOL],
        "shallow_nearest_dbar": SHALLOW_NEAREST_DBAR,
        "units": {"temp": "degC", "psal": "PSU", "maxpres": "dbar",
                  "time_days": "days since 1982-01-01 (fractional)",
                  "lon": "degrees east in [-180, 180)"},
        "normalisation": "RAW — not z-scored and not anomalised; E-076 §2.5 "
                         "leaves both to the sampler",
        "footprint": {"log2_fp": fs.LOG2_FP_ARGO, "log2_dt": fs.LOG2_DT_ARGO,
                      "note": "E-076 §2.6 — a point profile taken in minutes; "
                              "both clamp to the -4 floor"},
        "qc_policy": QC_POLICY,
        "epoch": str(START), "pentad_days": PENTAD_DAYS,
        "date_range": [str(ctx.d_lo), str(ctx.d_hi)],
        "bins": [ctx.b_lo, ctx.b_hi],
        "per_year": per_year,
        "counts": counts_all,
        "filled_levels_hist_t": [int(x) for x in hist_t],
        "filled_levels_hist_s": [int(x) for x in hist_s],
        "duplicates_dropped": int(n_dupe),
        "rows_before_dedupe": int(n_raw),
        "days_missing": len(missing_all),
        "days_missing_list": sorted(missing_all)[:500],
        "days_missing_list_truncated": len(missing_all) > 500,
        "source": (f"{GDAC}/geo/<basin>/YYYY/MM/YYYYMMDD_prof.nc"
                   if not ctx.source_dir else f"file://{ctx.source_dir}"),
        "source_mirror": f"{GDAC_MIRROR}/geo/...",
        "index_source": f"{GDAC}/{INDEX_NAME}",
        "basins": list(BASINS),
        "throughput": ctx.throughput.report(),
        "builder_git_sha": git_sha(),
        "built_at": utcnow(),
    }
    stats = read_json(os.path.join(work, "index_stats.json"), {})
    meta["index_date_update_max"] = stats.get("date_update_max")
    meta["index_profiles"] = stats.get("n_profiles")
    meta["sha256"] = {n: sha256(p) for n, p in sorted(files.items())}
    atomic_json(os.path.join(dest, "store.json"), meta)
    print(f"  store: {N:,} profiles ({n_dupe:,} duplicate(s) dropped) in "
          f"{live:,} live bin(s) -> {dest}")
    return meta


def stage_profiles(ctx):
    """Stream every (basin, day), extract, then assemble the store."""
    work = ctx.work
    # Refuse before anything is streamed (ml/CLAUDE.md §5.18): the store is
    # small, but the pipeline needs room for <= 8 daily files, the per-year
    # extracts and the store itself, and over 90 % used is unusable rather
    # than a warning — a full box computes and reports nothing.
    f7.disk_guard(work, {"store": 3e8, "year extracts": 3e8,
                         "sources in flight": 2e8}, headroom=2e9)
    ctx.prog.stage_start("profiles", len(ctx.years))
    for y in ctx.years:
        if marked(work, f"profiles/{y}") and not ctx.a.force:
            print(f"  {y}: already extracted — skipping")
            continue
        ctx.prog.stage_start(f"profiles {y}", len(ctx.tasks(y)))
        extract_year(ctx, y)
    rep = ctx.throughput.report()
    if rep["files"]:
        print(f"  streamed {rep['files']} file(s), {rep['mb']} MB in "
              f"{rep['elapsed_s']}s — {rep['mb_per_s']} MB/s, "
              f"{rep['files_per_s']} files/s, projected full pull "
              f"{rep['projected_full_pull_h']} h")
        ctx.throughput.check(rep)
    ctx.prog.stage_start("profiles store", 1)
    meta = assemble_store(ctx)
    mark(work, "profiles")                        # AFTER the store is on disk
    ctx.prog.item("store", 1, {"N": meta["N"],
                               "duplicates_dropped": meta["duplicates_dropped"],
                               "days_missing": meta["days_missing"],
                               "throughput": rep})
    return meta


# ========================================================== stage: publish ==
def stage_publish(ctx):
    """Upload the store, DOWNLOAD EACH FILE BACK and compare sha256.

    Family 7's `stage_publish`, file for file: an upload that returns 200 is
    not evidence the bytes are retrievable (ml/CLAUDE.md §0.2), so a publish
    that cannot verify FAILS the job.
    """
    from huggingface_hub import hf_hub_download
    work = ctx.work
    dest = ctx.store
    api, repo, tok = hub_repo()
    names = sorted([n for n in os.listdir(dest) if n.endswith(".npy")]) \
        + ["store.json"]
    for n in names:
        if not os.path.exists(os.path.join(dest, n)):
            sys.exit(f"cannot publish: {os.path.join(dest, n)} is missing")
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=False)
    ctx.prog.stage_start("publish", len(names))
    entries = []
    scratch = os.path.join(ctx.scratch, "verify")
    for i, n in enumerate(names, 1):
        p = os.path.join(dest, n)
        src = sha256(p)
        api.upload_file(path_or_fileobj=p, path_in_repo=f"{HF_PREFIX}/{n}",
                        repo_id=repo, repo_type="dataset",
                        commit_message=f"family 8 ({RECIPE}): {n}")
        shutil.rmtree(scratch, ignore_errors=True)
        back = hf_hub_download(repo, f"{HF_PREFIX}/{n}", repo_type="dataset",
                               token=tok, local_dir=scratch)
        got = sha256(back)
        shutil.rmtree(scratch, ignore_errors=True)
        if got != src:
            sys.exit(f"RESTORE MISMATCH {n}: uploaded {src}, downloaded {got} "
                     f"— the publish is not trustworthy")
        entries.append({"name": n, "bytes": os.path.getsize(p), "sha256": src})
        ctx.prog.item(n, i, {"sha256": src[:16]})

    store_meta = read_json(os.path.join(dest, "store.json"), {})
    man = {"recipe": RECIPE, "stem": STEM, "repo": repo, "prefix": HF_PREFIX,
           "N": store_meta.get("N"), "levels": store_meta.get("levels"),
           "date_range": store_meta.get("date_range"),
           "builder_git_sha": git_sha(), "built_at": utcnow(),
           "files": entries}
    mp = os.path.join(work, "manifest.json")
    atomic_json(mp, man)
    api.upload_file(path_or_fileobj=mp,
                    path_in_repo=f"{HF_PREFIX}/manifest.json",
                    repo_id=repo, repo_type="dataset",
                    commit_message=f"family 8 ({RECIPE}): manifest")
    mark(work, "publish")
    print(f"  publish: {len(entries)} file(s) verified by restore -> "
          f"https://huggingface.co/datasets/{repo}/tree/main/{HF_PREFIX}")


# =================================================================== driver ==
STAGE_FN = {"index": stage_index, "profiles": stage_profiles,
            "publish": stage_publish}


def run_stages(ctx, stages):
    work = ctx.work
    for s in stages:
        for dep in DEPS.get(s, []):
            if not marked(work, dep):
                sys.exit(f"stage {s!r} needs {dep!r} first (E-076 §3: stage "
                         f"order is fixed) — {marker(work, dep)} is missing")
        if marked(work, s) and not ctx.a.force:
            print(f"stage {s}: already done — skipping (--force to redo)")
            continue
        t0 = time.time()
        print(f"\n=== stage {s} ===", flush=True)
        STAGE_FN[s](ctx)
        print(f"=== stage {s} done in {time.time() - t0:.1f}s ===", flush=True)


# ==================================================================== smoke ==
SMOKE_START = "2014-12-27"
SMOKE_END = "2015-01-07"          # 12 days, across a bin AND a year boundary
SMOKE_BASINS = ("atlantic_ocean", "pacific_ocean")


def write_prof_nc(path, profs, n_levels=None):
    """A GDAC-style daily `*_prof.nc`, with the REAL names, dims and fills.

    Used by `--smoke` and by `tests/test_build_family8_argo.py`, so the
    synthetic file the extractor is tested against has exactly the shape of
    the file it meets on the box: N_PROF x N_LEVELS, PRES/TEMP/PSAL with their
    `_ADJUSTED` twins and `_QC` flags, DATA_MODE, JULD in days since
    1950-01-01, LATITUDE/LONGITUDE, POSITION_QC, JULD_QC, PLATFORM_NUMBER as
    (N_PROF, 8) of |S1, CYCLE_NUMBER and DIRECTION.

    A profile dict carries: wmo, cycle, direction, mode, juld, lat, lon,
    position_qc, juld_qc, pres, temp, psal and optional per-sample qc arrays
    (`pres_qc`, `temp_qc`, `psal_qc`, default b"1") and optional
    `pres_adjusted` / `temp_adjusted` / `psal_adjusted` (default: the raw
    arrays, so a D-mode profile has adjusted values to read).
    """
    import netCDF4 as ncdf
    n_prof = len(profs)
    nl = n_levels or max(1, max(len(p["pres"]) for p in profs))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    d = ncdf.Dataset(path, "w", format="NETCDF4")
    d.createDimension("N_PROF", n_prof)
    d.createDimension("N_LEVELS", nl)
    d.createDimension("STRING8", 8)
    d.createDimension("DATE_TIME", 14)

    def s1(name, dims, values, fill=b" "):
        v = d.createVariable(name, "S1", dims, fill_value=fill)
        v[:] = values
        return v

    pn = np.full((n_prof, 8), b" ", "S1")
    for i, p in enumerate(profs):
        s = f"{int(p['wmo']):>8d}".encode()
        pn[i] = np.frombuffer(s, "S1")
    s1("PLATFORM_NUMBER", ("N_PROF", "STRING8"), pn)
    cn = d.createVariable("CYCLE_NUMBER", "i4", ("N_PROF",), fill_value=99999)
    cn[:] = np.asarray([int(p["cycle"]) for p in profs], np.int32)
    s1("DIRECTION", ("N_PROF",),
       np.asarray([p.get("direction", b"A") for p in profs], "S1"))
    s1("DATA_MODE", ("N_PROF",),
       np.asarray([p.get("mode", b"D") for p in profs], "S1"))
    jv = d.createVariable("JULD", "f8", ("N_PROF",), fill_value=FILL_JULD)
    jv.units = "days since 1950-01-01 00:00:00 UTC"
    jv[:] = np.asarray([p["juld"] for p in profs], np.float64)
    s1("JULD_QC", ("N_PROF",),
       np.asarray([p.get("juld_qc", b"1") for p in profs], "S1"))
    for nm, key in (("LATITUDE", "lat"), ("LONGITUDE", "lon")):
        v = d.createVariable(nm, "f8", ("N_PROF",), fill_value=FILL_F)
        v[:] = np.asarray([p[key] for p in profs], np.float64)
    s1("POSITION_QC", ("N_PROF",),
       np.asarray([p.get("position_qc", b"1") for p in profs], "S1"))

    for base in ("PRES", "TEMP", "PSAL"):
        key = base.lower()
        for suffix in ("", "_ADJUSTED"):
            arr = np.full((n_prof, nl), FILL_F, np.float32)
            qc = np.full((n_prof, nl), b" ", "S1")
            for i, p in enumerate(profs):
                src = (p.get(key + "_adjusted", p[key]) if suffix
                       else p[key])
                src = np.asarray(src, np.float32)
                arr[i, :len(src)] = src
                q = p.get(key + "_qc", None)
                q = (np.full(len(src), b"1", "S1") if q is None
                     else np.asarray(q, "S1"))
                qc[i, :len(src)] = q
            v = d.createVariable(base + suffix, "f4",
                                 ("N_PROF", "N_LEVELS"), fill_value=FILL_F)
            v.units = {"PRES": "decibar", "TEMP": "degree_Celsius",
                       "PSAL": "psu"}[base]
            v[:] = arr
            s1(base + suffix + "_QC", ("N_PROF", "N_LEVELS"), qc)
    d.title = "Argo float vertical profile (synthetic, earth/family8 smoke)"
    d.close()
    return path


def _smoke_profile(wmo, cycle, day_frac, lat, lon, mode=b"D", bad=None):
    """One synthetic profile with a LINEAR T(p) and S(p) — exact to interpolate.

    Samples every 20 dbar from 5 to 2000, so every level is bracketed within
    its tolerance (25 dbar shallow, 50, 100) and a linear profile must come
    back EXACTLY at all sixteen levels. That exactness is the point: it turns
    the interpolation test into an identity rather than a tolerance.
    """
    pres = np.arange(5.0, 2001.0, 20.0)
    t0, dt_dp = 20.0 + (lat % 7), -0.008
    s0, ds_dp = 35.0 + 0.001 * (lon % 5), 0.0004
    temp = t0 + dt_dp * pres
    psal = s0 + ds_dp * pres
    p = {"wmo": wmo, "cycle": cycle, "direction": b"A", "mode": mode,
         "juld": day_frac, "lat": lat, "lon": lon, "pres": pres,
         "temp": temp, "psal": psal,
         "truth_t": t0 + dt_dp * LEVELS_ARR, "truth_s": s0 + ds_dp * LEVELS_ARR}
    if bad == "position":
        p["position_qc"] = b"4"
    elif bad == "juld":
        p["juld_qc"] = b"4"
    elif bad == "flag4":
        # one flag-4 TEMPERATURE sample: the level it would have bracketed
        # must go NaN, and salinity at the same level must be untouched.
        q = np.full(len(pres), b"1", "S1")
        q[pres == 285.0] = b"4"
        q[pres == 305.0] = b"4"
        p["temp_qc"] = q
        p["truth_t"][LEVELS_ARR == 300.0] = np.nan
    return p


def make_smoke_sources(root, d_lo, d_hi, n_per_day=15, seed=20260907):
    """Two basins x 12 days x ~15 profiles, plus the index text. Returns TRUTH."""
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    days = [d_lo + dt.timedelta(days=k) for k in range((d_hi - d_lo).days + 1)]
    truth, lines = {}, []
    wmo0 = 3900000
    for bi, basin in enumerate(SMOKE_BASINS):
        for di, day in enumerate(days):
            profs = []
            for j in range(n_per_day):
                wmo = wmo0 + bi * 1000 + j
                cycle = 10 + di
                frac = (day - JULD_EPOCH).days + 0.1 + 0.05 * j
                lat = float(rng.uniform(-55, 60))
                lon = float(rng.uniform(-179, 179) if bi == 0
                            else rng.uniform(120, 179))
                mode = b"R" if j == 1 else (b"A" if j == 2 else b"D")
                bad = None
                if j == 3:
                    bad = "position"
                elif j == 4:
                    bad = "juld"
                elif j == 5:
                    bad = "flag4"
                p = _smoke_profile(wmo, cycle, frac, lat, lon, mode, bad)
                if mode == b"R":
                    # an R-mode profile must be read from the RAW variables:
                    # its adjusted twin is deliberately WRONG, so a builder
                    # that read the adjusted values would fail the check.
                    p["temp_adjusted"] = p["temp"] + 100.0
                    p["psal_adjusted"] = p["psal"] + 100.0
                profs.append(p)
                lines.append(f"{basin[:2]}/{wmo}/profiles/D{wmo}_{cycle:03d}.nc,"
                             f"{day:%Y%m%d}120000,{lat:.3f},{lon:.3f},A,846,"
                             f"AO,20260101000000")
                # A profile the policy KEEPS goes into the truth; the two
                # whose metadata QC is 4 must not be in the store at all.
                if bad in (None, "flag4"):
                    key = int(dedupe_key(np.array([wmo]), np.array([cycle]),
                                         np.array([b"A"]))[0])
                    tt = p["truth_t"].copy()
                    truth[key] = {
                        "wmo": wmo, "cycle": cycle, "mode": mode,
                        "lat": lat, "lon": lon,
                        "time_days": frac - JULD_OFFSET,
                        "bin": int(np.floor((frac - JULD_OFFSET) / PENTAD_DAYS)),
                        "temp": tt, "psal": p["truth_s"].copy()}
            write_prof_nc(os.path.join(root, "geo", basin, f"{day.year:04d}",
                                       f"{day.month:02d}",
                                       f"{day:%Y%m%d}_prof.nc"), profs)
    # A DUPLICATE across the basin boundary: the same (wmo, cycle, direction)
    # in the pacific file of the last day as in the atlantic one. Only the
    # first, in (basin, date) order, may survive.
    dup_day = days[-1]
    dup = _smoke_profile(wmo0, 10 + len(days) - 1, (dup_day - JULD_EPOCH).days + 0.1,
                         12.0, 150.0, b"D")
    pac = os.path.join(root, "geo", SMOKE_BASINS[1], f"{dup_day.year:04d}",
                       f"{dup_day.month:02d}", f"{dup_day:%Y%m%d}_prof.nc")
    existing = _read_prof_nc(pac)
    write_prof_nc(pac, existing + [dup])

    hdr = ("# Title : Profile directory file of the Argo GDAC (synthetic)\n"
           "# Date of update : 20260907000000\n"
           "file,date,latitude,longitude,ocean,profiler_type,institution,"
           "date_update\n")
    with gzip.open(os.path.join(root, INDEX_NAME), "wt") as fh:
        fh.write(hdr)
        fh.write("\n".join(lines) + "\n")
    return truth


def _read_prof_nc(path):
    """Read a synthetic file back into the dict form `write_prof_nc` takes."""
    import netCDF4 as ncdf
    d = ncdf.Dataset(path)
    d.set_auto_mask(False)
    n = d.dimensions["N_PROF"].size
    pn = _chars(d.variables["PLATFORM_NUMBER"][:])
    out = []
    for i in range(n):
        pres = np.asarray(d.variables["PRES"][i], np.float64)
        ok = pres != FILL_F
        out.append({
            "wmo": int(b"".join(bytes(c) for c in pn[i]).strip()),
            "cycle": int(d.variables["CYCLE_NUMBER"][i]),
            "direction": bytes(d.variables["DIRECTION"][i]),
            "mode": bytes(d.variables["DATA_MODE"][i]),
            "juld": float(d.variables["JULD"][i]),
            "lat": float(d.variables["LATITUDE"][i]),
            "lon": float(d.variables["LONGITUDE"][i]),
            "position_qc": bytes(d.variables["POSITION_QC"][i]),
            "juld_qc": bytes(d.variables["JULD_QC"][i]),
            "pres": pres[ok],
            "temp": np.asarray(d.variables["TEMP"][i], np.float64)[ok],
            "psal": np.asarray(d.variables["PSAL"][i], np.float64)[ok],
            "pres_adjusted": np.asarray(d.variables["PRES_ADJUSTED"][i],
                                        np.float64)[ok],
            "temp_adjusted": np.asarray(d.variables["TEMP_ADJUSTED"][i],
                                        np.float64)[ok],
            "psal_adjusted": np.asarray(d.variables["PSAL_ADJUSTED"][i],
                                        np.float64)[ok],
            "pres_qc": _chars(d.variables["PRES_QC"][i])[ok],
            "temp_qc": _chars(d.variables["TEMP_QC"][i])[ok],
            "psal_qc": _chars(d.variables["PSAL_QC"][i])[ok],
        })
    d.close()
    return out


def run_smoke(root=None, keep=False, start=SMOKE_START, end=SMOKE_END,
              jobs=1):
    """Generate the synthetic sources, run index + profiles, CHECK the store."""
    tmp = root or tempfile.mkdtemp(prefix="f8smoke_")
    src = os.path.join(tmp, "src")
    work = os.path.join(tmp, "work")
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    d_lo = dt.date(*(int(x) for x in start.split("-")))
    d_hi = dt.date(*(int(x) for x in end.split("-")))
    print(f"smoke     sources -> {src}")
    truth = make_smoke_sources(src, d_lo, d_hi)
    print(f"smoke     {len(truth)} truth profile(s) in "
          f"{time.time() - t0:.1f}s")

    ap = argparse.Namespace(work=work, source_dir=src, start=start, end=end,
                            force=False, stage="all", smoke=True, jobs=jobs,
                            attempts=1, index_anchors=200, index_pentads=3,
                            dl_workers=2, inflight=4)
    ctx = Ctx(ap)
    print(f"axis      {bin_start(ctx.b_lo, 5)} .. {bin_start(ctx.b_hi, 5)}  "
          f"bins {ctx.b_lo}..{ctx.b_hi} (recipe {RECIPE})")
    run_stages(ctx, ["index", "profiles"])
    out = check_smoke(work, truth)
    print(f"\nsmoke     OK in {time.time() - t0:.1f}s — {out}")
    if not keep and root is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return work, truth


def check_smoke(work, truth):
    """The store against the truth the generator kept — values and order."""
    st = fs.ArgoStore(os.path.join(work, STEM))
    assert st.N == len(truth), (
        f"store holds {st.N} profiles, the generator kept {len(truth)} "
        f"(a duplicate that survived, or a good profile that was dropped)")
    b = np.asarray(st["bin"])
    t = np.asarray(st["time_days"])
    assert np.all(np.diff(b) >= 0), "the store is not sorted by bin"
    for i in range(st.N - 1):
        if b[i] == b[i + 1]:
            assert t[i] <= t[i + 1], f"rows {i},{i+1} are out of time order"
    off = st.bin_offsets
    assert off[0] == 0 and off[-1] == st.N
    assert np.all(np.diff(off) >= 0), "bin_offsets is not monotone"
    for bb in np.unique(b):
        lo, hi = int(off[bb]), int(off[bb + 1])
        assert np.all(b[lo:hi] == bb), f"bin {bb} CSR slice holds other bins"

    key = dedupe_key(np.asarray(st["wmo"]), np.asarray(st["cycle"]),
                     np.full(st.N, b"A"))
    seen = set()
    for i, k in enumerate(key):
        k = int(k)
        assert k in truth, f"row {i} (wmo {st['wmo'][i]}) is not in the truth"
        assert k not in seen, f"duplicate row {i}"
        seen.add(k)
        tr = truth[k]
        assert int(b[i]) == tr["bin"], (int(b[i]), tr["bin"])
        assert abs(float(t[i]) - tr["time_days"]) < 1e-3
        assert abs(float(st["lat"][i]) - tr["lat"]) < 1e-3
        assert abs(float(st["lon"][i]) - tr["lon"]) < 1e-3
        for col, want in (("temp", tr["temp"]), ("psal", tr["psal"])):
            got = np.asarray(st[col][i], np.float32)
            fin = np.isfinite(want)
            assert np.array_equal(np.isfinite(got), fin), (
                f"row {i} {col}: filled levels {np.isfinite(got)} != {fin}")
            assert np.allclose(got[fin], want[fin], atol=2e-2), (
                f"row {i} {col}: {got[fin]} != {want[fin]}")
    # and the reader answers with the profiles the store holds. R_max is 50 km
    # so the only observation the search can find is the anchor's own
    # profile — the synthetic floats are scattered over the whole globe.
    i = st.N // 2
    tok = st.knearest(float(st["lat"][i]), float(st["lon"][i]),
                      int(b[i]), k=3, R_max_km=50.0, T_max_days=30.0)
    assert tok["n_R"] >= 1 and tok["valid"][0]
    assert int(tok["row"][0]) == i, (int(tok["row"][0]), i)
    assert tok["dist_km"][0] < 1e-3, tok["dist_km"][0]
    assert np.all(tok["dt_days"][tok["valid"]] >= 0.0)
    assert np.all(b[tok["row"][tok["valid"]]] <= b[i])
    return (f"N={st.N} · bins {int(b.min())}..{int(b.max())} · "
            f"{int((np.diff(off) > 0).sum())} live bin(s)")


# ===================================================================== main ==
def main():
    ap = argparse.ArgumentParser(
        description="Build the family-8 Argo observation store (recipe "
                    "f8argo_l0). See ml/plans/E076_family8_nearest_"
                    "observations.md §3.")
    ap.add_argument("--work", default=os.path.join(CACHE, "family8"),
                    help="the build directory: index, per-year extracts, the "
                         "store, markers, progress.json")
    ap.add_argument("--stage", default="all", choices=["all"] + STAGES,
                    help="one stage, or `all`. Order is fixed: profiles needs "
                         "index, publish needs profiles.")
    ap.add_argument("--source-dir", default="",
                    help="read the index and every daily file from local "
                         "files — no network at all. Used by the tests and by "
                         "--smoke.")
    ap.add_argument("--smoke", action="store_true",
                    help="generate tiny synthetic sources in a temp dir and "
                         "run index + profiles, in seconds")
    ap.add_argument("--smoke-dir", default="",
                    help="with --smoke: keep the temp tree here")
    ap.add_argument("--start", default=str(ARGO_START),
                    help="first day to extract. The STORE's axis always covers "
                         "bins 0..3141 (1982-2024); before 2004 the float "
                         "array is too sparse to be worth the requests and "
                         "'none within R' is the honest answer (E-076 §2.3).")
    ap.add_argument("--end", default=str(END))
    ap.add_argument("--jobs", type=int, default=0,
                    help="netCDF parser processes (default: cpu_count capped "
                         f"at {MAX_PROCS}; 1 runs inline)")
    ap.add_argument("--dl-workers", type=int, default=0,
                    help=f"concurrent download connections (default "
                         f"{DL_WORKERS}). The job is DOWNLOAD-BOUND and "
                         f"Ifremer throttles per connection, so this is the "
                         f"knob that decides whether the pull fits in a day.")
    ap.add_argument("--inflight", type=int, default=0,
                    help=f"daily files allowed on disk at once (default "
                         f"{MAX_INFLIGHT}; 4-13 MB each, so 24 is ~300 MB)")
    ap.add_argument("--attempts", type=int, default=3,
                    help="download attempts per daily file before the YEAR "
                         "fails. A 404 from every mirror is not an attempt "
                         "failure — it is a missing day.")
    ap.add_argument("--index-anchors", type=int, default=2000,
                    help="random anchors for the k-th-neighbour measurement")
    ap.add_argument("--index-pentads", type=int, default=12,
                    help="pentads sampled for the nearest-neighbour measurement")
    ap.add_argument("--force", action="store_true",
                    help="redo a stage whose .done marker exists")
    a = ap.parse_args()

    if a.smoke:
        run_smoke(root=a.smoke_dir or None, keep=bool(a.smoke_dir))
        return 0

    ctx = Ctx(a)
    print(f"axis      {bin_start(ctx.b_lo, 5)} .. {bin_start(ctx.b_hi, 5)}  "
          f"bins {ctx.b_lo}..{ctx.b_hi} of 0..{N_BINS - 1} (recipe {RECIPE})")
    print(f"levels    {NLEV} RG pressures {LEVELS[0]:.0f}..{LEVELS[-1]:.0f} dbar")
    print(f"source    {GDAC}/geo/{{{','.join(BASINS)}}}/YYYY/MM/*_prof.nc"
          if not ctx.source_dir else f"source    {ctx.source_dir}")
    print(f"stream    {ctx.dl_workers} download connection(s), "
          f"<= {ctx.max_inflight} file(s) on disk, {ctx.n_proc} parser "
          f"process(es); mirror first ({GDAC_MIRROR.split('/')[2]}), "
          f"{GDAC.split('/')[2]} as fallback")
    stages = STAGES if a.stage == "all" else [a.stage]
    run_stages(ctx, stages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
