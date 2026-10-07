#!/usr/bin/env python3
"""E-088 · Per-year, per-calendar-month SUM, COUNT and M2 of the gridded stores,
so the Data tab can compose a mean (and a standard deviation) over ANY period
from a few range reads instead of reading every native frame.

WHAT THIS WRITES. For one sharded tier-G store (`ml/family1/sharded.py` — one
file of zstd tiles per five-day bin), per (calendar month, channel, year, cell):

    <out>/<store>/sum.npy     [12, C, Y, H, W] float32 — Σ of that month's
                              finite frames, in the store's PHYSICAL units,
                              float64-accumulated, rounded once; 0.0 where the
                              count is 0
    <out>/<store>/count.npy   [12, C, Y, H, W] uint8 — how many frames were
                              finite (≤ 31 daily, ≤ 124 six-hourly; asserted)
    <out>/<store>/m2.npy      [12, C, Y, H, W] float32 — Σ (x − mean_ym)² over
                              those frames, about THAT year-month's own mean
                              (two-pass, float64); 0.0 where count ≤ 1
    <out>/<store>/stats.json  years, frames present / possible per (year,
                              month), record span, channels and units (and
                              pressure levels), grid, licence, the source
                              store's sha256 block digest, git sha, UTC, and
                              the formulas below

Same axis order and offset formula as E-086 (`ml/export_family7_monthly.py`):
plane (m, c, y) starts at header_len + ((m·C + c)·Y + (y − year_first))·H·W·
itemsize, and a run of years for one (m, c) is one contiguous range.

    mean over a set of (year, month) cells K   = Σ_K sum / Σ_K count
    variance (population, ddof 0) over K       = (Σ_K m2 + Σ_K n_k (mean_k − mean)²) / Σ_K n_k
                                                  with mean_k = sum_k / n_k  (Chan et al.)

THE MONTH OF A FRAME is the true CALENDAR month (UTC) of the frame's own start
instant (`sharded.frame_datetime`). This is deliberately NOT E-086's
month-of-bin rule, which exists only to reproduce the model's climatology (a
five-day bin is charged to the month it opens in). So a July normal of the
daily SST store here and the five-day tensor's July normal differ slightly at
month edges: a bin opening on 30 June is June to the tensor and five-sixths
July here.

READING. Each worker takes one calendar year: it downloads the year's bins
(the shard and its index, sha256-checked against the store's own store.json),
decodes every frame through `sharded.ShardedGroup.read_frame` — the reader,
not a second decoder — and writes its year's planes into the output memmaps.
Nothing else is held: disk use is one year of bins per worker.

    python3 ml/export_gridded_monthly.py export --store family7_2d/oisst025d --out DIR [--base URL|DIR] [--workers N] [--keep DIR]
    python3 ml/export_gridded_monthly.py verify --store family7_2d/oisst025d --out DIR --keep DIR
    python3 ml/export_gridded_monthly.py run --stores a,b,... --out DIR       (the box: export, verify, upload, free — per store)
    python3 ml/export_gridded_monthly.py probe --store family1_gf/oc4k --month 2023-07 --out DIR
    python3 ml/export_gridded_monthly.py fixture
"""
import argparse
import calendar
import datetime as dt
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.parse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from family1 import sharded as sh                               # noqa: E402

PLAN = "ml/plans/E088_monthly_sums_gridded.md"
PLAN_URL = f"https://blauewelt.github.io/earth/docs.html?f={PLAN}"
HUB = "https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/"
AXES = ["month", "channel", "year", "lat", "lon"]
FILES = ("sum.npy", "count.npy", "m2.npy")
PHASE1 = ("family7_2d/glorys025d", "family7_2d/oisst025d",
          "family7_2d/ncep100d", "family7_2d/occci025d",
          "family1_2/era5_t", "family1_2/era5_q", "family1_2/era5_u",
          "family1_2/era5_v")
REGISTRY = {"family7_2d": "family7_2d/family72d.json",
            "family1_2": "family1_2/family12.json"}
# The falsifier's period-mean box and years (§5 of the plan): a small North
# Atlantic box — ocean for every ocean store, sea for ERA5 — and three whole
# years every phase-1 store covers.
PERIOD_YEARS = (2010, 2011, 2012)
PERIOD_BOX = (38.0, 42.0, -42.0, -38.0)          # lat_lo, lat_hi, lon_lo, lon_hi
MONTH_RULE = ("the CALENDAR month (UTC) of each frame's own start instant "
              "(sharded.frame_datetime) — not E-086's month-of-bin rule")
LAYOUT = ("sum.npy, m2.npy: [month, channel, year, lat, lon] float32 in the "
          "store's physical units; count.npy the same shape, uint8. Plane "
          "(m, c, y) starts at header_len + ((m * C + c) * Y + (y - "
          "year_first)) * H * W * itemsize; a run of years for one (m, c) is "
          "one contiguous range. sum and m2 are 0.0 where count is 0.")
COMBINE = ("mean over cells K = Σ sum / Σ count (NaN where Σ count = 0); "
           "population variance = (Σ m2 + Σ n_k (sum_k / n_k − mean)²) / "
           "Σ n_k, summing over the (year, month) cells with n_k > 0")


# =============================================================== helpers ====
def now_utc():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_sha():
    import subprocess
    try:
        r = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or "unknown"
    except Exception:                                         # noqa: BLE001
        return "unknown"


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, path)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def store_name(store):
    return store.rstrip("/").split("/")[-1]


# ================================================================ source ====
class Source:
    """A STORE (its store.json + <group>/…) on a local directory or the Hub."""

    def __init__(self, base, attempts=6):
        self.base = str(base).rstrip("/")
        self.url = self.base.startswith(("http://", "https://"))
        self.attempts = attempts
        self._sess = None
        self.bytes = 0

    def _session(self):
        if self._sess is None:
            import requests
            s = requests.Session()
            s.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=32))
            self._sess = s
        return self._sess

    def get(self, rel):
        if not self.url:
            with open(os.path.join(self.base, rel), "rb") as fh:
                b = fh.read()
            self.bytes += len(b)
            return b
        url = self.base + "/" + urllib.parse.quote(rel)
        tok = os.environ.get("HF_TOKEN", "").strip()
        hdr = {"Authorization": f"Bearer {tok}"} if tok else {}
        last, a, waited = None, 0, 0.0
        while True:
            try:
                r = self._session().get(url, headers=hdr, timeout=300)
                if r.status_code == 200:
                    self.bytes += len(r.content)
                    return r.content
                if r.status_code == 429:
                    # The Hub's rate limit is a QUEUE, not a failure: wait as
                    # long as it says (Retry-After, or `t=` in RateLimit),
                    # for up to RATE_WAIT_CAP_S in all, then carry on.
                    nap = rate_limit_wait(r.headers)
                    waited += nap
                    if waited > RATE_WAIT_CAP_S:
                        raise SystemExit(f"{url}: rate-limited for "
                                         f"{waited:.0f} s in all — giving up")
                    print(f"    429 from the Hub — waiting {nap:.0f} s",
                          flush=True)
                    time.sleep(nap)
                    continue
                if r.status_code in (500, 502, 503, 504):
                    raise ConnectionError(f"HTTP {r.status_code}")
                raise SystemExit(f"{url}: HTTP {r.status_code}")
            except SystemExit:
                raise
            except Exception as e:                            # noqa: BLE001
                last = e
                a += 1
                if a >= self.attempts:
                    raise SystemExit(f"{url}: gave up ({last})")
                time.sleep(min(60, 3 * 2 ** a))


RATE_WAIT_CAP_S = 3600


def rate_limit_wait(headers):
    """Seconds to wait after a 429, from the Hub's own headers (+ jitter)."""
    import random
    import re
    ra = headers.get("Retry-After")
    if ra and str(ra).strip().isdigit():
        base = float(ra)
    else:
        m = re.search(r"t=(\d+)", headers.get("RateLimit", "") or "")
        base = float(m.group(1)) if m else 30.0
    return min(310.0, base + 1.0 + random.random() * 4.0)


def open_store(base, store):
    """(src, meta, spec, shard_index, group) — store.json, the group's
    tile_grid.json and shard_index.npy, each checked against store.json's own
    sha256 block where it lists them."""
    src = Source(base)
    meta = json.loads(src.get("store.json"))
    groups = sorted(meta.get("groups") or {})
    g = store_name(store)
    if g not in groups:
        if len(groups) != 1:
            raise SystemExit(f"{store}: groups {groups}, expected {g}")
        g = groups[0]
    if meta.get("tier") != "G" or meta.get("layout") != "sharded":
        raise SystemExit(f"{store}: not a sharded tier-G store")
    spec = json.loads(src.get(f"{g}/tile_grid.json"))
    if spec.get("format") != sh.FORMAT:
        raise SystemExit(f"{store}: tile_grid format {spec.get('format')}")
    raw = src.get(f"{g}/shard_index.npy")
    want = (meta.get("sha256") or {}).get(f"{g}/shard_index.npy")
    if want and sha256_bytes(raw) != want:
        raise SystemExit(f"{store}: shard_index.npy sha256 differs from "
                         f"store.json")
    arr = sh.load_shard_index(raw)
    return src, meta, spec, arr, g


# ============================================================== calendar ====
def frame_times(arr, spec):
    """[(bin, frame, datetime)] of every PRESENT frame, time-ordered."""
    F, fs = spec["frames_per_bin"], spec["frame_seconds"]
    out = []
    for r in arr:
        b, mask = int(r["bin"]), int(r["frame_mask"])
        for f in range(F):
            if mask >> f & 1:
                out.append((b, f, sh.frame_datetime(b, f, fs)))
    out.sort(key=lambda x: x[2])
    return out


def month_bounds(y, m):
    """[start, end) datetimes of calendar month m (0-based) of year y."""
    a = dt.datetime(y, m + 1, 1)
    b = dt.datetime(y + (m == 11), 1 if m == 11 else m + 2, 1)
    return a, b


def frames_possible(y, m, lo, hi, fs):
    """Frame instants the record COULD hold in (y, m): epoch-aligned multiples
    of frame_seconds in [month start, month end) ∩ [record lo, record hi]."""
    a, b = month_bounds(y, m)
    a, b = max(a, lo), min(b, hi + dt.timedelta(seconds=1))
    if b <= a:
        return 0
    e = dt.datetime(sh.EPOCH.year, sh.EPOCH.month, sh.EPOCH.day)
    s0 = (a - e).total_seconds()
    s1 = (b - e).total_seconds()
    k0 = -(-int(s0) // fs)
    k1 = -(-int(s1) // fs)
    return max(0, k1 - k0)


def calendar_tables(times, fs):
    """years, frames_present [Y,12], frames_possible [Y,12], record span."""
    if not times:
        raise SystemExit("no present frame in the store")
    lo, hi = times[0][2], times[-1][2]
    y0, y1 = lo.year, hi.year
    Y = y1 - y0 + 1
    pres = np.zeros((Y, 12), np.int64)
    for _, _, t in times:
        pres[t.year - y0, t.month - 1] += 1
    poss = np.array([[frames_possible(y, m, lo, hi, fs) for m in range(12)]
                     for y in range(y0, y1 + 1)], np.int64)
    if (pres > poss).any():
        raise SystemExit("more frames present than possible in a month — "
                         "the frame calendar is not what the spec says")
    return list(range(y0, y1 + 1)), pres, poss, lo, hi


# ================================================================ worker ====
def _bin_files(g, b):
    return [f"{g}/{sh.shard_relpath(b)}", f"{g}/{sh.index_relpath(b)}"]


def fetch_bins(src_base, g, bins, dest, sha_map, threads=4):
    """Download every file of `bins` into dest/<rel>, sha256-checked."""
    src = Source(src_base)

    def one(rel):
        p = os.path.join(dest, rel)
        if os.path.exists(p):
            return 0
        b = src.get(rel)
        want = sha_map.get(rel)
        if want is None:
            raise SystemExit(f"{rel}: not in store.json's sha256 block")
        if sha256_bytes(b) != want:
            raise SystemExit(f"{rel}: sha256 differs from store.json — the "
                             f"bytes are not the published store's")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p + ".tmp", "wb") as fh:
            fh.write(b)
        os.replace(p + ".tmp", p)
        return len(b)

    rels = [r for b in bins for r in _bin_files(g, b)]
    with ThreadPoolExecutor(max_workers=threads) as ex:
        return sum(ex.map(one, rels))


def month_stats(frames, band=64):
    """(sum f64, count i32, m2 f64) over axis 0 of a [n, H, W, C] float32
    stack with NaN for missing — two-pass, in row bands to bound memory."""
    n, H, W, C = frames.shape
    s = np.zeros((H, W, C), np.float64)
    c = np.zeros((H, W, C), np.int32)
    m2 = np.zeros((H, W, C), np.float64)
    for r0 in range(0, H, band):
        blk = frames[:, r0:r0 + band]
        fin = np.isfinite(blk)
        cnt = fin.sum(axis=0, dtype=np.int32)
        tot = np.sum(blk, axis=0, where=fin, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = tot / cnt
        d = blk.astype(np.float64) - mean[None]
        np.square(d, out=d)
        m2b = np.sum(d, axis=0, where=fin, dtype=np.float64)
        s[r0:r0 + band], c[r0:r0 + band], m2[r0:r0 + band] = tot, cnt, m2b
    return s, c, m2


def year_worker(job):
    """One calendar year of one store -> its planes in the output memmaps."""
    (base, g, y, yi, bins, months, out_dir, keep_dir, keep_bins, sha_map,
     threads) = job
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix=f"gm_{g}_{y}_", dir=out_dir)
    try:
        got = fetch_bins(base, g, bins, tmp, sha_map, threads)
        spec_path = os.path.join(out_dir, "_spec", g, "tile_grid.json")
        os.makedirs(os.path.join(tmp, g), exist_ok=True)
        shutil.copy(spec_path, os.path.join(tmp, g, "tile_grid.json"))
        grp = sh.ShardedGroup(os.path.join(tmp, g))
        S = np.load(os.path.join(out_dir, "sum.npy"), mmap_mode="r+")
        N = np.load(os.path.join(out_dir, "count.npy"), mmap_mode="r+")
        M = np.load(os.path.join(out_dir, "m2.npy"), mmap_mode="r+")
        present = [0] * 12
        for m in range(12):
            fl = months[m]
            if not fl:
                continue
            frs = [grp.read_frame(b, f) for b, f in fl]
            if any(x is None for x in frs):
                raise SystemExit(f"{g} {y}-{m + 1}: a frame the shard index "
                                 f"lists as present is absent in its shard")
            st = np.stack(frs)
            del frs
            s, c, m2 = month_stats(st)
            del st
            if c.max() > len(fl) or c.max() > 255:
                raise SystemExit(f"{g} {y}-{m + 1}: count {c.max()} > "
                                 f"{len(fl)} frames")
            S[m, :, yi] = np.moveaxis(s, -1, 0).astype(np.float32)
            N[m, :, yi] = np.moveaxis(c, -1, 0).astype(np.uint8)
            M[m, :, yi] = np.moveaxis(m2, -1, 0).astype(np.float32)
            present[m] = len(fl)
        S.flush(), N.flush(), M.flush()
        del S, N, M
        if keep_dir and keep_bins:
            for b in keep_bins:
                for rel in _bin_files(g, b):
                    q = os.path.join(keep_dir, rel)
                    if not os.path.exists(q):
                        os.makedirs(os.path.dirname(q), exist_ok=True)
                        shutil.copy(os.path.join(tmp, rel), q)
        return dict(year=y, present=present, bytes=got,
                    seconds=time.time() - t0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ================================================================ export ====
def falsifier_months(years, pres):
    """[(y, m)]: the first and last months with data (partial ones included)
    and one month per decade the record touches."""
    have = [(y, m) for yi, y in enumerate(years) for m in range(12)
            if pres[yi, m] > 0]
    pick = [have[0], have[-1]]
    for d in range(years[0] // 10 * 10, years[-1] + 1, 10):
        cand = [ym for ym in have if d <= ym[0] < d + 10]
        if not cand:
            continue
        mid = cand[len(cand) // 2]
        if mid not in pick:
            pick.append(mid)
    return sorted(set(pick))


def grid_axes(spec):
    """(lat [H], lon [W], registration text) of PIXEL positions."""
    gr = spec["grid"]
    H, W = spec["H"], spec["W"]
    align = str(gr.get("align", ""))
    if align.startswith("point"):
        lat = gr["y0"] + gr["dy"] * np.arange(H)
        lon = gr["x0"] + gr["dx"] * np.arange(W)
    else:
        lat = gr["y0"] + gr["dy"] * (np.arange(H) + 0.5)
        lon = gr["x0"] + gr["dx"] * (np.arange(W) + 0.5)
    reg = (gr.get("align") or gr.get("coordinates") or
           "lon = x0 + (col + 0.5) * dx, lat = y0 + (row + 0.5) * dy")
    return lat, lon, reg


def export(store, out, *, base=None, workers=8, threads=4, keep=None,
           years=None, period_years=PERIOD_YEARS):
    """Write <out>/{sum,count,m2}.npy + stats.json for `store`."""
    from numpy.lib.format import open_memmap
    t_start = time.time()
    base = base or HUB + store
    src, meta, spec, arr, g = open_store(base, store)
    H, W, C = spec["H"], spec["W"], spec["C"]
    fs = spec["frame_seconds"]
    if spec["dtype"] != "float16":
        raise SystemExit(f"{store}: dtype {spec['dtype']} — phase 1 covers "
                         f"float16 stores only (uint8 decodes the same way "
                         f"but is not in scope)")
    times = frame_times(arr, spec)
    yrs, pres, poss, lo, hi = calendar_tables(times, fs)
    Y = len(yrs)
    if int(pres.max()) > 255:
        raise SystemExit(f"{store}: {pres.max()} frames in one month — uint8")
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "_spec", g), exist_ok=True)
    write_json(os.path.join(out, "_spec", g, "tile_grid.json"), spec)
    shape = (12, C, Y, H, W)
    for name, dtp in (("sum.npy", "<f4"), ("count.npy", "u1"),
                      ("m2.npy", "<f4")):
        open_memmap(os.path.join(out, name), mode="w+", dtype=dtp,
                    shape=shape).flush()
    by_ym = {}
    for b, f, t in times:
        by_ym.setdefault((t.year, t.month - 1), []).append((b, f))
    fmonths = falsifier_months(yrs, pres)
    keep_years = set(y for y in period_years if y in yrs)
    jobs = []
    for yi, y in enumerate(yrs):
        if years and y not in years:
            continue
        months = [by_ym.get((y, m), []) for m in range(12)]
        bins = sorted({b for fl in months for b, _ in fl})
        kb = set()
        if keep:
            for (fy, fm) in fmonths:
                if fy == y:
                    kb |= {b for b, _ in months[fm]}
            if y in keep_years:
                kb |= set(bins)
        jobs.append((base, g, y, yi, bins, months, out, keep, sorted(kb),
                     meta.get("sha256") or {}, threads))
    print(f"[{store}] {len(times)} frames, years {yrs[0]}–{yrs[-1]} ({Y}), "
          f"{len(jobs)} year job(s) on {workers} worker(s); output "
          f"{12 * C * Y * H * W * 9 / 1e9:.2f} GB",
          flush=True)
    done, nbytes = [], 0
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        for r in ex.map(year_worker, jobs):
            done.append(r)
            nbytes += r["bytes"]
            print(f"  [{store}] {r['year']}: {sum(r['present'])} frames, "
                  f"{r['bytes'] / 1e6:.0f} MB in {r['seconds']:.0f} s · "
                  f"{len(done)}/{len(jobs)} · {nbytes / 1e9:.1f} GB so far",
                  flush=True)
    for r in done:
        yi = yrs.index(r["year"])
        if r["present"] != pres[yi].tolist():
            raise SystemExit(f"{store} {r['year']}: frames summed "
                             f"{r['present']} != calendar {pres[yi].tolist()}")
    lat, lon, reg = grid_axes(spec)
    chans = [c["name"] for c in spec["channels"]]
    blk = meta.get("sha256") or {}
    digest = hashlib.sha256(json.dumps(blk, sort_keys=True).encode()).hexdigest()
    st = dict(
        _source="ml/export_gridded_monthly.py — do not hand-edit",
        plan=PLAN_URL, store=store, group=g,
        title=meta.get("title"), licence=meta.get("licence"),
        chans=chans, channels=spec["channels"],
        units={c["name"]: c["unit"] for c in spec["channels"]},
        levels_hpa=spec.get("levels_hpa"), channel_axis=spec.get("channel_axis"),
        dtype_source=spec["dtype"], frame_seconds=fs,
        frames_per_bin=spec["frames_per_bin"],
        axes=AXES, shape=list(shape), sum_dtype="float32",
        count_dtype="uint8", m2_dtype="float32",
        units_note="physical units as the store stores them",
        year_first=yrs[0], year_last=yrs[-1], n_years=Y, years=yrs,
        frames_present=pres.tolist(), frames_possible=poss.tolist(),
        max_count=int(pres.max()),
        record_first=lo.strftime("%Y-%m-%dT%H:%M:%SZ"),
        record_last=hi.strftime("%Y-%m-%dT%H:%M:%SZ"),
        grid=dict(H=H, W=W, lat0=float(lat[0]), lon0=float(lon[0]),
                  dlat=float(lat[1] - lat[0]), dlon=float(lon[1] - lon[0]),
                  south_first=bool(lat[1] > lat[0]), coordinates="pixel "
                  "positions (lat0 + row * dlat, lon0 + col * dlon)",
                  registration=reg, crs=spec["grid"].get("crs")),
        month_rule=MONTH_RULE, layout=LAYOUT, combine=COMBINE,
        source_sha256_block_digest=digest,
        source_shard_index_sha256=blk.get(f"{g}/shard_index.npy"),
        source_bytes_read=int(nbytes),
        falsifier_months=[[y, m + 1] for y, m in fmonths],
        falsifier_period_years=sorted(keep_years),
        falsifier_period_box=list(PERIOD_BOX),
        git_sha=git_sha(), generated_utc=now_utc(),
        export_seconds=round(time.time() - t_start, 1))
    shutil.rmtree(os.path.join(out, "_spec"), ignore_errors=True)
    write_json(os.path.join(out, "stats.json"), st)     # LAST: done-marker
    print(f"[{store}] exported in {(time.time() - t_start) / 60:.1f} min, "
          f"read {nbytes / 1e9:.1f} GB", flush=True)
    return st


# ================================================================ verify ====
def compose(S, N, M, mask=None):
    """(mean f64, n i64, std f64, mean_bound f64) over axis 0 of stacked
    (year, month) cells [K, ...] — the formulas the index states."""
    S = np.asarray(S, np.float64)
    M = np.asarray(M, np.float64)
    n_k = np.asarray(N, np.int64)
    if mask is not None:
        S, M, n_k = S[mask], M[mask], n_k[mask]
    n = n_k.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, S.sum(axis=0) / np.maximum(n, 1), np.nan)
        mk = np.where(n_k > 0, S / np.maximum(n_k, 1), 0.0)
        between = np.where(n_k > 0, n_k * (mk - mean[None]) ** 2, 0.0)
        var = (M.sum(axis=0) + between.sum(axis=0)) / np.maximum(n, 1)
        std = np.where(n > 0, np.sqrt(np.maximum(var, 0.0)), np.nan)
        bound = (0.5 * np.spacing(np.abs(S).astype(np.float32))
                 .astype(np.float64).sum(axis=0)) / np.maximum(n, 1)
    bound = bound + 4 * np.finfo(np.float64).eps * np.abs(np.nan_to_num(mean))
    return mean, n, std, bound


STD_REL_TOL = 1e-6     # |Δstd| ≤ STD_REL_TOL · (|mean| + std) — plan §4


def check(mean, n, std, bound, ref_mean, ref_n, ref_std, what):
    """Composed vs native. Returns a result dict; never raises."""
    r = dict(what=what, n_cells=int((ref_n > 0).sum()), ok=True, why=[])
    if not np.array_equal(n, ref_n):
        r["ok"] = False
        r["why"].append(f"counts differ in {int((n != ref_n).sum())} cells")
    fin = ref_n > 0
    if fin.any():
        d = np.abs(mean[fin] - ref_mean[fin])
        r["max_abs_mean"] = float(d.max())
        r["max_mean_over_bound"] = float((d / bound[fin]).max())
        if (d > bound[fin]).any():
            r["ok"] = False
            r["why"].append(f"{int((d > bound[fin]).sum())} cells beyond the "
                            f"mean bound")
        ds = np.abs(std[fin] - ref_std[fin])
        tol = STD_REL_TOL * (np.abs(ref_mean[fin]) + ref_std[fin]) + 1e-12
        r["max_abs_std"] = float(ds.max())
        r["max_std_over_tol"] = float((ds / tol).max())
        if (ds > tol).any():
            r["ok"] = False
            r["why"].append(f"{int((ds > tol).sum())} cells beyond the std "
                            f"tolerance")
    return r


def native_stats(stack):
    """numpy over native frames: (nanmean, count, nanstd ddof 0), float64."""
    x = stack.astype(np.float64)
    n = np.isfinite(x).sum(axis=0)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(x, axis=0), n, np.nanstd(x, axis=0)


def box_slices(spec, box=PERIOD_BOX):
    lat, lon, _ = grid_axes(spec)
    rows = np.flatnonzero((lat >= box[0]) & (lat <= box[1]))
    cols = np.flatnonzero((lon >= box[2]) & (lon <= box[3]))
    return slice(int(rows[0]), int(rows[-1]) + 1), \
        slice(int(cols[0]), int(cols[-1]) + 1)


def verify(store, out, keep, report=None, base=None):
    """THE FALSIFIER (plan §4): sampled planes and a period box mean, both
    against numpy over the native frames read back through sharded.py from the
    kept (sha256-checked) bins. Raises SystemExit on failure after writing
    the report."""
    st = json.load(open(os.path.join(out, "stats.json"), encoding="utf-8"))
    g = st["group"]
    spec_src = Source(base or HUB + store)
    spec = json.loads(spec_src.get(f"{g}/tile_grid.json"))
    gd = os.path.join(keep, g)
    with open(os.path.join(gd, "tile_grid.json"), "w") as fh:
        json.dump(spec, fh)
    grp = sh.ShardedGroup(gd)
    S = np.load(os.path.join(out, "sum.npy"), mmap_mode="r")
    N = np.load(os.path.join(out, "count.npy"), mmap_mode="r")
    M = np.load(os.path.join(out, "m2.npy"), mmap_mode="r")
    _, _, spec2, arr, _ = open_store(base or HUB + store, store)
    times = frame_times(arr, spec2)
    y0, chans = st["year_first"], st["chans"]
    res = dict(store=store, generated_utc=now_utc(), planes=[], period=None,
               std_rel_tol=STD_REL_TOL, ok=True, failures=[])
    for y, mo in st["falsifier_months"]:
        m = mo - 1
        fl = [(b, f) for b, f, t in times if t.year == y and t.month == mo]
        stack = np.stack([grp.read_frame(b, f) for b, f in fl])
        rm, rn, rs = native_stats(stack)
        yi = y - y0
        for c, name in enumerate(chans):
            mean, n, std, bound = compose(S[m, c, yi][None], N[m, c, yi][None],
                                          M[m, c, yi][None])
            r = check(mean, n, std, bound, rm[..., c], rn[..., c], rs[..., c],
                      f"{y}-{mo:02d} {name}")
            r.update(year=y, month=mo, channel=name, frames=len(fl),
                     frames_possible=st["frames_possible"][yi][m])
            res["planes"].append(r)
            if not r["ok"]:
                res["failures"].append(r)
    py = list(st.get("falsifier_period_years") or [])
    if py:
        rs_, cs_ = box_slices(spec)
        fl = [(b, f) for b, f, t in times if t.year in py]
        T = spec["tile"]
        tys = sorted({r // T for r in range(rs_.start, rs_.stop)})
        txs = sorted({c // T for c in range(cs_.start, cs_.stop)})
        parts = []
        for b, f in fl:
            fr = np.full((spec["H"], spec["W"], spec["C"]), np.nan, np.float32)
            for ty in tys:
                for tx in txs:
                    t = grp.read_tile(b, f, ty, tx, crop=True)
                    r0, r1 = spec["row_extents"][ty]
                    c0, c1 = spec["col_extents"][tx]
                    fr[r0:r1, c0:c1] = t
            parts.append(fr[rs_, cs_])
        rm, rn, rsd = native_stats(np.stack(parts))
        yis = [y - y0 for y in py]
        out_r = []
        for c, name in enumerate(chans):
            Sc = np.stack([S[m, c, yi][rs_, cs_] for yi in yis
                           for m in range(12)])
            Nc = np.stack([N[m, c, yi][rs_, cs_] for yi in yis
                           for m in range(12)])
            Mc = np.stack([M[m, c, yi][rs_, cs_] for yi in yis
                           for m in range(12)])
            mean, n, std, bound = compose(Sc, Nc, Mc)
            r = check(mean, n, std, bound, rm[..., c], rn[..., c],
                      rsd[..., c], f"{py[0]}–{py[-1]} box {name}")
            r.update(channel=name, years=py, frames=len(fl),
                     box=list(PERIOD_BOX))
            out_r.append(r)
            if not r["ok"]:
                res["failures"].append(r)
        res["period"] = out_r
    res["ok"] = not res["failures"]
    for k in ("max_abs_mean", "max_mean_over_bound", "max_abs_std",
              "max_std_over_tol"):
        vals = [p.get(k, 0.0) for p in res["planes"] + (res["period"] or [])]
        res[k] = float(max(vals)) if vals else 0.0
    res["n_planes"] = len(res["planes"])
    if report:
        write_json(report, res)
    print(f"  verify [{store}]: {res['n_planes']} planes + period box · "
          f"max mean/bound {res['max_mean_over_bound']:.3f} · max |Δmean| "
          f"{res['max_abs_mean']:.3g} · max std/tol "
          f"{res['max_std_over_tol']:.3f} · {'OK' if res['ok'] else 'FAIL'}",
          flush=True)
    if not res["ok"]:
        raise SystemExit(f"VERIFY FAILED for {store}: "
                         f"{[f['what'] for f in res['failures']][:10]}")
    return res


# =================================================================== run ====
def hub_has(store, src_sha):
    """True when the store's monthly/stats.json on the Hub was computed from
    the same source sha256 block (a finished store from an earlier box)."""
    try:
        b = Source(HUB + store + "/monthly").get("stats.json")
        st = json.loads(b)
        return st.get("source_sha256_block_digest") == src_sha
    except (SystemExit, OSError, ValueError):
        return False


def run(stores, out, *, workers=8, threads=4, upload=True):
    """The box: per store export -> verify -> upload (one commit, with its
    manifest and verify report) -> free the disk. A finished store is skipped
    (its monthly/stats.json on the Hub names the same source), so a lost box
    redoes at most the store it was on (ml/CLAUDE.md §5.26)."""
    import publish_gridded_monthly_index as P
    summary = {}
    for store in stores:
        meta = json.loads(Source(HUB + store).get("store.json"))
        digest = hashlib.sha256(json.dumps(meta.get("sha256") or {},
                                           sort_keys=True).encode()).hexdigest()
        if hub_has(store, digest):
            print(f"[{store}] already published from this source — skipped",
                  flush=True)
            summary[store] = "skipped (already on the Hub)"
            continue
        d = os.path.join(out, store_name(store))
        keep = os.path.join(out, "_keep_" + store_name(store))
        shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(keep, ignore_errors=True)
        st = export(store, d, workers=workers, threads=threads, keep=keep)
        rep = verify(store, d, keep, report=os.path.join(d, "verify.json"))
        shutil.rmtree(keep, ignore_errors=True)
        if upload:
            P.upload_store(store, d)
            shutil.rmtree(d, ignore_errors=True)
        summary[store] = dict(years=[st["year_first"], st["year_last"]],
                              bytes_read=st["source_bytes_read"],
                              minutes=round(st["export_seconds"] / 60, 1),
                              verify_ok=rep["ok"])
        write_json(os.path.join(out, "run_summary.json"), summary)
    return summary


# ================================================================= probe ====
def probe(store, month, out, threads=8, level=15):
    """Phase 2: what a per-year monthly sum + count of a FINE store would cost.
    Streams one calendar month, accumulates per-cell sums and counts, and
    measures the monthly union coverage and the zstd size of the month's sum
    (float32) and count (uint8) planes stored as the store's own tiles."""
    import zstandard
    t0 = time.time()
    src, meta, spec, arr, g = open_store(HUB + store, store)
    y, mo = (int(x) for x in month.split("-"))
    times = [x for x in frame_times(arr, spec)
             if x[2].year == y and x[2].month == mo]
    bins = sorted({b for b, _, _ in times})
    H, W, C, T = spec["H"], spec["W"], spec["C"], spec["tile"]
    nty, ntx = spec["n_tiles_y"], spec["n_tiles_x"]
    tmp = tempfile.mkdtemp(prefix="gm_probe_", dir=out)
    try:
        nbytes = fetch_bins(HUB + store, g, bins, tmp, meta.get("sha256") or {},
                            threads)
        with open(os.path.join(tmp, g, "tile_grid.json"), "w") as fh:
            json.dump(spec, fh)
        grp = sh.ShardedGroup(os.path.join(tmp, g))
        cctx = zstandard.ZstdCompressor(level=level)
        sum_b = cnt_b = tiles_cov = 0
        cov = np.zeros(C, np.int64)
        for ty in range(nty):
            for tx in range(ntx):
                s = np.zeros((T, T, C), np.float64)
                n = np.zeros((T, T, C), np.int32)
                for b, f, _ in times:
                    t = grp.read_tile(b, f, ty, tx)
                    if t is None:
                        continue
                    fin = np.isfinite(t)
                    s += np.where(fin, t, 0.0)
                    n += fin
                if not n.any():
                    continue
                tiles_cov += 1
                r0, r1 = spec["row_extents"][ty]
                c0, c1 = spec["col_extents"][tx]
                cov += (n[:r1 - r0, :c1 - c0] > 0).reshape(-1, C).sum(axis=0)
                sum_b += len(cctx.compress(s.astype(np.float32).tobytes()))
                cnt_b += len(cctx.compress(np.minimum(n, 255)
                                           .astype(np.uint8).tobytes()))
        months_record = len({(t.year, t.month)
                             for _, _, t in frame_times(arr, spec)})
        dense = H * W * C
        res = dict(store=store, month=month, frames=len(times), bins=len(bins),
                   bytes_read=int(nbytes), H=H, W=W, C=C, tile=T,
                   dense_cells_per_month=dense,
                   dense_bytes_sum_count_per_month=dense * 5,
                   union_coverage=(cov / (H * W)).tolist(),
                   tiles_with_data=tiles_cov, tiles_total=nty * ntx,
                   zstd_level=level, sum_bytes_compressed=int(sum_b),
                   count_bytes_compressed=int(cnt_b),
                   months_in_record=months_record,
                   record_estimate_bytes=int((sum_b + cnt_b) * months_record),
                   record_estimate_dense_bytes=int(dense * 5 * months_record),
                   seconds=round(time.time() - t0, 1))
        write_json(os.path.join(out, f"probe_{store_name(store)}.json"), res)
        print(f"  probe [{store} {month}]: {len(times)} frames, "
              f"{nbytes / 1e9:.2f} GB read, coverage "
              f"{[round(c, 3) for c in res['union_coverage']]}, "
              f"{tiles_cov}/{nty * ntx} tiles, sum+count "
              f"{(sum_b + cnt_b) / 1e6:.1f} MB/month compressed vs "
              f"{dense * 5 / 1e6:.0f} MB dense", flush=True)
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# =============================================================== fixture ====
FIXTURE_OUT = os.path.join(ROOT, "data", "gridded_monthly", "fixture")


def run_fixture(out=FIXTURE_OUT):
    """Build two tiny sharded stores with the real tier-G framework (one daily
    across a year boundary with gaps and an absent day, one six-hourly), run
    export -> verify -> index --local on them. Rewrites `out`."""
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import make_gridded_monthly_fixture as F
    import publish_gridded_monthly_index as P
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    src_dir = tempfile.mkdtemp(prefix="gm_fix_")
    try:
        stores = F.build(src_dir)                    # {store: local dir}
        for store, base in stores.items():
            d = os.path.join(out, store_name(store))
            keep = os.path.join(src_dir, "_keep_" + store_name(store))
            export(store, d, base=base, workers=2, threads=2, keep=keep)
            verify(store, d, keep, report=os.path.join(d, "verify.json"),
                   base=base)
        rel = os.path.relpath(out, ROOT)
        P.main(["index", "--local", "--out", out, "--stores",
                ",".join(stores), "--no-cors", "--fixture-dir", rel + "/",
                "--index", os.path.join(out, "gridded_monthly_index.json")])
    finally:
        shutil.rmtree(src_dir, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("export", "verify", "run", "probe"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--base", default="")
        if name == "run":
            p.add_argument("--stores", default=",".join(PHASE1))
            p.add_argument("--no-upload", action="store_true")
        else:
            p.add_argument("--store", required=True)
        if name in ("export", "verify"):
            p.add_argument("--keep", default="")
        if name == "probe":
            p.add_argument("--month", required=True)
    sub.add_parser("fixture")
    a = ap.parse_args(argv)
    if a.cmd == "fixture":
        run_fixture()
    elif a.cmd == "export":
        export(a.store, a.out, base=a.base or None, workers=a.workers,
               threads=a.threads, keep=a.keep or None)
    elif a.cmd == "verify":
        verify(a.store, a.out, a.keep, report=os.path.join(a.out,
                                                           "verify.json"),
               base=a.base or None)
    elif a.cmd == "run":
        run([s for s in a.stores.split(",") if s], a.out, workers=a.workers,
            threads=a.threads, upload=not a.no_upload)
    else:
        os.makedirs(a.out, exist_ok=True)
        probe(a.store, a.month, a.out, threads=a.threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
