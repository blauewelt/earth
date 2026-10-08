#!/usr/bin/env python3
"""E-089 · Long-period-average sums for the FINE family 1.gf grids
(`ml/plans/E089_fine_grid_monthly_sums.md`): per (calendar month, year,
channel, cell) the SUM of the month's finite frames, their COUNT and M2 (the
sum of squared deviations about that year-month's own mean), in two layers:

  1. NATIVE — at the store's own resolution, one file per (year, month) of
     independently zstd-compressed tiles in the store's own 256 x 256 tiling,
     empty tiles not stored (`ml/family1/monthly_tiles.py`):
         <out>/native/tile_grid.json
         <out>/native/<yyyy>/m_<yyyy>-<mm>.zst  +  .idx.npy
  2. POOLED — an EXACT 0.25° layer in E-088's dense layout
     ([month, channel, year, lat, lon], 128-byte .npy header):
         <out>/pooled025/sum.npy    float32
         <out>/pooled025/count.npy  uint16 (a 0.25° cell pools up to
                                    13 x 13 x 31 = 5,239 daily or 7 x 7 x 248
                                    = 12,152 three-hourly values)
         <out>/pooled025/m2.npy     float32
     A 0.25° cell holds EVERY native pixel whose CENTRE lies in it (cells
     half-open, [south, north) x [west, east); a centre within 1e-6 of a cell
     of an edge is ON that edge). For the 4 km grids that is an exact 6 x 6
     block; for the 2 km SST (0.02°: 12.5 pixels per 0.25°) and the
     geostationary IR (360/9896°) the blocks alternate 12/13 and 6/7 pixels.
     The pooled sum is float32(Σ over the block of the PUBLISHED native float32
     sums, in float64) — exact before that one rounding (a float64 holds the
     sum of up to 169 such values exactly), so the pooled sum and count ARE
     the native layer block-summed; m2 combines the block with Chan's
     formula m2 = Σ m2_k + Σ n_k (mean_k − mean)².
  plus <out>/stats.json (tables, grids, the pool map, the source digest) and
  <out>/verify.json (the falsifier's report).

    mean over a set K of (year, month) cells  = Σ_K sum / Σ_K count
    population variance over K               = (Σ_K m2 + Σ_K n_k (sum_k/n_k − mean)²) / Σ_K n_k

THE MONTH of a frame is the true CALENDAR month (UTC) of its own start
instant — E-088's rule. Physical units: float16 stores as stored; the uint8
`irtb` store ("K - 160") is summed in KELVIN.

    python3 ml/export_fine_monthly.py export --store family1_gf/oc4k --out DIR [--keep DIR]
    python3 ml/export_fine_monthly.py verify --store family1_gf/oc4k --out DIR --keep DIR
    python3 ml/export_fine_monthly.py run --stores a,b --out DIR      (the box: per store export -> verify -> upload + read-back -> free)
    python3 ml/export_fine_monthly.py fixture                          (data/gridded_monthly/fixture_fine/)
"""
import argparse
import datetime as dt
import hashlib
import json
import mmap
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import export_gridded_monthly as X                               # noqa: E402
from family1 import sharded as sh                                # noqa: E402
from family1 import monthly_tiles as mt                          # noqa: E402

PLAN = "ml/plans/E089_fine_grid_monthly_sums.md"
PLAN_URL = f"https://blauewelt.github.io/earth/docs.html?f={PLAN}"
FINE = ("family1_gf/oc4k", "family1_gf/pace4k", "family1_gf/sst_acspo02",
        "family1_gf/irtb")
POOL_DEG = 0.25
SNAP = 1e-6                     # cells: a centre this close to an edge is on it
AXES = list(X.AXES)
POOLED = (("sum", "<f4"), ("count", "<u2"), ("m2", "<f4"))
COUNT_MAX_POOLED = 65535
STD_REL_TOL = X.STD_REL_TOL
# The falsifier's boxes: (year, months, (lat_lo, lat_hi, lon_lo, lon_hi)).
# A whole mid-record year in one region and a three-month season in another;
# the ocean stores share theirs, the tropical IR band has its own.
OCEAN_NA = (38.0, 42.0, -42.0, -38.0)       # North Atlantic, south of the Grand Banks
BENGUELA = (-36.0, -32.0, 14.0, 18.0)       # off south-western Africa
TROP_ATL = (0.0, 4.0, -30.0, -26.0)         # the Atlantic ITCZ
TROP_IND = (-10.0, -6.0, 100.0, 104.0)      # west of Sumatra
JJA = (6, 7, 8)
ALL = tuple(range(1, 13))
BOXES = {
    "family1_gf/oc4k": [(2015, ALL, OCEAN_NA), (2000, JJA, BENGUELA)],
    "family1_gf/pace4k": [(2025, ALL, OCEAN_NA), (2024, JJA, BENGUELA)],
    "family1_gf/sst_acspo02": [(2015, ALL, OCEAN_NA), (2005, JJA, BENGUELA)],
    "family1_gf/irtb": [(2015, ALL, TROP_ATL), (2005, JJA, TROP_IND)],
}
MONTH_RULE = X.MONTH_RULE
COMBINE = X.COMBINE
POOL_RULE = ("a pooled cell holds every native pixel whose CENTRE lies in it; "
             "cells are half-open [south, north) x [west, east) on the 0.25° "
             "grid whose edges are multiples of 0.25° (cell-registered: cell "
             "centres at -89.875, …), and a centre within 1e-6 of a cell "
             "(2.5e-7°) of an edge lies ON that edge; `rows` / `cols` give each "
             "pooled row's / column's native [start, stop)")
NATIVE_LAYOUT = (
    "native/<yyyy>/m_<yyyy>-<mm>.zst + .idx.npy, one pair per (year, month) "
    "with at least one frame: the store's own tiles, each tile's parts sum "
    "(float32), count (uint8), m2 (float32) — T x T x C, C order [row, col, "
    "channel], little-endian, one zstd frame per (tile, part) — concatenated "
    "tiles row-major and parts sum, count, m2 within a tile (so a tile's "
    "parts are ADJACENT). The index is int64 [n_tiles_y, n_tiles_x, 3, 2] of "
    "(offset, length), header index_header_bytes; length 0 for all three "
    "parts = the tile holds no observation that month (sum 0, count 0, m2 0, "
    "nothing stored). Edge tiles are padded with zeros (count 0).")
POOLED_LAYOUT = (
    "pooled025/sum.npy, m2.npy: [month, channel, year, lat, lon] float32; "
    "count.npy the same shape uint16; lat south-first; plane (m, c, y) starts "
    "at header_len + ((m * C + c) * n_years + (y - year_first)) * plane_bytes "
    "— E-088's layout and formula. sum and m2 are 0.0 where count is 0.")


# ============================================================ geometry ====
def pixel_centres(spec):
    lat, lon, _ = X.grid_axes(spec)
    return np.asarray(lat, np.float64), np.asarray(lon, np.float64)


def cell_of(v, origin, deg):
    t = (np.asarray(v, np.float64) - origin) / deg
    r = np.round(t)
    t = np.where(np.abs(t - r) < SNAP, r, t)
    return np.floor(t).astype(np.int64)


def _runs(ids, n_cells, what):
    """[start, stop) of each cell id in a monotone id sequence, cells
    consecutive; refuses gaps and non-monotone maps."""
    d = np.diff(ids)
    if not ((d >= 0).all() or (d <= 0).all()) or (np.abs(d) > 1).any():
        raise SystemExit(f"pool map: {what} cells are not a monotone run of "
                         f"consecutive cells")
    if ids.min() < 0 or ids.max() >= n_cells:
        raise SystemExit(f"pool map: {what} cell {ids.min()}..{ids.max()} "
                         f"outside 0..{n_cells - 1}")
    lo, hi = int(ids.min()), int(ids.max())
    out = []
    for j in range(lo, hi + 1):
        k = np.flatnonzero(ids == j)
        out.append([int(k[0]), int(k[-1]) + 1])
    return lo, out


def pool_map(spec, deg=POOL_DEG):
    """How the native pixels of a tile_grid map onto the cell-registered
    `deg` grid. Pooled rows are SOUTH-FIRST: pooled row p is global cell row
    j0 + p, its native rows [rows[p][0], rows[p][1])."""
    lat, lon = pixel_centres(spec)
    nlat, nlon = int(round(180 / deg)), int(round(360 / deg))
    J = cell_of(lat, -90.0, deg)
    I = cell_of(lon, -180.0, deg)
    j0, rows = _runs(J, nlat, "row")
    i0, cols = _runs(I, nlon, "column")
    hs = sorted({b - a for a, b in rows})
    ws = sorted({b - a for a, b in cols})
    # every native pixel inside a cell, so the pooled cells partition the grid
    if sum(b - a for a, b in rows) != spec["H"] or \
            sum(b - a for a, b in cols) != spec["W"]:
        raise SystemExit("pool map: the cells do not cover every pixel once")
    return dict(deg=deg, Hp=len(rows), Wp=len(cols), j0=j0, i0=i0,
                lat0=-90.0 + (j0 + 0.5) * deg, lon0=-180.0 + (i0 + 0.5) * deg,
                rows=rows, cols=cols, jrow=(J - j0), icol=(I - i0),
                block_rows=hs, block_cols=ws,
                exact_blocks=(len(hs) == 1 and len(ws) == 1),
                north_first=bool(lat[-1] < lat[0]))


def pooled_grid_record(pm):
    """The index's / stats.json's description of the pooled grid."""
    bh, bw = pm["block_rows"], pm["block_cols"]
    note = (f"exact {bh[0]} x {bw[0]} blocks of native pixels" if
            pm["exact_blocks"] else
            f"{'/'.join(map(str, bh))} native rows x {'/'.join(map(str, bw))} "
            f"native columns per cell (centre-binned)")
    return dict(H=pm["Hp"], W=pm["Wp"], deg=pm["deg"], lat0=pm["lat0"],
                lon0=pm["lon0"], dlat=pm["deg"], dlon=pm["deg"],
                south_first=True, global_row0=pm["j0"], global_col0=pm["i0"],
                coordinates=("cell centres (lat0 + row * dlat, lon0 + col * "
                             "dlon); cell edges at multiples of dlat"),
                registration=("cell — row r covers latitudes [lat0 - dlat/2 + "
                              "r dlat, lat0 + dlat/2 + r dlat); NOT family 7's "
                              "point-registered g025 grid (whose centres are "
                              "the multiples of 0.25°)"),
                rule=POOL_RULE, block_note=note,
                exact_blocks=pm["exact_blocks"], rows=pm["rows"],
                cols=pm["cols"], native_north_first=pm["north_first"])


def segments(ids):
    """(starts, cell ids) of the runs of equal ids in a monotone sequence."""
    ids = np.asarray(ids)
    if not len(ids):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    st = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    return st, ids[st]


def add_cells(acc, a, jr, ic):
    """acc[J, I, …] += the sum of `a` [h, w, …] over each (row run, col run);
    jr / ic are the pooled cell of each of a's rows / columns."""
    rs, J = segments(jr)
    cs, I = segments(ic)
    b = np.add.reduceat(np.add.reduceat(a, rs, axis=0), cs, axis=1)
    acc[np.ix_(J, I)] += b


# ============================================================ pooling =====
def pool_month(S, N, M2, pm):
    """Native month (S f32, N u8, M2 f32, [H, W, C]) -> pooled (sum f32,
    count u16, m2 f32, [Hp, Wp, C]), row band by row band; exact sums and
    counts, Chan's m2."""
    Hp, Wp, C = pm["Hp"], pm["Wp"], S.shape[2]
    cs = np.array([a for a, _ in pm["cols"]], np.int64)
    widths = np.array([b - a for a, b in pm["cols"]], np.int64)
    Sp = np.zeros((Hp, Wp, C), np.float32)
    Np = np.zeros((Hp, Wp, C), np.uint16)
    Mp = np.zeros((Hp, Wp, C), np.float32)
    for p, (r0, r1) in enumerate(pm["rows"]):
        s = S[r0:r1].astype(np.float64)
        n = N[r0:r1].astype(np.int64)
        m2 = M2[r0:r1].astype(np.float64)
        sb = np.add.reduceat(s.sum(axis=0), cs, axis=0)
        nb = np.add.reduceat(n.sum(axis=0), cs, axis=0)
        if nb.max(initial=0) > COUNT_MAX_POOLED:
            raise SystemExit(f"pooled count {nb.max()} > uint16")
        with np.errstate(invalid="ignore", divide="ignore"):
            mb = np.where(nb > 0, sb / np.maximum(nb, 1), 0.0)
            mk = np.where(n > 0, s / np.maximum(n, 1), 0.0)
        me = np.repeat(mb, widths, axis=0)[None]
        btw = np.where(n > 0, n * (mk - me) ** 2, 0.0)
        m2b = np.add.reduceat((m2 + btw).sum(axis=0), cs, axis=0)
        Sp[p], Np[p] = sb.astype(np.float32), nb.astype(np.uint16)
        Mp[p] = np.where(nb > 1, m2b, 0.0).astype(np.float32)
    return Sp, Np, Mp


# ======================================================= local bins =======
class LocalBins:
    """The bins of one month on local disk: their indices in memory, their
    shards memory-mapped, tiles decoded with the group's own rules."""

    def __init__(self, gdir, spec):
        self.gdir, self.spec = gdir, spec
        self._d = sh.zstd().ZstdDecompressor()
        self._bins = {}
        T, C = spec["tile"], spec["C"]
        self._dt = sh.DTYPES[spec["dtype"]]
        self._want = T * T * C * self._dt.itemsize

    def _open(self, b):
        if b not in self._bins:
            idx = np.load(os.path.join(self.gdir, sh.index_relpath(b)),
                          allow_pickle=False)
            p = os.path.join(self.gdir, sh.shard_relpath(b))
            fh = open(p, "rb")
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) if \
                os.path.getsize(p) else b""
            self._bins[b] = (idx, fh, mm)
        return self._bins[b]

    def close(self, keep=()):
        for b in list(self._bins):
            if b in keep:
                continue
            idx, fh, mm = self._bins.pop(b)
            if mm:
                mm.close()
            fh.close()

    def tile(self, b, f, ty, tx):
        """float32 with NaN for missing, cropped to the tile's real extent."""
        sp = self.spec
        T, C = sp["tile"], sp["C"]
        idx, _, mm = self._open(b)
        off, n = int(idx[f, ty, tx, 0]), int(idx[f, ty, tx, 1])
        r0, r1 = sp["row_extents"][ty]
        c0, c1 = sp["col_extents"][tx]
        if off == sh.FRAME_ABSENT:
            raise SystemExit(f"bin {b} frame {f}: listed present by the shard "
                             f"index, absent in its shard")
        if n == 0:
            return np.full((r1 - r0, c1 - c0, C), np.nan, np.float32)
        raw = self._d.decompress(mm[off:off + n], max_output_size=self._want)
        if len(raw) != self._want:
            raise SystemExit(f"bin {b}: a tile decompressed to {len(raw)}")
        t = np.frombuffer(raw, self._dt).reshape(T, T, C)[:r1 - r0, :c1 - c0]
        return sh.decode(t, sp["dtype"])


def tile_stats(st):
    """(sum f64, count i32, m2 f64) over axis 0 of a [n, h, w, C] float32
    stack, NaN = missing: float64 sums of the float32 values (exact for
    float16 / integer inputs), then Σ (x − mean)² frame by frame in float64."""
    fin = np.isfinite(st)
    cnt = fin.sum(axis=0, dtype=np.int32)
    x = np.where(fin, st, np.float32(0))
    tot = x.sum(axis=0, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
    m2 = np.zeros(tot.shape, np.float64)
    for i in range(st.shape[0]):
        d = x[i].astype(np.float64) - mean
        d *= d
        m2 += np.where(fin[i], d, 0.0)
    return tot, cnt, m2


def month_native(lb, frames, spec, offset):
    """(S f32, N u8, M2 f32) [H, W, C] of one month's frames, tile by tile."""
    H, W, C = spec["H"], spec["W"], spec["C"]
    S = np.zeros((H, W, C), np.float32)
    N = np.zeros((H, W, C), np.uint8)
    M = np.zeros((H, W, C), np.float32)
    for ty in range(spec["n_tiles_y"]):
        r0, r1 = spec["row_extents"][ty]
        for tx in range(spec["n_tiles_x"]):
            c0, c1 = spec["col_extents"][tx]
            st = np.stack([lb.tile(b, f, ty, tx) for b, f in frames])
            if offset:
                st += np.float32(offset)
            s, c, m2 = tile_stats(st)
            if c.max(initial=0) > min(len(frames), 255):
                raise SystemExit(f"count {c.max()} > {len(frames)} frames")
            S[r0:r1, c0:c1] = s.astype(np.float32)
            N[r0:r1, c0:c1] = c.astype(np.uint8)
            M[r0:r1, c0:c1] = np.where(c > 1, m2, 0.0).astype(np.float32)
    return S, N, M


# ============================================================ worker ======
def run_worker(job):
    """A run of consecutive (year, month)s: per month download its bins
    (sha256-checked), compute the native statistics, write the month file,
    read it back, pool it into the 0.25° memmaps, free the bins no later
    month of the run needs."""
    (base, g, months, out, keep_dir, keep_months, sha_map, threads, offset,
     pm, y0, workers_tag) = job
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix=f"fm_{g}_{months[0][0]}{months[0][1]:02d}_",
                           dir=out)
    gdir = os.path.join(tmp, g)
    os.makedirs(gdir, exist_ok=True)
    spec = json.load(open(os.path.join(out, "_spec", "tile_grid.json")))
    mspec = json.load(open(os.path.join(out, "native", "tile_grid.json")))
    reader = mt.MonthlyTiles(os.path.join(out, "native"))
    cctx = sh.zstd().ZstdCompressor(level=int(mspec["compression"]["level"]))
    res, nbytes = [], 0
    try:
        P = {k: np.load(os.path.join(out, "pooled025", f"{k}.npy"),
                        mmap_mode="r+") for k, _ in POOLED}
        lb = LocalBins(gdir, spec)
        for i, (y, m, frames) in enumerate(months):
            tm = time.time()
            bins = sorted({b for b, _ in frames})
            nbytes += X.fetch_bins(base, g, bins, tmp, sha_map, threads)
            S, N, M = month_native(lb, frames, spec, offset)
            sp_ = os.path.join(out, "native", mt.shard_relpath(y, m))
            ip_ = os.path.join(out, "native", mt.index_relpath(y, m))
            w = mt.write_month(mspec, S, N, M, sp_, ip_, cctx)
            # read the month back through the reader, tile by tile
            reader._idx.pop((y, m), None)
            for ty in range(mspec["n_tiles_y"]):
                r0, r1 = mspec["row_extents"][ty]
                for tx in range(mspec["n_tiles_x"]):
                    c0, c1 = mspec["col_extents"][tx]
                    t = reader.read_tile(y, m, ty, tx)
                    for k, a in (("sum", S), ("count", N), ("m2", M)):
                        if not np.array_equal(t[k], a[r0:r1, c0:c1]):
                            raise SystemExit(f"{y}-{m:02d} tile {ty},{tx}: "
                                             f"{k} does not read back")
            mt.check_structure(reader.index(y, m), os.path.getsize(sp_),
                               mspec, sp_)
            Sp, Np, Mp = pool_month(S, N, M, pm)
            yi = y - y0
            for c in range(S.shape[2]):
                P["sum"][m - 1, c, yi] = Sp[..., c]
                P["count"][m - 1, c, yi] = Np[..., c]
                P["m2"][m - 1, c, yi] = Mp[..., c]
            del S, N, M, Sp, Np, Mp
            if keep_dir and (y, m) in keep_months:
                for b in bins:
                    for rel in X._bin_files(g, b):
                        q = os.path.join(keep_dir, rel)
                        if not os.path.exists(q):
                            os.makedirs(os.path.dirname(q), exist_ok=True)
                            shutil.copy(os.path.join(tmp, rel), q)
            nxt = {b for b, _ in months[i + 1][2]} if i + 1 < len(months) \
                else set()
            lb.close(keep=nxt)
            for b in bins:
                if b not in nxt:
                    for rel in X._bin_files(g, b):
                        p = os.path.join(tmp, rel)
                        if os.path.exists(p):
                            os.remove(p)
            res.append(dict(year=y, month=m, frames=len(frames),
                            tiles_stored=w["tiles_stored"],
                            shard_bytes=w["nbytes"],
                            seconds=round(time.time() - tm, 1)))
            print(f"    [{g}] {y}-{m:02d}: {len(frames)} frames, "
                  f"{w['tiles_stored']} tiles, {w['nbytes'] / 1e6:.1f} MB in "
                  f"{time.time() - tm:.0f} s", flush=True)
        for v in P.values():
            v.flush()
        del P
        lb.close()
        return dict(months=res, bytes=nbytes, seconds=time.time() - t0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def auto_workers(requested, spec):
    """Workers bounded by RAM: a worker holds one native month (sum, count,
    m2: 9 bytes per cell) and a tile stack."""
    per = spec["H"] * spec["W"] * spec["C"] * 9 * 1.3 + 1.2e9
    avail = 8e9
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
    except OSError:                                          # pragma: no cover
        pass
    return max(1, min(int(requested), int(0.7 * avail / per)))


# ============================================================ export ======
def keep_months_of(store, years, pres, boxes):
    have = {(y, m + 1) for yi, y in enumerate(years) for m in range(12)
            if pres[yi, m] > 0}
    fm = [(y, m + 1) for y, m in X.falsifier_months(years, pres)]
    bm = [(y, m) for y, ms, _ in boxes for m in ms if (y, m) in have]
    return sorted(set(fm)), sorted(set(bm))


def export(store, out, *, base=None, workers=8, threads=4, keep=None,
           boxes=None, deg=POOL_DEG, months_per_job=None, only=None):
    """`only` [(y, m)] restricts the export to those months (a trial: the
    stats then describe only them, and the falsifier samples only them)."""
    from numpy.lib.format import open_memmap
    t_start = time.time()
    base = base or X.HUB + store
    src, meta, spec, arr, g = X.open_store(base, store)
    H, W, C = spec["H"], spec["W"], spec["C"]
    fs = spec["frame_seconds"]
    offset, chans = mt.physical(spec)
    times = X.frame_times(arr, spec)
    yrs, pres, poss, lo, hi = X.calendar_tables(times, fs)
    Y = len(yrs)
    if int(pres.max()) > 255:
        raise SystemExit(f"{store}: {pres.max()} frames in one month — the "
                         f"native count is uint8")
    pm = pool_map(spec, deg)
    os.makedirs(os.path.join(out, "_spec"), exist_ok=True)
    os.makedirs(os.path.join(out, "native"), exist_ok=True)
    os.makedirs(os.path.join(out, "pooled025"), exist_ok=True)
    X.write_json(os.path.join(out, "_spec", "tile_grid.json"), spec)
    mspec = mt.make_spec(spec)
    X.write_json(os.path.join(out, "native", "tile_grid.json"), mspec)
    pshape = (12, C, Y, pm["Hp"], pm["Wp"])
    for k, d in POOLED:
        open_memmap(os.path.join(out, "pooled025", f"{k}.npy"), mode="w+",
                    dtype=d, shape=pshape).flush()
    by_ym = {}
    for b, f, t in times:
        by_ym.setdefault((t.year, t.month), []).append((b, f))
    seq = sorted(by_ym)
    boxes = BOXES.get(store, []) if boxes is None else boxes
    fmonths, bmonths = keep_months_of(store, yrs, pres, boxes)
    if only:
        only = {(int(y), int(m)) for y, m in only}
        seq = [k for k in seq if k in only]
        fmonths = [k for k in fmonths if k in only] or seq[:1]
        boxes = [(y, tuple(m for m in ms if (y, m) in only), bx)
                 for y, ms, bx in boxes]
        boxes = [b for b in boxes if b[1]]
        bmonths = [k for k in bmonths if k in only]
    keep_set = set(fmonths) | set(bmonths)
    workers = auto_workers(workers, spec)
    if months_per_job is None:
        # >= 4 jobs per worker: a job is its months in sequence, so long jobs
        # leave the last few running alone on one core (run #1's ACSPO tail:
        # 10-month jobs at ~400 s a month). The price is one bin downloaded
        # twice per job boundary (~5 % at 3 months a job).
        months_per_job = int(np.clip(len(seq) // max(1, 4 * workers), 1, 12))
    runs = [seq[i:i + months_per_job] for i in range(0, len(seq),
                                                     months_per_job)]
    jobs = [(base, g, [(y, m, by_ym[(y, m)]) for y, m in r], out, keep,
             keep_set, meta.get("sha256") or {}, threads, offset, pm, yrs[0],
             None) for r in runs]
    print(f"[{store}] {len(times)} frames in {len(seq)} months, "
          f"{yrs[0]}–{yrs[-1]}; {len(jobs)} job(s) of ≤ {months_per_job} "
          f"month(s) on {workers} worker(s); pooled {pm['Hp']} x {pm['Wp']} "
          f"({pooled_grid_record(pm)['block_note']}), "
          f"{12 * C * Y * pm['Hp'] * pm['Wp'] * 10 / 1e9:.2f} GB", flush=True)
    done, nbytes = [], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(run_worker, jobs):
            done += r["months"]
            nbytes += r["bytes"]
            print(f"  [{store}] {len(done)}/{len(seq)} months · "
                  f"{nbytes / 1e9:.1f} GB read · "
                  f"{(time.time() - t_start) / 60:.1f} min", flush=True)
    done.sort(key=lambda e: (e["year"], e["month"]))
    if [(e["year"], e["month"]) for e in done] != seq or not seq:
        raise SystemExit(f"{store}: months written != months in the record")
    for e in done:
        yi = e["year"] - yrs[0]
        if e["frames"] != pres[yi, e["month"] - 1]:
            raise SystemExit(f"{store} {e['year']}-{e['month']}: frames "
                             f"{e['frames']} != calendar")
    blk = meta.get("sha256") or {}
    digest = hashlib.sha256(json.dumps(blk, sort_keys=True)
                            .encode()).hexdigest()
    st = dict(
        _source="ml/export_fine_monthly.py — do not hand-edit",
        plan=PLAN_URL, store=store, group=g, title=meta.get("title"),
        licence=meta.get("licence"), chans=[c["name"] for c in chans],
        channels=chans, units={c["name"]: c["unit"] for c in chans},
        value_offset=offset, dtype_source=spec["dtype"],
        source_channels=spec["channels"], frame_seconds=fs,
        frames_per_bin=spec["frames_per_bin"], axes=AXES,
        year_first=yrs[0], year_last=yrs[-1], n_years=Y, years=yrs,
        frames_present=pres.tolist(), frames_possible=poss.tolist(),
        max_count=int(pres.max()),
        record_first=lo.strftime("%Y-%m-%dT%H:%M:%SZ"),
        record_last=hi.strftime("%Y-%m-%dT%H:%M:%SZ"),
        month_rule=MONTH_RULE, combine=COMBINE,
        native=dict(layout=NATIVE_LAYOUT, spec="native/tile_grid.json",
                    H=H, W=W, C=C, tile=spec["tile"],
                    n_tiles_y=spec["n_tiles_y"], n_tiles_x=spec["n_tiles_x"],
                    months_columns=["year", "month", "frames", "tiles_stored",
                                    "shard_bytes"],
                    months=[[e["year"], e["month"], e["frames"],
                             e["tiles_stored"], e["shard_bytes"]]
                            for e in done],
                    shard_bytes_total=int(sum(e["shard_bytes"]
                                              for e in done))),
        pooled=dict(layout=POOLED_LAYOUT, shape=list(pshape),
                    sum_dtype="float32", count_dtype="uint16",
                    m2_dtype="float32", grid=pooled_grid_record(pm),
                    count_note=("uint16: a 0.25° cell pools up to "
                                f"{max(pm['block_rows'])} x "
                                f"{max(pm['block_cols'])} pixels x "
                                f"{int(pres.max())} frames = "
                                f"{max(pm['block_rows']) * max(pm['block_cols']) * int(pres.max())}"
                                " values, past uint8")),
        source_sha256_block_digest=digest,
        source_shard_index_sha256=blk.get(f"{g}/shard_index.npy"),
        source_bytes_read=int(nbytes),
        falsifier_months=[list(x) for x in fmonths],
        falsifier_boxes=[dict(year=y, months=list(ms), box=list(bx))
                         for y, ms, bx in boxes],
        trial_months=(sorted([list(k) for k in only]) if only else None),
        git_sha=X.git_sha(), generated_utc=X.now_utc(),
        export_seconds=round(time.time() - t_start, 1))
    shutil.rmtree(os.path.join(out, "_spec"), ignore_errors=True)
    X.write_json(os.path.join(out, "stats.json"), st)       # LAST: done-marker
    print(f"[{store}] exported in {(time.time() - t_start) / 60:.1f} min, read "
          f"{nbytes / 1e9:.1f} GB, native "
          f"{st['native']['shard_bytes_total'] / 1e9:.2f} GB", flush=True)
    return st


# ============================================================ verify ======
def _ulp_half(a):
    return 0.5 * np.spacing(np.abs(np.asarray(a, np.float32))).astype(
        np.float64)


def _check_pooled(ref_mean, ref_n, ref_std, n, mean, std, bound, what):
    return X.check(mean, n, std, bound, ref_mean, ref_n, ref_std, what)


def _direct(sum_, cnt, m2):
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, sum_ / np.maximum(cnt, 1), np.nan)
        std = np.where(cnt > 0, np.sqrt(np.maximum(m2, 0) /
                                        np.maximum(cnt, 1)), np.nan)
    return mean, std


def verify_month_planes(args):
    """Every pixel and every 0.25° cell of one (year, month), every channel:
    the native layer and the pooled layer against numpy over the month's
    native frames, read tile by tile through `sharded.ShardedGroup` from the
    kept, sha256-checked bins."""
    out, keep, g, y, m, frames, pm_rec, offset, st = args
    spec = json.load(open(os.path.join(keep, g, "tile_grid.json")))
    grp = sh.ShardedGroup(os.path.join(keep, g))
    rd = mt.MonthlyTiles(os.path.join(out, "native"))
    C, yi = spec["C"], y - st["year_first"]
    pm = dict(pm_rec, jrow=np.asarray(pm_rec["jrow"]),
              icol=np.asarray(pm_rec["icol"]))
    Hp, Wp = pm["Hp"], pm["Wp"]
    acc_s = np.zeros((Hp, Wp, C), np.float64)
    acc_n = np.zeros((Hp, Wp, C), np.int64)
    acc_b = np.zeros((Hp, Wp, C), np.float64)       # Σ ½ulp(native sum)
    nat = []
    nty, ntx = spec["n_tiles_y"], spec["n_tiles_x"]

    def stack(ty, tx):
        s = np.stack([grp.read_tile(b, f, ty, tx, crop=True)
                      for b, f in frames]).astype(np.float32)
        if offset:
            s += np.float32(offset)
        return s

    for ty in range(nty):
        r0, r1 = spec["row_extents"][ty]
        for tx in range(ntx):
            c0, c1 = spec["col_extents"][tx]
            x = stack(ty, tx)
            rm, rn, rs = X.native_stats(x)
            t = rd.read_tile(y, m, ty, tx)
            for c in range(C):
                mean, n, std, bound = X.compose(t["sum"][None, ..., c],
                                                t["count"][None, ..., c],
                                                t["m2"][None, ..., c])
                nat.append(X.check(mean, n, std, bound, rm[..., c],
                                   rn[..., c], rs[..., c], f"{c}"))
            fin = np.isfinite(x)
            px_s = np.sum(x, axis=0, where=fin, dtype=np.float64)
            px_n = fin.sum(axis=0).astype(np.int64)
            jr, ic = pm["jrow"][r0:r1], pm["icol"][c0:c1]
            add_cells(acc_s, px_s, jr, ic)
            add_cells(acc_n, px_n, jr, ic)
            add_cells(acc_b, _ulp_half(t["sum"]), jr, ic)
    with np.errstate(invalid="ignore", divide="ignore"):
        mcell = np.where(acc_n > 0, acc_s / np.maximum(acc_n, 1), 0.0)
    acc_q = np.zeros((Hp, Wp, C), np.float64)
    for ty in range(nty):
        r0, r1 = spec["row_extents"][ty]
        for tx in range(ntx):
            c0, c1 = spec["col_extents"][tx]
            x = stack(ty, tx).astype(np.float64)
            jr, ic = pm["jrow"][r0:r1], pm["icol"][c0:c1]
            d = x - mcell[np.ix_(jr, ic)][None]
            q = np.sum(d * d, axis=0, where=np.isfinite(x))
            add_cells(acc_q, q, jr, ic)
    P = {k: np.load(os.path.join(out, "pooled025", f"{k}.npy"),
                    mmap_mode="r") for k, _ in POOLED}
    rm, rs = _direct(acc_s, acc_n, acc_q)
    res = []
    for c, name in enumerate(st["chans"]):
        sel = [r for r in nat[c::C]]
        nr = dict(what=f"{y}-{m:02d} {name} native", layer="native", year=y,
                  month=m, channel=name, frames=len(frames),
                  ok=all(r["ok"] for r in sel),
                  why=sum((r["why"] for r in sel), []),
                  n_cells=int(sum(r["n_cells"] for r in sel)))
        for k in ("max_abs_mean", "max_mean_over_bound", "max_abs_std",
                  "max_std_over_tol"):
            nr[k] = float(max([r.get(k, 0.0) for r in sel] or [0.0]))
        res.append(nr)
        Sp = np.asarray(P["sum"][m - 1, c, yi])
        Np = np.asarray(P["count"][m - 1, c, yi])
        Mp = np.asarray(P["m2"][m - 1, c, yi])
        mean, n, std, bound = X.compose(Sp[None], Np[None], Mp[None])
        bound = bound + acc_b[..., c] / np.maximum(n, 1)
        r = X.check(mean, n, std, bound, rm[..., c], acc_n[..., c],
                    rs[..., c], f"{y}-{m:02d} {name} pooled")
        r.update(layer="pooled", year=y, month=m, channel=name,
                 frames=len(frames))
        res.append(r)
    return res


def _region(pm, box):
    """Pooled cell rows / cols (relative) and native rows / cols of a box."""
    lat_lo, lat_hi, lon_lo, lon_hi = box
    deg = pm["deg"]
    J0 = int(round((lat_lo + 90) / deg)) - pm["j0"]
    J1 = int(round((lat_hi + 90) / deg)) - pm["j0"]
    I0 = int(round((lon_lo + 180) / deg)) - pm["i0"]
    I1 = int(round((lon_hi + 180) / deg)) - pm["i0"]
    J0, I0 = max(J0, 0), max(I0, 0)
    J1, I1 = min(J1, pm["Hp"]), min(I1, pm["Wp"])
    if J1 <= J0 or I1 <= I0:
        raise SystemExit(f"box {box} is outside the pooled grid")
    rr = [pm["rows"][j] for j in range(J0, J1)]
    cc = [pm["cols"][i] for i in range(I0, I1)]
    r0, r1 = min(a for a, _ in rr), max(b for _, b in rr)
    c0, c1 = min(a for a, _ in cc), max(b for _, b in cc)
    return (J0, J1, I0, I1), (r0, r1, c0, c1)


def verify_box(args):
    """A period box (a whole year or a season) composed from its monthly
    planes in BOTH layers, against numpy over every native frame of the
    period in the box."""
    out, keep, g, year, months, box, fr_by_month, pm_rec, offset, st = args
    spec = json.load(open(os.path.join(keep, g, "tile_grid.json")))
    grp = sh.ShardedGroup(os.path.join(keep, g))
    rd = mt.MonthlyTiles(os.path.join(out, "native"))
    pm = dict(pm_rec, jrow=np.asarray(pm_rec["jrow"]),
              icol=np.asarray(pm_rec["icol"]))
    (J0, J1, I0, I1), (r0, r1, c0, c1) = _region(pm, box)
    T, C = spec["tile"], spec["C"]
    tys = range(r0 // T, (r1 - 1) // T + 1)
    txs = range(c0 // T, (c1 - 1) // T + 1)

    def region(get):
        a = None
        for ty in tys:
            for tx in txs:
                t = get(ty, tx)
                if a is None:
                    a = np.full((r1 - r0, c1 - c0) + t.shape[2:], np.nan,
                                t.dtype if t.dtype.kind == "f" else
                                np.float64)
                R0, R1 = spec["row_extents"][ty]
                C0, C1 = spec["col_extents"][tx]
                ra, rb = max(R0, r0), min(R1, r1)
                ca, cb = max(C0, c0), min(C1, c1)
                a[ra - r0:rb - r0, ca - c0:cb - c0] = \
                    t[ra - R0:rb - R0, ca - C0:cb - C0]
        return a

    ms = [mo for mo in months if fr_by_month.get(mo)]
    frames = [bf for mo in ms for bf in fr_by_month[mo]]
    x = np.stack([region(lambda ty, tx, b=b, f=f:
                         grp.read_tile(b, f, ty, tx, crop=True))
                  for b, f in frames]).astype(np.float32)
    if offset:
        x += np.float32(offset)
    rm, rn, rsd = X.native_stats(x)
    nat = {k: [] for k in mt.PART_NAMES}
    for mo in ms:
        for k in mt.PART_NAMES:
            nat[k].append(region(lambda ty, tx, k=k:
                                 rd.read_tile(year, mo, ty, tx)[k]
                                 .astype(np.float64)))
    yi = year - st["year_first"]
    P = {k: np.load(os.path.join(out, "pooled025", f"{k}.npy"),
                    mmap_mode="r") for k, _ in POOLED}
    jr, ic = pm["jrow"][r0:r1] - J0, pm["icol"][c0:c1] - I0
    keep_r = (jr >= 0) & (jr < J1 - J0)
    keep_c = (ic >= 0) & (ic < I1 - I0)
    xs = x[:, keep_r][:, :, keep_c].astype(np.float64)
    jr, ic = jr[keep_r], ic[keep_c]
    fin = np.isfinite(xs)
    hp, wp = J1 - J0, I1 - I0
    cs = np.zeros((hp, wp, C))
    cn = np.zeros((hp, wp, C), np.int64)
    add_cells(cs, np.sum(xs, axis=0, where=fin), jr, ic)
    add_cells(cn, fin.sum(axis=0).astype(np.int64), jr, ic)
    with np.errstate(invalid="ignore", divide="ignore"):
        mc = np.where(cn > 0, cs / np.maximum(cn, 1), 0.0)
    d = xs - mc[np.ix_(jr, ic)][None]
    cq = np.zeros((hp, wp, C))
    add_cells(cq, np.sum(d * d, axis=0, where=fin), jr, ic)
    pmean, pstd = _direct(cs, cn, cq)
    ub = np.zeros((hp, wp, C))
    for a in nat["sum"]:
        add_cells(ub, _ulp_half(a[keep_r][:, keep_c]), jr, ic)
    res = []
    for c, name in enumerate(st["chans"]):
        mean, n, std, bound = X.compose(np.stack([a[..., c]
                                                  for a in nat["sum"]]),
                                        np.stack([a[..., c]
                                                  for a in nat["count"]]),
                                        np.stack([a[..., c]
                                                  for a in nat["m2"]]))
        r = X.check(mean, n, std, bound, rm[..., c], rn[..., c],
                    rsd[..., c], f"{year} {ms} box {name} native")
        r.update(layer="native", year=year, months=ms, box=list(box),
                 channel=name, frames=len(frames))
        res.append(r)
        Sp = np.stack([np.asarray(P["sum"][mo - 1, c, yi, J0:J1, I0:I1])
                       for mo in ms])
        Np = np.stack([np.asarray(P["count"][mo - 1, c, yi, J0:J1, I0:I1])
                       for mo in ms])
        Mp = np.stack([np.asarray(P["m2"][mo - 1, c, yi, J0:J1, I0:I1])
                       for mo in ms])
        mean, n, std, bound = X.compose(Sp, Np, Mp)
        bound = bound + ub[..., c] / np.maximum(n, 1)
        r = X.check(mean, n, std, bound, pmean[..., c], cn[..., c],
                    pstd[..., c], f"{year} {ms} box {name} pooled")
        r.update(layer="pooled", year=year, months=ms, box=list(box),
                 channel=name, frames=len(frames))
        res.append(r)
    return res


def blocksum_month(args):
    """The pooled layer of one (year, month) against the NATIVE layer
    block-summed by an independent route (whole-array reduceat on the
    published native file): sum and count BIT-EXACT, the std of m2 within the
    tolerance (m2 here by the algebraic form Σm2 + Σs²/n − S²/N); also sum = 0
    and m2 = 0 wherever the native count is 0."""
    out, y, m, pm_rec, st = args
    rd = mt.MonthlyTiles(os.path.join(out, "native"))
    a = rd.read_month(y, m)
    S, N, M = a["sum"], a["count"], a["m2"]
    if (S[N == 0] != 0).any() or (M[N <= 1] != 0).any():
        return dict(year=y, month=m, ok=False,
                    why=["a nonzero sum or m2 where the count is 0 / 1"])
    rows = pm_rec["rows"]
    rs = np.array(sorted(a_ for a_, _ in rows), np.int64)
    flip = pm_rec["north_first"]
    cs = np.array([a_ for a_, _ in pm_rec["cols"]], np.int64)
    P = {k: np.load(os.path.join(out, "pooled025", f"{k}.npy"),
                    mmap_mode="r") for k, _ in POOLED}
    yi = y - st["year_first"]
    r = dict(year=y, month=m, ok=True, why=[], max_abs_sum=0.0,
             max_std_over_tol=0.0)
    for c in range(S.shape[2]):
        def bs(v):
            b = np.add.reduceat(np.add.reduceat(v, rs, axis=0), cs, axis=1)
            return b[::-1] if flip else b
        s64 = S[..., c].astype(np.float64)
        n64 = N[..., c].astype(np.int64)
        Sb, Nb = bs(s64), bs(n64)
        with np.errstate(invalid="ignore", divide="ignore"):
            q = np.where(n64 > 0, s64 * s64 / np.maximum(n64, 1), 0.0)
        Mb = bs(M[..., c].astype(np.float64)) + bs(q) - np.where(
            Nb > 0, Sb * Sb / np.maximum(Nb, 1), 0.0)
        ps = np.asarray(P["sum"][m - 1, c, yi])
        pn = np.asarray(P["count"][m - 1, c, yi]).astype(np.int64)
        pq = np.asarray(P["m2"][m - 1, c, yi]).astype(np.float64)
        if not np.array_equal(Sb.astype(np.float32), ps):
            r["ok"] = False
            r["why"].append(f"channel {c}: pooled sum is not the native "
                            f"block sum in {int((Sb.astype(np.float32) != ps).sum())} cells")
        if not np.array_equal(Nb, pn):
            r["ok"] = False
            r["why"].append(f"channel {c}: pooled count is not the native "
                            f"block count")
        r["max_abs_sum"] = max(r["max_abs_sum"], float(np.abs(
            Sb.astype(np.float32).astype(np.float64) - ps).max(initial=0)))
        fin = pn > 0
        if fin.any():
            mean = Sb[fin] / pn[fin]
            sa = np.sqrt(np.maximum(Mb[fin], 0) / pn[fin])
            sb = np.sqrt(np.maximum(pq[fin], 0) / pn[fin])
            tol = STD_REL_TOL * (np.abs(mean) + sa) + 1e-12
            rat = float((np.abs(sa - sb) / tol).max())
            r["max_std_over_tol"] = max(r["max_std_over_tol"], rat)
            if rat > 1:
                r["ok"] = False
                r["why"].append(f"channel {c}: pooled m2 off the native "
                                f"Chan combination ({rat:.2f} x tol)")
    return r


def verify(store, out, keep, report=None, base=None, workers=4):
    """THE FALSIFIER (plan §5). Raises SystemExit on any failure, after
    writing the report."""
    st = json.load(open(os.path.join(out, "stats.json"), encoding="utf-8"))
    g = st["group"]
    base = base or X.HUB + store
    _, _, spec, arr, _ = X.open_store(base, store)
    gd = os.path.join(keep, g)
    os.makedirs(gd, exist_ok=True)
    with open(os.path.join(gd, "tile_grid.json"), "w") as fh:
        json.dump(spec, fh)
    pm = pool_map(spec, st["pooled"]["grid"]["deg"])
    if pm["rows"] != st["pooled"]["grid"]["rows"] or \
            pm["cols"] != st["pooled"]["grid"]["cols"]:
        raise SystemExit("the pool map differs from stats.json's")
    pm_rec = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
              for k, v in pm.items()}
    by = {}
    for b, f, t in X.frame_times(arr, spec):
        by.setdefault((t.year, t.month), []).append((b, f))
    off = st["value_offset"]
    t0 = time.time()
    res = dict(store=store, generated_utc=X.now_utc(), std_rel_tol=STD_REL_TOL,
               planes=[], boxes=[], blocksum=[], ok=True, failures=[])
    jobs = [(out, keep, g, y, m, by[(y, m)], pm_rec, off, st)
            for y, m in st["falsifier_months"]]
    bjobs = [(out, keep, g, bx["year"], bx["months"], tuple(bx["box"]),
              {mo: by.get((bx["year"], mo), []) for mo in bx["months"]},
              pm_rec, off, st) for bx in st["falsifier_boxes"]]
    kjobs = [(out, y, m, pm_rec, st) for y, m, *_ in st["native"]["months"]]
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        for rr in ex.map(verify_month_planes, jobs):
            res["planes"] += rr
        for rr in ex.map(verify_box, bjobs):
            res["boxes"] += rr
        for rr in ex.map(blocksum_month, kjobs, chunksize=4):
            res["blocksum"].append(rr)
    for r in res["planes"] + res["boxes"] + res["blocksum"]:
        if not r["ok"]:
            res["failures"].append(r)
    res["ok"] = not res["failures"]
    for layer in ("native", "pooled"):
        sel = [r for r in res["planes"] + res["boxes"]
               if r.get("layer") == layer]
        res[layer] = {k: float(max([r.get(k, 0.0) for r in sel] or [0.0]))
                      for k in ("max_abs_mean", "max_mean_over_bound",
                                "max_abs_std", "max_std_over_tol")}
        res[layer]["n_checks"] = len(sel)
    res["blocksum_summary"] = dict(
        months=len(res["blocksum"]),
        max_abs_sum=float(max([r["max_abs_sum"] for r in res["blocksum"]
                               if "max_abs_sum" in r] or [0.0])),
        max_std_over_tol=float(max([r.get("max_std_over_tol", 0.0)
                                    for r in res["blocksum"]] or [0.0])))
    res["seconds"] = round(time.time() - t0, 1)
    if report:
        X.write_json(report, res)
    nv, pv, bk = res["native"], res["pooled"], res["blocksum_summary"]
    print(f"  verify [{store}]: {len(jobs)} months x {len(st['chans'])} ch "
          f"(every pixel and pooled cell) + {len(bjobs)} boxes + "
          f"{bk['months']} block sums · NATIVE max |Δmean| "
          f"{nv['max_abs_mean']:.3g} (/bound {nv['max_mean_over_bound']:.3f})"
          f" max |Δstd| {nv['max_abs_std']:.3g} (/tol "
          f"{nv['max_std_over_tol']:.3f}) · POOLED max |Δmean| "
          f"{pv['max_abs_mean']:.3g} (/bound {pv['max_mean_over_bound']:.3f})"
          f" max |Δstd| {pv['max_abs_std']:.3g} (/tol "
          f"{pv['max_std_over_tol']:.3f}) · BLOCK SUM max |Δsum| "
          f"{bk['max_abs_sum']:.3g}, std/tol {bk['max_std_over_tol']:.3f} · "
          f"{'OK' if res['ok'] else 'FAIL'} in {res['seconds']:.0f} s",
          flush=True)
    if not res["ok"]:
        raise SystemExit(f"VERIFY FAILED for {store}: "
                         f"{[f.get('what', (f.get('year'), f.get('month'))) for f in res['failures']][:10]}")
    return res


# =============================================================== run ======
def hub_has(store, src_sha):
    try:
        st = json.loads(X.Source(X.HUB + store + "/monthly").get("stats.json"))
        return (st.get("source_sha256_block_digest") == src_sha
                and st.get("_source", "").startswith("ml/export_fine_monthly")
                and not st.get("trial_months"))
    except (SystemExit, OSError, ValueError):
        return False


def run(stores, out, *, workers=8, threads=4, upload=True, verify_workers=4):
    """The box: per store export -> verify -> upload + read-back -> free."""
    import publish_fine_monthly_index as P
    summary = {}
    for store in stores:
        meta = json.loads(X.Source(X.HUB + store).get("store.json"))
        digest = hashlib.sha256(json.dumps(meta.get("sha256") or {},
                                           sort_keys=True).encode()).hexdigest()
        if hub_has(store, digest):
            print(f"[{store}] already published from this source — skipped",
                  flush=True)
            summary[store] = "skipped (already on the Hub)"
            continue
        d = os.path.join(out, X.store_name(store))
        keep = os.path.join(out, "_keep_" + X.store_name(store))
        shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(keep, ignore_errors=True)
        t0 = time.time()
        st = export(store, d, workers=workers, threads=threads, keep=keep)
        t1 = time.time()
        rep = verify(store, d, keep, report=os.path.join(d, "verify.json"),
                     workers=verify_workers)
        t2 = time.time()
        shutil.rmtree(keep, ignore_errors=True)
        up = None
        if upload:
            up = P.upload_store(store, d)
            shutil.rmtree(d, ignore_errors=True)
        summary[store] = dict(
            years=[st["year_first"], st["year_last"]],
            bytes_read=st["source_bytes_read"],
            native_bytes=st["native"]["shard_bytes_total"],
            minutes_export=round((t1 - t0) / 60, 1),
            minutes_verify=round((t2 - t1) / 60, 1),
            minutes_upload=round((time.time() - t2) / 60, 1),
            verify_ok=rep["ok"], native=rep["native"], pooled=rep["pooled"],
            blocksum=rep["blocksum_summary"],
            uploaded_bytes=(up or {}).get("bytes"),
            readback_ok=(up or {}).get("readback_ok"))
        X.write_json(os.path.join(out, "run_summary.json"), summary)
    return summary


# =========================================================== fixture ======
FIXTURE_OUT = os.path.join(ROOT, "data", "gridded_monthly", "fixture_fine")


def run_fixture(out=FIXTURE_OUT):
    """Two tiny stores built with the real tier-G framework
    (`tests/make_fine_monthly_fixture.py`), export -> verify -> index --local,
    plus expected.json (numpy's answers for the reader's tests)."""
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import make_fine_monthly_fixture as F
    import publish_fine_monthly_index as P
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    src_dir = tempfile.mkdtemp(prefix="fm_fix_")
    try:
        stores = F.build(src_dir)
        for store, base in stores.items():
            d = os.path.join(out, X.store_name(store))
            keep = os.path.join(src_dir, "_keep_" + X.store_name(store))
            export(store, d, base=base, workers=2, threads=2, keep=keep,
                   boxes=F.BOXES[store], deg=F.DEG)
            verify(store, d, keep, report=os.path.join(d, "verify.json"),
                   base=base, workers=2)
        rel = os.path.relpath(out, ROOT)
        P.main(["index", "--local", "--out", out, "--stores",
                ",".join(stores), "--no-cors", "--fixture-dir", rel + "/",
                "--index", os.path.join(out, "gridded_monthly_fine_index.json")])
        F.write_expected(stores, out)
    finally:
        shutil.rmtree(src_dir, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("export", "verify", "run"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--verify-workers", type=int, default=4)
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--base", default="")
        if name == "run":
            p.add_argument("--stores", default=",".join(FINE))
            p.add_argument("--no-upload", action="store_true")
        else:
            p.add_argument("--store", required=True)
            p.add_argument("--keep", default="")
            p.add_argument("--only", default="",
                           help="export: comma list of YYYY-MM (a trial)")
    sub.add_parser("fixture")
    a = ap.parse_args(argv)
    if a.cmd == "fixture":
        run_fixture()
    elif a.cmd == "export":
        only = [tuple(int(v) for v in x.split("-"))
                for x in a.only.split(",") if x] or None
        export(a.store, a.out, base=a.base or None, workers=a.workers,
               threads=a.threads, keep=a.keep or None, only=only)
    elif a.cmd == "verify":
        verify(a.store, a.out, a.keep, report=os.path.join(a.out,
                                                           "verify.json"),
               base=a.base or None, workers=a.verify_workers)
    else:
        run([s for s in a.stores.split(",") if s], a.out, workers=a.workers,
            threads=a.threads, upload=not a.no_upload,
            verify_workers=a.verify_workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
