#!/usr/bin/env python3
"""Writes `data/family1_fixture/` — the stores `tests/f1data.test.mjs` reads (E-084 §6).

    python3 tests/make_family1_fixture.py          # rewrites the fixture

PLAIN ENGLISH. `src/f1data.js` reads family 1.gf's stores straight from the
public Hub in the browser. To test it without the network, this script builds
three tiny stores with the REAL builders — the same `index | fetch | assemble`
stages `ml/build_family1_stores.py` runs on a box, driven by three toy
adapters — so the bytes the reader is tested on are bytes the real writers
produced (`ml/family1/sharded.py` for the grids, the family-10 column
assembler for the points), not a JavaScript author's idea of them:

  fxgrid   float16, 2 channels, a GLOBAL 0.5° grid (360 x 720, row 0 north)
           cut into 32 x 32 tiles; 3 five-day bins (2009-12-25 .. 2010-01-08,
           so a year AND a month boundary fall inside); 2010-01-05 is absent
           upstream (index offset -1); one tile of 2009-12-27 is present and
           empty (length 0); ~30 % of the pixels valid, all of them inside two
           patches — one of them across the dateline — so the empty ocean
           costs nothing and the fixture stays small.
  fxtb     uint8 "K - 160" like `irtb`: 40 three-hourly frames per bin, a
           tropical band whose row 0 is the SOUTHERN row (dy > 0), one absent
           frame.
  fxpts    a tier-P store (schema 2), ~5,000 rows over 2009-12-20 ..
           2010-02-10 across many bins, platform ids above 2^53 so the reader
           has to keep them exact.

and a registry `family1gf.json` in the live registry's shape (a `groups` LIST;
plus one unbuilt and one private entry the reader must skip), and
`expected.json`: selections and their ground truth computed here with numpy,
INDEPENDENTLY of the JavaScript (pixel centres, dateline order, nanmean and
finite counts, coarse cells aligned to whole multiples of the resolution).

Deterministic (fixed seeds); < 2 MB in all. Needs numpy and zstandard.
"""
import argparse
import datetime as dt
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

import build_family10_stores as b10                             # noqa: E402
import build_family1_stores as b1                               # noqa: E402
import family10_store as f10                                    # noqa: E402
from family1 import sharded as sh                               # noqa: E402

OUT = os.path.join(ROOT, "data", "family1_fixture")
EPOCH = dt.datetime(1982, 1, 1)
EPOCH_UNIX = 378691200
BIN_S = 432000
BINS = (2044, 2045, 2046)            # 2009-12-25 .. 2010-01-08
WINDOW = ("2009-12-25", "2010-01-08")


def lic():
    return {"name": "test fixture", "redistribution": "yes",
            "derived_works": "free"}


def geo(H, W, x0, y0, dx, dy, row_order):
    return {"H": H, "W": W, "x0": x0, "y0": y0, "dx": dx, "dy": dy,
            "crs": "EPSG:4326", "row_order": row_order,
            "coordinates": "lon = x0 + (col + 0.5) * dx, lat = y0 + (row + "
                           "0.5) * dy"}


# ================================================================ fxgrid ===
G_H, G_W, G_RES = 360, 720, 0.5
G_GRID = geo(G_H, G_W, -180.0, 90.0, G_RES, -G_RES,
             "row 0 is the NORTHERNMOST row")
G_CH = (("sst", "degC", -2.5, 40.0),
        ("log_chl", "log10(mg m-3) - the base-10 LOGARITHM of chlorophyll",
         -4.0, 2.5))
G_ABSENT = (2046, 1)                 # 2010-01-05
G_EMPTY = (2044, 2)                  # 2009-12-27: one tile present and empty
G_TILE = 32


def g_centres():
    lat = 90.0 - (np.arange(G_H) + 0.5) * G_RES
    lon = -180.0 + (np.arange(G_W) + 0.5) * G_RES
    return lat, lon


def g_frame(b, f):
    """The float32 field handed to the writer, or None (absent)."""
    if (b, f) == G_ABSENT:
        return None
    rng = np.random.default_rng(1000 * b + f)
    lat, lon = g_centres()
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    patch = (((LON >= -30) & (LON <= 10) & (LAT >= -20) & (LAT <= 20))
             | (((LON >= 160) | (LON <= -160)) & (LAT >= -10) & (LAT <= 10)))
    a = np.empty((G_H, G_W, 2), np.float32)
    a[..., 0] = 15 + 10 * np.cos(np.radians(LAT)) + rng.normal(0, 2, LAT.shape)
    a[..., 1] = -1 + 0.01 * LAT + rng.normal(0, 0.3, LAT.shape)
    a[..., 0] = a[..., 0].clip(-2.4, 39.9)
    a[..., 1] = a[..., 1].clip(-3.9, 2.4)
    keep = patch[..., None] & (rng.random((G_H, G_W, 2)) < 0.3)
    a[~keep] = np.nan
    if (b, f) == G_EMPTY:
        # tile (ty, tx) = (6, 4): rows 192..223 (lat -6 .. -21.75), cols
        # 128..159 (lon -116 .. -100.25) is outside the patches anyway — so
        # empty a tile INSIDE patch A instead: (ty 5, tx 10), rows 160..191,
        # cols 320..351 (lon -20 .. -4.25, lat 10 .. -5.75)
        a[160:192, 320:352] = np.nan
    return a


class FxGrid(sh.GridAdapter):
    store = "fxgrid"
    title = "fixture: a global 0.5-degree float16 grid, two channels"
    family = "1gf"
    distribution = "public"
    licence = lic()
    channels = G_CH
    log2_fp = -2.0
    log2_dt = -2.32
    first_year = 2009
    frames_per_bin = 5
    frame_seconds = 86400
    dtype = "float16"
    tile = G_TILE
    zstd_level = 3
    grid = G_GRID
    sources = ("synthetic",)
    smoke_window = WINDOW

    def index(self, ctx):
        return {"dataset": "fixture"}

    def fetch_frames(self, ctx, wanted):
        for g, b, f in wanted:
            if b not in BINS:
                why = "before_record" if b < BINS[0] else "after_record"
                yield g, b, f, None, {"frame_missing": why}
                continue
            a = g_frame(b, f)
            if a is None:
                yield g, b, f, None, {"frame_missing": "absent_upstream"}
            else:
                yield g, b, f, a, {"files_read": 1}


# ================================================================== fxtb ===
T_H, T_W, T_RES = 60, 720, 0.5
T_GRID = geo(T_H, T_W, -180.0, -15.0, T_RES, T_RES,
             "row 0 is the SOUTHERNMOST row")
T_CH = (("tb", "K - 160 (uint8; add 160 for kelvin)", 0.0, 254.0),)
T_BINS = (2045, 2046)
T_ABSENT = (2045, 7)


def t_frame(b, f):
    if b not in T_BINS or (b, f) == T_ABSENT:
        return None
    rng = np.random.default_rng(7000 * b + f)
    lat = -15.0 + (np.arange(T_H) + 0.5) * T_RES
    lon = -180.0 + (np.arange(T_W) + 0.5) * T_RES
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    patch = (LON >= 0) & (LON <= 10) & (LAT >= -5) & (LAT <= 5)
    a = np.rint(100 + 40 * np.sin(f / 6.0) + rng.normal(0, 20, LAT.shape))
    a = a.clip(0, 254).astype(np.float32)
    a[~(patch & (rng.random(LAT.shape) < 0.5))] = np.nan
    return a[..., None]


class FxTb(sh.GridAdapter):
    store = "fxtb"
    title = "fixture: a uint8 K-160 three-hourly tropical band"
    family = "1gf"
    distribution = "public"
    licence = lic()
    channels = T_CH
    log2_fp = -2.0
    log2_dt = -7.9
    first_year = 2009
    frames_per_bin = 40
    frame_seconds = 10800
    dtype = "uint8"
    tile = G_TILE
    zstd_level = 3
    grid = T_GRID
    sources = ("synthetic",)
    smoke_window = ("2009-12-30", "2010-01-08")

    def index(self, ctx):
        return {"dataset": "fixture"}

    def fetch_frames(self, ctx, wanted):
        for g, b, f in wanted:
            if b not in T_BINS:
                why = "before_record" if b < T_BINS[0] else "after_record"
                yield g, b, f, None, {"frame_missing": why}
                continue
            a = t_frame(b, f)
            if a is None:
                yield g, b, f, None, {"frame_missing": "absent_upstream"}
            else:
                yield g, b, f, a, {"files_read": 1}


# ================================================================= fxpts ===
P_CH = (("temp", "degC", -2.5, 40.0), ("sal", "PSS-78", 0.0, 42.0),
        ("oxy", "umol/kg", 0.0, 600.0))
P_N = 5000
P_LO = (dt.date(2009, 12, 20) - EPOCH.date()).days * 86400
P_HI = (dt.date(2010, 2, 10) - EPOCH.date()).days * 86400 + 86399


def p_rows():
    rng = np.random.default_rng(20261005)
    t = rng.integers(P_LO, P_HI + 1, P_N).astype(np.int64)
    kind = rng.random(P_N)
    lat = rng.uniform(-60, 60, P_N)
    lon = rng.uniform(-180, 180, P_N)
    near = kind < 0.4                                  # the box cluster
    lon[near] = rng.uniform(-35, 15, near.sum())
    lat[near] = rng.uniform(-25, 25, near.sum())
    dl = (kind >= 0.4) & (kind < 0.8)                  # across the dateline
    lon[dl] = rng.uniform(165, 195, dl.sum())
    lon[lon >= 180] -= 360
    v = np.stack([rng.uniform(0, 30, P_N), rng.uniform(30, 38, P_N),
                  rng.uniform(150, 350, P_N)], axis=1)
    v[rng.random(v.shape) < 0.2] = np.nan
    v[np.isnan(v).all(axis=1), 0] = 10.0               # no row without a value
    platform = (np.int64(1) << 60) + rng.integers(0, 40, P_N)
    qc = rng.integers(0, 3, P_N)
    return t, lat, lon, v, platform, qc


class FxPts(b10.SourceAdapter):
    store = "fxpts"
    title = "fixture: ship-like point reports, three channels"
    family = "1gf"
    distribution = "public"
    licence = lic()
    channels = P_CH
    per_year = True
    first_year = 2009
    sources = ("synthetic",)
    smoke_window = ("2009-12-20", "2010-02-10")

    def index(self, ctx):
        return {"dataset": "fixture"}

    def fetch_year(self, ctx, year):
        t, lat, lon, v, platform, qc = p_rows()
        y0 = (dt.date(int(year), 1, 1) - EPOCH.date()).days * 86400
        y1 = (dt.date(int(year) + 1, 1, 1) - EPOCH.date()).days * 86400
        k = (t >= max(y0, ctx.t_lo)) & (t < y1) & (t <= ctx.t_hi)
        rows = self.pack(t[k], lat[k], lon[k], v[k], platform[k], qc[k])
        yield str(year), rows, {"rows": int(k.sum())}


# =============================================================== builders ==
def ns(cls, tmp):
    d = dict(store=cls.store, work=os.path.join(tmp, "work"),
             source_dir=os.path.join(tmp, "src"), start=cls.smoke_window[0],
             end=cls.smoke_window[1], stage="all", force=False, attempts=1,
             qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS,
             assemble="auto", parts_from_hub=False, push_parts=False,
             allow_missing_years=False, allow_unconfirmed_licence=False,
             probe_month="", smoke=False)
    os.makedirs(d["source_dir"], exist_ok=True)
    return argparse.Namespace(**d)


def build_grid(cls, tmp):
    ad = cls()
    ctx = b10.Ctx(ns(cls, tmp), adapter=ad, layout=b1.layout_for(ad))
    ctx = b1.prepare_grid_ctx(ctx)
    b10.run_stages(ctx, ["index", "fetch", "assemble", "check"],
                   stage_fn=b1.GRID_STAGE_FN, deps=b1.DEPS)
    return ctx.store


def build_points(cls, tmp):
    ad = cls()
    ctx = b10.Ctx(ns(cls, tmp), adapter=ad, layout=b1.layout_for(ad))
    b10.run_stages(ctx, ["index", "fetch", "assemble"], stage_fn=b1.STAGE_FN,
                   deps=b1.DEPS)
    b1.stage_check(ctx)
    return ctx.store


# ============================================================ references ===
def ts_of(b, f, fs):
    return b * BIN_S + f * fs


def utc(t82):
    return EPOCH + dt.timedelta(seconds=int(t82))


def month_key(t82):
    d = utc(t82)
    return d.year * 12 + d.month - 1


def month_start_unix(key):
    y, m = divmod(key, 12)
    return int((dt.datetime(y, m + 1, 1) - dt.datetime(1970, 1, 1))
               .total_seconds())


def hour_ok(h, hours):
    if hours is None:
        return True
    h0, h1 = hours
    if h0 == h1:
        return False
    return (h0 <= h < h1) if h0 < h1 else (h >= h0 or h < h1)


def sel_frames(frames, fs, F, bins, sel):
    """[(b, f, t82)] of the present frames the selection keeps, in order."""
    out = []
    for b in bins:
        for f in range(F):
            if frames(b, f) is None:
                continue
            t = ts_of(b, f, fs)
            d = utc(t)
            if not (sel["yearStart"] <= d.year <= sel["yearEnd"]):
                continue
            if d.month not in sel["months"]:
                continue
            if fs < 86400 and not hour_ok(d.hour, sel.get("hours")):
                continue
            out.append((b, f, t))
    return out


def axes(lat_c, lon_c, box):
    w, s, e, n = box["w"], box["s"], box["e"], box["n"]
    rows = np.where((lat_c >= s) & (lat_c <= n))[0]
    rows = rows[np.argsort(lat_c[rows])]
    if w <= e:
        cols = np.where((lon_c >= w) & (lon_c <= e))[0]
        lonU = lon_c[cols]
    else:
        c1 = np.where(lon_c >= w)[0]
        c2 = np.where(lon_c <= e)[0]
        cols = np.concatenate([c1, c2])
        lonU = np.concatenate([lon_c[c1], lon_c[c2] + 360.0])
    return rows, cols, lonU


def grid_reference(frames, fs, F, bins, lat_c, lon_c, sel, chan_idx,
                   offset=0.0):
    keep = sel_frames(frames, fs, F, bins, sel)
    rows, cols, lonU = axes(lat_c, lon_c, sel["bbox"])
    lat = lat_c[rows]
    stack = np.stack([frames(b, f)[np.ix_(rows, cols)][..., chan_idx]
                      for b, f, _ in keep]).astype(np.float64) + offset
    stack = stack.transpose(0, 3, 1, 2)                     # [T, C, H, W]
    step, res = sel["step"], sel["res"]
    if step == "native":
        keys = list(range(len(keep)))
        tstart = [EPOCH_UNIX + t for _, _, t in keep]
    elif step == "pentad":
        keys = [b for b, _, _ in keep]
        tstart = None
    elif step == "month":
        keys = [month_key(t) for _, _, t in keep]
        tstart = None
    else:
        keys = [0] * len(keep)
        tstart = None
    uk = sorted(set(keys))
    if step == "pentad":
        tstart = [EPOCH_UNIX + k * BIN_S for k in uk]
    elif step == "month":
        tstart = [month_start_unix(k) for k in uk]
    elif step == "all":
        tstart = [EPOCH_UNIX + keep[0][2]]
    if res == "native":
        rl, cl = np.arange(len(rows)), np.arange(len(cols))
        # longitude stays MONOTONIC across the dateline: a box with w > e
        # runs w .. 180 .. e + 360 (CF allows longitudes past 180)
        out_lat, out_lon = lat, lonU
    else:
        rk = np.floor(lat / res).astype(int)
        ck = np.floor(lonU / res).astype(int)
        ur, uc = np.unique(rk), np.unique(ck)
        rl = np.searchsorted(ur, rk)
        cl = np.searchsorted(uc, ck)
        out_lat = (ur + 0.5) * res
        out_lon = (uc + 0.5) * res
    Ho, Wo = len(out_lat), len(out_lon)
    C = len(chan_idx)
    S = np.zeros((len(uk), C, Ho, Wo))
    N = np.zeros((len(uk), C, Ho, Wo), np.int64)
    for i, k in enumerate(keys):
        si = uk.index(k)
        v = stack[i]
        fin = np.isfinite(v)
        for c in range(C):
            np.add.at(S[si, c], (rl[:, None].repeat(len(cl), 1),
                                 cl[None, :].repeat(len(rl), 0)),
                      np.where(fin[c], v[c], 0.0))
            np.add.at(N[si, c], (rl[:, None].repeat(len(cl), 1),
                                 cl[None, :].repeat(len(rl), 0)),
                      fin[c].astype(np.int64))
    if step == "native" and res == "native":
        data = stack.astype(np.float32)
        count = None
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            data = (S / N).astype(np.float32)
        data[N == 0] = np.nan
        count = N
    return {"time": tstart, "lat": out_lat.tolist(), "lon": out_lon.tolist(),
            "shape": list(data.shape), "data": nanlist(data),
            "count": None if count is None else count.ravel().tolist(),
            "frames": len(keep)}


def nanlist(a):
    a = np.asarray(a, np.float64).ravel()
    return [None if not np.isfinite(x) else float(x) for x in a]


def nanmean_check(frames, fs, F, bins, lat_c, lon_c, sel, chan_idx):
    """The pentad/month/all native-resolution means once more, by
    np.nanmean over the frame stack, against grid_reference's sum/count —
    so the reference itself is checked against numpy's own nanmean."""
    keep = sel_frames(frames, fs, F, bins, sel)
    rows, cols, _ = axes(lat_c, lon_c, sel["bbox"])
    stack = np.stack([frames(b, f)[np.ix_(rows, cols)][..., chan_idx]
                      for b, f, _ in keep]).astype(np.float64)
    return stack


def point_reference(t82, lat, lon, v, platform, qc, sel, chan_idx):
    d = [utc(x) for x in t82]
    ok = np.array([sel["yearStart"] <= x.year <= sel["yearEnd"]
                   and x.month in sel["months"]
                   and hour_ok(x.hour, sel.get("hours")) for x in d], bool)
    bx = sel.get("bbox")
    if bx:
        ok &= (lat >= bx["s"]) & (lat <= bx["n"])
        if bx["w"] <= bx["e"]:
            ok &= (lon >= bx["w"]) & (lon <= bx["e"])
        else:
            ok &= (lon >= bx["w"]) | (lon <= bx["e"])
    return ok


def bins_rows(off, bf, sel):
    """The rows the reader must read: every bin overlapping a selected
    (year, month), merged — exactly what the offsets say."""
    want = set()
    for y in range(sel["yearStart"], sel["yearEnd"] + 1):
        for m in sel["months"]:
            a = int((dt.datetime(y, m, 1) - EPOCH).total_seconds())
            ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
            z = int((dt.datetime(ny, nm, 1) - EPOCH).total_seconds()) - 1
            for b in range(a // BIN_S, z // BIN_S + 1):
                want.add(b)
    n = 0
    for b in want:
        i = b - bf
        if 0 <= i < len(off) - 1:
            n += int(off[i + 1] - off[i])
    return n


# ================================================================== main ===
def registry_entry(store_dir, name, ad, tier):
    sj = json.load(open(os.path.join(store_dir, "store.json")))
    e = {"name": name, "title": ad.title, "tier": tier,
         "channels": [{"name": n, "unit": u, "min": lo, "max": hi}
                      for (n, u, lo, hi) in ad.channels],
         "C": len(ad.channels), "built": True, "distribution": "public",
         "path": f"tensors/family1_gf/{name}", "family_version": "1.gf",
         "schema_version": sj.get("schema_version", 2),
         "time_dtype": sj.get("time_dtype", "int32")}
    if tier == "G":
        e.update({"cadence": ("daily" if ad.frame_seconds == 86400
                              else "3-hourly"),
                  "frames_per_bin": ad.frames_per_bin,
                  "frame_seconds": ad.frame_seconds, "dtype": ad.dtype,
                  "N": None, "bin_first": None, "bin_last": None})
        sg = {}
        for g, G in sj["groups"].items():
            tg = json.load(open(os.path.join(store_dir, g, "tile_grid.json")))
            arr = sh.load_shard_index(os.path.join(store_dir, g,
                                                   "shard_index.npy"))
            sg[g] = {"H": tg["H"], "W": tg["W"], "C": tg["C"],
                     "tile": tg["tile"], "n_tiles_x": tg["n_tiles_x"],
                     "n_tiles_y": tg["n_tiles_y"], "dtype": tg["dtype"],
                     "frames_per_bin": tg["frames_per_bin"],
                     "frame_seconds": tg["frame_seconds"],
                     "bin_first": int(arr["bin"].min()),
                     "bin_last": int(arr["bin"].max()),
                     "bins": int(len(arr)), "bytes": int(arr["nbytes"].sum()),
                     "prefix": g, "tile_grid": f"{g}/tile_grid.json",
                     "shard_index": f"{g}/shard_index.npy",
                     "valid_fraction": [float(x) for x in
                                        arr["valid_fraction"].mean(axis=0)]}
        e["store_groups"] = sg
        first = min(v["bin_first"] for v in sg.values())
        last = max(v["bin_last"] for v in sg.values())
        e["record_span"] = [str(sh.bin_start_date(first)),
                            str(sh.bin_start_date(last)
                                + dt.timedelta(days=4))]
    else:
        e.update({"cadence": "irregular (per observation)",
                  "frames_per_bin": None, "frame_seconds": None,
                  "N": sj["N"], "bin_first": sj["bin_first"],
                  "bin_last": sj["bin_last"], "n_bins": sj["n_bins"],
                  "store_groups": None,
                  "record_span": list(ad.smoke_window)})
    return e


def main():
    tmp = tempfile.mkdtemp(prefix="f1fixture_")
    try:
        built = {"fxgrid": build_grid(FxGrid, tmp + "/g"),
                 "fxtb": build_grid(FxTb, tmp + "/t"),
                 "fxpts": build_points(FxPts, tmp + "/p")}
        if os.path.isdir(OUT):
            shutil.rmtree(OUT)
        os.makedirs(OUT)
        for name, d in built.items():
            shutil.copytree(d, os.path.join(OUT, name))
        groups = [registry_entry(os.path.join(OUT, "fxgrid"), "fxgrid",
                                 FxGrid, "G"),
                  registry_entry(os.path.join(OUT, "fxtb"), "fxtb", FxTb, "G"),
                  registry_entry(os.path.join(OUT, "fxpts"), "fxpts", FxPts,
                                 "P"),
                  # what the reader must skip: an unbuilt store and a private
                  # one (seaice_asi's two reasons on the live registry)
                  {"name": "fxunbuilt", "tier": "G", "built": False,
                   "distribution": "public", "channels": [],
                   "path": "tensors/family1_gf/fxunbuilt"},
                  {"name": "fxprivate", "tier": "P", "built": True,
                   "distribution": "private", "channels": [],
                   "path": "tensors/family1_gf/fxprivate"}]
        reg = {"family": "family1_gf", "family_version": "1.gf",
               "hf_root": "tensors/family1_gf", "repo": "fixture",
               "epoch": "1982-01-01", "pentad_days": 5,
               "generated_utc": "fixture", "groups": groups}
        with open(os.path.join(OUT, "family1gf.json"), "w") as fh:
            json.dump(reg, fh, indent=1)
        expected = make_expected()
        with open(os.path.join(OUT, "expected.json"), "w") as fh:
            json.dump(expected, fh, separators=(",", ":"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    total = 0
    for dp, _, fs in os.walk(OUT):
        for f in fs:
            total += os.path.getsize(os.path.join(dp, f))
    print(f"wrote {OUT}: {total:,} bytes")
    if total > 2_000_000:
        sys.exit("the fixture is over 2 MB")


def f16(a):
    return None if a is None else a.astype(np.float16).astype(np.float32)


def make_expected():
    cases = []
    # ---- fxgrid: truth = what the writer was given, through float16 -------
    gfr = {(b, f): f16(g_frame(b, f)) for b in BINS for f in range(5)}
    frames = lambda b, f: gfr[(b, f)]                           # noqa: E731
    # the stored bytes ARE the given values through float16 — read the
    # fixture back with the Python reader and make sure, so the truth below
    # is the store's and not merely this script's
    grp = sh.ShardedGroup(os.path.join(OUT, "fxgrid", "fxgrid"))
    for (b, f), a in gfr.items():
        got = grp.read_frame(b, f)
        if a is None:
            assert got is None, (b, f)
        else:
            assert np.array_equal(np.isnan(got), np.isnan(a)), (b, f)
            assert np.array_equal(got[~np.isnan(a)], a[~np.isnan(a)]), (b, f)
    lat_c, lon_c = g_centres()
    allm = list(range(1, 13))
    base = {"store": "fxgrid", "channels": ["sst", "log_chl"],
            "yearStart": 2009, "yearEnd": 2010, "months": allm,
            "hours": None,
            # crosses a tile column (lon -4) and a tile row (lat -6), with
            # edges that are not on pixel or cell boundaries, and holds part
            # of the tile emptied on 2009-12-27
            "bbox": {"w": -8.3, "s": -9.2, "e": 0.4, "n": -1.8},
            "step": "native", "res": "native"}

    def gcase(name, **over):
        sel = dict(base, **over)
        ci = [[c[0] for c in G_CH].index(n) for n in sel["channels"]]
        r = grid_reference(frames, 86400, 5, BINS, lat_c, lon_c, sel, ci)
        if sel["step"] != "native" and sel["res"] == "native":
            st = nanmean_check(frames, 86400, 5, BINS, lat_c, lon_c, sel, ci)
            assert st.shape[0] == r["frames"]
        cases.append({"name": name, "sel": sel, "expect": r})

    gcase("grid native box")
    gcase("grid dateline box", bbox={"w": 175, "s": -4, "e": -176, "n": 3})
    gcase("grid january only", months=[1])
    gcase("grid 2010 only", yearStart=2010, yearEnd=2010)
    gcase("grid pentad mean", step="pentad")
    gcase("grid month mean", step="month", channels=["log_chl"])
    gcase("grid all mean", step="all")
    gcase("grid res 1 native step", res=1)
    gcase("grid res 1 month mean", res=1, step="month")
    gcase("grid dateline res 1 all", bbox={"w": 175, "s": -4, "e": -176,
                                           "n": 3}, res=1, step="all")
    # an absent frame is not in `time`; the all-NaN-in-box empty tile frame is
    full = grid_reference(frames, 86400, 5, BINS, lat_c, lon_c, base, [0, 1])
    assert full["frames"] == 14

    # ---- fxtb ------------------------------------------------------------
    tfr = {(b, f): t_frame(b, f) for b in T_BINS for f in range(40)}
    tframes = lambda b, f: tfr[(b, f)]                          # noqa: E731
    tlat = -15.0 + (np.arange(T_H) + 0.5) * T_RES
    tlon = -180.0 + (np.arange(T_W) + 0.5) * T_RES
    tbase = {"store": "fxtb", "channels": ["tb"], "yearStart": 2009,
             "yearEnd": 2010, "months": allm, "hours": None,
             "bbox": {"w": 0, "s": -5, "e": 10, "n": 5}, "step": "native",
             "res": "native"}

    def tcase(name, **over):
        sel = dict(tbase, **over)
        r = grid_reference(tframes, 10800, 40, T_BINS, tlat, tlon, sel, [0],
                           offset=160.0)
        cases.append({"name": name, "sel": sel, "expect": r})

    tcase("uint8 native kelvin hours 6-12", hours=[6, 12])
    tcase("uint8 hours across midnight res 1 all", hours=[21, 3], res=1,
          step="all")
    tcase("uint8 pentad", step="pentad")

    # ---- fxpts: truth = the store read by the Python reader --------------
    st = f10.Store(os.path.join(OUT, "fxpts"))
    t82 = np.asarray(st.time_s(), np.int64)
    lat = np.asarray(st["lat"], np.float32)
    lon = np.asarray(st["lon"], np.float32)
    v = np.asarray(st["values"], np.float32)
    platform = np.asarray(st["platform"], np.int64)
    qc = np.asarray(st["qc"], np.uint8)
    off = np.load(os.path.join(OUT, "fxpts", "bin_offsets.npy"))
    bf = int(json.load(open(os.path.join(OUT, "fxpts",
                                         "store.json")))["bin_first"])
    # the store holds what the adapter gave it (sorted by time inside bins)
    gt, glat, glon, gv, gp, gq = p_rows()
    assert len(t82) == P_N
    o = np.lexsort((gt, gt // BIN_S))
    assert np.array_equal(np.sort(t82), np.sort(gt))
    assert np.allclose(np.sort(lat), np.sort(glat.astype(np.float32)))
    del o
    pbase = {"store": "fxpts", "channels": ["temp", "sal", "oxy"],
             "yearStart": 2010, "yearEnd": 2010, "months": [1],
             "hours": [6, 18], "bbox": {"w": -30, "s": -20, "e": 10, "n": 20},
             "step": "native", "res": "native"}

    def prows(name, **over):
        sel = dict(pbase, **over)
        ci = [[c[0] for c in P_CH].index(n) for n in sel["channels"]]
        ok = point_reference(t82, lat, lon, v, platform, qc, sel, ci)
        cases.append({"name": name, "sel": sel, "expect": {
            "rowsRead": bins_rows(off, bf, sel),
            "n": int(ok.sum()),
            "time": (t82[ok] + EPOCH_UNIX).tolist(),
            "lat": lat[ok].astype(np.float64).tolist(),
            "lon": lon[ok].astype(np.float64).tolist(),
            "values": nanlist(v[ok][:, ci]),
            "platform": [str(x) for x in platform[ok]],
            "qc": qc[ok].tolist()}})

    prows("points box january hours 6-18")
    prows("points dateline all months", yearStart=2009, months=allm,
          hours=None, bbox={"w": 170, "s": -60, "e": -170, "n": 60},
          channels=["oxy", "temp"])
    prows("points no box february", months=[2], hours=None, bbox=None)

    def pbinned(name, **over):
        sel = dict(pbase, **over)
        ci = [[c[0] for c in P_CH].index(n) for n in sel["channels"]]
        ok = point_reference(t82, lat, lon, v, platform, qc, sel, ci)
        res = 0.25 if sel["res"] == "native" else sel["res"]
        step = "pentad" if sel["step"] == "native" else sel["step"]
        tt, la, lo_, vv = t82[ok], lat[ok].astype(np.float64), \
            lon[ok].astype(np.float64), v[ok][:, ci].astype(np.float64)
        bx = sel["bbox"]
        if bx["w"] > bx["e"]:
            lo_ = np.where(lo_ < bx["w"], lo_ + 360, lo_)
        if step == "pentad":
            sk = tt // BIN_S
            tstart = lambda k: EPOCH_UNIX + int(k) * BIN_S      # noqa: E731
        elif step == "month":
            sk = np.array([month_key(x) for x in tt])
            tstart = month_start_unix
        else:
            sk = np.zeros(len(tt), np.int64)
            tstart = lambda k: EPOCH_UNIX + int(tt.min()) // BIN_S * BIN_S  # noqa: E731,E501
        rk = np.floor(la / res).astype(int)
        ck = np.floor(lo_ / res).astype(int)
        cells = {}
        for i in range(len(tt)):
            key = (int(sk[i]), int(rk[i]), int(ck[i]))
            cells.setdefault(key, []).append(vv[i])
        out = []
        for key in sorted(cells):
            a = np.array(cells[key])
            fin = np.isfinite(a)
            cnt = fin.sum(axis=0)
            with np.errstate(invalid="ignore"):
                mean = np.where(cnt > 0, np.nansum(a, axis=0) /
                                np.maximum(cnt, 1), np.nan)
            ol = (key[2] + 0.5) * res
            out.append({"time": tstart(key[0]),
                        "lat": (key[1] + 0.5) * res,
                        "lon": ol,
                        "mean": nanlist(mean.astype(np.float32)),
                        "count": cnt.tolist()})
        cases.append({"name": name, "sel": sel,
                      "expect": {"cells": out, "steps": sorted(
                          {c["time"] for c in out})}})

    pbinned("points binned month 1 deg", yearStart=2009, months=allm,
            hours=None, step="month", res=1, channels=["temp", "oxy"])
    pbinned("points binned pentad dateline", yearStart=2009, months=allm,
            hours=None, step="pentad", res=1,
            bbox={"w": 170, "s": -60, "e": -170, "n": 60})
    return {"note": "written by tests/make_family1_fixture.py; values are "
                    "the stores' own (float16 through float32), NaN = null",
            "epoch_unix": EPOCH_UNIX, "cases": cases}


if __name__ == "__main__":
    main()
