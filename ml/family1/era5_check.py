#!/usr/bin/env python3
"""The measurements family 1.2's ERA5 probe adds to the standard one (E-085).

PLAIN ENGLISH. `build_family1_stores.py --stage probe` puts one real month
through the real adapter and the real writer and reports frames, bytes and
seconds. For the four ERA5 stores the planning session also needs to know
whether the stored numbers are RIGHT and what the rest of the record will
cost, so this script, run beside a probe, measures:

  stats      per channel: min, max, area-weighted global mean and a
             histogram over the month, read back from a store the real
             fetch + assemble wrote (t_500 should sit near 250-260 K in a
             January, u_200 should show the jets)
  readback   one frame through `sharded.ShardedGroup` against an
             INDEPENDENT read of the same instant from the source zarr with
             xarray + zarr (no code shared with the adapter), max |diff| per
             channel, and whether every difference is within half a float16
             step of the value (the only error the store is allowed)
  seam       the two source paths at instants BOTH archives hold: the
             producer's 1-degree regrid (path A) against this adapter's own
             regrid of the 0.25-degree archive (path B), per level — the
             evidence that the record has no step at the seam
  tail       a sample of path-B frames timed through the real reader and
             the real writer: source bytes and stored bytes per frame, MB/s
  tiles      the same frames encoded with 64- and 256-pixel tiles

and merges it, with the probe's own report and the process's peak RSS, into
`ml/family1/probes/<store>_<YYYY-MM>.json`.

Run (after the probe and a `--stage index,fetch,assemble` of the month):
  python3 ml/family1/era5_check.py --store era5_t --month 2015-01 \\
      --probe <work>/probe/era5_t/2015-01.json \\
      --store-dir <build>/era5_t/era5_t/era5_t --rss <rss.json> \\
      --out ml/family1/probes/era5_t_2015-01.json
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.dirname(HERE)
sys.path.insert(0, ML)

import build_family10_stores as f10b                            # noqa: E402
from build_family7 import atomic_json                           # noqa: E402
from family1 import sharded as sh                               # noqa: E402
from family1.adapters import REGISTRY                           # noqa: E402
from family1.adapters import _era5 as e5                        # noqa: E402


def area_weights(lat):
    w = np.cos(np.radians(lat))
    w[0] = w[-1] = np.sin(np.radians(0.25))          # the half-cell poles
    return w


def stats(store_dir, ad):
    """Every frame of the local store, read back, summarised per channel."""
    g = sh.ShardedGroup(store_dir)
    lat, _lon = e5.target_axes(1.0)
    w = area_weights(lat)[:, None]
    C = ad.C
    mn = np.full(C, np.inf)
    mx = np.full(C, -np.inf)
    mean_sum = np.zeros(C)
    n = 0
    frame_means = {nm: [] for nm in ad.channel_names}
    argmax = [None] * C
    frames = []
    for row in g.shard_index:
        b = int(row["bin"])
        for f in range(ad.frames_per_bin):
            fr = g.read_frame(b, f)
            if fr is None:
                continue
            frames.append((b, f))
            fm = (fr * w[:, :, None]).sum(axis=(0, 1)) / (w.sum() *
                                                         fr.shape[1])
            mean_sum += fm
            n += 1
            for k, nm in enumerate(ad.channel_names):
                frame_means[nm].append(float(fm[k]))
                v = fr[:, :, k]
                if np.nanmax(v) > mx[k]:
                    mx[k] = float(np.nanmax(v))
                    i, j = np.unravel_index(np.nanargmax(v), v.shape)
                    argmax[k] = {"bin": b, "frame": f,
                                 "instant": str(e5.frame_instant(b, f)),
                                 "lat": float(lat[i]),
                                 "lon": float(-180.0 + j)}
                mn[k] = min(mn[k], float(np.nanmin(v)))
    hist = {}
    for k, nm in enumerate(ad.channel_names):
        edges = np.linspace(mn[k], mx[k], 21)
        cnt = np.zeros(20, np.int64)
        for (b, f) in frames[::6]:                  # every sixth frame
            fr = g.read_frame(b, f)[:, :, k].ravel()
            cnt += np.histogram(fr[np.isfinite(fr)], edges)[0]
        hist[nm] = {"edges": [round(float(x), 6) for x in edges],
                    "counts": cnt.tolist(),
                    "frames_sampled": len(frames[::6])}
    return {
        "frames_read_back": n,
        "per_channel": {
            nm: {"min": round(float(mn[k]), 6), "max": round(float(mx[k]), 6),
                 "area_mean": round(float(mean_sum[k] / n), 6),
                 "area_mean_frames": [round(min(frame_means[nm]), 4),
                                      round(max(frame_means[nm]), 4)],
                 "argmax": argmax[k]}
            for k, nm in enumerate(ad.channel_names)},
        "histogram": hist,
        "area_weights": "cos(lat), pole rows sin(0.25 deg) (half cells)",
    }


def independent_frame(ad, when):
    """[181, 360, 13] float64 in the STORED unit, read with xarray + zarr
    from the 1-degree source archive — no adapter code involved."""
    import xarray as xr
    ds = xr.open_zarr(f"{e5.GCS}/{e5.ZARR_A}", consolidated=True,
                      chunks=None)
    da = ds[ad.src_name].sel(time=np.datetime64(when),
                             level=list(e5.LEVELS_HPA))
    da = da.assign_coords(longitude=((da.longitude + 180.0) % 360.0)
                          - 180.0).sortby("longitude").sortby("latitude")
    da = da.transpose("latitude", "longitude", "level")
    assert float(da.latitude[0]) == -90.0 and float(da.longitude[0]) == -180.0
    return da.values.astype(np.float64) * ad.scale


def readback(store_dir, ad, when):
    g = sh.ShardedGroup(store_dir)
    s = int((when - e5.EPOCH).total_seconds())
    b, rem = divmod(s, sh.BIN_SECONDS)
    f = rem // e5.FRAME_SECONDS
    got = g.read_frame(b, f).astype(np.float64)
    ref = independent_frame(ad, when)
    d = np.abs(got - ref)
    # half a float16 step at the value: the only error the store may carry.
    # The step is that of the binade holding |ref| (2^(e-10), subnormals
    # 2^-24), so a value just under a power of two that rounds UP to it is
    # judged by its own binade's step
    mag = np.abs(ref)
    e = np.floor(np.log2(np.maximum(mag, 2.0 ** -24)))
    step = 2.0 ** (np.maximum(e, -14.0) - 10.0)
    within = d <= step / 2 * (1 + 1e-9)
    return {"instant": str(when), "bin": b, "frame": f,
            "reference": ("xarray " + __import__("xarray").__version__
                          + " + zarr " + __import__("zarr").__version__
                          + f" on {e5.GCS}/{e5.ZARR_A}"),
            "max_abs_diff": {nm: float(d[:, :, k].max())
                             for k, nm in enumerate(ad.channel_names)},
            "max_abs_diff_all": float(d.max()),
            "within_half_float16_step": bool(within.all()),
            "values_compared": int(d.size)}


def seam(ad, instants):
    s = e5.Sources(e5.GCS, False, 1.0)
    out = {}
    for w in instants:
        _src, ia = "A", int((w - s.a_t0).total_seconds() // 3600) \
            - int(s.a_t[0])
        ib = int((w - s.b_t0).total_seconds() // 3600) - int(s.b_t[0])
        a, _ = s.frame(ad.src_name, "A", ia)
        b, _ = s.frame(ad.src_name, "B", ib)
        a, b = a * ad.scale, b * ad.scale
        d = np.abs(a - b)
        rel = d / np.maximum(np.abs(a), 1e-12)
        r16 = a.astype(np.float16)
        same16 = float((r16 == b.astype(np.float16)).mean())
        out[str(w)] = {
            "max_abs_diff": {nm: float(d[:, :, k].max())
                             for k, nm in enumerate(ad.channel_names)},
            "max_rel_diff": float(rel.max()),
            "float16_identical_fraction": round(same16, 6)}
    return out


def tail(ad, start, n, workers):
    """n consecutive six-hourly path-B frames from `start`: source bytes,
    seconds, and the bytes the real writer stores for each."""
    s = e5.Sources(e5.GCS, False, 1.0)
    spec = ad.specs()[ad.store]
    wr = sh.ShardWriter(spec)
    jobs = [start + dt.timedelta(hours=6 * i) for i in range(n)]

    def one(w):
        src, ti = s.where(w)
        assert src == "B", (w, src)
        t0 = time.time()
        arr, st = s.frame(ad.src_name, src, ti)
        return w, arr, st, time.time() - t0
    t0 = time.time()
    res = []
    with cf.ThreadPoolExecutor(workers) as ex:
        for w, arr, st, secs in ex.map(one, jobs):
            res.append((w, arr, st, secs))
    wall = time.time() - t0
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"),
                       f"era5_tail_{os.getpid()}")
    os.makedirs(tmp, exist_ok=True)
    stored = []
    for w, arr, _st, _s in res:
        v = arr.reshape(-1, ad.C) * ad.scale
        ad.mask_bounds(v)
        e = wr.write_bin(0, [v.reshape(arr.shape)] + [None] * 19,
                         os.path.join(tmp, "x.zst"),
                         os.path.join(tmp, "x.idx.npy"))
        stored.append(e["nbytes"])
    sb = [st["bytes"] for _w, _a, st, _s in res]
    return {"first": str(start), "frames": n, "workers": workers,
            "source_bytes_per_frame": {"mean": float(np.mean(sb)),
                                       "min": int(min(sb)),
                                       "max": int(max(sb))},
            "range_requests_per_frame": float(np.mean(
                [st["requests"] for _w, _a, st, _s in res])),
            "blosc_blocks_per_frame": float(np.mean(
                [st["blocks"] for _w, _a, st, _s in res])),
            "wall_seconds": round(wall, 2),
            "MB_per_s": round(sum(sb) / 1e6 / wall, 1),
            "seconds_per_frame_wall": round(wall / n, 3),
            "stored_bytes_per_frame": float(np.mean(stored))}


def tiles(store_dir, ad, n=8):
    g = sh.ShardedGroup(store_dir)
    rows = g.shard_index
    out = {}
    for T in (64, 256):
        cls = type(ad)
        a2 = cls()
        a2.tile = T
        wr = sh.ShardWriter(a2.specs()[ad.store])
        tot = 0
        tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"),
                           f"era5_tiles_{os.getpid()}")
        os.makedirs(tmp, exist_ok=True)
        for i in range(n):
            b = int(rows[i % len(rows)]["bin"])
            fr = g.read_frame(b, i % ad.frames_per_bin)
            e = wr.write_bin(b, [fr.astype(np.float32)] + [None] * 19,
                             os.path.join(tmp, "x.zst"),
                             os.path.join(tmp, "x.idx.npy"))
            tot += e["nbytes"]
        out[str(T)] = {"bytes_per_frame": tot / n,
                       "tiles_per_frame": a2.specs()[ad.store]["n_tiles_y"]
                       * a2.specs()[ad.store]["n_tiles_x"]}
    # WHAT-IF, NOT THE LAYOUT: the same 64-pixel tiles compressed with their
    # bytes reordered — channel-planar [channel, row, col], and planar with
    # the two bytes of each float16 split into two planes (a byte shuffle).
    # `sharded.py` fixes C order [row, col, channel]; these numbers say what
    # a layout option would buy, for the planning session to decide.
    import zstandard
    cz = zstandard.ZstdCompressor(level=sh.DEFAULT_LEVEL)
    T = ad.tile
    alt = {"c_order_row_col_channel": 0, "planar_channel_row_col": 0,
           "planar_byte_shuffled": 0}
    for i in range(n):
        b = int(rows[i % len(rows)]["bin"])
        fr = g.read_frame(b, i % ad.frames_per_bin, raw=True)
        H, W, C = fr.shape
        for y0 in range(0, H, T):
            for x0 in range(0, W, T):
                t = np.full((T, T, C), np.nan, np.float16)
                blk = fr[y0:y0 + T, x0:x0 + T]
                t[:blk.shape[0], :blk.shape[1]] = blk
                alt["c_order_row_col_channel"] += len(cz.compress(
                    t.tobytes()))
                pl = np.ascontiguousarray(t.transpose(2, 0, 1))
                alt["planar_channel_row_col"] += len(cz.compress(
                    pl.tobytes()))
                by = pl.view(np.uint8).reshape(C, T, T, 2)
                alt["planar_byte_shuffled"] += len(cz.compress(
                    np.ascontiguousarray(by.transpose(3, 0, 1, 2)).tobytes()))
    out["what_if_64"] = {k: v / n for k, v in alt.items()}
    out["what_if_note"] = ("bytes per frame for the same tiles with the "
                           "bytes reordered before zstd; only "
                           "c_order_row_col_channel is what sharded.py "
                           "writes")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", required=True)
    ap.add_argument("--month", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--rss", default="")
    ap.add_argument("--rss-build", default="")
    ap.add_argument("--readback", default="2015-01-15T12:00")
    ap.add_argument("--seam", default="2015-01-01T00:00,2021-12-31T18:00")
    ap.add_argument("--tail-start", default="2023-01-01T00:00")
    ap.add_argument("--tail-frames", type=int, default=24)
    ap.add_argument("--workers", type=int, default=e5.WORKERS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ad = REGISTRY[a.store]()
    probe = json.load(open(a.probe))
    t0 = time.time()
    ex = {"what": __doc__.split("\n\n")[0].strip()}
    ex["stats"] = stats(a.store_dir, ad)
    ex["readback"] = readback(a.store_dir, ad,
                              dt.datetime.fromisoformat(a.readback))
    ex["seam"] = seam(ad, [dt.datetime.fromisoformat(x)
                           for x in a.seam.split(",")])
    f10b.NET_BYTES["n"] = 0
    ex["tail"] = tail(ad, dt.datetime.fromisoformat(a.tail_start),
                      a.tail_frames, a.workers)
    ex["tiles"] = tiles(a.store_dir, ad)
    sm = json.load(open(os.path.join(os.path.dirname(a.store_dir),
                                     "store.json")))
    ex["local_store"] = {
        "dir_note": "the month's real fetch + assemble (lane m01 of 2015: "
                    "the bins whose first day is in January)",
        "groups": sm["groups"], "counts_frames_from":
            sm["counts"].get("frames_from"),
        "lanes_by_year": sm.get("lanes_by_year")}
    if a.rss:
        ex["process_probe"] = json.load(open(a.rss))
    if a.rss_build:
        ex["process_fetch_assemble"] = json.load(open(a.rss_build))
    ex["check_seconds"] = round(time.time() - t0, 1)
    ex["check_peak_rss_mb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    probe["era5"] = ex
    atomic_json(a.out, probe)
    print(json.dumps({k: v for k, v in ex.items()
                      if k not in ("stats",)}, indent=1)[:6000])
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
